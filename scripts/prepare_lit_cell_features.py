#!/usr/bin/env python3
"""Rebuild the frozen LIT-Cell causal feature grid from LaChance raw data."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "experiments" / "publication"
CONTRACT_PATH = ROOT / "evidence" / "raw_context_v2_feature_contract.json"
MOVIES = (1, 2, 3, 4, 5, 6)

if str(PUBLICATION) not in sys.path:
    sys.path.insert(0, str(PUBLICATION))


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def digest_file(path: Path, algorithm: str = "sha256", chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open(newline="") as handle:
        columns = next(csv.reader(handle))
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1
    return rows, len(columns)


def artifact_status(path: Path, reference: dict[str, Any] | None, check: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "matches_reference": False}
    rows, columns = csv_shape(path)
    status: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": rows,
        "columns": columns,
        "bytes": path.stat().st_size,
    }
    if check == "hash":
        status["sha256"] = digest_file(path)
    if reference is not None:
        dimensions_match = rows == int(reference["rows"]) and columns == int(reference["columns"])
        if check == "off":
            status["matches_reference"] = None
        elif check == "schema":
            status["matches_reference"] = dimensions_match
        else:
            status["matches_reference"] = dimensions_match and status["bytes"] == int(reference["bytes"]) and status["sha256"] == reference["sha256"]
    return status


def table_path(table_root: Path, movie: int) -> Path:
    return table_root / "MDCK_Bulk" / f"MDCK_Bulk_{movie:02d}_tracks.csv"


def load_track_coordinates(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    aliases = {
        "frame": "frame" if "frame" in header else "FRAME",
        "track_id": "track_id" if "track_id" in header else "TRACK_ID",
        "x_px": "x_px",
        "y_px": "y_px",
    }
    missing = [source for source in aliases.values() if source not in header]
    if missing:
        raise ValueError(f"{path} is missing track columns: {missing}")
    frame = pd.read_csv(path, usecols=list(aliases.values())).rename(
        columns={source: target for target, source in aliases.items()}
    )
    for column in ("frame", "track_id", "x_px", "y_px"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["frame", "track_id", "x_px", "y_px"]).copy()
    frame["frame"] = frame["frame"].astype(int)
    frame["track_id"] = frame["track_id"].astype(int)
    return frame.loc[frame["track_id"].ge(0)].copy()


def central_track_crop(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    side_fraction = math.sqrt(float(fraction))
    centers = frame.groupby("track_id", sort=False)[["x_px", "y_px"]].median().reset_index()
    x_mid = float(centers["x_px"].median())
    y_mid = float(centers["y_px"].median())
    x_half = 0.5 * side_fraction * max(float(centers["x_px"].max() - centers["x_px"].min()), 1.0)
    y_half = 0.5 * side_fraction * max(float(centers["y_px"].max() - centers["y_px"].min()), 1.0)
    keep = centers.loc[
        centers["x_px"].between(x_mid - x_half, x_mid + x_half)
        & centers["y_px"].between(y_mid - y_half, y_mid + y_half),
        "track_id",
    ].to_numpy()
    if not len(keep):
        x_scale = max(float(centers["x_px"].std()), 1.0)
        y_scale = max(float(centers["y_px"].std()), 1.0)
        score = np.square((centers["x_px"].to_numpy(float) - x_mid) / x_scale)
        score += np.square((centers["y_px"].to_numpy(float) - y_mid) / y_scale)
        count = min(len(centers), max(50, int(math.ceil(float(fraction) * len(centers)))))
        keep = centers.iloc[np.argsort(score)[:count]]["track_id"].to_numpy()
    return frame.loc[frame["track_id"].isin(keep)].copy()


def build_tracking_index(table_root: Path, output: Path, crop_fraction: float) -> Path:
    parts: list[pd.DataFrame] = []
    for movie in MOVIES:
        frame = central_track_crop(load_track_coordinates(table_path(table_root, movie)), crop_fraction)
        frame = frame.drop_duplicates(["track_id", "frame"], keep="first")
        frame = frame.sort_values(["track_id", "frame"])
        frame.insert(0, "sequence_name", f"{movie:02d}")
        frame.insert(0, "sequence", movie)
        frame.insert(0, "dataset", "MDCK_Bulk")
        parts.append(frame[["dataset", "sequence", "sequence_name", "frame", "track_id", "x_px", "y_px"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(output, index=False)
    return output


def verify_raw_zip(path: Path, contract: dict[str, Any], hash_file: bool) -> dict[str, Any]:
    reference = contract["raw_source"]
    status = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return status
    status["bytes"] = path.stat().st_size
    status["size_matches"] = status["bytes"] == int(reference["bytes"])
    if hash_file:
        status["md5"] = digest_file(path, "md5")
        status["md5_matches"] = status["md5"] == reference["md5"]
    return status


def extract_sequence_stacks(raw_zip: Path, stack_dir: Path) -> list[Path]:
    stack_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with zipfile.ZipFile(raw_zip) as archive:
        members = archive.namelist()
        for movie in MOVIES:
            suffix = f"/{movie:02d}.tif"
            candidates = sorted(name for name in members if name.endswith(suffix) or name == f"{movie:02d}.tif")
            if len(candidates) != 1:
                raise RuntimeError(f"expected one TIFF stack for movie {movie}, found {candidates}")
            info = archive.getinfo(candidates[0])
            target = stack_dir / f"{movie:02d}.tif"
            if not target.exists() or target.stat().st_size != info.file_size:
                temporary = target.with_suffix(".tif.part")
                with archive.open(candidates[0]) as source, temporary.open("wb") as destination:
                    while chunk := source.read(16 * 1024 * 1024):
                        destination.write(chunk)
                temporary.replace(target)
            outputs.append(target)
    return outputs


def verify_stacks(stack_dir: Path, expected_frames: int) -> list[dict[str, Any]]:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("tifffile is required to validate the raw stacks") from exc
    rows: list[dict[str, Any]] = []
    for movie in MOVIES:
        path = stack_dir / f"{movie:02d}.tif"
        row: dict[str, Any] = {"movie": movie, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            with tifffile.TiffFile(path) as stack:
                row["frames"] = len(stack.pages)
                row["shape"] = list(stack.pages[0].shape)
                row["valid"] = len(stack.pages) == expected_frames
        rows.append(row)
    return rows


def build_feature_grid(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    import build_lachance_observability_feature_grid as observability
    import build_lachance_raw_context_v2_grid as raw_context
    import run_lachance_feature_reconnaissance as reconnaissance
    import run_lachance_multiscale_image_feature_probe as multiscale
    import run_lachance_tissue_flow_feature_probe as tissue_flow

    if not args.resume or not paths["multiscale_image"].exists():
        frame = multiscale.extract_multiscale_features(
            image_index_path=paths["tracking_index"],
            table_root=args.table_root,
            stack_dir=args.stack_dir,
            dataset="MDCK_Bulk",
            radii=[8, 16, 24, 40],
            max_rows=0,
            seed=42,
        )
        frame.to_csv(paths["multiscale_image"], index=False)
        del frame
        gc.collect()

    if not args.resume or not paths["tissue_flow"].exists():
        frame = tissue_flow.extract_tissue_flow_features(
            SimpleNamespace(
                dataset="MDCK_Bulk",
                table_root=args.table_root,
                stack_dir=args.stack_dir,
                point_index=paths["tracking_index"],
                sequences="1,2,3,4,5,6",
                frames="1:49:1",
                radii="64,128,256",
                downsample=16,
                flow_radius=5,
                flow_num_warp=5,
                max_points_per_frame=1024,
                seed=42,
            )
        )
        frame.to_csv(paths["tissue_flow"], index=False)
        del frame
        gc.collect()

    if not args.resume or not paths["combined"].exists():
        frame = reconnaissance.load_combined(paths["multiscale_image"], paths["tissue_flow"], "MDCK_Bulk")
        frame.to_csv(paths["combined"], index=False)
        del frame
        gc.collect()

    if not args.resume or not paths["observability"].exists():
        frame = observability.build(
            SimpleNamespace(
                input=paths["combined"],
                table_root=args.table_root,
                dataset="MDCK_Bulk",
                density_radii="40,80,120,240,320",
                keep_only_keys_and_obs=False,
            )
        )
        frame.to_csv(paths["observability"], index=False)
        del frame
        gc.collect()

    if not args.resume or not paths["raw_context_v2"].exists():
        frame = raw_context.build_grid(
            SimpleNamespace(
                input=paths["observability"],
                dataset="MDCK_Bulk",
                table_root=args.table_root,
                stack_dir=args.stack_dir,
                radii="24,56,96",
                temporal_radii="24,56",
                temporal_lags="1,2,4",
                offset_scale=1.25,
                neighbour_radius=24,
                neighbour_k=8,
                max_rows=0,
                seed=42,
            )
        )
        frame.to_csv(paths["raw_context_v2"], index=False)
        del frame
        gc.collect()


def output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "tracking_index": out_dir / "tracking_index.csv",
        "multiscale_image": out_dir / "multiscale_image_features.csv",
        "tissue_flow": out_dir / "tissue_flow_features.csv",
        "combined": out_dir / "combined_feature_grid.csv",
        "observability": out_dir / "observability_feature_grid.csv",
        "raw_context_v2": out_dir / "raw_context_v2_feature_grid.csv",
    }


def require_inputs(args: argparse.Namespace, need_stacks: bool) -> None:
    missing = [str(table_path(args.table_root, movie)) for movie in MOVIES if not table_path(args.table_root, movie).is_file()]
    if missing:
        raise FileNotFoundError(f"missing MDCK_Bulk track tables: {missing}")
    if need_stacks:
        invalid = [row for row in verify_stacks(args.stack_dir, 49) if not row.get("valid")]
        if invalid:
            raise RuntimeError(f"raw stack validation failed: {invalid}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "index", "build", "all", "verify"], default="preflight")
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--stack-dir", type=Path, required=True)
    parser.add_argument("--raw-zip", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-check", choices=["off", "schema", "hash"], default="hash")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.table_root = args.table_root.resolve()
    args.stack_dir = args.stack_dir.resolve()
    args.raw_zip = args.raw_zip.resolve() if args.raw_zip else None
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    paths = output_paths(args.out_dir)

    raw_status = None
    if args.raw_zip:
        raw_status = verify_raw_zip(args.raw_zip, contract, args.reference_check == "hash")
        if args.mode in {"preflight", "all"}:
            raw_valid = bool(raw_status.get("size_matches")) and (
                args.reference_check != "hash" or bool(raw_status.get("md5_matches"))
            )
            if not raw_valid:
                raise RuntimeError(f"raw ZIP does not match the frozen source contract: {raw_status}")
    if args.mode == "all":
        if args.raw_zip is None:
            raise ValueError("--mode all requires --raw-zip")
        extract_sequence_stacks(args.raw_zip, args.stack_dir)

    need_stacks = args.mode in {"build", "all"}
    require_inputs(args, need_stacks=need_stacks)
    if args.mode in {"index", "build", "all"} and (not args.resume or not paths["tracking_index"].exists()):
        build_tracking_index(args.table_root, paths["tracking_index"], crop_fraction=0.025)
    if args.mode in {"build", "all"} and args.resume and args.reference_check != "off":
        for name, path in paths.items():
            if path.exists():
                status = artifact_status(path, contract["stage_artifacts"][name], args.reference_check)
                if status.get("matches_reference") is False:
                    raise RuntimeError(
                        f"refusing to resume from a non-reference {name} artifact; "
                        "use --no-resume to rebuild it"
                    )
    if args.mode in {"build", "all"}:
        build_feature_grid(args, paths)

    references = contract["stage_artifacts"]
    stages = {
        name: artifact_status(path, references[name], args.reference_check)
        for name, path in paths.items()
    }
    stack_status = verify_stacks(args.stack_dir, int(contract["raw_source"]["frames_per_stack"]))
    report = {
        "schema": "lit-cell-feature-preparation-v1",
        "mode": args.mode,
        "reference_check": args.reference_check,
        "raw_zip": raw_status,
        "stacks": stack_status,
        "stages": stages,
        "final_grid": str(paths["raw_context_v2"]),
    }
    report_path = args.out_dir / "feature_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    required_stages = {
        "index": {"tracking_index"},
        "build": set(paths),
        "all": set(paths),
        "verify": set(paths),
    }.get(args.mode, set())
    if required_stages and args.reference_check != "off":
        failed = [
            name
            for name in sorted(required_stages)
            if stages[name].get("matches_reference") is False
        ]
        if failed:
            raise RuntimeError(f"feature artifact verification failed: {failed}")
    print(report_path)


if __name__ == "__main__":
    main()
