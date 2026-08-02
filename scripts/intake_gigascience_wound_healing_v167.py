#!/usr/bin/env python3
"""Manifest-first intake for the Zaritsky et al. wound-healing dataset.

The runner queries the official GigaDB API (dataset 100118), records provenance
before download, supports resumable file transfer, safely extracts selected
per-experiment archives, and assigns an explicit causal availability time to
raw frames, ROI masks, motion fields, and wound-front derivatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "100118"
DOI = "10.5524/100118"
API_BASE = "https://gigadb.org/gigadb/api"
DATACITE_URL = f"https://api.datacite.org/dois/{DOI}"
DEFAULT_DATA_ROOT = ROOT / "new_data" / "gigascience_wound_healing_v167"
DEFAULT_OUT = ROOT / "outputs" / "gigascience_wound_healing_intake_v167_2026-07-28"
CHUNK_SIZE = 8 << 20
EXTRACTION_COLUMNS = [
    "experiment",
    "archive",
    "status",
    "archive_sha256",
    "members",
    "uncompressed_bytes",
    "marker",
]


CONDITIONS: dict[str, tuple[str, str]] = {}
for _name in ("SN29_L1", "SN29_L5", "SN29_L6", "SN77_L1", "SN77_L2", "SN77_L8"):
    CONDITIONS[_name] = ("DA3", "control")
for _name in ("SN29_L8", "SN29_L9", "SN29_L10", "SN29_L11", "SN29_L12"):
    CONDITIONS[_name] = ("DA3", "HGF")
for _name in ("SN77_L20", "SN77_L28", "SN77_L29", "SN77_L30"):
    CONDITIONS[_name] = ("DA3", "PHA")
for _name in ("SN77_L33", "SN77_L35", "SN77_L36", "SN77_L38", "SN77_L40", "SN77_L41"):
    CONDITIONS[_name] = ("DA3", "PHA_HGF")
for _name in ("DKWH7_L1", "DKWH7_L3", "DKWH7_L10", "DKWH7_L15", "DKWH7_L16"):
    CONDITIONS[_name] = ("MDCK", "control")
for _name in ("DKWH7_L4", "DKWH7_L5", "DKWH7_L6", "DKWH7_L18", "DKWH7_L19"):
    CONDITIONS[_name] = ("MDCK", "HGF")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def api_get(path: str, *, params: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/{path.lstrip('/')}",
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Airi-GigaScience-intake-v167/1.0"},
    )
    response.raise_for_status()
    return response.json()


def source_inventory(timeout: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    dataset = api_get(
        "dataset/get_dataset/",
        params={"dataset_id": DATASET_ID},
        timeout=timeout,
    )["data"]["data"]
    files = api_get(
        "dataset/list_dataset_files/",
        params={"dataset_id": DATASET_ID, "per_page": 100},
        timeout=timeout,
    )["data"]
    rows = files["data"]
    total = int(files["pagination"]["total"])
    if len(rows) != total:
        raise RuntimeError(f"GigaDB API returned {len(rows)} of {total} files")
    datacite_response = requests.get(
        DATACITE_URL,
        timeout=timeout,
        headers={"User-Agent": "Airi-GigaScience-intake-v167/1.0"},
    )
    datacite_response.raise_for_status()
    datacite = datacite_response.json()["data"]["attributes"]
    return dataset, rows, datacite


def file_kind(file_name: str, data_type: str) -> tuple[str, str | None]:
    lower = file_name.lower()
    experiment: str | None = None
    if lower.endswith(".tar.gz"):
        experiment = file_name[: -len(".tar.gz")]
        return "processed_experiment_archive", experiment
    if lower.endswith((".lsm", ".tif", ".tiff")):
        experiment = file_name.rsplit(".", 1)[0]
        return "raw_movie", experiment
    if file_name == "100118_AllData.tar":
        return "complete_archive", None
    if lower.endswith((".m", ".mat", ".txt")):
        return "metadata_or_script", None
    if data_type.lower() == "directory":
        return "remote_directory", None
    return "other", None


def select_file(kind: str, experiment: str | None, scope: str) -> bool:
    cell_line = CONDITIONS.get(experiment or "", ("", ""))[0]
    if scope == "none":
        return False
    if scope == "metadata":
        return kind == "metadata_or_script"
    if scope == "processed-mdck":
        return kind == "metadata_or_script" or (
            kind == "processed_experiment_archive" and cell_line == "MDCK"
        )
    if scope == "processed":
        return kind in {"metadata_or_script", "processed_experiment_archive"}
    if scope == "raw-mdck":
        return kind == "metadata_or_script" or (
            kind in {"processed_experiment_archive", "raw_movie"}
            and cell_line == "MDCK"
        )
    if scope == "raw":
        return kind in {
            "metadata_or_script",
            "processed_experiment_archive",
            "raw_movie",
        }
    if scope == "complete-archive":
        return kind == "complete_archive"
    if scope == "all":
        # The complete archive duplicates every individual source file.
        return kind not in {"remote_directory", "complete_archive"}
    raise ValueError(scope)


def existing_download_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    part = path.with_name(path.name + ".part")
    return int(part.stat().st_size) if part.is_file() else 0


def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    timeout: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size == expected_size:
        return {
            "status": "already_complete",
            "bytes": expected_size,
            "sha256": sha256_file(destination),
        }
    if destination.exists():
        destination.unlink()
    offset = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "Airi-GigaScience-intake-v167/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = requests.get(
        url,
        stream=True,
        timeout=(timeout, max(timeout, 120)),
        headers=headers,
    )
    if offset and response.status_code == 200:
        part.unlink(missing_ok=True)
        offset = 0
    elif offset and response.status_code != 206:
        response.raise_for_status()
        raise RuntimeError(
            f"Server ignored/resisted resume for {destination}: {response.status_code}"
        )
    else:
        response.raise_for_status()
    mode = "ab" if offset else "wb"
    with part.open(mode) as handle:
        for block in response.iter_content(chunk_size=CHUNK_SIZE):
            if block:
                handle.write(block)
    actual = int(part.stat().st_size)
    if expected_size > 0 and actual != expected_size:
        raise RuntimeError(
            f"Size mismatch for {destination.name}: expected {expected_size}, got {actual}"
        )
    part.replace(destination)
    return {
        "status": "downloaded",
        "bytes": actual,
        "sha256": sha256_file(destination),
    }


def safe_extract(archive: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted = 0
    total_bytes = 0
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Unsupported archive member: {member.name}")
        handle.extractall(destination, members=members)
        extracted = len(members)
        total_bytes = int(sum(max(0, int(member.size)) for member in members))
    marker = destination / ".v167_extracted.json"
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive.resolve()),
                "archive_sha256": sha256_file(archive),
                "members": extracted,
                "uncompressed_bytes": total_bytes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "extracted",
        "members": extracted,
        "uncompressed_bytes": total_bytes,
        "marker": str(marker.resolve()),
    }


def scalar_from_mat(value: Any) -> float | None:
    try:
        array = np.asarray(value).squeeze()
        if array.size != 1:
            return None
        number = float(array.item())
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def find_experiment_params(extracted_root: Path, experiment: str) -> Path | None:
    candidates = sorted(extracted_root.rglob("experimentParams.mat"))
    exact = [
        path
        for path in candidates
        if experiment.lower() in "/".join(path.parts).lower()
    ]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def metadata_parameter_lookup(data_root: Path) -> dict[str, dict[str, float]]:
    path = data_root / "downloads" / "metadata.mat"
    if not path.is_file():
        return {}
    try:
        metadata = loadmat(path, simplify_cells=True)["metaData"]
        names = np.asarray(metadata["experimentNames"], dtype=object).reshape(-1)
    except (KeyError, OSError, TypeError, ValueError, NotImplementedError):
        return {}
    fields = {
        "pixel_size_um": "pixelSize",
        "time_per_frame_min": "timePerFrame",
        "time_phase1_frames": "timePhase1",
    }
    lookup: dict[str, dict[str, float]] = {str(name): {} for name in names}
    for output_name, source_name in fields.items():
        values = np.asarray(metadata.get(source_name, []), dtype=object).reshape(-1)
        if len(values) != len(names):
            continue
        for name, value in zip(names, values):
            scalar = scalar_from_mat(value)
            if scalar is not None:
                lookup[str(name)][output_name] = scalar
    return lookup


def experiment_manifest(
    files: pd.DataFrame,
    data_root: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metadata_lookup = metadata_parameter_lookup(data_root)
    replicate_by_experiment: dict[str, int] = {}
    for cell_line, condition in sorted(set(CONDITIONS.values())):
        names = sorted(
            name
            for name, values in CONDITIONS.items()
            if values == (cell_line, condition)
        )
        replicate_by_experiment.update(
            {name: replicate for replicate, name in enumerate(names, start=1)}
        )
    for order, (experiment, (cell_line, condition)) in enumerate(
        sorted(CONDITIONS.items()), start=1
    ):
        subset = files[files["experiment"] == experiment]
        raw = subset[subset["kind"] == "raw_movie"]
        processed = subset[subset["kind"] == "processed_experiment_archive"]
        archive_path = (
            data_root / "downloads" / str(processed.iloc[0]["file_name"])
            if len(processed)
            else None
        )
        raw_path = (
            data_root / "downloads" / str(raw.iloc[0]["file_name"])
            if len(raw)
            else None
        )
        extracted = data_root / "processed" / experiment
        params_path = find_experiment_params(extracted, experiment) if extracted.exists() else None
        params: dict[str, Any] = {}
        params_load_error = ""
        if params_path:
            try:
                params = loadmat(params_path, simplify_cells=True)
            except (OSError, ValueError, NotImplementedError) as error:
                params = {}
                params_load_error = f"{type(error).__name__}: {error}"
        pixel_size = scalar_from_mat(
            params.get("pixelSize", params.get("picelSize"))
        )
        time_per_frame = scalar_from_mat(params.get("timePerFrame"))
        max_time = scalar_from_mat(params.get("maxTime"))
        time_phase1 = scalar_from_mat(params.get("timePhase1"))
        fallback = metadata_lookup.get(experiment, {})
        pixel_size = pixel_size or fallback.get("pixel_size_um")
        time_per_frame = time_per_frame or fallback.get("time_per_frame_min")
        time_phase1 = time_phase1 or fallback.get("time_phase1_frames")
        image_indices = []
        if extracted.exists():
            for image_path in extracted.rglob("*.tif"):
                try:
                    image_indices.append(int(image_path.stem))
                except ValueError:
                    continue
        if max_time is None and image_indices:
            max_time = float(max(image_indices))
        params_source = (
            "experimentParams.mat"
            if params
            else ("metadata.mat" if fallback else "unavailable")
        )
        records.append(
            {
                "experiment": experiment,
                "cell_line": cell_line,
                "condition": condition,
                "replicate": replicate_by_experiment[experiment],
                "outer_unit": experiment,
                "outer_fold_order": order,
                "raw_file": str(raw_path.resolve()) if raw_path else "",
                "raw_expected_bytes": int(raw.iloc[0]["file_size"]) if len(raw) else 0,
                "raw_available": bool(raw_path and raw_path.is_file()),
                "processed_archive": (
                    str(archive_path.resolve()) if archive_path else ""
                ),
                "processed_expected_bytes": (
                    int(processed.iloc[0]["file_size"]) if len(processed) else 0
                ),
                "processed_available": bool(
                    archive_path and archive_path.is_file()
                ),
                "extracted_dir": str(extracted.resolve()),
                "extracted": (extracted / ".v167_extracted.json").is_file(),
                "experiment_params": str(params_path.resolve()) if params_path else "",
                "experiment_params_source": params_source,
                "experiment_params_load_error": params_load_error,
                "pixel_size_um": pixel_size,
                "time_per_frame_min": time_per_frame,
                "max_time_min": max_time,
                "time_phase1_frames": time_phase1,
                "roi_available": bool(list(extracted.rglob("*_roi.mat"))) if extracted.exists() else False,
                "motion_field_available": bool(list(extracted.rglob("*_mf.mat"))) if extracted.exists() else False,
                "identity_preserving_tracks_available": False,
                "persistent_cell_contract": "not_established",
            }
        )
    return pd.DataFrame(records)


def availability_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "raw_frame_t",
                "source": "raw DIC movie",
                "transition": "state at t",
                "available_at_time": "t",
                "issue_time_feature": True,
                "target_or_privileged": False,
                "causal_note": "Current frame is observed before issuing t->t+1.",
            },
            {
                "variable": "roi_mask_t",
                "source": "official ROI or causal segmentation of frame t",
                "transition": "state at t",
                "available_at_time": "t",
                "issue_time_feature": True,
                "target_or_privileged": False,
                "causal_note": "Allowed only if computed without frames after t.",
            },
            {
                "variable": "motion_field_t_minus_1_to_t",
                "source": "official MF dxs/dys",
                "transition": "t-1 -> t",
                "available_at_time": "t",
                "issue_time_feature": True,
                "target_or_privileged": False,
                "causal_note": "Completed field transition.",
            },
            {
                "variable": "motion_field_t_to_t_plus_1",
                "source": "official MF dxs/dys",
                "transition": "t -> t+1",
                "available_at_time": "t+1",
                "issue_time_feature": False,
                "target_or_privileged": True,
                "causal_note": "Primary next-field target; forbidden at issue time t.",
            },
            {
                "variable": "front_contour_t",
                "source": "ROI boundary at frame t",
                "transition": "state at t",
                "available_at_time": "t",
                "issue_time_feature": True,
                "target_or_privileged": False,
                "causal_note": "Derived from current causal ROI only.",
            },
            {
                "variable": "front_velocity_t_to_t_plus_1",
                "source": "difference of front contours",
                "transition": "t -> t+1",
                "available_at_time": "t+1",
                "issue_time_feature": False,
                "target_or_privileged": True,
                "causal_note": "Secondary target; not a current feature.",
            },
            {
                "variable": "speed_kymograph_full",
                "source": "published full-experiment derivative",
                "transition": "whole sequence",
                "available_at_time": "experiment_end",
                "issue_time_feature": False,
                "target_or_privileged": True,
                "causal_note": "Diagnostic only unless recomputed causally up to t.",
            },
            {
                "variable": "treatment_condition",
                "source": "experiment metadata",
                "transition": "experiment-level",
                "available_at_time": "before t=0",
                "issue_time_feature": True,
                "target_or_privileged": False,
                "causal_note": "Use for stratification; never split frames across folds.",
            },
        ]
    )


def data_contract(
    dataset: dict[str, Any],
    datacite: dict[str, Any],
    experiments: pd.DataFrame,
    files: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "doi": DOI,
                "title": dataset["title"],
                "publisher": dataset["publisher"],
                "publication_date": dataset["publication_date"],
                "license": "; ".join(
                    str(item.get("rights", ""))
                    for item in datacite.get("rightsList", [])
                ),
                "declared_size": "; ".join(datacite.get("sizes", [])),
                "experiments": int(len(experiments)),
                "da3_experiments": int((experiments["cell_line"] == "DA3").sum()),
                "mdck_experiments": int((experiments["cell_line"] == "MDCK").sum()),
                "conditions": int(
                    experiments[["cell_line", "condition"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "api_files": int(len(files)),
                "complete_archive_bytes": int(
                    files.loc[
                        files["kind"] == "complete_archive", "file_size"
                    ].sum()
                ),
                "processed_archive_bytes": int(
                    files.loc[
                        files["kind"] == "processed_experiment_archive",
                        "file_size",
                    ].sum()
                ),
                "raw_movie_bytes": int(
                    files.loc[files["kind"] == "raw_movie", "file_size"].sum()
                ),
                "outer_unit": "experiment",
                "primary_target": "next causal motion/innovation field",
                "persistent_cell_target": "secondary_after_identity_audit",
                "target_leakage": False,
                "source_api": f"{API_BASE}/dataset/get_dataset/?dataset_id={DATASET_ID}",
                "article_doi": "10.1186/s13742-015-0049-6",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--download-scope",
        choices=[
            "none",
            "metadata",
            "processed-mdck",
            "processed",
            "raw-mdck",
            "raw",
            "complete-archive",
            "all",
        ],
        default="none",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--reserve-free-gib", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = data_root / "source_inventory.json"
    if args.offline:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        dataset = source["dataset"]
        source_rows = source["files"]
        datacite = source["datacite"]
    else:
        dataset, source_rows, datacite = source_inventory(args.timeout_seconds)
        source = {
            "dataset": dataset,
            "files": source_rows,
            "datacite": datacite,
            "retrieved_unix": time.time(),
        }
        source_path.write_text(
            json.dumps(finite(source), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    records: list[dict[str, Any]] = []
    for row in source_rows:
        kind, experiment = file_kind(
            str(row["file_name"]), str(row.get("data_type", ""))
        )
        cell_line, condition = CONDITIONS.get(experiment or "", ("", ""))
        destination = data_root / "downloads" / str(row["file_name"])
        selected = select_file(kind, experiment, args.download_scope)
        records.append(
            {
                "file_id": int(row["id"]),
                "file_name": str(row["file_name"]),
                "kind": kind,
                "experiment": experiment or "",
                "cell_line": cell_line,
                "condition": condition,
                "description": str(row.get("description", "")),
                "data_type": str(row.get("data_type", "")),
                "file_size": int(row.get("file_size") or 0),
                "url": str(row.get("url", "")),
                "server_md5": str(
                    (row.get("file_attributes") or {}).get("MD5 checksum", "")
                ),
                "selected": selected,
                "destination": str(destination.resolve()),
                "local_exists": destination.is_file(),
                "local_bytes": (
                    int(destination.stat().st_size)
                    if destination.is_file()
                    else existing_download_size(destination)
                ),
                "local_sha256": (
                    sha256_file(destination) if destination.is_file() else ""
                ),
                "download_status": "not_requested",
            }
        )
    files = pd.DataFrame(records)
    selected_indices = files.index[files["selected"]].tolist()
    if args.max_files > 0:
        selected_indices = selected_indices[: int(args.max_files)]
        files.loc[files["selected"], "selected"] = False
        files.loc[selected_indices, "selected"] = True

    missing_bytes = 0
    for index in selected_indices:
        expected = int(files.at[index, "file_size"])
        destination = Path(str(files.at[index, "destination"]))
        present = existing_download_size(destination)
        missing_bytes += max(0, expected - present)
    disk = shutil.disk_usage(data_root)
    reserve = int(float(args.reserve_free_gib) * (1024**3))
    if missing_bytes > max(0, disk.free - reserve):
        raise RuntimeError(
            "Insufficient disk for selected download scope: "
            f"missing={missing_bytes / 1e9:.2f} GB, "
            f"free={disk.free / 1e9:.2f} GB, "
            f"reserve={reserve / 1e9:.2f} GB"
        )

    for index in selected_indices:
        destination = Path(str(files.at[index, "destination"]))
        result = download_file(
            str(files.at[index, "url"]),
            destination,
            int(files.at[index, "file_size"]),
            int(args.timeout_seconds),
        )
        files.at[index, "download_status"] = result["status"]
        files.at[index, "local_exists"] = True
        files.at[index, "local_bytes"] = int(result["bytes"])
        files.at[index, "local_sha256"] = str(result["sha256"])

    extraction_rows: list[dict[str, Any]] = []
    if args.extract:
        archives = files[
            (files["kind"] == "processed_experiment_archive")
            & files["local_exists"].astype(bool)
        ]
        for row in archives.itertuples(index=False):
            archive = Path(row.destination)
            destination = data_root / "processed" / str(row.experiment)
            marker = destination / ".v167_extracted.json"
            if marker.is_file():
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                if marker_payload.get("archive_sha256") == row.local_sha256:
                    extraction_rows.append(
                        {
                            "experiment": row.experiment,
                            "archive": str(archive),
                            "status": "already_extracted",
                            **marker_payload,
                        }
                    )
                    continue
            extraction_rows.append(
                {
                    "experiment": row.experiment,
                    "archive": str(archive),
                    **safe_extract(archive, destination),
                }
            )

    experiments = experiment_manifest(files, data_root)
    availability = availability_contract()
    contract = data_contract(dataset, datacite, experiments, files)
    files.to_csv(out_dir / "gigascience_source_file_manifest.csv", index=False)
    experiments.to_csv(
        out_dir / "gigascience_experiment_manifest.csv", index=False
    )
    availability.to_csv(
        out_dir / "gigascience_availability_time_contract.csv", index=False
    )
    contract.to_csv(out_dir / "gigascience_data_contract.csv", index=False)
    pd.DataFrame(extraction_rows, columns=EXTRACTION_COLUMNS).to_csv(
        out_dir / "gigascience_extraction_manifest.csv", index=False
    )

    expected_experiments = set(CONDITIONS)
    source_experiments = set(
        files.loc[
            files["kind"] == "processed_experiment_archive", "experiment"
        ].astype(str)
    )
    audit = {
        "schema_version": 1,
        "status": (
            "pass"
            if len(files) == 68
            and source_experiments == expected_experiments
            and len(experiments) == 31
            else "fail"
        ),
        "dataset_id": DATASET_ID,
        "doi": DOI,
        "source_files": int(len(files)),
        "experiments": int(len(experiments)),
        "processed_archives": int(
            (files["kind"] == "processed_experiment_archive").sum()
        ),
        "raw_movies": int((files["kind"] == "raw_movie").sum()),
        "selected_files": int(files["selected"].sum()),
        "selected_missing_bytes_before_download": int(missing_bytes),
        "free_bytes_before_download": int(disk.free),
        "reserve_bytes": int(reserve),
        "download_scope": args.download_scope,
        "target_leakage": False,
        "identity_preserving_tracks_available": False,
        "elapsed_seconds": time.time() - started,
        "source_inventory_sha256": sha256_file(source_path),
    }
    (out_dir / "gigascience_intake_audit.json").write_text(
        json.dumps(finite(audit), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(finite({"out_dir": out_dir, **audit}), indent=2))


if __name__ == "__main__":
    run(parse_args())
