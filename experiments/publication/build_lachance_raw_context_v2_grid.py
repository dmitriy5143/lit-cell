#!/usr/bin/env python3
"""Build richer raw-context v2 features for LaChance trajectories.

This builder adds causal features that were missing from the first raw-context
pass:

- larger structure-tensor morphology at the focal cell;
- non-centered front/back/left/right windows in the ego-motion frame;
- temporal appearance differences over multiple past lags;
- neighbour raw-context summaries from nearby tracked cells.

The script starts from an existing aligned feature grid, keeps its rows, and
appends `rc_*` columns.  No future frames or target-derived labels are used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_image_feature_extraction as ife  # noqa: E402
import run_lachance_image_feature_probe as ifp  # noqa: E402

DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "lachance_feature_reconnaissance_ms_tf_mdck_bulk_h1h4h6_seed42_2026-06-15"
    / "combined_feature_grid.csv"
)
DEFAULT_STACK_DIR = (
    ROOT
    / "new_data"
    / "lachance_epithelia"
    / "raw_timelapse"
    / "extracted_stacks"
    / "MDCK_Bulk_Timelapse_Data_Sample_Tissues"
)
DEFAULT_TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"
DEFAULT_OUT = ROOT / "outputs" / "lachance_raw_context_v2_grid_2026-06-17"
KEYS = ["dataset", "sequence", "frame", "track_id", "x_px", "y_px"]
EPS = 1e-8


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text or "").split(",") if p.strip()]


def read_frame(stack_dir: Path, sequence: int, frame: int, cache: dict[tuple[int, int], np.ndarray]) -> np.ndarray | None:
    if frame < 0:
        return None
    key = (int(sequence), int(frame))
    if key in cache:
        return cache[key]
    path = stack_dir / f"{int(sequence):02d}.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        image = ife.read_tiff_page(path, int(frame))
    except Exception:
        return None
    cache[key] = image
    if len(cache) > 16:
        for old in list(cache.keys())[:-16]:
            cache.pop(old, None)
    return image


def prefix_features(prefix: str, feats: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in feats.items():
        if key == "img_orientation":
            continue
        out[prefix + key.replace("img_", "")] = float(value)
    return out


def safe_patch(image: np.ndarray | None, x: float, y: float, radius: int) -> np.ndarray:
    if image is None:
        return np.zeros((0, 0), dtype=np.float32)
    return ife.crop_patch(image, x, y, int(radius))


def structure_tensor_features(patch: np.ndarray) -> dict[str, float]:
    if patch.size == 0:
        return {
            "struct_energy": 0.0,
            "struct_coherence": 0.0,
            "struct_orient_cos": 0.0,
            "struct_orient_sin": 0.0,
            "struct_anisotropy": 0.0,
        }
    p = ife.normalize_image(patch)
    if p.size == 0 or min(p.shape) < 3:
        return {
            "struct_energy": 0.0,
            "struct_coherence": 0.0,
            "struct_orient_cos": 0.0,
            "struct_orient_sin": 0.0,
            "struct_anisotropy": 0.0,
        }
    gy, gx = np.gradient(p.astype(np.float32, copy=False))
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    disc = math.sqrt(max((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy, 0.0))
    l1 = 0.5 * (trace + disc)
    l2 = 0.5 * (trace - disc)
    theta = 0.5 * math.atan2(2.0 * jxy, jxx - jyy + EPS)
    coherence = (l1 - l2) / max(l1 + l2, EPS)
    return {
        "struct_energy": float(trace),
        "struct_coherence": float(coherence),
        "struct_orient_cos": float(math.cos(theta)),
        "struct_orient_sin": float(math.sin(theta)),
        "struct_anisotropy": float(l1 / max(l2, EPS)),
    }


def raw_patch_features(image: np.ndarray | None, x: float, y: float, radius: int) -> dict[str, float]:
    patch = safe_patch(image, x, y, radius)
    if patch.size == 0:
        base = {}
    else:
        base = ife.patch_features(patch)
    base.update(structure_tensor_features(patch))
    return {k: float(v) for k, v in base.items() if k != "img_orientation"}


def selected_patch_vector(feats: dict[str, float]) -> np.ndarray:
    keys = [
        "img_mean",
        "img_std",
        "img_grad_mean",
        "img_grad_p90",
        "img_fg_frac",
        "img_centroid_dx",
        "img_centroid_dy",
        "img_elongation",
        "img_orient_cos",
        "img_orient_sin",
        "struct_energy",
        "struct_coherence",
        "struct_orient_cos",
        "struct_orient_sin",
    ]
    return np.asarray([float(feats.get(k, 0.0)) for k in keys], dtype=np.float32)


def ego_axes(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray([float(dx), float(dy)], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        axis = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        axis = v / n
    tangent = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    return axis, tangent, n


def load_tracks_for_rows(table_root: Path, dataset: str, rows: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    frames_by_seq: dict[int, set[int]] = {}
    max_lag = max(lags) if lags else 0
    for seq, part in rows.groupby("sequence"):
        frames = set(int(f) for f in part["frame"].unique())
        for lag in range(max_lag + 1):
            frames.update(int(f) - lag for f in part["frame"].unique() if int(f) - lag >= 0)
        frames_by_seq[int(seq)] = frames
    tables = []
    for seq, frames in frames_by_seq.items():
        path = table_root / dataset / f"{dataset}_{seq:02d}_tracks.csv"
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in ifp.TRACK_COLS if c in header.columns]
        table = pd.read_csv(path, usecols=usecols)
        table = table[table["frame"].isin(frames)].copy()
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    tracks = pd.concat(tables, ignore_index=True)
    tracks["sequence"] = tracks["sequence"].astype(int)
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)
    return tracks


def track_lookup(tracks: pd.DataFrame) -> dict[tuple[int, int, int], tuple[float, float]]:
    out: dict[tuple[int, int, int], tuple[float, float]] = {}
    for row in tracks[["sequence", "frame", "track_id", "x_px", "y_px"]].itertuples(index=False):
        out[(int(row.sequence), int(row.frame), int(row.track_id))] = (float(row.x_px), float(row.y_px))
    return out


def add_current_velocity(rows: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    vel_cols = ["dataset", "sequence", "frame", "track_id", "dx_px", "dy_px", "QUALITY"]
    existing = [c for c in vel_cols if c in tracks.columns]
    cur = tracks[existing].copy()
    merged = rows.merge(cur, on=["dataset", "sequence", "frame", "track_id"], how="left")
    merged["dx_px"] = merged.get("dx_px", 0.0).fillna(0.0)
    merged["dy_px"] = merged.get("dy_px", 0.0).fillna(0.0)
    if "QUALITY" not in merged.columns:
        merged["QUALITY"] = 0.0
    merged["QUALITY"] = merged["QUALITY"].fillna(0.0)
    return merged


def focal_and_noncenter_features(
    row: pd.Series,
    image: np.ndarray | None,
    *,
    radii: list[int],
    offset_scale: float,
) -> dict[str, float]:
    x = float(row["x_px"])
    y = float(row["y_px"])
    axis, tangent, speed = ego_axes(float(row.get("dx_px", 0.0)), float(row.get("dy_px", 0.0)))
    out: dict[str, float] = {"rc_self_speed": speed}
    for radius in radii:
        center = raw_patch_features(image, x, y, radius)
        out.update(prefix_features(f"rc_c_r{radius}_", center))
        offset = float(radius) * float(offset_scale)
        positions = {
            "front": np.asarray([x, y], dtype=np.float32) + axis * offset,
            "back": np.asarray([x, y], dtype=np.float32) - axis * offset,
            "left": np.asarray([x, y], dtype=np.float32) + tangent * offset,
            "right": np.asarray([x, y], dtype=np.float32) - tangent * offset,
        }
        patch_feats: dict[str, dict[str, float]] = {}
        for name, pos in positions.items():
            feats = raw_patch_features(image, float(pos[0]), float(pos[1]), radius)
            patch_feats[name] = feats
            out.update(prefix_features(f"rc_nc_r{radius}_{name}_", feats))
        for a, b in (("front", "back"), ("left", "right")):
            fa = patch_feats[a]
            fb = patch_feats[b]
            for key in ("img_mean", "img_std", "img_grad_mean", "img_fg_frac", "struct_energy", "struct_coherence"):
                short = key.replace("img_", "")
                out[f"rc_nc_r{radius}_{a}_minus_{b}_{short}"] = float(fa.get(key, 0.0) - fb.get(key, 0.0))
    return out


def temporal_lag_features(
    row: pd.Series,
    *,
    current_image: np.ndarray | None,
    frame_cache: dict[tuple[int, int], np.ndarray],
    stack_dir: Path,
    lookup: dict[tuple[int, int, int], tuple[float, float]],
    lags: list[int],
    radii: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    sequence = int(row["sequence"])
    frame = int(row["frame"])
    track_id = int(row["track_id"])
    x = float(row["x_px"])
    y = float(row["y_px"])
    for lag in lags:
        past_frame = frame - int(lag)
        past_pos = lookup.get((sequence, past_frame, track_id))
        has = float(past_pos is not None and past_frame >= 0)
        past_image = read_frame(stack_dir, sequence, past_frame, frame_cache) if has else None
        px, py = past_pos if past_pos is not None else (x, y)
        out[f"rc_lag{lag}_has"] = has
        for radius in radii:
            cur = raw_patch_features(current_image, x, y, radius)
            prev = raw_patch_features(past_image, px, py, radius)
            for key in (
                "img_mean",
                "img_std",
                "img_grad_mean",
                "img_grad_p90",
                "img_fg_frac",
                "img_centroid_dx",
                "img_centroid_dy",
                "img_elongation",
                "struct_energy",
                "struct_coherence",
            ):
                short = key.replace("img_", "")
                out[f"rc_lag{lag}_r{radius}_delta_{short}"] = float(cur.get(key, 0.0) - prev.get(key, 0.0))
    return out


def neighbour_summary_features(
    group: pd.DataFrame,
    image: np.ndarray | None,
    *,
    radius: int,
    k: int,
) -> dict[int, dict[str, float]]:
    if cKDTree is None or len(group) <= 1:
        return {int(i): {} for i in group.index}
    pos = group[["x_px", "y_px"]].to_numpy(np.float32)
    take = min(int(k) + 1, len(group))
    dist, nbr = cKDTree(pos).query(pos, k=take)
    dist = np.atleast_2d(dist)
    nbr = np.atleast_2d(nbr)
    if dist.shape[0] != len(group):
        dist = dist.T
        nbr = nbr.T
    vectors = []
    for row in group.itertuples(index=False):
        feats = raw_patch_features(image, float(row.x_px), float(row.y_px), radius)
        vectors.append(selected_patch_vector(feats))
    mat = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 14), dtype=np.float32)
    out: dict[int, dict[str, float]] = {}
    indices = group.index.to_numpy(np.int64)
    for i, row_index in enumerate(indices):
        valid = np.isfinite(dist[i, 1:])
        nb = nbr[i, 1:][valid].astype(np.int64)[: int(k)]
        dd = dist[i, 1:][valid].astype(np.float32)[: int(k)]
        row_out: dict[str, float] = {
            "rc_nei_count": float(len(nb)),
            "rc_nei_dist_mean": float(np.mean(dd)) if len(dd) else 0.0,
            "rc_nei_dist_min": float(np.min(dd)) if len(dd) else 0.0,
        }
        if len(nb):
            vals = mat[nb]
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            names = [
                "mean",
                "std",
                "grad_mean",
                "grad_p90",
                "fg_frac",
                "centroid_dx",
                "centroid_dy",
                "elongation",
                "orient_cos",
                "orient_sin",
                "struct_energy",
                "struct_coherence",
                "struct_orient_cos",
                "struct_orient_sin",
            ]
            for j, name in enumerate(names):
                row_out[f"rc_nei_r{radius}_{name}_mean"] = float(mean[j])
                row_out[f"rc_nei_r{radius}_{name}_std"] = float(std[j])
        out[int(row_index)] = row_out
    return out


def build_grid(args: argparse.Namespace) -> pd.DataFrame:
    base = pd.read_csv(args.input)
    if args.dataset:
        base = base[base["dataset"].eq(args.dataset)].copy()
    if args.max_rows > 0 and len(base) > args.max_rows:
        base = base.sample(n=int(args.max_rows), random_state=int(args.seed)).sort_values(
            ["sequence", "frame", "track_id"]
        )
    base["sequence"] = base["sequence"].astype(int)
    base["frame"] = base["frame"].astype(int)
    base["track_id"] = base["track_id"].astype(int)
    lags = parse_ints(args.temporal_lags)
    tracks = load_tracks_for_rows(args.table_root, args.dataset, base, lags)
    work = add_current_velocity(base[KEYS].copy(), tracks)
    lookup = track_lookup(tracks)
    radii = parse_ints(args.radii)
    temporal_radii = parse_ints(args.temporal_radii)
    neighbour_radius = int(args.neighbour_radius)
    rows: list[dict[str, Any]] = []
    frame_cache: dict[tuple[int, int], np.ndarray] = {}
    for (sequence, frame), group in work.groupby(["sequence", "frame"], sort=True):
        image = read_frame(args.stack_dir, int(sequence), int(frame), frame_cache)
        nei = neighbour_summary_features(group, image, radius=neighbour_radius, k=int(args.neighbour_k))
        print(f"[raw-v2] seq={int(sequence):02d} frame={int(frame):04d} rows={len(group)}", flush=True)
        for idx, row in group.iterrows():
            out: dict[str, Any] = {
                "dataset": args.dataset,
                "sequence": int(row["sequence"]),
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "x_px": float(row["x_px"]),
                "y_px": float(row["y_px"]),
            }
            out.update(
                focal_and_noncenter_features(
                    row,
                    image,
                    radii=radii,
                    offset_scale=float(args.offset_scale),
                )
            )
            out.update(
                temporal_lag_features(
                    row,
                    current_image=image,
                    frame_cache=frame_cache,
                    stack_dir=args.stack_dir,
                    lookup=lookup,
                    lags=lags,
                    radii=temporal_radii,
                )
            )
            out.update(nei.get(int(idx), {}))
            rows.append(out)
    rc = pd.DataFrame(rows)
    rc_cols = [c for c in rc.columns if c.startswith("rc_")]
    rc[rc_cols] = rc[rc_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    merged = base.merge(rc, on=KEYS, how="inner")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--radii", default="24,56,96")
    parser.add_argument("--temporal-radii", default="24,56")
    parser.add_argument("--temporal-lags", default="1,2,4")
    parser.add_argument("--offset-scale", type=float, default=1.25)
    parser.add_argument("--neighbour-radius", type=int, default=24)
    parser.add_argument("--neighbour-k", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args)
    out_path = args.out_dir / "raw_context_v2_feature_grid.csv"
    grid.to_csv(out_path, index=False)
    rc_cols = [c for c in grid.columns if c.startswith("rc_")]
    status = {
        "args": finite_json(vars(args)),
        "out_path": out_path,
        "rows": int(len(grid)),
        "cols": int(len(grid.columns)),
        "rc_cols": int(len(rc_cols)),
        "rc_center_cols": int(sum(c.startswith("rc_c_") for c in rc_cols)),
        "rc_noncenter_cols": int(sum(c.startswith("rc_nc_") for c in rc_cols)),
        "rc_temporal_cols": int(sum(c.startswith("rc_lag") for c in rc_cols)),
        "rc_neighbour_cols": int(sum(c.startswith("rc_nei") for c in rc_cols)),
    }
    (args.out_dir / "raw_context_v2_status.json").write_text(
        json.dumps(finite_json(status), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
