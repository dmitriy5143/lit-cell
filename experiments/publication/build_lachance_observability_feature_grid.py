#!/usr/bin/env python3
"""Build causal observability feature grids for LaChance forecasting.

This builder does not create new targets and does not use future frames.  It
derives extra state descriptors from an already aligned feature grid:

    morphology/polarity proxies
    tissue-flow lag / coherence proxies
    boundary/front geometry
    kNN density/crowding
    explicit shape-flow interaction terms

The output keeps original columns and appends `obs_*` columns so downstream
triage can test the new packet independently.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_image_feature_probe as ifp  # noqa: E402

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore[assignment]


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "lachance_feature_reconnaissance_ms_tf_mdck_bulk_h1h4h6_seed42_2026-06-15"
    / "combined_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_observability_feature_grid_2026-06-17"
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


def col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return df[name].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norm, EPS)


def available_radii(cols: list[str], prefix: str) -> list[int]:
    radii: set[int] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_r(\d+)_")
    for c in cols:
        m = pattern.match(c)
        if m:
            radii.add(int(m.group(1)))
    return sorted(radii)


def read_track_subset(table_root: Path, dataset: str, sequences: list[int], frames: set[int]) -> pd.DataFrame:
    tables = [ifp.read_track_table(table_root, dataset, seq, frames) for seq in sequences]
    keep = ["dataset", "sequence", "frame", "track_id", "dx_px", "dy_px", "QUALITY"]
    tracks = pd.concat(tables, ignore_index=True)
    return tracks[[c for c in keep if c in tracks.columns]].drop_duplicates(["dataset", "sequence", "frame", "track_id"])


def merge_tracks(features: pd.DataFrame, table_root: Path, dataset: str) -> pd.DataFrame:
    seqs = sorted(int(s) for s in features["sequence"].dropna().unique())
    frames = set(int(f) for f in features["frame"].dropna().unique())
    tracks = read_track_subset(table_root, dataset, seqs, frames)
    merged = features.merge(tracks, on=["dataset", "sequence", "frame", "track_id"], how="left")
    for c in ("dx_px", "dy_px", "QUALITY"):
        if c not in merged.columns:
            merged[c] = 0.0
    merged[["dx_px", "dy_px", "QUALITY"]] = merged[["dx_px", "dy_px", "QUALITY"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return merged


def add_boundary_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = len(df)
    dist = np.zeros(n, dtype=np.float32)
    nx = np.zeros(n, dtype=np.float32)
    ny = np.zeros(n, dtype=np.float32)
    tangent_x = np.zeros(n, dtype=np.float32)
    tangent_y = np.zeros(n, dtype=np.float32)
    radial = np.zeros(n, dtype=np.float32)
    front = np.zeros(n, dtype=np.float32)
    tangential_motion = np.zeros(n, dtype=np.float32)
    edge_balance = np.zeros(n, dtype=np.float32)
    own = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    own_u = unit(own)
    for _, part in df.groupby(["dataset", "sequence", "frame"], sort=False):
        ids = part.index.to_numpy(np.int64)
        x = part["x_px"].to_numpy(np.float32)
        y = part["y_px"].to_numpy(np.float32)
        if len(ids) < 2:
            continue
        x0, x1 = np.percentile(x, [1.0, 99.0])
        y0, y1 = np.percentile(y, [1.0, 99.0])
        w = max(float(x1 - x0), 1.0)
        h = max(float(y1 - y0), 1.0)
        left = (x - x0) / w
        right = (x1 - x) / w
        top = (y - y0) / h
        bottom = (y1 - y) / h
        stack = np.stack([left, right, top, bottom], axis=1)
        side = np.argmin(stack, axis=1)
        normal = np.zeros((len(ids), 2), dtype=np.float32)
        normal[side == 0] = (-1.0, 0.0)
        normal[side == 1] = (1.0, 0.0)
        normal[side == 2] = (0.0, -1.0)
        normal[side == 3] = (0.0, 1.0)
        tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
        cx = (x - (x0 + x1) * 0.5) / w
        cy = (y - (y0 + y1) * 0.5) / h
        ou = own_u[ids]
        dist[ids] = np.min(stack, axis=1)
        nx[ids] = normal[:, 0]
        ny[ids] = normal[:, 1]
        tangent_x[ids] = tangent[:, 0]
        tangent_y[ids] = tangent[:, 1]
        radial[ids] = np.sqrt(cx * cx + cy * cy)
        front[ids] = np.sum(ou * normal, axis=1)
        tangential_motion[ids] = np.sum(ou * tangent, axis=1)
        edge_balance[ids] = (left - right) + (top - bottom)
    out["obs_boundary_dist"] = dist
    out["obs_boundary_normal_x"] = nx
    out["obs_boundary_normal_y"] = ny
    out["obs_boundary_tangent_x"] = tangent_x
    out["obs_boundary_tangent_y"] = tangent_y
    out["obs_boundary_radial"] = radial
    out["obs_boundary_self_front"] = front
    out["obs_boundary_self_tangent"] = tangential_motion
    out["obs_boundary_edge_balance"] = edge_balance
    return out


def add_density_features(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for density features")
    out = pd.DataFrame(index=df.index)
    n = len(df)
    nearest = np.zeros(n, dtype=np.float32)
    mean8 = np.zeros(n, dtype=np.float32)
    mean32 = np.zeros(n, dtype=np.float32)
    crowd_x = np.zeros(n, dtype=np.float32)
    crowd_y = np.zeros(n, dtype=np.float32)
    counts = {float(r): np.zeros(n, dtype=np.float32) for r in radii}
    for _, part in df.groupby(["dataset", "sequence", "frame"], sort=False):
        ids = part.index.to_numpy(np.int64)
        pos = part[["x_px", "y_px"]].to_numpy(np.float32)
        if len(ids) <= 1:
            continue
        tree = cKDTree(pos)
        k = min(33, len(ids))
        dist, nbr = tree.query(pos, k=k)
        dist = np.atleast_2d(dist)
        nbr = np.atleast_2d(nbr)
        if dist.shape[0] != len(ids):
            dist = dist.T
            nbr = nbr.T
        d = dist[:, 1:]
        nb = nbr[:, 1:]
        nearest[ids] = np.nan_to_num(d[:, 0], nan=0.0, posinf=0.0)
        mean8[ids] = np.nanmean(d[:, : min(8, d.shape[1])], axis=1)
        mean32[ids] = np.nanmean(d, axis=1)
        rel_mean = np.zeros((len(ids), 2), dtype=np.float32)
        take = min(16, nb.shape[1])
        if take > 0:
            valid_nb = np.clip(nb[:, :take].astype(np.int64), 0, len(ids) - 1)
            rel = pos[valid_nb] - pos[:, None, :]
            rel_mean = np.nanmean(rel, axis=1)
        crowd_x[ids] = rel_mean[:, 0]
        crowd_y[ids] = rel_mean[:, 1]
        for r in radii:
            # subtract self
            c = np.asarray([len(x) - 1 for x in tree.query_ball_point(pos, float(r))], dtype=np.float32)
            counts[float(r)][ids] = c / max(math.pi * float(r) * float(r), 1.0)
    out["obs_knn_nearest_dist"] = nearest
    out["obs_knn_mean8_dist"] = mean8
    out["obs_knn_mean32_dist"] = mean32
    out["obs_crowd_vector_x"] = crowd_x
    out["obs_crowd_vector_y"] = crowd_y
    out["obs_crowd_vector_norm"] = np.sqrt(crowd_x * crowd_x + crowd_y * crowd_y)
    for r, values in counts.items():
        out[f"obs_density_r{int(r)}"] = values
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def add_flow_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    own = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    own_u = unit(own)
    for r in available_radii(list(df.columns), "tf"):
        cur_u = col(df, f"tf_r{r}_cur_u_mean")
        cur_v = col(df, f"tf_r{r}_cur_v_mean")
        prev_u = col(df, f"tf_r{r}_prev_u_mean")
        prev_v = col(df, f"tf_r{r}_prev_v_mean")
        cur_mag = col(df, f"tf_r{r}_cur_mag_mean")
        prev_mag = col(df, f"tf_r{r}_prev_mag_mean")
        cur_std = np.sqrt(np.square(col(df, f"tf_r{r}_cur_u_std")) + np.square(col(df, f"tf_r{r}_cur_v_std")))
        prev_std = np.sqrt(np.square(col(df, f"tf_r{r}_prev_u_std")) + np.square(col(df, f"tf_r{r}_prev_v_std")))
        cur = np.stack([cur_u, cur_v], axis=1)
        prev = np.stack([prev_u, prev_v], axis=1)
        cur_unit = unit(cur)
        prev_unit = unit(prev)
        accel = cur - prev
        prefix = f"obs_flow_r{r}_"
        out[prefix + "lag_delta_u"] = accel[:, 0]
        out[prefix + "lag_delta_v"] = accel[:, 1]
        out[prefix + "lag_delta_mag"] = cur_mag - prev_mag
        out[prefix + "accel_norm"] = np.linalg.norm(accel, axis=1)
        out[prefix + "cur_prev_cos"] = np.sum(cur_unit * prev_unit, axis=1)
        out[prefix + "cur_self_cos"] = np.sum(cur_unit * own_u, axis=1)
        out[prefix + "accel_self_proj"] = np.sum(accel * own_u, axis=1)
        out[prefix + "coherence"] = cur_mag / np.maximum(cur_std, EPS)
        out[prefix + "prev_coherence"] = prev_mag / np.maximum(prev_std, EPS)
        out[prefix + "coherence_delta"] = out[prefix + "coherence"] - out[prefix + "prev_coherence"]
        if f"tf_r{r}_cur_center_u" in df.columns and f"tf_r{r}_cur_center_v" in df.columns:
            center = np.stack([col(df, f"tf_r{r}_cur_center_u"), col(df, f"tf_r{r}_cur_center_v")], axis=1)
            out[prefix + "center_mean_disagree"] = np.linalg.norm(center - cur, axis=1)
        if f"tf_r{r}_cur_front_back_proj" in df.columns:
            out[prefix + "front_back_proj"] = col(df, f"tf_r{r}_cur_front_back_proj")
            out[prefix + "front_back_tangent"] = col(df, f"tf_r{r}_cur_front_back_tangent")
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def add_polarity_shape_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    own = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    own_u = unit(own)
    own_perp = np.stack([-own_u[:, 1], own_u[:, 0]], axis=1)
    ms_r = available_radii(list(df.columns), "ms")
    tf_r = available_radii(list(df.columns), "tf")
    # Prefer broad morphology and local/mid-scale flow, but keep all available.
    for r in ms_r:
        for phase in ("cur", "prev", "delta"):
            oc = col(df, f"ms_r{r}_{phase}_orient_cos")
            os = col(df, f"ms_r{r}_{phase}_orient_sin")
            orient = np.stack([oc, os], axis=1)
            cdx = col(df, f"ms_r{r}_{phase}_centroid_dx")
            cdy = col(df, f"ms_r{r}_{phase}_centroid_dy")
            centroid = np.stack([cdx, cdy], axis=1)
            elong = col(df, f"ms_r{r}_{phase}_elongation")
            prefix = f"obs_polarity_m{r}_{phase}_"
            out[prefix + "orient_self_proj"] = np.sum(orient * own_u, axis=1)
            out[prefix + "orient_self_tangent"] = np.sum(orient * own_perp, axis=1)
            out[prefix + "centroid_self_proj"] = np.sum(centroid * own_u, axis=1)
            out[prefix + "centroid_self_tangent"] = np.sum(centroid * own_perp, axis=1)
            out[prefix + "centroid_norm"] = np.linalg.norm(centroid, axis=1)
            out[prefix + "elong_self_abs"] = elong * np.abs(out[prefix + "orient_self_proj"])
            out[prefix + "elong_tangent_abs"] = elong * np.abs(out[prefix + "orient_self_tangent"])
    for mr in ms_r:
        oc = col(df, f"ms_r{mr}_cur_orient_cos")
        os = col(df, f"ms_r{mr}_cur_orient_sin")
        orient = np.stack([oc, os], axis=1)
        cdx = col(df, f"ms_r{mr}_cur_centroid_dx")
        cdy = col(df, f"ms_r{mr}_cur_centroid_dy")
        centroid = np.stack([cdx, cdy], axis=1)
        elong = col(df, f"ms_r{mr}_cur_elongation")
        for fr in tf_r:
            fu = col(df, f"tf_r{fr}_cur_u_mean")
            fv = col(df, f"tf_r{fr}_cur_v_mean")
            flow = np.stack([fu, fv], axis=1)
            flow_u = unit(flow)
            flow_perp = np.stack([-flow_u[:, 1], flow_u[:, 0]], axis=1)
            prefix = f"obs_shape_flow_m{mr}_f{fr}_"
            out[prefix + "orient_flow_proj"] = np.sum(orient * flow_u, axis=1)
            out[prefix + "orient_flow_tangent"] = np.sum(orient * flow_perp, axis=1)
            out[prefix + "centroid_flow_proj"] = np.sum(centroid * flow_u, axis=1)
            out[prefix + "centroid_flow_tangent"] = np.sum(centroid * flow_perp, axis=1)
            out[prefix + "elong_flow_abs"] = elong * np.abs(out[prefix + "orient_flow_proj"])
            out[prefix + "elong_flow_tangent_abs"] = elong * np.abs(out[prefix + "orient_flow_tangent"])
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def build(args: argparse.Namespace) -> pd.DataFrame:
    features = pd.read_csv(args.input)
    if args.dataset:
        features = features[features["dataset"].eq(args.dataset)].copy()
    if features.empty:
        raise ValueError(f"No rows for dataset={args.dataset} in {args.input}")
    features["sequence"] = features["sequence"].astype(int)
    features["frame"] = features["frame"].astype(int)
    features["track_id"] = features["track_id"].astype(int)
    work = merge_tracks(features, args.table_root, args.dataset)
    parts = [work]
    parts.append(add_boundary_features(work))
    parts.append(add_density_features(work, parse_ints(args.density_radii)))
    parts.append(add_flow_lag_features(work))
    parts.append(add_polarity_shape_flow_features(work))
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    obs_cols = [c for c in out.columns if c.startswith("obs_")]
    out[obs_cols] = out[obs_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    if args.keep_only_keys_and_obs:
        keep = [c for c in KEYS if c in out.columns] + obs_cols
        out = out[keep].copy()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--table-root", type=Path, default=ifp.DEFAULT_TABLE_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset", default="MDCK_Bulk")
    p.add_argument("--density-radii", default="40,80,120,240,320")
    p.add_argument("--keep-only-keys-and-obs", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = build(args)
    out_path = args.out_dir / "observability_feature_grid.csv"
    grid.to_csv(out_path, index=False)
    status = {
        "args": finite_json(vars(args)),
        "out_path": str(out_path),
        "rows": int(len(grid)),
        "cols": int(len(grid.columns)),
        "obs_cols": int(sum(c.startswith("obs_") for c in grid.columns)),
        "ms_cols": int(sum(c.startswith("ms_") for c in grid.columns)),
        "tf_cols": int(sum(c.startswith("tf_") for c in grid.columns)),
    }
    (args.out_dir / "observability_feature_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
