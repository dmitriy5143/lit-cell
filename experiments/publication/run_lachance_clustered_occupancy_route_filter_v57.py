#!/usr/bin/env python3
"""v57 clustered occupancy route filter.

This runner tests the next video/coordinate fusion hypothesis without using a
flat scalar reranker:

    fixed clean-best route basis
    + route/cluster trajectory hypotheses
    + causal mask-size / neighbour occupancy feasibility
    -> cluster-level veto / sparse route mixture

The visual/state branch is deliberately translated into explicit feasibility
variables first: radius, contact/free-space, neighbour occupancy, swept path
clearance and reliability.  Target/future residuals are used only for training
risk labels and diagnostics, never for inference features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_dense_state_target_reformulation_sweep_v25 as v25  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_visual_evidence_risk_probe_v42 as v42  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "clustered_occupancy_route_filter_v57_2026-07-13"
DEFAULT_STATE_GRID = ROOT / "outputs" / "temporal_identity_seg_state_v40_bulk_seed42_2026-07-08" / "temporal_identity_seg_state_v40.csv"
KEY_COLS = ["dataset", "sequence", "frame", "track_id"]
EPS = 1e-8

STATE_DEFAULTS = {
    "segf_sam_box_radius": 42.0,
    "segf_area_frac": 0.10,
    "segf_equivalent_diameter_norm": 0.70,
    "segf_reliability": 0.70,
    "segf_contact_pressure": 0.15,
    "segf_free_front_frac": 0.186,
    "segf_free_back_frac": 0.186,
    "segf_free_left_frac": 0.186,
    "segf_free_right_frac": 0.186,
    "segf_boundary_grad": 0.0,
    "segf_front_back_balance": 0.0,
    "segf_polarity_balance_norm": 0.0,
    "segf_center_inside": 0.0,
}

STATE_ALIASES = {
    "segf_sam_box_radius": [
        "segf_sam_box_radius",
        "msam_sam_box_radius",
        "cpose_sam_box_radius",
        "cpose_box_radius",
        "mi_lag0_sam_box_radius",
        "mi_large_sam_box_radius",
    ],
    "segf_area_frac": [
        "segf_area_frac",
        "msam_area_frac",
        "cpose_area_frac",
        "mi_lag0_area_frac",
        "mi_lag0_mi_mask_area_frac",
        "mi_large_area_frac",
        "mi_large_mi_mask_area_frac",
    ],
    "segf_equivalent_diameter_norm": [
        "segf_equivalent_diameter_norm",
        "msam_equivalent_diameter_norm",
        "cpose_equivalent_diameter_norm",
    ],
    "segf_reliability": [
        "segf_reliability",
        "segf_sam_score",
        "msam_reliability",
        "msam_sam_score",
        "cpose_reliability",
        "cpose_score",
    ],
    "segf_contact_pressure": [
        "segf_contact_pressure",
        "segf_neighbor_near_mask_frac",
        "msam_contact_pressure",
        "msam_neighbor_near_mask_frac",
        "cpose_contact_pressure",
        "cpose_neighbor_near_mask_frac",
        "mi_lag0_contact_boundary_frac",
        "mi_lag0_mi_contact_boundary_frac",
        "mi_large_contact_boundary_frac",
        "mi_large_mi_contact_boundary_frac",
    ],
    "segf_free_front_frac": [
        "segf_free_front_frac",
        "msam_free_front_frac",
        "cpose_free_front_frac",
        "mi_lag0_free_front_frac",
        "mi_large_free_front_frac",
    ],
    "segf_free_back_frac": [
        "segf_free_back_frac",
        "msam_free_back_frac",
        "cpose_free_back_frac",
        "mi_lag0_free_back_frac",
        "mi_large_free_back_frac",
    ],
    "segf_free_left_frac": [
        "segf_free_left_frac",
        "msam_free_left_frac",
        "cpose_free_left_frac",
        "mi_lag0_free_left_frac",
        "mi_large_free_left_frac",
    ],
    "segf_free_right_frac": [
        "segf_free_right_frac",
        "msam_free_right_frac",
        "cpose_free_right_frac",
        "mi_lag0_free_right_frac",
        "mi_large_free_right_frac",
    ],
    "segf_boundary_grad": [
        "segf_boundary_grad",
        "msam_boundary_grad",
        "cpose_boundary_grad",
        "mi_lag0_mi_mask_boundary_grad",
        "mi_large_mi_mask_boundary_grad",
    ],
    "segf_front_back_balance": [
        "segf_front_back_balance",
        "msam_front_back_balance",
        "cpose_front_back_balance",
        "mi_lag0_front_back_balance",
        "mi_large_front_back_balance",
    ],
    "segf_polarity_balance_norm": [
        "segf_polarity_balance_norm",
        "msam_polarity_balance_norm",
        "cpose_polarity_balance_norm",
        "segf_front_back_balance",
        "msam_front_back_balance",
        "cpose_front_back_balance",
        "mi_lag0_front_back_balance",
        "mi_large_front_back_balance",
    ],
    "segf_center_inside": [
        "segf_center_inside",
        "msam_center_inside",
        "cpose_center_inside",
        "mi_lag0_mi_mask_center_inside",
        "mi_large_mi_mask_center_inside",
    ],
}


@dataclass
class SplitEvidence:
    raw: np.ndarray
    names: list[str]
    rule_risk: np.ndarray
    hard_violation: np.ndarray
    coverage: np.ndarray


def parse_csv(text: str | list[str]) -> list[str]:
    if isinstance(text, list):
        return [str(x) for x in text]
    if str(text).strip().lower() == "all":
        return ["all"]
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return [int(float(s)) for s in parse_csv(text)]


def parse_floats(text: str | list[float]) -> list[float]:
    if isinstance(text, list):
        return [float(x) for x in text]
    return [float(s) for s in parse_csv(text)]


def safe(x: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def zrow(x: np.ndarray) -> np.ndarray:
    x = safe(x)
    return ((x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)).astype(np.float32)


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return v42.softmax_np(safe(x), axis=axis)


def flat_to_steps(x: np.ndarray, max_horizon: int) -> np.ndarray:
    return safe(x).reshape(len(x), int(max_horizon), 2)


def route_actual_steps(route_residual: np.ndarray, base: np.ndarray, max_horizon: int) -> np.ndarray:
    """Route residual predictions -> absolute per-step displacement predictions."""
    n, k, d = route_residual.shape
    residual = route_residual.reshape(n, k, int(max_horizon), 2)
    return (base[:, None, None, :] + residual).astype(np.float32)


def route_cumulative(route_residual: np.ndarray, base: np.ndarray, max_horizon: int) -> np.ndarray:
    return np.cumsum(route_actual_steps(route_residual, base, max_horizon), axis=2).astype(np.float32)


def endpoint_rows(label: str, residual_flat: np.ndarray, basis: v26.RouteBasis, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=basis.arrays.steps_test,
        base=basis.arrays.base_test,
        residual_pred=flat_to_steps(residual_flat, args.max_horizon),
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def route_mix(route: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(safe(route) * safe(weights)[:, :, None], axis=1).astype(np.float32)


def endpoint_error_matrix(route: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n, k, _ = route.shape
    pred = route.reshape(n, k, args.max_horizon, 2)
    true = true_flat.reshape(n, args.max_horizon, 2)
    err = np.zeros((n, k), dtype=np.float32)
    for h in args.horizons:
        h = int(h)
        p = np.sum(pred[:, :, :h, :], axis=2)
        y = np.sum(true[:, :h, :], axis=1)[:, None, :]
        err += np.sum((p - y) ** 2, axis=-1).astype(np.float32)
    return np.sqrt(err / max(len(args.horizons), 1)).astype(np.float32)


def hmax_error_matrix(route: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    h = max(args.horizons)
    pred = route.reshape(len(route), route.shape[1], args.max_horizon, 2)
    true = true_flat.reshape(len(true_flat), args.max_horizon, 2)
    p = np.sum(pred[:, :, :h, :], axis=2)
    y = np.sum(true[:, :h, :], axis=1)[:, None, :]
    return np.sqrt(np.sum((p - y) ** 2, axis=-1)).astype(np.float32)


def topm_weights_from_logits(logits: np.ndarray, *, top_m: int, temperature: float) -> np.ndarray:
    n, k = logits.shape
    c = max(1, min(int(top_m), k))
    order = np.argsort(-logits, axis=1)[:, :c]
    masked = np.full((n, k), -1e9, dtype=np.float32)
    rows = np.arange(n)[:, None]
    masked[rows, order] = logits[rows, order] / max(float(temperature), 1e-6)
    return softmax_np(masked, axis=1)


def tune_logits(
    *,
    logits_val: np.ndarray,
    logits_test: np.ndarray,
    route_val: np.ndarray,
    route_test: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for top_m in parse_ints(args.v57_top_m_grid):
        for temp in parse_floats(args.v57_temperature_grid):
            wv = topm_weights_from_logits(logits_val, top_m=top_m, temperature=temp)
            pv = route_mix(route_val, wv)
            rmse = v16.endpoint_rmse_flat(pv, y_val, args)
            if best is None or rmse < best["val_rmse"]:
                wt = topm_weights_from_logits(logits_test, top_m=top_m, temperature=temp)
                best = {
                    "label": label,
                    "top_m": int(top_m),
                    "temperature": float(temp),
                    "val_rmse": float(rmse),
                    "weights_test": wt,
                    "pred_test": route_mix(route_test, wt),
                    "logits_test": logits_test.astype(np.float32),
                }
    assert best is not None
    return best


def load_position_table(args: argparse.Namespace) -> pd.DataFrame:
    cols = ["dataset", "sequence", "frame", "track_id", "x_px", "y_px", "dx_px", "dy_px", "QUALITY"]
    use = pd.read_csv(args.features, usecols=lambda c: c in cols)
    if "dataset" in use.columns:
        use = use[use["dataset"].astype(str).eq(str(args.dataset))].copy()
    use["sequence"] = use["sequence"].astype(int)
    use["frame"] = use["frame"].astype(int)
    use["track_id"] = use["track_id"].astype(int)
    for c in ["x_px", "y_px", "dx_px", "dy_px", "QUALITY"]:
        if c not in use.columns:
            use[c] = 0.0 if c != "QUALITY" else 1.0
        use[c] = pd.to_numeric(use[c], errors="coerce").fillna(0.0).astype(float)
    return use


def load_state_grid(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=KEY_COLS)
    state = pd.read_csv(path)
    missing = [c for c in KEY_COLS if c not in state.columns]
    if missing:
        raise RuntimeError(f"state grid misses key columns {missing}: {path}")
    for c in ["sequence", "frame", "track_id"]:
        state[c] = state[c].astype(int)
    state = state.drop_duplicates(KEY_COLS).copy()
    for canonical, aliases in STATE_ALIASES.items():
        if canonical in state.columns:
            continue
        for alias in aliases:
            if alias in state.columns:
                state[canonical] = pd.to_numeric(state[alias], errors="coerce")
                break
    if "segf_reliability" not in state.columns:
        fallback_cols = [c for c in state.columns if c.endswith("_fallback") or c.endswith("_fallback_used")]
        if fallback_cols:
            f = pd.to_numeric(state[fallback_cols[0]], errors="coerce").fillna(1.0)
            state["segf_reliability"] = np.clip(1.0 - f.to_numpy(float), 0.0, 1.0)
    return state


def load_state_grids(spec: Any) -> pd.DataFrame:
    """Load one or more state grids and coalesce them into canonical segf_* fields.

    The first path has highest priority. Later grids fill missing rows or
    missing canonical feasibility variables. Only key columns plus canonical
    explicit state variables are exposed to v57, so raw video embeddings cannot
    silently become the occupancy filter.
    """
    paths = [Path(p).expanduser() for p in parse_csv(str(spec))]
    frames: list[pd.DataFrame] = []
    for priority, path in enumerate(paths):
        df = load_state_grid(path)
        if df.empty:
            continue
        keep = KEY_COLS + [c for c in STATE_DEFAULTS if c in df.columns]
        cur = df[keep].copy()
        cur["_v57_state_source_priority"] = int(priority)
        frames.append(cur)
    if not frames:
        return pd.DataFrame(columns=KEY_COLS)

    out: pd.DataFrame | None = None
    for cur in frames:
        if out is None:
            out = cur
            continue
        merged = out.merge(cur, on=KEY_COLS, how="outer", suffixes=("", "__new"))
        for col in STATE_DEFAULTS:
            new_col = f"{col}__new"
            if col not in merged.columns and new_col in merged.columns:
                merged[col] = merged[new_col]
            elif new_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[new_col])
        source_new = "_v57_state_source_priority__new"
        if source_new in merged.columns:
            if "_v57_state_source_priority" not in merged.columns:
                merged["_v57_state_source_priority"] = merged[source_new]
            else:
                merged["_v57_state_source_priority"] = merged["_v57_state_source_priority"].combine_first(merged[source_new])
        drop_cols = [c for c in merged.columns if c.endswith("__new")]
        out = merged.drop(columns=drop_cols)
    assert out is not None
    return out.drop_duplicates(KEY_COLS)


def merge_state(pos: pd.DataFrame, state: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if state.empty:
        out = pos.copy()
        out["_v57_state_match"] = False
    else:
        cols = [c for c in state.columns if c not in KEY_COLS or c in KEY_COLS]
        out = pos.merge(state[cols], on=KEY_COLS, how="left", indicator="_v57_merge")
        out["_v57_state_match"] = out["_v57_merge"].eq("both")
        out = out.drop(columns=["_v57_merge"])
    stats: dict[str, float] = {"state_coverage": float(out["_v57_state_match"].mean()) if len(out) else 0.0}
    for col, default in STATE_DEFAULTS.items():
        if col not in out.columns:
            out[col] = float(default)
        med = pd.to_numeric(out[col], errors="coerce").median()
        if not math.isfinite(float(med)):
            med = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(float(med)).astype(float)
        stats[f"{col}_median"] = float(med)
    return out, stats


def split_state(split_df: pd.DataFrame, state_pos: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    cols = [
        "dataset",
        "sequence",
        "frame",
        "track_id",
        "_v57_state_match",
        "segf_sam_box_radius",
        "segf_area_frac",
        "segf_equivalent_diameter_norm",
        "segf_reliability",
        "segf_contact_pressure",
        "segf_free_front_frac",
        "segf_free_back_frac",
        "segf_free_left_frac",
        "segf_free_right_frac",
        "segf_boundary_grad",
        "segf_front_back_balance",
        "segf_polarity_balance_norm",
        "segf_center_inside",
    ]
    use = state_pos[[c for c in cols if c in state_pos.columns]].drop_duplicates(KEY_COLS)
    keys = split_df[KEY_COLS].copy()
    merged = keys.merge(use, on=KEY_COLS, how="left")
    cov = float(pd.to_numeric(merged["_v57_state_match"], errors="coerce").fillna(0).astype(bool).mean()) if len(merged) else 0.0
    for c in cols:
        if c not in merged.columns and c not in KEY_COLS:
            merged[c] = 0.0
    return merged, cov


def estimate_radius(df: pd.DataFrame, *, mode: str, median_radius: float, rng: np.random.Generator) -> np.ndarray:
    mode = str(mode)
    if mode == "zero":
        return np.zeros(len(df), dtype=np.float32)
    if mode == "median":
        return np.full(len(df), float(median_radius), dtype=np.float32)
    if mode == "direct":
        r = pd.to_numeric(df.get("segf_sam_box_radius", median_radius), errors="coerce").fillna(median_radius).to_numpy(float)
        # The SAM prompt box radius is a loose upper-bound; use a conservative
        # effective radius for collision feasibility.
        return np.clip(r * 0.55, 4.0, 80.0).astype(np.float32)
    if mode == "area":
        area = pd.to_numeric(df.get("segf_area_frac", 0.10), errors="coerce").fillna(0.10).to_numpy(float)
        r = np.sqrt(np.maximum(area, 1e-5) / math.pi) * float(median_radius) / max(math.sqrt(0.10 / math.pi), 1e-6)
        return np.clip(r, 4.0, 80.0).astype(np.float32)
    if mode == "shuffled":
        r = estimate_radius(df, mode="direct", median_radius=median_radius, rng=rng)
        return r[rng.permutation(len(r))].astype(np.float32)
    raise ValueError(f"unknown radius mode: {mode}")


def transform_cumulative(cum: np.ndarray, control: str) -> np.ndarray:
    if control in {
        "real",
        "no_occupancy",
        "zero_radius",
        "median_radius",
        "radius_shuffled",
        "wrong_cell_radius",
        "neighbor_velocity_shuffled",
        "wrong_frame_neighbors",
        "same_density_random_occupancy",
    }:
        return cum
    if control == "route_reverse":
        return -cum
    if control == "route_lateral":
        out = cum.copy()
        x = out[..., 0].copy()
        y = out[..., 1].copy()
        out[..., 0] = -y
        out[..., 1] = x
        return out
    if control == "route_roll":
        return np.roll(cum, shift=1, axis=1)
    raise ValueError(f"unknown route transform/control: {control}")


def frame_groups(full: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    return {key: g.reset_index(drop=True) for key, g in full.groupby(["sequence", "frame"], sort=False)}


def frame_keys_by_sequence(groups: dict[tuple[int, int], pd.DataFrame]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for key in groups:
        out.setdefault(int(key[0]), []).append(key)
    return out


def local_flow_for_row(nei_xy: np.ndarray, nei_v: np.ndarray, d: np.ndarray, max_neighbors: int) -> tuple[np.ndarray, float]:
    if len(nei_xy) == 0:
        return np.zeros(2, dtype=np.float32), 0.0
    dist = np.linalg.norm(d, axis=1)
    order = np.argsort(dist)[: max(1, min(int(max_neighbors), len(dist)))]
    flow = np.mean(nei_v[order], axis=0).astype(np.float32)
    density = float(len(order) / max(math.pi * (float(np.percentile(dist[order], 75)) + 1.0) ** 2, 1.0))
    return flow, density


def compute_evidence_split(
    *,
    split_df: pd.DataFrame,
    split_state_df: pd.DataFrame,
    full_groups: dict[tuple[int, int], pd.DataFrame],
    route_residual: np.ndarray,
    base: np.ndarray,
    args: argparse.Namespace,
    control: str,
    radius_mode: str,
    median_radius: float,
    seed: int,
) -> SplitEvidence:
    rng = np.random.default_rng(int(seed))
    keys_by_seq = frame_keys_by_sequence(full_groups)
    n, k, _ = route_residual.shape
    h = int(args.max_horizon)
    cum = transform_cumulative(route_cumulative(route_residual, base, h), control)
    steps = np.diff(np.concatenate([np.zeros((n, k, 1, 2), dtype=np.float32), cum], axis=2), axis=2)
    central_radius = estimate_radius(split_state_df, mode=radius_mode, median_radius=median_radius, rng=rng)
    if control == "zero_radius":
        central_radius = np.zeros_like(central_radius)
    if control == "median_radius":
        central_radius = np.full_like(central_radius, float(median_radius))
    if control == "radius_shuffled":
        central_radius = central_radius[rng.permutation(len(central_radius))]
    if control == "wrong_cell_radius":
        central_radius = central_radius[rng.permutation(len(central_radius))]

    names = [
        "overlap_max",
        "overlap_mean",
        "close_frac",
        "clearance_min",
        "clearance_mean",
        "ahead_density",
        "behind_density",
        "side_density",
        "flow_align",
        "flow_mismatch",
        "path_norm",
        "endpoint_norm",
        "step_max",
        "accel_mean",
        "turn_mean",
        "jump_excess",
        "free_support",
        "contact_pressure",
        "reliability",
        "boundary_grad",
        "state_covered",
        "neighbor_count",
        "local_density",
        "radius_px",
    ]
    feat = np.zeros((n, k, len(names)), dtype=np.float32)
    hard = np.zeros((n, k), dtype=bool)
    covered = pd.to_numeric(split_state_df.get("_v57_state_match", False), errors="coerce").fillna(0).astype(bool).to_numpy()

    for i, row in split_df.reset_index(drop=True).iterrows():
        key = (int(row["sequence"]), int(row["frame"]))
        if control == "wrong_frame_neighbors":
            candidates = [x for x in keys_by_seq.get(int(row["sequence"]), []) if x != key]
            if candidates:
                key = candidates[int(rng.integers(0, len(candidates)))]
        g = full_groups.get(key)
        center = np.array([float(row["x_px"]), float(row["y_px"])], dtype=np.float32)
        if g is None or g.empty:
            nei_xy = np.zeros((0, 2), dtype=np.float32)
            nei_v = np.zeros((0, 2), dtype=np.float32)
            nei_r = np.zeros((0,), dtype=np.float32)
        else:
            mask = g["track_id"].astype(int).to_numpy() != int(row["track_id"])
            gx = g.loc[mask, ["x_px", "y_px"]].to_numpy(np.float32)
            gv = g.loc[mask, ["dx_px", "dy_px"]].to_numpy(np.float32)
            if control == "neighbor_velocity_shuffled" and len(gv):
                gv = gv[rng.permutation(len(gv))]
            gr = pd.to_numeric(g.loc[mask, "_v57_radius_px"], errors="coerce").fillna(float(median_radius)).to_numpy(np.float32)
            d0 = gx - center[None, :]
            dist0 = np.linalg.norm(d0, axis=1)
            order = np.argsort(dist0)[: max(1, min(int(args.v57_neighbor_k), len(dist0)))]
            nei_xy = gx[order]
            nei_v = gv[order]
            nei_r = gr[order]
        rel0 = nei_xy - center[None, :] if len(nei_xy) else np.zeros((0, 2), dtype=np.float32)
        if control == "same_density_random_occupancy" and len(nei_xy):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
            perm = rng.permutation(len(rel0))
            rel0 = (rel0 @ rot.T)[perm]
            nei_xy = center[None, :] + rel0
            nei_v = nei_v[perm] if len(nei_v) else nei_v
            nei_r = nei_r[perm] if len(nei_r) else nei_r
        local_flow, local_density = local_flow_for_row(nei_xy, nei_v, rel0, max_neighbors=int(args.v57_flow_k))

        st = split_state_df.iloc[i]
        free_front = float(st.get("segf_free_front_frac", 0.186))
        free_back = float(st.get("segf_free_back_frac", 0.186))
        free_side = 0.5 * float(st.get("segf_free_left_frac", 0.186)) + 0.5 * float(st.get("segf_free_right_frac", 0.186))
        contact = float(st.get("segf_contact_pressure", 0.15))
        reliability = float(st.get("segf_reliability", 0.70))
        boundary = float(st.get("segf_boundary_grad", 0.0))
        radius_i = float(central_radius[i])

        endpoint = cum[i, :, -1, :]
        endpoint_norm = np.linalg.norm(endpoint, axis=1)
        path_norm = np.sum(np.linalg.norm(steps[i], axis=2), axis=1)
        step_norm = np.linalg.norm(steps[i], axis=2)
        step_max = np.max(step_norm, axis=1)
        accel = np.diff(steps[i], axis=1)
        accel_mean = np.mean(np.linalg.norm(accel, axis=2), axis=1) if h > 1 else np.zeros(k)
        step_a = steps[i, :, :-1, :]
        step_b = steps[i, :, 1:, :]
        denom = np.maximum(np.linalg.norm(step_a, axis=2) * np.linalg.norm(step_b, axis=2), EPS)
        turn = 1.0 - np.sum(step_a * step_b, axis=2) / denom if h > 1 else np.zeros((k, 1))
        turn_mean = np.mean(np.nan_to_num(turn, nan=0.0), axis=1)
        current_speed = float(np.linalg.norm([row.get("dx_px", 0.0), row.get("dy_px", 0.0)]))
        jump_excess = np.maximum(0.0, step_max - float(args.v57_jump_factor) * max(current_speed, 1.0))

        if len(nei_xy) == 0 or control == "no_occupancy":
            overlap_max = np.zeros(k, dtype=np.float32)
            overlap_mean = np.zeros(k, dtype=np.float32)
            close_frac = np.zeros(k, dtype=np.float32)
            clearance_min = np.full(k, 999.0, dtype=np.float32)
            clearance_mean = np.full(k, 999.0, dtype=np.float32)
            ahead_density = np.zeros(k, dtype=np.float32)
            behind_density = np.zeros(k, dtype=np.float32)
            side_density = np.zeros(k, dtype=np.float32)
        else:
            overlaps = []
            clearances = []
            close_flags = []
            for tau in range(h):
                if args.v57_occupancy_mode == "static":
                    pred_nei = nei_xy
                elif args.v57_occupancy_mode == "velocity":
                    pred_nei = nei_xy + float(tau + 1) * nei_v
                elif args.v57_occupancy_mode == "clipped_velocity":
                    nv = nei_v.copy()
                    norm = np.linalg.norm(nv, axis=1)
                    clip = np.minimum(1.0, float(args.v57_neighbor_velocity_clip) / np.maximum(norm, EPS))
                    pred_nei = nei_xy + float(tau + 1) * nv * clip[:, None]
                elif args.v57_occupancy_mode == "local_flow":
                    pred_nei = nei_xy + float(tau + 1) * local_flow[None, :]
                else:
                    raise ValueError(f"unknown occupancy mode {args.v57_occupancy_mode}")
                cand_pos = center[None, None, :] + cum[i, :, tau, :][:, None, :]
                rel = pred_nei[None, :, :] - cand_pos
                dist = np.linalg.norm(rel, axis=2)
                rsum = float(args.v57_radius_scale) * radius_i + float(args.v57_neighbor_radius_scale) * nei_r[None, :]
                margin = dist - rsum
                clearances.append(margin)
                ov = np.maximum(0.0, -margin / np.maximum(rsum, 1.0))
                overlaps.append(ov)
                close_flags.append((margin < float(args.v57_close_margin_px)).astype(np.float32))
            ov_all = np.stack(overlaps, axis=2)
            cl_all = np.stack(clearances, axis=2)
            cf_all = np.stack(close_flags, axis=2)
            overlap_max = np.max(ov_all, axis=(1, 2))
            overlap_mean = np.mean(ov_all, axis=(1, 2))
            close_frac = np.mean(cf_all, axis=(1, 2))
            clearance_min = np.min(cl_all, axis=(1, 2))
            clearance_mean = np.mean(cl_all, axis=(1, 2))

            unit = endpoint / np.maximum(endpoint_norm[:, None], EPS)
            proj = rel0[None, :, :] @ unit[:, :, None]
            proj = proj[:, :, 0]
            perp = np.sqrt(np.maximum(np.sum(rel0 * rel0, axis=1)[None, :] - proj * proj, 0.0))
            corridor = perp < max(float(args.v57_corridor_width_px), 2.0 * radius_i)
            ahead_density = np.mean((proj > 0.0) & corridor, axis=1).astype(np.float32)
            behind_density = np.mean((proj < 0.0) & corridor, axis=1).astype(np.float32)
            side_density = np.mean(~corridor, axis=1).astype(np.float32)

        flow_norm = float(np.linalg.norm(local_flow))
        flow_align = np.sum(endpoint * local_flow[None, :], axis=1) / np.maximum(endpoint_norm * flow_norm, EPS)
        flow_align = np.nan_to_num(flow_align, nan=0.0).astype(np.float32)
        flow_mismatch = np.maximum(0.0, 1.0 - flow_align)
        cur_v = np.array([float(row.get("dx_px", 0.0)), float(row.get("dy_px", 0.0))], dtype=np.float32)
        cur_norm = float(np.linalg.norm(cur_v))
        cur_align = np.sum(endpoint * cur_v[None, :], axis=1) / np.maximum(endpoint_norm * cur_norm, EPS)
        cur_align = np.nan_to_num(cur_align, nan=0.0)
        free_support = (
            np.maximum(cur_align, 0.0) * free_front
            + np.maximum(-cur_align, 0.0) * free_back
            + (1.0 - np.abs(np.clip(cur_align, -1.0, 1.0))) * free_side
        ).astype(np.float32)

        vals = {
            "overlap_max": overlap_max,
            "overlap_mean": overlap_mean,
            "close_frac": close_frac,
            "clearance_min": clearance_min,
            "clearance_mean": clearance_mean,
            "ahead_density": ahead_density,
            "behind_density": behind_density,
            "side_density": side_density,
            "flow_align": flow_align,
            "flow_mismatch": flow_mismatch,
            "path_norm": path_norm,
            "endpoint_norm": endpoint_norm,
            "step_max": step_max,
            "accel_mean": accel_mean,
            "turn_mean": turn_mean,
            "jump_excess": jump_excess,
            "free_support": free_support,
            "contact_pressure": np.full(k, contact, dtype=np.float32),
            "reliability": np.full(k, reliability, dtype=np.float32),
            "boundary_grad": np.full(k, boundary, dtype=np.float32),
            "state_covered": np.full(k, float(covered[i]), dtype=np.float32),
            "neighbor_count": np.full(k, float(len(nei_xy)), dtype=np.float32),
            "local_density": np.full(k, local_density, dtype=np.float32),
            "radius_px": np.full(k, radius_i, dtype=np.float32),
        }
        for j, name in enumerate(names):
            feat[i, :, j] = vals[name]
        hard[i] = (
            (overlap_max > float(args.v57_hard_overlap))
            | (close_frac > float(args.v57_hard_close_frac))
            | (jump_excess > float(args.v57_hard_jump_excess_px))
        )

    def col(name: str) -> np.ndarray:
        return feat[:, :, names.index(name)]

    clearance_pen = np.maximum(0.0, -col("clearance_min") / max(median_radius, 1.0))
    rule = (
        1.35 * col("overlap_max")
        + 0.75 * col("overlap_mean")
        + 0.55 * col("close_frac")
        + 0.40 * clearance_pen
        + 0.28 * col("ahead_density")
        + 0.20 * col("contact_pressure")
        + 0.08 * col("flow_mismatch")
        + 0.06 * col("turn_mean")
        + 0.018 * col("jump_excess")
        - 0.28 * col("free_support")
        - 0.08 * col("reliability")
    ).astype(np.float32)
    return SplitEvidence(
        raw=np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        names=names,
        rule_risk=np.nan_to_num(rule, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        hard_violation=hard,
        coverage=covered.astype(np.float32),
    )


def fit_risk_model(
    train_ev: SplitEvidence,
    val_ev: SplitEvidence,
    test_ev: SplitEvidence,
    train_err: np.ndarray,
    args: argparse.Namespace,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n, k, f = train_ev.raw.shape
    feature_set = str(feature_set)
    path_names = {"path_norm", "endpoint_norm", "step_max", "accel_mean", "turn_mean", "jump_excess"}
    occupancy_names = {
        "overlap_max",
        "overlap_mean",
        "close_frac",
        "clearance_min",
        "clearance_mean",
        "ahead_density",
        "behind_density",
        "side_density",
        "free_support",
        "contact_pressure",
        "boundary_grad",
        "state_covered",
        "neighbor_count",
        "local_density",
        "radius_px",
    }
    flow_names = {"flow_align", "flow_mismatch"}
    if feature_set == "full":
        idx = list(range(f))
        include_route_id = True
    elif feature_set == "full_no_route_id":
        idx = list(range(f))
        include_route_id = False
    elif feature_set == "occupancy_only":
        idx = [i for i, name in enumerate(train_ev.names) if name in occupancy_names]
        include_route_id = False
    elif feature_set == "occupancy_flow":
        idx = [i for i, name in enumerate(train_ev.names) if name in occupancy_names or name in flow_names]
        include_route_id = False
    elif feature_set == "no_path_no_route":
        idx = [i for i, name in enumerate(train_ev.names) if name not in path_names]
        include_route_id = False
    elif feature_set == "path_only":
        idx = [i for i, name in enumerate(train_ev.names) if name in path_names]
        include_route_id = True
    else:
        raise ValueError(f"unknown v57 risk feature set: {feature_set}")
    if not idx:
        raise RuntimeError(f"empty risk feature set: {feature_set}")

    xtr3 = train_ev.raw[:, :, idx]
    xva3 = val_ev.raw[:, :, idx]
    xte3 = test_ev.raw[:, :, idx]
    if include_route_id:
        route_id_train = np.tile(np.arange(k, dtype=np.float32)[None, :, None] / max(k - 1, 1), (n, 1, 1))
        route_id_val = np.tile(np.arange(k, dtype=np.float32)[None, :, None] / max(k - 1, 1), (val_ev.raw.shape[0], 1, 1))
        route_id_test = np.tile(np.arange(k, dtype=np.float32)[None, :, None] / max(k - 1, 1), (test_ev.raw.shape[0], 1, 1))
        xtr3 = np.concatenate([xtr3, route_id_train], axis=2)
        xva3 = np.concatenate([xva3, route_id_val], axis=2)
        xte3 = np.concatenate([xte3, route_id_test], axis=2)

    xtr = xtr3.reshape(n * k, xtr3.shape[2])
    ytr = train_err.reshape(n * k)
    xva = xva3.reshape(val_ev.raw.shape[0] * k, xva3.shape[2])
    xte = xte3.reshape(test_ev.raw.shape[0] * k, xte3.shape[2])

    if args.v57_risk_model == "ridge":
        sc = StandardScaler()
        ztr = sc.fit_transform(xtr)
        zva = sc.transform(xva)
        zte = sc.transform(xte)
        model = Ridge(alpha=float(args.v57_ridge_alpha))
        model.fit(ztr, ytr)
        pred_tr = model.predict(ztr).reshape(n, k).astype(np.float32)
        pred_va = model.predict(zva).reshape(val_ev.raw.shape[0], k).astype(np.float32)
        pred_te = model.predict(zte).reshape(test_ev.raw.shape[0], k).astype(np.float32)
        return pred_tr, pred_va, pred_te, {
            "risk_model": "ridge",
            "risk_alpha": float(args.v57_ridge_alpha),
            "risk_feature_set": feature_set,
            "risk_feature_dim": int(xtr.shape[1]),
            "risk_include_route_id": bool(include_route_id),
        }

    model = HistGradientBoostingRegressor(
        max_iter=int(args.v57_hgbdt_iter),
        learning_rate=float(args.v57_hgbdt_lr),
        max_leaf_nodes=int(args.v57_hgbdt_leaf_nodes),
        l2_regularization=float(args.v57_hgbdt_l2),
        random_state=int(args.seed) + 57001,
    )
    model.fit(xtr, ytr)
    return (
        model.predict(xtr).reshape(n, k).astype(np.float32),
        model.predict(xva).reshape(val_ev.raw.shape[0], k).astype(np.float32),
        model.predict(xte).reshape(test_ev.raw.shape[0], k).astype(np.float32),
        {
            "risk_model": "hgbdt",
            "hgbdt_iter": int(args.v57_hgbdt_iter),
            "hgbdt_leaf_nodes": int(args.v57_hgbdt_leaf_nodes),
            "risk_feature_set": feature_set,
            "risk_feature_dim": int(xtr.shape[1]),
            "risk_include_route_id": bool(include_route_id),
        },
    )


def combine_and_tune(
    *,
    label: str,
    prior_val: np.ndarray,
    prior_test: np.ndarray,
    risk_val: np.ndarray,
    risk_test: np.ndarray,
    hard_val: np.ndarray | None,
    hard_test: np.ndarray | None,
    basis: v26.RouteBasis,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    best: dict[str, Any] | None = None
    best_logits = None
    for alpha in parse_floats(args.v57_prior_alpha_grid):
        for beta in parse_floats(args.v57_risk_beta_grid):
            lv = float(alpha) * zrow(np.log(np.maximum(prior_val, EPS))) - float(beta) * zrow(risk_val)
            lt = float(alpha) * zrow(np.log(np.maximum(prior_test, EPS))) - float(beta) * zrow(risk_test)
            for veto_q in parse_floats(args.v57_veto_quantile_grid):
                lv2 = lv.copy()
                lt2 = lt.copy()
                if veto_q < 1.0 and hard_val is not None and hard_test is not None:
                    # hard flags always veto; additionally veto top-risk candidates
                    thr = np.quantile(risk_val, float(veto_q), axis=1, keepdims=True)
                    vmask = hard_val | (risk_val >= thr)
                    thr_t = np.quantile(risk_test, float(veto_q), axis=1, keepdims=True)
                    tmask = hard_test | (risk_test >= thr_t)
                    # Keep at least one candidate per row.
                    for mask, logits in [(vmask, lv2), (tmask, lt2)]:
                        all_bad = mask.all(axis=1)
                        if np.any(all_bad):
                            keep = np.argmin(risk_val if logits is lv2 else risk_test, axis=1)
                            mask[np.arange(len(mask)), keep] = False
                        logits[mask] = -1e9
                tuned = tune_logits(
                    logits_val=lv2,
                    logits_test=lt2,
                    route_val=basis.route_val,
                    route_test=basis.route_test,
                    y_val=basis.y_val,
                    args=args,
                    label=label,
                )
                if best is None or tuned["val_rmse"] < best["val_rmse"]:
                    best = {**tuned, "prior_alpha": float(alpha), "risk_beta": float(beta), "veto_quantile": float(veto_q)}
                    best_logits = lt2
    assert best is not None and best_logits is not None
    rows = endpoint_rows(
        label,
        best["pred_test"],
        basis,
        args,
        {
            "stage": "v57_filter",
            **extra,
            "top_m": best["top_m"],
            "temperature": best["temperature"],
            "val_rmse": best["val_rmse"],
            "prior_alpha": best["prior_alpha"],
            "risk_beta": best["risk_beta"],
            "veto_quantile": best["veto_quantile"],
        },
    )
    diag = {
        "method": label,
        "hmax": max(args.horizons),
        "hmax_rmse": float([r["rmse"] for r in rows if r["horizon"] == max(args.horizons)][0]),
        "top_m": best["top_m"],
        "temperature": best["temperature"],
        "val_rmse": best["val_rmse"],
        "prior_alpha": best["prior_alpha"],
        "risk_beta": best["risk_beta"],
        "veto_quantile": best["veto_quantile"],
        **extra,
    }
    return rows, diag, best["weights_test"]


def oracle_retention_rows(
    *,
    label: str,
    risk: np.ndarray,
    hard: np.ndarray,
    err: np.ndarray,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    oracle = np.argmin(err, axis=1)
    rows = []
    for q in parse_floats(args.v57_veto_quantile_grid):
        if q >= 1.0:
            mask = hard.copy()
        else:
            thr = np.quantile(risk, q, axis=1, keepdims=True)
            mask = hard | (risk >= thr)
        all_bad = mask.all(axis=1)
        if np.any(all_bad):
            keep = np.argmin(risk, axis=1)
            mask[np.arange(len(mask)), keep] = False
        retained = ~mask[np.arange(len(mask)), oracle]
        rows.append(
            {
                "method": label,
                "veto_quantile": float(q),
                "oracle_retention": float(np.mean(retained)),
                "candidate_veto_frac": float(np.mean(mask)),
                "rows_all_veto_before_rescue": float(np.mean(all_bad)),
                "oracle_error_mean": float(np.mean(err[np.arange(len(err)), oracle])),
                **extra,
            }
        )
    return rows


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    contract: pd.DataFrame,
    metrics: pd.DataFrame,
    diag: pd.DataFrame,
    retention: pd.DataFrame,
) -> None:
    lines = [
        "# v57 Clustered Occupancy Route Filter Decision Report",
        "",
        f"Dataset: `{args.dataset}`, seed `{args.seed}`.",
        "",
        "## Contract",
        contract.to_markdown(index=False) if not contract.empty else "No contract rows.",
        "",
        "## Best h6 Metrics",
    ]
    if not metrics.empty:
        sub = metrics[metrics["horizon"].eq(max(args.horizons))].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "control", "radius_mode", "occupancy_mode", "top_m", "temperature", "prior_alpha", "risk_beta", "veto_quantile"] if c in sub.columns]
        lines.append(sub[cols].head(40).to_markdown(index=False))
    else:
        lines.append("No metrics.")
    lines.extend(["", "## Diagnostics"])
    if not diag.empty:
        lines.append(diag.sort_values("hmax_rmse").head(40).to_markdown(index=False))
    else:
        lines.append("No diagnostics.")
    lines.extend(["", "## Oracle Retention"])
    if not retention.empty:
        lines.append(retention.head(80).to_markdown(index=False))
    else:
        lines.append("No retention rows.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- v57 passes only if real occupancy beats no/shuffled/radius/route controls and keeps oracle retention high.",
            "- A small gain with poor oracle retention means the filter is too destructive.",
            "- If real occupancy is indistinguishable from controls, the next visual cross-encoder should not be launched blindly.",
        ]
    )
    (out_dir / "v57_decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    np.random.seed(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # v26 expects dense_features as the route-basis feature table.  For v57 we
    # intentionally keep state-grid separate, so route basis uses args.features.
    args.dense_features = args.features

    device = closure.device_from_arg(args.device)
    basis = v26.build_route_basis(args, args.out_dir)
    pos = load_position_table(args)
    state = load_state_grids(args.state_grid)
    pos_state, state_stats = merge_state(pos, state)
    rng = np.random.default_rng(int(args.seed) + 5700)
    median_direct_radius = float(np.nanmedian(estimate_radius(pos_state, mode="direct", median_radius=42.0, rng=rng)))
    pos_state["_v57_radius_px"] = estimate_radius(pos_state, mode="direct", median_radius=median_direct_radius, rng=rng)
    groups = frame_groups(pos_state)

    split_state_train, cov_tr = split_state(basis.split.train, pos_state)
    split_state_val, cov_va = split_state(basis.split.val, pos_state)
    split_state_test, cov_te = split_state(basis.split.test, pos_state)

    contract = pd.DataFrame(
        [
            {"item": "features", "path": str(args.features), "exists": Path(args.features).exists()},
            {
                "item": "state_grid",
                "path": str(args.state_grid),
                "exists": all(Path(p).expanduser().exists() for p in parse_csv(str(args.state_grid))),
            },
            {"item": "state_grid_rows_after_union", "value": int(len(state))},
            {"item": "route_count", "value": int(basis.route_test.shape[1])},
            {"item": "train_rows", "value": int(len(basis.split.train))},
            {"item": "val_rows", "value": int(len(basis.split.val))},
            {"item": "test_rows", "value": int(len(basis.split.test))},
            {"item": "full_position_rows", "value": int(len(pos_state))},
            {"item": "full_state_coverage", "value": float(state_stats.get("state_coverage", 0.0))},
            {"item": "train_state_coverage", "value": cov_tr},
            {"item": "val_state_coverage", "value": cov_va},
            {"item": "test_state_coverage", "value": cov_te},
            {"item": "median_effective_radius_px", "value": median_direct_radius},
        ]
    )
    contract.to_csv(args.out_dir / "v57_data_contract.csv", index=False)
    pd.DataFrame([state_stats]).to_csv(args.out_dir / "v57_radius_calibration.csv", index=False)

    prior_tr = np.maximum(safe(basis.prior.probs_train), EPS)
    prior_va = np.maximum(safe(basis.prior.probs_val), EPS)
    prior_te = np.maximum(safe(basis.prior.probs_test), EPS)
    train_err = hmax_error_matrix(basis.route_train, basis.y_train, args)
    val_err = hmax_error_matrix(basis.route_val, basis.y_val, args)
    test_err = hmax_error_matrix(basis.route_test, basis.y_test, args)

    metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    feature_summaries: list[dict[str, Any]] = []

    prior_tuned = tune_logits(
        logits_val=np.log(prior_va),
        logits_test=np.log(prior_te),
        route_val=basis.route_val,
        route_test=basis.route_test,
        y_val=basis.y_val,
        args=args,
        label="v57_coord_prior_only",
    )
    metric_rows.extend(
        endpoint_rows(
            "v57_coord_prior_only",
            prior_tuned["pred_test"],
            basis,
            args,
            {
                "stage": "baseline",
                "control": "coord_prior_only",
                "top_m": prior_tuned["top_m"],
                "temperature": prior_tuned["temperature"],
                "val_rmse": prior_tuned["val_rmse"],
            },
        )
    )
    diag_rows.append(
        {
            "method": "v57_coord_prior_only",
            "hmax": max(args.horizons),
            "hmax_rmse": float([r["rmse"] for r in metric_rows if r["method"] == "v57_coord_prior_only" and r["horizon"] == max(args.horizons)][0]),
            "control": "coord_prior_only",
            "top_m": prior_tuned["top_m"],
            "temperature": prior_tuned["temperature"],
            "val_rmse": prior_tuned["val_rmse"],
        }
    )

    controls = parse_csv(args.v57_controls)
    if "all" in controls:
        controls = [
            "real",
            "no_occupancy",
            "zero_radius",
            "median_radius",
            "radius_shuffled",
            "wrong_cell_radius",
            "neighbor_velocity_shuffled",
            "wrong_frame_neighbors",
            "same_density_random_occupancy",
            "route_reverse",
            "route_lateral",
            "route_roll",
        ]
    radius_modes = parse_csv(args.radius_modes)
    occupancy_modes = parse_csv(args.occupancy_modes)

    for occupancy_mode in occupancy_modes:
        args.v57_occupancy_mode = occupancy_mode
        for radius_mode in radius_modes:
            for control in controls:
                if control == "zero_radius":
                    rm = "zero"
                elif control == "median_radius":
                    rm = "median"
                elif control == "radius_shuffled":
                    rm = "shuffled"
                else:
                    rm = radius_mode
                label_base = f"v57_{control}_{occupancy_mode}_{rm}"
                train_ev = compute_evidence_split(
                    split_df=basis.split.train,
                    split_state_df=split_state_train,
                    full_groups=groups,
                    route_residual=basis.route_train,
                    base=basis.arrays.base_train,
                    args=args,
                    control=control,
                    radius_mode=rm,
                    median_radius=median_direct_radius,
                    seed=args.seed + 5711,
                )
                val_ev = compute_evidence_split(
                    split_df=basis.split.val,
                    split_state_df=split_state_val,
                    full_groups=groups,
                    route_residual=basis.route_val,
                    base=basis.arrays.base_val,
                    args=args,
                    control=control,
                    radius_mode=rm,
                    median_radius=median_direct_radius,
                    seed=args.seed + 5712,
                )
                test_ev = compute_evidence_split(
                    split_df=basis.split.test,
                    split_state_df=split_state_test,
                    full_groups=groups,
                    route_residual=basis.route_test,
                    base=basis.arrays.base_test,
                    args=args,
                    control=control,
                    radius_mode=rm,
                    median_radius=median_direct_radius,
                    seed=args.seed + 5713,
                )

                pd.DataFrame(
                    [
                        {
                            "control": control,
                            "radius_mode": rm,
                            "occupancy_mode": occupancy_mode,
                            "split": "test",
                            "risk_mean": float(np.mean(test_ev.rule_risk)),
                            "risk_std": float(np.std(test_ev.rule_risk)),
                            "hard_violation_frac": float(np.mean(test_ev.hard_violation)),
                            "coverage": float(np.mean(test_ev.coverage)),
                            "feature_dim": int(test_ev.raw.shape[2]),
                        }
                    ]
                ).to_csv(args.out_dir / f"v57_feature_summary_{label_base}.csv", index=False)
                feature_summaries.append(
                    {
                        "control": control,
                        "radius_mode": rm,
                        "occupancy_mode": occupancy_mode,
                        "risk_mean": float(np.mean(test_ev.rule_risk)),
                        "hard_violation_frac": float(np.mean(test_ev.hard_violation)),
                        "coverage": float(np.mean(test_ev.coverage)),
                    }
                )

                extra = {"control": control, "radius_mode": rm, "occupancy_mode": occupancy_mode, "risk_source": "rule"}
                rows, diag, weights = combine_and_tune(
                    label=f"{label_base}_rule",
                    prior_val=prior_va,
                    prior_test=prior_te,
                    risk_val=val_ev.rule_risk,
                    risk_test=test_ev.rule_risk,
                    hard_val=val_ev.hard_violation,
                    hard_test=test_ev.hard_violation,
                    basis=basis,
                    args=args,
                    extra=extra,
                )
                metric_rows.extend(rows)
                diag_rows.append(diag)
                retention_rows.extend(
                    oracle_retention_rows(
                        label=f"{label_base}_rule",
                        risk=test_ev.rule_risk,
                        hard=test_ev.hard_violation,
                        err=test_err,
                        args=args,
                        extra=extra,
                    )
                )

                if control in set(parse_csv(args.v57_learned_controls)):
                    for feature_set in parse_csv(args.v57_risk_feature_sets):
                        risk_tr, risk_va, risk_te, risk_meta = fit_risk_model(
                            train_ev,
                            val_ev,
                            test_ev,
                            train_err,
                            args,
                            feature_set=feature_set,
                        )
                        extra2 = {**extra, "risk_source": "learned", **risk_meta}
                        rows, diag, _weights = combine_and_tune(
                            label=f"{label_base}_{risk_meta['risk_model']}_{feature_set}",
                            prior_val=prior_va,
                            prior_test=prior_te,
                            risk_val=risk_va,
                            risk_test=risk_te,
                            hard_val=val_ev.hard_violation,
                            hard_test=test_ev.hard_violation,
                            basis=basis,
                            args=args,
                            extra=extra2,
                        )
                        metric_rows.extend(rows)
                        diag_rows.append(diag)
                        retention_rows.extend(
                            oracle_retention_rows(
                                label=f"{label_base}_{risk_meta['risk_model']}_{feature_set}",
                                risk=risk_te,
                                hard=test_ev.hard_violation,
                                err=test_err,
                                args=args,
                                extra=extra2,
                            )
                        )

                # Save one compact feature sample for the real branch.
                if control == "real" and occupancy_mode == occupancy_modes[0] and rm == radius_modes[0]:
                    flat = test_ev.raw.reshape(-1, test_ev.raw.shape[2])
                    sample = pd.DataFrame(flat[: min(len(flat), 5000)], columns=test_ev.names)
                    sample.to_csv(args.out_dir / "v57_candidate_violation_features.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    diag = pd.DataFrame(diag_rows)
    retention = pd.DataFrame(retention_rows)
    feature_summary = pd.DataFrame(feature_summaries)
    metrics.to_csv(args.out_dir / "v57_filter_summary.csv", index=False)
    diag.to_csv(args.out_dir / "v57_controls.csv", index=False)
    retention.to_csv(args.out_dir / "v57_oracle_retention.csv", index=False)
    feature_summary.to_csv(args.out_dir / "v57_neighbor_occupancy_audit.csv", index=False)

    # Route/cluster feasibility table at route-expert cluster level.
    if not diag.empty:
        diag.sort_values("hmax_rmse").to_csv(args.out_dir / "v57_cluster_feasibility.csv", index=False)

    (args.out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, contract, metrics, diag, retention)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "metric_rows": int(len(metrics)),
                "diagnostic_rows": int(len(diag)),
                "retention_rows": int(len(retention)),
                "elapsed_sec": float(time.time() - t0),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=audit.DEFAULT_FEATURES)
    ap.add_argument("--state-grid", type=Path, default=DEFAULT_STATE_GRID)
    ap.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    ap.add_argument("--dataset", default="MDCK_Bulk")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seq", default="1,2,3,4")
    ap.add_argument("--val-seq", default="5")
    ap.add_argument("--test-seq", default="6")
    ap.add_argument("--horizons", default="1,2,4,6")
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--max-features-per-family", type=int, default=160)
    ap.add_argument("--max-all-features", type=int, default=384)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")

    # v26/v16-compatible route-basis settings.
    ap.add_argument("--dense-features", type=Path, default=audit.DEFAULT_FEATURES)
    ap.add_argument("--dense-max-cols", type=int, default=192)
    ap.add_argument("--generator-max-train-rows", type=int, default=3000)
    ap.add_argument("--generator-max-val-rows", type=int, default=1000)
    ap.add_argument("--generator-max-test-rows", type=int, default=1500)
    ap.add_argument("--generator-posterior-epochs", type=int, default=8)
    ap.add_argument("--generator-student-epochs", type=int, default=8)
    ap.add_argument("--generator-learned-route-epochs", type=int, default=6)
    ap.add_argument("--generator-candidate-k", type=int, default=32)
    ap.add_argument("--generator-oracle-k", default="8,16,32")
    ap.add_argument("--generator-variant", default="context_velocity")
    ap.add_argument("--generator-prior-model", default="logistic", choices=["logistic", "hgbdt"])
    ap.add_argument("--generator-base-mixes", default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    ap.add_argument("--generator-calibrators", default="correction_context,stacked_context")
    ap.add_argument("--generator-max-context-features", type=int, default=384)
    ap.add_argument("--v25-route-k", type=int, default=12)
    ap.add_argument("--v25-velocity-max-cols", type=int, default=160)
    ap.add_argument("--v16c-generator-variant", default="context_velocity")
    ap.add_argument("--v16c-top-c", type=int, default=8)
    ap.add_argument("--v16c-max-context-features", type=int, default=384)
    ap.add_argument("--v16c-ridge-alphas", default="0.1,0.3,1,3,10,30,100,300,1000,3000")

    # v57 feasibility settings.
    ap.add_argument("--radius-modes", default="direct,median,shuffled")
    ap.add_argument("--occupancy-modes", default="static,velocity,clipped_velocity,local_flow")
    ap.add_argument("--v57-controls", default="all")
    ap.add_argument(
        "--v57-learned-controls",
        default="real,no_occupancy,radius_shuffled,wrong_cell_radius,neighbor_velocity_shuffled,wrong_frame_neighbors,same_density_random_occupancy,route_reverse,route_lateral",
    )
    ap.add_argument("--v57-neighbor-k", type=int, default=32)
    ap.add_argument("--v57-flow-k", type=int, default=16)
    ap.add_argument("--v57-radius-scale", type=float, default=0.85)
    ap.add_argument("--v57-neighbor-radius-scale", type=float, default=0.85)
    ap.add_argument("--v57-close-margin-px", type=float, default=12.0)
    ap.add_argument("--v57-corridor-width-px", type=float, default=96.0)
    ap.add_argument("--v57-neighbor-velocity-clip", type=float, default=18.0)
    ap.add_argument("--v57-jump-factor", type=float, default=3.0)
    ap.add_argument("--v57-hard-overlap", type=float, default=0.55)
    ap.add_argument("--v57-hard-close-frac", type=float, default=0.45)
    ap.add_argument("--v57-hard-jump-excess-px", type=float, default=24.0)
    ap.add_argument("--v57-top-m-grid", default="1,2,4,8,12")
    ap.add_argument("--v57-temperature-grid", default="0.35,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v57-prior-alpha-grid", default="0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v57-risk-beta-grid", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v57-veto-quantile-grid", default="1.0,0.95,0.90,0.80")
    ap.add_argument("--v57-risk-model", default="hgbdt", choices=["hgbdt", "ridge"])
    ap.add_argument("--v57-risk-feature-sets", default="full")
    ap.add_argument("--v57-hgbdt-iter", type=int, default=180)
    ap.add_argument("--v57-hgbdt-lr", type=float, default=0.045)
    ap.add_argument("--v57-hgbdt-leaf-nodes", type=int, default=31)
    ap.add_argument("--v57-hgbdt-l2", type=float, default=0.02)
    ap.add_argument("--v57-ridge-alpha", type=float, default=100.0)
    args = ap.parse_args()

    if args.smoke:
        args.generator_max_train_rows = min(args.generator_max_train_rows, 900)
        args.generator_max_val_rows = min(args.generator_max_val_rows, 300)
        args.generator_max_test_rows = min(args.generator_max_test_rows, 400)
        args.generator_posterior_epochs = min(args.generator_posterior_epochs, 3)
        args.generator_student_epochs = min(args.generator_student_epochs, 3)
        args.generator_learned_route_epochs = min(args.generator_learned_route_epochs, 3)
        args.generator_candidate_k = min(args.generator_candidate_k, 16)
        args.generator_oracle_k = "4,8,16"
        args.radius_modes = "direct,median"
        args.occupancy_modes = "static,velocity"
        args.v57_controls = "real,no_occupancy,radius_shuffled,wrong_frame_neighbors,same_density_random_occupancy,route_reverse,route_lateral"
        args.v57_learned_controls = "real,no_occupancy,same_density_random_occupancy"
        args.v57_risk_feature_sets = "full,occupancy_only"
        args.v57_top_m_grid = "1,2,4,8"
        args.v57_temperature_grid = "0.5,1.0,1.5"
        args.v57_prior_alpha_grid = "0.75,1.0,1.5"
        args.v57_risk_beta_grid = "0.0,0.5,1.0,2.0"
        args.v57_veto_quantile_grid = "1.0,0.9"
        args.v57_hgbdt_iter = min(args.v57_hgbdt_iter, 80)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
