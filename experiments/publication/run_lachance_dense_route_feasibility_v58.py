#!/usr/bin/env python3
"""v58 dense route-feasibility field.

This runner closes the stronger route feasibility hypothesis:

    fixed coordinate route clusters
    + dense frame-level occupancy field from all tracked cells
    + mask/state radius where available, adaptive radius elsewhere
    + swept corridor along every candidate trajectory
    -> cluster-level feasibility/risk
    -> sparse coordinate+feasibility route mixture

Compared with v57, this is not a sparse per-row state feature injection.  The
field is built for every tracked cell in every frame, so candidate routes can be
checked against the actual local tissue occupancy even when the central cell has
no extracted visual-state row.  Target/future is used only for supervised risk
labels and diagnostics, never for inference features.
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
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_clustered_occupancy_route_filter_v57 as v57  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "dense_route_feasibility_v58_2026-07-13"
DEFAULT_STATE_GRID = (
    ROOT / "outputs" / "temporal_identity_seg_state_v40_bulk_seed42_2026-07-08" / "temporal_identity_seg_state_v40.csv"
)
KEY_COLS = ["dataset", "sequence", "frame", "track_id"]
EPS = 1e-8


@dataclass
class DenseField:
    table: pd.DataFrame
    groups: dict[tuple[int, int], pd.DataFrame]
    stats: dict[str, float]


def parse_csv(text: str | list[str]) -> list[str]:
    return v57.parse_csv(text)


def parse_ints(text: str | list[int]) -> list[int]:
    return v57.parse_ints(text)


def parse_floats(text: str | list[float]) -> list[float]:
    return v57.parse_floats(text)


def safe(x: Any) -> np.ndarray:
    return v57.safe(x)


def route_actual_steps(route_residual: np.ndarray, base: np.ndarray, max_horizon: int) -> np.ndarray:
    return v57.route_actual_steps(route_residual, base, max_horizon)


def route_cumulative(route_residual: np.ndarray, base: np.ndarray, max_horizon: int) -> np.ndarray:
    return v57.route_cumulative(route_residual, base, max_horizon)


def zrow(x: np.ndarray) -> np.ndarray:
    return v57.zrow(x)


def topm_weights_from_logits(logits: np.ndarray, *, top_m: int, temperature: float) -> np.ndarray:
    return v57.topm_weights_from_logits(logits, top_m=top_m, temperature=temperature)


def endpoint_rows(label: str, residual_flat: np.ndarray, basis: v26.RouteBasis, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return v57.endpoint_rows(label, residual_flat, basis, args, extra)


def hmax_error_matrix(route: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return v57.hmax_error_matrix(route, true_flat, args)


def route_mix(route: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return v57.route_mix(route, weights)


def local_nn_distance(xy: np.ndarray) -> np.ndarray:
    if len(xy) <= 1:
        return np.full(len(xy), np.nan, dtype=np.float32)
    d = xy[:, None, :] - xy[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1).astype(np.float32)


def build_dense_field(pos: pd.DataFrame, state: pd.DataFrame, args: argparse.Namespace) -> DenseField:
    pos_state, state_stats = v57.merge_state(pos, state)
    rng = np.random.default_rng(int(args.seed) + 5801)
    mask_radius = v57.estimate_radius(pos_state, mode="direct", median_radius=float(args.v58_default_radius_px), rng=rng)
    pos_state["_v58_mask_radius_px"] = mask_radius

    nn_dist = np.zeros(len(pos_state), dtype=np.float32)
    source = np.full(len(pos_state), "adaptive", dtype=object)
    radius = np.zeros(len(pos_state), dtype=np.float32)
    matched = pos_state["_v57_state_match"].astype(bool).to_numpy()
    offset = 0
    frames = []
    for _, g in pos_state.groupby(["sequence", "frame"], sort=False):
        idx = g.index.to_numpy()
        xy = g[["x_px", "y_px"]].to_numpy(np.float32)
        nn = local_nn_distance(xy)
        nn_dist[idx] = nn
        # Adaptive radius: robustly bounded by local spacing.  This is not a
        # visual mask, but it turns sparse mask coverage into a dense physical
        # occupancy field instead of leaving most rows unmodelled.
        adaptive = np.minimum(
            float(args.v58_default_radius_px),
            float(args.v58_nn_radius_factor) * np.nan_to_num(nn, nan=float(args.v58_default_radius_px) * 2.0),
        )
        adaptive = np.clip(adaptive, float(args.v58_radius_min_px), float(args.v58_radius_max_px))
        r = adaptive.astype(np.float32)
        local_matched = matched[idx]
        r[local_matched] = np.clip(mask_radius[idx][local_matched], float(args.v58_radius_min_px), float(args.v58_radius_max_px))
        radius[idx] = r
        source[idx[local_matched]] = "mask_state"
        frames.append(len(g))
        offset += len(g)
    pos_state["_v58_nn_dist_px"] = nn_dist
    pos_state["_v58_radius_px"] = radius
    pos_state["_v58_radius_source"] = source
    # Reliability is explicit when available; otherwise lower but non-zero for
    # adaptive tracking-derived radius.
    reliability = pd.to_numeric(pos_state.get("segf_reliability", 0.70), errors="coerce").fillna(0.70).to_numpy(float)
    reliability = np.where(matched, reliability, float(args.v58_adaptive_reliability))
    pos_state["_v58_reliability"] = np.clip(reliability, 0.0, 1.0).astype(np.float32)

    groups = {key: g.reset_index(drop=True) for key, g in pos_state.groupby(["sequence", "frame"], sort=False)}
    stats = {
        **state_stats,
        "dense_rows": float(len(pos_state)),
        "dense_frames": float(len(groups)),
        "mask_state_coverage": float(np.mean(matched)) if len(matched) else 0.0,
        "radius_mean": float(np.mean(radius)) if len(radius) else 0.0,
        "radius_median": float(np.median(radius)) if len(radius) else 0.0,
        "radius_mask_state_mean": float(np.mean(radius[matched])) if np.any(matched) else 0.0,
        "radius_adaptive_mean": float(np.mean(radius[~matched])) if np.any(~matched) else 0.0,
        "nn_dist_median": float(np.nanmedian(nn_dist)) if len(nn_dist) else 0.0,
    }
    return DenseField(pos_state, groups, stats)


def frame_keys_by_sequence(groups: dict[tuple[int, int], pd.DataFrame]) -> dict[int, list[tuple[int, int]]]:
    return v57.frame_keys_by_sequence(groups)


def transformed_path(cum: np.ndarray, control: str) -> np.ndarray:
    if control in {
        "real",
        "no_field",
        "zero_radius",
        "median_radius",
        "radius_shuffled",
        "wrong_cell_radius",
        "neighbor_velocity_shuffled",
        "wrong_frame_field",
        "same_density_random_field",
    }:
        return cum
    return v57.transform_cumulative(cum, control)


def randomize_same_density(
    *,
    rel: np.ndarray,
    vel: np.ndarray,
    rad: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(rel) == 0:
        return rel, vel, rad
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    perm = rng.permutation(len(rel))
    return (rel @ rot.T)[perm].astype(np.float32), vel[perm].astype(np.float32), rad[perm].astype(np.float32)


def candidate_sample_points(cum_i: np.ndarray, steps_i: np.ndarray, samples_per_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate sample positions, fractional times and segment ids.

    cum_i shape: k,h,2; steps_i shape: k,h,2.
    """
    k, h, _ = cum_i.shape
    prev = np.concatenate([np.zeros((k, 1, 2), dtype=np.float32), cum_i[:, :-1, :]], axis=1)
    ts = []
    pts = []
    seg = []
    for tau in range(h):
        for s in range(int(samples_per_step)):
            frac = float(s + 1) / float(samples_per_step)
            pts.append(prev[:, tau, :] + frac * steps_i[:, tau, :])
            ts.append(np.full(k, float(tau) + frac, dtype=np.float32))
            seg.append(np.full(k, tau, dtype=np.int32))
    return np.stack(pts, axis=1).astype(np.float32), np.stack(ts, axis=1).astype(np.float32), np.stack(seg, axis=1)


def compute_dense_evidence_split(
    *,
    split_df: pd.DataFrame,
    split_state_df: pd.DataFrame,
    field: DenseField,
    route_residual: np.ndarray,
    base: np.ndarray,
    args: argparse.Namespace,
    control: str,
    occupancy_mode: str,
    seed: int,
) -> v57.SplitEvidence:
    rng = np.random.default_rng(int(seed))
    keys_by_seq = frame_keys_by_sequence(field.groups)
    n, k, _ = route_residual.shape
    h = int(args.max_horizon)
    cum = transformed_path(route_cumulative(route_residual, base, h), control)
    steps = np.diff(np.concatenate([np.zeros((n, k, 1, 2), dtype=np.float32), cum], axis=2), axis=2)

    names = [
        "overlap_max",
        "overlap_mean",
        "close_frac",
        "clearance_min",
        "clearance_mean",
        "corridor_collision_frac",
        "corridor_free_frac",
        "corridor_pressure_mean",
        "corridor_pressure_max",
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
        "radius_source_mask_frac",
        "nn_clearance_ratio",
        "swept_min_margin",
        "swept_mean_margin",
        "swept_overlap_auc",
        "same_track_jump_guard",
    ]
    feat = np.zeros((n, k, len(names)), dtype=np.float32)
    hard = np.zeros((n, k), dtype=bool)
    covered = pd.to_numeric(split_state_df.get("_v57_state_match", False), errors="coerce").fillna(0).astype(bool).to_numpy()

    for i, row in split_df.reset_index(drop=True).iterrows():
        key = (int(row["sequence"]), int(row["frame"]))
        if control == "wrong_frame_field":
            candidates = [x for x in keys_by_seq.get(int(row["sequence"]), []) if x != key]
            if candidates:
                key = candidates[int(rng.integers(0, len(candidates)))]
        g = field.groups.get(key)
        center = np.array([float(row["x_px"]), float(row["y_px"])], dtype=np.float32)
        cur_v = np.array([float(row.get("dx_px", 0.0)), float(row.get("dy_px", 0.0))], dtype=np.float32)
        if g is None or g.empty:
            nei_xy = np.zeros((0, 2), dtype=np.float32)
            nei_v = np.zeros((0, 2), dtype=np.float32)
            nei_r = np.zeros((0,), dtype=np.float32)
            nei_rel = np.zeros((0, 2), dtype=np.float32)
        else:
            mask = g["track_id"].astype(int).to_numpy() != int(row["track_id"])
            gx = g.loc[mask, ["x_px", "y_px"]].to_numpy(np.float32)
            gv = g.loc[mask, ["dx_px", "dy_px"]].to_numpy(np.float32)
            gr = pd.to_numeric(g.loc[mask, "_v58_radius_px"], errors="coerce").fillna(float(args.v58_default_radius_px)).to_numpy(np.float32)
            if control == "neighbor_velocity_shuffled" and len(gv):
                gv = gv[rng.permutation(len(gv))]
            rel_all = gx - center[None, :]
            dist_all = np.linalg.norm(rel_all, axis=1)
            order = np.argsort(dist_all)[: max(1, min(int(args.v58_neighbor_k), len(dist_all)))]
            nei_xy = gx[order]
            nei_v = gv[order]
            nei_r = gr[order]
            nei_rel = rel_all[order]
        if control == "same_density_random_field" and len(nei_xy):
            nei_rel, nei_v, nei_r = randomize_same_density(rel=nei_rel, vel=nei_v, rad=nei_r, rng=rng)
            nei_xy = center[None, :] + nei_rel

        if control in {"zero_radius", "no_field"}:
            nei_r = np.zeros_like(nei_r)
        elif control == "median_radius":
            nei_r = np.full_like(nei_r, float(args.v58_default_radius_px))
        elif control in {"radius_shuffled", "wrong_cell_radius"} and len(nei_r):
            nei_r = nei_r[rng.permutation(len(nei_r))]

        st = split_state_df.iloc[i]
        central_radius = float(st.get("_v58_radius_px", st.get("segf_sam_box_radius", args.v58_default_radius_px)))
        if not math.isfinite(central_radius) or central_radius <= 0:
            central_radius = float(args.v58_default_radius_px)
        central_radius = float(np.clip(central_radius * float(args.v58_central_radius_scale), args.v58_radius_min_px, args.v58_radius_max_px))
        if control == "zero_radius":
            central_radius = 0.0
        elif control == "median_radius":
            central_radius = float(args.v58_default_radius_px)
        reliability = float(st.get("segf_reliability", args.v58_adaptive_reliability))
        if not bool(covered[i]):
            reliability = min(reliability, float(args.v58_adaptive_reliability))
        free_front = float(st.get("segf_free_front_frac", 0.186))
        free_back = float(st.get("segf_free_back_frac", 0.186))
        free_side = 0.5 * float(st.get("segf_free_left_frac", 0.186)) + 0.5 * float(st.get("segf_free_right_frac", 0.186))
        contact = float(st.get("segf_contact_pressure", 0.15))
        boundary = float(st.get("segf_boundary_grad", 0.0))

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
        current_speed = float(np.linalg.norm(cur_v))
        jump_excess = np.maximum(0.0, step_max - float(args.v58_jump_factor) * max(current_speed, 1.0))

        if len(nei_xy) == 0 or control == "no_field":
            overlap_max = np.zeros(k, dtype=np.float32)
            overlap_mean = np.zeros(k, dtype=np.float32)
            close_frac = np.zeros(k, dtype=np.float32)
            clearance_min = np.full(k, 999.0, dtype=np.float32)
            clearance_mean = np.full(k, 999.0, dtype=np.float32)
            corridor_collision_frac = np.zeros(k, dtype=np.float32)
            corridor_free_frac = np.ones(k, dtype=np.float32)
            corridor_pressure_mean = np.zeros(k, dtype=np.float32)
            corridor_pressure_max = np.zeros(k, dtype=np.float32)
            swept_min_margin = np.full(k, 999.0, dtype=np.float32)
            swept_mean_margin = np.full(k, 999.0, dtype=np.float32)
            swept_overlap_auc = np.zeros(k, dtype=np.float32)
            ahead_density = np.zeros(k, dtype=np.float32)
            behind_density = np.zeros(k, dtype=np.float32)
            side_density = np.zeros(k, dtype=np.float32)
            local_flow = np.zeros(2, dtype=np.float32)
            local_density = 0.0
            radius_source_mask_frac = 0.0
            nn_clearance_ratio = np.ones(k, dtype=np.float32)
        else:
            sample_pts, sample_t, _seg = candidate_sample_points(cum[i], steps[i], int(args.v58_samples_per_step))
            s_count = sample_pts.shape[1]
            local_flow, local_density = v57.local_flow_for_row(nei_xy, nei_v, nei_rel, max_neighbors=int(args.v58_flow_k))
            if occupancy_mode == "static":
                pred_nei_base = np.repeat(nei_xy[None, :, :], s_count, axis=0)
            elif occupancy_mode == "velocity":
                pred_nei_base = nei_xy[None, :, :] + sample_t[0, :, None, None] * nei_v[None, :, :]
            elif occupancy_mode == "clipped_velocity":
                nv = nei_v.copy()
                norm = np.linalg.norm(nv, axis=1)
                clip = np.minimum(1.0, float(args.v58_neighbor_velocity_clip) / np.maximum(norm, EPS))
                pred_nei_base = nei_xy[None, :, :] + sample_t[0, :, None, None] * (nv * clip[:, None])[None, :, :]
            elif occupancy_mode == "local_flow":
                pred_nei_base = nei_xy[None, :, :] + sample_t[0, :, None, None] * local_flow[None, None, :]
            else:
                raise ValueError(f"unknown occupancy mode {occupancy_mode}")

            margins = []
            overlaps = []
            close = []
            pressure = []
            for s in range(s_count):
                cand_pos = center[None, None, :] + sample_pts[:, s, :][:, None, :]
                rel = pred_nei_base[s][None, :, :] - cand_pos
                dist = np.linalg.norm(rel, axis=2)
                rsum = float(args.v58_radius_scale) * central_radius + float(args.v58_neighbor_radius_scale) * nei_r[None, :]
                margin = dist - rsum
                ov = np.maximum(0.0, -margin / np.maximum(rsum, 1.0))
                margins.append(margin)
                overlaps.append(ov)
                close.append((margin < float(args.v58_close_margin_px)).astype(np.float32))
                pressure.append(np.exp(-np.maximum(margin, -60.0) / max(float(args.v58_pressure_scale_px), 1.0)))
            margins_all = np.stack(margins, axis=2)
            overlaps_all = np.stack(overlaps, axis=2)
            close_all = np.stack(close, axis=2)
            pressure_all = np.stack(pressure, axis=2)
            overlap_max = np.max(overlaps_all, axis=(1, 2))
            overlap_mean = np.mean(overlaps_all, axis=(1, 2))
            close_frac = np.mean(close_all, axis=(1, 2))
            clearance_min = np.min(margins_all, axis=(1, 2))
            clearance_mean = np.mean(margins_all, axis=(1, 2))
            swept_min_margin = clearance_min.copy()
            swept_mean_margin = clearance_mean.copy()
            swept_overlap_auc = np.mean(overlaps_all, axis=(1, 2))
            corridor_collision_frac = np.mean(margins_all < 0.0, axis=(1, 2)).astype(np.float32)
            corridor_free_frac = np.mean(margins_all > float(args.v58_free_margin_px), axis=(1, 2)).astype(np.float32)
            corridor_pressure_mean = np.mean(pressure_all, axis=(1, 2)).astype(np.float32)
            corridor_pressure_max = np.max(pressure_all, axis=(1, 2)).astype(np.float32)

            unit = endpoint / np.maximum(endpoint_norm[:, None], EPS)
            proj = nei_rel[None, :, :] @ unit[:, :, None]
            proj = proj[:, :, 0]
            perp = np.sqrt(np.maximum(np.sum(nei_rel * nei_rel, axis=1)[None, :] - proj * proj, 0.0))
            corridor = perp < np.maximum(float(args.v58_corridor_width_px), 2.0 * central_radius)
            ahead_density = np.mean((proj > 0.0) & corridor, axis=1).astype(np.float32)
            behind_density = np.mean((proj < 0.0) & corridor, axis=1).astype(np.float32)
            side_density = np.mean(~corridor, axis=1).astype(np.float32)
            mask_source = field.groups.get(key)
            if mask_source is not None and "_v57_state_match" in mask_source.columns:
                neighbor_state = mask_source["track_id"].astype(int).to_numpy() != int(row["track_id"])
                state_vals = mask_source.loc[neighbor_state, "_v57_state_match"].astype(bool).to_numpy()
                radius_source_mask_frac = float(np.mean(state_vals)) if len(state_vals) else 0.0
            else:
                radius_source_mask_frac = 0.0
            nn = float(st.get("_v58_nn_dist_px", np.nan)) if "_v58_nn_dist_px" in split_state_df.columns else np.nan
            if not math.isfinite(nn) or nn <= 0:
                nn = float(np.nanmedian(field.table["_v58_nn_dist_px"].to_numpy(float)))
            nn_clearance_ratio = clearance_min / max(nn, 1.0)

        flow_norm = float(np.linalg.norm(local_flow)) if "local_flow" in locals() else 0.0
        flow_align = np.sum(endpoint * local_flow[None, :], axis=1) / np.maximum(endpoint_norm * flow_norm, EPS) if flow_norm > 0 else np.zeros(k)
        flow_align = np.nan_to_num(flow_align, nan=0.0).astype(np.float32)
        flow_mismatch = np.maximum(0.0, 1.0 - flow_align)
        cur_norm = float(np.linalg.norm(cur_v))
        cur_align = np.sum(endpoint * cur_v[None, :], axis=1) / np.maximum(endpoint_norm * cur_norm, EPS) if cur_norm > 0 else np.zeros(k)
        cur_align = np.nan_to_num(cur_align, nan=0.0)
        free_support = (
            np.maximum(cur_align, 0.0) * free_front
            + np.maximum(-cur_align, 0.0) * free_back
            + (1.0 - np.abs(np.clip(cur_align, -1.0, 1.0))) * free_side
        ).astype(np.float32)
        same_track_jump_guard = np.maximum(0.0, path_norm - float(args.v58_max_path_factor) * max(current_speed * h, 1.0)).astype(np.float32)

        vals = {
            "overlap_max": overlap_max,
            "overlap_mean": overlap_mean,
            "close_frac": close_frac,
            "clearance_min": clearance_min,
            "clearance_mean": clearance_mean,
            "corridor_collision_frac": corridor_collision_frac,
            "corridor_free_frac": corridor_free_frac,
            "corridor_pressure_mean": corridor_pressure_mean,
            "corridor_pressure_max": corridor_pressure_max,
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
            "radius_px": np.full(k, central_radius, dtype=np.float32),
            "radius_source_mask_frac": np.full(k, radius_source_mask_frac, dtype=np.float32),
            "nn_clearance_ratio": nn_clearance_ratio,
            "swept_min_margin": swept_min_margin,
            "swept_mean_margin": swept_mean_margin,
            "swept_overlap_auc": swept_overlap_auc,
            "same_track_jump_guard": same_track_jump_guard,
        }
        for j, name in enumerate(names):
            feat[i, :, j] = vals[name]
        hard[i] = (
            (overlap_max > float(args.v58_hard_overlap))
            | (corridor_collision_frac > float(args.v58_hard_collision_frac))
            | (jump_excess > float(args.v58_hard_jump_excess_px))
            | (same_track_jump_guard > float(args.v58_hard_path_excess_px))
        )

    def col(name: str) -> np.ndarray:
        return feat[:, :, names.index(name)]

    clearance_pen = np.maximum(0.0, -col("swept_min_margin") / max(float(args.v58_default_radius_px), 1.0))
    rule = (
        1.25 * col("overlap_max")
        + 0.55 * col("overlap_mean")
        + 0.65 * col("corridor_collision_frac")
        + 0.35 * col("corridor_pressure_mean")
        + 0.30 * col("corridor_pressure_max")
        + 0.40 * clearance_pen
        + 0.22 * col("ahead_density")
        + 0.12 * col("contact_pressure")
        + 0.08 * col("flow_mismatch")
        + 0.04 * col("turn_mean")
        + 0.012 * col("jump_excess")
        + 0.010 * col("same_track_jump_guard")
        - 0.18 * col("free_support")
        - 0.18 * col("corridor_free_frac")
        - 0.05 * col("reliability")
    ).astype(np.float32)

    return v57.SplitEvidence(
        raw=np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        names=names,
        rule_risk=np.nan_to_num(rule, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        hard_violation=hard,
        coverage=np.ones(n, dtype=np.float32),
    )


def fit_risk_model(
    train_ev: v57.SplitEvidence,
    val_ev: v57.SplitEvidence,
    test_ev: v57.SplitEvidence,
    train_err: np.ndarray,
    args: argparse.Namespace,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n, k, f = train_ev.raw.shape
    feature_set = str(feature_set)
    path_names = {"path_norm", "endpoint_norm", "step_max", "accel_mean", "turn_mean", "jump_excess", "same_track_jump_guard"}
    dense_names = {
        "overlap_max",
        "overlap_mean",
        "close_frac",
        "clearance_min",
        "clearance_mean",
        "corridor_collision_frac",
        "corridor_free_frac",
        "corridor_pressure_mean",
        "corridor_pressure_max",
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
        "radius_source_mask_frac",
        "nn_clearance_ratio",
        "swept_min_margin",
        "swept_mean_margin",
        "swept_overlap_auc",
    }
    flow_names = {"flow_align", "flow_mismatch"}
    if feature_set == "full":
        idx = list(range(f))
        include_route_id = True
    elif feature_set == "full_no_route_id":
        idx = list(range(f))
        include_route_id = False
    elif feature_set == "dense_only":
        idx = [i for i, name in enumerate(train_ev.names) if name in dense_names]
        include_route_id = False
    elif feature_set == "dense_flow":
        idx = [i for i, name in enumerate(train_ev.names) if name in dense_names or name in flow_names]
        include_route_id = False
    elif feature_set == "no_path_no_route":
        idx = [i for i, name in enumerate(train_ev.names) if name not in path_names]
        include_route_id = False
    elif feature_set == "path_only":
        idx = [i for i, name in enumerate(train_ev.names) if name in path_names]
        include_route_id = True
    else:
        raise ValueError(f"unknown feature set: {feature_set}")
    if not idx:
        raise RuntimeError(f"empty feature set: {feature_set}")

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
    xva = xva3.reshape(val_ev.raw.shape[0] * k, xva3.shape[2])
    xte = xte3.reshape(test_ev.raw.shape[0] * k, xte3.shape[2])
    ytr = train_err.reshape(n * k)

    if args.v58_risk_model == "ridge":
        sc = StandardScaler()
        ztr = sc.fit_transform(xtr)
        zva = sc.transform(xva)
        zte = sc.transform(xte)
        model = Ridge(alpha=float(args.v58_ridge_alpha))
        model.fit(ztr, ytr)
        pred_tr = model.predict(ztr).reshape(n, k).astype(np.float32)
        pred_va = model.predict(zva).reshape(val_ev.raw.shape[0], k).astype(np.float32)
        pred_te = model.predict(zte).reshape(test_ev.raw.shape[0], k).astype(np.float32)
        return pred_tr, pred_va, pred_te, {
            "risk_model": "ridge",
            "risk_feature_set": feature_set,
            "risk_feature_dim": int(xtr.shape[1]),
            "risk_include_route_id": bool(include_route_id),
            "risk_alpha": float(args.v58_ridge_alpha),
        }

    model = HistGradientBoostingRegressor(
        max_iter=int(args.v58_hgbdt_iter),
        learning_rate=float(args.v58_hgbdt_lr),
        max_leaf_nodes=int(args.v58_hgbdt_leaf_nodes),
        l2_regularization=float(args.v58_hgbdt_l2),
        random_state=int(args.seed) + 58017,
    )
    model.fit(xtr, ytr)
    return (
        model.predict(xtr).reshape(n, k).astype(np.float32),
        model.predict(xva).reshape(val_ev.raw.shape[0], k).astype(np.float32),
        model.predict(xte).reshape(test_ev.raw.shape[0], k).astype(np.float32),
        {
            "risk_model": "hgbdt",
            "risk_feature_set": feature_set,
            "risk_feature_dim": int(xtr.shape[1]),
            "risk_include_route_id": bool(include_route_id),
            "hgbdt_iter": int(args.v58_hgbdt_iter),
        },
    )


def split_state(split_df: pd.DataFrame, field: DenseField) -> tuple[pd.DataFrame, float]:
    cols = KEY_COLS + [
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
        "_v58_nn_dist_px",
        "_v58_radius_px",
        "_v58_reliability",
    ]
    use = field.table[[c for c in cols if c in field.table.columns]].drop_duplicates(KEY_COLS)
    merged = split_df[KEY_COLS].merge(use, on=KEY_COLS, how="left")
    cov = float(pd.to_numeric(merged["_v57_state_match"], errors="coerce").fillna(0).astype(bool).mean()) if len(merged) else 0.0
    for col, default in v57.STATE_DEFAULTS.items():
        if col not in merged.columns:
            merged[col] = default
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default).astype(float)
    for c in ["_v57_state_match", "_v58_nn_dist_px", "_v58_radius_px", "_v58_reliability"]:
        if c not in merged.columns:
            merged[c] = 0.0
    return merged, cov


def tune_logits(logits_val: np.ndarray, logits_test: np.ndarray, basis: v26.RouteBasis, args: argparse.Namespace, label: str) -> dict[str, Any]:
    return v57.tune_logits(
        logits_val=logits_val,
        logits_test=logits_test,
        route_val=basis.route_val,
        route_test=basis.route_test,
        y_val=basis.y_val,
        args=args,
        label=label,
    )


def combine_and_tune(
    *,
    label: str,
    prior_val: np.ndarray,
    prior_test: np.ndarray,
    risk_val: np.ndarray,
    risk_test: np.ndarray,
    hard_val: np.ndarray,
    hard_test: np.ndarray,
    basis: v26.RouteBasis,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, diag, _weights = v57.combine_and_tune(
        label=label,
        prior_val=prior_val,
        prior_test=prior_test,
        risk_val=risk_val,
        risk_test=risk_test,
        hard_val=hard_val,
        hard_test=hard_test,
        basis=basis,
        args=args,
        extra=extra,
    )
    return rows, diag


def write_report(out_dir: Path, args: argparse.Namespace, contract: pd.DataFrame, metrics: pd.DataFrame, diag: pd.DataFrame, retention: pd.DataFrame) -> None:
    lines = [
        "# v58 Dense Route-Feasibility Field Report",
        "",
        f"Dataset: `{args.dataset}`, seed `{args.seed}`.",
        "",
        "## Contract",
        contract.to_markdown(index=False) if not contract.empty else "No contract rows.",
        "",
        "## Best h6 Metrics",
    ]
    if not metrics.empty:
        hmax = max(args.horizons)
        sub = metrics[metrics["horizon"].eq(hmax)].sort_values("rmse")
        cols = [
            c
            for c in [
                "method",
                "rmse",
                "r2",
                "control",
                "occupancy_mode",
                "risk_source",
                "risk_model",
                "risk_feature_set",
                "top_m",
                "temperature",
                "prior_alpha",
                "risk_beta",
                "veto_quantile",
            ]
            if c in sub.columns
        ]
        lines.append(sub[cols].head(60).to_markdown(index=False))
    else:
        lines.append("No metrics.")
    lines.extend(["", "## Diagnostics"])
    if not diag.empty:
        lines.append(diag.sort_values("hmax_rmse").head(80).to_markdown(index=False))
    else:
        lines.append("No diagnostics.")
    lines.extend(["", "## Oracle Retention"])
    if not retention.empty:
        lines.append(retention.head(120).to_markdown(index=False))
    else:
        lines.append("No retention rows.")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "- Pass requires `real_dense` / real field to beat no-field and hard controls.",
            "- If `path_only` or corrupted controls win, the route-feasibility transfer mechanism is still not validated.",
            "- If dense-only beats controls but final RMSE is weak, the field is a useful conditioner but needs a stronger cross-modal selector.",
        ]
    )
    (out_dir / "v58_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    np.random.seed(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.dense_features = args.features

    _device = closure.device_from_arg(args.device)
    basis = v26.build_route_basis(args, args.out_dir / "route_basis")
    pos = v57.load_position_table(args)
    state = v57.load_state_grids(args.state_grid)
    field = build_dense_field(pos, state, args)
    split_state_train, cov_tr = split_state(basis.split.train, field)
    split_state_val, cov_va = split_state(basis.split.val, field)
    split_state_test, cov_te = split_state(basis.split.test, field)

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
            {"item": "dense_field_rows", "value": int(field.stats["dense_rows"])},
            {"item": "dense_field_frames", "value": int(field.stats["dense_frames"])},
            {"item": "mask_state_coverage_full", "value": float(field.stats["mask_state_coverage"])},
            {"item": "mask_state_coverage_train", "value": cov_tr},
            {"item": "mask_state_coverage_val", "value": cov_va},
            {"item": "mask_state_coverage_test", "value": cov_te},
            {"item": "radius_median", "value": float(field.stats["radius_median"])},
            {"item": "radius_adaptive_mean", "value": float(field.stats["radius_adaptive_mean"])},
            {"item": "nn_dist_median", "value": float(field.stats["nn_dist_median"])},
        ]
    )
    contract.to_csv(args.out_dir / "v58_data_contract.csv", index=False)
    pd.DataFrame([field.stats]).to_csv(args.out_dir / "v58_dense_field_stats.csv", index=False)

    prior_tr = np.maximum(safe(basis.prior.probs_train), EPS)
    prior_va = np.maximum(safe(basis.prior.probs_val), EPS)
    prior_te = np.maximum(safe(basis.prior.probs_test), EPS)
    train_err = hmax_error_matrix(basis.route_train, basis.y_train, args)
    test_err = hmax_error_matrix(basis.route_test, basis.y_test, args)

    metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    prior_tuned = tune_logits(np.log(prior_va), np.log(prior_te), basis, args, "v58_coord_prior_only")
    metric_rows.extend(
        endpoint_rows(
            "v58_coord_prior_only",
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
            "method": "v58_coord_prior_only",
            "hmax": max(args.horizons),
            "hmax_rmse": float([r["rmse"] for r in metric_rows if r["method"] == "v58_coord_prior_only" and r["horizon"] == max(args.horizons)][0]),
            "control": "coord_prior_only",
            "top_m": prior_tuned["top_m"],
            "temperature": prior_tuned["temperature"],
            "val_rmse": prior_tuned["val_rmse"],
        }
    )

    controls = parse_csv(args.v58_controls)
    if "all" in controls:
        controls = [
            "real",
            "no_field",
            "zero_radius",
            "median_radius",
            "radius_shuffled",
            "wrong_cell_radius",
            "neighbor_velocity_shuffled",
            "wrong_frame_field",
            "same_density_random_field",
            "route_reverse",
            "route_lateral",
            "route_roll",
        ]
    learned_controls = set(parse_csv(args.v58_learned_controls))
    occupancy_modes = parse_csv(args.occupancy_modes)

    for occupancy_mode in occupancy_modes:
        for control in controls:
            label_base = f"v58_{control}_{occupancy_mode}"
            train_ev = compute_dense_evidence_split(
                split_df=basis.split.train,
                split_state_df=split_state_train,
                field=field,
                route_residual=basis.route_train,
                base=basis.arrays.base_train,
                args=args,
                control=control,
                occupancy_mode=occupancy_mode,
                seed=args.seed + 5811,
            )
            val_ev = compute_dense_evidence_split(
                split_df=basis.split.val,
                split_state_df=split_state_val,
                field=field,
                route_residual=basis.route_val,
                base=basis.arrays.base_val,
                args=args,
                control=control,
                occupancy_mode=occupancy_mode,
                seed=args.seed + 5812,
            )
            test_ev = compute_dense_evidence_split(
                split_df=basis.split.test,
                split_state_df=split_state_test,
                field=field,
                route_residual=basis.route_test,
                base=basis.arrays.base_test,
                args=args,
                control=control,
                occupancy_mode=occupancy_mode,
                seed=args.seed + 5813,
            )
            audit_rows.append(
                {
                    "control": control,
                    "occupancy_mode": occupancy_mode,
                    "risk_mean": float(np.mean(test_ev.rule_risk)),
                    "risk_std": float(np.std(test_ev.rule_risk)),
                    "hard_violation_frac": float(np.mean(test_ev.hard_violation)),
                    "coverage": float(np.mean(test_ev.coverage)),
                    "feature_dim": int(test_ev.raw.shape[2]),
                    "corridor_collision_mean": float(np.mean(test_ev.raw[:, :, test_ev.names.index("corridor_collision_frac")])),
                    "corridor_free_mean": float(np.mean(test_ev.raw[:, :, test_ev.names.index("corridor_free_frac")])),
                    "swept_min_margin_mean": float(np.mean(test_ev.raw[:, :, test_ev.names.index("swept_min_margin")])),
                }
            )
            extra = {"control": control, "occupancy_mode": occupancy_mode, "risk_source": "rule"}
            rows, diag = combine_and_tune(
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
                v57.oracle_retention_rows(
                    label=f"{label_base}_rule",
                    risk=test_ev.rule_risk,
                    hard=test_ev.hard_violation,
                    err=test_err,
                    args=args,
                    extra=extra,
                )
            )

            if control in learned_controls:
                for feature_set in parse_csv(args.v58_risk_feature_sets):
                    _rtr, rva, rte, risk_meta = fit_risk_model(train_ev, val_ev, test_ev, train_err, args, feature_set)
                    extra2 = {**extra, "risk_source": "learned", **risk_meta}
                    rows, diag = combine_and_tune(
                        label=f"{label_base}_{risk_meta['risk_model']}_{feature_set}",
                        prior_val=prior_va,
                        prior_test=prior_te,
                        risk_val=rva,
                        risk_test=rte,
                        hard_val=val_ev.hard_violation,
                        hard_test=test_ev.hard_violation,
                        basis=basis,
                        args=args,
                        extra=extra2,
                    )
                    metric_rows.extend(rows)
                    diag_rows.append(diag)
                    retention_rows.extend(
                        v57.oracle_retention_rows(
                            label=f"{label_base}_{risk_meta['risk_model']}_{feature_set}",
                            risk=rte,
                            hard=test_ev.hard_violation,
                            err=test_err,
                            args=args,
                            extra=extra2,
                        )
                    )
            if control == "real" and occupancy_mode == occupancy_modes[0]:
                flat = test_ev.raw.reshape(-1, test_ev.raw.shape[2])
                pd.DataFrame(flat[: min(len(flat), 6000)], columns=test_ev.names).to_csv(
                    args.out_dir / "v58_dense_candidate_features.csv",
                    index=False,
                )

    metrics = pd.DataFrame(metric_rows)
    diag = pd.DataFrame(diag_rows)
    retention = pd.DataFrame(retention_rows)
    audit_df = pd.DataFrame(audit_rows)
    metrics.to_csv(args.out_dir / "v58_dense_route_feasibility_summary.csv", index=False)
    diag.to_csv(args.out_dir / "v58_dense_route_feasibility_diagnostics.csv", index=False)
    retention.to_csv(args.out_dir / "v58_oracle_retention.csv", index=False)
    audit_df.to_csv(args.out_dir / "v58_dense_field_audit.csv", index=False)
    if not diag.empty:
        diag.sort_values("hmax_rmse").to_csv(args.out_dir / "v58_cluster_feasibility.csv", index=False)
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

    # v26/v16 route-basis compatibility.
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

    # Dense field / feasibility.
    ap.add_argument("--occupancy-modes", default="static,velocity,clipped_velocity,local_flow")
    ap.add_argument("--v58-controls", default="all")
    ap.add_argument(
        "--v58-learned-controls",
        default="real,no_field,radius_shuffled,wrong_cell_radius,neighbor_velocity_shuffled,wrong_frame_field,same_density_random_field,route_reverse,route_lateral",
    )
    ap.add_argument("--v58-risk-feature-sets", default="full,full_no_route_id,dense_only,dense_flow,no_path_no_route,path_only")
    ap.add_argument("--v58-risk-model", default="hgbdt", choices=["hgbdt", "ridge"])
    ap.add_argument("--v58-hgbdt-iter", type=int, default=160)
    ap.add_argument("--v58-hgbdt-lr", type=float, default=0.045)
    ap.add_argument("--v58-hgbdt-leaf-nodes", type=int, default=31)
    ap.add_argument("--v58-hgbdt-l2", type=float, default=0.02)
    ap.add_argument("--v58-ridge-alpha", type=float, default=100.0)
    ap.add_argument("--v58-neighbor-k", type=int, default=64)
    ap.add_argument("--v58-flow-k", type=int, default=16)
    ap.add_argument("--v58-samples-per-step", type=int, default=4)
    ap.add_argument("--v58-default-radius-px", type=float, default=23.1)
    ap.add_argument("--v58-radius-min-px", type=float, default=5.0)
    ap.add_argument("--v58-radius-max-px", type=float, default=42.0)
    ap.add_argument("--v58-nn-radius-factor", type=float, default=0.34)
    ap.add_argument("--v58-adaptive-reliability", type=float, default=0.45)
    ap.add_argument("--v58-central-radius-scale", type=float, default=0.55)
    ap.add_argument("--v58-radius-scale", type=float, default=0.85)
    ap.add_argument("--v58-neighbor-radius-scale", type=float, default=0.85)
    ap.add_argument("--v58-close-margin-px", type=float, default=10.0)
    ap.add_argument("--v58-free-margin-px", type=float, default=8.0)
    ap.add_argument("--v58-pressure-scale-px", type=float, default=18.0)
    ap.add_argument("--v58-corridor-width-px", type=float, default=92.0)
    ap.add_argument("--v58-neighbor-velocity-clip", type=float, default=18.0)
    ap.add_argument("--v58-jump-factor", type=float, default=3.0)
    ap.add_argument("--v58-max-path-factor", type=float, default=2.6)
    ap.add_argument("--v58-hard-overlap", type=float, default=0.55)
    ap.add_argument("--v58-hard-collision-frac", type=float, default=0.22)
    ap.add_argument("--v58-hard-jump-excess-px", type=float, default=24.0)
    ap.add_argument("--v58-hard-path-excess-px", type=float, default=36.0)

    # Shared tuning grids expected by v57 combine_and_tune.
    ap.add_argument("--v57-top-m-grid", default="1,2,4,8,12")
    ap.add_argument("--v57-temperature-grid", default="0.35,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v57-prior-alpha-grid", default="0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v57-risk-beta-grid", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v57-veto-quantile-grid", default="1.0,0.97,0.95,0.90")

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
        args.occupancy_modes = "static,velocity"
        args.v58_controls = "real,no_field,same_density_random_field,wrong_frame_field,route_lateral"
        args.v58_learned_controls = "real,no_field,same_density_random_field"
        args.v58_risk_feature_sets = "full,dense_only,path_only"
        args.v58_hgbdt_iter = min(args.v58_hgbdt_iter, 80)
        args.v58_neighbor_k = min(args.v58_neighbor_k, 32)
        args.v58_samples_per_step = min(args.v58_samples_per_step, 3)
        args.v57_top_m_grid = "1,2,4,8"
        args.v57_temperature_grid = "0.5,1.0,1.5"
        args.v57_prior_alpha_grid = "0.75,1.0,1.5"
        args.v57_risk_beta_grid = "0.0,0.5,1.0,2.0"
        args.v57_veto_quantile_grid = "1.0,0.95"
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
