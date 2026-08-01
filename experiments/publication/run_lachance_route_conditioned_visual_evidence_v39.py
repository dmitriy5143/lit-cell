#!/usr/bin/env python3
"""Route-conditioned visual evidence validator v39.

v38 tested visual state as global state tokens. This runner tests the missing
transfer mechanism: every candidate route gets its own visual compatibility
features computed from the route direction and tracking-aligned mask/state
variables.

The core idea is:

    route_k direction / curvature / endpoint
    + visual front/back/left/right/free/contact/polarity state
    -> evidence_k
    -> route validator

Target/future is used only for teacher labels and training losses, never as an
inference feature.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_visual_state_route_validator_v38 as v38  # noqa: E402
import run_lachance_visual_state_target_v32 as v32  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_conditioned_visual_evidence_v39_2026-07-08"
DEFAULT_ALIGNED_ROOT = ROOT / "outputs" / "visual_temporal_target_v37_aligned_bulk_seed42_2026-07-07"
DEFAULT_FEATURES = DEFAULT_ALIGNED_ROOT / "raw_context_aligned_extracted_4784.csv"
DEFAULT_SEGF = DEFAULT_ALIGNED_ROOT / "seg_foundation_aligned_pointbox_as_segf.csv"
EPS = 1e-8


def parse_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str | list[int]) -> list[int]:
    return audit.parse_ints(text) if not isinstance(text, list) else [int(x) for x in text]


def safe_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def col_map(names: list[str]) -> dict[str, int]:
    return {str(c): i for i, c in enumerate(names)}


def get_col(x: np.ndarray, cmap: dict[str, int], name: str, default: float = 0.0) -> np.ndarray:
    if name in cmap:
        return x[:, cmap[name]].astype(np.float32)
    return np.full((len(x),), float(default), dtype=np.float32)


def visual_state_packet(
    packets: dict[str, v32.Packet],
    variant: str,
    *,
    coord_dim: int,
    feature_mode: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[str]]:
    name = {
        "real": "coord_plus_seg_foundation_state",
        "zero": "coord_plus_seg_foundation_state_zero",
        "row_shuffled": "coord_plus_seg_foundation_state_row_shuffled",
        "same_frame_wrong_cell": "coord_plus_seg_foundation_state_same_frame_wrong_cell",
        "time_shuffled": "coord_plus_seg_foundation_state_time_shuffled",
    }.get(variant)
    if name is None:
        # fixed dimensional zero state matching the real feature list
        real = packets["coord_plus_seg_foundation_state"]
        names = real.feature_names[coord_dim:]
        return (
            np.zeros((len(real.train), len(names)), dtype=np.float32),
            np.zeros((len(real.val), len(names)), dtype=np.float32),
            np.zeros((len(real.test), len(names)), dtype=np.float32),
        ), names
    p = packets[name]
    xtr = p.train[:, coord_dim:].astype(np.float32)
    xva = p.val[:, coord_dim:].astype(np.float32)
    xte = p.test[:, coord_dim:].astype(np.float32)
    names = p.feature_names[coord_dim:]
    keep = v38.filter_visual_cols(names, feature_mode)
    if keep:
        return (xtr[:, keep], xva[:, keep], xte[:, keep]), [names[i] for i in keep]
    return (
        np.zeros((len(xtr), 0), dtype=np.float32),
        np.zeros((len(xva), 0), dtype=np.float32),
        np.zeros((len(xte), 0), dtype=np.float32),
    ), []


def current_velocity(df: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    if {"dx_px", "dy_px"}.issubset(df.columns):
        v = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    elif {"coord_dx", "coord_dy"}.issubset(df.columns):
        v = df[["coord_dx", "coord_dy"]].fillna(0.0).to_numpy(np.float32)
    else:
        v = np.asarray(base, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    fallback = n[:, 0] < 1e-6
    if np.any(fallback):
        v = v.copy()
        v[fallback] = np.array([1.0, 0.0], dtype=np.float32)
        n[fallback] = 1.0
    return (v / np.maximum(n, 1e-6)).astype(np.float32)


def sector_mix(front: np.ndarray, back: np.ndarray, left: np.ndarray, right: np.ndarray, par: np.ndarray, side: np.ndarray) -> np.ndarray:
    wf = np.maximum(par, 0.0)
    wb = np.maximum(-par, 0.0)
    wl = np.maximum(side, 0.0)
    wr = np.maximum(-side, 0.0)
    s = np.maximum(wf + wb + wl + wr, 1e-6)
    return (wf * front[:, None] + wb * back[:, None] + wl * left[:, None] + wr * right[:, None]) / s


def prioritized_visual_cols(names: list[str], max_cols: int, *, cross: bool) -> list[int]:
    if max_cols <= 0 or not names:
        return []
    tokens = (
        "front",
        "back",
        "left",
        "right",
        "balance",
        "free",
        "contact",
        "boundary",
        "neighbor",
        "orient",
        "centroid",
        "velocity",
        "grad",
        "area",
        "eccentric",
        "solidity",
        "quality",
        "score",
        "fallback",
        "extract",
        "temporal",
        "track",
        "history",
        "lag",
        "delta",
        "mean",
        "std",
        "span",
        "stable",
        "stability",
        "iou",
        "xor",
        "aligned",
        "available",
        "reliability",
        "pressure",
        "anisotropy",
        "age",
    )
    # Cross terms are most useful for directional/state variables; raw repeated
    # state can afford a slightly broader fallback.
    priority = [i for i, name in enumerate(names) if any(t in name.lower() for t in tokens)]
    if not cross:
        priority.extend(i for i in range(len(names)) if i not in set(priority))
    if not priority:
        priority = list(range(len(names)))
    seen: set[int] = set()
    out: list[int] = []
    for i in priority:
        if i not in seen:
            seen.add(i)
            out.append(i)
        if len(out) >= max_cols:
            break
    return out


def route_visual_evidence(
    *,
    route_pred: np.ndarray,
    base_step: np.ndarray,
    split_df: pd.DataFrame,
    visual: np.ndarray,
    visual_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    n, k, _ = route_pred.shape
    max_h = int(args.max_horizon)
    residual_steps = route_pred.reshape(n, k, max_h, 2)
    full_steps = residual_steps + base_step[:, None, None, :]
    endpoint = np.sum(full_steps[:, :, :max_h, :], axis=2)
    route_norm = np.linalg.norm(endpoint, axis=2)
    route_u = endpoint / np.maximum(route_norm[:, :, None], 1e-6)

    self_u = current_velocity(split_df, base_step)
    self_p = np.stack([-self_u[:, 1], self_u[:, 0]], axis=1).astype(np.float32)
    par = np.sum(route_u * self_u[:, None, :], axis=2).astype(np.float32)
    side = np.sum(route_u * self_p[:, None, :], axis=2).astype(np.float32)

    cmap = col_map(visual_names)
    # Raw visual components. Defaults are neutral, not optimistic.
    orient = np.stack(
        [
            get_col(visual, cmap, "segf_orient_x", 0.0),
            get_col(visual, cmap, "segf_orient_y", 0.0),
        ],
        axis=1,
    )
    orient_norm = np.linalg.norm(orient, axis=1, keepdims=True)
    orient_u = orient / np.maximum(orient_norm, 1e-6)
    centroid = np.stack(
        [
            get_col(visual, cmap, "segf_centroid_dx", 0.0),
            get_col(visual, cmap, "segf_centroid_dy", 0.0),
        ],
        axis=1,
    )
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_u = centroid / np.maximum(centroid_norm, 1e-6)
    orient_align = np.sum(route_u * orient_u[:, None, :], axis=2).astype(np.float32)
    centroid_align = np.sum(route_u * centroid_u[:, None, :], axis=2).astype(np.float32)

    front_frac = get_col(visual, cmap, "segf_front_frac", 0.0)
    back_frac = get_col(visual, cmap, "segf_back_frac", 0.0)
    left_frac = get_col(visual, cmap, "segf_left_frac", 0.0)
    right_frac = get_col(visual, cmap, "segf_right_frac", 0.0)
    front_int = get_col(visual, cmap, "segf_front_intensity", 0.0)
    back_int = get_col(visual, cmap, "segf_back_intensity", 0.0)
    left_int = get_col(visual, cmap, "segf_left_intensity", 0.0)
    right_int = get_col(visual, cmap, "segf_right_intensity", 0.0)
    free_front = get_col(visual, cmap, "segf_free_front_frac", 0.5)
    free_back = get_col(visual, cmap, "segf_free_back_frac", 0.5)
    free_left = get_col(visual, cmap, "segf_free_left_frac", 0.5)
    free_right = get_col(visual, cmap, "segf_free_right_frac", 0.5)
    grad_front = get_col(visual, cmap, "segf_front_grad", 0.0)
    grad_back = get_col(visual, cmap, "segf_back_grad", 0.0)
    # We only have front/back gradients in this grid; side gradients are neutral.
    zero = np.zeros(n, dtype=np.float32)
    dir_frac = sector_mix(front_frac, back_frac, left_frac, right_frac, par, side).astype(np.float32)
    dir_int = sector_mix(front_int, back_int, left_int, right_int, par, side).astype(np.float32)
    dir_free = sector_mix(free_front, free_back, free_left, free_right, par, side).astype(np.float32)
    dir_grad = sector_mix(grad_front, grad_back, zero, zero, par, side).astype(np.float32)

    fb_balance = get_col(visual, cmap, "segf_front_back_balance", 0.0)
    lr_balance = get_col(visual, cmap, "segf_left_right_balance", 0.0)
    directional_balance = par * fb_balance[:, None] + side * lr_balance[:, None]
    boundary_grad = get_col(visual, cmap, "segf_boundary_grad", 0.0)
    near_mask = get_col(visual, cmap, "segf_neighbor_near_mask_frac", 0.0)
    on_boundary = get_col(visual, cmap, "segf_neighbor_on_boundary_frac", 0.0)
    nn_dist = get_col(visual, cmap, "segf_nn_dist_crop_norm", 0.0)
    sam_score = get_col(visual, cmap, "segf_sam_score", 0.0)
    fallback = get_col(visual, cmap, "segf_sam_fallback", 0.0)
    extract_ok = get_col(visual, cmap, "segf_extract_ok", 1.0)
    eccentricity = get_col(visual, cmap, "segf_eccentricity", 0.0)
    area = get_col(visual, cmap, "segf_area_frac", 0.0)
    solidity = get_col(visual, cmap, "segf_solidity", 0.0)

    contact_pressure = (1.0 - dir_free) * (near_mask[:, None] + on_boundary[:, None] + np.maximum(boundary_grad[:, None], 0.0))
    route_speed = route_norm / max(float(max_h), 1.0)
    route_turn = np.linalg.norm(np.diff(full_steps, axis=2), axis=3).mean(axis=2) if max_h > 1 else np.zeros((n, k), dtype=np.float32)
    visual_quality = (extract_ok * (1.0 - fallback) * np.maximum(sam_score, 0.0))[:, None]
    support = (
        0.35 * dir_free
        + 0.25 * np.maximum(orient_align, 0.0)
        + 0.20 * np.maximum(directional_balance, 0.0)
        + 0.20 * visual_quality
        - 0.25 * contact_pressure
    )
    conflict = contact_pressure + np.maximum(-orient_align, 0.0) * (1.0 - dir_free)

    ev = np.stack(
        [
            par,
            side,
            np.abs(side),
            route_norm,
            route_speed,
            route_turn,
            orient_align,
            np.abs(orient_align),
            centroid_align,
            dir_frac,
            dir_int,
            dir_free,
            dir_grad,
            directional_balance,
            contact_pressure,
            conflict,
            support,
            support * route_norm,
            conflict * route_norm,
            boundary_grad[:, None].repeat(k, axis=1),
            near_mask[:, None].repeat(k, axis=1),
            on_boundary[:, None].repeat(k, axis=1),
            nn_dist[:, None].repeat(k, axis=1),
            visual_quality.repeat(k, axis=1),
            eccentricity[:, None].repeat(k, axis=1),
            area[:, None].repeat(k, axis=1),
            solidity[:, None].repeat(k, axis=1),
            (dir_free - contact_pressure),
        ],
        axis=2,
    )
    blocks = [ev.astype(np.float32)]

    raw_cols = prioritized_visual_cols(visual_names, int(args.v39_raw_visual_candidate_cols), cross=False)
    if raw_cols:
        raw = visual[:, raw_cols].astype(np.float32)
        raw = np.repeat(raw[:, None, :], k, axis=1)
        blocks.append(raw)

    cross_cols = prioritized_visual_cols(visual_names, int(args.v39_cross_visual_cols), cross=True)
    if cross_cols:
        vis = visual[:, cross_cols].astype(np.float32)
        route_primitives = np.stack(
            [
                par,
                side,
                np.abs(side),
                route_speed,
                route_turn,
                route_norm / max(float(max_h), 1.0),
            ],
            axis=2,
        ).astype(np.float32)
        cross = route_primitives[:, :, :, None] * vis[:, None, None, :]
        blocks.append(cross.reshape(n, k, -1).astype(np.float32))

    out = np.concatenate(blocks, axis=2)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_candidate_with_evidence(
    *,
    route_pred: np.ndarray,
    probs: np.ndarray,
    base_step: np.ndarray,
    split_df: pd.DataFrame,
    visual: np.ndarray,
    visual_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    cand = v38.candidate_features(route_pred, probs, args)
    if int(args.v39_use_visual_evidence):
        evidence = route_visual_evidence(
            route_pred=route_pred,
            base_step=base_step,
            split_df=split_df,
            visual=visual,
            visual_names=visual_names,
            args=args,
        )
        cand = np.concatenate([cand, evidence], axis=2).astype(np.float32)
    return cand


def run_variant(
    *,
    variant: str,
    visual_variant: str,
    visual_feature_mode: str,
    packets: dict[str, v32.Packet],
    basis: v26.RouteBasis,
    teacher: v38.TeacherLabels,
    args: argparse.Namespace,
    device,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    coord_dim = packets["coord_all_context"].train.shape[1]
    (vtr, vva, vte), vnames = visual_state_packet(
        packets,
        visual_variant,
        coord_dim=coord_dim,
        feature_mode=visual_feature_mode,
    )
    cand_tr = build_candidate_with_evidence(
        route_pred=basis.route_train,
        probs=basis.prior.probs_train,
        base_step=basis.arrays.base_train,
        split_df=basis.split.train,
        visual=vtr,
        visual_names=vnames,
        args=args,
    )
    cand_va = build_candidate_with_evidence(
        route_pred=basis.route_val,
        probs=basis.prior.probs_val,
        base_step=basis.arrays.base_val,
        split_df=basis.split.val,
        visual=vva,
        visual_names=vnames,
        args=args,
    )
    cand_te = build_candidate_with_evidence(
        route_pred=basis.route_test,
        probs=basis.prior.probs_test,
        base_step=basis.arrays.base_test,
        split_df=basis.split.test,
        visual=vte,
        visual_names=vnames,
        args=args,
    )
    cand_tr, cand_va, cand_te = v38.standardize_candidate_features(cand_tr, cand_va, cand_te)
    return v38.train_variant(
        variant=variant,
        visual_variant="no_visual",
        packets=packets,
        basis=basis,
        cand_train=cand_tr,
        cand_val=cand_va,
        cand_test=cand_te,
        teacher=teacher,
        args=args,
        out_dir=args.out_dir,
        device=device,
        use_cross_attention=bool(args.v39_use_context_cross_attention),
        use_candidate_self_attention=bool(args.v39_use_candidate_self_attention),
        use_route_teacher=True,
        visual_feature_mode="all",
        drop_families=(),
        use_uncertainty=bool(args.v39_use_uncertainty),
    )


def write_report(out_dir: Path, summary: pd.DataFrame, diagnostics: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    lines = ["# v39 Route-Conditioned Visual Evidence", ""]
    hmax = max(args.horizons)
    h6 = summary[summary["horizon"].eq(hmax)].sort_values("rmse")
    cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "best_val_hmax_rmse", "visual_source", "visual_feature_mode"] if c in h6.columns]
    lines.append("## Best h6")
    lines.append(h6[cols].head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Diagnostics")
    dcols = [c for c in ["method", "variant", "hmax_rmse", "route_top1", "route_top3", "ndcg_at3", "ndcg_at8", "expected_candidate_error_mean", "oracle_gap_closed_vs_prior"] if c in diagnostics.columns]
    lines.append(diagnostics.sort_values("hmax_rmse")[dcols].head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Decision")
    lines.append("```json")
    lines.append(json.dumps(v38.finite_json(decision), indent=2, ensure_ascii=False))
    lines.append("```")
    (out_dir / "route_conditioned_visual_evidence_v39_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    audit.set_global_seed(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.generator_max_train_rows < 0:
        args.generator_max_train_rows = args.max_train_rows
    if args.generator_max_val_rows < 0:
        args.generator_max_val_rows = args.max_val_rows
    if args.generator_max_test_rows < 0:
        args.generator_max_test_rows = args.max_test_rows
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 450)
        args.generator_max_train_rows = args.max_train_rows
        args.generator_max_val_rows = args.max_val_rows
        args.generator_max_test_rows = args.max_test_rows
        args.generator_posterior_epochs = min(args.generator_posterior_epochs, 3)
        args.generator_student_epochs = min(args.generator_student_epochs, 3)
        args.generator_learned_route_epochs = min(args.generator_learned_route_epochs, 2)
        args.v38_epochs = min(args.v38_epochs, 4)
        args.v39_variants = "evidence_full,evidence_no_visual,evidence_row_shuffled"
    device = v38.device_from_arg(args.device)
    basis = v26.build_route_basis(args, args.out_dir / "route_basis")
    packets = v32.build_packets(args, basis.arrays, basis.split)
    teacher = v38.build_teacher_labels(args, basis)
    base_metrics, base_diag = v38.baseline_rows(basis, teacher, args)
    hmax = max(args.horizons)
    prior = base_metrics[base_metrics["horizon"].eq(hmax) & base_metrics["variant"].astype(str).eq("prior_topm")]
    oracle = base_metrics[base_metrics["horizon"].eq(hmax) & base_metrics["variant"].astype(str).eq("oracle")]
    args._v38_prior_hmax_rmse = float(prior.iloc[0]["rmse"]) if not prior.empty else None
    args._v38_oracle_hmax_rmse = float(oracle.iloc[0]["rmse"]) if not oracle.empty else None

    variant_specs = {
        "evidence_full": ("real", "all"),
        "evidence_no_visual": ("no_visual", "all"),
        "evidence_zero": ("zero", "all"),
        "evidence_row_shuffled": ("row_shuffled", "all"),
        "evidence_same_frame_wrong_cell": ("same_frame_wrong_cell", "all"),
        "evidence_time_shuffled": ("time_shuffled", "all"),
        "evidence_shape_only": ("real", "shape_only"),
        "evidence_shape_row_shuffled": ("row_shuffled", "shape_only"),
        "evidence_shape_same_frame_wrong_cell": ("same_frame_wrong_cell", "shape_only"),
        "evidence_shape_time_shuffled": ("time_shuffled", "shape_only"),
        "evidence_polarity_only": ("real", "polarity_only"),
        "evidence_polarity_row_shuffled": ("row_shuffled", "polarity_only"),
        "evidence_polarity_same_frame_wrong_cell": ("same_frame_wrong_cell", "polarity_only"),
        "evidence_polarity_time_shuffled": ("time_shuffled", "polarity_only"),
        "evidence_history_only": ("real", "history_only"),
        "evidence_history_row_shuffled": ("row_shuffled", "history_only"),
        "evidence_history_same_frame_wrong_cell": ("same_frame_wrong_cell", "history_only"),
        "evidence_history_time_shuffled": ("time_shuffled", "history_only"),
        "evidence_polarity_history_only": ("real", "polarity_history_only"),
        "evidence_polarity_history_row_shuffled": ("row_shuffled", "polarity_history_only"),
        "evidence_polarity_history_same_frame_wrong_cell": ("same_frame_wrong_cell", "polarity_history_only"),
        "evidence_polarity_history_time_shuffled": ("time_shuffled", "polarity_history_only"),
        "evidence_contact_only": ("real", "contact_only"),
        "evidence_contact_row_shuffled": ("row_shuffled", "contact_only"),
        "evidence_contact_same_frame_wrong_cell": ("same_frame_wrong_cell", "contact_only"),
        "evidence_contact_time_shuffled": ("time_shuffled", "contact_only"),
        "evidence_quality_only": ("real", "quality_only"),
        "evidence_no_contact": ("real", "no_contact"),
        "evidence_no_polarity": ("real", "no_polarity"),
    }
    metric_parts = [base_metrics]
    diag_rows = base_diag.to_dict(orient="records")
    logs = []
    for variant in parse_csv(args.v39_variants):
        if variant not in variant_specs:
            raise ValueError(f"Unknown v39 variant: {variant}")
        visual_source, mode = variant_specs[variant]
        rows, diag, log = run_variant(
            variant=variant,
            visual_variant=visual_source,
            visual_feature_mode=mode,
            packets=packets,
            basis=basis,
            teacher=teacher,
            args=args,
            device=device,
        )
        rows["visual_source"] = visual_source
        rows["visual_feature_mode"] = mode
        diag.update({"visual_source": visual_source, "visual_feature_mode": mode})
        log["visual_source"] = visual_source
        log["visual_feature_mode"] = mode
        metric_parts.append(rows)
        diag_rows.append(diag)
        logs.append(log)

    summary = pd.concat(metric_parts, ignore_index=True, sort=False)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.insert(0, "seed", int(args.seed))
    diagnostics.insert(0, "dataset", str(args.dataset))
    train_log = pd.concat(logs, ignore_index=True, sort=False) if logs else pd.DataFrame()
    if not train_log.empty:
        train_log.insert(0, "seed", int(args.seed))
        train_log.insert(0, "dataset", str(args.dataset))

    summary.to_csv(args.out_dir / "route_conditioned_visual_evidence_v39_summary.csv", index=False)
    diagnostics.to_csv(args.out_dir / "route_conditioned_visual_evidence_v39_diagnostics.csv", index=False)
    train_log.to_csv(args.out_dir / "route_conditioned_visual_evidence_v39_train_log.csv", index=False)

    h6 = summary[summary["horizon"].eq(hmax)].sort_values("rmse")
    best = h6.iloc[0].to_dict() if not h6.empty else {}
    full = h6[h6["variant"].astype(str).eq("evidence_full")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("evidence_full")].empty else {}
    no_visual = h6[h6["variant"].astype(str).eq("evidence_no_visual")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("evidence_no_visual")].empty else {}
    controls = h6[h6["variant"].astype(str).isin(["evidence_zero", "evidence_row_shuffled", "evidence_same_frame_wrong_cell", "evidence_time_shuffled"])]
    best_control = controls.iloc[0].to_dict() if not controls.empty else {}
    prior_row = h6[h6["variant"].astype(str).eq("prior_topm")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("prior_topm")].empty else {}
    oracle_row = h6[h6["variant"].astype(str).eq("oracle")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("oracle")].empty else {}
    decision = {
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "best_hmax": best,
        "full_hmax": full,
        "no_visual_hmax": no_visual,
        "best_visual_control_hmax": best_control,
        "prior_baseline_hmax": prior_row,
        "oracle_hmax": oracle_row,
        "full_beats_no_visual": bool(full and no_visual and float(full["rmse"]) < float(no_visual["rmse"])),
        "full_beats_visual_controls": bool(full and best_control and float(full["rmse"]) < float(best_control["rmse"])),
        "full_beats_prior": bool(full and prior_row and float(full["rmse"]) < float(prior_row["rmse"])),
        "hard_pass_h6_le_16": bool(full and float(full["rmse"]) <= 16.0),
    }
    (args.out_dir / "route_conditioned_visual_evidence_v39_decision.json").write_text(json.dumps(v38.finite_json(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(v38.finite_json(vars(args)), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.out_dir, summary, diagnostics, decision, args)
    print(json.dumps({"out_dir": str(args.out_dir), "best_hmax": v38.finite_json(best), "decision": v38.finite_json(decision)}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--dense-features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    ap.add_argument("--dataset", default="MDCK_Bulk")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seq", default="1,2,3,4")
    ap.add_argument("--val-seq", default="5")
    ap.add_argument("--test-seq", default="6")
    ap.add_argument("--horizons", default="1,2,4,6")
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--max-test-rows", type=int, default=0)
    ap.add_argument("--max-features-per-family", type=int, default=160)
    ap.add_argument("--max-all-features", type=int, default=384)
    ap.add_argument("--device", default="auto")
    # v26/v16 generator arguments.
    ap.add_argument("--generator-max-train-rows", type=int, default=-1)
    ap.add_argument("--generator-max-val-rows", type=int, default=-1)
    ap.add_argument("--generator-max-test-rows", type=int, default=-1)
    ap.add_argument("--generator-posterior-epochs", type=int, default=4)
    ap.add_argument("--generator-student-epochs", type=int, default=4)
    ap.add_argument("--generator-learned-route-epochs", type=int, default=3)
    ap.add_argument("--generator-candidate-k", type=int, default=32)
    ap.add_argument("--generator-oracle-k", default="8,16,32")
    ap.add_argument("--generator-variant", default="context_velocity")
    ap.add_argument("--generator-prior-model", default="logistic")
    ap.add_argument("--generator-base-mixes", default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    ap.add_argument("--generator-calibrators", default="correction_context,stacked_context")
    ap.add_argument("--generator-max-context-features", type=int, default=384)
    ap.add_argument("--dense-max-cols", type=int, default=256)
    ap.add_argument("--v25-velocity-max-cols", type=int, default=160)
    ap.add_argument("--v25-route-k", type=int, default=12)
    # Visual grids for v32 packet construction.
    ap.add_argument("--object-grid", type=Path, default=ROOT / "outputs" / "lachance_object_centric_mask_grid_bulk_seed42_2026-07-03" / "object_centric_mask_feature_grid.csv")
    ap.add_argument("--temporal-grid", type=Path, default=ROOT / "outputs" / "temporal_mask_change_medium_bulk_seed42_2026-07-04" / "multiseed_instance_mask_feature_grid.csv")
    ap.add_argument("--multiseed-grid", type=Path, default=ROOT / "outputs" / "multiseed_instance_mask_medium_bulk_seed42_2026-07-04" / "multiseed_instance_mask_feature_grid.csv")
    ap.add_argument("--seg-foundation-grid", type=Path, default=DEFAULT_SEGF)
    ap.add_argument("--visual-tokens", default="area,perimeter,eccentricity,solidity,extent,major,minor,elongation,orient,velocity,centroid,front,back,left,right,balance,intensity,grad,free,contact,boundary,neighbor,seed,center,available,quality,fallback,track_aligned")
    ap.add_argument("--max-object-cols", type=int, default=0)
    ap.add_argument("--max-temporal-cols", type=int, default=0)
    ap.add_argument("--max-multiseed-cols", type=int, default=0)
    ap.add_argument("--max-seg-cols", type=int, default=160)
    ap.add_argument("--max-interaction-cols", type=int, default=120)
    # Model args reused by v38.train_variant.
    ap.add_argument("--v38-hidden", type=int, default=160)
    ap.add_argument("--v38-heads", type=int, default=4)
    ap.add_argument("--v38-layers", type=int, default=2)
    ap.add_argument("--v38-dropout", type=float, default=0.08)
    ap.add_argument("--v38-epochs", type=int, default=26)
    ap.add_argument("--v38-batch-size", type=int, default=256)
    ap.add_argument("--v38-eval-batch-size", type=int, default=512)
    ap.add_argument("--v38-lr", type=float, default=7e-4)
    ap.add_argument("--v38-weight-decay", type=float, default=1e-4)
    ap.add_argument("--v38-grad-clip", type=float, default=2.0)
    ap.add_argument("--v38-rmse-weight", type=float, default=0.75)
    ap.add_argument("--v38-kl-weight", type=float, default=1.0)
    ap.add_argument("--v38-ce-weight", type=float, default=0.50)
    ap.add_argument("--v38-rank-weight", type=float, default=0.20)
    ap.add_argument("--v38-entropy-weight", type=float, default=0.02)
    ap.add_argument("--v38-uncertainty-weight", type=float, default=0.02)
    ap.add_argument("--v38-huber-beta", type=float, default=1.5)
    ap.add_argument("--v38-rank-margin", type=float, default=1.0)
    ap.add_argument("--v38-entropy-floor", type=float, default=0.25)
    ap.add_argument("--v38-correction-scale", type=float, default=0.0)
    ap.add_argument("--v38-teacher-temperature", type=float, default=4.0)
    ap.add_argument("--v38-max-family-cols", type=int, default=96)
    ap.add_argument("--v38-include-all-context-token", type=int, default=1)
    ap.add_argument("--v38-top-m-grid", default="1,2,4,8,12")
    ap.add_argument("--v38-temperature-grid", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    # v39.
    ap.add_argument("--v39-variants", default="evidence_full,evidence_no_visual,evidence_zero,evidence_row_shuffled,evidence_same_frame_wrong_cell,evidence_time_shuffled,evidence_shape_only,evidence_polarity_only,evidence_contact_only,evidence_quality_only,evidence_no_contact,evidence_no_polarity")
    ap.add_argument("--v39-use-visual-evidence", type=int, default=1)
    ap.add_argument("--v39-use-context-cross-attention", type=int, default=1)
    ap.add_argument("--v39-use-candidate-self-attention", type=int, default=1)
    ap.add_argument("--v39-use-uncertainty", type=int, default=1)
    ap.add_argument("--v39-raw-visual-candidate-cols", type=int, default=32)
    ap.add_argument("--v39-cross-visual-cols", type=int, default=48)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
