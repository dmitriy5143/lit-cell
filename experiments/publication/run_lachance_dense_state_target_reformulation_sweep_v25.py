#!/usr/bin/env python3
"""Dense-state gate -> target reformulation sweep v25.

This runner is intentionally diagnostic rather than another scalar reranker.
It answers two questions:

1. Do dense visual/mask/state variables contain causal route signal after
   wrong-cell/time-shuffle controls?
2. If dense-state is weak, does a different target coordinate system improve
   final coordinate RMSE when reconstructed back to h1/h2/h4/h6 endpoints?

The target/future is allowed only for labels, route regimes and metrics.  All
inference feature packets are causal and train-normalized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_object_centric_mask_gate as objgate  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_target_formulation_audit_v2 as target_v2  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402
import run_lachance_visual_late_fusion_gate_v21 as v21  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "dense_state_target_v25_2026-07-04"
DEFAULT_FULL_FEATURES = audit.DEFAULT_FEATURES
DEFAULT_DENSE_FEATURES = (
    ROOT
    / "outputs"
    / "seg_tracking_foundation_v21_medium_downstream_bulk_seed42_2026-07-04"
    / "raw_context_v2_medium_visual_keys.csv"
)

OBJECT_GRID = (
    ROOT
    / "outputs"
    / "lachance_object_centric_mask_grid_bulk_seed42_2026-07-03"
    / "object_centric_mask_feature_grid.csv"
)
MULTISEED_DIR = ROOT / "outputs" / "multiseed_instance_mask_medium_bulk_seed42_2026-07-04"
TEMPORAL_DIR = ROOT / "outputs" / "temporal_mask_change_medium_bulk_seed42_2026-07-04"
SEG_FOUNDATION_DIR = ROOT / "outputs" / "seg_tracking_foundation_v21_medium_fusion_controls_bulk_seed42_2026-07-04"

KEY_COLS = ["dataset", "sequence", "frame", "track_id"]
EPS = 1e-8


@dataclass
class DenseSpec:
    family: str
    control: str
    path: Path
    prefix: str
    derived: str = ""


def parse_ints(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return audit.parse_ints(text)


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def safe_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def standardize_splits(
    xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xtr.shape[1] == 0:
        return xtr, xva, xte
    scaler = StandardScaler()
    ztr = scaler.fit_transform(safe_matrix(xtr)).astype(np.float32)
    zva = scaler.transform(safe_matrix(xva)).astype(np.float32)
    zte = scaler.transform(safe_matrix(xte)).astype(np.float32)
    return (
        np.clip(np.nan_to_num(ztr), -8.0, 8.0).astype(np.float32),
        np.clip(np.nan_to_num(zva), -8.0, 8.0).astype(np.float32),
        np.clip(np.nan_to_num(zte), -8.0, 8.0).astype(np.float32),
    )


def endpoint_residual_rmse(true: np.ndarray, pred: np.ndarray, horizons: list[int]) -> float:
    vals = []
    for h in horizons:
        p = np.sum(pred[:, : int(h), :], axis=1)
        y = np.sum(true[:, : int(h), :], axis=1)
        vals.append(np.sum((p - y) ** 2, axis=-1))
    return float(np.sqrt(np.mean(np.stack(vals, axis=1))))


def endpoint_direction_cos(true: np.ndarray, pred: np.ndarray, h: int) -> float:
    p = np.sum(pred[:, : int(h), :], axis=1)
    y = np.sum(true[:, : int(h), :], axis=1)
    den = np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1), EPS)
    return float(np.mean(np.sum(p * y, axis=1) / den))


def split_keys(split_df: pd.DataFrame) -> pd.DataFrame:
    return split_df[KEY_COLS].reset_index(drop=True)


def select_dense_columns(grid: pd.DataFrame, prefix: str, max_cols: int) -> list[str]:
    cols = [c for c in grid.columns if c.startswith(prefix) and c not in KEY_COLS + ["split"]]
    if not cols:
        return []
    if max_cols <= 0 or len(cols) <= int(max_cols):
        return cols
    x = audit.safe_matrix(grid, cols)
    var = np.nan_to_num(np.var(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-int(max_cols) :]
    return [cols[int(i)] for i in keep]


def load_dense_split(
    grid_path: Path,
    prefix: str,
    split: audit.seq.SplitData,
    *,
    max_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not grid_path.exists():
        raise FileNotFoundError(grid_path)
    grid = pd.read_csv(grid_path)
    missing = [c for c in KEY_COLS if c not in grid.columns]
    if missing:
        raise RuntimeError(f"{grid_path} misses key columns: {missing}")
    cols = select_dense_columns(grid, prefix, max_cols)
    if not cols:
        raise RuntimeError(f"{grid_path} has no columns with prefix {prefix!r}")
    use = grid[KEY_COLS + cols].drop_duplicates(KEY_COLS)

    def merge(df: pd.DataFrame) -> tuple[np.ndarray, float]:
        merged = split_keys(df).merge(use, on=KEY_COLS, how="left", indicator=True)
        coverage = float(np.mean(merged["_merge"].eq("both"))) if len(merged) else 0.0
        return audit.safe_matrix(merged, cols), coverage

    xtr, ctr = merge(split.train)
    xva, cva = merge(split.val)
    xte, cte = merge(split.test)
    meta = {
        "path": str(grid_path),
        "prefix": prefix,
        "n_cols": int(len(cols)),
        "coverage_train": ctr,
        "coverage_val": cva,
        "coverage_test": cte,
    }
    return xtr.astype(np.float32), xva.astype(np.float32), xte.astype(np.float32), meta


def make_derived_control(
    mats: tuple[np.ndarray, np.ndarray, np.ndarray],
    split: audit.seq.SplitData,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtr, xva, xte = mats
    if control == "real":
        return mats
    if control == "zero":
        return (np.zeros_like(xtr), np.zeros_like(xva), np.zeros_like(xte))
    if control == "row_shuffled":
        rng = np.random.default_rng(seed)
        return (
            xtr[rng.permutation(len(xtr))],
            xva[np.random.default_rng(seed + 1).permutation(len(xva))],
            xte[np.random.default_rng(seed + 2).permutation(len(xte))],
        )
    if control == "same_frame_wrong_cell":
        return (
            objgate.same_frame_wrong_cell(xtr, split.train, seed + 11),
            objgate.same_frame_wrong_cell(xva, split.val, seed + 12),
            objgate.same_frame_wrong_cell(xte, split.test, seed + 13),
        )
    if control == "time_shuffled":
        return (
            objgate.time_shuffle(xtr, split.train, seed + 21),
            objgate.time_shuffle(xva, split.val, seed + 22),
            objgate.time_shuffle(xte, split.test, seed + 23),
        )
    raise ValueError(f"unknown derived dense control: {control}")


def dense_specs(args: argparse.Namespace) -> list[DenseSpec]:
    specs: list[DenseSpec] = []
    requested = set(parse_strs(args.dense_families))
    if "object" in requested:
        for ctrl in ["real", "zero", "row_shuffled", "same_frame_wrong_cell", "time_shuffled"]:
            specs.append(DenseSpec("object_mask", ctrl, args.object_grid, "oc_", derived=ctrl))
    if "multiseed" in requested:
        specs.extend(
            [
                DenseSpec("multiseed_mask", "real", MULTISEED_DIR / "multiseed_instance_mask_feature_grid.csv", "mi_"),
                DenseSpec("multiseed_mask", "same_frame_wrong_cell", MULTISEED_DIR / "multiseed_instance_mask_sameframe_wrong_grid.csv", "mi_"),
                DenseSpec("multiseed_mask", "time_shuffled", MULTISEED_DIR / "multiseed_instance_mask_time_shuffled_grid.csv", "mi_"),
            ]
        )
    if "temporal" in requested:
        specs.extend(
            [
                DenseSpec("temporal_mask", "real", TEMPORAL_DIR / "multiseed_instance_mask_feature_grid.csv", "mi_"),
                DenseSpec("temporal_mask", "same_frame_wrong_cell", TEMPORAL_DIR / "multiseed_instance_mask_sameframe_wrong_grid.csv", "mi_"),
                DenseSpec("temporal_mask", "time_shuffled", TEMPORAL_DIR / "multiseed_instance_mask_time_shuffled_grid.csv", "mi_"),
                DenseSpec("temporal_mask", "temporal_shuffled", TEMPORAL_DIR / "multiseed_instance_mask_temporal_shuffled_grid.csv", "mi_"),
            ]
        )
    if "seg_foundation" in requested:
        specs.extend(
            [
                DenseSpec("seg_foundation", "real", SEG_FOUNDATION_DIR / "visual_real_feature_grid.csv", "segf_"),
                DenseSpec("seg_foundation", "zero", SEG_FOUNDATION_DIR / "visual_zero_feature_grid.csv", "segf_"),
                DenseSpec("seg_foundation", "row_shuffled", SEG_FOUNDATION_DIR / "visual_shuffled_feature_grid.csv", "segf_"),
                DenseSpec("seg_foundation", "wrong_cell", SEG_FOUNDATION_DIR / "visual_wrong_cell_feature_grid.csv", "segf_"),
                DenseSpec("seg_foundation", "same_frame_wrong_cell", SEG_FOUNDATION_DIR / "visual_same_frame_wrong_cell_feature_grid.csv", "segf_"),
                DenseSpec("seg_foundation", "time_shuffled", SEG_FOUNDATION_DIR / "visual_time_shuffled_feature_grid.csv", "segf_"),
            ]
        )
    return specs


def fit_dense_probe(
    *,
    variant: str,
    family: str,
    control: str,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    labels: dict[str, Any],
    arrays: audit.SplitArrays,
    horizons: list[int],
    seed: int,
    feature_meta: dict[str, Any],
) -> dict[str, Any]:
    xtr_s, xva_s, xte_s = standardize_splits(xtr, xva, xte)
    k = int(labels["k"])
    clf = LogisticRegression(max_iter=700, C=0.35, class_weight="balanced", random_state=seed + 25001)
    clf.fit(xtr_s, labels["train"])
    proba_raw = clf.predict_proba(xte_s)
    proba = np.full((len(xte_s), k), 1e-7, dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        proba[:, int(cls)] = proba_raw[:, j]
    proba /= np.maximum(proba.sum(axis=1, keepdims=True), EPS)

    ytr = arrays.residual_train.reshape(len(arrays.residual_train), -1)
    yva = arrays.residual_val.reshape(len(arrays.residual_val), -1)
    best_alpha = 0.0
    best_val = float("inf")
    best_model: Ridge | None = None
    for alpha in [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]:
        reg = Ridge(alpha=float(alpha))
        reg.fit(xtr_s, ytr)
        pred_va = reg.predict(xva_s).reshape(arrays.residual_val.shape).astype(np.float32)
        score = endpoint_residual_rmse(arrays.residual_val, pred_va, horizons)
        if score < best_val:
            best_val = score
            best_alpha = float(alpha)
            best_model = reg
    assert best_model is not None
    pred = best_model.predict(xte_s).reshape(arrays.residual_test.shape).astype(np.float32)
    try:
        nll = float(log_loss(labels["test"], np.clip(proba, 1e-7, 1.0), labels=np.arange(k)))
    except Exception:
        nll = float("nan")
    out = {
        "stage": "A_dense_state_gate",
        "variant": variant,
        "family": family,
        "control": control,
        "feature_dim": int(xtr.shape[1]),
        "route_k": k,
        "route_top1": float(np.mean(np.argmax(proba, axis=1) == labels["test"])),
        "route_top3": float(top_k_accuracy_score(labels["test"], proba, k=min(3, k), labels=np.arange(k))),
        "route_nll": nll,
        "residual_endpoint_rmse": endpoint_residual_rmse(arrays.residual_test, pred, horizons),
        "residual_hmax_cos": endpoint_direction_cos(arrays.residual_test, pred, max(horizons)),
        "ridge_alpha": best_alpha,
        "val_endpoint_rmse": best_val,
    }
    out.update(feature_meta)
    return out


def stage_a_dense_gate(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, audit.SplitArrays, audit.seq.SplitData]:
    dense_args = argparse.Namespace(**vars(args))
    dense_args.features = args.dense_features
    dense_args.max_train_rows = args.dense_max_train_rows
    dense_args.max_val_rows = args.dense_max_val_rows
    dense_args.max_test_rows = args.dense_max_test_rows
    dense_args.max_horizon = max(parse_ints(args.horizons))
    dense_args.horizons = args.horizons
    dense_args.train_seq = args.train_seq
    dense_args.val_seq = args.val_seq
    dense_args.test_seq = args.test_seq
    arrays, split = audit.prepare_data(dense_args)
    horizons = parse_ints(args.horizons)
    labels = objgate.route_labels(arrays, args.v25_route_k, args.seed)
    velocity_blocks, _ = v10.build_velocity_blocks(split, max_cols=args.v25_velocity_max_cols)
    ctx = (arrays.x_train["all_context"], arrays.x_val["all_context"], arrays.x_test["all_context"])
    vel = velocity_blocks["all"]
    base_variants: list[tuple[str, str, str, tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]] = [
        ("context_only", "coordinate", "real", ctx, {"coverage_train": 1.0, "coverage_val": 1.0, "coverage_test": 1.0}),
        ("velocity_only", "velocity", "real", vel, {"coverage_train": 1.0, "coverage_val": 1.0, "coverage_test": 1.0}),
        (
            "context_plus_velocity",
            "coordinate_velocity",
            "real",
            tuple(np.concatenate([a, b], axis=1) for a, b in zip(ctx, vel, strict=False)),
            {"coverage_train": 1.0, "coverage_val": 1.0, "coverage_test": 1.0},
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, family, control, mats, meta in base_variants:
        rows.append(
            fit_dense_probe(
                variant=name,
                family=family,
                control=control,
                xtr=mats[0],
                xva=mats[1],
                xte=mats[2],
                labels=labels,
                arrays=arrays,
                horizons=horizons,
                seed=args.seed,
                feature_meta=meta,
            )
        )

    loaded_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for spec in dense_specs(args):
        try:
            key = (str(spec.path), spec.prefix)
            if key not in loaded_cache:
                loaded_cache[key] = load_dense_split(spec.path, spec.prefix, split, max_cols=args.dense_max_cols)
            raw_tr, raw_va, raw_te, meta = loaded_cache[key]
            mats = (raw_tr, raw_va, raw_te)
            if spec.derived:
                mats = make_derived_control(mats, split, spec.derived, args.seed + abs(hash(spec.family)) % 10000)
            variants = [
                (f"{spec.family}_{spec.control}_only", mats),
                (f"context_plus_{spec.family}_{spec.control}", tuple(np.concatenate([a, b], axis=1) for a, b in zip(ctx, mats, strict=False))),
                (
                    f"context_velocity_{spec.family}_{spec.control}",
                    tuple(np.concatenate([a, b, c], axis=1) for a, b, c in zip(ctx, vel, mats, strict=False)),
                ),
            ]
            for name, mats2 in variants:
                rows.append(
                    fit_dense_probe(
                        variant=name,
                        family=spec.family,
                        control=spec.control,
                        xtr=mats2[0],
                        xva=mats2[1],
                        xte=mats2[2],
                        labels=labels,
                        arrays=arrays,
                        horizons=horizons,
                        seed=args.seed,
                        feature_meta=meta,
                    )
                )
        except Exception as exc:
            rows.append(
                {
                    "stage": "A_dense_state_gate",
                    "variant": f"{spec.family}_{spec.control}_ERROR",
                    "family": spec.family,
                    "control": spec.control,
                    "error": repr(exc),
                    "path": str(spec.path),
                }
            )
    dense = pd.DataFrame(rows)
    dense.to_csv(out_dir / "dense_state_gate.csv", index=False)
    return dense, arrays, split


def make_v16_args(base_args: argparse.Namespace, out_dir: Path, *, extra: DenseSpec | None = None, features: Path | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    qrc.add_common_args(parser)
    v21.add_v16_args(parser)
    ns = parser.parse_args([])
    ns.features = Path(features or base_args.dense_features)
    ns.table_root = base_args.table_root
    ns.dataset = base_args.dataset
    ns.out_dir = out_dir
    ns.seed = int(base_args.seed)
    ns.train_seq = base_args.train_seq
    ns.val_seq = base_args.val_seq
    ns.test_seq = base_args.test_seq
    ns.max_horizon = max(parse_ints(base_args.horizons))
    ns.horizons = base_args.horizons
    ns.max_train_rows = int(base_args.generator_max_train_rows)
    ns.max_val_rows = int(base_args.generator_max_val_rows)
    ns.max_test_rows = int(base_args.generator_max_test_rows)
    ns.max_features_per_family = int(base_args.max_features_per_family)
    ns.max_all_features = int(base_args.max_all_features)
    ns.posterior_epochs = int(base_args.generator_posterior_epochs)
    ns.student_epochs = int(base_args.generator_student_epochs)
    ns.learned_route_epochs = int(base_args.generator_learned_route_epochs)
    ns.candidate_k = int(base_args.generator_candidate_k)
    ns.oracle_k = base_args.generator_oracle_k
    ns.device = base_args.device
    ns.v10_velocity_max_cols = int(base_args.v25_velocity_max_cols)
    ns.v12_route_k = int(base_args.v25_route_k)
    ns.v12_prior_model = base_args.generator_prior_model
    ns.v16c_generator_variant = base_args.generator_variant
    ns.v16c_base_mixes = base_args.generator_base_mixes
    ns.v16c_calibrators = base_args.generator_calibrators
    ns.v16c_max_context_features = int(base_args.generator_max_context_features)
    ns.extra_feature_grid = None
    ns.extra_feature_prefixes = ""
    ns.extra_feature_block_name = "none"
    ns.extra_feature_max_cols = int(base_args.dense_max_cols)
    ns.extra_feature_merge_all_context = "false"
    if extra is not None:
        ns.extra_feature_grid = Path(extra.path)
        ns.extra_feature_prefixes = extra.prefix
        ns.extra_feature_block_name = f"{extra.family}_{extra.control}"
        ns.extra_feature_merge_all_context = "true"
    return ns


def stage_b_generator_conditioning(args: argparse.Namespace, out_dir: Path, dense_gate: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    variant_specs: list[tuple[str, DenseSpec | None]] = [("v16_no_dense", None)]
    real_candidates = dense_gate[
        dense_gate["variant"].astype(str).str.startswith("context_velocity_")
        & dense_gate["control"].astype(str).eq("real")
        & dense_gate["route_top3"].notna()
    ].copy()
    if "coverage_test" in real_candidates:
        real_candidates = real_candidates[
            pd.to_numeric(real_candidates["coverage_test"], errors="coerce").fillna(0.0).ge(float(args.dense_min_coverage_for_generator))
        ].copy()
    if not real_candidates.empty:
        top_families = (
            real_candidates.sort_values(["residual_endpoint_rmse", "route_top3"], ascending=[True, False])["family"]
            .drop_duplicates()
            .head(args.generator_dense_top_families)
        )
        for family in top_families:
            match = [s for s in dense_specs(args) if s.family == family and s.control == "real"]
            ctrl = [s for s in dense_specs(args) if s.family == family and s.control in {"same_frame_wrong_cell", "time_shuffled", "wrong_cell", "row_shuffled"}]
            if match:
                variant_specs.append((f"v16_dense_{family}_real", match[0]))
            if ctrl:
                variant_specs.append((f"v16_dense_{family}_{ctrl[0].control}", ctrl[0]))
    else:
        for spec in dense_specs(args):
            if spec.control == "real":
                variant_specs.append((f"v16_dense_{spec.family}_real", spec))
                break

    for name, spec in variant_specs[: int(args.generator_max_variants)]:
        run_dir = out_dir / f"stage_b_{name}"
        try:
            v16_args = make_v16_args(args, run_dir, extra=spec, features=args.dense_features)
            v16.run(v16_args)
            summary_path = run_dir / "route_balanced_calibrator_v16_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df.insert(0, "dense_variant", name)
                df.insert(1, "dense_family", spec.family if spec is not None else "none")
                df.insert(2, "dense_control", spec.control if spec is not None else "none")
                rows.append(df)
        except Exception as exc:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "dense_variant": name,
                            "dense_family": spec.family if spec is not None else "none",
                            "dense_control": spec.control if spec is not None else "none",
                            "stage": "B_generator_conditioning",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=3),
                        }
                    ]
                )
            )
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_dir / "generator_conditioning_ablation.csv", index=False)
    return out


def stage_c_target_reformulation(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    targs = argparse.Namespace(
        features=args.features,
        table_root=args.table_root,
        out_dir=out_dir / "stage_c_target_v2",
        dataset=args.dataset,
        seed=int(args.seed),
        train_sequences=args.train_seq,
        val_sequences=args.val_seq,
        test_sequences=args.test_seq,
        horizons=args.horizons,
        models=args.target_models,
        blocks=args.target_blocks,
        basis_kinds=args.target_basis_kinds,
        form_regex=args.target_form_regex,
        max_train_rows=int(args.target_max_train_rows),
        max_val_rows=int(args.target_max_val_rows),
        max_test_rows=int(args.target_max_test_rows),
        controls=args.target_controls,
        include_controls=bool(args.target_include_controls),
        route_probe=bool(args.target_route_probe),
        route_reprs=args.target_route_reprs,
        route_ks=args.target_route_ks,
        stratify=bool(args.target_stratify),
        stratify_forms=args.target_stratify_forms,
        stratify_blocks=args.target_stratify_blocks,
    )
    summary, route_probe, stratified = target_v2.run(targs)
    summary.to_csv(out_dir / "target_reformulation_metrics.csv", index=False)
    route_probe.to_csv(out_dir / "target_reformulation_route_probe.csv", index=False)
    stratified.to_csv(out_dir / "target_reformulation_stratified.csv", index=False)
    return summary, route_probe


def dense_family_decisions(dense: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if dense.empty or "family" not in dense.columns:
        return pd.DataFrame(rows)
    for family, sub in dense.groupby("family", dropna=False):
        sub = sub.copy()
        real = sub[sub["control"].eq("real") & sub["variant"].astype(str).str.startswith("context_velocity_")]
        controls = sub[~sub["control"].eq("real") & sub["variant"].astype(str).str.startswith("context_velocity_")]
        if real.empty:
            continue
        best_real = real.sort_values(["route_top3", "residual_endpoint_rmse"], ascending=[False, True]).iloc[0]
        best_control_top3 = float(controls["route_top3"].max()) if not controls.empty and "route_top3" in controls else float("nan")
        best_control_rmse = float(controls["residual_endpoint_rmse"].min()) if not controls.empty and "residual_endpoint_rmse" in controls else float("nan")
        route_gain = float(best_real["route_top3"] - best_control_top3) if math.isfinite(best_control_top3) else float("nan")
        rmse_gain = (
            float((best_control_rmse - best_real["residual_endpoint_rmse"]) / max(abs(best_control_rmse), EPS) * 100.0)
            if math.isfinite(best_control_rmse)
            else float("nan")
        )
        rows.append(
            {
                "stage": "A_dense_decision",
                "family": family,
                "best_real_variant": best_real["variant"],
                "best_real_route_top3": float(best_real["route_top3"]),
                "best_control_route_top3": best_control_top3,
                "route_top3_delta_vs_best_control": route_gain,
                "best_real_residual_endpoint_rmse": float(best_real["residual_endpoint_rmse"]),
                "best_control_residual_endpoint_rmse": best_control_rmse,
                "rmse_gain_vs_best_control_pct": rmse_gain,
                "dense_gate_pass": bool((math.isfinite(route_gain) and route_gain >= 0.03) or (math.isfinite(rmse_gain) and rmse_gain >= 1.0)),
            }
        )
    return pd.DataFrame(rows)


def generator_decision(gen: pd.DataFrame) -> pd.DataFrame:
    if gen.empty or "horizon" not in gen.columns or "rmse" not in gen.columns:
        return pd.DataFrame()
    hmax = int(pd.to_numeric(gen["horizon"], errors="coerce").max())
    sub = gen[pd.to_numeric(gen["horizon"], errors="coerce").eq(hmax)].copy()
    if sub.empty:
        return pd.DataFrame()
    best = sub.sort_values("rmse").groupby("dense_variant").head(1)
    nod = best[best["dense_variant"].eq("v16_no_dense")]
    base_rmse = float(nod.iloc[0]["rmse"]) if not nod.empty else float("nan")
    rows = []
    for _, row in best.iterrows():
        rmse = float(row["rmse"])
        rows.append(
            {
                "stage": "B_generator_decision",
                "dense_variant": row["dense_variant"],
                "dense_family": row.get("dense_family", ""),
                "dense_control": row.get("dense_control", ""),
                "horizon": hmax,
                "best_method": row["method"],
                "rmse": rmse,
                "r2": float(row.get("r2", float("nan"))),
                "gain_vs_no_dense_pct": float((base_rmse - rmse) / max(abs(base_rmse), EPS) * 100.0) if math.isfinite(base_rmse) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse")


def target_decision(target: pd.DataFrame) -> pd.DataFrame:
    if target.empty:
        return pd.DataFrame()
    hmax = int(pd.to_numeric(target["horizon"], errors="coerce").max())
    sub = target[target["horizon"].eq(hmax) & target["control"].eq("real")].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [
        "target_form",
        "target_family",
        "feature_block",
        "model",
        "rmse_px",
        "base_rmse_px",
        "gain_vs_base_pct",
        "r2",
        "cosine",
        "magnitude_ratio",
    ]
    top = sub.sort_values("rmse_px").head(20)[[c for c in cols if c in sub.columns]].copy()
    top.insert(0, "stage", "C_target_decision")
    top.insert(1, "horizon", hmax)
    return top


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    dense: pd.DataFrame,
    gen: pd.DataFrame,
    target: pd.DataFrame,
    dense_decision: pd.DataFrame,
    gen_decision: pd.DataFrame,
    target_top: pd.DataFrame,
    errors: list[dict[str, Any]],
) -> None:
    lines: list[str] = ["# Dense-State Gate -> Target Reformulation v25", ""]
    lines += [
        f"- dataset: `{args.dataset}`",
        f"- seed: `{args.seed}`",
        f"- coordinate features: `{args.features}`",
        f"- dense/aligned features: `{args.dense_features}`",
        f"- rows dense train/val/test: `{args.dense_max_train_rows}/{args.dense_max_val_rows}/{args.dense_max_test_rows}`",
        "",
    ]
    lines.append("## Stage A: Dense-State Gate")
    if not dense_decision.empty:
        lines.append(dense_decision.to_markdown(index=False))
    else:
        lines.append("_No dense decision rows._")
    lines.append("")
    if not dense.empty and "route_top3" in dense.columns:
        cols = [c for c in ["variant", "family", "control", "route_top1", "route_top3", "route_nll", "residual_endpoint_rmse", "coverage_test"] if c in dense.columns]
        lines.append("Top dense probes:")
        lines.append(dense.dropna(subset=["route_top3"]).sort_values("route_top3", ascending=False)[cols].head(18).to_markdown(index=False))
        lines.append("")

    lines.append("## Stage B: Dense-State -> Generator Conditioning")
    if not gen_decision.empty:
        lines.append(gen_decision.to_markdown(index=False))
    else:
        lines.append("_No generator conditioning rows or stage skipped._")
    lines.append("")

    lines.append("## Stage C: Target Reformulation")
    if not target_top.empty:
        lines.append(target_top.to_markdown(index=False))
    else:
        lines.append("_No target reformulation rows._")
    lines.append("")

    dense_pass = bool((not dense_decision.empty) and dense_decision["dense_gate_pass"].fillna(False).any())
    target_internal_pass = bool((not target_top.empty) and float(target_top.iloc[0].get("gain_vs_base_pct", 0.0)) >= float(args.target_soft_pass_pct))
    target_clean_pass = bool(
        (not target_top.empty)
        and math.isfinite(float(args.clean_reference_h6_rmse))
        and float(target_top.iloc[0].get("rmse_px", float("inf"))) <= float(args.clean_reference_h6_rmse) * (1.0 - float(args.target_clean_pass_pct) / 100.0)
    )
    gen_pass = bool((not gen_decision.empty) and gen_decision["gain_vs_no_dense_pct"].fillna(-999).max() >= float(args.generator_soft_pass_pct))
    lines.append("## Decision")
    lines.append(f"- dense-state usable as causal state: `{dense_pass}`")
    lines.append(f"- dense-state improves generator conditioning: `{gen_pass}`")
    lines.append(f"- target reformulation improves its internal reference: `{target_internal_pass}`")
    lines.append(f"- target reformulation beats clean-best h6 reference: `{target_clean_pass}`")
    if dense_pass:
        lines.append("- Keep dense-state active, but only the families that beat wrong/time controls; do not inject all visual tokens blindly.")
    else:
        lines.append("- Dense-state remains diagnostic/reserve: real features did not cleanly beat hard controls enough to be trusted as causal conditioning.")
    if target_clean_pass:
        lines.append("- Continue target-first architecture as a quality route: build the next model around the winning target view, then reconstruct endpoints.")
    elif target_internal_pass:
        lines.append("- Target reformulation is useful diagnostically, but not yet a clean-best quality route; do not replace the current backbone with it.")
    else:
        lines.append("- Target reformulation did not deliver a large coordinate-RMSE break in this sweep; the next route is stronger observability/data or a more explicit state target.")
    if errors:
        lines.append("")
        lines.append("## Errors / Skipped Pieces")
        for err in errors:
            lines.append(f"- `{err.get('stage')}`: `{err.get('error')}`")
    (out_dir / "dense_state_target_v25_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.horizons = ",".join(map(str, parse_ints(args.horizons)))
    errors: list[dict[str, Any]] = []
    dense = pd.DataFrame()
    dense_decision_df = pd.DataFrame()
    gen = pd.DataFrame()
    gen_decision_df = pd.DataFrame()
    target = pd.DataFrame()
    target_top = pd.DataFrame()

    try:
        dense, _, _ = stage_a_dense_gate(args, args.out_dir)
        dense_decision_df = dense_family_decisions(dense)
    except Exception as exc:
        errors.append({"stage": "A_dense_state_gate", "error": repr(exc), "traceback": traceback.format_exc(limit=8)})

    if not args.skip_generator_conditioning:
        try:
            gen = stage_b_generator_conditioning(args, args.out_dir, dense)
            gen_decision_df = generator_decision(gen)
        except Exception as exc:
            errors.append({"stage": "B_generator_conditioning", "error": repr(exc), "traceback": traceback.format_exc(limit=8)})
            gen = pd.DataFrame(errors[-1:])
            gen.to_csv(args.out_dir / "generator_conditioning_ablation.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.out_dir / "generator_conditioning_ablation.csv", index=False)

    if not args.skip_target_reformulation:
        try:
            target, _ = stage_c_target_reformulation(args, args.out_dir)
            target_top = target_decision(target)
        except Exception as exc:
            errors.append({"stage": "C_target_reformulation", "error": repr(exc), "traceback": traceback.format_exc(limit=8)})
            target = pd.DataFrame(errors[-1:])
            target.to_csv(args.out_dir / "target_reformulation_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.out_dir / "target_reformulation_metrics.csv", index=False)

    summary_parts = [df for df in [dense_decision_df, gen_decision_df, target_top] if not df.empty]
    summary = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    summary.to_csv(args.out_dir / "dense_state_target_v25_summary.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    if errors:
        (args.out_dir / "errors.json").write_text(json.dumps(finite_json(errors), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, dense, gen, target, dense_decision_df, gen_decision_df, target_top, errors)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "errors": len(errors)}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FULL_FEATURES)
    parser.add_argument("--dense-features", type=Path, default=DEFAULT_DENSE_FEATURES)
    parser.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seq", default="1,2,3,4")
    parser.add_argument("--val-seq", default="5")
    parser.add_argument("--test-seq", default="6")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--max-features-per-family", type=int, default=160)
    parser.add_argument("--max-all-features", type=int, default=384)
    parser.add_argument("--dense-max-train-rows", type=int, default=3000)
    parser.add_argument("--dense-max-val-rows", type=int, default=1000)
    parser.add_argument("--dense-max-test-rows", type=int, default=1500)
    parser.add_argument("--dense-max-cols", type=int, default=192)
    parser.add_argument("--dense-families", default="object,multiseed,temporal,seg_foundation")
    parser.add_argument("--dense-min-coverage-for-generator", type=float, default=0.25)
    parser.add_argument("--object-grid", type=Path, default=OBJECT_GRID)
    parser.add_argument("--v25-route-k", type=int, default=12)
    parser.add_argument("--v25-velocity-max-cols", type=int, default=160)
    parser.add_argument("--skip-generator-conditioning", action="store_true")
    parser.add_argument("--generator-max-train-rows", type=int, default=3000)
    parser.add_argument("--generator-max-val-rows", type=int, default=1000)
    parser.add_argument("--generator-max-test-rows", type=int, default=1500)
    parser.add_argument("--generator-posterior-epochs", type=int, default=8)
    parser.add_argument("--generator-student-epochs", type=int, default=8)
    parser.add_argument("--generator-learned-route-epochs", type=int, default=6)
    parser.add_argument("--generator-candidate-k", type=int, default=32)
    parser.add_argument("--generator-oracle-k", default="8,16,32")
    parser.add_argument("--generator-variant", default="context_velocity")
    parser.add_argument("--generator-prior-model", default="logistic", choices=["logistic", "hgbdt"])
    parser.add_argument("--generator-base-mixes", default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    parser.add_argument("--generator-calibrators", default="correction_context,stacked_context")
    parser.add_argument("--generator-max-context-features", type=int, default=384)
    parser.add_argument("--generator-dense-top-families", type=int, default=2)
    parser.add_argument("--generator-max-variants", type=int, default=4)
    parser.add_argument("--generator-soft-pass-pct", type=float, default=1.0)
    parser.add_argument("--skip-target-reformulation", action="store_true")
    parser.add_argument("--target-models", default="ridge")
    parser.add_argument("--target-blocks", default="trajectory_only,obs_context_core,ms_shape_tf_alignment_rc_core,ms_all_tf_all_rc,rc_all")
    parser.add_argument("--target-basis-kinds", default="self,flow64,flow128,flow256,boundary")
    parser.add_argument("--target-form-regex", default="")
    parser.add_argument("--target-max-train-rows", type=int, default=16000)
    parser.add_argument("--target-max-val-rows", type=int, default=5000)
    parser.add_argument("--target-max-test-rows", type=int, default=6000)
    parser.add_argument("--target-controls", default="")
    parser.add_argument("--target-include-controls", action="store_true")
    parser.add_argument("--target-route-probe", action="store_true")
    parser.add_argument("--target-route-reprs", default="global_residual,self_frame_residual,flow128_frame_residual,boundary_frame_residual,shape_unit")
    parser.add_argument("--target-route-ks", default="8,16")
    parser.add_argument("--target-stratify", action="store_true")
    parser.add_argument("--target-stratify-forms", default="global_residual,self_frame,flow128_frame,boundary_frame,overcomplete")
    parser.add_argument("--target-stratify-blocks", default="ms_all_tf_all_rc,ms_shape_tf_alignment_rc_core")
    parser.add_argument("--target-soft-pass-pct", type=float, default=2.0)
    parser.add_argument("--clean-reference-h6-rmse", type=float, default=16.96)
    parser.add_argument("--target-clean-pass-pct", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.dense_max_train_rows = min(args.dense_max_train_rows, 1200)
        args.dense_max_val_rows = min(args.dense_max_val_rows, 400)
        args.dense_max_test_rows = min(args.dense_max_test_rows, 600)
        args.dense_max_cols = min(args.dense_max_cols, 96)
        args.generator_max_train_rows = min(args.generator_max_train_rows, 900)
        args.generator_max_val_rows = min(args.generator_max_val_rows, 300)
        args.generator_max_test_rows = min(args.generator_max_test_rows, 400)
        args.generator_posterior_epochs = min(args.generator_posterior_epochs, 3)
        args.generator_student_epochs = min(args.generator_student_epochs, 3)
        args.generator_learned_route_epochs = min(args.generator_learned_route_epochs, 3)
        args.generator_candidate_k = min(args.generator_candidate_k, 16)
        args.generator_oracle_k = "4,8,16"
        args.generator_max_variants = min(args.generator_max_variants, 2)
        args.target_max_train_rows = min(args.target_max_train_rows, 1200)
        args.target_max_val_rows = min(args.target_max_val_rows, 400)
        args.target_max_test_rows = min(args.target_max_test_rows, 600)
        args.target_blocks = "trajectory_only,ms_shape_tf_alignment_rc_core,ms_all_tf_all_rc"
        args.target_basis_kinds = "self,flow128,boundary"
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
