#!/usr/bin/env python3
"""Raw-state closure + route/state-conditioned architecture v26.

This runner closes two linked questions:

1. Do raw-video-derived state variables provide causal route observability after
   wrong-cell / time-shuffle / row-shuffle controls?
2. If yes or partly yes, can a causal regime router over a fixed v16/v12 route
   basis improve final h1/h2/h4/h6 prediction without repeating the failed v17
   fully trainable expert-bank path?

The target/future is allowed for training labels, route-oracle diagnostics and
losses only. Inference features are causal.
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
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_dense_state_target_reformulation_sweep_v25 as v25  # noqa: E402
import run_lachance_object_centric_mask_gate as objgate  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "raw_state_route_architecture_v26_2026-07-06"
DEFAULT_RAW_STACK = (
    ROOT
    / "new_data"
    / "lachance_epithelia"
    / "raw_timelapse"
    / "extracted_stacks"
    / "MDCK_Bulk_Timelapse_Data_Sample_Tissues"
)
KEY_COLS = ["dataset", "sequence", "frame", "track_id"]
EPS = 1e-8


@dataclass
class SplitMatrices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    meta: dict[str, Any]


@dataclass
class RouteBasis:
    arrays: audit.SplitArrays
    split: audit.seq.SplitData
    labels: Any
    prior: Any
    bank: Any
    route_train: np.ndarray
    route_val: np.ndarray
    route_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    oracle_labels_train: np.ndarray
    oracle_labels_val: np.ndarray
    oracle_labels_test: np.ndarray
    route_data_meta: dict[str, Any]


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return audit.parse_ints(text)


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in parse_strs(text)]


def safe_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def concat_parts(parts: list[np.ndarray]) -> np.ndarray:
    valid = [safe_matrix(p) for p in parts if p is not None and np.asarray(p).size > 0]
    if not valid:
        raise RuntimeError("empty feature packet")
    return np.concatenate(valid, axis=1).astype(np.float32)


def standardize_splits(
    xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(safe_matrix(xtr)).astype(np.float32)
    zva = scaler.transform(safe_matrix(xva)).astype(np.float32)
    zte = scaler.transform(safe_matrix(xte)).astype(np.float32)
    return (
        np.clip(np.nan_to_num(ztr), -9.0, 9.0).astype(np.float32),
        np.clip(np.nan_to_num(zva), -9.0, 9.0).astype(np.float32),
        np.clip(np.nan_to_num(zte), -9.0, 9.0).astype(np.float32),
        scaler,
    )


def padded_proba(clf: LogisticRegression, x: np.ndarray, k: int) -> np.ndarray:
    raw = clf.predict_proba(x)
    out = np.full((len(x), int(k)), 1e-7, dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        ci = int(cls)
        if 0 <= ci < k:
            out[:, ci] = raw[:, j]
    out /= np.maximum(out.sum(axis=1, keepdims=True), EPS)
    return out


def entropy_rows(weights: np.ndarray) -> dict[str, float]:
    ent = -np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1)
    active = np.sum(weights > 1e-4, axis=1)
    usage = weights.mean(axis=0)
    return {
        "route_entropy_mean": float(np.mean(ent)),
        "route_entropy_std": float(np.std(ent)),
        "active_routes_mean": float(np.mean(active)),
        "route_usage_entropy": float(-np.sum(usage * np.log(np.maximum(usage, EPS)))),
        "max_route_usage": float(np.max(usage)) if len(usage) else float("nan"),
    }


def residual_endpoint_rmse(true: np.ndarray, pred: np.ndarray, horizons: list[int]) -> float:
    errs = []
    for h in horizons:
        h = int(h)
        p = np.sum(pred[:, :h, :], axis=1)
        y = np.sum(true[:, :h, :], axis=1)
        errs.append(np.sum((p - y) ** 2, axis=-1))
    return float(np.sqrt(np.mean(np.stack(errs, axis=1))))


def route_oracle_labels(route_pred: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Best route expert per sample under multi-horizon endpoint error."""
    n, k, dim = route_pred.shape
    steps_pred = route_pred.reshape(n, k, args.max_horizon, 2)
    true = true_flat.reshape(n, args.max_horizon, 2)
    err = np.zeros((n, k), dtype=np.float64)
    for h in args.horizons:
        h = int(h)
        p = np.sum(steps_pred[:, :, :h, :], axis=2)
        y = np.sum(true[:, :h, :], axis=1)[:, None, :]
        err += np.sum((p - y) ** 2, axis=-1)
    return np.argmin(err, axis=1).astype(np.int64)


def topm_temperature_weights(
    probs: np.ndarray,
    *,
    top_m: int,
    temperature: float,
    entropy_blend: float = 0.0,
) -> np.ndarray:
    probs = np.maximum(safe_matrix(probs), 1e-8)
    n, r = probs.shape
    c = max(1, min(int(top_m), r))
    temp = max(float(temperature), 1e-3)
    order = np.argsort(-probs, axis=1)[:, :c]
    chosen = np.take_along_axis(probs, order, axis=1)
    logits = np.log(np.maximum(chosen, 1e-8)) / temp
    logits -= np.max(logits, axis=1, keepdims=True)
    vals = np.exp(logits)
    vals /= np.maximum(vals.sum(axis=1, keepdims=True), EPS)
    w = np.zeros_like(probs, dtype=np.float32)
    rows = np.arange(n)[:, None]
    w[rows, order] = vals.astype(np.float32)
    blend = float(entropy_blend)
    if blend > 0:
        w = (1.0 - blend) * w + blend * np.full_like(w, 1.0 / float(r), dtype=np.float32)
        w /= np.maximum(w.sum(axis=1, keepdims=True), EPS)
    return w.astype(np.float32)


def mix_routes(route_pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(route_pred * weights[:, :, None], axis=1).astype(np.float32)


def tune_weights(
    *,
    probs_train: np.ndarray,
    probs_val: np.ndarray,
    probs_test: np.ndarray,
    route_train: np.ndarray,
    route_val: np.ndarray,
    route_test: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    entropy_blends: list[float],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for top_m in parse_ints(args.v26_top_m_grid):
        for temp in parse_floats(args.v26_temperature_grid):
            for blend in entropy_blends:
                wva = topm_temperature_weights(probs_val, top_m=top_m, temperature=temp, entropy_blend=blend)
                pred_va = mix_routes(route_val, wva)
                rmse = v16.endpoint_rmse_flat(pred_va, y_val, args)
                if best is None or rmse < float(best["val_endpoint_rmse"]):
                    wtr = topm_temperature_weights(probs_train, top_m=top_m, temperature=temp, entropy_blend=blend)
                    wte = topm_temperature_weights(probs_test, top_m=top_m, temperature=temp, entropy_blend=blend)
                    best = {
                        "top_m": int(top_m),
                        "temperature": float(temp),
                        "entropy_blend": float(blend),
                        "val_endpoint_rmse": float(rmse),
                        "weights_train": wtr,
                        "weights_val": wva,
                        "weights_test": wte,
                        "mix_train": mix_routes(route_train, wtr),
                        "mix_val": pred_va,
                        "mix_test": mix_routes(route_test, wte),
                    }
    assert best is not None
    return best


def clip_correction(pred_corr: np.ndarray, train_target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scale = float(args.v26_correction_scale)
    if scale <= 0:
        return pred_corr.astype(np.float32)
    train_norm = np.linalg.norm(train_target.reshape(len(train_target), -1), axis=1)
    limit = float(scale * np.nanmedian(train_norm))
    if not math.isfinite(limit) or limit <= 0:
        return pred_corr.astype(np.float32)
    corr = pred_corr.astype(np.float32).copy()
    norm = np.linalg.norm(corr.reshape(len(corr), -1), axis=1)
    mult = np.minimum(1.0, limit / np.maximum(norm, EPS)).astype(np.float32)
    return (corr * mult[:, None]).astype(np.float32)


def fit_bounded_ridge_correction(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    base_tr: np.ndarray,
    base_va: np.ndarray,
    base_te: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(safe_matrix(xtr))
    zva = scaler.transform(safe_matrix(xva))
    zte = scaler.transform(safe_matrix(xte))
    best_alpha = float("nan")
    best_rmse = float("inf")
    best_model: Ridge | None = None
    for alpha in parse_floats(args.v26_ridge_alphas):
        model = Ridge(alpha=float(alpha))
        model.fit(ztr, ytr - base_tr)
        corr_va = clip_correction(model.predict(zva).astype(np.float32), ytr, args)
        pred_va = base_va + corr_va
        rmse = v16.endpoint_rmse_flat(pred_va, yva, args)
        if rmse < best_rmse:
            best_rmse = float(rmse)
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None
    corr_te = clip_correction(best_model.predict(zte).astype(np.float32), ytr, args)
    return (base_te + corr_te).astype(np.float32), best_alpha, best_rmse


def route_probe(
    *,
    variant: str,
    target_name: str,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    labels_train: np.ndarray,
    labels_val: np.ndarray,
    labels_test: np.ndarray,
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    feature_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    k = int(max(labels_train.max(initial=0), labels_val.max(initial=0), labels_test.max(initial=0)) + 1)
    xtr_s, xva_s, xte_s, _ = standardize_splits(xtr, xva, xte)
    clf = LogisticRegression(
        max_iter=int(args.v26_probe_max_iter),
        C=float(args.v26_probe_c),
        class_weight="balanced",
        random_state=int(args.seed) + 26001,
    )
    clf.fit(xtr_s, labels_train)
    pva = padded_proba(clf, xva_s, k)
    pte = padded_proba(clf, xte_s, k)

    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    best_alpha = float("nan")
    best_val = float("inf")
    best_model: Ridge | None = None
    for alpha in parse_floats(args.v26_ridge_alphas):
        model = Ridge(alpha=float(alpha))
        model.fit(xtr_s, ytr)
        pred_va = model.predict(xva_s).astype(np.float32)
        rmse = v16.endpoint_rmse_flat(pred_va, yva, args)
        if rmse < best_val:
            best_val = float(rmse)
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None
    pred_te = best_model.predict(xte_s).reshape(arrays.residual_test.shape).astype(np.float32)
    try:
        nll = float(log_loss(labels_test, np.clip(pte, 1e-7, 1.0), labels=np.arange(k)))
    except Exception:
        nll = float("nan")
    out: dict[str, Any] = {
        "stage": "B_regime_encoder_probe",
        "variant": variant,
        "target": target_name,
        "feature_dim": int(xtr.shape[1]),
        "route_k": k,
        "val_route_top1": float(np.mean(np.argmax(pva, axis=1) == labels_val)),
        "route_top1": float(np.mean(np.argmax(pte, axis=1) == labels_test)),
        "route_top3": float(top_k_accuracy_score(labels_test, pte, k=min(3, k), labels=np.arange(k))),
        "route_nll": nll,
        "residual_endpoint_rmse": residual_endpoint_rmse(arrays.residual_test, pred_te, args.horizons),
        "ridge_alpha": best_alpha,
        "val_endpoint_rmse": best_val,
    }
    if feature_meta:
        out.update(feature_meta)
    return out


def preflight(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rows = []
    paths = {
        "features": args.features,
        "dense_features": args.dense_features,
        "raw_stack": args.raw_stack,
        "v16_reference": args.v16_reference_csv,
    }
    for name, path in paths.items():
        p = Path(path)
        rows.append(
            {
                "item": name,
                "path": str(p),
                "exists": bool(p.exists()),
                "is_dir": bool(p.is_dir()),
                "file_count": int(sum(1 for _ in p.glob("*"))) if p.exists() and p.is_dir() else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "v26_preflight.csv", index=False)
    return out


def make_v16_args(args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    ns = v25.make_v16_args(args, out_dir, extra=None, features=args.dense_features)
    ns.horizons = parse_ints(ns.horizons)
    ns.oracle_k = parse_ints(ns.oracle_k)
    ns.max_horizon = max(ns.horizons)
    return ns


def build_route_basis(args: argparse.Namespace, out_dir: Path) -> RouteBasis:
    """Build fixed v16/v12 expert trajectories once and keep them fixed."""
    base_args = make_v16_args(args, out_dir / "stage_route_basis")
    base_args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, labels, prior, bank, _packs, gate, meta = v16.build_route_data(base_args, device)
    split_arrays, split = audit.prepare_data(base_args)
    if len(split_arrays.residual_train) != len(arrays.residual_train):
        raise RuntimeError("prepare_data mismatch while building v26 route basis")

    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    yte = audit.flatten_residual(arrays.residual_test).astype(np.float32)
    rtr = v16.route_outputs(bank, prior.x_train)
    rva = v16.route_outputs(bank, prior.x_val)
    rte = v16.route_outputs(bank, prior.x_test)

    oracle_tr = route_oracle_labels(rtr, ytr, base_args)
    oracle_va = route_oracle_labels(rva, yva, base_args)
    oracle_te = route_oracle_labels(rte, yte, base_args)
    route_data_meta = {
        "v16_gate": gate.to_dict(orient="records") if isinstance(gate, pd.DataFrame) else [],
        "v16_meta": meta,
        "route_count": int(rtr.shape[1]),
        "route_dim": int(rtr.shape[2]),
        "feature_dim": int(prior.x_train.shape[1]),
    }
    (out_dir / "v26_route_basis_meta.json").write_text(
        json.dumps(audit.finite_json(route_data_meta), indent=2),
        encoding="utf-8",
    )
    return RouteBasis(
        arrays=arrays,
        split=split,
        labels=labels,
        prior=prior,
        bank=bank,
        route_train=rtr,
        route_val=rva,
        route_test=rte,
        y_train=ytr,
        y_val=yva,
        y_test=yte,
        oracle_labels_train=oracle_tr,
        oracle_labels_val=oracle_va,
        oracle_labels_test=oracle_te,
        route_data_meta=route_data_meta,
    )


def choose_dense_family(args: argparse.Namespace, dense_gate: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    decision = v25.dense_family_decisions(dense_gate)
    if decision.empty:
        return None, decision
    decision = decision.copy()
    decision["coverage_ok"] = True
    real_cov = dense_gate[
        dense_gate["control"].astype(str).eq("real")
        & dense_gate["variant"].astype(str).str.startswith("context_velocity_")
    ][["family", "coverage_test"]].copy()
    if not real_cov.empty:
        cov = real_cov.groupby("family")["coverage_test"].max()
        decision["coverage_test"] = decision["family"].map(cov).astype(float)
        decision["coverage_ok"] = decision["coverage_test"].ge(float(args.raw_state_min_coverage))
    decision["score"] = (
        decision["route_top3_delta_vs_best_control"].fillna(-999.0) * 10.0
        + decision["rmse_gain_vs_best_control_pct"].fillna(-999.0)
    )
    eligible = decision[decision["coverage_ok"]].sort_values("score", ascending=False)
    if eligible.empty:
        eligible = decision.sort_values("score", ascending=False)
    family = str(eligible.iloc[0]["family"]) if not eligible.empty else None
    return family, decision


def load_dense_family(args: argparse.Namespace, family: str | None, split: audit.seq.SplitData) -> dict[str, SplitMatrices]:
    if not family:
        return {}
    specs = [s for s in v25.dense_specs(args) if s.family == family]
    loaded: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    out: dict[str, SplitMatrices] = {}
    for spec in specs:
        try:
            key = (str(spec.path), spec.prefix)
            if key not in loaded:
                loaded[key] = v25.load_dense_split(spec.path, spec.prefix, split, max_cols=int(args.dense_max_cols))
            xtr, xva, xte, meta = loaded[key]
            mats = (xtr, xva, xte)
            if spec.derived:
                mats = v25.make_derived_control(mats, split, spec.derived, int(args.seed) + 26100)
            out[spec.control] = SplitMatrices(mats[0], mats[1], mats[2], {"family": spec.family, "control": spec.control, **meta})
        except Exception as exc:
            out[f"{spec.control}_ERROR"] = SplitMatrices(
                np.zeros((len(split.train), 0), dtype=np.float32),
                np.zeros((len(split.val), 0), dtype=np.float32),
                np.zeros((len(split.test), 0), dtype=np.float32),
                {"family": spec.family, "control": spec.control, "error": repr(exc)},
            )
    if "row_shuffled" not in out and "real" in out:
        mats = v25.make_derived_control(
            (out["real"].train, out["real"].val, out["real"].test),
            split,
            "row_shuffled",
            int(args.seed) + 26150,
        )
        out["row_shuffled"] = SplitMatrices(mats[0], mats[1], mats[2], {**out["real"].meta, "control": "row_shuffled_derived"})
    return out


def feature_packets(args: argparse.Namespace, basis: RouteBasis, dense: dict[str, SplitMatrices]) -> dict[str, SplitMatrices]:
    arrays = basis.arrays
    split = basis.split
    velocity, _names = v10.build_velocity_blocks(split, max_cols=int(args.v25_velocity_max_cols))
    ctx = SplitMatrices(arrays.x_train["all_context"], arrays.x_val["all_context"], arrays.x_test["all_context"], {"family": "coord", "control": "real"})
    vel = SplitMatrices(velocity["all"][0], velocity["all"][1], velocity["all"][2], {"family": "velocity", "control": "real"})

    self_parts = []
    for key in ["self", "flow", "quality"]:
        if key in arrays.x_train:
            self_parts.append((arrays.x_train[key], arrays.x_val[key], arrays.x_test[key]))
    if self_parts:
        nt = concat_parts([p[0] for p in self_parts])
        nv = concat_parts([p[1] for p in self_parts])
        ne = concat_parts([p[2] for p in self_parts])
    else:
        nt, nv, ne = ctx.train, ctx.val, ctx.test
    no_topology = SplitMatrices(nt, nv, ne, {"family": "coord_no_topology_proxy", "control": "real"})

    packets: dict[str, SplitMatrices] = {
        "coord_only": ctx,
        "coord_velocity": SplitMatrices(
            concat_parts([ctx.train, vel.train]),
            concat_parts([ctx.val, vel.val]),
            concat_parts([ctx.test, vel.test]),
            {"family": "coord_velocity", "control": "real"},
        ),
        "coord_velocity_topology": SplitMatrices(
            concat_parts([ctx.train, vel.train]),
            concat_parts([ctx.val, vel.val]),
            concat_parts([ctx.test, vel.test]),
            {"family": "coord_velocity_topology", "control": "real"},
        ),
        "no_topology_proxy": SplitMatrices(
            concat_parts([no_topology.train, vel.train]),
            concat_parts([no_topology.val, vel.val]),
            concat_parts([no_topology.test, vel.test]),
            {"family": "no_topology_proxy", "control": "real"},
        ),
    }
    if "real" in dense and dense["real"].train.shape[1] > 0:
        d = dense["real"]
        packets["coord_velocity_raw_state_real"] = SplitMatrices(
            concat_parts([ctx.train, vel.train, d.train]),
            concat_parts([ctx.val, vel.val, d.val]),
            concat_parts([ctx.test, vel.test, d.test]),
            d.meta,
        )
    for ctrl in ["row_shuffled", "wrong_cell", "same_frame_wrong_cell", "time_shuffled"]:
        if ctrl in dense and dense[ctrl].train.shape[1] > 0:
            d = dense[ctrl]
            packets[f"coord_velocity_raw_state_{ctrl}"] = SplitMatrices(
                concat_parts([ctx.train, vel.train, d.train]),
                concat_parts([ctx.val, vel.val, d.val]),
                concat_parts([ctx.test, vel.test, d.test]),
                d.meta,
            )
            break
    return packets


def run_raw_state_gate(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    dense_gate, _arrays, _split = v25.stage_a_dense_gate(args, out_dir)
    dense_gate.to_csv(out_dir / "v26_raw_state_gate.csv", index=False)
    best_family, decision = choose_dense_family(args, dense_gate)
    decision.to_csv(out_dir / "v26_raw_state_gate_decision.csv", index=False)
    return dense_gate, decision, best_family


def run_regime_probe(args: argparse.Namespace, basis: RouteBasis, packets: dict[str, SplitMatrices], out_dir: Path) -> pd.DataFrame:
    residual_labels = objgate.route_labels(basis.arrays, int(args.v25_route_k), int(args.seed))
    targets = [
        ("residual_signature", residual_labels["train"], residual_labels["val"], residual_labels["test"]),
        ("fixed_route_oracle", basis.oracle_labels_train, basis.oracle_labels_val, basis.oracle_labels_test),
    ]
    rows: list[dict[str, Any]] = []
    probe_variants = [
        "coord_only",
        "coord_velocity",
        "coord_velocity_topology",
        "no_topology_proxy",
        "coord_velocity_raw_state_real",
        "coord_velocity_raw_state_row_shuffled",
        "coord_velocity_raw_state_wrong_cell",
        "coord_velocity_raw_state_same_frame_wrong_cell",
        "coord_velocity_raw_state_time_shuffled",
    ]
    for variant in probe_variants:
        if variant not in packets:
            continue
        mats = packets[variant]
        for target_name, ytr, yva, yte in targets:
            try:
                rows.append(
                    route_probe(
                        variant=variant,
                        target_name=target_name,
                        xtr=mats.train,
                        xva=mats.val,
                        xte=mats.test,
                        labels_train=ytr,
                        labels_val=yva,
                        labels_test=yte,
                        arrays=basis.arrays,
                        args=args,
                        feature_meta=mats.meta,
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "stage": "B_regime_encoder_probe",
                        "variant": variant,
                        "target": target_name,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "v26_regime_encoder_probe.csv", index=False)
    return out


def classifier_probs(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    k: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtr_s, xva_s, xte_s, _ = standardize_splits(xtr, xva, xte)
    clf = LogisticRegression(
        max_iter=int(args.v26_router_max_iter),
        C=float(args.v26_router_c),
        class_weight="balanced",
        random_state=int(args.seed) + 26200,
    )
    clf.fit(xtr_s, ytr)
    return padded_proba(clf, xtr_s, k), padded_proba(clf, xva_s, k), padded_proba(clf, xte_s, k)


def add_metric_rows(
    rows: list[dict[str, Any]],
    basis: RouteBasis,
    pred_flat: np.ndarray,
    label: str,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> None:
    rows.extend(v16.metric_rows(basis.arrays, v16.flat_to_steps(pred_flat, args), label, args, extra))


def run_single_router_variant(
    *,
    name: str,
    feature_packet: SplitMatrices | None,
    probs_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    basis: RouteBasis,
    args: argparse.Namespace,
    entropy_blends: list[float],
    calibrate: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    k = int(basis.route_train.shape[1])
    if probs_override is not None:
        ptr, pva, pte = probs_override
        router_dim = int(ptr.shape[1])
        router_source = "fixed_prior"
        packet_train = basis.prior.x_train
        packet_val = basis.prior.x_val
        packet_test = basis.prior.x_test
    else:
        assert feature_packet is not None
        ptr, pva, pte = classifier_probs(
            xtr=feature_packet.train,
            xva=feature_packet.val,
            xte=feature_packet.test,
            ytr=basis.oracle_labels_train,
            k=k,
            args=args,
        )
        router_dim = int(feature_packet.train.shape[1])
        router_source = "oracle_label_logistic"
        packet_train = feature_packet.train
        packet_val = feature_packet.val
        packet_test = feature_packet.test

    tuned = tune_weights(
        probs_train=ptr,
        probs_val=pva,
        probs_test=pte,
        route_train=basis.route_train,
        route_val=basis.route_val,
        route_test=basis.route_test,
        y_val=basis.y_val,
        args=args,
        entropy_blends=entropy_blends,
    )
    pred = tuned["mix_test"]
    route_stats = entropy_rows(tuned["weights_test"])
    extra = {
        "stage": "C_v26_route_state_architecture",
        "variant": name,
        "router_source": router_source,
        "router_dim": router_dim,
        "top_m": tuned["top_m"],
        "temperature": tuned["temperature"],
        "entropy_blend": tuned["entropy_blend"],
        "val_endpoint_rmse": tuned["val_endpoint_rmse"],
        "calibrated": False,
        **route_stats,
    }
    rows: list[dict[str, Any]] = []
    add_metric_rows(rows, basis, pred, name, args, extra)

    diag = {
        "variant": name,
        "router_source": router_source,
        "router_dim": router_dim,
        "top_m": tuned["top_m"],
        "temperature": tuned["temperature"],
        "entropy_blend": tuned["entropy_blend"],
        "val_endpoint_rmse": tuned["val_endpoint_rmse"],
        **route_stats,
    }

    if calibrate:
        ftr = v16.build_features(
            route_pred=basis.route_train,
            probs=tuned["weights_train"],
            x=packet_train,
            mix=tuned["mix_train"],
            args=args,
            kind="stacked_top_context",
        )
        fva = v16.build_features(
            route_pred=basis.route_val,
            probs=tuned["weights_val"],
            x=packet_val,
            mix=tuned["mix_val"],
            args=args,
            kind="stacked_top_context",
        )
        fte = v16.build_features(
            route_pred=basis.route_test,
            probs=tuned["weights_test"],
            x=packet_test,
            mix=tuned["mix_test"],
            args=args,
            kind="stacked_top_context",
        )
        pred_cal, alpha, val_rmse = fit_bounded_ridge_correction(
            xtr=ftr,
            xva=fva,
            xte=fte,
            ytr=basis.y_train,
            yva=basis.y_val,
            base_tr=tuned["mix_train"],
            base_va=tuned["mix_val"],
            base_te=tuned["mix_test"],
            args=args,
        )
        cal_label = f"{name}_bounded_calibration"
        cal_extra = {
            **extra,
            "variant": cal_label,
            "calibrated": True,
            "alpha": alpha,
            "val_endpoint_rmse": val_rmse,
        }
        add_metric_rows(rows, basis, pred_cal, cal_label, args, cal_extra)
        diag[f"{cal_label}_alpha"] = alpha
        diag[f"{cal_label}_val_endpoint_rmse"] = val_rmse
    return rows, diag


def run_route_architecture(args: argparse.Namespace, basis: RouteBasis, packets: dict[str, SplitMatrices], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    k = int(basis.route_train.shape[1])

    uniform = np.full((len(basis.y_train), k), 1.0 / float(k), dtype=np.float32)
    uniform_val = np.full((len(basis.y_val), k), 1.0 / float(k), dtype=np.float32)
    uniform_test = np.full((len(basis.y_test), k), 1.0 / float(k), dtype=np.float32)

    # Fixed uniform calibrator: a strong v16-style reference on the same aligned rows.
    uniform_mix_tr = mix_routes(basis.route_train, uniform)
    uniform_mix_va = mix_routes(basis.route_val, uniform_val)
    uniform_mix_te = mix_routes(basis.route_test, uniform_test)
    xtr, xva, xte = v16.select_context(
        basis.prior.x_train,
        basis.prior.x_val,
        basis.prior.x_test,
        int(args.v16c_max_context_features),
    )
    ftr = v16.build_features(route_pred=basis.route_train, probs=uniform, x=xtr, mix=uniform_mix_tr, args=args, kind="stacked_top_context")
    fva = v16.build_features(route_pred=basis.route_val, probs=uniform_val, x=xva, mix=uniform_mix_va, args=args, kind="stacked_top_context")
    fte = v16.build_features(route_pred=basis.route_test, probs=uniform_test, x=xte, mix=uniform_mix_te, args=args, kind="stacked_top_context")
    pred_fixed, alpha, val_rmse = v16.fit_ridge_endpoint_predict(xtr=ftr, xva=fva, xte=fte, ytr=basis.y_train, yva=basis.y_val, args=args)
    add_metric_rows(
        rows,
        basis,
        pred_fixed,
        "v26_fixed_uniform_calibrator",
        args,
        {
            "stage": "C_v26_route_state_architecture",
            "variant": "v26_fixed_uniform_calibrator",
            "router_source": "uniform",
            "router_dim": int(xtr.shape[1]),
            "calibrated": True,
            "alpha": alpha,
            "val_endpoint_rmse": val_rmse,
            **entropy_rows(uniform_test),
        },
    )
    diag_rows.append({"variant": "v26_fixed_uniform_calibrator", "alpha": alpha, "val_endpoint_rmse": val_rmse, **entropy_rows(uniform_test)})

    # Prior-only / no-regime auxiliary: what v12/v16 already thought probable.
    variant_rows, diag = run_single_router_variant(
        name="v26_no_regime_aux_prior_router",
        feature_packet=None,
        probs_override=(basis.prior.probs_train, basis.prior.probs_val, basis.prior.probs_test),
        basis=basis,
        args=args,
        entropy_blends=[0.0],
        calibrate=True,
    )
    rows.extend(variant_rows)
    diag_rows.append(diag)

    router_variants = [
        ("v26_regime_router", "coord_velocity", [0.0], False),
        ("v26_regime_router_entropy", "coord_velocity", parse_floats(args.v26_entropy_blend_grid), False),
        ("v26_regime_router_calibrated", "coord_velocity", parse_floats(args.v26_entropy_blend_grid), True),
        ("v26_no_velocity", "coord_only", [0.0], True),
        ("v26_no_topology", "no_topology_proxy", [0.0], True),
        ("v26_regime_router_raw_state", "coord_velocity_raw_state_real", parse_floats(args.v26_entropy_blend_grid), True),
        ("v26_raw_state_shuffled", "coord_velocity_raw_state_row_shuffled", parse_floats(args.v26_entropy_blend_grid), True),
        ("v26_raw_state_wrong_or_time_control", "coord_velocity_raw_state_wrong_cell", [0.0], True),
        ("v26_raw_state_wrong_or_time_control", "coord_velocity_raw_state_same_frame_wrong_cell", [0.0], True),
        ("v26_raw_state_wrong_or_time_control", "coord_velocity_raw_state_time_shuffled", [0.0], True),
    ]
    seen_names: set[str] = set()
    for name, packet_name, blends, calibrate in router_variants:
        if packet_name not in packets:
            continue
        # Keep only one hard raw-state control if several are available.
        final_name = name
        if name in seen_names:
            continue
        seen_names.add(name)
        variant_rows, diag = run_single_router_variant(
            name=final_name,
            feature_packet=packets[packet_name],
            probs_override=None,
            basis=basis,
            args=args,
            entropy_blends=blends,
            calibrate=calibrate,
        )
        rows.extend(variant_rows)
        diag_rows.append(diag)

    metrics = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    metrics.insert(0, "seed", int(args.seed))
    metrics.insert(0, "dataset", str(args.dataset))
    if not diag_df.empty:
        diag_df.insert(0, "seed", int(args.seed))
        diag_df.insert(0, "dataset", str(args.dataset))
    metrics.to_csv(out_dir / "v26_route_mixture_metrics.csv", index=False)
    diag_df.to_csv(out_dir / "v26_route_mixture_diagnostics.csv", index=False)
    return metrics, diag_df


def architecture_decision(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    if metrics.empty or "horizon" not in metrics.columns:
        return pd.DataFrame(rows)
    hmax = max(args.horizons)
    sub = metrics[pd.to_numeric(metrics["horizon"], errors="coerce").eq(hmax)].copy()
    if sub.empty:
        return pd.DataFrame(rows)
    best = sub.sort_values("rmse").groupby("method", as_index=False).head(1).sort_values("rmse")
    clean_ref = float(args.clean_reference_h6_rmse)
    best_row = best.iloc[0]
    no_reg = best[best["method"].astype(str).str.contains("no_regime_aux", regex=False)]
    raw = best[best["method"].astype(str).str.contains("raw_state") & ~best["method"].astype(str).str.contains("shuffled")]
    raw_ctrl = best[best["method"].astype(str).str.contains("raw_state_shuffled|wrong_or_time", regex=True)]
    rows.append(
        {
            "stage": "D_v26_architecture_decision",
            "decision_item": "best_hmax",
            "horizon": hmax,
            "best_method": best_row["method"],
            "best_rmse": float(best_row["rmse"]),
            "best_r2": float(best_row.get("r2", float("nan"))),
            "gain_vs_clean_reference_pct": float((clean_ref - float(best_row["rmse"])) / max(abs(clean_ref), EPS) * 100.0),
            "hard_pass_h6_le_16": bool(float(best_row["rmse"]) <= float(args.v26_hard_h6_rmse)),
            "soft_pass_vs_clean_ref": bool((clean_ref - float(best_row["rmse"])) / max(abs(clean_ref), EPS) * 100.0 >= float(args.v26_soft_gain_pct)),
        }
    )
    if not no_reg.empty:
        nr = no_reg.sort_values("rmse").iloc[0]
        rows.append(
            {
                "stage": "D_v26_architecture_decision",
                "decision_item": "best_no_regime_aux",
                "horizon": hmax,
                "best_method": nr["method"],
                "best_rmse": float(nr["rmse"]),
                "delta_best_minus_no_regime": float(float(best_row["rmse"]) - float(nr["rmse"])),
            }
        )
    if not raw.empty and not raw_ctrl.empty:
        rr = raw.sort_values("rmse").iloc[0]
        rc = raw_ctrl.sort_values("rmse").iloc[0]
        rows.append(
            {
                "stage": "D_v26_architecture_decision",
                "decision_item": "raw_state_vs_control",
                "horizon": hmax,
                "raw_state_method": rr["method"],
                "raw_state_rmse": float(rr["rmse"]),
                "control_method": rc["method"],
                "control_rmse": float(rc["rmse"]),
                "raw_state_better_than_control": bool(float(rr["rmse"]) < float(rc["rmse"])),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    preflight_df: pd.DataFrame,
    dense_gate: pd.DataFrame,
    dense_decision: pd.DataFrame,
    selected_family: str | None,
    regime_probe: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    errors: list[dict[str, Any]],
) -> None:
    lines: list[str] = ["# v26 Raw-State Route/Architecture Sweep", ""]
    lines.append("## Setup")
    lines.append(f"- dataset: `{args.dataset}`, seed: `{args.seed}`")
    lines.append(f"- coordinate features: `{args.features}`")
    lines.append(f"- dense/aligned features: `{args.dense_features}`")
    lines.append(f"- selected raw-state family: `{selected_family or 'none'}`")
    lines.append("")

    lines.append("## Preflight")
    lines.append(preflight_df.to_markdown(index=False))
    lines.append("")

    if not dense_decision.empty:
        lines.append("## Stage A: Raw-State Gate")
        cols = [
            c
            for c in [
                "family",
                "best_real_variant",
                "best_real_route_top3",
                "best_control_route_top3",
                "route_top3_delta_vs_best_control",
                "best_real_residual_endpoint_rmse",
                "best_control_residual_endpoint_rmse",
                "rmse_gain_vs_best_control_pct",
                "coverage_test",
                "dense_gate_pass",
            ]
            if c in dense_decision.columns
        ]
        lines.append(dense_decision[cols].sort_values(cols[-2] if "coverage_test" in cols else cols[0], ascending=False).to_markdown(index=False))
        lines.append("")
    elif not dense_gate.empty:
        lines.append("## Stage A: Raw-State Gate")
        lines.append("No family-level decision table was produced; see `v26_raw_state_gate.csv`.")
        lines.append("")

    if not regime_probe.empty:
        lines.append("## Stage B: Causal Regime Encoder Probe")
        sub = regime_probe[regime_probe["target"].eq("fixed_route_oracle")].copy()
        if sub.empty:
            sub = regime_probe.copy()
        cols = [c for c in ["variant", "target", "route_top3", "route_nll", "residual_endpoint_rmse", "feature_dim", "family", "control"] if c in sub.columns]
        lines.append(sub.sort_values(["route_top3", "residual_endpoint_rmse"], ascending=[False, True])[cols].head(30).to_markdown(index=False))
        lines.append("")

    if not metrics.empty:
        lines.append("## Stage C: v26 Route/State Architecture")
        for h in args.horizons:
            sub = metrics[metrics["horizon"].eq(int(h))].sort_values("rmse")
            cols = [
                c
                for c in [
                    "method",
                    "rmse",
                    "r2",
                    "angular_cosine",
                    "magnitude_ratio",
                    "variant",
                    "top_m",
                    "temperature",
                    "entropy_blend",
                    "calibrated",
                    "route_entropy_mean",
                    "max_route_usage",
                ]
                if c in sub.columns
            ]
            lines.append(f"### h{h}")
            lines.append(sub[cols].head(40).to_markdown(index=False))
            lines.append("")

    if not ablation.empty:
        lines.append("## Decision")
        lines.append(ablation.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Decision")
        lines.append("No architecture decision could be computed.")
        lines.append("")

    if errors:
        lines.append("## Errors")
        for err in errors[:10]:
            lines.append(f"- `{err.get('stage', 'unknown')}`: {err.get('error', '')}")
        lines.append("")

    lines.append("## Interpretation Rules")
    lines.append("- Raw-state is accepted only if real beats wrong/time/row controls with enough coverage.")
    lines.append("- v26 is accepted only if fixed route basis + causal router/calibration beats the v16 reference without collapsed route entropy.")
    lines.append("- If route oracle stays strong but router cannot approach it, the limit is causal observability rather than candidate construction.")
    lines.append("")
    (out_dir / "v26_decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []

    preflight_df = preflight(args, args.out_dir)

    try:
        dense_gate, dense_decision, selected_family = run_raw_state_gate(args, args.out_dir)
    except Exception as exc:
        errors.append({"stage": "A_raw_state_gate", "error": repr(exc), "traceback": traceback.format_exc()})
        dense_gate = pd.DataFrame()
        dense_decision = pd.DataFrame()
        selected_family = None

    try:
        basis = build_route_basis(args, args.out_dir)
    except Exception as exc:
        errors.append({"stage": "route_basis", "error": repr(exc), "traceback": traceback.format_exc()})
        (args.out_dir / "errors.json").write_text(json.dumps(audit.finite_json(errors), indent=2), encoding="utf-8")
        write_report(args.out_dir, args, preflight_df, dense_gate, dense_decision, selected_family, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), errors)
        raise

    try:
        dense_mats = load_dense_family(args, selected_family, basis.split)
        packets = feature_packets(args, basis, dense_mats)
    except Exception as exc:
        errors.append({"stage": "feature_packets", "error": repr(exc), "traceback": traceback.format_exc()})
        dense_mats = {}
        packets = feature_packets(args, basis, dense_mats)

    try:
        regime_probe = run_regime_probe(args, basis, packets, args.out_dir)
    except Exception as exc:
        errors.append({"stage": "B_regime_probe", "error": repr(exc), "traceback": traceback.format_exc()})
        regime_probe = pd.DataFrame()

    try:
        metrics, diag = run_route_architecture(args, basis, packets, args.out_dir)
    except Exception as exc:
        errors.append({"stage": "C_route_architecture", "error": repr(exc), "traceback": traceback.format_exc()})
        metrics = pd.DataFrame()
        diag = pd.DataFrame()

    ablation = architecture_decision(metrics, args)
    if not ablation.empty:
        ablation.to_csv(args.out_dir / "v26_ablation.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.out_dir / "v26_ablation.csv", index=False)

    summary_parts = []
    if not dense_decision.empty:
        summary_parts.append(dense_decision.assign(summary_stage="raw_state"))
    if not ablation.empty:
        summary_parts.append(ablation.assign(summary_stage="architecture"))
    summary = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    summary.to_csv(args.out_dir / "v26_raw_state_route_architecture_summary.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    if errors:
        (args.out_dir / "errors.json").write_text(json.dumps(audit.finite_json(errors), indent=2), encoding="utf-8")

    write_report(args.out_dir, args, preflight_df, dense_gate, dense_decision, selected_family, regime_probe, metrics, ablation, errors)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "selected_raw_state_family": selected_family,
                "dense_rows": int(len(dense_gate)),
                "probe_rows": int(len(regime_probe)),
                "metric_rows": int(len(metrics)),
                "errors": int(len(errors)),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=v25.DEFAULT_FULL_FEATURES)
    ap.add_argument("--dense-features", type=Path, default=v25.DEFAULT_DENSE_FEATURES)
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
    ap.add_argument("--raw-stack", type=Path, default=DEFAULT_RAW_STACK)
    ap.add_argument("--v16-reference-csv", type=Path, default=ROOT / "outputs" / "v16_route_balanced_calibrator_full60k_3seed_aggregate_2026-07-03.csv")

    # Dense-state gate arguments mirror v25 so we can reuse its controlled loader.
    ap.add_argument("--dense-max-train-rows", type=int, default=3000)
    ap.add_argument("--dense-max-val-rows", type=int, default=1000)
    ap.add_argument("--dense-max-test-rows", type=int, default=1500)
    ap.add_argument("--dense-max-cols", type=int, default=192)
    ap.add_argument("--dense-families", default="object,multiseed,temporal,seg_foundation")
    ap.add_argument("--dense-min-coverage-for-generator", type=float, default=0.25)
    ap.add_argument("--raw-state-min-coverage", type=float, default=0.80)
    ap.add_argument("--object-grid", type=Path, default=v25.OBJECT_GRID)

    # Route-basis / v16-compatible settings.
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
    ap.add_argument("--generator-dense-top-families", type=int, default=2)
    ap.add_argument("--generator-max-variants", type=int, default=4)
    ap.add_argument("--v25-route-k", type=int, default=12)
    ap.add_argument("--v25-velocity-max-cols", type=int, default=160)

    # v16 fields used directly by v26 feature builders.
    ap.add_argument("--v16c-generator-variant", default="context_velocity")
    ap.add_argument("--v16c-top-c", type=int, default=8)
    ap.add_argument("--v16c-max-context-features", type=int, default=384)
    ap.add_argument("--v16c-ridge-alphas", default="0.1,0.3,1,3,10,30,100,300,1000,3000")

    # v26 router/calibrator.
    ap.add_argument("--v26-top-m-grid", default="1,2,4,8,12")
    ap.add_argument("--v26-temperature-grid", default="0.35,0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--v26-entropy-blend-grid", default="0.0,0.03,0.07,0.12")
    ap.add_argument("--v26-router-c", type=float, default=0.45)
    ap.add_argument("--v26-router-max-iter", type=int, default=900)
    ap.add_argument("--v26-probe-c", type=float, default=0.35)
    ap.add_argument("--v26-probe-max-iter", type=int, default=800)
    ap.add_argument("--v26-ridge-alphas", default="0.1,0.3,1,3,10,30,100,300,1000,3000")
    ap.add_argument("--v26-correction-scale", type=float, default=0.85)
    ap.add_argument("--clean-reference-h6-rmse", type=float, default=16.96)
    ap.add_argument("--v26-hard-h6-rmse", type=float, default=16.0)
    ap.add_argument("--v26-soft-gain-pct", type=float, default=3.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

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
        args.v26_top_m_grid = "1,2,4,8"
        args.v26_temperature_grid = "0.5,1.0,1.5"
        args.v26_entropy_blend_grid = "0.0,0.07"
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
