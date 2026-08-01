#!/usr/bin/env python3
"""Route-balanced expert calibrator v16b.

The first v16-from-scratch MLP route experts were unstable.  This runner uses
the already useful v12 Ridge route experts as fixed generators, then trains only
lightweight calibrated mixture/correction models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_conditioned_generator_v12 as v12  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_cluster_order_selector_v14 as v14  # noqa: E402
import run_lachance_cluster_mixture_generator_v15 as v15  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_balanced_calibrator_v16_2026-07-03"
EPS = 1e-8


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(s) for s in parse_strs(text)]


def metric_rows(arrays: audit.SplitArrays, pred_steps: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred_steps,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def flat_to_steps(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return x.reshape(len(x), args.max_horizon, 2).astype(np.float32)


def endpoint_rmse_flat(pred: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> float:
    pred_steps = flat_to_steps(pred, args)
    true_steps = flat_to_steps(true_flat, args)
    vals = []
    for h in args.horizons:
        p = np.sum(pred_steps[:, :h], axis=1)
        y = np.sum(true_steps[:, :h], axis=1)
        vals.append(np.mean(np.sum((p - y) ** 2, axis=1)))
    return float(np.sqrt(np.mean(vals)))


def top_weights(probs: np.ndarray, top_c: int, mode: str, power: float = 1.0) -> np.ndarray:
    n, r = probs.shape
    if mode == "all_uniform":
        return np.full((n, r), 1.0 / float(r), dtype=np.float32)
    c = max(1, min(int(top_c), r))
    order = np.argsort(-probs, axis=1)[:, :c]
    weights = np.zeros_like(probs, dtype=np.float32)
    if mode == "uniform":
        vals = np.full((n, c), 1.0 / float(c), dtype=np.float32)
    elif mode == "prior":
        vals = np.take_along_axis(probs, order, axis=1).astype(np.float64)
        vals = np.power(np.maximum(vals, 1e-8), float(power))
        vals = vals / np.maximum(np.sum(vals, axis=1, keepdims=True), EPS)
        vals = vals.astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    rows = np.arange(n)[:, None]
    weights[rows, order] = vals
    return weights


def mix_route_outputs(route_pred: np.ndarray, probs: np.ndarray, top_c: int, mode: str, power: float = 1.0) -> np.ndarray:
    w = top_weights(probs, top_c, mode, power)
    return np.sum(w[:, :, None] * route_pred, axis=1).astype(np.float32)


def route_outputs(bank: v12.ExpertBank, x: np.ndarray) -> np.ndarray:
    preds = []
    for model in bank.models:
        preds.append(model.predict(x).astype(np.float32))
    return np.stack(preds, axis=1).astype(np.float32)


def select_context(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray, max_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xtr.shape[1] <= int(max_cols):
        return xtr, xva, xte
    var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-int(max_cols) :]
    return xtr[:, keep], xva[:, keep], xte[:, keep]


def prior_stats(probs: np.ndarray) -> np.ndarray:
    order = np.sort(probs, axis=1)[:, ::-1]
    ent = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=1, keepdims=True)
    margin = (order[:, :1] - order[:, 1:2]) if probs.shape[1] > 1 else order[:, :1]
    return np.concatenate([probs, order[:, : min(4, probs.shape[1])], ent, margin], axis=1).astype(np.float32)


def sorted_top_route_outputs(route_pred: np.ndarray, probs: np.ndarray, top_c: int) -> np.ndarray:
    n, r, d = route_pred.shape
    c = max(1, min(int(top_c), r))
    order = np.argsort(-probs, axis=1)[:, :c]
    return route_pred[np.arange(n)[:, None], order].reshape(n, c * d).astype(np.float32)


def build_features(
    *,
    route_pred: np.ndarray,
    probs: np.ndarray,
    x: np.ndarray,
    mix: np.ndarray,
    args: argparse.Namespace,
    kind: str,
) -> np.ndarray:
    parts = [mix.astype(np.float32), prior_stats(probs)]
    if kind in {"correction_context", "stacked_context", "stacked_top_context"}:
        parts.append(x.astype(np.float32))
    if kind in {"stacked", "stacked_context"}:
        parts.append(route_pred.reshape(len(route_pred), -1).astype(np.float32))
    if kind in {"stacked_top", "stacked_top_context"}:
        parts.append(sorted_top_route_outputs(route_pred, probs, args.v16c_top_c))
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_ridge_predict(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, float, float]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(xtr)
    zva = scaler.transform(xva)
    zte = scaler.transform(xte)
    best_alpha = float(alphas[0])
    best_rmse = float("inf")
    best_model = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(ztr, ytr)
        pred = model.predict(zva).astype(np.float32)
        rmse = float(np.sqrt(np.mean((pred - yva) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None
    return best_model.predict(zte).astype(np.float32), best_alpha, best_rmse


def fit_ridge_endpoint_predict(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(xtr)
    zva = scaler.transform(xva)
    zte = scaler.transform(xte)
    best_alpha = None
    best_rmse = float("inf")
    best_model = None
    for alpha in parse_floats(args.v16c_ridge_alphas):
        model = Ridge(alpha=float(alpha))
        model.fit(ztr, ytr)
        pred = model.predict(zva).astype(np.float32)
        rmse = endpoint_rmse_flat(pred, yva, args)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None and best_alpha is not None
    return best_model.predict(zte).astype(np.float32), best_alpha, best_rmse


def build_route_data(args: argparse.Namespace, device: Any) -> tuple[audit.SplitArrays, v12.RouteLabels, v12.RoutePrior, v12.ExpertBank, dict[str, seq.CandidatePack], pd.DataFrame, dict[str, Any]]:
    arrays, split = audit.prepare_data(args)
    extra_meta = rpr.attach_extra_feature_block(arrays, split, args)
    velocity_blocks, velocity_names = v10.build_velocity_blocks(split, max_cols=args.v10_velocity_max_cols)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    decomp = v12.decomposition_features(student, arrays, blocks, args, device)
    labels = v12.fit_route_labels(arrays, args)
    xtr_raw, xva_raw, xte_raw, names = v12.build_route_feature_matrix(
        arrays=arrays,
        split=split,
        velocity_blocks=velocity_blocks,
        decomp=decomp,
        variant=args.v16c_generator_variant,
        args=args,
    )
    prior = v12.fit_prior_model(name=args.v16c_generator_variant, xtr_raw=xtr_raw, xva_raw=xva_raw, xte_raw=xte_raw, labels=labels, args=args, feature_names=names)
    bank = v12.fit_expert_bank(prior, labels, arrays, args)
    packs = {
        "train": v12.generate_expert_candidates(name=args.v16c_generator_variant, prior=prior, bank=bank, probs=prior.probs_train, x=prior.x_train, residual_true=arrays.residual_train, arrays_base=arrays.base_train, args=args, split_name="train"),
        "val": v12.generate_expert_candidates(name=args.v16c_generator_variant, prior=prior, bank=bank, probs=prior.probs_val, x=prior.x_val, residual_true=arrays.residual_val, arrays_base=arrays.base_val, args=args, split_name="val"),
        "test": v12.generate_expert_candidates(name=args.v16c_generator_variant, prior=prior, bank=bank, probs=prior.probs_test, x=prior.x_test, residual_true=arrays.residual_test, arrays_base=arrays.base_test, args=args, split_name="test"),
    }
    gate = pd.DataFrame(v12.prior_gate_rows(prior, labels))
    meta = {"extra_feature": extra_meta, "velocity_names": velocity_names, "route_k": labels.k, "expert_meta": bank.meta, "feature_dim": prior.feature_dim}
    return arrays, labels, prior, bank, packs, gate, meta


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    args.horizons = audit.parse_ints(args.horizons)
    args.oracle_k = audit.parse_ints(args.oracle_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, labels, prior, bank, packs, gate, meta = build_route_data(args, device)

    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    yte = audit.flatten_residual(arrays.residual_test).astype(np.float32)
    rtr = route_outputs(bank, prior.x_train)
    rva = route_outputs(bank, prior.x_val)
    rte = route_outputs(bank, prior.x_test)
    xtr, xva, xte = select_context(prior.x_train, prior.x_val, prior.x_test, args.v16c_max_context_features)

    rows: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    rows.extend(metric_rows(arrays, seq.mean_candidate_residual(packs["test"]), "v12_candidate_mean", args, {"stage": "candidate_mean"}))
    for k in args.oracle_k:
        kk = min(int(k), args.candidate_k)
        rows.extend(metric_rows(arrays, seq.oracle_residual(packs["test"], arrays.residual_test, kk), f"v12_oracle@{kk}", args, {"stage": "candidate_oracle", "oracle_k": kk}))
    cl = v14.make_cluster_pack(packs["test"], arrays.residual_test, arrays.base_test, args, method="route", rep="medoid", cluster_count=8)
    rows.extend(metric_rows(arrays, v15.weighted_cluster_residual(cl, v15.cluster_prior_weights(packs["test"], cl, 8, "uniform")), "v15_route_medoid_c8_uniform_mix", args, {"stage": "v15_fixed_route_mix"}))

    fixed_specs = [
        ("expert_top4_uniform", 4, "uniform", 1.0),
        ("expert_top8_uniform", 8, "uniform", 1.0),
        ("expert_top8_prior_p050", 8, "prior", 0.5),
        ("expert_top8_prior_p100", 8, "prior", 1.0),
        ("expert_all_uniform", 999, "all_uniform", 1.0),
    ]
    mixes = {}
    for name, top_c, mode, power in fixed_specs:
        pred = mix_route_outputs(rte, prior.probs_test, top_c, mode, power)
        rows.extend(metric_rows(arrays, flat_to_steps(pred, args), f"v16c_{name}", args, {"stage": "v16c_fixed_expert_mix", "variant": name}))
        mixes[name] = (
            mix_route_outputs(rtr, prior.probs_train, top_c, mode, power),
            mix_route_outputs(rva, prior.probs_val, top_c, mode, power),
            pred,
        )

    for base_name in parse_strs(args.v16c_base_mixes):
        if base_name not in mixes:
            continue
        mtr, mva, mte = mixes[base_name]
        for kind in parse_strs(args.v16c_calibrators):
            ftr = build_features(route_pred=rtr, probs=prior.probs_train, x=xtr, mix=mtr, args=args, kind=kind)
            fva = build_features(route_pred=rva, probs=prior.probs_val, x=xva, mix=mva, args=args, kind=kind)
            fte = build_features(route_pred=rte, probs=prior.probs_test, x=xte, mix=mte, args=args, kind=kind)
            if kind.startswith("correction"):
                pred_corr, alpha, val_rmse = fit_ridge_endpoint_predict(xtr=ftr, xva=fva, xte=fte, ytr=ytr - mtr, yva=yva - mva, args=args)
                pred = mte + pred_corr
                label = f"v16c_{base_name}_{kind}_ridge_correction"
            else:
                pred, alpha, val_rmse = fit_ridge_endpoint_predict(xtr=ftr, xva=fva, xte=fte, ytr=ytr, yva=yva, args=args)
                label = f"v16c_{base_name}_{kind}_ridge_stack"
            rows.extend(metric_rows(arrays, flat_to_steps(pred, args), label, args, {"stage": "v16c_calibrated_expert_mix", "variant": label, "alpha": alpha, "val_rmse": val_rmse}))
            diag.append({"variant": label, "base_mix": base_name, "calibrator": kind, "alpha": alpha, "val_endpoint_rmse": val_rmse})

    summary = pd.DataFrame(rows)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    diag_df = pd.DataFrame(diag)
    if not diag_df.empty:
        diag_df.insert(0, "seed", int(args.seed))
        diag_df.insert(0, "dataset", str(args.dataset))
    if not gate.empty:
        gate.insert(0, "seed", int(args.seed))
        gate.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "route_balanced_calibrator_v16_summary.csv", index=False)
    diag_df.to_csv(args.out_dir / "route_balanced_calibrator_v16_diagnostics.csv", index=False)
    gate.to_csv(args.out_dir / "route_balanced_calibrator_v16_prior_gate.csv", index=False)
    (args.out_dir / "route_balanced_calibrator_v16_meta.json").write_text(json.dumps(audit.finite_json(meta), indent=2), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag_df, gate)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diag_rows": len(diag_df)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, gate: pd.DataFrame) -> None:
    lines = ["# v16 Route-Balanced Calibrator", ""]
    if not gate.empty:
        lines.append("## Route Prior Gate")
        lines.append(gate[gate["split"].eq("test")].to_markdown(index=False))
        lines.append("")
    for h in args.horizons:
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "alpha", "val_rmse", "oracle_k"] if c in sub.columns]
        lines.append(f"## h{h}")
        lines.append(sub[cols].head(60).to_markdown(index=False))
        lines.append("")
    if not diag.empty:
        lines.append("## Diagnostics")
        lines.append(diag.sort_values("val_endpoint_rmse").head(30).to_markdown(index=False))
    (out_dir / "route_balanced_calibrator_v16_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(out_dir=DEFAULT_OUT)
    parser.add_argument("--extra-feature-grid", type=Path, default=v12.DEFAULT_OBJECT_GRID)
    parser.add_argument("--extra-feature-prefixes", type=str, default="oc_")
    parser.add_argument("--extra-feature-block-name", type=str, default="object_mask")
    parser.add_argument("--extra-feature-max-cols", type=int, default=256)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--v10-velocity-max-cols", type=int, default=160)
    parser.add_argument("--v12-route-k", type=int, default=12)
    parser.add_argument("--v12-min-route-cluster-size", type=int, default=40)
    parser.add_argument("--v12-prior-model", type=str, default="logistic", choices=["logistic", "hgbdt"])
    parser.add_argument("--v12-prior-max-iter", type=int, default=500)
    parser.add_argument("--v12-prior-c", type=float, default=0.35)
    parser.add_argument("--v12-hgbdt-iter", type=int, default=160)
    parser.add_argument("--v12-hgbdt-lr", type=float, default=0.05)
    parser.add_argument("--v12-hgbdt-leaf-nodes", type=int, default=31)
    parser.add_argument("--v12-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v12-max-route-features", type=int, default=768)
    parser.add_argument("--v12-include-decomposition", action="store_true")
    parser.add_argument("--v12-expert-alpha", type=float, default=300.0)
    parser.add_argument("--v12-min-expert-samples", type=int, default=80)
    parser.add_argument("--v12-error-pool-max", type=int, default=2500)
    parser.add_argument("--v12-top-route-modes", type=int, default=4)
    parser.add_argument("--v12-route-prob-power", type=float, default=1.5)
    parser.add_argument("--v12-error-noise-scale", type=float, default=0.75)
    parser.add_argument("--v12-noise-jitter", type=float, default=0.02)
    parser.add_argument("--v16c-generator-variant", type=str, default="context_velocity")
    parser.add_argument("--v16c-top-c", type=int, default=8)
    parser.add_argument("--v16c-max-context-features", type=int, default=384)
    parser.add_argument("--v16c-base-mixes", type=str, default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    parser.add_argument("--v16c-calibrators", type=str, default="correction_context,stacked_top_context,stacked_context")
    parser.add_argument("--v16c-ridge-alphas", type=str, default="0.1,0.3,1,3,10,30,100,300,1000,3000")
    args = parser.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 400)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = "4,8,16"
        args.v16c_base_mixes = "expert_top8_uniform"
        args.v16c_calibrators = "correction_context,stacked_top_context"
    run(args)


if __name__ == "__main__":
    main()
