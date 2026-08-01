#!/usr/bin/env python3
"""v42 visual-route evidence risk probe.

This runner tests the transfer mechanism more directly than v38/v39/v41.
Instead of asking a larger attention model to discover everything at once, it
flattens route candidates and trains a candidate-risk observer:

    route candidate k + route-specific visual evidence(C, k) -> risk_k

The target future is used only to build train-time risk labels.  Inference
features are causal: route candidates, coordinate context and tracking-aligned
visual state.  A visual transfer mechanism is accepted only if real visual
state beats matched row/time/wrong-cell controls.
"""

from __future__ import annotations

import json
import sys
import time
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
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_conditioned_visual_evidence_v39 as v39  # noqa: E402
import run_lachance_visual_state_route_validator_v38 as v38  # noqa: E402
import run_lachance_visual_state_target_v32 as v32  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "visual_evidence_risk_probe_v42_bulk_seed42_2026-07-08"
EPS = 1e-8


def parse_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 4:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) < EPS or float(np.std(bb)) < EPS:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    p = np.exp(z)
    return (p / np.maximum(p.sum(axis=axis, keepdims=True), EPS)).astype(np.float32)


def topm_weights_from_risk(risk: np.ndarray, *, top_m: int, temperature: float) -> np.ndarray:
    n, k = risk.shape
    top_m = min(max(int(top_m), 1), k)
    order = np.argsort(risk, axis=1)[:, :top_m]
    mask = np.full((n, k), -1e9, dtype=np.float32)
    rows = np.arange(n)[:, None]
    mask[rows, order] = -risk[rows, order] / max(float(temperature), 1e-6)
    return softmax_np(mask, axis=1)


def route_mix(route: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(weights[:, :, None] * route, axis=1).astype(np.float32)


def tune_risk_weights(
    risk_val: np.ndarray,
    route_val: np.ndarray,
    y_val: np.ndarray,
    args: Any,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for top_m in v38.parse_ints(args.v38_top_m_grid):
        for temp in v38.parse_floats(args.v38_temperature_grid):
            w = topm_weights_from_risk(risk_val, top_m=int(top_m), temperature=float(temp))
            pred = route_mix(route_val, w)
            rmse = v38.endpoint_rmse_flat_np(pred, y_val, args, max(args.horizons))
            if best is None or rmse < best["val_rmse"]:
                best = {"top_m": int(top_m), "temperature": float(temp), "val_rmse": float(rmse)}
    assert best is not None
    return best


def standardize_flat(
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k, d = xtr.shape
    sc = StandardScaler()
    ztr = sc.fit_transform(xtr.reshape(-1, d)).reshape(n, k, d)
    zva = sc.transform(xva.reshape(-1, d)).reshape(xva.shape)
    zte = sc.transform(xte.reshape(-1, d)).reshape(xte.shape)
    return (
        np.clip(np.nan_to_num(ztr), -8, 8).astype(np.float32),
        np.clip(np.nan_to_num(zva), -8, 8).astype(np.float32),
        np.clip(np.nan_to_num(zte), -8, 8).astype(np.float32),
    )


def visual_triplet(
    packets: dict[str, v32.Packet],
    *,
    visual_variant: str,
    visual_feature_mode: str,
    coord_dim: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[str]]:
    return v39.visual_state_packet(
        packets,
        visual_variant,
        coord_dim=coord_dim,
        feature_mode=visual_feature_mode,
    )


def build_candidate_triplet(
    *,
    packets: dict[str, v32.Packet],
    basis: v26.RouteBasis,
    args: Any,
    visual_variant: str,
    visual_feature_mode: str,
    raw_cols: int,
    cross_cols: int,
    use_visual_evidence: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    old_raw = int(args.v39_raw_visual_candidate_cols)
    old_cross = int(args.v39_cross_visual_cols)
    old_use = int(args.v39_use_visual_evidence)
    args.v39_raw_visual_candidate_cols = int(raw_cols)
    args.v39_cross_visual_cols = int(cross_cols)
    args.v39_use_visual_evidence = int(use_visual_evidence)
    try:
        coord_dim = packets["coord_all_context"].train.shape[1]
        (vtr, vva, vte), vnames = visual_triplet(
            packets,
            visual_variant=visual_variant,
            visual_feature_mode=visual_feature_mode,
            coord_dim=coord_dim,
        )
        xtr = v39.build_candidate_with_evidence(
            route_pred=basis.route_train,
            probs=basis.prior.probs_train,
            base_step=basis.arrays.base_train,
            split_df=basis.split.train,
            visual=vtr,
            visual_names=vnames,
            args=args,
        )
        xva = v39.build_candidate_with_evidence(
            route_pred=basis.route_val,
            probs=basis.prior.probs_val,
            base_step=basis.arrays.base_val,
            split_df=basis.split.val,
            visual=vva,
            visual_names=vnames,
            args=args,
        )
        xte = v39.build_candidate_with_evidence(
            route_pred=basis.route_test,
            probs=basis.prior.probs_test,
            base_step=basis.arrays.base_test,
            split_df=basis.split.test,
            visual=vte,
            visual_names=vnames,
            args=args,
        )
        xtr, xva, xte = standardize_flat(xtr, xva, xte)
        return xtr, xva, xte, len(vnames)
    finally:
        args.v39_raw_visual_candidate_cols = old_raw
        args.v39_cross_visual_cols = old_cross
        args.v39_use_visual_evidence = old_use


def fit_predict_risk(
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    teacher: v38.TeacherLabels,
    *,
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n, k, d = xtr.shape
    xtr_f = xtr.reshape(n * k, d)
    ytr_f = teacher.err_train.reshape(n * k)
    xva_f = xva.reshape(xva.shape[0] * k, d)
    xte_f = xte.reshape(xte.shape[0] * k, d)
    if model_name == "ridge":
        model = Ridge(alpha=30.0)
    elif model_name == "hgbdt":
        model = HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            random_state=int(seed) + 42001,
        )
    else:
        raise ValueError(f"Unknown risk model: {model_name}")
    model.fit(xtr_f, ytr_f)
    ptr = model.predict(xtr_f).reshape(n, k).astype(np.float32)
    pva = model.predict(xva_f).reshape(xva.shape[0], k).astype(np.float32)
    pte = model.predict(xte_f).reshape(xte.shape[0], k).astype(np.float32)
    meta = {
        "risk_model": model_name,
        "candidate_dim": int(d),
        "train_risk_corr": safe_corr(ptr, teacher.err_train),
        "val_risk_corr": safe_corr(pva, teacher.err_val),
        "test_risk_corr": safe_corr(pte, teacher.err_test),
    }
    return ptr, pva, pte, meta


def evaluate_risk_variant(
    *,
    label: str,
    risk_val: np.ndarray,
    risk_test: np.ndarray,
    basis: v26.RouteBasis,
    teacher: v38.TeacherLabels,
    args: Any,
    extra: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    best = tune_risk_weights(risk_val, basis.route_val, basis.y_val, args)
    wte = topm_weights_from_risk(risk_test, top_m=int(best["top_m"]), temperature=float(best["temperature"]))
    pred = route_mix(basis.route_test, wte)
    rows, diag = v38.evaluate_np(
        label=label,
        pred_flat=pred,
        logits=-risk_test,
        weights=wte,
        basis=basis,
        teacher=teacher,
        args=args,
        extra={
            **extra,
            "top_m": int(best["top_m"]),
            "temperature": float(best["temperature"]),
            "best_val_hmax_rmse": float(best["val_rmse"]),
        },
    )
    diag.update(
        {
            **extra,
            "top_m": int(best["top_m"]),
            "temperature": float(best["temperature"]),
            "best_val_hmax_rmse": float(best["val_rmse"]),
            "risk_error_corr": safe_corr(risk_test, teacher.err_test),
        }
    )
    return pd.DataFrame(rows), diag


def write_report(out_dir: Path, summary: pd.DataFrame, diagnostics: pd.DataFrame, decision: dict[str, Any]) -> None:
    lines = ["# v42 Visual Evidence Risk Probe", ""]
    h6 = summary[summary["horizon"].eq(6)].sort_values("rmse") if not summary.empty else pd.DataFrame()
    cols = [
        c
        for c in [
            "method",
            "variant",
            "fusion_config",
            "visual_source",
            "visual_feature_mode",
            "rmse",
            "r2",
            "risk_error_corr",
            "route_top3",
            "ndcg_at8",
            "best_val_hmax_rmse",
        ]
        if c in h6.columns
    ]
    lines.append("## h6 Leaderboard")
    lines.append(h6[cols].head(60).to_markdown(index=False) if not h6.empty else "No rows.")
    lines.append("")
    lines.append("## Diagnostics")
    if not diagnostics.empty:
        dcols = [
            c
            for c in [
                "variant",
                "fusion_config",
                "visual_source",
                "visual_feature_mode",
                "hmax_rmse",
                "risk_error_corr",
                "route_top1",
                "route_top3",
                "ndcg_at8",
                "expected_candidate_error_mean",
                "oracle_gap_closed_vs_prior",
                "candidate_dim",
            ]
            if c in diagnostics.columns
        ]
        lines.append(diagnostics.sort_values("hmax_rmse")[dcols].head(80).to_markdown(index=False))
    lines.append("")
    lines.append("## Decision")
    lines.append("```json")
    lines.append(json.dumps(audit.finite_json(decision), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Pass condition: real visual evidence must beat no_visual and matched row/time/wrong-cell controls.")
    lines.append("- This is a transfer-mechanism probe, not a final architecture claim.")
    (out_dir / "visual_evidence_risk_probe_v42_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: Any) -> None:
    t0 = time.time()
    audit.set_global_seed(int(args.seed))
    args.horizons = v38.parse_ints(args.horizons)
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

    basis = v26.build_route_basis(args, args.out_dir / "route_basis")
    packets = v32.build_packets(args, basis.arrays, basis.split)
    teacher = v38.build_teacher_labels(args, basis)
    base_metrics, base_diag = v38.baseline_rows(basis, teacher, args)
    hmax = max(args.horizons)
    prior = base_metrics[base_metrics["horizon"].eq(hmax) & base_metrics["variant"].astype(str).eq("prior_topm")]
    oracle = base_metrics[base_metrics["horizon"].eq(hmax) & base_metrics["variant"].astype(str).eq("oracle")]
    args._v38_prior_hmax_rmse = float(prior.iloc[0]["rmse"]) if not prior.empty else None
    args._v38_oracle_hmax_rmse = float(oracle.iloc[0]["rmse"]) if not oracle.empty else None

    fusion_specs = [
        ("base_no_visual", 0, 0, False),
        ("formula_only", 0, 0, True),
        ("cross64", 0, 64, True),
        ("cross128", 0, 128, True),
        ("raw96", 96, 0, True),
        ("both48_cross96", 48, 96, True),
    ]
    if args.smoke:
        fusion_specs = fusion_specs[:3]
    variant_specs = [
        ("no_visual", "no_visual", "all"),
        ("polarity_history_real", "real", "polarity_history_only"),
        ("polarity_history_row", "row_shuffled", "polarity_history_only"),
        ("polarity_history_wrong", "same_frame_wrong_cell", "polarity_history_only"),
        ("polarity_history_time", "time_shuffled", "polarity_history_only"),
        ("polarity_real", "real", "polarity_only"),
        ("polarity_wrong", "same_frame_wrong_cell", "polarity_only"),
        ("history_real", "real", "history_only"),
        ("history_time", "time_shuffled", "history_only"),
    ]
    if args.smoke:
        variant_specs = variant_specs[:4]

    metric_parts = [base_metrics]
    diag_rows = base_diag.to_dict(orient="records")
    train_rows = []
    for fusion_name, raw_cols, cross_cols, use_visual in fusion_specs:
        for variant_name, visual_source, mode in variant_specs:
            if fusion_name == "base_no_visual" and variant_name != "no_visual":
                continue
            if fusion_name != "base_no_visual" and variant_name == "no_visual":
                continue
            xtr, xva, xte, visual_dim = build_candidate_triplet(
                packets=packets,
                basis=basis,
                args=args,
                visual_variant=visual_source,
                visual_feature_mode=mode,
                raw_cols=raw_cols,
                cross_cols=cross_cols,
                use_visual_evidence=use_visual,
            )
            for risk_model in (["ridge", "hgbdt"] if not args.smoke else ["hgbdt"]):
                risk_tr, risk_va, risk_te, meta = fit_predict_risk(
                    xtr,
                    xva,
                    xte,
                    teacher,
                    model_name=risk_model,
                    seed=int(args.seed),
                )
                label = f"v42_{fusion_name}_{variant_name}_{risk_model}"
                rows, diag = evaluate_risk_variant(
                    label=label,
                    risk_val=risk_va,
                    risk_test=risk_te,
                    basis=basis,
                    teacher=teacher,
                    args=args,
                    extra={
                        "stage": "v42_visual_evidence_risk_probe",
                        "variant": variant_name,
                        "fusion_config": fusion_name,
                        "visual_source": visual_source,
                        "visual_feature_mode": mode,
                        "raw_visual_candidate_cols": int(raw_cols),
                        "cross_visual_cols": int(cross_cols),
                        "risk_model": risk_model,
                        "visual_dim": int(visual_dim),
                        "candidate_dim": int(xtr.shape[2]),
                        "prior_hmax_rmse": args._v38_prior_hmax_rmse,
                        "oracle_hmax_rmse": args._v38_oracle_hmax_rmse,
                        **meta,
                    },
                )
                metric_parts.append(rows)
                diag_rows.append(diag)
                train_rows.append({"method": label, **meta})

    summary = pd.concat(metric_parts, ignore_index=True, sort=False)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.insert(0, "seed", int(args.seed))
    diagnostics.insert(0, "dataset", str(args.dataset))
    train_log = pd.DataFrame(train_rows)
    if not train_log.empty:
        train_log.insert(0, "seed", int(args.seed))
        train_log.insert(0, "dataset", str(args.dataset))

    summary.to_csv(args.out_dir / "visual_evidence_risk_probe_v42_summary.csv", index=False)
    diagnostics.to_csv(args.out_dir / "visual_evidence_risk_probe_v42_diagnostics.csv", index=False)
    train_log.to_csv(args.out_dir / "visual_evidence_risk_probe_v42_train_log.csv", index=False)

    h6 = summary[summary["horizon"].eq(hmax)].sort_values("rmse")
    deploy = h6[~h6["variant"].astype(str).isin(["oracle"])] if "variant" in h6.columns else h6
    real = deploy[deploy["variant"].astype(str).str.contains("_real", na=False)]
    controls = deploy[
        deploy["variant"].astype(str).str.contains("_row|_wrong|_time", regex=True, na=False)
    ]
    no_visual = deploy[deploy["variant"].astype(str).eq("no_visual")]
    decision = {
        "elapsed_sec": time.time() - t0,
        "best_hmax": deploy.iloc[0].to_dict() if not deploy.empty else {},
        "best_real": real.iloc[0].to_dict() if not real.empty else {},
        "best_control": controls.iloc[0].to_dict() if not controls.empty else {},
        "best_no_visual": no_visual.iloc[0].to_dict() if not no_visual.empty else {},
        "real_beats_controls": bool(not real.empty and not controls.empty and float(real.iloc[0]["rmse"]) < float(controls.iloc[0]["rmse"])),
        "real_beats_no_visual": bool(not real.empty and not no_visual.empty and float(real.iloc[0]["rmse"]) < float(no_visual.iloc[0]["rmse"])),
        "prior_hmax_rmse": args._v38_prior_hmax_rmse,
        "oracle_hmax_rmse": args._v38_oracle_hmax_rmse,
    }
    (args.out_dir / "visual_evidence_risk_probe_v42_decision.json").write_text(json.dumps(audit.finite_json(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.out_dir, summary, diagnostics, decision)
    print(json.dumps({"out_dir": str(args.out_dir), "decision": audit.finite_json(decision)}, indent=2, ensure_ascii=False))


def main() -> None:
    # Reuse v39's parser so route-basis and visual-packet arguments stay
    # identical across v39/v41/v42.  v42-specific choices are intentionally
    # fixed in this diagnostic runner to keep the comparison small and matched.
    args = v39.parse_args()
    if str(args.out_dir) == str(v39.DEFAULT_OUT):
        args.out_dir = DEFAULT_OUT
    run(args)


if __name__ == "__main__":
    main()
