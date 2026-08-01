#!/usr/bin/env python3
"""Deep route-observability gap diagnostic after v26.

This script is deliberately forensic, not another architecture attempt.  It
rebuilds the fixed v16/v12 route basis from a previous v26 run config and asks:

- how strong is the fixed route oracle;
- how ambiguous are the oracle route labels;
- can stronger causal probes predict the oracle route;
- are opportunity samples separable from low-opportunity / ambiguous samples;
- is the remaining gap mostly candidate construction, router capacity or causal
  observability.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402


DEFAULT_RUN = ROOT / "outputs" / "raw_state_route_architecture_v26_fullcoord_bulk_seed42_2026-07-06"
DEFAULT_OUT = ROOT / "outputs" / "route_observability_gap_diagnostic_v27_2026-07-06"
EPS = 1e-8


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def as_path_config(value: Any) -> Any:
    if isinstance(value, str) and (
        value.endswith(".csv")
        or value.endswith(".json")
        or value.startswith("outputs/")
        or value.startswith("/Users/")
        or value.startswith("new_data/")
    ):
        return Path(value)
    return value


def load_v26_config(run_dir: Path, out_dir: Path) -> argparse.Namespace:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg = {k: as_path_config(v) for k, v in cfg.items()}
    cfg["out_dir"] = out_dir
    cfg["horizons"] = v26.parse_ints(cfg.get("horizons", [1, 2, 4, 6]))
    cfg["max_horizon"] = max(cfg["horizons"])
    return argparse.Namespace(**cfg)


def endpoint_vec(flat: np.ndarray, h: int, max_horizon: int) -> np.ndarray:
    return flat.reshape(len(flat), max_horizon, 2)[:, : int(h), :].sum(axis=1)


def route_endpoint_vec(route_pred: np.ndarray, h: int, max_horizon: int) -> np.ndarray:
    n, k, _d = route_pred.shape
    return route_pred.reshape(n, k, max_horizon, 2)[:, :, : int(h), :].sum(axis=2)


def rmse_endpoint(pred_flat: np.ndarray, true_flat: np.ndarray, h: int, max_horizon: int) -> float:
    p = endpoint_vec(pred_flat, h, max_horizon)
    y = endpoint_vec(true_flat, h, max_horizon)
    return float(np.sqrt(np.mean(np.sum((p - y) ** 2, axis=1))))


def per_sample_endpoint_error(pred_flat: np.ndarray, true_flat: np.ndarray, h: int, max_horizon: int) -> np.ndarray:
    p = endpoint_vec(pred_flat, h, max_horizon)
    y = endpoint_vec(true_flat, h, max_horizon)
    return np.sqrt(np.sum((p - y) ** 2, axis=1)).astype(np.float32)


def per_route_sq_error(route_pred: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    true_steps = true_flat.reshape(len(true_flat), args.max_horizon, 2)
    pred_steps = route_pred.reshape(len(route_pred), route_pred.shape[1], args.max_horizon, 2)
    err = np.zeros((len(route_pred), route_pred.shape[1]), dtype=np.float64)
    for h in args.horizons:
        p = pred_steps[:, :, : int(h), :].sum(axis=2)
        y = true_steps[:, : int(h), :].sum(axis=1)[:, None, :]
        err += np.sum((p - y) ** 2, axis=-1)
    return err.astype(np.float32)


def route_oracle_pred(route_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return route_pred[np.arange(len(route_pred)), labels].astype(np.float32)


def class_proba_full(model: Any, x: np.ndarray, k: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.full((len(x), k), 1e-7, dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        ci = int(cls)
        if 0 <= ci < k:
            out[:, ci] = raw[:, j]
    out /= np.maximum(out.sum(axis=1, keepdims=True), EPS)
    return out


def standardize(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(v26.safe_matrix(xtr)).astype(np.float32),
        scaler.transform(v26.safe_matrix(xva)).astype(np.float32),
        scaler.transform(v26.safe_matrix(xte)).astype(np.float32),
    )


def fit_probe(
    *,
    model_name: str,
    packet_name: str,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    subset_masks: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    k = int(max(ytr.max(initial=0), yva.max(initial=0), yte.max(initial=0)) + 1)
    ztr, zva, zte = standardize(xtr, xva, xte)
    if model_name == "logistic":
        model = LogisticRegression(max_iter=900, C=0.45, class_weight="balanced", random_state=seed + 27001)
        model.fit(ztr, ytr)
        pva = class_proba_full(model, zva, k)
        pte = class_proba_full(model, zte, k)
    elif model_name == "hgbdt":
        model = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=seed + 27002,
        )
        model.fit(ztr, ytr)
        pva = class_proba_full(model, zva, k)
        pte = class_proba_full(model, zte, k)
    else:
        raise ValueError(model_name)
    row: dict[str, Any] = {
        "stage": "route_observability_probe",
        "packet": packet_name,
        "model": model_name,
        "feature_dim": int(xtr.shape[1]),
        "route_k": k,
        "val_top1": float(np.mean(np.argmax(pva, axis=1) == yva)),
        "test_top1": float(np.mean(np.argmax(pte, axis=1) == yte)),
        "test_top3": float(top_k_accuracy_score(yte, pte, k=min(3, k), labels=np.arange(k))),
        "test_nll": float(log_loss(yte, np.clip(pte, 1e-7, 1.0), labels=np.arange(k))),
    }
    for name, mask in subset_masks.items():
        mask = np.asarray(mask, dtype=bool)
        if int(mask.sum()) < 10:
            continue
        row[f"{name}_n"] = int(mask.sum())
        row[f"{name}_top1"] = float(np.mean(np.argmax(pte[mask], axis=1) == yte[mask]))
        row[f"{name}_top3"] = float(top_k_accuracy_score(yte[mask], pte[mask], k=min(3, k), labels=np.arange(k)))
    return row, pte


def fit_residual_probe(
    *,
    packet_name: str,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ztr, zva, zte = standardize(xtr, xva, xte)
    best_alpha = None
    best_val = float("inf")
    best_model = None
    for alpha in [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]:
        model = Ridge(alpha=float(alpha))
        model.fit(ztr, ytr)
        pred_va = model.predict(zva).astype(np.float32)
        rmse = v16.endpoint_rmse_flat(pred_va, yva, args)
        if rmse < best_val:
            best_val = float(rmse)
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None and best_alpha is not None
    pred = best_model.predict(zte).astype(np.float32)
    return {
        "stage": "residual_direct_probe",
        "packet": packet_name,
        "feature_dim": int(xtr.shape[1]),
        "alpha": best_alpha,
        "val_endpoint_rmse": best_val,
        "test_endpoint_rmse": v16.endpoint_rmse_flat(pred, yte, args),
        "h1_rmse": rmse_endpoint(pred, yte, 1, args.max_horizon),
        "h6_rmse": rmse_endpoint(pred, yte, max(args.horizons), args.max_horizon),
    }


def quantile_rows(name: str, values: np.ndarray) -> list[dict[str, Any]]:
    qs = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    vals = np.quantile(values.astype(float), qs)
    return [{"metric": name, "quantile": q, "value": float(v)} for q, v in zip(qs, vals, strict=False)]


def label_distribution_rows(target: str, labels_by_split: dict[str, np.ndarray], k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    probs: dict[str, np.ndarray] = {}
    for split, labels in labels_by_split.items():
        counts = np.bincount(labels.astype(int), minlength=k).astype(np.float64)
        prob = counts / max(float(counts.sum()), EPS)
        probs[split] = prob
        ent = -float(np.sum(prob * np.log(np.maximum(prob, EPS))))
        for route, (cnt, pr) in enumerate(zip(counts, prob, strict=False)):
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "route": int(route),
                    "count": int(cnt),
                    "prob": float(pr),
                    "entropy": ent,
                    "max_prob": float(prob.max()),
                }
            )
    shift = []
    train = probs.get("train")
    if train is not None:
        for split, prob in probs.items():
            if split == "train":
                continue
            mix = 0.5 * (train + prob)
            kl_train = np.sum(train * (np.log(np.maximum(train, EPS)) - np.log(np.maximum(mix, EPS))))
            kl_split = np.sum(prob * (np.log(np.maximum(prob, EPS)) - np.log(np.maximum(mix, EPS))))
            js = 0.5 * (kl_train + kl_split)
            tv = 0.5 * np.sum(np.abs(train - prob))
            shift.append({"target": target, "compare": f"train_vs_{split}", "js_divergence": float(js), "total_variation": float(tv)})
    return pd.DataFrame(rows), pd.DataFrame(shift)


def expert_diversity_rows(route_pred: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    endpoints = route_endpoint_vec(route_pred, max(args.horizons), args.max_horizon)
    n, k, _ = endpoints.shape
    vals = []
    mins = []
    for i in range(n):
        d = []
        for a in range(k):
            for b in range(a + 1, k):
                d.append(float(np.linalg.norm(endpoints[i, a] - endpoints[i, b])))
        arr = np.asarray(d, dtype=np.float32)
        vals.append(float(arr.mean()))
        mins.append(float(arr.min()))
    rows = quantile_rows("route_expert_pairwise_hmax_mean_distance", np.asarray(vals, dtype=np.float32))
    rows += quantile_rows("route_expert_pairwise_hmax_min_distance", np.asarray(mins, dtype=np.float32))
    return pd.DataFrame(rows)


def make_report(out_dir: Path, args: argparse.Namespace, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Route Observability Gap Diagnostic v27", ""]
    lines.append("## Setup")
    lines.append(f"- source v26 run: `{args.source_run_dir}`")
    lines.append(f"- dataset: `{args.dataset}`, seed: `{args.seed}`")
    lines.append(f"- horizons: `{args.horizons}`")
    lines.append("")

    for title, key, sort_cols in [
        ("Oracle And Error Budget", "oracle_budget", ["horizon", "rmse"]),
        ("Route Ambiguity", "ambiguity", ["metric", "quantile"]),
        ("Route Distribution Shift", "route_shift", ["target", "compare"]),
        ("Expert Diversity", "expert_diversity", ["metric", "quantile"]),
        ("Route Probe", "route_probe", ["test_top3", "test_nll"]),
        ("Direct Residual Probe", "residual_probe", ["test_endpoint_rmse"]),
        ("Opportunity Strata", "strata", ["stratum"]),
    ]:
        df = tables.get(key, pd.DataFrame())
        if df.empty:
            continue
        lines.append(f"## {title}")
        show = df.copy()
        if all(c in show.columns for c in sort_cols):
            ascending = [True] * len(sort_cols)
            if sort_cols[0] == "test_top3":
                ascending[0] = False
            show = show.sort_values(sort_cols, ascending=ascending)
        lines.append(show.head(80).to_markdown(index=False))
        lines.append("")

    lines.append("## Diagnostic Reading")
    lines.append("- If fixed-route oracle is far better than learned mixtures, candidates exist.")
    lines.append("- If HGBDT route-top3 remains weak, the bottleneck is not just linear probe capacity.")
    lines.append("- If oracle margins are small for many samples, labels are intrinsically ambiguous and scalar/top1 route selection is ill-posed.")
    lines.append("- If high-opportunity/high-margin samples are still poorly predicted, causal observability is the dominant limitation.")
    (out_dir / "route_observability_gap_diagnostic_v27_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cli: argparse.Namespace) -> None:
    out_dir = cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args = load_v26_config(cli.v26_run_dir, out_dir)
    args.source_run_dir = str(cli.v26_run_dir)
    # The diagnostic should be deterministic and not mutate the original run dir.
    args.out_dir = out_dir

    basis = v26.build_route_basis(args, out_dir)
    packets = v26.feature_packets(args, basis, {})
    hmax = max(args.horizons)

    # Core predictions.
    zero = np.zeros_like(basis.y_test, dtype=np.float32)
    oracle_test = route_oracle_pred(basis.route_test, basis.oracle_labels_test)
    prior_top = v26.topm_temperature_weights(basis.prior.probs_test, top_m=basis.route_test.shape[1], temperature=2.0)
    prior_mix = v26.mix_routes(basis.route_test, prior_top)
    uniform_w = np.full((len(basis.y_test), basis.route_test.shape[1]), 1.0 / basis.route_test.shape[1], dtype=np.float32)
    uniform_mix = v26.mix_routes(basis.route_test, uniform_w)

    budget_rows = []
    for h in args.horizons:
        for name, pred in [
            ("clean_backbone_zero_residual", zero),
            ("fixed_route_oracle", oracle_test),
            ("fixed_route_prior_all_temp2", prior_mix),
            ("fixed_route_uniform", uniform_mix),
        ]:
            budget_rows.append(
                {
                    "stage": "oracle_budget",
                    "method": name,
                    "horizon": int(h),
                    "rmse": rmse_endpoint(pred, basis.y_test, int(h), args.max_horizon),
                }
            )
    oracle_budget = pd.DataFrame(budget_rows)

    # Ambiguity and opportunity.
    sqerr = per_route_sq_error(basis.route_test, basis.y_test, args)
    order = np.argsort(sqerr, axis=1)
    best = sqerr[np.arange(len(sqerr)), order[:, 0]]
    second = sqerr[np.arange(len(sqerr)), order[:, 1]]
    rel_margin = (second - best) / np.maximum(second, EPS)
    best_rmse = np.sqrt(best / max(len(args.horizons), 1)).astype(np.float32)
    prior_err = per_sample_endpoint_error(prior_mix, basis.y_test, hmax, args.max_horizon)
    oracle_err = per_sample_endpoint_error(oracle_test, basis.y_test, hmax, args.max_horizon)
    clean_err = per_sample_endpoint_error(zero, basis.y_test, hmax, args.max_horizon)
    opportunity = prior_err - oracle_err
    target_norm = np.linalg.norm(endpoint_vec(basis.y_test, hmax, args.max_horizon), axis=1)

    ambiguity = pd.DataFrame(
        quantile_rows("route_rel_margin", rel_margin)
        + quantile_rows("route_oracle_multi_horizon_rmse_per_sample", best_rmse)
        + quantile_rows("h6_clean_error", clean_err)
        + quantile_rows("h6_prior_minus_oracle_opportunity", opportunity)
        + quantile_rows("h6_target_norm", target_norm)
    )
    ambiguity["ambiguous_margin_lt_0p05"] = float(np.mean(rel_margin < 0.05))
    ambiguity["ambiguous_margin_lt_0p10"] = float(np.mean(rel_margin < 0.10))

    dist_oracle, shift_oracle = label_distribution_rows(
        "fixed_route_oracle",
        {
            "train": basis.oracle_labels_train,
            "val": basis.oracle_labels_val,
            "test": basis.oracle_labels_test,
        },
        basis.route_train.shape[1],
    )
    dist_residual, shift_residual = label_distribution_rows(
        "residual_signature",
        {
            "train": basis.labels.train,
            "val": basis.labels.val,
            "test": basis.labels.test,
        },
        int(basis.labels.k),
    )
    route_distribution = pd.concat([dist_oracle, dist_residual], ignore_index=True)
    route_shift = pd.concat([shift_oracle, shift_residual], ignore_index=True)
    expert_diversity = expert_diversity_rows(basis.route_test, args)

    subset_masks = {
        "high_margin_q75": rel_margin >= np.quantile(rel_margin, 0.75),
        "low_margin_q25": rel_margin <= np.quantile(rel_margin, 0.25),
        "high_opportunity_q75": opportunity >= np.quantile(opportunity, 0.75),
        "high_target_norm_q75": target_norm >= np.quantile(target_norm, 0.75),
    }

    probe_rows = []
    residual_rows = []
    for packet_name in ["coord_only", "coord_velocity", "no_topology_proxy"]:
        if packet_name not in packets:
            continue
        p = packets[packet_name]
        residual_rows.append(
            fit_residual_probe(
                packet_name=packet_name,
                xtr=p.train,
                xva=p.val,
                xte=p.test,
                ytr=basis.y_train,
                yva=basis.y_val,
                yte=basis.y_test,
                args=args,
            )
        )
        for model_name in parse_strs(cli.models):
            row, _pte = fit_probe(
                model_name=model_name,
                packet_name=packet_name,
                xtr=p.train,
                xva=p.val,
                xte=p.test,
                ytr=basis.oracle_labels_train,
                yva=basis.oracle_labels_val,
                yte=basis.oracle_labels_test,
                subset_masks=subset_masks,
                seed=int(args.seed),
            )
            probe_rows.append(row)
    route_probe = pd.DataFrame(probe_rows)
    residual_probe = pd.DataFrame(residual_rows)

    strata_rows = []
    for name, mask in subset_masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 10:
            continue
        strata_rows.append(
            {
                "stratum": name,
                "n": int(mask.sum()),
                "mean_rel_margin": float(np.mean(rel_margin[mask])),
                "clean_h6_rmse": float(np.sqrt(np.mean(clean_err[mask] ** 2))),
                "prior_h6_rmse": float(np.sqrt(np.mean(prior_err[mask] ** 2))),
                "oracle_h6_rmse": float(np.sqrt(np.mean(oracle_err[mask] ** 2))),
                "opportunity_mean": float(np.mean(opportunity[mask])),
                "target_norm_mean": float(np.mean(target_norm[mask])),
            }
        )
    strata = pd.DataFrame(strata_rows)

    tables = {
        "oracle_budget": oracle_budget,
        "ambiguity": ambiguity,
        "route_distribution": route_distribution,
        "route_shift": route_shift,
        "expert_diversity": expert_diversity,
        "route_probe": route_probe,
        "residual_probe": residual_probe,
        "strata": strata,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    make_report(out_dir, args, tables)
    print(json.dumps({"out_dir": str(out_dir), **{f"{k}_rows": len(v) for k, v in tables.items()}}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v26-run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", default="logistic,hgbdt")
    return ap.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
