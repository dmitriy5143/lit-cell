#!/usr/bin/env python3
"""v70 frozen-v52 anchor route search.

v69 answered whether a search/filter over the fixed route bank can beat a local
route prior.  This runner answers the sharper question raised after v69:

    can route search help when it is placed on top of the real v52 clean-best?

The v52 prediction is a frozen deterministic anchor.  Candidate routes are used
only as bounded deviations or as an augmented candidate set containing the v52
anchor itself.  Validation may choose blend=0, in which case the search is
explicitly rejected and the output remains the v52 anchor.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_flow_state_cleanbest_integration_v52 as v52  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_clustered_occupancy_route_filter_v57 as v57  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "v52_anchor_route_search_v70_2026-07-14"
EPS = 1e-8


def parse_csv(text: str | list[str]) -> list[str]:
    if isinstance(text, list):
        return [str(x) for x in text]
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
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def flat_to_steps(x: np.ndarray, max_horizon: int) -> np.ndarray:
    return safe(x).reshape(len(x), int(max_horizon), 2)


def endpoint_rmse_flat(pred: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> float:
    return v16.endpoint_rmse_flat(safe(pred), safe(true_flat), args)


def endpoint_rows(label: str, residual_flat: np.ndarray, basis: v26.RouteBasis, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=basis.arrays.steps_test,
        base=basis.arrays.base_test,
        residual_pred=flat_to_steps(residual_flat, args.max_horizon),
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def endpoint_error_matrix_to_flat(route: np.ndarray, target_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n, k, _ = route.shape
    pred = route.reshape(n, k, args.max_horizon, 2)
    true = target_flat.reshape(n, args.max_horizon, 2)
    err = np.zeros((n, k), dtype=np.float32)
    for h in args.horizons:
        h = int(h)
        p = np.sum(pred[:, :, :h, :], axis=2)
        y = np.sum(true[:, :h, :], axis=1)[:, None, :]
        err += np.sum((p - y) ** 2, axis=-1).astype(np.float32)
    return np.sqrt(err / max(len(args.horizons), 1)).astype(np.float32)


def route_mix(route: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(safe(route) * safe(weights)[:, :, None], axis=1).astype(np.float32)


def topm_weights(logits: np.ndarray, top_m: int, temp: float) -> np.ndarray:
    return v57.topm_weights_from_logits(safe(logits), top_m=int(top_m), temperature=float(temp))


def build_v52_anchor(basis: v26.RouteBasis, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild the v52 best-style anchor from the already built route basis."""
    local = copy.copy(args)
    local.flow_packets = "real"
    local.use_reliability_head = False
    local.bounded_calibration = bool(args.v70_v52_bounded)
    local.horizons = list(args.horizons)
    local.max_horizon = int(args.max_horizon)
    local.v16c_base_mixes = args.v70_v52_base_mix
    local.v16c_calibrators = args.v70_v52_calibrator
    local.v16c_ridge_alphas = args.v16c_ridge_alphas
    local.max_train_rows = int(args.generator_max_train_rows)
    local.max_val_rows = int(args.generator_max_val_rows)
    local.max_test_rows = int(args.generator_max_test_rows)
    local.bounded_quantile = float(args.v70_bounded_quantile)
    local.bounded_scale = float(args.v70_bounded_scale)

    ytr = audit.flatten_residual(basis.arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(basis.arrays.residual_val).astype(np.float32)
    rtr = v16.route_outputs(basis.bank, basis.prior.x_train)
    rva = v16.route_outputs(basis.bank, basis.prior.x_val)
    rte = v16.route_outputs(basis.bank, basis.prior.x_test)
    xtr0, xva0, xte0 = v52.select_context(basis.prior.x_train, basis.prior.x_val, basis.prior.x_test, local.v16c_max_context_features)

    split_flow, flow_cols = v52.prepare_sampled_split_with_flow(local)
    packets = v52.build_flow_packets(split_flow, flow_cols, local)
    packet = packets["clean_best_real_flow"]
    xtr = np.concatenate([xtr0, packet.train], axis=1).astype(np.float32)
    xva = np.concatenate([xva0, packet.val], axis=1).astype(np.float32)
    xte = np.concatenate([xte0, packet.test], axis=1).astype(np.float32)

    if args.v70_v52_base_mix == "expert_top4_uniform":
        top_c, mode, power = 4, "uniform", 1.0
    elif args.v70_v52_base_mix == "expert_top8_uniform":
        top_c, mode, power = 8, "uniform", 1.0
    elif args.v70_v52_base_mix == "expert_all_uniform":
        top_c, mode, power = 999, "all_uniform", 1.0
    else:
        raise ValueError(f"unsupported v52 base mix {args.v70_v52_base_mix}")
    mtr = v16.mix_route_outputs(rtr, basis.prior.probs_train, top_c, mode, power)
    mva = v16.mix_route_outputs(rva, basis.prior.probs_val, top_c, mode, power)
    mte = v16.mix_route_outputs(rte, basis.prior.probs_test, top_c, mode, power)

    ftr = v52.build_calibration_features(route_pred=rtr, probs=basis.prior.probs_train, context=xtr, mix=mtr, kind=args.v70_v52_calibrator, top_c=local.v16c_top_c)
    fva = v52.build_calibration_features(route_pred=rva, probs=basis.prior.probs_val, context=xva, mix=mva, kind=args.v70_v52_calibrator, top_c=local.v16c_top_c)
    fte = v52.build_calibration_features(route_pred=rte, probs=basis.prior.probs_test, context=xte, mix=mte, kind=args.v70_v52_calibrator, top_c=local.v16c_top_c)
    ptr, pva, pte, alpha, val_rmse = v52.fit_ridge_endpoint_all(
        xtr=ftr,
        xva=fva,
        xte=fte,
        ytr=ytr,
        yva=yva,
        horizons=local.horizons,
        max_horizon=local.max_horizon,
        alphas=parse_floats(local.v16c_ridge_alphas),
    )
    if bool(args.v70_v52_bounded):
        ptr = v52.bound_prediction(mtr, ptr, ytr, mtr, local)
        pva = v52.bound_prediction(mva, pva, ytr, mtr, local)
        pte = v52.bound_prediction(mte, pte, ytr, mtr, local)
    meta = {
        "v52_anchor_base_mix": args.v70_v52_base_mix,
        "v52_anchor_calibrator": args.v70_v52_calibrator,
        "v52_anchor_alpha": float(alpha),
        "v52_anchor_val_endpoint_rmse": float(val_rmse),
        "v52_anchor_bounded": bool(args.v70_v52_bounded),
        "v52_anchor_feature_dim": int(ftr.shape[1]),
        "v52_anchor_flow_dim": int(packet.train.shape[1]),
    }
    return {"train": ptr.astype(np.float32), "val": pva.astype(np.float32), "test": pte.astype(np.float32), "mix_train": mtr, "mix_val": mva, "mix_test": mte}, meta


def candidate_bank(basis: v26.RouteBasis, anchor: dict[str, np.ndarray], args: argparse.Namespace, mode: str, shrink: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rtr, rva, rte = basis.route_train, basis.route_val, basis.route_test
    ptr, pva, pte = anchor["train"], anchor["val"], anchor["test"]
    k = rtr.shape[1]
    if mode == "route":
        prior_tr, prior_va, prior_te = basis.prior.probs_train, basis.prior.probs_val, basis.prior.probs_test
        return rtr, rva, rte, prior_tr, prior_va, prior_te, {"candidate_mode": mode, "candidate_count": int(k), "shrink": 1.0}
    if mode == "augmented_anchor":
        prior_mass = float(args.v70_anchor_prior_mass)
        ptr3, pva3, pte3 = ptr[:, None, :], pva[:, None, :], pte[:, None, :]
        prior_tr = np.concatenate([np.full((len(rtr), 1), prior_mass, dtype=np.float32), (1.0 - prior_mass) * basis.prior.probs_train], axis=1)
        prior_va = np.concatenate([np.full((len(rva), 1), prior_mass, dtype=np.float32), (1.0 - prior_mass) * basis.prior.probs_val], axis=1)
        prior_te = np.concatenate([np.full((len(rte), 1), prior_mass, dtype=np.float32), (1.0 - prior_mass) * basis.prior.probs_test], axis=1)
        prior_tr /= np.maximum(prior_tr.sum(axis=1, keepdims=True), EPS)
        prior_va /= np.maximum(prior_va.sum(axis=1, keepdims=True), EPS)
        prior_te /= np.maximum(prior_te.sum(axis=1, keepdims=True), EPS)
        return (
            np.concatenate([ptr3, rtr], axis=1).astype(np.float32),
            np.concatenate([pva3, rva], axis=1).astype(np.float32),
            np.concatenate([pte3, rte], axis=1).astype(np.float32),
            prior_tr.astype(np.float32),
            prior_va.astype(np.float32),
            prior_te.astype(np.float32),
            {"candidate_mode": mode, "candidate_count": int(k + 1), "shrink": 1.0, "anchor_prior_mass": prior_mass},
        )
    if mode == "shrink_to_v52":
        lam = float(shrink)
        ctr = ptr[:, None, :] + lam * (rtr - ptr[:, None, :])
        cva = pva[:, None, :] + lam * (rva - pva[:, None, :])
        cte = pte[:, None, :] + lam * (rte - pte[:, None, :])
        return ctr.astype(np.float32), cva.astype(np.float32), cte.astype(np.float32), basis.prior.probs_train, basis.prior.probs_val, basis.prior.probs_test, {"candidate_mode": mode, "candidate_count": int(k), "shrink": lam}
    raise ValueError(f"unknown candidate bank mode {mode}")


def path_features(route: np.ndarray, base: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n, k, _ = route.shape
    residual = route.reshape(n, k, args.max_horizon, 2)
    steps = residual + base[:, None, None, :].astype(np.float32)
    endpoint = np.sum(steps, axis=2)
    step_norm = np.linalg.norm(steps, axis=3)
    path_norm = np.sum(step_norm, axis=2)
    endpoint_norm = np.linalg.norm(endpoint, axis=2)
    step_max = np.max(step_norm, axis=2)
    accel = np.diff(steps, axis=2)
    accel_mean = np.mean(np.linalg.norm(accel, axis=3), axis=2) if args.max_horizon > 1 else np.zeros((n, k), dtype=np.float32)
    step_a = steps[:, :, :-1, :]
    step_b = steps[:, :, 1:, :]
    denom = np.maximum(np.linalg.norm(step_a, axis=3) * np.linalg.norm(step_b, axis=3), EPS)
    turn = 1.0 - np.sum(step_a * step_b, axis=3) / denom if args.max_horizon > 1 else np.zeros((n, k, 1), dtype=np.float32)
    turn_mean = np.mean(np.nan_to_num(turn, nan=0.0), axis=2)
    base_speed = np.linalg.norm(base, axis=1)[:, None]
    jump_excess = np.maximum(0.0, step_max - float(args.v70_jump_factor) * np.maximum(base_speed, 1.0))
    efficiency = endpoint_norm / np.maximum(path_norm, 1e-6)
    feat = np.stack([path_norm, endpoint_norm, step_max, accel_mean, turn_mean, jump_excess, efficiency], axis=2)
    return safe(feat)


def fit_path_risk(
    route_train: np.ndarray,
    route_val: np.ndarray,
    route_test: np.ndarray,
    basis: v26.RouteBasis,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    xtr3 = path_features(route_train, basis.arrays.base_train, args)
    xva3 = path_features(route_val, basis.arrays.base_val, args)
    xte3 = path_features(route_test, basis.arrays.base_test, args)
    n, k, f = xtr3.shape
    rid_tr = np.tile(np.arange(k, dtype=np.float32)[None, :, None] / max(k - 1, 1), (n, 1, 1))
    rid_va = np.tile(np.arange(k, dtype=np.float32)[None, :, None] / max(k - 1, 1), (len(route_val), 1, 1))
    rid_te = np.tile(np.arange(k, dtype=np.float32)[None, :, None] / max(k - 1, 1), (len(route_test), 1, 1))
    xtr = np.concatenate([xtr3, rid_tr], axis=2).reshape(n * k, f + 1)
    xva = np.concatenate([xva3, rid_va], axis=2).reshape(len(route_val) * k, f + 1)
    xte = np.concatenate([xte3, rid_te], axis=2).reshape(len(route_test) * k, f + 1)
    ytr = endpoint_error_matrix_to_flat(route_train, basis.y_train, args).reshape(-1)
    sc = StandardScaler()
    ztr = sc.fit_transform(xtr)
    zva = sc.transform(xva)
    zte = sc.transform(xte)
    model = HistGradientBoostingRegressor(
        max_iter=int(args.v70_hgbdt_iter),
        learning_rate=float(args.v70_hgbdt_lr),
        max_leaf_nodes=int(args.v70_hgbdt_leaf_nodes),
        l2_regularization=float(args.v70_hgbdt_l2),
        random_state=int(args.seed) + 7001,
    )
    model.fit(ztr, ytr)
    return (
        model.predict(ztr).reshape(n, k).astype(np.float32),
        model.predict(zva).reshape(len(route_val), k).astype(np.float32),
        model.predict(zte).reshape(len(route_test), k).astype(np.float32),
        {"risk_model": "hgbdt_path", "risk_feature_dim": int(f + 1), "hgbdt_iter": int(args.v70_hgbdt_iter)},
    )


def zscore_by_row(x: np.ndarray) -> np.ndarray:
    return ((safe(x) - np.mean(safe(x), axis=1, keepdims=True)) / np.maximum(np.std(safe(x), axis=1, keepdims=True), 1e-6)).astype(np.float32)


def tune_anchor_search(
    *,
    label: str,
    route_val: np.ndarray,
    route_test: np.ndarray,
    prior_val: np.ndarray,
    prior_test: np.ndarray,
    risk_val: np.ndarray,
    risk_test: np.ndarray,
    anchor_val: np.ndarray,
    anchor_test: np.ndarray,
    basis: v26.RouteBasis,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dist_val = endpoint_error_matrix_to_flat(route_val, anchor_val, args)
    dist_test = endpoint_error_matrix_to_flat(route_test, anchor_test, args)
    best: dict[str, Any] | None = None
    for alpha in parse_floats(args.v70_prior_alpha_grid):
        for beta in parse_floats(args.v70_risk_beta_grid):
            for eta in parse_floats(args.v70_anchor_eta_grid):
                lv = float(alpha) * zscore_by_row(np.log(np.maximum(prior_val, EPS))) - float(beta) * zscore_by_row(risk_val) - float(eta) * zscore_by_row(dist_val)
                lt = float(alpha) * zscore_by_row(np.log(np.maximum(prior_test, EPS))) - float(beta) * zscore_by_row(risk_test) - float(eta) * zscore_by_row(dist_test)
                for top_m in parse_ints(args.v70_top_m_grid):
                    for temp in parse_floats(args.v70_temperature_grid):
                        wv = topm_weights(lv, top_m, temp)
                        wt = topm_weights(lt, top_m, temp)
                        mv = route_mix(route_val, wv)
                        mt = route_mix(route_test, wt)
                        for blend in parse_floats(args.v70_blend_grid):
                            pv = (1.0 - float(blend)) * anchor_val + float(blend) * mv
                            rmse = endpoint_rmse_flat(pv, basis.y_val, args)
                            if best is None or rmse < best["val_rmse"]:
                                pt = (1.0 - float(blend)) * anchor_test + float(blend) * mt
                                best = {
                                    "pred_test": pt.astype(np.float32),
                                    "weights_test": wt.astype(np.float32),
                                    "logits_test": lt.astype(np.float32),
                                    "val_rmse": float(rmse),
                                    "top_m": int(top_m),
                                    "temperature": float(temp),
                                    "blend": float(blend),
                                    "prior_alpha": float(alpha),
                                    "risk_beta": float(beta),
                                    "anchor_eta": float(eta),
                                }
    assert best is not None
    rows = endpoint_rows(
        label,
        best["pred_test"],
        basis,
        args,
        {
            "stage": "v70_v52_anchor_search",
            "top_m": int(best["top_m"]),
            "temperature": float(best["temperature"]),
            "blend_from_v52_to_route_mix": float(best["blend"]),
            "prior_alpha": float(best["prior_alpha"]),
            "risk_beta": float(best["risk_beta"]),
            "anchor_eta": float(best["anchor_eta"]),
            "val_rmse": float(best["val_rmse"]),
            **extra,
        },
    )
    hmax = max(args.horizons)
    hrow = [r for r in rows if int(r["horizon"]) == int(hmax)][0]
    diag = {
        "method": label,
        "hmax": int(hmax),
        "hmax_rmse": float(hrow["rmse"]),
        "hmax_r2": float(hrow["r2"]),
        "top_m": int(best["top_m"]),
        "temperature": float(best["temperature"]),
        "blend_from_v52_to_route_mix": float(best["blend"]),
        "prior_alpha": float(best["prior_alpha"]),
        "risk_beta": float(best["risk_beta"]),
        "anchor_eta": float(best["anchor_eta"]),
        "val_rmse": float(best["val_rmse"]),
        "weight_entropy_mean": float(np.mean(-np.sum(best["weights_test"] * np.log(np.maximum(best["weights_test"], EPS)), axis=1))),
        **extra,
    }
    return rows, diag


def write_report(out_dir: Path, args: argparse.Namespace, contract: pd.DataFrame, metrics: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = [
        "# v70 Frozen-v52 Anchor Route Search Report",
        "",
        f"Dataset: `{args.dataset}`, seed `{args.seed}`.",
        "",
        "## Contract",
        contract.to_markdown(index=False) if not contract.empty else "No contract.",
        "",
        "## Best h6 / hmax",
    ]
    if not metrics.empty:
        hmax = max(args.horizons)
        sub = metrics[metrics["horizon"].eq(hmax)].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "candidate_mode", "shrink", "top_m", "temperature", "blend_from_v52_to_route_mix", "prior_alpha", "risk_beta", "anchor_eta"] if c in sub.columns]
        lines.append(sub[cols].head(60).to_markdown(index=False))
    else:
        lines.append("No metrics.")
    lines.extend(["", "## Diagnostics"])
    lines.append(diag.sort_values("hmax_rmse").head(80).to_markdown(index=False) if not diag.empty else "No diagnostics.")
    lines.extend(
        [
            "",
            "## Reading",
            "- If best blend is `0`, validation prefers frozen v52 and the route search is rejected.",
            "- If best blend is positive and h6 beats the v52 anchor, search genuinely helps on top of v52.",
        ]
    )
    (out_dir / "v70_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    np.random.seed(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.dense_features = args.features
    _device = closure.device_from_arg(args.device)

    basis = v26.build_route_basis(args, args.out_dir / "route_basis")
    anchor, anchor_meta = build_v52_anchor(basis, args)
    metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = [
        {"item": "dataset", "value": str(args.dataset)},
        {"item": "seed", "value": int(args.seed)},
        {"item": "train_rows", "value": int(len(basis.split.train))},
        {"item": "val_rows", "value": int(len(basis.split.val))},
        {"item": "test_rows", "value": int(len(basis.split.test))},
        {"item": "route_count", "value": int(basis.route_test.shape[1])},
    ]
    contract_rows.extend({"item": k, "value": v} for k, v in anchor_meta.items())

    metric_rows.extend(endpoint_rows("v70_v52_anchor", anchor["test"], basis, args, {"stage": "v52_anchor", **anchor_meta}))
    oracle_idx = np.argmin(endpoint_error_matrix_to_flat(basis.route_test, basis.y_test, args), axis=1)
    metric_rows.extend(endpoint_rows("v70_fixed_route_oracle", basis.route_test[np.arange(len(basis.route_test)), oracle_idx], basis, args, {"stage": "oracle"}))

    for mode in parse_csv(args.v70_candidate_modes):
        shrink_values = parse_floats(args.v70_shrink_grid) if mode == "shrink_to_v52" else [1.0]
        for shrink in shrink_values:
            rtr, rva, rte, ptr, pva, pte, cmeta = candidate_bank(basis, anchor, args, mode, shrink)
            _risk_tr, risk_va, risk_te, rmeta = fit_path_risk(rtr, rva, rte, basis, args)
            label = f"v70_{mode}_shrink{float(shrink):.2f}_pathrisk"
            rows, diag = tune_anchor_search(
                label=label,
                route_val=rva,
                route_test=rte,
                prior_val=pva,
                prior_test=pte,
                risk_val=risk_va,
                risk_test=risk_te,
                anchor_val=anchor["val"],
                anchor_test=anchor["test"],
                basis=basis,
                args=args,
                extra={**cmeta, **rmeta},
            )
            metric_rows.extend(rows)
            diag_rows.append(diag)

    metrics = pd.DataFrame(metric_rows)
    diag = pd.DataFrame(diag_rows)
    contract = pd.DataFrame(contract_rows)
    if not metrics.empty:
        metrics.insert(0, "dataset", str(args.dataset))
        metrics.insert(1, "seed", int(args.seed))
    if not diag.empty:
        diag.insert(0, "dataset", str(args.dataset))
        diag.insert(1, "seed", int(args.seed))
    metrics.to_csv(args.out_dir / "v70_anchor_search_summary.csv", index=False)
    diag.to_csv(args.out_dir / "v70_anchor_search_diagnostics.csv", index=False)
    contract.to_csv(args.out_dir / "v70_contract.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, contract, metrics, diag)
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": int(len(metrics)), "diagnostics": int(len(diag)), "elapsed_sec": time.time() - t0}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=audit.DEFAULT_FEATURES)
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

    # v26/v16/v52 compatibility.
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
    ap.add_argument("--extra-feature-grid", type=Path, default=v52.v12.DEFAULT_OBJECT_GRID)
    ap.add_argument("--extra-feature-prefixes", type=str, default="oc_")
    ap.add_argument("--extra-feature-block-name", type=str, default="object_mask")
    ap.add_argument("--extra-feature-max-cols", type=int, default=256)
    ap.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    ap.add_argument("--v10-velocity-max-cols", type=int, default=160)
    ap.add_argument("--local-ks", type=str, default="8,16,32")
    ap.add_argument("--local-radii", type=str, default="64,128,256")
    ap.add_argument("--ridge-alphas", type=str, default="30,100,300,1000,3000,10000")

    # v52 anchor and route-search knobs.
    ap.add_argument("--v70-v52-base-mix", default="expert_top8_uniform", choices=["expert_top4_uniform", "expert_top8_uniform", "expert_all_uniform"])
    ap.add_argument("--v70-v52-calibrator", default="stacked_context", choices=["stacked_context", "stacked_top_context", "correction_context"])
    ap.add_argument("--v70-v52-bounded", action="store_true", default=True)
    ap.add_argument("--no-v70-v52-bounded", action="store_false", dest="v70_v52_bounded")
    ap.add_argument("--v70-bounded-quantile", type=float, default=0.95)
    ap.add_argument("--v70-bounded-scale", type=float, default=1.25)
    ap.add_argument("--v70-candidate-modes", default="route,augmented_anchor,shrink_to_v52")
    ap.add_argument("--v70-shrink-grid", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--v70-anchor-prior-mass", type=float, default=0.35)
    ap.add_argument("--v70-top-m-grid", default="1,2,4,8,12")
    ap.add_argument("--v70-temperature-grid", default="0.35,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--v70-blend-grid", default="0,0.05,0.10,0.20,0.35,0.50,0.75,1.0")
    ap.add_argument("--v70-prior-alpha-grid", default="0,0.5,1.0,2.0")
    ap.add_argument("--v70-risk-beta-grid", default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--v70-anchor-eta-grid", default="0,0.25,0.5,1.0,2.0")
    ap.add_argument("--v70-jump-factor", type=float, default=3.0)
    ap.add_argument("--v70-hgbdt-iter", type=int, default=120)
    ap.add_argument("--v70-hgbdt-lr", type=float, default=0.045)
    ap.add_argument("--v70-hgbdt-leaf-nodes", type=int, default=31)
    ap.add_argument("--v70-hgbdt-l2", type=float, default=0.02)
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
        args.v70_candidate_modes = "route,augmented_anchor,shrink_to_v52"
        args.v70_shrink_grid = "0.5,1.0"
        args.v70_top_m_grid = "1,4,8"
        args.v70_temperature_grid = "0.5,1.0,2.0"
        args.v70_blend_grid = "0,0.1,0.35,0.75,1.0"
        args.v70_prior_alpha_grid = "0,1.0"
        args.v70_risk_beta_grid = "0,0.5,1.0"
        args.v70_anchor_eta_grid = "0,0.5,1.0"
        args.v70_hgbdt_iter = min(args.v70_hgbdt_iter, 60)
    return args


if __name__ == "__main__":
    run(parse_args())
