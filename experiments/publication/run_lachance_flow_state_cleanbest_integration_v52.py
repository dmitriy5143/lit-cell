#!/usr/bin/env python3
"""Flow-state clean-best integration v52.

This runner closes the v51 branch in the intended way:

    clean-best v16/v23 fixed route basis
    + raw_context_v2/full60k
    + causal local tissue-flow conditioner
    + reliability/noise auxiliary signal
    + bounded sequence calibration

It deliberately does not use KMeans route labels as a final hard selector and
does not use video/mask/raw-state features.  Target/future is used only for
supervised calibration labels, reliability labels, and diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_flow_regime_route_generator_v51 as v51  # noqa: E402
import run_lachance_h1_sequence_raw_context_decoder as seq  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_route_conditioned_generator_v12 as v12  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402
from run_lachance_regime_magnitude_sequence_decoder import apply_train_position_norm  # noqa: E402


DEFAULT_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_raw_context_v2_grid_bulk_full60k_2026-06-19"
    / "raw_context_v2_feature_grid.csv"
)
DEFAULT_EDGE_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_track_only_feature_grid_mdck_edge_v41_seed7_42_123_compact_2026-07-01"
    / "combined_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "flow_state_cleanbest_integration_v52_2026-07-10"
EPS = 1e-8
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class FlowPacket:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    feature_names: list[str]
    control: str


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def parse_floats(text: str) -> list[float]:
    return [float(s) for s in parse_strs(text)]


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return np.clip(x, -1.0e4, 1.0e4).astype(np.float32, copy=False)


def flat_to_steps(x: np.ndarray, max_horizon: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(len(x), int(max_horizon), 2)


def endpoint_rmse_flat(pred: np.ndarray, true_flat: np.ndarray, horizons: list[int], max_horizon: int) -> float:
    pred_steps = flat_to_steps(pred, max_horizon)
    true_steps = flat_to_steps(true_flat, max_horizon)
    vals = []
    for h in horizons:
        p = np.sum(pred_steps[:, : int(h)], axis=1)
        y = np.sum(true_steps[:, : int(h)], axis=1)
        vals.append(np.mean(np.sum((p - y) ** 2, axis=1)))
    return float(np.sqrt(np.mean(vals)))


def endpoint_error_h(pred_flat: np.ndarray, true_flat: np.ndarray, h: int, max_horizon: int) -> np.ndarray:
    pred_steps = flat_to_steps(pred_flat, max_horizon)
    true_steps = flat_to_steps(true_flat, max_horizon)
    p = np.sum(pred_steps[:, : int(h)], axis=1)
    y = np.sum(true_steps[:, : int(h)], axis=1)
    return np.sqrt(np.sum((p - y) ** 2, axis=1)).astype(np.float32)


def select_context(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray, max_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xtr.shape[1] <= int(max_cols):
        return xtr.astype(np.float32), xva.astype(np.float32), xte.astype(np.float32)
    var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-int(max_cols) :]
    return xtr[:, keep].astype(np.float32), xva[:, keep].astype(np.float32), xte[:, keep].astype(np.float32)


def fit_ridge_endpoint_all(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    horizons: list[int],
    max_horizon: int,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    scaler = StandardScaler()
    ztr = np.clip(np.nan_to_num(scaler.fit_transform(xtr).astype(np.float32)), -8.0, 8.0)
    zva = np.clip(np.nan_to_num(scaler.transform(xva).astype(np.float32)), -8.0, 8.0)
    zte = np.clip(np.nan_to_num(scaler.transform(xte).astype(np.float32)), -8.0, 8.0)
    best_model: Ridge | None = None
    best_alpha = float(alphas[0])
    best_rmse = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), solver="svd")
        model.fit(ztr, ytr)
        pred = np.nan_to_num(model.predict(zva).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        score = endpoint_rmse_flat(pred, yva, horizons, max_horizon)
        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)
            best_model = model
    if best_model is None:
        raise RuntimeError("ridge fit failed")
    return (
        np.nan_to_num(best_model.predict(ztr).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(best_model.predict(zva).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(best_model.predict(zte).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        best_alpha,
        best_rmse,
    )


def add_v0_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["v0_dx"] = out["dx_px"].fillna(0.0).astype(float)
    out["v0_dy"] = out["dy_px"].fillna(0.0).astype(float)
    out["v0_speed"] = np.sqrt(out["v0_dx"] * out["v0_dx"] + out["v0_dy"] * out["v0_dy"])
    return out


def prepare_sampled_split_with_flow(args: argparse.Namespace) -> tuple[seq.SplitData, list[str]]:
    features = pd.read_csv(args.features)
    full = seq.build_sequence_table(
        features=features,
        table_root=Path(args.table_root),
        dataset=args.dataset,
        max_horizon=args.max_horizon,
    )
    split_full = seq.make_split(full, parse_ints(args.train_seq), parse_ints(args.val_seq), parse_ints(args.test_seq), args.seed)
    split_full = apply_train_position_norm(split_full)

    train = add_v0_columns(split_full.train)
    val = add_v0_columns(split_full.val)
    test = add_v0_columns(split_full.test)
    train["_split_name"] = "train"
    val["_split_name"] = "val"
    test["_split_name"] = "test"
    pool = pd.concat([train, val, test], ignore_index=True)
    pool["_row_uid"] = np.arange(len(pool), dtype=np.int64)

    train = pool[pool["_split_name"].eq("train")].copy()
    val = pool[pool["_split_name"].eq("val")].copy()
    test = pool[pool["_split_name"].eq("test")].copy()
    train_s = seq.sample_rows(train, args.max_train_rows, args.seed + 11)
    val_s = seq.sample_rows(val, args.max_val_rows, args.seed + 23)
    test_s = seq.sample_rows(test, args.max_test_rows, args.seed + 37)

    ks = parse_ints(args.local_ks)
    radii = parse_floats(args.local_radii)
    train_f, flow_cols = v51.add_local_flow_features(pool, train_s, ks, radii)
    val_f, _ = v51.add_local_flow_features(pool, val_s, ks, radii)
    test_f, _ = v51.add_local_flow_features(pool, test_s, ks, radii)

    rename = {c: f"v52{c}" for c in flow_cols}
    train_f = train_f.rename(columns=rename)
    val_f = val_f.rename(columns=rename)
    test_f = test_f.rename(columns=rename)
    flow_names = [rename[c] for c in flow_cols]
    return seq.SplitData(train=train_f, val=val_f, test=test_f), flow_names


def row_shuffle(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return x[rng.permutation(len(x))].astype(np.float32)


def time_shuffle_matrix(x: np.ndarray, df: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros_like(x, dtype=np.float32)
    for seq_id, idx in df.groupby("sequence", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        out[idx_arr] = x[idx_arr[rng.permutation(len(idx_arr))]]
    return out.astype(np.float32)


def build_flow_packets(split: seq.SplitData, flow_cols: list[str], args: argparse.Namespace) -> dict[str, FlowPacket]:
    real = (
        safe_matrix(split.train, flow_cols),
        safe_matrix(split.val, flow_cols),
        safe_matrix(split.test, flow_cols),
    )
    zeros = tuple(np.zeros_like(m, dtype=np.float32) for m in real)
    packets: dict[str, FlowPacket] = {
        "clean_best_no_flow": FlowPacket("clean_best_no_flow", np.zeros((len(split.train), 0), np.float32), np.zeros((len(split.val), 0), np.float32), np.zeros((len(split.test), 0), np.float32), [], "none"),
        "clean_best_real_flow": FlowPacket("clean_best_real_flow", real[0], real[1], real[2], flow_cols, "real"),
        "clean_best_no_flow_state": FlowPacket("clean_best_no_flow_state", zeros[0], zeros[1], zeros[2], flow_cols, "zero"),
        "clean_best_shuffled_flow": FlowPacket(
            "clean_best_shuffled_flow",
            row_shuffle(real[0], args.seed + 5201),
            row_shuffle(real[1], args.seed + 5202),
            row_shuffle(real[2], args.seed + 5203),
            flow_cols,
            "row_shuffled",
        ),
        "clean_best_time_shuffled_flow": FlowPacket(
            "clean_best_time_shuffled_flow",
            time_shuffle_matrix(real[0], split.train, args.seed + 5211),
            time_shuffle_matrix(real[1], split.val, args.seed + 5212),
            time_shuffle_matrix(real[2], split.test, args.seed + 5213),
            flow_cols,
            "time_shuffled",
        ),
    }
    requested = parse_strs(args.flow_packets)
    if requested:
        alias = {
            "real": "clean_best_real_flow",
            "no_flow": "clean_best_no_flow",
            "zero": "clean_best_no_flow_state",
            "shuffled": "clean_best_shuffled_flow",
            "time_shuffled": "clean_best_time_shuffled_flow",
        }
        keep = {alias.get(x, x) for x in requested}
        packets = {k: v for k, v in packets.items() if k in keep}
    return packets


def prepare_route_data_with_v16(args: argparse.Namespace, device: Any):
    # Use the exact v16 route-basis machinery.  This keeps v52 a conditioner
    # integration test rather than a new route-generator implementation.
    return v16.build_route_data(args, device)


def route_oracle_rows(arrays: audit.SplitArrays, packs: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(v16.metric_rows(arrays, v16.seq.mean_candidate_residual(packs["test"]), "v12_candidate_mean", args, {"stage": "candidate_mean"}))
    for k in parse_ints(args.oracle_k):
        kk = min(int(k), int(args.candidate_k))
        rows.extend(v16.metric_rows(arrays, v16.seq.oracle_residual(packs["test"], arrays.residual_test, kk), f"v12_oracle@{kk}", args, {"stage": "candidate_oracle", "oracle_k": kk}))
    return rows


def prior_stats(probs: np.ndarray) -> np.ndarray:
    return v16.prior_stats(probs)


def build_calibration_features(
    *,
    route_pred: np.ndarray,
    probs: np.ndarray,
    context: np.ndarray,
    mix: np.ndarray,
    kind: str,
    top_c: int,
) -> np.ndarray:
    parts = [mix.astype(np.float32), prior_stats(probs)]
    if kind in {"correction_context", "stacked_context", "stacked_top_context"}:
        parts.append(context.astype(np.float32))
    if kind in {"stacked", "stacked_context"}:
        parts.append(route_pred.reshape(len(route_pred), -1).astype(np.float32))
    if kind in {"stacked_top", "stacked_top_context"}:
        parts.append(v16.sorted_top_route_outputs(route_pred, probs, top_c))
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_reliability(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr_flat: np.ndarray,
    yva_flat: np.ndarray,
    yte_flat: np.ndarray,
    base_tr_flat: np.ndarray,
    base_va_flat: np.ndarray,
    base_te_flat: np.ndarray,
    args: argparse.Namespace,
    packet: str,
    base_mix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    h = max(args.horizons)
    err_tr = np.log1p(endpoint_error_h(base_tr_flat, ytr_flat, h, args.max_horizon))
    err_va = np.log1p(endpoint_error_h(base_va_flat, yva_flat, h, args.max_horizon))
    err_te = np.log1p(endpoint_error_h(base_te_flat, yte_flat, h, args.max_horizon))

    scaler = StandardScaler()
    ztr = np.clip(np.nan_to_num(scaler.fit_transform(xtr).astype(np.float32)), -8.0, 8.0)
    zva = np.clip(np.nan_to_num(scaler.transform(xva).astype(np.float32)), -8.0, 8.0)
    zte = np.clip(np.nan_to_num(scaler.transform(xte).astype(np.float32)), -8.0, 8.0)

    folds = max(1, int(args.reliability_oof_folds))
    pred_tr = np.zeros(len(ztr), dtype=np.float32)
    if folds > 1 and len(ztr) >= folds * 10:
        kf = KFold(n_splits=folds, shuffle=True, random_state=int(args.seed) + 5299)
        for fold, (ii, jj) in enumerate(kf.split(ztr)):
            model = HistGradientBoostingRegressor(
                max_iter=int(args.hgbdt_iter),
                learning_rate=float(args.hgbdt_lr),
                max_leaf_nodes=int(args.hgbdt_max_leaf_nodes),
                l2_regularization=float(args.hgbdt_l2),
                random_state=int(args.seed) + 5300 + fold,
            )
            model.fit(ztr[ii], err_tr[ii])
            pred_tr[jj] = model.predict(ztr[jj]).astype(np.float32)
    else:
        pred_tr[:] = float(np.mean(err_tr))

    model = HistGradientBoostingRegressor(
        max_iter=int(args.hgbdt_iter),
        learning_rate=float(args.hgbdt_lr),
        max_leaf_nodes=int(args.hgbdt_max_leaf_nodes),
        l2_regularization=float(args.hgbdt_l2),
        random_state=int(args.seed) + 5311,
    )
    model.fit(ztr, err_tr)
    pred_va = model.predict(zva).astype(np.float32)
    pred_te = model.predict(zte).astype(np.float32)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    rows: list[dict[str, Any]] = []
    for split_name, pred, actual in [
        ("train_oof", pred_tr, err_tr),
        ("val", pred_va, err_va),
        ("test", pred_te, err_te),
    ]:
        rows.append(
            {
                "packet": packet,
                "base_mix": base_mix,
                "split": split_name,
                "pred_error_corr": corr(np.expm1(pred), np.expm1(actual)),
                "pred_log_error_rmse": audit.rmse(actual[:, None], pred[:, None]),
                "actual_error_mean": float(np.mean(np.expm1(actual))),
                "pred_error_mean": float(np.mean(np.expm1(pred))),
            }
        )
    q = pd.qcut(pd.Series(pred_te), q=4, labels=False, duplicates="drop")
    actual_te = np.expm1(err_te)
    for b in sorted(q.dropna().unique()):
        mask = q.to_numpy() == int(b)
        rows.append(
            {
                "packet": packet,
                "base_mix": base_mix,
                "split": "test",
                "bin": f"pred_error_q{int(b)+1}",
                "rows": int(mask.sum()),
                "actual_error_mean": float(np.mean(actual_te[mask])) if mask.any() else float("nan"),
                "pred_error_mean": float(np.mean(np.expm1(pred_te[mask]))) if mask.any() else float("nan"),
            }
        )
    return pred_tr[:, None], pred_va[:, None], pred_te[:, None], rows


def bound_prediction(base_flat: np.ndarray, pred_flat: np.ndarray, ytr_flat: np.ndarray, base_tr_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    corr = np.asarray(pred_flat - base_flat, dtype=np.float32)
    train_corr = np.asarray(ytr_flat - base_tr_flat, dtype=np.float32)
    cap = float(np.quantile(np.linalg.norm(train_corr, axis=1), float(args.bounded_quantile))) * float(args.bounded_scale)
    cap = max(cap, 1.0)
    norm = np.linalg.norm(corr, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / np.maximum(norm, EPS))
    return (base_flat + corr * scale).astype(np.float32)


def run_single(seed: int, args: argparse.Namespace, *, dataset_tag: str = "primary") -> dict[str, pd.DataFrame]:
    local = copy.copy(args)
    local.seed = int(seed)
    local.horizons = parse_ints(args.horizons) if isinstance(args.horizons, str) else list(args.horizons)
    np.random.seed(local.seed)
    device = closure.device_from_arg(local.device)

    arrays, labels, prior, bank, packs, gate, meta = prepare_route_data_with_v16(local, device)
    split_flow, flow_cols = prepare_sampled_split_with_flow(local)
    packets = build_flow_packets(split_flow, flow_cols, local)

    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    yte = audit.flatten_residual(arrays.residual_test).astype(np.float32)
    rtr = v16.route_outputs(bank, prior.x_train)
    rva = v16.route_outputs(bank, prior.x_val)
    rte = v16.route_outputs(bank, prior.x_test)
    xtr0, xva0, xte0 = select_context(prior.x_train, prior.x_val, prior.x_test, local.v16c_max_context_features)

    fixed_specs = [
        ("expert_top4_uniform", 4, "uniform", 1.0),
        ("expert_top8_uniform", 8, "uniform", 1.0),
        ("expert_top8_prior_p050", 8, "prior", 0.5),
        ("expert_top8_prior_p100", 8, "prior", 1.0),
        ("expert_all_uniform", 999, "all_uniform", 1.0),
    ]
    mixes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    oracle_rows = route_oracle_rows(arrays, packs, local)

    for name, top_c, mode, power in fixed_specs:
        mixes[name] = (
            v16.mix_route_outputs(rtr, prior.probs_train, top_c, mode, power),
            v16.mix_route_outputs(rva, prior.probs_val, top_c, mode, power),
            v16.mix_route_outputs(rte, prior.probs_test, top_c, mode, power),
        )
        rows.extend(
            v16.metric_rows(
                arrays,
                flat_to_steps(mixes[name][2], local.max_horizon),
                f"v52_{name}_fixed",
                local,
                {"stage": "fixed_route_mix", "packet": "fixed_no_calibration", "base_mix": name, "dataset_tag": dataset_tag},
            )
        )

    base_mix_names = parse_strs(local.v16c_base_mixes)
    calibrators = parse_strs(local.v16c_calibrators)
    alphas = parse_floats(local.v16c_ridge_alphas)
    for packet_name, packet in packets.items():
        xtr = np.concatenate([xtr0, packet.train], axis=1).astype(np.float32)
        xva = np.concatenate([xva0, packet.val], axis=1).astype(np.float32)
        xte = np.concatenate([xte0, packet.test], axis=1).astype(np.float32)
        for base_name in base_mix_names:
            if base_name not in mixes:
                continue
            mtr, mva, mte = mixes[base_name]
            rel_tr = rel_va = rel_te = None
            if bool(local.use_reliability_head):
                rel_tr, rel_va, rel_te, rr = fit_reliability(
                    xtr=xtr,
                    xva=xva,
                    xte=xte,
                    ytr_flat=ytr,
                    yva_flat=yva,
                    yte_flat=yte,
                    base_tr_flat=mtr,
                    base_va_flat=mva,
                    base_te_flat=mte,
                    args=local,
                    packet=packet_name,
                    base_mix=base_name,
                )
                rel_rows.extend(rr)
            for kind in calibrators:
                ftr = build_calibration_features(route_pred=rtr, probs=prior.probs_train, context=xtr, mix=mtr, kind=kind, top_c=local.v16c_top_c)
                fva = build_calibration_features(route_pred=rva, probs=prior.probs_val, context=xva, mix=mva, kind=kind, top_c=local.v16c_top_c)
                fte = build_calibration_features(route_pred=rte, probs=prior.probs_test, context=xte, mix=mte, kind=kind, top_c=local.v16c_top_c)
                feature_variants = [("plain", ftr, fva, fte, False)]
                if rel_tr is not None and rel_va is not None and rel_te is not None:
                    feature_variants.append(
                        (
                            "reliability",
                            np.concatenate([ftr, rel_tr], axis=1).astype(np.float32),
                            np.concatenate([fva, rel_va], axis=1).astype(np.float32),
                            np.concatenate([fte, rel_te], axis=1).astype(np.float32),
                            True,
                        )
                    )
                for suffix, gtr, gva, gte, uses_rel in feature_variants:
                    if kind.startswith("correction"):
                        ptr, pva, pte_corr, alpha, val_rmse = fit_ridge_endpoint_all(
                            xtr=gtr,
                            xva=gva,
                            xte=gte,
                            ytr=ytr - mtr,
                            yva=yva - mva,
                            horizons=local.horizons,
                            max_horizon=local.max_horizon,
                            alphas=alphas,
                        )
                        pte = mte + pte_corr
                        ptr_abs = mtr + ptr
                        label = f"v52_{packet_name}_{base_name}_{kind}_{suffix}_ridge_correction"
                    else:
                        ptr_abs, pva, pte, alpha, val_rmse = fit_ridge_endpoint_all(
                            xtr=gtr,
                            xva=gva,
                            xte=gte,
                            ytr=ytr,
                            yva=yva,
                            horizons=local.horizons,
                            max_horizon=local.max_horizon,
                            alphas=alphas,
                        )
                        label = f"v52_{packet_name}_{base_name}_{kind}_{suffix}_ridge_stack"
                    rows.extend(
                        v16.metric_rows(
                            arrays,
                            flat_to_steps(pte, local.max_horizon),
                            label,
                            local,
                            {
                                "stage": "v52_cleanbest_calibration",
                                "packet": packet_name,
                                "flow_control": packet.control,
                                "base_mix": base_name,
                                "calibrator": kind,
                                "feature_variant": suffix,
                                "uses_reliability": bool(uses_rel),
                                "bounded": False,
                                "alpha": alpha,
                                "val_endpoint_rmse": val_rmse,
                                "dataset_tag": dataset_tag,
                            },
                        )
                    )
                    diag.append(
                        {
                            "packet": packet_name,
                            "flow_control": packet.control,
                            "base_mix": base_name,
                            "calibrator": kind,
                            "feature_variant": suffix,
                            "uses_reliability": bool(uses_rel),
                            "bounded": False,
                            "alpha": alpha,
                            "val_endpoint_rmse": val_rmse,
                            "feature_dim": int(gtr.shape[1]),
                            "flow_dim": int(packet.train.shape[1]),
                            "dataset_tag": dataset_tag,
                        }
                    )
                    if bool(local.bounded_calibration):
                        pte_b = bound_prediction(mte, pte, ytr, mtr, local)
                        rows.extend(
                            v16.metric_rows(
                                arrays,
                                flat_to_steps(pte_b, local.max_horizon),
                                label + "_bounded",
                                local,
                                {
                                    "stage": "v52_bounded_sequence_calibration",
                                    "packet": packet_name,
                                    "flow_control": packet.control,
                                    "base_mix": base_name,
                                    "calibrator": kind,
                                    "feature_variant": suffix,
                                    "uses_reliability": bool(uses_rel),
                                    "bounded": True,
                                    "alpha": alpha,
                                    "val_endpoint_rmse": val_rmse,
                                    "dataset_tag": dataset_tag,
                                },
                            )
                        )

    summary = pd.DataFrame(rows)
    summary.insert(0, "seed", int(local.seed))
    summary.insert(0, "dataset", str(local.dataset))
    diag_df = pd.DataFrame(diag)
    if not diag_df.empty:
        diag_df.insert(0, "seed", int(local.seed))
        diag_df.insert(0, "dataset", str(local.dataset))
    rel_df = pd.DataFrame(rel_rows)
    if not rel_df.empty:
        rel_df.insert(0, "seed", int(local.seed))
        rel_df.insert(0, "dataset", str(local.dataset))
    oracle = pd.DataFrame(oracle_rows)
    if not oracle.empty:
        oracle.insert(0, "seed", int(local.seed))
        oracle.insert(0, "dataset", str(local.dataset))
        oracle["dataset_tag"] = dataset_tag
    gate = gate.copy()
    if not gate.empty:
        gate.insert(0, "seed", int(local.seed))
        gate.insert(0, "dataset", str(local.dataset))
        gate["dataset_tag"] = dataset_tag
    meta_df = pd.DataFrame(
        [
            {
                "dataset": local.dataset,
                "seed": local.seed,
                "dataset_tag": dataset_tag,
                "route_k": labels.k,
                "route_feature_dim": prior.feature_dim,
                "flow_cols": len(flow_cols),
                "train_rows": len(arrays.residual_train),
                "val_rows": len(arrays.residual_val),
                "test_rows": len(arrays.residual_test),
            }
        ]
    )
    return {
        "summary": summary,
        "ablation": diag_df,
        "reliability": rel_df,
        "oracle": oracle,
        "gate": gate,
        "meta": meta_df,
    }


def best_control_rows(summary: pd.DataFrame) -> pd.DataFrame:
    h6 = summary[summary["horizon"].eq(6)].copy()
    if h6.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in h6.groupby(["dataset", "seed", "dataset_tag", "packet"], dropna=False):
        r = sub.sort_values("rmse").iloc[0]
        rows.append(r.to_dict())
    out = pd.DataFrame(rows)
    real = out[out["packet"].eq("clean_best_real_flow")][["dataset", "seed", "dataset_tag", "rmse"]].rename(columns={"rmse": "real_best_h6"})
    out = out.merge(real, on=["dataset", "seed", "dataset_tag"], how="left")
    out["delta_vs_real_h6"] = out["rmse"] - out["real_best_h6"]
    return out


def aggregate_3seed(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "dataset_tag", "method", "packet", "horizon", "stage"]
    return (
        summary.groupby(group_cols, dropna=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), r2_mean=("r2", "mean"), r2_std=("r2", "std"), n=("seed", "nunique"))
        .reset_index()
        .sort_values(["dataset_tag", "horizon", "rmse_mean"])
    )


def write_report(out_dir: Path, args: argparse.Namespace, tables: dict[str, pd.DataFrame]) -> None:
    summary = tables["summary"]
    controls = tables["controls"]
    rel = tables["reliability"]
    agg = tables["aggregate"]
    lines = ["# v52 Flow-State Clean-Best Integration", ""]
    lines.append("## Decision Tables")
    if not summary.empty:
        for tag in sorted(summary["dataset_tag"].unique()):
            sub = summary[summary["dataset_tag"].eq(tag)]
            lines.append(f"### {tag}")
            h6 = sub[sub["horizon"].eq(6)].sort_values("rmse")
            cols = [c for c in ["dataset", "seed", "method", "packet", "rmse", "r2", "stage", "base_mix", "calibrator", "feature_variant", "bounded", "uses_reliability", "alpha", "val_endpoint_rmse"] if c in h6.columns]
            lines.append(h6[cols].head(40).to_markdown(index=False))
            lines.append("")
    if not controls.empty:
        lines.append("## h6 Control Winners")
        cols = [c for c in ["dataset", "seed", "dataset_tag", "packet", "method", "rmse", "r2", "delta_vs_real_h6", "stage"] if c in controls.columns]
        lines.append(controls[cols].sort_values(["dataset_tag", "seed", "rmse"]).to_markdown(index=False))
        lines.append("")
    if not agg.empty:
        lines.append("## Aggregate")
        h6 = agg[agg["horizon"].eq(6)].sort_values("rmse_mean")
        cols = [c for c in ["dataset", "dataset_tag", "method", "packet", "rmse_mean", "rmse_std", "r2_mean", "n", "stage"] if c in h6.columns]
        lines.append(h6[cols].head(40).to_markdown(index=False))
        lines.append("")
    if not rel.empty:
        lines.append("## Reliability")
        cols = [c for c in ["dataset", "seed", "packet", "base_mix", "split", "pred_error_corr", "actual_error_mean", "pred_error_mean", "bin", "rows"] if c in rel.columns]
        lines.append(rel[cols].head(80).to_markdown(index=False))
        lines.append("")
    lines.append("## Interpretation Guard")
    lines.append("- `real_flow` must beat no-flow and shuffled/time-shuffled controls to count.")
    lines.append("- If gains are below 1% or reproduced by controls, v52 is closed as non-breakthrough.")
    lines.append("- Edge is a guard for flow transfer; Bulk full60k remains the primary clean-best integration target.")
    (out_dir / "v52_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_smoke(args: argparse.Namespace) -> None:
    args.max_train_rows = min(int(args.max_train_rows), 3000)
    args.max_val_rows = min(int(args.max_val_rows), 1000)
    args.max_test_rows = min(int(args.max_test_rows), 1500)
    args.posterior_epochs = min(int(args.posterior_epochs), 4)
    args.student_epochs = min(int(args.student_epochs), 4)
    args.candidate_k = min(int(args.candidate_k), 16)
    args.oracle_k = "4,8,16"
    args.v16c_base_mixes = "expert_top8_uniform"
    args.v16c_calibrators = "correction_context,stacked_top_context"
    args.flow_packets = "real,no_flow,shuffled"
    args.reliability_oof_folds = min(int(args.reliability_oof_folds), 2)
    args.hgbdt_iter = min(int(args.hgbdt_iter), 60)


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        configure_smoke(args)
    args.horizons = parse_ints(args.horizons) if isinstance(args.horizons, str) else list(args.horizons)

    all_tables = {k: [] for k in ["summary", "ablation", "reliability", "oracle", "gate", "meta"]}
    seeds = parse_ints(args.confirm_seeds) if str(args.confirm_seeds).strip() else [int(args.seed)]
    for seed in seeds:
        res = run_single(seed, args, dataset_tag="bulk")
        for k, v in res.items():
            all_tables[k].append(v)

    if bool(args.edge_guard) and not args.smoke:
        if args.edge_features.exists():
            edge_args = copy.copy(args)
            edge_args.dataset = "MDCK_Edge"
            edge_args.features = args.edge_features
            edge_args.train_seq = args.edge_train_seq
            edge_args.val_seq = args.edge_val_seq
            edge_args.test_seq = args.edge_test_seq
            edge_args.max_train_rows = min(int(args.max_train_rows), int(args.edge_max_train_rows))
            edge_args.max_val_rows = min(int(args.max_val_rows), int(args.edge_max_val_rows))
            edge_args.max_test_rows = min(int(args.max_test_rows), int(args.edge_max_test_rows))
            res = run_single(int(args.seed), edge_args, dataset_tag="edge_guard")
            for k, v in res.items():
                all_tables[k].append(v)
        else:
            all_tables["meta"].append(pd.DataFrame([{"dataset": "MDCK_Edge", "dataset_tag": "edge_guard", "status": "skipped_missing_edge_features", "path": str(args.edge_features)}]))

    tables = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in all_tables.items()}
    tables["controls"] = best_control_rows(tables["summary"])
    tables["aggregate"] = aggregate_3seed(tables["summary"])

    tables["summary"].to_csv(args.out_dir / "v52_flow_cleanbest_summary.csv", index=False)
    tables["ablation"].to_csv(args.out_dir / "v52_flow_cleanbest_ablation.csv", index=False)
    tables["reliability"].to_csv(args.out_dir / "v52_flow_reliability.csv", index=False)
    tables["controls"].to_csv(args.out_dir / "v52_flow_controls.csv", index=False)
    tables["oracle"].to_csv(args.out_dir / "v52_flow_route_oracle_gap.csv", index=False)
    tables["aggregate"].to_csv(args.out_dir / "v52_3seed_aggregate.csv", index=False)
    tables["gate"].to_csv(args.out_dir / "v52_route_prior_gate.csv", index=False)
    tables["meta"].to_csv(args.out_dir / "v52_meta.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, tables)
    print(json.dumps({"out_dir": str(args.out_dir), "elapsed_sec": time.time() - t0, "summary_rows": len(tables["summary"])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(features=DEFAULT_FEATURES, out_dir=DEFAULT_OUT)
    parser.add_argument("--flow-v52", action="store_true")
    parser.add_argument("--flow-packets", type=str, default="real,no_flow,zero,shuffled,time_shuffled")
    parser.add_argument("--use-reliability-head", action="store_true")
    parser.add_argument("--bounded-calibration", action="store_true")
    parser.add_argument("--confirm-seeds", type=str, default="")
    parser.add_argument("--edge-guard", action="store_true")
    parser.add_argument("--edge-features", type=Path, default=DEFAULT_EDGE_FEATURES)
    parser.add_argument("--edge-train-seq", type=str, default="2,3,4,5")
    parser.add_argument("--edge-val-seq", type=str, default="6")
    parser.add_argument("--edge-test-seq", type=str, default="7")
    parser.add_argument("--edge-max-train-rows", type=int, default=30000)
    parser.add_argument("--edge-max-val-rows", type=int, default=8000)
    parser.add_argument("--edge-max-test-rows", type=int, default=10000)
    parser.add_argument("--local-ks", type=str, default="8,16,32")
    parser.add_argument("--local-radii", type=str, default="64,128,256")
    parser.add_argument("--ridge-alphas", type=str, default="30,100,300,1000,3000,10000")
    parser.add_argument("--hgbdt-iter", type=int, default=120)
    parser.add_argument("--reliability-oof-folds", type=int, default=3)
    parser.add_argument("--bounded-quantile", type=float, default=0.95)
    parser.add_argument("--bounded-scale", type=float, default=1.25)
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
    run(args)


if __name__ == "__main__":
    main()
