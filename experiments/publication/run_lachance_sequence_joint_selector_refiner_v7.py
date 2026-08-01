#!/usr/bin/env python3
"""Sequence/Joint Candidate Selector-Refiner v7 for LaChance trajectories.

This runner is intentionally focused on the current bottleneck:

    strong K64 candidate cloud exists,
    but v3/fps prototypes and scalar risk models do not recover the oracle.

v7 keeps the full candidate set and trains a joint candidate transformer to
score all trajectories.  Inference uses sparse top-M mixtures and reports
route/source diagnostics.  Target futures are used only for training labels and
metrics, never as inference features.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "sequence_joint_selector_refiner_v7_2026-07-02"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def maybe_shuffle(x: np.ndarray, seed: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.permutation(len(x))]


def standardize_3d(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    flat = train.reshape(-1, train.shape[-1])
    mean = np.nanmean(flat, axis=0, keepdims=True)
    std = np.nanstd(flat, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def z(x: np.ndarray) -> np.ndarray:
        y = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return z(train), z(val), z(test), {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


def route_signature(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Compact trajectory signature used for route-regime pseudo-labels."""
    n = residual.shape[0]
    endpoints = np.stack([np.sum(residual[:, : int(h), :], axis=1) for h in horizons], axis=1)
    speed = np.linalg.norm(residual, axis=-1)
    parts = [
        residual.reshape(n, -1),
        endpoints.reshape(n, -1),
        np.mean(speed, axis=1, keepdims=True),
        np.std(speed, axis=1, keepdims=True),
        speed[:, :1],
        speed[:, -1:],
    ]
    if residual.shape[1] > 1:
        prev = residual[:, :-1, :]
        nxt = residual[:, 1:, :]
        denom = np.maximum(np.linalg.norm(prev, axis=-1) * np.linalg.norm(nxt, axis=-1), EPS)
        turn_cos = np.sum(prev * nxt, axis=-1) / denom
        turn_sin = (prev[..., 0] * nxt[..., 1] - prev[..., 1] * nxt[..., 0]) / denom
        parts.extend([np.mean(turn_cos, axis=1, keepdims=True), np.mean(turn_sin, axis=1, keepdims=True)])
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def route_signature_candidates(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    n, k, h, d = residual.shape
    flat = residual.reshape(n * k, h, d)
    return route_signature(flat, horizons).reshape(n, k, -1).astype(np.float32)


def padded_proba(model: LogisticRegression, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.full((len(x), n_classes), 1e-6, dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    out /= np.maximum(out.sum(axis=1, keepdims=True), EPS)
    return out.astype(np.float32)


def top3_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    if p.shape[1] <= 1:
        return float("nan")
    try:
        return float(top_k_accuracy_score(y, p, k=min(3, p.shape[1]), labels=np.arange(p.shape[1])))
    except Exception:
        order = np.argsort(-p, axis=1)[:, : min(3, p.shape[1])]
        return float(np.mean([int(y[i]) in set(order[i]) for i in range(len(y))]))


def fit_video_route_hints(
    *,
    arrays: audit.SplitArrays,
    edge_train: np.ndarray,
    edge_val: np.ndarray,
    edge_test: np.ndarray,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cand_test: seq.CandidatePack,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sig_train = route_signature(arrays.residual_train, args.horizons)
    sig_val = route_signature(arrays.residual_val, args.horizons)
    sig_test = route_signature(arrays.residual_test, args.horizons)
    sig_scaler = StandardScaler()
    ztr = sig_scaler.fit_transform(sig_train).astype(np.float32)
    zva = sig_scaler.transform(sig_val).astype(np.float32)
    zte = sig_scaler.transform(sig_test).astype(np.float32)
    k = min(int(args.video_hint_k), max(2, len(ztr) // 25))
    km = KMeans(n_clusters=k, n_init=20, random_state=int(args.seed) + 51001)
    ytr = km.fit_predict(ztr).astype(np.int64)
    yva = km.predict(zva).astype(np.int64)
    yte = km.predict(zte).astype(np.int64)

    x_scaler = StandardScaler()
    xtr = x_scaler.fit_transform(edge_train).astype(np.float32)
    xva = x_scaler.transform(edge_val).astype(np.float32)
    xte = x_scaler.transform(edge_test).astype(np.float32)
    clf = LogisticRegression(
        max_iter=int(args.video_hint_max_iter),
        C=float(args.video_hint_c),
        class_weight="balanced",
        random_state=int(args.seed) + 52001,
    )
    clf.fit(xtr, ytr)
    ptr = padded_proba(clf, xtr, k)
    pva = padded_proba(clf, xva, k)
    pte = padded_proba(clf, xte, k)

    def assign_candidates(cand: seq.CandidatePack) -> np.ndarray:
        sig = route_signature_candidates(cand.residual, args.horizons)
        z = sig_scaler.transform(sig.reshape(-1, sig.shape[-1])).astype(np.float32)
        d = np.sum((z[:, None, :] - km.cluster_centers_[None, :, :]) ** 2, axis=-1)
        return np.argmin(d, axis=1).reshape(sig.shape[:2]).astype(np.int64)

    cm_tr = assign_candidates(cand_train)
    cm_va = assign_candidates(cand_val)
    cm_te = assign_candidates(cand_test)
    rows = []
    for split_name, y, p in [("train", ytr, ptr), ("val", yva, pva), ("test", yte, pte)]:
        rows.append(
            {
                "split": split_name,
                "route_k": int(k),
                "video_hint_top1": float(np.mean(np.argmax(p, axis=1) == y)),
                "video_hint_top3": top3_accuracy(y, p),
                "video_hint_nll": float(log_loss(y, np.clip(p, 1e-6, 1.0), labels=np.arange(k))),
                "mean_entropy": float(-np.mean(np.sum(p * np.log(np.maximum(p, EPS)), axis=1))),
            }
        )
    pack = {
        "k": int(k),
        "prob_train": ptr,
        "prob_val": pva,
        "prob_test": pte,
        "candidate_mode_train": cm_tr,
        "candidate_mode_val": cm_va,
        "candidate_mode_test": cm_te,
    }
    return pack, pd.DataFrame(rows)


def _fit_multioutput_ridge(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train).astype(np.float32)
    xva = scaler.transform(x_val).astype(np.float32)
    xte = scaler.transform(x_test).astype(np.float32)
    best: tuple[float, Ridge, float] | None = None
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
        model = Ridge(alpha=float(alpha))
        model.fit(xtr, y_train)
        pred_val = model.predict(xva).astype(np.float32)
        score = float(np.sqrt(np.mean(np.square(pred_val - y_val))))
        if best is None or score < best[0]:
            best = (score, model, float(alpha))
    assert best is not None
    model = best[1]
    return (
        model.predict(xtr).astype(np.float32),
        model.predict(xva).astype(np.float32),
        model.predict(xte).astype(np.float32),
        {"model": "ridge", "alpha": float(best[2]), "val_flat_rmse": float(best[0]), "feature_dim": int(x_train.shape[1])},
    )


def _fit_multioutput_hgbdt(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train).astype(np.float32)
    xva = scaler.transform(x_val).astype(np.float32)
    xte = scaler.transform(x_test).astype(np.float32)
    pred_tr: list[np.ndarray] = []
    pred_va: list[np.ndarray] = []
    pred_te: list[np.ndarray] = []
    for j in range(y_train.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=int(args.video_residual_hgbdt_iter),
            learning_rate=float(args.video_residual_hgbdt_lr),
            max_leaf_nodes=int(args.video_residual_hgbdt_leaf_nodes),
            l2_regularization=float(args.video_residual_hgbdt_l2),
            random_state=int(args.seed) + 53000 + j,
        )
        model.fit(xtr, y_train[:, j])
        pred_tr.append(model.predict(xtr).astype(np.float32))
        pred_va.append(model.predict(xva).astype(np.float32))
        pred_te.append(model.predict(xte).astype(np.float32))
    pred_val = np.column_stack(pred_va).astype(np.float32)
    return (
        np.column_stack(pred_tr).astype(np.float32),
        pred_val,
        np.column_stack(pred_te).astype(np.float32),
        {
            "model": "hgbdt",
            "val_flat_rmse": float(np.sqrt(np.mean(np.square(pred_val - y_val)))),
            "feature_dim": int(x_train.shape[1]),
            "outputs": int(y_train.shape[1]),
        },
    )


def fit_video_residual_hints(
    *,
    arrays: audit.SplitArrays,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    edge_train: np.ndarray,
    edge_val: np.ndarray,
    edge_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal video-to-residual teacher used as candidate compatibility hint.

    The teacher never uses the target at inference.  During training it learns
    residual steps from video/edge-context features, then the critic receives
    only compatibility between each candidate and this video-implied residual.
    """

    ytr = arrays.residual_train.reshape(len(arrays.residual_train), -1).astype(np.float32)
    yva = arrays.residual_val.reshape(len(arrays.residual_val), -1).astype(np.float32)
    yte = arrays.residual_test.reshape(len(arrays.residual_test), -1).astype(np.float32)
    if args.video_residual_include_context:
        xtr = np.concatenate([ctx_train, edge_train], axis=1).astype(np.float32)
        xva = np.concatenate([ctx_val, edge_val], axis=1).astype(np.float32)
        xte = np.concatenate([ctx_test, edge_test], axis=1).astype(np.float32)
        input_mode = "context_plus_edge"
    else:
        xtr, xva, xte = edge_train, edge_val, edge_test
        input_mode = "edge_only"
    if args.video_residual_model == "hgbdt":
        ptr, pva, pte, info = _fit_multioutput_hgbdt(xtr, xva, xte, ytr, yva, args)
    else:
        ptr, pva, pte, info = _fit_multioutput_ridge(xtr, xva, xte, ytr, yva)
    info["input_mode"] = input_mode

    def reshape(pred: np.ndarray) -> np.ndarray:
        return pred.reshape(len(pred), arrays.residual_train.shape[1], 2).astype(np.float32)

    pred_train = reshape(ptr)
    pred_val = reshape(pva)
    pred_test = reshape(pte)
    rows: list[dict[str, Any]] = []
    for split_name, truth, pred in [
        ("train", arrays.residual_train, pred_train),
        ("val", arrays.residual_val, pred_val),
        ("test", arrays.residual_test, pred_test),
    ]:
        for h in args.horizons:
            yt = np.sum(truth[:, : int(h), :], axis=1)
            yp = np.sum(pred[:, : int(h), :], axis=1)
            rows.append(
                {
                    "split": split_name,
                    "model": info.get("model", args.video_residual_model),
                    "horizon": int(h),
                    "rmse": float(np.sqrt(np.mean(np.square(yt - yp)))),
                    "r2": audit.r2_score_np(yt, yp),
                    "flat_val_rmse": float(info.get("val_flat_rmse", np.nan)),
                    "feature_dim": int(info.get("feature_dim", edge_train.shape[1])),
                    "input_mode": input_mode,
                }
            )
    pack = {
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "info": info,
    }
    return pack, pd.DataFrame(rows)


def select_context_block(arrays: audit.SplitArrays, block: str, max_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    max_features = max(1, int(max_features))
    if block not in arrays.x_train:
        return (
            np.zeros((len(arrays.residual_train), 1), dtype=np.float32),
            np.zeros((len(arrays.residual_val), 1), dtype=np.float32),
            np.zeros((len(arrays.residual_test), 1), dtype=np.float32),
            {"block": block, "enabled": False, "reason": "missing"},
        )
    xtr = np.nan_to_num(arrays.x_train[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xva = np.nan_to_num(arrays.x_val[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xte = np.nan_to_num(arrays.x_test[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if xtr.shape[1] > max_features:
        var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-max_features:]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
    xtr, xva, xte, scaler = seq.standardize(xtr, xva, xte)
    return xtr, xva, xte, {"block": block, "enabled": True, "dim": int(xtr.shape[1]), "scaler": finite_json(scaler)}


def candidate_source_fraction(pack: seq.CandidatePack, take: np.ndarray) -> dict[str, float]:
    if pack.features.shape[-1] < 3:
        return {}
    src = pack.features[np.arange(len(take)), take, -3:]
    return {
        "source_generic_frac": float(np.mean(src[:, 0] > 0.5)),
        "source_route_frac": float(np.mean(src[:, 1] > 0.5)),
        "source_learned_frac": float(np.mean(src[:, 2] > 0.5)),
    }


def build_feature_pack(
    *,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cand_test: seq.CandidatePack,
    arrays: audit.SplitArrays,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    edge_train: np.ndarray,
    edge_val: np.ndarray,
    edge_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    args: argparse.Namespace,
    use_context: bool = True,
    use_axis: bool = True,
    use_edge: bool = True,
    shuffle_context: bool = False,
    shuffle_axis: bool = False,
    shuffle_edge: bool = False,
    route_hints: dict[str, Any] | None = None,
    use_route_hints: bool = True,
    shuffle_route_hints: bool = False,
    residual_hints: dict[str, Any] | None = None,
    use_residual_hints: bool = True,
    shuffle_residual_hints: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    def route_logits(cand: seq.CandidatePack) -> np.ndarray:
        if cand.logprob is not None and cand.logprob.size:
            return np.nan_to_num(cand.logprob.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return np.zeros(cand.residual.shape[:2], dtype=np.float32)

    def base_features(cand: seq.CandidatePack, base: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        feat, _ = qrc.query_sequence_features(
            query_pred=cand.residual,
            base=base,
            route_logits=route_logits(cand),
            ctx=ctx,
            horizons=args.horizons,
            include_context=False,
            include_query_id=args.risk_include_query_id,
        )
        return np.concatenate([feat, cand.features.astype(np.float32)], axis=-1).astype(np.float32)

    ftr = base_features(cand_train, arrays.base_train, ctx_train)
    fva = base_features(cand_val, arrays.base_val, ctx_val)
    fte = base_features(cand_test, arrays.base_test, ctx_test)

    if use_axis:
        axtr, axnames = rpr.component_route_features(
            query_pred=cand_train.residual,
            component_pred=component_axes.train,
            horizons=args.horizons,
            temperature=args.component_attention_temperature,
        )
        axva, _ = rpr.component_route_features(
            query_pred=cand_val.residual,
            component_pred=component_axes.val,
            horizons=args.horizons,
            temperature=args.component_attention_temperature,
        )
        axte, _ = rpr.component_route_features(
            query_pred=cand_test.residual,
            component_pred=component_axes.test,
            horizons=args.horizons,
            temperature=args.component_attention_temperature,
        )
        if shuffle_axis:
            axtr = maybe_shuffle(axtr, args.seed + 37001, True)
            axva = maybe_shuffle(axva, args.seed + 37002, True)
            axte = maybe_shuffle(axte, args.seed + 37003, True)
        ftr = np.concatenate([ftr, axtr], axis=-1).astype(np.float32)
        fva = np.concatenate([fva, axva], axis=-1).astype(np.float32)
        fte = np.concatenate([fte, axte], axis=-1).astype(np.float32)
    else:
        axnames = []

    if not use_context:
        ctx_train = np.zeros_like(ctx_train)
        ctx_val = np.zeros_like(ctx_val)
        ctx_test = np.zeros_like(ctx_test)
    if shuffle_context:
        ctx_train = maybe_shuffle(ctx_train, args.seed + 38001, True)
        ctx_val = maybe_shuffle(ctx_val, args.seed + 38002, True)
        ctx_test = maybe_shuffle(ctx_test, args.seed + 38003, True)
    if use_context and ctx_train.shape[1] > 0:
        ftr = np.concatenate([ftr, np.repeat(ctx_train[:, None, :], ftr.shape[1], axis=1)], axis=-1).astype(np.float32)
        fva = np.concatenate([fva, np.repeat(ctx_val[:, None, :], fva.shape[1], axis=1)], axis=-1).astype(np.float32)
        fte = np.concatenate([fte, np.repeat(ctx_test[:, None, :], fte.shape[1], axis=1)], axis=-1).astype(np.float32)

    if not use_edge:
        edge_train = np.zeros_like(edge_train)
        edge_val = np.zeros_like(edge_val)
        edge_test = np.zeros_like(edge_test)
    if shuffle_edge:
        edge_train = maybe_shuffle(edge_train, args.seed + 39001, True)
        edge_val = maybe_shuffle(edge_val, args.seed + 39002, True)
        edge_test = maybe_shuffle(edge_test, args.seed + 39003, True)
    if use_edge and edge_train.shape[1] > 0:
        ftr = np.concatenate([ftr, np.repeat(edge_train[:, None, :], ftr.shape[1], axis=1)], axis=-1).astype(np.float32)
        fva = np.concatenate([fva, np.repeat(edge_val[:, None, :], fva.shape[1], axis=1)], axis=-1).astype(np.float32)
        fte = np.concatenate([fte, np.repeat(edge_test[:, None, :], fte.shape[1], axis=1)], axis=-1).astype(np.float32)

    if route_hints is not None and use_route_hints:
        def hint_features(split_name: str, seed: int) -> np.ndarray:
            prob = route_hints[f"prob_{split_name}"].astype(np.float32)
            cmode = route_hints[f"candidate_mode_{split_name}"].astype(np.int64)
            if shuffle_route_hints:
                prob = maybe_shuffle(prob, seed, True)
            k = int(route_hints["k"])
            picked = np.take_along_axis(prob, cmode, axis=1)[:, :, None]
            entropy = -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=1, keepdims=True)
            entropy_rep = np.repeat(entropy[:, None, :], cmode.shape[1], axis=1)
            prob_rep = np.repeat(prob[:, None, :], cmode.shape[1], axis=1)
            onehot = np.zeros((cmode.shape[0], cmode.shape[1], k), dtype=np.float32)
            rows = np.arange(cmode.shape[0])[:, None]
            cols = np.arange(cmode.shape[1])[None, :]
            onehot[rows, cols, cmode] = 1.0
            top_mode = np.argmax(prob, axis=1)
            mismatch = (cmode != top_mode[:, None]).astype(np.float32)[:, :, None]
            return np.concatenate([picked, 1.0 - picked, entropy_rep, mismatch, prob_rep, onehot], axis=-1).astype(np.float32)

        htr = hint_features("train", args.seed + 39101)
        hva = hint_features("val", args.seed + 39102)
        hte = hint_features("test", args.seed + 39103)
        ftr = np.concatenate([ftr, htr], axis=-1).astype(np.float32)
        fva = np.concatenate([fva, hva], axis=-1).astype(np.float32)
        fte = np.concatenate([fte, hte], axis=-1).astype(np.float32)

    if residual_hints is not None and use_residual_hints:
        def residual_hint_features(split_name: str, cand: seq.CandidatePack, seed: int) -> np.ndarray:
            hint = residual_hints[f"pred_{split_name}"].astype(np.float32)
            if shuffle_residual_hints:
                hint = maybe_shuffle(hint, seed, True)
            hint_rep = hint[:, None, :, :]
            diff = cand.residual.astype(np.float32) - hint_rep
            step_l2 = np.sqrt(np.sum(np.square(diff), axis=-1) + EPS)
            step_mean = np.mean(step_l2, axis=2, keepdims=True)
            step_max = np.max(step_l2, axis=2, keepdims=True)
            step_first = step_l2[:, :, :1]
            step_last = step_l2[:, :, -1:]
            parts = [step_mean, step_max, step_first, step_last]
            for h in args.horizons:
                ce = np.sum(cand.residual[:, :, : int(h), :], axis=2)
                he = np.sum(hint[:, : int(h), :], axis=1)[:, None, :]
                ed = ce - he
                e_l2 = np.sqrt(np.sum(np.square(ed), axis=-1) + EPS)[:, :, None]
                c_norm = np.linalg.norm(ce, axis=-1)
                h_norm = np.linalg.norm(he, axis=-1)
                cos = np.sum(ce * he, axis=-1) / np.maximum(c_norm * h_norm, EPS)
                mag_delta = np.abs(c_norm - h_norm)[:, :, None]
                parts.extend([e_l2.astype(np.float32), cos[:, :, None].astype(np.float32), mag_delta.astype(np.float32)])
            cand_acc = np.diff(cand.residual.astype(np.float32), axis=2)
            hint_acc = np.diff(hint.astype(np.float32), axis=1)[:, None, :, :]
            if cand_acc.shape[2] > 0:
                acc_l2 = np.sqrt(np.sum(np.square(cand_acc - hint_acc), axis=-1) + EPS)
                parts.extend([np.mean(acc_l2, axis=2, keepdims=True), np.max(acc_l2, axis=2, keepdims=True)])
            hint_mag = np.linalg.norm(hint, axis=-1)
            hint_summary = np.concatenate(
                [
                    np.mean(hint_mag, axis=1, keepdims=True),
                    np.std(hint_mag, axis=1, keepdims=True),
                    hint[:, 0, :],
                    hint[:, -1, :],
                ],
                axis=1,
            ).astype(np.float32)
            parts.append(np.repeat(hint_summary[:, None, :], cand.residual.shape[1], axis=1))
            return np.nan_to_num(np.concatenate(parts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        rhtr = residual_hint_features("train", cand_train, args.seed + 39201)
        rhva = residual_hint_features("val", cand_val, args.seed + 39202)
        rhte = residual_hint_features("test", cand_test, args.seed + 39203)
        ftr = np.concatenate([ftr, rhtr], axis=-1).astype(np.float32)
        fva = np.concatenate([fva, rhva], axis=-1).astype(np.float32)
        fte = np.concatenate([fte, rhte], axis=-1).astype(np.float32)

    ftr, fva, fte, scaler = standardize_3d(ftr, fva, fte)
    target_train = qrc.risk_endpoint_errors(cand_train.residual, arrays.residual_train, args)
    target_val = qrc.risk_endpoint_errors(cand_val.residual, arrays.residual_val, args)
    target_test = qrc.risk_endpoint_errors(cand_test.residual, arrays.residual_test, args)
    return (
        {
            "feat_train": ftr,
            "feat_val": fva,
            "feat_test": fte,
            "err_train": target_train,
            "err_val": target_val,
            "err_test": target_test,
        },
        {
            "feature_dim": int(ftr.shape[-1]),
            "scaler": finite_json(scaler),
            "axis_features": axnames[:32],
            "use_context": use_context,
            "use_axis": use_axis,
            "use_edge": use_edge,
            "use_route_hints": bool(route_hints is not None and use_route_hints),
            "shuffle_route_hints": bool(shuffle_route_hints),
            "use_residual_hints": bool(residual_hints is not None and use_residual_hints),
            "shuffle_residual_hints": bool(shuffle_residual_hints),
        },
    )


class JointSelectorV7(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(layers)))
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.block = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 7))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self.in_proj(x))
        return {"risk": self.score(h).squeeze(-1), "block_logits": self.block(h)}


def train_v7(pack: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device, *, shuffled_labels: bool = False) -> tuple[JointSelectorV7, pd.DataFrame, float]:
    model = JointSelectorV7(pack["feat_train"].shape[-1], args.v7_hidden, args.v7_heads, args.v7_layers, args.v7_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.v7_lr, weight_decay=args.v7_weight_decay)
    err_train = pack["err_train"]
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 41001)
        err_train = err_train[rng.permutation(len(err_train))]
    err_val = pack["err_val"]
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack["feat_train"])
    for epoch in range(args.v7_epochs):
        model.train()
        losses = []
        for idx in closure.batches(n, args.v7_batch_size, args.seed + 42000 + epoch):
            x = to_tensor(pack["feat_train"][idx], device)
            err = to_tensor(err_train[idx], device)
            risk = model(x)["risk"]
            soft = qrc.soft_labels_from_error(err, args.v7_label_temperature)
            listwise = -torch.mean(torch.sum(soft * F.log_softmax(-risk, dim=1), dim=1))
            target_log = torch.log1p(err)
            target_z = (target_log - target_log.mean(dim=1, keepdim=True)) / torch.clamp(target_log.std(dim=1, keepdim=True), min=1e-3)
            pred_z = (risk - risk.mean(dim=1, keepdim=True)) / torch.clamp(risk.std(dim=1, keepdim=True), min=1e-3)
            rank = F.smooth_l1_loss(pred_z, target_z)
            entropy = -torch.mean(torch.sum(torch.softmax(-risk, dim=1) * F.log_softmax(-risk, dim=1), dim=1))
            loss = args.v7_listwise_weight * listwise + args.v7_rank_weight * rank + args.v7_entropy_weight * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == args.v7_epochs - 1 or epoch % max(1, args.v7_epochs // 5) == 0:
            risk_val = predict_v7(model, pack["feat_val"], args, device)
            corr = qrc.risk_error_corr(risk_val, err_val)
            # Rank-correlation is the primary validation signal for selector.
            val = -corr
            rows.append({"epoch": int(epoch), "train_loss": float(np.mean(losses)), "val_risk_error_corr": float(corr)})
            if val < best_val:
                best_val = float(val)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), float(-best_val)


def predict_v7(model: JointSelectorV7, feat: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for idx in closure.batches(len(feat), args.v7_batch_size, 43001, shuffle=False):
            out.append(model(to_tensor(feat[idx], device))["risk"].cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def sparse_topm_prediction(risk: np.ndarray, residual: np.ndarray, m: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    m = min(int(m), residual.shape[1])
    idx = np.argsort(risk, axis=1)[:, :m]
    rr = np.take_along_axis(residual, idx[:, :, None, None], axis=1)
    rs = np.take_along_axis(risk, idx, axis=1)
    w = qrc.softmax_np(-rs / max(float(temperature), 1e-6), axis=1)
    pred = np.sum(w[:, :, None, None] * rr, axis=1).astype(np.float32)
    return pred, idx


def tune_topm(risk: np.ndarray, residual: np.ndarray, true: np.ndarray, args: argparse.Namespace) -> tuple[int, float, float]:
    best_m = parse_ints(args.v7_topm)[0]
    best_t = args.v7_temperatures[0]
    best = float("inf")
    for m in parse_ints(args.v7_topm):
        for t in args.v7_temperatures:
            pred, _ = sparse_topm_prediction(risk, residual, int(m), float(t))
            rmse = qrc.residual_endpoint_rmse(true, pred, args.horizons)
            if rmse < best:
                best = float(rmse)
                best_m = int(m)
                best_t = float(t)
    return best_m, best_t, best


def add_metric_rows(rows: list[dict[str, Any]], arrays: audit.SplitArrays, pred: np.ndarray, args: argparse.Namespace, label: str, extra: dict[str, Any]) -> None:
    rows.extend(audit.endpoint_metrics(steps_true=arrays.steps_test, base=arrays.base_test, residual_pred=pred, horizons=args.horizons, label=label, extra=extra))


def run_variant(
    *,
    name: str,
    rows: list[dict[str, Any]],
    logs: list[pd.DataFrame],
    diagnostics: list[dict[str, Any]],
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cand_test: seq.CandidatePack,
    arrays: audit.SplitArrays,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    edge_train: np.ndarray,
    edge_val: np.ndarray,
    edge_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    route_hints: dict[str, Any] | None,
    residual_hints: dict[str, Any] | None,
    args: argparse.Namespace,
    device: torch.device,
    use_context: bool = True,
    use_axis: bool = True,
    use_edge: bool = True,
    shuffle_context: bool = False,
    shuffle_axis: bool = False,
    shuffle_edge: bool = False,
    use_route_hints: bool = True,
    shuffle_route_hints: bool = False,
    use_residual_hints: bool = True,
    shuffle_residual_hints: bool = False,
    shuffled_labels: bool = False,
    meta_out: dict[str, Any] | None = None,
) -> None:
    pack, meta = build_feature_pack(
        cand_train=cand_train,
        cand_val=cand_val,
        cand_test=cand_test,
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        edge_train=edge_train,
        edge_val=edge_val,
        edge_test=edge_test,
        component_axes=component_axes,
        args=args,
        use_context=use_context,
        use_axis=use_axis,
        use_edge=use_edge,
        shuffle_context=shuffle_context,
        shuffle_axis=shuffle_axis,
        shuffle_edge=shuffle_edge,
        route_hints=route_hints,
        use_route_hints=use_route_hints,
        shuffle_route_hints=shuffle_route_hints,
        residual_hints=residual_hints,
        use_residual_hints=use_residual_hints,
        shuffle_residual_hints=shuffle_residual_hints,
    )
    if meta_out is not None:
        meta_out[name] = finite_json(meta)
    model, log, val_corr = train_v7(pack, args, device, shuffled_labels=shuffled_labels)
    logs.append(log.assign(variant=name))
    risk_val = predict_v7(model, pack["feat_val"], args, device)
    risk_test = predict_v7(model, pack["feat_test"], args, device)
    best_m, best_t, val_rmse = tune_topm(risk_val, cand_val.residual, arrays.residual_val, args)
    pred_sparse, top_idx = sparse_topm_prediction(risk_test, cand_test.residual, best_m, best_t)
    pred_top = cand_test.residual[np.arange(len(risk_test)), np.argmin(risk_test, axis=1)]
    pred_dense = qrc.weighted_residual(cand_test.residual, qrc.softmax_np(-risk_test / best_t, axis=1))
    corr = qrc.risk_error_corr(risk_test, pack["err_test"])
    add_metric_rows(rows, arrays, pred_sparse, args, f"{name}_sparse_topm", {"stage": "v7_sparse_topm", "variant": name, "top_m": best_m, "temperature": best_t, "risk_error_corr": corr, "val_selector_rmse": val_rmse, "val_risk_error_corr": val_corr})
    add_metric_rows(rows, arrays, pred_top, args, f"{name}_top1", {"stage": "v7_top1", "variant": name, "top_m": 1, "risk_error_corr": corr, "val_risk_error_corr": val_corr})
    add_metric_rows(rows, arrays, pred_dense, args, f"{name}_dense", {"stage": "v7_dense", "variant": name, "temperature": best_t, "risk_error_corr": corr, "val_risk_error_corr": val_corr})
    take = np.argmin(risk_test, axis=1)
    diagnostics.append({"variant": name, "risk_error_corr": float(corr), "val_risk_error_corr": float(val_corr), "best_top_m": int(best_m), "best_temperature": float(best_t), **candidate_source_fraction(cand_test, take)})


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    extra_feature_meta = rpr.attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)

    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_scaler = rpr.prepare_context(args, arrays, posterior, student, blocks, device)
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v7 requires component axes")

    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand_train = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    edge_train, edge_val, edge_test, edge_meta = select_context_block(arrays, args.v7_edge_block, args.v7_edge_max_features)
    route_hints = None
    route_hint_probe = pd.DataFrame()
    if args.use_video_route_hints:
        route_hints, route_hint_probe = fit_video_route_hints(
            arrays=arrays,
            edge_train=edge_train,
            edge_val=edge_val,
            edge_test=edge_test,
            cand_train=cand_train,
            cand_val=cand_val,
            cand_test=cand_test,
            args=args,
        )
    residual_hints = None
    residual_hint_probe = pd.DataFrame()
    if args.use_video_residual_hints:
        residual_hints, residual_hint_probe = fit_video_residual_hints(
            arrays=arrays,
            ctx_train=ctx_train,
            ctx_val=ctx_val,
            ctx_test=ctx_test,
            edge_train=edge_train,
            edge_val=edge_val,
            edge_test=edge_test,
            args=args,
        )

    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "context": finite_json(ctx_scaler),
        "extra_feature": finite_json(extra_feature_meta),
        "edge": finite_json(edge_meta),
        "video_residual_hints": finite_json(residual_hints["info"] if residual_hints is not None else None),
    }

    add_metric_rows(rows, arrays, seq.mean_candidate_residual(cand_test), args, "candidate_mean", {"stage": "candidate_control"})
    for k in args.oracle_k:
        add_metric_rows(rows, arrays, seq.oracle_residual(cand_test, arrays.residual_test, int(k)), args, f"candidate_oracle@{k}", {"stage": "candidate_oracle", "oracle_k": int(k)})

    run_variant(
        name="v7_full",
        rows=rows,
        logs=logs,
        diagnostics=diagnostics,
        cand_train=cand_train,
        cand_val=cand_val,
        cand_test=cand_test,
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        edge_train=edge_train,
        edge_val=edge_val,
        edge_test=edge_test,
        component_axes=component_axes,
        route_hints=route_hints,
        residual_hints=residual_hints,
        args=args,
        device=device,
        meta_out=meta,
    )
    if args.include_controls:
        for name, kwargs in [
            ("v7_no_context", {"use_context": False}),
        ("v7_no_axis", {"use_axis": False}),
        ("v7_no_edge", {"use_edge": False}),
        ("v7_shuffled_context", {"shuffle_context": True}),
        ("v7_shuffled_axis", {"shuffle_axis": True}),
        ("v7_shuffled_edge", {"shuffle_edge": True}),
        ("v7_shuffled_labels", {"shuffled_labels": True}),
    ]:
            run_variant(
                name=name,
                rows=rows,
                logs=logs,
                diagnostics=diagnostics,
                cand_train=cand_train,
                cand_val=cand_val,
                cand_test=cand_test,
                arrays=arrays,
                ctx_train=ctx_train,
                ctx_val=ctx_val,
                ctx_test=ctx_test,
                edge_train=edge_train,
                edge_val=edge_val,
                edge_test=edge_test,
                component_axes=component_axes,
                route_hints=route_hints,
                residual_hints=residual_hints,
                args=args,
                device=device,
                meta_out=meta,
                **kwargs,
            )
        if route_hints is not None:
            for name, kwargs in [
                ("v7_no_video_hints", {"use_route_hints": False}),
                ("v7_shuffled_video_hints", {"shuffle_route_hints": True}),
            ]:
                run_variant(
                    name=name,
                    rows=rows,
                    logs=logs,
                    diagnostics=diagnostics,
                    cand_train=cand_train,
                    cand_val=cand_val,
                    cand_test=cand_test,
                    arrays=arrays,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    edge_train=edge_train,
                    edge_val=edge_val,
                    edge_test=edge_test,
                    component_axes=component_axes,
                    route_hints=route_hints,
                    residual_hints=residual_hints,
                    args=args,
                    device=device,
                    meta_out=meta,
                    **kwargs,
                )
        if residual_hints is not None:
            for name, kwargs in [
                ("v7_no_video_residual_hints", {"use_residual_hints": False}),
                ("v7_shuffled_video_residual_hints", {"shuffle_residual_hints": True}),
            ]:
                run_variant(
                    name=name,
                    rows=rows,
                    logs=logs,
                    diagnostics=diagnostics,
                    cand_train=cand_train,
                    cand_val=cand_val,
                    cand_test=cand_test,
                    arrays=arrays,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    edge_train=edge_train,
                    edge_val=edge_val,
                    edge_test=edge_test,
                    component_axes=component_axes,
                    route_hints=route_hints,
                    residual_hints=residual_hints,
                    args=args,
                    device=device,
                    meta_out=meta,
                    **kwargs,
                )

    summary = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    if not summary.empty:
        summary.insert(0, "seed", int(args.seed))
        summary.insert(0, "dataset", str(args.dataset))
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "sequence_joint_selector_refiner_v7_summary.csv", index=False)
    diag.to_csv(args.out_dir / "sequence_joint_selector_refiner_v7_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "sequence_joint_selector_refiner_v7_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    if not route_hint_probe.empty:
        route_hint_probe.to_csv(args.out_dir / "video_route_hint_probe.csv", index=False)
    if not residual_hint_probe.empty:
        residual_hint_probe.to_csv(args.out_dir / "video_residual_hint_probe.csv", index=False)
    component_axes.probe.to_csv(args.out_dir / "sequence_joint_selector_refiner_v7_component_axis_probe.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = ["# Sequence/Joint Selector-Refiner v7 Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- v7_edge_block: `{args.v7_edge_block}`")
    lines.append(f"- extra_feature_grid: `{args.extra_feature_grid}`")
    lines.append(f"- use_video_route_hints: `{args.use_video_route_hints}`")
    lines.append(f"- use_video_residual_hints: `{args.use_video_residual_hints}`")
    lines.append(f"- video_residual_model: `{args.video_residual_model}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{int(h)}")
        sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
        for _, row in sub.head(36).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, stage={row.get('stage', '')}")
    if not diag.empty:
        lines.append("\n## Selector Diagnostics")
        for _, row in diag.sort_values("risk_error_corr", ascending=False).iterrows():
            lines.append(
                f"- `{row['variant']}`: corr={row['risk_error_corr']:.3f}, "
                f"topM={row['best_top_m']}, T={row['best_temperature']:.3f}, "
                f"source learned={row.get('source_learned_frac', np.nan):.3f}"
            )
    hint_path = out_dir / "video_route_hint_probe.csv"
    if hint_path.exists():
        lines.append("\n## Video Route Hint Probe")
        probe = pd.read_csv(hint_path)
        lines.append(probe.to_markdown(index=False))
    residual_hint_path = out_dir / "video_residual_hint_probe.csv"
    if residual_hint_path.exists():
        lines.append("\n## Video Residual Hint Probe")
        probe = pd.read_csv(residual_hint_path)
        lines.append(probe.to_markdown(index=False))
    lines.append("\n## Decision")
    lines.append("- Pass only if v7_full beats no_axis/no_edge/no_context and shuffled controls.")
    lines.append("- If v7_full remains near candidate_mean, the blocker is missing observability or the need for a stronger video/query-centric architecture.")
    (out_dir / "sequence_joint_selector_refiner_v7_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="ef_")
    parser.add_argument("--extra-feature-block-name", type=str, default="explicit_edge")
    parser.add_argument("--extra-feature-max-cols", type=int, default=128)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--component-axis-blocks", type=str, default="self,flow,explicit_edge,all_context")
    parser.add_argument("--component-include-student-axis", action="store_true")
    parser.add_argument("--component-axis-model", type=str, default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--component-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--component-axis-max-features", type=int, default=256)
    parser.add_argument("--component-axis-hidden", type=int, default=128)
    parser.add_argument("--component-axis-epochs", type=int, default=16)
    parser.add_argument("--component-axis-lr", type=float, default=8e-4)
    parser.add_argument("--component-axis-weight-decay", type=float, default=1e-4)
    parser.add_argument("--component-axis-dropout", type=float, default=0.05)
    parser.add_argument("--component-attention-temperature", type=float, default=6.0)
    parser.add_argument("--v7-edge-block", type=str, default="explicit_edge")
    parser.add_argument("--v7-edge-max-features", type=int, default=128)
    parser.add_argument("--v7-hidden", type=int, default=192)
    parser.add_argument("--v7-heads", type=int, default=4)
    parser.add_argument("--v7-layers", type=int, default=2)
    parser.add_argument("--v7-dropout", type=float, default=0.05)
    parser.add_argument("--v7-epochs", type=int, default=16)
    parser.add_argument("--v7-batch-size", type=int, default=384)
    parser.add_argument("--v7-lr", type=float, default=7e-4)
    parser.add_argument("--v7-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v7-label-temperature", type=float, default=8.0)
    parser.add_argument("--v7-listwise-weight", type=float, default=1.0)
    parser.add_argument("--v7-rank-weight", type=float, default=0.25)
    parser.add_argument("--v7-entropy-weight", type=float, default=0.002)
    parser.add_argument("--v7-topm", type=str, default="1,2,4,8,16")
    parser.add_argument("--v7-temperatures", type=str, default="0.10,0.15,0.25,0.35,0.5,0.75,1.0,1.5")
    parser.add_argument("--use-video-route-hints", action="store_true")
    parser.add_argument("--video-hint-k", type=int, default=8)
    parser.add_argument("--video-hint-max-iter", type=int, default=500)
    parser.add_argument("--video-hint-c", type=float, default=1.0)
    parser.add_argument("--use-video-residual-hints", action="store_true")
    parser.add_argument("--video-residual-model", type=str, default="ridge", choices=["ridge", "hgbdt"])
    parser.add_argument("--video-residual-include-context", action="store_true")
    parser.add_argument("--video-residual-hgbdt-iter", type=int, default=120)
    parser.add_argument("--video-residual-hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--video-residual-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--video-residual-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    args.v7_temperatures = [float(x) for x in str(args.v7_temperatures).split(",") if x.strip()]
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 2000)
        args.max_val_rows = min(args.max_val_rows, 700)
        args.max_test_rows = min(args.max_test_rows, 900)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.learned_route_epochs = min(args.learned_route_epochs, 4)
        args.v7_epochs = min(args.v7_epochs, 4)
        args.candidate_k = min(args.candidate_k, 32)
        args.oracle_k = [8, min(16, args.candidate_k), args.candidate_k]
        args.max_all_features = min(args.max_all_features, 192)
        args.max_critic_context_features = min(args.max_critic_context_features, 192)
    run(args)


if __name__ == "__main__":
    main()
