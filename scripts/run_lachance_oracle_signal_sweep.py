#!/usr/bin/env python3
"""Broad oracle-signal sweep for LaChance candidate clouds.

This is intentionally less strict than the critic-v2 gate.  It is a fast
hypothesis miner: build the expensive candidate split once, then try many cheap
scorers/aggregators/decoders to find something that can close the oracle gap.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.multioutput import MultiOutputRegressor

from scripts import run_lachance_candidate_oracle as co  # noqa: E402
from scripts import run_lachance_transition_critic_v2 as v2  # noqa: E402


EPS = 1e-8
DEFAULT_FEATURE_SETS = [
    "full",
    "baseline_config_backward",
    "no_physics",
    "dynamic_only",
    "no_backward",
    "no_soft_neighbour",
    "oz_only",
]
DEFAULT_METHODS = [
    "family_mean_error",
    "candidate_name_mean_error",
    "ridge_error",
    "ridge_error_soft_blend",
    "ridge_error_gate_linear",
    "ridge_error_gate_hgb",
    "sgd_top3",
    "hgb_error",
    "extra_trees_error",
    "ridge_vector_mean",
    "ridge_residual_mean",
    "cloud_ridge_vector",
    "cloud_hgb_vector",
    "cloud_extra_trees_vector",
    "torch_mlp_error",
    "torch_set_error",
    "torch_gain_weighted_soft",
    "torch_gain_weighted_soft_blend",
    "torch_radius_quality_bce",
    "torch_radius_quality_bce_blend",
    "cloud_mlp_residual",
]


def stable_seed(*parts: object, base: int = 0) -> int:
    text = "::".join(map(str, parts))
    value = int(base)
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def sanitize_features(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.clip(x, -8.0, 8.0),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


@dataclass
class SplitBundle:
    train_split: co.CandidateSplit
    val_split: co.CandidateSplit
    test_split: co.CandidateSplit
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    scale_px: float
    feature_names: list[str]
    feature_groups: list[str]
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_candidates: np.ndarray
    val_candidates: np.ndarray
    test_candidates: np.ndarray
    train_target: np.ndarray
    val_target: np.ndarray
    test_target: np.ndarray
    info: dict[str, Any]


def parse_csv(text: str | None, default: list[str]) -> list[str]:
    if text is None or str(text).strip() == "":
        return list(default)
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_int_csv(text: str | None, default: list[int]) -> list[int]:
    return [int(x) for x in parse_csv(text, [str(v) for v in default])]


def vector_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    rmse = v2.vector_rmse_arrays(pred, y)
    return {
        "rmse": rmse,
        "r2": co.vector_r2_from_arrays(y, pred),
        "angular_cosine": v2.mean_cosine_arrays(pred, y),
        "magnitude_ratio": v2.magnitude_ratio_arrays(pred, y),
    }


def candidate_arrays(split: co.CandidateSplit, node_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return v2.candidate_arrays(split, node_idx)


def oracle_prediction(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    idx = np.argmin(np.linalg.norm(candidates - target[:, None, :], axis=-1), axis=1)
    return candidates[np.arange(len(idx)), idx]


def sample_flat_rows(n_nodes: int, k: int, max_rows: int, seed: int) -> np.ndarray:
    total = int(n_nodes * k)
    if max_rows <= 0 or total <= max_rows:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=int(max_rows), replace=False)).astype(np.int64)


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def reshape_scores(scores_flat: np.ndarray, node_count: int, candidate_count: int) -> np.ndarray:
    return np.asarray(scores_flat, dtype=np.float32).reshape(node_count, candidate_count)


def candidate_error(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(candidates - target[:, None, :], axis=-1).astype(np.float32)


def topk_positive_labels(candidates: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    err = candidate_error(candidates, target)
    k = max(1, min(int(k), err.shape[1]))
    kth = np.partition(err, kth=k - 1, axis=1)[:, k - 1]
    return (err <= kth[:, None] + 1e-6).astype(np.int64)


def softmax_weights(scores: np.ndarray, temp: float) -> np.ndarray:
    shifted = (scores - scores.max(axis=1, keepdims=True)) / max(float(temp), 1e-4)
    w = np.exp(np.clip(shifted, -40.0, 20.0))
    return w / np.maximum(w.sum(axis=1, keepdims=True), EPS)


def pred_from_scores(candidates: np.ndarray, scores: np.ndarray, mode: str, temp: float = 1.0) -> np.ndarray:
    n, k, _ = candidates.shape
    if mode == "top1":
        idx = np.argmax(scores, axis=1)
        return candidates[np.arange(n), idx]
    if mode.startswith("top") and mode.endswith("_mean"):
        top_k = int(mode.removeprefix("top").removesuffix("_mean"))
        top_k = min(max(top_k, 1), k)
        idx = np.argsort(-scores, axis=1)[:, :top_k]
        return candidates[np.arange(n)[:, None], idx].mean(axis=1)
    if mode.startswith("soft_t"):
        temp = float(mode.removeprefix("soft_t"))
    weights = softmax_weights(scores, temp)
    return np.sum(weights[:, :, None] * candidates, axis=1)


def tune_temp_on_val(candidates: np.ndarray, target: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_temp = 1.0
    best_rmse = float("inf")
    for temp in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0):
        rmse = vector_metrics(pred_from_scores(candidates, scores, "soft", temp=temp), target)["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_temp = float(temp)
    return best_temp, best_rmse


def scorer_rows(
    *,
    cell_type: str,
    horizon: int,
    seed: int,
    method: str,
    feature_set: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    bundle: SplitBundle,
    baseline_rmse: float,
    proposal_rmse: float,
    oracle_rmse: float,
) -> list[dict[str, Any]]:
    temp, val_rmse = tune_temp_on_val(bundle.val_candidates, bundle.val_target, val_scores)
    rows: list[dict[str, Any]] = []
    for agg in ["top1", "top3_mean", "top5_mean", "soft", "soft_t0.5", "soft_t1.5", "soft_t5.0"]:
        pred = pred_from_scores(bundle.test_candidates, test_scores, agg, temp=temp)
        metrics = vector_metrics(pred, bundle.test_target)
        err = candidate_error(bundle.test_candidates, bundle.test_target)
        top_order = np.argsort(-test_scores, axis=1)
        oracle_label = np.argmin(err, axis=1)
        top3_hit = np.mean([oracle_label[i] in set(top_order[i, : min(3, top_order.shape[1])]) for i in range(len(oracle_label))])
        rows.append(
            {
                "cell_type": cell_type,
                "horizon": horizon,
                "seed": seed,
                "method": method,
                "feature_set": feature_set,
                "aggregator": agg,
                **metrics,
                "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                "oracle_gap_closed_pct": (
                    (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                ),
                "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                "val_soft_rmse": val_rmse,
                "softmax_temp": temp,
                "score_error_corr": v2.score_error_corr(bundle.test_candidates, bundle.test_target, test_scores),
                "ndcg_at_5": v2.ndcg_at_k(bundle.test_candidates, bundle.test_target, test_scores, k=min(5, test_scores.shape[1])),
                "oracle_top1_hit": float(np.mean(np.argmax(test_scores, axis=1) == oracle_label)),
                "oracle_top3_hit": float(top3_hit),
            }
        )
    return rows


def train_family_mean_scores(bundle: SplitBundle, by: str) -> tuple[np.ndarray, np.ndarray]:
    err = candidate_error(bundle.train_candidates, bundle.train_target)
    if by == "family":
        keys = list(bundle.train_split.pack.families)
        val_keys = list(bundle.val_split.pack.families)
        test_keys = list(bundle.test_split.pack.families)
    else:
        keys = list(bundle.train_split.pack.names)
        val_keys = list(bundle.val_split.pack.names)
        test_keys = list(bundle.test_split.pack.names)
    means: dict[str, float] = {}
    global_mean = float(err.mean())
    for idx, key in enumerate(keys):
        means[str(key)] = float(err[:, idx].mean())

    def scores_for(keys_local: list[str], node_count: int) -> np.ndarray:
        vals = np.array([-means.get(str(key), global_mean) for key in keys_local], dtype=np.float32)
        return np.repeat(vals[None, :], node_count, axis=0)

    return scores_for(val_keys, len(bundle.val_idx)), scores_for(test_keys, len(bundle.test_idx))


def train_flat_error_model(
    method: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, k, _ = train_x.shape
    rows = sample_flat_rows(n, k, max_rows, seed)
    x_train = flatten(train_x)[rows]
    if method in {"ridge_error", "hgb_error", "extra_trees_error"}:
        y_train = candidate_error(bundle.train_candidates, bundle.train_target).reshape(-1)[rows]
        if method == "ridge_error":
            model = Ridge(alpha=3.0)
        elif method == "hgb_error":
            model = HistGradientBoostingRegressor(
                max_iter=90,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=seed,
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=80,
                max_depth=16,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            )
        model.fit(x_train, y_train)
        val_pred = model.predict(flatten(val_x))
        test_pred = model.predict(flatten(test_x))
        return (
            reshape_scores(-val_pred, val_x.shape[0], val_x.shape[1]),
            reshape_scores(-test_pred, test_x.shape[0], test_x.shape[1]),
        )

    if method == "sgd_top3":
        y_train = topk_positive_labels(bundle.train_candidates, bundle.train_target, k=3).reshape(-1)[rows]
        clf = SGDClassifier(
            loss="log_loss",
            alpha=2e-4,
            penalty="elasticnet",
            l1_ratio=0.10,
            max_iter=1500,
            class_weight="balanced",
            random_state=seed,
            tol=1e-4,
        )
        clf.fit(x_train, y_train)
        val_score = clf.decision_function(flatten(val_x))
        test_score = clf.decision_function(flatten(test_x))
        return (
            reshape_scores(val_score, val_x.shape[0], val_x.shape[1]),
            reshape_scores(test_score, test_x.shape[0], test_x.shape[1]),
        )
    raise ValueError(method)


def train_ridge_error_scores(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k, _ = train_x.shape
    rows = sample_flat_rows(n, k, max_rows, seed)
    x_train = flatten(train_x)[rows]
    y_train = candidate_error(bundle.train_candidates, bundle.train_target).reshape(-1)[rows]
    model = Ridge(alpha=3.0)
    model.fit(x_train, y_train)
    train_pred = model.predict(flatten(train_x))
    val_pred = model.predict(flatten(val_x))
    test_pred = model.predict(flatten(test_x))
    return (
        reshape_scores(-train_pred, train_x.shape[0], train_x.shape[1]),
        reshape_scores(-val_pred, val_x.shape[0], val_x.shape[1]),
        reshape_scores(-test_pred, test_x.shape[0], test_x.shape[1]),
    )


def tune_alpha_on_val(proposal: np.ndarray, alt_pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    best_alpha = 1.0
    best_rmse = float("inf")
    for alpha in np.linspace(0.0, 1.5, 31):
        pred = proposal + float(alpha) * (alt_pred - proposal)
        rmse = vector_metrics(pred, target)["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha, best_rmse


def tune_temp_alpha_on_val(
    proposal: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float, float]:
    best_temp = 1.0
    best_alpha = 1.0
    best_rmse = float("inf")
    for temp in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        alt_pred = pred_from_scores(candidates, scores, "soft", temp=temp)
        for alpha in np.linspace(0.0, 1.6, 33):
            pred = proposal + float(alpha) * (alt_pred - proposal)
            rmse = vector_metrics(pred, target)["rmse"]
            if rmse < best_rmse:
                best_rmse = rmse
                best_temp = float(temp)
                best_alpha = float(alpha)
    return best_temp, best_alpha, best_rmse


def optimal_gate_target(proposal: np.ndarray, alt_pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    direction = alt_pred - proposal
    denom = np.sum(direction * direction, axis=1)
    numer = np.sum((target - proposal) * direction, axis=1)
    gate = np.zeros(len(proposal), dtype=np.float32)
    valid = denom > 1e-6
    gate[valid] = np.clip(numer[valid] / denom[valid], 0.0, 1.0)
    return gate


def confidence_gate_features(
    candidates: np.ndarray,
    proposal: np.ndarray,
    alt_pred: np.ndarray,
    scores: np.ndarray,
    temp: float,
) -> np.ndarray:
    weights = softmax_weights(scores, temp)
    entropy = -np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1, keepdims=True)
    entropy = entropy / max(np.log(scores.shape[1]), EPS)
    sorted_scores = np.sort(scores, axis=1)
    top_margin = (sorted_scores[:, -1] - sorted_scores[:, -2])[:, None] if scores.shape[1] > 1 else np.zeros_like(entropy)
    score_std = scores.std(axis=1, keepdims=True)
    score_range = (scores.max(axis=1) - scores.min(axis=1))[:, None]
    best_score = scores.max(axis=1, keepdims=True)
    mean_score = scores.mean(axis=1, keepdims=True)

    rel = candidates - proposal[:, None, :]
    cloud_center = candidates.mean(axis=1)
    cloud_disp = np.linalg.norm(candidates - cloud_center[:, None, :], axis=-1)
    rel_norm = np.linalg.norm(rel, axis=-1)
    step = alt_pred - proposal
    pieces = [
        entropy,
        top_margin,
        score_std,
        score_range,
        best_score,
        mean_score,
        np.linalg.norm(proposal, axis=1, keepdims=True),
        np.linalg.norm(alt_pred, axis=1, keepdims=True),
        np.linalg.norm(step, axis=1, keepdims=True),
        rel_norm.mean(axis=1, keepdims=True),
        rel_norm.std(axis=1, keepdims=True),
        rel_norm.min(axis=1, keepdims=True),
        rel_norm.max(axis=1, keepdims=True),
        cloud_disp.mean(axis=1, keepdims=True),
        cloud_disp.std(axis=1, keepdims=True),
        weights.max(axis=1, keepdims=True),
        weights.std(axis=1, keepdims=True),
    ]
    return np.nan_to_num(np.concatenate(pieces, axis=1), nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)


def train_ridge_error_confidence_blend(
    method: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    train_scores, val_scores, test_scores = train_ridge_error_scores(
        train_x,
        val_x,
        test_x,
        bundle,
        max_rows=max_rows,
        seed=seed,
    )
    temp, val_soft_rmse = tune_temp_on_val(bundle.val_candidates, bundle.val_target, val_scores)
    train_alt = pred_from_scores(bundle.train_candidates, train_scores, "soft", temp=temp)
    val_alt = pred_from_scores(bundle.val_candidates, val_scores, "soft", temp=temp)
    test_alt = pred_from_scores(bundle.test_candidates, test_scores, "soft", temp=temp)
    train_proposal = bundle.train_split.proposal_px[bundle.train_idx].astype(np.float32)
    val_proposal = bundle.val_split.proposal_px[bundle.val_idx].astype(np.float32)
    test_proposal = bundle.test_split.proposal_px[bundle.test_idx].astype(np.float32)

    if method == "ridge_error_soft_blend":
        alpha, val_blend_rmse = tune_alpha_on_val(val_proposal, val_alt, bundle.val_target)
        pred = test_proposal + alpha * (test_alt - test_proposal)
        return pred.astype(np.float32), {
            "softmax_temp": temp,
            "val_soft_rmse": val_soft_rmse,
            "blend_alpha": alpha,
            "val_blend_rmse": val_blend_rmse,
            "gate_mean": float(alpha),
        }

    y_gate = optimal_gate_target(train_proposal, train_alt, bundle.train_target)
    x_train = confidence_gate_features(bundle.train_candidates, train_proposal, train_alt, train_scores, temp)
    x_val = confidence_gate_features(bundle.val_candidates, val_proposal, val_alt, val_scores, temp)
    x_test = confidence_gate_features(bundle.test_candidates, test_proposal, test_alt, test_scores, temp)
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-5)
    x_train = sanitize_features((x_train - mean) / std)
    x_val = sanitize_features((x_val - mean) / std)
    x_test = sanitize_features((x_test - mean) / std)
    if method == "ridge_error_gate_linear":
        gate_model: Any = Ridge(alpha=2.0)
    elif method == "ridge_error_gate_hgb":
        gate_model = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=seed,
        )
    else:
        raise ValueError(method)
    gate_model.fit(x_train, y_gate)
    val_gate = np.clip(gate_model.predict(x_val), 0.0, 1.0).astype(np.float32)
    test_gate = np.clip(gate_model.predict(x_test), 0.0, 1.0).astype(np.float32)

    best_scale = 1.0
    best_val = float("inf")
    for scale in np.linspace(0.0, 1.5, 31):
        gate = np.clip(float(scale) * val_gate, 0.0, 1.0)
        pred = val_proposal + gate[:, None] * (val_alt - val_proposal)
        rmse = vector_metrics(pred, bundle.val_target)["rmse"]
        if rmse < best_val:
            best_val = rmse
            best_scale = float(scale)
    final_gate = np.clip(best_scale * test_gate, 0.0, 1.0)
    pred = test_proposal + final_gate[:, None] * (test_alt - test_proposal)
    return pred.astype(np.float32), {
        "softmax_temp": temp,
        "val_soft_rmse": val_soft_rmse,
        "blend_alpha": best_scale,
        "val_blend_rmse": best_val,
        "gate_mean": float(final_gate.mean()),
    }


def train_candidate_vector_decoder(
    method: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    n, k, _ = train_x.shape
    rows = sample_flat_rows(n, k, max_rows, seed)
    x_train = flatten(train_x)[rows]
    if method == "ridge_vector_mean":
        y_train = np.repeat(bundle.train_target[:, None, :], k, axis=1).reshape(-1, 2)[rows]
        model = Ridge(alpha=10.0)
        model.fit(x_train, y_train)
        pred = model.predict(flatten(test_x)).reshape(test_x.shape[0], test_x.shape[1], 2)
        return np.mean(pred, axis=1)
    if method == "ridge_residual_mean":
        residual = bundle.train_target[:, None, :] - bundle.train_candidates
        y_train = residual.reshape(-1, 2)[rows]
        model = Ridge(alpha=10.0)
        model.fit(x_train, y_train)
        residual_pred = model.predict(flatten(test_x)).reshape(test_x.shape[0], test_x.shape[1], 2)
        pred = bundle.test_candidates + residual_pred
        return np.mean(pred, axis=1)
    raise ValueError(method)


def cloud_features(x: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    stats = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        candidates.mean(axis=1),
        candidates.std(axis=1),
        np.linalg.norm(candidates, axis=-1).min(axis=1)[:, None],
        np.linalg.norm(candidates, axis=-1).mean(axis=1)[:, None],
        np.linalg.norm(candidates, axis=-1).max(axis=1)[:, None],
    ]
    return np.concatenate(stats, axis=1).astype(np.float32)


def train_cloud_decoder(
    method: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    seed: int,
) -> np.ndarray:
    x_train = cloud_features(train_x, bundle.train_candidates)
    x_test = cloud_features(test_x, bundle.test_candidates)
    if method == "cloud_ridge_vector":
        model = Ridge(alpha=20.0)
    elif method == "cloud_hgb_vector":
        base = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed,
        )
        model = MultiOutputRegressor(base)
    elif method == "cloud_extra_trees_vector":
        model = ExtraTreesRegressor(
            n_estimators=96,
            max_depth=14,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(method)
    model.fit(x_train, bundle.train_target)
    return model.predict(x_test).astype(np.float32)


def train_torch_error_scorer(
    method: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    device: torch.device,
    *,
    epochs: int,
    batch_nodes: int,
    hidden_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Neural scorer trained to predict candidate error and listwise order."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feature_dim = train_x.shape[-1]
    if method == "torch_set_error":
        model = co.CandidateSetReranker(
            feature_dim,
            hidden_dim=hidden_dim,
            layers=2,
            heads=4,
        ).to(device)
    elif method == "torch_mlp_error":
        model = co.CandidateReranker(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(method)

    train_err = candidate_error(bundle.train_candidates, bundle.train_target) / float(bundle.scale_px)
    val_err = candidate_error(bundle.val_candidates, bundle.val_target) / float(bundle.scale_px)
    train_score_target = -train_err.astype(np.float32)
    val_score_target = -val_err.astype(np.float32)
    train_best = np.argmin(train_err, axis=1).astype(np.int64)
    val_best = np.argmin(val_err, axis=1).astype(np.int64)
    train_soft = co.candidate_soft_oracle_targets(
        bundle.train_candidates / float(bundle.scale_px),
        bundle.train_target / float(bundle.scale_px),
        temperature=0.10,
        topk=min(6, bundle.train_candidates.shape[1]),
    )[0]
    val_soft = co.candidate_soft_oracle_targets(
        bundle.val_candidates / float(bundle.scale_px),
        bundle.val_target / float(bundle.scale_px),
        temperature=0.10,
        topk=min(6, bundle.val_candidates.shape[1]),
    )[0]

    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    vx = torch.tensor(val_x, dtype=torch.float32, device=device)
    target_score = torch.tensor(train_score_target, dtype=torch.float32, device=device)
    v_target_score = torch.tensor(val_score_target, dtype=torch.float32, device=device)
    best = torch.tensor(train_best, dtype=torch.long, device=device)
    v_best = torch.tensor(val_best, dtype=torch.long, device=device)
    soft = torch.tensor(train_soft, dtype=torch.float32, device=device)
    v_soft = torch.tensor(val_soft, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience = 8

    def loss_fn(logits: torch.Tensor, score_target: torch.Tensor, labels: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        reg = torch.nn.functional.smooth_l1_loss(logits, score_target)
        ce = torch.mean(torch.sum(-soft_target * torch.nn.functional.log_softmax(logits, dim=1), dim=1))
        pos = logits.gather(1, labels[:, None]).squeeze(1)
        neg = logits.masked_fill(torch.nn.functional.one_hot(labels, logits.shape[1]).bool(), -1e9).max(dim=1).values
        rank = torch.relu(0.15 - pos + neg).mean()
        return reg + 0.25 * ce + 0.08 * rank

    batch_nodes = max(int(batch_nodes), 128)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_x.shape[0])
        for start in range(0, len(order), batch_nodes):
            idx = order[start : start + batch_nodes]
            logits = model(tx[idx])
            loss = loss_fn(logits, target_score[idx], best[idx], soft[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(vx)
            val_loss = float(loss_fn(val_logits, v_target_score, v_best, v_soft).detach().cpu())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = co.score_reranker(model, val_x, device=device, batch_nodes=2048)
    test_scores = co.score_reranker(model, test_x, device=device, batch_nodes=2048)
    return val_scores, test_scores


def train_torch_oracle_gain_scorer(
    method: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    device: torch.device,
    *,
    epochs: int,
    batch_nodes: int,
    hidden_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Train a candidate scorer from target-derived labels, gated by oracle gain.

    The true target is used only to build train/validation supervision.  The
    model input remains candidate/context features, matching inference.
    """

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feature_dim = train_x.shape[-1]
    model = co.CandidateReranker(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    scale = float(bundle.scale_px)

    train_cand_np = (bundle.train_candidates / scale).astype(np.float32)
    val_cand_np = (bundle.val_candidates / scale).astype(np.float32)
    train_target_np = (bundle.train_target / scale).astype(np.float32)
    val_target_np = (bundle.val_target / scale).astype(np.float32)
    train_prop_np = (bundle.train_split.proposal_px[bundle.train_idx] / scale).astype(np.float32)
    val_prop_np = (bundle.val_split.proposal_px[bundle.val_idx] / scale).astype(np.float32)

    train_soft_np = co.candidate_soft_oracle_targets(
        train_cand_np,
        train_target_np,
        temperature=0.10,
        topk=min(6, train_cand_np.shape[1]),
    )[0]
    val_soft_np = co.candidate_soft_oracle_targets(
        val_cand_np,
        val_target_np,
        temperature=0.10,
        topk=min(6, val_cand_np.shape[1]),
    )[0]

    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    vx = torch.tensor(val_x, dtype=torch.float32, device=device)
    tc = torch.tensor(train_cand_np, dtype=torch.float32, device=device)
    vc = torch.tensor(val_cand_np, dtype=torch.float32, device=device)
    tt = torch.tensor(train_target_np, dtype=torch.float32, device=device)
    vt = torch.tensor(val_target_np, dtype=torch.float32, device=device)
    tp = torch.tensor(train_prop_np, dtype=torch.float32, device=device)
    vp = torch.tensor(val_prop_np, dtype=torch.float32, device=device)
    tq = torch.tensor(train_soft_np, dtype=torch.float32, device=device)
    vq = torch.tensor(val_soft_np, dtype=torch.float32, device=device)

    def oracle_weight(cand: torch.Tensor, target: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
        d = torch.linalg.norm(cand - target[:, None, :], dim=-1)
        err_oracle = d.min(dim=1).values
        err_base = torch.linalg.norm(proposal - target, dim=-1)
        return ((err_base - err_oracle) / err_base.clamp_min(1e-6)).clamp(0.0, 1.0).detach()

    train_weight = oracle_weight(tc, tt, tp)
    val_weight = oracle_weight(vc, vt, vp)
    opt = torch.optim.AdamW(model.parameters(), lr=1.3e-3, weight_decay=2e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience = 8

    def loss_fn(
        logits: torch.Tensor,
        cand: torch.Tensor,
        target: torch.Tensor,
        proposal: torch.Tensor,
        soft_target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.softmax(logits, dim=1)
        pred_cloud = torch.sum(weights[:, :, None] * cand, dim=1)
        point = torch.nn.functional.smooth_l1_loss(pred_cloud, target, reduction="none").sum(dim=1)
        if method in {"torch_gain_weighted_soft", "torch_gain_weighted_soft_blend"}:
            rank = torch.sum(
                -soft_target.detach() * torch.nn.functional.log_softmax(logits, dim=1),
                dim=1,
            )
        elif method in {"torch_radius_quality_bce", "torch_radius_quality_bce_blend"}:
            d = torch.linalg.norm(cand - target[:, None, :], dim=-1)
            d_min = d.min(dim=1, keepdim=True).values
            pos = d <= d_min + 0.05
            neg = d >= d_min + 0.20
            valid = (pos | neg).float()
            labels = pos.float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            rank = (bce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(method)
        # Strongly train selector only where the cloud has an actual oracle gain.
        # Keep a small point loss so logits do not become unconstrained elsewhere.
        return torch.mean(weight * (rank + 0.35 * point) + 0.05 * point + 1e-4 * logits.square().mean(dim=1))

    batch_nodes = max(int(batch_nodes), 128)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_x.shape[0])
        for start in range(0, len(order), batch_nodes):
            idx = order[start : start + batch_nodes]
            logits = model(tx[idx])
            loss = loss_fn(logits, tc[idx], tt[idx], tp[idx], tq[idx], train_weight[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(vx)
            val_loss = float(loss_fn(val_logits, vc, vt, vp, vq, val_weight).detach().cpu())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = co.score_reranker(model, val_x, device=device, batch_nodes=2048)
    test_scores = co.score_reranker(model, test_x, device=device, batch_nodes=2048)
    extra = {
        "oracle_weight_train_mean": float(train_weight.detach().cpu().mean()),
        "oracle_weight_val_mean": float(val_weight.detach().cpu().mean()),
        "quality_best_epoch": float(best_epoch),
        "quality_val_loss": float(best_val),
    }
    return val_scores, test_scores, extra


def train_torch_oracle_gain_blend(
    method: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    device: torch.device,
    *,
    epochs: int,
    batch_nodes: int,
    hidden_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    score_method = method.removesuffix("_blend")
    val_scores, test_scores, extra = train_torch_oracle_gain_scorer(
        score_method,
        train_x,
        val_x,
        test_x,
        bundle,
        device,
        epochs=epochs,
        batch_nodes=batch_nodes,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    val_proposal = bundle.val_split.proposal_px[bundle.val_idx].astype(np.float32)
    test_proposal = bundle.test_split.proposal_px[bundle.test_idx].astype(np.float32)
    temp, alpha, val_rmse = tune_temp_alpha_on_val(
        val_proposal,
        bundle.val_candidates,
        bundle.val_target,
        val_scores,
    )
    cloud_pred = pred_from_scores(bundle.test_candidates, test_scores, "soft", temp=temp)
    pred = test_proposal + alpha * (cloud_pred - test_proposal)
    extra.update(
        {
            "softmax_temp": float(temp),
            "blend_alpha": float(alpha),
            "val_blend_rmse": float(val_rmse),
        }
    )
    return pred.astype(np.float32), extra


class CloudResidualMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.08),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CloudBasisResidualHead(torch.nn.Module):
    def __init__(
        self,
        cand_feat_dim: int,
        ctx_dim: int,
        hidden_dim: int,
        tau: float = 0.75,
    ) -> None:
        super().__init__()
        in_dim = 2 + int(cand_feat_dim) + int(ctx_dim)
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.06),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.score = torch.nn.Linear(hidden_dim, 1)
        pooled_dim = hidden_dim * 3 + int(ctx_dim)
        self.delta = torch.nn.Sequential(
            torch.nn.Linear(pooled_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(pooled_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        # Conservative initialization: start near proposal and only use cloud
        # if validation proves the candidate basis is helpful.
        torch.nn.init.zeros_(self.delta[-1].weight)
        torch.nn.init.zeros_(self.delta[-1].bias)
        torch.nn.init.constant_(self.gate[-1].bias, -2.0)
        self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(max(tau, 1e-3)))))

    def forward(
        self,
        proposal: torch.Tensor,
        candidates: torch.Tensor,
        ctx: torch.Tensor,
        cand_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, cand_count, _ = candidates.shape
        v = candidates - proposal[:, None, :]
        ctx_k = ctx[:, None, :].expand(batch, cand_count, ctx.shape[-1])
        x = torch.cat([v, cand_feat, ctx_k], dim=-1)
        e = self.phi(x)
        tau = torch.exp(self.log_tau).clamp(0.15, 5.0)
        logits = self.score(e).squeeze(-1) / tau
        alpha = torch.softmax(logits, dim=1)
        basis = torch.sum(alpha[:, :, None] * v, dim=1)
        mean_pool = e.mean(dim=1)
        max_pool = e.max(dim=1).values
        std_pool = e.std(dim=1, unbiased=False)
        z = torch.cat([mean_pool, max_pool, std_pool, ctx], dim=-1)
        radii = torch.linalg.norm(v, dim=-1)
        kth = min(cand_count - 1, max(0, int(np.ceil(0.8 * cand_count)) - 1))
        radius = torch.sort(radii, dim=1).values[:, kth : kth + 1].clamp_min(1e-4)
        delta = radius * torch.tanh(self.delta(z))
        gate = torch.sigmoid(self.gate(z)).squeeze(-1)
        pred = proposal + gate[:, None] * (basis + delta)
        return pred, alpha, gate, delta, basis


def cloud_context_features(
    x: np.ndarray,
    candidates: np.ndarray,
    proposal: np.ndarray,
) -> np.ndarray:
    rel = candidates - proposal[:, None, :]
    mag = np.linalg.norm(candidates, axis=-1)
    rel_mag = np.linalg.norm(rel, axis=-1)
    q = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    pieces = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        candidates.mean(axis=1),
        candidates.std(axis=1),
        candidates.min(axis=1),
        candidates.max(axis=1),
        rel.mean(axis=1),
        rel.std(axis=1),
        np.quantile(mag, q, axis=1).T,
        np.quantile(rel_mag, q, axis=1).T,
        proposal,
        np.linalg.norm(proposal, axis=1, keepdims=True),
    ]
    return np.nan_to_num(np.concatenate(pieces, axis=1), nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)


def cloud_basis_context_features(
    x: np.ndarray,
    candidates: np.ndarray,
    proposal: np.ndarray,
    scale_px: float,
) -> np.ndarray:
    scale = float(max(scale_px, 1e-6))
    cand = candidates / scale
    prop = proposal / scale
    rel = cand - prop[:, None, :]
    rel_norm = np.linalg.norm(rel, axis=-1)
    cand_norm = np.linalg.norm(cand, axis=-1)
    pieces = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        rel.mean(axis=1),
        rel.std(axis=1),
        rel.min(axis=1),
        rel.max(axis=1),
        prop,
        np.linalg.norm(prop, axis=1, keepdims=True),
        rel_norm.mean(axis=1, keepdims=True),
        rel_norm.std(axis=1, keepdims=True),
        rel_norm.min(axis=1, keepdims=True),
        rel_norm.max(axis=1, keepdims=True),
        cand_norm.mean(axis=1, keepdims=True),
        cand_norm.std(axis=1, keepdims=True),
    ]
    return np.nan_to_num(np.concatenate(pieces, axis=1), nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)


def train_cloud_mlp_residual(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_proposal = bundle.train_split.proposal_px[bundle.train_idx].astype(np.float32)
    val_proposal = bundle.val_split.proposal_px[bundle.val_idx].astype(np.float32)
    test_proposal = bundle.test_split.proposal_px[bundle.test_idx].astype(np.float32)
    x_train = cloud_context_features(train_x, bundle.train_candidates, train_proposal)
    x_val = cloud_context_features(val_x, bundle.val_candidates, val_proposal)
    x_test = cloud_context_features(test_x, bundle.test_candidates, test_proposal)
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-5)
    x_train = sanitize_features((x_train - mean) / std)
    x_val = sanitize_features((x_val - mean) / std)
    x_test = sanitize_features((x_test - mean) / std)
    scale = float(bundle.scale_px)
    y_train = (bundle.train_target - train_proposal) / scale
    y_val = (bundle.val_target - val_proposal) / scale
    model = CloudResidualMLP(x_train.shape[1], hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    tx = torch.tensor(x_train, dtype=torch.float32, device=device)
    ty = torch.tensor(y_train, dtype=torch.float32, device=device)
    vx = torch.tensor(x_val, dtype=torch.float32, device=device)
    vy = torch.tensor(y_val, dtype=torch.float32, device=device)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    batch_size = max(int(batch_size), 128)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(x_train.shape[0])
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = model(tx[idx])
            loss = torch.nn.functional.smooth_l1_loss(pred, ty[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.smooth_l1_loss(model(vx), vy).detach().cpu())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        residual = model(torch.tensor(x_test, dtype=torch.float32, device=device)).detach().cpu().numpy() * scale
    return test_proposal + residual


def train_cloud_basis_residual(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    bundle: SplitBundle,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(bundle.scale_px)
    train_proposal = bundle.train_split.proposal_px[bundle.train_idx].astype(np.float32)
    val_proposal = bundle.val_split.proposal_px[bundle.val_idx].astype(np.float32)
    test_proposal = bundle.test_split.proposal_px[bundle.test_idx].astype(np.float32)

    train_ctx = cloud_basis_context_features(train_x, bundle.train_candidates, train_proposal, scale)
    val_ctx = cloud_basis_context_features(val_x, bundle.val_candidates, val_proposal, scale)
    test_ctx = cloud_basis_context_features(test_x, bundle.test_candidates, test_proposal, scale)
    ctx_mean = train_ctx.mean(axis=0, keepdims=True)
    ctx_std = np.maximum(train_ctx.std(axis=0, keepdims=True), 1e-5)
    train_ctx = sanitize_features((train_ctx - ctx_mean) / ctx_std)
    val_ctx = sanitize_features((val_ctx - ctx_mean) / ctx_std)
    test_ctx = sanitize_features((test_ctx - ctx_mean) / ctx_std)

    train_cand = (bundle.train_candidates / scale).astype(np.float32)
    val_cand = (bundle.val_candidates / scale).astype(np.float32)
    test_cand = (bundle.test_candidates / scale).astype(np.float32)
    train_prop = (train_proposal / scale).astype(np.float32)
    val_prop = (val_proposal / scale).astype(np.float32)
    test_prop = (test_proposal / scale).astype(np.float32)
    train_target = (bundle.train_target / scale).astype(np.float32)
    val_target = (bundle.val_target / scale).astype(np.float32)

    train_soft = co.candidate_soft_oracle_targets(
        train_cand,
        train_target,
        temperature=0.10,
        topk=min(6, train_cand.shape[1]),
    )[0]
    val_soft = co.candidate_soft_oracle_targets(
        val_cand,
        val_target,
        temperature=0.10,
        topk=min(6, val_cand.shape[1]),
    )[0]

    model = CloudBasisResidualHead(
        cand_feat_dim=train_x.shape[-1],
        ctx_dim=train_ctx.shape[-1],
        hidden_dim=hidden_dim,
        tau=0.75,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    tc = torch.tensor(train_cand, dtype=torch.float32, device=device)
    tp = torch.tensor(train_prop, dtype=torch.float32, device=device)
    tctx = torch.tensor(train_ctx, dtype=torch.float32, device=device)
    ty = torch.tensor(train_target, dtype=torch.float32, device=device)
    tq = torch.tensor(train_soft, dtype=torch.float32, device=device)
    vx = torch.tensor(val_x, dtype=torch.float32, device=device)
    vc = torch.tensor(val_cand, dtype=torch.float32, device=device)
    vp = torch.tensor(val_prop, dtype=torch.float32, device=device)
    vctx = torch.tensor(val_ctx, dtype=torch.float32, device=device)
    vy = torch.tensor(val_target, dtype=torch.float32, device=device)
    vq = torch.tensor(val_soft, dtype=torch.float32, device=device)

    def loss_fn(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: torch.Tensor,
        soft_target: torch.Tensor,
        gate: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        main = torch.nn.functional.smooth_l1_loss(pred, target)
        kl = torch.mean(
            torch.sum(
                soft_target * (torch.log(torch.clamp(soft_target, min=1e-8)) - torch.log(torch.clamp(alpha, min=1e-8))),
                dim=1,
            )
        )
        delta_reg = torch.mean(torch.sum(delta * delta, dim=1))
        gate_reg = torch.mean(gate)
        return main + 0.07 * kl + 1e-3 * delta_reg + 5e-4 * gate_reg

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    batch_size = max(int(batch_size), 128)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_x.shape[0])
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred, alpha, gate, delta, _basis = model(tp[idx], tc[idx], tctx[idx], tx[idx])
            loss = loss_fn(pred, ty[idx], alpha, tq[idx], gate, delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred, alpha, gate, delta, _basis = model(vp, vc, vctx, vx)
            val_loss = float(loss_fn(pred, vy, alpha, vq, gate, delta).detach().cpu())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred, alpha, gate, delta, basis = model(
            torch.tensor(test_prop, dtype=torch.float32, device=device),
            torch.tensor(test_cand, dtype=torch.float32, device=device),
            torch.tensor(test_ctx, dtype=torch.float32, device=device),
            torch.tensor(test_x, dtype=torch.float32, device=device),
        )
        pred_np = pred.detach().cpu().numpy() * scale
        alpha_np = alpha.detach().cpu().numpy()
        gate_np = gate.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()
        basis_np = basis.detach().cpu().numpy()
    entropy = -np.sum(alpha_np * np.log(np.maximum(alpha_np, 1e-8)), axis=1) / max(np.log(alpha_np.shape[1]), EPS)
    extra = {
        "basis_best_epoch": float(best_epoch),
        "basis_val_loss": float(best_val),
        "gate_mean": float(np.mean(gate_np)),
        "gate_p90": float(np.quantile(gate_np, 0.90)),
        "alpha_entropy_mean": float(np.mean(entropy)),
        "delta_norm_mean": float(np.mean(np.linalg.norm(delta_np, axis=1)) * scale),
        "basis_norm_mean": float(np.mean(np.linalg.norm(basis_np, axis=1)) * scale),
        "softmax_temp": float(np.exp(float(model.log_tau.detach().cpu())).clip(0.15, 5.0)),
    }
    return pred_np.astype(np.float32), extra


def build_bundle(cell_type: str, horizon: int, seed: int, args: argparse.Namespace, device: torch.device) -> SplitBundle:
    train_split, val_split, test_split, scale_px, info = v2.train_backbone_and_candidates(
        cell_type,
        horizon,
        seed,
        args,
        device,
    )
    train_idx = co.choose_target_nodes(train_split.mask, args.reranker_train_nodes, seed + 11)
    val_idx = co.choose_target_nodes(val_split.mask, args.reranker_val_nodes, seed + 22)
    test_idx = np.flatnonzero(test_split.mask).astype(np.int64, copy=False)
    if args.max_test_nodes > 0 and len(test_idx) > args.max_test_nodes:
        rng = np.random.default_rng(seed + 33)
        test_idx = np.sort(rng.choice(test_idx, size=args.max_test_nodes, replace=False))
    family_to_idx = {family: idx for idx, family in enumerate(sorted(set(train_split.pack.families)))}
    train_packet = v2.build_feature_packet(train_split, train_idx, family_to_idx, scale_px)
    val_packet = v2.build_feature_packet(val_split, val_idx, family_to_idx, scale_px)
    test_packet = v2.build_feature_packet(test_split, test_idx, family_to_idx, scale_px)
    train_candidates, train_target = candidate_arrays(train_split, train_idx)
    val_candidates, val_target = candidate_arrays(val_split, val_idx)
    test_candidates, test_target = candidate_arrays(test_split, test_idx)
    return SplitBundle(
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        scale_px=scale_px,
        feature_names=train_packet.names,
        feature_groups=train_packet.groups,
        train_x=train_packet.values,
        val_x=val_packet.values,
        test_x=test_packet.values,
        train_candidates=train_candidates,
        val_candidates=val_candidates,
        test_candidates=test_candidates,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        info=info,
    )


def run_setting(cell_type: str, horizon: int, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[sweep] build {cell_type} h{horizon} seed={seed}", flush=True)
    bundle = build_bundle(cell_type, horizon, seed, args, device)
    baseline_pred = bundle.test_split.self_flow_px[bundle.test_idx]
    proposal_pred = bundle.test_split.proposal_px[bundle.test_idx]
    oracle_pred = oracle_prediction(bundle.test_candidates, bundle.test_target)
    baseline_rmse = vector_metrics(baseline_pred, bundle.test_target)["rmse"]
    proposal_rmse = vector_metrics(proposal_pred, bundle.test_target)["rmse"]
    oracle_rmse = vector_metrics(oracle_pred, bundle.test_target)["rmse"]

    rows: list[dict[str, Any]] = []
    fixed_defs = [
        ("self_flow", baseline_pred),
        ("proposal", proposal_pred),
        ("oracle_all", oracle_pred),
    ]
    for name, pred in fixed_defs:
        metrics = vector_metrics(pred, bundle.test_target)
        rows.append(
            {
                "cell_type": cell_type,
                "horizon": horizon,
                "seed": seed,
                "method": name,
                "feature_set": "fixed",
                "aggregator": "fixed",
                **metrics,
                "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                "oracle_gap_closed_pct": (
                    (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                ),
                "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                "val_soft_rmse": np.nan,
                "softmax_temp": np.nan,
                "score_error_corr": np.nan,
                "ndcg_at_5": np.nan,
                "oracle_top1_hit": np.nan,
                "oracle_top3_hit": np.nan,
            }
        )

    for feature_set in args.feature_sets:
        mask = v2.feature_mask(bundle.feature_groups, bundle.feature_names, feature_set)
        tr_x, va_x, te_x, _, _ = v2.standardize_train_val_test(
            bundle.train_x[..., mask],
            bundle.val_x[..., mask],
            bundle.test_x[..., mask],
        )
        tr_x = sanitize_features(tr_x)
        va_x = sanitize_features(va_x)
        te_x = sanitize_features(te_x)
        print(
            f"[sweep] {cell_type} h{horizon} s{seed} feature_set={feature_set} dim={tr_x.shape[-1]}",
            flush=True,
        )
        for method in args.methods:
            try:
                if method == "family_mean_error":
                    val_scores, test_scores = train_family_mean_scores(bundle, by="family")
                    rows.extend(
                        scorer_rows(
                            cell_type=cell_type,
                            horizon=horizon,
                            seed=seed,
                            method=method,
                            feature_set=feature_set,
                            val_scores=val_scores,
                            test_scores=test_scores,
                            bundle=bundle,
                            baseline_rmse=baseline_rmse,
                            proposal_rmse=proposal_rmse,
                            oracle_rmse=oracle_rmse,
                        )
                    )
                elif method == "candidate_name_mean_error":
                    val_scores, test_scores = train_family_mean_scores(bundle, by="name")
                    rows.extend(
                        scorer_rows(
                            cell_type=cell_type,
                            horizon=horizon,
                            seed=seed,
                            method=method,
                            feature_set=feature_set,
                            val_scores=val_scores,
                            test_scores=test_scores,
                            bundle=bundle,
                            baseline_rmse=baseline_rmse,
                            proposal_rmse=proposal_rmse,
                            oracle_rmse=oracle_rmse,
                        )
                    )
                elif method in {"ridge_error", "sgd_top3", "hgb_error", "extra_trees_error"}:
                    val_scores, test_scores = train_flat_error_model(
                        method,
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        max_rows=args.max_flat_train_rows,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    rows.extend(
                        scorer_rows(
                            cell_type=cell_type,
                            horizon=horizon,
                            seed=seed,
                            method=method,
                            feature_set=feature_set,
                            val_scores=val_scores,
                            test_scores=test_scores,
                            bundle=bundle,
                            baseline_rmse=baseline_rmse,
                            proposal_rmse=proposal_rmse,
                            oracle_rmse=oracle_rmse,
                        )
                    )
                elif method in {"ridge_error_soft_blend", "ridge_error_gate_linear", "ridge_error_gate_hgb"}:
                    pred, extra = train_ridge_error_confidence_blend(
                        method,
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        max_rows=args.max_flat_train_rows,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    metrics = vector_metrics(pred, bundle.test_target)
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "feature_set": feature_set,
                            "aggregator": "confidence_blend",
                            **metrics,
                            "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                            "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                            "oracle_gap_closed_pct": (
                                (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                            ),
                            "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                            "val_soft_rmse": extra.get("val_soft_rmse", np.nan),
                            "softmax_temp": extra.get("softmax_temp", np.nan),
                            "blend_alpha": extra.get("blend_alpha", np.nan),
                            "val_blend_rmse": extra.get("val_blend_rmse", np.nan),
                            "gate_mean": extra.get("gate_mean", np.nan),
                            "score_error_corr": np.nan,
                            "ndcg_at_5": np.nan,
                            "oracle_top1_hit": np.nan,
                            "oracle_top3_hit": np.nan,
                        }
                    )
                elif method in {"ridge_vector_mean", "ridge_residual_mean"}:
                    pred = train_candidate_vector_decoder(
                        method,
                        tr_x,
                        te_x,
                        bundle,
                        max_rows=args.max_flat_train_rows,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    metrics = vector_metrics(pred, bundle.test_target)
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "feature_set": feature_set,
                            "aggregator": "candidate_mean_decoder",
                            **metrics,
                            "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                            "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                            "oracle_gap_closed_pct": (
                                (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                            ),
                            "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                            "val_soft_rmse": np.nan,
                            "softmax_temp": np.nan,
                            "score_error_corr": np.nan,
                            "ndcg_at_5": np.nan,
                            "oracle_top1_hit": np.nan,
                            "oracle_top3_hit": np.nan,
                        }
                    )
                elif method in {"cloud_ridge_vector", "cloud_hgb_vector", "cloud_extra_trees_vector"}:
                    pred = train_cloud_decoder(
                        method,
                        tr_x,
                        te_x,
                        bundle,
                        stable_seed(method, feature_set, base=seed),
                    )
                    metrics = vector_metrics(pred, bundle.test_target)
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "feature_set": feature_set,
                            "aggregator": "cloud_decoder",
                            **metrics,
                            "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                            "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                            "oracle_gap_closed_pct": (
                                (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                            ),
                            "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                            "val_soft_rmse": np.nan,
                            "softmax_temp": np.nan,
                            "score_error_corr": np.nan,
                            "ndcg_at_5": np.nan,
                            "oracle_top1_hit": np.nan,
                            "oracle_top3_hit": np.nan,
                        }
                    )
                elif method in {"torch_mlp_error", "torch_set_error"}:
                    val_scores, test_scores = train_torch_error_scorer(
                        method,
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        device,
                        epochs=args.torch_epochs,
                        batch_nodes=args.torch_batch_nodes,
                        hidden_dim=args.torch_hidden_dim,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    rows.extend(
                        scorer_rows(
                            cell_type=cell_type,
                            horizon=horizon,
                            seed=seed,
                            method=method,
                            feature_set=feature_set,
                            val_scores=val_scores,
                            test_scores=test_scores,
                            bundle=bundle,
                            baseline_rmse=baseline_rmse,
                            proposal_rmse=proposal_rmse,
                            oracle_rmse=oracle_rmse,
                        )
                    )
                elif method in {"torch_gain_weighted_soft", "torch_radius_quality_bce"}:
                    val_scores, test_scores, _extra = train_torch_oracle_gain_scorer(
                        method,
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        device,
                        epochs=args.torch_epochs,
                        batch_nodes=args.torch_batch_nodes,
                        hidden_dim=args.torch_hidden_dim,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    rows.extend(
                        scorer_rows(
                            cell_type=cell_type,
                            horizon=horizon,
                            seed=seed,
                            method=method,
                            feature_set=feature_set,
                            val_scores=val_scores,
                            test_scores=test_scores,
                            bundle=bundle,
                            baseline_rmse=baseline_rmse,
                            proposal_rmse=proposal_rmse,
                            oracle_rmse=oracle_rmse,
                        )
                    )
                elif method in {"torch_gain_weighted_soft_blend", "torch_radius_quality_bce_blend"}:
                    pred, extra = train_torch_oracle_gain_blend(
                        method,
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        device,
                        epochs=args.torch_epochs,
                        batch_nodes=args.torch_batch_nodes,
                        hidden_dim=args.torch_hidden_dim,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    metrics = vector_metrics(pred, bundle.test_target)
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "feature_set": feature_set,
                            "aggregator": "oracle_gain_blend",
                            **metrics,
                            "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                            "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                            "oracle_gap_closed_pct": (
                                (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                            ),
                            "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                            "val_soft_rmse": np.nan,
                            "softmax_temp": extra.get("softmax_temp", np.nan),
                            "blend_alpha": extra.get("blend_alpha", np.nan),
                            "val_blend_rmse": extra.get("val_blend_rmse", np.nan),
                            "oracle_weight_train_mean": extra.get("oracle_weight_train_mean", np.nan),
                            "oracle_weight_val_mean": extra.get("oracle_weight_val_mean", np.nan),
                            "quality_best_epoch": extra.get("quality_best_epoch", np.nan),
                            "quality_val_loss": extra.get("quality_val_loss", np.nan),
                            "score_error_corr": np.nan,
                            "ndcg_at_5": np.nan,
                            "oracle_top1_hit": np.nan,
                            "oracle_top3_hit": np.nan,
                        }
                    )
                elif method == "cloud_mlp_residual":
                    pred = train_cloud_mlp_residual(
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        device,
                        epochs=args.torch_epochs,
                        batch_size=args.torch_batch_nodes,
                        hidden_dim=args.torch_hidden_dim,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    metrics = vector_metrics(pred, bundle.test_target)
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "feature_set": feature_set,
                            "aggregator": "cloud_residual_decoder",
                            **metrics,
                            "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                            "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                            "oracle_gap_closed_pct": (
                                (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                            ),
                            "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                            "val_soft_rmse": np.nan,
                            "softmax_temp": np.nan,
                            "score_error_corr": np.nan,
                            "ndcg_at_5": np.nan,
                            "oracle_top1_hit": np.nan,
                            "oracle_top3_hit": np.nan,
                        }
                    )
                elif method == "cloud_basis_residual":
                    pred, extra = train_cloud_basis_residual(
                        tr_x,
                        va_x,
                        te_x,
                        bundle,
                        device,
                        epochs=args.torch_epochs,
                        batch_size=args.torch_batch_nodes,
                        hidden_dim=args.torch_hidden_dim,
                        seed=stable_seed(method, feature_set, base=seed),
                    )
                    metrics = vector_metrics(pred, bundle.test_target)
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "feature_set": feature_set,
                            "aggregator": "basis_residual_decoder",
                            **metrics,
                            "gain_vs_self_flow_pct": co.gain_pct(baseline_rmse, metrics["rmse"]),
                            "gain_vs_proposal_pct": co.gain_pct(proposal_rmse, metrics["rmse"]),
                            "oracle_gap_closed_pct": (
                                (proposal_rmse - metrics["rmse"]) / max(proposal_rmse - oracle_rmse, EPS) * 100.0
                            ),
                            "gap_to_oracle_rmse": metrics["rmse"] - oracle_rmse,
                            "val_soft_rmse": np.nan,
                            "softmax_temp": extra.get("softmax_temp", np.nan),
                            "basis_best_epoch": extra.get("basis_best_epoch", np.nan),
                            "basis_val_loss": extra.get("basis_val_loss", np.nan),
                            "gate_mean": extra.get("gate_mean", np.nan),
                            "gate_p90": extra.get("gate_p90", np.nan),
                            "alpha_entropy_mean": extra.get("alpha_entropy_mean", np.nan),
                            "delta_norm_mean": extra.get("delta_norm_mean", np.nan),
                            "basis_norm_mean": extra.get("basis_norm_mean", np.nan),
                            "score_error_corr": np.nan,
                            "ndcg_at_5": np.nan,
                            "oracle_top1_hit": np.nan,
                            "oracle_top3_hit": np.nan,
                        }
                    )
                else:
                    raise ValueError(method)
            except Exception as exc:
                rows.append(
                    {
                        "cell_type": cell_type,
                        "horizon": horizon,
                        "seed": seed,
                        "method": method,
                        "feature_set": feature_set,
                        "aggregator": "ERROR",
                        "rmse": np.nan,
                        "r2": np.nan,
                        "angular_cosine": np.nan,
                        "magnitude_ratio": np.nan,
                        "gain_vs_self_flow_pct": np.nan,
                        "gain_vs_proposal_pct": np.nan,
                        "oracle_gap_closed_pct": np.nan,
                        "gap_to_oracle_rmse": np.nan,
                        "error": repr(exc),
                    }
                )
                print(f"[sweep][ERROR] {cell_type} h{horizon} {feature_set}/{method}: {exc}", flush=True)
    summary = pd.DataFrame(rows)
    deployable = summary[summary["rmse"].notna() & ~summary["method"].isin(["oracle_all"])].copy()
    best = (
        deployable
        .sort_values("rmse")
        .groupby(["cell_type", "horizon", "seed"], as_index=False)
        .head(args.best_rows_per_setting)
    )
    print(
        f"[sweep] done {cell_type} h{horizon} seed={seed}: best={best.iloc[0]['method']}/{best.iloc[0]['feature_set']} "
        f"{best.iloc[0]['aggregator']} RMSE={best.iloc[0]['rmse']:.3f}",
        flush=True,
    )
    return summary, best


def make_plots(out_dir: Path, summary: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = summary[summary["rmse"].notna() & ~summary["method"].isin(["oracle_all"])].copy()
    if df.empty:
        return
    for (cell_type, horizon), group in df.groupby(["cell_type", "horizon"]):
        top = group.sort_values("rmse").head(20).copy()
        top["label"] = top["method"] + "\n" + top["feature_set"] + "\n" + top["aggregator"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(top["label"], top["rmse"], color="#4C78A8")
        ax.set_title(f"Oracle signal sweep best RMSE: {cell_type} h{horizon}")
        ax.set_ylabel("RMSE px")
        ax.tick_params(axis="x", rotation=75, labelsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"sweep_best_{cell_type}_h{horizon}.png", dpi=180)
        plt.close(fig)


def write_report(out_dir: Path, summary: pd.DataFrame, best: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# LaChance oracle signal broad sweep",
        "",
        "This is a hypothesis-mining run, not a final gate. The goal is to find any scorer/decoder route that starts closing the candidate oracle gap.",
        "",
        f"- datasets: `{', '.join(args.cell_types)}`",
        f"- horizons: `{', '.join(map(str, args.horizons))}`",
        f"- seeds: `{', '.join(map(str, args.seeds))}`",
        f"- feature sets: `{', '.join(args.feature_sets)}`",
        f"- methods: `{', '.join(args.methods)}`",
        "",
        "## Best Rows",
        "",
    ]
    cols = [
        "cell_type",
        "horizon",
        "seed",
        "method",
        "feature_set",
        "aggregator",
        "rmse",
        "r2",
        "gain_vs_proposal_pct",
        "oracle_gap_closed_pct",
        "gap_to_oracle_rmse",
    ]
    if best.empty:
        lines.append("No valid rows.")
    else:
        lines.append(best.loc[:, [c for c in cols if c in best.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Per Setting Min", ""])
    if not summary.empty:
        per = (
            summary[summary["rmse"].notna() & ~summary["method"].isin(["oracle_all"])]
            .sort_values("rmse")
            .groupby(["cell_type", "horizon", "seed"], as_index=False)
            .head(1)
        )
        lines.append(per.loc[:, [c for c in cols if c in per.columns]].to_markdown(index=False, floatfmt=".4f"))
    (out_dir / "oracle_signal_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def append_plan_update(out_dir: Path, best: pd.DataFrame) -> None:
    plan = REPO_ROOT / "research_plan_prior_gradient_reranker_teacher_2026-06-10.md"
    if not plan.exists():
        return
    text = plan.read_text(encoding="utf-8")
    marker = "## 2026-06-12: Broad oracle-signal sweep"
    if marker in text:
        return
    if best.empty:
        summary = "- Sweep запущен, но валидных строк нет."
    else:
        top = best.sort_values("rmse").head(8)
        summary = top.loc[
            :,
            ["cell_type", "horizon", "seed", "method", "feature_set", "aggregator", "rmse", "oracle_gap_closed_pct"],
        ].to_markdown(index=False, floatfmt=".3f")
    update = f"""

{marker}

- Реализован broad runner `scripts/run_lachance_oracle_signal_sweep.py`.
- Цель: не доказывать одну гипотезу, а быстро перебрать scorer/aggregator/decoder варианты и найти зацеп за oracle gap.
- Output directory: `{out_dir}`.

Top rows:

{summary}
"""
    plan.write_text(text.rstrip() + update + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--table-root", dest="data_root", type=Path, default=v2.la.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/lachance_oracle_signal_sweep"))
    parser.add_argument("--cell-types", default="MDCK_Bulk,MDCK_Edge")
    parser.add_argument("--horizons", default="6,4")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--reranker-train-nodes", type=int, default=12000)
    parser.add_argument("--reranker-val-nodes", type=int, default=5000)
    parser.add_argument("--max-test-nodes", type=int, default=0)
    parser.add_argument("--max-flat-train-rows", type=int, default=180000)
    parser.add_argument("--best-rows-per-setting", type=int, default=25)
    parser.add_argument("--torch-epochs", type=int, default=28)
    parser.add_argument("--torch-batch-nodes", type=int, default=768)
    parser.add_argument("--torch-hidden-dim", type=int, default=160)
    parser.add_argument("--device", default="auto")

    # Backbone/candidate args mirrored from critic v2.
    parser.add_argument("--split-mode", choices=["movie", "frame"], default="movie")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-movies", type=int, default=8)
    parser.add_argument("--max-tracks-per-movie", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--crop-fraction", type=float, default=0.08)
    parser.add_argument("--r-cut-px", type=float, default=50.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temporal-epochs", type=int, default=10)
    parser.add_argument("--flow-epochs", type=int, default=8)
    parser.add_argument("--mp-epochs", type=int, default=20)
    parser.add_argument("--backbone-batch-size", type=int, default=4096)
    parser.add_argument("--sequence-balanced-loss", action="store_true")
    parser.add_argument("--mp-layers", type=int, default=3)
    parser.add_argument("--mp-hidden-dim", type=int, default=72)
    parser.add_argument("--mp-edge-hidden-dim", type=int, default=56)
    parser.add_argument("--mp-max-delta-norm", type=float, default=1.35)
    parser.add_argument("--mp-lr", type=float, default=1.2e-3)
    parser.add_argument("--mp-social-l2", type=float, default=0.0015)
    parser.add_argument("--mp-flow-gate-l2", type=float, default=0.0005)
    parser.add_argument("--sobol-count", type=int, default=8)
    parser.add_argument("--gaussian-count", type=int, default=4)
    parser.add_argument("--backbone-variant", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plan-update", action="store_true")
    args = parser.parse_args()
    args.cell_types = parse_csv(args.cell_types, ["MDCK_Bulk", "MDCK_Edge"])
    args.horizons = parse_int_csv(args.horizons, [6, 4])
    args.seeds = parse_int_csv(args.seeds, [42])
    args.feature_sets = parse_csv(args.feature_sets, DEFAULT_FEATURE_SETS)
    args.methods = parse_csv(args.methods, DEFAULT_METHODS)
    if args.smoke:
        args.max_movies = min(args.max_movies, 3)
        args.crop_fraction = min(args.crop_fraction, 0.02)
        args.temporal_epochs = min(args.temporal_epochs, 1)
        args.flow_epochs = min(args.flow_epochs, 1)
        args.mp_epochs = min(args.mp_epochs, 1)
        args.reranker_train_nodes = min(args.reranker_train_nodes, 384)
        args.reranker_val_nodes = min(args.reranker_val_nodes, 256)
        args.max_test_nodes = min(args.max_test_nodes if args.max_test_nodes > 0 else 384, 384)
        args.max_flat_train_rows = min(args.max_flat_train_rows, 25000)
        args.torch_epochs = min(args.torch_epochs, 5)
        args.torch_batch_nodes = min(args.torch_batch_nodes, 384)
        args.sobol_count = min(args.sobol_count, 4)
        args.gaussian_count = min(args.gaussian_count, 2)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    all_summary: list[pd.DataFrame] = []
    all_best: list[pd.DataFrame] = []
    for cell_type in args.cell_types:
        for horizon in args.horizons:
            for seed in args.seeds:
                summary, best = run_setting(cell_type, horizon, seed, args, device)
                all_summary.append(summary)
                all_best.append(best)
                interim = pd.concat(all_summary, ignore_index=True)
                interim.to_csv(args.out_dir / "oracle_signal_sweep_summary.csv", index=False)
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    best_df = pd.concat(all_best, ignore_index=True) if all_best else pd.DataFrame()
    summary_df.to_csv(args.out_dir / "oracle_signal_sweep_summary.csv", index=False)
    best_df.to_csv(args.out_dir / "oracle_signal_sweep_best.csv", index=False)
    with (args.out_dir / "oracle_signal_sweep_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "cell_types": args.cell_types,
                "horizons": args.horizons,
                "seeds": args.seeds,
                "feature_sets": args.feature_sets,
                "methods": args.methods,
                "smoke": args.smoke,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    make_plots(args.out_dir, summary_df)
    write_report(args.out_dir, summary_df, best_df, args)
    if not args.no_plan_update:
        append_plan_update(args.out_dir, best_df)
    print(f"[sweep] done: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
