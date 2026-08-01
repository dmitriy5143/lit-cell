#!/usr/bin/env python3
"""Query-level risk calibrator for LaChance route candidates.

This runner tests the current bottleneck hypothesis:

    good h1..h6 route-query trajectories already exist,
    but the route logits / weighted mixture are not calibrated to trajectory risk.

It reuses the existing decomposition/generator/RouteQueryRefiner stack from
``run_lachance_sequence_critic_refiner.py`` and trains a separate causal risk
head over whole route-query trajectories.  The target future is used only for
training risk labels and validation, never as an inference feature.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "query_risk_calibrator_2026-06-27"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def weighted_residual(query_pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(weights[:, :, None, None] * query_pred, axis=1).astype(np.float32)


def residual_rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return audit.rmse(audit.flatten_residual(true), audit.flatten_residual(pred))


def residual_endpoint_rmse(true: np.ndarray, pred: np.ndarray, horizons: list[int], weights: list[float] | None = None) -> float:
    errs = []
    for h in horizons:
        p = np.sum(pred[:, : int(h), :], axis=1)
        t = np.sum(true[:, : int(h), :], axis=1)
        errs.append(np.sum((p - t) ** 2, axis=-1))
    stacked = np.stack(errs, axis=-1)
    if weights is None or len(weights) == 0:
        score = np.mean(stacked)
    else:
        w = np.asarray(weights, dtype=np.float32)
        w = w / np.sum(w)
        score = np.mean(np.sum(stacked * w[None, :], axis=-1))
    return float(np.sqrt(score))


def endpoint_errors(query_pred: np.ndarray, true: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Average endpoint squared error per query across requested horizons."""
    errs = []
    for h in horizons:
        qp = np.sum(query_pred[:, :, : int(h), :], axis=2)
        yt = np.sum(true[:, : int(h), :], axis=1)
        errs.append(np.sum((qp - yt[:, None, :]) ** 2, axis=-1))
    return np.mean(np.stack(errs, axis=-1), axis=-1).astype(np.float32)


def weighted_endpoint_errors(
    query_pred: np.ndarray,
    true: np.ndarray,
    horizons: list[int],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted endpoint squared error per query across requested horizons."""
    if not horizons:
        raise ValueError("horizons must be non-empty")
    if weights is None or len(weights) == 0:
        return endpoint_errors(query_pred, true, horizons)
    if len(weights) != len(horizons):
        raise ValueError(f"risk weights length {len(weights)} does not match horizons length {len(horizons)}")
    w = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(w).all() or float(np.sum(np.abs(w))) <= EPS:
        raise ValueError("risk weights must be finite and non-zero")
    w = w / np.sum(w)
    errs = []
    for h in horizons:
        qp = np.sum(query_pred[:, :, : int(h), :], axis=2)
        yt = np.sum(true[:, : int(h), :], axis=1)
        errs.append(np.sum((qp - yt[:, None, :]) ** 2, axis=-1))
    stacked = np.stack(errs, axis=-1)
    return np.sum(stacked * w[None, None, :], axis=-1).astype(np.float32)


def risk_label_horizons(args: argparse.Namespace) -> list[int]:
    text = str(getattr(args, "risk_label_horizons", "") or "").strip()
    if text:
        return parse_ints(text)
    return list(args.horizons)


def risk_label_weights(args: argparse.Namespace, horizons: list[int]) -> list[float] | None:
    text = str(getattr(args, "risk_horizon_weights", "") or "").strip()
    if not text:
        return None
    weights = [float(x) for x in text.split(",") if x.strip()]
    if len(weights) != len(horizons):
        raise ValueError("--risk-horizon-weights must match --risk-label-horizons length")
    return weights


def risk_endpoint_errors(query_pred: np.ndarray, true: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    horizons = risk_label_horizons(args)
    weights = risk_label_weights(args, horizons)
    return weighted_endpoint_errors(query_pred, true, horizons, weights)


def query_oracle_residual(query_pred: np.ndarray, true: np.ndarray, horizons: list[int]) -> np.ndarray:
    err = endpoint_errors(query_pred, true, horizons)
    take = np.argmin(err, axis=1)
    return query_pred[np.arange(len(take)), take].astype(np.float32)


def top_query_residual(query_pred: np.ndarray, route_logits: np.ndarray) -> np.ndarray:
    take = np.argmax(route_logits, axis=1)
    return query_pred[np.arange(len(take)), take].astype(np.float32)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    ez = np.exp(z)
    return (ez / np.maximum(np.sum(ez, axis=axis, keepdims=True), EPS)).astype(np.float32)


def tune_temperature(
    logits: np.ndarray,
    query_pred: np.ndarray,
    true: np.ndarray,
    temps: list[float],
) -> tuple[float, np.ndarray, float]:
    best_t = float(temps[0])
    best_pred = weighted_residual(query_pred, softmax_np(logits / best_t, axis=1))
    best_rmse = residual_rmse(true, best_pred)
    for t in temps[1:]:
        pred = weighted_residual(query_pred, softmax_np(logits / float(t), axis=1))
        rmse = residual_rmse(true, pred)
        if rmse < best_rmse:
            best_t = float(t)
            best_rmse = float(rmse)
            best_pred = pred
    return best_t, best_pred, best_rmse


def query_sequence_features(
    *,
    query_pred: np.ndarray,
    base: np.ndarray,
    route_logits: np.ndarray,
    ctx: np.ndarray,
    horizons: list[int],
    include_context: bool,
    include_query_id: bool,
) -> tuple[np.ndarray, list[str]]:
    n, q, hmax, _ = query_pred.shape
    total_steps = base[:, None, None, :] + query_pred
    endpoints = np.cumsum(total_steps, axis=2)
    base_end = np.stack([base * float(h) for h in horizons], axis=1)
    endpoint_sel = np.stack([endpoints[:, :, int(h) - 1, :] for h in horizons], axis=2)
    endpoint_delta = endpoint_sel - base_end[:, None, :, :]
    speed = np.linalg.norm(total_steps, axis=-1)
    acc = np.diff(total_steps, axis=2)
    acc_norm = np.linalg.norm(acc, axis=-1) if hmax > 1 else np.zeros((n, q, 1), dtype=np.float32)
    path_len = np.sum(speed, axis=2, keepdims=True)
    net = endpoint_sel[:, :, -1, :]
    net_norm = np.linalg.norm(net, axis=-1, keepdims=True)
    persistence = net_norm / np.maximum(path_len, EPS)
    if hmax > 1:
        prev = total_steps[:, :, :-1, :]
        nxt = total_steps[:, :, 1:, :]
        denom = np.maximum(np.linalg.norm(prev, axis=-1) * np.linalg.norm(nxt, axis=-1), EPS)
        turn_cos = np.sum(prev * nxt, axis=-1) / denom
        turn_sin = (prev[..., 0] * nxt[..., 1] - prev[..., 1] * nxt[..., 0]) / denom
    else:
        turn_cos = np.zeros((n, q, 1), dtype=np.float32)
        turn_sin = np.zeros((n, q, 1), dtype=np.float32)

    route_prob = softmax_np(route_logits, axis=1)
    route_rank = np.argsort(np.argsort(-route_logits, axis=1), axis=1).astype(np.float32)
    route_rank = route_rank / max(float(q - 1), 1.0)

    pieces: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        a = arr.reshape(n, q, -1).astype(np.float32)
        pieces.append(a)
        for j in range(a.shape[-1]):
            names.append(f"{name}_{j}" if a.shape[-1] > 1 else name)

    add("query_residual", query_pred)
    add("total_steps", total_steps)
    add("endpoint", endpoint_sel)
    add("endpoint_delta_base", endpoint_delta)
    add("endpoint_mag", np.linalg.norm(endpoint_sel, axis=-1))
    add("endpoint_delta_mag", np.linalg.norm(endpoint_delta, axis=-1))
    add("speed_mean", np.mean(speed, axis=2, keepdims=True))
    add("speed_std", np.std(speed, axis=2, keepdims=True))
    add("speed_first", speed[:, :, :1])
    add("speed_last", speed[:, :, -1:])
    add("acc_mean", np.mean(acc_norm, axis=2, keepdims=True))
    add("acc_std", np.std(acc_norm, axis=2, keepdims=True))
    add("path_len", path_len)
    add("net_norm", net_norm)
    add("persistence", persistence)
    add("turn_cos_mean", np.mean(turn_cos, axis=2, keepdims=True))
    add("turn_cos_std", np.std(turn_cos, axis=2, keepdims=True))
    add("turn_sin_mean", np.mean(turn_sin, axis=2, keepdims=True))
    add("route_logit", route_logits[:, :, None])
    add("route_prob", route_prob[:, :, None])
    add("route_rank", route_rank[:, :, None])
    if include_query_id:
        eye = np.eye(q, dtype=np.float32)[None, :, :].repeat(n, axis=0)
        add("query_id", eye)
    if include_context and ctx.shape[1] > 0:
        c = ctx[:, None, :].repeat(q, axis=1)
        add("ctx", c)
    feat = np.concatenate(pieces, axis=-1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names


def append_context_to_set_features(features: np.ndarray, ctx: np.ndarray, include_context: bool) -> np.ndarray:
    if not include_context or ctx.shape[1] == 0:
        return features.astype(np.float32)
    n, k, _ = features.shape
    c = ctx[:, None, :].repeat(k, axis=1)
    return np.concatenate([features, c], axis=-1).astype(np.float32)


def standardize_query_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tr = train.reshape(-1, train.shape[-1])
    va = val.reshape(-1, val.shape[-1])
    te = test.reshape(-1, test.shape[-1])
    tr_z, va_z, te_z, scaler = seq.standardize(tr, va, te)
    return tr_z.reshape(train.shape), va_z.reshape(val.shape), te_z.reshape(test.shape), scaler


class QueryRiskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QueryRiskSetTransformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, heads: int, layers: int):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(layers)))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


def make_risk_model(in_dim: int, args: argparse.Namespace) -> nn.Module:
    if getattr(args, "risk_model", "mlp") == "set_transformer":
        return QueryRiskSetTransformer(
            in_dim,
            args.risk_hidden,
            args.risk_dropout,
            args.risk_transformer_heads,
            args.risk_transformer_layers,
        )
    return QueryRiskMLP(in_dim, args.risk_hidden, args.risk_dropout)


@dataclass
class QueryOutputs:
    query_pred: np.ndarray
    route_logits: np.ndarray
    route_probs: np.ndarray
    weighted_pred: np.ndarray
    top_pred: np.ndarray
    query_oracle: np.ndarray


def extract_queries(
    model: nn.Module,
    ctx: np.ndarray,
    cand: seq.CandidatePack,
    true: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> QueryOutputs:
    model.eval()
    qpreds, logits, probs, weighted, top = [], [], [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, 9201, shuffle=False):
            pred, _, weights, route_logits, _, query_pred = seq.critic_forward(
                model,
                to_tensor(cand.features[idx], device),
                to_tensor(ctx[idx], device),
                to_tensor(cand.residual[idx], device),
                arch="route_query",
            )
            if query_pred is None:
                raise RuntimeError("Query extraction requires critic_arch=route_query")
            qpreds.append(query_pred.cpu().numpy())
            logits.append(route_logits.cpu().numpy())
            probs.append(weights.cpu().numpy())
            weighted.append(pred.cpu().numpy())
            top_idx = torch.argmax(route_logits, dim=1)
            top_q = query_pred[torch.arange(len(top_idx), device=device), top_idx]
            top.append(top_q.cpu().numpy())
    query_np = np.concatenate(qpreds, axis=0).astype(np.float32)
    logits_np = np.concatenate(logits, axis=0).astype(np.float32)
    probs_np = np.concatenate(probs, axis=0).astype(np.float32)
    weighted_np = np.concatenate(weighted, axis=0).astype(np.float32)
    top_np = np.concatenate(top, axis=0).astype(np.float32)
    return QueryOutputs(
        query_pred=query_np,
        route_logits=logits_np,
        route_probs=probs_np,
        weighted_pred=weighted_np,
        top_pred=top_np,
        query_oracle=query_oracle_residual(query_np, true, args.horizons),
    )


def soft_labels_from_error(err: torch.Tensor, temperature: float) -> torch.Tensor:
    e = err - torch.min(err, dim=1, keepdim=True).values
    return torch.softmax(-e / max(float(temperature), 1e-6), dim=1)


def train_risk_model(
    feat_train: np.ndarray,
    feat_val: np.ndarray,
    q_train: QueryOutputs,
    q_val: QueryOutputs,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
    shuffled_labels: bool,
) -> tuple[QueryRiskMLP, pd.DataFrame, float]:
    model = make_risk_model(feat_train.shape[-1], args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.risk_lr, weight_decay=args.risk_weight_decay)
    train_err_np = risk_endpoint_errors(q_train.query_pred, residual_train, args)
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 17001)
        train_err_np = train_err_np[rng.permutation(len(train_err_np))]
    val_err_np = risk_endpoint_errors(q_val.query_pred, residual_val, args)
    n = len(feat_train)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []

    for epoch in range(args.risk_epochs):
        model.train()
        losses = []
        for idx in closure.batches(n, args.risk_batch_size, args.seed + 18100 + epoch):
            x = to_tensor(feat_train[idx], device)
            err = to_tensor(train_err_np[idx], device)
            pred_risk = model(x)
            q_soft = soft_labels_from_error(err, args.risk_oracle_temperature)
            listwise = -torch.mean(torch.sum(q_soft * F.log_softmax(-pred_risk, dim=1), dim=1))
            target_log = torch.log1p(err)
            target_z = (target_log - target_log.mean(dim=1, keepdim=True)) / torch.clamp(target_log.std(dim=1, keepdim=True), min=1e-3)
            pred_z = (pred_risk - pred_risk.mean(dim=1, keepdim=True)) / torch.clamp(pred_risk.std(dim=1, keepdim=True), min=1e-3)
            mse = F.smooth_l1_loss(pred_z, target_z)
            loss = args.risk_listwise_weight * listwise + args.risk_mse_weight * mse
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == args.risk_epochs - 1 or epoch % max(1, args.risk_epochs // 5) == 0:
            model.eval()
            risks = predict_risk(model, feat_val, args, device=device)
            temp, pred, rmse = tune_risk_temperature(risks, q_val.query_pred, residual_val, args.risk_temperatures, args)
            top_pred = q_val.query_pred[np.arange(len(risks)), np.argmin(risks, axis=1)]
            top_rmse = residual_rmse(residual_val, top_pred)
            corr = risk_error_corr(risks, val_err_np)
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_weighted_rmse": float(rmse),
                    "val_top_rmse": float(top_rmse),
                    "val_best_temp": float(temp),
                    "val_risk_error_corr": float(corr),
                }
            )
            if rmse < best_val:
                best_val = float(rmse)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    risks_val = predict_risk(model, feat_val, args, device=device)
    best_temp, _, _ = tune_risk_temperature(risks_val, q_val.query_pred, residual_val, args.risk_temperatures, args)
    return model, pd.DataFrame(rows), float(best_temp)


def predict_risk(model: QueryRiskMLP, feat: np.ndarray, args: argparse.Namespace, *, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for idx in closure.batches(len(feat), args.risk_batch_size, 19100, shuffle=False):
            out.append(model(to_tensor(feat[idx], device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def tune_risk_temperature(
    risk: np.ndarray,
    query_pred: np.ndarray,
    true: np.ndarray,
    temps: list[float],
    args: argparse.Namespace,
) -> tuple[float, np.ndarray, float]:
    logits = -risk
    if getattr(args, "risk_temperature_metric", "full_residual") != "risk_label":
        return tune_temperature(logits, query_pred, true, temps)
    horizons = risk_label_horizons(args)
    weights = risk_label_weights(args, horizons)
    best_t = float(temps[0])
    best_pred = weighted_residual(query_pred, softmax_np(logits / best_t, axis=1))
    best_rmse = residual_endpoint_rmse(true, best_pred, horizons, weights)
    for t in temps[1:]:
        pred = weighted_residual(query_pred, softmax_np(logits / float(t), axis=1))
        rmse = residual_endpoint_rmse(true, pred, horizons, weights)
        if rmse < best_rmse:
            best_t = float(t)
            best_rmse = float(rmse)
            best_pred = pred
    return best_t, best_pred, best_rmse


def risk_error_corr(risk: np.ndarray, err: np.ndarray) -> float:
    a = risk.reshape(-1)
    b = err.reshape(-1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fit_predict_sklearn_risk(
    *,
    model_kind: str,
    feat_train: np.ndarray,
    feat_val: np.ndarray,
    feat_test: np.ndarray,
    q_train: QueryOutputs,
    q_val: QueryOutputs,
    q_test: QueryOutputs,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    residual_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, float]]:
    train_err = risk_endpoint_errors(q_train.query_pred, residual_train, args)
    val_err = risk_endpoint_errors(q_val.query_pred, residual_val, args)
    test_err = risk_endpoint_errors(q_test.query_pred, residual_test, args)
    xtr = feat_train.reshape(-1, feat_train.shape[-1])
    xva = feat_val.reshape(-1, feat_val.shape[-1])
    xte = feat_test.reshape(-1, feat_test.shape[-1])
    ytr = np.log1p(train_err.reshape(-1))
    if model_kind == "hgbdt":
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("sklearn HistGradientBoostingRegressor is unavailable")
        model = HistGradientBoostingRegressor(
            max_iter=args.hgbdt_max_iter,
            learning_rate=args.hgbdt_lr,
            max_leaf_nodes=args.hgbdt_max_leaf_nodes,
            l2_regularization=args.hgbdt_l2,
            random_state=args.seed,
        )
    elif model_kind == "ridge":
        if Ridge is None:
            raise RuntimeError("sklearn Ridge is unavailable")
        model = Ridge(alpha=args.risk_ridge_alpha, random_state=args.seed)
    else:
        raise ValueError(f"Unknown sklearn risk model: {model_kind}")
    model.fit(xtr, ytr)
    val_risk = model.predict(xva).reshape(feat_val.shape[:2]).astype(np.float32)
    test_risk = model.predict(xte).reshape(feat_test.shape[:2]).astype(np.float32)
    best_temp, _, val_rmse = tune_risk_temperature(val_risk, q_val.query_pred, residual_val, args.risk_temperatures, args)
    diag = {
        "val_weighted_rmse": float(val_rmse),
        "val_risk_error_corr": risk_error_corr(val_risk, val_err),
        "test_risk_error_corr": risk_error_corr(test_risk, test_err),
    }
    return test_risk, float(best_temp), diag


def fit_predict_sklearn_set_risk(
    *,
    model_kind: str,
    feat_train: np.ndarray,
    feat_val: np.ndarray,
    feat_test: np.ndarray,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    residual_test: np.ndarray,
    true_train: np.ndarray,
    true_val: np.ndarray,
    true_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, float]]:
    train_err = risk_endpoint_errors(residual_train, true_train, args)
    val_err = risk_endpoint_errors(residual_val, true_val, args)
    test_err = risk_endpoint_errors(residual_test, true_test, args)
    xtr = feat_train.reshape(-1, feat_train.shape[-1])
    xva = feat_val.reshape(-1, feat_val.shape[-1])
    xte = feat_test.reshape(-1, feat_test.shape[-1])
    ytr = np.log1p(train_err.reshape(-1))
    if model_kind == "hgbdt":
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("sklearn HistGradientBoostingRegressor is unavailable")
        model = HistGradientBoostingRegressor(
            max_iter=args.hgbdt_max_iter,
            learning_rate=args.hgbdt_lr,
            max_leaf_nodes=args.hgbdt_max_leaf_nodes,
            l2_regularization=args.hgbdt_l2,
            random_state=args.seed,
        )
    elif model_kind == "ridge":
        if Ridge is None:
            raise RuntimeError("sklearn Ridge is unavailable")
        model = Ridge(alpha=args.risk_ridge_alpha, random_state=args.seed)
    else:
        raise ValueError(f"Unknown sklearn risk model: {model_kind}")
    model.fit(xtr, ytr)
    val_risk = model.predict(xva).reshape(feat_val.shape[:2]).astype(np.float32)
    test_risk = model.predict(xte).reshape(feat_test.shape[:2]).astype(np.float32)
    best_temp, _, val_rmse = tune_risk_temperature(val_risk, residual_val, true_val, args.risk_temperatures, args)
    diag = {
        "val_weighted_rmse": float(val_rmse),
        "val_risk_error_corr": risk_error_corr(val_risk, val_err),
        "test_risk_error_corr": risk_error_corr(test_risk, test_err),
    }
    return test_risk, float(best_temp), diag


def endpoint_rows(arrays: audit.SplitArrays, pred: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    args.critic_arch = "route_query"

    arrays, split = audit.prepare_data(args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)

    ctx_blocks = closure.variant_blocks(args.critic_context_variant, arrays.x_train)
    ctx_train_raw = seq.flatten_blocks(arrays.x_train, ctx_blocks)
    ctx_val_raw = seq.flatten_blocks(arrays.x_val, ctx_blocks)
    ctx_test_raw = seq.flatten_blocks(arrays.x_test, ctx_blocks)
    if args.add_decomposition_context:
        pred_ctx_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
        pred_ctx_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
        pred_ctx_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
        ctx_train_raw = np.concatenate([ctx_train_raw, seq.decomposition_context_features(pred_ctx_train, mode_k=args.mode_k)], axis=1)
        ctx_val_raw = np.concatenate([ctx_val_raw, seq.decomposition_context_features(pred_ctx_val, mode_k=args.mode_k)], axis=1)
        ctx_test_raw = np.concatenate([ctx_test_raw, seq.decomposition_context_features(pred_ctx_test, mode_k=args.mode_k)], axis=1)
    if ctx_train_raw.shape[1] > args.max_critic_context_features:
        var = np.var(ctx_train_raw, axis=0)
        keep = np.argsort(var)[-args.max_critic_context_features :]
        ctx_train_raw = ctx_train_raw[:, keep]
        ctx_val_raw = ctx_val_raw[:, keep]
        ctx_test_raw = ctx_test_raw[:, keep]
    ctx_train, ctx_val, ctx_test, ctx_scaler = seq.standardize(ctx_train_raw, ctx_val_raw, ctx_test_raw)

    route_train_log = pd.DataFrame()
    route_ctx_train = route_ctx_val = route_ctx_test = None
    route_model = None
    hybrid_budgets = seq.resolve_hybrid_budgets(args) if args.candidate_generator == "hybrid" else {"generic": 0, "route": 0, "learned": 0}
    needs_learned_route = args.candidate_generator == "learned_route" or (
        args.candidate_generator == "hybrid" and hybrid_budgets.get("learned", 0) > 0
    )
    route_ctx_scaler: dict[str, Any] = {}
    if needs_learned_route:
        route_blocks = closure.variant_blocks(args.learned_route_context_variant, arrays.x_train)
        route_ctx_train_raw = seq.flatten_blocks(arrays.x_train, route_blocks)
        route_ctx_val_raw = seq.flatten_blocks(arrays.x_val, route_blocks)
        route_ctx_test_raw = seq.flatten_blocks(arrays.x_test, route_blocks)
        if args.learned_route_add_decomposition_context:
            pred_route_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
            pred_route_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
            pred_route_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
            route_ctx_train_raw = np.concatenate([route_ctx_train_raw, seq.decomposition_context_features(pred_route_train, mode_k=args.mode_k)], axis=1)
            route_ctx_val_raw = np.concatenate([route_ctx_val_raw, seq.decomposition_context_features(pred_route_val, mode_k=args.mode_k)], axis=1)
            route_ctx_test_raw = np.concatenate([route_ctx_test_raw, seq.decomposition_context_features(pred_route_test, mode_k=args.mode_k)], axis=1)
        if route_ctx_train_raw.shape[1] > args.max_learned_route_context_features:
            var = np.var(route_ctx_train_raw, axis=0)
            keep = np.argsort(var)[-args.max_learned_route_context_features :]
            route_ctx_train_raw = route_ctx_train_raw[:, keep]
            route_ctx_val_raw = route_ctx_val_raw[:, keep]
            route_ctx_test_raw = route_ctx_test_raw[:, keep]
        route_ctx_train, route_ctx_val, route_ctx_test, route_ctx_scaler = seq.standardize(route_ctx_train_raw, route_ctx_val_raw, route_ctx_test_raw)
        route_model, route_train_log = seq.train_learned_route_generator(
            route_ctx_train,
            route_ctx_val,
            arrays.residual_train,
            arrays.residual_val,
            posterior.mode_soft_train,
            posterior.mode_soft_val,
            args,
            device=device,
        )

    if args.candidate_generator == "learned_route":
        cand_train = seq.generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_train, args, split_name="train", device=device)
        cand_val = seq.generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_val, args, split_name="val", device=device)
        cand_test = seq.generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_test, args, split_name="test", device=device)
    elif args.candidate_generator == "hybrid":
        cand_train = seq.generate_hybrid_candidates(arrays, posterior, student, blocks, route_model, route_ctx_train, args, split_name="train", device=device)
        cand_val = seq.generate_hybrid_candidates(arrays, posterior, student, blocks, route_model, route_ctx_val, args, split_name="val", device=device)
        cand_test = seq.generate_hybrid_candidates(arrays, posterior, student, blocks, route_model, route_ctx_test, args, split_name="test", device=device)
    else:
        cand_train = seq.generate_candidates(arrays, posterior, student, blocks, args, split_name="train", device=device)
        cand_val = seq.generate_candidates(arrays, posterior, student, blocks, args, split_name="val", device=device)
        cand_test = seq.generate_candidates(arrays, posterior, student, blocks, args, split_name="test", device=device)

    cand_train_z, cand_val_z, cand_test_z, cand_scaler = seq.standardize(
        cand_train.features.reshape(-1, cand_train.features.shape[-1]),
        cand_val.features.reshape(-1, cand_val.features.shape[-1]),
        cand_test.features.reshape(-1, cand_test.features.shape[-1]),
    )
    cand_train.features = cand_train_z.reshape(cand_train.features.shape)
    cand_val.features = cand_val_z.reshape(cand_val.features.shape)
    cand_test.features = cand_test_z.reshape(cand_test.features.shape)

    critic, critic_log = seq.train_critic(
        ctx_train,
        ctx_val,
        cand_train,
        cand_val,
        arrays.residual_train,
        arrays.residual_val,
        posterior.mu_train,
        posterior.mode_soft_train,
        args,
        device=device,
        use_context=True,
    )

    q_train = extract_queries(critic, ctx_train, cand_train, arrays.residual_train, args, device=device)
    q_val = extract_queries(critic, ctx_val, cand_val, arrays.residual_val, args, device=device)
    q_test = extract_queries(critic, ctx_test, cand_test, arrays.residual_test, args, device=device)

    rows: list[dict[str, Any]] = []
    rows.extend(endpoint_rows(arrays, q_test.weighted_pred, "route_logits_weighted", args, {"stage": "route_baseline"}))
    rows.extend(endpoint_rows(arrays, q_test.top_pred, "route_logits_top", args, {"stage": "route_baseline"}))
    rows.extend(endpoint_rows(arrays, q_test.query_oracle, "query_oracle", args, {"stage": "route_oracle"}))
    rows.extend(endpoint_rows(arrays, seq.mean_candidate_residual(cand_test), "candidate_mean", args, {"stage": "candidate_control"}))
    for k in args.oracle_k:
        rows.extend(endpoint_rows(arrays, seq.oracle_residual(cand_test, arrays.residual_test, int(k)), f"candidate_oracle@{k}", args, {"stage": "candidate_oracle", "oracle_k": int(k)}))

    temp_grid = args.route_temperatures
    best_route_t, _, _ = tune_temperature(q_val.route_logits, q_val.query_pred, arrays.residual_val, temp_grid)
    route_temp_pred = weighted_residual(q_test.query_pred, softmax_np(q_test.route_logits / best_route_t, axis=1))
    rows.extend(endpoint_rows(arrays, route_temp_pred, "route_logits_temp_tuned", args, {"stage": "route_baseline", "temperature": best_route_t}))

    risk_variants = [
        ("risk_full", True, False, False),
        ("risk_no_context", False, False, False),
        ("risk_shuffled_labels", True, True, False),
        ("risk_shuffled_context", True, False, True),
    ]
    risk_logs: list[pd.DataFrame] = []
    scaler_meta: dict[str, Any] = {"context": finite_json(ctx_scaler), "route_context": finite_json(route_ctx_scaler), "candidate": finite_json(cand_scaler)}
    for name, include_context, shuffled_labels, shuffled_context in risk_variants:
        ctx_train_use, ctx_val_use, ctx_test_use = ctx_train, ctx_val, ctx_test
        if shuffled_context:
            rng = np.random.default_rng(args.seed + 22001)
            ctx_train_use = ctx_train_use[rng.permutation(len(ctx_train_use))]
            ctx_val_use = ctx_val_use[rng.permutation(len(ctx_val_use))]
            ctx_test_use = ctx_test_use[rng.permutation(len(ctx_test_use))]
        feat_train, feat_names = query_sequence_features(
            query_pred=q_train.query_pred,
            base=arrays.base_train,
            route_logits=q_train.route_logits,
            ctx=ctx_train_use,
            horizons=args.horizons,
            include_context=include_context,
            include_query_id=args.risk_include_query_id,
        )
        feat_val, _ = query_sequence_features(
            query_pred=q_val.query_pred,
            base=arrays.base_val,
            route_logits=q_val.route_logits,
            ctx=ctx_val_use,
            horizons=args.horizons,
            include_context=include_context,
            include_query_id=args.risk_include_query_id,
        )
        feat_test, _ = query_sequence_features(
            query_pred=q_test.query_pred,
            base=arrays.base_test,
            route_logits=q_test.route_logits,
            ctx=ctx_test_use,
            horizons=args.horizons,
            include_context=include_context,
            include_query_id=args.risk_include_query_id,
        )
        feat_train, feat_val, feat_test, scaler = standardize_query_features(feat_train, feat_val, feat_test)
        scaler_meta[name] = finite_json(scaler)
        risk_model, risk_log, best_temp = train_risk_model(
            feat_train,
            feat_val,
            q_train,
            q_val,
            arrays.residual_train,
            arrays.residual_val,
            args,
            device=device,
            shuffled_labels=shuffled_labels,
        )
        risk_log = risk_log.assign(variant=name)
        risk_logs.append(risk_log)
        test_risk = predict_risk(risk_model, feat_test, args, device=device)
        risk_weighted = weighted_residual(q_test.query_pred, softmax_np(-test_risk / best_temp, axis=1))
        risk_top = q_test.query_pred[np.arange(len(test_risk)), np.argmin(test_risk, axis=1)]
        err_test = endpoint_errors(q_test.query_pred, arrays.residual_test, args.horizons)
        corr = risk_error_corr(test_risk, err_test)
        rows.extend(endpoint_rows(arrays, risk_weighted, f"{name}_weighted", args, {"stage": "query_risk", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr}))
        rows.extend(endpoint_rows(arrays, risk_top, f"{name}_top", args, {"stage": "query_risk_top", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr}))

    if not args.skip_sklearn_risk:
        sklearn_variants = [
            ("risk_hgbdt_full", "hgbdt", True),
            ("risk_hgbdt_no_context", "hgbdt", False),
            ("risk_ridge_full", "ridge", True),
        ]
        for name, kind, include_context in sklearn_variants:
            try:
                feat_train, feat_names = query_sequence_features(
                    query_pred=q_train.query_pred,
                    base=arrays.base_train,
                    route_logits=q_train.route_logits,
                    ctx=ctx_train,
                    horizons=args.horizons,
                    include_context=include_context,
                    include_query_id=args.risk_include_query_id,
                )
                feat_val, _ = query_sequence_features(
                    query_pred=q_val.query_pred,
                    base=arrays.base_val,
                    route_logits=q_val.route_logits,
                    ctx=ctx_val,
                    horizons=args.horizons,
                    include_context=include_context,
                    include_query_id=args.risk_include_query_id,
                )
                feat_test, _ = query_sequence_features(
                    query_pred=q_test.query_pred,
                    base=arrays.base_test,
                    route_logits=q_test.route_logits,
                    ctx=ctx_test,
                    horizons=args.horizons,
                    include_context=include_context,
                    include_query_id=args.risk_include_query_id,
                )
                feat_train, feat_val, feat_test, scaler = standardize_query_features(feat_train, feat_val, feat_test)
                scaler_meta[name] = finite_json(scaler)
                test_risk, best_temp, diag = fit_predict_sklearn_risk(
                    model_kind=kind,
                    feat_train=feat_train,
                    feat_val=feat_val,
                    feat_test=feat_test,
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    residual_train=arrays.residual_train,
                    residual_val=arrays.residual_val,
                    residual_test=arrays.residual_test,
                    args=args,
                )
                risk_weighted = weighted_residual(q_test.query_pred, softmax_np(-test_risk / best_temp, axis=1))
                risk_top = q_test.query_pred[np.arange(len(test_risk)), np.argmin(test_risk, axis=1)]
                corr = float(diag.get("test_risk_error_corr", risk_error_corr(test_risk, endpoint_errors(q_test.query_pred, arrays.residual_test, args.horizons))))
                rows.extend(endpoint_rows(arrays, risk_weighted, f"{name}_weighted", args, {"stage": "query_risk_sklearn", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr, **diag}))
                rows.extend(endpoint_rows(arrays, risk_top, f"{name}_top", args, {"stage": "query_risk_sklearn_top", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr, **diag}))
            except Exception as exc:
                risk_logs.append(pd.DataFrame([{"variant": name, "error": repr(exc)}]))

        candidate_sklearn_variants = [
            ("candidate_hgbdt_full", "hgbdt", True),
            ("candidate_hgbdt_no_context", "hgbdt", False),
            ("candidate_ridge_full", "ridge", True),
        ]
        for name, kind, include_context in candidate_sklearn_variants:
            try:
                feat_train = append_context_to_set_features(cand_train.features, ctx_train, include_context)
                feat_val = append_context_to_set_features(cand_val.features, ctx_val, include_context)
                feat_test = append_context_to_set_features(cand_test.features, ctx_test, include_context)
                test_risk, best_temp, diag = fit_predict_sklearn_set_risk(
                    model_kind=kind,
                    feat_train=feat_train,
                    feat_val=feat_val,
                    feat_test=feat_test,
                    residual_train=cand_train.residual,
                    residual_val=cand_val.residual,
                    residual_test=cand_test.residual,
                    true_train=arrays.residual_train,
                    true_val=arrays.residual_val,
                    true_test=arrays.residual_test,
                    args=args,
                )
                risk_weighted = weighted_residual(cand_test.residual, softmax_np(-test_risk / best_temp, axis=1))
                risk_top = cand_test.residual[np.arange(len(test_risk)), np.argmin(test_risk, axis=1)]
                corr = float(diag.get("test_risk_error_corr", risk_error_corr(test_risk, endpoint_errors(cand_test.residual, arrays.residual_test, args.horizons))))
                rows.extend(endpoint_rows(arrays, risk_weighted, f"{name}_weighted", args, {"stage": "candidate_risk_sklearn", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr, **diag}))
                rows.extend(endpoint_rows(arrays, risk_top, f"{name}_top", args, {"stage": "candidate_risk_sklearn_top", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr, **diag}))
            except Exception as exc:
                risk_logs.append(pd.DataFrame([{"variant": name, "error": repr(exc)}]))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "query_risk_summary.csv", index=False)
    critic_log.to_csv(args.out_dir / "route_query_critic_train_log.csv", index=False)
    if risk_logs:
        pd.concat(risk_logs, ignore_index=True).to_csv(args.out_dir / "query_risk_train_log.csv", index=False)
    if not route_train_log.empty:
        route_train_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    diagnostics = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "candidate_generator": args.candidate_generator,
                "candidate_k": args.candidate_k,
                "mode_k": args.mode_k,
                "query_count": q_test.query_pred.shape[1],
                "route_temp_best_val": best_route_t,
                "query_feature_dim_full": int(feat_names.__len__()),
                "baseline_h6_rmse": float(summary[(summary["method"].eq("route_logits_weighted")) & (summary["horizon"].eq(6))]["rmse"].iloc[0]) if np.any((summary["method"].eq("route_logits_weighted")) & (summary["horizon"].eq(6))) else np.nan,
            }
        ]
    )
    diagnostics.to_csv(args.out_dir / "query_risk_diagnostics.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diagnostics)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    lines = ["# Query Risk Calibrator Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- critic_context_variant: `{args.critic_context_variant}`")
    lines.append(f"- add_decomposition_context: `{args.add_decomposition_context}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{h}")
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        for _, row in sub.head(16).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, gain={row['gain_vs_base_pct']:.2f}%")
    lines.append("\n## Gate Reading")
    h6 = summary[summary["horizon"].eq(6)] if 6 in args.horizons else summary[summary["horizon"].eq(max(args.horizons))]
    if not h6.empty:
        base = h6[h6["method"].eq("route_logits_weighted")]
        oracle = h6[h6["method"].eq("query_oracle")]
        risks = h6[h6["stage"].astype(str).str.contains("query_risk", na=False)]
        if not base.empty and not oracle.empty and not risks.empty:
            best = risks.sort_values("rmse").iloc[0]
            base_rmse = float(base.iloc[0]["rmse"])
            oracle_rmse = float(oracle.iloc[0]["rmse"])
            gap = max(base_rmse - oracle_rmse, 1e-8)
            closed = (base_rmse - float(best["rmse"])) / gap * 100.0
            lines.append(f"- best risk variant: `{best['method']}` RMSE={best['rmse']:.3f}")
            lines.append(f"- route baseline RMSE={base_rmse:.3f}; query oracle RMSE={oracle_rmse:.3f}; gap closed={closed:.2f}%")
    lines.append("\n## Interpretation")
    lines.append("- Pass if risk_full beats route_logits_weighted and controls, and closes >=20% of the query-oracle gap.")
    lines.append("- If risk_no_context ~= risk_full, query trajectory geometry is doing most of the work and causal context is still weak.")
    lines.append("- If shuffled_labels performs similarly, the risk head is not learning a real selector.")
    (out_dir / "query_risk_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", type=Path, default=audit.DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    parser.add_argument("--dataset", type=str, default="MDCK_Bulk")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seq", type=str, default="1,2,3,4")
    parser.add_argument("--val-seq", type=str, default="5")
    parser.add_argument("--test-seq", type=str, default="6")
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--horizons", type=str, default="1,2,4,6")
    parser.add_argument("--max-train-rows", type=int, default=18000)
    parser.add_argument("--max-val-rows", type=int, default=5000)
    parser.add_argument("--max-test-rows", type=int, default=7000)
    parser.add_argument("--max-features-per-family", type=int, default=160)
    parser.add_argument("--max-all-features", type=int, default=384)
    parser.add_argument("--history-windows", type=str, default="4,8,16,32,64")
    parser.add_argument("--history-flat-lags", type=int, default=32)
    parser.add_argument("--add-history", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--mode-k", type=int, default=12)
    parser.add_argument("--mode-temperature", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--posterior-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=16)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--posterior-beta", type=float, default=1e-3)
    parser.add_argument("--mode-loss-weight", type=float, default=0.30)
    parser.add_argument("--recon-loss-weight", type=float, default=0.20)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--oracle-k", type=str, default="8,16,32")
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument("--generator-variant", type=str, default="no_history")
    parser.add_argument("--candidate-generator", type=str, default="generic", choices=["generic", "route_conditioned", "learned_route", "hybrid"])
    parser.add_argument("--hybrid-generic-k", type=int, default=0)
    parser.add_argument("--hybrid-route-k", type=int, default=0)
    parser.add_argument("--hybrid-learned-k", type=int, default=0)
    parser.add_argument("--route-center-mix", type=float, default=0.55)
    parser.add_argument("--route-noise-scale", type=float, default=0.85)
    parser.add_argument("--route-mode-count", type=int, default=0)
    parser.add_argument("--route-include-centers", action="store_true")
    parser.add_argument("--learned-route-context-variant", type=str, default="no_history")
    parser.add_argument("--learned-route-add-decomposition-context", action="store_true")
    parser.add_argument("--learned-route-hidden", type=int, default=160)
    parser.add_argument("--learned-route-epochs", type=int, default=20)
    parser.add_argument("--learned-route-sample-scale", type=float, default=0.65)
    parser.add_argument("--learned-route-deterministic-first", action="store_true")
    parser.add_argument("--learned-route-recon-weight", type=float, default=0.60)
    parser.add_argument("--learned-route-query-weight", type=float, default=0.30)
    parser.add_argument("--learned-route-mode-weight", type=float, default=0.20)
    parser.add_argument("--learned-route-nll-weight", type=float, default=0.05)
    parser.add_argument("--learned-route-softmin-weight", type=float, default=0.15)
    parser.add_argument("--learned-route-diversity-weight", type=float, default=0.03)
    parser.add_argument("--learned-route-entropy-weight", type=float, default=0.01)
    parser.add_argument("--learned-route-softmin-temp", type=float, default=0.25)
    parser.add_argument("--learned-route-diversity-temp", type=float, default=16.0)
    parser.add_argument("--max-learned-route-context-features", type=int, default=384)
    parser.add_argument("--critic-context-variant", type=str, default="no_history")
    parser.add_argument("--add-decomposition-context", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--candidate-mode-temperature", type=float, default=1.0)
    parser.add_argument("--critic-arch", type=str, default="route_query")
    parser.add_argument("--critic-hidden", type=int, default=160)
    parser.add_argument("--critic-heads", type=int, default=4)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--critic-epochs", type=int, default=20)
    parser.add_argument("--critic-batch-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--correction-scale", type=float, default=1.5)
    parser.add_argument("--listwise-weight", type=float, default=0.35)
    parser.add_argument("--teacher-central", action="store_true")
    parser.add_argument("--oracle-label-weight", type=float, default=0.35)
    parser.add_argument("--teacher-latent-label-weight", type=float, default=0.45)
    parser.add_argument("--teacher-mode-label-weight", type=float, default=0.20)
    parser.add_argument("--teacher-latent-weight", type=float, default=0.20)
    parser.add_argument("--teacher-mode-weight", type=float, default=0.15)
    parser.add_argument("--route-loss-weight", type=float, default=0.20)
    parser.add_argument("--pairwise-rank-weight", type=float, default=0.10)
    parser.add_argument("--query-candidate-weight", type=float, default=0.20)
    parser.add_argument("--query-reg-weight", type=float, default=0.02)
    parser.add_argument("--query-oracle-weight", type=float, default=0.70)
    parser.add_argument("--query-mode-weight", type=float, default=0.30)
    parser.add_argument("--query-route-oracle-weight", type=float, default=0.0)
    parser.add_argument("--query-route-oracle-temperature", type=float, default=0.25)
    parser.add_argument("--teacher-latent-temperature", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--oracle-temperature", type=float, default=0.18)
    parser.add_argument("--synthetic-critic-pretrain", action="store_true")
    parser.add_argument("--synthetic-pretrain-epochs", type=int, default=3)
    parser.add_argument("--synthetic-pretrain-n", type=int, default=4096)
    parser.add_argument("--max-critic-context-features", type=int, default=320)
    parser.add_argument("--skip-candidate-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--risk-hidden", type=int, default=192)
    parser.add_argument("--risk-model", type=str, default="mlp", choices=["mlp", "set_transformer"])
    parser.add_argument("--risk-transformer-heads", type=int, default=4)
    parser.add_argument("--risk-transformer-layers", type=int, default=2)
    parser.add_argument("--risk-epochs", type=int, default=24)
    parser.add_argument("--risk-batch-size", type=int, default=512)
    parser.add_argument("--risk-lr", type=float, default=8e-4)
    parser.add_argument("--risk-weight-decay", type=float, default=1e-4)
    parser.add_argument("--risk-dropout", type=float, default=0.05)
    parser.add_argument("--risk-oracle-temperature", type=float, default=8.0)
    parser.add_argument("--risk-listwise-weight", type=float, default=1.0)
    parser.add_argument("--risk-mse-weight", type=float, default=0.25)
    parser.add_argument("--risk-temperatures", type=str, default="0.15,0.25,0.35,0.5,0.75,1.0,1.5,2.0,3.0")
    parser.add_argument(
        "--risk-label-horizons",
        type=str,
        default="",
        help="Optional comma-separated horizons used only for risk labels; evaluation still uses --horizons.",
    )
    parser.add_argument(
        "--risk-horizon-weights",
        type=str,
        default="",
        help="Optional comma-separated weights matching --risk-label-horizons.",
    )
    parser.add_argument(
        "--risk-temperature-metric",
        type=str,
        default="full_residual",
        choices=["full_residual", "risk_label"],
        help="Validation metric for risk softmax temperature tuning.",
    )
    parser.add_argument("--route-temperatures", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0")
    parser.add_argument("--risk-include-query-id", action="store_true")
    parser.add_argument("--skip-sklearn-risk", action="store_true")
    parser.add_argument("--hgbdt-max-iter", type=int, default=180)
    parser.add_argument("--hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--hgbdt-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--risk-ridge-alpha", type=float, default=10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 4000)
        args.max_val_rows = min(args.max_val_rows, 1500)
        args.max_test_rows = min(args.max_test_rows, 2000)
        args.posterior_epochs = min(args.posterior_epochs, 8)
        args.student_epochs = min(args.student_epochs, 8)
        args.learned_route_epochs = min(args.learned_route_epochs, 6)
        args.critic_epochs = min(args.critic_epochs, 8)
        args.risk_epochs = min(args.risk_epochs, 8)
        args.hgbdt_max_iter = min(args.hgbdt_max_iter, 80)
        args.synthetic_pretrain_epochs = min(args.synthetic_pretrain_epochs, 2)
        args.synthetic_pretrain_n = min(args.synthetic_pretrain_n, 1024)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.max_all_features = min(args.max_all_features, 192)
        args.history_flat_lags = min(args.history_flat_lags, 16)
    run(args)


if __name__ == "__main__":
    main()
