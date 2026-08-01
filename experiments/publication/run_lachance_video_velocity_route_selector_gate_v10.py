#!/usr/bin/env python3
"""Video/velocity route-observability gate and v10 selector-refiner.

This runner tests a narrower question than the previous v8/v9 attempts:

    do causal velocity and tracking-aligned video features make the hidden
    route/candidate choice more observable?

Only signals that pass the cheap route/prototype/residual probes should be
trusted in the heavier selector-refiner.  Futures are used only for training
labels and metrics, never as inference features.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_graph_memory_critic_v4 as v4  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_route_query_pruned_refiner_v9 as v9  # noqa: E402
import run_lachance_route_query_prototype_refiner_v9p as v9p  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_sequence_joint_selector_refiner_v7 as v7  # noqa: E402
import run_lachance_agentic_sequence_refiner_v8 as v8  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "video_velocity_route_selector_gate_v10_2026-07-03"
DEFAULT_SMART_VIDEO_GRID = (
    ROOT
    / "outputs"
    / "video_observability_v2_smart_medium_bulk_seed42_2026-07-01"
    / "smart_video_feature_grid.csv"
)
EPS = 1e-8
FAMILIES = ("context", "video", "axis", "edge", "own", "velocity")


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def to_bool_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.bool, device=device)


def maybe_shuffle(x: np.ndarray, seed: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    rng = np.random.default_rng(int(seed))
    return x[rng.permutation(len(x))]


def standardize_2d(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def z(x: np.ndarray) -> np.ndarray:
        out = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
        return np.clip(np.nan_to_num(out), -8.0, 8.0).astype(np.float32)

    return z(train), z(val), z(test), {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


def family_fraction(weights: np.ndarray, family_ids: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    denom = float(np.maximum(np.sum(weights), EPS))
    for i, name in enumerate(FAMILIES):
        out[f"gate_frac_{name}"] = float(np.sum(weights[family_ids == i]) / denom)
    return out


def safe_topk_accuracy(y: np.ndarray, p: np.ndarray, k: int) -> float:
    if p.shape[1] <= 1:
        return float("nan")
    kk = min(int(k), p.shape[1])
    try:
        return float(top_k_accuracy_score(y, p, k=kk, labels=np.arange(p.shape[1])))
    except Exception:
        order = np.argsort(-p, axis=1)[:, :kk]
        return float(np.mean([int(y[i]) in set(order[i]) for i in range(len(y))]))


def padded_proba(model: LogisticRegression, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.full((len(x), n_classes), 1e-6, dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    out /= np.maximum(out.sum(axis=1, keepdims=True), EPS)
    return out.astype(np.float32)


def select_variance_columns(df: pd.DataFrame, cols: list[str], max_cols: int) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return []
    if len(cols) <= int(max_cols):
        return cols
    x = audit.safe_matrix(df, cols)
    var = np.nan_to_num(np.var(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-int(max_cols) :]
    return [cols[int(i)] for i in keep]


def matrix_from_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return audit.safe_matrix(df, cols).astype(np.float32)


def base_velocity_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dx = df["dx_px"].fillna(0.0).to_numpy(np.float32) if "dx_px" in df else np.zeros(len(df), dtype=np.float32)
    dy = df["dy_px"].fillna(0.0).to_numpy(np.float32) if "dy_px" in df else np.zeros(len(df), dtype=np.float32)
    speed = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    denom = np.maximum(speed, 1e-6)
    feats = [
        dx,
        dy,
        speed,
        dx / denom,
        dy / denom,
        speed * speed,
    ]
    names = ["inst_dx", "inst_dy", "inst_speed", "inst_unit_x", "inst_unit_y", "inst_speed2"]
    if "proposal_norm" in df:
        feats.append(df["proposal_norm"].fillna(0.0).to_numpy(np.float32))
        names.append("proposal_norm")
    if "rc_self_speed" in df:
        feats.append(df["rc_self_speed"].fillna(0.0).to_numpy(np.float32))
        names.append("rc_self_speed")
    return np.stack(feats, axis=1).astype(np.float32), names


def build_velocity_blocks(
    split: audit.seq.SplitData,
    *,
    max_cols: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, list[str]]]:
    train, val, test = split.train, split.val, split.test
    all_cols = list(train.columns)
    inst_tr, inst_names = base_velocity_features(train)
    inst_va, _ = base_velocity_features(val)
    inst_te, _ = base_velocity_features(test)

    def has_any(c: str, keys: tuple[str, ...]) -> bool:
        lc = c.lower()
        return any(k in lc for k in keys)

    multi_cols = [
        c
        for c in all_cols
        if has_any(c, ("lag", "prev", "accel", "jerk", "turn", "cur_prev", "delta"))
        and has_any(c, ("dx", "dy", "u", "v", "mag", "speed", "cos", "sin", "proj", "tangent"))
    ]
    rel_cols = [
        c
        for c in all_cols
        if has_any(c, ("own_minus", "center_mean_disagree", "front_back", "align", "closing", "stretch", "shear", "div", "curl", "coherence"))
    ]
    flow_cols = [
        c
        for c in all_cols
        if (c.startswith("tf_") or c.startswith("obs_flow_"))
        and has_any(c, ("cur", "prev", "lag", "accel", "coherence", "proj", "front_back", "div", "curl", "shear"))
    ]
    multi_cols = select_variance_columns(train, list(dict.fromkeys(multi_cols + flow_cols)), max_cols)
    rel_cols = select_variance_columns(train, rel_cols, max_cols)

    blocks_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "instant": (inst_tr, inst_va, inst_te),
        "multi_lag": (
            np.concatenate([inst_tr, matrix_from_cols(train, multi_cols)], axis=1),
            np.concatenate([inst_va, matrix_from_cols(val, multi_cols)], axis=1),
            np.concatenate([inst_te, matrix_from_cols(test, multi_cols)], axis=1),
        ),
        "relative": (
            np.concatenate([inst_tr, matrix_from_cols(train, rel_cols)], axis=1),
            np.concatenate([inst_va, matrix_from_cols(val, rel_cols)], axis=1),
            np.concatenate([inst_te, matrix_from_cols(test, rel_cols)], axis=1),
        ),
    }
    all_names = list(dict.fromkeys(inst_names + multi_cols + rel_cols))
    blocks_raw["all"] = (
        np.concatenate([inst_tr, matrix_from_cols(train, [c for c in all_names if c not in inst_names])], axis=1),
        np.concatenate([inst_va, matrix_from_cols(val, [c for c in all_names if c not in inst_names])], axis=1),
        np.concatenate([inst_te, matrix_from_cols(test, [c for c in all_names if c not in inst_names])], axis=1),
    )
    names = {
        "instant": inst_names,
        "multi_lag": inst_names + multi_cols,
        "relative": inst_names + rel_cols,
        "all": all_names,
    }
    blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mode, (tr, va, te) in blocks_raw.items():
        blocks[mode] = standardize_2d(tr, va, te)[:3]
    return blocks, names


def route_labels(
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sig_tr = v7.route_signature(arrays.residual_train, args.horizons)
    sig_va = v7.route_signature(arrays.residual_val, args.horizons)
    sig_te = v7.route_signature(arrays.residual_test, args.horizons)
    scaler = StandardScaler()
    ztr = scaler.fit_transform(sig_tr).astype(np.float32)
    zva = scaler.transform(sig_va).astype(np.float32)
    zte = scaler.transform(sig_te).astype(np.float32)
    k = min(int(args.v10_route_k), max(2, len(ztr) // 25))
    km = KMeans(n_clusters=k, n_init=20, random_state=int(args.seed) + 71001)
    ytr = km.fit_predict(ztr).astype(np.int64)
    yva = km.predict(zva).astype(np.int64)
    yte = km.predict(zte).astype(np.int64)
    return {"k": k, "train": ytr, "val": yva, "test": yte, "scaler": scaler, "kmeans": km}


def endpoint_residual_rmse(true: np.ndarray, pred: np.ndarray, horizons: list[int]) -> float:
    errs = []
    for h in horizons:
        p = np.sum(pred[:, : int(h), :], axis=1)
        t = np.sum(true[:, : int(h), :], axis=1)
        errs.append(np.sum((p - t) ** 2, axis=-1))
    return float(np.sqrt(np.mean(np.stack(errs, axis=1))))


def endpoint_direction_cos(true: np.ndarray, pred: np.ndarray, h: int) -> float:
    p = np.sum(pred[:, : int(h), :], axis=1)
    t = np.sum(true[:, : int(h), :], axis=1)
    den = np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1), EPS)
    return float(np.mean(np.sum(p * t, axis=1) / den))


def fit_route_probe(
    *,
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    labels: dict[str, Any],
    arrays: audit.SplitArrays,
    proto_train: seq.CandidatePack,
    proto_val: seq.CandidatePack,
    proto_test: seq.CandidatePack,
    args: argparse.Namespace,
) -> dict[str, Any]:
    xtr, xva, xte, _ = standardize_2d(x_train, x_val, x_test)
    k = int(labels["k"])
    clf = LogisticRegression(
        max_iter=int(args.v10_route_probe_iter),
        C=float(args.v10_route_probe_c),
        class_weight="balanced",
        random_state=int(args.seed) + 72001,
    )
    clf.fit(xtr, labels["train"])
    pte = padded_proba(clf, xte, k)
    pva = padded_proba(clf, xva, k)

    ytr = arrays.residual_train.reshape(len(arrays.residual_train), -1)
    yva = arrays.residual_val.reshape(len(arrays.residual_val), -1)
    yte = arrays.residual_test.reshape(len(arrays.residual_test), -1)
    best_alpha = 10.0
    best_val = float("inf")
    best_model: Ridge | None = None
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        model = Ridge(alpha=float(alpha))
        model.fit(xtr, ytr)
        pred_val = model.predict(xva).reshape(arrays.residual_val.shape).astype(np.float32)
        rmse_val = endpoint_residual_rmse(arrays.residual_val, pred_val, args.horizons)
        if rmse_val < best_val:
            best_val = rmse_val
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None
    pred_test = best_model.predict(xte).reshape(arrays.residual_test.shape).astype(np.float32)

    err_train = qrc.risk_endpoint_errors(proto_train.residual, arrays.residual_train, args)
    err_val = qrc.risk_endpoint_errors(proto_val.residual, arrays.residual_val, args)
    err_test = qrc.risk_endpoint_errors(proto_test.residual, arrays.residual_test, args)
    idx_train = np.argmin(err_train, axis=1).astype(np.int64)
    idx_test = np.argmin(err_test, axis=1).astype(np.int64)
    q = proto_train.residual.shape[1]
    cand_clf = LogisticRegression(
        max_iter=int(args.v10_route_probe_iter),
        C=float(args.v10_route_probe_c),
        class_weight="balanced",
        random_state=int(args.seed) + 73001,
    )
    cand_clf.fit(xtr, idx_train)
    cand_pte = padded_proba(cand_clf, xte, q)

    return {
        "variant": name,
        "feature_dim": int(x_train.shape[1]),
        "route_top1": float(np.mean(np.argmax(pte, axis=1) == labels["test"])),
        "route_top3": safe_topk_accuracy(labels["test"], pte, 3),
        "route_nll": float(log_loss(labels["test"], np.clip(pte, 1e-6, 1.0), labels=np.arange(k))),
        "val_route_top3": safe_topk_accuracy(labels["val"], pva, 3),
        "residual_h6_cos": endpoint_direction_cos(arrays.residual_test, pred_test, max(args.horizons)),
        "residual_endpoint_rmse": endpoint_residual_rmse(arrays.residual_test, pred_test, args.horizons),
        "residual_ridge_alpha": best_alpha,
        "proto_oracle_top1": float(np.mean(np.argmax(cand_pte, axis=1) == idx_test)),
        "proto_oracle_top3": safe_topk_accuracy(idx_test, cand_pte, 3),
        "proto_oracle_nll": float(log_loss(idx_test, np.clip(cand_pte, 1e-6, 1.0), labels=np.arange(q))),
        "proto_val_oracle_mean_err": float(np.mean(np.min(err_val, axis=1))),
        "proto_test_oracle_mean_err": float(np.mean(np.min(err_test, axis=1))),
    }


class RouteQueryVelocityRefinerV10(nn.Module):
    def __init__(
        self,
        *,
        cand_feat_dim: int,
        ctx_dim: int,
        video_dim: int,
        velocity_dim: int,
        edge_dim: int,
        own_dim: int,
        n_lags: int,
        n_neighbours: int,
        max_horizon: int,
        n_axes: int,
        hidden: int,
        heads: int,
        layers: int,
        route_queries: int,
        dropout: float,
        router_topk: int,
        router_disabled: bool,
    ) -> None:
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.router_topk = int(router_topk)
        self.router_disabled = bool(router_disabled)
        self.cand_step_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.cand_feat_proj = nn.Sequential(nn.Linear(max(1, cand_feat_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.ctx_proj = nn.Sequential(nn.Linear(max(1, ctx_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.video_proj = nn.Sequential(nn.Linear(max(1, video_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.velocity_proj = nn.Sequential(nn.Linear(max(1, velocity_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.axis_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_proj = nn.Sequential(nn.Linear(max(1, edge_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.own_proj = nn.Sequential(nn.Linear(max(1, own_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.time_embed = nn.Embedding(max(1, max_horizon), hidden)
        self.axis_embed = nn.Embedding(max(1, n_axes), hidden)
        self.lag_embed = nn.Embedding(max(1, n_lags), hidden)
        self.rank_embed = nn.Embedding(max(1, n_neighbours), hidden)
        self.family_embed = nn.Embedding(len(FAMILIES), hidden)
        self.query_embed = nn.Parameter(torch.randn(route_queries, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cand_step_encoder = nn.TransformerEncoder(copy.deepcopy(layer), num_layers=1)
        self.cand_encoder = nn.TransformerEncoder(copy.deepcopy(layer), num_layers=max(1, layers))
        self.state_encoder = nn.TransformerEncoder(copy.deepcopy(layer), num_layers=max(1, layers))
        self.router = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.query_state_cross = nn.MultiheadAttention(hidden, max(1, heads), dropout=dropout, batch_first=True)
        self.query_candidate_cross = nn.MultiheadAttention(hidden, max(1, heads), dropout=dropout, batch_first=True)
        self.query_encoder = nn.TransformerEncoder(copy.deepcopy(layer), num_layers=max(1, layers))
        self.query_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.cand_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.logvar = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def encode_candidate(self, cand_res: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        n, q, h, _ = cand_res.shape
        step = self.cand_step_proj(cand_res.reshape(n * q, h, 2))
        step = step + self.time_embed(torch.arange(h, device=cand_res.device))[None, :, :]
        step_h = self.cand_step_encoder(step).mean(dim=1).reshape(n, q, -1)
        if cand_feat.shape[-1] == 0:
            cand_feat = torch.zeros((n, q, 1), dtype=cand_res.dtype, device=cand_res.device)
        return self.cand_encoder(step_h + self.cand_feat_proj(cand_feat))

    def encode_state(
        self,
        ctx: torch.Tensor,
        video_feat: torch.Tensor,
        velocity: torch.Tensor,
        axis_steps: torch.Tensor,
        edge_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        own_seq: torch.Tensor,
        center_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = ctx.shape[0]
        tokens: list[torch.Tensor] = []
        valid_parts: list[torch.Tensor] = []
        family_parts: list[torch.Tensor] = []

        def add(tok: torch.Tensor, valid: torch.Tensor, fam: int) -> None:
            tokens.append(tok + self.family_embed(torch.tensor(fam, device=tok.device))[None, None, :])
            valid_parts.append(valid)
            family_parts.append(torch.full(valid.shape, fam, dtype=torch.long, device=tok.device))

        if ctx.shape[-1] == 0:
            ctx = torch.zeros((n, 1), dtype=velocity.dtype, device=velocity.device)
        add(self.ctx_proj(ctx)[:, None, :], torch.ones((n, 1), dtype=torch.bool, device=ctx.device), 0)
        if video_feat.shape[-1] == 0:
            video_feat = torch.zeros((n, 1), dtype=velocity.dtype, device=velocity.device)
        add(self.video_proj(video_feat)[:, None, :], torch.ones((n, 1), dtype=torch.bool, device=ctx.device), 1)
        if velocity.shape[-1] == 0:
            velocity = torch.zeros((n, 1), dtype=ctx.dtype, device=ctx.device)
        add(self.velocity_proj(velocity)[:, None, :], torch.ones((n, 1), dtype=torch.bool, device=ctx.device), 5)

        if axis_steps.shape[1] > 0:
            axis = self.axis_proj(axis_steps.mean(dim=2))
            axis = axis + self.axis_embed(torch.arange(axis.shape[1], device=ctx.device))[None, :, :]
            add(axis, torch.ones((n, axis.shape[1]), dtype=torch.bool, device=ctx.device), 2)

        lags, neigh = edge_seq.shape[1], edge_seq.shape[2]
        edge_h = self.edge_proj(edge_seq.reshape(n, lags * neigh, -1)).reshape(n, lags, neigh, -1)
        edge_h = edge_h + self.lag_embed(torch.arange(lags, device=ctx.device))[None, :, None, :]
        edge_h = edge_h + self.rank_embed(torch.arange(neigh, device=ctx.device))[None, None, :, :]
        add(edge_h.reshape(n, lags * neigh, -1), valid_mask.reshape(n, lags * neigh), 3)

        own = self.own_proj(own_seq.reshape(n, lags, -1))
        own = own + self.lag_embed(torch.arange(lags, device=ctx.device))[None, :, :]
        add(own, center_mask, 4)

        state = torch.cat(tokens, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        family = torch.cat(family_parts, dim=1)
        no_valid = ~torch.any(valid, dim=1)
        if torch.any(no_valid):
            valid = valid.clone()
            valid[no_valid, 0] = True
        padding = ~valid
        raw_gate = self.router(state).squeeze(-1).masked_fill(padding, -1e4)
        if self.router_disabled:
            raw_gate = torch.where(padding, torch.full_like(raw_gate, -1e4), torch.zeros_like(raw_gate))
        gate = torch.softmax(raw_gate, dim=1)
        if (not self.router_disabled) and self.router_topk > 0 and self.router_topk < gate.shape[1]:
            _, idx = torch.topk(gate, k=int(self.router_topk), dim=1)
            mask = torch.zeros_like(gate, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            raw_gate = raw_gate.masked_fill(~mask, -1e4)
            gate = torch.softmax(raw_gate, dim=1)
            padding = padding | (~mask)
        encoded = self.state_encoder(state * gate[:, :, None], src_key_padding_mask=padding)
        return encoded, padding, gate, family

    def forward(
        self,
        cand_res: torch.Tensor,
        cand_feat: torch.Tensor,
        ctx: torch.Tensor,
        video_feat: torch.Tensor,
        velocity: torch.Tensor,
        axis_steps: torch.Tensor,
        edge_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        own_seq: torch.Tensor,
        center_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cand_h = self.encode_candidate(cand_res, cand_feat)
        state, state_padding, gate, family = self.encode_state(
            ctx, video_feat, velocity, axis_steps, edge_seq, valid_mask, own_seq, center_mask
        )
        n = cand_h.shape[0]
        query = self.query_embed[None, :, :].repeat(n, 1, 1)
        q_state, _ = self.query_state_cross(query, state, state, key_padding_mask=state_padding, need_weights=False)
        query = self.query_encoder(query + q_state)
        q_cand, attn = self.query_candidate_cross(query, cand_h, cand_h, need_weights=True, average_attn_weights=False)
        query = self.query_encoder(query + q_cand)
        q_score = self.query_score(query).squeeze(-1)
        attn_mean = torch.mean(attn, dim=1)
        q_prob = torch.softmax(-q_score, dim=1)
        cand_from_query = torch.sum(q_prob[:, :, None] * attn_mean, dim=1)
        cand_direct = torch.softmax(-self.cand_score(cand_h).squeeze(-1), dim=1)
        cand_prob = 0.5 * cand_from_query + 0.5 * cand_direct
        risk = -torch.log(torch.clamp(cand_prob, min=1e-8))
        return {"risk": risk, "gate": gate, "family": family, "logvar": self.logvar(torch.sum(q_prob[:, :, None] * query, dim=1))}


def prepare_pack_v10(
    *,
    arrays: audit.SplitArrays,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cand_test: seq.CandidatePack,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    residual_hints: dict[str, Any],
    edge_memory: dict[str, np.ndarray],
    velocity_blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    use_context: bool = True,
    use_video: bool = True,
    shuffle_video: bool = False,
    use_edge: bool = True,
    shuffle_edge: bool = False,
    use_axis: bool = True,
    shuffle_axis: bool = False,
    use_velocity: bool = True,
    shuffle_velocity: bool = False,
    velocity_mode: str = "all",
    router_disabled: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pack, meta = v9.prepare_pack(
        arrays=arrays,
        cand_train=cand_train,
        cand_val=cand_val,
        cand_test=cand_test,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        component_axes=component_axes,
        residual_hints=residual_hints,
        edge_memory=edge_memory,
        args=args,
        use_context=use_context,
        use_video=use_video,
        shuffle_video=shuffle_video,
        use_edge=use_edge,
        shuffle_edge=shuffle_edge,
        use_axis=use_axis,
        shuffle_axis=shuffle_axis,
        router_disabled=router_disabled,
    )
    mode = velocity_mode if velocity_mode in velocity_blocks else "all"
    vel_train, vel_val, vel_test = velocity_blocks[mode]
    if not use_velocity:
        vel_train = np.zeros_like(vel_train)
        vel_val = np.zeros_like(vel_val)
        vel_test = np.zeros_like(vel_test)
    if shuffle_velocity:
        vel_train = maybe_shuffle(vel_train, args.seed + 76001, True)
        vel_val = maybe_shuffle(vel_val, args.seed + 76002, True)
        vel_test = maybe_shuffle(vel_test, args.seed + 76003, True)
    pack["velocity_train"] = vel_train.astype(np.float32)
    pack["velocity_val"] = vel_val.astype(np.float32)
    pack["velocity_test"] = vel_test.astype(np.float32)
    # Use a compact video feature vector as video token, not only residual steps.
    for split_name in ("train", "val", "test"):
        res = pack[f"video_{split_name}"].reshape(len(pack[f"video_{split_name}"]), -1)
        pack[f"video_{split_name}"] = res.astype(np.float32)
    meta.update(
        {
            "use_velocity": bool(use_velocity),
            "shuffle_velocity": bool(shuffle_velocity),
            "velocity_mode": mode,
            "velocity_dim": int(vel_train.shape[1]),
        }
    )
    return pack, meta


def forward_v10(model: RouteQueryVelocityRefinerV10, pack: dict[str, np.ndarray], split_name: str, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        to_tensor(pack[f"cand_{split_name}"][idx], device),
        to_tensor(pack[f"feat_{split_name}"][idx], device),
        to_tensor(pack[f"ctx_{split_name}"][idx], device),
        to_tensor(pack[f"video_{split_name}"][idx], device),
        to_tensor(pack[f"velocity_{split_name}"][idx], device),
        to_tensor(pack[f"axis_{split_name}"][idx], device),
        to_tensor(pack[f"edge_{split_name}"][idx], device),
        to_bool_tensor(pack[f"valid_{split_name}"][idx], device),
        to_tensor(pack[f"own_{split_name}"][idx], device),
        to_bool_tensor(pack[f"center_{split_name}"][idx], device),
    )


def train_v10(
    pack: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    *,
    shuffled_labels: bool = False,
    router_disabled: bool = False,
) -> tuple[RouteQueryVelocityRefinerV10, pd.DataFrame, float]:
    model = RouteQueryVelocityRefinerV10(
        cand_feat_dim=pack["feat_train"].shape[-1],
        ctx_dim=pack["ctx_train"].shape[-1],
        video_dim=pack["video_train"].shape[-1],
        velocity_dim=pack["velocity_train"].shape[-1],
        edge_dim=pack["edge_train"].shape[-1],
        own_dim=pack["own_train"].shape[-1],
        n_lags=pack["edge_train"].shape[1],
        n_neighbours=pack["edge_train"].shape[2],
        max_horizon=args.max_horizon,
        n_axes=pack["axis_train"].shape[1],
        hidden=args.v10_hidden,
        heads=args.v10_heads,
        layers=args.v10_layers,
        route_queries=args.v10_route_queries,
        dropout=args.v10_dropout,
        router_topk=args.v10_router_topk,
        router_disabled=router_disabled,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.v10_lr, weight_decay=args.v10_weight_decay)
    err_train = pack["err_train"]
    target_end_train = pack["target_end_train"]
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 77001)
        perm = rng.permutation(len(err_train))
        err_train = err_train[perm]
        target_end_train = target_end_train[perm]
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack["cand_train"])
    for epoch in range(int(args.v10_epochs)):
        model.train()
        losses: list[float] = []
        for idx in closure.batches(n, args.v10_batch_size, args.seed + 78000 + epoch):
            out = forward_v10(model, pack, "train", idx, device)
            risk = out["risk"]
            err = to_tensor(err_train[idx], device)
            soft = qrc.soft_labels_from_error(err, args.v10_label_temperature)
            listwise = -torch.mean(torch.sum(soft * F.log_softmax(-risk, dim=1), dim=1))
            target_log = torch.log1p(err)
            target_z = (target_log - target_log.mean(dim=1, keepdim=True)) / torch.clamp(target_log.std(dim=1, keepdim=True), min=1e-3)
            pred_z = (risk - risk.mean(dim=1, keepdim=True)) / torch.clamp(risk.std(dim=1, keepdim=True), min=1e-3)
            rank = F.smooth_l1_loss(pred_z, target_z)
            weights = torch.softmax(-risk / max(float(args.v10_train_temperature), 1e-6), dim=1)
            reg = v8.mixture_endpoint_loss(
                to_tensor(pack["cand_train"][idx], device),
                weights,
                to_tensor(target_end_train[idx], device),
                args.horizons,
            )
            gate = out["gate"]
            gate_entropy = -torch.mean(torch.sum(gate * torch.log(torch.clamp(gate, min=1e-8)), dim=1))
            cand_entropy = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, min=1e-8)), dim=1))
            loss = (
                args.v10_listwise_weight * listwise
                + args.v10_rank_weight * rank
                + args.v10_reg_weight * reg
                - args.v10_candidate_entropy_weight * cand_entropy
                + args.v10_gate_entropy_weight * gate_entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.v10_clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == args.v10_epochs - 1 or epoch % max(1, args.v10_epochs // 5) == 0:
            risk_val = predict_v10(model, pack, "val", args, device)
            m, t, val_rmse = v8.tune_topm(risk_val, pack["cand_val"], pack["target_steps_val"], args)
            corr = qrc.risk_error_corr(risk_val, pack["err_val"])
            rows.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(losses)),
                    "val_selector_rmse": float(val_rmse),
                    "val_risk_error_corr": float(corr),
                    "top_m": int(m),
                    "temperature": float(t),
                }
            )
            if val_rmse < best_val:
                best_val = float(val_rmse)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_val


def predict_v10(model: RouteQueryVelocityRefinerV10, pack: dict[str, np.ndarray], split_name: str, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for idx in closure.batches(len(pack[f"cand_{split_name}"]), args.v10_batch_size, args.seed + 79001, shuffle=False):
            out.append(forward_v10(model, pack, split_name, idx, device)["risk"].detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def gate_diagnostics(model: RouteQueryVelocityRefinerV10, pack: dict[str, np.ndarray], split_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    weights: list[np.ndarray] = []
    families: list[np.ndarray] = []
    entropies: list[float] = []
    with torch.no_grad():
        for idx in closure.batches(len(pack[f"cand_{split_name}"]), args.v10_batch_size, args.seed + 80001, shuffle=False):
            out = forward_v10(model, pack, split_name, idx, device)
            gate = out["gate"].detach().cpu().numpy()
            fam = out["family"].detach().cpu().numpy()
            weights.append(gate.reshape(-1))
            families.append(fam.reshape(-1))
            entropies.append(float(np.mean(-np.sum(gate * np.log(np.maximum(gate, EPS)), axis=1))))
    diag = family_fraction(np.concatenate(weights), np.concatenate(families))
    diag["gate_entropy"] = float(np.mean(entropies)) if entropies else float("nan")
    return diag


def add_metric_rows(rows: list[dict[str, Any]], arrays: audit.SplitArrays, pred: np.ndarray, args: argparse.Namespace, label: str, extra: dict[str, Any]) -> None:
    rows.extend(audit.endpoint_metrics(steps_true=arrays.steps_test, base=arrays.base_test, residual_pred=pred, horizons=args.horizons, label=label, extra=extra))


def run_v10_variant(
    *,
    name: str,
    rows: list[dict[str, Any]],
    logs: list[pd.DataFrame],
    diagnostics: list[dict[str, Any]],
    arrays: audit.SplitArrays,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cand_test: seq.CandidatePack,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    residual_hints: dict[str, Any],
    edge_memory: dict[str, np.ndarray],
    velocity_blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
    use_context: bool = True,
    use_video: bool = True,
    shuffle_video: bool = False,
    use_edge: bool = True,
    shuffle_edge: bool = False,
    use_axis: bool = True,
    shuffle_axis: bool = False,
    use_velocity: bool = True,
    shuffle_velocity: bool = False,
    velocity_mode: str = "all",
    router_disabled: bool = False,
    shuffled_labels: bool = False,
    meta_out: dict[str, Any] | None = None,
) -> None:
    pack, meta = prepare_pack_v10(
        arrays=arrays,
        cand_train=cand_train,
        cand_val=cand_val,
        cand_test=cand_test,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        component_axes=component_axes,
        residual_hints=residual_hints,
        edge_memory=edge_memory,
        velocity_blocks=velocity_blocks,
        args=args,
        use_context=use_context,
        use_video=use_video,
        shuffle_video=shuffle_video,
        use_edge=use_edge,
        shuffle_edge=shuffle_edge,
        use_axis=use_axis,
        shuffle_axis=shuffle_axis,
        use_velocity=use_velocity,
        shuffle_velocity=shuffle_velocity,
        velocity_mode=velocity_mode,
        router_disabled=router_disabled,
    )
    if meta_out is not None:
        meta_out[name] = finite_json(meta)
    model, log, val_rmse = train_v10(pack, args, device, shuffled_labels=shuffled_labels, router_disabled=router_disabled)
    logs.append(log.assign(variant=name))
    risk_val = predict_v10(model, pack, "val", args, device)
    risk_test = predict_v10(model, pack, "test", args, device)
    best_m, best_t, tuned_val = v8.tune_topm(risk_val, cand_val.residual, arrays.residual_val, args)
    pred_sparse, _ = v8.sparse_topm_prediction(risk_test, cand_test.residual, best_m, best_t)
    pred_dense = qrc.weighted_residual(cand_test.residual, qrc.softmax_np(-risk_test / best_t, axis=1))
    pred_top = cand_test.residual[np.arange(len(risk_test)), np.argmin(risk_test, axis=1)]
    err_test = qrc.risk_endpoint_errors(cand_test.residual, arrays.residual_test, args)
    corr = qrc.risk_error_corr(risk_test, err_test)
    gdiag = gate_diagnostics(model, pack, "test", args, device)
    extra = {
        "variant": name,
        "risk_error_corr": corr,
        "val_selector_rmse": tuned_val,
        "best_top_m": best_m,
        "temperature": best_t,
        **gdiag,
    }
    add_metric_rows(rows, arrays, pred_sparse, args, f"{name}_sparse_topm", {"stage": "v10_sparse_topm", **extra})
    add_metric_rows(rows, arrays, pred_dense, args, f"{name}_dense", {"stage": "v10_dense", **extra})
    add_metric_rows(rows, arrays, pred_top, args, f"{name}_top1", {"stage": "v10_top1", **extra, "best_top_m": 1})
    diagnostics.append(
        {
            "variant": name,
            "risk_error_corr": float(corr),
            "val_selector_rmse": float(val_rmse),
            "tuned_val_selector_rmse": float(tuned_val),
            "best_top_m": int(best_m),
            "temperature": float(best_t),
            "top1_mean_index": float(np.mean(np.argmin(risk_test, axis=1))),
            **gdiag,
        }
    )


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    gate: pd.DataFrame,
    summary: pd.DataFrame,
    diag: pd.DataFrame,
    video_probe: pd.DataFrame,
) -> None:
    lines = ["# Video/Velocity Route Selector Gate v10 Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- primary KPI: selector/route gate before final RMSE")
    lines.append(f"- historical h6 reference: component-aware `17.13`, v9p `17.16`, prototype oracle `9.78`")
    lines.append("")
    if not gate.empty:
        lines.append("## Route/Feature Gate")
        cols = [
            "variant",
            "feature_dim",
            "route_top3",
            "route_nll",
            "residual_endpoint_rmse",
            "residual_h6_cos",
            "proto_oracle_top3",
        ]
        lines.append(gate[[c for c in cols if c in gate.columns]].sort_values("route_top3", ascending=False).to_markdown(index=False))
    if not summary.empty:
        lines.append("\n## Selector Metrics")
        for h in args.horizons:
            lines.append(f"### h{int(h)}")
            sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
            for _, row in sub.head(40).iterrows():
                lines.append(
                    f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, "
                    f"stage={row.get('stage', '')}"
                )
    if not diag.empty:
        lines.append("\n## V10 Diagnostics")
        for _, row in diag.sort_values("risk_error_corr", ascending=False).iterrows():
            lines.append(
                f"- `{row['variant']}`: corr={row['risk_error_corr']:.3f}, "
                f"val={row['val_selector_rmse']:.3f}, topM={int(row['best_top_m'])}, "
                f"T={row['temperature']:.3f}, vel={row.get('gate_frac_velocity', float('nan')):.3f}, "
                f"video={row.get('gate_frac_video', float('nan')):.3f}, edge={row.get('gate_frac_edge', float('nan')):.3f}, "
                f"axis={row.get('gate_frac_axis', float('nan')):.3f}"
            )
    if not video_probe.empty:
        lines.append("\n## Video Residual Teacher Probe")
        lines.append(video_probe.to_markdown(index=False))
    lines.append("\n## Decision Logic")
    lines.append("- Hard pass: h6 beats `17.13` by >=3% and real video/velocity controls degrade logically.")
    lines.append("- Soft pass: selector improves 1-2% and route gate shows real > shuffled controls.")
    lines.append("- Fail: route gate is clean but selector remains near `17.1-17.4`, or video equals shuffled.")
    (out_dir / "video_velocity_route_selector_gate_v10_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)

    arrays, split = audit.prepare_data(args)
    extra_feature_meta = rpr.attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    velocity_blocks, velocity_names = build_velocity_blocks(split, max_cols=args.v10_velocity_max_cols)

    edge_memory, edge_memory_meta = v4.load_edge_memory_cache(
        args.edge_sequence_cache,
        split,
        max_lags=args.v10_edge_max_lags,
        max_neighbours=args.v10_edge_max_neighbours,
        min_found_frac=args.v10_min_edge_found_frac,
    )

    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_meta = rpr.prepare_context(args, arrays, posterior, student, blocks, device)
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v10 requires component axes")

    cand_args = argparse.Namespace(**vars(args))
    cand_args.candidate_generator = args.prototype_source_candidate_generator
    cand_args.candidate_k = int(args.prototype_source_candidate_k)
    cand_args.oracle_k = [min(int(k), cand_args.candidate_k) for k in args.oracle_k]
    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(
        cand_args, arrays, posterior, student, blocks, device
    )
    src_train = rpr.generate_candidates_for_split(cand_args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    src_val = rpr.generate_candidates_for_split(cand_args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    src_test = rpr.generate_candidates_for_split(cand_args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    proto_train, _ = v9p.make_prototype_pack(
        src_train, arrays, split_name="train", method=args.prototype_method, k=args.prototype_k, args=args, seed=args.seed + 8001
    )
    proto_val, _ = v9p.make_prototype_pack(
        src_val, arrays, split_name="val", method=args.prototype_method, k=args.prototype_k, args=args, seed=args.seed + 9001
    )
    proto_test, _ = v9p.make_prototype_pack(
        src_test, arrays, split_name="test", method=args.prototype_method, k=args.prototype_k, args=args, seed=args.seed + 10001
    )

    video_train, video_val, video_test, video_meta = v7.select_context_block(arrays, args.v10_video_block, args.v10_video_max_features)
    residual_hints, video_probe = v7.fit_video_residual_hints(
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        edge_train=video_train,
        edge_val=video_val,
        edge_test=video_test,
        args=args,
    )

    labels = route_labels(arrays, args)
    zero_video = np.zeros_like(video_train), np.zeros_like(video_val), np.zeros_like(video_test)
    sh_video = (
        maybe_shuffle(video_train, args.seed + 81001, True),
        maybe_shuffle(video_val, args.seed + 81002, True),
        maybe_shuffle(video_test, args.seed + 81003, True),
    )
    sh_vel = tuple(maybe_shuffle(v, args.seed + 82000 + i, True) for i, v in enumerate(velocity_blocks["all"]))
    probe_variants: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
        ("context_only", (ctx_train, ctx_val, ctx_test)),
        ("velocity_instant", velocity_blocks["instant"]),
        ("velocity_multi_lag", velocity_blocks["multi_lag"]),
        ("velocity_relative", velocity_blocks["relative"]),
        ("velocity_all", velocity_blocks["all"]),
        ("velocity_all_shuffled", sh_vel),
        ("context_plus_velocity_all", tuple(np.concatenate([c, v], axis=1) for c, v in zip((ctx_train, ctx_val, ctx_test), velocity_blocks["all"]))),
        ("video_only_real", (video_train, video_val, video_test)),
        ("video_only_shuffled", sh_video),
        ("video_only_zero", zero_video),
        ("context_plus_video_real", tuple(np.concatenate([c, v], axis=1) for c, v in zip((ctx_train, ctx_val, ctx_test), (video_train, video_val, video_test)))),
        ("context_plus_video_shuffled", tuple(np.concatenate([c, v], axis=1) for c, v in zip((ctx_train, ctx_val, ctx_test), sh_video))),
        (
            "context_plus_velocity_video",
            tuple(np.concatenate([c, v, sv], axis=1) for c, v, sv in zip((ctx_train, ctx_val, ctx_test), velocity_blocks["all"], (video_train, video_val, video_test))),
        ),
    ]
    gate_rows = [
        fit_route_probe(
            name=name,
            x_train=mat[0],
            x_val=mat[1],
            x_test=mat[2],
            labels=labels,
            arrays=arrays,
            proto_train=proto_train,
            proto_val=proto_val,
            proto_test=proto_test,
            args=args,
        )
        for name, mat in probe_variants
    ]
    gate_df = pd.DataFrame(gate_rows)
    gate_df.insert(0, "seed", int(args.seed))
    gate_df.insert(0, "dataset", str(args.dataset))

    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "extra_feature": finite_json(extra_feature_meta),
        "context": finite_json(ctx_meta),
        "edge_memory": finite_json(edge_memory_meta),
        "video_block": finite_json(video_meta),
        "velocity_names": velocity_names,
        "video_residual_teacher": finite_json(residual_hints.get("info", {})),
        "prototype": {
            "method": args.prototype_method,
            "k": int(args.prototype_k),
            "source_candidate_generator": args.prototype_source_candidate_generator,
            "source_candidate_k": int(args.prototype_source_candidate_k),
        },
    }

    add_metric_rows(rows, arrays, seq.mean_candidate_residual(src_test), args, "source_candidate_mean", {"stage": "source_candidate_control"})
    for k in args.oracle_k:
        kk = min(int(k), src_test.residual.shape[1])
        add_metric_rows(rows, arrays, seq.oracle_residual(src_test, arrays.residual_test, kk), args, f"source_candidate_oracle@{kk}", {"stage": "source_candidate_oracle", "oracle_k": kk})
    add_metric_rows(rows, arrays, seq.mean_candidate_residual(proto_test), args, f"{args.prototype_method}{args.prototype_k}_mean", {"stage": "prototype_mean"})
    add_metric_rows(rows, arrays, qrc.query_oracle_residual(proto_test.residual, arrays.residual_test, args.horizons), args, f"{args.prototype_method}{args.prototype_k}_oracle", {"stage": "prototype_oracle"})

    variants: list[tuple[str, dict[str, Any]]] = [
        ("v10_velocity_video_full", {}),
        ("v10_velocity_only", {"use_video": False, "use_context": False}),
        ("v10_video_only", {"use_velocity": False, "use_context": False}),
        ("v10_no_velocity", {"use_velocity": False}),
        ("v10_velocity_instant", {"velocity_mode": "instant"}),
        ("v10_velocity_multi_lag", {"velocity_mode": "multi_lag"}),
        ("v10_shuffled_velocity", {"shuffle_velocity": True}),
        ("v10_no_video", {"use_video": False}),
        ("v10_shuffled_video", {"shuffle_video": True}),
        ("v10_no_edge", {"use_edge": False}),
        ("v10_no_axes", {"use_axis": False}),
        ("v10_edge_axes_velocity_only", {"use_context": False, "use_video": False}),
        ("v10_router_disabled", {"router_disabled": True}),
        ("v10_shuffled_labels", {"shuffled_labels": True}),
    ]
    requested = {s.strip() for s in str(args.v10_variant_list).split(",") if s.strip()}
    for name, kwargs in variants:
        if requested and name not in requested:
            continue
        run_v10_variant(
            name=name,
            rows=rows,
            logs=logs,
            diagnostics=diagnostics,
            arrays=arrays,
            cand_train=proto_train,
            cand_val=proto_val,
            cand_test=proto_test,
            ctx_train=ctx_train,
            ctx_val=ctx_val,
            ctx_test=ctx_test,
            component_axes=component_axes,
            residual_hints=residual_hints,
            edge_memory=edge_memory,
            velocity_blocks=velocity_blocks,
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

    gate_df.to_csv(args.out_dir / "video_velocity_route_gate_summary.csv", index=False)
    summary.to_csv(args.out_dir / "video_velocity_route_selector_v10_summary.csv", index=False)
    diag.to_csv(args.out_dir / "video_velocity_route_selector_v10_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "video_velocity_route_selector_v10_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "video_velocity_route_selector_v10_route_train_log.csv", index=False)
    video_probe.to_csv(args.out_dir / "video_velocity_route_selector_v10_video_teacher_probe.csv", index=False)
    component_axes.probe.to_csv(args.out_dir / "video_velocity_route_selector_v10_component_axis_probe.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, gate_df, summary, diag, video_probe)
    print(json.dumps({"out_dir": str(args.out_dir), "gate_rows": len(gate_df), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(out_dir=DEFAULT_OUT)
    parser.add_argument("--edge-sequence-cache", type=Path, required=True)
    parser.add_argument("--extra-feature-grid", type=Path, default=DEFAULT_SMART_VIDEO_GRID)
    parser.add_argument("--extra-feature-prefixes", type=str, default="sv_")
    parser.add_argument("--extra-feature-block-name", type=str, default="smart_video")
    parser.add_argument("--extra-feature-max-cols", type=int, default=256)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--prototype-source-candidate-generator", type=str, default="generic")
    parser.add_argument("--prototype-source-candidate-k", type=int, default=32)
    parser.add_argument("--prototype-method", type=str, default="fps_shape")
    parser.add_argument("--prototype-k", type=int, default=16)
    parser.add_argument("--component-axis-blocks", type=str, default="self,flow,morphology,boundary,crowding,raw_context,smart_video,all_context")
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
    parser.add_argument("--video-residual-model", type=str, default="hgbdt", choices=["ridge", "hgbdt"])
    parser.add_argument("--video-residual-include-context", action="store_true")
    parser.add_argument("--video-residual-hgbdt-iter", type=int, default=120)
    parser.add_argument("--video-residual-hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--video-residual-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--video-residual-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v10-video-block", type=str, default="smart_video")
    parser.add_argument("--v10-video-max-features", type=int, default=192)
    parser.add_argument("--v10-velocity-max-cols", type=int, default=96)
    parser.add_argument("--v10-route-k", type=int, default=12)
    parser.add_argument("--v10-route-probe-iter", type=int, default=350)
    parser.add_argument("--v10-route-probe-c", type=float, default=0.35)
    parser.add_argument("--v10-hidden", type=int, default=192)
    parser.add_argument("--v10-heads", type=int, default=4)
    parser.add_argument("--v10-layers", type=int, default=2)
    parser.add_argument("--v10-route-queries", type=int, default=12)
    parser.add_argument("--v10-router-topk", type=int, default=24)
    parser.add_argument("--v10-dropout", type=float, default=0.05)
    parser.add_argument("--v10-epochs", type=int, default=10)
    parser.add_argument("--v10-batch-size", type=int, default=192)
    parser.add_argument("--v10-lr", type=float, default=7e-4)
    parser.add_argument("--v10-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v10-label-temperature", type=float, default=8.0)
    parser.add_argument("--v10-train-temperature", type=float, default=0.75)
    parser.add_argument("--v10-listwise-weight", type=float, default=1.0)
    parser.add_argument("--v10-rank-weight", type=float, default=0.5)
    parser.add_argument("--v10-reg-weight", type=float, default=0.25)
    parser.add_argument("--v10-candidate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--v10-gate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--v10-clip-grad", type=float, default=5.0)
    parser.add_argument("--v10-topm", type=str, default="1,2,4,8,16")
    parser.add_argument("--v10-temperatures", type=str, default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--v10-edge-max-lags", type=int, default=0)
    parser.add_argument("--v10-edge-max-neighbours", type=int, default=0)
    parser.add_argument("--v10-min-edge-found-frac", type=float, default=0.98)
    parser.add_argument("--v10-variant-list", type=str, default="")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    # Compatibility aliases for v8/v9 helper functions.
    args.v8_topm = args.v10_topm
    args.v8_temperatures = [float(x) for x in str(args.v10_temperatures).split(",") if x.strip()]
    args.v8_batch_size = args.v10_batch_size
    args.v8_video_edge_block = args.v10_video_block
    args.v8_video_edge_max_features = args.v10_video_max_features
    args.v8_lr = args.v10_lr
    args.v8_weight_decay = args.v10_weight_decay
    args.v8_epochs = args.v10_epochs
    args.v8_label_temperature = args.v10_label_temperature
    args.v8_train_temperature = args.v10_train_temperature
    args.v8_clip_grad = args.v10_clip_grad
    args.v8_rank_weight = args.v10_rank_weight
    args.v8_reg_weight = args.v10_reg_weight
    args.v8_entropy_weight = args.v10_candidate_entropy_weight
    args.v9_topm = args.v10_topm
    args.v9_temperatures = [float(x) for x in str(args.v10_temperatures).split(",") if x.strip()]
    args.v9_batch_size = args.v10_batch_size
    args.v9_hidden = args.v10_hidden
    args.v9_heads = args.v10_heads
    args.v9_layers = args.v10_layers
    args.v9_route_queries = args.v10_route_queries
    args.v9_router_topk = args.v10_router_topk
    args.v9_dropout = args.v10_dropout
    args.v9_epochs = args.v10_epochs
    args.v9_lr = args.v10_lr
    args.v9_weight_decay = args.v10_weight_decay
    args.v9_label_temperature = args.v10_label_temperature
    args.v9_train_temperature = args.v10_train_temperature
    args.v9_listwise_weight = args.v10_listwise_weight
    args.v9_rank_weight = args.v10_rank_weight
    args.v9_reg_weight = args.v10_reg_weight
    args.v9_candidate_entropy_weight = args.v10_candidate_entropy_weight
    args.v9_gate_entropy_weight = args.v10_gate_entropy_weight
    args.v9_clip_grad = args.v10_clip_grad
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1200)
        args.max_val_rows = min(args.max_val_rows, 400)
        args.max_test_rows = min(args.max_test_rows, 600)
        args.posterior_epochs = min(args.posterior_epochs, 3)
        args.student_epochs = min(args.student_epochs, 3)
        args.v10_epochs = min(args.v10_epochs, 3)
        args.component_axis_epochs = min(args.component_axis_epochs, 3)
        args.video_residual_hgbdt_iter = min(args.video_residual_hgbdt_iter, 40)
        args.prototype_source_candidate_k = min(args.prototype_source_candidate_k, 16)
        args.prototype_k = min(args.prototype_k, 8)
        args.oracle_k = [8, args.prototype_source_candidate_k]
        args.v10_hidden = min(args.v10_hidden, 128)
        args.v10_layers = min(args.v10_layers, 1)
        args.v9_hidden = args.v10_hidden
        args.v9_layers = args.v10_layers
    if args.v10_edge_max_lags == 0:
        args.v10_edge_max_lags = 4
    if args.v10_edge_max_neighbours == 0:
        args.v10_edge_max_neighbours = 8
    run(args)


if __name__ == "__main__":
    main()
