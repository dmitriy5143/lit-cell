#!/usr/bin/env python3
"""Agentic Sequence/Graph Trajectory Critic-Refiner v8 for LaChance cells.

This runner is a compact "cell-agent" version of the selector/refiner:

    candidate trajectory actions
    + clean context / decomposition axes / video residual teacher / neighbour memory
    -> query-centric cross-attention critic
    -> sparse top-M trajectory mixture

The target future is used only for training labels/losses and metrics.  All
inference features are causal.
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
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_sequence_joint_selector_refiner_v7 as v7  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "agentic_sequence_refiner_v8_2026-07-02"
EPS = 1e-8


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


def standardize_2d(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tr, va, te, scaler = seq.standardize(train, val, test)
    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32), finite_json(scaler)


def standardize_3d(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return v7.standardize_3d(train, val, test)


def route_logits(cand: seq.CandidatePack) -> np.ndarray:
    if cand.logprob is not None and cand.logprob.size:
        return np.nan_to_num(cand.logprob.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.zeros(cand.residual.shape[:2], dtype=np.float32)


def build_candidate_features(
    cand: seq.CandidatePack,
    base: np.ndarray,
    ctx: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
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


def endpoints_from_residual(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    return np.stack([np.sum(residual[:, : int(h), :], axis=1) for h in horizons], axis=1).astype(np.float32)


def candidate_endpoints(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    return np.stack([np.sum(residual[:, :, : int(h), :], axis=2) for h in horizons], axis=2).astype(np.float32)


def sparse_topm_prediction(risk: np.ndarray, residual: np.ndarray, m: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    m = min(int(m), residual.shape[1])
    idx = np.argsort(risk, axis=1)[:, :m]
    rr = np.take_along_axis(residual, idx[:, :, None, None], axis=1)
    rs = np.take_along_axis(risk, idx, axis=1)
    w = qrc.softmax_np(-rs / max(float(temperature), 1e-6), axis=1)
    pred = np.sum(w[:, :, None, None] * rr, axis=1).astype(np.float32)
    return pred, idx


def tune_topm(risk: np.ndarray, residual: np.ndarray, true: np.ndarray, args: argparse.Namespace) -> tuple[int, float, float]:
    best_m = parse_ints(args.v8_topm)[0]
    best_t = args.v8_temperatures[0]
    best = float("inf")
    for m in parse_ints(args.v8_topm):
        for t in args.v8_temperatures:
            pred, _ = sparse_topm_prediction(risk, residual, int(m), float(t))
            score = qrc.residual_endpoint_rmse(true, pred, args.horizons)
            if score < best:
                best = float(score)
                best_m = int(m)
                best_t = float(t)
    return best_m, best_t, best


class AgenticCriticV8(nn.Module):
    def __init__(
        self,
        *,
        cand_feat_dim: int,
        ctx_dim: int,
        edge_dim: int,
        own_dim: int,
        n_lags: int,
        n_neighbours: int,
        max_horizon: int,
        n_axes: int,
        hidden: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.hidden = int(hidden)
        self.n_lags = int(n_lags)
        self.n_neighbours = int(n_neighbours)

        self.cand_step_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.cand_feat_proj = nn.Sequential(nn.Linear(max(1, cand_feat_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.ctx_proj = nn.Sequential(nn.Linear(max(1, ctx_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.video_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.axis_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_proj = nn.Sequential(nn.Linear(max(1, edge_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.own_proj = nn.Sequential(nn.Linear(max(1, own_dim), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.time_embed = nn.Embedding(max(1, max_horizon), hidden)
        self.axis_embed = nn.Embedding(max(1, n_axes), hidden)
        self.lag_embed = nn.Embedding(max(1, n_lags), hidden)
        self.rank_embed = nn.Embedding(max(1, n_neighbours), hidden)
        self.type_embed = nn.Embedding(5, hidden)
        self.type_gate_logits = nn.Parameter(torch.tensor([1.5, 1.5, 0.5, -1.0, -1.0], dtype=torch.float32))

        step_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cand_step_encoder = nn.TransformerEncoder(step_layer, num_layers=1)
        self.state_encoder = nn.TransformerEncoder(copy.deepcopy(step_layer), num_layers=max(1, int(layers)))
        self.cross = nn.MultiheadAttention(hidden, max(1, int(heads)), dropout=dropout, batch_first=True)
        self.candidate_encoder = nn.TransformerEncoder(copy.deepcopy(step_layer), num_layers=max(1, int(layers)))
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.logvar = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def encode_candidate(self, cand_res: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        n, q, h, _ = cand_res.shape
        step = self.cand_step_proj(cand_res.reshape(n * q, h, 2))
        step = step + self.time_embed(torch.arange(h, device=cand_res.device))[None, :, :]
        step_h = self.cand_step_encoder(step).mean(dim=1).reshape(n, q, -1)
        if cand_feat.shape[-1] == 0:
            cand_feat = torch.zeros((n, q, 1), dtype=cand_res.dtype, device=cand_res.device)
        return step_h + self.cand_feat_proj(cand_feat)

    def encode_state(
        self,
        ctx: torch.Tensor,
        video_res: torch.Tensor,
        axis_steps: torch.Tensor,
        edge_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        own_seq: torch.Tensor,
        center_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = ctx.shape[0]
        gates = torch.sigmoid(self.type_gate_logits)
        if ctx.shape[-1] == 0:
            ctx = torch.zeros((n, 1), dtype=video_res.dtype, device=video_res.device)
        tokens = [(self.ctx_proj(ctx)[:, None, :] + self.type_embed(torch.tensor(0, device=ctx.device))[None, None, :]) * gates[0]]
        valid_parts = [torch.ones((n, 1), dtype=torch.bool, device=ctx.device)]

        h = video_res.shape[1]
        video = self.video_proj(video_res) + self.time_embed(torch.arange(h, device=ctx.device))[None, :, :]
        tokens.append((video + self.type_embed(torch.tensor(1, device=ctx.device))[None, None, :]) * gates[1])
        valid_parts.append(torch.ones((n, h), dtype=torch.bool, device=ctx.device))

        if axis_steps.shape[1] > 0:
            axis = self.axis_proj(axis_steps.mean(dim=2))
            axis = axis + self.axis_embed(torch.arange(axis.shape[1], device=ctx.device))[None, :, :]
            tokens.append((axis + self.type_embed(torch.tensor(2, device=ctx.device))[None, None, :]) * gates[2])
            valid_parts.append(torch.ones((n, axis.shape[1]), dtype=torch.bool, device=ctx.device))

        lags = edge_seq.shape[1]
        neigh = edge_seq.shape[2]
        edge_h = self.edge_proj(edge_seq.reshape(n, lags * neigh, -1)).reshape(n, lags, neigh, -1)
        edge_h = edge_h + self.lag_embed(torch.arange(lags, device=ctx.device))[None, :, None, :]
        edge_h = edge_h + self.rank_embed(torch.arange(neigh, device=ctx.device))[None, None, :, :]
        tokens.append((edge_h.reshape(n, lags * neigh, -1) + self.type_embed(torch.tensor(3, device=ctx.device))[None, None, :]) * gates[3])
        valid_parts.append(valid_mask.reshape(n, lags * neigh))

        own = self.own_proj(own_seq.reshape(n, lags, -1))
        own = own + self.lag_embed(torch.arange(lags, device=ctx.device))[None, :, :]
        tokens.append((own + self.type_embed(torch.tensor(4, device=ctx.device))[None, None, :]) * gates[4])
        valid_parts.append(center_mask)

        state = torch.cat(tokens, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        no_valid = ~torch.any(valid, dim=1)
        if torch.any(no_valid):
            valid = valid.clone()
            valid[no_valid, 0] = True
        padding = ~valid
        return self.state_encoder(state, src_key_padding_mask=padding), padding

    def forward(
        self,
        cand_res: torch.Tensor,
        cand_feat: torch.Tensor,
        ctx: torch.Tensor,
        video_res: torch.Tensor,
        axis_steps: torch.Tensor,
        edge_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        own_seq: torch.Tensor,
        center_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cand_h = self.encode_candidate(cand_res, cand_feat)
        state, padding = self.encode_state(ctx, video_res, axis_steps, edge_seq, valid_mask, own_seq, center_mask)
        cross, _ = self.cross(cand_h, state, state, key_padding_mask=padding, need_weights=False)
        cand_h = self.candidate_encoder(cand_h + cross)
        return {
            "risk": self.score(cand_h).squeeze(-1),
            "logvar": self.logvar(cand_h),
        }


def prepare_pack(
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
    args: argparse.Namespace,
    use_context: bool = True,
    use_video: bool = True,
    shuffle_video: bool = False,
    use_edge: bool = True,
    shuffle_edge: bool = False,
    use_axis: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feat_train = build_candidate_features(cand_train, arrays.base_train, ctx_train, args)
    feat_val = build_candidate_features(cand_val, arrays.base_val, ctx_val, args)
    feat_test = build_candidate_features(cand_test, arrays.base_test, ctx_test, args)
    feat_train, feat_val, feat_test, feat_scaler = standardize_3d(feat_train, feat_val, feat_test)

    ctx_tr, ctx_va, ctx_te = ctx_train, ctx_val, ctx_test
    if not use_context:
        ctx_tr, ctx_va, ctx_te = np.zeros_like(ctx_train), np.zeros_like(ctx_val), np.zeros_like(ctx_test)
    ctx_tr, ctx_va, ctx_te, ctx_scaler = standardize_2d(ctx_tr, ctx_va, ctx_te)

    video_train = residual_hints["pred_train"].astype(np.float32)
    video_val = residual_hints["pred_val"].astype(np.float32)
    video_test = residual_hints["pred_test"].astype(np.float32)
    if not use_video:
        video_train, video_val, video_test = np.zeros_like(video_train), np.zeros_like(video_val), np.zeros_like(video_test)
    if shuffle_video:
        video_train = maybe_shuffle(video_train, args.seed + 51001, True)
        video_val = maybe_shuffle(video_val, args.seed + 51002, True)
        video_test = maybe_shuffle(video_test, args.seed + 51003, True)

    axis_train = component_axes.train.astype(np.float32)
    axis_val = component_axes.val.astype(np.float32)
    axis_test = component_axes.test.astype(np.float32)
    if not use_axis:
        axis_train = np.zeros((len(axis_train), 0, args.max_horizon, 2), dtype=np.float32)
        axis_val = np.zeros((len(axis_val), 0, args.max_horizon, 2), dtype=np.float32)
        axis_test = np.zeros((len(axis_test), 0, args.max_horizon, 2), dtype=np.float32)

    edge_train, edge_val, edge_test, own_train, own_val, own_test, edge_scaler = v4.standardize_edge_memory(
        edge_memory["edge_train"],
        edge_memory["edge_val"],
        edge_memory["edge_test"],
        edge_memory["own_train"],
        edge_memory["own_val"],
        edge_memory["own_test"],
        edge_memory["valid_train"],
        edge_memory["center_train"],
    )
    valid_train = edge_memory["valid_train"].astype(bool)
    valid_val = edge_memory["valid_val"].astype(bool)
    valid_test = edge_memory["valid_test"].astype(bool)
    center_train = edge_memory["center_train"].astype(bool)
    center_val = edge_memory["center_val"].astype(bool)
    center_test = edge_memory["center_test"].astype(bool)
    if not use_edge:
        edge_train = np.zeros_like(edge_train)
        edge_val = np.zeros_like(edge_val)
        edge_test = np.zeros_like(edge_test)
        own_train = np.zeros_like(own_train)
        own_val = np.zeros_like(own_val)
        own_test = np.zeros_like(own_test)
        valid_train[:] = False
        valid_val[:] = False
        valid_test[:] = False
        center_train[:] = False
        center_val[:] = False
        center_test[:] = False
    if shuffle_edge:
        edge_train = maybe_shuffle(edge_train, args.seed + 52001, True)
        edge_val = maybe_shuffle(edge_val, args.seed + 52002, True)
        edge_test = maybe_shuffle(edge_test, args.seed + 52003, True)
        own_train = maybe_shuffle(own_train, args.seed + 52101, True)
        own_val = maybe_shuffle(own_val, args.seed + 52102, True)
        own_test = maybe_shuffle(own_test, args.seed + 52103, True)

    target_err_train = qrc.risk_endpoint_errors(cand_train.residual, arrays.residual_train, args)
    target_err_val = qrc.risk_endpoint_errors(cand_val.residual, arrays.residual_val, args)
    target_err_test = qrc.risk_endpoint_errors(cand_test.residual, arrays.residual_test, args)
    pack = {
        "cand_train": cand_train.residual.astype(np.float32),
        "cand_val": cand_val.residual.astype(np.float32),
        "cand_test": cand_test.residual.astype(np.float32),
        "feat_train": feat_train,
        "feat_val": feat_val,
        "feat_test": feat_test,
        "ctx_train": ctx_tr,
        "ctx_val": ctx_va,
        "ctx_test": ctx_te,
        "video_train": video_train.astype(np.float32),
        "video_val": video_val.astype(np.float32),
        "video_test": video_test.astype(np.float32),
        "axis_train": axis_train,
        "axis_val": axis_val,
        "axis_test": axis_test,
        "edge_train": edge_train,
        "edge_val": edge_val,
        "edge_test": edge_test,
        "own_train": own_train,
        "own_val": own_val,
        "own_test": own_test,
        "valid_train": valid_train,
        "valid_val": valid_val,
        "valid_test": valid_test,
        "center_train": center_train,
        "center_val": center_val,
        "center_test": center_test,
        "err_train": target_err_train,
        "err_val": target_err_val,
        "err_test": target_err_test,
        "target_end_train": endpoints_from_residual(arrays.residual_train, args.horizons),
        "target_end_val": endpoints_from_residual(arrays.residual_val, args.horizons),
        "target_end_test": endpoints_from_residual(arrays.residual_test, args.horizons),
        "target_steps_train": arrays.residual_train.astype(np.float32),
        "target_steps_val": arrays.residual_val.astype(np.float32),
        "target_steps_test": arrays.residual_test.astype(np.float32),
    }
    meta = {
        "candidate_feature_scaler": finite_json(feat_scaler),
        "context_scaler": finite_json(ctx_scaler),
        "edge_scaler": finite_json(edge_scaler),
        "use_context": bool(use_context),
        "use_video": bool(use_video),
        "shuffle_video": bool(shuffle_video),
        "use_edge": bool(use_edge),
        "shuffle_edge": bool(shuffle_edge),
        "use_axis": bool(use_axis),
        "feature_dim": int(feat_train.shape[-1]),
        "ctx_dim": int(ctx_tr.shape[-1]),
        "axis_count": int(axis_train.shape[1]),
        "edge_shape": list(edge_train.shape),
    }
    return pack, meta


def forward_v8(model: AgenticCriticV8, pack: dict[str, np.ndarray], split_name: str, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        to_tensor(pack[f"cand_{split_name}"][idx], device),
        to_tensor(pack[f"feat_{split_name}"][idx], device),
        to_tensor(pack[f"ctx_{split_name}"][idx], device),
        to_tensor(pack[f"video_{split_name}"][idx], device),
        to_tensor(pack[f"axis_{split_name}"][idx], device),
        to_tensor(pack[f"edge_{split_name}"][idx], device),
        to_bool_tensor(pack[f"valid_{split_name}"][idx], device),
        to_tensor(pack[f"own_{split_name}"][idx], device),
        to_bool_tensor(pack[f"center_{split_name}"][idx], device),
    )


def mixture_endpoint_loss(residual: torch.Tensor, weights: torch.Tensor, target_end: torch.Tensor, horizons: list[int]) -> torch.Tensor:
    pred_steps = torch.sum(weights[:, :, None, None] * residual, dim=1)
    parts = [torch.sum(pred_steps[:, : int(h), :], dim=1) for h in horizons]
    pred_end = torch.stack(parts, dim=1).contiguous()
    return F.smooth_l1_loss(pred_end, target_end.contiguous())


def train_v8(
    pack: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    *,
    shuffled_labels: bool = False,
) -> tuple[AgenticCriticV8, pd.DataFrame, float]:
    model = AgenticCriticV8(
        cand_feat_dim=pack["feat_train"].shape[-1],
        ctx_dim=pack["ctx_train"].shape[-1],
        edge_dim=pack["edge_train"].shape[-1],
        own_dim=pack["own_train"].shape[-1],
        n_lags=pack["edge_train"].shape[1],
        n_neighbours=pack["edge_train"].shape[2],
        max_horizon=args.max_horizon,
        n_axes=pack["axis_train"].shape[1],
        hidden=args.v8_hidden,
        heads=args.v8_heads,
        layers=args.v8_layers,
        dropout=args.v8_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.v8_lr, weight_decay=args.v8_weight_decay)
    err_train = pack["err_train"]
    target_end_train = pack["target_end_train"]
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 54001)
        perm = rng.permutation(len(err_train))
        err_train = err_train[perm]
        target_end_train = target_end_train[perm]
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack["cand_train"])
    for epoch in range(int(args.v8_epochs)):
        model.train()
        losses: list[float] = []
        for idx in closure.batches(n, args.v8_batch_size, args.seed + 55000 + epoch):
            out = forward_v8(model, pack, "train", idx, device)
            risk = out["risk"]
            err = to_tensor(err_train[idx], device)
            soft = qrc.soft_labels_from_error(err, args.v8_label_temperature)
            listwise = -torch.mean(torch.sum(soft * F.log_softmax(-risk, dim=1), dim=1))
            target_log = torch.log1p(err)
            target_z = (target_log - target_log.mean(dim=1, keepdim=True)) / torch.clamp(target_log.std(dim=1, keepdim=True), min=1e-3)
            pred_z = (risk - risk.mean(dim=1, keepdim=True)) / torch.clamp(risk.std(dim=1, keepdim=True), min=1e-3)
            rank = F.smooth_l1_loss(pred_z, target_z)
            weights = torch.softmax(-risk / max(float(args.v8_train_temperature), 1e-6), dim=1)
            reg = mixture_endpoint_loss(
                to_tensor(pack["cand_train"][idx], device),
                weights,
                to_tensor(target_end_train[idx], device),
                args.horizons,
            )
            entropy = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, min=1e-8)), dim=1))
            loss = (
                args.v8_listwise_weight * listwise
                + args.v8_rank_weight * rank
                + args.v8_reg_weight * reg
                - args.v8_entropy_weight * entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.v8_clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == args.v8_epochs - 1 or epoch % max(1, args.v8_epochs // 4) == 0:
            risk_val = predict_v8(model, pack, "val", args, device)
            m, t, val_rmse = tune_topm(risk_val, pack["cand_val"], pack["target_steps_val"], args)
            corr = qrc.risk_error_corr(risk_val, pack["err_val"])
            rows.append({"epoch": int(epoch), "train_loss": float(np.mean(losses)), "val_selector_rmse": float(val_rmse), "val_risk_error_corr": float(corr), "top_m": int(m), "temperature": float(t)})
            if val_rmse < best_val:
                best_val = float(val_rmse)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_val


def predict_v8(model: AgenticCriticV8, pack: dict[str, np.ndarray], split_name: str, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    n = len(pack[f"cand_{split_name}"])
    with torch.no_grad():
        for idx in closure.batches(n, args.v8_batch_size, args.seed + 56001, shuffle=False):
            out.append(forward_v8(model, pack, split_name, idx, device)["risk"].detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def add_metric_rows(rows: list[dict[str, Any]], arrays: audit.SplitArrays, pred: np.ndarray, args: argparse.Namespace, label: str, extra: dict[str, Any]) -> None:
    rows.extend(audit.endpoint_metrics(steps_true=arrays.steps_test, base=arrays.base_test, residual_pred=pred, horizons=args.horizons, label=label, extra=extra))


def run_variant(
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
    args: argparse.Namespace,
    device: torch.device,
    use_context: bool = True,
    use_video: bool = True,
    shuffle_video: bool = False,
    use_edge: bool = True,
    shuffle_edge: bool = False,
    use_axis: bool = True,
    shuffled_labels: bool = False,
    meta_out: dict[str, Any] | None = None,
) -> None:
    pack, meta = prepare_pack(
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
    )
    if meta_out is not None:
        meta_out[name] = finite_json(meta)
    model, log, val_rmse = train_v8(pack, args, device, shuffled_labels=shuffled_labels)
    logs.append(log.assign(variant=name))
    risk_val = predict_v8(model, pack, "val", args, device)
    risk_test = predict_v8(model, pack, "test", args, device)
    best_m, best_t, tuned_val = tune_topm(risk_val, cand_val.residual, arrays.residual_val, args)
    pred_sparse, top_idx = sparse_topm_prediction(risk_test, cand_test.residual, best_m, best_t)
    pred_dense = qrc.weighted_residual(cand_test.residual, qrc.softmax_np(-risk_test / best_t, axis=1))
    pred_top = cand_test.residual[np.arange(len(risk_test)), np.argmin(risk_test, axis=1)]
    corr = qrc.risk_error_corr(risk_test, qrc.risk_endpoint_errors(cand_test.residual, arrays.residual_test, args))
    extra = {"variant": name, "risk_error_corr": corr, "val_selector_rmse": tuned_val, "best_top_m": best_m, "temperature": best_t}
    add_metric_rows(rows, arrays, pred_sparse, args, f"{name}_sparse_topm", {"stage": "agentic_v8_sparse_topm", **extra})
    add_metric_rows(rows, arrays, pred_dense, args, f"{name}_dense", {"stage": "agentic_v8_dense", **extra})
    add_metric_rows(rows, arrays, pred_top, args, f"{name}_top1", {"stage": "agentic_v8_top1", **extra, "best_top_m": 1})
    diagnostics.append({
        "variant": name,
        "risk_error_corr": float(corr),
        "val_selector_rmse": float(val_rmse),
        "tuned_val_selector_rmse": float(tuned_val),
        "best_top_m": int(best_m),
        "temperature": float(best_t),
        "top1_mean_index": float(np.mean(np.argmin(risk_test, axis=1))),
    })


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    extra_feature_meta = rpr.attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    edge_memory, edge_memory_meta = v4.load_edge_memory_cache(
        args.edge_sequence_cache,
        split,
        max_lags=args.v8_edge_max_lags,
        max_neighbours=args.v8_edge_max_neighbours,
        min_found_frac=args.v8_min_edge_found_frac,
    )

    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_meta = rpr.prepare_context(args, arrays, posterior, student, blocks, device)
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v8 requires component axes")
    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand_train = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    edge_ctx_train, edge_ctx_val, edge_ctx_test, edge_ctx_meta = v7.select_context_block(arrays, args.v8_video_edge_block, args.v8_video_edge_max_features)
    residual_hints, video_probe = v7.fit_video_residual_hints(
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        edge_train=edge_ctx_train,
        edge_val=edge_ctx_val,
        edge_test=edge_ctx_test,
        args=args,
    )

    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "extra_feature": finite_json(extra_feature_meta),
        "context": finite_json(ctx_meta),
        "edge_memory": finite_json(edge_memory_meta),
        "video_edge": finite_json(edge_ctx_meta),
        "video_residual_teacher": finite_json(residual_hints.get("info", {})),
    }

    add_metric_rows(rows, arrays, seq.mean_candidate_residual(cand_test), args, "candidate_mean", {"stage": "candidate_control"})
    for k in args.oracle_k:
        add_metric_rows(rows, arrays, seq.oracle_residual(cand_test, arrays.residual_test, int(k)), args, f"candidate_oracle@{k}", {"stage": "candidate_oracle", "oracle_k": int(k)})

    run_variant(
        name="v8_full",
        rows=rows,
        logs=logs,
        diagnostics=diagnostics,
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
        device=device,
        meta_out=meta,
    )
    if args.include_controls:
        requested = {s.strip() for s in str(args.v8_variant_list).split(",") if s.strip()}
        for name, kwargs in [
            ("v8_no_video_teacher", {"use_video": False}),
            ("v8_shuffled_video_teacher", {"shuffle_video": True}),
            ("v8_no_edge_memory", {"use_edge": False}),
            ("v8_shuffled_edge_memory", {"shuffle_edge": True}),
            ("v8_no_decomposition_axes", {"use_axis": False}),
            ("v8_no_context", {"use_context": False}),
            # Lean combined variants. The main run showed that single-token
            # ablations can hide a noisy interaction between context/video/axes.
            # These keep the sequence critic agentic, but force it to rely on
            # candidate actions plus the most stable causal memory packets.
            ("v8_edge_only", {"use_context": False, "use_video": False, "use_axis": False}),
            ("v8_edge_video", {"use_context": False, "use_axis": False}),
            ("v8_edge_axes", {"use_context": False, "use_video": False}),
            ("v8_no_context_no_video", {"use_context": False, "use_video": False}),
            ("v8_no_context_no_axes", {"use_context": False, "use_axis": False}),
            ("v8_shuffled_labels", {"shuffled_labels": True}),
        ]:
            if requested and name not in requested:
                continue
            run_variant(
                name=name,
                rows=rows,
                logs=logs,
                diagnostics=diagnostics,
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
    summary.to_csv(args.out_dir / "agentic_v8_summary.csv", index=False)
    diag.to_csv(args.out_dir / "agentic_v8_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "agentic_v8_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "agentic_v8_learned_route_train_log.csv", index=False)
    video_probe.to_csv(args.out_dir / "agentic_v8_video_teacher_probe.csv", index=False)
    component_axes.probe.to_csv(args.out_dir / "agentic_v8_component_axis_probe.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag, video_probe)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, video_probe: pd.DataFrame) -> None:
    lines = ["# Agentic Sequence/Graph Critic-Refiner v8 Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- edge_sequence_cache: `{args.edge_sequence_cache}`")
    lines.append(f"- video_residual_model: `{args.video_residual_model}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{int(h)}")
        sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
        for _, row in sub.head(32).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, stage={row.get('stage', '')}")
    if not diag.empty:
        lines.append("\n## Diagnostics")
        for _, row in diag.sort_values("risk_error_corr", ascending=False).iterrows():
            lines.append(
                f"- `{row['variant']}`: corr={row['risk_error_corr']:.3f}, "
                f"val={row['val_selector_rmse']:.3f}, topM={int(row['best_top_m'])}, T={row['temperature']:.3f}"
            )
    if not video_probe.empty:
        lines.append("\n## Video Teacher Probe")
        lines.append(video_probe.to_markdown(index=False))
    lines.append("\n## Decision")
    lines.append("- Hard pass if h6 beats 18.8 and full beats no-video/no-edge/no-axis/shuffled controls.")
    lines.append("- Soft pass if h6 beats the prior v7 video-residual baseline 19.627 with consistent controls.")
    (out_dir / "agentic_v8_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(out_dir=DEFAULT_OUT)
    parser.add_argument("--edge-sequence-cache", type=Path, required=True)
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="ef_")
    parser.add_argument("--extra-feature-block-name", type=str, default="explicit_edge")
    parser.add_argument("--extra-feature-max-cols", type=int, default=128)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--component-axis-blocks", type=str, default="self,flow,explicit_edge,all_context")
    parser.add_argument("--component-include-student-axis", action="store_true")
    parser.add_argument("--component-axis-model", type=str, default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--component-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--component-axis-max-features", type=int, default=192)
    parser.add_argument("--component-axis-hidden", type=int, default=128)
    parser.add_argument("--component-axis-epochs", type=int, default=12)
    parser.add_argument("--component-axis-lr", type=float, default=8e-4)
    parser.add_argument("--component-axis-weight-decay", type=float, default=1e-4)
    parser.add_argument("--component-axis-dropout", type=float, default=0.05)
    parser.add_argument("--component-attention-temperature", type=float, default=6.0)
    parser.add_argument("--v8-video-edge-block", type=str, default="explicit_edge")
    parser.add_argument("--v8-video-edge-max-features", type=int, default=128)
    parser.add_argument("--video-residual-model", type=str, default="hgbdt", choices=["ridge", "hgbdt"])
    parser.add_argument("--video-residual-include-context", action="store_true")
    parser.add_argument("--video-residual-hgbdt-iter", type=int, default=120)
    parser.add_argument("--video-residual-hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--video-residual-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--video-residual-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v8-hidden", type=int, default=192)
    parser.add_argument("--v8-heads", type=int, default=4)
    parser.add_argument("--v8-layers", type=int, default=2)
    parser.add_argument("--v8-dropout", type=float, default=0.05)
    parser.add_argument("--v8-epochs", type=int, default=8)
    parser.add_argument("--v8-batch-size", type=int, default=256)
    parser.add_argument("--v8-lr", type=float, default=7e-4)
    parser.add_argument("--v8-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v8-label-temperature", type=float, default=8.0)
    parser.add_argument("--v8-train-temperature", type=float, default=0.75)
    parser.add_argument("--v8-listwise-weight", type=float, default=1.0)
    parser.add_argument("--v8-rank-weight", type=float, default=0.25)
    parser.add_argument("--v8-reg-weight", type=float, default=0.50)
    parser.add_argument("--v8-entropy-weight", type=float, default=0.002)
    parser.add_argument("--v8-clip-grad", type=float, default=5.0)
    parser.add_argument("--v8-topm", type=str, default="1,2,4,8,16")
    parser.add_argument("--v8-temperatures", type=str, default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--v8-edge-max-lags", type=int, default=0)
    parser.add_argument("--v8-edge-max-neighbours", type=int, default=0)
    parser.add_argument("--v8-min-edge-found-frac", type=float, default=0.98)
    parser.add_argument("--v8-variant-list", type=str, default="")
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    args.v8_temperatures = [float(x) for x in str(args.v8_temperatures).split(",") if x.strip()]
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1200)
        args.max_val_rows = min(args.max_val_rows, 400)
        args.max_test_rows = min(args.max_test_rows, 600)
        args.posterior_epochs = min(args.posterior_epochs, 3)
        args.student_epochs = min(args.student_epochs, 3)
        args.learned_route_epochs = min(args.learned_route_epochs, 3)
        args.v8_epochs = min(args.v8_epochs, 3)
        args.candidate_k = min(args.candidate_k, 32)
        args.oracle_k = [8, min(16, args.candidate_k), args.candidate_k]
        args.max_all_features = min(args.max_all_features, 192)
        args.max_critic_context_features = min(args.max_critic_context_features, 192)
        args.v8_hidden = min(args.v8_hidden, 128)
        args.v8_layers = min(args.v8_layers, 1)
    run(args)


if __name__ == "__main__":
    main()
