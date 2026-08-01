#!/usr/bin/env python3
"""Route-Query Pruned Component Refiner v9 for LaChance trajectories.

This runner tests the current strongest hypothesis:

    candidate trajectories
    + edge memory / decomposition axes / optional video teacher
    -> learned token router-pruner
    -> route-query decoder
    -> sparse top-M / min-risk refined trajectory

Targets/futures are used only for training labels and metrics.  Inference
features are causal.
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
import run_lachance_agentic_sequence_refiner_v8 as v8  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_query_pruned_refiner_v9_2026-07-02"
EPS = 1e-8
FAMILIES = ("context", "video", "axis", "edge", "own")


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


def one_hot_family_mask(enabled: dict[str, bool]) -> dict[str, bool]:
    return {k: bool(enabled.get(k, False)) for k in FAMILIES}


def family_fraction(weights: np.ndarray, family_ids: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    denom = float(np.maximum(np.sum(weights), EPS))
    for i, name in enumerate(FAMILIES):
        out[f"gate_frac_{name}"] = float(np.sum(weights[family_ids == i]) / denom)
    return out


class RouteQueryPrunedRefinerV9(nn.Module):
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
        route_queries: int,
        dropout: float,
        router_topk: int,
        router_disabled: bool,
    ) -> None:
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.hidden = int(hidden)
        self.n_lags = int(n_lags)
        self.n_neighbours = int(n_neighbours)
        self.route_queries = int(route_queries)
        self.router_topk = int(router_topk)
        self.router_disabled = bool(router_disabled)

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
        video_res: torch.Tensor,
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
            ctx = torch.zeros((n, 1), dtype=video_res.dtype, device=video_res.device)
        add(self.ctx_proj(ctx)[:, None, :], torch.ones((n, 1), dtype=torch.bool, device=ctx.device), 0)

        h = video_res.shape[1]
        video = self.video_proj(video_res) + self.time_embed(torch.arange(h, device=ctx.device))[None, :, :]
        add(video, torch.ones((n, h), dtype=torch.bool, device=ctx.device), 1)

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
        raw_gate = self.router(state).squeeze(-1)
        raw_gate = raw_gate.masked_fill(padding, -1e4)
        if self.router_disabled:
            gate = torch.softmax(raw_gate * 0.0 + torch.where(padding, torch.full_like(raw_gate, -1e4), torch.zeros_like(raw_gate)), dim=1)
        else:
            gate = torch.softmax(raw_gate, dim=1)
            if self.router_topk > 0 and self.router_topk < gate.shape[1]:
                vals, idx = torch.topk(gate, k=int(self.router_topk), dim=1)
                mask = torch.zeros_like(gate, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                raw_gate = raw_gate.masked_fill(~mask, -1e4)
                gate = torch.softmax(raw_gate, dim=1)
                padding = padding | (~mask)
        gated_state = state * gate[:, :, None]
        encoded = self.state_encoder(gated_state, src_key_padding_mask=padding)
        return encoded, padding, gate, family

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
        state, state_padding, gate, family = self.encode_state(ctx, video_res, axis_steps, edge_seq, valid_mask, own_seq, center_mask)
        n = cand_h.shape[0]
        query = self.query_embed[None, :, :].repeat(n, 1, 1)
        q_state, _ = self.query_state_cross(query, state, state, key_padding_mask=state_padding, need_weights=False)
        query = self.query_encoder(query + q_state)
        q_cand, attn = self.query_candidate_cross(query, cand_h, cand_h, need_weights=True, average_attn_weights=False)
        query = self.query_encoder(query + q_cand)
        q_score = self.query_score(query).squeeze(-1)
        attn_mean = torch.mean(attn, dim=1)  # n, route_queries, candidates
        q_prob = torch.softmax(-q_score, dim=1)
        cand_from_query = torch.sum(q_prob[:, :, None] * attn_mean, dim=1)
        cand_direct = torch.softmax(-self.cand_score(cand_h).squeeze(-1), dim=1)
        cand_prob = 0.5 * cand_from_query + 0.5 * cand_direct
        risk = -torch.log(torch.clamp(cand_prob, min=1e-8))
        return {
            "risk": risk,
            "query_score": q_score,
            "gate": gate,
            "family": family,
            "logvar": self.logvar(torch.sum(q_prob[:, :, None] * query, dim=1)),
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
    shuffle_axis: bool = False,
    router_disabled: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pack, meta = v8.prepare_pack(
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
    axis_feature_names: list[str] = []
    if use_axis and component_axes.train.shape[1] > 0:
        axtr, axis_feature_names = rpr.component_route_features(
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
            axtr = maybe_shuffle(axtr, args.seed + 97001, True)
            axva = maybe_shuffle(axva, args.seed + 97002, True)
            axte = maybe_shuffle(axte, args.seed + 97003, True)
        axtr, axva, axte, axis_scaler = v8.standardize_3d(axtr, axva, axte)
        pack["feat_train"] = np.concatenate([pack["feat_train"], axtr], axis=-1).astype(np.float32)
        pack["feat_val"] = np.concatenate([pack["feat_val"], axva], axis=-1).astype(np.float32)
        pack["feat_test"] = np.concatenate([pack["feat_test"], axte], axis=-1).astype(np.float32)
        meta["axis_candidate_feature_scaler"] = finite_json(axis_scaler)
    meta["router_disabled"] = bool(router_disabled)
    meta["shuffle_axis"] = bool(shuffle_axis)
    meta["axis_candidate_feature_count"] = int(len(axis_feature_names))
    return pack, meta


def forward_v9(model: RouteQueryPrunedRefinerV9, pack: dict[str, np.ndarray], split_name: str, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
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


def train_v9(
    pack: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    *,
    shuffled_labels: bool = False,
    router_disabled: bool = False,
) -> tuple[RouteQueryPrunedRefinerV9, pd.DataFrame, float]:
    model = RouteQueryPrunedRefinerV9(
        cand_feat_dim=pack["feat_train"].shape[-1],
        ctx_dim=pack["ctx_train"].shape[-1],
        edge_dim=pack["edge_train"].shape[-1],
        own_dim=pack["own_train"].shape[-1],
        n_lags=pack["edge_train"].shape[1],
        n_neighbours=pack["edge_train"].shape[2],
        max_horizon=args.max_horizon,
        n_axes=pack["axis_train"].shape[1],
        hidden=args.v9_hidden,
        heads=args.v9_heads,
        layers=args.v9_layers,
        route_queries=args.v9_route_queries,
        dropout=args.v9_dropout,
        router_topk=args.v9_router_topk,
        router_disabled=router_disabled,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.v9_lr, weight_decay=args.v9_weight_decay)
    err_train = pack["err_train"]
    target_end_train = pack["target_end_train"]
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 91001)
        perm = rng.permutation(len(err_train))
        err_train = err_train[perm]
        target_end_train = target_end_train[perm]
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack["cand_train"])
    for epoch in range(int(args.v9_epochs)):
        model.train()
        losses: list[float] = []
        for idx in closure.batches(n, args.v9_batch_size, args.seed + 92000 + epoch):
            out = forward_v9(model, pack, "train", idx, device)
            risk = out["risk"]
            err = to_tensor(err_train[idx], device)
            soft = qrc.soft_labels_from_error(err, args.v9_label_temperature)
            listwise = -torch.mean(torch.sum(soft * F.log_softmax(-risk, dim=1), dim=1))
            target_log = torch.log1p(err)
            target_z = (target_log - target_log.mean(dim=1, keepdim=True)) / torch.clamp(target_log.std(dim=1, keepdim=True), min=1e-3)
            pred_z = (risk - risk.mean(dim=1, keepdim=True)) / torch.clamp(risk.std(dim=1, keepdim=True), min=1e-3)
            rank = F.smooth_l1_loss(pred_z, target_z)
            weights = torch.softmax(-risk / max(float(args.v9_train_temperature), 1e-6), dim=1)
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
                args.v9_listwise_weight * listwise
                + args.v9_rank_weight * rank
                + args.v9_reg_weight * reg
                - args.v9_candidate_entropy_weight * cand_entropy
                + args.v9_gate_entropy_weight * gate_entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.v9_clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == args.v9_epochs - 1 or epoch % max(1, args.v9_epochs // 5) == 0:
            risk_val = predict_v9(model, pack, "val", args, device)
            m, t, val_rmse = v8.tune_topm(risk_val, pack["cand_val"], pack["target_steps_val"], args)
            corr = qrc.risk_error_corr(risk_val, pack["err_val"])
            rows.append({"epoch": int(epoch), "train_loss": float(np.mean(losses)), "val_selector_rmse": float(val_rmse), "val_risk_error_corr": float(corr), "top_m": int(m), "temperature": float(t)})
            if val_rmse < best_val:
                best_val = float(val_rmse)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_val


def predict_v9(model: RouteQueryPrunedRefinerV9, pack: dict[str, np.ndarray], split_name: str, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    n = len(pack[f"cand_{split_name}"])
    with torch.no_grad():
        for idx in closure.batches(n, args.v9_batch_size, args.seed + 93001, shuffle=False):
            out.append(forward_v9(model, pack, split_name, idx, device)["risk"].detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def gate_diagnostics(model: RouteQueryPrunedRefinerV9, pack: dict[str, np.ndarray], split_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    weights: list[np.ndarray] = []
    families: list[np.ndarray] = []
    entropies: list[float] = []
    with torch.no_grad():
        for idx in closure.batches(len(pack[f"cand_{split_name}"]), args.v9_batch_size, args.seed + 94001, shuffle=False):
            out = forward_v9(model, pack, split_name, idx, device)
            gate = out["gate"].detach().cpu().numpy()
            fam = out["family"].detach().cpu().numpy()
            weights.append(gate.reshape(-1))
            families.append(fam.reshape(-1))
            entropies.append(float(np.mean(-np.sum(gate * np.log(np.maximum(gate, EPS)), axis=1))))
    w = np.concatenate(weights)
    f = np.concatenate(families)
    diag = family_fraction(w, f)
    diag["gate_entropy"] = float(np.mean(entropies)) if entropies else float("nan")
    return diag


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
    shuffle_axis: bool = False,
    router_disabled: bool = False,
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
        shuffle_axis=shuffle_axis,
        router_disabled=router_disabled,
    )
    if meta_out is not None:
        meta_out[name] = finite_json(meta)
    model, log, val_rmse = train_v9(pack, args, device, shuffled_labels=shuffled_labels, router_disabled=router_disabled)
    logs.append(log.assign(variant=name))
    risk_val = predict_v9(model, pack, "val", args, device)
    risk_test = predict_v9(model, pack, "test", args, device)
    best_m, best_t, tuned_val = v8.tune_topm(risk_val, cand_val.residual, arrays.residual_val, args)
    pred_sparse, _ = v8.sparse_topm_prediction(risk_test, cand_test.residual, best_m, best_t)
    pred_dense = qrc.weighted_residual(cand_test.residual, qrc.softmax_np(-risk_test / best_t, axis=1))
    pred_top = cand_test.residual[np.arange(len(risk_test)), np.argmin(risk_test, axis=1)]
    err_test = qrc.risk_endpoint_errors(cand_test.residual, arrays.residual_test, args)
    corr = qrc.risk_error_corr(risk_test, err_test)
    gdiag = gate_diagnostics(model, pack, "test", args, device)
    extra = {"variant": name, "risk_error_corr": corr, "val_selector_rmse": tuned_val, "best_top_m": best_m, "temperature": best_t, **gdiag}
    add_metric_rows(rows, arrays, pred_sparse, args, f"{name}_sparse_topm", {"stage": "v9_sparse_topm", **extra})
    add_metric_rows(rows, arrays, pred_dense, args, f"{name}_dense", {"stage": "v9_dense", **extra})
    add_metric_rows(rows, arrays, pred_top, args, f"{name}_top1", {"stage": "v9_top1", **extra, "best_top_m": 1})
    diagnostics.append({
        "variant": name,
        "risk_error_corr": float(corr),
        "val_selector_rmse": float(val_rmse),
        "tuned_val_selector_rmse": float(tuned_val),
        "best_top_m": int(best_m),
        "temperature": float(best_t),
        "top1_mean_index": float(np.mean(np.argmin(risk_test, axis=1))),
        **gdiag,
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
        max_lags=args.v9_edge_max_lags,
        max_neighbours=args.v9_edge_max_neighbours,
        min_found_frac=args.v9_min_edge_found_frac,
    )

    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_meta = rpr.prepare_context(args, arrays, posterior, student, blocks, device)
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v9 requires component axes")
    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand_train = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    edge_ctx_train, edge_ctx_val, edge_ctx_test, edge_ctx_meta = v7.select_context_block(arrays, args.v9_video_edge_block, args.v9_video_edge_max_features)
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

    variants: list[tuple[str, dict[str, Any]]] = [
        ("v9_full_pruned", {}),
        ("v9_no_video", {"use_video": False}),
        ("v9_shuffled_video", {"shuffle_video": True}),
        ("v9_no_edge", {"use_edge": False}),
        ("v9_no_axes", {"use_axis": False}),
        ("v9_shuffled_axes", {"shuffle_axis": True}),
        ("v9_no_context", {"use_context": False}),
        ("v9_edge_axes_only", {"use_context": False, "use_video": False}),
        ("v9_router_disabled", {"router_disabled": True}),
        ("v9_shuffled_labels", {"shuffled_labels": True}),
    ]
    requested = {s.strip() for s in str(args.v9_variant_list).split(",") if s.strip()}
    for name, kwargs in variants:
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
    summary.to_csv(args.out_dir / "route_query_pruned_v9_summary.csv", index=False)
    diag.to_csv(args.out_dir / "route_query_pruned_v9_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "route_query_pruned_v9_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "route_query_pruned_v9_learned_route_train_log.csv", index=False)
    video_probe.to_csv(args.out_dir / "route_query_pruned_v9_video_teacher_probe.csv", index=False)
    component_axes.probe.to_csv(args.out_dir / "route_query_pruned_v9_component_axis_probe.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag, video_probe)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, video_probe: pd.DataFrame) -> None:
    lines = ["# Route-Query Pruned Component Refiner v9 Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- route_queries: `{args.v9_route_queries}`")
    lines.append(f"- router_topk: `{args.v9_router_topk}`")
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
                f"val={row['val_selector_rmse']:.3f}, topM={int(row['best_top_m'])}, "
                f"T={row['temperature']:.3f}, gate_entropy={row.get('gate_entropy', float('nan')):.3f}, "
                f"edge={row.get('gate_frac_edge', float('nan')):.3f}, axis={row.get('gate_frac_axis', float('nan')):.3f}, "
                f"video={row.get('gate_frac_video', float('nan')):.3f}, context={row.get('gate_frac_context', float('nan')):.3f}"
            )
    if not video_probe.empty:
        lines.append("\n## Video Teacher Probe")
        lines.append(video_probe.to_markdown(index=False))
    lines.append("\n## Decision Gates")
    lines.append("- Hard pass: Bulk seed42 h6 <= 16.9 and full beats no_edge/no_axes/router_disabled.")
    lines.append("- Soft pass: Bulk seed42 h6 beats component-aware ~17.31.")
    lines.append("- Fail: v9 remains around 18+ or full is no better than lean controls.")
    (out_dir / "route_query_pruned_v9_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--v9-video-edge-block", type=str, default="explicit_edge")
    parser.add_argument("--v9-video-edge-max-features", type=int, default=128)
    parser.add_argument("--video-residual-model", type=str, default="hgbdt", choices=["ridge", "hgbdt"])
    parser.add_argument("--video-residual-include-context", action="store_true")
    parser.add_argument("--video-residual-hgbdt-iter", type=int, default=120)
    parser.add_argument("--video-residual-hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--video-residual-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--video-residual-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v9-hidden", type=int, default=192)
    parser.add_argument("--v9-heads", type=int, default=4)
    parser.add_argument("--v9-layers", type=int, default=2)
    parser.add_argument("--v9-route-queries", type=int, default=12)
    parser.add_argument("--v9-router-topk", type=int, default=24)
    parser.add_argument("--v9-dropout", type=float, default=0.05)
    parser.add_argument("--v9-epochs", type=int, default=10)
    parser.add_argument("--v9-batch-size", type=int, default=192)
    parser.add_argument("--v9-lr", type=float, default=7e-4)
    parser.add_argument("--v9-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v9-label-temperature", type=float, default=8.0)
    parser.add_argument("--v9-train-temperature", type=float, default=0.75)
    parser.add_argument("--v9-listwise-weight", type=float, default=1.0)
    parser.add_argument("--v9-rank-weight", type=float, default=0.5)
    parser.add_argument("--v9-reg-weight", type=float, default=0.25)
    parser.add_argument("--v9-candidate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--v9-gate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--v9-clip-grad", type=float, default=5.0)
    parser.add_argument("--v9-topm", type=str, default="1,2,4,8,16")
    parser.add_argument("--v9-temperatures", type=str, default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--v9-edge-max-lags", type=int, default=0)
    parser.add_argument("--v9-edge-max-neighbours", type=int, default=0)
    parser.add_argument("--v9-min-edge-found-frac", type=float, default=0.98)
    parser.add_argument("--v9-variant-list", type=str, default="")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    args.v8_topm = args.v9_topm
    args.v8_temperatures = [float(x) for x in str(args.v9_temperatures).split(",") if x.strip()]
    args.v8_video_edge_block = args.v9_video_edge_block
    args.v8_video_edge_max_features = args.v9_video_edge_max_features
    args.v8_edge_max_lags = args.v9_edge_max_lags
    args.v8_edge_max_neighbours = args.v9_edge_max_neighbours
    args.v8_min_edge_found_frac = args.v9_min_edge_found_frac
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1200)
        args.max_val_rows = min(args.max_val_rows, 400)
        args.max_test_rows = min(args.max_test_rows, 600)
        args.posterior_epochs = min(args.posterior_epochs, 3)
        args.student_epochs = min(args.student_epochs, 3)
        args.learned_route_epochs = min(args.learned_route_epochs, 3)
        args.v9_epochs = min(args.v9_epochs, 3)
        args.candidate_k = min(args.candidate_k, 32)
    run(args)


if __name__ == "__main__":
    main()
