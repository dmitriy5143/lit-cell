#!/usr/bin/env python3
"""Graph-memory route-conditioned critic/refiner v4 for LaChance trajectories.

This is the "battle" version after v3:

    candidate route sequence tokens
    + candidate-to-axis decomposition attention
    + causal edge-memory tokens (lag x neighbour)
    + candidate-to-graph cross attention
    + causal global context
    -> per-horizon route weights
    -> full step-sequence correction
    -> endpoint residual prediction + uncertainty

The target future is used only for training losses and metric/oracle labels.
All inference features are causal: past/self motion, decomposition student axes,
candidate trajectories generated from the causal generator, and same/past-frame
neighbour memory from the edge-sequence cache.
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

import run_lachance_axis_conditioned_distillation_critic as axis  # noqa: E402
import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_sequence_graph_critic_v3 as v3  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "graph_memory_critic_v4_2026-07-01"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def to_bool_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.bool, device=device)


def residual_endpoints(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    return axis.residual_endpoints(residual, horizons)


def candidate_endpoints(query_pred: np.ndarray, horizons: list[int]) -> np.ndarray:
    return axis.candidate_endpoints(query_pred, horizons)


def endpoint_rows_from_residual_endpoints(
    *,
    arrays: audit.SplitArrays,
    residual_endpoint_pred: np.ndarray,
    horizons: list[int],
    label: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return axis.endpoint_rows_from_residual_endpoints(
        arrays=arrays,
        residual_endpoint_pred=residual_endpoint_pred,
        horizons=horizons,
        label=label,
        extra=extra,
    )


def maybe_shuffle(x: np.ndarray, seed: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.permutation(len(x))]


def standardize_candidate_steps(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return v3.standardize_candidate_steps(train, val, test)


def standardize_2d(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tr, va, te, scaler = seq.standardize(train, val, test)
    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32), finite_json(scaler)


def split_keys(df: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    return [
        (str(row.dataset), int(row.sequence), int(row.frame), int(row.track_id))
        for row in df[["dataset", "sequence", "frame", "track_id"]].itertuples(index=False)
    ]


def load_edge_memory_cache(
    cache_path: Path,
    split: seq.SplitData,
    *,
    max_lags: int = 0,
    max_neighbours: int = 0,
    min_found_frac: float = 0.98,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if cache_path is None or not Path(cache_path).exists():
        raise FileNotFoundError(cache_path)
    cache = np.load(cache_path, allow_pickle=True)
    key_map: dict[tuple[str, int, int, int], int] = {}
    duplicate_keys = 0
    for i in range(len(cache["key_track_id"])):
        key = (
            str(cache["key_dataset"][i]),
            int(cache["key_sequence"][i]),
            int(cache["key_frame"][i]),
            int(cache["key_track_id"][i]),
        )
        if key in key_map:
            duplicate_keys += 1
        key_map[key] = int(i)

    edge_all = cache["edge_features"].astype(np.float32)
    own_all = cache["own_features"].astype(np.float32)
    valid_all = cache["valid_mask"].astype(bool)
    center_all = cache["center_present"].astype(bool)
    lags = cache["lags"].astype(np.int32)
    edge_order = [str(x) for x in cache["feature_order"].tolist()]
    own_order = [str(x) for x in cache["own_feature_order"].tolist()]

    if int(max_lags) > 0:
        edge_all = edge_all[:, : int(max_lags)]
        own_all = own_all[:, : int(max_lags)]
        valid_all = valid_all[:, : int(max_lags)]
        center_all = center_all[:, : int(max_lags)]
        lags = lags[: int(max_lags)]
    if int(max_neighbours) > 0:
        edge_all = edge_all[:, :, : int(max_neighbours)]
        valid_all = valid_all[:, :, : int(max_neighbours)]

    def take(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keys = split_keys(part)
        edge = np.zeros((len(keys),) + edge_all.shape[1:], dtype=np.float32)
        own = np.zeros((len(keys),) + own_all.shape[1:], dtype=np.float32)
        valid = np.zeros((len(keys),) + valid_all.shape[1:], dtype=bool)
        center = np.zeros((len(keys),) + center_all.shape[1:], dtype=bool)
        found = np.zeros(len(keys), dtype=bool)
        for row_idx, key in enumerate(keys):
            src = key_map.get(key)
            if src is None:
                continue
            edge[row_idx] = edge_all[src]
            own[row_idx] = own_all[src]
            valid[row_idx] = valid_all[src]
            center[row_idx] = center_all[src]
            found[row_idx] = True
        return edge, own, valid, center, found

    tr_edge, tr_own, tr_valid, tr_center, tr_found = take(split.train)
    va_edge, va_own, va_valid, va_center, va_found = take(split.val)
    te_edge, te_own, te_valid, te_center, te_found = take(split.test)
    meta = {
        "cache_path": str(cache_path),
        "lags": lags.tolist(),
        "edge_feature_order": edge_order,
        "own_feature_order": own_order,
        "edge_shape_train": list(tr_edge.shape),
        "own_shape_train": list(tr_own.shape),
        "found_train_frac": float(np.mean(tr_found)) if len(tr_found) else 0.0,
        "found_val_frac": float(np.mean(va_found)) if len(va_found) else 0.0,
        "found_test_frac": float(np.mean(te_found)) if len(te_found) else 0.0,
        "valid_edge_train_frac": float(np.mean(tr_valid)) if tr_valid.size else 0.0,
        "center_present_train_frac": float(np.mean(tr_center)) if tr_center.size else 0.0,
        "duplicate_key_count": int(duplicate_keys),
    }
    if duplicate_keys:
        raise ValueError(f"edge sequence cache has {duplicate_keys} duplicate keys; rebuild with stable row ids")
    low = {k: meta[k] for k in ["found_train_frac", "found_val_frac", "found_test_frac"] if meta[k] < float(min_found_frac)}
    if low:
        raise ValueError(
            "edge sequence cache coverage is too low for a valid v4 run: "
            + json.dumps(low, ensure_ascii=False)
            + f"; rebuild cache for all split rows or lower --v4-min-edge-found-frac (current {min_found_frac})"
        )
    return {
        "edge_train": tr_edge,
        "edge_val": va_edge,
        "edge_test": te_edge,
        "own_train": tr_own,
        "own_val": va_own,
        "own_test": te_own,
        "valid_train": tr_valid,
        "valid_val": va_valid,
        "valid_test": te_valid,
        "center_train": tr_center,
        "center_val": va_center,
        "center_test": te_center,
    }, meta


def standardize_edge_memory(
    edge_train: np.ndarray,
    edge_val: np.ndarray,
    edge_test: np.ndarray,
    own_train: np.ndarray,
    own_val: np.ndarray,
    own_test: np.ndarray,
    valid_train: np.ndarray,
    center_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    def fit_stats(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.size == 0:
            return np.zeros((x.shape[-1],), dtype=np.float32), np.ones((x.shape[-1],), dtype=np.float32)
        flat = x.reshape(-1, x.shape[-1])
        m = mask.reshape(-1)
        if np.any(m):
            use = flat[m]
        else:
            use = flat
        mean = np.nanmean(use, axis=0).astype(np.float32)
        std = np.nanstd(use, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return mean, std

    edge_mean, edge_std = fit_stats(edge_train, valid_train)
    own_mean, own_std = fit_stats(own_train, center_train)

    def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        y = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) - mean.reshape(*([1] * (x.ndim - 1)), -1)) / std.reshape(*([1] * (x.ndim - 1)), -1)
        return np.clip(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0).astype(np.float32)

    meta = {
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
        "own_mean": own_mean.tolist(),
        "own_std": own_std.tolist(),
    }
    return (
        transform(edge_train, edge_mean, edge_std),
        transform(edge_val, edge_mean, edge_std),
        transform(edge_test, edge_mean, edge_std),
        transform(own_train, own_mean, own_std),
        transform(own_val, own_mean, own_std),
        transform(own_test, own_mean, own_std),
        meta,
    )


class GraphMemoryCriticV4(nn.Module):
    def __init__(
        self,
        *,
        step_dim: int,
        axis_dim: int,
        ctx_dim: int,
        edge_dim: int,
        own_dim: int,
        n_axes: int,
        n_horizons: int,
        max_horizon: int,
        n_lags: int,
        n_neighbours: int,
        hidden: int,
        heads: int,
        graph_layers: int,
        cand_layers: int,
        dropout: float,
        correction_scale: float,
        context_router_pruner: bool = False,
        axis_graph_cross_attention: bool = False,
        axis_score_router: bool = False,
        sequence_graph_refiner: bool = False,
        v6_route_score_source: str = "full",
    ):
        super().__init__()
        self.n_horizons = int(n_horizons)
        self.max_horizon = int(max_horizon)
        self.n_lags = int(n_lags)
        self.n_neighbours = int(n_neighbours)
        self.correction_scale = float(correction_scale)
        self.context_router_pruner = bool(context_router_pruner)
        self.axis_graph_cross_attention = bool(axis_graph_cross_attention)
        self.axis_score_router = bool(axis_score_router)
        self.sequence_graph_refiner = bool(sequence_graph_refiner)
        self.v6_route_score_source = str(v6_route_score_source)
        self.ctx_dim = int(ctx_dim)
        heads = max(1, int(heads))
        hidden = int(hidden)

        self.step_proj = nn.Sequential(nn.Linear(step_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.time_embed = nn.Embedding(max_horizon, hidden)
        step_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.step_encoder = nn.TransformerEncoder(step_layer, num_layers=1)

        self.axis_proj = nn.Sequential(nn.Linear(max(axis_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.axis_embed = nn.Embedding(max(1, int(n_axes)), hidden)
        self.axis_cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.axis_graph_cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.candidate_axis_graph_cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)

        ctx_in = max(ctx_dim, 1)
        self.ctx_feature_gate_logits = nn.Parameter(torch.zeros(ctx_in))
        self.ctx_sample_gate = nn.Sequential(nn.Linear(ctx_in, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.ctx_proj = nn.Sequential(nn.Linear(ctx_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_proj = nn.Sequential(nn.Linear(max(edge_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.own_proj = nn.Sequential(nn.Linear(max(own_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.lag_embed = nn.Embedding(max(1, n_lags), hidden)
        self.rank_embed = nn.Embedding(max(1, n_neighbours), hidden)
        self.own_token = nn.Parameter(torch.zeros(1, 1, hidden))

        graph_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph_encoder = nn.TransformerEncoder(graph_layer, num_layers=max(1, int(graph_layers)))
        self.graph_cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)

        cand_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(cand_layer, num_layers=max(1, int(cand_layers)))

        self.axis_graph_gate = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.axis_component_score_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )
        self.axis_component_gate_head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Conservative horizon-wise component score. A scalar score path helped
        # h1 but over-steered long horizons; per-horizon scale lets the model
        # use component votes only where validation signal supports them.
        self.axis_score_scale = nn.Parameter(torch.full((n_horizons,), -2.5))
        fuse_dim = hidden * (5 if self.axis_graph_cross_attention else 4)
        self.fuse = nn.Sequential(nn.LayerNorm(fuse_dim), nn.Linear(fuse_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.score_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_horizons))
        self.sequence_correction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, max_horizon * 2))
        self.logvar_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.temp = nn.Parameter(torch.tensor(0.0))

        # v6 modules are initialized after all v4/v5 layers so enabling or
        # disabling v6 does not silently change the old path initialization.
        dyn_dim = max_horizon * 2 + max_horizon + max(1, max_horizon - 1) + n_horizons
        pair_dim = n_horizons + 3
        self.v6_dyn_proj = nn.Sequential(
            nn.LayerNorm(dyn_dim),
            nn.Linear(dyn_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.v6_pair_feat_proj = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.v6_pair_score = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.v6_refine_fuse = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.v6_refine_gate = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.v6_route_score_head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )
        self.v6_route_score_scale = nn.Parameter(torch.full((n_horizons,), -2.5))
        self.v6_refine_scale = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.v6_refine_fuse[-1].weight)
        nn.init.zeros_(self.v6_refine_fuse[-1].bias)
        nn.init.constant_(self.v6_refine_gate[-1].bias, -2.0)

    def encode_context(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = ctx.shape[0]
        if ctx.shape[-1] == 0:
            ctx = torch.zeros((n, 1), dtype=ctx.dtype, device=ctx.device)
        if not self.context_router_pruner:
            return self.ctx_proj(ctx), torch.ones((), dtype=ctx.dtype, device=ctx.device), torch.ones((n, 1), dtype=ctx.dtype, device=ctx.device)
        feature_gate = torch.sigmoid(self.ctx_feature_gate_logits).reshape(1, -1)
        gated_ctx = ctx * feature_gate
        sample_gate = torch.sigmoid(self.ctx_sample_gate(gated_ctx))
        return self.ctx_proj(gated_ctx) * sample_gate, feature_gate.mean(), sample_gate

    def encode_candidate_steps(self, cand_steps: torch.Tensor) -> torch.Tensor:
        n, q, h, _ = cand_steps.shape
        step = self.step_proj(cand_steps.reshape(n * q, h, -1))
        t = self.time_embed(torch.arange(h, device=cand_steps.device))[None, :, :]
        return self.step_encoder(step + t).mean(dim=1).reshape(n, q, -1)

    def encode_axis_tokens(self, axis_feat: torch.Tensor, cand_h: torch.Tensor) -> torch.Tensor:
        n, q, _ = cand_h.shape
        if axis_feat.shape[-1] == 0 or axis_feat.shape[2] == 0:
            return torch.zeros((n, q, 1, cand_h.shape[-1]), dtype=cand_h.dtype, device=cand_h.device)
        c = axis_feat.shape[2]
        ah = self.axis_proj(axis_feat.reshape(n * q, c, -1))
        eid = self.axis_embed(torch.arange(c, device=cand_h.device))[None, :, :]
        return (ah + eid).reshape(n, q, c, -1)

    def encode_axes(self, cand_h: torch.Tensor, axis_feat: torch.Tensor) -> torch.Tensor:
        n, q, d = cand_h.shape
        if axis_feat.shape[-1] == 0 or axis_feat.shape[2] == 0:
            return torch.zeros_like(cand_h)
        axis_tokens = self.encode_axis_tokens(axis_feat, cand_h).reshape(n * q, axis_feat.shape[2], -1)
        query = cand_h.reshape(n * q, 1, d)
        out, _ = self.axis_cross(query, axis_tokens, axis_tokens, need_weights=False)
        return out.reshape(n, q, d)

    def encode_axis_graph(
        self,
        cand_h: torch.Tensor,
        axis_feat: torch.Tensor,
        memory: torch.Tensor,
        key_padding: torch.Tensor,
        axis_h: torch.Tensor,
        graph_h: torch.Tensor,
        ctx_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, q, d = cand_h.shape
        if (not self.axis_graph_cross_attention) or axis_feat.shape[-1] == 0 or axis_feat.shape[2] == 0:
            component_tokens = torch.zeros((n, q, 1, d), dtype=cand_h.dtype, device=cand_h.device)
            component_gate = torch.ones((n, q, 1), dtype=cand_h.dtype, device=cand_h.device)
            return torch.zeros_like(cand_h), torch.ones((n, q, 1), dtype=cand_h.dtype, device=cand_h.device), component_tokens, component_gate
        c = axis_feat.shape[2]
        axis_tokens = self.encode_axis_tokens(axis_feat, cand_h).reshape(n * q, c, d)
        memory_rep = memory[:, None, :, :].expand(n, q, memory.shape[1], d).reshape(n * q, memory.shape[1], d)
        padding_rep = key_padding[:, None, :].expand(n, q, key_padding.shape[1]).reshape(n * q, key_padding.shape[1])
        axis_graph_tokens, _ = self.axis_graph_cross(
            axis_tokens,
            memory_rep,
            memory_rep,
            key_padding_mask=padding_rep,
            need_weights=False,
        )
        query = cand_h.reshape(n * q, 1, d)
        axis_graph_h, _ = self.candidate_axis_graph_cross(query, axis_graph_tokens, axis_graph_tokens, need_weights=False)
        axis_graph_h = axis_graph_h.reshape(n, q, d)
        component_tokens = axis_graph_tokens.reshape(n, q, c, d)
        ctx_rep = ctx_h[:, None, :].expand(-1, q, -1)
        gate_in = torch.cat([cand_h, axis_h, graph_h, ctx_rep], dim=-1)
        gate = torch.sigmoid(self.axis_graph_gate(gate_in))
        comp_gate_in = torch.cat(
            [
                component_tokens,
                cand_h[:, :, None, :].expand(-1, -1, c, -1),
                ctx_h[:, None, None, :].expand(-1, q, c, -1),
            ],
            dim=-1,
        )
        component_gate = torch.softmax(self.axis_component_gate_head(comp_gate_in).squeeze(-1), dim=-1)
        return axis_graph_h * gate, gate, component_tokens, component_gate

    def encode_graph_memory(
        self,
        edge_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        own_seq: torch.Tensor,
        center_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, l, k, _ = edge_seq.shape
        edge_h = self.edge_proj(edge_seq.reshape(n, l * k, -1)).reshape(n, l, k, -1)
        lag_ids = torch.arange(l, device=edge_seq.device)
        rank_ids = torch.arange(k, device=edge_seq.device)
        edge_h = edge_h + self.lag_embed(lag_ids)[None, :, None, :] + self.rank_embed(rank_ids)[None, None, :, :]
        edge_tokens = edge_h.reshape(n, l * k, -1)
        edge_valid = valid_mask.reshape(n, l * k)

        own_h = self.own_proj(own_seq.reshape(n, l, -1)) + self.lag_embed(lag_ids)[None, :, :] + self.own_token
        tokens = torch.cat([edge_tokens, own_h], dim=1)
        valid = torch.cat([edge_valid, center_mask], dim=1)
        no_valid = ~torch.any(valid, dim=1)
        if torch.any(no_valid):
            valid = valid.clone()
            valid[no_valid, 0] = True
        key_padding = ~valid
        memory = self.graph_encoder(tokens, src_key_padding_mask=key_padding)
        return memory, key_padding

    def encode_candidate_dynamics(self, cand_residual_steps: torch.Tensor, cand_end: torch.Tensor) -> torch.Tensor:
        n, q, t, _ = cand_residual_steps.shape
        step_flat = cand_residual_steps.reshape(n, q, t * 2)
        step_norm = torch.linalg.norm(cand_residual_steps, dim=-1)
        if t > 1:
            accel_norm = torch.linalg.norm(cand_residual_steps[:, :, 1:] - cand_residual_steps[:, :, :-1], dim=-1)
        else:
            accel_norm = torch.zeros((n, q, 1), dtype=cand_residual_steps.dtype, device=cand_residual_steps.device)
        endpoint_norm = torch.linalg.norm(cand_end, dim=-1)
        dyn = torch.cat([step_flat, step_norm, accel_norm, endpoint_norm], dim=-1)
        return self.v6_dyn_proj(dyn)

    def encode_pair_relation(self, cand_h: torch.Tensor, cand_end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, q, d = cand_h.shape
        if q <= 1:
            return torch.zeros_like(cand_h), torch.zeros((), dtype=cand_h.dtype, device=cand_h.device)
        diff = cand_end[:, :, None, :, :] - cand_end[:, None, :, :, :]
        dist = torch.log1p(torch.linalg.norm(diff, dim=-1))
        final_dist = dist[:, :, :, -1:]
        mean_dist = dist.mean(dim=-1, keepdim=True)
        final_end = cand_end[:, :, -1, :]
        final_norm = torch.clamp(torch.linalg.norm(final_end, dim=-1, keepdim=True), min=1e-6)
        cos = (final_end[:, :, None, :] * final_end[:, None, :, :]).sum(dim=-1, keepdim=True)
        cos = cos / torch.clamp(final_norm[:, :, None, :] * final_norm[:, None, :, :], min=1e-6)
        pair_feat = torch.cat([dist, final_dist, mean_dist, cos], dim=-1)
        pair_h = self.v6_pair_feat_proj(pair_feat)
        cand_i = cand_h[:, :, None, :].expand(-1, -1, q, -1)
        cand_j = cand_h[:, None, :, :].expand(-1, q, -1, -1)
        pair_logits = self.v6_pair_score(torch.cat([cand_i, cand_j, pair_h], dim=-1)).squeeze(-1)
        eye = torch.eye(q, dtype=torch.bool, device=cand_h.device)
        pair_logits = pair_logits.masked_fill(eye[None, :, :], -1e4)
        pair_attn = torch.softmax(pair_logits, dim=-1)
        pair_context = torch.sum(pair_attn[:, :, :, None] * (cand_j + pair_h), dim=2)
        entropy = -torch.mean(torch.sum(torch.clamp(pair_attn, min=1e-8) * torch.log(torch.clamp(pair_attn, min=1e-8)), dim=-1))
        return pair_context, entropy

    def forward(
        self,
        cand_steps: torch.Tensor,
        axis_feat: torch.Tensor,
        ctx: torch.Tensor,
        edge_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        own_seq: torch.Tensor,
        center_mask: torch.Tensor,
        cand_residual_steps: torch.Tensor,
        cand_end: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n, q, _, _ = cand_steps.shape
        cand_step_h = self.encode_candidate_steps(cand_steps)
        axis_h = self.encode_axes(cand_step_h, axis_feat)

        ctx_h, ctx_feature_gate_mean, ctx_sample_gate = self.encode_context(ctx)

        memory, key_padding = self.encode_graph_memory(edge_seq, valid_mask, own_seq, center_mask)
        graph_h, _ = self.graph_cross(cand_step_h, memory, memory, key_padding_mask=key_padding, need_weights=False)
        ctx_rep = ctx_h[:, None, :].expand(-1, q, -1)
        if self.axis_graph_cross_attention:
            axis_graph_h, axis_graph_gate, axis_component_tokens, axis_component_gate = self.encode_axis_graph(
                cand_step_h,
                axis_feat,
                memory,
                key_padding,
                axis_h,
                graph_h,
                ctx_h,
            )
            fused_in = torch.cat([cand_step_h, axis_h, graph_h, axis_graph_h, ctx_rep], dim=-1)
        else:
            axis_graph_gate = torch.ones((n, q, 1), dtype=cand_step_h.dtype, device=cand_step_h.device)
            axis_component_tokens = torch.zeros((n, q, 1, cand_step_h.shape[-1]), dtype=cand_step_h.dtype, device=cand_step_h.device)
            axis_component_gate = torch.ones((n, q, 1), dtype=cand_step_h.dtype, device=cand_step_h.device)
            fused_in = torch.cat([cand_step_h, axis_h, graph_h, ctx_rep], dim=-1)
        fused = self.fuse(fused_in)
        cand_h = self.candidate_encoder(fused)
        if self.sequence_graph_refiner:
            dyn_h = self.encode_candidate_dynamics(cand_residual_steps, cand_end)
            component_h = torch.sum(axis_component_gate[:, :, :, None] * axis_component_tokens, dim=2)
            pair_h, v6_pair_entropy = self.encode_pair_relation(cand_h, cand_end)
            v6_fuse_in = torch.cat([cand_h, dyn_h, component_h, pair_h, ctx_rep], dim=-1)
            route_component_h = component_h
            if self.v6_route_score_source == "pair_dyn":
                route_component_h = torch.zeros_like(component_h)
            v6_route_logits = self.v6_route_score_head(torch.cat([dyn_h, route_component_h, pair_h], dim=-1)).permute(0, 2, 1).contiguous()
            v6_route_score_scale = torch.clamp(F.softplus(self.v6_route_score_scale), min=0.0, max=0.75).reshape(1, self.n_horizons, 1)
            v6_delta = self.v6_refine_fuse(v6_fuse_in)
            v6_gate = torch.sigmoid(self.v6_refine_gate(v6_fuse_in))
            v6_scale = torch.clamp(F.softplus(self.v6_refine_scale), min=0.0, max=1.0)
            cand_h = cand_h + v6_scale * v6_gate * v6_delta
        else:
            v6_gate = torch.zeros((n, q, 1), dtype=cand_h.dtype, device=cand_h.device)
            v6_scale = torch.zeros((), dtype=cand_h.dtype, device=cand_h.device)
            v6_pair_entropy = torch.zeros((), dtype=cand_h.dtype, device=cand_h.device)
            v6_route_logits = torch.zeros((n, self.n_horizons, q), dtype=cand_h.dtype, device=cand_h.device)
            v6_route_score_scale = torch.zeros((1, self.n_horizons, 1), dtype=cand_h.dtype, device=cand_h.device)

        base_logits = self.score_head(cand_h).permute(0, 2, 1).contiguous()
        logits = base_logits
        if self.axis_score_router and self.axis_graph_cross_attention:
            axis_scores = self.axis_component_score_head(axis_component_tokens)  # n, q, c, horizons
            axis_logits = torch.einsum("nqc,nqch->nqh", axis_component_gate, axis_scores).permute(0, 2, 1).contiguous()
            axis_scale = torch.clamp(F.softplus(self.axis_score_scale), min=0.0, max=0.75).reshape(1, self.n_horizons, 1)
            logits = logits + axis_scale * axis_logits
            axis_scale_out = axis_scale.reshape(self.n_horizons)
        else:
            axis_scale_out = torch.zeros((self.n_horizons,), dtype=cand_h.dtype, device=cand_h.device)
        if self.sequence_graph_refiner:
            logits = logits + v6_route_score_scale * v6_route_logits
        temp = torch.clamp(F.softplus(self.temp) + 0.15, min=0.20, max=6.0)
        weights = torch.softmax(logits / temp, dim=-1)
        mixture_end = torch.einsum("nhq,nqhd->nhd", weights, cand_end)
        mixture_steps = torch.einsum("nhq,nqtd->nhtd", weights, cand_residual_steps)
        pooled = torch.einsum("nhq,nqd->nhd", weights, cand_h)
        seq_corr = torch.tanh(self.sequence_correction_head(pooled).reshape(n, self.n_horizons, self.max_horizon, 2)) * self.correction_scale
        pred_steps = mixture_steps + seq_corr
        pred_end_parts = []
        for hi, horizon in enumerate(range(1, self.n_horizons + 1)):
            # Placeholder, overwritten by actual horizon ids in endpoint_from_steps.
            pred_end_parts.append(pred_steps[:, hi, :horizon, :].sum(dim=1))
        pred_end = torch.stack(pred_end_parts, dim=1)
        # If horizons are not [1..n], caller will pass a horizon mask through
        # endpoint_select below; this field is overwritten in wrapper using
        # actual horizon indices when needed.
        logvar = self.logvar_head(pooled)
        return {
            "pred": pred_end,
            "mixture": mixture_end,
            "weights": weights,
            "logits": logits,
            "correction_steps": seq_corr,
            "pred_steps": pred_steps,
            "mixture_steps": mixture_steps,
            "logvar": logvar,
            "ctx_feature_gate_mean": ctx_feature_gate_mean,
            "ctx_sample_gate": ctx_sample_gate,
            "axis_graph_gate": axis_graph_gate,
            "axis_component_gate": axis_component_gate,
            "axis_score_scale": axis_scale_out,
            "v6_refine_gate": v6_gate,
            "v6_refine_scale": v6_scale,
            "v6_pair_entropy": v6_pair_entropy,
            "v6_route_score_scale": v6_route_score_scale.reshape(self.n_horizons),
        }


def endpoint_from_pred_steps(pred_steps: torch.Tensor, horizons: list[int]) -> torch.Tensor:
    parts = []
    for hi, h in enumerate(horizons):
        parts.append(pred_steps[:, hi, : int(h), :].sum(dim=1))
    return torch.stack(parts, dim=1)


def v4_loss(
    out: dict[str, torch.Tensor],
    target_end: torch.Tensor,
    target_steps: torch.Tensor,
    target_soft: torch.Tensor,
    target_rank: torch.Tensor,
    horizons: list[int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = endpoint_from_pred_steps(out["pred_steps"], horizons).contiguous()
    target_end = target_end.contiguous()
    target_steps = target_steps.contiguous()
    target_soft = target_soft.contiguous()
    target_rank = target_rank.contiguous()
    weights = torch.clamp(out["weights"].contiguous(), min=1e-8)
    logits = out["logits"].contiguous()
    logvar = torch.clamp(out["logvar"].contiguous(), min=args.v4_logvar_min, max=args.v4_logvar_max)

    reg = F.smooth_l1_loss(pred, target_end)
    listwise = -torch.mean(torch.sum(target_soft * torch.log(weights), dim=-1))
    score_z = (logits - logits.mean(dim=-1, keepdim=True)) / torch.clamp(logits.std(dim=-1, keepdim=True), min=1e-3)
    rank_z = (target_rank - target_rank.mean(dim=-1, keepdim=True)) / torch.clamp(target_rank.std(dim=-1, keepdim=True), min=1e-3)
    rank = F.smooth_l1_loss(score_z, rank_z)
    nll = 0.5 * torch.mean((target_end - pred).pow(2) * torch.exp(-logvar) + logvar)

    max_head = -1
    pred_steps_max = out["pred_steps"][:, max_head].contiguous()
    seq_loss = F.smooth_l1_loss(pred_steps_max, target_steps)
    pred_acc = pred_steps_max[:, 1:] - pred_steps_max[:, :-1]
    targ_acc = target_steps[:, 1:] - target_steps[:, :-1]
    accel = F.smooth_l1_loss(pred_acc, targ_acc)
    correction_l2 = torch.mean(out["correction_steps"].pow(2))
    entropy = -torch.mean(torch.sum(weights * torch.log(weights), dim=-1))
    context_gate_l1 = torch.mean(out.get("ctx_sample_gate", torch.ones_like(logvar[..., :1])))

    loss = (
        args.v4_reg_weight * reg
        + args.v4_listwise_weight * listwise
        + args.v4_rank_weight * rank
        + args.v4_nll_weight * nll
        + args.v4_sequence_weight * seq_loss
        + args.v4_accel_weight * accel
        + args.v4_correction_l2_weight * correction_l2
        + args.v4_context_pruner_l1_weight * context_gate_l1
        - args.v4_entropy_weight * entropy
    )
    return loss, {
        "reg": float(reg.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "rank": float(rank.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "sequence": float(seq_loss.detach().cpu()),
        "accel": float(accel.detach().cpu()),
        "correction_l2": float(correction_l2.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "context_gate_l1": float(context_gate_l1.detach().cpu()),
    }


def forward_model(model: GraphMemoryCriticV4, pack: dict[str, np.ndarray], split_name: str, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        to_tensor(pack[f"steps_{split_name}"][idx], device),
        to_tensor(pack[f"axis_{split_name}"][idx], device),
        to_tensor(pack[f"ctx_{split_name}"][idx], device),
        to_tensor(pack[f"edge_seq_{split_name}"][idx], device),
        to_bool_tensor(pack[f"valid_{split_name}"][idx], device),
        to_tensor(pack[f"own_seq_{split_name}"][idx], device),
        to_bool_tensor(pack[f"center_{split_name}"][idx], device),
        to_tensor(pack[f"cand_res_{split_name}"][idx], device),
        to_tensor(pack[f"cand_end_{split_name}"][idx], device),
    )


def predict_v4(model: GraphMemoryCriticV4, pack: dict[str, np.ndarray], split_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    out_parts: dict[str, list[np.ndarray]] = {
        "pred": [],
        "mixture": [],
        "weights": [],
        "logits": [],
        "pred_steps": [],
        "correction_steps": [],
        "logvar": [],
        "ctx_sample_gate": [],
        "axis_graph_gate": [],
        "axis_component_gate": [],
        "axis_score_scale": [],
        "v6_refine_gate": [],
        "v6_refine_scale": [],
        "v6_pair_entropy": [],
        "v6_route_score_scale": [],
    }
    n = len(pack[f"steps_{split_name}"])
    with torch.no_grad():
        for idx in closure.batches(n, args.v4_batch_size, args.seed + 91001, shuffle=False):
            out = forward_model(model, pack, split_name, idx, device)
            pred = endpoint_from_pred_steps(out["pred_steps"], args.horizons)
            out_parts["pred"].append(pred.detach().cpu().numpy())
            for key in [
                "mixture",
                "weights",
                "logits",
                "pred_steps",
                "correction_steps",
                "logvar",
                "ctx_sample_gate",
                "axis_graph_gate",
                "axis_component_gate",
                "v6_refine_gate",
            ]:
                out_parts[key].append(out[key].detach().cpu().numpy())
            scale_np = out["axis_score_scale"].detach().cpu().numpy().astype(np.float32)
            if scale_np.ndim == 0:
                out_parts["axis_score_scale"].append(np.full((len(idx),), float(scale_np), dtype=np.float32))
            else:
                out_parts["axis_score_scale"].append(np.tile(scale_np.reshape(1, -1), (len(idx), 1)).astype(np.float32))
            out_parts["v6_refine_scale"].append(
                np.full((len(idx),), float(out["v6_refine_scale"].detach().cpu()), dtype=np.float32)
            )
            out_parts["v6_pair_entropy"].append(
                np.full((len(idx),), float(out["v6_pair_entropy"].detach().cpu()), dtype=np.float32)
            )
            route_scale_np = out["v6_route_score_scale"].detach().cpu().numpy().astype(np.float32)
            out_parts["v6_route_score_scale"].append(np.tile(route_scale_np.reshape(1, -1), (len(idx), 1)).astype(np.float32))
    return {key: np.concatenate(vals, axis=0).astype(np.float32) for key, vals in out_parts.items()}


def train_v4_variant(
    pack: dict[str, np.ndarray],
    args: argparse.Namespace,
    *,
    device: torch.device,
    variant: str,
    shuffled_labels: bool = False,
) -> tuple[GraphMemoryCriticV4, pd.DataFrame, float]:
    set_global_seed(args.seed + 97531)
    model = GraphMemoryCriticV4(
        step_dim=pack["steps_train"].shape[-1],
        axis_dim=pack["axis_train"].shape[-1],
        ctx_dim=pack["ctx_train"].shape[-1],
        edge_dim=pack["edge_seq_train"].shape[-1],
        own_dim=pack["own_seq_train"].shape[-1],
        n_axes=pack["axis_train"].shape[2],
        n_horizons=len(args.horizons),
        max_horizon=args.max_horizon,
        n_lags=pack["edge_seq_train"].shape[1],
        n_neighbours=pack["edge_seq_train"].shape[2],
        hidden=args.v4_hidden,
        heads=args.v4_heads,
        graph_layers=args.v4_graph_layers,
        cand_layers=args.v4_candidate_layers,
        dropout=args.v4_dropout,
        correction_scale=args.v4_correction_scale,
        context_router_pruner=args.v4_context_router_pruner,
        axis_graph_cross_attention=args.v5_axis_graph_cross_attention,
        axis_score_router=args.v5_axis_score_router,
        sequence_graph_refiner=args.v6_sequence_graph_refiner,
        v6_route_score_source=args.v6_route_score_source,
    ).to(device)
    target_soft = pack["target_soft_train"]
    target_rank = pack["target_rank_train"]
    target_end = pack["target_end_train"]
    target_steps = pack["target_steps_train"]
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 92001)
        perm = rng.permutation(len(target_soft))
        target_soft = target_soft[perm]
        target_rank = target_rank[perm]
        target_end = target_end[perm]
        target_steps = target_steps[perm]
    opt = torch.optim.AdamW(model.parameters(), lr=args.v4_lr, weight_decay=args.v4_weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack["steps_train"])
    for epoch in range(args.v4_epochs):
        model.train()
        losses = []
        parts_acc: list[dict[str, float]] = []
        for idx in closure.batches(n, args.v4_batch_size, args.seed + 93001 + epoch):
            out = forward_model(model, pack, "train", idx, device)
            loss, parts = v4_loss(
                out,
                to_tensor(target_end[idx], device),
                to_tensor(target_steps[idx], device),
                to_tensor(target_soft[idx], device),
                to_tensor(target_rank[idx], device),
                args.horizons,
                args,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.v4_clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            parts_acc.append(parts)
        if epoch == args.v4_epochs - 1 or epoch % max(1, args.v4_epochs // 5) == 0:
            pred_val = predict_v4(model, pack, "val", args, device)["pred"]
            val_rmse = float(np.sqrt(np.mean(np.sum((pred_val - pack["target_end_val"]) ** 2, axis=-1))))
            row = {"variant": variant, "epoch": int(epoch), "train_loss": float(np.mean(losses)), "val_endpoint_rmse": val_rmse}
            if parts_acc:
                for key in parts_acc[0]:
                    row[f"train_{key}"] = float(np.mean([p[key] for p in parts_acc]))
            rows.append(row)
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_val


def build_v4_pack(
    *,
    q_train: qrc.QueryOutputs,
    q_val: qrc.QueryOutputs,
    q_test: qrc.QueryOutputs,
    arrays: audit.SplitArrays,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    edge_memory: dict[str, np.ndarray],
    args: argparse.Namespace,
    use_context: bool = True,
    use_axis: bool = True,
    use_edge: bool = True,
    shuffle_context: bool = False,
    shuffle_edge: bool = False,
    shuffle_axis: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    steps_train, steps_val, steps_test, step_scaler = standardize_candidate_steps(q_train.query_pred, q_val.query_pred, q_test.query_pred)

    axis_train, _ = axis.axis_candidate_features(query_pred=q_train.query_pred, component_pred=component_axes.train, horizons=args.horizons)
    axis_val, _ = axis.axis_candidate_features(query_pred=q_val.query_pred, component_pred=component_axes.val, horizons=args.horizons)
    axis_test, axis_names = axis.axis_candidate_features(query_pred=q_test.query_pred, component_pred=component_axes.test, horizons=args.horizons)
    axis_train, axis_val, axis_test, axis_scaler = axis.standardize_axis_features(axis_train, axis_val, axis_test)
    if (not use_axis) or axis_train.shape[-1] == 0:
        axis_train = np.zeros_like(axis_train)
        axis_val = np.zeros_like(axis_val)
        axis_test = np.zeros_like(axis_test)
    if shuffle_axis:
        axis_train = maybe_shuffle(axis_train, args.seed + 94001, True)
        axis_val = maybe_shuffle(axis_val, args.seed + 94002, True)
        axis_test = maybe_shuffle(axis_test, args.seed + 94003, True)

    edge_train, edge_val, edge_test, own_train, own_val, own_test, edge_scaler = standardize_edge_memory(
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
        valid_train = np.zeros_like(valid_train)
        valid_val = np.zeros_like(valid_val)
        valid_test = np.zeros_like(valid_test)
        center_train = np.zeros_like(center_train)
        center_val = np.zeros_like(center_val)
        center_test = np.zeros_like(center_test)
    if shuffle_edge:
        edge_train = maybe_shuffle(edge_train, args.seed + 95001, True)
        own_train = maybe_shuffle(own_train, args.seed + 95001, True)
        valid_train = maybe_shuffle(valid_train, args.seed + 95001, True)
        center_train = maybe_shuffle(center_train, args.seed + 95001, True)
        edge_val = maybe_shuffle(edge_val, args.seed + 95002, True)
        own_val = maybe_shuffle(own_val, args.seed + 95002, True)
        valid_val = maybe_shuffle(valid_val, args.seed + 95002, True)
        center_val = maybe_shuffle(center_val, args.seed + 95002, True)
        edge_test = maybe_shuffle(edge_test, args.seed + 95003, True)
        own_test = maybe_shuffle(own_test, args.seed + 95003, True)
        valid_test = maybe_shuffle(valid_test, args.seed + 95003, True)
        center_test = maybe_shuffle(center_test, args.seed + 95003, True)

    ctx_train_use, ctx_val_use, ctx_test_use = ctx_train, ctx_val, ctx_test
    if not use_context:
        ctx_train_use = np.zeros_like(ctx_train_use)
        ctx_val_use = np.zeros_like(ctx_val_use)
        ctx_test_use = np.zeros_like(ctx_test_use)
    if shuffle_context:
        ctx_train_use = maybe_shuffle(ctx_train_use, args.seed + 96001, True)
        ctx_val_use = maybe_shuffle(ctx_val_use, args.seed + 96002, True)
        ctx_test_use = maybe_shuffle(ctx_test_use, args.seed + 96003, True)

    target_err_train = axis.target_candidate_errors(q_train.query_pred, arrays.residual_train, args.horizons)
    target_err_val = axis.target_candidate_errors(q_val.query_pred, arrays.residual_val, args.horizons)
    target_err_test = axis.target_candidate_errors(q_test.query_pred, arrays.residual_test, args.horizons)
    pack = {
        "steps_train": steps_train,
        "steps_val": steps_val,
        "steps_test": steps_test,
        "cand_res_train": q_train.query_pred.astype(np.float32),
        "cand_res_val": q_val.query_pred.astype(np.float32),
        "cand_res_test": q_test.query_pred.astype(np.float32),
        "axis_train": axis_train,
        "axis_val": axis_val,
        "axis_test": axis_test,
        "ctx_train": ctx_train_use.astype(np.float32),
        "ctx_val": ctx_val_use.astype(np.float32),
        "ctx_test": ctx_test_use.astype(np.float32),
        "edge_seq_train": edge_train,
        "edge_seq_val": edge_val,
        "edge_seq_test": edge_test,
        "own_seq_train": own_train,
        "own_seq_val": own_val,
        "own_seq_test": own_test,
        "valid_train": valid_train,
        "valid_val": valid_val,
        "valid_test": valid_test,
        "center_train": center_train,
        "center_val": center_val,
        "center_test": center_test,
        "cand_end_train": candidate_endpoints(q_train.query_pred, args.horizons),
        "cand_end_val": candidate_endpoints(q_val.query_pred, args.horizons),
        "cand_end_test": candidate_endpoints(q_test.query_pred, args.horizons),
        "target_steps_train": arrays.residual_train.astype(np.float32),
        "target_steps_val": arrays.residual_val.astype(np.float32),
        "target_steps_test": arrays.residual_test.astype(np.float32),
        "target_end_train": residual_endpoints(arrays.residual_train, args.horizons),
        "target_end_val": residual_endpoints(arrays.residual_val, args.horizons),
        "target_end_test": residual_endpoints(arrays.residual_test, args.horizons),
        "target_soft_train": axis.soft_labels_from_distance_np(target_err_train, args.v4_label_temperature),
        "target_soft_val": axis.soft_labels_from_distance_np(target_err_val, args.v4_label_temperature),
        "target_rank_train": -np.log1p(target_err_train).astype(np.float32),
        "target_rank_val": -np.log1p(target_err_val).astype(np.float32),
        "target_err_test": target_err_test,
    }
    meta = {
        "step_scaler": step_scaler,
        "axis_scaler": finite_json(axis_scaler),
        "axis_feature_names": axis_names,
        "edge_scaler": edge_scaler,
        "use_context": bool(use_context),
        "use_axis": bool(use_axis),
        "use_edge": bool(use_edge),
        "shuffle_context": bool(shuffle_context),
        "shuffle_edge": bool(shuffle_edge),
        "shuffle_axis": bool(shuffle_axis),
    }
    return pack, meta


def add_v4_variant_rows(
    *,
    rows: list[dict[str, Any]],
    logs: list[pd.DataFrame],
    diagnostics: list[dict[str, Any]],
    arrays: audit.SplitArrays,
    q_train: qrc.QueryOutputs,
    q_val: qrc.QueryOutputs,
    q_test: qrc.QueryOutputs,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    edge_memory: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    name: str,
    use_context: bool = True,
    use_axis: bool = True,
    use_edge: bool = True,
    shuffle_context: bool = False,
    shuffle_edge: bool = False,
    shuffle_axis: bool = False,
    shuffled_labels: bool = False,
    scalers: dict[str, Any] | None = None,
) -> None:
    pack, meta = build_v4_pack(
        q_train=q_train,
        q_val=q_val,
        q_test=q_test,
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        component_axes=component_axes,
        edge_memory=edge_memory,
        args=args,
        use_context=use_context,
        use_axis=use_axis,
        use_edge=use_edge,
        shuffle_context=shuffle_context,
        shuffle_edge=shuffle_edge,
        shuffle_axis=shuffle_axis,
    )
    if scalers is not None:
        scalers[name] = finite_json(meta)
    model, log, best_val = train_v4_variant(pack, args, device=device, variant=name, shuffled_labels=shuffled_labels)
    logs.append(log.assign(variant=name))
    pred = predict_v4(model, pack, "test", args, device)
    rows.extend(
        endpoint_rows_from_residual_endpoints(
            arrays=arrays,
            residual_endpoint_pred=pred["pred"],
            horizons=args.horizons,
            label=name,
            extra={"stage": "graph_memory_critic_v4", "variant": name, "val_endpoint_rmse": best_val},
        )
    )
    rows.extend(
        endpoint_rows_from_residual_endpoints(
            arrays=arrays,
            residual_endpoint_pred=pred["mixture"],
            horizons=args.horizons,
            label=f"{name}_mixture",
            extra={"stage": "graph_memory_critic_v4_mixture", "variant": name, "val_endpoint_rmse": best_val},
        )
    )
    top_idx = np.argmax(pred["logits"].mean(axis=1), axis=1)
    top_res = q_test.query_pred[np.arange(len(top_idx)), top_idx]
    rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=top_res,
            horizons=args.horizons,
            label=f"{name}_top",
            extra={"stage": "graph_memory_critic_v4_top", "variant": name, "val_endpoint_rmse": best_val},
        )
    )
    err = qrc.endpoint_errors(q_test.query_pred, arrays.residual_test, args.horizons)
    scores = pred["logits"].mean(axis=1)
    axis_component_gate = pred.get("axis_component_gate", np.ones((len(scores), 1, 1), dtype=np.float32))
    axis_component_entropy = -np.mean(
        np.sum(axis_component_gate * np.log(np.clip(axis_component_gate, 1e-8, 1.0)), axis=-1)
    )
    diagnostics.append(
        {
            "variant": name,
            "val_endpoint_rmse": float(best_val),
            "mean_step_correction_norm": float(np.mean(np.linalg.norm(pred["correction_steps"], axis=-1))),
            "mean_weight_entropy": float(axis.router_entropy_np(pred["weights"])),
            "risk_error_corr": float(qrc.risk_error_corr(-scores, err)),
            "mean_context_sample_gate": float(np.mean(pred.get("ctx_sample_gate", np.ones((len(scores), 1), dtype=np.float32)))),
            "mean_axis_graph_gate": float(np.mean(pred.get("axis_graph_gate", np.ones((len(scores), 1, 1), dtype=np.float32)))),
            "mean_axis_component_entropy": float(axis_component_entropy),
            "mean_axis_score_scale": float(np.mean(pred.get("axis_score_scale", np.zeros((len(scores),), dtype=np.float32)))),
            "mean_v6_refine_gate": float(np.mean(pred.get("v6_refine_gate", np.zeros((len(scores), 1, 1), dtype=np.float32)))),
            "mean_v6_refine_scale": float(np.mean(pred.get("v6_refine_scale", np.zeros((len(scores),), dtype=np.float32)))),
            "mean_v6_pair_entropy": float(np.mean(pred.get("v6_pair_entropy", np.zeros((len(scores),), dtype=np.float32)))),
            "mean_v6_route_score_scale": float(np.mean(pred.get("v6_route_score_scale", np.zeros((len(scores),), dtype=np.float32)))),
        }
    )
    write_partial_outputs(args, rows, logs, diagnostics)


def write_partial_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    logs: list[pd.DataFrame],
    diagnostics: list[dict[str, Any]],
) -> None:
    """Persist partial v4 outputs after every trained variant.

    Full v4/v4.1 runs can take a while because every seed often trains several
    controls.  Writing partial outputs makes long guards auditable even if a
    later variant is interrupted.
    """

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        partial = pd.DataFrame(rows)
        if "dataset" not in partial.columns:
            partial.insert(0, "seed", int(args.seed))
            partial.insert(0, "dataset", str(args.dataset))
        partial.to_csv(args.out_dir / "graph_memory_critic_v4_partial_summary.csv", index=False)
    if diagnostics:
        diag = pd.DataFrame(diagnostics)
        if "dataset" not in diag.columns:
            diag.insert(0, "seed", int(args.seed))
            diag.insert(0, "dataset", str(args.dataset))
        diag.to_csv(args.out_dir / "graph_memory_critic_v4_partial_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "graph_memory_critic_v4_partial_train_log.csv", index=False)


def v4_variant_enabled(args: argparse.Namespace, suffix: str) -> bool:
    raw = str(getattr(args, "v4_variant_list", "") or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    allowed = {part.strip() for part in raw.split(",") if part.strip()}
    return suffix in allowed


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    extra_feature_meta = rpr.attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    edge_memory, edge_memory_meta = load_edge_memory_cache(
        args.edge_sequence_cache,
        split,
        max_lags=args.v4_edge_max_lags,
        max_neighbours=args.v4_edge_max_neighbours,
        min_found_frac=args.v4_min_edge_found_frac,
    )
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_scaler = rpr.prepare_context(args, arrays, posterior, student, blocks, device)
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v4 requires at least two component axes")
    component_axes.probe.to_csv(args.out_dir / "graph_memory_v4_component_axis_probe.csv", index=False)

    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand_train = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    scalers: dict[str, Any] = {
        "context": finite_json(ctx_scaler),
        "extra_feature": finite_json(extra_feature_meta),
        "edge_memory": finite_json(edge_memory_meta),
    }
    rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=seq.mean_candidate_residual(cand_test),
            horizons=args.horizons,
            label="candidate_mean",
            extra={"stage": "candidate_control"},
        )
    )
    for k in args.oracle_k:
        rows.extend(
            audit.endpoint_metrics(
                steps_true=arrays.steps_test,
                base=arrays.base_test,
                residual_pred=rpr.proto.oracle_from_set(cand_test.residual[:, : int(k)], arrays.residual_test, args.horizons),
                horizons=args.horizons,
                label=f"candidate_endpoint_oracle@{k}",
                extra={"stage": "candidate_endpoint_oracle", "oracle_k": int(k)},
            )
        )

    methods = [s.strip() for s in str(args.prototype_methods).split(",") if s.strip()]
    counts = parse_ints(args.prototype_k)
    for method in methods:
        for m in counts:
            if int(m) > args.candidate_k:
                continue
            ptr, itr = rpr.build_prototype_set(cand_train, method, int(m), seed=args.seed + 97001)
            pva, iva = rpr.build_prototype_set(cand_val, method, int(m), seed=args.seed + 97002)
            pte, ite = rpr.build_prototype_set(cand_test, method, int(m), seed=args.seed + 97003)
            q_train = rpr.make_query_outputs(ptr, itr, method, args.candidate_k, arrays.residual_train, args.horizons)
            q_val = rpr.make_query_outputs(pva, iva, method, args.candidate_k, arrays.residual_val, args.horizons)
            q_test = rpr.make_query_outputs(pte, ite, method, args.candidate_k, arrays.residual_test, args.horizons)
            prefix = f"{method}{int(m)}"
            rows.extend(
                audit.endpoint_metrics(
                    steps_true=arrays.steps_test,
                    base=arrays.base_test,
                    residual_pred=q_test.query_oracle,
                    horizons=args.horizons,
                    label=f"{prefix}_oracle",
                    extra={"stage": "prototype_oracle", "prototype": method, "prototype_k": int(m)},
                )
            )
            rows.extend(
                audit.endpoint_metrics(
                    steps_true=arrays.steps_test,
                    base=arrays.base_test,
                    residual_pred=np.mean(q_test.query_pred, axis=1).astype(np.float32),
                    horizons=args.horizons,
                    label=f"{prefix}_mean",
                    extra={"stage": "prototype_mean", "prototype": method, "prototype_k": int(m)},
                )
            )
            if v4_variant_enabled(args, "full"):
                add_v4_variant_rows(
                    rows=rows,
                    logs=logs,
                    diagnostics=diagnostics,
                    arrays=arrays,
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    component_axes=component_axes,
                    edge_memory=edge_memory,
                    args=args,
                    device=device,
                    name=f"{prefix}_v4_full",
                    scalers=scalers,
                )
            if args.include_v4_ablations:
                if v4_variant_enabled(args, "no_edge_memory"):
                    add_v4_variant_rows(
                        rows=rows,
                        logs=logs,
                        diagnostics=diagnostics,
                        arrays=arrays,
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        edge_memory=edge_memory,
                        args=args,
                        device=device,
                        name=f"{prefix}_v4_no_edge_memory",
                        use_edge=False,
                        scalers=scalers,
                    )
                if v4_variant_enabled(args, "no_axis"):
                    add_v4_variant_rows(
                        rows=rows,
                        logs=logs,
                        diagnostics=diagnostics,
                        arrays=arrays,
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        edge_memory=edge_memory,
                        args=args,
                        device=device,
                        name=f"{prefix}_v4_no_axis",
                        use_axis=False,
                        scalers=scalers,
                    )
                if v4_variant_enabled(args, "no_context"):
                    add_v4_variant_rows(
                        rows=rows,
                        logs=logs,
                        diagnostics=diagnostics,
                        arrays=arrays,
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        edge_memory=edge_memory,
                        args=args,
                        device=device,
                        name=f"{prefix}_v4_no_context",
                        use_context=False,
                        scalers=scalers,
                    )
                if args.include_controls:
                    if v4_variant_enabled(args, "shuffled_edge_memory"):
                        add_v4_variant_rows(
                            rows=rows,
                            logs=logs,
                            diagnostics=diagnostics,
                            arrays=arrays,
                            q_train=q_train,
                            q_val=q_val,
                            q_test=q_test,
                            ctx_train=ctx_train,
                            ctx_val=ctx_val,
                            ctx_test=ctx_test,
                            component_axes=component_axes,
                            edge_memory=edge_memory,
                            args=args,
                            device=device,
                            name=f"{prefix}_v4_shuffled_edge_memory",
                            shuffle_edge=True,
                            scalers=scalers,
                        )
                    if v4_variant_enabled(args, "shuffled_context"):
                        add_v4_variant_rows(
                            rows=rows,
                            logs=logs,
                            diagnostics=diagnostics,
                            arrays=arrays,
                            q_train=q_train,
                            q_val=q_val,
                            q_test=q_test,
                            ctx_train=ctx_train,
                            ctx_val=ctx_val,
                            ctx_test=ctx_test,
                            component_axes=component_axes,
                            edge_memory=edge_memory,
                            args=args,
                            device=device,
                            name=f"{prefix}_v4_shuffled_context",
                            shuffle_context=True,
                            scalers=scalers,
                        )
                    if v4_variant_enabled(args, "shuffled_labels"):
                        add_v4_variant_rows(
                            rows=rows,
                            logs=logs,
                            diagnostics=diagnostics,
                            arrays=arrays,
                            q_train=q_train,
                            q_val=q_val,
                            q_test=q_test,
                            ctx_train=ctx_train,
                            ctx_val=ctx_val,
                            ctx_test=ctx_test,
                            component_axes=component_axes,
                            edge_memory=edge_memory,
                            args=args,
                            device=device,
                            name=f"{prefix}_v4_shuffled_labels",
                            shuffled_labels=True,
                            scalers=scalers,
                        )

    summary = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    if not summary.empty:
        summary.insert(0, "seed", int(args.seed))
        summary.insert(0, "dataset", str(args.dataset))
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "graph_memory_critic_v4_summary.csv", index=False)
    diag.to_csv(args.out_dir / "graph_memory_critic_v4_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "graph_memory_critic_v4_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(scalers), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = ["# Graph-Memory Critic-Refiner v4 Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- edge_sequence_cache: `{args.edge_sequence_cache}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- prototype_methods: `{args.prototype_methods}`")
    lines.append(f"- prototype_k: `{args.prototype_k}`")
    lines.append(f"- component_axis_blocks: `{args.component_axis_blocks}`")
    lines.append(f"- include_v4_ablations: `{args.include_v4_ablations}`")
    lines.append(f"- include_controls: `{args.include_controls}`")
    lines.append(f"- v5_axis_graph_cross_attention: `{args.v5_axis_graph_cross_attention}`")
    lines.append(f"- v5_axis_score_router: `{args.v5_axis_score_router}`")
    lines.append(f"- v6_sequence_graph_refiner: `{args.v6_sequence_graph_refiner}`")
    lines.append(f"- v6_route_score_source: `{args.v6_route_score_source}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{int(h)}")
        sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
        for _, row in sub.head(32).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, stage={row.get('stage', '')}")
    if not diag.empty:
        lines.append("\n## Diagnostics")
        for _, row in diag.sort_values("val_endpoint_rmse").head(32).iterrows():
            lines.append(
                f"- `{row['variant']}`: val={row['val_endpoint_rmse']:.3f}, "
                f"corr={row['mean_step_correction_norm']:.3f}, entropy={row['mean_weight_entropy']:.3f}, "
                f"risk_corr={row['risk_error_corr']:.3f}, "
                f"axis_graph_gate={row.get('mean_axis_graph_gate', float('nan')):.3f}, "
                f"axis_comp_entropy={row.get('mean_axis_component_entropy', float('nan')):.3f}, "
                f"axis_score_scale={row.get('mean_axis_score_scale', float('nan')):.3f}, "
                f"v6_gate={row.get('mean_v6_refine_gate', float('nan')):.3f}, "
                f"v6_scale={row.get('mean_v6_refine_scale', float('nan')):.3f}, "
                f"v6_pair_entropy={row.get('mean_v6_pair_entropy', float('nan')):.3f}, "
                f"v6_route_scale={row.get('mean_v6_route_score_scale', float('nan')):.3f}"
            )
    lines.append("\n## Decision Notes")
    lines.append("- Pass only if full beats no_edge_memory/no_axis/no_context and shuffled controls.")
    lines.append("- If no_edge_memory is competitive, edge observability is still not entering route choice.")
    lines.append("- If no_axis is much worse but full remains far from oracle, decomposition is useful but selector/observability remains the bottleneck.")
    (out_dir / "graph_memory_critic_v4_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.add_argument("--edge-sequence-cache", type=Path, required=True)
    parser.add_argument("--prototype-methods", type=str, default="fps_shape")
    parser.add_argument("--prototype-k", type=str, default="16")
    parser.add_argument("--include-v4-ablations", action="store_true")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--v4-variant-list", type=str, default="all")

    parser.add_argument("--component-axis-blocks", type=str, default="self,flow,all_context")
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
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="ef_")
    parser.add_argument("--extra-feature-block-name", type=str, default="explicit_edge")
    parser.add_argument("--extra-feature-max-cols", type=int, default=128)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")

    parser.add_argument("--v4-hidden", type=int, default=192)
    parser.add_argument("--v4-heads", type=int, default=4)
    parser.add_argument("--v4-graph-layers", type=int, default=2)
    parser.add_argument("--v4-candidate-layers", type=int, default=2)
    parser.add_argument("--v4-dropout", type=float, default=0.05)
    parser.add_argument("--v4-epochs", type=int, default=16)
    parser.add_argument("--v4-batch-size", type=int, default=256)
    parser.add_argument("--v4-lr", type=float, default=6e-4)
    parser.add_argument("--v4-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v4-correction-scale", type=float, default=0.75)
    parser.add_argument("--v4-label-temperature", type=float, default=8.0)
    parser.add_argument("--v4-reg-weight", type=float, default=1.0)
    parser.add_argument("--v4-listwise-weight", type=float, default=0.35)
    parser.add_argument("--v4-rank-weight", type=float, default=0.10)
    parser.add_argument("--v4-nll-weight", type=float, default=0.03)
    parser.add_argument("--v4-sequence-weight", type=float, default=0.20)
    parser.add_argument("--v4-accel-weight", type=float, default=0.05)
    parser.add_argument("--v4-correction-l2-weight", type=float, default=0.003)
    parser.add_argument("--v4-entropy-weight", type=float, default=0.002)
    parser.add_argument("--v4-context-router-pruner", action="store_true")
    parser.add_argument("--v4-context-pruner-l1-weight", type=float, default=0.0)
    parser.add_argument("--v4-logvar-min", type=float, default=-6.0)
    parser.add_argument("--v4-logvar-max", type=float, default=6.0)
    parser.add_argument("--v4-clip-grad", type=float, default=5.0)
    parser.add_argument("--v4-edge-max-lags", type=int, default=0)
    parser.add_argument("--v4-edge-max-neighbours", type=int, default=0)
    parser.add_argument("--v4-min-edge-found-frac", type=float, default=0.98)
    parser.add_argument("--v5-axis-graph-cross-attention", action="store_true")
    parser.add_argument("--v5-axis-score-router", action="store_true")
    parser.add_argument("--v6-sequence-graph-refiner", action="store_true")
    parser.add_argument("--v6-route-score-source", type=str, default="full", choices=["full", "pair_dyn"])
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    if args.horizons != sorted(set(args.horizons)):
        raise ValueError(f"--horizons must be sorted unique positive ints, got {args.horizons}")
    if max(args.horizons) > int(args.max_horizon):
        raise ValueError(f"max(horizons)={max(args.horizons)} exceeds --max-horizon={args.max_horizon}")
    if int(args.horizons[-1]) != int(args.max_horizon):
        raise ValueError(
            "v4 sequence loss expects the last reported horizon to equal --max-horizon; "
            f"got horizons={args.horizons}, max_horizon={args.max_horizon}"
        )
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1600)
        args.max_val_rows = min(args.max_val_rows, 600)
        args.max_test_rows = min(args.max_test_rows, 800)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.learned_route_epochs = min(args.learned_route_epochs, 3)
        args.v4_epochs = min(args.v4_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.prototype_k = "8"
        args.v4_hidden = min(args.v4_hidden, 96)
        args.v4_graph_layers = min(args.v4_graph_layers, 1)
        args.v4_candidate_layers = min(args.v4_candidate_layers, 1)
        args.v4_batch_size = min(args.v4_batch_size, 256)
        args.max_all_features = min(args.max_all_features, 192)
        args.max_critic_context_features = min(args.max_critic_context_features, 192)
        if args.v4_edge_max_lags == 0:
            args.v4_edge_max_lags = 4
        if args.v4_edge_max_neighbours == 0:
            args.v4_edge_max_neighbours = 8
    run(args)


if __name__ == "__main__":
    main()
