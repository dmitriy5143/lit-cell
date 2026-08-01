#!/usr/bin/env python3
"""v99: matched causal-online neural architecture benchmark for LaChance tracks.

Every model receives the same completed track history and is trained to issue an
h1 displacement before the next observation exists.  At evaluation time models
are replayed chronologically.  Their first-step predictions are accumulated for
h2/h4/h6 with the same implementation as v97.

The AgentFormer, MTR, and QCNet entries are cell-domain architectural adapters,
not executions of the official repositories.  The AgentFormer adapter has an
agent-aware spatiotemporal encoder and a CVAE posterior that may see the future
target during training only.  Validation and test always use its causal prior.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import TransformerConv


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_causal_innovation_state_space_v97 as v97  # noqa: E402
import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


KEYS = ["sequence", "frame", "track_id"]
DEFAULT_CACHE = ROOT / "outputs/joint_innovation_field_v84_anchor_cache_dense_bulk_2026-07-18"
DEFAULT_OUT = ROOT / "outputs/online_architecture_benchmark_v99_smoke"
MODEL_LABELS = {
    "temporal_gru": "Temporal GRU h1",
    "temporal_lstm": "Temporal LSTM h1",
    "temporal_transformer": "Temporal Transformer encoder h1",
    "pyg_transformerconv": "PyG TransformerConv h1",
    "social_lstm": "Social-LSTM/social-pooling h1 adapter",
    "agentformer_cell_adapter": "AgentFormer-style CVAE cell adapter (not official; no DLow stage)",
    "mtr_cell_adapter": "MTR-style query cell adapter (not official)",
    "qcnet_cell_adapter": "QCNet-style query-centric cell adapter (not official)",
}


def safe(value: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def parse_ints(value: str | Iterable[int]) -> list[int]:
    return [int(item) for item in parse_strings(value)]


def parse_horizon_weights(value: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for token in parse_strings(value):
        horizon, weight = token.split(":", 1)
        result[int(horizon)] = float(weight)
    if not result or sum(result.values()) <= 0:
        raise ValueError("validation horizon weights must be positive")
    return result


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def subset_bundle(bundle: v84.AnchorBundle, limit: int) -> v84.AnchorBundle:
    if limit <= 0 or len(bundle.rows) <= limit:
        return bundle
    selected: list[int] = []
    for _key, raw_indices in bundle.rows.groupby(["sequence", "track_id"], sort=True).groups.items():
        selected.extend(int(index) for index in raw_indices)
        if len(selected) >= limit:
            break
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_rows = bundle.rows.iloc[selected_array]
    local_order = np.lexsort(
        (
            selected_rows["track_id"].to_numpy(),
            selected_rows["frame"].to_numpy(),
            selected_rows["sequence"].to_numpy(),
        )
    )
    order = selected_array[local_order]
    return v84.AnchorBundle(
        name=bundle.name,
        rows=bundle.rows.iloc[order].reset_index(drop=True),
        anchor_residual=bundle.anchor_residual[order],
        base=bundle.base[order],
        target_steps=bundle.target_steps[order],
        meta={**bundle.meta, "v99_row_limit": int(limit)},
    )


@dataclass
class SplitArrays:
    history: np.ndarray
    neighbours: np.ndarray
    anchor: np.ndarray
    target: np.ndarray
    chain: np.ndarray
    observed_max_frame: np.ndarray


@dataclass
class BenchmarkData:
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle]
    splits: tuple[SplitArrays, SplitArrays, SplitArrays]
    history_scaler: StandardScaler
    neighbour_scaler: StandardScaler
    anchor_scaler: StandardScaler
    target_scaler: StandardScaler
    history_dim: int
    neighbour_dim: int
    history_lags: int
    neighbours_k: int


def row_lookup(bundle: v84.AnchorBundle) -> dict[tuple[int, int, int], int]:
    return {
        (int(sequence), int(frame), int(track)): index
        for index, (sequence, frame, track) in enumerate(
            bundle.rows[KEYS].itertuples(index=False, name=None)
        )
    }


def build_chain(bundle: v84.AnchorBundle, max_horizon: int) -> np.ndarray:
    lookup = row_lookup(bundle)
    result = np.full((len(bundle.rows), max_horizon), -1, dtype=np.int64)
    for index, (sequence, frame, track) in enumerate(
        bundle.rows[KEYS].itertuples(index=False, name=None)
    ):
        for offset in range(max_horizon):
            found = lookup.get((int(sequence), int(frame) + offset, int(track)))
            if found is None:
                break
            result[index, offset] = int(found)
    return result


def raw_history(bundle: v84.AnchorBundle, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Completed-transition history ending at the current observation."""
    rows = bundle.rows.reset_index(drop=True)
    lookup = row_lookup(bundle)
    values = np.zeros((len(rows), lags, 8), dtype=np.float32)
    max_frame = np.full(len(rows), -1, dtype=np.int64)
    for index, row in enumerate(rows.itertuples(index=False)):
        sequence = int(getattr(row, "sequence"))
        frame = int(getattr(row, "frame"))
        track = int(getattr(row, "track_id"))
        current_x = float(getattr(row, "x_px"))
        current_y = float(getattr(row, "y_px"))
        for slot, offset in enumerate(range(-(lags - 1), 1)):
            previous = lookup.get((sequence, frame + offset, track))
            if previous is None:
                continue
            source = rows.iloc[previous]
            dx = float(source["dx_px"])
            dy = float(source["dy_px"])
            values[index, slot] = (
                dx,
                dy,
                float(source["x_px"]) - current_x,
                float(source["y_px"]) - current_y,
                math.hypot(dx, dy),
                float(offset) / max(lags - 1, 1),
                1.0 if offset == 0 else 0.0,
                1.0,
            )
            max_frame[index] = max(max_frame[index], frame + offset)
    return values, max_frame


def dense_neighbour_history(bundle: v84.AnchorBundle, lags: int, k: int) -> np.ndarray:
    rows = bundle.rows.reset_index(drop=True)
    lookup = row_lookup(bundle)
    output = np.zeros((len(rows), k, lags, 8), dtype=np.float32)
    groups = rows.groupby(["sequence", "frame"], sort=False).groups
    for (_sequence, _frame), raw_indices in groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        xy = rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float32)
        if len(indices) <= 1:
            continue
        distance = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(distance, np.inf)
        nearest = np.argsort(distance, axis=1)[:, : min(k, len(indices) - 1)]
        for focal_local, focal_index in enumerate(indices):
            focal_row = rows.iloc[focal_index]
            sequence = int(focal_row["sequence"])
            frame = int(focal_row["frame"])
            focal_track = int(focal_row["track_id"])
            for rank, neighbour_local in enumerate(nearest[focal_local]):
                neighbour_index = int(indices[neighbour_local])
                neighbour_track = int(rows.iloc[neighbour_index]["track_id"])
                for slot, offset in enumerate(range(-(lags - 1), 1)):
                    focal_past = lookup.get((sequence, frame + offset, focal_track))
                    neighbour_past = lookup.get((sequence, frame + offset, neighbour_track))
                    if focal_past is None or neighbour_past is None:
                        continue
                    focal_source = rows.iloc[focal_past]
                    source = rows.iloc[neighbour_past]
                    rel_x = float(source["x_px"] - focal_source["x_px"])
                    rel_y = float(source["y_px"] - focal_source["y_px"])
                    dx = float(source["dx_px"])
                    dy = float(source["dy_px"])
                    output[focal_index, rank, slot] = (
                        rel_x,
                        rel_y,
                        dx,
                        dy,
                        math.hypot(dx, dy),
                        math.hypot(rel_x, rel_y),
                        float(offset) / max(lags - 1, 1),
                        1.0,
                    )
    return output


def fit_masked_scaler(values: np.ndarray, mask_index: int) -> StandardScaler:
    flat = values.reshape(-1, values.shape[-1])
    valid = flat[:, mask_index] > 0.5
    if not np.any(valid):
        raise RuntimeError("no valid causal history tokens")
    return StandardScaler().fit(flat[valid, :mask_index])


def apply_masked_scaler(values: np.ndarray, scaler: StandardScaler, mask_index: int) -> np.ndarray:
    result = values.copy().astype(np.float32)
    flat = result.reshape(-1, result.shape[-1])
    mask = flat[:, mask_index] > 0.5
    flat[mask, :mask_index] = np.clip(scaler.transform(flat[mask, :mask_index]), -8.0, 8.0)
    flat[~mask, :mask_index] = 0.0
    return result


def prepare_data(args: argparse.Namespace) -> BenchmarkData:
    bundles_raw = v85.load_anchor_cache(args.anchor_cache)
    limits = (args.train_rows, args.val_rows, args.test_rows)
    bundles = tuple(subset_bundle(bundle, limit) for bundle, limit in zip(bundles_raw, limits))
    histories: list[np.ndarray] = []
    neighbours: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for bundle in bundles:
        history, max_frame = raw_history(bundle, args.history_lags)
        histories.append(history)
        neighbours.append(dense_neighbour_history(bundle, args.history_lags, args.neighbours_k))
        observed.append(max_frame)
    history_scaler = fit_masked_scaler(histories[0], 7)
    neighbour_scaler = fit_masked_scaler(neighbours[0], 7)
    histories = [apply_masked_scaler(value, history_scaler, 7) for value in histories]
    neighbours = [apply_masked_scaler(value, neighbour_scaler, 7) for value in neighbours]

    anchor_scaler = StandardScaler().fit(bundles[0].anchor_steps[:, 0])
    target_scaler = StandardScaler().fit(bundles[0].target_steps[:, 0])
    arrays: list[SplitArrays] = []
    for split, bundle in enumerate(bundles):
        arrays.append(
            SplitArrays(
                history=safe(histories[split]),
                neighbours=safe(neighbours[split]),
                anchor=np.clip(anchor_scaler.transform(bundle.anchor_steps[:, 0]), -8.0, 8.0).astype(np.float32),
                target=target_scaler.transform(bundle.target_steps[:, 0]).astype(np.float32),
                chain=build_chain(bundle, max(args.horizons)),
                observed_max_frame=observed[split],
            )
        )
    return BenchmarkData(
        bundles=(bundles[0], bundles[1], bundles[2]),
        splits=(arrays[0], arrays[1], arrays[2]),
        history_scaler=history_scaler,
        neighbour_scaler=neighbour_scaler,
        anchor_scaler=anchor_scaler,
        target_scaler=target_scaler,
        history_dim=histories[0].shape[-1],
        neighbour_dim=neighbours[0].shape[-1],
        history_lags=args.history_lags,
        neighbours_k=args.neighbours_k,
    )


class TemporalTransformerH1(nn.Module):
    def __init__(self, history_dim: int, hidden: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(history_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position = nn.Parameter(torch.randn(1, 64, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden))
        self.anchor = nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, history: torch.Tensor, neighbours: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.token(history) + self.position[:, : history.shape[1]]
        cls = self.cls.expand(len(history), -1, -1) + self.anchor(anchor)[:, None]
        padding = torch.cat([torch.zeros((len(history), 1), dtype=torch.bool, device=history.device), history[:, :, 7] < 0.5], dim=1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1), src_key_padding_mask=padding)
        return self.out(encoded[:, 0]), {}


class TemporalRNNH1(nn.Module):
    """Matched-capacity causal recurrent baseline over the focal-cell history."""

    def __init__(
        self,
        kind: str,
        history_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kind not in {"gru", "lstm"}:
            raise ValueError(f"Unknown recurrent kind: {kind}")
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent(
            history_dim,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.anchor = nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        history: torch.Tensor,
        neighbours: torch.Tensor,
        anchor: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del neighbours, target
        output, _state = self.recurrent(history)
        # Prepared rows are right-aligned at the issue frame. Invalid early
        # padding, when present, therefore cannot replace the final causal state.
        state = output[:, -1] + self.anchor(anchor)
        return self.out(state), {}


class SocialLSTMH1(nn.Module):
    def __init__(self, history_dim: int, neighbour_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.focal = nn.LSTM(history_dim, hidden, batch_first=True)
        self.neighbour = nn.LSTM(neighbour_dim, hidden, batch_first=True)
        self.social_message = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.anchor = nn.Linear(2, hidden)
        self.out = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, history: torch.Tensor, neighbours: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, (focal, _) = self.focal(history)
        b, k, t, d = neighbours.shape
        _, (encoded, _) = self.neighbour(neighbours.reshape(b * k, t, d))
        encoded = encoded[-1].reshape(b, k, -1)
        relative = neighbours[:, :, -1, :2]
        message = self.social_message(torch.cat([encoded, relative], dim=-1))
        valid = neighbours[:, :, :, 7].amax(dim=2) > 0.5
        message = message.masked_fill(~valid[:, :, None], -1e4)
        pooled = message.max(dim=1).values
        pooled = torch.where(valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        state = focal[-1] + self.anchor(anchor)
        return self.out(torch.cat([state, pooled], dim=-1)), {}


class PyGTransformerConvH1(nn.Module):
    def __init__(self, history_dim: int, neighbour_dim: int, hidden: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.focal = nn.GRU(history_dim, hidden, batch_first=True)
        self.neighbour = nn.GRU(neighbour_dim, hidden, batch_first=True)
        self.anchor = nn.Linear(2, hidden)
        self.convs = nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, concat=True, beta=True, edge_dim=neighbour_dim, dropout=dropout)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, history: torch.Tensor, neighbours: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, k, t, d = neighbours.shape
        _, focal = self.focal(history)
        _, neighbour = self.neighbour(neighbours.reshape(b * k, t, d))
        nodes = torch.cat([focal[-1, :, None, :] + self.anchor(anchor)[:, None], neighbour[-1].reshape(b, k, self.hidden)], dim=1)
        nodes = nodes.reshape(b * (k + 1), self.hidden)
        base = torch.arange(b, device=history.device) * (k + 1)
        src_parts: list[torch.Tensor] = []
        dst_parts: list[torch.Tensor] = []
        attrs: list[torch.Tensor] = []
        for rank in range(k):
            neighbour_index = base + rank + 1
            valid = neighbours[:, rank, :, 7].amax(dim=1) > 0.5
            src_parts.extend([neighbour_index[valid], base[valid]])
            dst_parts.extend([base[valid], neighbour_index[valid]])
            edge = neighbours[valid, rank, -1]
            reverse = edge.clone()
            reverse[:, :2] *= -1
            attrs.extend([edge, reverse])
        if src_parts and sum(len(part) for part in src_parts):
            edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
            edge_attr = torch.cat(attrs, dim=0)
            for conv, norm in zip(self.convs, self.norms):
                nodes = norm(nodes + F.gelu(conv(nodes, edge_index, edge_attr)))
        central = nodes.reshape(b, k + 1, self.hidden)[:, 0]
        return self.out(central), {}


class AgentSceneEncoder(nn.Module):
    def __init__(self, history_dim: int, neighbour_dim: int, hidden: int, heads: int, layers: int, lags: int, k: int, dropout: float) -> None:
        super().__init__()
        self.focal = nn.Linear(history_dim, hidden)
        self.neighbour = nn.Linear(neighbour_dim, hidden)
        self.time = nn.Parameter(torch.randn(1, 1, lags, hidden) * 0.02)
        self.agent = nn.Parameter(torch.randn(1, k + 1, 1, hidden) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden))
        self.anchor = nn.Linear(2, hidden)

    def forward(self, history: torch.Tensor, neighbours: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k, t, _ = neighbours.shape
        focal = self.focal(history)[:, None] + self.time[:, :, :t] + self.agent[:, :1]
        other = self.neighbour(neighbours) + self.time[:, :, :t] + self.agent[:, 1 : k + 1]
        agents = torch.cat([focal, other], dim=1)
        tokens = agents.reshape(b, (k + 1) * t, -1)
        valid = torch.cat([history[:, None, :, 7], neighbours[:, :, :, 7]], dim=1).reshape(b, -1) > 0.5
        cls = self.cls.expand(b, -1, -1) + self.anchor(anchor)[:, None]
        padding = torch.cat([torch.zeros((b, 1), dtype=torch.bool, device=history.device), ~valid], dim=1)
        memory = self.encoder(torch.cat([cls, tokens], dim=1), src_key_padding_mask=padding)
        return memory, memory[:, 0], padding


class AgentFormerCellAdapter(nn.Module):
    """AgentFormer-style online CVAE adapter; not the official two-stage model.

    The primary v99 table needs a matched h1 model, so this adapter retains
    agent-aware spatiotemporal attention and the train-only future posterior,
    but it does not reproduce AgentFormer's separate DLow sampler.  The latter
    belongs in the fixed-origin probabilistic benchmark.
    """

    def __init__(self, history_dim: int, neighbour_dim: int, hidden: int, heads: int, layers: int, lags: int, k: int, dropout: float, latent: int) -> None:
        super().__init__()
        self.scene = AgentSceneEncoder(history_dim, neighbour_dim, hidden, heads, layers, lags, k, dropout)
        self.prior_mu = nn.Linear(hidden, latent)
        self.prior_logvar = nn.Linear(hidden, latent)
        self.future = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.posterior_mu = nn.Linear(hidden * 2, latent)
        self.posterior_logvar = nn.Linear(hidden * 2, latent)
        self.out = nn.Sequential(nn.Linear(hidden + latent, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, history: torch.Tensor, neighbours: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, scene, _ = self.scene(history, neighbours, anchor)
        prior_mu = self.prior_mu(scene)
        prior_logvar = torch.clamp(self.prior_logvar(scene), -5.0, 3.0)
        if self.training and target is not None:
            future = self.future(target)
            posterior_input = torch.cat([scene, future], dim=-1)
            posterior_mu = self.posterior_mu(posterior_input)
            posterior_logvar = torch.clamp(self.posterior_logvar(posterior_input), -5.0, 3.0)
            latent = posterior_mu + torch.randn_like(posterior_mu) * torch.exp(0.5 * posterior_logvar)
            kl = 0.5 * torch.mean(
                prior_logvar - posterior_logvar
                + (torch.exp(posterior_logvar) + (posterior_mu - prior_mu).square()) / torch.exp(prior_logvar).clamp_min(1e-6)
                - 1.0
            )
        else:
            latent = prior_mu
            kl = scene.new_zeros(())
        return self.out(torch.cat([scene, latent], dim=-1)), {"kl": kl}


class QueryCellAdapter(nn.Module):
    """Query-based MTR/QCNet-style cell adapter; not an official implementation."""

    def __init__(self, kind: str, history_dim: int, neighbour_dim: int, hidden: int, heads: int, layers: int, lags: int, k: int, dropout: float, modes: int) -> None:
        super().__init__()
        self.kind = kind
        self.scene = AgentSceneEncoder(history_dim, neighbour_dim, hidden, heads, layers, lags, k, dropout)
        self.queries = nn.Parameter(torch.randn(1, modes, hidden) * 0.03)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.trajectory = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.logit = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, history: torch.Tensor, neighbours: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory, scene, padding = self.scene(history, neighbours, anchor)
        query = self.queries.expand(len(history), -1, -1) + scene[:, None]
        cross, _ = self.cross(self.norm(query), self.norm(memory), self.norm(memory), key_padding_mask=padding, need_weights=False)
        query = query + cross
        if self.kind == "qcnet":
            refined, _ = self.self_attention(self.norm(query), self.norm(query), self.norm(query), need_weights=False)
            query = query + refined
        modes = self.trajectory(query)
        logits = self.logit(query).squeeze(-1)
        probability = torch.softmax(logits, dim=-1)
        mean = torch.sum(probability[:, :, None] * modes, dim=1)
        aux: dict[str, torch.Tensor] = {"modes": modes, "logits": logits}
        return mean, aux


def build_model(name: str, data: BenchmarkData, args: argparse.Namespace) -> nn.Module:
    common = (data.history_dim, data.neighbour_dim, args.hidden)
    if name == "temporal_gru":
        return TemporalRNNH1("gru", data.history_dim, args.hidden, args.layers, args.dropout)
    if name == "temporal_lstm":
        return TemporalRNNH1("lstm", data.history_dim, args.hidden, args.layers, args.dropout)
    if name == "temporal_transformer":
        return TemporalTransformerH1(data.history_dim, args.hidden, args.heads, args.layers, args.dropout)
    if name == "pyg_transformerconv":
        return PyGTransformerConvH1(*common, args.heads, args.layers, args.dropout)
    if name == "social_lstm":
        return SocialLSTMH1(*common, args.dropout)
    if name == "agentformer_cell_adapter":
        return AgentFormerCellAdapter(*common, args.heads, args.layers, data.history_lags, data.neighbours_k, args.dropout, args.latent_dim)
    if name == "mtr_cell_adapter":
        return QueryCellAdapter("mtr", *common, args.heads, args.layers, data.history_lags, data.neighbours_k, args.dropout, args.modes)
    if name == "qcnet_cell_adapter":
        return QueryCellAdapter("qcnet", *common, args.heads, args.layers, data.history_lags, data.neighbours_k, args.dropout, args.modes)
    raise ValueError(f"unknown model {name!r}")


def tensors_for_indices(data: BenchmarkData, split: int, indices: np.ndarray, input_variant: str, device: torch.device) -> tuple[torch.Tensor, ...]:
    arrays = data.splits[split]
    anchor = arrays.anchor[indices].copy()
    if input_variant == "raw_coordinate":
        anchor[:] = 0.0
    elif input_variant != "v52_anchor":
        raise ValueError(f"unknown input variant {input_variant!r}")
    return (
        torch.from_numpy(arrays.history[indices]).to(device),
        torch.from_numpy(arrays.neighbours[indices]).to(device),
        torch.from_numpy(anchor).to(device),
        torch.from_numpy(arrays.target[indices]).to(device),
    )


def auxiliary_terms(mean: torch.Tensor, aux: dict[str, torch.Tensor], target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    result = mean.new_zeros(())
    if "kl" in aux:
        result = result + args.kl_weight * aux["kl"]
    if "modes" in aux:
        modes = aux["modes"]
        logits = aux["logits"]
        distance = torch.sum((modes - target[:, None]).square(), dim=-1)
        best = torch.argmin(distance.detach(), dim=1)
        chosen = modes[torch.arange(len(target), device=target.device), best]
        pairwise = torch.cdist(modes, modes)
        off_diagonal = 1.0 - torch.eye(modes.shape[1], device=modes.device)[None]
        diversity = (pairwise * off_diagonal).sum() / off_diagonal.sum().clamp_min(1.0) / max(len(modes), 1)
        result = result + args.query_best_weight * F.smooth_l1_loss(chosen, target)
        result = result + args.query_ce_weight * F.cross_entropy(logits, best)
        result = result - args.query_diversity_weight * torch.clamp(diversity, max=5.0)
    return result


def physical_prediction(data: BenchmarkData, normalized: np.ndarray) -> np.ndarray:
    return data.target_scaler.inverse_transform(normalized).astype(np.float32)


def replay_prediction(model: nn.Module, data: BenchmarkData, split: int, input_variant: str, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    bundle = data.bundles[split]
    result = np.zeros((len(bundle.rows), 2), dtype=np.float32)
    assigned = np.zeros(len(bundle.rows), dtype=bool)
    posterior_future_calls = 0
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for (_sequence, _frame), raw_indices in bundle.rows.groupby(["sequence", "frame"], sort=True).groups.items():
            frame_indices = np.asarray(list(raw_indices), dtype=np.int64)
            for start in range(0, len(frame_indices), args.eval_batch_size):
                indices = frame_indices[start : start + args.eval_batch_size]
                history, neighbours, anchor, _target = tensors_for_indices(data, split, indices, input_variant, device)
                mean, _aux = model(history, neighbours, anchor, None)
                result[indices] = physical_prediction(data, mean.detach().cpu().numpy())
                assigned[indices] = True
    elapsed = time.perf_counter() - started
    frames = bundle.rows["frame"].to_numpy(np.int64)
    violations = int(np.sum(data.splits[split].observed_max_frame > frames))
    audit = {
        "rows": int(len(bundle.rows)),
        "predicted_rows": int(assigned.sum()),
        "coverage": float(assigned.mean()) if len(assigned) else 0.0,
        "history_after_issue_violations": violations,
        "future_target_in_inference_calls": posterior_future_calls,
        "predict_before_observe": bool(violations == 0 and assigned.all()),
        "elapsed_sec": elapsed,
        "ms_per_cell": 1000.0 * elapsed / max(len(bundle.rows), 1),
    }
    return result, audit


def validation_score(data: BenchmarkData, prediction: np.ndarray, weights: dict[int, float]) -> float:
    return v97.weighted_rolling_score(data.bundles[1], prediction, weights)


def cumulative_training_loss(model: nn.Module, data: BenchmarkData, starts: np.ndarray, input_variant: str, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.cumulative_weight <= 0 or len(starts) == 0:
        return torch.zeros((), device=device)
    chain = data.splits[0].chain[starts]
    terms: list[torch.Tensor] = []
    target_scale = torch.as_tensor(data.target_scaler.scale_, dtype=torch.float32, device=device)
    target_mean = torch.as_tensor(data.target_scaler.mean_, dtype=torch.float32, device=device)
    for horizon in args.cumulative_horizons:
        valid = np.all(chain[:, :horizon] >= 0, axis=1)
        if not np.any(valid):
            continue
        indices = chain[valid, :horizon].reshape(-1)
        history, neighbours, anchor, target = tensors_for_indices(data, 0, indices, input_variant, device)
        mean, _ = model(history, neighbours, anchor, target)
        mean_physical = mean * target_scale + target_mean
        target_physical = target * target_scale + target_mean
        mean_sum = mean_physical.reshape(-1, horizon, 2).sum(dim=1)
        target_sum = target_physical.reshape(-1, horizon, 2).sum(dim=1)
        terms.append(F.smooth_l1_loss(mean_sum / target_scale, target_sum / target_scale))
    return torch.stack(terms).mean() if terms else torch.zeros((), device=device)


def train_one(name: str, input_variant: str, seed: int, data: BenchmarkData, args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    seed_everything(seed)
    model = build_model(name, data, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.arange(len(data.bundles[0].rows), dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = -1
    patience = args.patience
    logs: list[dict[str, Any]] = []
    started = time.perf_counter()
    weights = parse_horizon_weights(args.validation_horizon_weights)
    rng = np.random.default_rng(seed + 991)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        for (index_tensor,) in loader:
            indices = index_tensor.numpy().astype(np.int64)
            history, neighbours, anchor, target = tensors_for_indices(data, 0, indices, input_variant, device)
            optimizer.zero_grad(set_to_none=True)
            mean, aux = model(history, neighbours, anchor, target)
            loss = F.smooth_l1_loss(mean, target) + auxiliary_terms(mean, aux, target, args)
            if args.cumulative_weight > 0:
                candidates = indices[np.all(data.splits[0].chain[indices] >= 0, axis=1)]
                if len(candidates):
                    starts = rng.choice(candidates, size=min(args.auxiliary_batch_size, len(candidates)), replace=False)
                    loss = loss + args.cumulative_weight * cumulative_training_loss(model, data, starts, input_variant, args, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        validation_prediction, _ = replay_prediction(model, data, 1, input_variant, args, device)
        score = validation_score(data, validation_prediction, weights)
        logs.append({"model": name, "input_variant": input_variant, "seed": seed, "epoch": epoch + 1, "train_loss": total / max(seen, 1), "validation_score": score})
        print(f"[v99] {name}/{input_variant}/seed{seed} epoch={epoch + 1} loss={total/max(seen,1):.5f} val={score:.5f}", flush=True)
        if score < best_score - args.min_delta:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is None:
        raise RuntimeError(f"{name}/{input_variant} did not produce a checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    metadata = {
        "model": name,
        "display_name": MODEL_LABELS[name],
        "input_variant": input_variant,
        "seed": seed,
        "parameters": parameter_count(model),
        "train_elapsed_sec": elapsed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "official_model": False if "adapter" in name else None,
    }
    return model, pd.DataFrame(logs), metadata


def save_checkpoint(model: nn.Module, path: Path, metadata: dict[str, Any], data: BenchmarkData, args: argparse.Namespace) -> None:
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata,
            "architecture_args": {key: getattr(args, key) for key in ("hidden", "heads", "layers", "dropout", "latent_dim", "modes", "history_lags", "neighbours_k")},
            "scalers": {
                "history_mean": data.history_scaler.mean_, "history_scale": data.history_scaler.scale_,
                "neighbour_mean": data.neighbour_scaler.mean_, "neighbour_scale": data.neighbour_scaler.scale_,
                "anchor_mean": data.anchor_scaler.mean_, "anchor_scale": data.anchor_scaler.scale_,
                "target_mean": data.target_scaler.mean_, "target_scale": data.target_scaler.scale_,
            },
        },
        path,
    )


def data_contract_rows(data: BenchmarkData, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, bundle in zip(("train", "validation", "test"), data.bundles):
        rows.append(
            {
                "split": split_name,
                "bundle": bundle.name,
                "sequences": ",".join(str(int(value)) for value in sorted(bundle.rows.sequence.unique())),
                "rows": len(bundle.rows),
                "tracks": bundle.rows[["sequence", "track_id"]].drop_duplicates().shape[0],
                "frame_min": int(bundle.rows.frame.min()),
                "frame_max": int(bundle.rows.frame.max()),
                "history_lags": args.history_lags,
                "neighbours_k": args.neighbours_k,
                "scaling": "train_only",
                "target": "next displacement h1",
                "contract": "predict before next observation; rolling first-step accumulation",
            }
        )
    return rows


def write_report(args: argparse.Namespace, metrics: pd.DataFrame, metadata: pd.DataFrame, audit: pd.DataFrame, contract: pd.DataFrame) -> None:
    h1 = metrics[metrics.horizon.eq(1)].sort_values("component_rmse")
    h6 = metrics[metrics.horizon.eq(6)].sort_values("component_rmse")
    lines = [
        "# v99 Causal-Online Architecture Benchmark",
        "",
        "## Contract",
        "",
        "- Every prediction is an h1 displacement issued before the next observation.",
        "- h2/h4/h6 are sums of consecutive first-step predictions from chronological replay.",
        "- Every architecture uses the same history length, neighbour budget, movie split, train-only scaling, and validation objective.",
        "- `agentformer_cell_adapter`, `mtr_cell_adapter`, and `qcnet_cell_adapter` are domain adaptations, not official repository runs; the AgentFormer row does not include its separate DLow stage.",
        "- AgentFormer posterior sees the next displacement during training only; validation/test call the causal prior.",
        "",
        "## Smoke Result" if args.smoke else "## Result",
        "",
        f"- Evaluated models: `{len(metadata)}`.",
        f"- Test prediction coverage: `{audit.coverage.min():.6f}` to `{audit.coverage.max():.6f}`.",
        f"- Total causal violations: `{int(audit.history_after_issue_violations.sum() + audit.future_target_in_inference_calls.sum())}`.",
        "",
        "### h1",
        "",
        h1[["method", "component_rmse", "r2", "cosine", "magnitude_ratio", "n_rows"]].to_markdown(index=False),
        "",
        "### rolling h6",
        "",
        h6[["method", "component_rmse", "r2", "cosine", "magnitude_ratio", "n_rows"]].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Smoke mode verifies implementation and causal equivalence; it is not a ranking claim. Full comparisons require matched seeds and the same frozen movie-level evaluation protocol.",
    ]
    (args.out_dir / "v99_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.patience = 1
        args.hidden = min(args.hidden, 64)
        args.layers = min(args.layers, 1)
        args.train_rows = args.train_rows or 1200
        args.val_rows = args.val_rows or 400
        args.test_rows = args.test_rows or 600
        args.auxiliary_batch_size = min(args.auxiliary_batch_size, 8)
    args.horizons = parse_ints(args.horizons)
    args.cumulative_horizons = parse_ints(args.cumulative_horizons)
    models = parse_strings(args.models)
    inputs = parse_strings(args.input_variants)
    seeds = parse_ints(args.seeds)
    unknown = [name for name in models if name not in MODEL_LABELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if not set(inputs).issubset({"raw_coordinate", "v52_anchor"}):
        raise ValueError(f"unknown input variants: {inputs}")
    device = choose_device(args.device)
    print(f"[v99] device={device} preparing data", flush=True)
    data = prepare_data(args)
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    archive: dict[str, np.ndarray] = {}
    for seed in seeds:
        for input_variant in inputs:
            for name in models:
                model, train_log, metadata = train_one(name, input_variant, seed, data, args, device)
                key = f"{name}__{input_variant}__seed{seed}"
                checkpoint = args.out_dir / "checkpoints" / f"{key}.pt"
                save_checkpoint(model, checkpoint, metadata, data, args)
                validation_prediction, validation_audit = replay_prediction(model, data, 1, input_variant, args, device)
                test_prediction, test_audit = replay_prediction(model, data, 2, input_variant, args, device)
                archive[f"validation__{key}"] = validation_prediction
                archive[f"test__{key}"] = test_prediction
                method = key
                metric_rows.extend(v97.rolling_metric_rows(data.bundles[2], test_prediction, args.horizons, method, {"model": name, "input_variant": input_variant, "seed": seed, "selection": "validation_only"}))
                for split_name, replay_audit in (("validation", validation_audit), ("test", test_audit)):
                    audit_rows.append({"method": method, "split": split_name, **replay_audit})
                metadata_rows.append({**metadata, "checkpoint": str(checkpoint), "validation_ms_per_cell": validation_audit["ms_per_cell"], "test_ms_per_cell": test_audit["ms_per_cell"], "device": str(device)})
                logs.append(train_log)
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
    metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)
    metadata = pd.DataFrame(metadata_rows)
    train_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    contract = pd.DataFrame(data_contract_rows(data, args))
    metrics.to_csv(args.out_dir / "v99_online_summary.csv", index=False)
    audit.to_csv(args.out_dir / "v99_causal_audit.csv", index=False)
    metadata.to_csv(args.out_dir / "v99_model_metadata.csv", index=False)
    train_log.to_csv(args.out_dir / "v99_train_log.csv", index=False)
    contract.to_csv(args.out_dir / "v99_data_contract.csv", index=False)
    np.savez_compressed(args.out_dir / "v99_predictions.npz", **archive)
    config = {**vars(args), "anchor_cache": str(args.anchor_cache), "out_dir": str(args.out_dir), "device_resolved": str(device)}
    (args.out_dir / "v99_config.json").write_text(json.dumps(finite(config), indent=2), encoding="utf-8")
    write_report(args, metrics, metadata, audit, contract)
    status = {
        "ok": True,
        "models": len(metadata),
        "causal_violations": int(audit.history_after_issue_violations.sum() + audit.future_target_in_inference_calls.sum()),
        "minimum_test_coverage": float(audit[audit.split.eq("test")].coverage.min()),
        "smoke": bool(args.smoke),
    }
    (args.out_dir / "v99_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print((args.out_dir / "v99_benchmark_report.md").read_text(encoding="utf-8"), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", default=",".join(MODEL_LABELS))
    parser.add_argument("--input-variants", default="raw_coordinate,v52_anchor")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--cumulative-horizons", default="2,4,6")
    parser.add_argument("--validation-horizon-weights", default="1:0.90,2:0.05,4:0.03,6:0.02")
    parser.add_argument("--history-lags", type=int, default=8)
    parser.add_argument("--neighbours-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--auxiliary-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--cumulative-weight", type=float, default=0.10)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--query-best-weight", type=float, default=0.20)
    parser.add_argument("--query-ce-weight", type=float, default=0.05)
    parser.add_argument("--query-diversity-weight", type=float, default=0.002)
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=0)
    parser.add_argument("--test-rows", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    try:
        run(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"ok": False, "error": repr(error), "traceback": traceback.format_exc(), "elapsed_sec": time.time() - started}
        (args.out_dir / "v99_error.json").write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
