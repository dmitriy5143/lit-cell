#!/usr/bin/env python3
"""v96: online graph state-space Seq2Seq correction over the dense v52 anchor.

The runner treats already observed one-step v52 errors as measurements of a
hidden cell-specific interaction state.  A recurrent predict/update block keeps
that state by track id, a current-frame graph exchanges filtered states between
neighbours, and an autoregressive decoder emits six residual steps.  Open-loop
and receding-h1 metrics are reported under separate contracts.

Target/future residuals are used only in losses and evaluation.  Every
inference measurement is an error from a forecast issued at an earlier frame.
Training anchor errors must therefore come from the existing movie-held-out OOF
cache produced by v84/v85.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


EPS = 1e-8
KEYS = ["sequence", "frame", "track_id"]
DEFAULT_OUT = ROOT / "outputs" / "graph_state_space_seq2seq_v96_2026-07-20"


@dataclass(frozen=True)
class Variant:
    name: str
    use_graph: bool = True
    use_update: bool = True
    use_context: bool = True
    shuffled_measurement: bool = False
    shuffled_edges: bool = False
    shuffled_edge_attr: bool = False
    use_edge_measurement: bool = True
    graph_operator: str = "transformer"
    use_frame_pool: bool = False
    use_history: bool = False
    shuffled_history: bool = False
    use_history_innovation: bool = True


@dataclass
class FrameSpec:
    sequence: int
    frame: int
    indices: np.ndarray
    track_ids: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


@dataclass
class Prepared:
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle]
    static: tuple[np.ndarray, np.ndarray, np.ndarray]
    history: tuple[np.ndarray, np.ndarray, np.ndarray]
    measurement: tuple[np.ndarray, np.ndarray, np.ndarray]
    measurement_mask: tuple[np.ndarray, np.ndarray, np.ndarray]
    targets: tuple[np.ndarray, np.ndarray, np.ndarray]
    anchor_decoder: tuple[np.ndarray, np.ndarray, np.ndarray]
    frames: tuple[list[FrameSpec], list[FrameSpec], list[FrameSpec]]
    error_scaler: StandardScaler
    anchor_scaler: StandardScaler
    static_scaler: StandardScaler
    context_names: list[str]
    static_dim: int


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def safe(value: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_context_quotas(value: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        prefix, count = token.rsplit(":", 1)
        quotas[prefix.strip()] = int(count)
    return quotas


def choose_context_columns(path: Path, quotas: dict[str, int]) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    preferred_tokens = (
        "flow",
        "velocity",
        "speed",
        "align",
        "coherence",
        "div",
        "curl",
        "shear",
        "density",
        "crowd",
        "boundary",
        "front_back",
        "orient",
        "elong",
        "centroid",
        "quality",
        "contact",
        "free",
        "delta",
        "grad",
    )
    selected: list[str] = []
    for prefix, quota in quotas.items():
        family = [c for c in columns if c.startswith(prefix)]
        preferred = [c for c in family if any(token in c.lower() for token in preferred_tokens)]
        fallback = [c for c in family if c not in preferred]
        selected.extend((preferred + fallback)[: max(int(quota), 0)])
    return list(dict.fromkeys(selected))


def load_context(
    path: Path,
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    quotas: dict[str, int],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[str], list[dict[str, Any]]]:
    selected = choose_context_columns(path, quotas)
    usecols = set(KEYS + selected)
    table = pd.read_csv(path, usecols=lambda c: c in usecols)
    for key in KEYS:
        table[key] = table[key].astype(int)
    table = table.drop_duplicates(KEYS)
    arrays: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for bundle in bundles:
        merged = bundle.rows[KEYS].merge(table, on=KEYS, how="left", validate="one_to_one")
        values = merged[selected].to_numpy(np.float32) if selected else np.zeros((len(merged), 0), np.float32)
        missing = float(np.mean(~np.isfinite(values))) if values.size else 0.0
        coverage = float(np.mean(np.any(np.isfinite(values), axis=1))) if values.size else 1.0
        arrays.append(safe(values))
        diagnostics.append({"split": bundle.name, "rows": len(bundle.rows), "context_features": len(selected), "context_missing_fraction": missing, "context_row_coverage": coverage})
    return (arrays[0], arrays[1], arrays[2]), selected, diagnostics


def previous_innovation(bundle: v84.AnchorBundle) -> tuple[np.ndarray, np.ndarray]:
    rows = bundle.rows.reset_index(drop=True)
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(rows[KEYS].itertuples(index=False, name=None))
    }
    values = np.zeros((len(rows), 2), dtype=np.float32)
    mask = np.zeros(len(rows), dtype=np.float32)
    one_step_error = bundle.errors[:, 0].astype(np.float32)
    for i, (sequence, frame, track) in enumerate(rows[KEYS].itertuples(index=False, name=None)):
        previous = lookup.get((int(sequence), int(frame) - 1, int(track)))
        if previous is None:
            continue
        values[i] = one_step_error[previous]
        mask[i] = 1.0
    return values, mask


def shuffle_measurement_within_frame(
    values: np.ndarray,
    mask: np.ndarray,
    rows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    shuffled_mask = mask.copy()
    for _, raw_indices in rows.groupby(["sequence", "frame"], sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        permutation = rng.permutation(indices)
        shuffled[indices] = values[permutation]
        shuffled_mask[indices] = mask[permutation]
    return shuffled, shuffled_mask


def shuffle_rows_within_frame(values: np.ndarray, rows: pd.DataFrame, seed: int) -> np.ndarray:
    shuffled = values.copy()
    rng = np.random.default_rng(seed)
    for _, raw_indices in rows.groupby(["sequence", "frame"], sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        shuffled[indices] = values[rng.permutation(indices)]
    return shuffled


def frame_scale(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 1.0
    tree = cKDTree(xy)
    distance, _ = tree.query(xy, k=2)
    nearest = distance[:, 1]
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    return max(float(np.median(positive)) if len(positive) else 1.0, 1.0)


def make_frames(
    bundle: v84.AnchorBundle,
    measurement: np.ndarray,
    measurement_mask: np.ndarray,
    velocity_scale: np.ndarray,
    k: int,
    shuffled_edges: bool,
    shuffled_edge_attr: bool,
    use_edge_measurement: bool,
    seed: int,
) -> list[FrameSpec]:
    frames: list[FrameSpec] = []
    velocity = bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    for frame_no, ((sequence, frame), raw_indices) in enumerate(bundle.rows.groupby(["sequence", "frame"], sort=True).groups.items()):
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        xy = bundle.rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float32)
        tracks = bundle.rows.iloc[indices]["track_id"].to_numpy(np.int64)
        if len(indices) > 1:
            scale = frame_scale(xy)
            xy_norm = (xy - np.mean(xy, axis=0, keepdims=True)) / scale
            tree = cKDTree(xy_norm)
            kk = min(int(k) + 1, len(indices))
            distance, neighbours = tree.query(xy_norm, k=kk)
            if neighbours.ndim == 1:
                neighbours = neighbours[:, None]
                distance = distance[:, None]
            dst = np.repeat(np.arange(len(indices), dtype=np.int64), max(0, kk - 1))
            src = neighbours[:, 1:kk].reshape(-1).astype(np.int64)
            dist = distance[:, 1:kk].reshape(-1).astype(np.float32)
            rel = xy_norm[src] - xy_norm[dst]
            local_velocity = velocity[indices] / np.maximum(velocity_scale[None], 1e-4)
            relative_velocity = local_velocity[src] - local_velocity[dst]
            local_measurement = measurement[indices] * measurement_mask[indices, None]
            measurement_delta = local_measurement[src] - local_measurement[dst]
            pair_mask = (measurement_mask[indices][src] * measurement_mask[indices][dst])[:, None]
            edge_attr = np.concatenate(
                [rel, dist[:, None], relative_velocity, measurement_delta, pair_mask],
                axis=1,
            ).astype(np.float32)
            if not use_edge_measurement:
                edge_attr[:, 5:8] = 0.0
            if shuffled_edges and len(src):
                rng = np.random.default_rng(seed + frame_no)
                src = rng.permutation(src)
            if shuffled_edge_attr and len(src):
                rng = np.random.default_rng(seed + 50000 + frame_no)
                edge_attr = edge_attr[rng.permutation(len(edge_attr))]
            edge_index = np.stack([src, dst], axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 8), dtype=np.float32)
        frames.append(
            FrameSpec(
                sequence=int(sequence),
                frame=int(frame),
                indices=indices,
                track_ids=tracks,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
        )
    return frames


def prepare(
    args: argparse.Namespace,
    variant: Variant,
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    context: tuple[np.ndarray, np.ndarray, np.ndarray],
    context_names: list[str],
) -> Prepared:
    train, val, test = bundles
    train_xy = train.rows[["x_px", "y_px"]].to_numpy(np.float32)
    xy_low = np.percentile(train_xy, 1, axis=0)
    xy_high = np.percentile(train_xy, 99, axis=0)
    frame_max = float(train.rows["frame"].max())
    anchor_state = [v85.anchor_state(bundle, xy_low, xy_high - xy_low, frame_max)[0] for bundle in bundles]
    if variant.use_context:
        static_raw = [safe(np.concatenate([anchor, extra], axis=1)) for anchor, extra in zip(anchor_state, context)]
    else:
        static_raw = [safe(anchor) for anchor in anchor_state]
    static_scaler = StandardScaler().fit(static_raw[0])
    static = tuple(np.clip(static_scaler.transform(values), -8.0, 8.0).astype(np.float32) for values in static_raw)

    measurement_raw: list[np.ndarray] = []
    measurement_masks: list[np.ndarray] = []
    for split_no, bundle in enumerate(bundles):
        values, mask = previous_innovation(bundle)
        if variant.shuffled_measurement:
            values, mask = shuffle_measurement_within_frame(values, mask, bundle.rows, int(args.seed) + 900 + split_no)
        if not variant.use_update:
            values = np.zeros_like(values)
            mask = np.zeros_like(mask)
        measurement_raw.append(values)
        measurement_masks.append(mask)

    scaler_source = (
        train.errors[:, 0]
        if bool(getattr(args, "one_step_scaler", False))
        else train.errors.reshape(-1, 2)
    )
    error_scaler = StandardScaler().fit(scaler_source)
    measurement = tuple(error_scaler.transform(values).astype(np.float32) for values in measurement_raw)
    targets = tuple(error_scaler.transform(bundle.errors.reshape(-1, 2)).reshape(len(bundle.rows), 6, 2).astype(np.float32) for bundle in bundles)
    velocity_scaler = StandardScaler().fit(train.rows[["dx_px", "dy_px"]].to_numpy(np.float32))
    history_values: list[np.ndarray] = []
    for split_no, bundle in enumerate(bundles):
        raw_history, history_mask, _ = v85.temporal_history(bundle, int(args.history_lags))
        normalized_error = error_scaler.transform(raw_history[:, :, :2].reshape(-1, 2)).reshape(raw_history.shape[0], raw_history.shape[1], 2)
        normalized_velocity = velocity_scaler.transform(raw_history[:, :, 2:].reshape(-1, 2)).reshape(raw_history.shape[0], raw_history.shape[1], 2)
        if not variant.use_history_innovation:
            normalized_error[:] = 0.0
        token_history = np.concatenate(
            [normalized_error, normalized_velocity, history_mask[:, :, None]],
            axis=2,
        ).astype(np.float32)
        token_history[:, :, :4] *= history_mask[:, :, None]
        if variant.shuffled_history:
            token_history = shuffle_rows_within_frame(
                token_history,
                bundle.rows,
                int(args.seed) + 7000 + split_no,
            )
        history_values.append(safe(token_history))
    anchor_scaler = StandardScaler().fit(train.anchor_steps.reshape(-1, 2))
    anchor_decoder = tuple(anchor_scaler.transform(bundle.anchor_steps.reshape(-1, 2)).reshape(len(bundle.rows), 6, 2).astype(np.float32) for bundle in bundles)
    velocity_scale = np.std(train.rows[["dx_px", "dy_px"]].to_numpy(np.float32), axis=0)
    frames = tuple(
        make_frames(
            bundle,
            measurement_values,
            mask,
            velocity_scale,
            int(args.graph_k),
            variant.shuffled_edges,
            variant.shuffled_edge_attr,
            variant.use_edge_measurement,
            int(args.seed) + 1200 + split_no * 100,
        )
        for split_no, (bundle, measurement_values, mask) in enumerate(zip(bundles, measurement, measurement_masks))
    )
    return Prepared(
        bundles=bundles,
        static=static,
        history=(history_values[0], history_values[1], history_values[2]),
        measurement=measurement,
        measurement_mask=(measurement_masks[0], measurement_masks[1], measurement_masks[2]),
        targets=targets,
        anchor_decoder=anchor_decoder,
        frames=(frames[0], frames[1], frames[2]),
        error_scaler=error_scaler,
        anchor_scaler=anchor_scaler,
        static_scaler=static_scaler,
        context_names=context_names if variant.use_context else [],
        static_dim=static[0].shape[1],
    )


class StructuredConsensus(nn.Module):
    """Difference-based neighbour update with learned edge gates."""

    def __init__(self, hidden: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden),
            nn.Sigmoid(),
        )
        self.state_message = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.zeros_like(state)
        source, destination = edge_index[0], edge_index[1]
        difference = state[source] - state[destination]
        message = self.edge_gate(edge_attr) * self.state_message(difference)
        aggregated = torch.zeros_like(state)
        aggregated.index_add_(0, destination, message)
        degree = torch.zeros((len(state), 1), dtype=state.dtype, device=state.device)
        degree.index_add_(0, destination, torch.ones((len(destination), 1), dtype=state.dtype, device=state.device))
        return aggregated / torch.clamp(degree, min=1.0)


class EdgeFieldConditioner(nn.Module):
    """Aggregate causal relative fields without importing neighbour identity."""

    def __init__(self, hidden: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_message = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.zeros_like(state)
        destination = edge_index[1]
        message = self.edge_message(edge_attr)
        aggregated = torch.zeros_like(state)
        aggregated.index_add_(0, destination, message)
        degree = torch.zeros((len(state), 1), dtype=state.dtype, device=state.device)
        degree.index_add_(0, destination, torch.ones((len(destination), 1), dtype=state.dtype, device=state.device))
        return aggregated / torch.clamp(degree, min=1.0)


class GraphStateSpaceSeq2Seq(nn.Module):
    def __init__(
        self,
        static_dim: int,
        hidden: int,
        heads: int,
        graph_layers: int,
        horizon_dim: int,
        use_graph: bool,
        use_update: bool,
        graph_operator: str,
        use_frame_pool: bool,
        use_history: bool,
        history_lags: int,
        history_layers: int,
        correction_bound: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.hidden = int(hidden)
        self.use_graph = bool(use_graph)
        self.use_update = bool(use_update)
        self.graph_operator = str(graph_operator)
        self.use_frame_pool = bool(use_frame_pool)
        self.use_history = bool(use_history)
        self.history_lags = int(history_lags)
        self.correction_bound = float(correction_bound)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        if self.use_history:
            self.history_token = nn.Sequential(nn.Linear(5, hidden), nn.LayerNorm(hidden), nn.SiLU())
            self.history_cls = nn.Parameter(torch.zeros(1, 1, hidden))
            self.history_position = nn.Parameter(torch.randn(1, self.history_lags + 1, hidden) * 0.02)
            history_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 3,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.history_encoder = nn.TransformerEncoder(
                history_layer,
                num_layers=int(history_layers),
                norm=nn.LayerNorm(hidden),
            )
            self.history_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            self.history_fuse_norm = nn.LayerNorm(hidden)
        self.initial_state = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.transition = nn.GRUCell(hidden, hidden)
        self.measurement_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 2))
        self.update_encoder = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gain = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(hidden)
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        self.graph_gates = nn.ModuleList()
        if self.use_graph:
            for _ in range(int(graph_layers)):
                if self.graph_operator == "transformer":
                    self.graph_convs.append(
                        TransformerConv(
                            hidden,
                            hidden // heads,
                            heads=heads,
                            concat=True,
                            beta=True,
                            edge_dim=8,
                            dropout=dropout,
                        )
                    )
                elif self.graph_operator == "consensus":
                    self.graph_convs.append(StructuredConsensus(hidden, 8, dropout))
                elif self.graph_operator == "edge_field":
                    self.graph_convs.append(EdgeFieldConditioner(hidden, 8, dropout))
                else:
                    raise ValueError(f"Unknown graph operator: {self.graph_operator}")
                self.graph_norms.append(nn.LayerNorm(hidden))
                self.graph_gates.append(nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid()))
            self.raw_graph_strength = nn.Parameter(torch.tensor(-2.0))
        if self.use_frame_pool:
            self.frame_pool_encoder = nn.Sequential(
                nn.Linear(hidden * 2 + 5, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.frame_pool_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            self.frame_pool_norm = nn.LayerNorm(hidden)
            self.raw_pool_strength = nn.Parameter(torch.tensor(-2.0))
        self.horizon_embedding = nn.Embedding(6, horizon_dim)
        self.decoder = nn.GRUCell(4 + horizon_dim, hidden)
        self.mean_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.logscale_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 2))
        self.raw_df = nn.Parameter(torch.tensor(1.5))

    @property
    def degrees_of_freedom(self) -> torch.Tensor:
        return F.softplus(self.raw_df) + 2.1

    def forward_frame(
        self,
        static: torch.Tensor,
        history: torch.Tensor,
        measurement: torch.Tensor,
        measurement_mask: torch.Tensor,
        previous_state: torch.Tensor,
        has_previous_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        anchor_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.static_encoder(static)
        if self.use_history:
            history_tokens = self.history_token(history)
            cls = self.history_cls.expand(len(static), -1, -1)
            sequence = torch.cat([cls, history_tokens], dim=1) + self.history_position
            padding = history[:, :, 4] < 0.5
            padding = torch.cat(
                [torch.zeros((len(static), 1), dtype=torch.bool, device=static.device), padding],
                dim=1,
            )
            history_state = self.history_encoder(sequence, src_key_padding_mask=padding)[:, 0]
            history_gate = self.history_gate(torch.cat([encoded, history_state], dim=-1))
            encoded = self.history_fuse_norm(encoded + history_gate * history_state)
        initialized = self.initial_state(encoded)
        recurrent_input = torch.where(has_previous_state[:, None] > 0.5, previous_state, initialized)
        prior = self.transition(encoded, recurrent_input)
        predicted_measurement = self.measurement_head(prior)
        innovation = measurement - predicted_measurement
        update = self.update_encoder(torch.cat([innovation, measurement, measurement_mask[:, None]], dim=-1))
        gain = self.gain(torch.cat([prior, encoded, innovation, measurement_mask[:, None]], dim=-1))
        if self.use_update:
            posterior = self.update_norm(prior + measurement_mask[:, None] * gain * update)
        else:
            posterior = prior
            gain = torch.zeros_like(gain)
        if self.use_graph and edge_index.shape[1] > 0:
            for conv, norm, gate_layer in zip(self.graph_convs, self.graph_norms, self.graph_gates):
                graph_update = F.silu(conv(posterior, edge_index, edge_attr))
                graph_gate = gate_layer(torch.cat([posterior, graph_update], dim=-1))
                graph_strength = torch.sigmoid(self.raw_graph_strength)
                posterior = norm(posterior + graph_strength * graph_gate * graph_update)
        if self.use_frame_pool:
            state_mean = posterior.mean(dim=0)
            state_std = torch.sqrt(torch.mean(torch.square(posterior - state_mean), dim=0) + 1e-6)
            observed_count = torch.clamp(measurement_mask.sum(), min=1.0)
            measurement_mean = torch.sum(measurement * measurement_mask[:, None], dim=0) / observed_count
            centered_measurement = (measurement - measurement_mean) * measurement_mask[:, None]
            measurement_std = torch.sqrt(torch.sum(torch.square(centered_measurement), dim=0) / observed_count + 1e-6)
            observation_fraction = measurement_mask.mean().reshape(1)
            pooled = self.frame_pool_encoder(
                torch.cat([state_mean, state_std, measurement_mean, measurement_std, observation_fraction], dim=0)[None]
            ).expand(len(posterior), -1)
            pool_gate = self.frame_pool_gate(torch.cat([posterior, pooled], dim=-1))
            pool_strength = torch.sigmoid(self.raw_pool_strength)
            posterior = self.frame_pool_norm(posterior + pool_strength * pool_gate * pooled)

        decoder_state = posterior
        previous_correction = torch.zeros((len(static), 2), dtype=static.dtype, device=static.device)
        means: list[torch.Tensor] = []
        logscales: list[torch.Tensor] = []
        for horizon in range(6):
            horizon_ids = torch.full((len(static),), horizon, dtype=torch.long, device=static.device)
            decoder_input = torch.cat(
                [anchor_steps[:, horizon], previous_correction, self.horizon_embedding(horizon_ids)],
                dim=-1,
            )
            decoder_state = self.decoder(decoder_input, decoder_state)
            raw_mean = self.mean_head(decoder_state)
            mean = self.correction_bound * torch.tanh(raw_mean / max(self.correction_bound, 1e-4))
            logscale = torch.clamp(self.logscale_head(decoder_state), -4.5, 2.5)
            means.append(mean)
            logscales.append(logscale)
            previous_correction = mean
        return posterior, torch.stack(means, dim=1), torch.stack(logscales, dim=1), predicted_measurement, gain


def frame_tensors(
    prep: Prepared,
    split: int,
    frame: FrameSpec,
    device: torch.device,
    measurement_stride: int = 1,
) -> tuple[torch.Tensor, ...]:
    idx = frame.indices
    mask = prep.measurement_mask[split][idx].copy()
    if int(measurement_stride) > 1 and (frame.frame % int(measurement_stride)) != 0:
        mask[:] = 0.0
    return (
        torch.from_numpy(prep.static[split][idx]).to(device),
        torch.from_numpy(prep.history[split][idx]).to(device),
        torch.from_numpy(prep.measurement[split][idx]).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(prep.targets[split][idx]).to(device),
        torch.from_numpy(prep.anchor_decoder[split][idx]).to(device),
        torch.from_numpy(frame.edge_index).to(device),
        torch.from_numpy(frame.edge_attr).to(device),
    )


def gather_state(
    frame: FrameSpec,
    state_cache: dict[int, tuple[int, torch.Tensor]],
    hidden: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous: list[torch.Tensor] = []
    present: list[float] = []
    zero = torch.zeros(hidden, dtype=torch.float32, device=device)
    for track in frame.track_ids:
        cached = state_cache.get(int(track))
        if cached is not None and cached[0] == frame.frame - 1:
            previous.append(cached[1])
            present.append(1.0)
        else:
            previous.append(zero)
            present.append(0.0)
    return torch.stack(previous, dim=0), torch.tensor(present, dtype=torch.float32, device=device)


def store_state(frame: FrameSpec, posterior: torch.Tensor, state_cache: dict[int, tuple[int, torch.Tensor]]) -> None:
    for local_index, track in enumerate(frame.track_ids):
        state_cache[int(track)] = (frame.frame, posterior[local_index])


def objective(
    model: GraphStateSpaceSeq2Seq,
    mean: torch.Tensor,
    logscale: torch.Tensor,
    target: torch.Tensor,
    predicted_measurement: torch.Tensor,
    measurement: torch.Tensor,
    measurement_mask: torch.Tensor,
    gain: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    step_weights = torch.tensor([1.0, 0.85, 0.70, 0.58, 0.50, 0.45], device=mean.device)[None, :, None]
    step_huber = torch.mean(F.smooth_l1_loss(mean, target, reduction="none") * step_weights)
    endpoint_weights = {1: 1.0, 2: 0.75, 4: 0.55, 6: 0.45}
    cumulative_mean = torch.cumsum(mean, dim=1)
    cumulative_target = torch.cumsum(target, dim=1)
    endpoint_terms = [
        weight
        * F.smooth_l1_loss(
            cumulative_mean[:, horizon - 1].contiguous(),
            cumulative_target[:, horizon - 1].contiguous(),
        )
        for horizon, weight in endpoint_weights.items()
    ]
    endpoint_loss = torch.stack(endpoint_terms).mean()
    scale = torch.exp(logscale).clamp_min(1e-3)
    distribution = torch.distributions.StudentT(model.degrees_of_freedom, loc=mean, scale=scale)
    nll = -torch.mean(distribution.log_prob(target) * step_weights)
    if torch.any(measurement_mask > 0.5):
        measurement_error = F.smooth_l1_loss(predicted_measurement, measurement, reduction="none").mean(dim=1)
        measurement_loss = torch.sum(measurement_error * measurement_mask) / torch.clamp(measurement_mask.sum(), min=1.0)
    else:
        measurement_loss = mean.new_tensor(0.0)
    temporal_loss = F.smooth_l1_loss(mean[:, 1:] - mean[:, :-1], target[:, 1:] - target[:, :-1])
    gain_penalty = torch.mean(torch.square(gain))
    loss = (
        step_huber
        + float(args.endpoint_weight) * endpoint_loss
        + float(args.nll_weight) * nll
        + float(args.measurement_weight) * measurement_loss
        + float(args.temporal_weight) * temporal_loss
        + float(args.gain_weight) * gain_penalty
    )
    return loss, {
        "step_huber": float(step_huber.detach().cpu()),
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "measurement_loss": float(measurement_loss.detach().cpu()),
        "temporal_loss": float(temporal_loss.detach().cpu()),
        "gain_mean": float(gain.detach().mean().cpu()),
    }


def sequence_frames(frames: list[FrameSpec]) -> dict[int, list[FrameSpec]]:
    grouped: dict[int, list[FrameSpec]] = {}
    for frame in frames:
        grouped.setdefault(frame.sequence, []).append(frame)
    for values in grouped.values():
        values.sort(key=lambda item: item.frame)
    return grouped


def train_epoch(
    model: GraphStateSpaceSeq2Seq,
    prep: Prepared,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    grouped = sequence_frames(prep.frames[0])
    rng = np.random.default_rng(int(args.seed) + epoch)
    sequence_order = list(grouped)
    rng.shuffle(sequence_order)
    records: list[dict[str, float]] = []
    for sequence in sequence_order:
        state_cache: dict[int, tuple[int, torch.Tensor]] = {}
        optimizer.zero_grad(set_to_none=True)
        chunk_losses: list[torch.Tensor] = []
        for frame_no, frame in enumerate(grouped[sequence]):
            static, history, measurement, mask, target, anchor, edge_index, edge_attr = frame_tensors(prep, 0, frame, device)
            previous, has_previous = gather_state(frame, state_cache, model.hidden, device)
            posterior, mean, logscale, predicted_measurement, gain = model.forward_frame(
                static,
                history,
                measurement,
                mask,
                previous,
                has_previous,
                edge_index,
                edge_attr,
                anchor,
            )
            loss, details = objective(model, mean, logscale, target, predicted_measurement, measurement, mask, gain, args)
            chunk_losses.append(loss)
            records.append(details)
            store_state(frame, posterior, state_cache)
            boundary = len(chunk_losses) >= int(args.tbptt_frames) or frame_no == len(grouped[sequence]) - 1
            if boundary:
                chunk_loss = torch.stack(chunk_losses).mean()
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state_cache = {track: (last_frame, state.detach()) for track, (last_frame, state) in state_cache.items()}
                chunk_losses = []
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]} if records else {}


@torch.no_grad()
def infer(
    model: GraphStateSpaceSeq2Seq,
    prep: Prepared,
    split: int,
    device: torch.device,
    measurement_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    n_rows = len(prep.bundles[split].rows)
    mean = np.zeros((n_rows, 6, 2), dtype=np.float32)
    logscale = np.zeros_like(mean)
    predicted_measurement = np.zeros((n_rows, 2), dtype=np.float32)
    gain = np.zeros(n_rows, dtype=np.float32)
    grouped = sequence_frames(prep.frames[split])
    for sequence in sorted(grouped):
        state_cache: dict[int, tuple[int, torch.Tensor]] = {}
        for frame in grouped[sequence]:
            static, history, measurement, mask, _target, anchor, edge_index, edge_attr = frame_tensors(
                prep,
                split,
                frame,
                device,
                measurement_stride=measurement_stride,
            )
            previous, has_previous = gather_state(frame, state_cache, model.hidden, device)
            posterior, frame_mean, frame_logscale, frame_measurement, frame_gain = model.forward_frame(
                static,
                history,
                measurement,
                mask,
                previous,
                has_previous,
                edge_index,
                edge_attr,
                anchor,
            )
            indices = frame.indices
            mean[indices] = frame_mean.detach().cpu().numpy()
            logscale[indices] = frame_logscale.detach().cpu().numpy()
            predicted_measurement[indices] = frame_measurement.detach().cpu().numpy()
            gain[indices] = frame_gain.detach().mean(dim=1).cpu().numpy()
            store_state(frame, posterior, state_cache)
            state_cache = {track: (last_frame, state.detach()) for track, (last_frame, state) in state_cache.items()}
    return mean, logscale, predicted_measurement, gain


def decode_correction(prep: Prepared, normalized: np.ndarray) -> np.ndarray:
    return prep.error_scaler.inverse_transform(normalized.reshape(-1, 2)).reshape(normalized.shape).astype(np.float32)


def validation_score(bundle: v84.AnchorBundle, correction: np.ndarray) -> float:
    prediction = bundle.anchor_steps + correction
    horizon_weights = {1: 0.40, 2: 0.25, 4: 0.20, 6: 0.15}
    values = []
    weights = []
    for horizon, weight in horizon_weights.items():
        target = bundle.target_steps[:, :horizon].sum(axis=1)
        pred = prediction[:, :horizon].sum(axis=1)
        values.append(float(np.mean(np.square(target - pred))))
        weights.append(weight)
    return float(math.sqrt(np.average(values, weights=weights)))


def train_variant(
    prep: Prepared,
    variant: Variant,
    args: argparse.Namespace,
) -> tuple[GraphStateSpaceSeq2Seq, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    # Matched ablations share initialization. Architectures with different
    # tensor shapes still initialize independently where shapes diverge.
    torch.manual_seed(int(args.seed))
    model = GraphStateSpaceSeq2Seq(
        static_dim=prep.static_dim,
        hidden=int(args.hidden),
        heads=int(args.heads),
        graph_layers=int(args.graph_layers),
        horizon_dim=int(args.horizon_dim),
        use_graph=variant.use_graph,
        use_update=variant.use_update,
        graph_operator=variant.graph_operator,
        use_frame_pool=variant.use_frame_pool,
        use_history=variant.use_history,
        history_lags=int(args.history_lags),
        history_layers=int(args.history_layers),
        correction_bound=float(args.correction_bound),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    patience = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = train_epoch(model, prep, optimizer, device, args, epoch)
        val_mean, _val_logscale, _val_measurement, _val_gain = infer(model, prep, 1, device)
        val_correction = decode_correction(prep, val_mean)
        score = validation_score(prep.bundles[1], val_correction)
        row = {"variant": variant.name, "epoch": epoch, "val_weighted_endpoint_rmse": score, **train_stats}
        logs.append(row)
        print(f"[v96] {variant.name} epoch={epoch:02d} val={score:.6f}", flush=True)
        if np.isfinite(score) and score < best_score - float(args.min_delta):
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint for {variant.name}")
    model.load_state_dict(best_state)
    model.to(device)
    val_mean, _val_logscale, _val_measurement, _val_gain = infer(model, prep, 1, device)
    test_mean, test_logscale, test_measurement, test_gain = infer(model, prep, 2, device)
    val_correction = decode_correction(prep, val_mean)
    test_correction = decode_correction(prep, test_mean)
    test_scale = np.exp(test_logscale) * prep.error_scaler.scale_[None, None, :]
    metadata = {
        "best_epoch": best_epoch,
        "best_val_weighted_endpoint_rmse": best_score,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "degrees_of_freedom": float(model.degrees_of_freedom.detach().cpu()),
        "gain_mean": float(np.mean(test_gain)),
        "gain_p10": float(np.quantile(test_gain, 0.10)),
        "gain_p90": float(np.quantile(test_gain, 0.90)),
        "measurement_prediction_rmse_normalized": float(np.sqrt(np.mean(np.square(test_measurement - prep.measurement[2])))),
        "graph_operator": variant.graph_operator if variant.use_graph else "none",
        "graph_strength": (
            float(torch.sigmoid(model.raw_graph_strength).detach().cpu()) if variant.use_graph else 0.0
        ),
        "use_frame_pool": bool(variant.use_frame_pool),
        "pool_strength": (
            float(torch.sigmoid(model.raw_pool_strength).detach().cpu()) if variant.use_frame_pool else 0.0
        ),
        "use_history": bool(variant.use_history),
        "shuffled_history": bool(variant.shuffled_history),
        "use_history_innovation": bool(variant.use_history_innovation),
        "device": str(device),
    }
    return model, val_correction, test_correction, test_scale.astype(np.float32), test_gain, metadata, logs


def variants_from_args(args: argparse.Namespace) -> list[Variant]:
    available = {
        "ogif_full": Variant("ogif_full"),
        "ogif_no_graph": Variant("ogif_no_graph", use_graph=False),
        "ogif_no_update": Variant("ogif_no_update", use_update=False),
        "ogif_no_context": Variant("ogif_no_context", use_context=False),
        "ogif_shuffled_measurement": Variant("ogif_shuffled_measurement", shuffled_measurement=True),
        "ogif_shuffled_edges": Variant("ogif_shuffled_edges", shuffled_edges=True),
        "ogif_core": Variant("ogif_core", use_context=False),
        "ogif_core_no_graph": Variant("ogif_core_no_graph", use_graph=False, use_context=False),
        "ogif_core_no_update": Variant("ogif_core_no_update", use_update=False, use_context=False),
        "ogif_core_shuffled_measurement": Variant("ogif_core_shuffled_measurement", use_context=False, shuffled_measurement=True),
        "ogif_core_shuffled_edges": Variant("ogif_core_shuffled_edges", use_context=False, shuffled_edges=True),
        "ogif_core_consensus": Variant("ogif_core_consensus", use_context=False, graph_operator="consensus"),
        "ogif_core_consensus_shuffled_edges": Variant("ogif_core_consensus_shuffled_edges", use_context=False, shuffled_edges=True, graph_operator="consensus"),
        "ogif_core_edge_field": Variant("ogif_core_edge_field", use_context=False, graph_operator="edge_field"),
        "ogif_core_edge_field_shuffled_attr": Variant("ogif_core_edge_field_shuffled_attr", use_context=False, shuffled_edge_attr=True, graph_operator="edge_field"),
        "ogif_core_edge_field_no_edge_measurement": Variant("ogif_core_edge_field_no_edge_measurement", use_context=False, use_edge_measurement=False, graph_operator="edge_field"),
        "ogif_core_edge_field_no_update": Variant("ogif_core_edge_field_no_update", use_context=False, use_update=False, graph_operator="edge_field"),
        "ogif_core_pool": Variant("ogif_core_pool", use_graph=False, use_context=False, use_frame_pool=True),
        "ogif_core_pool_no_update": Variant("ogif_core_pool_no_update", use_graph=False, use_update=False, use_context=False, use_frame_pool=True),
        "ogif_core_pool_shuffled_measurement": Variant("ogif_core_pool_shuffled_measurement", use_graph=False, use_context=False, shuffled_measurement=True, use_frame_pool=True),
        "ogif_core_graph_pool": Variant("ogif_core_graph_pool", use_context=False, use_frame_pool=True),
        "ogif_temporal": Variant("ogif_temporal", use_graph=False, use_context=False, use_history=True),
        "ogif_temporal_graph": Variant("ogif_temporal_graph", use_context=False, use_history=True),
        "ogif_temporal_shuffled_history": Variant("ogif_temporal_shuffled_history", use_graph=False, use_context=False, use_history=True, shuffled_history=True),
        "ogif_temporal_velocity_history": Variant("ogif_temporal_velocity_history", use_graph=False, use_context=False, use_history=True, use_history_innovation=False),
        "ogif_temporal_no_innovation": Variant("ogif_temporal_no_innovation", use_graph=False, use_update=False, use_context=False, use_history=True, use_history_innovation=False),
        "ogif_temporal_history_only": Variant("ogif_temporal_history_only", use_graph=False, use_update=False, use_context=False, use_history=True),
        "ogif_temporal_history_only_shuffled": Variant("ogif_temporal_history_only_shuffled", use_graph=False, use_update=False, use_context=False, use_history=True, shuffled_history=True),
        "ogif_temporal_graph_history_only": Variant("ogif_temporal_graph_history_only", use_update=False, use_context=False, use_history=True),
        "ogif_temporal_graph_history_only_shuffled_history": Variant("ogif_temporal_graph_history_only_shuffled_history", use_update=False, use_context=False, use_history=True, shuffled_history=True),
        "ogif_temporal_graph_history_only_shuffled_edges": Variant("ogif_temporal_graph_history_only_shuffled_edges", use_update=False, use_context=False, use_history=True, shuffled_edges=True),
    }
    names = [token.strip() for token in str(args.variants).split(",") if token.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown variants: {missing}")
    return [available[name] for name in names]


def rolling_rows(
    bundle: v84.AnchorBundle,
    prediction_steps: np.ndarray,
    horizons: list[int],
    method: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = bundle.rows.reset_index(drop=True)
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(rows[KEYS].itertuples(index=False, name=None))
    }
    output: list[dict[str, Any]] = []
    for horizon in horizons:
        targets: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for i, (sequence, frame, track) in enumerate(rows[KEYS].itertuples(index=False, name=None)):
            chain = [lookup.get((int(sequence), int(frame) + offset, int(track))) for offset in range(horizon)]
            if any(index is None for index in chain):
                continue
            chain_indices = np.asarray(chain, dtype=np.int64)
            targets.append(bundle.target_steps[i, :horizon].sum(axis=0))
            predictions.append(prediction_steps[chain_indices, 0].sum(axis=0))
        target = safe(np.asarray(targets))
        prediction = safe(np.asarray(predictions))
        row: dict[str, Any] = {
            "method": method,
            "contract": "streaming_receding_h1",
            "horizon": int(horizon),
            "component_rmse": v84.component_rmse(target, prediction),
            "vector_rmse": v84.vector_rmse(target, prediction),
            "r2": v84.vector_r2(target, prediction),
            "cosine": v84.cosine_mean(target, prediction),
            "magnitude_ratio": v84.magnitude_ratio(target, prediction),
            "n_rows": int(len(target)),
        }
        if extra:
            row.update(extra)
        output.append(row)
    return output


def uncertainty_rows(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    scale: np.ndarray,
    eta: float,
    degrees_of_freedom: float,
    variant: str,
) -> list[dict[str, Any]]:
    if eta <= 0:
        return [{"variant": variant, "horizon": 1, "eta": eta, "nll": np.nan, "coverage_50": np.nan, "coverage_90": np.nan, "uncertainty_error_corr": np.nan}]
    target = bundle.target_steps[:, 0]
    mean = prediction[:, 0]
    effective_scale = np.maximum(scale[:, 0] * float(eta), 1e-3)
    standardized = (target - mean) / effective_scale
    nll = -np.mean(student_t.logpdf(standardized, df=degrees_of_freedom) - np.log(effective_scale))
    q50 = float(student_t.ppf(0.75, df=degrees_of_freedom))
    q90 = float(student_t.ppf(0.95, df=degrees_of_freedom))
    absolute = np.abs(target - mean)
    coverage_50 = float(np.mean(absolute <= q50 * effective_scale))
    coverage_90 = float(np.mean(absolute <= q90 * effective_scale))
    uncertainty = np.mean(effective_scale, axis=1)
    error = np.linalg.norm(target - mean, axis=1)
    correlation = float(np.corrcoef(uncertainty, error)[0, 1]) if np.std(uncertainty) > 1e-8 else 0.0
    return [{"variant": variant, "horizon": 1, "eta": eta, "nll": float(nll), "coverage_50": coverage_50, "coverage_90": coverage_90, "uncertainty_error_corr": correlation}]


def paired_direct_rows(
    bundle: v84.AnchorBundle,
    prediction_steps: np.ndarray,
    horizons: list[int],
    method: str,
) -> list[dict[str, Any]]:
    rows = bundle.rows.reset_index(drop=True)
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(rows[KEYS].itertuples(index=False, name=None))
    }
    output: list[dict[str, Any]] = []
    for horizon in horizons:
        valid: list[int] = []
        for i, (sequence, frame, track) in enumerate(rows[KEYS].itertuples(index=False, name=None)):
            if all(lookup.get((int(sequence), int(frame) + offset, int(track))) is not None for offset in range(horizon)):
                valid.append(i)
        idx = np.asarray(valid, dtype=np.int64)
        target = bundle.target_steps[idx, :horizon].sum(axis=1)
        prediction = prediction_steps[idx, :horizon].sum(axis=1)
        output.append(
            {
                "method": method,
                "contract": "single_shot_paired_rows",
                "horizon": horizon,
                "component_rmse": v84.component_rmse(target, prediction),
                "vector_rmse": v84.vector_rmse(target, prediction),
                "r2": v84.vector_r2(target, prediction),
                "cosine": v84.cosine_mean(target, prediction),
                "magnitude_ratio": v84.magnitude_ratio(target, prediction),
                "n_rows": len(idx),
            }
        )
    return output


def load_v88_reference(path: Path | None, test: v84.AnchorBundle) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    archive = np.load(path)
    required = ["anchor_seed_mean", "graph_all_models_mean"]
    if any(key not in archive for key in required):
        raise RuntimeError(f"v88 file lacks required keys: {required}")
    reference = {key: safe(archive[key]) for key in required}
    for key, values in reference.items():
        if values.shape != test.target_steps.shape:
            raise RuntimeError(f"v88 {key} shape {values.shape} != target {test.target_steps.shape}")
    return reference


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundles = v85.load_anchor_cache(args.anchor_cache)
    context, context_names, context_diagnostics = load_context(args.features, bundles, parse_context_quotas(args.context_quotas))
    variants = variants_from_args(args)
    open_loop_metrics: list[dict[str, Any]] = []
    streaming_metrics: list[dict[str, Any]] = []
    paired_metrics: list[dict[str, Any]] = []
    uncertainty_metrics: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    horizons = parse_ints(args.horizons)
    eta_grid = parse_floats(args.eta_grid)
    test = bundles[2]

    open_loop_metrics.extend(v84.metric_rows(test, test.anchor_steps, horizons, "v96_v52_anchor", {"variant": "anchor", "control": "real", "contract": "open_loop"}))
    streaming_metrics.extend(rolling_rows(test, test.anchor_steps, horizons, "v96_v52_anchor_rolling", {"variant": "anchor", "control": "real"}))
    paired_metrics.extend(paired_direct_rows(test, test.anchor_steps, horizons, "v96_v52_anchor_direct_paired"))

    v88_reference = load_v88_reference(args.v88_predictions, test)
    if v88_reference:
        open_loop_metrics.extend(v84.metric_rows(test, v88_reference["graph_all_models_mean"], horizons, "v88_graph_all_models_mean", {"variant": "v88", "control": "real", "contract": "open_loop"}))
        streaming_metrics.extend(rolling_rows(test, v88_reference["graph_all_models_mean"], horizons, "v88_graph_all_models_mean_rolling", {"variant": "v88", "control": "real"}))
        paired_metrics.extend(paired_direct_rows(test, v88_reference["graph_all_models_mean"], horizons, "v88_graph_all_models_mean_direct_paired"))

    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    for variant in variants:
        print(f"[v96] preparing {variant.name}", flush=True)
        prep = prepare(args, variant, bundles, context, context_names)
        print(f"[v96] training {variant.name}", flush=True)
        model, val_correction, test_correction, test_scale, test_gain, metadata, variant_logs = train_variant(prep, variant, args)
        eta, val_score = v84.tune_eta(prep.bundles[1], val_correction, horizons, eta_grid)
        streaming_eta, streaming_val_score = v84.tune_eta(prep.bundles[1], val_correction, [1], eta_grid)
        test_prediction = prep.bundles[2].anchor_steps + float(eta) * test_correction
        streaming_prediction = prep.bundles[2].anchor_steps + float(streaming_eta) * test_correction
        control = "real"
        if variant.shuffled_measurement:
            control = "shuffled_measurement"
        elif variant.shuffled_history:
            control = "shuffled_history"
        elif variant.shuffled_edge_attr:
            control = "shuffled_edge_attr"
        elif variant.shuffled_edges:
            control = "shuffled_edges"
        elif not variant.use_update:
            control = "no_update"
        elif not variant.use_graph:
            control = "no_graph"
        elif not variant.use_context:
            control = "no_context"
        extra = {
            "variant": variant.name,
            "control": control,
            "contract": "open_loop",
            "eta": eta,
            "streaming_eta": streaming_eta,
            "val_score": val_score,
            "streaming_val_score": streaming_val_score,
            "best_epoch": metadata["best_epoch"],
        }
        open_loop_metrics.extend(v84.metric_rows(prep.bundles[2], test_prediction, horizons, f"v96_{variant.name}", extra))
        streaming_metrics.extend(rolling_rows(prep.bundles[2], streaming_prediction, horizons, f"v96_{variant.name}_rolling", {"variant": variant.name, "control": control, "eta": streaming_eta}))
        paired_metrics.extend(paired_direct_rows(prep.bundles[2], test_prediction, horizons, f"v96_{variant.name}_direct_paired"))
        uncertainty_metrics.extend(uncertainty_rows(prep.bundles[2], streaming_prediction, test_scale, streaming_eta, metadata["degrees_of_freedom"], variant.name))
        diagnostics.append({"variant": variant.name, "control": control, "eta": eta, "streaming_eta": streaming_eta, "val_score": val_score, "streaming_val_score": streaming_val_score, **metadata})
        logs.extend(variant_logs)
        predictions[f"{variant.name}__test_prediction"] = safe(test_prediction)
        predictions[f"{variant.name}__test_streaming_prediction"] = safe(streaming_prediction)
        predictions[f"{variant.name}__val_correction"] = safe(val_correction)
        predictions[f"{variant.name}__test_correction"] = safe(test_correction)
        predictions[f"{variant.name}__test_scale"] = safe(test_scale)
        predictions[f"{variant.name}__test_gain"] = safe(test_gain)
        torch.save({"state_dict": model.state_dict(), "metadata": metadata, "static_dim": prep.static_dim}, args.out_dir / f"{variant.name}.pt")

        if variant.name in {
            "ogif_full",
            "ogif_core",
            "ogif_no_context",
            "ogif_core_consensus",
            "ogif_core_edge_field",
            "ogif_core_pool",
            "ogif_core_graph_pool",
            "ogif_temporal",
            "ogif_temporal_graph",
        }:
            for stride in parse_ints(args.measurement_strides):
                if stride == 1:
                    continue
                normalized, _logscale, _measurement, _gain = infer(model, prep, 2, device, measurement_stride=stride)
                correction = decode_correction(prep, normalized)
                prediction = prep.bundles[2].anchor_steps + float(streaming_eta) * correction
                streaming_metrics.extend(
                    rolling_rows(
                        prep.bundles[2],
                        prediction,
                        horizons,
                        f"v96_{variant.name}_rolling_update_stride{stride}",
                        {"variant": variant.name, "control": f"measurement_stride_{stride}", "eta": streaming_eta},
                    )
                )

    open_df = pd.DataFrame(open_loop_metrics)
    streaming_df = pd.DataFrame(streaming_metrics)
    paired_df = pd.DataFrame(paired_metrics)
    uncertainty_df = pd.DataFrame(uncertainty_metrics)
    diagnostics_df = pd.DataFrame(diagnostics)
    open_df.to_csv(args.out_dir / "v96_open_loop_metrics.csv", index=False)
    streaming_df.to_csv(args.out_dir / "v96_streaming_metrics.csv", index=False)
    paired_df.to_csv(args.out_dir / "v96_paired_contract_metrics.csv", index=False)
    uncertainty_df.to_csv(args.out_dir / "v96_uncertainty.csv", index=False)
    diagnostics_df.to_csv(args.out_dir / "v96_state_diagnostics.csv", index=False)
    pd.DataFrame(logs).to_csv(args.out_dir / "v96_train_log.csv", index=False)
    pd.DataFrame(context_diagnostics).to_csv(args.out_dir / "v96_data_contract.csv", index=False)
    np.savez_compressed(args.out_dir / "v96_predictions.npz", **predictions)
    (args.out_dir / "v96_context_features.json").write_text(json.dumps(context_names, indent=2), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(finite(vars(args)), indent=2), encoding="utf-8")

    hmax = max(horizons)
    open_h = open_df[open_df.horizon.eq(hmax)].sort_values("component_rmse")
    rolling_h = streaming_df[streaming_df.horizon.eq(hmax)].sort_values("component_rmse")
    anchor_open = float(open_h[open_h.method.eq("v96_v52_anchor")].iloc[0].component_rmse)
    best = open_h.iloc[0]
    best_gain = (anchor_open - float(best.component_rmse)) / anchor_open * 100.0
    lines = [
        "# v96 Graph State-Space Seq2Seq Report",
        "",
        "## Open-loop h6",
        "",
        open_h[["method", "component_rmse", "r2", "cosine", "magnitude_ratio", "variant", "control"]].to_markdown(index=False),
        "",
        "## Receding-h1 h6-equivalent",
        "",
        rolling_h[["method", "component_rmse", "r2", "cosine", "magnitude_ratio", "variant", "control", "n_rows"]].to_markdown(index=False),
        "",
        "## Decision",
        "",
        f"- Same-row v52 open-loop h6: `{anchor_open:.6f}`.",
        f"- Best open-loop result: `{float(best.component_rmse):.6f}` (`{best_gain:.3f}%` vs v52).",
    ]
    full_h = open_h[open_h.variant.eq("ogif_full")]
    controls_h = open_h[open_h.control.isin(["no_graph", "no_update", "shuffled_measurement", "shuffled_edges"])]
    if not full_h.empty:
        full_value = float(full_h.iloc[0].component_rmse)
        full_gain = (anchor_open - full_value) / anchor_open * 100.0
        lines.append(f"- OGIF full h6: `{full_value:.6f}` (`{full_gain:.3f}%`).")
        if full_gain >= 3.0 and (controls_h.empty or full_value < float(controls_h.component_rmse.min())):
            lines.append("- Hard gate passed: recursive filtered state beats the anchor and hard controls.")
        elif full_gain >= 1.0 and (controls_h.empty or full_value <= float(controls_h.component_rmse.min())):
            lines.append("- Soft gate passed: retain OGIF and confirm across seeds/Edge.")
        else:
            lines.append("- Gate not passed: do not add switching/diffusion before repairing the continuous filter.")
    lines.append(f"- Elapsed: `{(time.time() - started) / 3600.0:.2f} h`.")
    (args.out_dir / "v96_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "v96_decision_report.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-cache", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--v88-predictions", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", default="ogif_full,ogif_no_graph,ogif_no_update,ogif_shuffled_measurement,ogif_shuffled_edges")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--measurement-strides", default="1,2,3,6")
    parser.add_argument("--context-quotas", default="ms_:16,tf_:48,rc_:16,obs_:48")
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=1)
    parser.add_argument("--history-lags", type=int, default=6)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--horizon-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--tbptt-frames", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.06)
    parser.add_argument("--correction-bound", type=float, default=3.5)
    parser.add_argument("--endpoint-weight", type=float, default=0.35)
    parser.add_argument("--nll-weight", type=float, default=0.08)
    parser.add_argument("--measurement-weight", type=float, default=0.15)
    parser.add_argument("--temporal-weight", type=float, default=0.08)
    parser.add_argument("--gain-weight", type=float, default=0.002)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--eta-grid", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1,1.25")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    try:
        run(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        }
        (args.out_dir / "v96_error.json").write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
