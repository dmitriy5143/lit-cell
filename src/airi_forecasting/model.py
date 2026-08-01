"""Neural prior used by the causal sequential forecasting architecture."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class CausalInnovationStateSpaceForecaster(nn.Module):
    """Neural innovation filter with learned process and observation scales."""

    def __init__(
        self,
        static_dim: int,
        hidden: int,
        history_lags: int,
        correction_bound: float,
        dropout: float,
        use_update: bool,
        use_graph: bool,
        graph_heads: int,
        output_mode: str = "anchor_residual",
        target_mean: np.ndarray | None = None,
        target_scale: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if use_graph and hidden % graph_heads:
            raise ValueError("hidden must be divisible by graph_heads")
        self.hidden = int(hidden)
        self.history_lags = int(history_lags)
        self.correction_bound = float(correction_bound)
        self.dropout = float(dropout)
        self.use_update = bool(use_update)
        self.use_graph = bool(use_graph)
        self.output_mode = str(output_mode)
        if self.output_mode not in {"anchor_residual", "direct"}:
            raise ValueError(f"Unknown output mode: {self.output_mode}")
        target_mean_array = (
            np.zeros(2, dtype=np.float32)
            if target_mean is None
            else np.asarray(target_mean, dtype=np.float32)
        )
        target_scale_array = (
            np.ones(2, dtype=np.float32)
            if target_scale is None
            else np.asarray(target_scale, dtype=np.float32)
        )
        self.register_buffer(
            "target_mean",
            torch.as_tensor(target_mean_array, dtype=torch.float32),
        )
        self.register_buffer(
            "target_scale",
            torch.as_tensor(
                np.maximum(target_scale_array, 1e-4),
                dtype=torch.float32,
            ),
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.history_input = nn.Sequential(
            nn.Linear(5, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
        )
        self.history_encoder = nn.GRU(hidden // 2, hidden, batch_first=True)
        self.history_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self.history_norm = nn.LayerNorm(hidden)
        self.initial_state = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.transition = nn.GRUCell(hidden, hidden)
        self.process_scale_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.observation_scale_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.update_encoder = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(hidden * 2 + 7, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.update_norm = nn.LayerNorm(hidden)

        if self.use_graph:
            self.graph_conv = TransformerConv(
                hidden,
                hidden // graph_heads,
                heads=graph_heads,
                concat=True,
                beta=True,
                edge_dim=8,
                dropout=dropout,
            )
            self.graph_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Sigmoid(),
            )
            self.graph_norm = nn.LayerNorm(hidden)
            self.raw_graph_strength = nn.Parameter(torch.tensor(-2.0))

        self.forecast_trunk = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, 2)
        self.scale_head = nn.Linear(hidden, 2)
        self.raw_df = nn.Parameter(torch.tensor(1.5))

    @property
    def degrees_of_freedom(self) -> torch.Tensor:
        return F.softplus(self.raw_df) + 2.1

    def forward_step(
        self,
        static: torch.Tensor,
        history: torch.Tensor,
        innovation: torch.Tensor,
        measurement_mask: torch.Tensor,
        previous_state: torch.Tensor,
        has_previous_state: torch.Tensor,
        anchor_h1: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        encoded = self.static_encoder(static)
        history_tokens = self.history_input(torch.flip(history, dims=[1]))
        history_state = self.history_encoder(history_tokens)[1][-1]
        history_gate = self.history_gate(torch.cat([encoded, history_state], dim=-1))
        encoded = self.history_norm(encoded + history_gate * history_state)

        initialized = self.initial_state(encoded)
        recurrent = torch.where(
            has_previous_state[:, None] > 0.5,
            previous_state,
            initialized,
        )
        prior = self.transition(encoded, recurrent)
        process_scale = (
            F.softplus(
                self.process_scale_head(torch.cat([prior, encoded], dim=-1))
            )
            + 1e-3
        )
        observation_scale = (
            F.softplus(
                self.observation_scale_head(torch.cat([prior, encoded], dim=-1))
            )
            + 1e-3
        )
        total_scale = torch.sqrt(
            torch.square(process_scale)
            + torch.square(observation_scale)
            + 1e-6
        )
        normalized_innovation = innovation / total_scale
        noise_packet = torch.cat(
            [
                normalized_innovation,
                torch.log(process_scale),
                torch.log(observation_scale),
                measurement_mask[:, None],
            ],
            dim=-1,
        )
        update = self.update_encoder(noise_packet)
        analytic_gain = torch.square(process_scale) / (
            torch.square(process_scale)
            + torch.square(observation_scale)
            + 1e-6
        )
        analytic_logit = torch.logit(
            torch.clamp(
                analytic_gain.mean(dim=1, keepdim=True),
                1e-4,
                1.0 - 1e-4,
            )
        )
        learned_logit = self.gain_head(
            torch.cat([prior, encoded, noise_packet], dim=-1)
        )
        gain = torch.sigmoid(learned_logit + analytic_logit)
        if self.use_update:
            posterior = self.update_norm(
                prior + measurement_mask[:, None] * gain * update
            )
        else:
            posterior = prior
            gain = torch.zeros_like(gain)

        if self.use_graph and edge_index.shape[1] > 0:
            graph_update = F.silu(
                self.graph_conv(posterior, edge_index, edge_attr)
            )
            graph_gate = self.graph_gate(
                torch.cat([posterior, graph_update], dim=-1)
            )
            posterior = self.graph_norm(
                posterior
                + torch.sigmoid(self.raw_graph_strength)
                * graph_gate
                * graph_update
            )

        forecast = self.forecast_trunk(
            torch.cat([posterior, encoded, anchor_h1], dim=-1)
        )
        raw_mean = self.mean_head(forecast)
        mean = self.correction_bound * torch.tanh(
            raw_mean / max(self.correction_bound, 1e-4)
        )
        aleatoric = F.softplus(self.scale_head(forecast)) + 1e-3
        predictive_scale = torch.sqrt(
            torch.square(aleatoric) + torch.square(process_scale) + 1e-6
        )
        return (
            posterior,
            mean,
            predictive_scale,
            process_scale,
            observation_scale,
            gain,
        )

    def normalize_direct_target(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.target_mean) / self.target_scale

    def denormalize_direct_target(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.target_scale + self.target_mean
