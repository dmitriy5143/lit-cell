#!/usr/bin/env python3
"""Next-generation LaChance graph message-passing benchmark.

This runner keeps the proven LaChance data protocol and temporal/flow encoders,
but replaces the conservative one-hop structural decoder with a learned
flow-gated, multi-hop equivariant message-passing decoder.

The scientific question is narrow: can a stronger prior-to-message interface
materially improve held-out displacement prediction on MDCK while keeping
negative/weak controls interpretable?
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import data_protocol as la  # noqa: E402

arch = la.arch

DEFAULT_OUT = ROOT / "outputs" / "lachance_nextgen_message_passing"
CELL_TYPES = la.CELL_TYPES
VARIANTS = ("mp_gated_full", "mp_gated_no_velocity", "mp_gated_radial")


def finite_json(value: Any) -> Any:
    return la.finite_json(value)


def vector_metrics_from_norm(
    pred_norm: torch.Tensor,
    graph: arch.GraphTensors,
    norm: arch.Normalizer,
) -> tuple[np.ndarray, dict[str, float]]:
    pred_px = arch.to_px(pred_norm, norm)
    mask = graph.target_valid.detach().cpu().numpy()
    y_px = graph.y_px.detach().cpu().numpy()
    return pred_px, arch.vector_metrics(y_px[mask], pred_px[mask], 1)


def vector_metrics_from_norm_np(
    pred_norm: np.ndarray,
    graph: arch.GraphTensors,
    norm: arch.Normalizer,
) -> tuple[np.ndarray, dict[str, float]]:
    pred_px = pred_norm * norm.target_std + norm.target_mean
    mask = graph.target_valid.detach().cpu().numpy()
    y_px = graph.y_px.detach().cpu().numpy()
    return pred_px, arch.vector_metrics(y_px[mask], pred_px[mask], 1)


def sequence_groups(graph: arch.GraphTensors) -> torch.Tensor:
    return arch.sequence_groups(graph)


def fit_linear_calibrator(
    features: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
) -> np.ndarray:
    x = features[mask].detach().cpu().numpy().astype(np.float64)
    y = target[mask].detach().cpu().numpy().astype(np.float64)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[finite]
    y = y[finite]
    if len(x) == 0:
        coef = np.zeros((features.shape[1], target.shape[1]), dtype=np.float32)
        return coef
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
    main = x[:, :-1]
    mean = main.mean(axis=0, keepdims=True)
    std = np.maximum(main.std(axis=0, keepdims=True), 1e-6)
    z = np.concatenate([(main - mean) / std, x[:, -1:]], axis=1)
    penalty = np.eye(z.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coef_z = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    coef = np.zeros_like(coef_z)
    coef[:-1] = coef_z[:-1] / std.T
    coef[-1] = coef_z[-1] - (mean / std) @ coef_z[:-1]
    return coef.astype(np.float32)


def apply_linear_calibrator(features: torch.Tensor, coef: np.ndarray) -> np.ndarray:
    x = features.detach().cpu().numpy().astype(np.float32)
    return x @ coef


def bias_column(reference: torch.Tensor) -> torch.Tensor:
    return torch.ones((reference.shape[0], 1), dtype=reference.dtype, device=reference.device)


class EquivariantMPDecoder(nn.Module):
    """Multi-hop message passing with physically readable vector bases."""

    def __init__(
        self,
        *,
        self_dim: int = 48,
        flow_dim: int = 24,
        edge_dim: int = 12,
        hidden_dim: int = 72,
        edge_hidden_dim: int = 56,
        layers: int = 2,
        max_delta_norm: float = 1.35,
        mode: str = "full",
    ) -> None:
        super().__init__()
        if mode not in {"full", "no_velocity", "radial"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.layers = int(layers)
        self.max_delta_norm = float(max_delta_norm)
        node_extra = 2 + 2 + 3  # self_pred, flow_pred, quality/speed/log-degree
        node_in = self_dim + flow_dim + node_extra
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        # src node, dst node, normalized edge features, radial, rel velocity,
        # shear, closing, front/side, source/destination speed.
        edge_in = 2 * hidden_dim + edge_dim + 2 + 2 + 2 + 1 + 2 + 2
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(edge_in, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, edge_hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(self.layers)
            ]
        )
        self.edge_gate = nn.ModuleList(
            [nn.Linear(edge_hidden_dim, 1) for _ in range(self.layers)]
        )
        self.edge_coeff = nn.ModuleList(
            [nn.Linear(edge_hidden_dim, 6) for _ in range(self.layers)]
        )
        self.node_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + edge_hidden_dim + 2 + 3, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.layers)
            ]
        )
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.layers)])
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        for gate in self.edge_gate:
            nn.init.normal_(gate.weight, std=1e-3)
            nn.init.zeros_(gate.bias)
            gate.bias.data.fill_(-0.25)
        for coeff in self.edge_coeff:
            nn.init.normal_(coeff.weight, std=1e-3)
            nn.init.zeros_(coeff.bias)
        last = self.node_head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)
        # node gate moderately open; flow gate initialized near 1.0 because
        # self_flow is the strong baseline on MDCK.
        last.bias.data[0] = 0.25
        last.bias.data[1] = math.log(2.0)

    def _edge_context(
        self,
        graph: arch.GraphTensors,
        node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = graph.src, graph.dst
        radial = graph.radial
        rel_velocity = graph.rel_velocity
        shear = graph.shear
        if self.mode == "radial":
            rel_velocity = torch.zeros_like(rel_velocity)
            shear = torch.zeros_like(shear)
        elif self.mode == "no_velocity":
            rel_velocity = torch.zeros_like(rel_velocity)
            shear = torch.zeros_like(shear)
        own = graph.own_direction[dst]
        front = torch.sum(radial * own, dim=1, keepdim=True)
        side = torch.sqrt(torch.clamp(1.0 - front.square(), min=0.0))
        src_speed = graph.speed_norm[src]
        dst_speed = graph.speed_norm[dst]
        edge_input = torch.cat(
            [
                node[src],
                node[dst],
                graph.edge_features,
                radial,
                rel_velocity,
                shear,
                graph.closing,
                front,
                side,
                src_speed,
                dst_speed,
            ],
            dim=1,
        )
        reliability = torch.sqrt(
            torch.clamp(graph.quality[src] * graph.quality[dst], min=0.0)
        )
        tangent = torch.stack([-radial[:, 1], radial[:, 0]], dim=1)
        bases = torch.stack(
            [
                radial,
                tangent,
                rel_velocity,
                shear,
                graph.current_velocity[src],
                graph.current_velocity[dst],
            ],
            dim=1,
        )
        if self.mode == "radial":
            bases = torch.stack(
                [
                    radial,
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                ],
                dim=1,
            )
        elif self.mode == "no_velocity":
            bases = torch.stack(
                [
                    radial,
                    tangent,
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                    torch.zeros_like(radial),
                ],
                dim=1,
            )
        return edge_input, reliability, bases, front

    def forward(
        self,
        graph: arch.GraphTensors,
        self_state: torch.Tensor,
        flow_state: torch.Tensor,
        self_pred: torch.Tensor,
        flow_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        n = graph.history.shape[0]
        log_degree = torch.log1p(graph.degree) / math.log(10.0)
        node = self.node_proj(
            torch.cat(
                [
                    self_state,
                    flow_state,
                    self_pred,
                    flow_pred,
                    graph.quality,
                    graph.speed_norm,
                    log_degree,
                ],
                dim=1,
            )
        )
        edge_gate_values: list[torch.Tensor] = []
        coeff_values: list[torch.Tensor] = []
        effective_values: list[torch.Tensor] = []
        field = torch.zeros((n, 2), dtype=node.dtype, device=node.device)
        scalar_agg = torch.zeros((n, self.edge_mlps[0][-2].out_features), dtype=node.dtype, device=node.device)
        for layer_id in range(self.layers):
            edge_input, reliability, bases, _ = self._edge_context(graph, node)
            edge_hidden = self.edge_mlps[layer_id](edge_input)
            edge_gate = torch.sigmoid(self.edge_gate[layer_id](edge_hidden))
            coeff = 0.75 * torch.tanh(self.edge_coeff[layer_id](edge_hidden))
            message = torch.sum(coeff[:, :, None] * bases, dim=1)
            weighted_vec = reliability * edge_gate * message
            weighted_hidden = reliability * edge_gate * edge_hidden
            effective = arch.scatter_sum(reliability * edge_gate, graph.dst, n).clamp_min(1e-4)
            field = arch.scatter_sum(weighted_vec, graph.dst, n) / effective
            scalar_agg = arch.scatter_sum(weighted_hidden, graph.dst, n) / effective
            update = self.node_updates[layer_id](
                torch.cat(
                    [node, scalar_agg, field, graph.quality, graph.speed_norm, log_degree],
                    dim=1,
                )
            )
            node = self.node_norms[layer_id](node + 0.65 * update)
            edge_gate_values.append(edge_gate)
            coeff_values.append(coeff)
            effective_values.append(effective)
        node_out = self.node_head(
            torch.cat([node, field, graph.quality, graph.speed_norm, log_degree], dim=1)
        )
        node_gate = torch.sigmoid(node_out[:, 0:1])
        flow_gate = 1.5 * torch.sigmoid(node_out[:, 1:2])
        mobility = 0.20 + 1.55 * torch.sigmoid(node_out[:, 2:4])
        scale = 0.55 + 1.20 * torch.sigmoid(node_out[:, 4:5])
        residual_bias = 0.10 * torch.tanh(node_out[:, 5:6])
        own = graph.own_direction
        parallel_scalar = torch.sum(field * own, dim=1, keepdim=True)
        parallel = parallel_scalar * own
        perpendicular = field - parallel
        field = scale * (
            mobility[:, 0:1] * parallel + mobility[:, 1:2] * perpendicular
        )
        # A tiny speed-aligned residual lets the model express acceleration-like
        # corrections without turning the decoder into an arbitrary 2-D head.
        field = field + residual_bias * own
        norm = torch.linalg.vector_norm(field, dim=1, keepdim=True)
        shrink = torch.where(
            norm > 1e-5,
            torch.tanh(norm) / norm.clamp_min(1e-6),
            torch.ones_like(norm),
        )
        delta = self.max_delta_norm * node_gate * field * shrink
        return delta, flow_gate, {
            "node_gate": node_gate,
            "flow_gate": flow_gate,
            "edge_gate": torch.cat(edge_gate_values, dim=0),
            "coeff": torch.cat(coeff_values, dim=0),
            "effective_degree": torch.stack(effective_values, dim=0).mean(dim=0),
            "field": field,
        }


@dataclass
class EncodedBase:
    self_pred: torch.Tensor
    self_state: torch.Tensor
    flow_pred: torch.Tensor
    flow_state: torch.Tensor


@torch.no_grad()
def encode_all(
    temporal: arch.TemporalSelfEncoder,
    flow: arch.CoarseFlowEncoder,
    graph: arch.GraphTensors,
) -> EncodedBase:
    self_pred, self_state, flow_pred, flow_state = arch.encode_base(
        temporal, flow, graph
    )
    return EncodedBase(self_pred, self_state, flow_pred, flow_state)


def train_mp_decoder(
    train: arch.GraphTensors,
    val: arch.GraphTensors,
    train_enc: EncodedBase,
    val_enc: EncodedBase,
    *,
    variant: str,
    seed: int,
    epochs: int,
    sequence_balanced_loss: bool,
    layers: int,
    hidden_dim: int,
    edge_hidden_dim: int,
    max_delta_norm: float,
    lr: float,
    social_l2: float,
    flow_gate_l2: float,
) -> tuple[EquivariantMPDecoder, dict[str, float]]:
    mode = {
        "mp_gated_full": "full",
        "mp_gated_no_velocity": "no_velocity",
        "mp_gated_radial": "radial",
    }[variant]
    arch.set_seed(seed + 50_000)
    model = EquivariantMPDecoder(
        edge_dim=train.edge_features.shape[1],
        hidden_dim=hidden_dim,
        edge_hidden_dim=edge_hidden_dim,
        layers=layers,
        max_delta_norm=max_delta_norm,
        mode=mode,
    ).to(train.history.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    best, best_val, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        delta, flow_gate, diag = model(
            train,
            train_enc.self_state.detach(),
            train_enc.flow_state.detach(),
            train_enc.self_pred.detach(),
            train_enc.flow_pred.detach(),
        )
        pred = train_enc.self_pred.detach() + flow_gate * train_enc.flow_pred.detach() + delta
        loss = arch.masked_vector_mse(
            pred,
            train.y_norm,
            train.target_valid,
            sequence_groups(train) if sequence_balanced_loss else None,
        )
        active_delta = delta[train.target_valid]
        active_flow_gate = flow_gate[train.target_valid]
        loss = (
            loss
            + float(social_l2) * torch.mean(torch.sum(active_delta.square(), dim=1))
            + float(flow_gate_l2) * torch.mean((active_flow_gate - 1.0).square())
            + 1e-6 * diag["edge_gate"].mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            delta, flow_gate, _ = model(
                val,
                val_enc.self_state,
                val_enc.flow_state,
                val_enc.self_pred,
                val_enc.flow_pred,
            )
            pred = val_enc.self_pred + flow_gate * val_enc.flow_pred + delta
            score = float(
                arch.masked_vector_mse(
                    pred,
                    val.y_norm,
                    val.target_valid,
                    sequence_groups(val) if sequence_balanced_loss else None,
                )
            )
        if score < best_val - 1e-6:
            best, best_val, best_epoch = copy.deepcopy(model.state_dict()), score, epoch + 1
        elif epoch + 1 - best_epoch >= 22:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_norm_mse": best_val}


@torch.no_grad()
def evaluate_mp(
    model: EquivariantMPDecoder,
    graph: arch.GraphTensors,
    enc: EncodedBase,
    norm: arch.Normalizer,
) -> tuple[np.ndarray, dict[str, float]]:
    delta, flow_gate, diag = model(
        graph,
        enc.self_state,
        enc.flow_state,
        enc.self_pred,
        enc.flow_pred,
    )
    base = enc.self_pred + flow_gate * enc.flow_pred
    pred = base + delta
    pred_px, metrics = vector_metrics_from_norm(pred, graph, norm)
    mask = graph.target_valid.detach().cpu().numpy()
    y_px = graph.y_px.detach().cpu().numpy()
    delta_px = delta.detach().cpu().numpy() * norm.target_std
    residual_px = y_px - arch.to_px(base, norm)
    finite = mask & np.isfinite(residual_px).all(axis=1)
    dot = np.sum(delta_px[finite] * residual_px[finite], axis=1)
    denom = np.maximum(
        np.linalg.norm(delta_px[finite], axis=1)
        * np.linalg.norm(residual_px[finite], axis=1),
        1e-8,
    )
    coeff = diag["coeff"].detach().cpu().numpy()
    metrics.update(
        {
            "social_magnitude_mean_px": float(np.mean(np.linalg.norm(delta_px[mask], axis=1))),
            "social_magnitude_p90_px": float(np.quantile(np.linalg.norm(delta_px[mask], axis=1), 0.9)),
            "social_residual_cosine": float(np.mean(dot / denom)),
            "node_gate_mean": float(diag["node_gate"][graph.target_valid].mean().cpu()),
            "node_gate_p90": float(torch.quantile(diag["node_gate"][graph.target_valid], 0.9).cpu()),
            "flow_gate_mean": float(flow_gate[graph.target_valid].mean().cpu()),
            "flow_gate_std": float(flow_gate[graph.target_valid].std().cpu()),
            "flow_gate_p10": float(torch.quantile(flow_gate[graph.target_valid].reshape(-1), 0.1).cpu()),
            "flow_gate_p90": float(torch.quantile(flow_gate[graph.target_valid].reshape(-1), 0.9).cpu()),
            "edge_gate_mean": float(diag["edge_gate"].mean().cpu()),
            "effective_degree_mean": float(diag["effective_degree"][graph.target_valid].mean().cpu()),
        }
    )
    for idx in range(coeff.shape[1]):
        metrics[f"basis_coeff_abs_mean_{idx}"] = float(np.mean(np.abs(coeff[:, idx])))
    return pred_px, metrics


@torch.no_grad()
def predict_mp_components(
    model: EquivariantMPDecoder,
    graph: arch.GraphTensors,
    enc: EncodedBase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    delta, flow_gate, diag = model(
        graph,
        enc.self_state,
        enc.flow_state,
        enc.self_pred,
        enc.flow_pred,
    )
    gated_flow = flow_gate * enc.flow_pred
    pred = enc.self_pred + gated_flow + delta
    return pred, gated_flow, delta, diag


def zero_social_metrics(base_metrics: dict[str, float]) -> dict[str, float]:
    out = dict(base_metrics)
    out.update(
        {
            "social_magnitude_mean_px": 0.0,
            "social_magnitude_p90_px": 0.0,
            "social_residual_cosine": float("nan"),
            "node_gate_mean": 0.0,
            "node_gate_p90": 0.0,
            "flow_gate_mean": 1.0,
            "flow_gate_std": 0.0,
            "flow_gate_p10": 1.0,
            "flow_gate_p90": 1.0,
            "edge_gate_mean": 0.0,
            "effective_degree_mean": 0.0,
        }
    )
    return out


def run_cell_type(
    cell_type: str,
    *,
    table_root: Path,
    split_mode: str,
    split_seed: int,
    max_movies: int,
    max_tracks_per_movie: int,
    frame_stride: int,
    smooth_window: int,
    crop_fraction: float,
    r_cut_px: float | None,
    horizon: int,
    seeds: list[int],
    variants: list[str],
    device: torch.device,
    k: int,
    temporal_epochs: int,
    flow_epochs: int,
    mp_epochs: int,
    batch_size: int,
    sequence_balanced_loss: bool,
    layers: int,
    hidden_dim: int,
    edge_hidden_dim: int,
    max_delta_norm: float,
    lr: float,
    social_l2: float,
    flow_gate_l2: float,
    emit_guarded: bool,
    guard_threshold_pct: float,
    emit_calibrated: bool,
    calibration_alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, meta = la.load_lachance_dataset(
        cell_type,
        table_root=table_root,
        split_mode=split_mode,
        split_seed=split_seed,
        max_movies=max_movies,
        max_tracks_per_movie=max_tracks_per_movie,
        frame_stride=frame_stride,
        smooth_window=smooth_window,
        crop_fraction=crop_fraction,
        r_cut_px=r_cut_px,
    )
    graphs, norm, coverage = la.prepare_dataset(
        cell_type, raw, meta, horizon=horizon, k=k, device=device
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[{cell_type}] seed={seed} temporal", flush=True)
        temporal, temporal_info = arch.train_temporal(
            graphs["train"],
            graphs["val"],
            seed=seed,
            epochs=temporal_epochs,
            batch_size=batch_size,
            sequence_balanced_loss=sequence_balanced_loss,
        )
        zero_flow = arch.CoarseFlowEncoder(graphs["train"].flow.shape[1]).to(device)
        for parameter in zero_flow.parameters():
            torch.nn.init.zeros_(parameter)
        self_pred_px, self_metrics = arch.evaluate(
            temporal, zero_flow, None, graphs["test"], norm, variant="self_only"
        )
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "self_only",
                **self_metrics,
                **arch.sequence_metric_fields(graphs["test"], self_pred_px),
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        with torch.no_grad():
            train_self, _ = temporal(graphs["train"].history)
            val_self, _ = temporal(graphs["val"].history)
        print(f"[{cell_type}] seed={seed} flow", flush=True)
        flow, flow_info = arch.train_flow(
            graphs["train"],
            graphs["val"],
            train_self.detach(),
            val_self.detach(),
            seed=seed,
            epochs=flow_epochs,
            batch_size=batch_size,
            sequence_balanced_loss=sequence_balanced_loss,
        )
        flow_pred_px, flow_metrics = arch.evaluate(
            temporal, flow, None, graphs["test"], norm, variant="self_flow"
        )
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "self_flow",
                **flow_metrics,
                **arch.sequence_metric_fields(graphs["test"], flow_pred_px),
                **{f"flow_{k0}": v for k0, v in flow_info.items()},
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        encoded = {split: encode_all(temporal, flow, graph) for split, graph in graphs.items()}
        if emit_calibrated:
            flow_val_features = torch.cat(
                [
                    encoded["val"].self_pred,
                    encoded["val"].flow_pred,
                    bias_column(encoded["val"].self_pred),
                ],
                dim=1,
            )
            flow_test_features = torch.cat(
                [
                    encoded["test"].self_pred,
                    encoded["test"].flow_pred,
                    bias_column(encoded["test"].self_pred),
                ],
                dim=1,
            )
            flow_coef = fit_linear_calibrator(
                flow_val_features,
                graphs["val"].y_norm,
                graphs["val"].target_valid,
                alpha=calibration_alpha,
            )
            flow_cal_pred_norm = apply_linear_calibrator(flow_test_features, flow_coef)
            flow_cal_pred_px, flow_cal_metrics = vector_metrics_from_norm_np(
                flow_cal_pred_norm, graphs["test"], norm
            )
            rows.append(
                {
                    "dataset": cell_type,
                    "seed": seed,
                    "variant": "self_flow_calibrated",
                    "calibration_alpha": calibration_alpha,
                    "calibration_features": "self_pred+flow_pred+bias",
                    **flow_cal_metrics,
                    **arch.sequence_metric_fields(graphs["test"], flow_cal_pred_px),
                    **arch.paired_block_bootstrap(
                        graphs["test"],
                        flow_pred_px,
                        flow_cal_pred_px,
                        seed=seed + 920_001,
                    ),
                    **{f"flow_{k0}": v for k0, v in flow_info.items()},
                    **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
                }
            )
        for variant in variants:
            print(f"[{cell_type}] seed={seed} nextgen {variant}", flush=True)
            model, info = train_mp_decoder(
                graphs["train"],
                graphs["val"],
                encoded["train"],
                encoded["val"],
                variant=variant,
                seed=seed,
                epochs=mp_epochs,
                sequence_balanced_loss=sequence_balanced_loss,
                layers=layers,
                hidden_dim=hidden_dim,
                edge_hidden_dim=edge_hidden_dim,
                max_delta_norm=max_delta_norm,
                lr=lr,
                social_l2=social_l2,
                flow_gate_l2=flow_gate_l2,
            )
            base_val = float(
                arch.masked_vector_mse(
                    encoded["val"].self_pred + encoded["val"].flow_pred,
                    graphs["val"].y_norm,
                    graphs["val"].target_valid,
                    sequence_groups(graphs["val"]) if sequence_balanced_loss else None,
                )
            )
            val_gain = arch.relative_gain(base_val, float(info["best_val_norm_mse"]))
            pred_px, metrics = evaluate_mp(model, graphs["test"], encoded["test"], norm)
            rows.append(
                {
                    "dataset": cell_type,
                    "seed": seed,
                    "variant": variant,
                    "stage_val_gain_pct": val_gain,
                    "guard_active": False,
                    **metrics,
                    **arch.sequence_metric_fields(graphs["test"], pred_px),
                    **arch.paired_block_bootstrap(
                        graphs["test"],
                        flow_pred_px,
                        pred_px,
                        seed=seed + 900_001,
                    ),
                    **{f"nextgen_{k0}": v for k0, v in info.items()},
                    **{f"flow_{k0}": v for k0, v in flow_info.items()},
                    **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
                }
            )
            if emit_calibrated:
                val_pred, val_gated_flow, val_delta, _ = predict_mp_components(
                    model, graphs["val"], encoded["val"]
                )
                test_pred, test_gated_flow, test_delta, _ = predict_mp_components(
                    model, graphs["test"], encoded["test"]
                )
                del val_pred, test_pred
                mp_val_features = torch.cat(
                    [
                        encoded["val"].self_pred,
                        val_gated_flow,
                        val_delta,
                        bias_column(encoded["val"].self_pred),
                    ],
                    dim=1,
                )
                mp_test_features = torch.cat(
                    [
                        encoded["test"].self_pred,
                        test_gated_flow,
                        test_delta,
                        bias_column(encoded["test"].self_pred),
                    ],
                    dim=1,
                )
                mp_coef = fit_linear_calibrator(
                    mp_val_features,
                    graphs["val"].y_norm,
                    graphs["val"].target_valid,
                    alpha=calibration_alpha,
                )
                mp_cal_pred_norm = apply_linear_calibrator(mp_test_features, mp_coef)
                mp_cal_pred_px, mp_cal_metrics = vector_metrics_from_norm_np(
                    mp_cal_pred_norm, graphs["test"], norm
                )
                rows.append(
                    {
                        "dataset": cell_type,
                        "seed": seed,
                        "variant": f"{variant}_calibrated",
                        "stage_val_gain_pct": val_gain,
                        "calibration_alpha": calibration_alpha,
                        "calibration_features": "self_pred+gated_flow+social_delta+bias",
                        **mp_cal_metrics,
                        **arch.sequence_metric_fields(graphs["test"], mp_cal_pred_px),
                        **arch.paired_block_bootstrap(
                            graphs["test"],
                            flow_pred_px,
                            mp_cal_pred_px,
                            seed=seed + 930_001,
                        ),
                        **{f"nextgen_{k0}": v for k0, v in info.items()},
                        **{f"flow_{k0}": v for k0, v in flow_info.items()},
                        **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
                    }
                )
            if emit_guarded:
                guard_active = val_gain >= float(guard_threshold_pct)
                guarded_pred_px = pred_px if guard_active else flow_pred_px
                guarded_metrics = metrics if guard_active else zero_social_metrics(flow_metrics)
                rows.append(
                    {
                        "dataset": cell_type,
                        "seed": seed,
                        "variant": f"{variant}_guarded",
                        "stage_val_gain_pct": val_gain,
                        "guard_active": guard_active,
                        **guarded_metrics,
                        **arch.sequence_metric_fields(graphs["test"], guarded_pred_px),
                        **arch.paired_block_bootstrap(
                            graphs["test"],
                            flow_pred_px,
                            guarded_pred_px,
                            seed=seed + 910_001,
                        ),
                        **{f"nextgen_{k0}": v for k0, v in info.items()},
                        **{f"flow_{k0}": v for k0, v in flow_info.items()},
                        **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
                    }
                )
            print(
                f"[{cell_type}] seed={seed} {variant}: "
                f"rmse={metrics['rmse_px']:.5f}px "
                f"social={metrics['social_magnitude_mean_px']:.4f}px "
                f"flow_gate={metrics['flow_gate_mean']:.3f}",
                flush=True,
            )
        if device.type == "mps":
            torch.mps.empty_cache()
    return pd.DataFrame(rows), coverage


def summarize(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, part in summary.groupby("dataset"):
        pivot = part.pivot_table(index="seed", columns="variant", values="rmse_px")
        base = pivot["self_flow"] if "self_flow" in pivot else pivot["self_only"]
        for variant in pivot.columns:
            gains = (base - pivot[variant]) / base * 100.0
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seeds": int(pivot[variant].notna().sum()),
                    "rmse_px_mean": float(pivot[variant].mean()),
                    "rmse_px_std": float(pivot[variant].std(ddof=0)),
                    "gain_vs_self_flow_pct_mean": float(gains.mean()),
                    "gain_vs_self_flow_pct_min": float(gains.min()),
                    "positive_seed_fraction": float((gains > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: dict[str, Any],
    out_dir: Path,
) -> None:
    diag_cols = [
        "dataset",
        "variant",
        "rmse_px",
        "r2_vec",
        "social_magnitude_mean_px",
        "social_residual_cosine",
        "node_gate_mean",
        "flow_gate_mean",
        "edge_gate_mean",
        "effective_degree_mean",
    ]
    available = [col for col in diag_cols if col in summary.columns]
    means = (
        summary[available]
        .groupby(["dataset", "variant"], as_index=False)
        .mean(numeric_only=True)
    )
    lines = [
        "# LaChance Next-Gen Message Passing",
        "",
        "This benchmark tests a stronger flow-gated, multi-hop equivariant graph decoder.",
        "It is an architecture-quality gate, not an OZ-prior validation.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(finite_json(coverage), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Aggregate Test Metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Mean Diagnostics",
        "",
        means.to_markdown(index=False),
        "",
        "## Decision Rule",
        "",
        "A successful next-gen branch must beat `self_flow` and the legacy structural decoder on held-out movies, while keeping MDA/HUVEC controls interpretable.",
    ]
    (out_dir / "lachance_nextgen_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_gain(aggregate: pd.DataFrame, out_dir: Path) -> None:
    if aggregate.empty:
        return
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    plot_df = aggregate[aggregate["variant"].ne("self_only")].copy()
    labels = plot_df["dataset"] + "\n" + plot_df["variant"]
    ax.bar(np.arange(len(plot_df)), plot_df["gain_vs_self_flow_pct_mean"], color="#345995")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(np.arange(len(plot_df)), labels, rotation=25, ha="right")
    ax.set_ylabel("Gain over self + flow (%)")
    ax.set_title("Next-gen message passing gate")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "fig01_nextgen_gain.png", dpi=260)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-types", nargs="+", choices=CELL_TYPES, default=["MDCK_Edge"])
    parser.add_argument("--table-root", type=Path, default=la.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-mode", choices=["movie", "frame"], default="movie")
    parser.add_argument("--split-seed", type=int, default=20260608)
    parser.add_argument("--max-movies", type=int, default=8)
    parser.add_argument("--max-tracks-per-movie", type=int, default=0)
    parser.add_argument("--crop-fraction", type=float, default=0.08)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--r-cut-px", type=float, default=50.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["mp_gated_full"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temporal-epochs", type=int, default=35)
    parser.add_argument("--flow-epochs", type=int, default=25)
    parser.add_argument("--mp-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sequence-balanced-loss", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=72)
    parser.add_argument("--edge-hidden-dim", type=int, default=56)
    parser.add_argument("--max-delta-norm", type=float, default=1.35)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--social-l2", type=float, default=1e-4)
    parser.add_argument("--flow-gate-l2", type=float, default=2e-5)
    parser.add_argument("--emit-guarded", action="store_true")
    parser.add_argument("--guard-threshold-pct", type=float, default=0.10)
    parser.add_argument("--emit-calibrated", action="store_true")
    parser.add_argument("--calibration-alpha", type=float, default=1e-2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.cell_types = args.cell_types[:1]
        args.max_movies = min(args.max_movies, 4)
        args.seeds = args.seeds[:1]
        args.temporal_epochs = min(args.temporal_epochs, 3)
        args.flow_epochs = min(args.flow_epochs, 3)
        args.mp_epochs = min(args.mp_epochs, 3)
        args.hidden_dim = min(args.hidden_dim, 40)
        args.edge_hidden_dim = min(args.edge_hidden_dim, 32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = arch.select_device(args.device)
    print(f"device={device}", flush=True)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    all_rows: list[pd.DataFrame] = []
    all_coverage: dict[str, Any] = {}
    for cell_type in args.cell_types:
        rows, coverage = run_cell_type(
            cell_type,
            table_root=args.table_root,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            max_movies=args.max_movies,
            max_tracks_per_movie=args.max_tracks_per_movie,
            frame_stride=args.frame_stride,
            smooth_window=args.smooth_window,
            crop_fraction=args.crop_fraction,
            r_cut_px=args.r_cut_px,
            horizon=args.horizon,
            seeds=args.seeds,
            variants=args.variants,
            device=device,
            k=args.k,
            temporal_epochs=args.temporal_epochs,
            flow_epochs=args.flow_epochs,
            mp_epochs=args.mp_epochs,
            batch_size=args.batch_size,
            sequence_balanced_loss=args.sequence_balanced_loss,
            layers=args.layers,
            hidden_dim=args.hidden_dim,
            edge_hidden_dim=args.edge_hidden_dim,
            max_delta_norm=args.max_delta_norm,
            lr=args.lr,
            social_l2=args.social_l2,
            flow_gate_l2=args.flow_gate_l2,
            emit_guarded=args.emit_guarded,
            guard_threshold_pct=args.guard_threshold_pct,
            emit_calibrated=args.emit_calibrated,
            calibration_alpha=args.calibration_alpha,
        )
        rows.to_csv(args.out_dir / f"lachance_nextgen_summary_{cell_type}.csv", index=False)
        all_rows.append(rows)
        all_coverage[cell_type] = coverage
    summary = pd.concat(all_rows, ignore_index=True)
    aggregate = summarize(summary)
    summary.to_csv(args.out_dir / "lachance_nextgen_summary.csv", index=False)
    aggregate.to_csv(args.out_dir / "lachance_nextgen_aggregate.csv", index=False)
    (args.out_dir / "coverage.json").write_text(
        json.dumps(finite_json(all_coverage), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_gain(aggregate, args.out_dir)
    write_report(summary, aggregate, all_coverage, args.out_dir)
    print(aggregate.to_string(index=False), flush=True)
    print(args.out_dir / "lachance_nextgen_report.md", flush=True)


if __name__ == "__main__":
    main()
