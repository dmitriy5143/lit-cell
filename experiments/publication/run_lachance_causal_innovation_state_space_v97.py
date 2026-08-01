#!/usr/bin/env python3
"""v97: causal online h1 forecasting with learned innovation filtering.

The model is evaluated under a strict streaming contract.  At frame ``t`` it
issues a one-step prediction.  At frame ``t+1`` the completed transition is
available, its error becomes an innovation measurement, and only then may the
hidden state be updated.  No target at or after the prediction time is used.

Unlike v96, v97 is trained for this exact contract: chronological track replay,
one-step Huber + Student-t NLL, cumulative rolling losses, explicit process and
observation noise heads, and measurement dropout/noise augmentation.  The
graph is optional and is treated as an ablation rather than part of the core.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
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

import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_graph_state_space_seq2seq_v96 as v96  # noqa: E402
import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


EPS = 1e-8
KEYS = ["sequence", "frame", "track_id"]
DEFAULT_OUT = ROOT / "outputs" / "causal_innovation_state_space_v97_2026-07-21"


@dataclass(frozen=True)
class TrainVariant:
    name: str
    use_context: bool = True
    use_update: bool = True
    use_graph: bool = False
    output_mode: str = "anchor_residual"
    track_only: bool = False


@dataclass
class ReplayEntry:
    frame: int
    state: torch.Tensor
    prediction: torch.Tensor
    last_innovation: torch.Tensor


@dataclass
class WindowEntry:
    frame: int
    predictions: list[torch.Tensor]
    targets: list[torch.Tensor]


@dataclass
class ReplayResult:
    prediction: np.ndarray
    scale: np.ndarray
    process_scale: np.ndarray
    observation_scale: np.ndarray
    gain: np.ndarray
    measurement_mask: np.ndarray


def safe(value: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def parse_ints(value: str | Iterable[int]) -> list[int]:
    if not isinstance(value, str):
        return [int(x) for x in value]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if not isinstance(value, str):
        return [float(x) for x in value]
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_strings(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_horizon_weights(value: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for token in parse_strings(value):
        horizon, weight = token.split(":", maxsplit=1)
        weights[int(horizon)] = float(weight)
    if not weights or any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError(f"Invalid horizon weights: {value!r}")
    return weights


def device_from_args(args: argparse.Namespace) -> torch.device:
    if args.device != "auto":
        return torch.device(args.device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_variants(args: argparse.Namespace) -> list[TrainVariant]:
    available = {
        "v97_core": TrainVariant("v97_core", use_context=True, use_update=True, use_graph=False),
        "v97_no_context": TrainVariant("v97_no_context", use_context=False, use_update=True, use_graph=False),
        "v97_no_update": TrainVariant("v97_no_update", use_context=True, use_update=False, use_graph=False),
        "v97_graph": TrainVariant("v97_graph", use_context=True, use_update=True, use_graph=True),
        "v97_direct": TrainVariant(
            "v97_direct",
            use_context=False,
            use_update=True,
            use_graph=False,
            output_mode="direct",
        ),
        "v97_direct_context": TrainVariant(
            "v97_direct_context",
            use_context=True,
            use_update=True,
            use_graph=False,
            output_mode="direct",
        ),
        "v97_track_only": TrainVariant(
            "v97_track_only",
            use_context=False,
            use_update=True,
            use_graph=False,
            output_mode="direct",
            track_only=True,
        ),
    }
    names = parse_strings(args.variants)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown v97 variants: {missing}")
    return [available[name] for name in names]


def load_prepared(args: argparse.Namespace, variant: TrainVariant) -> v96.Prepared:
    bundles = v85.load_anchor_cache(args.anchor_cache)
    if variant.track_only:
        track_bundles: list[v84.AnchorBundle] = []
        for bundle in bundles:
            current_velocity = bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
            track_bundles.append(
                v84.AnchorBundle(
                    name=bundle.name,
                    rows=bundle.rows.copy(),
                    anchor_residual=np.zeros_like(bundle.anchor_residual, dtype=np.float32),
                    base=current_velocity,
                    target_steps=bundle.target_steps.copy(),
                    meta={
                        **bundle.meta,
                        "anchor_method": "constant_velocity_track_only",
                        "track_only": True,
                    },
                )
            )
        bundles = tuple(track_bundles)
    quotas = v96.parse_context_quotas(args.context_quotas)
    context, names, _diagnostics = v96.load_context(args.features, bundles, quotas)
    legacy_variant = v96.Variant(
        name=variant.name,
        use_graph=variant.use_graph,
        use_update=variant.use_update,
        use_context=variant.use_context,
        use_history=True,
        use_history_innovation=True,
    )
    return v96.prepare(args, legacy_variant, bundles, context, names)


class CausalInnovationStateSpaceForecaster(nn.Module):
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
        self.use_update = bool(use_update)
        self.use_graph = bool(use_graph)
        self.output_mode = str(output_mode)
        if self.output_mode not in {"anchor_residual", "direct"}:
            raise ValueError(f"Unknown output mode: {self.output_mode}")
        target_mean_array = np.zeros(2, dtype=np.float32) if target_mean is None else np.asarray(target_mean, dtype=np.float32)
        target_scale_array = np.ones(2, dtype=np.float32) if target_scale is None else np.asarray(target_scale, dtype=np.float32)
        self.register_buffer("target_mean", torch.as_tensor(target_mean_array, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(np.maximum(target_scale_array, 1e-4), dtype=torch.float32))

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.history_input = nn.Sequential(nn.Linear(5, hidden // 2), nn.LayerNorm(hidden // 2), nn.SiLU())
        self.history_encoder = nn.GRU(hidden // 2, hidden, batch_first=True)
        self.history_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.history_norm = nn.LayerNorm(hidden)

        self.initial_state = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.transition = nn.GRUCell(hidden, hidden)

        self.process_scale_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 2)
        )
        self.observation_scale_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 2)
        )
        self.update_encoder = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.gain_head = nn.Sequential(
            nn.Linear(hidden * 2 + 7, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
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
            self.graph_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
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
        recurrent = torch.where(has_previous_state[:, None] > 0.5, previous_state, initialized)
        prior = self.transition(encoded, recurrent)

        process_scale = F.softplus(self.process_scale_head(torch.cat([prior, encoded], dim=-1))) + 1e-3
        observation_scale = F.softplus(self.observation_scale_head(torch.cat([prior, encoded], dim=-1))) + 1e-3
        total_scale = torch.sqrt(torch.square(process_scale) + torch.square(observation_scale) + 1e-6)
        normalized_innovation = innovation / total_scale
        noise_packet = torch.cat(
            [normalized_innovation, torch.log(process_scale), torch.log(observation_scale), measurement_mask[:, None]],
            dim=-1,
        )
        update = self.update_encoder(noise_packet)
        analytic_gain = torch.square(process_scale) / (
            torch.square(process_scale) + torch.square(observation_scale) + 1e-6
        )
        analytic_logit = torch.logit(torch.clamp(analytic_gain.mean(dim=1, keepdim=True), 1e-4, 1.0 - 1e-4))
        learned_logit = self.gain_head(torch.cat([prior, encoded, noise_packet], dim=-1))
        gain = torch.sigmoid(learned_logit + analytic_logit)
        if self.use_update:
            posterior = self.update_norm(prior + measurement_mask[:, None] * gain * update)
        else:
            posterior = prior
            gain = torch.zeros_like(gain)

        if self.use_graph and edge_index.shape[1] > 0:
            graph_update = F.silu(self.graph_conv(posterior, edge_index, edge_attr))
            graph_gate = self.graph_gate(torch.cat([posterior, graph_update], dim=-1))
            posterior = self.graph_norm(
                posterior + torch.sigmoid(self.raw_graph_strength) * graph_gate * graph_update
            )

        forecast = self.forecast_trunk(torch.cat([posterior, encoded, anchor_h1], dim=-1))
        raw_mean = self.mean_head(forecast)
        mean = self.correction_bound * torch.tanh(raw_mean / max(self.correction_bound, 1e-4))
        aleatoric = F.softplus(self.scale_head(forecast)) + 1e-3
        predictive_scale = torch.sqrt(torch.square(aleatoric) + torch.square(process_scale) + 1e-6)
        return posterior, mean, predictive_scale, process_scale, observation_scale, gain

    def normalize_direct_target(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.target_mean) / self.target_scale

    def denormalize_direct_target(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.target_scale + self.target_mean


def inverse_error_tensor(prep: v96.Prepared, normalized: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(prep.error_scaler.mean_, dtype=normalized.dtype, device=normalized.device)
    scale = torch.as_tensor(prep.error_scaler.scale_, dtype=normalized.dtype, device=normalized.device)
    return normalized * scale + mean


def physical_scale_tensor(prep: v96.Prepared, normalized_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(prep.error_scaler.scale_, dtype=normalized_scale.dtype, device=normalized_scale.device)
    return normalized_scale * scale


def prediction_from_output(
    model: CausalInnovationStateSpaceForecaster,
    prep: v96.Prepared,
    mean: torch.Tensor,
    anchor_physical: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    if model.output_mode == "direct":
        direct = model.denormalize_direct_target(mean)
        return anchor_physical + float(eta) * (direct - anchor_physical)
    return anchor_physical + float(eta) * inverse_error_tensor(prep, mean)


def target_for_output(
    model: CausalInnovationStateSpaceForecaster,
    residual_target: torch.Tensor,
    physical_target: torch.Tensor,
) -> torch.Tensor:
    if model.output_mode == "direct":
        return model.normalize_direct_target(physical_target)
    return residual_target


def output_scale_physical(
    model: CausalInnovationStateSpaceForecaster,
    prep: v96.Prepared,
    normalized_scale: torch.Tensor,
) -> torch.Tensor:
    if model.output_mode == "direct":
        return normalized_scale * model.target_scale
    return physical_scale_tensor(prep, normalized_scale)


def grouped_frames(prep: v96.Prepared, split: int) -> dict[int, list[v96.FrameSpec]]:
    return v96.sequence_frames(prep.frames[split])


def previous_row_lookup(bundle: v84.AnchorBundle) -> np.ndarray:
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None))
    }
    out = np.full(len(bundle.rows), -1, dtype=np.int64)
    for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None)):
        out[i] = lookup.get((int(sequence), int(frame) - 1, int(track)), -1)
    return out


def gather_replay_inputs(
    frame: v96.FrameSpec,
    cache: dict[int, ReplayEntry],
    bundle: v84.AnchorBundle,
    previous_rows: np.ndarray,
    hidden: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[ReplayEntry | None]]:
    zero_state = torch.zeros(hidden, dtype=torch.float32, device=device)
    states: list[torch.Tensor] = []
    present: list[float] = []
    innovations: list[torch.Tensor] = []
    entries: list[ReplayEntry | None] = []
    for local_index, track in enumerate(frame.track_ids):
        row_index = int(frame.indices[local_index])
        previous_index = int(previous_rows[row_index])
        entry = cache.get(int(track))
        valid = entry is not None and entry.frame == frame.frame - 1 and previous_index >= 0
        if valid:
            observed = torch.as_tensor(
                bundle.target_steps[previous_index, 0], dtype=torch.float32, device=device
            )
            states.append(entry.state)
            innovations.append(observed - entry.prediction)
            present.append(1.0)
            entries.append(entry)
        else:
            states.append(zero_state)
            innovations.append(torch.zeros(2, dtype=torch.float32, device=device))
            present.append(0.0)
            entries.append(None)
    return (
        torch.stack(states),
        torch.tensor(present, dtype=torch.float32, device=device),
        torch.stack(innovations),
        entries,
    )


def deterministic_rng(seed: int, sequence: int, frame: int, salt: int = 0) -> np.random.Generator:
    mixed = int(seed) * 1_000_003 + int(sequence) * 10_007 + int(frame) * 101 + int(salt)
    return np.random.default_rng(mixed % (2**32 - 1))


def measurement_policy(
    innovation_physical: torch.Tensor,
    mask: torch.Tensor,
    entries: list[ReplayEntry | None],
    reservoir: list[torch.Tensor],
    prep: v96.Prepared,
    frame: v96.FrameSpec,
    *,
    control: str,
    update_stride: int,
    missing_rate: float,
    coordinate_noise_px: float,
    seed: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    real = innovation_physical
    used = real.clone()
    out_mask = mask.clone()
    valid = torch.nonzero(out_mask > 0.5, as_tuple=False).flatten()

    if control == "no_update":
        out_mask.zero_()
    elif control == "wrong_cell":
        if len(valid) > 1:
            used_valid = used[valid]
            used[valid] = torch.roll(used_valid, shifts=1, dims=0)
    elif control == "time_shuffled":
        if reservoir and len(valid):
            rng = deterministic_rng(seed, frame.sequence, frame.frame, 719)
            chosen = rng.integers(0, len(reservoir), size=len(valid))
            replacement = torch.stack([reservoir[int(i)].to(used.device) for i in chosen])
            used[valid] = replacement
        else:
            out_mask.zero_()
    elif control == "delayed":
        for local_index in valid.tolist():
            entry = entries[int(local_index)]
            if entry is None:
                out_mask[int(local_index)] = 0.0
            else:
                used[int(local_index)] = entry.last_innovation.to(used.device)
    elif control != "real":
        raise ValueError(f"Unknown measurement control: {control}")

    if int(update_stride) > 1 and frame.frame % int(update_stride) != 0:
        out_mask.zero_()

    if missing_rate > 0:
        if training:
            keep = torch.rand(len(out_mask), device=out_mask.device) >= float(missing_rate)
            out_mask *= keep.float()
        else:
            rng = deterministic_rng(seed, frame.sequence, frame.frame, 991)
            keep = torch.as_tensor(
                rng.random(len(out_mask)) >= float(missing_rate), dtype=torch.float32, device=out_mask.device
            )
            out_mask *= keep

    if coordinate_noise_px > 0:
        # A displacement is the difference of two noisy coordinates.
        sigma = math.sqrt(2.0) * float(coordinate_noise_px)
        if training:
            noise = torch.randn_like(used) * sigma
        else:
            rng = deterministic_rng(seed, frame.sequence, frame.frame, 1237)
            noise = torch.as_tensor(rng.normal(0.0, sigma, used.shape), dtype=used.dtype, device=used.device)
        used = used + noise * out_mask[:, None]

    scale = torch.as_tensor(prep.error_scaler.scale_, dtype=used.dtype, device=used.device)
    normalized = used / torch.clamp(scale, min=1e-4)
    return normalized, out_mask


def append_cumulative_terms(
    frame: v96.FrameSpec,
    prediction: torch.Tensor,
    target: torch.Tensor,
    windows: dict[int, WindowEntry],
    horizons: list[int],
    displacement_scale: torch.Tensor,
) -> list[torch.Tensor]:
    terms: list[torch.Tensor] = []
    max_horizon = max(horizons, default=1)
    for local_index, track_value in enumerate(frame.track_ids):
        track = int(track_value)
        old = windows.get(track)
        if old is None or old.frame != frame.frame - 1:
            predictions: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
        else:
            predictions = old.predictions
            targets = old.targets
        predictions = (predictions + [prediction[local_index]])[-max_horizon:]
        targets = (targets + [target[local_index]])[-max_horizon:]
        windows[track] = WindowEntry(frame.frame, predictions, targets)
        for horizon in horizons:
            if horizon <= 1 or len(predictions) < horizon:
                continue
            pred_sum = torch.stack(predictions[-horizon:]).sum(dim=0)
            target_sum = torch.stack(targets[-horizon:]).sum(dim=0)
            terms.append(F.smooth_l1_loss(pred_sum / displacement_scale, target_sum / displacement_scale))
    return terms


def detach_replay_cache(cache: dict[int, ReplayEntry]) -> dict[int, ReplayEntry]:
    return {
        track: ReplayEntry(
            frame=entry.frame,
            state=entry.state.detach(),
            prediction=entry.prediction.detach(),
            last_innovation=entry.last_innovation.detach(),
        )
        for track, entry in cache.items()
    }


def detach_windows(windows: dict[int, WindowEntry]) -> dict[int, WindowEntry]:
    return {
        track: WindowEntry(
            frame=entry.frame,
            predictions=[value.detach() for value in entry.predictions],
            targets=[value.detach() for value in entry.targets],
        )
        for track, entry in windows.items()
    }


def frame_arrays(prep: v96.Prepared, split: int, frame: v96.FrameSpec, device: torch.device) -> tuple[torch.Tensor, ...]:
    indices = frame.indices
    return (
        torch.as_tensor(prep.static[split][indices], dtype=torch.float32, device=device),
        torch.as_tensor(prep.history[split][indices], dtype=torch.float32, device=device),
        torch.as_tensor(prep.anchor_decoder[split][indices, 0], dtype=torch.float32, device=device),
        torch.as_tensor(prep.bundles[split].anchor_steps[indices, 0], dtype=torch.float32, device=device),
        torch.as_tensor(prep.targets[split][indices, 0], dtype=torch.float32, device=device),
        torch.as_tensor(prep.bundles[split].target_steps[indices, 0], dtype=torch.float32, device=device),
        torch.as_tensor(frame.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(frame.edge_attr, dtype=torch.float32, device=device),
    )


def step_objective(
    model: CausalInnovationStateSpaceForecaster,
    mean: torch.Tensor,
    predictive_scale: torch.Tensor,
    target: torch.Tensor,
    innovation: torch.Tensor,
    measurement_mask: torch.Tensor,
    process_scale: torch.Tensor,
    observation_scale: torch.Tensor,
    gain: torch.Tensor,
    cumulative_terms: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    huber = F.smooth_l1_loss(mean, target)
    distribution = torch.distributions.StudentT(model.degrees_of_freedom, loc=mean, scale=predictive_scale)
    nll = -distribution.log_prob(target).mean()

    total_noise = torch.sqrt(torch.square(process_scale) + torch.square(observation_scale) + 1e-6)
    if torch.any(measurement_mask > 0.5):
        innovation_dist = torch.distributions.StudentT(model.degrees_of_freedom, loc=0.0, scale=total_noise)
        per_row = -innovation_dist.log_prob(innovation).mean(dim=1)
        innovation_nll = torch.sum(per_row * measurement_mask) / torch.clamp(measurement_mask.sum(), min=1.0)
    else:
        innovation_nll = mean.new_tensor(0.0)

    cumulative = torch.stack(cumulative_terms).mean() if cumulative_terms else mean.new_tensor(0.0)
    # The two scales are separately parameterized but only their sum is fully
    # identifiable.  Weak priors keep the decomposition finite without forcing
    # a physical interpretation unsupported by tracking labels.
    process_prior = torch.square(torch.log(process_scale + 1e-6) - math.log(float(args.process_scale_prior))).mean()
    observation_prior = torch.square(
        torch.log(observation_scale + 1e-6) - math.log(float(args.observation_scale_prior))
    ).mean()
    noise_regularization = process_prior + observation_prior
    gain_regularization = torch.square(gain).mean()
    loss = (
        huber
        + float(args.nll_weight) * nll
        + float(args.cumulative_weight) * cumulative
        + float(args.innovation_nll_weight) * innovation_nll
        + float(args.noise_regularization_weight) * noise_regularization
        + float(args.gain_weight) * gain_regularization
    )
    return loss, {
        "huber": float(huber.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "cumulative": float(cumulative.detach().cpu()),
        "innovation_nll": float(innovation_nll.detach().cpu()),
        "process_scale": float(process_scale.detach().mean().cpu()),
        "observation_scale": float(observation_scale.detach().mean().cpu()),
        "gain": float(gain.detach().mean().cpu()),
    }


def train_epoch(
    model: CausalInnovationStateSpaceForecaster,
    prep: v96.Prepared,
    variant: TrainVariant,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    bundle = prep.bundles[0]
    previous_rows = previous_row_lookup(bundle)
    grouped = grouped_frames(prep, 0)
    displacement_scale = torch.as_tensor(
        np.maximum(np.std(bundle.target_steps[:, 0], axis=0), 1.0), dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(int(args.seed) + epoch * 37)
    order = list(grouped)
    rng.shuffle(order)
    records: list[dict[str, float]] = []
    cumulative_horizons = parse_ints(args.cumulative_horizons)

    for sequence in order:
        replay_cache: dict[int, ReplayEntry] = {}
        windows: dict[int, WindowEntry] = {}
        reservoir: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        chunk_losses: list[torch.Tensor] = []
        for frame_number, frame in enumerate(grouped[sequence]):
            static, history, anchor_normalized, anchor_physical, target_normalized, target_physical, edge_index, edge_attr = frame_arrays(
                prep, 0, frame, device
            )
            previous_state, has_previous, innovation_physical, entries = gather_replay_inputs(
                frame, replay_cache, bundle, previous_rows, model.hidden, device
            )
            innovation, measurement_mask = measurement_policy(
                innovation_physical,
                has_previous,
                entries,
                reservoir,
                prep,
                frame,
                control="real" if variant.use_update else "no_update",
                update_stride=1,
                missing_rate=float(args.train_missing_rate),
                coordinate_noise_px=float(args.train_coordinate_noise_px),
                seed=int(args.seed) + epoch,
                training=True,
            )
            posterior, mean, predictive_scale, process_scale, observation_scale, gain = model.forward_step(
                static,
                history,
                innovation,
                measurement_mask,
                previous_state,
                has_previous,
                anchor_normalized,
                edge_index,
                edge_attr,
            )
            final_prediction = prediction_from_output(model, prep, mean, anchor_physical)
            output_target = target_for_output(model, target_normalized, target_physical)
            cumulative_terms = append_cumulative_terms(
                frame,
                final_prediction,
                target_physical,
                windows,
                cumulative_horizons,
                displacement_scale,
            )
            loss, details = step_objective(
                model,
                mean,
                predictive_scale,
                output_target,
                innovation,
                measurement_mask,
                process_scale,
                observation_scale,
                gain,
                cumulative_terms,
                args,
            )
            chunk_losses.append(loss)
            records.append(details)
            for local_index, track_value in enumerate(frame.track_ids):
                replay_cache[int(track_value)] = ReplayEntry(
                    frame=frame.frame,
                    state=posterior[local_index],
                    prediction=final_prediction[local_index],
                    last_innovation=innovation_physical[local_index].detach(),
                )
            reservoir.extend([value.detach().cpu() for value in innovation_physical[has_previous > 0.5]])
            if len(reservoir) > int(args.reservoir_size):
                reservoir = reservoir[-int(args.reservoir_size) :]

            boundary = len(chunk_losses) >= int(args.tbptt_frames) or frame_number == len(grouped[sequence]) - 1
            if boundary:
                chunk_loss = torch.stack(chunk_losses).mean()
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                replay_cache = detach_replay_cache(replay_cache)
                windows = detach_windows(windows)
                chunk_losses = []
    if not records:
        return {}
    return {key: float(np.mean([row[key] for row in records])) for key in records[0]}


@torch.no_grad()
def replay_inference(
    model: CausalInnovationStateSpaceForecaster,
    prep: v96.Prepared,
    split: int,
    device: torch.device,
    *,
    eta: float,
    control: str = "real",
    update_stride: int = 1,
    missing_rate: float = 0.0,
    coordinate_noise_px: float = 0.0,
    seed: int = 42,
) -> ReplayResult:
    model.eval()
    bundle = prep.bundles[split]
    previous_rows = previous_row_lookup(bundle)
    n = len(bundle.rows)
    prediction = np.zeros((n, 2), dtype=np.float32)
    scale = np.zeros((n, 2), dtype=np.float32)
    process_out = np.zeros((n, 2), dtype=np.float32)
    observation_out = np.zeros((n, 2), dtype=np.float32)
    gain_out = np.zeros(n, dtype=np.float32)
    mask_out = np.zeros(n, dtype=np.float32)

    for sequence in sorted(grouped_frames(prep, split)):
        replay_cache: dict[int, ReplayEntry] = {}
        reservoir: list[torch.Tensor] = []
        for frame in grouped_frames(prep, split)[sequence]:
            static, history, anchor_normalized, anchor_physical, _target_normalized, _target_physical, edge_index, edge_attr = frame_arrays(
                prep, split, frame, device
            )
            previous_state, has_previous, innovation_physical, entries = gather_replay_inputs(
                frame, replay_cache, bundle, previous_rows, model.hidden, device
            )
            innovation, measurement_mask = measurement_policy(
                innovation_physical,
                has_previous,
                entries,
                reservoir,
                prep,
                frame,
                control=control,
                update_stride=update_stride,
                missing_rate=missing_rate,
                coordinate_noise_px=coordinate_noise_px,
                seed=seed,
                training=False,
            )
            posterior, mean, predictive_scale, process_scale, observation_scale, gain = model.forward_step(
                static,
                history,
                innovation,
                measurement_mask,
                previous_state,
                has_previous,
                anchor_normalized,
                edge_index,
                edge_attr,
            )
            final_prediction = prediction_from_output(model, prep, mean, anchor_physical, eta)
            physical_scale = output_scale_physical(model, prep, predictive_scale)
            indices = frame.indices
            prediction[indices] = final_prediction.cpu().numpy()
            scale[indices] = physical_scale.cpu().numpy()
            process_out[indices] = physical_scale_tensor(prep, process_scale).cpu().numpy()
            observation_out[indices] = physical_scale_tensor(prep, observation_scale).cpu().numpy()
            gain_out[indices] = gain.mean(dim=1).cpu().numpy()
            mask_out[indices] = measurement_mask.cpu().numpy()
            for local_index, track_value in enumerate(frame.track_ids):
                replay_cache[int(track_value)] = ReplayEntry(
                    frame=frame.frame,
                    state=posterior[local_index].detach(),
                    prediction=final_prediction[local_index].detach(),
                    last_innovation=innovation_physical[local_index].detach(),
                )
            reservoir.extend([value.detach().cpu() for value in innovation_physical[has_previous > 0.5]])
            if len(reservoir) > 4096:
                reservoir = reservoir[-4096:]
    return ReplayResult(prediction, scale, process_out, observation_out, gain_out, mask_out)


def rolling_examples(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None))
    }
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    starts: list[int] = []
    clusters: list[str] = []
    for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None)):
        chain = [lookup.get((int(sequence), int(frame) + offset, int(track))) for offset in range(int(horizon))]
        if any(index is None for index in chain):
            continue
        indices = np.asarray(chain, dtype=np.int64)
        targets.append(bundle.target_steps[i, :horizon].sum(axis=0))
        predictions.append(prediction[indices].sum(axis=0))
        starts.append(i)
        clusters.append(f"{int(sequence)}:{int(track)}")
    return safe(targets), safe(predictions), np.asarray(starts, dtype=np.int64), np.asarray(clusters, dtype=object)


def rolling_metric_rows(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    horizons: list[int],
    method: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target, pred, _starts, _clusters = rolling_examples(bundle, prediction, horizon)
        row: dict[str, Any] = {
            "method": method,
            "contract": "streaming_receding_h1",
            "horizon": int(horizon),
            "component_rmse": v84.component_rmse(target, pred),
            "vector_rmse": v84.vector_rmse(target, pred),
            "r2": v84.vector_r2(target, pred),
            "cosine": v84.cosine_mean(target, pred),
            "magnitude_ratio": v84.magnitude_ratio(target, pred),
            "n_rows": int(len(target)),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def weighted_rolling_score(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    weights: dict[int, float] | None = None,
) -> float:
    weights = weights or {1: 0.45, 2: 0.25, 4: 0.18, 6: 0.12}
    scores = []
    values = []
    for horizon, weight in weights.items():
        target, pred, _starts, _clusters = rolling_examples(bundle, prediction, horizon)
        scores.append(v84.component_rmse(target, pred))
        values.append(weight)
    return float(np.average(scores, weights=values))


def calibrate_uncertainty(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: float,
    factors: list[float],
) -> float:
    target = bundle.target_steps[:, 0]
    best_factor = 1.0
    best_nll = float("inf")
    for factor in factors:
        effective = np.maximum(scale * float(factor), 1e-3)
        standardized = (target - prediction) / effective
        nll = -float(np.mean(student_t.logpdf(standardized, df=degrees_of_freedom) - np.log(effective)))
        if nll < best_nll:
            best_nll = nll
            best_factor = float(factor)
    return best_factor


def uncertainty_rows(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    scale: np.ndarray,
    horizons: list[int],
    method: str,
    degrees_of_freedom: float,
    calibration_factor: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    effective_step = np.maximum(scale * float(calibration_factor), 1e-3)
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None))
    }
    for horizon in horizons:
        target, pred, starts, _clusters = rolling_examples(bundle, prediction, horizon)
        cumulative_scale: list[np.ndarray] = []
        for start in starts:
            sequence, frame, track = bundle.rows.iloc[int(start)][KEYS]
            indices = [lookup[(int(sequence), int(frame) + offset, int(track))] for offset in range(horizon)]
            cumulative_scale.append(np.sqrt(np.sum(np.square(effective_step[indices]), axis=0)))
        cumulative_scale_array = np.maximum(safe(cumulative_scale), 1e-3)
        standardized = (target - pred) / cumulative_scale_array
        if horizon == 1:
            nll = -float(
                np.mean(
                    student_t.logpdf(standardized, df=degrees_of_freedom)
                    - np.log(cumulative_scale_array)
                )
            )
            q50 = float(student_t.ppf(0.75, df=degrees_of_freedom))
            q90 = float(student_t.ppf(0.95, df=degrees_of_freedom))
            approximation = "student_t_exact"
        else:
            nll = -float(np.mean(norm.logpdf(standardized) - np.log(cumulative_scale_array)))
            q50 = float(norm.ppf(0.75))
            q90 = float(norm.ppf(0.95))
            approximation = "independent_moment_normal"
        absolute = np.abs(target - pred)
        coverage50 = float(np.mean(absolute <= q50 * cumulative_scale_array))
        coverage90 = float(np.mean(absolute <= q90 * cumulative_scale_array))
        uncertainty = cumulative_scale_array.mean(axis=1)
        error = np.linalg.norm(target - pred, axis=1)
        corr = float(np.corrcoef(uncertainty, error)[0, 1]) if np.std(uncertainty) > EPS else 0.0
        rows.append(
            {
                "method": method,
                "horizon": horizon,
                "nll": nll,
                "coverage_50": coverage50,
                "coverage_90": coverage90,
                "calibration_error": abs(coverage50 - 0.5) + abs(coverage90 - 0.9),
                "uncertainty_error_corr": corr,
                "scale_factor": calibration_factor,
                "approximation": approximation,
            }
        )
    return rows


def rolling_scale_array(
    bundle: v84.AnchorBundle,
    scale: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Accumulate per-step scales for the same rolling examples as the metrics."""

    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(
            bundle.rows[KEYS].itertuples(index=False, name=None)
        )
    }
    cumulative: list[np.ndarray] = []
    effective = np.maximum(safe(scale), 1e-3)
    for start in starts:
        sequence, frame, track = bundle.rows.iloc[int(start)][KEYS]
        indices = [
            lookup[(int(sequence), int(frame) + offset, int(track))]
            for offset in range(int(horizon))
        ]
        cumulative.append(np.sqrt(np.sum(np.square(effective[indices]), axis=0)))
    return np.maximum(safe(cumulative), 1e-3)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    level = min(math.ceil((len(values) + 1) * float(coverage)) / len(values), 1.0)
    try:
        return float(np.quantile(values, level, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, level, interpolation="higher"))


def conformal_uncertainty_rows(
    validation_bundle: v84.AnchorBundle,
    validation_prediction: np.ndarray,
    validation_scale: np.ndarray,
    test_bundle: v84.AnchorBundle,
    test_prediction: np.ndarray,
    test_scale: np.ndarray,
    horizons: list[int],
    method: str,
) -> list[dict[str, Any]]:
    """Validation-only normalized split-conformal marginal intervals."""

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        val_target, val_pred, val_starts, _ = rolling_examples(
            validation_bundle, validation_prediction, horizon
        )
        test_target, test_pred, test_starts, _ = rolling_examples(
            test_bundle, test_prediction, horizon
        )
        val_cumulative_scale = rolling_scale_array(
            validation_bundle, validation_scale, val_starts, horizon
        )
        test_cumulative_scale = rolling_scale_array(test_bundle, test_scale, test_starts, horizon)
        calibration_scores = np.abs(val_target - val_pred) / val_cumulative_scale
        q50 = finite_sample_quantile(calibration_scores, 0.50)
        q90 = finite_sample_quantile(calibration_scores, 0.90)
        absolute = np.abs(test_target - test_pred)
        coverage50 = float(np.mean(absolute <= q50 * test_cumulative_scale))
        coverage90 = float(np.mean(absolute <= q90 * test_cumulative_scale))
        rows.append(
            {
                "method": method,
                "horizon": int(horizon),
                "coverage_50": coverage50,
                "coverage_90": coverage90,
                "calibration_error": abs(coverage50 - 0.50) + abs(coverage90 - 0.90),
                "q50": q50,
                "q90": q90,
                "mean_half_width_50": float(np.mean(q50 * test_cumulative_scale)),
                "mean_half_width_90": float(np.mean(q90 * test_cumulative_scale)),
                "validation_scores": int(calibration_scores.size),
                "test_values": int(absolute.size),
                "calibration": "normalized_split_conformal_validation_only",
            }
        )
    return rows


def tune_eta(
    model: CausalInnovationStateSpaceForecaster,
    prep: v96.Prepared,
    device: torch.device,
    eta_grid: list[float],
    seed: int,
    horizon_weights: dict[int, float] | None = None,
) -> tuple[float, float]:
    best_eta = 1.0
    best_score = float("inf")
    for eta in eta_grid:
        result = replay_inference(model, prep, 1, device, eta=eta, seed=seed)
        score = weighted_rolling_score(prep.bundles[1], result.prediction, horizon_weights)
        if score < best_score:
            best_score = score
            best_eta = float(eta)
    return best_eta, best_score


def train_model(
    prep: v96.Prepared,
    variant: TrainVariant,
    args: argparse.Namespace,
) -> tuple[CausalInnovationStateSpaceForecaster, float, dict[str, Any], list[dict[str, Any]]]:
    device = device_from_args(args)
    seed_everything(int(args.seed))
    direct_target = prep.bundles[0].target_steps[:, 0].astype(np.float32)
    direct_target_mean = direct_target.mean(axis=0)
    direct_target_scale = np.maximum(direct_target.std(axis=0), 1e-4)
    model = CausalInnovationStateSpaceForecaster(
        static_dim=prep.static_dim,
        hidden=int(args.hidden),
        history_lags=int(args.history_lags),
        correction_bound=float(args.correction_bound),
        dropout=float(args.dropout),
        use_update=variant.use_update,
        use_graph=variant.use_graph,
        graph_heads=int(args.graph_heads),
        output_mode=variant.output_mode,
        target_mean=direct_target_mean,
        target_scale=direct_target_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    patience = 0
    logs: list[dict[str, Any]] = []
    horizon_weights = parse_horizon_weights(args.validation_horizon_weights)

    for epoch in range(1, int(args.epochs) + 1):
        stats = train_epoch(model, prep, variant, optimizer, device, args, epoch)
        val_result = replay_inference(model, prep, 1, device, eta=1.0, seed=int(args.seed))
        val_score = weighted_rolling_score(prep.bundles[1], val_result.prediction, horizon_weights)
        row = {"variant": variant.name, "epoch": epoch, "val_weighted_rolling_rmse": val_score, **stats}
        logs.append(row)
        print(f"[v97] {variant.name} epoch={epoch:02d} val={val_score:.6f}", flush=True)
        if np.isfinite(val_score) and val_score < best_score - float(args.min_delta):
            best_score = val_score
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
    eta, tuned_score = tune_eta(
        model,
        prep,
        device,
        parse_floats(args.eta_grid),
        int(args.seed),
        horizon_weights,
    )
    metadata = {
        "best_epoch": best_epoch,
        "best_val_score_eta1": best_score,
        "best_val_score_tuned": tuned_score,
        "eta": eta,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "degrees_of_freedom": float(model.degrees_of_freedom.detach().cpu()),
        "device": str(device),
        "use_context": variant.use_context,
        "track_only": variant.track_only,
        "use_update": variant.use_update,
        "use_graph": variant.use_graph,
        "output_mode": variant.output_mode,
        "target_mean": direct_target_mean.tolist(),
        "target_scale": direct_target_scale.tolist(),
        "graph_strength": (
            float(torch.sigmoid(model.raw_graph_strength).detach().cpu()) if variant.use_graph else 0.0
        ),
    }
    return model, eta, metadata, logs


def velocity_history(bundle: v84.AnchorBundle, lags: int) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (int(sequence), int(frame), int(track)): i
        for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None))
    }
    velocity = bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    values = np.zeros((len(bundle.rows), lags, 2), dtype=np.float32)
    mask = np.zeros((len(bundle.rows), lags), dtype=np.float32)
    for i, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None)):
        for position, lag in enumerate(range(lags - 1, -1, -1)):
            index = lookup.get((int(sequence), int(frame) - lag, int(track)))
            if index is None:
                continue
            values[i, position] = velocity[index]
            mask[i, position] = 1.0
    return values, mask


class RecurrentH1Baseline(nn.Module):
    def __init__(self, kind: str, hidden: int) -> None:
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent(3, hidden, num_layers=2, batch_first=True, dropout=0.05)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _state = self.recurrent(values)
        return self.head(output[:, -1])


def fit_recurrent_baseline(
    kind: str,
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    train, val, test = bundles
    histories = [velocity_history(bundle, int(args.baseline_history_lags)) for bundle in bundles]
    velocity_scaler = StandardScaler().fit(histories[0][0].reshape(-1, 2))
    target_scaler = StandardScaler().fit(train.target_steps[:, 0])
    tensors: list[np.ndarray] = []
    for values, mask in histories:
        normalized = velocity_scaler.transform(values.reshape(-1, 2)).reshape(values.shape)
        normalized *= mask[:, :, None]
        tensors.append(np.concatenate([normalized, mask[:, :, None]], axis=2).astype(np.float32))
    targets = [target_scaler.transform(bundle.target_steps[:, 0]).astype(np.float32) for bundle in bundles]
    seed_everything(int(args.seed) + (101 if kind == "gru" else 211))
    model = RecurrentH1Baseline(kind, int(args.baseline_hidden)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    dataset = TensorDataset(torch.from_numpy(tensors[0]), torch.from_numpy(targets[0]))
    loader = DataLoader(dataset, batch_size=int(args.baseline_batch_size), shuffle=True)
    best_state = None
    best_score = float("inf")
    validation_weights = parse_horizon_weights(args.validation_horizon_weights)
    patience = 0
    best_epoch = 0
    for epoch in range(1, int(args.baseline_epochs) + 1):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch.to(device))
            loss = F.smooth_l1_loss(pred, y_batch.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_norm = model(torch.from_numpy(tensors[1]).to(device)).cpu().numpy()
        val_prediction = target_scaler.inverse_transform(val_norm)
        score = weighted_rolling_score(val, val_prediction, validation_weights)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is None:
        raise RuntimeError(f"No finite {kind} baseline")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        test_norm = model(torch.from_numpy(tensors[2]).to(device)).cpu().numpy()
    return target_scaler.inverse_transform(test_norm).astype(np.float32), {
        "kind": kind,
        "best_epoch": best_epoch,
        "val_weighted_rolling_rmse": best_score,
        "validation_horizon_weights": dict(validation_weights),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def transition_matrix(kind: str, turn_rate: float = 0.0) -> np.ndarray:
    if kind == "cv":
        return np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
    if kind == "ca":
        return np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    if kind in {"turn_left", "turn_right"}:
        angle = float(turn_rate) * (1.0 if kind == "turn_left" else -1.0)
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
    raise ValueError(kind)


H_KALMAN = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


def gaussian_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
    covariance = covariance + np.eye(2) * 1e-6
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return 1e-30
    exponent = -0.5 * float(residual.T @ np.linalg.solve(covariance, residual))
    return max(float(math.exp(exponent - 0.5 * (2 * math.log(2 * math.pi) + logdet))), 1e-30)


def kalman_update(x: np.ndarray, p: np.ndarray, z: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, float]:
    observation_cov = np.eye(2) * float(r)
    residual = z - H_KALMAN @ x
    innovation_cov = H_KALMAN @ p @ H_KALMAN.T + observation_cov
    gain = p @ H_KALMAN.T @ np.linalg.inv(innovation_cov)
    posterior = x + gain @ residual
    identity = np.eye(4)
    covariance = (identity - gain @ H_KALMAN) @ p @ (identity - gain @ H_KALMAN).T + gain @ observation_cov @ gain.T
    return posterior, covariance, gaussian_likelihood(residual, innovation_cov)


def kalman_predictions(bundle: v84.AnchorBundle, kind: str, q: float, r: float) -> np.ndarray:
    prediction = np.zeros((len(bundle.rows), 2), dtype=np.float32)
    cache: dict[tuple[int, int], tuple[int, np.ndarray, np.ndarray]] = {}
    for index, row in bundle.rows.sort_values(KEYS).iterrows():
        sequence, frame, track = int(row.sequence), int(row.frame), int(row.track_id)
        key = (sequence, track)
        z = np.asarray([row.dx_px, row.dy_px], dtype=np.float64)
        cached = cache.get(key)
        matrix = transition_matrix(kind)
        if cached is None or cached[0] != frame - 1:
            state = np.asarray([z[0], z[1], 0.0, 0.0], dtype=np.float64)
            covariance = np.eye(4) * max(float(r), 1.0)
        else:
            prior_state = matrix @ cached[1]
            prior_covariance = matrix @ cached[2] @ matrix.T + np.eye(4) * float(q)
            state, covariance, _likelihood = kalman_update(prior_state, prior_covariance, z, r)
        prediction[int(index)] = (matrix @ state)[:2]
        cache[key] = (frame, state, covariance)
    return prediction


def imm_predictions(bundle: v84.AnchorBundle, q: float, r: float, turn_rate: float) -> np.ndarray:
    kinds = ["cv", "ca", "turn_left", "turn_right"]
    matrices = [transition_matrix(kind, turn_rate) for kind in kinds]
    modes = len(kinds)
    transition = np.full((modes, modes), 0.04 / (modes - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.96)
    prediction = np.zeros((len(bundle.rows), 2), dtype=np.float32)
    cache: dict[tuple[int, int], tuple[int, list[np.ndarray], list[np.ndarray], np.ndarray]] = {}

    for index, row in bundle.rows.sort_values(KEYS).iterrows():
        sequence, frame, track = int(row.sequence), int(row.frame), int(row.track_id)
        key = (sequence, track)
        z = np.asarray([row.dx_px, row.dy_px], dtype=np.float64)
        cached = cache.get(key)
        if cached is None or cached[0] != frame - 1:
            states = [np.asarray([z[0], z[1], 0.0, 0.0], dtype=np.float64) for _ in kinds]
            covariances = [np.eye(4) * max(float(r), 1.0) for _ in kinds]
            probabilities = np.full(modes, 1.0 / modes)
        else:
            old_states, old_covariances, old_probabilities = cached[1], cached[2], cached[3]
            normalizers = old_probabilities @ transition
            states = []
            covariances = []
            likelihoods = np.zeros(modes, dtype=np.float64)
            for destination in range(modes):
                mixing = old_probabilities * transition[:, destination] / max(normalizers[destination], EPS)
                mixed_state = sum(float(mixing[source]) * old_states[source] for source in range(modes))
                mixed_covariance = np.zeros((4, 4), dtype=np.float64)
                for source in range(modes):
                    difference = old_states[source] - mixed_state
                    mixed_covariance += float(mixing[source]) * (
                        old_covariances[source] + np.outer(difference, difference)
                    )
                matrix = matrices[destination]
                prior_state = matrix @ mixed_state
                prior_covariance = matrix @ mixed_covariance @ matrix.T + np.eye(4) * float(q)
                posterior, covariance, likelihood = kalman_update(prior_state, prior_covariance, z, r)
                states.append(posterior)
                covariances.append(covariance)
                likelihoods[destination] = likelihood
            probabilities = normalizers * likelihoods
            total_probability = float(probabilities.sum())
            if np.isfinite(total_probability) and total_probability > 0.0:
                probabilities /= total_probability
            else:
                normalizer_total = float(normalizers.sum())
                probabilities = (
                    normalizers / normalizer_total
                    if np.isfinite(normalizer_total) and normalizer_total > 0.0
                    else np.full(modes, 1.0 / modes, dtype=np.float64)
                )
        mode_predictions = np.stack([(matrix @ state)[:2] for matrix, state in zip(matrices, states)])
        prediction[int(index)] = np.sum(probabilities[:, None] * mode_predictions, axis=0)
        cache[key] = (frame, states, covariances, probabilities)
    return prediction


def tune_classical_filters(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    _train, val, test = bundles
    predictions: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    validation_weights = parse_horizon_weights(args.validation_horizon_weights)
    q_grid = parse_floats(args.kalman_q_grid)
    r_grid = parse_floats(args.kalman_r_grid)
    for kind in ("cv", "ca"):
        best: tuple[float, float, float] | None = None
        for q in q_grid:
            for r in r_grid:
                val_prediction = kalman_predictions(val, kind, q, r)
                score = weighted_rolling_score(val, val_prediction, validation_weights)
                if best is None or score < best[0]:
                    best = (score, q, r)
        assert best is not None
        predictions[f"kalman_{kind}"] = kalman_predictions(test, kind, best[1], best[2])
        diagnostics.append(
            {
                "method": f"kalman_{kind}",
                "val_score": best[0],
                "q": best[1],
                "r": best[2],
                "validation_horizon_weights": json.dumps(validation_weights, sort_keys=True),
            }
        )

    best_imm: tuple[float, float, float, float] | None = None
    for q in q_grid:
        for r in r_grid:
            for turn in parse_floats(args.imm_turn_grid):
                val_prediction = imm_predictions(val, q, r, turn)
                score = weighted_rolling_score(val, val_prediction, validation_weights)
                if best_imm is None or score < best_imm[0]:
                    best_imm = (score, q, r, turn)
    assert best_imm is not None
    predictions["imm_cv_ca_turn"] = imm_predictions(test, best_imm[1], best_imm[2], best_imm[3])
    diagnostics.append(
        {
            "method": "imm_cv_ca_turn",
            "val_score": best_imm[0],
            "q": best_imm[1],
            "r": best_imm[2],
            "turn_rate": best_imm[3],
            "validation_horizon_weights": json.dumps(validation_weights, sort_keys=True),
        }
    )
    return predictions, diagnostics


def external_prediction(path: Path | None, keys: list[str], expected_rows: int) -> tuple[np.ndarray | None, str | None]:
    if path is None or not path.exists():
        return None, None
    archive = np.load(path)
    for key in keys:
        if key in archive and archive[key].shape[0] == expected_rows:
            values = archive[key]
            return (values[:, 0] if values.ndim == 3 else values).astype(np.float32), key
    return None, None


def cluster_bootstrap_delta(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    baseline: np.ndarray,
    horizon: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    target, pred, _starts, clusters = rolling_examples(bundle, prediction, horizon)
    target_base, base, _starts_base, clusters_base = rolling_examples(bundle, baseline, horizon)
    if not np.array_equal(clusters, clusters_base) or not np.allclose(target, target_base):
        raise RuntimeError("Bootstrap contracts do not match")
    unique = np.unique(clusters)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    deltas = np.zeros(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([cluster_indices[value] for value in sampled])
        model_rmse = float(np.sqrt(np.mean(np.square(target[indices] - pred[indices]))))
        base_rmse = float(np.sqrt(np.mean(np.square(target[indices] - base[indices]))))
        deltas[repeat] = base_rmse - model_rmse
    return {
        "horizon": horizon,
        "delta_rmse": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_positive": float(np.mean(deltas > 0)),
        "bootstrap_repeats": repeats,
        "clusters": len(unique),
    }


def causal_audit(prep: v96.Prepared) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split, bundle in zip(("train", "val", "test"), prep.bundles):
        previous = previous_row_lookup(bundle)
        valid = previous >= 0
        current_frames = bundle.rows["frame"].to_numpy(np.int64)
        source_frames = np.full(len(bundle.rows), -1, dtype=np.int64)
        source_frames[valid] = bundle.rows.iloc[previous[valid]]["frame"].to_numpy(np.int64)
        violation = valid & (source_frames != current_frames - 1)
        records.append(
            {
                "split": split,
                "rows": len(bundle.rows),
                "movies": ",".join(map(str, sorted(bundle.rows.sequence.unique()))),
                "measurement_rows": int(valid.sum()),
                "measurement_coverage": float(valid.mean()),
                "causal_violations": int(violation.sum()),
                "prediction_time": "frame_t",
                "latest_measurement": "completed_transition_t-1_to_t",
                "target_time": "frame_t+1",
                "scalers": "train_only",
                "train_anchor": "movie_held_out_oof" if split == "train" else "held_out_movie",
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    horizons = parse_ints(args.horizons)
    variants = train_variants(args)
    if args.smoke:
        args.epochs = min(int(args.epochs), 3)
        args.patience = min(int(args.patience), 2)
        args.baseline_epochs = min(int(args.baseline_epochs), 3)
        args.variants = ",".join([variant.name for variant in variants[:2]])
        variants = variants[:2]
        args.kalman_q_grid = "0.5,4"
        args.kalman_r_grid = "0.5,4"
        args.imm_turn_grid = "0.15"

    all_metrics: list[dict[str, Any]] = []
    robustness_metrics: list[dict[str, Any]] = []
    uncertainty_metrics: list[dict[str, Any]] = []
    conformal_uncertainty_metrics: list[dict[str, Any]] = []
    noise_diagnostics: list[dict[str, Any]] = []
    train_logs: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []
    prediction_archive: dict[str, np.ndarray] = {}
    trained: dict[str, tuple[CausalInnovationStateSpaceForecaster, v96.Prepared, float, dict[str, Any]]] = {}

    for variant in variants:
        print(f"[v97] preparing {variant.name}", flush=True)
        prep = load_prepared(args, variant)
        audit = causal_audit(prep)
        if int(audit.causal_violations.sum()) != 0:
            raise RuntimeError("Causal audit failed")
        print(f"[v97] training {variant.name}", flush=True)
        model, eta, metadata, logs = train_model(prep, variant, args)
        device = device_from_args(args)
        if args.checkpoint_only:
            train_logs.extend(logs)
            model_metadata.append({"variant": variant.name, **metadata})
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": metadata,
                    "variant": variant.__dict__,
                    "args": finite(vars(args)),
                },
                args.out_dir / f"{variant.name}.pt",
            )
            continue
        result = replay_inference(model, prep, 2, device, eta=eta, seed=int(args.seed))
        val_result = replay_inference(model, prep, 1, device, eta=eta, seed=int(args.seed))
        scale_factor = calibrate_uncertainty(
            prep.bundles[1],
            val_result.prediction,
            val_result.scale,
            metadata["degrees_of_freedom"],
            parse_floats(args.uncertainty_scale_grid),
        )
        all_metrics.extend(
            rolling_metric_rows(
                prep.bundles[2],
                result.prediction,
                horizons,
                variant.name,
                {"family": "v97", "control": "real", "eta": eta},
            )
        )
        uncertainty_metrics.extend(
            uncertainty_rows(
                prep.bundles[2],
                result.prediction,
                result.scale,
                horizons,
                variant.name,
                metadata["degrees_of_freedom"],
                scale_factor,
            )
        )
        conformal_uncertainty_metrics.extend(
            conformal_uncertainty_rows(
                prep.bundles[1],
                val_result.prediction,
                val_result.scale,
                prep.bundles[2],
                result.prediction,
                result.scale,
                horizons,
                variant.name,
            )
        )
        noise_diagnostics.append(
            {
                "variant": variant.name,
                "process_scale_mean": float(result.process_scale.mean()),
                "observation_scale_mean": float(result.observation_scale.mean()),
                "gain_mean": float(result.gain.mean()),
                "gain_p10": float(np.quantile(result.gain, 0.10)),
                "gain_p90": float(np.quantile(result.gain, 0.90)),
                "measurement_coverage": float(result.measurement_mask.mean()),
                "degrees_of_freedom": metadata["degrees_of_freedom"],
                "uncertainty_scale_factor": scale_factor,
            }
        )
        train_logs.extend(logs)
        model_metadata.append({"variant": variant.name, **metadata})
        prediction_archive[f"{variant.name}__prediction"] = result.prediction
        prediction_archive[f"{variant.name}__scale"] = result.scale
        trained[variant.name] = (model, prep, eta, metadata)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": metadata,
                "variant": variant.__dict__,
                "args": finite(vars(args)),
            },
            args.out_dir / f"{variant.name}.pt",
        )

    if args.checkpoint_only:
        pd.DataFrame(train_logs).to_csv(
            args.out_dir / "v97_train_log.csv",
            index=False,
        )
        pd.DataFrame(model_metadata).to_csv(
            args.out_dir / "v97_model_metadata.csv",
            index=False,
        )
        (args.out_dir / "run_config.json").write_text(
            json.dumps(finite(vars(args)), indent=2),
            encoding="utf-8",
        )
        print(
            "[v97] checkpoint-only: skipped test robustness, baselines, and bootstrap",
            flush=True,
        )
        return

    if args.evaluation_variant == "auto":
        eligible = [
            name
            for name, (_model, _prep, _eta, metadata) in trained.items()
            if bool(next(variant.use_update for variant in variants if variant.name == name))
        ]
        evaluation_variant = min(
            eligible,
            key=lambda name: float(trained[name][3]["best_val_score_tuned"]),
        )
    else:
        evaluation_variant = str(args.evaluation_variant)
        if evaluation_variant not in trained:
            raise ValueError(
                f"evaluation variant {evaluation_variant!r} was not trained; available={sorted(trained)}"
            )
    core_model, core_prep, core_eta, core_metadata = trained[evaluation_variant]
    device = device_from_args(args)
    test_bundle = core_prep.bundles[2]

    for stride in parse_ints(args.update_strides):
        result = replay_inference(
            core_model, core_prep, 2, device, eta=core_eta, update_stride=stride, seed=int(args.seed)
        )
        rows = rolling_metric_rows(
            test_bundle,
            result.prediction,
            horizons,
            f"{evaluation_variant}_update_stride{stride}",
            {"family": "cadence", "control": "real", "update_stride": stride, "missing_rate": 0.0, "coordinate_noise_px": 0.0, "evaluation_variant": evaluation_variant},
        )
        robustness_metrics.extend(rows)
        prediction_archive[f"{evaluation_variant}__stride{stride}"] = result.prediction

    for missing_rate in parse_floats(args.missing_rates):
        if missing_rate <= 0:
            continue
        result = replay_inference(
            core_model,
            core_prep,
            2,
            device,
            eta=core_eta,
            missing_rate=missing_rate,
            seed=int(args.seed),
        )
        robustness_metrics.extend(
            rolling_metric_rows(
                test_bundle,
                result.prediction,
                horizons,
                f"{evaluation_variant}_missing_{missing_rate:g}",
                {"family": "missingness", "control": "random_missing", "update_stride": 1, "missing_rate": missing_rate, "coordinate_noise_px": 0.0, "evaluation_variant": evaluation_variant},
            )
        )

    for noise in parse_floats(args.coordinate_noise_grid):
        if noise <= 0:
            continue
        result = replay_inference(
            core_model,
            core_prep,
            2,
            device,
            eta=core_eta,
            coordinate_noise_px=noise,
            seed=int(args.seed),
        )
        robustness_metrics.extend(
            rolling_metric_rows(
                test_bundle,
                result.prediction,
                horizons,
                f"{evaluation_variant}_coord_noise_{noise:g}px",
                {"family": "tracking_noise", "control": "innovation_coordinate_noise", "update_stride": 1, "missing_rate": 0.0, "coordinate_noise_px": noise, "evaluation_variant": evaluation_variant},
            )
        )

    for control in ("wrong_cell", "time_shuffled", "delayed", "no_update"):
        result = replay_inference(
            core_model, core_prep, 2, device, eta=core_eta, control=control, seed=int(args.seed)
        )
        robustness_metrics.extend(
            rolling_metric_rows(
                test_bundle,
                result.prediction,
                horizons,
                f"{evaluation_variant}_{control}",
                {"family": "causal_control", "control": control, "update_stride": 1, "missing_rate": 0.0, "coordinate_noise_px": 0.0, "evaluation_variant": evaluation_variant},
            )
        )

    constant_velocity = test_bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    baseline_predictions: dict[str, np.ndarray] = {"constant_velocity": constant_velocity}
    anchor_method = str(test_bundle.meta.get("anchor_method", "v52_rolling"))
    anchor_prediction = test_bundle.anchor_steps[:, 0].astype(np.float32)
    if anchor_method != "constant_velocity" or not np.allclose(anchor_prediction, constant_velocity):
        baseline_predictions[anchor_method] = anchor_prediction
    classical_predictions, classical_diagnostics = tune_classical_filters(core_prep.bundles, args)
    baseline_predictions.update(classical_predictions)
    recurrent_diagnostics: list[dict[str, Any]] = []
    if not args.skip_recurrent_baselines:
        for kind in ("lstm", "gru"):
            prediction, diagnostics = fit_recurrent_baseline(kind, core_prep.bundles, args, device)
            baseline_predictions[f"{kind}_h1"] = prediction
            recurrent_diagnostics.append(diagnostics)

    v88_prediction, v88_key = external_prediction(
        args.v88_predictions,
        ["graph_all_models_mean", "graph_seed_convex_validation", "anchor_seed_mean"],
        len(test_bundle.rows),
    )
    if v88_prediction is not None:
        baseline_predictions["v88_rolling"] = v88_prediction
    v96_prediction, v96_key = external_prediction(
        args.v96_predictions,
        ["ogif_core_streaming", "ogif_core__test_streaming_prediction", "ogif_core__test_prediction"],
        len(test_bundle.rows),
    )
    if v96_prediction is not None:
        baseline_predictions["v96_rolling"] = v96_prediction

    for method, prediction in baseline_predictions.items():
        all_metrics.extend(
            rolling_metric_rows(
                test_bundle,
                prediction,
                horizons,
                method,
                {"family": "baseline", "control": "real", "eta": np.nan},
            )
        )
        prediction_archive[f"baseline__{method}"] = prediction

    bootstrap_rows: list[dict[str, Any]] = []
    core_prediction = prediction_archive[f"{evaluation_variant}__prediction"]
    for baseline_name in ("v96_rolling", "imm_cv_ca_turn", "v88_rolling", "v52_rolling"):
        baseline = baseline_predictions.get(baseline_name)
        if baseline is None:
            continue
        row = cluster_bootstrap_delta(
            test_bundle,
            core_prediction,
            baseline,
            max(horizons),
            int(args.bootstrap_repeats),
            int(args.seed) + 880,
        )
        row.update(model=evaluation_variant, baseline=baseline_name)
        bootstrap_rows.append(row)

    causal = causal_audit(core_prep)
    data_contract = pd.DataFrame(
        [
            {
                "anchor_cache": str(args.anchor_cache),
                "features": str(args.features),
                "train_movies": ",".join(map(str, sorted(core_prep.bundles[0].rows.sequence.unique()))),
                "val_movies": ",".join(map(str, sorted(core_prep.bundles[1].rows.sequence.unique()))),
                "test_movies": ",".join(map(str, sorted(core_prep.bundles[2].rows.sequence.unique()))),
                "train_rows": len(core_prep.bundles[0].rows),
                "val_rows": len(core_prep.bundles[1].rows),
                "test_rows": len(core_prep.bundles[2].rows),
                "test_tracks": core_prep.bundles[2].rows.track_id.nunique(),
                "context_features": len(core_prep.context_names),
                "v88_key": v88_key,
                "v96_key": v96_key,
                "evaluation_variant": evaluation_variant,
                "anchor_method": anchor_method,
                "component_rmse_definition": "sqrt(mean((dx_error,dy_error)^2))",
                "contract": "one-step predictions issued before next observation; cumulative rolling evaluation",
            }
        ]
    )

    metrics_df = pd.DataFrame(all_metrics)
    robustness_df = pd.DataFrame(robustness_metrics)
    uncertainty_df = pd.DataFrame(uncertainty_metrics)
    conformal_uncertainty_df = pd.DataFrame(conformal_uncertainty_metrics)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    metrics_df.to_csv(args.out_dir / "v97_online_summary.csv", index=False)
    robustness_df.to_csv(args.out_dir / "v97_robustness.csv", index=False)
    robustness_df[robustness_df.family.eq("cadence")].to_csv(args.out_dir / "v97_update_cadence.csv", index=False)
    uncertainty_df.to_csv(args.out_dir / "v97_uncertainty.csv", index=False)
    conformal_uncertainty_df.to_csv(
        args.out_dir / "v97_conformal_uncertainty.csv", index=False
    )
    pd.DataFrame(noise_diagnostics).to_csv(args.out_dir / "v97_noise_decomposition.csv", index=False)
    pd.DataFrame(train_logs).to_csv(args.out_dir / "v97_train_log.csv", index=False)
    pd.DataFrame(model_metadata).to_csv(args.out_dir / "v97_model_metadata.csv", index=False)
    pd.DataFrame(classical_diagnostics + recurrent_diagnostics).to_csv(args.out_dir / "v97_baseline_diagnostics.csv", index=False)
    bootstrap_df.to_csv(args.out_dir / "v97_bootstrap.csv", index=False)
    causal.to_csv(args.out_dir / "v97_causal_audit.csv", index=False)
    data_contract.to_csv(args.out_dir / "v97_data_contract.csv", index=False)
    np.savez_compressed(args.out_dir / "v97_predictions.npz", **prediction_archive)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite(vars(args)), indent=2), encoding="utf-8")

    hmax = max(horizons)
    h_table = metrics_df[metrics_df.horizon.eq(hmax)].sort_values("component_rmse")
    cadence_h = robustness_df[(robustness_df.horizon.eq(hmax)) & robustness_df.family.eq("cadence")].sort_values("update_stride")
    controls_h = robustness_df[(robustness_df.horizon.eq(hmax)) & robustness_df.family.eq("causal_control")]
    core_h = h_table[h_table.method.eq(evaluation_variant)]
    core_rmse = float(core_h.iloc[0].component_rmse) if not core_h.empty else float("nan")
    v96_h = h_table[h_table.method.eq("v96_rolling")]
    imm_h = h_table[h_table.method.eq("imm_cv_ca_turn")]
    coordinate_unit = str(
        test_bundle.meta.get("contract", {}).get("coordinate_unit", "pixel")
    )
    target_pass = core_rmse < 9.5 if coordinate_unit == "pixel" else None
    statistical_pass = False
    if not bootstrap_df.empty:
        relevant = bootstrap_df[bootstrap_df.baseline.isin(["v96_rolling", "imm_cv_ca_turn"])]
        statistical_pass = bool(len(relevant) and np.all(relevant.ci_low > 0))
    lines = [
        "# v97 Causal Innovation State-Space Forecaster",
        "",
        "## Streaming cumulative h6",
        "",
        h_table[["method", "component_rmse", "r2", "cosine", "magnitude_ratio", "family", "control"]].to_markdown(index=False),
        "",
        "## Observation cadence",
        "",
        cadence_h[["method", "update_stride", "component_rmse", "r2", "n_rows"]].to_markdown(index=False),
        "",
        "## Causal controls",
        "",
        controls_h[["method", "control", "component_rmse", "r2"]].sort_values("component_rmse").to_markdown(index=False),
        "",
        "## Decision",
        "",
        f"- Validation-selected evaluation variant: `{evaluation_variant}`.",
        f"- Selected v97 cumulative h6 RMSE: `{core_rmse:.6f}`.",
        (
            f"- Numeric target `<9.5 px`: `{target_pass}`."
            if target_pass is not None
            else (
                "- Numeric target `<9.5 px`: not applicable to "
                f"`{coordinate_unit}` coordinates."
            )
        ),
        f"- Statistically better than available v96/IMM baselines: `{statistical_pass}`.",
        f"- Causal audit violations: `{int(causal.causal_violations.sum())}`.",
    ]
    if not v96_h.empty:
        lines.append(f"- Same-contract v96 reference: `{float(v96_h.iloc[0].component_rmse):.6f}`.")
    if not imm_h.empty:
        lines.append(f"- Same-contract IMM reference: `{float(imm_h.iloc[0].component_rmse):.6f}`.")
    lines.extend(
        [
            "- The rolling metric is not a six-step single-shot forecast: a new observation is assimilated before each next prediction.",
            "- Process/observation scales are separately parameterized but are not claimed as physically identifiable without independent tracking-noise labels.",
            f"- Elapsed: `{(time.time() - started) / 3600.0:.2f} h`.",
        ]
    )
    (args.out_dir / "v97_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "v97_decision_report.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-cache", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--v88-predictions", type=Path)
    parser.add_argument("--v96-predictions", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", default="v97_core,v97_no_context,v97_no_update,v97_graph")
    parser.add_argument(
        "--evaluation-variant",
        default="auto",
        help="Variant used for cadence/robustness/bootstrap; auto selects an update-enabled variant by validation score.",
    )
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--cumulative-horizons", default="2,4,6")
    parser.add_argument(
        "--validation-horizon-weights",
        default="1:0.45,2:0.25,4:0.18,6:0.12",
        help="Validation-only checkpoint/eta score. Use 1:0.65,2:0.2,4:0.1,6:0.05 for the h1-strict audit.",
    )
    parser.add_argument("--update-strides", default="1,2,3,6,1000000")
    parser.add_argument("--missing-rates", default="0.1,0.2,0.4")
    parser.add_argument("--coordinate-noise-grid", default="0.25,0.5,1.0")
    parser.add_argument("--context-quotas", default="ms_:16,tf_:48,rc_:16,obs_:48")
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--history-lags", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--tbptt-frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.06)
    parser.add_argument("--correction-bound", type=float, default=3.5)
    parser.add_argument("--nll-weight", type=float, default=0.12)
    parser.add_argument("--cumulative-weight", type=float, default=0.30)
    parser.add_argument("--innovation-nll-weight", type=float, default=0.06)
    parser.add_argument("--noise-regularization-weight", type=float, default=0.002)
    parser.add_argument("--gain-weight", type=float, default=0.001)
    parser.add_argument("--process-scale-prior", type=float, default=0.75)
    parser.add_argument("--observation-scale-prior", type=float, default=0.35)
    parser.add_argument("--train-missing-rate", type=float, default=0.15)
    parser.add_argument("--train-coordinate-noise-px", type=float, default=0.35)
    parser.add_argument(
        "--one-step-scaler",
        action="store_true",
        help="Fit innovation normalization from causal h1 residuals only.",
    )
    parser.add_argument("--reservoir-size", type=int, default=4096)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--eta-grid", default="0,0.1,0.2,0.35,0.5,0.75,1,1.25")
    parser.add_argument("--uncertainty-scale-grid", default="0.5,0.75,1,1.25,1.5,2,3")
    parser.add_argument("--kalman-q-grid", default="0.1,0.5,1,4,16")
    parser.add_argument("--kalman-r-grid", default="0.1,0.5,1,4,16")
    parser.add_argument("--imm-turn-grid", default="0.08,0.15,0.25,0.4")
    parser.add_argument("--baseline-history-lags", type=int, default=8)
    parser.add_argument("--baseline-hidden", type=int, default=96)
    parser.add_argument("--baseline-epochs", type=int, default=24)
    parser.add_argument("--baseline-batch-size", type=int, default=1024)
    parser.add_argument("--skip-recurrent-baselines", action="store_true")
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help=(
            "Stop after validation-selected model checkpoints are written. "
            "Used when a downstream runner regenerates all replay metrics."
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=3000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
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
        (args.out_dir / "v97_error.json").write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
