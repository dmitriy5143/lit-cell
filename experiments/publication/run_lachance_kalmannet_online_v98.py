#!/usr/bin/env python3
"""v98: KalmanNet comparator for the causal online h1 benchmark.

This runner adapts the official KalmanNet architecture #2 to cell-motion
forecasting while preserving the v97 streaming contract.  At frame ``t`` the
filter may assimilate only the observation available at ``t`` and then issues
the displacement prediction for ``t -> t+1``.  Publication mode defines this
event as ``H F x_post_t - observed_x_t``; the historical
``H(F-I)x_post_t`` state increment remains an explicit legacy ablation.  The
future displacement is used only by the training loss and metrics.

The implementation keeps known CV/CA transition and position-observation
models.  Only the Kalman gain is learned from the four normalized sequences
used by official KalmanNet: observation difference, innovation, state
evolution difference, and state update difference.  The Q, Sigma, and S GRU
flow follows:
https://github.com/KalmanNet/KalmanNet_TSP/blob/main/KNet/KalmanNet_nn.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_causal_innovation_state_space_v97 as v97  # noqa: E402
import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


EPS = 1e-8
KEYS = ["sequence", "frame", "track_id"]
DEFAULT_OUT = ROOT / "outputs" / "kalmannet_online_v98_2026-07-21"
OFFICIAL_ROOT = ROOT / "external" / "state_estimation_sota" / "KalmanNet_TSP"
PREDICTIVE_EVENTS = ("next_observation_prior", "state_increment")
PUBLICATION_PREDICTIVE_EVENT = "next_observation_prior"
LEGACY_PREDICTIVE_EVENT = "state_increment"
PUBLICATION_VALIDATION_WEIGHTS = "1:0.90,2:0.05,4:0.03,6:0.02"


@dataclass(frozen=True)
class FrameBatch:
    sequence: int
    frame: int
    indices: np.ndarray
    track_ids: np.ndarray


@dataclass
class KNetEntry:
    frame: int
    posterior: torch.Tensor
    previous_posterior: torch.Tensor
    prior: torch.Tensor
    observation: torch.Tensor
    h_q: torch.Tensor
    h_sigma: torch.Tensor
    h_s: torch.Tensor
    last_obs_diff: torch.Tensor
    last_innovation: torch.Tensor


@dataclass
class KNetReplayResult:
    prediction: np.ndarray
    gain_norm: np.ndarray
    gain_trace: np.ndarray
    measurement_mask: np.ndarray


def safe(value: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if not isinstance(value, str):
        return [str(item) for item in value]
    return [item.strip() for item in value.split(",") if item.strip()]


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def bundle_contract_hashes(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split, bundle in zip(("train", "val", "test"), bundles):
        hashes[f"{split}_key_sha256"] = array_sha256(
            bundle.rows[KEYS].to_numpy(np.int64, copy=True)
        )
        hashes[f"{split}_h1_target_sha256"] = array_sha256(bundle.target_steps[:, 0])
    return hashes


def predictive_mean(
    posterior: torch.Tensor,
    next_prior: torch.Tensor,
    observed_position: torch.Tensor,
    observation_matrix: torch.Tensor,
    predictive_event: str,
) -> torch.Tensor:
    if predictive_event == PUBLICATION_PREDICTIVE_EVENT:
        # x_t is already observed.  H F x_post_t is the prior mean for x_{t+1};
        # subtracting the observed x_t expresses that prior as the benchmark
        # displacement without using any information from t+1.
        return next_prior @ observation_matrix.T - observed_position
    if predictive_event == LEGACY_PREDICTIVE_EVENT:
        return next_prior @ observation_matrix.T - posterior @ observation_matrix.T
    raise ValueError(f"Unknown predictive event: {predictive_event}")


def state_dimensions(kind: str) -> tuple[int, int]:
    if kind == "cv":
        return 4, 2
    if kind == "ca":
        return 6, 2
    raise ValueError(f"Unknown state model: {kind}")


def transition_observation(
    kind: str,
    *,
    dt: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return standard position-space CV/CA F and H matrices."""
    if kind == "cv":
        matrix = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        observation = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    elif kind == "ca":
        half_dt2 = 0.5 * dt * dt
        matrix = np.asarray(
            [
                [1.0, 0.0, dt, 0.0, half_dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, half_dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        observation = np.asarray(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown state model: {kind}")
    return (
        torch.as_tensor(matrix, dtype=dtype, device=device),
        torch.as_tensor(observation, dtype=dtype, device=device),
    )


def initial_state(kind: str, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
    if kind == "cv":
        return torch.cat([positions, velocities], dim=-1)
    acceleration = torch.zeros_like(velocities)
    return torch.cat([positions, velocities, acceleration], dim=-1)


class OfficialKalmanGainNetwork(nn.Module):
    """Official KalmanNet architecture #2 expressed with GRUCell steps.

    The dimensions and forward/backward covariance flow mirror
    ``KNet/KalmanNet_nn.py``.  GRUCell is used because different tracks enter
    and leave each microscopy frame, so every cell carries its own recurrent
    covariance state.
    """

    def __init__(self, state_dim: int, observation_dim: int, in_mult: int = 5, out_mult: int = 40):
        super().__init__()
        self.m = int(state_dim)
        self.n = int(observation_dim)
        self.in_mult = int(in_mult)
        self.out_mult = int(out_mult)

        self.fc5 = nn.Sequential(nn.Linear(self.m, self.m * self.in_mult), nn.ReLU())
        self.gru_q = nn.GRUCell(self.m * self.in_mult, self.m * self.m)

        self.fc6 = nn.Sequential(nn.Linear(self.m, self.m * self.in_mult), nn.ReLU())
        self.gru_sigma = nn.GRUCell(self.m * self.m + self.m * self.in_mult, self.m * self.m)

        self.fc1 = nn.Sequential(nn.Linear(self.m * self.m, self.n * self.n), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(2 * self.n, 2 * self.n * self.in_mult), nn.ReLU())
        self.gru_s = nn.GRUCell(self.n * self.n + 2 * self.n * self.in_mult, self.n * self.n)

        fc2_in = self.m * self.m + self.n * self.n
        self.fc2 = nn.Sequential(
            nn.Linear(fc2_in, fc2_in * self.out_mult),
            nn.ReLU(),
            nn.Linear(fc2_in * self.out_mult, self.m * self.n),
        )
        self.fc3 = nn.Sequential(nn.Linear(self.n * self.n + self.m * self.n, self.m * self.m), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(2 * self.m * self.m, self.m * self.m), nn.ReLU())

    def initial_hidden(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.eye(self.m, dtype=dtype, device=device).reshape(1, -1).repeat(batch, 1) * 0.01
        sigma = torch.eye(self.m, dtype=dtype, device=device).reshape(1, -1).repeat(batch, 1)
        s = torch.eye(self.n, dtype=dtype, device=device).reshape(1, -1).repeat(batch, 1)
        return q, sigma, s

    def forward(
        self,
        obs_diff: torch.Tensor,
        innovation: torch.Tensor,
        evolution_diff: torch.Tensor,
        update_diff: torch.Tensor,
        h_q: torch.Tensor,
        h_sigma: torch.Tensor,
        h_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_diff = F.normalize(obs_diff, p=2, dim=-1, eps=1e-12)
        innovation = F.normalize(innovation, p=2, dim=-1, eps=1e-12)
        evolution_diff = F.normalize(evolution_diff, p=2, dim=-1, eps=1e-12)
        update_diff = F.normalize(update_diff, p=2, dim=-1, eps=1e-12)

        q_out = self.gru_q(self.fc5(update_diff), h_q)
        sigma_out = self.gru_sigma(torch.cat([q_out, self.fc6(evolution_diff)], dim=-1), h_sigma)
        s_out = self.gru_s(
            torch.cat([self.fc1(sigma_out), self.fc7(torch.cat([obs_diff, innovation], dim=-1))], dim=-1),
            h_s,
        )
        gain_flat = self.fc2(torch.cat([sigma_out, s_out], dim=-1))

        sigma_feedback = self.fc4(torch.cat([sigma_out, self.fc3(torch.cat([s_out, gain_flat], dim=-1))], dim=-1))
        gain = gain_flat.reshape(-1, self.m, self.n)
        return gain, q_out, sigma_feedback, s_out


class KalmanNetOnlineForecaster(nn.Module):
    def __init__(self, kind: str, in_mult: int, out_mult: int):
        super().__init__()
        self.kind = str(kind)
        self.state_dim, self.observation_dim = state_dimensions(self.kind)
        self.gain_network = OfficialKalmanGainNetwork(
            self.state_dim, self.observation_dim, in_mult=in_mult, out_mult=out_mult
        )

    def matrices(
        self, *, dt: float, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return transition_observation(self.kind, dt=dt, dtype=dtype, device=device)


def grouped_frames(bundle: v84.AnchorBundle) -> dict[int, list[FrameBatch]]:
    groups: dict[int, list[FrameBatch]] = {}
    rows = bundle.rows
    for (sequence, frame), indices in rows.groupby(["sequence", "frame"], sort=True).groups.items():
        index_array = np.asarray(list(indices), dtype=np.int64)
        batch = FrameBatch(
            int(sequence),
            int(frame),
            index_array,
            rows.loc[index_array, "track_id"].to_numpy(np.int64),
        )
        groups.setdefault(int(sequence), []).append(batch)
    for sequence in groups:
        groups[sequence].sort(key=lambda item: item.frame)
    return groups


def detach_entry(entry: KNetEntry) -> KNetEntry:
    return KNetEntry(
        frame=entry.frame,
        posterior=entry.posterior.detach(),
        previous_posterior=entry.previous_posterior.detach(),
        prior=entry.prior.detach(),
        observation=entry.observation.detach(),
        h_q=entry.h_q.detach(),
        h_sigma=entry.h_sigma.detach(),
        h_s=entry.h_s.detach(),
        last_obs_diff=entry.last_obs_diff.detach(),
        last_innovation=entry.last_innovation.detach(),
    )


def detach_windows(windows: dict[int, v97.WindowEntry]) -> dict[int, v97.WindowEntry]:
    return {
        key: v97.WindowEntry(
            frame=value.frame,
            predictions=[item.detach() for item in value.predictions],
            targets=[item.detach() for item in value.targets],
        )
        for key, value in windows.items()
    }


def policy_features(
    obs_diff: torch.Tensor,
    innovation: torch.Tensor,
    valid: torch.Tensor,
    entries: list[KNetEntry | None],
    reservoir: list[tuple[torch.Tensor, torch.Tensor]],
    frame: FrameBatch,
    *,
    control: str,
    update_stride: int,
    missing_rate: float,
    coordinate_noise_px: float,
    seed: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    used_obs_diff = obs_diff.clone()
    used_innovation = innovation.clone()
    mask = valid.clone()
    valid_indices = torch.nonzero(mask > 0.5, as_tuple=False).flatten()

    if control == "no_update":
        mask.zero_()
    elif control == "wrong_cell":
        if len(valid_indices) > 1:
            used_obs_diff[valid_indices] = torch.roll(used_obs_diff[valid_indices], shifts=1, dims=0)
            used_innovation[valid_indices] = torch.roll(used_innovation[valid_indices], shifts=1, dims=0)
    elif control == "time_shuffled":
        if reservoir and len(valid_indices):
            rng = v97.deterministic_rng(seed, frame.sequence, frame.frame, 9801)
            selected = rng.integers(0, len(reservoir), size=len(valid_indices))
            used_obs_diff[valid_indices] = torch.stack(
                [reservoir[int(index)][0].to(obs_diff.device) for index in selected]
            )
            used_innovation[valid_indices] = torch.stack(
                [reservoir[int(index)][1].to(innovation.device) for index in selected]
            )
        else:
            mask.zero_()
    elif control == "delayed":
        for local_index in valid_indices.tolist():
            entry = entries[int(local_index)]
            if entry is None:
                mask[int(local_index)] = 0.0
            else:
                used_obs_diff[int(local_index)] = entry.last_obs_diff.to(obs_diff.device)
                used_innovation[int(local_index)] = entry.last_innovation.to(innovation.device)
    elif control != "real":
        raise ValueError(f"Unknown control: {control}")

    if int(update_stride) > 1 and frame.frame % int(update_stride) != 0:
        mask.zero_()

    if missing_rate > 0:
        if training:
            keep = torch.rand(len(mask), device=mask.device) >= float(missing_rate)
        else:
            rng = v97.deterministic_rng(seed, frame.sequence, frame.frame, 9827)
            keep = torch.as_tensor(rng.random(len(mask)) >= float(missing_rate), device=mask.device)
        mask *= keep.float()

    if coordinate_noise_px > 0:
        sigma = math.sqrt(2.0) * float(coordinate_noise_px)
        if training:
            noise = torch.randn_like(used_innovation) * sigma
        else:
            rng = v97.deterministic_rng(seed, frame.sequence, frame.frame, 9851)
            noise = torch.as_tensor(
                rng.normal(0.0, sigma, used_innovation.shape),
                dtype=used_innovation.dtype,
                device=used_innovation.device,
            )
        used_obs_diff = used_obs_diff + noise * mask[:, None]
        used_innovation = used_innovation + noise * mask[:, None]

    return used_obs_diff, used_innovation, mask


def forward_frame(
    model: KalmanNetOnlineForecaster,
    bundle: v84.AnchorBundle,
    frame: FrameBatch,
    cache: dict[int, KNetEntry],
    reservoir: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    control: str,
    update_stride: int,
    missing_rate: float,
    coordinate_noise_px: float,
    transition_dt_scale: float,
    seed: int,
    training: bool,
    predictive_event: str = LEGACY_PREDICTIVE_EVENT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    rows = bundle.rows.loc[frame.indices]
    positions = torch.as_tensor(rows[["x_px", "y_px"]].to_numpy(np.float32), device=device)
    velocities = torch.as_tensor(rows[["dx_px", "dy_px"]].to_numpy(np.float32), device=device)
    batch = len(frame.indices)
    dtype = positions.dtype
    transition, observation_matrix = model.matrices(
        dt=float(transition_dt_scale), dtype=dtype, device=device
    )

    initial = initial_state(model.kind, positions, velocities)
    hq_init, hsigma_init, hs_init = model.gain_network.initial_hidden(batch, device, dtype)
    previous_state: list[torch.Tensor] = []
    previous_previous: list[torch.Tensor] = []
    previous_prior: list[torch.Tensor] = []
    previous_observation: list[torch.Tensor] = []
    h_q: list[torch.Tensor] = []
    h_sigma: list[torch.Tensor] = []
    h_s: list[torch.Tensor] = []
    entries: list[KNetEntry | None] = []
    valid_values: list[float] = []

    for local_index, track_value in enumerate(frame.track_ids):
        entry = cache.get(int(track_value))
        valid_entry = entry is not None and entry.frame == frame.frame - 1
        entries.append(entry if valid_entry else None)
        valid_values.append(float(valid_entry))
        if valid_entry:
            assert entry is not None
            previous_state.append(entry.posterior)
            previous_previous.append(entry.previous_posterior)
            previous_prior.append(entry.prior)
            previous_observation.append(entry.observation)
            h_q.append(entry.h_q)
            h_sigma.append(entry.h_sigma)
            h_s.append(entry.h_s)
        else:
            previous_state.append(initial[local_index])
            previous_previous.append(initial[local_index])
            previous_prior.append(initial[local_index])
            previous_observation.append(positions[local_index])
            h_q.append(hq_init[local_index])
            h_sigma.append(hsigma_init[local_index])
            h_s.append(hs_init[local_index])

    state_previous = torch.stack(previous_state)
    state_previous_previous = torch.stack(previous_previous)
    prior_previous = torch.stack(previous_prior)
    observation_previous = torch.stack(previous_observation)
    h_q_tensor = torch.stack(h_q)
    h_sigma_tensor = torch.stack(h_sigma)
    h_s_tensor = torch.stack(h_s)
    valid = torch.as_tensor(valid_values, dtype=dtype, device=device)

    state_prior = state_previous @ transition.T
    predicted_observation = state_prior @ observation_matrix.T
    real_obs_diff = positions - observation_previous
    real_innovation = positions - predicted_observation
    evolution_diff = state_previous - state_previous_previous
    update_diff = state_previous - prior_previous
    used_obs_diff, used_innovation, measurement_mask = policy_features(
        real_obs_diff,
        real_innovation,
        valid,
        entries,
        reservoir,
        frame,
        control=control,
        update_stride=update_stride,
        missing_rate=missing_rate,
        coordinate_noise_px=coordinate_noise_px,
        seed=seed,
        training=training,
    )

    gain, next_h_q, next_h_sigma, next_h_s = model.gain_network(
        used_obs_diff,
        used_innovation,
        evolution_diff,
        update_diff,
        h_q_tensor,
        h_sigma_tensor,
        h_s_tensor,
    )
    correction = torch.bmm(gain, used_innovation.unsqueeze(-1)).squeeze(-1)
    updated = state_prior + correction
    posterior = torch.where(measurement_mask[:, None] > 0.5, updated, state_prior)
    posterior = torch.where(valid[:, None] > 0.5, posterior, initial)

    hidden_mask = measurement_mask[:, None] > 0.5
    final_h_q = torch.where(hidden_mask, next_h_q, h_q_tensor)
    final_h_sigma = torch.where(hidden_mask, next_h_sigma, h_sigma_tensor)
    final_h_s = torch.where(hidden_mask, next_h_s, h_s_tensor)
    final_observation = torch.where(hidden_mask, positions, observation_previous)
    # Keep the real completed residual for the delayed-innovation control.  If
    # we stored ``used_innovation`` here, the delayed control would recursively
    # delay its own already delayed value and degenerate into no-update.
    real_history_mask = valid[:, None] > 0.5
    final_obs_diff = torch.where(real_history_mask, real_obs_diff, torch.zeros_like(real_obs_diff))
    final_innovation = torch.where(real_history_mask, real_innovation, torch.zeros_like(real_innovation))

    next_prior = posterior @ transition.T
    prediction = predictive_mean(
        posterior,
        next_prior,
        positions,
        observation_matrix,
        predictive_event,
    )

    for local_index, track_value in enumerate(frame.track_ids):
        cache[int(track_value)] = KNetEntry(
            frame=frame.frame,
            posterior=posterior[local_index],
            previous_posterior=state_previous[local_index],
            prior=state_prior[local_index] if valid[local_index] > 0.5 else initial[local_index],
            observation=final_observation[local_index],
            h_q=final_h_q[local_index],
            h_sigma=final_h_sigma[local_index],
            h_s=final_h_s[local_index],
            last_obs_diff=final_obs_diff[local_index],
            last_innovation=final_innovation[local_index],
        )

    fresh_reservoir = [
        (real_obs_diff[index].detach().cpu(), real_innovation[index].detach().cpu())
        for index in torch.nonzero(valid > 0.5, as_tuple=False).flatten().tolist()
    ]
    return prediction, gain, measurement_mask, fresh_reservoir


def step_loss(prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, loss_name: str) -> torch.Tensor:
    normalized_prediction = prediction / scale
    normalized_target = target / scale
    if loss_name == "mse":
        return F.mse_loss(normalized_prediction, normalized_target)
    if loss_name == "huber":
        return F.smooth_l1_loss(normalized_prediction, normalized_target)
    raise ValueError(loss_name)


def train_epoch(
    model: KalmanNetOnlineForecaster,
    bundle: v84.AnchorBundle,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    frames = grouped_frames(bundle)
    sequences = list(frames)
    rng = np.random.default_rng(int(args.seed) + epoch * 101)
    rng.shuffle(sequences)
    target_scale = torch.as_tensor(
        np.maximum(np.std(bundle.target_steps[:, 0], axis=0), 1.0), dtype=torch.float32, device=device
    )
    cumulative_horizons = v97.parse_ints(args.cumulative_horizons)
    records: list[dict[str, float]] = []

    for sequence in sequences:
        cache: dict[int, KNetEntry] = {}
        windows: dict[int, v97.WindowEntry] = {}
        reservoir: list[tuple[torch.Tensor, torch.Tensor]] = []
        chunk_losses: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        sequence_frames = frames[sequence]
        if int(args.max_train_frames) > 0:
            sequence_frames = sequence_frames[: int(args.max_train_frames)]

        for frame_number, frame in enumerate(sequence_frames):
            prediction, _gain, _mask, additions = forward_frame(
                model,
                bundle,
                frame,
                cache,
                reservoir,
                device,
                control="real",
                update_stride=1,
                missing_rate=float(args.train_missing_rate),
                coordinate_noise_px=float(args.train_coordinate_noise_px),
                transition_dt_scale=1.0,
                seed=int(args.seed) + epoch,
                training=True,
                predictive_event=str(args.predictive_event),
            )
            target = torch.as_tensor(bundle.target_steps[frame.indices, 0], dtype=torch.float32, device=device)
            h1_loss = step_loss(prediction, target, target_scale, str(args.loss))
            cumulative_terms = v97.append_cumulative_terms(
                frame,
                prediction,
                target,
                windows,
                cumulative_horizons,
                target_scale,
            )
            cumulative_loss = (
                torch.stack(cumulative_terms).mean() if cumulative_terms else torch.zeros((), device=device)
            )
            loss = h1_loss + float(args.cumulative_weight) * cumulative_loss
            chunk_losses.append(loss)
            records.append(
                {
                    "loss": float(loss.detach()),
                    "h1_loss": float(h1_loss.detach()),
                    "cumulative_loss": float(cumulative_loss.detach()),
                }
            )
            reservoir.extend(additions)
            if len(reservoir) > int(args.reservoir_size):
                reservoir = reservoir[-int(args.reservoir_size) :]

            boundary = len(chunk_losses) >= int(args.tbptt_frames) or frame_number == len(sequence_frames) - 1
            if boundary:
                torch.stack(chunk_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                cache = {track: detach_entry(entry) for track, entry in cache.items()}
                windows = detach_windows(windows)
                chunk_losses = []

    if not records:
        raise RuntimeError("No chronological training frames were available")
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


@torch.no_grad()
def replay(
    model: KalmanNetOnlineForecaster,
    bundle: v84.AnchorBundle,
    device: torch.device,
    *,
    control: str = "real",
    update_stride: int = 1,
    missing_rate: float = 0.0,
    coordinate_noise_px: float = 0.0,
    transition_dt_scale: float = 1.0,
    seed: int = 42,
    max_frames: int = 0,
    predictive_event: str = LEGACY_PREDICTIVE_EVENT,
) -> KNetReplayResult:
    model.eval()
    output = np.full((len(bundle.rows), 2), np.nan, dtype=np.float32)
    gain_norm = np.full(len(bundle.rows), np.nan, dtype=np.float32)
    gain_trace = np.full(len(bundle.rows), np.nan, dtype=np.float32)
    measurement_mask = np.zeros(len(bundle.rows), dtype=np.float32)
    frames = grouped_frames(bundle)

    for sequence in sorted(frames):
        cache: dict[int, KNetEntry] = {}
        reservoir: list[tuple[torch.Tensor, torch.Tensor]] = []
        sequence_frames = frames[sequence]
        if int(max_frames) > 0:
            sequence_frames = sequence_frames[: int(max_frames)]
        for frame in sequence_frames:
            prediction, gain, mask, additions = forward_frame(
                model,
                bundle,
                frame,
                cache,
                reservoir,
                device,
                control=control,
                update_stride=update_stride,
                missing_rate=missing_rate,
                coordinate_noise_px=coordinate_noise_px,
                transition_dt_scale=transition_dt_scale,
                seed=seed,
                training=False,
                predictive_event=predictive_event,
            )
            indices = frame.indices
            output[indices] = prediction.cpu().numpy()
            gain_norm[indices] = torch.linalg.vector_norm(gain, dim=(-2, -1)).cpu().numpy()
            diagonal = torch.diagonal(gain[:, :2, :2], dim1=-2, dim2=-1).mean(dim=-1)
            gain_trace[indices] = diagonal.cpu().numpy()
            measurement_mask[indices] = mask.cpu().numpy()
            cache = {track: detach_entry(entry) for track, entry in cache.items()}
            reservoir.extend(additions)
            if len(reservoir) > 4096:
                reservoir = reservoir[-4096:]

    missing = ~np.isfinite(output).all(axis=1)
    if missing.any():
        # This path is used only by deliberately truncated smoke evaluation.
        output[missing] = bundle.rows.loc[missing, ["dx_px", "dy_px"]].to_numpy(np.float32)
    return KNetReplayResult(output, gain_norm, gain_trace, measurement_mask)


def train_model(
    kind: str,
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[KalmanNetOnlineForecaster, dict[str, Any], list[dict[str, Any]]]:
    model = KalmanNetOnlineForecaster(kind, int(args.in_mult), int(args.out_mult)).to(device)
    checkpoint_path = args.out_dir / f"kalmannet_{kind}.pt"
    start_epoch = 1
    if args.resume_dir:
        resume_path = Path(args.resume_dir) / checkpoint_path.name
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device, weights_only=False)
            checkpoint_event = str(
                payload.get("metadata", {}).get(
                    "predictive_event",
                    payload.get("args", {}).get("predictive_event", LEGACY_PREDICTIVE_EVENT),
                )
            )
            if checkpoint_event != str(args.predictive_event):
                raise ValueError(
                    "Checkpoint predictive-event mismatch: "
                    f"checkpoint={checkpoint_event}, requested={args.predictive_event}. "
                    f"Use --predictive-event {checkpoint_event} to reproduce this checkpoint."
                )
            model.load_state_dict(payload["state_dict"])
            start_epoch = int(payload.get("metadata", {}).get("best_epoch", 0)) + 1
        elif args.eval_only:
            raise FileNotFoundError(resume_path)
    elif args.eval_only:
        raise ValueError("--eval-only requires --resume-dir")

    weights = v97.parse_horizon_weights(args.validation_horizon_weights)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = max(0, start_epoch - 1)
    patience = 0
    logs: list[dict[str, Any]] = []

    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
        for epoch in range(start_epoch, int(args.epochs) + 1):
            train_values = train_epoch(model, bundles[0], optimizer, device, args, epoch)
            val_result = replay(
                model,
                bundles[1],
                device,
                seed=int(args.seed),
                max_frames=int(args.max_eval_frames),
                predictive_event=str(args.predictive_event),
            )
            val_score = v97.weighted_rolling_score(bundles[1], val_result.prediction, weights)
            row = {"state_model": kind, "epoch": epoch, "val_score": val_score, **train_values}
            logs.append(row)
            print(
                f"[v98:{kind}] epoch={epoch:03d} loss={train_values['loss']:.6f} val={val_score:.6f}",
                flush=True,
            )
            if val_score < best_score - float(args.min_delta):
                best_score = float(val_score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(args.patience):
                    break
        model.load_state_dict(best_state)
    else:
        best_score = float("nan")

    metadata = {
        "state_model": kind,
        "best_epoch": best_epoch,
        "best_val_weighted_rmse": best_score,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "state_dim": model.state_dim,
        "observation_dim": model.observation_dim,
        "architecture": "official_kalmannet_architecture_2_grucell_equivalent",
        "transition": "known_standard_position_cv_or_ca",
        "observation": "known_position_H",
        "loss": str(args.loss),
        "cumulative_weight": float(args.cumulative_weight),
        "validation_horizon_weights": str(args.validation_horizon_weights),
        "predictive_event": str(args.predictive_event),
        "point_mean_definition": (
            "H_F_x_post_t_minus_observed_x_t"
            if args.predictive_event == PUBLICATION_PREDICTIVE_EVENT
            else "H_F_x_post_t_minus_H_x_post_t"
        ),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "args": finite(vars(args)),
        },
        checkpoint_path,
    )
    return model, metadata, logs


def causal_audit(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    movie_sets: list[set[int]] = []
    for split_name, bundle in zip(("train", "val", "test"), bundles):
        rows = bundle.rows
        required = set(KEYS + ["x_px", "y_px", "dx_px", "dy_px"])
        missing_columns = sorted(required.difference(rows.columns))
        duplicated = int(rows.duplicated(KEYS).sum())
        previous = v97.previous_row_lookup(bundle)
        valid = previous >= 0
        frames = rows["frame"].to_numpy(np.int64)
        source_frames = np.full(len(rows), -1, dtype=np.int64)
        source_frames[valid] = rows.iloc[previous[valid]]["frame"].to_numpy(np.int64)
        temporal_violations = int((valid & (source_frames != frames - 1)).sum())
        movies = set(map(int, rows.sequence.unique()))
        movie_sets.append(movies)
        records.append(
            {
                "split": split_name,
                "rows": len(rows),
                "movies": ",".join(map(str, sorted(movies))),
                "missing_required_columns": ",".join(missing_columns),
                "duplicate_keys": duplicated,
                "chronology_violations": temporal_violations,
                "future_target_inference_features": 0,
                "prediction_issued": "after_observation_t_before_target_t_plus_1",
                "causal_violations": int(bool(missing_columns)) + duplicated + temporal_violations,
            }
        )
    overlap = sum(len(movie_sets[left] & movie_sets[right]) for left in range(3) for right in range(left + 1, 3))
    records.append(
        {
            "split": "cross_split",
            "rows": sum(len(bundle.rows) for bundle in bundles),
            "movies": "",
            "missing_required_columns": "",
            "duplicate_keys": 0,
            "chronology_violations": 0,
            "future_target_inference_features": 0,
            "prediction_issued": "movie_held_out_splits",
            "causal_violations": int(overlap),
        }
    )
    return pd.DataFrame(records)


def metric_rows(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    horizons: list[int],
    method: str,
    family: str,
    **extra: Any,
) -> list[dict[str, Any]]:
    return v97.rolling_metric_rows(
        bundle,
        prediction,
        horizons,
        method,
        {"family": family, **extra},
    )


def no_future_sentinel_audit(
    model: KalmanNetOnlineForecaster,
    bundle: v84.AnchorBundle,
    device: torch.device,
    seed: int,
    predictive_event: str = LEGACY_PREDICTIVE_EVENT,
) -> dict[str, Any]:
    original = replay(model, bundle, device, seed=seed, predictive_event=predictive_event)
    sentinel = copy.copy(bundle)
    sentinel.target_steps = np.full_like(bundle.target_steps, 1.2345678e6)
    changed = replay(model, sentinel, device, seed=seed, predictive_event=predictive_event)
    prediction_delta = float(np.max(np.abs(original.prediction - changed.prediction)))
    gain_delta = float(np.max(np.abs(original.gain_norm - changed.gain_norm)))
    return {
        "target_sentinel": 1234567.8,
        "max_abs_prediction_delta": prediction_delta,
        "max_abs_gain_delta": gain_delta,
        "predictive_event": predictive_event,
        "row_key_sha256": array_sha256(bundle.rows[KEYS].to_numpy(np.int64, copy=True)),
        "original_h1_target_sha256": array_sha256(bundle.target_steps[:, 0]),
        "sentinel_h1_target_sha256": array_sha256(sentinel.target_steps[:, 0]),
        "future_placeholder_read_at_inference": bool(prediction_delta > 0.0 or gain_delta > 0.0),
        "pass": bool(prediction_delta == 0.0 and gain_delta == 0.0),
    }


def official_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    v97.seed_everything(int(args.seed))
    device = v97.device_from_args(args)
    bundles = v85.load_anchor_cache(args.anchor_cache)
    horizons = v97.parse_ints(args.horizons)
    models = parse_strings(args.models)
    unknown = sorted(set(models).difference({"cv", "ca"}))
    if unknown:
        raise ValueError(f"Unknown state models: {unknown}")

    if args.smoke:
        args.epochs = min(int(args.epochs), 2)
        args.patience = 1
        args.max_train_frames = int(args.max_train_frames) or 16
        args.max_eval_frames = 0
        args.bootstrap_repeats = min(int(args.bootstrap_repeats), 100)

    contract_hashes = bundle_contract_hashes(bundles)
    audit = causal_audit(bundles).assign(predictive_event=str(args.predictive_event))
    if int(audit.causal_violations.sum()) != 0:
        raise RuntimeError(f"Causal audit failed:\n{audit.to_string(index=False)}")

    summary: list[dict[str, Any]] = []
    robustness: list[dict[str, Any]] = []
    mismatch: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    validation_scores: dict[str, float] = {}
    trained: dict[str, KalmanNetOnlineForecaster] = {}

    for kind in models:
        print(f"[v98] training KalmanNet-{kind.upper()} on {device}", flush=True)
        model, metadata, model_logs = train_model(kind, bundles, args, device)
        result = replay(
            model,
            bundles[2],
            device,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        val_result = replay(
            model,
            bundles[1],
            device,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        method = f"kalmannet_{kind}"
        summary.extend(metric_rows(bundles[2], result.prediction, horizons, method, "kalmannet", control="real"))
        predictions[f"{method}__prediction"] = result.prediction
        predictions[f"{method}__gain_norm"] = result.gain_norm
        validation_scores[kind] = v97.weighted_rolling_score(
            bundles[1], val_result.prediction, v97.parse_horizon_weights(args.validation_horizon_weights)
        )
        gain_rows.append(
            {
                "method": method,
                "gain_norm_mean": float(np.nanmean(result.gain_norm)),
                "gain_norm_p10": float(np.nanquantile(result.gain_norm, 0.10)),
                "gain_norm_p90": float(np.nanquantile(result.gain_norm, 0.90)),
                "position_gain_trace_mean": float(np.nanmean(result.gain_trace)),
                "measurement_coverage": float(result.measurement_mask.mean()),
            }
        )
        logs.extend(model_logs)
        metadata_rows.append(metadata)
        trained[kind] = model

    winner = min(validation_scores, key=validation_scores.get)
    winner_model = trained[winner]
    winner_method = f"kalmannet_{winner}"
    print(f"[v98] validation-selected robustness model: {winner_method}", flush=True)
    sentinel = no_future_sentinel_audit(
        winner_model,
        bundles[2],
        device,
        seed=int(args.seed),
        predictive_event=str(args.predictive_event),
    )
    if not bool(sentinel["pass"]):
        raise RuntimeError(f"No-future sentinel audit failed: {sentinel}")

    for stride in v97.parse_ints(args.update_strides):
        result = replay(
            winner_model,
            bundles[2],
            device,
            update_stride=stride,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        robustness.extend(
            metric_rows(
                bundles[2],
                result.prediction,
                horizons,
                f"{winner_method}_stride{stride}",
                "cadence",
                control="real",
                update_stride=stride,
                missing_rate=0.0,
                coordinate_noise_px=0.0,
            )
        )

    for missing_rate in v97.parse_floats(args.missing_rates):
        if missing_rate <= 0:
            continue
        result = replay(
            winner_model,
            bundles[2],
            device,
            missing_rate=missing_rate,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        robustness.extend(
            metric_rows(
                bundles[2],
                result.prediction,
                horizons,
                f"{winner_method}_missing_{missing_rate:g}",
                "missingness",
                control="random_missing",
                update_stride=1,
                missing_rate=missing_rate,
                coordinate_noise_px=0.0,
            )
        )

    for noise in v97.parse_floats(args.coordinate_noise_grid):
        if noise <= 0:
            continue
        result = replay(
            winner_model,
            bundles[2],
            device,
            coordinate_noise_px=noise,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        robustness.extend(
            metric_rows(
                bundles[2],
                result.prediction,
                horizons,
                f"{winner_method}_noise_{noise:g}px",
                "tracking_noise",
                control="coordinate_noise",
                update_stride=1,
                missing_rate=0.0,
                coordinate_noise_px=noise,
            )
        )

    for control in ("wrong_cell", "time_shuffled", "delayed", "no_update"):
        result = replay(
            winner_model,
            bundles[2],
            device,
            control=control,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        robustness.extend(
            metric_rows(
                bundles[2],
                result.prediction,
                horizons,
                f"{winner_method}_{control}",
                "causal_control",
                control=control,
                update_stride=1,
                missing_rate=0.0,
                coordinate_noise_px=0.0,
            )
        )

    for dt_scale in v97.parse_floats(args.transition_dt_scales):
        if math.isclose(dt_scale, 1.0):
            continue
        result = replay(
            winner_model,
            bundles[2],
            device,
            transition_dt_scale=dt_scale,
            seed=int(args.seed),
            predictive_event=str(args.predictive_event),
        )
        mismatch.extend(
            metric_rows(
                bundles[2],
                result.prediction,
                horizons,
                f"{winner_method}_transition_dt_{dt_scale:g}",
                "process_model_mismatch",
                control="real",
                transition_dt_scale=dt_scale,
            )
        )

    constant_velocity = bundles[2].rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    summary.extend(metric_rows(bundles[2], constant_velocity, horizons, "constant_velocity", "baseline", control="real"))
    predictions["baseline__constant_velocity"] = constant_velocity

    summary_df = pd.DataFrame(summary)
    robustness_df = pd.DataFrame(robustness)
    mismatch_df = pd.DataFrame(mismatch)
    gain_df = pd.DataFrame(gain_rows)
    for frame in (summary_df, robustness_df, mismatch_df, gain_df):
        frame["predictive_event"] = str(args.predictive_event)
    hmax = max(horizons)
    summary_h = summary_df[summary_df.horizon.eq(hmax)].sort_values("component_rmse")
    controls_h = robustness_df[
        robustness_df.horizon.eq(hmax) & robustness_df.family.eq("causal_control")
    ].sort_values("component_rmse")

    bootstrap_rows: list[dict[str, Any]] = []
    winner_prediction = predictions[f"{winner_method}__prediction"]
    bootstrap = v97.cluster_bootstrap_delta(
        bundles[2],
        winner_prediction,
        constant_velocity,
        hmax,
        int(args.bootstrap_repeats),
        int(args.seed) + 98,
    )
    bootstrap.update(model=winner_method, baseline="constant_velocity")
    bootstrap_rows.append(bootstrap)

    data_contract = pd.DataFrame(
        [
            {
                "anchor_cache": str(args.anchor_cache),
                "train_movies": ",".join(map(str, sorted(bundles[0].rows.sequence.unique()))),
                "val_movies": ",".join(map(str, sorted(bundles[1].rows.sequence.unique()))),
                "test_movies": ",".join(map(str, sorted(bundles[2].rows.sequence.unique()))),
                "train_rows": len(bundles[0].rows),
                "val_rows": len(bundles[1].rows),
                "test_rows": len(bundles[2].rows),
                "models": ",".join(models),
                "selected_model": winner_method,
                "predictive_event": str(args.predictive_event),
                "publication_mode": bool(args.predictive_event == PUBLICATION_PREDICTIVE_EVENT),
                "point_mean_definition": (
                    "H_F_x_post_t_minus_observed_x_t"
                    if args.predictive_event == PUBLICATION_PREDICTIVE_EVENT
                    else "H_F_x_post_t_minus_H_x_post_t"
                ),
                "prior_or_posterior": (
                    "next_observation_prior_mean_after_posterior_t_update"
                    if args.predictive_event == PUBLICATION_PREDICTIVE_EVENT
                    else "legacy_posterior_state_increment"
                ),
                "predictive_covariance": "unavailable_point_only_comparator",
                "contract": "observation_t -> posterior_t -> issue prediction before observation_t_plus_1; rolling accumulation",
                "inference_inputs": "current/past positions, completed displacement for initialization, recurrent filter state",
                "future_target_inference_features": 0,
                "official_reference": "https://github.com/KalmanNet/KalmanNet_TSP",
                "validation_horizon_weights": str(args.validation_horizon_weights),
                **contract_hashes,
            }
        ]
    )

    predictions["contract__row_keys"] = bundles[2].rows[KEYS].to_numpy(np.int64, copy=True)
    predictions["contract__test_key_sha256"] = np.asarray(contract_hashes["test_key_sha256"])
    predictions["contract__test_h1_target_sha256"] = np.asarray(
        contract_hashes["test_h1_target_sha256"]
    )
    predictions["contract__predictive_event"] = np.asarray(str(args.predictive_event))

    summary_df.to_csv(args.out_dir / "v98_online_summary.csv", index=False)
    robustness_df.to_csv(args.out_dir / "v98_robustness.csv", index=False)
    robustness_df[robustness_df.family.eq("cadence")].to_csv(
        args.out_dir / "v98_update_cadence.csv", index=False
    )
    mismatch_df.to_csv(args.out_dir / "v98_model_mismatch.csv", index=False)
    gain_df.to_csv(args.out_dir / "v98_gain_diagnostics.csv", index=False)
    pd.DataFrame(logs).to_csv(args.out_dir / "v98_train_log.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(args.out_dir / "v98_model_metadata.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.out_dir / "v98_bootstrap.csv", index=False)
    audit.to_csv(args.out_dir / "v98_causal_audit.csv", index=False)
    data_contract.to_csv(args.out_dir / "v98_data_contract.csv", index=False)
    np.savez_compressed(args.out_dir / "v98_predictions.npz", **predictions)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(finite(vars(args)), indent=2), encoding="utf-8"
    )
    (args.out_dir / "v98_no_future_sentinel.json").write_text(
        json.dumps(finite(sentinel), indent=2), encoding="utf-8"
    )
    (args.out_dir / "v98_provenance.json").write_text(
        json.dumps(
            {
                "official_repository": "https://github.com/KalmanNet/KalmanNet_TSP",
                "official_commit": official_commit(),
                "fidelity": "source-faithful Architecture #2 domain adaptation using per-track GRUCell state",
                "task": "causal online next-displacement forecasting",
                "predictive_event": str(args.predictive_event),
                "point_mean_definition": data_contract.point_mean_definition.iloc[0],
                "predictive_covariance": "unavailable_point_only_comparator",
                "validation_horizon_weights": str(args.validation_horizon_weights),
                "contract_hashes": contract_hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# v98 KalmanNet Online Comparator",
        "",
        "## Rolling cumulative h6",
        "",
        summary_h[["method", "component_rmse", "r2", "cosine", "magnitude_ratio", "family"]].to_markdown(index=False),
        "",
        "## Causal controls",
        "",
        controls_h[["method", "control", "component_rmse", "r2"]].to_markdown(index=False),
        "",
        "## Contract",
        "",
        f"- Validation-selected state model: `{winner_method}`.",
        f"- Predictive event: `{args.predictive_event}` (`{data_contract.point_mean_definition.iloc[0]}`).",
        f"- Causal audit violations: `{int(audit.causal_violations.sum())}`.",
        f"- No-future target sentinel pass: `{sentinel['pass']}` (prediction delta `{sentinel['max_abs_prediction_delta']}`).",
        "- Every prediction is emitted after observing frame t and before the target transition t->t+1.",
        "- CV/CA F and position H are fixed; the official architecture-2 recurrent flow learns only the Kalman gain.",
        "- Random observation dropout/noise are applied only to completed measurements during training.",
        "- This is an adaptation from state estimation to next-displacement forecasting; it is not an unmodified upstream experiment.",
        f"- Elapsed: `{(time.time() - started) / 3600.0:.3f} h`.",
    ]
    report = "\n".join(lines) + "\n"
    (args.out_dir / "v98_decision_report.md").write_text(report, encoding="utf-8")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", default="cv,ca")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--cumulative-horizons", default="2,4,6")
    parser.add_argument(
        "--predictive-event",
        choices=PREDICTIVE_EVENTS,
        default=PUBLICATION_PREDICTIVE_EVENT,
        help=(
            "Publication mode scores H F x_post_t - observed_x_t. "
            "state_increment preserves the historical H(F-I)x_post_t ablation."
        ),
    )
    parser.add_argument("--validation-horizon-weights", default=PUBLICATION_VALIDATION_WEIGHTS)
    parser.add_argument("--update-strides", default="1,2,3,6,1000000")
    parser.add_argument("--missing-rates", default="0.1,0.2,0.4")
    parser.add_argument("--coordinate-noise-grid", default="0.25,0.5,1.0")
    parser.add_argument("--transition-dt-scales", default="0.75,1.25")
    parser.add_argument("--in-mult", type=int, default=5)
    parser.add_argument("--out-mult", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--tbptt-frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--cumulative-weight", type=float, default=0.0)
    parser.add_argument("--train-missing-rate", type=float, default=0.15)
    parser.add_argument("--train-coordinate-noise-px", type=float, default=0.35)
    parser.add_argument("--reservoir-size", type=int, default=4096)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=3000)
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-eval-frames", type=int, default=0)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--eval-only", action="store_true")
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
        (args.out_dir / "v98_error.json").write_text(
            json.dumps(finite(payload), indent=2), encoding="utf-8"
        )
        print(payload["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
