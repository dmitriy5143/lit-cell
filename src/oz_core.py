#!/usr/bin/env python3
"""Causal self + flow + OZ-structural influence architecture.

This study fixes a subtle target-availability leak in the earlier prototype:
the graph at time t contains every cell with a causal history, including cells
without a valid t+1 target.  The target mask is used only by losses and metrics.

The model follows the staged plan:

    temporal self -> coarse flow -> structural social response -> optional joint

The social decoder is intentionally constrained to observable equivariant
vector bases.  It cannot emit an arbitrary two-dimensional edge message.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
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
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_clean_spatial_identifiability_test import (  # noqa: E402
    DATASETS,
    HISTORY,
    load_dataset,
    prior_functions,
    vector_metrics,
)


OUT_DIR = ROOT / "outputs" / "oz_full_architecture_study"
VARIANTS = ("geometry_structural", "oz_structural")
FLOW_COLUMNS = (
    "position_x_centered",
    "position_y_centered",
    "boundary_left_norm",
    "boundary_right_norm",
    "boundary_top_norm",
    "boundary_bottom_norm",
    "boundary_min_norm",
    "flow_global_x",
    "flow_global_y",
    "flow_global_relative_x",
    "flow_global_relative_y",
    "flow_global_speed_mean",
    "flow_far_x",
    "flow_far_y",
    "flow_far_relative_x",
    "flow_far_relative_y",
    "flow_far_dispersion",
    "flow_far_count",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    return torch.device(name)


def history_columns() -> list[tuple[str, str, str]]:
    return [
        (f"self_dx_lag{lag}", f"self_dy_lag{lag}", f"self_speed_lag{lag}")
        for lag in reversed(range(HISTORY - 1))
    ]


def extract_history(samples: pd.DataFrame) -> np.ndarray:
    parts = [
        samples[[dx, dy, speed]].to_numpy(np.float32)
        for dx, dy, speed in history_columns()
    ]
    return np.stack(parts, axis=1)


def build_causal_samples(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Build nodes without conditioning graph membership on a future target."""
    rows: list[dict[str, Any]] = []
    for _, g0 in df.groupby("GLOBAL_TRACK_ID", sort=False):
        g = g0.sort_values("FRAME").reset_index(drop=True)
        frames = g["FRAME"].to_numpy(int)
        pos = g[["x_px", "y_px"]].to_numpy(float)
        step = g[["raw_dx", "raw_dy"]].to_numpy(float)
        for idx in range(HISTORY - 1, len(g)):
            start = idx - HISTORY + 1
            if not np.all(np.diff(frames[start : idx + 1]) == 1):
                continue
            history_step = step[start + 1 : idx + 1]
            if not np.isfinite(history_step).all():
                continue
            valid_target = (
                idx + horizon < len(g)
                and frames[idx + horizon] == frames[idx] + horizon
                and g.loc[idx, "split"] == g.loc[idx + horizon, "split"]
            )
            if valid_target:
                target = pos[idx + horizon] - pos[idx]
            else:
                target = np.array([np.nan, np.nan], dtype=float)
            row: dict[str, Any] = {
                "SEQ_ID": int(g.loc[idx, "SEQ_ID"]),
                "FRAME": int(frames[idx]),
                "TRACK_ID": int(g.loc[idx, "TRACK_ID"]),
                "GLOBAL_TRACK_ID": str(g.loc[idx, "GLOBAL_TRACK_ID"]),
                "split": str(g.loc[idx, "split"]),
                "history_valid": True,
                "target_valid": bool(valid_target),
                "target_dx": float(target[0]),
                "target_dy": float(target[1]),
                "current_dx": float(history_step[-1, 0]),
                "current_dy": float(history_step[-1, 1]),
                "current_speed": float(np.linalg.norm(history_step[-1])),
                "current_x_px": float(pos[idx, 0]),
                "current_y_px": float(pos[idx, 1]),
                "quality_proxy": float(g.loc[idx, "quality_proxy"]),
            }
            for lag in range(HISTORY - 1):
                reverse = HISTORY - 2 - lag
                dx, dy = history_step[reverse]
                row[f"self_dx_lag{lag}"] = float(dx)
                row[f"self_dy_lag{lag}"] = float(dy)
                row[f"self_speed_lag{lag}"] = float(math.hypot(dx, dy))
            rows.append(row)
    return pd.DataFrame(rows)


def add_visible_context_nodes(
    samples: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    """Add every visible cell in forecasted frames as a context-only node."""
    frame_keys = samples[["SEQ_ID", "FRAME"]].drop_duplicates()
    visible = raw.merge(frame_keys, on=["SEQ_ID", "FRAME"], how="inner")
    existing = pd.MultiIndex.from_frame(
        samples[["SEQ_ID", "FRAME", "TRACK_ID"]]
    )
    visible_index = pd.MultiIndex.from_frame(
        visible[["SEQ_ID", "FRAME", "TRACK_ID"]]
    )
    missing = visible.loc[~visible_index.isin(existing)].copy()
    if missing.empty:
        return samples
    context = pd.DataFrame(
        {
            "SEQ_ID": missing["SEQ_ID"].astype(int),
            "FRAME": missing["FRAME"].astype(int),
            "TRACK_ID": missing["TRACK_ID"].astype(int),
            "GLOBAL_TRACK_ID": missing["GLOBAL_TRACK_ID"].astype(str),
            "split": missing["split"].astype(str),
            "history_valid": False,
            "target_valid": False,
            "target_dx": np.nan,
            "target_dy": np.nan,
            "current_dx": missing["raw_dx"].fillna(0.0).astype(float),
            "current_dy": missing["raw_dy"].fillna(0.0).astype(float),
            "current_x_px": missing["x_px"].astype(float),
            "current_y_px": missing["y_px"].astype(float),
            "quality_proxy": missing["quality_proxy"].fillna(0.0).astype(float),
        }
    )
    context["current_speed"] = np.hypot(
        context["current_dx"], context["current_dy"]
    )
    for lag in range(HISTORY - 1):
        context[f"self_dx_lag{lag}"] = 0.0
        context[f"self_dy_lag{lag}"] = 0.0
        context[f"self_speed_lag{lag}"] = 0.0
    return (
        pd.concat([samples, context], ignore_index=True)
        .sort_values(["SEQ_ID", "FRAME", "TRACK_ID"])
        .reset_index(drop=True)
    )


def add_causal_flow_features(samples: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add position, boundary and deliberately coarse causal flow controls."""
    out = samples.copy()
    width = float(DATASETS[dataset]["frame_width_px"])
    height = float(DATASETS[dataset]["frame_height_px"])
    x = out["current_x_px"].to_numpy(float)
    y = out["current_y_px"].to_numpy(float)
    out["position_x_centered"] = (x - 0.5 * width) / width
    out["position_y_centered"] = (y - 0.5 * height) / height
    out["boundary_left_norm"] = np.clip(x / width, 0.0, 1.0)
    out["boundary_right_norm"] = np.clip((width - x) / width, 0.0, 1.0)
    out["boundary_top_norm"] = np.clip(y / height, 0.0, 1.0)
    out["boundary_bottom_norm"] = np.clip((height - y) / height, 0.0, 1.0)
    out["boundary_min_norm"] = out[
        [
            "boundary_left_norm",
            "boundary_right_norm",
            "boundary_top_norm",
            "boundary_bottom_norm",
        ]
    ].min(axis=1)

    flow = np.zeros((len(out), 11), dtype=np.float32)
    for _, idx0 in out.groupby(["SEQ_ID", "FRAME"], sort=False).groups.items():
        idx = np.asarray(list(idx0), dtype=np.int64)
        pos = out.loc[idx, ["current_x_px", "current_y_px"]].to_numpy(float)
        vel = out.loc[idx, ["current_dx", "current_dy"]].to_numpy(float)
        speed = np.linalg.norm(vel, axis=1)
        n = len(idx)
        r_cut = float(DATASETS[dataset]["r_cut_px"])
        if n > 1:
            tree = cKDTree(pos)
            local_sets = tree.query_ball_point(pos, r_cut)
            global_mean = np.zeros((n, 2), dtype=float)
            global_speed = np.zeros(n, dtype=float)
            for row, local in enumerate(local_sets):
                outside = np.ones(n, dtype=bool)
                outside[np.asarray(local, dtype=int)] = False
                if not np.any(outside):
                    outside[:] = True
                    outside[row] = False
                global_mean[row] = vel[outside].mean(axis=0)
                global_speed[row] = speed[outside].mean()
        else:
            global_mean = np.zeros((n, 2), dtype=float)
            global_speed = np.zeros(n, dtype=float)
        if n > 1:
            take = min(33, n)
            dist, nbr = tree.query(pos, k=take)
            dist = np.asarray(dist)
            nbr = np.asarray(nbr)
            if dist.ndim == 1:
                dist = dist[:, None]
                nbr = nbr[:, None]
            dist, nbr = dist[:, 1:], nbr[:, 1:]
        else:
            dist = np.empty((n, 0))
            nbr = np.empty((n, 0), dtype=int)
        start, stop = min(8, dist.shape[1]), min(32, dist.shape[1])
        if stop > start:
            far_dist = dist[:, start:stop]
            far_vel = vel[nbr[:, start:stop]]
            bandwidth = max(2.0 * r_cut, 1.0)
            weight = np.exp(-0.5 * np.square(far_dist / bandwidth))
            weight *= far_dist > r_cut
            denom = np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)
            far_mean = np.sum(weight[:, :, None] * far_vel, axis=1) / denom
            variance = np.sum(
                weight[:, :, None] * np.square(far_vel - far_mean[:, None, :]),
                axis=1,
            ) / denom
            dispersion = np.sqrt(np.maximum(variance.sum(axis=1), 0.0))
            count = (far_dist > r_cut).sum(axis=1).astype(float)
            missing = count == 0
            far_mean[missing] = global_mean[missing]
            dispersion[missing] = 0.0
        else:
            far_mean = global_mean
            dispersion = np.zeros(n, dtype=float)
            count = np.zeros(n, dtype=float)
        flow[idx] = np.column_stack(
            [
                global_mean,
                global_mean - vel,
                global_speed,
                far_mean,
                far_mean - vel,
                dispersion,
                count,
            ]
        )
    names = (
        "flow_global_x",
        "flow_global_y",
        "flow_global_relative_x",
        "flow_global_relative_y",
        "flow_global_speed_mean",
        "flow_far_x",
        "flow_far_y",
        "flow_far_relative_x",
        "flow_far_relative_y",
        "flow_far_dispersion",
        "flow_far_count",
    )
    out.loc[:, names] = flow
    return out


@dataclass
class Normalizer:
    hist_mean: np.ndarray
    hist_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    flow_mean: np.ndarray
    flow_std: np.ndarray
    edge_mean: np.ndarray | None = None
    edge_std: np.ndarray | None = None
    prior_std: np.ndarray | None = None


@dataclass
class GraphArrays:
    frame: np.ndarray
    seq_id: np.ndarray
    track_id: np.ndarray
    y_px: np.ndarray
    history_valid: np.ndarray
    target_valid: np.ndarray
    history: np.ndarray
    flow: np.ndarray
    current_velocity: np.ndarray
    quality: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    radial: np.ndarray
    rel_velocity: np.ndarray
    shear: np.ndarray
    closing: np.ndarray
    edge_features: np.ndarray
    c_correct: np.ndarray
    force_correct: np.ndarray
    c_shuffled: np.ndarray
    force_shuffled: np.ndarray
    degree: np.ndarray


@dataclass
class GraphTensors:
    frame: np.ndarray
    seq_id: np.ndarray
    track_id: np.ndarray
    y_px: torch.Tensor
    y_norm: torch.Tensor
    history_valid: torch.Tensor
    target_valid: torch.Tensor
    history: torch.Tensor
    flow: torch.Tensor
    current_velocity: torch.Tensor
    own_direction: torch.Tensor
    speed_norm: torch.Tensor
    quality: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    radial: torch.Tensor
    rel_velocity: torch.Tensor
    shear: torch.Tensor
    closing: torch.Tensor
    edge_features: torch.Tensor
    c_correct: torch.Tensor
    force_correct: torch.Tensor
    c_shuffled: torch.Tensor
    force_shuffled: torch.Tensor
    degree: torch.Tensor


def fit_normalizer(train: pd.DataFrame) -> Normalizer:
    history_all = extract_history(train)
    history = history_all[train["history_valid"].to_numpy(bool)]
    valid = train["target_valid"].to_numpy(bool)
    y = train.loc[valid, ["target_dx", "target_dy"]].to_numpy(np.float32)
    flow = train.loc[:, FLOW_COLUMNS].to_numpy(np.float32)
    hist_mean = np.nanmean(history, axis=(0, 1))
    hist_std = np.maximum(np.nanstd(history, axis=(0, 1)), 1e-4)
    velocity_scale = max(
        float(
            np.sqrt(
                0.5
                * (
                    np.nanvar(history[:, :, 0])
                    + np.nanvar(history[:, :, 1])
                )
            )
        ),
        1e-4,
    )
    hist_std[:2] = velocity_scale
    target_mean = np.nanmean(y, axis=0)
    target_scale = max(
        float(np.sqrt(0.5 * (np.nanvar(y[:, 0]) + np.nanvar(y[:, 1])))),
        1e-4,
    )
    return Normalizer(
        hist_mean=hist_mean,
        hist_std=hist_std,
        target_mean=target_mean,
        target_std=np.full(2, target_scale, dtype=np.float32),
        flow_mean=np.nanmean(flow, axis=0),
        flow_std=np.maximum(np.nanstd(flow, axis=0), 1e-4),
    )


def _shuffle_within_frames(
    edge_frame: np.ndarray,
    c: np.ndarray,
    force: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c_out, force_out = c.copy(), force.copy()
    for frame_id in np.unique(edge_frame):
        idx = np.flatnonzero(edge_frame == frame_id)
        if len(idx) > 1:
            perm = rng.permutation(idx)
            c_out[idx], force_out[idx] = c[perm], force[perm]
    return c_out, force_out


def build_graph_arrays(
    samples: pd.DataFrame,
    dataset: str,
    *,
    k: int,
    shuffle_seed: int,
) -> GraphArrays:
    samples = samples.reset_index(drop=True)
    n = len(samples)
    pos = samples[["current_x_px", "current_y_px"]].to_numpy(np.float32)
    vel = samples[["current_dx", "current_dy"]].to_numpy(np.float32)
    speed = np.linalg.norm(vel, axis=1)
    quality = np.clip(
        samples["quality_proxy"].fillna(0.0).to_numpy(np.float32), 0.0, 1.0
    )
    history_valid = samples["history_valid"].to_numpy(np.float32)
    r_cut = float(DATASETS[dataset]["r_cut_px"])
    c_fn, dc_fn = prior_functions(dataset)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    frame_counter = 0
    for _, idx0 in samples.groupby(["SEQ_ID", "FRAME"], sort=False).groups.items():
        idx = np.asarray(list(idx0), dtype=np.int64)
        if len(idx) > 1:
            take = min(k + 1, len(idx))
            dist, nbr = cKDTree(pos[idx]).query(pos[idx], k=take)
            dist, nbr = np.asarray(dist), np.asarray(nbr)
            if dist.ndim == 1:
                dist, nbr = dist[:, None], nbr[:, None]
            dist, nbr = dist[:, 1:], nbr[:, 1:]
            valid = dist <= r_cut
            local_dst = np.repeat(np.arange(len(idx)), dist.shape[1])
            local_src = nbr.reshape(-1)
            keep = valid.reshape(-1)
            if np.any(keep):
                local_dst, local_src = local_dst[keep], local_src[keep]
                src_parts.append(idx[local_src])
                dst_parts.append(idx[local_dst])
                frame_parts.append(
                    np.full(int(keep.sum()), frame_counter, dtype=np.int64)
                )
        frame_counter += 1
    src = np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64)
    edge_frame = (
        np.concatenate(frame_parts) if frame_parts else np.empty(0, dtype=np.int64)
    )
    rel_pos = pos[src] - pos[dst]
    distance = np.linalg.norm(rel_pos, axis=1)
    radial = rel_pos / np.maximum(distance[:, None], 1e-6)
    tangent = np.column_stack((-radial[:, 1], radial[:, 0])).astype(np.float32)
    rel_velocity = vel[src] - vel[dst]
    radial_velocity = np.sum(rel_velocity * radial, axis=1)
    transverse_velocity = np.sum(rel_velocity * tangent, axis=1)
    closing = np.maximum(-radial_velocity, 0.0)
    shear = transverse_velocity[:, None] * tangent
    degree = np.bincount(dst, minlength=n).astype(np.float32)
    speed_scale = max(float(np.std(speed)), float(np.median(speed)), 1e-3)
    src_speed, dst_speed = speed[src], speed[dst]
    alignment = np.sum(vel[src] * vel[dst], axis=1) / np.maximum(
        src_speed * dst_speed, 1e-6
    )
    edge_features = np.column_stack(
        [
            distance / r_cut,
            np.log1p(degree[src]) / math.log1p(max(k, 1)),
            np.log1p(degree[dst]) / math.log1p(max(k, 1)),
            quality[src],
            quality[dst],
            src_speed / speed_scale,
            dst_speed / speed_scale,
            radial_velocity / speed_scale,
            np.abs(transverse_velocity) / speed_scale,
            np.clip(alignment, -1.0, 1.0),
            history_valid[src],
            history_valid[dst],
        ]
    ).astype(np.float32)
    c = c_fn(distance).astype(np.float32)
    force = (-dc_fn(distance)).astype(np.float32)
    c_shuffled, force_shuffled = _shuffle_within_frames(
        edge_frame, c, force, shuffle_seed
    )
    y = samples[["target_dx", "target_dy"]].to_numpy(np.float32)
    y[~samples["target_valid"].to_numpy(bool)] = 0.0
    return GraphArrays(
        frame=samples["FRAME"].to_numpy(np.int64),
        seq_id=samples["SEQ_ID"].to_numpy(np.int64),
        track_id=samples["TRACK_ID"].to_numpy(np.int64),
        y_px=y,
        history_valid=samples["history_valid"].to_numpy(bool),
        target_valid=samples["target_valid"].to_numpy(bool),
        history=extract_history(samples),
        flow=samples.loc[:, FLOW_COLUMNS].to_numpy(np.float32),
        current_velocity=vel,
        quality=quality,
        src=src,
        dst=dst,
        radial=radial.astype(np.float32),
        rel_velocity=rel_velocity.astype(np.float32),
        shear=shear.astype(np.float32),
        closing=closing.astype(np.float32),
        edge_features=edge_features,
        c_correct=c,
        force_correct=force,
        c_shuffled=c_shuffled.astype(np.float32),
        force_shuffled=force_shuffled.astype(np.float32),
        degree=degree,
    )


def fit_graph_normalization(train: GraphArrays, norm: Normalizer) -> None:
    norm.edge_mean = train.edge_features.mean(axis=0)
    norm.edge_std = np.maximum(train.edge_features.std(axis=0), 1e-4)
    # Preserve the physical zero: scale c and -dc/dr, but never center them.
    norm.prior_std = np.maximum(
        np.std(np.column_stack([train.c_correct, train.force_correct]), axis=0),
        1e-4,
    )


def graph_to_tensors(
    graph: GraphArrays,
    norm: Normalizer,
    device: torch.device,
) -> GraphTensors:
    assert norm.edge_mean is not None
    assert norm.edge_std is not None
    assert norm.prior_std is not None
    hist = (graph.history - norm.hist_mean) / norm.hist_std
    flow = (graph.flow - norm.flow_mean) / norm.flow_std
    y_norm = (graph.y_px - norm.target_mean) / norm.target_std
    y_norm[~graph.target_valid] = 0.0
    velocity_scale = float(norm.hist_std[0])
    current_velocity = graph.current_velocity / velocity_scale
    current_speed = np.linalg.norm(current_velocity, axis=1)
    own_direction = current_velocity / np.maximum(current_speed[:, None], 1e-6)

    def ft(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    return GraphTensors(
        frame=graph.frame,
        seq_id=graph.seq_id,
        track_id=graph.track_id,
        y_px=ft(graph.y_px),
        y_norm=ft(y_norm),
        history_valid=torch.as_tensor(
            graph.history_valid, dtype=torch.bool, device=device
        ),
        target_valid=torch.as_tensor(graph.target_valid, dtype=torch.bool, device=device),
        history=ft(np.clip(hist, -15.0, 15.0)),
        flow=ft(np.clip(flow, -15.0, 15.0)),
        current_velocity=ft(current_velocity),
        own_direction=ft(own_direction),
        speed_norm=ft(current_speed[:, None]),
        quality=ft(graph.quality[:, None]),
        src=torch.as_tensor(graph.src, dtype=torch.long, device=device),
        dst=torch.as_tensor(graph.dst, dtype=torch.long, device=device),
        radial=ft(graph.radial),
        rel_velocity=ft(graph.rel_velocity / velocity_scale),
        shear=ft(graph.shear / velocity_scale),
        closing=ft(
            graph.closing[:, None] / max(velocity_scale, 1e-4)
        ),
        edge_features=ft(
            np.clip(
                (graph.edge_features - norm.edge_mean) / norm.edge_std,
                -12.0,
                12.0,
            )
        ),
        c_correct=ft(np.clip(graph.c_correct / norm.prior_std[0], -12.0, 12.0)[:, None]),
        force_correct=ft(
            np.clip(graph.force_correct / norm.prior_std[1], -12.0, 12.0)[:, None]
        ),
        c_shuffled=ft(
            np.clip(graph.c_shuffled / norm.prior_std[0], -12.0, 12.0)[:, None]
        ),
        force_shuffled=ft(
            np.clip(graph.force_shuffled / norm.prior_std[1], -12.0, 12.0)[:, None]
        ),
        degree=ft(graph.degree[:, None]),
    )


class TemporalSelfEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(3, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(history)
        state = self.norm(hidden[-1])
        return self.head(state), state


class CoarseFlowEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, state_dim: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(state_dim, 2)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(features)
        return self.head(state), state


def scatter_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(
        (n, values.shape[1]), dtype=values.dtype, device=values.device
    )
    out.index_add_(0, index, values)
    return out


def scatter_max(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.full(
        (n, values.shape[1]),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    expanded = index[:, None].expand(-1, values.shape[1])
    out.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


class StructuralInfluenceDecoder(nn.Module):
    """Context-aware structural encoder plus constrained equivariant response."""

    def __init__(
        self,
        self_dim: int = 48,
        flow_dim: int = 24,
        edge_dim: int = 10,
        edge_state_dim: int = 16,
        hidden_dim: int = 64,
        max_delta_norm: float = 0.9,
    ) -> None:
        super().__init__()
        self.max_delta_norm = float(max_delta_norm)
        node_state_dim = self_dim + flow_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * node_state_dim + edge_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_state_dim),
            nn.SiLU(),
        )
        self.structural_dim = 4 * edge_state_dim + 14
        context_dim = edge_state_dim + 2 * self.structural_dim + 2
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_gate = nn.Linear(hidden_dim, 1)
        self.edge_coeff = nn.Linear(hidden_dim, 4)
        self.prior_amplitude = nn.Linear(hidden_dim, 1)
        self.node_net = nn.Sequential(
            nn.Linear(node_state_dim + self.structural_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        self.prior_strength_raw = nn.Parameter(torch.tensor(-2.5))
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        for layer in (self.edge_gate, self.edge_coeff, self.prior_amplitude):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)
        self.edge_gate.bias.data.fill_(-1.0)
        last = self.node_net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        last.bias.data[0] = -2.0

    def _prior(
        self, graph: GraphTensors, variant: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if variant == "geometry_structural":
            return torch.zeros_like(graph.c_correct), torch.zeros_like(
                graph.force_correct
            )
        if variant == "oz_structural":
            return graph.c_correct, graph.force_correct
        if variant == "oz_shuffled":
            return graph.c_shuffled, graph.force_shuffled
        if variant == "oz_sign_flipped":
            return -graph.c_correct, -graph.force_correct
        raise ValueError(f"Unknown variant: {variant}")

    def _structural_state(
        self,
        edge_state: torch.Tensor,
        graph: GraphTensors,
        c_value: torch.Tensor,
        force_value: torch.Tensor,
    ) -> torch.Tensor:
        n = graph.history.shape[0]
        degree = torch.clamp(graph.degree, min=1.0)
        summed = scatter_sum(edge_state, graph.dst, n)
        mean = summed / degree
        second = scatter_sum(edge_state.square(), graph.dst, n) / degree
        variance = torch.clamp(second - mean.square(), min=0.0)
        maximum = scatter_max(edge_state, graph.dst, n)
        sqrt_sum = summed / torch.sqrt(degree)
        reliability = torch.sqrt(
            torch.clamp(
                graph.quality[graph.src] * graph.quality[graph.dst], min=0.0
            )
        )

        def mean_vec(value: torch.Tensor) -> torch.Tensor:
            return scatter_sum(value, graph.dst, n) / degree

        directed = torch.cat(
            [
                mean_vec(graph.radial),
                scatter_sum(graph.radial, graph.dst, n) / torch.sqrt(degree),
                mean_vec(graph.rel_velocity),
                mean_vec(graph.closing * graph.radial),
                mean_vec(c_value * graph.radial),
                mean_vec(force_value * graph.radial),
                torch.log1p(graph.degree) / math.log(10.0),
                mean_vec(reliability),
            ],
            dim=1,
        )
        return torch.cat([sqrt_sum, mean, variance, maximum, directed], dim=1)

    def forward(
        self,
        graph: GraphTensors,
        self_state: torch.Tensor,
        flow_state: torch.Tensor,
        *,
        variant: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        c_value, force_value = self._prior(graph, variant)
        node_state = torch.cat([self_state, flow_state], dim=1)
        edge_input = torch.cat(
            [
                node_state[graph.src],
                node_state[graph.dst],
                graph.edge_features,
                c_value,
                force_value,
            ],
            dim=1,
        )
        edge_state = self.edge_encoder(edge_input)
        structural = self._structural_state(
            edge_state, graph, c_value, force_value
        )
        context = self.context(
            torch.cat(
                [
                    edge_state,
                    structural[graph.src],
                    structural[graph.dst],
                    c_value,
                    force_value,
                ],
                dim=1,
            )
        )
        reliability = torch.sqrt(
            torch.clamp(
                graph.quality[graph.src] * graph.quality[graph.dst], min=0.0
            )
        )
        edge_gate = torch.sigmoid(self.edge_gate(context))
        coeff = 0.40 * torch.tanh(self.edge_coeff(context))
        prior_amplitude = 2.0 * torch.sigmoid(self.prior_amplitude(context))
        prior_strength = 0.40 * torch.sigmoid(self.prior_strength_raw)
        message = (
            coeff[:, 0:1] * graph.radial
            + coeff[:, 1:2] * graph.closing * graph.radial
            + coeff[:, 2:3] * graph.rel_velocity
            + coeff[:, 3:4] * graph.shear
            + prior_strength * prior_amplitude * force_value * graph.radial
        )
        weighted = reliability * edge_gate * message
        n = graph.history.shape[0]
        summed = scatter_sum(weighted, graph.dst, n)
        effective_degree = scatter_sum(
            reliability * edge_gate, graph.dst, n
        ).clamp_min(1e-4)
        mean_field = summed / effective_degree
        sqrt_field = summed / torch.sqrt(effective_degree)

        node_out = self.node_net(
            torch.cat(
                [
                    node_state,
                    structural,
                    torch.log1p(graph.degree) / math.log(10.0),
                    graph.quality,
                    graph.speed_norm,
                ],
                dim=1,
            )
        )
        # Deterministic hard-sigmoid keeps gradients in the interior while
        # allowing an exact no-interaction state at inference.
        node_gate = torch.clamp(
            1.2 * torch.sigmoid(node_out[:, 0:1]) - 0.1,
            min=0.0,
            max=1.0,
        )
        mix = torch.softmax(node_out[:, 1:3], dim=1)
        mobility = 0.20 + 1.30 * torch.sigmoid(node_out[:, 3:5])
        scale = 0.5 + torch.sigmoid(node_out[:, 5:6])
        field = scale * (
            mix[:, 0:1] * mean_field + mix[:, 1:2] * sqrt_field
        )
        parallel_scalar = torch.sum(
            field * graph.own_direction, dim=1, keepdim=True
        )
        parallel = parallel_scalar * graph.own_direction
        perpendicular = field - parallel
        field = (
            mobility[:, 0:1] * parallel
            + mobility[:, 1:2] * perpendicular
        )
        norm = torch.linalg.vector_norm(field, dim=1, keepdim=True)
        shrink = torch.where(
            norm > 1e-5,
            torch.tanh(norm) / norm.clamp_min(1e-6),
            torch.ones_like(norm),
        )
        delta = self.max_delta_norm * node_gate * field * shrink
        return delta, {
            "structural": structural,
            "edge_gate": edge_gate,
            "node_gate": node_gate,
            "coeff": coeff,
            "prior_amplitude": prior_amplitude,
            "prior_strength": prior_strength,
            "mix": mix,
            "mobility": mobility,
            "effective_degree": effective_degree,
        }


def masked_vector_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    per_node = torch.sum((pred[mask] - target[mask]) ** 2, dim=1)
    if groups is None:
        return torch.mean(per_node)
    local_groups = groups[mask]
    losses = []
    for group in torch.unique(local_groups):
        group_mask = local_groups == group
        if torch.any(group_mask):
            losses.append(torch.mean(per_node[group_mask]))
    if not losses:
        return torch.mean(per_node)
    return torch.mean(torch.stack(losses))


def sequence_groups(graph: GraphTensors) -> torch.Tensor:
    return torch.as_tensor(
        graph.seq_id, dtype=torch.long, device=graph.y_norm.device
    )


def train_temporal(
    train: GraphTensors,
    val: GraphTensors,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    sequence_balanced_loss: bool = False,
) -> tuple[TemporalSelfEncoder, dict[str, float]]:
    set_seed(seed)
    model = TemporalSelfEncoder().to(train.history.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        train.history[train.target_valid],
        train.y_norm[train.target_valid],
        sequence_groups(train)[train.target_valid],
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best, best_val, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0
    for epoch in range(epochs):
        model.train()
        for history, target, seq_id in loader:
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(history)
            if sequence_balanced_loss:
                loss = masked_vector_mse(
                    pred,
                    target,
                    torch.ones(len(target), dtype=torch.bool, device=target.device),
                    seq_id,
                )
            else:
                loss = torch.mean(torch.sum((pred - target) ** 2, dim=1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred, _ = model(val.history)
            score = float(
                masked_vector_mse(
                    pred,
                    val.y_norm,
                    val.target_valid,
                    sequence_groups(val) if sequence_balanced_loss else None,
                )
            )
        if score < best_val - 1e-6:
            best, best_val, best_epoch = (
                copy.deepcopy(model.state_dict()),
                score,
                epoch + 1,
            )
        elif epoch + 1 - best_epoch >= 12:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_norm_mse": best_val}


def train_flow(
    train: GraphTensors,
    val: GraphTensors,
    train_self: torch.Tensor,
    val_self: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    sequence_balanced_loss: bool = False,
) -> tuple[CoarseFlowEncoder, dict[str, float]]:
    set_seed(seed + 1_000)
    model = CoarseFlowEncoder(train.flow.shape[1]).to(train.flow.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    mask = train.target_valid
    dataset = TensorDataset(
        train.flow[mask],
        train.y_norm[mask],
        train_self[mask],
        sequence_groups(train)[mask],
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best, best_val, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0
    for epoch in range(epochs):
        model.train()
        for flow, target, base, seq_id in loader:
            optimizer.zero_grad(set_to_none=True)
            delta, _ = model(flow)
            if sequence_balanced_loss:
                loss = masked_vector_mse(
                    base + delta,
                    target,
                    torch.ones(len(target), dtype=torch.bool, device=target.device),
                    seq_id,
                )
            else:
                loss = torch.mean(torch.sum((base + delta - target) ** 2, dim=1))
            loss = loss + 5e-4 * torch.mean(torch.sum(delta.square(), dim=1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            delta, _ = model(val.flow)
            score = float(
                masked_vector_mse(
                    val_self + delta,
                    val.y_norm,
                    val.target_valid,
                    sequence_groups(val) if sequence_balanced_loss else None,
                )
            )
        if score < best_val - 1e-6:
            best, best_val, best_epoch = (
                copy.deepcopy(model.state_dict()),
                score,
                epoch + 1,
            )
        elif epoch + 1 - best_epoch >= 14:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_norm_mse": best_val}


def train_social(
    train: GraphTensors,
    val: GraphTensors,
    train_base: torch.Tensor,
    val_base: torch.Tensor,
    train_self_state: torch.Tensor,
    val_self_state: torch.Tensor,
    train_flow_state: torch.Tensor,
    val_flow_state: torch.Tensor,
    *,
    variant: str,
    seed: int,
    epochs: int,
    sequence_balanced_loss: bool = False,
) -> tuple[StructuralInfluenceDecoder, dict[str, float]]:
    set_seed(seed + 10_000)
    model = StructuralInfluenceDecoder(edge_dim=train.edge_features.shape[1]).to(
        train.history.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=3e-4)
    best, best_val, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        delta, diag = model(
            train,
            train_self_state,
            train_flow_state,
            variant=variant,
        )
        pred = train_base + delta
        loss = masked_vector_mse(
            pred,
            train.y_norm,
            train.target_valid,
            sequence_groups(train) if sequence_balanced_loss else None,
        )
        active_delta = delta[train.target_valid]
        loss = (
            loss
            + 5e-4 * torch.mean(torch.sum(active_delta.square(), dim=1))
            + 2e-5 * diag["node_gate"].mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            delta, _ = model(
                val, val_self_state, val_flow_state, variant=variant
            )
            score = float(
                masked_vector_mse(
                    val_base + delta,
                    val.y_norm,
                    val.target_valid,
                    sequence_groups(val) if sequence_balanced_loss else None,
                )
            )
        if score < best_val - 1e-6:
            best, best_val, best_epoch = (
                copy.deepcopy(model.state_dict()),
                score,
                epoch + 1,
            )
        elif epoch + 1 - best_epoch >= 18:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_norm_mse": best_val}


def joint_finetune(
    temporal: TemporalSelfEncoder,
    flow: CoarseFlowEncoder,
    social: StructuralInfluenceDecoder,
    train: GraphTensors,
    val: GraphTensors,
    *,
    variant: str,
    epochs: int,
    sequence_balanced_loss: bool = False,
) -> dict[str, float]:
    parameters = [
        {"params": temporal.parameters(), "lr": 8e-5},
        {"params": flow.parameters(), "lr": 1.2e-4},
        {"params": social.parameters(), "lr": 2e-4},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=3e-4)
    best = {
        "temporal": copy.deepcopy(temporal.state_dict()),
        "flow": copy.deepcopy(flow.state_dict()),
        "social": copy.deepcopy(social.state_dict()),
    }
    best_val, best_epoch = float("inf"), 0
    for epoch in range(epochs):
        temporal.train()
        flow.train()
        social.train()
        optimizer.zero_grad(set_to_none=True)
        self_pred, self_state = temporal(train.history)
        self_pred = torch.where(
            train.history_valid[:, None], self_pred, torch.zeros_like(self_pred)
        )
        self_state = torch.where(
            train.history_valid[:, None], self_state, torch.zeros_like(self_state)
        )
        flow_pred, flow_state = flow(train.flow)
        social_pred, diag = social(
            train, self_state, flow_state, variant=variant
        )
        pred = self_pred + flow_pred + social_pred
        loss = masked_vector_mse(
            pred,
            train.y_norm,
            train.target_valid,
            sequence_groups(train) if sequence_balanced_loss else None,
        )
        loss = loss + 5e-4 * torch.mean(
            torch.sum(social_pred[train.target_valid].square(), dim=1)
        )
        loss = loss + 2e-5 * diag["node_gate"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(temporal.parameters())
            + list(flow.parameters())
            + list(social.parameters()),
            3.0,
        )
        optimizer.step()
        temporal.eval()
        flow.eval()
        social.eval()
        with torch.no_grad():
            self_pred, self_state = temporal(val.history)
            self_pred = torch.where(
                val.history_valid[:, None], self_pred, torch.zeros_like(self_pred)
            )
            self_state = torch.where(
                val.history_valid[:, None], self_state, torch.zeros_like(self_state)
            )
            flow_pred, flow_state = flow(val.flow)
            social_pred, _ = social(
                val, self_state, flow_state, variant=variant
            )
            score = float(
                masked_vector_mse(
                    self_pred + flow_pred + social_pred,
                    val.y_norm,
                    val.target_valid,
                    sequence_groups(val) if sequence_balanced_loss else None,
                )
            )
        if score < best_val - 1e-6:
            best_val, best_epoch = score, epoch + 1
            best = {
                "temporal": copy.deepcopy(temporal.state_dict()),
                "flow": copy.deepcopy(flow.state_dict()),
                "social": copy.deepcopy(social.state_dict()),
            }
        elif epoch + 1 - best_epoch >= 8:
            break
    temporal.load_state_dict(best["temporal"])
    flow.load_state_dict(best["flow"])
    social.load_state_dict(best["social"])
    temporal.eval()
    flow.eval()
    social.eval()
    return {"best_epoch": best_epoch, "best_val_norm_mse": best_val}


def to_px(pred_norm: torch.Tensor, norm: Normalizer) -> np.ndarray:
    return pred_norm.detach().cpu().numpy() * norm.target_std + norm.target_mean


@torch.no_grad()
def encode_base(
    temporal: TemporalSelfEncoder,
    flow: CoarseFlowEncoder,
    graph: GraphTensors,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    self_pred, self_state = temporal(graph.history)
    self_pred = torch.where(
        graph.history_valid[:, None], self_pred, torch.zeros_like(self_pred)
    )
    self_state = torch.where(
        graph.history_valid[:, None], self_state, torch.zeros_like(self_state)
    )
    flow_pred, flow_state = flow(graph.flow)
    return self_pred, self_state, flow_pred, flow_state


@torch.no_grad()
def evaluate(
    temporal: TemporalSelfEncoder,
    flow: CoarseFlowEncoder,
    social: StructuralInfluenceDecoder | None,
    graph: GraphTensors,
    norm: Normalizer,
    *,
    variant: str,
) -> tuple[np.ndarray, dict[str, float]]:
    self_pred, self_state, flow_pred, flow_state = encode_base(
        temporal, flow, graph
    )
    if social is None:
        delta = torch.zeros_like(self_pred)
        pred = self_pred if variant == "self_only" else self_pred + flow_pred
        diag: dict[str, torch.Tensor] = {}
    else:
        delta, diag = social(
            graph, self_state, flow_state, variant=variant
        )
        pred = self_pred + flow_pred + delta
    mask = graph.target_valid.detach().cpu().numpy()
    pred_px = to_px(pred, norm)
    y_px = graph.y_px.detach().cpu().numpy()
    metrics = vector_metrics(y_px[mask], pred_px[mask], 1)
    delta_px = delta.detach().cpu().numpy() * norm.target_std
    if social is not None:
        residual_px = y_px - to_px(self_pred + flow_pred, norm)
        finite = mask & np.isfinite(residual_px).all(axis=1)
        dot = np.sum(delta_px[finite] * residual_px[finite], axis=1)
        denom = np.maximum(
            np.linalg.norm(delta_px[finite], axis=1)
            * np.linalg.norm(residual_px[finite], axis=1),
            1e-8,
        )
        metrics.update(
            {
                "social_magnitude_mean_px": float(
                    np.mean(np.linalg.norm(delta_px[mask], axis=1))
                ),
                "social_magnitude_p90_px": float(
                    np.quantile(np.linalg.norm(delta_px[mask], axis=1), 0.9)
                ),
                "social_residual_cosine": float(np.mean(dot / denom)),
                "node_gate_mean": float(diag["node_gate"][graph.target_valid].mean().cpu()),
                "node_gate_zero_fraction": float(
                    (diag["node_gate"][graph.target_valid] == 0).float().mean().cpu()
                ),
                "node_gate_p90": float(
                    torch.quantile(
                        diag["node_gate"][graph.target_valid].reshape(-1), 0.9
                    ).cpu()
                ),
                "edge_gate_mean": float(diag["edge_gate"].mean().cpu()),
                "prior_strength": float(diag["prior_strength"].cpu()),
                "prior_amplitude_mean": float(diag["prior_amplitude"].mean().cpu()),
                "prior_amplitude_std": float(diag["prior_amplitude"].std().cpu()),
                "effective_degree_mean": float(
                    diag["effective_degree"][graph.target_valid].mean().cpu()
                ),
                "mean_mix_mean": float(diag["mix"][graph.target_valid, 0].mean().cpu()),
                "mobility_parallel_mean": float(
                    diag["mobility"][graph.target_valid, 0].mean().cpu()
                ),
                "mobility_perpendicular_mean": float(
                    diag["mobility"][graph.target_valid, 1].mean().cpu()
                ),
                **{
                    f"basis_coeff_abs_mean_{i}": float(
                        diag["coeff"][:, i].abs().mean().cpu()
                    )
                    for i in range(4)
                },
            }
        )
    else:
        metrics["social_magnitude_mean_px"] = 0.0
    return pred_px, metrics


def prepare_dataset(
    dataset: str,
    *,
    k: int,
    device: torch.device,
) -> tuple[dict[str, GraphTensors], Normalizer, dict[str, Any]]:
    raw = load_dataset(dataset)
    samples = build_causal_samples(raw)
    samples = add_visible_context_nodes(samples, raw)
    samples = add_causal_flow_features(samples, dataset)
    parts = {
        split: samples[samples["split"].eq(split)].copy().reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    norm = fit_normalizer(parts["train"])
    arrays = {
        split: build_graph_arrays(
            part,
            dataset,
            k=k,
            shuffle_seed=20260607 + split_id * 1009,
        )
        for split_id, (split, part) in enumerate(parts.items())
    }
    fit_graph_normalization(arrays["train"], norm)
    graphs = {
        split: graph_to_tensors(array, norm, device)
        for split, array in arrays.items()
    }
    coverage: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        valid = int(arrays[split].target_valid.sum())
        total = int(len(arrays[split].target_valid))
        coverage[f"{split}_causal_nodes"] = total
        coverage[f"{split}_target_nodes"] = valid
        coverage[f"{split}_context_only_nodes"] = total - valid
        coverage[f"{split}_historyless_context_nodes"] = int(
            (~arrays[split].history_valid).sum()
        )
        coverage[f"{split}_edges"] = int(len(arrays[split].src))
        if len(arrays[split].src):
            context_edge = (
                ~arrays[split].target_valid[arrays[split].src]
                | ~arrays[split].target_valid[arrays[split].dst]
            )
            coverage[f"{split}_edges_with_context_only_node"] = int(
                context_edge.sum()
            )
    return graphs, norm, coverage


def relative_gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0


def sequence_metric_fields(
    graph: GraphTensors,
    pred_px: np.ndarray,
) -> dict[str, float]:
    mask = graph.target_valid.detach().cpu().numpy()
    y = graph.y_px.detach().cpu().numpy()
    fields: dict[str, float] = {}
    for seq_id in np.unique(graph.seq_id):
        local = mask & (graph.seq_id == seq_id)
        if np.any(local):
            fields[f"rmse_seq{int(seq_id)}_px"] = vector_metrics(
                y[local], pred_px[local], 1
            )["rmse_px"]
    return fields


def paired_block_bootstrap(
    graph: GraphTensors,
    base_pred_px: np.ndarray,
    candidate_pred_px: np.ndarray,
    *,
    seed: int,
    repeats: int = 800,
    block_frames: int = 20,
) -> dict[str, float]:
    """Paired temporal-block bootstrap, stratified by sequence."""
    mask = graph.target_valid.detach().cpu().numpy()
    y = graph.y_px.detach().cpu().numpy()[mask]
    base = base_pred_px[mask]
    candidate = candidate_pred_px[mask]
    seq = graph.seq_id[mask]
    frame = graph.frame[mask]
    base_error = np.sum(np.square(base - y), axis=1)
    candidate_error = np.sum(np.square(candidate - y), axis=1)
    block = frame // block_frames
    rng = np.random.default_rng(seed)
    strata: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for seq_id in np.unique(seq):
        seq_mask = seq == seq_id
        block_ids = np.unique(block[seq_mask])
        base_sse, candidate_sse, counts = [], [], []
        for block_id in block_ids:
            local = seq_mask & (block == block_id)
            base_sse.append(base_error[local].sum())
            candidate_sse.append(candidate_error[local].sum())
            counts.append(local.sum())
        strata.append(
            (
                np.asarray(base_sse, dtype=float),
                np.asarray(candidate_sse, dtype=float),
                np.asarray(counts, dtype=float),
            )
        )
    gains = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        total_base, total_candidate, total_n = 0.0, 0.0, 0.0
        for base_sse, candidate_sse, counts in strata:
            sample = rng.integers(0, len(counts), size=len(counts))
            total_base += base_sse[sample].sum()
            total_candidate += candidate_sse[sample].sum()
            total_n += counts[sample].sum()
        base_rmse = math.sqrt(total_base / max(total_n, 1.0))
        candidate_rmse = math.sqrt(total_candidate / max(total_n, 1.0))
        gains[repeat] = relative_gain(base_rmse, candidate_rmse)
    observed = relative_gain(
        math.sqrt(base_error.mean()), math.sqrt(candidate_error.mean())
    )
    return {
        "paired_gain_over_flow_pct": observed,
        "block_bootstrap_gain_mean_pct": float(gains.mean()),
        "block_bootstrap_gain_ci_low_pct": float(np.quantile(gains, 0.025)),
        "block_bootstrap_gain_ci_high_pct": float(np.quantile(gains, 0.975)),
    }


def run_dataset(
    dataset: str,
    *,
    seeds: list[int],
    variants: list[str],
    device: torch.device,
    k: int,
    temporal_epochs: int,
    flow_epochs: int,
    social_epochs: int,
    joint_epochs: int,
    batch_size: int,
    joint_threshold_pct: float,
    sequence_balanced_loss: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    graphs, norm, coverage = prepare_dataset(dataset, k=k, device=device)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[{dataset}] seed={seed} stage A: temporal self", flush=True)
        temporal, temporal_info = train_temporal(
            graphs["train"],
            graphs["val"],
            seed=seed,
            epochs=temporal_epochs,
            batch_size=batch_size,
            sequence_balanced_loss=sequence_balanced_loss,
        )
        zero_flow = CoarseFlowEncoder(graphs["train"].flow.shape[1]).to(device)
        for parameter in zero_flow.parameters():
            nn.init.zeros_(parameter)
        self_pred_px, self_metrics = evaluate(
            temporal,
            zero_flow,
            None,
            graphs["test"],
            norm,
            variant="self_only",
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variant": "self_only",
                "joint_finetuned": False,
                **self_metrics,
                **sequence_metric_fields(graphs["test"], self_pred_px),
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )

        with torch.no_grad():
            train_self, _ = temporal(graphs["train"].history)
            val_self, _ = temporal(graphs["val"].history)
        print(f"[{dataset}] seed={seed} stage B: coarse flow", flush=True)
        flow, flow_info = train_flow(
            graphs["train"],
            graphs["val"],
            train_self.detach(),
            val_self.detach(),
            seed=seed,
            epochs=flow_epochs,
            batch_size=batch_size,
            sequence_balanced_loss=sequence_balanced_loss,
        )
        flow_pred_px, flow_metrics = evaluate(
            temporal,
            flow,
            None,
            graphs["test"],
            norm,
            variant="self_flow",
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variant": "self_flow",
                "joint_finetuned": False,
                **flow_metrics,
                **sequence_metric_fields(graphs["test"], flow_pred_px),
                **{f"flow_{k0}": v for k0, v in flow_info.items()},
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        encoded = {
            split: encode_base(temporal, flow, graph)
            for split, graph in graphs.items()
        }
        for variant in variants:
            print(f"[{dataset}] seed={seed} stage C: {variant}", flush=True)
            train_base = encoded["train"][0] + encoded["train"][2]
            val_base = encoded["val"][0] + encoded["val"][2]
            social, social_info = train_social(
                graphs["train"],
                graphs["val"],
                train_base.detach(),
                val_base.detach(),
                encoded["train"][1].detach(),
                encoded["val"][1].detach(),
                encoded["train"][3].detach(),
                encoded["val"][3].detach(),
                variant=variant,
                seed=seed,
                epochs=social_epochs,
                sequence_balanced_loss=sequence_balanced_loss,
            )
            base_val = masked_vector_mse(
                val_base,
                graphs["val"].y_norm,
                graphs["val"].target_valid,
                sequence_groups(graphs["val"]) if sequence_balanced_loss else None,
            ).item()
            social_val = float(social_info["best_val_norm_mse"])
            val_gain = relative_gain(base_val, social_val)
            joint_info: dict[str, float] = {}
            joint_used = val_gain >= joint_threshold_pct and joint_epochs > 0
            candidate_temporal = temporal
            candidate_flow = flow
            if joint_used:
                print(
                    f"[{dataset}] seed={seed} stage D: joint fine-tune "
                    f"(val gain {val_gain:.3f}%)",
                    flush=True,
                )
                candidate_temporal = copy.deepcopy(temporal)
                candidate_flow = copy.deepcopy(flow)
                joint_info = joint_finetune(
                    candidate_temporal,
                    candidate_flow,
                    social,
                    graphs["train"],
                    graphs["val"],
                    variant=variant,
                    epochs=joint_epochs,
                    sequence_balanced_loss=sequence_balanced_loss,
                )
            pred_px, metrics = evaluate(
                candidate_temporal,
                candidate_flow,
                social,
                graphs["test"],
                norm,
                variant=variant,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "joint_finetuned": joint_used,
                    "stage_c_val_gain_pct": val_gain,
                    **metrics,
                    **sequence_metric_fields(graphs["test"], pred_px),
                    **paired_block_bootstrap(
                        graphs["test"],
                        flow_pred_px,
                        pred_px,
                        seed=seed + 100_003,
                    ),
                    **{f"social_{k0}": v for k0, v in social_info.items()},
                    **{f"joint_{k0}": v for k0, v in joint_info.items()},
                    **{f"flow_{k0}": v for k0, v in flow_info.items()},
                    **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
                }
            )
            print(
                f"[{dataset}] seed={seed} {variant}: "
                f"test RMSE={metrics['rmse_px']:.6f}, "
                f"social={metrics['social_magnitude_mean_px']:.4f}px",
                flush=True,
            )
            if variant == "oz_structural":
                controls = {
                    "oz_inference_no_prior": "geometry_structural",
                    "oz_inference_shuffled": "oz_shuffled",
                    "oz_inference_sign_flipped": "oz_sign_flipped",
                }
                for control_name, intervention in controls.items():
                    control_pred_px, control_metrics = evaluate(
                        candidate_temporal,
                        candidate_flow,
                        social,
                        graphs["test"],
                        norm,
                        variant=intervention,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "variant": control_name,
                            "control_of": "oz_structural",
                            "joint_finetuned": joint_used,
                            **control_metrics,
                            **sequence_metric_fields(
                                graphs["test"], control_pred_px
                            ),
                            **paired_block_bootstrap(
                                graphs["test"],
                                flow_pred_px,
                                control_pred_px,
                                seed=seed + 200_003,
                            ),
                        }
                    )
        del temporal, flow, encoded
        if device.type == "mps":
            torch.mps.empty_cache()
    return pd.DataFrame(rows), coverage


def plot_results(summary: pd.DataFrame, out_dir: Path) -> None:
    order = [
        variant
        for variant in (
            "self_only",
            "self_flow",
            *VARIANTS,
            "oz_inference_no_prior",
            "oz_inference_shuffled",
            "oz_inference_sign_flipped",
        )
        if variant in set(summary["variant"])
    ]
    labels = {
        "self_only": "Self",
        "self_flow": "Self + flow",
        "geometry_structural": "+ structural geometry",
        "oz_structural": "+ OZ structural response",
        "oz_inference_no_prior": "OZ weights, prior off",
        "oz_inference_shuffled": "OZ weights, shuffled",
        "oz_inference_sign_flipped": "OZ weights, sign-flipped",
    }
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.2 * len(datasets), 4.2),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, dataset in zip(axes[0], datasets):
        part = summary[summary["dataset"].eq(dataset)]
        pivot = part.pivot(index="seed", columns="variant", values="rmse_px")
        base = pivot["self_flow"]
        gain = pd.DataFrame(
            {
                variant: (base - pivot[variant]) / base * 100.0
                for variant in order
                if variant != "self_only"
            }
        )
        variants = list(gain.columns)
        ax.bar(
            np.arange(len(variants)),
            gain.mean().to_numpy(),
            yerr=gain.std().fillna(0.0).to_numpy(),
            color="#567C8D",
            capsize=3,
        )
        ax.axhline(0.0, color="#303030", linewidth=0.8)
        ax.set_xticks(
            np.arange(len(variants)),
            [labels[v] for v in variants],
            rotation=22,
            ha="right",
        )
        ax.set_ylabel("Gain over self + flow (%)")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "fig01_full_architecture_gain.png", dpi=300)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    coverage: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    ablation_threshold_pct: float,
) -> None:
    means = (
        summary.groupby(["dataset", "variant"], as_index=False)
        .agg(
            rmse_px=("rmse_px", "mean"),
            rmse_std=("rmse_px", "std"),
            r2_vec=("r2_vec", "mean"),
            social_px=("social_magnitude_mean_px", "mean"),
            residual_cos=("social_residual_cosine", "mean"),
            node_gate=("node_gate_mean", "mean"),
        )
        .fillna(0.0)
    )
    lines = [
        "# Full causal OZ architecture study",
        "",
        "Архитектура разделяет causal temporal self, coarse flow и локальный "
        "structural social response. Состав графа в момент t больше не зависит "
        "от наличия таргета t+1: context-only клетки входят в message passing, "
        "а target mask применяется только к loss и метрикам.",
        "",
        "## Покрытие и причинность",
        "",
        "```text",
    ]
    for dataset, values in coverage.items():
        lines.append(f"{dataset}: {json.dumps(values, sort_keys=True)}")
    lines.extend(["```", "", "## Средние test-результаты", "", "```text"])
    lines.append(means.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    lines.extend(["```", "", "## Решение по абляциям", ""])
    for dataset in summary["dataset"].unique():
        pivot = summary[summary["dataset"].eq(dataset)].pivot(
            index="seed", columns="variant", values="rmse_px"
        )
        for variant in [
            v for v in ("geometry_structural", "oz_structural") if v in pivot
        ]:
            gains = (pivot["self_flow"] - pivot[variant]) / pivot["self_flow"] * 100.0
            required_positive = math.ceil(0.8 * len(gains))
            positive_seeds = int((gains > 0).sum())
            candidate_rows = summary[
                summary["dataset"].eq(dataset)
                & summary["variant"].eq(variant)
            ].set_index("seed")
            sequence_gains: list[float] = []
            for seq_id in (0, 1):
                column = f"rmse_seq{seq_id}_px"
                if column not in summary.columns:
                    continue
                base_seq = (
                    summary[
                        summary["dataset"].eq(dataset)
                        & summary["variant"].eq("self_flow")
                    ]
                    .set_index("seed")[column]
                    .reindex(gains.index)
                )
                candidate_seq = candidate_rows[column].reindex(gains.index)
                sequence_gains.append(
                    float(((base_seq - candidate_seq) / base_seq * 100.0).mean())
                )
            bootstrap_positive = int(
                (
                    candidate_rows["block_bootstrap_gain_ci_low_pct"]
                    .reindex(gains.index)
                    .fillna(-np.inf)
                    > 0
                ).sum()
            )
            stable = positive_seeds >= required_positive
            sequence_ok = bool(sequence_gains) and min(sequence_gains) >= 0.10
            bootstrap_ok = bootstrap_positive >= required_positive
            eligible = (
                float(gains.mean()) >= ablation_threshold_pct
                and stable
                and sequence_ok
                and bootstrap_ok
            )
            lines.append(
                f"- {dataset} {variant}: {gains.mean():.4f}% mean gain, "
                f"{positive_seeds}/{len(gains)} positive seeds, "
                f"sequence gains {sequence_gains}, "
                f"{bootstrap_positive}/{len(gains)} block-bootstrap CI>0; "
                f"full ablations {'разрешены' if eligible else 'не запускаются'}."
            )
    lines.extend(
        [
            "",
            "## Интерпретируемые ограничения",
            "",
            "1. Structural encoder хранит sum/mean/variance/max и directed moments; "
            "degree не стирается softmax-нормировкой.",
            "2. Context-aware edge gate получает состояния обоих узлов и их "
            "многотельное окружение.",
            "3. Decoder ограничен radial, closing, relative-velocity и shear "
            "базисами. Свободного 2D edge-вектора нет.",
            "4. OZ zero сохраняется при нормализации: c(r) и -dc/dr масштабируются "
            "train-only стандартным отклонением без вычитания среднего.",
            "5. Joint fine-tuning включается только после заметного validation "
            "gain, чтобы шумовой social branch не портил self/flow.",
        ]
    )
    (out_dir / "full_architecture_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=["HSC", "PSC"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "geometry_structural",
            "oz_structural",
            "oz_shuffled",
            "oz_sign_flipped",
        ],
        default=list(VARIANTS),
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temporal-epochs", type=int, default=70)
    parser.add_argument("--flow-epochs", type=int, default=60)
    parser.add_argument("--social-epochs", type=int, default=90)
    parser.add_argument("--joint-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--joint-threshold-pct", type=float, default=0.20)
    parser.add_argument("--ablation-threshold-pct", type=float, default=0.20)
    parser.add_argument(
        "--sequence-balanced-loss",
        action="store_true",
        help="Average training and validation loss per sequence before averaging.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets = args.datasets[:1]
        args.seeds = args.seeds[:1]
        args.temporal_epochs = min(args.temporal_epochs, 3)
        args.flow_epochs = min(args.flow_epochs, 3)
        args.social_epochs = min(args.social_epochs, 3)
        args.joint_epochs = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"device={device}", flush=True)
    all_rows: list[pd.DataFrame] = []
    coverage: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        summary, dataset_coverage = run_dataset(
            dataset,
            seeds=args.seeds,
            variants=args.variants,
            device=device,
            k=args.k,
            temporal_epochs=args.temporal_epochs,
            flow_epochs=args.flow_epochs,
            social_epochs=args.social_epochs,
            joint_epochs=args.joint_epochs,
            batch_size=args.batch_size,
            joint_threshold_pct=args.joint_threshold_pct,
            sequence_balanced_loss=args.sequence_balanced_loss,
        )
        summary.to_csv(
            args.out_dir / f"full_architecture_summary_{dataset.lower()}.csv",
            index=False,
        )
        all_rows.append(summary)
        coverage[dataset] = dataset_coverage
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(args.out_dir / "full_architecture_summary.csv", index=False)
    (args.out_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_results(combined, args.out_dir)
    write_report(
        combined,
        coverage,
        args.out_dir,
        ablation_threshold_pct=args.ablation_threshold_pct,
    )
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
