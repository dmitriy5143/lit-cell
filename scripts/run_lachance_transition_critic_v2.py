#!/usr/bin/env python3
"""Offline learned transition critic v2 for LaChance candidate trajectories.

This runner intentionally stays outside the final forecasting backbone.  It
reuses the causal candidate generator/backbone from
``run_lachance_candidate_oracle.py`` and asks a narrower question:

Can a calibrated learned critic rank or mix candidate future trajectories
better than dynamic-only, no-physics, OZ-only and shuffled controls?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts import run_lachance_architecture_study as la  # noqa: E402
from scripts import run_lachance_candidate_oracle as co  # noqa: E402
from scripts import run_lachance_nextgen_message_passing as ng  # noqa: E402

arch = la.arch

EPS = 1e-6
BASELINE_FEATURE_SET = "baseline_config_backward"
DEFAULT_FEATURE_SETS = [
    "full",
    "no_physics",
    "dynamic_only",
    "no_structural",
    "no_density_pressure",
    "no_topology",
    "no_backward",
    "no_soft_neighbour",
    "oz_only",
    "shuffled_state",
    "time_shuffled",
]
ALL_FEATURE_SETS = [BASELINE_FEATURE_SET, *DEFAULT_FEATURE_SETS]


@dataclass
class FeaturePacket:
    values: np.ndarray
    names: list[str]
    groups: list[str]


@dataclass
class CriticTrainResult:
    model: torch.nn.Module
    temp: float
    best_val_rmse: float
    best_epoch: int
    train_loss: float


def parse_csv(value: str | None, default: Iterable[str | int]) -> list[str]:
    if value is None or str(value).strip() == "":
        return [str(x) for x in default]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value: str | None, default: Iterable[int]) -> list[int]:
    return [int(x) for x in parse_csv(value, default)]


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def unit_np(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def cos_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(unit_np(a) * unit_np(b), axis=-1)


def vector_rmse_arrays(pred_px: np.ndarray, y_px: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(pred_px - y_px), axis=1))))


def mean_cosine_arrays(pred_px: np.ndarray, y_px: np.ndarray) -> float:
    denom = np.maximum(
        np.linalg.norm(pred_px, axis=1) * np.linalg.norm(y_px, axis=1),
        EPS,
    )
    return float(np.mean(np.sum(pred_px * y_px, axis=1) / denom))


def magnitude_ratio_arrays(pred_px: np.ndarray, y_px: np.ndarray) -> float:
    return float(
        np.mean(np.linalg.norm(pred_px, axis=1))
        / max(float(np.mean(np.linalg.norm(y_px, axis=1))), EPS)
    )


def argmin_labels_from_arrays(candidates_px: np.ndarray, target_px: np.ndarray) -> np.ndarray:
    err = np.sum(np.square(candidates_px - target_px[:, None, :]), axis=2)
    return np.argmin(err, axis=1).astype(np.int64)


def tune_temperature_arrays(
    candidates_px: np.ndarray,
    target_px: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    best_temp = 1.0
    best_rmse = float("inf")
    for temp in (0.15, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        shifted = (scores - scores.max(axis=1, keepdims=True)) / max(float(temp), 1e-3)
        weights = np.exp(np.clip(shifted, -40.0, 20.0))
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)
        pred = np.sum(weights[:, :, None] * candidates_px, axis=1)
        rmse = vector_rmse_arrays(pred, target_px)
        if rmse < best_rmse:
            best_rmse = rmse
            best_temp = float(temp)
    return best_temp, best_rmse


def add_named(
    parts: list[np.ndarray],
    names: list[str],
    groups: list[str],
    arr: np.ndarray,
    prefix: str,
    group: str,
    feature_names: list[str] | None = None,
) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Feature block {prefix!r} must be [N,K,D], got {arr.shape}")
    parts.append(arr)
    width = arr.shape[-1]
    if feature_names is None:
        feature_names = [f"{prefix}_{i:02d}" for i in range(width)]
    if len(feature_names) != width:
        raise ValueError(f"{prefix}: expected {width} names, got {len(feature_names)}")
    names.extend([f"{prefix}__{name}" for name in feature_names])
    groups.extend([group] * width)


def node_state_arrays(graph: la.LachanceGraphData) -> dict[str, np.ndarray]:
    quality = to_numpy(graph.quality).reshape(-1).astype(np.float32)
    speed = to_numpy(graph.speed_norm).reshape(-1).astype(np.float32)
    degree = to_numpy(graph.degree).reshape(-1).astype(np.float32)
    pos = to_numpy(graph.current_pos_px).astype(np.float32)
    center = pos.mean(axis=0, keepdims=True)
    rel = pos - center
    radius = np.linalg.norm(rel, axis=1)
    max_radius = float(np.percentile(radius, 95) + EPS)
    boundary = np.clip(radius / max_radius, 0.0, 2.0).astype(np.float32)
    return {
        "quality": quality,
        "speed": speed,
        "degree": degree,
        "boundary": boundary,
    }


def build_named_base_features(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    family_to_idx: dict[str, int],
    scale_px: float,
) -> FeaturePacket:
    """Causal candidate/statistical features with explicit groups for masks."""

    parts: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    cand = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2).astype(np.float32)
    proposal = split.proposal_px[node_idx].astype(np.float32)
    self_flow = split.self_flow_px[node_idx].astype(np.float32)
    self_only = split.self_px[node_idx].astype(np.float32)
    n, k, _ = cand.shape

    cand_norm = cand / float(scale_px)
    proposal_rep = proposal[:, None, :] / float(scale_px)
    self_flow_rep = self_flow[:, None, :] / float(scale_px)
    self_rep = self_only[:, None, :] / float(scale_px)
    offset_prop = cand_norm - proposal_rep
    offset_flow = cand_norm - self_flow_rep
    offset_self = cand_norm - self_rep

    add_named(
        parts,
        names,
        groups,
        cand_norm,
        "candidate",
        "geometry",
        ["dx_norm", "dy_norm"],
    )
    add_named(
        parts,
        names,
        groups,
        np.linalg.norm(cand, axis=-1, keepdims=True) / float(scale_px),
        "candidate",
        "geometry",
        ["mag_norm"],
    )
    add_named(
        parts,
        names,
        groups,
        offset_prop,
        "proposal_offset",
        "geometry",
        ["dx", "dy"],
    )
    add_named(
        parts,
        names,
        groups,
        np.linalg.norm(offset_prop, axis=-1, keepdims=True),
        "proposal_offset",
        "geometry",
        ["mag"],
    )
    add_named(
        parts,
        names,
        groups,
        offset_flow,
        "self_flow_offset",
        "geometry",
        ["dx", "dy"],
    )
    add_named(
        parts,
        names,
        groups,
        offset_self,
        "self_offset",
        "geometry",
        ["dx", "dy"],
    )
    add_named(
        parts,
        names,
        groups,
        cos_np(cand, proposal[:, None, :])[..., None],
        "candidate_stats",
        "candidate_stats",
        ["cos_to_proposal"],
    )
    add_named(
        parts,
        names,
        groups,
        cos_np(cand, self_flow[:, None, :])[..., None],
        "candidate_stats",
        "candidate_stats",
        ["cos_to_self_flow"],
    )
    add_named(
        parts,
        names,
        groups,
        cos_np(cand, self_only[:, None, :])[..., None],
        "candidate_stats",
        "candidate_stats",
        ["cos_to_self"],
    )
    add_named(
        parts,
        names,
        groups,
        (
            np.linalg.norm(cand, axis=-1)
            / (np.linalg.norm(proposal, axis=-1)[:, None] + EPS)
        )[..., None],
        "candidate_stats",
        "candidate_stats",
        ["mag_ratio_to_proposal"],
    )

    field_groups = {
        "force": "structural",
        "c_radial": "structural",
        "rel_velocity": "dynamic",
        "shear": "dynamic",
        "closing": "dynamic",
    }
    for field_name, group in field_groups.items():
        if field_name not in split.fields:
            continue
        field = split.fields[field_name][node_idx].astype(np.float32)
        field_rep = field[:, None, :]
        field_norm = np.linalg.norm(field, axis=-1, keepdims=True)
        add_named(
            parts,
            names,
            groups,
            np.repeat(field_rep / float(scale_px), k, axis=1),
            f"field_{field_name}",
            group,
            ["dx_norm", "dy_norm"],
        )
        add_named(
            parts,
            names,
            groups,
            np.repeat(field_norm[:, None, :] / float(scale_px), k, axis=1),
            f"field_{field_name}",
            group,
            ["mag_norm"],
        )
        add_named(
            parts,
            names,
            groups,
            cos_np(offset_prop, field_rep / float(scale_px))[..., None],
            f"field_{field_name}",
            group,
            ["cos_to_candidate_offset"],
        )

    state = node_state_arrays(split.graph)
    state_block = np.stack(
        [
            state["quality"][node_idx],
            state["speed"][node_idx],
            state["degree"][node_idx],
            state["boundary"][node_idx],
        ],
        axis=-1,
    ).astype(np.float32)
    state_block = np.repeat(state_block[:, None, :], k, axis=1)
    add_named(
        parts,
        names,
        groups,
        state_block,
        "state",
        "state",
        ["quality", "speed", "degree", "boundary"],
    )

    onehot = np.zeros((n, k, len(family_to_idx)), dtype=np.float32)
    for cand_i, family in enumerate(split.pack.families):
        if family in family_to_idx:
            onehot[:, cand_i, family_to_idx[family]] = 1.0
    if onehot.shape[-1] > 0:
        ordered_families = [None] * len(family_to_idx)
        for family, idx in family_to_idx.items():
            ordered_families[idx] = family
        add_named(
            parts,
            names,
            groups,
            onehot,
            "family",
            "family",
            [str(family).replace("/", "_") for family in ordered_families],
        )

    return FeaturePacket(np.concatenate(parts, axis=-1), names, groups)


def add_legacy_config_features(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    scale_px: float,
    parts: list[np.ndarray],
    names: list[str],
    groups: list[str],
) -> None:
    config = co.configuration_candidate_features(
        split,
        node_idx,
        scale_px=scale_px,
        include_joint=False,
    ).astype(np.float32)
    # Existing order from run_lachance_candidate_oracle:
    # future_dist-current_dist, abs(stretch), deltaE, c_future, collision,
    # log distance ratio, candidate-neighbour motion mismatch, motion cosine.
    mapping = [
        ("config_distance", "density_pressure", ["future_minus_current_dist_norm"]),
        ("config_distance", "density_pressure", ["abs_stretch_norm"]),
        ("config_structural", "structural", ["delta_c_energy"]),
        ("config_structural", "structural", ["future_c"]),
        ("config_contact", "density_pressure", ["collision_pressure"]),
        ("config_contact", "density_pressure", ["log_dist_ratio"]),
        ("config_motion", "alignment", ["candidate_neighbour_motion_mismatch"]),
        ("config_motion", "alignment", ["candidate_neighbour_motion_cos"]),
    ]
    for idx, (prefix, group, feature_names) in enumerate(mapping):
        add_named(parts, names, groups, config[..., idx : idx + 1], prefix, group, feature_names)


def add_backward_features(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    scale_px: float,
    parts: list[np.ndarray],
    names: list[str],
    groups: list[str],
) -> None:
    back = co.backward_consistency_features(
        split,
        node_idx,
        scale_px=scale_px,
        include_joint=False,
    ).astype(np.float32)
    add_named(
        parts,
        names,
        groups,
        back,
        "backward_consistency",
        "backward",
        [
            "reverse_offset_mag",
            "reverse_offset_dx",
            "reverse_offset_dy",
            "reverse_cos_self",
            "reverse_cos_flow",
            "reverse_cos_proposal",
        ],
    )


def add_soft_neighbour_features(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    scale_px: float,
    parts: list[np.ndarray],
    names: list[str],
    groups: list[str],
) -> None:
    soft = co.soft_neighbour_route_features(
        split,
        node_idx,
        scale_px=scale_px,
        source_temperature=0.35,
        source_topk=8,
        route_temperature=0.50,
    ).astype(np.float32)
    add_named(
        parts,
        names,
        groups,
        soft,
        "soft_neighbour_route",
        "soft_neighbour",
        [
            "mean_dx",
            "mean_dy",
            "offset_mean_dx",
            "offset_mean_dy",
            "offset_mean_mag",
            "cos_to_mean",
            "top1_dx",
            "top1_dy",
            "offset_top1_dx",
            "offset_top1_dy",
            "offset_top1_mag",
            "cos_to_top1",
            "family_entropy",
            "soft_weight_sum",
        ],
    )


def active_transition_features(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    scale_px: float,
) -> FeaturePacket:
    """Conditional active-physics features for each node/candidate.

    The block is deliberately local and causal: neighbours are at t, candidate
    futures come from the generator, and neighbour future motion is approximated
    by the causal proposal/self-flow fields.  No target future is used.
    """

    graph = split.graph
    cand = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2).astype(np.float32)
    proposal_all = split.proposal_px.astype(np.float32)
    proposal = proposal_all[node_idx]
    self_flow = split.self_flow_px[node_idx].astype(np.float32)
    pos = to_numpy(graph.current_pos_px).astype(np.float32)
    velocity = to_numpy(graph.current_velocity).astype(np.float32)
    n, k, _ = cand.shape

    parts: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    local = np.full(pos.shape[0], -1, dtype=np.int64)
    local[node_idx] = np.arange(len(node_idx), dtype=np.int64)
    src_all = to_numpy(graph.src).astype(np.int64)
    dst_all = to_numpy(graph.dst).astype(np.int64)
    local_dst = local[dst_all]
    edge_mask = local_dst >= 0

    if not np.any(edge_mask):
        zeros = np.zeros((n, k, 1), dtype=np.float32)
        for prefix, group in [
            ("active_density_pressure", "density_pressure"),
            ("active_topology", "topology"),
            ("active_boundary_flow", "boundary"),
            ("active_polarity", "polarity"),
            ("active_alignment", "alignment"),
            ("active_structural", "structural"),
        ]:
            add_named(parts, names, groups, zeros, prefix, group, ["missing_graph"])
        return FeaturePacket(np.concatenate(parts, axis=-1), names, groups)

    src = src_all[edge_mask]
    dst = dst_all[edge_mask]
    ldst = local_dst[edge_mask]
    rel = pos[src] - pos[dst]
    dist = np.linalg.norm(rel, axis=-1) + EPS
    radial = rel / dist[:, None]
    tangent = np.stack([-radial[:, 1], radial[:, 0]], axis=-1)
    proposal_src = proposal_all[src]
    vel_src = velocity[src]
    vel_dst = velocity[dst]
    edge_density = to_numpy(graph.degree).reshape(-1).astype(np.float32)[dst]

    # Candidate-specific future relative geometry. Shape: [E,K,2].
    cand_dst = cand[ldst]
    future_rel = pos[src, None, :] + proposal_src[:, None, :] - (pos[dst, None, :] + cand_dst)
    future_dist = np.linalg.norm(future_rel, axis=-1) + EPS
    dr = future_dist - dist[:, None]
    rel_motion = proposal_src[:, None, :] - cand_dst
    relrad = np.sum(rel_motion * radial[:, None, :], axis=-1)
    reltan = np.sum(rel_motion * tangent[:, None, :], axis=-1)
    cand_unit = unit_np(cand_dst)
    proposal_unit = unit_np(proposal_src[:, None, :])
    vel_dst_unit = unit_np(vel_dst)
    vel_src_unit = unit_np(vel_src)

    r_cut = float(np.percentile(dist, 80) + EPS)
    density_weight = 1.0 / (1.0 + dist / r_cut)
    denom = co.scatter_sum_scalar_np(density_weight.astype(np.float32), ldst, n)[:, None] + EPS

    def aggregate(edge_values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        edge_values = np.asarray(edge_values, dtype=np.float32)
        if weights is None:
            weights_local = density_weight.astype(np.float32)[:, None]
        else:
            weights_local = np.asarray(weights, dtype=np.float32)
            if weights_local.ndim == 1:
                weights_local = weights_local[:, None]
        if edge_values.ndim == 1:
            edge_values = edge_values[:, None]
        if edge_values.ndim == 2:
            out = np.zeros((n, edge_values.shape[-1]), dtype=np.float32)
            for col in range(edge_values.shape[-1]):
                out[:, col] = co.scatter_sum_scalar_np(
                    (edge_values[:, col] * weights_local[:, 0]).astype(np.float32),
                    ldst,
                    n,
                )
            return out / denom
        out = np.zeros((n, k, edge_values.shape[-1]), dtype=np.float32)
        for cand_i in range(k):
            for col in range(edge_values.shape[-1]):
                out[:, cand_i, col] = co.scatter_sum_scalar_np(
                    (edge_values[:, cand_i, col] * weights_local[:, 0]).astype(np.float32),
                    ldst,
                    n,
                )
        return out / denom[:, None, :]

    def aggregate_scalar(edge_values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        return aggregate(edge_values[..., None], weights=weights)

    current_density = aggregate(edge_density)[:, 0]
    current_density_rep = np.repeat(current_density[:, None, None], k, axis=1)
    persistence = (future_dist <= r_cut).astype(np.float32)
    persistence_ag = aggregate_scalar(persistence)[:, :, 0]
    degree_change = np.abs(persistence_ag - 1.0)
    collision = np.exp(-0.5 * (future_dist / max(0.33 * r_cut, EPS)) ** 2)
    collision_ag = aggregate_scalar(collision)[:, :, 0]
    stretch_abs_ag = aggregate_scalar(np.abs(dr) / float(scale_px))[:, :, 0]
    compression_ag = aggregate_scalar(np.maximum(-relrad, 0.0) / float(scale_px))[:, :, 0]
    shear_ag = aggregate_scalar(np.abs(reltan) / float(scale_px))[:, :, 0]
    density_pressure_score = (
        -collision_ag - stretch_abs_ag - 0.5 * compression_ag - 0.25 * shear_ag
    )
    add_named(
        parts,
        names,
        groups,
        np.stack(
            [
                current_density_rep[:, :, 0],
                persistence_ag,
                degree_change,
                collision_ag,
                stretch_abs_ag,
                compression_ag,
                shear_ag,
                density_pressure_score,
            ],
            axis=-1,
        ),
        "active_density_pressure",
        "density_pressure",
        [
            "local_density",
            "future_edge_persistence_proxy",
            "degree_change_abs",
            "collision_pressure",
            "stretch_abs",
            "compression_release",
            "shear_abs",
            "score",
        ],
    )

    topology_score = persistence_ag - degree_change
    add_named(
        parts,
        names,
        groups,
        np.stack([persistence_ag, 1.0 - persistence_ag, -degree_change, topology_score], axis=-1),
        "active_topology",
        "topology",
        ["persistence", "turnover", "neg_degree_change", "score"],
    )

    center = pos.mean(axis=0, keepdims=True)
    rel_center = pos[node_idx] - center
    outward = unit_np(rel_center)
    tangent_node = np.stack([-outward[:, 1], outward[:, 0]], axis=-1)
    boundary_state = node_state_arrays(graph)["boundary"][node_idx]
    boundary_future = np.linalg.norm(pos[node_idx][:, None, :] + cand - center, axis=-1)
    boundary_future = boundary_future / (np.percentile(np.linalg.norm(pos - center, axis=1), 95) + EPS)
    outward_align = cos_np(cand, outward[:, None, :])
    tangent_align = np.sum(unit_np(cand) * tangent_node[:, None, :], axis=-1)
    radial_escape = boundary_future - boundary_state[:, None]
    boundary_score = outward_align * boundary_state[:, None] - np.abs(radial_escape)
    add_named(
        parts,
        names,
        groups,
        np.stack(
            [
                np.repeat(boundary_state[:, None], k, axis=1),
                boundary_future,
                outward_align,
                tangent_align,
                radial_escape,
                boundary_score,
            ],
            axis=-1,
        ),
        "active_boundary_flow",
        "boundary",
        [
            "boundary_current",
            "boundary_future",
            "outward_align",
            "tangent_align",
            "radial_escape",
            "score",
        ],
    )

    front = np.sum(radial * vel_dst_unit, axis=-1)
    front_ag = aggregate_scalar(front[:, None] * np.ones((len(src), k), dtype=np.float32))[:, :, 0]
    pair_motion_cos = np.sum(cand_unit * proposal_unit, axis=-1)
    pair_motion_cos_ag = aggregate_scalar(pair_motion_cos)[:, :, 0]
    neighbour_follow = np.sum(cand_unit * vel_src_unit[:, None, :], axis=-1)
    neighbour_follow_ag = aggregate_scalar(neighbour_follow)[:, :, 0]
    self_persistence = cos_np(cand, velocity[node_idx, None, :])
    front_drive = front_ag * self_persistence
    polarity_score = 0.4 * self_persistence + 0.3 * pair_motion_cos_ag + 0.3 * front_drive
    add_named(
        parts,
        names,
        groups,
        np.stack(
            [
                self_persistence,
                neighbour_follow_ag,
                pair_motion_cos_ag,
                front_ag,
                front_drive,
                polarity_score,
            ],
            axis=-1,
        ),
        "active_polarity",
        "polarity",
        [
            "self_persistence",
            "neighbour_follow",
            "pair_motion_cos",
            "frontness",
            "front_drive",
            "score",
        ],
    )

    local_flow = split.fields.get("rel_velocity", split.proposal_px)[node_idx].astype(np.float32)
    local_flow_align = cos_np(cand, local_flow[:, None, :])
    closing_consistency = -aggregate_scalar(np.abs(relrad) / float(scale_px))[:, :, 0]
    velocity_alignment = np.sum(vel_dst_unit[:, None, :] * vel_src_unit[:, None, :], axis=-1)
    velocity_alignment_ag = aggregate_scalar(
        velocity_alignment * np.ones((len(src), k), dtype=np.float32)
    )[:, :, 0]
    alignment_score = 0.5 * local_flow_align + 0.3 * velocity_alignment_ag + 0.2 * closing_consistency
    add_named(
        parts,
        names,
        groups,
        np.stack(
            [local_flow_align, velocity_alignment_ag, closing_consistency, alignment_score],
            axis=-1,
        ),
        "active_alignment",
        "alignment",
        ["local_flow_align", "velocity_alignment", "closing_consistency", "score"],
    )

    current_c = np.exp(-dist / max(float(scale_px) * 0.35, EPS)) * np.cos(dist / max(scale_px, EPS))
    future_c = np.exp(-future_dist / max(float(scale_px) * 0.35, EPS)) * np.cos(
        future_dist / max(scale_px, EPS)
    )
    delta_c = current_c[:, None] - future_c
    delta_c_ag = aggregate_scalar(delta_c)[:, :, 0]
    future_c_ag = aggregate_scalar(future_c)[:, :, 0]
    c_radial = split.fields.get("c_radial", np.zeros_like(split.proposal_px))[node_idx].astype(np.float32)
    c_radial_align = cos_np(cand, c_radial[:, None, :])
    structural_score = delta_c_ag + 0.2 * c_radial_align
    add_named(
        parts,
        names,
        groups,
        np.stack([delta_c_ag, future_c_ag, c_radial_align, structural_score], axis=-1),
        "active_structural",
        "structural",
        ["delta_c", "future_c", "c_radial_align", "score"],
    )

    return FeaturePacket(np.concatenate(parts, axis=-1), names, groups)


def build_feature_packet(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    family_to_idx: dict[str, int],
    scale_px: float,
) -> FeaturePacket:
    base = build_named_base_features(split, node_idx, family_to_idx, scale_px)
    parts = [base.values]
    names = list(base.names)
    groups = list(base.groups)
    add_legacy_config_features(split, node_idx, scale_px, parts, names, groups)
    add_backward_features(split, node_idx, scale_px, parts, names, groups)
    add_soft_neighbour_features(split, node_idx, scale_px, parts, names, groups)
    active = active_transition_features(split, node_idx, scale_px)
    parts.append(active.values)
    names.extend(active.names)
    groups.extend(active.groups)
    return FeaturePacket(np.concatenate(parts, axis=-1).astype(np.float32), names, groups)


def feature_mask(groups: list[str], names: list[str], feature_set: str) -> np.ndarray:
    groups_arr = np.asarray(groups)
    names_arr = np.asarray(names)
    keep = np.ones(len(groups), dtype=bool)

    def only(allowed: set[str]) -> np.ndarray:
        return np.asarray([group in allowed for group in groups], dtype=bool)

    if feature_set == "full":
        keep[:] = True
    elif feature_set == BASELINE_FEATURE_SET:
        keep = only(
            {
                "geometry",
                "candidate_stats",
                "state",
                "family",
                "dynamic",
                "structural",
                "density_pressure",
                "alignment",
                "backward",
            }
        )
        # Keep legacy config/backward, but drop explicit v2 active packets.
        keep &= ~np.char.startswith(names_arr.astype(str), "active_")
    elif feature_set == "no_physics":
        keep = only({"geometry", "candidate_stats", "state", "family"})
    elif feature_set == "dynamic_only":
        keep = only({"geometry", "candidate_stats", "state", "family", "dynamic"})
    elif feature_set == "no_structural":
        keep = groups_arr != "structural"
    elif feature_set == "no_density_pressure":
        keep = groups_arr != "density_pressure"
    elif feature_set == "no_topology":
        keep = groups_arr != "topology"
    elif feature_set == "no_backward":
        keep = groups_arr != "backward"
    elif feature_set == "no_soft_neighbour":
        keep = groups_arr != "soft_neighbour"
    elif feature_set == "oz_only":
        keep = only({"geometry", "candidate_stats", "state", "family", "structural"})
    elif feature_set in {"shuffled_state", "time_shuffled"}:
        keep[:] = True
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if int(keep.sum()) == 0:
        raise ValueError(f"Feature set {feature_set} selected zero features")
    return keep


def apply_feature_control(
    x: np.ndarray,
    groups: list[str],
    feature_set: str,
    seed: int,
) -> np.ndarray:
    if feature_set not in {"shuffled_state", "time_shuffled"}:
        return x
    rng = np.random.default_rng(seed)
    out = x.copy()
    groups_arr = np.asarray(groups)
    if feature_set == "shuffled_state":
        cols = np.where(groups_arr == "state")[0]
    else:
        cols = np.where(
            ~np.isin(groups_arr, np.asarray(["geometry", "candidate_stats", "state", "family"]))
        )[0]
    if len(cols) == 0:
        return out
    flat = out.reshape(-1, out.shape[-1])
    for col in cols:
        flat[:, col] = flat[rng.permutation(flat.shape[0]), col]
    return flat.reshape(out.shape)


def standardize_train_val_test(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-6
    train_z = (train_x - mean) / std
    val_z = (val_x - mean) / std
    test_z = (test_x - mean) / std
    return (
        train_z.astype(np.float32),
        val_z.astype(np.float32),
        test_z.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def make_model(model_name: str, feature_dim: int, hidden_dim: int) -> torch.nn.Module:
    if model_name == "mlp":
        return co.CandidateReranker(feature_dim=feature_dim, hidden_dim=hidden_dim)
    if model_name == "set_light":
        return co.CandidateSetReranker(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            layers=1,
            heads=4,
            dropout=0.10,
        )
    raise ValueError(f"Unknown critic v2 model: {model_name}")


def calibrated_loss(
    logits: torch.Tensor,
    candidate_px: torch.Tensor,
    target_px: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: torch.Tensor,
    scale_px: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    weights = torch.softmax(logits / temperature, dim=1)
    pred = torch.sum(weights.unsqueeze(-1) * candidate_px, dim=1)
    mse = torch.mean((pred - target_px) ** 2) / (float(scale_px) ** 2)
    log_prob = F.log_softmax(logits, dim=1)
    ce = torch.mean(torch.sum(-soft_targets * log_prob, dim=1))
    pos = logits.gather(1, labels[:, None]).squeeze(1)
    neg = logits.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), -1e9).max(dim=1).values
    rank = torch.relu(0.20 - pos + neg).mean()
    return mse + 0.20 * ce + 0.05 * rank


def train_critic_v2(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_candidates_px: np.ndarray,
    val_candidates_px: np.ndarray,
    train_target_px: np.ndarray,
    val_target_px: np.ndarray,
    model_name: str,
    device: torch.device,
    scale_px: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    seed: int,
) -> CriticTrainResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(model_name, train_x.shape[-1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)

    train_labels = argmin_labels_from_arrays(train_candidates_px, train_target_px)
    train_soft = co.candidate_soft_oracle_targets(
        train_candidates_px / float(scale_px),
        train_target_px / float(scale_px),
        temperature=0.12,
        topk=min(5, train_candidates_px.shape[1]),
    )[0]
    val_labels = argmin_labels_from_arrays(val_candidates_px, val_target_px)
    val_soft = co.candidate_soft_oracle_targets(
        val_candidates_px / float(scale_px),
        val_target_px / float(scale_px),
        temperature=0.12,
        topk=min(5, val_candidates_px.shape[1]),
    )[0]

    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    tc = torch.tensor(train_candidates_px, dtype=torch.float32, device=device)
    tt = torch.tensor(train_target_px, dtype=torch.float32, device=device)
    tl = torch.tensor(train_labels, dtype=torch.long, device=device)
    ts = torch.tensor(train_soft, dtype=torch.float32, device=device)

    vx = torch.tensor(val_x, dtype=torch.float32, device=device)
    vc = torch.tensor(val_candidates_px, dtype=torch.float32, device=device)
    vt = torch.tensor(val_target_px, dtype=torch.float32, device=device)
    vl = torch.tensor(val_labels, dtype=torch.long, device=device)
    vs = torch.tensor(val_soft, dtype=torch.float32, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    final_loss = float("nan")
    patience = 8
    stale = 0

    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_x.shape[0])
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = model(tx[batch])
            loss = calibrated_loss(
                logits,
                tc[batch],
                tt[batch],
                tl[batch],
                ts[batch],
                scale_px=scale_px,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else float("nan")

        model.eval()
        with torch.no_grad():
            val_logits = model(vx)
            _ = calibrated_loss(val_logits, vc, vt, vl, vs, scale_px=scale_px)
            weights = torch.softmax(val_logits, dim=1)
            pred = torch.sum(weights.unsqueeze(-1) * vc, dim=1)
            val_rmse = float(torch.sqrt(torch.mean(torch.sum((pred - vt) ** 2, dim=1))).cpu())
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_scores = co.score_reranker(model, val_x, device=device, batch_nodes=2048)
    temp, _ = tune_temperature_arrays(val_candidates_px, val_target_px, val_scores)
    return CriticTrainResult(model, float(temp), best_val, best_epoch, final_loss)


def candidate_arrays(split: co.CandidateSplit, node_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2).astype(np.float32)
    target = split.y_px[node_idx].astype(np.float32)
    return candidates, target


def score_error_corr(candidates: np.ndarray, target: np.ndarray, scores: np.ndarray) -> float:
    errors = np.linalg.norm(candidates - target[:, None, :], axis=-1)
    a = scores.reshape(-1)
    b = (-errors).reshape(-1)
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ndcg_at_k(candidates: np.ndarray, target: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    errors = np.linalg.norm(candidates - target[:, None, :], axis=-1)
    relevance = 1.0 / (1.0 + errors)
    order = np.argsort(-scores, axis=1)[:, :k]
    ideal = np.argsort(-relevance, axis=1)[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(np.take_along_axis(relevance, order, axis=1) * discounts[None, :], axis=1)
    idcg = np.sum(np.take_along_axis(relevance, ideal, axis=1) * discounts[None, :], axis=1)
    return float(np.mean(dcg / (idcg + EPS)))


def evaluate_scores(
    split: co.CandidateSplit,
    node_idx: np.ndarray,
    scores: np.ndarray,
    temp: float,
) -> dict[str, float]:
    raw = co.evaluate_reranker_scores(split, node_idx, scores, temperature=temp)
    candidates, target = candidate_arrays(split, node_idx)
    return {
        "rmse": raw["reranker_softmax_rmse_px"],
        "r2": raw["reranker_softmax_r2_vec"],
        "angular_cosine": mean_cosine_arrays(
            np.sum(
                (
                    np.exp(np.clip((scores - scores.max(axis=1, keepdims=True)) / max(temp, 1e-3), -40, 20))
                    / np.maximum(
                        np.exp(
                            np.clip(
                                (scores - scores.max(axis=1, keepdims=True)) / max(temp, 1e-3),
                                -40,
                                20,
                            )
                        ).sum(axis=1, keepdims=True),
                        EPS,
                    )
                )[:, :, None]
                * candidates,
                axis=1,
            ),
            target,
        ),
        "magnitude_ratio": magnitude_ratio_arrays(
            np.sum(
                (
                    np.exp(np.clip((scores - scores.max(axis=1, keepdims=True)) / max(temp, 1e-3), -40, 20))
                    / np.maximum(
                        np.exp(
                            np.clip(
                                (scores - scores.max(axis=1, keepdims=True)) / max(temp, 1e-3),
                                -40,
                                20,
                            )
                        ).sum(axis=1, keepdims=True),
                        EPS,
                    )
                )[:, :, None]
                * candidates,
                axis=1,
            ),
            target,
        ),
        "top1_rmse": raw["reranker_top1_rmse_px"],
        "top3_rmse": raw["reranker_top3_mean_rmse_px"],
        "oracle_candidate_acc": raw["reranker_oracle_candidate_acc"],
        "score_error_corr": score_error_corr(candidates, target, scores),
        "ndcg_at_5": ndcg_at_k(candidates, target, scores, k=min(5, candidates.shape[1])),
    }


def permutation_feature_probe(
    model: torch.nn.Module,
    test_x: np.ndarray,
    test_candidates: np.ndarray,
    test_target: np.ndarray,
    names: list[str],
    selected_mask: np.ndarray,
    device: torch.device,
    temp: float,
    seed: int,
    limit: int,
) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    base_scores = co.score_reranker(model, test_x, device=device, batch_nodes=2048)
    weights = np.exp(base_scores / max(temp, EPS))
    weights = weights / (weights.sum(axis=1, keepdims=True) + EPS)
    base_pred = np.sum(weights[..., None] * test_candidates, axis=1)
    base_rmse = vector_rmse_arrays(base_pred, test_target)

    selected_names = [name for name, keep in zip(names, selected_mask, strict=False) if keep]
    rows: list[dict[str, object]] = []
    max_features = min(limit, test_x.shape[-1])
    candidate_cols = list(range(test_x.shape[-1]))
    # Prefer interpretable non-one-hot feature names first.
    candidate_cols = sorted(
        candidate_cols,
        key=lambda i: (
            selected_names[i].startswith("family__"),
            selected_names[i].startswith("candidate__"),
            i,
        ),
    )[:max_features]
    flat = test_x.reshape(-1, test_x.shape[-1])
    for col in candidate_cols:
        perturbed = flat.copy()
        perturbed[:, col] = perturbed[rng.permutation(perturbed.shape[0]), col]
        perturbed = perturbed.reshape(test_x.shape).astype(np.float32)
        scores = co.score_reranker(model, perturbed, device=device, batch_nodes=2048)
        weights = np.exp(scores / max(temp, EPS))
        weights = weights / (weights.sum(axis=1, keepdims=True) + EPS)
        pred = np.sum(weights[..., None] * test_candidates, axis=1)
        rmse = vector_rmse_arrays(pred, test_target)
        rows.append(
            {
                "feature": selected_names[col],
                "base_rmse": base_rmse,
                "permuted_rmse": rmse,
                "delta_rmse": rmse - base_rmse,
            }
        )
    return pd.DataFrame(rows)


def train_backbone_and_candidates(
    cell_type: str,
    horizon: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[co.CandidateSplit, co.CandidateSplit, co.CandidateSplit, float, dict[str, object]]:
    raw, meta = la.load_lachance_dataset(
        cell_type,
        table_root=args.data_root,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        max_movies=args.max_movies,
        max_tracks_per_movie=args.max_tracks_per_movie,
        frame_stride=args.frame_stride,
        smooth_window=args.smooth_window,
        crop_fraction=args.crop_fraction,
        r_cut_px=args.r_cut_px,
    )
    graphs, norm, coverage = la.prepare_dataset(
        cell_type,
        raw,
        meta,
        horizon=horizon,
        k=args.k,
        device=device,
    )

    temporal_epochs = args.temporal_epochs
    flow_epochs = args.flow_epochs
    mp_epochs = args.mp_epochs
    if args.smoke:
        temporal_epochs = min(temporal_epochs, 4)
        flow_epochs = min(flow_epochs, 4)
        mp_epochs = min(mp_epochs, 5)

    print(f"[{cell_type}] seed={seed} temporal", flush=True)
    temporal, temporal_info = arch.train_temporal(
        graphs["train"],
        graphs["val"],
        seed=seed,
        epochs=temporal_epochs,
        batch_size=args.backbone_batch_size,
        sequence_balanced_loss=args.sequence_balanced_loss,
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
        batch_size=args.backbone_batch_size,
        sequence_balanced_loss=args.sequence_balanced_loss,
    )
    encoded = {split: ng.encode_all(temporal, flow, graph) for split, graph in graphs.items()}

    variant = args.backbone_variant or co.default_variant(cell_type)
    print(f"[{cell_type}] seed={seed} proposal {variant}", flush=True)
    model, model_info = ng.train_mp_decoder(
        graphs["train"],
        graphs["val"],
        encoded["train"],
        encoded["val"],
        variant=variant,
        seed=seed,
        epochs=mp_epochs,
        sequence_balanced_loss=args.sequence_balanced_loss,
        layers=args.mp_layers,
        hidden_dim=args.mp_hidden_dim,
        edge_hidden_dim=args.mp_edge_hidden_dim,
        max_delta_norm=args.mp_max_delta_norm,
        lr=args.mp_lr,
        social_l2=args.mp_social_l2,
        flow_gate_l2=args.mp_flow_gate_l2,
    )
    train_mask = graphs["train"].target_valid.detach().cpu().numpy()
    train_y = graphs["train"].y_px.detach().cpu().numpy()
    train_target_median = float(np.median(np.linalg.norm(train_y[train_mask], axis=1)))
    train_target_median = max(train_target_median, 1.0)

    train_split = co.make_candidate_split(
        model=model,
        graph=graphs["train"],
        enc=encoded["train"],
        norm=norm,
        horizon=horizon,
        train_target_median=train_target_median,
        seed=seed + 201,
        sobol_count=args.sobol_count,
        gaussian_count=args.gaussian_count,
    )
    val_split = co.make_candidate_split(
        model=model,
        graph=graphs["val"],
        enc=encoded["val"],
        norm=norm,
        horizon=horizon,
        train_target_median=train_target_median,
        seed=seed + 301,
        sobol_count=args.sobol_count,
        gaussian_count=args.gaussian_count,
    )
    test_split = co.make_candidate_split(
        model=model,
        graph=graphs["test"],
        enc=encoded["test"],
        norm=norm,
        horizon=horizon,
        train_target_median=train_target_median,
        seed=seed + 101,
        sobol_count=args.sobol_count,
        gaussian_count=args.gaussian_count,
    )
    info = {
        "coverage": coverage,
        "variant": variant,
        "train_target_median_mag_px": train_target_median,
        **{f"temporal_{key}": value for key, value in temporal_info.items()},
        **{f"flow_{key}": value for key, value in flow_info.items()},
        **{f"proposal_{key}": value for key, value in model_info.items()},
    }
    return train_split, val_split, test_split, float(train_target_median), info


def run_real_setting(
    cell_type: str,
    horizon: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[critic-v2] start {cell_type} h{horizon} seed={seed}", flush=True)
    train_split, val_split, test_split, scale_px, info = train_backbone_and_candidates(
        cell_type,
        horizon,
        seed,
        args,
        device,
    )

    train_idx = co.choose_target_nodes(
        train_split.mask,
        max_nodes=args.reranker_train_nodes if not args.smoke else min(args.reranker_train_nodes, 512),
        seed=seed + 101,
    )
    val_idx = co.choose_target_nodes(
        val_split.mask,
        max_nodes=args.reranker_val_nodes if not args.smoke else min(args.reranker_val_nodes, 384),
        seed=seed + 202,
    )
    test_idx = np.flatnonzero(test_split.mask).astype(np.int64, copy=False)
    if args.smoke and len(test_idx) > args.smoke_test_nodes:
        rng = np.random.default_rng(seed + 303)
        test_idx = rng.choice(test_idx, size=args.smoke_test_nodes, replace=False)

    family_to_idx = {family: idx for idx, family in enumerate(sorted(set(train_split.pack.families)))}
    train_packet = build_feature_packet(train_split, train_idx, family_to_idx, scale_px)
    val_packet = build_feature_packet(val_split, val_idx, family_to_idx, scale_px)
    test_packet = build_feature_packet(test_split, test_idx, family_to_idx, scale_px)
    if train_packet.names != val_packet.names or train_packet.names != test_packet.names:
        raise RuntimeError("Feature name mismatch between splits")

    train_candidates, train_target = candidate_arrays(train_split, train_idx)
    val_candidates, val_target = candidate_arrays(val_split, val_idx)
    test_candidates, test_target = candidate_arrays(test_split, test_idx)

    mask_for_metrics = np.zeros_like(test_split.mask, dtype=bool)
    mask_for_metrics[test_idx] = True
    base_rmse = co.vector_rmse(test_split.y_px, test_split.self_flow_px, mask_for_metrics)
    _, family_rows, _ = co.candidate_metrics(
        test_split.y_px,
        mask_for_metrics,
        test_split.pack,
        base_rmse=base_rmse,
    )
    family_rows.insert(0, "seed", seed)
    family_rows.insert(0, "horizon", horizon)
    family_rows.insert(0, "cell_type", cell_type)

    summary_rows: list[dict[str, object]] = []
    probe_frames: list[pd.DataFrame] = []

    summary_rows.append(
        {
            "cell_type": cell_type,
            "horizon": horizon,
            "seed": seed,
            "model": "fixed",
            "feature_set": "self_flow",
            "rmse": base_rmse,
            "r2": co.vector_r2_from_arrays(test_split.self_flow_px[test_idx], test_target),
            "angular_cosine": mean_cosine_arrays(test_split.self_flow_px[test_idx], test_target),
            "magnitude_ratio": magnitude_ratio_arrays(test_split.self_flow_px[test_idx], test_target),
            "temp": np.nan,
            "score_error_corr": np.nan,
            "ndcg_at_5": np.nan,
            "selected_features": 0,
            **info,
        }
    )
    proposal = test_split.proposal_px[test_idx]
    summary_rows.append(
        {
            "cell_type": cell_type,
            "horizon": horizon,
            "seed": seed,
            "model": "fixed",
            "feature_set": "proposal_backbone",
            "rmse": vector_rmse_arrays(proposal, test_target),
            "r2": co.vector_r2_from_arrays(proposal, test_target),
            "angular_cosine": mean_cosine_arrays(proposal, test_target),
            "magnitude_ratio": magnitude_ratio_arrays(proposal, test_target),
            "temp": np.nan,
            "score_error_corr": np.nan,
            "ndcg_at_5": np.nan,
            "selected_features": 0,
            **info,
        }
    )
    oracle_idx = np.argmin(np.linalg.norm(test_candidates - test_target[:, None, :], axis=-1), axis=1)
    oracle_pred = test_candidates[np.arange(len(test_idx)), oracle_idx]
    summary_rows.append(
        {
            "cell_type": cell_type,
            "horizon": horizon,
            "seed": seed,
            "model": "oracle",
            "feature_set": "oracle_all_candidates",
            "rmse": vector_rmse_arrays(oracle_pred, test_target),
            "r2": co.vector_r2_from_arrays(oracle_pred, test_target),
            "angular_cosine": mean_cosine_arrays(oracle_pred, test_target),
            "magnitude_ratio": magnitude_ratio_arrays(oracle_pred, test_target),
            "temp": np.nan,
            "score_error_corr": np.nan,
            "ndcg_at_5": np.nan,
            "selected_features": 0,
            **info,
        }
    )

    feature_sets = [BASELINE_FEATURE_SET, *args.feature_sets]
    models = args.critic_models
    for feature_set in feature_sets:
        selected = feature_mask(train_packet.groups, train_packet.names, feature_set)
        selected_names = [name for name, keep in zip(train_packet.names, selected, strict=False) if keep]
        tr_x, va_x, te_x, _, _ = standardize_train_val_test(
            train_packet.values[..., selected],
            val_packet.values[..., selected],
            test_packet.values[..., selected],
        )
        selected_groups = [group for group, keep in zip(train_packet.groups, selected, strict=False) if keep]
        tr_x = apply_feature_control(tr_x, selected_groups, feature_set, seed + 404)
        va_x = apply_feature_control(va_x, selected_groups, feature_set, seed + 505)
        te_x = apply_feature_control(te_x, selected_groups, feature_set, seed + 606)

        model_names = models
        if args.smoke and "set_light" in model_names and feature_set not in {"full"}:
            model_names = ["mlp"]
        for model_name in model_names:
            result = train_critic_v2(
                tr_x,
                va_x,
                train_candidates,
                val_candidates,
                train_target,
                val_target,
                model_name=model_name,
                device=device,
                scale_px=scale_px,
                epochs=args.epochs if not args.smoke else min(args.epochs, 5),
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                seed=seed + (sum(ord(ch) for ch in f"{feature_set}:{model_name}") % 10000),
            )
            scores = co.score_reranker(result.model, te_x, device=device, batch_nodes=2048)
            metrics = evaluate_scores(test_split, test_idx, scores, result.temp)
            summary_rows.append(
                {
                    "cell_type": cell_type,
                    "horizon": horizon,
                    "seed": seed,
                    "model": model_name,
                    "feature_set": feature_set,
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "angular_cosine": metrics["angular_cosine"],
                    "magnitude_ratio": metrics["magnitude_ratio"],
                    "temp": result.temp,
                    "score_error_corr": metrics["score_error_corr"],
                    "ndcg_at_5": metrics["ndcg_at_5"],
                    "selected_features": int(selected.sum()),
                    "best_val_rmse": result.best_val_rmse,
                    "best_epoch": result.best_epoch,
                    "train_loss": result.train_loss,
                    **info,
                }
            )
            if (
                feature_set == "full"
                and model_name == "mlp"
                and args.permutation_features > 0
            ):
                probe = permutation_feature_probe(
                    result.model,
                    te_x,
                    test_candidates,
                    test_target,
                    train_packet.names,
                    selected,
                    device,
                    result.temp,
                    seed=seed + 707,
                    limit=args.permutation_features if not args.smoke else min(args.permutation_features, 12),
                )
                if not probe.empty:
                    probe.insert(0, "seed", seed)
                    probe.insert(0, "horizon", horizon)
                    probe.insert(0, "cell_type", cell_type)
                    probe_frames.append(probe)
            print(
                f"[critic-v2] {cell_type} h{horizon} seed={seed} "
                f"{model_name}/{feature_set}: RMSE={metrics['rmse']:.3f} "
                f"R2={metrics['r2']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    probes = pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
    return summary, family_rows, probes


def build_gate_table(summary: pd.DataFrame, synthetic_gate: dict[str, object] | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame()
    learned = summary[summary["model"].isin(["mlp", "set_light"])].copy()
    for (cell_type, horizon, seed), group in learned.groupby(["cell_type", "horizon", "seed"]):
        def best(feature_set: str) -> pd.Series | None:
            sub = group[group["feature_set"] == feature_set]
            if sub.empty:
                return None
            return sub.sort_values("rmse").iloc[0]

        full = best("full")
        baseline = best(BASELINE_FEATURE_SET)
        no_physics = best("no_physics")
        dynamic = best("dynamic_only")
        oz = best("oz_only")
        shuffled = best("shuffled_state")
        time_shuffled = best("time_shuffled")
        if full is None:
            continue
        baseline_rmse = float(baseline["rmse"]) if baseline is not None else float("nan")
        no_physics_rmse = float(no_physics["rmse"]) if no_physics is not None else float("nan")
        dynamic_rmse = float(dynamic["rmse"]) if dynamic is not None else float("nan")
        shuffled_rmse = float(shuffled["rmse"]) if shuffled is not None else float("nan")
        time_shuffled_rmse = float(time_shuffled["rmse"]) if time_shuffled is not None else float("nan")
        control_best = np.nanmin([baseline_rmse, no_physics_rmse, dynamic_rmse])
        gain_vs_baseline = co.gain_pct(baseline_rmse, float(full["rmse"]))
        gain_vs_no_physics = co.gain_pct(no_physics_rmse, float(full["rmse"]))
        gain_vs_dynamic = co.gain_pct(dynamic_rmse, float(full["rmse"]))
        gain_vs_best = co.gain_pct(control_best, float(full["rmse"]))
        gain_vs_shuffled = co.gain_pct(shuffled_rmse, float(full["rmse"]))
        gain_vs_time_shuffled = co.gain_pct(time_shuffled_rmse, float(full["rmse"]))
        control_gate = (
            gain_vs_no_physics >= 1.0
            and gain_vs_dynamic >= 1.0
            and (np.isnan(gain_vs_shuffled) or gain_vs_shuffled >= 0.0)
            and (np.isnan(gain_vs_time_shuffled) or gain_vs_time_shuffled >= 0.0)
        )
        row_real_gate = bool(int(horizon) == 6 and gain_vs_baseline >= 5.0 and control_gate)
        rows.append(
            {
                "cell_type": cell_type,
                "horizon": int(horizon),
                "seed": int(seed),
                "full_model": full["model"],
                "full_rmse": float(full["rmse"]),
                "baseline_rmse": baseline_rmse,
                "no_physics_rmse": no_physics_rmse,
                "dynamic_rmse": dynamic_rmse,
                "oz_only_rmse": float(oz["rmse"]) if oz is not None else float("nan"),
                "shuffled_state_rmse": shuffled_rmse,
                "time_shuffled_rmse": time_shuffled_rmse,
                "gain_vs_baseline_pct": gain_vs_baseline,
                "gain_vs_no_physics_pct": gain_vs_no_physics,
                "gain_vs_dynamic_pct": gain_vs_dynamic,
                "gain_vs_best_control_pct": gain_vs_best,
                "gain_vs_shuffled_state_pct": gain_vs_shuffled,
                "gain_vs_time_shuffled_pct": gain_vs_time_shuffled,
                "control_gate_pass": control_gate,
                "real_gate_pass": row_real_gate,
                "synthetic_gate_source": synthetic_gate.get("source") if synthetic_gate else "not_requested",
                "synthetic_pretrain_allowed": bool(synthetic_gate.get("pretrain_allowed", False))
                if synthetic_gate
                else False,
            }
        )
    return pd.DataFrame(rows)


def load_existing_synthetic_gate(outputs_root: Path) -> dict[str, object]:
    """Use the latest active-transition synthetic/physics gate as a blocker.

    The v2 runner does not start synthetic pretraining automatically.  Until a
    dedicated v2 synthetic gate is added, we deliberately require an existing
    active-transition gate to show a clean positive signal; otherwise pretrain
    and teacher are blocked.
    """

    candidates = sorted(outputs_root.glob("active_transition_physics_gate*/active_transition_physics_gate.csv"))
    if not candidates:
        candidates = sorted(outputs_root.glob("active_transition_physics_gate*/combined_gate_summary.csv"))
    if not candidates:
        return {
            "source": "missing_existing_active_gate",
            "pretrain_allowed": False,
            "reason": "No existing active-transition gate output found; synthetic pretraining blocked.",
        }
    path = candidates[-1]
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive for field runs
        return {
            "source": str(path),
            "pretrain_allowed": False,
            "reason": f"Could not read existing gate: {exc}",
        }
    # Conservative: permit only if the table explicitly contains positive gate rows.
    if "gate_pass" in df.columns:
        allowed = bool(df["gate_pass"].fillna(False).all())
    elif "real_gate_pass" in df.columns:
        allowed = bool(df["real_gate_pass"].fillna(False).all())
    else:
        allowed = False
    return {
        "source": str(path),
        "pretrain_allowed": allowed,
        "reason": "Existing active-transition gate passed." if allowed else "Existing gate did not pass; pretraining blocked.",
    }


def make_plots(out_dir: Path, summary: pd.DataFrame, gate: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    learned = summary[summary["model"].isin(["mlp", "set_light"])].copy()
    if not learned.empty:
        pivot = (
            learned.groupby(["cell_type", "horizon", "feature_set"], as_index=False)["rmse"]
            .mean()
            .sort_values(["cell_type", "horizon", "rmse"])
        )
        for (cell_type, horizon), group in pivot.groupby(["cell_type", "horizon"]):
            fig, ax = plt.subplots(figsize=(10, 4.8))
            shown = group.head(12)
            ax.bar(shown["feature_set"], shown["rmse"], color="#4C78A8")
            ax.set_title(f"Critic v2 RMSE: {cell_type} h{horizon}")
            ax.set_ylabel("RMSE px")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(plot_dir / f"critic_v2_gate_{cell_type}_h{horizon}.png", dpi=180)
            plt.close(fig)
    if not gate.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        labels = gate.apply(lambda r: f"{r['cell_type']} h{int(r['horizon'])} s{int(r['seed'])}", axis=1)
        ax.bar(labels, gate["gain_vs_best_control_pct"], color="#59A14F")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(1.0, color="#E15759", linestyle="--", linewidth=1.0)
        ax.set_title("Critic v2 gain vs best control")
        ax.set_ylabel("RMSE gain, %")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        fig.savefig(plot_dir / "critic_v2_best_control_gain.png", dpi=180)
        plt.close(fig)


def write_status_report(
    out_dir: Path,
    summary: pd.DataFrame,
    gate: pd.DataFrame,
    synthetic_gate: dict[str, object] | None,
    args: argparse.Namespace,
) -> None:
    report = out_dir / "critic_v2_status_report.md"
    lines: list[str] = []
    lines.append("# Learned Calibrated Transition Critic v2 status")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "Runner tests an offline critic over causal LaChance candidate trajectories. "
        "It does not integrate into the final backbone and does not start teacher-student."
    )
    lines.append("")
    lines.append(f"- datasets: `{', '.join(args.cell_types)}`")
    lines.append(f"- horizons: `{', '.join(map(str, args.horizons))}`")
    lines.append(f"- seeds: `{', '.join(map(str, args.seeds))}`")
    lines.append(f"- feature sets: `{', '.join(args.feature_sets)}`")
    lines.append(f"- models: `{', '.join(args.critic_models)}`")
    lines.append("")
    if synthetic_gate:
        lines.append("## Synthetic gate")
        lines.append("")
        lines.append(f"- source: `{synthetic_gate.get('source')}`")
        lines.append(f"- pretrain allowed: `{bool(synthetic_gate.get('pretrain_allowed', False))}`")
        lines.append(f"- reason: {synthetic_gate.get('reason')}")
        lines.append("")
    else:
        lines.append("## Synthetic gate")
        lines.append("")
        lines.append("- not requested; synthetic pretraining is blocked by default.")
        lines.append("")

    lines.append("## Best learned rows")
    lines.append("")
    learned = summary[summary["model"].isin(["mlp", "set_light"])].copy()
    if learned.empty:
        lines.append("No learned critic rows were produced.")
    else:
        best = (
            learned.sort_values("rmse")
            .groupby(["cell_type", "horizon", "seed"], as_index=False)
            .head(3)
            .loc[:, ["cell_type", "horizon", "seed", "model", "feature_set", "rmse", "r2", "angular_cosine"]]
        )
        lines.append(best.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Gate")
    lines.append("")
    if gate.empty:
        lines.append("Gate table is empty.")
    else:
        shown = gate.loc[
            :,
            [
                "cell_type",
                "horizon",
                "seed",
                "full_rmse",
                "baseline_rmse",
                "no_physics_rmse",
                "dynamic_rmse",
                "gain_vs_best_control_pct",
                "real_gate_pass",
                "synthetic_pretrain_allowed",
            ],
        ]
        lines.append(shown.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    real_pass = bool(gate["real_gate_pass"].fillna(False).any()) if not gate.empty else False
    synthetic_pass = bool(synthetic_gate.get("pretrain_allowed", False)) if synthetic_gate else False
    lines.append("## Decision")
    lines.append("")
    if real_pass and synthetic_pass:
        lines.append(
            "Critic v2 passed the current real-data and synthetic gates. "
            "Pretraining/teacher-student can be considered in the next stage."
        )
    elif real_pass:
        lines.append(
            "Critic v2 has at least one real-data positive gate, but synthetic identifiability "
            "does not allow pretraining yet. Teacher-student remains blocked."
        )
    else:
        lines.append(
            "Critic v2 did not pass the real-data gate in this run. "
            "Do not build teacher-student on this critic yet; inspect feature probes and candidate families first."
        )
    lines.append("")
    lines.append("## Leakage controls")
    lines.append("")
    lines.append(
        "Candidate generation uses causal candidate families from the existing oracle runner. "
        "Feature packets use current positions, current velocities, proposal/self-flow fields, "
        "candidate coordinates and graph topology at t. The true future target is used only in losses "
        "and evaluation, not in inference features. Standardization is fit on train nodes only."
    )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def append_research_plan_update(out_dir: Path, gate: pd.DataFrame) -> None:
    plan_path = REPO_ROOT / "research_plan_prior_gradient_reranker_teacher_2026-06-10.md"
    if not plan_path.exists():
        return
    real_pass = bool(gate["real_gate_pass"].fillna(False).any()) if not gate.empty else False
    text = plan_path.read_text(encoding="utf-8")
    marker = "## 2026-06-12: Learned calibrated transition critic v2"
    if marker in text:
        return
    update = f"""

{marker}

- Реализован isolated runner `scripts/run_lachance_transition_critic_v2.py`.
- Scope: offline critic поверх causal candidate generator/backbone, без интеграции в финальную модель.
- Проверяемые каналы: candidate statistics, backward consistency, soft-neighbour route, active transition компоненты, topology/density/boundary/polarity/alignment/weak structural.
- Контроли: `no_physics`, `dynamic_only`, `oz_only`, `shuffled_state`, `time_shuffled`, компонентные ablations.
- Последний output directory: `{out_dir}`.
- Текущий gate-pass: `{real_pass}`. Teacher/student и synthetic pretrain разрешаются только после прохождения gate.
"""
    plan_path.write_text(text.rstrip() + update + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--table-root", dest="data_root", type=Path, default=la.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/lachance_transition_critic_v2"))
    parser.add_argument("--critic-v2", action="store_true", help="Compatibility flag; runner is always v2.")
    parser.add_argument("--critic-v2-model", default="mlp", help="Comma list: mlp,set_light")
    parser.add_argument("--critic-v2-loss", default="calibrated", choices=["calibrated"])
    parser.add_argument(
        "--critic-v2-feature-set",
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma list of feature sets.",
    )
    parser.add_argument("--cell-types", default="MDCK_Bulk,MDCK_Edge,MDAMB231,HUVEC")
    parser.add_argument("--horizons", default="6,4")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--reranker-train-nodes", type=int, default=35000)
    parser.add_argument("--reranker-val-nodes", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--split-mode", choices=["movie", "frame"], default="movie")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-movies", type=int, default=8)
    parser.add_argument("--max-tracks-per-movie", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--crop-fraction", type=float, default=0.08)
    parser.add_argument("--r-cut-px", type=float, default=50.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temporal-epochs", type=int, default=35)
    parser.add_argument("--flow-epochs", type=int, default=25)
    parser.add_argument("--mp-epochs", type=int, default=80)
    parser.add_argument("--backbone-batch-size", type=int, default=4096)
    parser.add_argument("--sequence-balanced-loss", action="store_true")
    parser.add_argument("--mp-layers", type=int, default=3)
    parser.add_argument("--mp-hidden-dim", type=int, default=72)
    parser.add_argument("--mp-edge-hidden-dim", type=int, default=56)
    parser.add_argument("--mp-max-delta-norm", type=float, default=1.35)
    parser.add_argument("--mp-lr", type=float, default=1.2e-3)
    parser.add_argument("--mp-social-l2", type=float, default=0.0015)
    parser.add_argument("--mp-flow-gate-l2", type=float, default=0.0005)
    parser.add_argument("--sobol-count", type=int, default=16)
    parser.add_argument("--gaussian-count", type=int, default=8)
    parser.add_argument("--backbone-variant", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic-identifiability-gate", action="store_true")
    parser.add_argument("--synthetic-pretrain-if-pass", action="store_true")
    parser.add_argument("--permutation-features", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-graphs", type=int, default=18)
    parser.add_argument("--smoke-test-nodes", type=int, default=512)
    parser.add_argument("--no-plan-update", action="store_true")
    args = parser.parse_args()
    args.cell_types = parse_csv(args.cell_types, ["MDCK_Bulk", "MDCK_Edge", "MDAMB231", "HUVEC"])
    args.horizons = parse_int_csv(args.horizons, [6, 4])
    args.seeds = parse_int_csv(args.seeds, [7, 42, 123])
    args.critic_models = parse_csv(args.critic_v2_model, ["mlp"])
    args.feature_sets = parse_csv(args.critic_v2_feature_set, DEFAULT_FEATURE_SETS)
    unknown = sorted(set(args.feature_sets) - set(DEFAULT_FEATURE_SETS))
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")
    for model_name in args.critic_models:
        if model_name not in {"mlp", "set_light"}:
            raise ValueError(f"Unknown critic model: {model_name}")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[critic-v2] device={device}", flush=True)

    synthetic_gate = None
    if args.synthetic_identifiability_gate:
        synthetic_gate = load_existing_synthetic_gate(Path("outputs"))
        if args.synthetic_pretrain_if_pass and not synthetic_gate.get("pretrain_allowed", False):
            print(
                "[critic-v2] synthetic pretrain requested but blocked by gate: "
                f"{synthetic_gate.get('reason')}",
                flush=True,
            )

    summary_frames: list[pd.DataFrame] = []
    family_frames: list[pd.DataFrame] = []
    probe_frames: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []

    for cell_type in args.cell_types:
        for horizon in args.horizons:
            for seed in args.seeds:
                try:
                    summary, family, probe = run_real_setting(cell_type, horizon, seed, args, device)
                    summary_frames.append(summary)
                    family_frames.append(family)
                    if not probe.empty:
                        probe_frames.append(probe)
                except Exception as exc:  # keep mass runs alive
                    print(f"[critic-v2][ERROR] {cell_type} h{horizon} seed={seed}: {exc}", flush=True)
                    errors.append(
                        {
                            "cell_type": cell_type,
                            "horizon": horizon,
                            "seed": seed,
                            "error": repr(exc),
                        }
                    )
                    if args.smoke:
                        raise

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    family_df = pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame()
    probe_df = pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
    gate_df = build_gate_table(summary_df, synthetic_gate)

    summary_df.to_csv(args.out_dir / "critic_v2_summary.csv", index=False)
    # Component ablation table is the learned summary subset; gate deltas live in critic_v2_gate.csv.
    learned = summary_df[summary_df.get("model", pd.Series(dtype=str)).isin(["mlp", "set_light"])]
    learned.to_csv(args.out_dir / "critic_v2_ablation.csv", index=False)
    gate_df.to_csv(args.out_dir / "critic_v2_gate.csv", index=False)
    probe_df.to_csv(args.out_dir / "critic_v2_feature_probe.csv", index=False)
    family_df.to_csv(args.out_dir / "critic_v2_candidate_family_oracle.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(args.out_dir / "critic_v2_errors.csv", index=False)
    with (args.out_dir / "critic_v2_run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "cell_types": args.cell_types,
                "horizons": args.horizons,
                "seeds": args.seeds,
                "feature_sets": args.feature_sets,
                "models": args.critic_models,
                "smoke": args.smoke,
                "synthetic_gate": synthetic_gate,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    make_plots(args.out_dir, summary_df, gate_df)
    write_status_report(args.out_dir, summary_df, gate_df, synthetic_gate, args)
    if not args.no_plan_update:
        append_research_plan_update(args.out_dir, gate_df)
    print(f"[critic-v2] done: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
