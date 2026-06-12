#!/usr/bin/env python3
"""Candidate-oracle gate for the LaChance route/state-aware backbone.

This script is intentionally placed before learned reranking.  It asks whether
simple, interpretable candidate families cover better future displacements than
the current deterministic backbone.  If best-of-K coverage is weak, a reranker
has nothing useful to select.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_architecture_study as la  # noqa: E402
import run_lachance_nextgen_message_passing as ng  # noqa: E402

arch = la.arch

DEFAULT_OUT = ROOT / "outputs" / "lachance_candidate_oracle"
DEFAULT_TABLE_ROOT = la.DEFAULT_TABLE_ROOT


@dataclass
class CandidatePack:
    names: list[str]
    families: list[str]
    values_px: np.ndarray  # [K, N, 2]


@dataclass
class CandidateSplit:
    graph: arch.GraphTensors
    mask: np.ndarray
    y_px: np.ndarray
    self_px: np.ndarray
    self_flow_px: np.ndarray
    proposal_px: np.ndarray
    fields: dict[str, np.ndarray]
    pack: CandidatePack


def finite_json(value: Any) -> Any:
    return la.finite_json(value)


def parse_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def default_variant(cell_type: str) -> str:
    if cell_type == "MDCK_Edge":
        return "mp_gated_radial"
    if cell_type == "MDCK_Bulk":
        return "mp_gated_velocity_state"
    return "mp_gated_velocity_state"


def vector_rmse(y_px: np.ndarray, pred_px: np.ndarray, mask: np.ndarray) -> float:
    err2 = np.sum(np.square(pred_px[mask] - y_px[mask]), axis=1)
    return float(np.sqrt(np.mean(err2)))


def vector_r2_from_arrays(y_px: np.ndarray, pred_px: np.ndarray) -> float:
    y = np.asarray(y_px, dtype=np.float64)
    pred = np.asarray(pred_px, dtype=np.float64)
    finite = np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    y = y[finite]
    pred = pred[finite]
    if len(y) == 0:
        return float("nan")
    sse = float(np.sum(np.square(pred - y)))
    centered = y - y.mean(axis=0, keepdims=True)
    sst = float(np.sum(np.square(centered)))
    return float(1.0 - sse / sst) if sst > 1e-12 else float("nan")


def vector_r2(y_px: np.ndarray, pred_px: np.ndarray, mask: np.ndarray) -> float:
    return vector_r2_from_arrays(y_px[mask], pred_px[mask])


def vector_mae(y_px: np.ndarray, pred_px: np.ndarray, mask: np.ndarray) -> float:
    err = np.linalg.norm(pred_px[mask] - y_px[mask], axis=1)
    return float(np.mean(err))


def gain_pct(base_rmse: float, rmse: float) -> float:
    return float((base_rmse - rmse) / max(base_rmse, 1e-8) * 100.0)


def np_unit(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, eps)


def scatter_sum_np(values: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, values.shape[1]), dtype=np.float32)
    np.add.at(out, dst, values.astype(np.float32, copy=False))
    return out


def scatter_sum_scalar_np(values: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    np.add.at(out, dst, values.astype(np.float32, copy=False))
    return out


def mean_edge_vec(
    graph: arch.GraphTensors,
    edge_vec: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
) -> np.ndarray:
    n = int(graph.history.shape[0])
    src = graph.src.detach().cpu().numpy()
    dst = graph.dst.detach().cpu().numpy()
    values = edge_vec.detach().cpu().numpy().astype(np.float32)
    if edge_weight is None:
        w = np.ones((len(dst), 1), dtype=np.float32)
    else:
        w = edge_weight.detach().cpu().numpy().astype(np.float32)
        if w.ndim == 1:
            w = w[:, None]
    del src
    num = scatter_sum_np(values * w, dst, n)
    den = scatter_sum_np(w, dst, n)
    return num / np.maximum(den, 1e-6)


def prior_vector_fields(graph: arch.GraphTensors, norm: arch.Normalizer, horizon: int) -> dict[str, np.ndarray]:
    reliability = torch.sqrt(
        torch.clamp(graph.quality[graph.src] * graph.quality[graph.dst], min=0.0)
    )
    force = graph.force_correct
    c_val = graph.c_correct
    force_shuffled = graph.force_shuffled
    c_shuffled = graph.c_shuffled
    radial = graph.radial

    force_field = mean_edge_vec(graph, force * radial, edge_weight=reliability)
    force_shuffled_field = mean_edge_vec(graph, force_shuffled * radial, edge_weight=reliability)
    c_field = mean_edge_vec(graph, c_val * radial, edge_weight=reliability)
    c_shuffled_field = mean_edge_vec(graph, c_shuffled * radial, edge_weight=reliability)
    rel_velocity = mean_edge_vec(graph, graph.rel_velocity, edge_weight=reliability)
    shear = mean_edge_vec(graph, graph.shear, edge_weight=reliability)
    closing = mean_edge_vec(graph, graph.closing * radial, edge_weight=reliability)

    velocity_scale = float(norm.hist_std[0])
    h = float(max(int(horizon), 1))
    return {
        "force": force_field,
        "force_sign_flipped": -force_field,
        "force_shuffled": force_shuffled_field,
        "c_radial": c_field,
        "c_radial_shuffled": c_shuffled_field,
        "rel_velocity": rel_velocity * velocity_scale * h,
        "shear": shear * velocity_scale * h,
        "closing": closing * velocity_scale * h,
    }


def sobol_offsets(n: int, count: int, scale_px: float, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed))
        # Sobol balance is best with powers of two.
        m = int(math.ceil(math.log2(max(count, 1))))
        vals = sampler.random_base2(m=m)[:count]
    except Exception:
        rng = np.random.default_rng(int(seed))
        vals = rng.random((count, 2))
    angles = 2.0 * np.pi * vals[:, 0]
    radii = scale_px * np.sqrt(vals[:, 1])
    offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
    return np.repeat(offsets[:, None, :].astype(np.float32), n, axis=1)


def add_candidate(
    values: list[np.ndarray],
    names: list[str],
    families: list[str],
    name: str,
    family: str,
    pred: np.ndarray,
) -> None:
    values.append(pred.astype(np.float32, copy=False))
    names.append(name)
    families.append(family)


def build_candidates(
    *,
    y_px: np.ndarray,
    mask: np.ndarray,
    self_px: np.ndarray,
    self_flow_px: np.ndarray,
    proposal_px: np.ndarray,
    fields: dict[str, np.ndarray],
    train_target_median_px: float,
    seed: int,
    sobol_count: int,
    gaussian_count: int,
) -> CandidatePack:
    del y_px
    n = int(self_flow_px.shape[0])
    values: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []

    zero = np.zeros_like(self_flow_px)
    flow_only = self_flow_px - self_px
    add_candidate(values, names, families, "self_only", "base", self_px)
    add_candidate(values, names, families, "self_flow", "base", self_flow_px)
    add_candidate(values, names, families, "flow_only", "base", flow_only)
    add_candidate(values, names, families, "proposal", "proposal", proposal_px)
    add_candidate(
        values,
        names,
        families,
        "proposal_self_flow_mean",
        "proposal",
        0.5 * (proposal_px + self_flow_px),
    )
    add_candidate(values, names, families, "zero", "base", zero)

    base_scales = {
        "small": 0.25 * train_target_median_px,
        "mid": 0.50 * train_target_median_px,
        "large": 0.85 * train_target_median_px,
    }
    anchor_map = {"sf": self_flow_px, "prop": proposal_px}
    for field_name in ("force", "force_sign_flipped", "force_shuffled", "c_radial", "c_radial_shuffled", "closing"):
        direction = np_unit(fields[field_name])
        for anchor_name, anchor in anchor_map.items():
            for scale_name, scale in base_scales.items():
                add_candidate(
                    values,
                    names,
                    families,
                    f"{anchor_name}_{field_name}_{scale_name}",
                    field_name,
                    anchor + direction * float(scale),
                )
                if field_name == "force":
                    add_candidate(
                        values,
                        names,
                        families,
                        f"{anchor_name}_{field_name}_minus_{scale_name}",
                        "force_negative_step",
                        anchor - direction * float(scale),
                    )

    for field_name in ("rel_velocity", "shear"):
        field_px = fields[field_name]
        for anchor_name, anchor in anchor_map.items():
            for beta in (0.35, 0.70, 1.0):
                add_candidate(
                    values,
                    names,
                    families,
                    f"{anchor_name}_{field_name}_b{beta:.2f}",
                    field_name,
                    anchor + float(beta) * field_px,
                )

    rng = np.random.default_rng(int(seed))
    base_scale = max(float(train_target_median_px), 1.0)
    if gaussian_count > 0:
        dirs = rng.normal(size=(gaussian_count, 2))
        dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
        mags = rng.uniform(0.15, 0.75, size=(gaussian_count, 1)) * base_scale
        offsets = (dirs * mags).astype(np.float32)
        for idx, offset in enumerate(offsets):
            add_candidate(
                values,
                names,
                families,
                f"gaussian_{idx:02d}",
                "gaussian",
                proposal_px + offset[None, :],
            )
    if sobol_count > 0:
        offsets = sobol_offsets(n, sobol_count, 0.80 * base_scale, seed + 99_001)
        for idx in range(sobol_count):
            add_candidate(
                values,
                names,
                families,
                f"sobol_{idx:02d}",
                "sobol",
                proposal_px + offsets[idx],
            )

    # A simple amplitude repair candidate: keep proposal direction but match the
    # train median target magnitude.  It tests the known under-amplitude issue
    # without using the test target.
    prop_dir = np_unit(proposal_px)
    repaired = prop_dir * train_target_median_px
    add_candidate(values, names, families, "proposal_train_median_magnitude", "amplitude", repaired)

    arr = np.stack(values, axis=0)
    # Avoid impossible explosions from noisy fields.
    max_abs = 5.0 * max(train_target_median_px, 1.0)
    arr = np.clip(arr, -max_abs, max_abs)
    # Keep non-target rows finite for easy vectorized math.
    arr[:, ~mask, :] = self_flow_px[None, ~mask, :]
    return CandidatePack(names=names, families=families, values_px=arr.astype(np.float32))


def candidate_metrics(
    y_px: np.ndarray,
    mask: np.ndarray,
    pack: CandidatePack,
    *,
    base_rmse: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    err = pack.values_px[:, mask, :] - y_px[mask][None, :, :]
    sq = np.sum(np.square(err), axis=2)
    dist = np.sqrt(sq)
    for idx, (name, family) in enumerate(zip(pack.names, pack.families)):
        rmse = float(np.sqrt(np.mean(sq[idx])))
        mae = float(np.mean(dist[idx]))
        rows.append(
            {
                "candidate": name,
                "family": family,
                "rmse_px": rmse,
                "mae_px": mae,
                "r2_vec": vector_r2_from_arrays(y_px[mask], pack.values_px[idx, mask, :]),
                "gain_vs_self_flow_pct": gain_pct(base_rmse, rmse),
            }
        )
    by_candidate = pd.DataFrame(rows).sort_values("rmse_px").reset_index(drop=True)

    family_rows: list[dict[str, Any]] = []
    families = np.asarray(pack.families)
    for family in sorted(set(pack.families)):
        idx = np.flatnonzero(families == family)
        best_sq = np.min(sq[idx], axis=0)
        best_local = idx[np.argmin(sq[idx], axis=0)]
        target_nodes = np.flatnonzero(mask)
        best_pred = pack.values_px[best_local, target_nodes, :]
        rmse = float(np.sqrt(np.mean(best_sq)))
        family_rows.append(
            {
                "family": family,
                "candidates": int(len(idx)),
                "oracle_rmse_px": rmse,
                "oracle_r2_vec": vector_r2_from_arrays(y_px[mask], best_pred),
                "oracle_gain_vs_self_flow_pct": gain_pct(base_rmse, rmse),
            }
        )
    group_defs = {
        "all": np.arange(len(pack.names)),
        "no_prior_gradient": np.flatnonzero(
            ~np.isin(
                families,
                [
                    "force",
                    "force_negative_step",
                    "force_sign_flipped",
                    "force_shuffled",
                    "c_radial",
                    "c_radial_shuffled",
                    "closing",
                ],
            )
        ),
        "physical_only": np.flatnonzero(
            np.isin(
                families,
                [
                    "force",
                    "force_negative_step",
                    "force_sign_flipped",
                    "c_radial",
                    "closing",
                    "rel_velocity",
                    "shear",
                ],
            )
        ),
        "prior_gradient_only": np.flatnonzero(np.isin(families, ["force", "c_radial", "closing"])),
        "prior_gradient_controls": np.flatnonzero(
            np.isin(families, ["force_negative_step", "force_sign_flipped", "force_shuffled", "c_radial_shuffled"])
        ),
        "no_random": np.flatnonzero(~np.isin(families, ["gaussian", "sobol"])),
        "anchors_only": np.flatnonzero(np.isin(families, ["base", "proposal", "amplitude"])),
    }
    for group, idx in group_defs.items():
        if len(idx) == 0:
            continue
        best_sq = np.min(sq[idx], axis=0)
        best_local = idx[np.argmin(sq[idx], axis=0)]
        target_nodes = np.flatnonzero(mask)
        best_pred = pack.values_px[best_local, target_nodes, :]
        rmse = float(np.sqrt(np.mean(best_sq)))
        family_rows.append(
            {
                "family": f"group:{group}",
                "candidates": int(len(idx)),
                "oracle_rmse_px": rmse,
                "oracle_r2_vec": vector_r2_from_arrays(y_px[mask], best_pred),
                "oracle_gain_vs_self_flow_pct": gain_pct(base_rmse, rmse),
            }
        )
    by_family = pd.DataFrame(family_rows).sort_values("oracle_rmse_px").reset_index(drop=True)

    best_idx = np.argmin(sq, axis=0)
    best_fam = families[best_idx]
    unique, counts = np.unique(best_fam, return_counts=True)
    hist = {str(k): int(v) for k, v in zip(unique, counts)}
    top_m: dict[str, float] = {}
    for m in (2, 3, 5):
        if len(pack.names) < m:
            continue
        order = np.argsort(sq, axis=0)[:m]
        target_nodes = np.flatnonzero(mask)
        mixture = np.mean(pack.values_px[order, target_nodes[None, :], :], axis=0)
        rmse = float(np.sqrt(np.mean(np.sum(np.square(mixture - y_px[mask]), axis=1))))
        top_m[f"top{m}_oracle_mixture_rmse_px"] = rmse
        top_m[f"top{m}_oracle_mixture_r2_vec"] = vector_r2_from_arrays(y_px[mask], mixture)
        top_m[f"top{m}_oracle_mixture_gain_pct"] = gain_pct(base_rmse, rmse)
    summary = {
        "oracle_best_family_histogram": hist,
        "candidate_count": int(len(pack.names)),
        **top_m,
    }
    return by_candidate, by_family, summary


class CandidateReranker(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(hidden_dim, mid_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, candidates, feature_dim = x.shape
        return self.net(x.reshape(batch * candidates, feature_dim)).reshape(batch, candidates)


class CandidateSetReranker(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        heads = int(max(1, min(heads, hidden_dim)))
        while hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        self.in_proj = torch.nn.Linear(feature_dim, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=int(max(layers, 1)))
        self.out = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            torch.nn.GELU(),
            torch.nn.Linear(max(hidden_dim // 2, 16), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.in_proj(x))
        return self.out(z).squeeze(-1)


def candidate_argmin_labels(split: CandidateSplit, node_idx: np.ndarray) -> np.ndarray:
    cand = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2)
    err = np.sum(np.square(cand - split.y_px[node_idx, None, :]), axis=2)
    return np.argmin(err, axis=1).astype(np.int64)


def candidate_soft_oracle_targets(
    cand_norm: np.ndarray,
    target_norm: np.ndarray,
    *,
    temperature: float,
    topk: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Soft train-only candidate distribution from true future distance.

    This is a teacher signal for training the scorer, not an inference input.
    Keeping only top-k candidates prevents the target from becoming a nearly
    uniform blur over obviously bad trajectories.
    """

    sq = np.sum(np.square(cand_norm - target_norm[:, None, :]), axis=2).astype(np.float32)
    logits = -sq / float(max(temperature, 1e-4))
    k = int(topk)
    if 0 < k < logits.shape[1]:
        keep = np.zeros_like(logits, dtype=bool)
        idx = np.argpartition(sq, kth=k - 1, axis=1)[:, :k]
        np.put_along_axis(keep, idx, True, axis=1)
        logits = np.where(keep, logits, -1.0e9)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(np.clip(logits, -80.0, 40.0)).astype(np.float32)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=1)
    best = np.min(np.sqrt(sq), axis=1)
    second = np.partition(np.sqrt(sq), kth=min(1, sq.shape[1] - 1), axis=1)[:, min(1, sq.shape[1] - 1)]
    stats = {
        "soft_oracle_temperature": float(temperature),
        "soft_oracle_topk": float(topk),
        "soft_oracle_entropy_mean": float(np.mean(entropy)),
        "soft_oracle_best_norm_error_mean": float(np.mean(best)),
        "soft_oracle_best_second_gap_mean": float(np.mean(second - best)),
    }
    return probs.astype(np.float32, copy=False), stats


def choose_target_nodes(mask: np.ndarray, max_nodes: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if max_nodes > 0 and len(idx) > max_nodes:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_nodes), replace=False))
    return idx.astype(np.int64, copy=False)


def cos_feature(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.sum(a * b, axis=-1, keepdims=True)
    den = np.linalg.norm(a, axis=-1, keepdims=True) * np.linalg.norm(b, axis=-1, keepdims=True)
    return num / np.maximum(den, eps)


def configuration_candidate_features(
    split: CandidateSplit,
    node_idx: np.ndarray,
    *,
    scale_px: float,
    include_joint: bool = False,
) -> np.ndarray:
    """Counterfactual future-configuration features for dst-node candidates.

    For each candidate displacement of a destination node, neighbours are moved
    with the backbone proposal.  This is not a full joint trajectory rollout,
    but it lets the scorer see whether a candidate preserves plausible local
    pair structure.
    """

    graph = split.graph
    if not hasattr(graph, "current_pos_px"):
        return np.zeros((len(node_idx), len(split.pack.names), 0), dtype=np.float32)

    pos = graph.current_pos_px.detach().cpu().numpy().astype(np.float32)
    src_all = graph.src.detach().cpu().numpy()
    dst_all = graph.dst.detach().cpu().numpy()
    n = int(pos.shape[0])
    local = np.full(n, -1, dtype=np.int64)
    local[node_idx] = np.arange(len(node_idx), dtype=np.int64)
    keep = local[dst_all] >= 0
    if not np.any(keep):
        feature_dim = 19 if include_joint else 8
        return np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)

    src = src_all[keep]
    dst = dst_all[keep]
    local_dst = local[dst]
    quality = graph.quality.detach().cpu().numpy().reshape(-1).astype(np.float32)
    weights = np.sqrt(np.clip(quality[src] * quality[dst], 0.0, None)).astype(np.float32)
    den = np.maximum(scatter_sum_scalar_np(weights, local_dst, len(node_idx)), 1e-6)

    current_rel = pos[src] - pos[dst]
    current_dist = np.linalg.norm(current_rel, axis=1).astype(np.float32)
    r_scale = float(max(np.median(current_dist), 1.0))
    close_scale = float(max(0.35 * r_scale, 1.0))
    c_current = np.exp(-0.5 * np.square(current_dist / r_scale)).astype(np.float32)

    src_future = pos[src] + split.proposal_px[src].astype(np.float32)
    neighbour_motion = split.proposal_px[src].astype(np.float32)
    scale = float(max(scale_px, 1.0))
    out = np.zeros((len(node_idx), len(split.pack.names), 8), dtype=np.float32)
    joint_out = np.zeros((len(node_idx), len(split.pack.names), 11), dtype=np.float32) if include_joint else None
    for k_idx in range(len(split.pack.names)):
        cand_motion = split.pack.values_px[k_idx, dst].astype(np.float32)
        dst_future = pos[dst] + cand_motion
        future_rel = src_future - dst_future
        future_dist = np.linalg.norm(future_rel, axis=1).astype(np.float32)
        c_future = np.exp(-0.5 * np.square(future_dist / r_scale)).astype(np.float32)
        collision = np.exp(-0.5 * np.square(future_dist / close_scale)).astype(np.float32)
        motion_diff = np.linalg.norm(neighbour_motion - cand_motion, axis=1).astype(np.float32)
        motion_dot = np.sum(neighbour_motion * cand_motion, axis=1)
        motion_den = np.maximum(
            np.linalg.norm(neighbour_motion, axis=1) * np.linalg.norm(cand_motion, axis=1),
            1e-6,
        )
        motion_cos = (motion_dot / motion_den).astype(np.float32)
        # E=-c.  DeltaE = E_future - E_current = c_current - c_future.
        values = [
            (future_dist - current_dist) / scale,
            np.abs(future_dist - current_dist) / scale,
            c_current - c_future,
            c_future,
            collision,
            np.log1p(future_dist) - np.log1p(current_dist),
            motion_diff / scale,
            motion_cos,
        ]
        for feat_idx, value in enumerate(values):
            out[:, k_idx, feat_idx] = scatter_sum_scalar_np(value * weights, local_dst, len(node_idx)) / den
        if joint_out is not None:
            src_joint_motion = split.pack.values_px[k_idx, src].astype(np.float32)
            src_joint_future = pos[src] + src_joint_motion
            joint_rel = src_joint_future - dst_future
            joint_dist = np.linalg.norm(joint_rel, axis=1).astype(np.float32)
            joint_c_future = np.exp(-0.5 * np.square(joint_dist / r_scale)).astype(np.float32)
            joint_collision = np.exp(-0.5 * np.square(joint_dist / close_scale)).astype(np.float32)
            joint_motion_diff = np.linalg.norm(src_joint_motion - cand_motion, axis=1).astype(np.float32)
            joint_motion_dot = np.sum(src_joint_motion * cand_motion, axis=1)
            joint_motion_den = np.maximum(
                np.linalg.norm(src_joint_motion, axis=1) * np.linalg.norm(cand_motion, axis=1),
                1e-6,
            )
            joint_motion_cos = (joint_motion_dot / joint_motion_den).astype(np.float32)
            mid_rel = (pos[src] + 0.5 * src_joint_motion) - (pos[dst] + 0.5 * cand_motion)
            mid_dist = np.linalg.norm(mid_rel, axis=1).astype(np.float32)
            mid_c = np.exp(-0.5 * np.square(mid_dist / r_scale)).astype(np.float32)
            mid_collision = np.exp(-0.5 * np.square(mid_dist / close_scale)).astype(np.float32)
            src_anchor_gap = np.linalg.norm(src_joint_motion - split.proposal_px[src], axis=1).astype(np.float32)
            joint_values = [
                (joint_dist - current_dist) / scale,
                np.abs(joint_dist - current_dist) / scale,
                c_current - joint_c_future,
                joint_c_future,
                joint_collision,
                joint_motion_diff / scale,
                joint_motion_cos,
                mid_c,
                mid_collision,
                c_current - mid_c,
                src_anchor_gap / scale,
            ]
            for feat_idx, value in enumerate(joint_values):
                joint_out[:, k_idx, feat_idx] = scatter_sum_scalar_np(value * weights, local_dst, len(node_idx)) / den
    if joint_out is not None:
        out = np.concatenate([out, joint_out], axis=2)
    return np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32, copy=False)


def source_candidate_soft_motion(
    split: CandidateSplit,
    *,
    scale_px: float,
    temperature: float,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inference-safe soft future motion for every node.

    This is not an oracle.  It favours candidates close to the backbone
    proposal/self-flow anchors and keeps a top-k soft distribution so that
    random candidates do not dominate neighbour futures.
    """

    cand = split.pack.values_px.astype(np.float32)
    scale = float(max(scale_px, 1.0))
    proposal = split.proposal_px.astype(np.float32)[None, :, :]
    self_flow = split.self_flow_px.astype(np.float32)[None, :, :]
    sq_prop = np.sum(np.square(cand - proposal), axis=2) / (scale * scale)
    sq_flow = np.sum(np.square(cand - self_flow), axis=2) / (scale * scale)
    family_bias = np.zeros(len(split.pack.names), dtype=np.float32)
    for idx, family in enumerate(split.pack.families):
        if family in {"base", "proposal", "amplitude"}:
            family_bias[idx] = 0.10
        elif family in {"rel_velocity", "shear", "closing"}:
            family_bias[idx] = 0.04
        elif family in {"gaussian", "sobol"}:
            family_bias[idx] = -0.03
    logits = -0.70 * sq_prop - 0.30 * sq_flow + family_bias[:, None]
    k = int(topk)
    if 0 < k < logits.shape[0]:
        keep = np.zeros_like(logits, dtype=bool)
        idx = np.argpartition(-logits, kth=k - 1, axis=0)[:k, :]
        np.put_along_axis(keep, idx, True, axis=0)
        logits = np.where(keep, logits, -1.0e9)
    logits = logits / float(max(temperature, 1e-4))
    logits = logits - np.max(logits, axis=0, keepdims=True)
    prob = np.exp(np.clip(logits, -80.0, 40.0)).astype(np.float32)
    prob = prob / np.maximum(prob.sum(axis=0, keepdims=True), 1e-8)
    soft_motion = np.einsum("kn,knd->nd", prob, cand).astype(np.float32)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-8)), axis=0)
    norm = math.log(float(max(2, min(k if k > 0 else len(split.pack.names), len(split.pack.names)))))
    entropy = (entropy / max(norm, 1e-6)).astype(np.float32)
    spread = np.sqrt(
        np.einsum("kn,knd->n", prob, np.square(cand - soft_motion[None, :, :]))
    ).astype(np.float32) / scale
    top_prob = np.max(prob, axis=0).astype(np.float32)
    return soft_motion, entropy, spread, top_prob


def soft_neighbour_route_features(
    split: CandidateSplit,
    node_idx: np.ndarray,
    *,
    scale_px: float,
    source_temperature: float,
    source_topk: int,
    route_temperature: float,
) -> np.ndarray:
    """Candidate-conditioned soft-neighbour future-configuration features.

    Unlike the first joint critic, neighbours do not blindly take the same
    candidate index.  For each target candidate, each neighbour softly chooses
    among several inference-safe future bases according to pair-configuration
    compatibility.
    """

    graph = split.graph
    feature_dim = 14
    if not hasattr(graph, "current_pos_px"):
        return np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)

    pos = graph.current_pos_px.detach().cpu().numpy().astype(np.float32)
    src_all = graph.src.detach().cpu().numpy()
    dst_all = graph.dst.detach().cpu().numpy()
    n = int(pos.shape[0])
    local = np.full(n, -1, dtype=np.int64)
    local[node_idx] = np.arange(len(node_idx), dtype=np.int64)
    keep = local[dst_all] >= 0
    if not np.any(keep):
        return np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)

    src = src_all[keep]
    dst = dst_all[keep]
    local_dst = local[dst]
    quality = graph.quality.detach().cpu().numpy().reshape(-1).astype(np.float32)
    reliability = np.sqrt(np.clip(quality[src] * quality[dst], 0.0, None)).astype(np.float32)
    den = np.maximum(scatter_sum_scalar_np(reliability, local_dst, len(node_idx)), 1e-6)

    current_rel = pos[src] - pos[dst]
    current_dist = np.linalg.norm(current_rel, axis=1).astype(np.float32)
    r_scale = float(max(np.median(current_dist), 1.0))
    close_scale = float(max(0.35 * r_scale, 1.0))
    c_current = np.exp(-0.5 * np.square(current_dist / r_scale)).astype(np.float32)
    scale = float(max(scale_px, 1.0))

    soft_motion_all, source_entropy_all, source_spread_all, source_top_prob_all = source_candidate_soft_motion(
        split,
        scale_px=scale_px,
        temperature=source_temperature,
        topk=source_topk,
    )
    proposal_motion = split.proposal_px[src].astype(np.float32)
    self_flow_motion = split.self_flow_px[src].astype(np.float32)
    soft_motion = soft_motion_all[src].astype(np.float32)
    source_entropy = source_entropy_all[src].astype(np.float32)
    source_spread = source_spread_all[src].astype(np.float32)
    source_top_prob = source_top_prob_all[src].astype(np.float32)

    out = np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)
    for k_idx in range(len(split.pack.names)):
        target_motion = split.pack.values_px[k_idx, dst].astype(np.float32)
        same_motion = split.pack.values_px[k_idx, src].astype(np.float32)
        bases = [proposal_motion, self_flow_motion, soft_motion, same_motion]
        basis_scores: list[np.ndarray] = []
        basis_dist: list[np.ndarray] = []
        basis_c: list[np.ndarray] = []
        basis_collision: list[np.ndarray] = []
        basis_motion_cos: list[np.ndarray] = []
        for basis_motion in bases:
            future_rel = (pos[src] + basis_motion) - (pos[dst] + target_motion)
            future_dist = np.linalg.norm(future_rel, axis=1).astype(np.float32)
            c_future = np.exp(-0.5 * np.square(future_dist / r_scale)).astype(np.float32)
            collision = np.exp(-0.5 * np.square(future_dist / close_scale)).astype(np.float32)
            motion_dot = np.sum(basis_motion * target_motion, axis=1)
            motion_den = np.maximum(
                np.linalg.norm(basis_motion, axis=1) * np.linalg.norm(target_motion, axis=1),
                1e-6,
            )
            motion_cos = (motion_dot / motion_den).astype(np.float32)
            stretch = np.abs(future_dist - current_dist) / scale
            score = -stretch - 0.35 * collision + 0.10 * motion_cos + 0.05 * c_future
            basis_scores.append(score.astype(np.float32))
            basis_dist.append(future_dist)
            basis_c.append(c_future)
            basis_collision.append(collision)
            basis_motion_cos.append(motion_cos)
        score_arr = np.stack(basis_scores, axis=0)
        score_arr = score_arr / float(max(route_temperature, 1e-4))
        score_arr = score_arr - np.max(score_arr, axis=0, keepdims=True)
        basis_w = np.exp(np.clip(score_arr, -60.0, 30.0)).astype(np.float32)
        basis_w = basis_w / np.maximum(basis_w.sum(axis=0, keepdims=True), 1e-8)
        soft_edge_motion = sum(basis_w[b_idx, :, None] * bases[b_idx] for b_idx in range(len(bases)))
        future_rel = (pos[src] + soft_edge_motion) - (pos[dst] + target_motion)
        future_dist = np.linalg.norm(future_rel, axis=1).astype(np.float32)
        c_future = np.exp(-0.5 * np.square(future_dist / r_scale)).astype(np.float32)
        collision = np.exp(-0.5 * np.square(future_dist / close_scale)).astype(np.float32)
        motion_diff = np.linalg.norm(soft_edge_motion - target_motion, axis=1).astype(np.float32)
        motion_dot = np.sum(soft_edge_motion * target_motion, axis=1)
        motion_den = np.maximum(
            np.linalg.norm(soft_edge_motion, axis=1) * np.linalg.norm(target_motion, axis=1),
            1e-6,
        )
        motion_cos = (motion_dot / motion_den).astype(np.float32)
        basis_entropy = -np.sum(basis_w * np.log(np.maximum(basis_w, 1e-8)), axis=0).astype(np.float32)
        basis_entropy = basis_entropy / math.log(float(len(bases)))
        values = [
            (future_dist - current_dist) / scale,
            np.abs(future_dist - current_dist) / scale,
            c_current - c_future,
            c_future,
            collision,
            np.log1p(future_dist) - np.log1p(current_dist),
            motion_diff / scale,
            motion_cos,
            basis_entropy,
            basis_w[0],
            basis_w[2],
            basis_w[3],
            np.linalg.norm(soft_edge_motion - proposal_motion, axis=1).astype(np.float32) / scale,
            source_entropy + source_spread - source_top_prob,
        ]
        for feat_idx, value in enumerate(values):
            out[:, k_idx, feat_idx] = scatter_sum_scalar_np(value * reliability, local_dst, len(node_idx)) / den
    return np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32, copy=False)


def backward_consistency_features(
    split: CandidateSplit,
    node_idx: np.ndarray,
    *,
    scale_px: float,
    include_joint: bool = False,
) -> np.ndarray:
    """Inference-time backward consistency features for candidate futures.

    A good candidate future should be locally reversible: after moving the
    target by a candidate and neighbours by their plausible future displacements,
    the reverse local motion field should point back toward the current target
    state.  This block uses no true future target.
    """

    graph = split.graph
    if not hasattr(graph, "current_pos_px"):
        feature_dim = 12 if include_joint else 6
        return np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)

    pos = graph.current_pos_px.detach().cpu().numpy().astype(np.float32)
    src_all = graph.src.detach().cpu().numpy()
    dst_all = graph.dst.detach().cpu().numpy()
    n = int(pos.shape[0])
    local = np.full(n, -1, dtype=np.int64)
    local[node_idx] = np.arange(len(node_idx), dtype=np.int64)
    keep = local[dst_all] >= 0
    feature_dim = 12 if include_joint else 6
    if not np.any(keep):
        return np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)

    src = src_all[keep]
    dst = dst_all[keep]
    local_dst = local[dst]
    quality = graph.quality.detach().cpu().numpy().reshape(-1).astype(np.float32)
    reliability = np.sqrt(np.clip(quality[src] * quality[dst], 0.0, None)).astype(np.float32)
    current_rel = pos[src] - pos[dst]
    current_dist = np.linalg.norm(current_rel, axis=1).astype(np.float32)
    r_scale = float(max(np.median(current_dist), 1.0))
    scale = float(max(scale_px, 1.0))
    current_velocity = graph.current_velocity.detach().cpu().numpy().astype(np.float32)
    own_reverse_node = -current_velocity[node_idx]

    def aggregate_reverse_features(
        k_idx: int,
        neighbour_motion: np.ndarray,
        target_motion: np.ndarray,
        offset: int,
        out: np.ndarray,
    ) -> None:
        dst_future = pos[dst] + target_motion
        src_future = pos[src] + neighbour_motion
        future_rel = src_future - dst_future
        future_dist = np.linalg.norm(future_rel, axis=1).astype(np.float32)
        future_c = np.exp(-0.5 * np.square(future_dist / r_scale)).astype(np.float32)
        w = reliability * future_c
        den = np.maximum(scatter_sum_scalar_np(w, local_dst, len(node_idx)), 1e-6)
        reverse_neighbour_motion = -neighbour_motion
        reverse_field_x = scatter_sum_scalar_np(reverse_neighbour_motion[:, 0] * w, local_dst, len(node_idx)) / den
        reverse_field_y = scatter_sum_scalar_np(reverse_neighbour_motion[:, 1] * w, local_dst, len(node_idx)) / den
        reverse_field = np.stack([reverse_field_x, reverse_field_y], axis=1).astype(np.float32)
        reverse_target = -split.pack.values_px[k_idx, node_idx].astype(np.float32)
        cycle_residual = split.pack.values_px[k_idx, node_idx].astype(np.float32) + reverse_field
        out[:, k_idx, offset + 0] = np.linalg.norm(cycle_residual, axis=1) / scale
        out[:, k_idx, offset + 1] = cos_feature(reverse_target[:, None, :], reverse_field[:, None, :]).reshape(-1)
        out[:, k_idx, offset + 2] = np.linalg.norm(reverse_field, axis=1) / scale
        out[:, k_idx, offset + 3] = np.linalg.norm(reverse_target - reverse_field, axis=1) / scale
        # Filled below at node level; kept as a fixed slot so the joint and
        # non-joint layouts stay stable.
        out[:, k_idx, offset + 4] = 0.0
        out[:, k_idx, offset + 5] = den / np.maximum(scatter_sum_scalar_np(reliability, local_dst, len(node_idx)), 1e-6)

    out = np.zeros((len(node_idx), len(split.pack.names), feature_dim), dtype=np.float32)
    proposal_motion = split.proposal_px[src].astype(np.float32)
    for k_idx in range(len(split.pack.names)):
        target_motion = split.pack.values_px[k_idx, dst].astype(np.float32)
        aggregate_reverse_features(k_idx, proposal_motion, target_motion, 0, out)
        if include_joint:
            joint_neighbour_motion = split.pack.values_px[k_idx, src].astype(np.float32)
            aggregate_reverse_features(k_idx, joint_neighbour_motion, target_motion, 6, out)
    # Fill own-reverse alignment separately in a stable vectorized way.
    for k_idx in range(len(split.pack.names)):
        reverse_target = -split.pack.values_px[k_idx, node_idx].astype(np.float32)
        out[:, k_idx, 4] = cos_feature(own_reverse_node[:, None, :], reverse_target[:, None, :]).reshape(-1)
        if include_joint:
            out[:, k_idx, 10] = out[:, k_idx, 4]
    return np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32, copy=False)


def build_reranker_features(
    split: CandidateSplit,
    node_idx: np.ndarray,
    family_to_idx: dict[str, int],
    *,
    scale_px: float,
    use_config_features: bool,
    use_joint_features: bool = False,
    use_backward_features: bool = False,
    use_soft_neighbour_features: bool = False,
    soft_neighbour_temperature: float = 0.35,
    soft_neighbour_topk: int = 8,
    soft_neighbour_route_temperature: float = 0.50,
) -> np.ndarray:
    cand = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2).astype(np.float32)
    node_count, candidate_count, _ = cand.shape
    scale = float(max(scale_px, 1.0))
    proposal = split.proposal_px[node_idx].astype(np.float32)
    self_flow = split.self_flow_px[node_idx].astype(np.float32)
    self_only = split.self_px[node_idx].astype(np.float32)
    offset_prop = cand - proposal[:, None, :]
    offset_flow = cand - self_flow[:, None, :]
    offset_self = cand - self_only[:, None, :]

    pieces: list[np.ndarray] = [
        cand / scale,
        np.linalg.norm(cand, axis=2, keepdims=True) / scale,
        offset_prop / scale,
        offset_flow / scale,
        offset_self / scale,
        np.linalg.norm(offset_prop, axis=2, keepdims=True) / scale,
        np.linalg.norm(offset_flow, axis=2, keepdims=True) / scale,
        cos_feature(cand, proposal[:, None, :]),
        cos_feature(cand, self_flow[:, None, :]),
        cos_feature(offset_prop, self_flow[:, None, :]),
    ]

    for name in ("force", "force_shuffled", "c_radial", "c_radial_shuffled", "closing", "rel_velocity", "shear"):
        field = split.fields[name][node_idx].astype(np.float32)
        field_norm = np.linalg.norm(field, axis=1, keepdims=True)[:, None, :] / scale
        pieces.append(np.repeat(field_norm, candidate_count, axis=1))
        pieces.append(cos_feature(offset_prop, field[:, None, :]))

    quality = split.graph.quality.detach().cpu().numpy().reshape(-1, 1)[node_idx].astype(np.float32)
    speed = split.graph.speed_norm.detach().cpu().numpy().reshape(-1, 1)[node_idx].astype(np.float32)
    degree = split.graph.degree.detach().cpu().numpy().reshape(-1, 1)[node_idx].astype(np.float32)
    node_feat = np.concatenate([quality, speed, degree], axis=1)[:, None, :]
    pieces.append(np.repeat(node_feat, candidate_count, axis=1))

    fam = np.zeros((candidate_count, len(family_to_idx)), dtype=np.float32)
    for idx, family in enumerate(split.pack.families):
        fam[idx, family_to_idx[family]] = 1.0
    pieces.append(np.repeat(fam[None, :, :], node_count, axis=0))
    if use_config_features:
        pieces.append(
            configuration_candidate_features(
                split,
                node_idx,
                scale_px=scale_px,
                include_joint=use_joint_features,
            )
        )
    if use_backward_features:
        pieces.append(
            backward_consistency_features(
                split,
                node_idx,
                scale_px=scale_px,
                include_joint=use_joint_features,
            )
        )
    if use_soft_neighbour_features:
        pieces.append(
            soft_neighbour_route_features(
                split,
                node_idx,
                scale_px=scale_px,
                source_temperature=soft_neighbour_temperature,
                source_topk=soft_neighbour_topk,
                route_temperature=soft_neighbour_route_temperature,
            )
        )

    out = np.concatenate(pieces, axis=2)
    return np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32, copy=False)


def standardize_features(
    train_x: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(flat.std(axis=0, keepdims=True).astype(np.float32), 1e-5)

    def norm(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - mean) / std, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

    return norm(train_x), [norm(x) for x in others]


@torch.no_grad()
def score_reranker(
    model: CandidateReranker,
    features: np.ndarray,
    device: torch.device,
    *,
    batch_nodes: int,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, features.shape[0], int(batch_nodes)):
        x = torch.from_numpy(features[start : start + batch_nodes]).to(device)
        chunks.append(model(x).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_reranker_scores(
    split: CandidateSplit,
    node_idx: np.ndarray,
    scores: np.ndarray,
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    cand = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2)
    y = split.y_px[node_idx]
    top1_idx = np.argmax(scores, axis=1)
    top1 = cand[np.arange(len(node_idx)), top1_idx]
    top3_idx = np.argsort(-scores, axis=1)[:, : min(3, scores.shape[1])]
    top3 = cand[np.arange(len(node_idx))[:, None], top3_idx].mean(axis=1)
    shifted = (scores - scores.max(axis=1, keepdims=True)) / max(float(temperature), 1e-3)
    weights = np.exp(np.clip(shifted, -40.0, 20.0))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    soft = np.sum(weights[:, :, None] * cand, axis=1)
    labels = candidate_argmin_labels(split, node_idx)
    acc = float(np.mean(top1_idx == labels))
    return {
        "reranker_top1_rmse_px": float(np.sqrt(np.mean(np.sum(np.square(top1 - y), axis=1)))),
        "reranker_top3_mean_rmse_px": float(np.sqrt(np.mean(np.sum(np.square(top3 - y), axis=1)))),
        "reranker_softmax_rmse_px": float(np.sqrt(np.mean(np.sum(np.square(soft - y), axis=1)))),
        "reranker_top1_r2_vec": vector_r2_from_arrays(y, top1),
        "reranker_top3_mean_r2_vec": vector_r2_from_arrays(y, top3),
        "reranker_softmax_r2_vec": vector_r2_from_arrays(y, soft),
        "reranker_oracle_candidate_acc": acc,
    }


def tune_softmax_temperature(
    split: CandidateSplit,
    node_idx: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    best_temp = 1.0
    best_rmse = float("inf")
    for temp in (0.15, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        rmse = evaluate_reranker_scores(split, node_idx, scores, temperature=temp)[
            "reranker_softmax_rmse_px"
        ]
        if rmse < best_rmse:
            best_temp = float(temp)
            best_rmse = float(rmse)
    return best_temp, best_rmse


def hand_prior_score_matrices(
    split: CandidateSplit,
    node_idx: np.ndarray,
    *,
    scale_px: float,
) -> dict[str, np.ndarray]:
    cfg = configuration_candidate_features(split, node_idx, scale_px=scale_px)
    cand = split.pack.values_px[:, node_idx, :].transpose(1, 0, 2).astype(np.float32)
    proposal = split.proposal_px[node_idx].astype(np.float32)
    offset_prop = cand - proposal[:, None, :]

    delta_e = cfg[:, :, 2]
    c_future = cfg[:, :, 3]
    collision = cfg[:, :, 4]
    abs_stretch = cfg[:, :, 1]
    motion_mismatch = cfg[:, :, 6]
    motion_cos = cfg[:, :, 7]

    force = split.fields["force"][node_idx].astype(np.float32)
    force_shuffled = split.fields["force_shuffled"][node_idx].astype(np.float32)
    c_radial = split.fields["c_radial"][node_idx].astype(np.float32)
    c_radial_shuffled = split.fields["c_radial_shuffled"][node_idx].astype(np.float32)
    closing = split.fields["closing"][node_idx].astype(np.float32)

    force_align = cos_feature(offset_prop, force[:, None, :]).squeeze(-1)
    force_flipped_align = -force_align
    force_shuffled_align = cos_feature(offset_prop, force_shuffled[:, None, :]).squeeze(-1)
    c_radial_align = cos_feature(offset_prop, c_radial[:, None, :]).squeeze(-1)
    c_radial_shuffled_align = cos_feature(offset_prop, c_radial_shuffled[:, None, :]).squeeze(-1)
    closing_align = cos_feature(offset_prop, closing[:, None, :]).squeeze(-1)

    # Higher score is better.  DeltaE is E_future - E_current for E=-c(r), so
    # lower deltaE means a more prior-compatible future local configuration.
    return {
        "hand_deltaE": -delta_e,
        "hand_future_c": c_future,
        "hand_deltaE_collision": -delta_e - 0.50 * collision,
        "hand_config_combo": (
            -delta_e
            - 0.50 * collision
            - 0.15 * abs_stretch
            - 0.10 * motion_mismatch
            + 0.10 * motion_cos
        ),
        "hand_force_align": force_align - 0.25 * collision,
        "hand_force_flipped_control": force_flipped_align - 0.25 * collision,
        "hand_force_shuffled_control": force_shuffled_align - 0.25 * collision,
        "hand_c_radial_align": c_radial_align - 0.25 * collision,
        "hand_c_radial_shuffled_control": c_radial_shuffled_align - 0.25 * collision,
        "hand_closing_align": closing_align - 0.25 * collision,
    }


def evaluate_hand_prior_scorers(
    val_split: CandidateSplit,
    test_split: CandidateSplit,
    *,
    scale_px: float,
    max_val_nodes: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    val_idx = choose_target_nodes(val_split.mask, max_val_nodes, seed + 11)
    test_idx = np.flatnonzero(test_split.mask).astype(np.int64, copy=False)
    val_scores = hand_prior_score_matrices(val_split, val_idx, scale_px=scale_px)
    test_scores = hand_prior_score_matrices(test_split, test_idx, scale_px=scale_px)
    base = vector_rmse(test_split.y_px, test_split.self_flow_px, test_split.mask)
    rows: list[dict[str, Any]] = []
    for name, scores in test_scores.items():
        temp, val_rmse = tune_softmax_temperature(val_split, val_idx, val_scores[name])
        metrics = evaluate_reranker_scores(test_split, test_idx, scores, temperature=temp)
        rows.append(
            {
                "hand_scorer": name,
                "temperature": temp,
                "val_softmax_rmse_px": val_rmse,
                "top1_rmse_px": metrics["reranker_top1_rmse_px"],
                "top1_r2_vec": metrics["reranker_top1_r2_vec"],
                "top3_mean_rmse_px": metrics["reranker_top3_mean_rmse_px"],
                "top3_mean_r2_vec": metrics["reranker_top3_mean_r2_vec"],
                "softmax_rmse_px": metrics["reranker_softmax_rmse_px"],
                "softmax_r2_vec": metrics["reranker_softmax_r2_vec"],
                "softmax_gain_vs_self_flow_pct": gain_pct(base, metrics["reranker_softmax_rmse_px"]),
                "oracle_candidate_acc": metrics["reranker_oracle_candidate_acc"],
            }
        )
    df = pd.DataFrame(rows).sort_values("softmax_rmse_px").reset_index(drop=True)
    best = df.iloc[0].to_dict() if len(df) else {}
    summary = {f"hand_prior_best_{k}": v for k, v in best.items()}
    return df, summary


def train_candidate_reranker(
    train_split: CandidateSplit,
    val_split: CandidateSplit,
    test_split: CandidateSplit,
    *,
    scale_px: float,
    seed: int,
    device: torch.device,
    epochs: int,
    max_train_nodes: int,
    max_val_nodes: int,
    hidden_dim: int,
    lr: float,
    batch_nodes: int,
    loss_mode: str,
    use_config_features: bool,
    use_joint_features: bool,
    use_backward_features: bool,
    use_soft_neighbour_features: bool,
    soft_neighbour_temperature: float,
    soft_neighbour_topk: int,
    soft_neighbour_route_temperature: float,
    soft_oracle_temperature: float,
    soft_oracle_topk: int,
    contrastive_margin: float,
    model_type: str,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + 90_000)
    family_to_idx = {fam: idx for idx, fam in enumerate(sorted(set(train_split.pack.families)))}
    train_idx = choose_target_nodes(train_split.mask, max_train_nodes, seed + 1)
    val_idx = choose_target_nodes(val_split.mask, max_val_nodes, seed + 2)
    test_idx = np.flatnonzero(test_split.mask).astype(np.int64, copy=False)

    train_x = build_reranker_features(
        train_split,
        train_idx,
        family_to_idx,
        scale_px=scale_px,
        use_config_features=use_config_features,
        use_joint_features=use_joint_features,
        use_backward_features=use_backward_features,
        use_soft_neighbour_features=use_soft_neighbour_features,
        soft_neighbour_temperature=soft_neighbour_temperature,
        soft_neighbour_topk=soft_neighbour_topk,
        soft_neighbour_route_temperature=soft_neighbour_route_temperature,
    )
    val_x = build_reranker_features(
        val_split,
        val_idx,
        family_to_idx,
        scale_px=scale_px,
        use_config_features=use_config_features,
        use_joint_features=use_joint_features,
        use_backward_features=use_backward_features,
        use_soft_neighbour_features=use_soft_neighbour_features,
        soft_neighbour_temperature=soft_neighbour_temperature,
        soft_neighbour_topk=soft_neighbour_topk,
        soft_neighbour_route_temperature=soft_neighbour_route_temperature,
    )
    test_x = build_reranker_features(
        test_split,
        test_idx,
        family_to_idx,
        scale_px=scale_px,
        use_config_features=use_config_features,
        use_joint_features=use_joint_features,
        use_backward_features=use_backward_features,
        use_soft_neighbour_features=use_soft_neighbour_features,
        soft_neighbour_temperature=soft_neighbour_temperature,
        soft_neighbour_topk=soft_neighbour_topk,
        soft_neighbour_route_temperature=soft_neighbour_route_temperature,
    )
    train_x, (val_x, test_x) = standardize_features(train_x, val_x, test_x)
    train_y = candidate_argmin_labels(train_split, train_idx)
    val_y = candidate_argmin_labels(val_split, val_idx)
    train_cand = train_split.pack.values_px[:, train_idx, :].transpose(1, 0, 2).astype(np.float32) / float(scale_px)
    val_cand = val_split.pack.values_px[:, val_idx, :].transpose(1, 0, 2).astype(np.float32) / float(scale_px)
    train_target = train_split.y_px[train_idx].astype(np.float32) / float(scale_px)
    val_target = val_split.y_px[val_idx].astype(np.float32) / float(scale_px)
    train_soft, soft_stats = candidate_soft_oracle_targets(
        train_cand,
        train_target,
        temperature=soft_oracle_temperature,
        topk=soft_oracle_topk,
    )
    val_soft, _ = candidate_soft_oracle_targets(
        val_cand,
        val_target,
        temperature=soft_oracle_temperature,
        topk=soft_oracle_topk,
    )

    if model_type == "set":
        model = CandidateSetReranker(train_x.shape[-1], hidden_dim=int(hidden_dim), layers=2, heads=4).to(device)
    else:
        model = CandidateReranker(train_x.shape[-1], hidden_dim=int(hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    rng = np.random.default_rng(int(seed) + 3)
    batch_nodes = int(max(batch_nodes, 64))

    val_tensor = torch.from_numpy(val_x).to(device)
    val_labels = torch.from_numpy(val_y).long().to(device)
    val_cand_tensor = torch.from_numpy(val_cand).to(device)
    val_target_tensor = torch.from_numpy(val_target).to(device)
    val_soft_tensor = torch.from_numpy(val_soft).to(device)

    def reranker_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        cand: torch.Tensor,
        target: torch.Tensor,
        soft_target: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, label_smoothing=0.03)
        soft_nce = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        weights = torch.softmax(logits, dim=1)
        pred = torch.sum(weights.unsqueeze(-1) * cand, dim=1)
        mix = F.mse_loss(pred, target)
        pos = logits.gather(1, labels[:, None])
        neg_logits = logits.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), -1.0e9)
        hard_neg = torch.max(neg_logits, dim=1, keepdim=True).values
        rank = F.relu(float(contrastive_margin) - pos + hard_neg).mean()
        if loss_mode == "ce":
            return ce
        if loss_mode == "soft":
            return mix + 0.20 * soft_nce
        if loss_mode == "contrastive":
            return mix + 0.20 * soft_nce + 0.05 * rank
        if loss_mode == "rank":
            return mix + 0.08 * ce + 0.10 * rank
        if loss_mode == "hybrid":
            return mix + 0.05 * ce
        return mix

    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), batch_nodes):
            idx = order[start : start + batch_nodes]
            xb = torch.from_numpy(train_x[idx]).to(device)
            yb = torch.from_numpy(train_y[idx]).long().to(device)
            cb = torch.from_numpy(train_cand[idx]).to(device)
            tb = torch.from_numpy(train_target[idx]).to(device)
            qb = torch.from_numpy(train_soft[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = reranker_loss(model(xb), yb, cb, tb, qb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(
                reranker_loss(
                    model(val_tensor),
                    val_labels,
                    val_cand_tensor,
                    val_target_tensor,
                    val_soft_tensor,
                ).detach().cpu()
            )
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = score_reranker(model, val_x, device, batch_nodes=batch_nodes)
    best_temp, best_val_softmax_rmse = tune_softmax_temperature(val_split, val_idx, val_scores)
    scores = score_reranker(model, test_x, device, batch_nodes=batch_nodes)
    metrics = evaluate_reranker_scores(test_split, test_idx, scores, temperature=best_temp)
    base = vector_rmse(test_split.y_px, test_split.self_flow_px, test_split.mask)
    metrics.update(
        {
            "reranker_best_epoch": float(best_epoch),
            "reranker_best_val_ce": float(best_val),
            "reranker_train_nodes": float(len(train_idx)),
            "reranker_val_nodes": float(len(val_idx)),
            "reranker_feature_dim": float(train_x.shape[-1]),
            "reranker_loss_mode": loss_mode,
            "reranker_model": model_type,
            "reranker_config_features": bool(use_config_features),
            "reranker_joint_features": bool(use_joint_features),
            "reranker_backward_features": bool(use_backward_features),
            "reranker_soft_neighbour_features": bool(use_soft_neighbour_features),
            "reranker_soft_neighbour_temperature": float(soft_neighbour_temperature),
            "reranker_soft_neighbour_topk": float(soft_neighbour_topk),
            "reranker_soft_neighbour_route_temperature": float(soft_neighbour_route_temperature),
            "reranker_soft_oracle_temperature": float(soft_oracle_temperature),
            "reranker_soft_oracle_topk": float(soft_oracle_topk),
            "reranker_contrastive_margin": float(contrastive_margin),
            "reranker_softmax_temperature": float(best_temp),
            "reranker_val_softmax_rmse_px": float(best_val_softmax_rmse),
            "reranker_top1_gain_vs_self_flow_pct": gain_pct(base, metrics["reranker_top1_rmse_px"]),
            "reranker_top3_mean_gain_vs_self_flow_pct": gain_pct(base, metrics["reranker_top3_mean_rmse_px"]),
            "reranker_softmax_gain_vs_self_flow_pct": gain_pct(base, metrics["reranker_softmax_rmse_px"]),
            **{f"reranker_{k}": v for k, v in soft_stats.items()},
        }
    )
    return metrics


def prior_gradient_sanity(graph: arch.GraphTensors, fields: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    force = fields["force"]
    flipped = fields["force_sign_flipped"]
    c_field = fields["c_radial"]
    rel = fields["rel_velocity"]
    shear = fields["shear"]
    dot = np.sum(force[mask] * flipped[mask], axis=1)
    denom = np.maximum(
        np.linalg.norm(force[mask], axis=1) * np.linalg.norm(flipped[mask], axis=1),
        1e-8,
    )
    out = {
        "force_norm_mean": float(np.mean(np.linalg.norm(force[mask], axis=1))),
        "force_norm_p90": float(np.quantile(np.linalg.norm(force[mask], axis=1), 0.9)),
        "c_field_norm_mean": float(np.mean(np.linalg.norm(c_field[mask], axis=1))),
        "rel_velocity_field_norm_mean_px": float(np.mean(np.linalg.norm(rel[mask], axis=1))),
        "shear_field_norm_mean_px": float(np.mean(np.linalg.norm(shear[mask], axis=1))),
        "force_vs_sign_flipped_cosine": float(np.mean(dot / denom)),
        "target_nodes": int(mask.sum()),
        "edges": int(len(graph.src)),
    }
    return out


def finite_difference_prior_check() -> dict[str, float]:
    def c_fn(r: float) -> float:
        return math.exp(-0.5 * (r / 50.0) ** 2)

    def energy(dst_x: np.ndarray, src_x: np.ndarray) -> float:
        r = float(np.linalg.norm(src_x - dst_x))
        return -c_fn(r)

    rng = np.random.default_rng(123)
    errors = []
    cosines = []
    for _ in range(128):
        src = rng.normal(size=2) * 25.0
        dst = rng.normal(size=2) * 25.0
        rel = src - dst
        r = max(float(np.linalg.norm(rel)), 1e-6)
        radial = rel / r
        force = (r / (50.0**2)) * math.exp(-0.5 * (r / 50.0) ** 2)
        # F = -grad_dst E for E=-c(r)
        analytic_force = force * radial
        eps = 1e-3
        grad = np.zeros(2)
        for d in range(2):
            step = np.zeros(2)
            step[d] = eps
            grad[d] = (energy(dst + step, src) - energy(dst - step, src)) / (2 * eps)
        numeric_force = -grad
        errors.append(float(np.linalg.norm(analytic_force - numeric_force)))
        denom = max(float(np.linalg.norm(analytic_force) * np.linalg.norm(numeric_force)), 1e-12)
        cosines.append(float(np.dot(analytic_force, numeric_force) / denom))
    return {
        "finite_diff_force_error_mean": float(np.mean(errors)),
        "finite_diff_force_error_p95": float(np.quantile(errors, 0.95)),
        "finite_diff_force_cosine_mean": float(np.mean(cosines)),
    }


@torch.no_grad()
def make_candidate_split(
    *,
    model: ng.EquivariantMPDecoder,
    graph: arch.GraphTensors,
    enc: ng.EncodedBase,
    norm: arch.Normalizer,
    horizon: int,
    train_target_median: float,
    seed: int,
    sobol_count: int,
    gaussian_count: int,
) -> CandidateSplit:
    proposal_norm, _, _, _ = ng.predict_mp_components(model, graph, enc)
    mask = graph.target_valid.detach().cpu().numpy()
    y_px = graph.y_px.detach().cpu().numpy()
    self_px = arch.to_px(enc.self_pred, norm)
    self_flow_px = arch.to_px(enc.self_pred + enc.flow_pred, norm)
    proposal_px = arch.to_px(proposal_norm, norm)
    fields = prior_vector_fields(graph, norm, horizon)
    pack = build_candidates(
        y_px=y_px,
        mask=mask,
        self_px=self_px,
        self_flow_px=self_flow_px,
        proposal_px=proposal_px,
        fields=fields,
        train_target_median_px=train_target_median,
        seed=seed,
        sobol_count=sobol_count,
        gaussian_count=gaussian_count,
    )
    return CandidateSplit(
        graph=graph,
        mask=mask,
        y_px=y_px,
        self_px=self_px,
        self_flow_px=self_flow_px,
        proposal_px=proposal_px,
        fields=fields,
        pack=pack,
    )


def run_one(
    cell_type: str,
    *,
    variant: str,
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
    k: int,
    seed: int,
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
    sobol_count: int,
    gaussian_count: int,
    train_reranker: bool,
    reranker_epochs: int,
    reranker_train_nodes: int,
    reranker_val_nodes: int,
    reranker_hidden_dim: int,
    reranker_lr: float,
    reranker_batch_nodes: int,
    reranker_loss: str,
    reranker_model: str,
    soft_oracle_temperature: float,
    soft_oracle_topk: int,
    contrastive_margin: float,
    config_critic_features: bool,
    joint_critic_features: bool,
    backward_critic_features: bool,
    soft_neighbour_critic_features: bool,
    soft_neighbour_temperature: float,
    soft_neighbour_topk: int,
    soft_neighbour_route_temperature: float,
    hand_prior_scorer: bool,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
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
    graphs, norm, coverage = la.prepare_dataset(cell_type, raw, meta, horizon=horizon, k=k, device=device)
    print(f"[{cell_type}] seed={seed} temporal", flush=True)
    temporal, temporal_info = arch.train_temporal(
        graphs["train"],
        graphs["val"],
        seed=seed,
        epochs=temporal_epochs,
        batch_size=batch_size,
        sequence_balanced_loss=sequence_balanced_loss,
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
    encoded = {split: ng.encode_all(temporal, flow, graph) for split, graph in graphs.items()}
    print(f"[{cell_type}] seed={seed} proposal {variant}", flush=True)
    model, model_info = ng.train_mp_decoder(
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
    train_mask = graphs["train"].target_valid.detach().cpu().numpy()
    train_y = graphs["train"].y_px.detach().cpu().numpy()
    train_target_median = float(np.median(np.linalg.norm(train_y[train_mask], axis=1)))
    train_target_median = max(train_target_median, 1.0)

    test_split = make_candidate_split(
        model=model,
        graph=graphs["test"],
        enc=encoded["test"],
        norm=norm,
        horizon=horizon,
        train_target_median=train_target_median,
        seed=seed + 101,
        sobol_count=sobol_count,
        gaussian_count=gaussian_count,
    )
    base_rmse = vector_rmse(test_split.y_px, test_split.self_flow_px, test_split.mask)
    proposal_rmse = vector_rmse(test_split.y_px, test_split.proposal_px, test_split.mask)
    by_candidate, by_family, oracle_extra = candidate_metrics(
        test_split.y_px,
        test_split.mask,
        test_split.pack,
        base_rmse=base_rmse,
    )
    reranker_extra: dict[str, Any] = {}
    hand_prior_extra: dict[str, Any] = {}
    hand_prior_df = pd.DataFrame()
    train_split: CandidateSplit | None = None
    val_split: CandidateSplit | None = None
    if train_reranker or hand_prior_scorer:
        train_split = make_candidate_split(
            model=model,
            graph=graphs["train"],
            enc=encoded["train"],
            norm=norm,
            horizon=horizon,
            train_target_median=train_target_median,
            seed=seed + 201,
            sobol_count=sobol_count,
            gaussian_count=gaussian_count,
        )
        val_split = make_candidate_split(
            model=model,
            graph=graphs["val"],
            enc=encoded["val"],
            norm=norm,
            horizon=horizon,
            train_target_median=train_target_median,
            seed=seed + 301,
            sobol_count=sobol_count,
            gaussian_count=gaussian_count,
        )
    if hand_prior_scorer:
        assert val_split is not None
        print(f"[{cell_type}] seed={seed} hand-prior-scorer", flush=True)
        hand_prior_df, hand_prior_extra = evaluate_hand_prior_scorers(
            val_split,
            test_split,
            scale_px=train_target_median,
            max_val_nodes=reranker_val_nodes,
            seed=seed,
        )
    if train_reranker:
        assert train_split is not None and val_split is not None
        print(f"[{cell_type}] seed={seed} reranker", flush=True)
        reranker_extra = train_candidate_reranker(
            train_split,
            val_split,
            test_split,
            scale_px=train_target_median,
            seed=seed,
            device=device,
            epochs=reranker_epochs,
            max_train_nodes=reranker_train_nodes,
            max_val_nodes=reranker_val_nodes,
            hidden_dim=reranker_hidden_dim,
            lr=reranker_lr,
            batch_nodes=reranker_batch_nodes,
            loss_mode=reranker_loss,
            use_config_features=config_critic_features,
            use_joint_features=joint_critic_features,
            use_backward_features=backward_critic_features,
            use_soft_neighbour_features=soft_neighbour_critic_features,
            soft_neighbour_temperature=soft_neighbour_temperature,
            soft_neighbour_topk=soft_neighbour_topk,
            soft_neighbour_route_temperature=soft_neighbour_route_temperature,
            soft_oracle_temperature=soft_oracle_temperature,
            soft_oracle_topk=soft_oracle_topk,
            contrastive_margin=contrastive_margin,
            model_type=reranker_model,
        )
    for frame in (by_candidate, by_family):
        frame.insert(0, "dataset", cell_type)
        frame.insert(1, "seed", seed)
        frame.insert(2, "proposal_variant", variant)
    if not hand_prior_df.empty:
        hand_prior_df.insert(0, "dataset", cell_type)
        hand_prior_df.insert(1, "seed", seed)
        hand_prior_df.insert(2, "proposal_variant", variant)
    best_row = by_family.loc[by_family["family"].eq("group:all")].iloc[0].to_dict()
    summary = {
        "dataset": cell_type,
        "seed": seed,
        "proposal_variant": variant,
        "self_flow_rmse_px": base_rmse,
        "self_flow_r2_vec": vector_r2(test_split.y_px, test_split.self_flow_px, test_split.mask),
        "proposal_rmse_px": proposal_rmse,
        "proposal_r2_vec": vector_r2(test_split.y_px, test_split.proposal_px, test_split.mask),
        "proposal_gain_vs_self_flow_pct": gain_pct(base_rmse, proposal_rmse),
        "train_target_median_mag_px": train_target_median,
        **oracle_extra,
        **{f"all_{k0}": v0 for k0, v0 in best_row.items() if k0 not in {"dataset", "seed", "proposal_variant", "family"}},
        **prior_gradient_sanity(test_split.graph, test_split.fields, test_split.mask),
        **hand_prior_extra,
        **reranker_extra,
        **finite_difference_prior_check(),
        **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
        **{f"flow_{k0}": v for k0, v in flow_info.items()},
        **{f"proposal_{k0}": v for k0, v in model_info.items()},
    }
    return by_candidate, by_family, hand_prior_df, summary, coverage


def plot_family(by_family: pd.DataFrame, out_dir: Path) -> None:
    if by_family.empty:
        return
    plot_df = by_family[by_family["family"].str.startswith("group:")].copy()
    if plot_df.empty:
        plot_df = by_family.copy()
    plot_df["label"] = plot_df["dataset"] + "\n" + plot_df["family"]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(plot_df)), 4.8))
    ax.bar(plot_df["label"], plot_df["oracle_gain_vs_self_flow_pct"], color="#315f9c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Oracle gain vs self_flow, %")
    ax.set_title("Candidate oracle coverage by family/group")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_candidate_oracle_family_gain.png", dpi=180)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    by_family: pd.DataFrame,
    by_candidate: pd.DataFrame,
    coverage: dict[str, Any],
    out_dir: Path,
) -> None:
    lines = [
        "# LaChance candidate oracle report",
        "",
        "This is a pre-reranker gate.  Oracle rows use true future only to measure candidate coverage; they are not deployable inference results.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Oracle families",
        "",
        by_family.head(80).to_markdown(index=False),
        "",
        "## Best candidates",
        "",
        by_candidate.head(80).to_markdown(index=False),
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(finite_json(coverage), indent=2, ensure_ascii=False),
        "```",
    ]
    (out_dir / "lachance_candidate_oracle_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-types", type=str, default="MDCK_Bulk,MDCK_Edge")
    parser.add_argument("--variant", type=str, default="auto")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-mode", choices=["movie", "frame"], default="movie")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-movies", type=int, default=8)
    parser.add_argument("--max-tracks-per-movie", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--crop-fraction", type=float, default=0.08)
    parser.add_argument("--r-cut-px", type=float, default=50.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--temporal-epochs", type=int, default=35)
    parser.add_argument("--flow-epochs", type=int, default=25)
    parser.add_argument("--mp-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sequence-balanced-loss", action="store_true")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=72)
    parser.add_argument("--edge-hidden-dim", type=int, default=56)
    parser.add_argument("--max-delta-norm", type=float, default=1.35)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--social-l2", type=float, default=0.0015)
    parser.add_argument("--flow-gate-l2", type=float, default=0.0005)
    parser.add_argument("--sobol-count", type=int, default=8)
    parser.add_argument("--gaussian-count", type=int, default=8)
    parser.add_argument("--train-reranker", action="store_true")
    parser.add_argument("--reranker-epochs", type=int, default=24)
    parser.add_argument("--reranker-train-nodes", type=int, default=25000)
    parser.add_argument("--reranker-val-nodes", type=int, default=12000)
    parser.add_argument("--reranker-hidden-dim", type=int, default=128)
    parser.add_argument("--reranker-lr", type=float, default=1.0e-3)
    parser.add_argument("--reranker-batch-nodes", type=int, default=512)
    parser.add_argument("--reranker-loss", choices=["ce", "mix", "hybrid", "soft", "contrastive", "rank"], default="mix")
    parser.add_argument("--reranker-model", choices=["mlp", "set"], default="mlp")
    parser.add_argument("--soft-oracle-temperature", type=float, default=0.10)
    parser.add_argument("--soft-oracle-topk", type=int, default=8)
    parser.add_argument("--contrastive-margin", type=float, default=0.25)
    parser.add_argument("--config-critic-features", action="store_true")
    parser.add_argument("--joint-critic-features", action="store_true")
    parser.add_argument("--backward-critic-features", action="store_true")
    parser.add_argument(
        "--soft-neighbour-critic-features",
        "--soft-neighbor-critic-features",
        dest="soft_neighbour_critic_features",
        action="store_true",
    )
    parser.add_argument(
        "--soft-neighbour-temperature",
        "--soft-neighbor-temperature",
        dest="soft_neighbour_temperature",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--soft-neighbour-topk",
        "--soft-neighbor-topk",
        dest="soft_neighbour_topk",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--soft-neighbour-route-temperature",
        "--soft-neighbor-route-temperature",
        dest="soft_neighbour_route_temperature",
        type=float,
        default=0.50,
    )
    parser.add_argument("--hand-prior-scorer", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_movies = min(args.max_movies, 3)
        args.crop_fraction = min(args.crop_fraction, 0.02)
        args.temporal_epochs = min(args.temporal_epochs, 4)
        args.flow_epochs = min(args.flow_epochs, 4)
        args.mp_epochs = min(args.mp_epochs, 5)
        args.sobol_count = min(args.sobol_count, 4)
        args.gaussian_count = min(args.gaussian_count, 4)
        args.hidden_dim = min(args.hidden_dim, 40)
        args.edge_hidden_dim = min(args.edge_hidden_dim, 32)
        args.reranker_epochs = min(args.reranker_epochs, 4)
        args.reranker_train_nodes = min(args.reranker_train_nodes, 2000)
        args.reranker_val_nodes = min(args.reranker_val_nodes, 1000)
        args.reranker_hidden_dim = min(args.reranker_hidden_dim, 64)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in parse_list(args.seeds)]
    all_candidates: list[pd.DataFrame] = []
    all_families: list[pd.DataFrame] = []
    all_hand_prior: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for cell_type in parse_list(args.cell_types):
        variant = default_variant(cell_type) if args.variant == "auto" else args.variant
        for seed in seeds:
            by_candidate, by_family, hand_prior, summary, cov = run_one(
                cell_type,
                variant=variant,
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
                k=args.k,
                seed=seed,
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
                sobol_count=args.sobol_count,
                gaussian_count=args.gaussian_count,
                train_reranker=args.train_reranker,
                reranker_epochs=args.reranker_epochs,
                reranker_train_nodes=args.reranker_train_nodes,
                reranker_val_nodes=args.reranker_val_nodes,
                reranker_hidden_dim=args.reranker_hidden_dim,
                reranker_lr=args.reranker_lr,
                reranker_batch_nodes=args.reranker_batch_nodes,
                reranker_loss=args.reranker_loss,
                reranker_model=args.reranker_model,
                soft_oracle_temperature=args.soft_oracle_temperature,
                soft_oracle_topk=args.soft_oracle_topk,
                contrastive_margin=args.contrastive_margin,
                config_critic_features=args.config_critic_features,
                joint_critic_features=args.joint_critic_features,
                backward_critic_features=args.backward_critic_features,
                soft_neighbour_critic_features=args.soft_neighbour_critic_features,
                soft_neighbour_temperature=args.soft_neighbour_temperature,
                soft_neighbour_topk=args.soft_neighbour_topk,
                soft_neighbour_route_temperature=args.soft_neighbour_route_temperature,
                hand_prior_scorer=args.hand_prior_scorer,
                device=device,
            )
            all_candidates.append(by_candidate)
            all_families.append(by_family)
            if not hand_prior.empty:
                all_hand_prior.append(hand_prior)
            summaries.append(summary)
            coverage[f"{cell_type}_seed{seed}"] = cov
            print(
                f"[{cell_type}] seed={seed}: self_flow={summary['self_flow_rmse_px']:.4f} "
                f"proposal={summary['proposal_rmse_px']:.4f} "
                f"oracle_all={summary['all_oracle_rmse_px']:.4f} "
                f"gain={summary['all_oracle_gain_vs_self_flow_pct']:.2f}%",
                flush=True,
            )
    by_candidate_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    by_family_df = pd.concat(all_families, ignore_index=True) if all_families else pd.DataFrame()
    hand_prior_df = pd.concat(all_hand_prior, ignore_index=True) if all_hand_prior else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    by_candidate_df.to_csv(args.out_dir / "candidate_metrics.csv", index=False)
    by_family_df.to_csv(args.out_dir / "candidate_family_oracle.csv", index=False)
    if not hand_prior_df.empty:
        hand_prior_df.to_csv(args.out_dir / "hand_prior_scorer_metrics.csv", index=False)
    summary_df.to_csv(args.out_dir / "candidate_oracle_summary.csv", index=False)
    (args.out_dir / "coverage.json").write_text(
        json.dumps(finite_json(coverage), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_family(by_family_df, args.out_dir)
    write_report(summary_df, by_family_df, by_candidate_df, coverage, args.out_dir)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
