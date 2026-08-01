#!/usr/bin/env python3
"""Visual-state conditioned route validator v38.

This runner tests the next video-aware hypothesis:

    fixed route-expert candidate trajectories
    + reliable visual/cell state
    + coordinate/velocity/flow context
    -> route validation network
    -> calibrated route weights
    -> final h1/h2/h4/h6 residual prediction

The target is used only to create train-time route teacher labels and losses.
Inference features are causal: route candidates, coordinate context and
tracking-aligned visual state.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_visual_state_target_v32 as v32  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "visual_state_route_validator_v38_2026-07-08"
DEFAULT_ALIGNED_ROOT = ROOT / "outputs" / "visual_temporal_target_v37_aligned_bulk_seed42_2026-07-07"
DEFAULT_FEATURES = DEFAULT_ALIGNED_ROOT / "raw_context_aligned_extracted_4784.csv"
DEFAULT_SEGF = DEFAULT_ALIGNED_ROOT / "seg_foundation_aligned_pointbox_as_segf.csv"
EPS = 1e-8


@dataclass
class FamilyBlocks:
    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    names: dict[str, list[str]]


@dataclass
class TeacherLabels:
    soft_train: np.ndarray
    soft_val: np.ndarray
    soft_test: np.ndarray
    hard_train: np.ndarray
    hard_val: np.ndarray
    hard_test: np.ndarray
    err_train: np.ndarray
    err_val: np.ndarray
    err_test: np.ndarray


def parse_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return audit.parse_ints(text)


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in parse_csv(text)]


def finite_json(x: Any) -> Any:
    return audit.finite_json(x)


def safe_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def standardize_triplet(
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    *,
    clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xtr.shape[1] == 0:
        return xtr, xva, xte
    sc = StandardScaler()
    ztr = sc.fit_transform(safe_matrix(xtr)).astype(np.float32)
    zva = sc.transform(safe_matrix(xva)).astype(np.float32)
    zte = sc.transform(safe_matrix(xte)).astype(np.float32)
    return (
        np.clip(np.nan_to_num(ztr), -clip, clip).astype(np.float32),
        np.clip(np.nan_to_num(zva), -clip, clip).astype(np.float32),
        np.clip(np.nan_to_num(zte), -clip, clip).astype(np.float32),
    )


def select_indices(names: list[str], tokens: tuple[str, ...], *, prefixes: tuple[str, ...] = ()) -> list[int]:
    out = []
    toks = tuple(t.lower() for t in tokens)
    prefs = tuple(p.lower() for p in prefixes)
    for i, name in enumerate(names):
        low = str(name).lower()
        if prefs and low.startswith(prefs):
            out.append(i)
        elif toks and any(t in low for t in toks):
            out.append(i)
    return list(dict.fromkeys(out))


def take_cols(x: np.ndarray, idx: list[int], max_cols: int) -> np.ndarray:
    if not idx:
        return np.zeros((len(x), 0), dtype=np.float32)
    idx = idx[: int(max_cols)]
    return safe_matrix(x[:, idx])


def packet_state_from_coord_plus(
    packets: dict[str, v32.Packet], name: str, coord_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    p = packets[name]
    return (
        p.train[:, coord_dim:].astype(np.float32),
        p.val[:, coord_dim:].astype(np.float32),
        p.test[:, coord_dim:].astype(np.float32),
        p.feature_names[coord_dim:],
    )


def filter_visual_cols(names: list[str], mode: str) -> list[int]:
    if mode == "all":
        return list(range(len(names)))
    groups = {
        "shape": ("area", "perimeter", "eccentric", "solidity", "extent", "diameter", "major", "minor"),
        "polarity": ("orient", "centroid", "front", "back", "left", "right", "balance", "intensity", "grad"),
        "history": ("lag", "delta", "mean_w", "std_w", "span_delta", "track", "age", "stability", "ema", "iou", "xor", "aligned", "available", "reliability"),
        "polarity_history": (
            "orient",
            "centroid",
            "front",
            "back",
            "left",
            "right",
            "balance",
            "intensity",
            "grad",
            "lag",
            "delta",
            "mean_w",
            "std_w",
            "span_delta",
            "track",
            "age",
            "stability",
            "ema",
            "reliability",
        ),
        "contact": ("free", "neighbor", "boundary", "nn_dist", "near_mask"),
        "quality": ("center_inside", "sam_", "fallback", "prompt", "box_radius", "extract_ok", "n_masks", "score"),
    }
    if mode.endswith("_only"):
        key = mode[: -len("_only")]
        toks = groups.get(key, ())
        return [i for i, n in enumerate(names) if any(t in n.lower() for t in toks)]
    if mode in groups:
        toks = groups[mode]
        return [i for i, n in enumerate(names) if any(t in n.lower() for t in toks)]
    if mode.startswith("no_"):
        key = mode[len("no_") :]
        toks = groups.get(key, ())
        return [i for i, n in enumerate(names) if not any(t in n.lower() for t in toks)]
    raise ValueError(f"Unknown visual feature mode: {mode}")


def build_family_blocks(
    args: argparse.Namespace,
    packets: dict[str, v32.Packet],
    *,
    visual_variant: str,
    visual_feature_mode: str = "all",
    drop_families: tuple[str, ...] = (),
) -> FamilyBlocks:
    coord = packets["coord_all_context"]
    coord_dim = coord.train.shape[1]
    names = coord.feature_names
    max_cols = int(args.v38_max_family_cols)

    groups: dict[str, list[int]] = {
        "self": select_indices(
            names,
            ("speed", "velocity", "dx", "dy", "accel", "quality", "proposal", "x_norm", "y_norm", "frame_norm"),
        ),
        "flow": select_indices(names, ("flow", "tf_", "diverg", "curl", "shear"), prefixes=("tf_",)),
        "morphology_context": select_indices(names, ("morph", "area", "eccentric", "polarity"), prefixes=("ms_",)),
        "crowding": select_indices(
            names,
            ("density", "neighbor", "neighbour", "degree", "crowd", "contact", "closing", "stretch", "align", "pressure"),
        ),
        "boundary": select_indices(names, ("boundary", "edge", "front", "normal", "tangent")),
        "raw_context": select_indices(names, (), prefixes=("rc_",)),
    }
    # Make sure every coordinate column is available at least through a compact all-context token.
    if int(args.v38_include_all_context_token):
        groups["coord_all"] = list(range(coord_dim))

    train: dict[str, np.ndarray] = {}
    val: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    fname: dict[str, list[str]] = {}
    drop = set(drop_families)
    for family, idx in groups.items():
        if family in drop:
            continue
        xtr = take_cols(coord.train, idx, max_cols)
        xva = take_cols(coord.val, idx, max_cols)
        xte = take_cols(coord.test, idx, max_cols)
        if xtr.shape[1] > 0:
            train[family], val[family], test[family] = standardize_triplet(xtr, xva, xte)
            fname[family] = [names[i] for i in idx[:max_cols]]

    visual_name = {
        "real": "coord_plus_seg_foundation_state",
        "zero": "coord_plus_seg_foundation_state_zero",
        "row_shuffled": "coord_plus_seg_foundation_state_row_shuffled",
        "same_frame_wrong_cell": "coord_plus_seg_foundation_state_same_frame_wrong_cell",
        "time_shuffled": "coord_plus_seg_foundation_state_time_shuffled",
    }.get(visual_variant)
    if visual_name and visual_name in packets:
        xtr, xva, xte, vnames = packet_state_from_coord_plus(packets, visual_name, coord_dim)
        keep = filter_visual_cols(vnames, visual_feature_mode)
        if keep:
            xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
            vnames = [vnames[i] for i in keep]
        else:
            xtr = np.zeros((len(xtr), 0), dtype=np.float32)
            xva = np.zeros((len(xva), 0), dtype=np.float32)
            xte = np.zeros((len(xte), 0), dtype=np.float32)
            vnames = []
        train["visual_state"], val["visual_state"], test["visual_state"] = standardize_triplet(xtr, xva, xte)
        fname["visual_state"] = vnames
    elif visual_variant == "no_visual":
        pass
    else:
        raise ValueError(f"Visual variant {visual_variant!r} is unavailable. Existing packets: {sorted(packets)[:20]}...")

    return FamilyBlocks(train=train, val=val, test=test, names=fname)


def route_endpoint_errors(route_pred: np.ndarray, true_flat: np.ndarray, horizons: list[int], max_h: int) -> np.ndarray:
    n, k, _ = route_pred.shape
    cand = route_pred.reshape(n, k, max_h, 2)
    true = true_flat.reshape(n, max_h, 2)
    vals = np.zeros((n, k), dtype=np.float32)
    for h in horizons:
        h = int(h)
        p = np.sum(cand[:, :, :h, :], axis=2)
        y = np.sum(true[:, :h, :], axis=1)[:, None, :]
        vals += np.sum((p - y) ** 2, axis=-1).astype(np.float32)
    return np.sqrt(vals / float(max(len(horizons), 1))).astype(np.float32)


def soft_labels_from_errors(errors: np.ndarray, tau: float) -> np.ndarray:
    logits = -errors.astype(np.float64) / max(float(tau), 1e-4)
    logits -= np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits)
    p /= np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p.astype(np.float32)


def build_teacher_labels(args: argparse.Namespace, basis: v26.RouteBasis) -> TeacherLabels:
    err_tr = route_endpoint_errors(basis.route_train, basis.y_train, args.horizons, args.max_horizon)
    err_va = route_endpoint_errors(basis.route_val, basis.y_val, args.horizons, args.max_horizon)
    err_te = route_endpoint_errors(basis.route_test, basis.y_test, args.horizons, args.max_horizon)
    tau = float(args.v38_teacher_temperature)
    if tau <= 0:
        tau = float(np.nanmedian(np.std(err_tr, axis=1)) + 1e-3)
    return TeacherLabels(
        soft_train=soft_labels_from_errors(err_tr, tau),
        soft_val=soft_labels_from_errors(err_va, tau),
        soft_test=soft_labels_from_errors(err_te, tau),
        hard_train=np.argmin(err_tr, axis=1).astype(np.int64),
        hard_val=np.argmin(err_va, axis=1).astype(np.int64),
        hard_test=np.argmin(err_te, axis=1).astype(np.int64),
        err_train=err_tr,
        err_val=err_va,
        err_test=err_te,
    )


def candidate_features(route_pred: np.ndarray, probs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n, k, d = route_pred.shape
    steps = route_pred.reshape(n, k, args.max_horizon, 2)
    parts = [route_pred.astype(np.float32)]
    endpoints = []
    for h in args.horizons:
        endpoints.append(np.sum(steps[:, :, : int(h), :], axis=2))
    parts.append(np.concatenate(endpoints, axis=2).astype(np.float32))
    speed = np.linalg.norm(steps, axis=-1)
    accel = np.diff(steps, axis=2)
    accel_norm = np.linalg.norm(accel, axis=-1) if accel.shape[2] else np.zeros((n, k, 1), dtype=np.float32)
    h6 = np.sum(steps[:, :, : args.max_horizon, :], axis=2)
    parts.append(
        np.stack(
            [
                speed.mean(axis=2),
                speed.std(axis=2),
                speed.max(axis=2),
                accel_norm.mean(axis=2),
                np.linalg.norm(h6, axis=-1),
                probs,
                np.log(np.maximum(probs, 1e-7)),
            ],
            axis=2,
        ).astype(np.float32)
    )
    out = np.concatenate(parts, axis=2).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_candidate_features(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k, d = xtr.shape
    sc = StandardScaler()
    ztr = sc.fit_transform(xtr.reshape(-1, d)).reshape(n, k, d)
    zva = sc.transform(xva.reshape(-1, d)).reshape(xva.shape)
    zte = sc.transform(xte.reshape(-1, d)).reshape(xte.shape)
    return (
        np.clip(np.nan_to_num(ztr), -8, 8).astype(np.float32),
        np.clip(np.nan_to_num(zva), -8, 8).astype(np.float32),
        np.clip(np.nan_to_num(zte), -8, 8).astype(np.float32),
    )


class VisualRouteValidator(nn.Module):
    def __init__(
        self,
        *,
        cand_dim: int,
        family_dims: dict[str, int],
        route_count: int,
        out_dim: int,
        hidden: int,
        heads: int,
        layers: int,
        dropout: float,
        use_cross_attention: bool,
        use_candidate_self_attention: bool,
        correction_scale: float,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.route_count = int(route_count)
        self.use_cross_attention = bool(use_cross_attention)
        self.use_candidate_self_attention = bool(use_candidate_self_attention)
        self.correction_scale = float(correction_scale)
        self.cand_proj = nn.Sequential(nn.Linear(cand_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.route_emb = nn.Embedding(route_count, hidden)
        self.family_enc = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
                for name, dim in family_dims.items()
                if dim > 0
            }
        )
        if self.use_cross_attention:
            self.cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(hidden)
        if self.use_candidate_self_attention and layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 3,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.self_attn = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.corr = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.logvar = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, cand: torch.Tensor, families: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k, _ = cand.shape
        route_ids = torch.arange(k, device=cand.device).view(1, k).expand(b, k)
        h = self.cand_proj(cand) + self.route_emb(route_ids)
        state_tokens = []
        for name, enc in self.family_enc.items():
            if name in families:
                state_tokens.append(enc(families[name]).unsqueeze(1))
        if state_tokens:
            state = torch.cat(state_tokens, dim=1)
        else:
            state = torch.zeros((b, 1, self.hidden), device=cand.device, dtype=h.dtype)
        if self.use_cross_attention:
            h2, _ = self.cross(query=h, key=state, value=state, need_weights=False)
            h = self.cross_norm(h + h2)
        if self.use_candidate_self_attention:
            h = self.self_attn(h)
        logits = self.score(h).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        state_pooled = state.mean(dim=1)
        joint = torch.cat([pooled, state_pooled], dim=1)
        corr = self.correction_scale * torch.tanh(self.corr(joint))
        logvar = self.logvar(joint).squeeze(-1)
        return logits, corr, logvar


def tensorize_family_blocks(blocks: FamilyBlocks, split: str, device: torch.device) -> dict[str, torch.Tensor]:
    src = {"train": blocks.train, "val": blocks.val, "test": blocks.test}[split]
    return {k: torch.from_numpy(v).float().to(device) for k, v in src.items() if v.shape[1] > 0}


def endpoint_rmse_flat_np(pred_flat: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace, horizon: int | None = None) -> float:
    pred = pred_flat.reshape(len(pred_flat), args.max_horizon, 2)
    true = true_flat.reshape(len(true_flat), args.max_horizon, 2)
    horizons = [int(horizon)] if horizon is not None else args.horizons
    vals = []
    for h in horizons:
        p = np.sum(pred[:, :h, :], axis=1)
        y = np.sum(true[:, :h, :], axis=1)
        vals.append(np.mean(np.sum((p - y) ** 2, axis=1)))
    return float(np.sqrt(np.mean(vals)))


def route_topk(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-logits, axis=1)[:, : min(k, logits.shape[1])]
    return float(np.mean(np.any(order == labels[:, None], axis=1)))


def ndcg_from_scores(scores: np.ndarray, errors: np.ndarray, k: int) -> float:
    kk = min(int(k), scores.shape[1])
    if kk <= 0:
        return float("nan")
    relevance = 1.0 / (1.0 + np.maximum(errors.astype(np.float64), 0.0))
    order = np.argsort(-scores, axis=1)[:, :kk]
    ideal = np.argsort(-relevance, axis=1)[:, :kk]
    discount = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
    dcg = np.sum(np.take_along_axis(relevance, order, axis=1) * discount[None, :], axis=1)
    idcg = np.sum(np.take_along_axis(relevance, ideal, axis=1) * discount[None, :], axis=1)
    return float(np.mean(dcg / np.maximum(idcg, EPS)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) < EPS or float(np.std(bb)) < EPS:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def per_sample_endpoint_error(pred_flat: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace, horizon: int) -> np.ndarray:
    pred = pred_flat.reshape(len(pred_flat), args.max_horizon, 2)
    true = true_flat.reshape(len(true_flat), args.max_horizon, 2)
    p = np.sum(pred[:, : int(horizon), :], axis=1)
    y = np.sum(true[:, : int(horizon), :], axis=1)
    return np.sqrt(np.sum((p - y) ** 2, axis=1)).astype(np.float32)


def evaluate_np(
    *,
    label: str,
    pred_flat: np.ndarray,
    logits: np.ndarray | None,
    weights: np.ndarray | None,
    basis: v26.RouteBasis,
    teacher: TeacherLabels,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    residual = pred_flat.reshape(len(pred_flat), args.max_horizon, 2).astype(np.float32)
    rows = audit.endpoint_metrics(
        steps_true=basis.arrays.steps_test,
        base=basis.arrays.base_test,
        residual_pred=residual,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )
    diag: dict[str, Any] = {"method": label}
    if logits is not None:
        diag.update(
            {
                "route_top1": route_topk(logits, teacher.hard_test, 1),
                "route_top3": route_topk(logits, teacher.hard_test, 3),
                "ndcg_at3": ndcg_from_scores(logits, teacher.err_test, 3),
                "ndcg_at8": ndcg_from_scores(logits, teacher.err_test, min(8, logits.shape[1])),
            }
        )
    if weights is not None:
        ent = -np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1)
        usage = weights.mean(axis=0)
        expected_candidate_error = np.sum(weights * teacher.err_test, axis=1)
        best_candidate_error = np.min(teacher.err_test, axis=1)
        chosen_candidate_error = teacher.err_test[np.arange(len(teacher.err_test)), np.argmax(weights, axis=1)]
        final_error_hmax = per_sample_endpoint_error(pred_flat, basis.y_test, args, max(args.horizons))
        prior_gap = None
        oracle_gap = None
        if extra.get("prior_hmax_rmse") is not None and extra.get("oracle_hmax_rmse") is not None:
            prior_gap = float(extra["prior_hmax_rmse"]) - float(extra["oracle_hmax_rmse"])
            if prior_gap > EPS:
                oracle_gap = (float(extra["prior_hmax_rmse"]) - endpoint_rmse_flat_np(pred_flat, basis.y_test, args, max(args.horizons))) / prior_gap
        diag.update(
            {
                "route_entropy_mean": float(np.mean(ent)),
                "route_entropy_norm_mean": float(np.mean(ent) / max(np.log(weights.shape[1]), EPS)),
                "max_route_usage": float(np.max(usage)),
                "active_routes_mean": float(np.mean(np.sum(weights > 1e-3, axis=1))),
                "expected_candidate_error_mean": float(np.mean(expected_candidate_error)),
                "chosen_candidate_error_mean": float(np.mean(chosen_candidate_error)),
                "best_candidate_error_mean": float(np.mean(best_candidate_error)),
                "expected_candidate_error_corr_final_error": safe_corr(expected_candidate_error, final_error_hmax),
                "max_weight_corr_final_error": safe_corr(np.max(weights, axis=1), final_error_hmax),
                "oracle_gap_closed_vs_prior": oracle_gap if oracle_gap is not None else float("nan"),
            }
        )
    hmax = max(args.horizons)
    diag["hmax_rmse"] = endpoint_rmse_flat_np(pred_flat, basis.y_test, args, hmax)
    return rows, diag


@torch.no_grad()
def model_predict(
    model: VisualRouteValidator,
    cand_x: np.ndarray,
    route_pred: np.ndarray,
    families_np: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    pred_all = []
    corr_all = []
    logvar_all = []
    n = len(cand_x)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        cand = torch.from_numpy(cand_x[sl]).float().to(device)
        route = torch.from_numpy(route_pred[sl]).float().to(device)
        fam = {k: torch.from_numpy(v[sl]).float().to(device) for k, v in families_np.items() if v.shape[1] > 0}
        logits, corr, logvar = model(cand, fam)
        weights = torch.softmax(logits, dim=1)
        mix = torch.sum(weights.unsqueeze(-1) * route, dim=1)
        pred = mix + corr
        logits_all.append(logits.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())
        corr_all.append(corr.detach().cpu().numpy())
        logvar_all.append(logvar.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all, axis=0).astype(np.float32)
    pred_np = np.concatenate(pred_all, axis=0).astype(np.float32)
    corr_np = np.concatenate(corr_all, axis=0).astype(np.float32)
    logvar_np = np.concatenate(logvar_all, axis=0).astype(np.float32)
    return logits_np, pred_np, corr_np, logvar_np


def train_variant(
    *,
    variant: str,
    visual_variant: str,
    packets: dict[str, v32.Packet],
    basis: v26.RouteBasis,
    cand_train: np.ndarray,
    cand_val: np.ndarray,
    cand_test: np.ndarray,
    teacher: TeacherLabels,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    use_cross_attention: bool = True,
    use_candidate_self_attention: bool = True,
    use_route_teacher: bool = True,
    visual_feature_mode: str = "all",
    drop_families: tuple[str, ...] = (),
    use_uncertainty: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    blocks = build_family_blocks(
        args,
        packets,
        visual_variant=visual_variant,
        visual_feature_mode=visual_feature_mode,
        drop_families=drop_families,
    )
    family_dims = {k: v.shape[1] for k, v in blocks.train.items() if v.shape[1] > 0}
    model = VisualRouteValidator(
        cand_dim=cand_train.shape[2],
        family_dims=family_dims,
        route_count=basis.route_train.shape[1],
        out_dim=basis.y_train.shape[1],
        hidden=int(args.v38_hidden),
        heads=int(args.v38_heads),
        layers=int(args.v38_layers),
        dropout=float(args.v38_dropout),
        use_cross_attention=use_cross_attention,
        use_candidate_self_attention=use_candidate_self_attention,
        correction_scale=float(args.v38_correction_scale),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.v38_lr), weight_decay=float(args.v38_weight_decay))
    n = len(cand_train)
    family_order = list(family_dims)
    tensors = [
        torch.from_numpy(cand_train).float(),
        torch.from_numpy(basis.route_train).float(),
        torch.from_numpy(basis.y_train).float(),
        torch.from_numpy(teacher.soft_train).float(),
        torch.from_numpy(teacher.hard_train).long(),
        torch.from_numpy(teacher.err_train).float(),
    ]
    for fam in family_order:
        tensors.append(torch.from_numpy(blocks.train[fam]).float())
    loader = DataLoader(TensorDataset(*tensors), batch_size=int(args.v38_batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    logs: list[dict[str, Any]] = []
    hmax = max(args.horizons)
    for epoch in range(1, int(args.v38_epochs) + 1):
        model.train()
        losses = []
        for batch in loader:
            cand = batch[0].to(device)
            route = batch[1].to(device)
            y = batch[2].to(device)
            soft = batch[3].to(device)
            hard = batch[4].to(device)
            err = batch[5].to(device)
            fam = {family_order[i]: batch[6 + i].to(device) for i in range(len(family_order))}
            logits, corr, logvar = model(cand, fam)
            weights = torch.softmax(logits, dim=1)
            mix = torch.sum(weights.unsqueeze(-1) * route, dim=1)
            pred = mix + corr
            huber = torch.nn.functional.smooth_l1_loss(pred, y, beta=float(args.v38_huber_beta))
            logp = torch.log_softmax(logits, dim=1)
            kl = -(soft * logp).sum(dim=1).mean()
            ce = torch.nn.functional.cross_entropy(logits, hard)
            worst = torch.argmax(err, dim=1)
            best_logit = logits.gather(1, hard[:, None]).squeeze(1)
            worst_logit = logits.gather(1, worst[:, None]).squeeze(1)
            rank = torch.relu(float(args.v38_rank_margin) - (best_logit - worst_logit)).mean()
            entropy = -(weights * torch.log(torch.clamp(weights, min=1e-8))).sum(dim=1).mean()
            entropy_floor = torch.relu(float(args.v38_entropy_floor) - entropy / math.log(weights.shape[1])).pow(2)
            # Light heteroscedastic diagnostic loss: route mixture squared error should correlate with logvar.
            sq = torch.mean((pred.detach() - y) ** 2, dim=1)
            nll = 0.5 * (torch.exp(-logvar) * sq + logvar).mean()
            uncertainty_weight = float(args.v38_uncertainty_weight) if use_uncertainty else 0.0
            loss = (
                float(args.v38_rmse_weight) * huber
                + (float(args.v38_kl_weight) * kl + float(args.v38_ce_weight) * ce if use_route_teacher else 0.0)
                + float(args.v38_rank_weight) * rank
                + float(args.v38_entropy_weight) * entropy_floor
                + uncertainty_weight * nll
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.v38_grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        logits_va, pred_va, _corr_va, _logvar_va = model_predict(model, cand_val, basis.route_val, blocks.val, device, int(args.v38_eval_batch_size))
        val_rmse = endpoint_rmse_flat_np(pred_va, basis.y_val, args, hmax)
        logs.append(
            {
                "variant": variant,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_hmax_rmse": val_rmse,
                "val_route_top3": route_topk(logits_va, teacher.hard_val, 3),
            }
        )
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    logits_te, pred_te, corr_te, logvar_te = model_predict(model, cand_test, basis.route_test, blocks.test, device, int(args.v38_eval_batch_size))
    weights_te = np.exp(logits_te - np.max(logits_te, axis=1, keepdims=True))
    weights_te /= np.maximum(weights_te.sum(axis=1, keepdims=True), EPS)
    rows, diag = evaluate_np(
        label=f"v38_{variant}",
        pred_flat=pred_te,
        logits=logits_te,
        weights=weights_te,
        basis=basis,
        teacher=teacher,
        args=args,
        extra={
            "stage": "v38_visual_state_route_validator",
            "variant": variant,
            "visual_variant": visual_variant,
            "use_cross_attention": use_cross_attention,
            "use_candidate_self_attention": use_candidate_self_attention,
            "use_route_teacher": use_route_teacher,
            "visual_feature_mode": visual_feature_mode,
            "drop_families": ",".join(drop_families),
            "use_uncertainty": use_uncertainty,
            "best_val_hmax_rmse": best_val,
            "family_dims": json.dumps(family_dims, sort_keys=True),
            "correction_norm_mean": float(np.mean(np.linalg.norm(corr_te.reshape(len(corr_te), -1), axis=1))),
            "logvar_mean": float(np.mean(logvar_te)),
            "prior_hmax_rmse": getattr(args, "_v38_prior_hmax_rmse", None),
            "oracle_hmax_rmse": getattr(args, "_v38_oracle_hmax_rmse", None),
        },
    )
    diag.update(
        {
            "variant": variant,
            "visual_variant": visual_variant,
            "best_val_hmax_rmse": best_val,
            "family_dims": json.dumps(family_dims, sort_keys=True),
            "use_cross_attention": use_cross_attention,
            "use_candidate_self_attention": use_candidate_self_attention,
            "use_route_teacher": use_route_teacher,
            "visual_feature_mode": visual_feature_mode,
            "drop_families": ",".join(drop_families),
            "use_uncertainty": use_uncertainty,
        }
    )
    return pd.DataFrame(rows), diag, pd.DataFrame(logs)


def baseline_rows(basis: v26.RouteBasis, teacher: TeacherLabels, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []
    k = basis.route_test.shape[1]
    uniform = np.full((len(basis.route_test), k), 1.0 / float(k), dtype=np.float32)
    uniform_pred = np.sum(basis.route_test * uniform[:, :, None], axis=1).astype(np.float32)
    r, d = evaluate_np(label="v38_baseline_uniform_route_mix", pred_flat=uniform_pred, logits=None, weights=uniform, basis=basis, teacher=teacher, args=args, extra={"stage": "baseline", "variant": "uniform"})
    rows.extend(r)
    diags.append(d | {"variant": "uniform", "stage": "baseline"})

    # Tune prior top-M/temperature on validation.
    best = None
    for top_m in parse_ints(args.v38_top_m_grid):
        for temp in parse_floats(args.v38_temperature_grid):
            wva = v26.topm_temperature_weights(basis.prior.probs_val, top_m=top_m, temperature=temp, entropy_blend=0.0)
            pva = np.sum(basis.route_val * wva[:, :, None], axis=1)
            rmse = endpoint_rmse_flat_np(pva, basis.y_val, args, max(args.horizons))
            if best is None or rmse < best["val_rmse"]:
                best = {"top_m": int(top_m), "temp": float(temp), "val_rmse": float(rmse)}
    assert best is not None
    wte = v26.topm_temperature_weights(basis.prior.probs_test, top_m=best["top_m"], temperature=best["temp"], entropy_blend=0.0)
    pte = np.sum(basis.route_test * wte[:, :, None], axis=1)
    r, d = evaluate_np(
        label=f"v38_baseline_prior_top{best['top_m']}_t{best['temp']}",
        pred_flat=pte,
        logits=basis.prior.probs_test,
        weights=wte,
        basis=basis,
        teacher=teacher,
        args=args,
        extra={"stage": "baseline", "variant": "prior_topm", "top_m": best["top_m"], "temperature": best["temp"], "best_val_hmax_rmse": best["val_rmse"]},
    )
    rows.extend(r)
    diags.append(d | {"variant": "prior_topm", "stage": "baseline", **best})

    oracle = basis.route_test[np.arange(len(basis.route_test)), teacher.hard_test]
    oracle_w = np.zeros((len(basis.route_test), k), dtype=np.float32)
    oracle_w[np.arange(len(basis.route_test)), teacher.hard_test] = 1.0
    r, d = evaluate_np(label="v38_oracle_route_choice", pred_flat=oracle, logits=-teacher.err_test, weights=oracle_w, basis=basis, teacher=teacher, args=args, extra={"stage": "oracle", "variant": "oracle"})
    rows.extend(r)
    diags.append(d | {"variant": "oracle", "stage": "oracle"})
    return pd.DataFrame(rows), pd.DataFrame(diags)


def write_report(out_dir: Path, metrics: pd.DataFrame, diagnostics: pd.DataFrame, train_log: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    lines = ["# v38 Visual-State Conditioned Route Validator", ""]
    lines.append("## Best h6")
    hmax = max(args.horizons)
    h6 = metrics[metrics["horizon"].eq(hmax)].sort_values("rmse")
    cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "visual_variant", "best_val_hmax_rmse", "route_top3", "route_entropy_norm_mean"] if c in h6.columns]
    lines.append(h6[cols].head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Diagnostics")
    if not diagnostics.empty:
        dcols = [c for c in ["method", "variant", "stage", "visual_variant", "hmax_rmse", "route_top1", "route_top3", "route_entropy_norm_mean", "max_route_usage", "best_val_hmax_rmse", "family_dims"] if c in diagnostics.columns]
        lines.append(diagnostics.sort_values("hmax_rmse")[dcols].head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Decision")
    lines.append("```json")
    lines.append(json.dumps(finite_json(decision), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    if not train_log.empty:
        lines.append("## Training Tail")
        lines.append(train_log.groupby("variant").tail(3).to_markdown(index=False))
    (out_dir / "visual_state_route_validator_v38_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    audit.set_global_seed(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.generator_max_train_rows < 0:
        args.generator_max_train_rows = args.max_train_rows
    if args.generator_max_val_rows < 0:
        args.generator_max_val_rows = args.max_val_rows
    if args.generator_max_test_rows < 0:
        args.generator_max_test_rows = args.max_test_rows
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 450)
        args.generator_max_train_rows = args.max_train_rows
        args.generator_max_val_rows = args.max_val_rows
        args.generator_max_test_rows = args.max_test_rows
        args.generator_posterior_epochs = min(args.generator_posterior_epochs, 3)
        args.generator_student_epochs = min(args.generator_student_epochs, 3)
        args.generator_learned_route_epochs = min(args.generator_learned_route_epochs, 2)
        args.v38_epochs = min(args.v38_epochs, 4)
        args.v38_hidden = min(args.v38_hidden, 96)
        args.v38_batch_size = min(args.v38_batch_size, 256)
        args.v38_variants = "full,no_visual,row_shuffled_visual"

    device = device_from_arg(args.device)
    # v26 builds fixed route experts and route-prior probabilities.
    basis = v26.build_route_basis(args, args.out_dir / "route_basis")
    packets = v32.build_packets(args, basis.arrays, basis.split)
    teacher = build_teacher_labels(args, basis)
    cand_tr = candidate_features(basis.route_train, basis.prior.probs_train, args)
    cand_va = candidate_features(basis.route_val, basis.prior.probs_val, args)
    cand_te = candidate_features(basis.route_test, basis.prior.probs_test, args)
    cand_tr, cand_va, cand_te = standardize_candidate_features(cand_tr, cand_va, cand_te)

    base_metrics, base_diag = baseline_rows(basis, teacher, args)
    hmax = max(args.horizons)
    prior_hmax = base_metrics[
        base_metrics["horizon"].eq(hmax) & base_metrics["variant"].astype(str).eq("prior_topm")
    ]
    oracle_hmax = base_metrics[
        base_metrics["horizon"].eq(hmax) & base_metrics["variant"].astype(str).eq("oracle")
    ]
    args._v38_prior_hmax_rmse = float(prior_hmax.iloc[0]["rmse"]) if not prior_hmax.empty else None
    args._v38_oracle_hmax_rmse = float(oracle_hmax.iloc[0]["rmse"]) if not oracle_hmax.empty else None
    metric_parts = [base_metrics]
    diag_rows: list[dict[str, Any]] = base_diag.to_dict(orient="records")
    logs = []
    variant_specs = {
        "full": dict(visual_variant="real"),
        "no_visual": dict(visual_variant="no_visual"),
        "zero_visual": dict(visual_variant="zero"),
        "row_shuffled_visual": dict(visual_variant="row_shuffled"),
        "same_frame_wrong_cell_visual": dict(visual_variant="same_frame_wrong_cell"),
        "time_shuffled_visual": dict(visual_variant="time_shuffled"),
        "no_cross_attention": dict(visual_variant="real", use_cross_attention=False),
        "no_candidate_self_attention": dict(visual_variant="real", use_candidate_self_attention=False),
        "no_route_teacher": dict(visual_variant="real", use_route_teacher=False),
        "no_uncertainty": dict(visual_variant="real", use_uncertainty=False),
        "no_contact": dict(visual_variant="real", visual_feature_mode="no_contact", drop_families=("crowding",)),
        "no_polarity": dict(visual_variant="real", visual_feature_mode="no_polarity"),
        "shape_only": dict(visual_variant="real", visual_feature_mode="shape_only"),
        "polarity_only": dict(visual_variant="real", visual_feature_mode="polarity_only"),
        "contact_only": dict(visual_variant="real", visual_feature_mode="contact_only"),
        "mask_quality_only": dict(visual_variant="real", visual_feature_mode="quality_only"),
        "no_neighbour_state": dict(visual_variant="real", drop_families=("crowding",)),
    }
    for variant in parse_csv(args.v38_variants):
        if variant not in variant_specs:
            raise ValueError(f"Unknown v38 variant {variant}; choices={sorted(variant_specs)}")
        spec = {
            "visual_variant": "real",
            "use_cross_attention": True,
            "use_candidate_self_attention": True,
            "use_route_teacher": True,
            "visual_feature_mode": "all",
            "drop_families": (),
            "use_uncertainty": True,
        }
        spec.update(variant_specs[variant])
        rows, diag, log = train_variant(
            variant=variant,
            visual_variant=spec["visual_variant"],
            packets=packets,
            basis=basis,
            cand_train=cand_tr,
            cand_val=cand_va,
            cand_test=cand_te,
            teacher=teacher,
            args=args,
            out_dir=args.out_dir,
            device=device,
            use_cross_attention=bool(spec["use_cross_attention"]),
            use_candidate_self_attention=bool(spec["use_candidate_self_attention"]),
            use_route_teacher=bool(spec["use_route_teacher"]),
            visual_feature_mode=str(spec["visual_feature_mode"]),
            drop_families=tuple(spec["drop_families"]),
            use_uncertainty=bool(spec["use_uncertainty"]),
        )
        metric_parts.append(rows)
        diag_rows.append(diag)
        logs.append(log)

    metrics = pd.concat(metric_parts, ignore_index=True, sort=False)
    metrics.insert(0, "seed", int(args.seed))
    metrics.insert(0, "dataset", str(args.dataset))
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.insert(0, "seed", int(args.seed))
    diagnostics.insert(0, "dataset", str(args.dataset))
    train_log = pd.concat(logs, ignore_index=True, sort=False) if logs else pd.DataFrame()
    if not train_log.empty:
        train_log.insert(0, "seed", int(args.seed))
        train_log.insert(0, "dataset", str(args.dataset))

    metrics.to_csv(args.out_dir / "visual_state_route_validator_v38_summary.csv", index=False)
    diagnostics.to_csv(args.out_dir / "visual_state_route_validator_v38_diagnostics.csv", index=False)
    train_log.to_csv(args.out_dir / "visual_state_route_validator_v38_train_log.csv", index=False)

    hmax = max(args.horizons)
    h6 = metrics[metrics["horizon"].eq(hmax)].sort_values("rmse")
    best = h6.iloc[0].to_dict() if not h6.empty else {}
    full = h6[h6["variant"].astype(str).eq("full")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("full")].empty else {}
    no_visual = h6[h6["variant"].astype(str).eq("no_visual")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("no_visual")].empty else {}
    control = h6[h6["variant"].astype(str).isin(["row_shuffled_visual", "same_frame_wrong_cell_visual", "time_shuffled_visual", "zero_visual"])].head(1)
    best_control = control.iloc[0].to_dict() if not control.empty else {}
    prior = h6[h6["variant"].astype(str).eq("prior_topm")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("prior_topm")].empty else {}
    oracle = h6[h6["variant"].astype(str).eq("oracle")].iloc[0].to_dict() if not h6[h6["variant"].astype(str).eq("oracle")].empty else {}
    decision = {
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "best_hmax": best,
        "full_hmax": full,
        "no_visual_hmax": no_visual,
        "best_visual_control_hmax": best_control,
        "prior_baseline_hmax": prior,
        "oracle_hmax": oracle,
        "full_beats_no_visual": bool(full and no_visual and float(full["rmse"]) < float(no_visual["rmse"])),
        "full_beats_visual_controls": bool(full and best_control and float(full["rmse"]) < float(best_control["rmse"])),
        "full_beats_prior": bool(full and prior and float(full["rmse"]) < float(prior["rmse"])),
        "hard_pass_h6_le_16": bool(full and float(full["rmse"]) <= 16.0),
    }
    (args.out_dir / "visual_state_route_validator_v38_decision.json").write_text(json.dumps(finite_json(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.out_dir, metrics, diagnostics, train_log, decision, args)
    print(json.dumps({"out_dir": str(args.out_dir), "best_hmax": finite_json(best), "decision": finite_json(decision)}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--dense-features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    ap.add_argument("--dataset", default="MDCK_Bulk")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seq", default="1,2,3,4")
    ap.add_argument("--val-seq", default="5")
    ap.add_argument("--test-seq", default="6")
    ap.add_argument("--horizons", default="1,2,4,6")
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--max-test-rows", type=int, default=0)
    ap.add_argument("--max-features-per-family", type=int, default=160)
    ap.add_argument("--max-all-features", type=int, default=384)
    ap.add_argument("--device", default="auto")

    # Route generator compatibility with v25/v26/v16.
    ap.add_argument("--generator-max-train-rows", type=int, default=-1)
    ap.add_argument("--generator-max-val-rows", type=int, default=-1)
    ap.add_argument("--generator-max-test-rows", type=int, default=-1)
    ap.add_argument("--generator-posterior-epochs", type=int, default=4)
    ap.add_argument("--generator-student-epochs", type=int, default=4)
    ap.add_argument("--generator-learned-route-epochs", type=int, default=3)
    ap.add_argument("--generator-candidate-k", type=int, default=32)
    ap.add_argument("--generator-oracle-k", default="8,16,32")
    ap.add_argument("--generator-variant", default="context_velocity")
    ap.add_argument("--generator-prior-model", default="logistic")
    ap.add_argument("--generator-base-mixes", default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    ap.add_argument("--generator-calibrators", default="correction_context,stacked_context")
    ap.add_argument("--generator-max-context-features", type=int, default=384)
    ap.add_argument("--dense-max-cols", type=int, default=256)
    ap.add_argument("--v25-velocity-max-cols", type=int, default=160)
    ap.add_argument("--v25-route-k", type=int, default=12)

    # Visual grids.
    ap.add_argument("--object-grid", type=Path, default=ROOT / "outputs" / "lachance_object_centric_mask_grid_bulk_seed42_2026-07-03" / "object_centric_mask_feature_grid.csv")
    ap.add_argument("--temporal-grid", type=Path, default=ROOT / "outputs" / "temporal_mask_change_medium_bulk_seed42_2026-07-04" / "multiseed_instance_mask_feature_grid.csv")
    ap.add_argument("--multiseed-grid", type=Path, default=ROOT / "outputs" / "multiseed_instance_mask_medium_bulk_seed42_2026-07-04" / "multiseed_instance_mask_feature_grid.csv")
    ap.add_argument("--seg-foundation-grid", type=Path, default=DEFAULT_SEGF)
    ap.add_argument("--visual-tokens", default="area,perimeter,eccentricity,solidity,extent,major,minor,elongation,orient,velocity,centroid,front,back,left,right,balance,intensity,grad,free,contact,boundary,neighbor,seed,center,available,quality,fallback,track_aligned")
    ap.add_argument("--max-object-cols", type=int, default=0)
    ap.add_argument("--max-temporal-cols", type=int, default=0)
    ap.add_argument("--max-multiseed-cols", type=int, default=0)
    ap.add_argument("--max-seg-cols", type=int, default=160)
    ap.add_argument("--max-interaction-cols", type=int, default=120)

    # v38 model.
    ap.add_argument("--v38-variants", default="full,no_visual,zero_visual,row_shuffled_visual,same_frame_wrong_cell_visual,time_shuffled_visual,no_cross_attention,no_candidate_self_attention,no_route_teacher")
    ap.add_argument("--v38-hidden", type=int, default=160)
    ap.add_argument("--v38-heads", type=int, default=4)
    ap.add_argument("--v38-layers", type=int, default=2)
    ap.add_argument("--v38-dropout", type=float, default=0.08)
    ap.add_argument("--v38-epochs", type=int, default=28)
    ap.add_argument("--v38-batch-size", type=int, default=384)
    ap.add_argument("--v38-eval-batch-size", type=int, default=512)
    ap.add_argument("--v38-lr", type=float, default=8e-4)
    ap.add_argument("--v38-weight-decay", type=float, default=1e-4)
    ap.add_argument("--v38-grad-clip", type=float, default=2.0)
    ap.add_argument("--v38-rmse-weight", type=float, default=1.0)
    ap.add_argument("--v38-kl-weight", type=float, default=0.50)
    ap.add_argument("--v38-ce-weight", type=float, default=0.20)
    ap.add_argument("--v38-rank-weight", type=float, default=0.10)
    ap.add_argument("--v38-entropy-weight", type=float, default=0.02)
    ap.add_argument("--v38-uncertainty-weight", type=float, default=0.02)
    ap.add_argument("--v38-huber-beta", type=float, default=1.5)
    ap.add_argument("--v38-rank-margin", type=float, default=1.0)
    ap.add_argument("--v38-entropy-floor", type=float, default=0.25)
    ap.add_argument("--v38-correction-scale", type=float, default=4.0)
    ap.add_argument("--v38-teacher-temperature", type=float, default=4.0)
    ap.add_argument("--v38-max-family-cols", type=int, default=96)
    ap.add_argument("--v38-include-all-context-token", type=int, default=1)
    ap.add_argument("--v38-top-m-grid", default="1,2,4,8,12")
    ap.add_argument("--v38-temperature-grid", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
