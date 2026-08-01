#!/usr/bin/env python3
"""Axis-conditioned candidate distillation critic for LaChance trajectories.

This runner implements the next selector/refiner hypothesis:

    route prototypes are useful,
    component-aware features help,
    but one scalar risk head is too weak.

The new module lets every learned component axis build its own distribution
over route prototypes. A horizon-specific router then mixes these axis-specific
candidate distributions.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "axis_conditioned_distillation_critic_2026-06-29"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def residual_endpoints(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    out = []
    for h in horizons:
        out.append(np.sum(residual[:, : int(h), :], axis=1))
    return np.stack(out, axis=1).astype(np.float32)


def candidate_endpoints(query_pred: np.ndarray, horizons: list[int]) -> np.ndarray:
    out = []
    for h in horizons:
        out.append(np.sum(query_pred[:, :, : int(h), :], axis=2))
    return np.stack(out, axis=2).astype(np.float32)  # n,q,hh,2


def endpoint_rows_from_residual_endpoints(
    *,
    arrays: audit.SplitArrays,
    residual_endpoint_pred: np.ndarray,
    horizons: list[int],
    label: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for j, h in enumerate(horizons):
        y = audit.endpoint_from_steps(arrays.steps_test, int(h))
        y_base = audit.base_rollout(arrays.base_test, int(h))
        y_hat = y_base + residual_endpoint_pred[:, j, :]
        row: dict[str, Any] = {
            "method": label,
            "horizon": int(h),
            "rmse": audit.rmse(y, y_hat),
            "r2": audit.r2_score_np(y, y_hat),
            "base_rmse": audit.rmse(y, y_base),
            "base_r2": audit.r2_score_np(y, y_base),
        }
        row["gain_vs_base_pct"] = audit.gain_pct(row["base_rmse"], row["rmse"])
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def target_candidate_errors(query_pred: np.ndarray, true: np.ndarray, horizons: list[int]) -> np.ndarray:
    qh = candidate_endpoints(query_pred, horizons)  # n,q,hh,2
    th = residual_endpoints(true, horizons)[:, None, :, :]
    err = np.linalg.norm(qh - th, axis=-1)  # n,q,hh
    return np.transpose(err, (0, 2, 1)).astype(np.float32)  # n,hh,q


def component_candidate_errors(query_pred: np.ndarray, component_pred: np.ndarray, horizons: list[int]) -> np.ndarray:
    qh = candidate_endpoints(query_pred, horizons)  # n,q,hh,2
    ch = np.stack([np.sum(component_pred[:, :, : int(h), :], axis=2) for h in horizons], axis=2)  # n,c,hh,2
    err = np.linalg.norm(qh[:, :, None, :, :] - ch[:, None, :, :, :], axis=-1)  # n,q,c,hh
    return np.transpose(err, (0, 3, 2, 1)).astype(np.float32)  # n,hh,c,q


def soft_labels_from_distance_np(dist: np.ndarray, temp: float) -> np.ndarray:
    z = -np.asarray(dist, dtype=np.float32) / max(float(temp), 1e-6)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z)
    return (ez / np.maximum(np.sum(ez, axis=-1, keepdims=True), EPS)).astype(np.float32)


def axis_candidate_features(
    *,
    query_pred: np.ndarray,
    component_pred: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Pair features for candidate k versus component axis a."""
    n, q, hmax, _ = query_pred.shape
    c = component_pred.shape[1]
    q_steps = query_pred[:, :, None, :, :]
    a_steps = component_pred[:, None, :, :, :]
    flat_q = query_pred.reshape(n, q, -1)
    flat_a = component_pred.reshape(n, c, -1)

    step_delta = q_steps - a_steps
    step_mse = np.mean(np.sum(step_delta * step_delta, axis=-1), axis=-1)  # n,q,c
    step_abs = np.mean(np.linalg.norm(step_delta, axis=-1), axis=-1)

    dot = np.sum(flat_q[:, :, None, :] * flat_a[:, None, :, :], axis=-1)
    q_norm = np.linalg.norm(flat_q, axis=-1)[:, :, None]
    a_norm = np.linalg.norm(flat_a, axis=-1)[:, None, :]
    traj_cos = dot / np.maximum(q_norm * a_norm, EPS)
    traj_dist = np.linalg.norm(flat_q[:, :, None, :] - flat_a[:, None, :, :], axis=-1)
    traj_mag_ratio = q_norm / np.maximum(a_norm, EPS)

    pieces: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        a = arr.reshape(n, q, c, -1).astype(np.float32)
        pieces.append(a)
        for j in range(a.shape[-1]):
            names.append(f"{name}_{j}" if a.shape[-1] > 1 else name)

    add("step_mse", step_mse)
    add("step_abs", step_abs)
    add("traj_cos", traj_cos)
    add("traj_dist", traj_dist)
    add("traj_mag_ratio", np.clip(traj_mag_ratio, 0.0, 8.0))

    for h in horizons:
        qh = np.sum(query_pred[:, :, : int(h), :], axis=2)
        ah = np.sum(component_pred[:, :, : int(h), :], axis=2)
        diff = qh[:, :, None, :] - ah[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        cdot = np.sum(qh[:, :, None, :] * ah[:, None, :, :], axis=-1)
        qmag = np.linalg.norm(qh, axis=-1)[:, :, None]
        amag = np.linalg.norm(ah, axis=-1)[:, None, :]
        cos = cdot / np.maximum(qmag * amag, EPS)
        ratio = qmag / np.maximum(amag, EPS)
        add(f"h{int(h)}_endpoint_dx", diff[..., 0])
        add(f"h{int(h)}_endpoint_dy", diff[..., 1])
        add(f"h{int(h)}_endpoint_dist", dist)
        add(f"h{int(h)}_endpoint_cos", cos)
        add(f"h{int(h)}_endpoint_mag_ratio", np.clip(ratio, 0.0, 8.0))

    feat = np.concatenate(pieces, axis=-1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat, names


def standardize_candidate_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return qrc.standardize_query_features(train, val, test)


def standardize_axis_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tr = train.reshape(-1, train.shape[-1])
    va = val.reshape(-1, val.shape[-1])
    te = test.reshape(-1, test.shape[-1])
    tr_z, va_z, te_z, scaler = seq.standardize(tr, va, te)
    return tr_z.reshape(train.shape), va_z.reshape(val.shape), te_z.reshape(test.shape), scaler


def _select_edge_compat_block(
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    block = str(args.candidate_edge_compat_block)
    max_features = max(1, int(args.candidate_edge_compat_max_features))
    if block not in arrays.x_train:
        ntr, nva, nte = len(arrays.residual_train), len(arrays.residual_val), len(arrays.residual_test)
        return (
            np.zeros((ntr, max_features), dtype=np.float32),
            np.zeros((nva, max_features), dtype=np.float32),
            np.zeros((nte, max_features), dtype=np.float32),
            {"enabled": False, "reason": f"missing block {block}", "block": block, "max_features": max_features},
        )
    xtr = np.nan_to_num(arrays.x_train[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xva = np.nan_to_num(arrays.x_val[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xte = np.nan_to_num(arrays.x_test[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if xtr.shape[1] == 0:
        return (
            np.zeros((len(xtr), max_features), dtype=np.float32),
            np.zeros((len(xva), max_features), dtype=np.float32),
            np.zeros((len(xte), max_features), dtype=np.float32),
            {"enabled": False, "reason": "empty block", "block": block, "max_features": max_features},
        )
    var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-min(max_features, xtr.shape[1]) :]
    names = list(arrays.feature_names.get(block, []))
    kept_names = [names[int(i)] if int(i) < len(names) else f"{block}_{int(i)}" for i in keep]
    return (
        xtr[:, keep].astype(np.float32),
        xva[:, keep].astype(np.float32),
        xte[:, keep].astype(np.float32),
        {
            "enabled": True,
            "block": block,
            "max_features": max_features,
            "selected_features": len(keep),
            "selected_names_preview": kept_names[:24],
        },
    )


def candidate_edge_compat_features(
    query_pred: np.ndarray,
    edge_x: np.ndarray,
    horizons: list[int],
    *,
    include_broadcast: bool,
) -> tuple[np.ndarray, list[str]]:
    """Candidate-conditioned interaction between route endpoints and edge context.

    Edge/context alone is candidate-independent.  Repeating it inside candidate
    tokens lets the candidate MLP learn interactions, but the stronger signal is
    the product between each candidate endpoint direction/magnitude and the
    selected edge descriptors.
    """
    n, q = query_pred.shape[:2]
    edge = np.nan_to_num(edge_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pieces: list[np.ndarray] = []
    names: list[str] = []
    if include_broadcast:
        pieces.append(np.repeat(edge[:, None, :], q, axis=1))
        names += [f"edge_broadcast_{i}" for i in range(edge.shape[1])]
    for h in horizons:
        endpoint = np.sum(query_pred[:, :, : int(h), :], axis=2).astype(np.float32)
        mag = np.linalg.norm(endpoint, axis=-1, keepdims=True)
        # Robust scale is handled by later train-only standardization; clipping
        # just prevents rare huge products from dominating early optimization.
        coeffs = np.concatenate(
            [
                np.clip(endpoint[..., :1], -128.0, 128.0),
                np.clip(endpoint[..., 1:2], -128.0, 128.0),
                np.clip(mag, 0.0, 256.0),
            ],
            axis=-1,
        )
        inter = coeffs[:, :, :, None] * edge[:, None, None, :]
        inter = inter.reshape(n, q, -1).astype(np.float32)
        pieces.append(inter)
        for c_name in ("dx", "dy", "mag"):
            names += [f"edge_h{int(h)}_{c_name}_x_{i}" for i in range(edge.shape[1])]
    feat = np.concatenate(pieces, axis=-1) if pieces else np.zeros((n, q, 0), dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat, names


@dataclass
class AxisFeaturePack:
    cand_train: np.ndarray
    cand_val: np.ndarray
    cand_test: np.ndarray
    axis_train: np.ndarray
    axis_val: np.ndarray
    axis_test: np.ndarray
    edge_train: np.ndarray
    edge_val: np.ndarray
    edge_test: np.ndarray
    cand_end_train: np.ndarray
    cand_end_val: np.ndarray
    cand_end_test: np.ndarray
    target_end_train: np.ndarray
    target_end_val: np.ndarray
    target_end_test: np.ndarray
    target_soft_train: np.ndarray
    target_soft_val: np.ndarray
    component_soft_train: np.ndarray
    component_soft_val: np.ndarray
    component_names: list[str]
    scalers: dict[str, Any]


class AxisConditionedCritic(nn.Module):
    def __init__(
        self,
        *,
        cand_dim: int,
        axis_dim: int,
        edge_dim: int,
        ctx_dim: int,
        n_axes: int,
        n_horizons: int,
        hidden: int,
        dropout: float,
        heads: int,
        layers: int,
        correction_scale: float,
    ):
        super().__init__()
        self.n_axes = int(n_axes)
        self.n_horizons = int(n_horizons)
        self.correction_scale = float(correction_scale)
        self.cand_proj = nn.Sequential(nn.Linear(cand_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.axis_proj = nn.Sequential(nn.Linear(axis_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.ctx_proj = nn.Sequential(nn.Linear(max(ctx_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.axis_embed = nn.Embedding(n_axes, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(layers)))
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )
        self.router_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons * 2),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons * 2),
        )

    def forward(
        self,
        cand_feat: torch.Tensor,
        axis_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        ctx: torch.Tensor,
        cand_end: torch.Tensor,
        *,
        uniform_router: bool = False,
    ) -> dict[str, torch.Tensor]:
        n, q, c, _ = axis_feat.shape
        if ctx.shape[-1] == 0:
            ctx = torch.zeros((n, 1), dtype=cand_feat.dtype, device=cand_feat.device)
        cand_h = self.cand_proj(cand_feat)  # n,q,d
        axis_h = self.axis_proj(axis_feat)  # n,q,c,d
        edge_h = self.edge_proj(edge_feat)  # n,c,d
        ctx_h = self.ctx_proj(ctx)  # n,d
        eid = self.axis_embed(torch.arange(c, device=cand_feat.device))[None, None, :, :]
        pair = cand_h[:, :, None, :] + axis_h + edge_h[:, None, :, :] + ctx_h[:, None, None, :] + eid
        pair = pair.reshape(n, q * c, -1)
        pair = self.token_encoder(pair).reshape(n, q, c, -1)
        score = self.score_head(pair)  # n,q,c,hh
        axis_logits = score.permute(0, 3, 2, 1).contiguous()  # n,hh,c,q
        axis_weights = torch.softmax(axis_logits, dim=-1)

        axis_context = pair.mean(dim=1) + ctx_h[:, None, :]
        router_logits = self.router_head(axis_context).permute(0, 2, 1).contiguous()  # n,hh,c
        if uniform_router:
            router = torch.full_like(router_logits, 1.0 / max(float(c), 1.0))
        else:
            router = torch.softmax(router_logits, dim=-1)

        final_weights = torch.sum(router[:, :, :, None] * axis_weights, dim=2)  # n,hh,q
        mixture_end = torch.einsum("nhq,nqhd->nhd", final_weights, cand_end)
        corr_axis = self.correction_head(axis_context).reshape(n, c, self.n_horizons, 2).permute(0, 2, 1, 3)
        correction = torch.sum(router[:, :, :, None] * corr_axis, dim=2)
        correction = torch.tanh(correction) * self.correction_scale
        pred_end = mixture_end + correction
        axis_logvar = self.logvar_head(axis_context).reshape(n, c, self.n_horizons, 2).permute(0, 2, 1, 3)
        logvar = torch.sum(router[:, :, :, None] * axis_logvar, dim=2)
        return {
            "pred_end": pred_end,
            "mixture_end": mixture_end,
            "correction": correction,
            "logvar": logvar,
            "final_weights": final_weights,
            "axis_logits": axis_logits,
            "axis_weights": axis_weights,
            "router": router,
            "router_logits": router_logits,
        }


def axis_distillation_loss(
    out: dict[str, torch.Tensor],
    target_end: torch.Tensor,
    target_soft: torch.Tensor,
    component_soft: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_end = out["pred_end"]
    final_weights = torch.clamp(out["final_weights"], min=1e-8)
    axis_logits = out["axis_logits"]
    router = torch.clamp(out["router"], min=1e-8)
    correction = out.get("correction")
    logvar = out.get("logvar")
    reg = F.smooth_l1_loss(pred_end.contiguous(), target_end.contiguous())
    listwise = -torch.mean(torch.sum(target_soft * torch.log(final_weights), dim=-1))
    axis_logprob = F.log_softmax(axis_logits, dim=-1)
    comp_ce = -torch.mean(torch.sum(component_soft * axis_logprob, dim=-1))
    router_entropy = -torch.mean(torch.sum(router * torch.log(router), dim=-1))

    # Diversity penalty: discourage all axes from producing identical candidate
    # distributions.  This is a light regularizer, not a hard constraint.
    aw = out["axis_weights"]
    aw_z = aw - aw.mean(dim=-1, keepdim=True)
    aw_z = aw_z / torch.clamp(torch.linalg.norm(aw_z, dim=-1, keepdim=True), min=1e-6)
    sim = torch.einsum("nhcq,nhdq->nhcd", aw_z, aw_z)
    c = sim.shape[-1]
    if c > 1:
        off_diag = (torch.sum(sim * sim, dim=(-1, -2)) - torch.sum(torch.diagonal(sim * sim, dim1=-1, dim2=-2), dim=-1))
        diversity = torch.mean(off_diag / float(c * (c - 1)))
    else:
        diversity = torch.zeros((), dtype=reg.dtype, device=reg.device)

    if correction is None:
        corr_l2 = torch.zeros((), dtype=reg.dtype, device=reg.device)
    else:
        corr_l2 = torch.mean(correction * correction)
    if logvar is None or args.axis_nll_weight <= 0:
        nll = torch.zeros((), dtype=reg.dtype, device=reg.device)
    else:
        logv = torch.clamp(logvar, min=args.axis_logvar_min, max=args.axis_logvar_max)
        err2 = (target_end - pred_end) ** 2
        nll = 0.5 * torch.mean(err2 * torch.exp(-logv) + logv)

    loss = (
        args.axis_regression_weight * reg
        + args.axis_target_listwise_weight * listwise
        + args.axis_component_listwise_weight * comp_ce
        + args.axis_router_entropy_weight * router_entropy
        + args.axis_diversity_weight * diversity
        + args.axis_correction_l2_weight * corr_l2
        + args.axis_nll_weight * nll
    )
    return loss, {
        "reg": float(reg.detach().cpu()),
        "target_listwise": float(listwise.detach().cpu()),
        "component_listwise": float(comp_ce.detach().cpu()),
        "router_entropy": float(router_entropy.detach().cpu()),
        "diversity": float(diversity.detach().cpu()),
        "correction_l2": float(corr_l2.detach().cpu()),
        "nll": float(nll.detach().cpu()),
    }


def predict_axis_model(
    model: AxisConditionedCritic,
    pack: AxisFeaturePack,
    split: str,
    args: argparse.Namespace,
    *,
    device: torch.device,
    uniform_router: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cand_feat = getattr(pack, f"cand_{split}")
    axis_feat = getattr(pack, f"axis_{split}")
    edge_feat = getattr(pack, f"edge_{split}")
    cand_end = getattr(pack, f"cand_end_{split}")
    ctx = getattr(pack, f"ctx_{split}", None)
    if ctx is None:
        raise AttributeError("AxisFeaturePack must receive ctx_* attributes before prediction")
    model.eval()
    pred, final_w, router, axis_w, logvar = [], [], [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(cand_feat), args.axis_batch_size, args.seed + 59001, shuffle=False):
            out = model(
                to_tensor(cand_feat[idx], device),
                to_tensor(axis_feat[idx], device),
                to_tensor(edge_feat[idx], device),
                to_tensor(ctx[idx], device),
                to_tensor(cand_end[idx], device),
                uniform_router=uniform_router,
            )
            pred.append(out["pred_end"].cpu().numpy())
            final_w.append(out["final_weights"].cpu().numpy())
            router.append(out["router"].cpu().numpy())
            axis_w.append(out["axis_weights"].cpu().numpy())
            logvar.append(out["logvar"].cpu().numpy())
    return (
        np.concatenate(pred, axis=0).astype(np.float32),
        {
            "final_weights": np.concatenate(final_w, axis=0).astype(np.float32),
            "router": np.concatenate(router, axis=0).astype(np.float32),
            "axis_weights": np.concatenate(axis_w, axis=0).astype(np.float32),
            "logvar": np.concatenate(logvar, axis=0).astype(np.float32),
        },
    )


def train_axis_model(
    pack: AxisFeaturePack,
    args: argparse.Namespace,
    *,
    device: torch.device,
    variant: str,
    uniform_router: bool = False,
    shuffled_axis_labels: bool = False,
) -> tuple[AxisConditionedCritic, pd.DataFrame, float]:
    model = AxisConditionedCritic(
        cand_dim=pack.cand_train.shape[-1],
        axis_dim=pack.axis_train.shape[-1],
        edge_dim=pack.edge_train.shape[-1],
        ctx_dim=pack.ctx_train.shape[-1],
        n_axes=pack.axis_train.shape[2],
        n_horizons=len(args.horizons),
        hidden=args.axis_hidden,
        dropout=args.axis_dropout,
        heads=args.axis_heads,
        layers=args.axis_layers,
        correction_scale=args.axis_correction_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.axis_lr, weight_decay=args.axis_weight_decay)
    target_soft_train = pack.target_soft_train
    comp_soft_train = pack.component_soft_train
    if shuffled_axis_labels:
        rng = np.random.default_rng(args.seed + 61003)
        comp_soft_train = comp_soft_train[rng.permutation(len(comp_soft_train))]
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack.cand_train)
    for epoch in range(args.axis_epochs):
        model.train()
        batch_losses: list[float] = []
        batch_parts: list[dict[str, float]] = []
        for idx in closure.batches(n, args.axis_batch_size, args.seed + 60000 + epoch):
            out = model(
                to_tensor(pack.cand_train[idx], device),
                to_tensor(pack.axis_train[idx], device),
                to_tensor(pack.edge_train[idx], device),
                to_tensor(pack.ctx_train[idx], device),
                to_tensor(pack.cand_end_train[idx], device),
                uniform_router=uniform_router,
            )
            loss, parts = axis_distillation_loss(
                out,
                to_tensor(pack.target_end_train[idx], device),
                to_tensor(target_soft_train[idx], device),
                to_tensor(comp_soft_train[idx], device),
                args,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
            batch_parts.append(parts)
        if epoch == args.axis_epochs - 1 or epoch % max(1, args.axis_epochs // 5) == 0:
            pred_val, diag = predict_axis_model(model, pack, "val", args, device=device, uniform_router=uniform_router)
            score = validation_score_endpoint(pack.target_end_val, pred_val, args)
            row = {
                "epoch": int(epoch),
                "variant": variant,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                "val_endpoint_rmse": float(score),
                "router_entropy_mean": float(router_entropy_np(diag["router"])),
            }
            for key in ("reg", "target_listwise", "component_listwise", "router_entropy", "diversity", "correction_l2", "nll"):
                vals = [p[key] for p in batch_parts if key in p]
                row[f"train_{key}"] = float(np.mean(vals)) if vals else float("nan")
            rows.append(row)
            if score < best_val:
                best_val = float(score)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), float(best_val)


def validation_score_endpoint(true_end: np.ndarray, pred_end: np.ndarray, args: argparse.Namespace) -> float:
    horizons = qrc.risk_label_horizons(args)
    weights = qrc.risk_label_weights(args, horizons)
    lookup = {int(h): i for i, h in enumerate(args.horizons)}
    idx = [lookup[int(h)] for h in horizons if int(h) in lookup]
    if not idx:
        idx = list(range(len(args.horizons)))
    errs = np.sum((true_end[:, idx, :] - pred_end[:, idx, :]) ** 2, axis=-1)
    if weights is not None and len(weights) == len(idx):
        w = np.asarray(weights, dtype=np.float32)
        w = w / np.sum(w)
        return float(np.sqrt(np.mean(np.sum(errs * w[None, :], axis=1))))
    return float(np.sqrt(np.mean(errs)))


def router_entropy_np(router: np.ndarray) -> float:
    r = np.clip(np.asarray(router, dtype=np.float32), 1e-8, 1.0)
    return float(np.mean(-np.sum(r * np.log(r), axis=-1)))


def _select_pad_axis_context(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    max_features = max(1, int(max_features))
    ntr, nva, nte = len(x_train), len(x_val), len(x_test)
    if x_train.shape[1] == 0:
        return (
            np.zeros((ntr, max_features), dtype=np.float32),
            np.zeros((nva, max_features), dtype=np.float32),
            np.zeros((nte, max_features), dtype=np.float32),
            [],
        )
    var = np.nan_to_num(np.var(x_train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-min(max_features, x_train.shape[1]) :]
    out_train = np.nan_to_num(x_train[:, keep], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out_val = np.nan_to_num(x_val[:, keep], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out_test = np.nan_to_num(x_test[:, keep], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if out_train.shape[1] < max_features:
        pad = max_features - out_train.shape[1]
        out_train = np.pad(out_train, ((0, 0), (0, pad))).astype(np.float32)
        out_val = np.pad(out_val, ((0, 0), (0, pad))).astype(np.float32)
        out_test = np.pad(out_test, ((0, 0), (0, pad))).astype(np.float32)
    return out_train, out_val, out_test, keep.tolist()


def build_edge_latent_axis_context(
    *,
    arrays: audit.SplitArrays,
    component_names: list[str],
    args: argparse.Namespace,
    shuffle_edge_context: bool = False,
    zero_edge_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build axis-specific pseudo-edge/context tokens.

    The current LaChance grid does not expose explicit i-j neighbour rows.
    This function therefore builds a structured edge-latent proxy from the
    already extracted context families: flow, boundary, crowding/topology,
    morphology and raw-context.  It is still axis-specific and train-only
    selected, but should be interpreted as pseudo-edge context rather than a
    true kNN edge list.
    """
    max_features = max(1, int(args.edge_latent_max_features))
    if (not args.use_edge_latent_context) or zero_edge_context:
        shape_train = (len(arrays.residual_train), len(component_names), max_features)
        shape_val = (len(arrays.residual_val), len(component_names), max_features)
        shape_test = (len(arrays.residual_test), len(component_names), max_features)
        return (
            np.zeros(shape_train, dtype=np.float32),
            np.zeros(shape_val, dtype=np.float32),
            np.zeros(shape_test, dtype=np.float32),
            {"mode": "zero", "max_features": max_features},
        )

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    meta: dict[str, Any] = {"mode": "block_proxy", "max_features": max_features, "axis_sources": {}}
    fallback = "all_context" if "all_context" in arrays.x_train else next(iter(arrays.x_train.keys()))
    for name in component_names:
        source = name if name in arrays.x_train else fallback if name != "decomposition_student" else fallback
        xtr = arrays.x_train.get(source, np.zeros((len(arrays.residual_train), 0), dtype=np.float32))
        xva = arrays.x_val.get(source, np.zeros((len(arrays.residual_val), 0), dtype=np.float32))
        xte = arrays.x_test.get(source, np.zeros((len(arrays.residual_test), 0), dtype=np.float32))
        etr, eva, ete, keep = _select_pad_axis_context(xtr, xva, xte, max_features)
        train_parts.append(etr)
        val_parts.append(eva)
        test_parts.append(ete)
        meta["axis_sources"][name] = {"source": source, "n_selected": len(keep)}

    edge_train = np.stack(train_parts, axis=1).astype(np.float32)
    edge_val = np.stack(val_parts, axis=1).astype(np.float32)
    edge_test = np.stack(test_parts, axis=1).astype(np.float32)
    if shuffle_edge_context:
        rng = np.random.default_rng(args.seed + 63001)
        edge_train = edge_train[rng.permutation(len(edge_train))]
        edge_val = edge_val[rng.permutation(len(edge_val))]
        edge_test = edge_test[rng.permutation(len(edge_test))]
        meta["shuffled"] = True
    else:
        meta["shuffled"] = False
    return edge_train, edge_val, edge_test, meta


def build_axis_feature_pack(
    *,
    q_train: qrc.QueryOutputs,
    q_val: qrc.QueryOutputs,
    q_test: qrc.QueryOutputs,
    arrays: audit.SplitArrays,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    args: argparse.Namespace,
    shuffle_axis_tokens: bool = False,
    shuffle_edge_context: bool = False,
    zero_edge_context: bool = False,
) -> AxisFeaturePack:
    comp_train = component_axes.train
    comp_val = component_axes.val
    comp_test = component_axes.test
    if shuffle_axis_tokens:
        rng = np.random.default_rng(args.seed + 62001)
        comp_train = comp_train[rng.permutation(len(comp_train))]
        comp_val = comp_val[rng.permutation(len(comp_val))]
        comp_test = comp_test[rng.permutation(len(comp_test))]

    cand_train, _ = qrc.query_sequence_features(
        query_pred=q_train.query_pred,
        base=arrays.base_train,
        route_logits=q_train.route_logits,
        ctx=ctx_train,
        horizons=args.horizons,
        include_context=False,
        include_query_id=args.risk_include_query_id,
    )
    cand_val, _ = qrc.query_sequence_features(
        query_pred=q_val.query_pred,
        base=arrays.base_val,
        route_logits=q_val.route_logits,
        ctx=ctx_val,
        horizons=args.horizons,
        include_context=False,
        include_query_id=args.risk_include_query_id,
    )
    cand_test, cand_names = qrc.query_sequence_features(
        query_pred=q_test.query_pred,
        base=arrays.base_test,
        route_logits=q_test.route_logits,
        ctx=ctx_test,
        horizons=args.horizons,
        include_context=False,
        include_query_id=args.risk_include_query_id,
    )
    candidate_edge_meta: dict[str, Any] = {"enabled": False}
    if args.use_candidate_edge_compat:
        edge_tr, edge_va, edge_te, candidate_edge_meta = _select_edge_compat_block(arrays, args)
        ce_train, ce_names = candidate_edge_compat_features(
            q_train.query_pred,
            edge_tr,
            args.horizons,
            include_broadcast=bool(args.candidate_edge_compat_include_broadcast),
        )
        ce_val, _ = candidate_edge_compat_features(
            q_val.query_pred,
            edge_va,
            args.horizons,
            include_broadcast=bool(args.candidate_edge_compat_include_broadcast),
        )
        ce_test, _ = candidate_edge_compat_features(
            q_test.query_pred,
            edge_te,
            args.horizons,
            include_broadcast=bool(args.candidate_edge_compat_include_broadcast),
        )
        cand_train = np.concatenate([cand_train, ce_train], axis=-1).astype(np.float32)
        cand_val = np.concatenate([cand_val, ce_val], axis=-1).astype(np.float32)
        cand_test = np.concatenate([cand_test, ce_test], axis=-1).astype(np.float32)
        cand_names = list(cand_names) + ce_names
        candidate_edge_meta["feature_dim_added"] = int(ce_train.shape[-1])
    cand_train, cand_val, cand_test, cand_scaler = standardize_candidate_features(cand_train, cand_val, cand_test)

    axis_train, _ = axis_candidate_features(query_pred=q_train.query_pred, component_pred=comp_train, horizons=args.horizons)
    axis_val, _ = axis_candidate_features(query_pred=q_val.query_pred, component_pred=comp_val, horizons=args.horizons)
    axis_test, axis_names = axis_candidate_features(query_pred=q_test.query_pred, component_pred=comp_test, horizons=args.horizons)
    axis_train, axis_val, axis_test, axis_scaler = standardize_axis_features(axis_train, axis_val, axis_test)
    edge_train, edge_val, edge_test, edge_meta = build_edge_latent_axis_context(
        arrays=arrays,
        component_names=component_axes.names,
        args=args,
        shuffle_edge_context=shuffle_edge_context,
        zero_edge_context=zero_edge_context,
    )

    target_err_train = target_candidate_errors(q_train.query_pred, arrays.residual_train, args.horizons)
    target_err_val = target_candidate_errors(q_val.query_pred, arrays.residual_val, args.horizons)
    comp_err_train = component_candidate_errors(q_train.query_pred, comp_train, args.horizons)
    comp_err_val = component_candidate_errors(q_val.query_pred, comp_val, args.horizons)

    pack = AxisFeaturePack(
        cand_train=cand_train,
        cand_val=cand_val,
        cand_test=cand_test,
        axis_train=axis_train,
        axis_val=axis_val,
        axis_test=axis_test,
        edge_train=edge_train,
        edge_val=edge_val,
        edge_test=edge_test,
        cand_end_train=candidate_endpoints(q_train.query_pred, args.horizons),
        cand_end_val=candidate_endpoints(q_val.query_pred, args.horizons),
        cand_end_test=candidate_endpoints(q_test.query_pred, args.horizons),
        target_end_train=residual_endpoints(arrays.residual_train, args.horizons),
        target_end_val=residual_endpoints(arrays.residual_val, args.horizons),
        target_end_test=residual_endpoints(arrays.residual_test, args.horizons),
        target_soft_train=soft_labels_from_distance_np(target_err_train, args.axis_target_temperature),
        target_soft_val=soft_labels_from_distance_np(target_err_val, args.axis_target_temperature),
        component_soft_train=soft_labels_from_distance_np(comp_err_train, args.axis_component_temperature),
        component_soft_val=soft_labels_from_distance_np(comp_err_val, args.axis_component_temperature),
        component_names=list(component_axes.names),
        scalers={
            "candidate": finite_json(cand_scaler),
            "axis": finite_json(axis_scaler),
            "candidate_feature_names": cand_names,
            "axis_feature_names": axis_names,
            "candidate_edge_compat": finite_json(candidate_edge_meta),
            "shuffled_axis_tokens": bool(shuffle_axis_tokens),
            "edge_latent_context": finite_json(edge_meta),
            "zero_edge_context": bool(zero_edge_context),
        },
    )
    pack.ctx_train = ctx_train.astype(np.float32)
    pack.ctx_val = ctx_val.astype(np.float32)
    pack.ctx_test = ctx_test.astype(np.float32)
    return pack


def add_axis_variant_rows(
    *,
    rows: list[dict[str, Any]],
    train_logs: list[pd.DataFrame],
    arrays: audit.SplitArrays,
    pack: AxisFeaturePack,
    args: argparse.Namespace,
    device: torch.device,
    name: str,
    uniform_router: bool = False,
    shuffled_axis_labels: bool = False,
) -> dict[str, Any]:
    model, log, best_val = train_axis_model(
        pack,
        args,
        device=device,
        variant=name,
        uniform_router=uniform_router,
        shuffled_axis_labels=shuffled_axis_labels,
    )
    train_logs.append(log)
    pred_test, diag = predict_axis_model(model, pack, "test", args, device=device, uniform_router=uniform_router)
    logv = np.clip(diag["logvar"], args.axis_logvar_min, args.axis_logvar_max)
    err2 = np.square(pack.target_end_test - pred_test)
    test_nll = float(0.5 * np.mean(err2 * np.exp(-logv) + logv))
    unc = np.sqrt(np.mean(np.exp(logv), axis=-1)).reshape(-1)
    err = np.linalg.norm(pack.target_end_test - pred_test, axis=-1).reshape(-1)
    if np.std(unc) < 1e-8 or np.std(err) < 1e-8:
        unc_err_corr = 0.0
    else:
        unc_err_corr = float(np.corrcoef(unc, err)[0, 1])
    rows.extend(
        endpoint_rows_from_residual_endpoints(
            arrays=arrays,
            residual_endpoint_pred=pred_test,
            horizons=args.horizons,
            label=name,
            extra={
                "stage": "axis_conditioned_distillation",
                "risk_variant": name,
                "val_endpoint_rmse": best_val,
                "router_entropy": router_entropy_np(diag["router"]),
                "test_gaussian_nll": test_nll,
                "uncertainty_error_corr": unc_err_corr,
            },
        )
    )
    return {
        "variant": name,
        "val_endpoint_rmse": float(best_val),
        "router_entropy_test": router_entropy_np(diag["router"]),
        "router_mean": np.mean(diag["router"], axis=0).tolist(),
        "final_weight_entropy": float(router_entropy_np(diag["final_weights"])),
        "test_gaussian_nll": test_nll,
        "uncertainty_error_corr": unc_err_corr,
    }


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    extra_feature_meta = rpr.attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_scaler = rpr.prepare_context(args, arrays, posterior, student, blocks, device)

    # Reuse the component-axis builder from the previous runner.  This makes
    # the new runner directly comparable to component-aware v1.
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("Axis-conditioned critic requires at least two component axes")
    component_axes.probe.to_csv(args.out_dir / "axis_component_axis_probe.csv", index=False)

    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand_train = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    rows: list[dict[str, Any]] = []
    rows.extend(rpr.endpoint_rows(arrays, seq.mean_candidate_residual(cand_test), "candidate_mean", args, {"stage": "candidate_control"}))
    for k in args.oracle_k:
        rows.extend(rpr.endpoint_rows(arrays, rpr.proto.oracle_from_set(cand_test.residual[:, : int(k)], arrays.residual_test, args.horizons), f"candidate_endpoint_oracle@{k}", args, {"stage": "candidate_endpoint_oracle", "oracle_k": int(k)}))

    risk_logs: list[pd.DataFrame] = []
    axis_logs: list[pd.DataFrame] = []
    scaler_meta: dict[str, Any] = {"context": finite_json(ctx_scaler)}
    axis_diag_rows: list[dict[str, Any]] = []
    methods = [s.strip() for s in args.prototype_methods.split(",") if s.strip()]
    counts = parse_ints(args.prototype_k)

    for method in methods:
        for m in counts:
            if m > args.candidate_k:
                continue
            ptr, itr = rpr.build_prototype_set(cand_train, method, m, seed=args.seed + 8001)
            pva, iva = rpr.build_prototype_set(cand_val, method, m, seed=args.seed + 9001)
            pte, ite = rpr.build_prototype_set(cand_test, method, m, seed=args.seed + 10001)
            q_train = rpr.make_query_outputs(ptr, itr, method, args.candidate_k, arrays.residual_train, args.horizons)
            q_val = rpr.make_query_outputs(pva, iva, method, args.candidate_k, arrays.residual_val, args.horizons)
            q_test = rpr.make_query_outputs(pte, ite, method, args.candidate_k, arrays.residual_test, args.horizons)
            prefix = f"{method}{m}"

            rows.extend(rpr.endpoint_rows(arrays, q_test.query_oracle, f"{prefix}_oracle", args, {"stage": "prototype_oracle", "prototype": method, "prototype_k": m}))
            rows.extend(rpr.endpoint_rows(arrays, q_test.weighted_pred, f"{prefix}_prior_weighted", args, {"stage": "prototype_prior", "prototype": method, "prototype_k": m}))
            rows.extend(rpr.endpoint_rows(arrays, np.mean(q_test.query_pred, axis=1).astype(np.float32), f"{prefix}_mean", args, {"stage": "prototype_mean", "prototype": method, "prototype_k": m}))

            if not args.skip_scalar_baselines:
                rpr.add_refiner_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_risk_full",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    device=device,
                    include_context=True,
                    shuffled_labels=False,
                    shuffled_context=False,
                    risk_logs=risk_logs,
                    scaler_meta=scaler_meta,
                )
                rpr.add_refiner_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_component_risk_full",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    device=device,
                    include_context=True,
                    shuffled_labels=False,
                    shuffled_context=False,
                    risk_logs=risk_logs,
                    scaler_meta=scaler_meta,
                    component_train=component_axes.train,
                    component_val=component_axes.val,
                    component_test=component_axes.test,
                    include_components=True,
                )

            pack = build_axis_feature_pack(
                q_train=q_train,
                q_val=q_val,
                q_test=q_test,
                arrays=arrays,
                ctx_train=ctx_train,
                ctx_val=ctx_val,
                ctx_test=ctx_test,
                component_axes=component_axes,
                args=args,
                shuffle_axis_tokens=False,
            )
            scaler_meta[f"{prefix}_axis_features"] = finite_json(pack.scalers)

            axis_diag_rows.append(
                add_axis_variant_rows(
                    rows=rows,
                    train_logs=axis_logs,
                    arrays=arrays,
                    pack=pack,
                    args=args,
                    device=device,
                    name=f"{prefix}_axis_full",
                )
            )
            axis_diag_rows.append(
                add_axis_variant_rows(
                    rows=rows,
                    train_logs=axis_logs,
                    arrays=arrays,
                    pack=pack,
                    args=args,
                    device=device,
                    name=f"{prefix}_axis_no_router",
                    uniform_router=True,
                )
            )
            if args.use_edge_latent_context:
                no_edge_pack = build_axis_feature_pack(
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    arrays=arrays,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    component_axes=component_axes,
                    args=args,
                    zero_edge_context=True,
                )
                axis_diag_rows.append(
                    add_axis_variant_rows(
                        rows=rows,
                        train_logs=axis_logs,
                        arrays=arrays,
                        pack=no_edge_pack,
                        args=args,
                        device=device,
                        name=f"{prefix}_axis_no_edge_context",
                    )
                )
            if args.include_controls:
                shuffled_pack = build_axis_feature_pack(
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    arrays=arrays,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    component_axes=component_axes,
                    args=args,
                    shuffle_axis_tokens=True,
                )
                axis_diag_rows.append(
                    add_axis_variant_rows(
                        rows=rows,
                        train_logs=axis_logs,
                        arrays=arrays,
                        pack=shuffled_pack,
                        args=args,
                        device=device,
                        name=f"{prefix}_axis_shuffled_tokens",
                    )
                )
                axis_diag_rows.append(
                    add_axis_variant_rows(
                        rows=rows,
                        train_logs=axis_logs,
                        arrays=arrays,
                        pack=pack,
                        args=args,
                        device=device,
                        name=f"{prefix}_axis_shuffled_component_labels",
                        shuffled_axis_labels=True,
                    )
                )
                if args.use_edge_latent_context:
                    shuffled_edge_pack = build_axis_feature_pack(
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        arrays=arrays,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        args=args,
                        shuffle_edge_context=True,
                    )
                    axis_diag_rows.append(
                        add_axis_variant_rows(
                            rows=rows,
                            train_logs=axis_logs,
                            arrays=arrays,
                            pack=shuffled_edge_pack,
                            args=args,
                            device=device,
                            name=f"{prefix}_axis_shuffled_edge_context",
                        )
                    )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "axis_conditioned_distillation_summary.csv", index=False)
    if risk_logs:
        pd.concat(risk_logs, ignore_index=True).to_csv(args.out_dir / "scalar_risk_train_log.csv", index=False)
    if axis_logs:
        pd.concat(axis_logs, ignore_index=True).to_csv(args.out_dir / "axis_distillation_train_log.csv", index=False)
    if axis_diag_rows:
        pd.DataFrame(axis_diag_rows).to_csv(args.out_dir / "axis_distillation_diagnostics.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    run_meta = finite_json(vars(args))
    run_meta["component_axis_names"] = component_axes.names
    run_meta["extra_feature_block"] = finite_json(extra_feature_meta)
    (args.out_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(scaler_meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = ["# Axis-Conditioned Candidate Distillation Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- prototype_methods: `{args.prototype_methods}`")
    lines.append(f"- prototype_k: `{args.prototype_k}`")
    lines.append(f"- component_axis_blocks: `{args.component_axis_blocks}`")
    lines.append(f"- component_axis_model: `{args.component_axis_model}`")
    if getattr(args, "extra_feature_grid", None):
        lines.append(f"- extra_feature_grid: `{args.extra_feature_grid}`")
        lines.append(f"- extra_feature_block_name: `{args.extra_feature_block_name}`")
        lines.append(f"- extra_feature_prefixes: `{args.extra_feature_prefixes}`")
        lines.append(f"- extra_feature_merge_all_context: `{args.extra_feature_merge_all_context}`")
    lines.append(f"- axis_hidden: `{args.axis_hidden}`")
    lines.append(f"- axis_correction_scale: `{args.axis_correction_scale}`")
    lines.append(f"- axis_nll_weight: `{args.axis_nll_weight}`")
    lines.append(f"- use_edge_latent_context: `{bool(args.use_edge_latent_context)}`")
    if args.use_edge_latent_context:
        lines.append(f"- edge_latent_max_features: `{args.edge_latent_max_features}`")
    lines.append(f"- use_candidate_edge_compat: `{bool(args.use_candidate_edge_compat)}`")
    if args.use_candidate_edge_compat:
        lines.append(f"- candidate_edge_compat_block: `{args.candidate_edge_compat_block}`")
        lines.append(f"- candidate_edge_compat_max_features: `{args.candidate_edge_compat_max_features}`")
        lines.append(f"- candidate_edge_compat_include_broadcast: `{bool(args.candidate_edge_compat_include_broadcast)}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{h}")
        sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
        for _, row in sub.head(32).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, gain={row['gain_vs_base_pct']:.2f}%")
    lines.append("\n## Decision Notes")
    lines.append("- The gate is passed only if `axis_full` beats scalar/component v1 and the shuffled-axis controls.")
    lines.append("- If `axis_no_router` is close to `axis_full`, the axis router is not yet learning cell-specific mechanism weights.")
    lines.append("- If shuffled-axis variants are close to full, the model is still exploiting candidate geometry rather than decomposed causal axes.")
    lines.append("- If `axis_no_edge_context` or `axis_shuffled_edge_context` is close to full, the pseudo-edge latent context is not yet a useful interaction signal.")
    (out_dir / "axis_conditioned_distillation_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.add_argument("--prototype-methods", type=str, default="fps_shape")
    parser.add_argument("--prototype-k", type=str, default="16")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--skip-scalar-baselines", action="store_true")
    parser.add_argument(
        "--component-axis-blocks",
        type=str,
        default="self,flow,morphology,boundary,crowding,raw_context,all_context",
    )
    parser.add_argument("--component-include-student-axis", action="store_true")
    parser.add_argument("--component-axis-model", type=str, default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--component-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--component-axis-max-features", type=int, default=256)
    parser.add_argument("--component-axis-hidden", type=int, default=128)
    parser.add_argument("--component-axis-epochs", type=int, default=16)
    parser.add_argument("--component-axis-lr", type=float, default=8e-4)
    parser.add_argument("--component-axis-weight-decay", type=float, default=1e-4)
    parser.add_argument("--component-axis-dropout", type=float, default=0.05)
    parser.add_argument("--component-attention-temperature", type=float, default=6.0)

    parser.add_argument("--axis-hidden", type=int, default=192)
    parser.add_argument("--axis-heads", type=int, default=4)
    parser.add_argument("--axis-layers", type=int, default=1)
    parser.add_argument("--axis-epochs", type=int, default=22)
    parser.add_argument("--axis-batch-size", type=int, default=384)
    parser.add_argument("--axis-lr", type=float, default=8e-4)
    parser.add_argument("--axis-weight-decay", type=float, default=1e-4)
    parser.add_argument("--axis-dropout", type=float, default=0.05)
    parser.add_argument("--axis-target-temperature", type=float, default=8.0)
    parser.add_argument("--axis-component-temperature", type=float, default=8.0)
    parser.add_argument("--axis-regression-weight", type=float, default=1.0)
    parser.add_argument("--axis-target-listwise-weight", type=float, default=0.25)
    parser.add_argument("--axis-component-listwise-weight", type=float, default=0.20)
    parser.add_argument("--axis-router-entropy-weight", type=float, default=0.0)
    parser.add_argument("--axis-diversity-weight", type=float, default=0.015)
    parser.add_argument("--axis-correction-scale", type=float, default=0.0)
    parser.add_argument("--axis-correction-l2-weight", type=float, default=0.001)
    parser.add_argument("--axis-nll-weight", type=float, default=0.0)
    parser.add_argument("--axis-logvar-min", type=float, default=-6.0)
    parser.add_argument("--axis-logvar-max", type=float, default=6.0)
    parser.add_argument("--use-edge-latent-context", action="store_true")
    parser.add_argument("--edge-latent-max-features", type=int, default=64)
    parser.add_argument("--use-candidate-edge-compat", action="store_true")
    parser.add_argument("--candidate-edge-compat-block", type=str, default="explicit_edge")
    parser.add_argument("--candidate-edge-compat-max-features", type=int, default=32)
    parser.add_argument("--candidate-edge-compat-include-broadcast", action="store_true")
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="nx_")
    parser.add_argument("--extra-feature-block-name", type=str, default="networkx")
    parser.add_argument("--extra-feature-max-cols", type=int, default=96)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    args = parser.parse_args()

    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 3000)
        args.max_val_rows = min(args.max_val_rows, 1000)
        args.max_test_rows = min(args.max_test_rows, 1200)
        args.posterior_epochs = min(args.posterior_epochs, 6)
        args.student_epochs = min(args.student_epochs, 6)
        args.learned_route_epochs = min(args.learned_route_epochs, 4)
        args.critic_epochs = min(args.critic_epochs, 6)
        args.risk_epochs = min(args.risk_epochs, 6)
        args.axis_epochs = min(args.axis_epochs, 6)
        args.hgbdt_max_iter = min(args.hgbdt_max_iter, 60)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.max_all_features = min(args.max_all_features, 192)
        args.max_critic_context_features = min(args.max_critic_context_features, 192)
    run(args)


if __name__ == "__main__":
    main()
