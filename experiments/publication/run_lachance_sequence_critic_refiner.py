#!/usr/bin/env python3
"""Sequence Trajectory Critic-Refiner for LaChance residual candidates.

This runner is the first explicit implementation of the fourth block in the
current research architecture:

    clean-best/context backbone
    -> decomposition posterior/student generator
    -> learned trajectory candidate cloud
    -> sequence critic-refiner

The critic is trained only with causal context and generated candidates at
inference time.  The target future is used during training only to construct
soft-oracle/listwise supervision and regression loss.
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


DEFAULT_FEATURES = audit.DEFAULT_FEATURES
DEFAULT_OUT = ROOT / "outputs" / "sequence_critic_refiner_2026-06-24"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def flatten_blocks(xdict: dict[str, np.ndarray], blocks: list[str]) -> np.ndarray:
    parts = [xdict[b].astype(np.float32) for b in blocks if b in xdict and xdict[b].shape[1] > 0]
    if not parts:
        return np.zeros((len(next(iter(xdict.values()))), 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    def z(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return z(train), z(val), z(test), {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}


def endpoint_residual(residual_steps: np.ndarray, h: int) -> np.ndarray:
    return np.sum(residual_steps[:, : int(h), :], axis=1)


def endpoint_residual_t(residual_steps: torch.Tensor, h: int) -> torch.Tensor:
    return torch.sum(residual_steps[:, : int(h), :], dim=1)


@dataclass
class CandidatePack:
    residual: np.ndarray  # n,k,h,2
    z: np.ndarray  # n,k,latent
    z_eps: np.ndarray  # n,k,latent
    logprob: np.ndarray  # n,k,1
    mode_prob: np.ndarray  # n,k,modes
    features: np.ndarray  # n,k,d
    oracle_dist: np.ndarray  # n,k
    route_mode: np.ndarray | None = None  # n,k assigned generator mode, if any


def decode_latents(
    posterior: closure.PosteriorPack,
    z: np.ndarray,
    *,
    max_horizon: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    return closure.decode_residual(posterior.model, posterior.scaler, z, max_horizon=max_horizon, device=device, batch_size=batch_size)


def build_candidate_features(
    *,
    residual: np.ndarray,
    base: np.ndarray,
    z_eps: np.ndarray,
    logprob: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, list[str]]:
    n, k, hmax, _ = residual.shape
    total_steps = base[:, None, None, :] + residual
    endpoints = np.cumsum(total_steps, axis=2)
    base_end = np.stack([base * float(h) for h in horizons], axis=1)  # n,hh,2
    endpoint_sel = np.stack([endpoints[:, :, h - 1, :] for h in horizons], axis=2)  # n,k,hh,2
    endpoint_delta_base = endpoint_sel - base_end[:, None, :, :]

    speed = np.linalg.norm(total_steps, axis=-1)
    acc = np.diff(total_steps, axis=2)
    acc_norm = np.linalg.norm(acc, axis=-1) if hmax > 1 else np.zeros((n, k, 1), dtype=np.float32)
    path_len = np.sum(speed, axis=2, keepdims=True)
    net = endpoint_sel[:, :, -1, :]
    net_norm = np.linalg.norm(net, axis=-1, keepdims=True)
    persistence = net_norm / np.maximum(path_len, EPS)
    step0 = total_steps[:, :, 0, :]
    final = total_steps[:, :, -1, :]
    dot = np.sum(step0 * final, axis=-1, keepdims=True)
    turn_end_cos = dot / np.maximum(np.linalg.norm(step0, axis=-1, keepdims=True) * np.linalg.norm(final, axis=-1, keepdims=True), EPS)
    if hmax > 1:
        prev = total_steps[:, :, :-1, :]
        nxt = total_steps[:, :, 1:, :]
        denom = np.maximum(np.linalg.norm(prev, axis=-1) * np.linalg.norm(nxt, axis=-1), EPS)
        turn_cos = np.sum(prev * nxt, axis=-1) / denom
        turn_sin = (prev[..., 0] * nxt[..., 1] - prev[..., 1] * nxt[..., 0]) / denom
    else:
        turn_cos = np.zeros((n, k, 1), dtype=np.float32)
        turn_sin = np.zeros((n, k, 1), dtype=np.float32)

    pieces: list[np.ndarray] = []
    names: list[str] = []
    def add(name: str, arr: np.ndarray) -> None:
        a = arr.reshape(n, k, -1).astype(np.float32)
        pieces.append(a)
        for j in range(a.shape[-1]):
            names.append(f"{name}_{j}" if a.shape[-1] > 1 else name)

    add("residual_steps", residual)
    add("total_steps", total_steps)
    add("endpoint", endpoint_sel)
    add("endpoint_delta_base", endpoint_delta_base)
    add("endpoint_mag", np.linalg.norm(endpoint_sel, axis=-1))
    add("endpoint_delta_mag", np.linalg.norm(endpoint_delta_base, axis=-1))
    add("speed_mean", np.mean(speed, axis=2, keepdims=True))
    add("speed_std", np.std(speed, axis=2, keepdims=True))
    add("speed_first", speed[:, :, :1])
    add("speed_last", speed[:, :, -1:])
    add("acc_mean", np.mean(acc_norm, axis=2, keepdims=True))
    add("acc_std", np.std(acc_norm, axis=2, keepdims=True))
    add("path_len", path_len)
    add("net_norm", net_norm)
    add("persistence", persistence)
    add("turn_end_cos", turn_end_cos)
    add("turn_cos_mean", np.mean(turn_cos, axis=2, keepdims=True))
    add("turn_cos_std", np.std(turn_cos, axis=2, keepdims=True))
    add("turn_sin_mean", np.mean(turn_sin, axis=2, keepdims=True))
    add("latent_eps", z_eps)
    add("latent_logprob", logprob)
    feat = np.concatenate(pieces, axis=-1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat, names


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    ez = np.exp(z)
    return (ez / np.maximum(np.sum(ez, axis=axis, keepdims=True), EPS)).astype(np.float32)


def candidate_mode_features(
    z: np.ndarray,
    pred: dict[str, np.ndarray],
    posterior: closure.PosteriorPack,
    *,
    mode_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = posterior.centers.astype(np.float32)
    diff = z[:, :, None, :] - centers[None, None, :, :]
    dist = np.sum(diff * diff, axis=-1).astype(np.float32)
    mode_prob = softmax_np(-dist / max(float(mode_temperature), 1e-6), axis=-1)
    nearest = np.argmin(dist, axis=-1)
    min_dist = np.take_along_axis(dist, nearest[:, :, None], axis=-1)
    sorted_dist = np.sort(dist, axis=-1)
    margin = (sorted_dist[:, :, 1:2] - sorted_dist[:, :, 0:1]).astype(np.float32) if dist.shape[-1] > 1 else np.zeros_like(min_dist)
    probs = softmax_np(pred["logits"], axis=-1)
    nearest_prob = np.take_along_axis(probs[:, None, :].repeat(z.shape[1], axis=1), nearest[:, :, None], axis=-1)
    log_probs = np.log(np.maximum(probs, EPS))
    # Mixture energy: low is compatible with a high-probability mode center.
    energy_terms = log_probs[:, None, :] - 0.5 * dist
    mix_energy = -np.log(np.maximum(np.sum(np.exp(energy_terms - np.max(energy_terms, axis=-1, keepdims=True)), axis=-1, keepdims=True), EPS)) - np.max(energy_terms, axis=-1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1, keepdims=True)
    entropy = entropy[:, None, :].repeat(z.shape[1], axis=1)
    logvar_mean = np.mean(pred["logvar"], axis=-1, keepdims=True)[:, None, :].repeat(z.shape[1], axis=1)
    unc_arr = np.asarray(pred["uncertainty"], dtype=np.float32)
    if unc_arr.ndim == 1:
        unc_arr = unc_arr[:, None]
    unc = unc_arr[:, None, :].repeat(z.shape[1], axis=1)
    z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
    mu_norm = np.linalg.norm(pred["mu"], axis=-1, keepdims=True)[:, None, :].repeat(z.shape[1], axis=1)
    out = np.concatenate([min_dist, margin, nearest_prob, mix_energy.astype(np.float32), entropy, logvar_mean, unc, z_norm, mu_norm], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), mode_prob.astype(np.float32)


def decomposition_context_features(pred: dict[str, np.ndarray], *, mode_k: int) -> np.ndarray:
    """Causal feature packet built by the decomposition student.

    This is the feature-builder part of the decomposition module: no future
    residual is used here, only the student prior outputs available at
    inference.
    """
    mode_p = softmax_np(pred["logits"], axis=-1)
    entropy = -np.sum(mode_p * np.log(np.maximum(mode_p, EPS)), axis=-1, keepdims=True)
    top_prob = np.max(mode_p, axis=-1, keepdims=True)
    margin = np.sort(mode_p, axis=-1)[:, -1:] - np.sort(mode_p, axis=-1)[:, -2:-1] if mode_k > 1 else top_prob
    unc = np.asarray(pred["uncertainty"], dtype=np.float32)
    if unc.ndim == 1:
        unc = unc[:, None]
    pieces = [
        pred["mu"],
        pred["logvar"],
        mode_p,
        entropy.astype(np.float32),
        top_prob.astype(np.float32),
        margin.astype(np.float32),
        unc.astype(np.float32),
        pred["gates"].astype(np.float32),
    ]
    return np.nan_to_num(np.concatenate(pieces, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def route_feature_packet(
    *,
    n: int,
    k: int,
    mode_k: int,
    route_mode: np.ndarray | None,
    mode_prior: np.ndarray | None,
    route_rank: np.ndarray | None,
) -> np.ndarray:
    """Candidate-family features for route-conditioned generation."""
    feat = np.zeros((n, k, mode_k + 2), dtype=np.float32)
    if route_mode is None:
        return feat
    route_mode = np.asarray(route_mode, dtype=np.int64)
    valid = (route_mode >= 0) & (route_mode < mode_k)
    rows = np.arange(n)[:, None].repeat(k, axis=1)
    cols = np.arange(k)[None, :].repeat(n, axis=0)
    feat[rows[valid], cols[valid], route_mode[valid]] = 1.0
    if mode_prior is not None:
        feat[:, :, mode_k : mode_k + 1] = mode_prior.astype(np.float32)
    if route_rank is not None:
        feat[:, :, mode_k + 1 : mode_k + 2] = route_rank.astype(np.float32)
    return feat


def generate_candidates(
    arrays: audit.SplitArrays,
    posterior: closure.PosteriorPack,
    student: closure.ComponentStudentPrior,
    blocks: list[str],
    args: argparse.Namespace,
    *,
    split_name: str,
    device: torch.device,
) -> CandidatePack:
    xdict = getattr(arrays, f"x_{split_name}")
    base = getattr(arrays, f"base_{split_name}")
    residual_true = getattr(arrays, f"residual_{split_name}")
    pred = closure.predict_student(student, xdict, blocks, device=device, batch_size=args.batch_size)
    rng = np.random.default_rng(args.seed + {"train": 3301, "val": 4401, "test": 5501}[split_name])
    n = len(base)
    std = np.exp(0.5 * np.clip(pred["logvar"], -8, 5)).astype(np.float32) * float(args.sample_scale)
    eps = rng.normal(size=(n, args.candidate_k, args.latent_dim)).astype(np.float32)
    route_mode = None
    mode_prior = None
    route_rank = None
    if args.candidate_generator == "route_conditioned":
        mode_p = softmax_np(pred["logits"], axis=-1)
        order = np.argsort(-mode_p, axis=1)
        route_count = int(args.route_mode_count) if int(args.route_mode_count) > 0 else int(args.mode_k)
        route_count = max(1, min(route_count, int(args.mode_k), int(args.candidate_k)))
        rank_idx = (np.arange(args.candidate_k, dtype=np.int64) % route_count)[None, :].repeat(n, axis=0)
        route_mode = np.take_along_axis(order[:, :route_count], rank_idx, axis=1).astype(np.int64)
        centers = posterior.centers[route_mode].astype(np.float32)
        mode_prior = np.take_along_axis(mode_p, route_mode, axis=1)[:, :, None].astype(np.float32)
        route_rank = (rank_idx.astype(np.float32) / max(float(route_count - 1), 1.0))[:, :, None]
        std_route = std[:, None, :] * float(args.route_noise_scale)
        if args.route_include_centers:
            eps[:, :route_count, :] = 0.0
        z = (1.0 - float(args.route_center_mix)) * pred["mu"][:, None, :] + float(args.route_center_mix) * centers + eps * std_route
    else:
        z = pred["mu"][:, None, :] + eps * std[:, None, :]
    z_flat = z.reshape(n * args.candidate_k, args.latent_dim)
    residual = decode_latents(posterior, z_flat, max_horizon=args.max_horizon, device=device, batch_size=args.batch_size)
    residual = residual.reshape(n, args.candidate_k, args.max_horizon, 2).astype(np.float32)
    z_eps = eps.astype(np.float32)
    logprob = -0.5 * np.sum(eps * eps + np.log(2.0 * np.pi), axis=-1, keepdims=True).astype(np.float32)
    true_flat = audit.flatten_residual(residual_true)
    cand_flat = residual.reshape(n, args.candidate_k, -1)
    oracle_dist = np.mean((cand_flat - true_flat[:, None, :]) ** 2, axis=-1).astype(np.float32)
    features, _ = build_candidate_features(residual=residual, base=base, z_eps=z_eps, logprob=logprob, horizons=args.horizons)
    mode_features, mode_prob = candidate_mode_features(z, pred, posterior, mode_temperature=args.candidate_mode_temperature)
    route_features = route_feature_packet(
        n=n,
        k=args.candidate_k,
        mode_k=args.mode_k,
        route_mode=route_mode,
        mode_prior=mode_prior,
        route_rank=route_rank,
    )
    features = np.concatenate([features, mode_features, route_features], axis=-1).astype(np.float32)
    return CandidatePack(
        residual=residual,
        z=z.astype(np.float32),
        z_eps=z_eps,
        logprob=logprob,
        mode_prob=mode_prob,
        features=features,
        oracle_dist=oracle_dist,
        route_mode=route_mode,
    )


class SequenceCriticRefiner(nn.Module):
    def __init__(
        self,
        cand_dim: int,
        ctx_dim: int,
        hidden: int,
        horizon: int,
        *,
        nhead: int,
        layers: int,
        dropout: float,
        correction_scale: float,
        use_context: bool,
        mode_k: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.correction_scale = float(correction_scale)
        self.use_context = bool(use_context)
        self.cand = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.ctx = nn.Sequential(
            nn.Linear(max(ctx_dim, 1), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.corr = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * 2),
        )
        self.route = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, mode_k),
        )
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, cand_x: torch.Tensor, ctx_x: torch.Tensor, cand_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = self.cand(cand_x)
        if self.use_context and ctx_x.shape[1] > 0:
            cx = self.ctx(ctx_x)
        else:
            cx = self.ctx(torch.zeros((cand_x.shape[0], 1), device=cand_x.device, dtype=cand_x.dtype))
        enc = self.encoder(ce + cx[:, None, :])
        scores = self.score(enc).squeeze(-1)
        temp = torch.clamp(F.softplus(self.temp), min=0.20, max=5.0)
        weights = torch.softmax(scores / temp, dim=1)
        pooled = torch.sum(weights[:, :, None] * enc, dim=1)
        mean_res = torch.sum(weights[:, :, None, None] * cand_residual, dim=1)
        pc = torch.cat([pooled, cx], dim=1)
        corr = torch.tanh(self.corr(pc)).view(-1, self.horizon, 2)
        route_logits = self.route(pc)
        pred = mean_res + self.correction_scale * corr
        return pred, scores, weights, route_logits


class RouteQueryRefiner(nn.Module):
    """MTR/MultiPath-like route-query adapter over generated candidates.

    Each learned query corresponds to a hidden motion regime.  A query attends
    over the candidate set, builds its own candidate mixture and residual
    correction, and the final prediction is a mixture over route queries.
    """

    def __init__(
        self,
        cand_dim: int,
        ctx_dim: int,
        hidden: int,
        horizon: int,
        mode_k: int,
        *,
        nhead: int,
        layers: int,
        dropout: float,
        correction_scale: float,
        use_context: bool,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.mode_k = int(mode_k)
        self.correction_scale = float(correction_scale)
        self.use_context = bool(use_context)
        self.cand = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.ctx = nn.Sequential(
            nn.Linear(max(ctx_dim, 1), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.query = nn.Parameter(torch.randn(mode_k, hidden) * 0.02)
        self.query_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.query_score = nn.Linear(hidden, hidden, bias=False)
        self.route_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.query_corr = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(
        self,
        cand_x: torch.Tensor,
        ctx_x: torch.Tensor,
        cand_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = self.cand(cand_x)
        if self.use_context and ctx_x.shape[1] > 0:
            cx = self.ctx(ctx_x)
        else:
            cx = self.ctx(torch.zeros((cand_x.shape[0], 1), device=cand_x.device, dtype=cand_x.dtype))
        tokens = self.encoder(ce + cx[:, None, :])
        q = self.query[None, :, :] + cx[:, None, :]
        q = self.query_proj(q)
        q_score = self.query_score(q)
        cand_scores = torch.einsum("nqh,nkh->nqk", q_score, tokens) / np.sqrt(float(tokens.shape[-1]))
        cand_weights = torch.softmax(cand_scores, dim=-1)
        query_feat = torch.einsum("nqk,nkh->nqh", cand_weights, tokens)
        query_res = torch.einsum("nqk,nkhd->nqhd", cand_weights, cand_residual)
        qc = torch.cat([query_feat, cx[:, None, :].expand(-1, self.mode_k, -1)], dim=-1)
        corr = torch.tanh(self.query_corr(qc)).view(-1, self.mode_k, self.horizon, 2)
        query_pred = query_res + self.correction_scale * corr
        route_logits = self.route_head(qc).squeeze(-1)
        route_probs = torch.softmax(route_logits, dim=-1)
        pred = torch.sum(route_probs[:, :, None, None] * query_pred, dim=1)
        # Candidate-level aggregate score is useful for old diagnostics/top1.
        scores = torch.sum(route_probs[:, :, None] * cand_scores, dim=1)
        return pred, scores, route_probs, route_logits, cand_scores, query_pred


class LearnedRouteGenerator(nn.Module):
    """Causal query-conditioned residual trajectory generator.

    Unlike the heuristic `route_conditioned` sampler, this module does not move
    latent samples toward KMeans centers.  It learns route queries that decode
    full h1..h6 residual trajectories directly from causal context.
    """

    def __init__(
        self,
        ctx_dim: int,
        hidden: int,
        horizon: int,
        mode_k: int,
        *,
        dropout: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.mode_k = int(mode_k)
        self.ctx = nn.Sequential(
            nn.Linear(max(ctx_dim, 1), hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.query = nn.Parameter(torch.randn(mode_k, hidden) * 0.02)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.route_logits = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.route_residual = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, horizon * 2),
        )
        self.route_logscale = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(self, ctx_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if ctx_x.shape[1] == 0:
            ctx_x = torch.zeros((ctx_x.shape[0], 1), device=ctx_x.device, dtype=ctx_x.dtype)
        cx = self.ctx(ctx_x)
        q = self.query[None, :, :].expand(ctx_x.shape[0], -1, -1)
        h = self.query_proj(torch.cat([q, cx[:, None, :].expand(-1, self.mode_k, -1)], dim=-1))
        logits = self.route_logits(h).squeeze(-1)
        residual = self.route_residual(h).view(-1, self.mode_k, self.horizon, 2)
        logscale = torch.clamp(self.route_logscale(h).view(-1, self.mode_k, self.horizon, 2), -4.0, 3.0)
        route_probs = torch.softmax(logits, dim=-1)
        return residual, logscale, logits, route_probs


def route_diversity_loss(route_residual: torch.Tensor, temperature: float) -> torch.Tensor:
    endpoint = torch.sum(route_residual, dim=2)
    diff = endpoint[:, :, None, :] - endpoint[:, None, :, :]
    dist = torch.sum(diff.pow(2), dim=-1)
    eye = torch.eye(dist.shape[1], device=dist.device, dtype=torch.bool)[None, :, :]
    dist = dist.masked_fill(eye, 1e6)
    return torch.mean(torch.exp(-dist / max(float(temperature), 1e-6)))


def route_softmin_loss(route_residual: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    err = torch.mean((route_residual.reshape(route_residual.shape[0], route_residual.shape[1], -1) - target.reshape(target.shape[0], 1, -1)).pow(2), dim=-1)
    return torch.mean(-float(temperature) * torch.logsumexp(-err / max(float(temperature), 1e-6), dim=1))


def soft_oracle_labels(dist: torch.Tensor, temperature: float) -> torch.Tensor:
    d = dist - torch.min(dist, dim=1, keepdim=True).values
    return torch.softmax(-d / max(float(temperature), 1e-6), dim=1)


def normalize_prob_t(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=1e-8)
    return x / torch.clamp(torch.sum(x, dim=1, keepdim=True), min=1e-8)


def pairwise_rank_loss(scores: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    diff_t = target_prob[:, :, None] - target_prob[:, None, :]
    sign = torch.sign(diff_t)
    weight = torch.abs(diff_t)
    diff_s = scores[:, :, None] - scores[:, None, :]
    mask = weight > 1e-5
    if not torch.any(mask):
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    loss = weight * F.softplus(-sign * diff_s)
    return torch.sum(loss[mask]) / torch.clamp(torch.sum(weight[mask]), min=1e-8)


def critic_forward(
    model: nn.Module,
    cand_x: torch.Tensor,
    ctx_x: torch.Tensor,
    cand_residual: torch.Tensor,
    *,
    arch: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if arch == "route_query":
        pred, scores, route_probs, route_logits, cand_scores, query_pred = model(cand_x, ctx_x, cand_residual)
        return pred, scores, route_probs, route_logits, cand_scores, query_pred
    pred, scores, weights, route_logits = model(cand_x, ctx_x, cand_residual)
    return pred, scores, weights, route_logits, None, None


def endpoint_loss(pred: torch.Tensor, target: torch.Tensor, horizons: list[int]) -> torch.Tensor:
    losses = []
    for h in horizons:
        losses.append(F.smooth_l1_loss(endpoint_residual_t(pred, h), endpoint_residual_t(target, h)))
    return torch.stack(losses).mean()


def evaluate_route_generator_raw(
    model: LearnedRouteGenerator,
    ctx: np.ndarray,
    residual_true: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    mixtures, oracles, entropies = [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.batch_size, 23001, shuffle=False):
            rr, _, logits, probs = model(to_tensor(ctx[idx], device))
            mix = torch.sum(probs[:, :, None, None] * rr, dim=1)
            yt = to_tensor(residual_true[idx], device)
            dist = torch.mean((rr.reshape(rr.shape[0], rr.shape[1], -1) - yt.reshape(yt.shape[0], 1, -1)).pow(2), dim=-1)
            best = torch.argmin(dist, dim=1)
            oracle = rr[torch.arange(len(best), device=device), best]
            mixtures.append(mix.cpu().numpy())
            oracles.append(oracle.cpu().numpy())
            entropies.append((-torch.sum(probs * torch.log(probs + 1e-8), dim=1)).cpu().numpy())
    mix_np = np.concatenate(mixtures, axis=0).astype(np.float32)
    oracle_np = np.concatenate(oracles, axis=0).astype(np.float32)
    return {
        "val_route_mixture_rmse": audit.rmse(audit.flatten_residual(residual_true), audit.flatten_residual(mix_np)),
        "val_route_oracle_rmse": audit.rmse(audit.flatten_residual(residual_true), audit.flatten_residual(oracle_np)),
        "val_route_entropy": float(np.mean(np.concatenate(entropies, axis=0))),
    }


def train_learned_route_generator(
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    mode_train: np.ndarray,
    mode_val: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[LearnedRouteGenerator, pd.DataFrame]:
    model = LearnedRouteGenerator(
        ctx_dim=ctx_train.shape[1],
        hidden=args.learned_route_hidden,
        horizon=args.max_horizon,
        mode_k=args.mode_k,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(args.learned_route_epochs):
        model.train()
        losses = []
        for idx in closure.batches(len(ctx_train), args.batch_size, args.seed + 24001 + epoch):
            cx = to_tensor(ctx_train[idx], device)
            yt = to_tensor(residual_train[idx], device)
            q_mode = to_tensor(mode_train[idx], device)
            rr, logscale, logits, probs = model(cx)
            mix = torch.sum(probs[:, :, None, None] * rr, dim=1)
            flat_err = torch.mean((rr.reshape(rr.shape[0], rr.shape[1], -1) - yt.reshape(yt.shape[0], 1, -1)).pow(2), dim=-1)
            query_weighted = torch.mean(torch.sum(q_mode * flat_err, dim=1))
            route_kl = torch.mean(
                torch.sum(q_mode * (torch.log(torch.clamp(q_mode, min=1e-8)) - F.log_softmax(logits, dim=-1)), dim=1)
            )
            recon = endpoint_loss(mix, yt, args.horizons) + F.smooth_l1_loss(mix.reshape(mix.shape[0], -1), yt.reshape(yt.shape[0], -1))
            var = torch.exp(2.0 * logscale)
            nll = 0.5 * torch.mean(torch.sum(q_mode * torch.mean((rr - yt[:, None, :, :]).pow(2) / torch.clamp(var, min=1e-5) + 2.0 * logscale, dim=(2, 3)), dim=1))
            softmin = route_softmin_loss(rr, yt, args.learned_route_softmin_temp)
            diversity = route_diversity_loss(rr, args.learned_route_diversity_temp)
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            loss = (
                args.learned_route_recon_weight * recon
                + args.learned_route_query_weight * query_weighted
                + args.learned_route_mode_weight * route_kl
                + args.learned_route_nll_weight * nll
                + args.learned_route_softmin_weight * softmin
                + args.learned_route_diversity_weight * diversity
                - args.learned_route_entropy_weight * entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == args.learned_route_epochs - 1 or epoch % max(1, args.learned_route_epochs // 5) == 0:
            val = evaluate_route_generator_raw(model, ctx_val, residual_val, args, device=device)
            if val["val_route_mixture_rmse"] < best_val:
                best_val = float(val["val_route_mixture_rmse"])
                best_state = copy.deepcopy(model.state_dict())
            rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_learned_route_generator(
    model: LearnedRouteGenerator,
    ctx: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    res, logscale, logits, probs = [], [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.batch_size, 25001, shuffle=False):
            rr, ls, lo, pr = model(to_tensor(ctx[idx], device))
            res.append(rr.cpu().numpy())
            logscale.append(ls.cpu().numpy())
            logits.append(lo.cpu().numpy())
            probs.append(pr.cpu().numpy())
    return {
        "residual": np.concatenate(res, axis=0).astype(np.float32),
        "logscale": np.concatenate(logscale, axis=0).astype(np.float32),
        "logits": np.concatenate(logits, axis=0).astype(np.float32),
        "probs": np.concatenate(probs, axis=0).astype(np.float32),
    }


def generate_learned_route_candidates(
    arrays: audit.SplitArrays,
    posterior: closure.PosteriorPack,
    student: closure.ComponentStudentPrior,
    blocks: list[str],
    route_model: LearnedRouteGenerator,
    route_ctx: np.ndarray,
    args: argparse.Namespace,
    *,
    split_name: str,
    device: torch.device,
) -> CandidatePack:
    xdict = getattr(arrays, f"x_{split_name}")
    base = getattr(arrays, f"base_{split_name}")
    residual_true = getattr(arrays, f"residual_{split_name}")
    student_pred = closure.predict_student(student, xdict, blocks, device=device, batch_size=args.batch_size)
    route_pred = predict_learned_route_generator(route_model, route_ctx, args, device=device)
    route_res = route_pred["residual"]
    route_logscale = route_pred["logscale"]
    route_probs = route_pred["probs"]
    n = len(base)
    route_count = min(args.mode_k, args.candidate_k)
    order = np.argsort(-route_probs, axis=1)
    rank_idx = (np.arange(args.candidate_k, dtype=np.int64) % route_count)[None, :].repeat(n, axis=0)
    route_mode = np.take_along_axis(order[:, :route_count], rank_idx, axis=1).astype(np.int64)
    rng = np.random.default_rng(args.seed + {"train": 6301, "val": 6401, "test": 6501}[split_name])
    eps = rng.normal(size=(n, args.candidate_k, args.max_horizon, 2)).astype(np.float32)
    if args.learned_route_deterministic_first:
        eps[:, :route_count, :, :] = 0.0
    rows = np.arange(n)[:, None].repeat(args.candidate_k, axis=1)
    cols = np.arange(args.candidate_k)[None, :].repeat(n, axis=0)
    residual = route_res[rows, route_mode] + eps * np.exp(route_logscale[rows, route_mode]) * float(args.learned_route_sample_scale)
    residual = residual.astype(np.float32)
    z_eps = eps.reshape(n, args.candidate_k, -1)
    if z_eps.shape[-1] >= args.latent_dim:
        z_eps_lat = z_eps[:, :, : args.latent_dim].astype(np.float32)
    else:
        z_eps_lat = np.pad(z_eps, ((0, 0), (0, 0), (0, args.latent_dim - z_eps.shape[-1]))).astype(np.float32)
    z = posterior.centers[route_mode].astype(np.float32)
    logprob_route = np.log(np.maximum(np.take_along_axis(route_probs, route_mode, axis=1), EPS))[:, :, None]
    logprob_noise = -0.5 * np.mean(z_eps * z_eps + np.log(2.0 * np.pi), axis=-1, keepdims=True)
    logprob = (logprob_route + logprob_noise).astype(np.float32)
    true_flat = audit.flatten_residual(residual_true)
    cand_flat = residual.reshape(n, args.candidate_k, -1)
    oracle_dist = np.mean((cand_flat - true_flat[:, None, :]) ** 2, axis=-1).astype(np.float32)
    features, _ = build_candidate_features(residual=residual, base=base, z_eps=z_eps_lat, logprob=logprob, horizons=args.horizons)
    mode_features, _ = candidate_mode_features(z, student_pred, posterior, mode_temperature=args.candidate_mode_temperature)
    mode_prob = np.zeros((n, args.candidate_k, args.mode_k), dtype=np.float32)
    mode_prob[rows, cols, route_mode] = 1.0
    mode_prior = np.take_along_axis(route_probs, route_mode, axis=1)[:, :, None].astype(np.float32)
    route_rank = (rank_idx.astype(np.float32) / max(float(route_count - 1), 1.0))[:, :, None]
    route_features = route_feature_packet(n=n, k=args.candidate_k, mode_k=args.mode_k, route_mode=route_mode, mode_prior=mode_prior, route_rank=route_rank)
    features = np.concatenate([features, mode_features, route_features], axis=-1).astype(np.float32)
    return CandidatePack(
        residual=residual,
        z=z,
        z_eps=z_eps_lat,
        logprob=logprob,
        mode_prob=mode_prob,
        features=features,
        oracle_dist=oracle_dist,
        route_mode=route_mode,
    )


def resolve_hybrid_budgets(args: argparse.Namespace) -> dict[str, int]:
    """Resolve fixed candidate budget across generator families."""
    total = int(args.candidate_k)
    budgets = {
        "generic": int(args.hybrid_generic_k),
        "route": int(args.hybrid_route_k),
        "learned": int(args.hybrid_learned_k),
    }
    if sum(budgets.values()) <= 0:
        generic = max(1, total // 2)
        route = max(1, total // 4)
        learned = max(0, total - generic - route)
        budgets = {"generic": generic, "route": route, "learned": learned}
    diff = total - sum(budgets.values())
    if diff > 0:
        budgets["generic"] += diff
    elif diff < 0:
        # Trim least stable branches first, preserving at least one generic
        # candidate when possible.
        need = -diff
        for key in ["learned", "route", "generic"]:
            keep_min = 1 if key == "generic" and total > 0 else 0
            take = min(max(0, budgets[key] - keep_min), need)
            budgets[key] -= take
            need -= take
            if need == 0:
                break
    return {k: max(0, int(v)) for k, v in budgets.items()}


def args_for_candidate_branch(args: argparse.Namespace, *, candidate_k: int, generator: str) -> argparse.Namespace:
    out = copy.copy(args)
    out.candidate_k = int(candidate_k)
    out.candidate_generator = generator
    return out


def concat_candidate_packs(packs: list[CandidatePack]) -> CandidatePack:
    packs = [p for p in packs if p.residual.shape[1] > 0]
    if not packs:
        raise ValueError("No candidate packs to concatenate")
    route_modes = []
    for p in packs:
        if p.route_mode is None:
            route_modes.append(np.full((p.residual.shape[0], p.residual.shape[1]), -1, dtype=np.int64))
        else:
            route_modes.append(p.route_mode.astype(np.int64))
    return CandidatePack(
        residual=np.concatenate([p.residual for p in packs], axis=1).astype(np.float32),
        z=np.concatenate([p.z for p in packs], axis=1).astype(np.float32),
        z_eps=np.concatenate([p.z_eps for p in packs], axis=1).astype(np.float32),
        logprob=np.concatenate([p.logprob for p in packs], axis=1).astype(np.float32),
        mode_prob=np.concatenate([p.mode_prob for p in packs], axis=1).astype(np.float32),
        features=np.concatenate([p.features for p in packs], axis=1).astype(np.float32),
        oracle_dist=np.concatenate([p.oracle_dist for p in packs], axis=1).astype(np.float32),
        route_mode=np.concatenate(route_modes, axis=1).astype(np.int64),
    )


def append_candidate_source(pack: CandidatePack, source_idx: int, source_count: int = 3) -> CandidatePack:
    src = np.zeros((pack.features.shape[0], pack.features.shape[1], source_count), dtype=np.float32)
    if 0 <= int(source_idx) < source_count:
        src[:, :, int(source_idx)] = 1.0
    pack.features = np.concatenate([pack.features, src], axis=-1).astype(np.float32)
    return pack


def generate_hybrid_candidates(
    arrays: audit.SplitArrays,
    posterior: closure.PosteriorPack,
    student: closure.ComponentStudentPrior,
    blocks: list[str],
    route_model: LearnedRouteGenerator | None,
    route_ctx: np.ndarray | None,
    args: argparse.Namespace,
    *,
    split_name: str,
    device: torch.device,
) -> CandidatePack:
    budgets = resolve_hybrid_budgets(args)
    packs: list[CandidatePack] = []
    if budgets["generic"] > 0:
        packs.append(
            append_candidate_source(
                generate_candidates(
                    arrays,
                    posterior,
                    student,
                    blocks,
                    args_for_candidate_branch(args, candidate_k=budgets["generic"], generator="generic"),
                    split_name=split_name,
                    device=device,
                ),
                0,
            )
        )
    if budgets["route"] > 0:
        packs.append(
            append_candidate_source(
                generate_candidates(
                    arrays,
                    posterior,
                    student,
                    blocks,
                    args_for_candidate_branch(args, candidate_k=budgets["route"], generator="route_conditioned"),
                    split_name=split_name,
                    device=device,
                ),
                1,
            )
        )
    if budgets["learned"] > 0:
        if route_model is None or route_ctx is None:
            raise ValueError("Hybrid learned budget requires trained route_model and route_ctx")
        packs.append(
            append_candidate_source(
                generate_learned_route_candidates(
                    arrays,
                    posterior,
                    student,
                    blocks,
                    route_model,
                    route_ctx,
                    args_for_candidate_branch(args, candidate_k=budgets["learned"], generator="learned_route"),
                    split_name=split_name,
                    device=device,
                ),
                2,
            )
        )
    return concat_candidate_packs(packs)


def synthetic_target_residual(n: int, horizon: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate short active-motion residual trajectories for critic pretraining."""
    base = rng.normal(0.0, 4.0, size=(n, 2)).astype(np.float32)
    residual = np.zeros((n, horizon, 2), dtype=np.float32)
    modes = rng.integers(0, 6, size=n)
    for i in range(n):
        direction = rng.normal(size=2).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), EPS)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
        amp = rng.uniform(0.2, 2.0)
        curve = rng.uniform(-1.0, 1.0)
        noise = rng.uniform(0.05, 0.35)
        for t in range(horizon):
            u = (t + 1) / float(horizon)
            if modes[i] == 0:  # straight drift
                step = amp * direction
            elif modes[i] == 1:  # acceleration
                step = amp * (0.4 + 1.2 * u) * direction
            elif modes[i] == 2:  # deceleration / stopping
                step = amp * (1.4 - 1.0 * u) * direction
            elif modes[i] == 3:  # turning route
                step = amp * (direction + curve * u * tangent)
            elif modes[i] == 4:  # flow-aligned with lateral fluctuation
                step = amp * direction + 0.35 * amp * np.sin(np.pi * u) * tangent
            else:  # noisy local mode
                step = amp * direction + rng.normal(0.0, amp * 0.35, size=2)
            residual[i, t] = step + rng.normal(0.0, noise, size=2)
    return base, residual


def synthetic_candidate_pack(
    *,
    n: int,
    k: int,
    horizon: int,
    latent_dim: int,
    cand_dim: int,
    horizons: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, CandidatePack]:
    base, target = synthetic_target_residual(n, horizon, rng)
    residual = np.zeros((n, k, horizon, 2), dtype=np.float32)
    for i in range(n):
        residual[i, 0] = target[i] + rng.normal(0.0, 0.15, size=(horizon, 2))
        for j in range(1, k):
            kind = j % 6
            if kind == 0:
                residual[i, j] = target[i] + rng.normal(0.0, 0.75 + 0.08 * j, size=(horizon, 2))
            elif kind == 1:
                residual[i, j] = target[i, ::-1] + rng.normal(0.0, 0.25, size=(horizon, 2))
            elif kind == 2:
                residual[i, j] = -0.5 * target[i] + rng.normal(0.0, 0.35, size=(horizon, 2))
            elif kind == 3:
                drift = np.mean(target[i], axis=0, keepdims=True)
                residual[i, j] = np.repeat(drift, horizon, axis=0) + rng.normal(0.0, 0.50, size=(horizon, 2))
            elif kind == 4:
                residual[i, j] = np.cumsum(rng.normal(0.0, 0.85, size=(horizon, 2)), axis=0)
            else:
                residual[i, j] = target[i] * rng.uniform(0.3, 1.7) + rng.normal(0.0, 0.45, size=(horizon, 2))
    z_eps = rng.normal(size=(n, k, latent_dim)).astype(np.float32)
    logprob = -0.5 * np.sum(z_eps * z_eps + np.log(2.0 * np.pi), axis=-1, keepdims=True).astype(np.float32)
    feat, _ = build_candidate_features(residual=residual, base=base, z_eps=z_eps, logprob=logprob, horizons=horizons)
    if feat.shape[-1] < cand_dim:
        feat = np.concatenate([feat, np.zeros((n, k, cand_dim - feat.shape[-1]), dtype=np.float32)], axis=-1)
    elif feat.shape[-1] > cand_dim:
        feat = feat[:, :, :cand_dim]
    flat = feat.reshape(-1, feat.shape[-1])
    feat = ((flat - flat.mean(axis=0, keepdims=True)) / np.maximum(flat.std(axis=0, keepdims=True), 1e-6)).reshape(feat.shape).astype(np.float32)
    dist = np.mean((residual.reshape(n, k, -1) - target.reshape(n, 1, -1)) ** 2, axis=-1).astype(np.float32)
    mode_prob = np.full((n, k, 1), 1.0, dtype=np.float32)
    pack = CandidatePack(
        residual=residual,
        z=z_eps,
        z_eps=z_eps,
        logprob=logprob,
        mode_prob=mode_prob,
        features=feat,
        oracle_dist=dist,
    )
    return target, pack


def synthetic_pretrain_critic(
    model: SequenceCriticRefiner,
    args: argparse.Namespace,
    *,
    cand_dim: int,
    ctx_dim: int,
    device: torch.device,
) -> None:
    if args.synthetic_pretrain_epochs <= 0 or args.synthetic_pretrain_n <= 0:
        return
    rng = np.random.default_rng(args.seed + 12345)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctx = rng.normal(0.0, 1.0, size=(args.synthetic_pretrain_n, max(ctx_dim, 1))).astype(np.float32)
    target, pack = synthetic_candidate_pack(
        n=args.synthetic_pretrain_n,
        k=args.candidate_k,
        horizon=args.max_horizon,
        latent_dim=args.latent_dim,
        cand_dim=cand_dim,
        horizons=args.horizons,
        rng=rng,
    )
    for epoch in range(args.synthetic_pretrain_epochs):
        model.train()
        for idx in closure.batches(args.synthetic_pretrain_n, args.critic_batch_size, args.seed + 15000 + epoch):
            cx = to_tensor(ctx[idx], device)
            cf = to_tensor(pack.features[idx], device)
            cr = to_tensor(pack.residual[idx], device)
            yt = to_tensor(target[idx], device)
            dist = to_tensor(pack.oracle_dist[idx], device)
            q = soft_oracle_labels(dist, args.oracle_temperature)
            pred, scores, weights, _, _, _ = critic_forward(model, cf, cx, cr, arch=args.critic_arch)
            listwise = -torch.mean(torch.sum(q * F.log_softmax(scores, dim=1), dim=1))
            reg = endpoint_loss(pred, yt, args.horizons) + F.smooth_l1_loss(pred.reshape(pred.shape[0], -1), yt.reshape(yt.shape[0], -1))
            entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))
            loss = reg + args.listwise_weight * listwise - args.entropy_weight * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def train_critic(
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    cand_train: CandidatePack,
    cand_val: CandidatePack,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    posterior_mu_train: np.ndarray,
    posterior_mode_train: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
    use_context: bool,
) -> tuple[SequenceCriticRefiner, pd.DataFrame]:
    if args.critic_arch == "route_query":
        model = RouteQueryRefiner(
            cand_dim=cand_train.features.shape[-1],
            ctx_dim=ctx_train.shape[1],
            hidden=args.critic_hidden,
            horizon=args.max_horizon,
            mode_k=args.mode_k,
            nhead=args.critic_heads,
            layers=args.critic_layers,
            dropout=args.dropout,
            correction_scale=args.correction_scale,
            use_context=use_context,
        ).to(device)
    else:
        model = SequenceCriticRefiner(
            cand_dim=cand_train.features.shape[-1],
            ctx_dim=ctx_train.shape[1],
            hidden=args.critic_hidden,
            horizon=args.max_horizon,
            nhead=args.critic_heads,
            layers=args.critic_layers,
            dropout=args.dropout,
            correction_scale=args.correction_scale,
            use_context=use_context,
            mode_k=args.mode_k,
        ).to(device)
    if args.synthetic_critic_pretrain:
        synthetic_pretrain_critic(
            model,
            args,
            cand_dim=cand_train.features.shape[-1],
            ctx_dim=ctx_train.shape[1],
            device=device,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows: list[dict[str, Any]] = []
    n = len(ctx_train)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(args.critic_epochs):
        model.train()
        train_losses = []
        for idx in closure.batches(n, args.critic_batch_size, args.seed + 6100 + epoch):
            cx = to_tensor(ctx_train[idx], device)
            cf = to_tensor(cand_train.features[idx], device)
            cr = to_tensor(cand_train.residual[idx], device)
            yt = to_tensor(residual_train[idx], device)
            dist = to_tensor(cand_train.oracle_dist[idx], device)
            z = to_tensor(cand_train.z[idx], device)
            mu_q = to_tensor(posterior_mu_train[idx], device)
            mode_q = to_tensor(posterior_mode_train[idx], device)
            q = soft_oracle_labels(dist, args.oracle_temperature)
            pred, scores, weights, route_logits, cand_scores, query_pred = critic_forward(model, cf, cx, cr, arch=args.critic_arch)
            logp = F.log_softmax(scores, dim=1)
            latent_dist = torch.mean((z - mu_q[:, None, :]).pow(2), dim=-1)
            q_latent = soft_oracle_labels(latent_dist, args.teacher_latent_temperature)
            teacher_latent = -torch.mean(torch.sum(q_latent * logp, dim=1))
            if cand_train.mode_prob.shape[-1] == posterior_mode_train.shape[-1]:
                mode_prob = to_tensor(cand_train.mode_prob[idx], device)
                mode_compat = torch.sum(mode_prob * mode_q[:, None, :], dim=-1)
                q_mode_candidate = normalize_prob_t(mode_compat)
                if args.critic_arch == "route_query":
                    pred_mode = weights
                else:
                    pred_mode = torch.sum(weights[:, :, None] * mode_prob, dim=1)
                teacher_mode = torch.mean(
                    torch.sum(mode_q * (torch.log(torch.clamp(mode_q, min=1e-8)) - torch.log(torch.clamp(pred_mode, min=1e-8))), dim=1)
                )
            else:
                q_mode_candidate = torch.full_like(q, 1.0 / q.shape[1])
                teacher_mode = torch.zeros((), device=device)
            if args.teacher_central:
                q_target = normalize_prob_t(
                    args.oracle_label_weight * q
                    + args.teacher_latent_label_weight * q_latent
                    + args.teacher_mode_label_weight * q_mode_candidate
                )
            else:
                q_target = q
            listwise = -torch.mean(torch.sum(q_target * logp, dim=1))
            reg = endpoint_loss(pred, yt, args.horizons) + F.smooth_l1_loss(pred.reshape(pred.shape[0], -1), yt.reshape(yt.shape[0], -1))
            route_loss = torch.mean(
                torch.sum(mode_q * (torch.log(torch.clamp(mode_q, min=1e-8)) - F.log_softmax(route_logits, dim=-1)), dim=1)
            )
            rank_loss = pairwise_rank_loss(scores, q_target)
            query_loss = torch.zeros((), device=device)
            query_reg = torch.zeros((), device=device)
            query_route_oracle_loss = torch.zeros((), device=device)
            if args.critic_arch == "route_query" and cand_scores is not None and query_pred is not None and cand_train.mode_prob.shape[-1] == posterior_mode_train.shape[-1]:
                query_logp = F.log_softmax(cand_scores, dim=-1)
                mode_prob = to_tensor(cand_train.mode_prob[idx], device)
                # Query m should attend to candidates compatible with posterior
                # mode m and still close to the true trajectory.
                q_query = args.query_oracle_weight * q[:, None, :] * mode_prob.transpose(1, 2) + args.query_mode_weight * mode_prob.transpose(1, 2)
                q_query = torch.clamp(q_query, min=1e-8)
                q_query = q_query / torch.clamp(torch.sum(q_query, dim=-1, keepdim=True), min=1e-8)
                query_loss = -torch.mean(torch.sum(q_query * query_logp, dim=-1))
                query_flat_err = torch.mean(
                    (
                        query_pred.reshape(query_pred.shape[0], query_pred.shape[1], -1)
                        - yt.reshape(yt.shape[0], 1, -1)
                    ).pow(2),
                    dim=-1,
                )
                q_route_oracle = soft_oracle_labels(query_flat_err, args.query_route_oracle_temperature)
                query_route_oracle_loss = -torch.mean(torch.sum(q_route_oracle * F.log_softmax(route_logits, dim=-1), dim=1))
                endpoint_err = []
                for h in args.horizons:
                    qe = torch.sum(query_pred[:, :, : int(h), :], dim=2)
                    yt_e = endpoint_residual_t(yt, h)
                    endpoint_err.append(torch.mean(torch.sum((qe - yt_e[:, None, :]).pow(2), dim=-1), dim=0))
                endpoint_err_t = torch.stack(endpoint_err).mean(dim=0)
                query_reg = torch.sum(torch.mean(mode_q, dim=0) * endpoint_err_t)
            entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))
            loss = (
                reg
                + args.listwise_weight * listwise
                + args.teacher_latent_weight * teacher_latent
                + args.teacher_mode_weight * teacher_mode
                + args.route_loss_weight * route_loss
                + args.pairwise_rank_weight * rank_loss
                + args.query_candidate_weight * query_loss
                + args.query_reg_weight * query_reg
                + args.query_route_oracle_weight * query_route_oracle_loss
                - args.entropy_weight * entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        if epoch == args.critic_epochs - 1 or epoch % max(1, args.critic_epochs // 5) == 0:
            val = evaluate_critic_raw(model, ctx_val, cand_val, residual_val, args, device=device)
            if val["val_residual_rmse"] < best_val:
                best_val = float(val["val_residual_rmse"])
                best_state = copy.deepcopy(model.state_dict())
            rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val})
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def evaluate_critic_raw(
    model: SequenceCriticRefiner,
    ctx: np.ndarray,
    cand: CandidatePack,
    residual_true: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds, top_preds, weight_entropy = [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, 8101, shuffle=False):
            pred, scores, weights, _, _, _ = critic_forward(
                model,
                to_tensor(cand.features[idx], device),
                to_tensor(ctx[idx], device),
                to_tensor(cand.residual[idx], device),
                arch=args.critic_arch,
            )
            top = torch.argmax(scores, dim=1)
            cr = to_tensor(cand.residual[idx], device)
            top_res = cr[torch.arange(len(top), device=device), top]
            preds.append(pred.cpu().numpy())
            top_preds.append(top_res.cpu().numpy())
            ent = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
            weight_entropy.append(ent.cpu().numpy())
    pred_np = np.concatenate(preds, axis=0).astype(np.float32)
    top_np = np.concatenate(top_preds, axis=0).astype(np.float32)
    ent_np = np.concatenate(weight_entropy, axis=0)
    return {
        "val_residual_rmse": audit.rmse(audit.flatten_residual(residual_true), audit.flatten_residual(pred_np)),
        "val_top_residual_rmse": audit.rmse(audit.flatten_residual(residual_true), audit.flatten_residual(top_np)),
        "weight_entropy": float(np.mean(ent_np)),
    }


def endpoint_rows(
    arrays: audit.SplitArrays,
    residual_pred: np.ndarray,
    label: str,
    args: argparse.Namespace,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=residual_pred,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def oracle_residual(cand: CandidatePack, residual_true: np.ndarray, k: int) -> np.ndarray:
    dist = cand.oracle_dist[:, :k]
    take = np.argmin(dist, axis=1)
    return cand.residual[np.arange(len(take)), take]


def mean_candidate_residual(cand: CandidatePack) -> np.ndarray:
    return np.mean(cand.residual, axis=1).astype(np.float32)


def evaluate_final(
    model: SequenceCriticRefiner,
    ctx_test: np.ndarray,
    cand_test: CandidatePack,
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    *,
    device: torch.device,
    label_prefix: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    model.eval()
    preds, top_preds, weights_all = [], [], []
    top_query_preds, query_oracle_preds = [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx_test), args.critic_batch_size, 9201, shuffle=False):
            pred, scores, weights, route_logits, _, query_pred = critic_forward(
                model,
                to_tensor(cand_test.features[idx], device),
                to_tensor(ctx_test[idx], device),
                to_tensor(cand_test.residual[idx], device),
                arch=args.critic_arch,
            )
            top = torch.argmax(scores, dim=1)
            cr = to_tensor(cand_test.residual[idx], device)
            top_res = cr[torch.arange(len(top), device=device), top]
            preds.append(pred.cpu().numpy())
            top_preds.append(top_res.cpu().numpy())
            weights_all.append(weights.cpu().numpy())
            if args.critic_arch == "route_query" and query_pred is not None:
                top_q = torch.argmax(route_logits, dim=1)
                tq = query_pred[torch.arange(len(top_q), device=device), top_q]
                yt = to_tensor(arrays.residual_test[idx], device)
                qdist = torch.mean((query_pred.reshape(query_pred.shape[0], query_pred.shape[1], -1) - yt.reshape(yt.shape[0], 1, -1)).pow(2), dim=-1)
                oq = torch.argmin(qdist, dim=1)
                qo = query_pred[torch.arange(len(oq), device=device), oq]
                top_query_preds.append(tq.cpu().numpy())
                query_oracle_preds.append(qo.cpu().numpy())
    pred_np = np.concatenate(preds, axis=0).astype(np.float32)
    top_np = np.concatenate(top_preds, axis=0).astype(np.float32)
    weights_np = np.concatenate(weights_all, axis=0).astype(np.float32)
    rows: list[dict[str, Any]] = []
    rows.extend(endpoint_rows(arrays, pred_np, f"{label_prefix}_weighted_refined", args, {"stage": "sequence_critic", "variant": label_prefix}))
    rows.extend(endpoint_rows(arrays, top_np, f"{label_prefix}_top1_scored", args, {"stage": "sequence_critic_top1", "variant": label_prefix}))
    if top_query_preds:
        top_query_np = np.concatenate(top_query_preds, axis=0).astype(np.float32)
        query_oracle_np = np.concatenate(query_oracle_preds, axis=0).astype(np.float32)
        rows.extend(endpoint_rows(arrays, top_query_np, f"{label_prefix}_top_route", args, {"stage": "route_query_top", "variant": label_prefix}))
        rows.extend(endpoint_rows(arrays, query_oracle_np, f"{label_prefix}_query_oracle", args, {"stage": "route_query_oracle", "variant": label_prefix}))
    rows.extend(endpoint_rows(arrays, mean_candidate_residual(cand_test), f"{label_prefix}_candidate_mean", args, {"stage": "candidate_control", "variant": label_prefix}))
    for k in args.oracle_k:
        rows.extend(endpoint_rows(arrays, oracle_residual(cand_test, arrays.residual_test, int(k)), f"{label_prefix}_oracle@{k}", args, {"stage": "candidate_oracle", "variant": label_prefix, "oracle_k": int(k)}))
    return rows, weights_np


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)

    ctx_blocks = closure.variant_blocks(args.critic_context_variant, arrays.x_train)
    ctx_train_raw = flatten_blocks(arrays.x_train, ctx_blocks)
    ctx_val_raw = flatten_blocks(arrays.x_val, ctx_blocks)
    ctx_test_raw = flatten_blocks(arrays.x_test, ctx_blocks)
    if args.add_decomposition_context:
        pred_ctx_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
        pred_ctx_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
        pred_ctx_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
        ctx_train_raw = np.concatenate([ctx_train_raw, decomposition_context_features(pred_ctx_train, mode_k=args.mode_k)], axis=1)
        ctx_val_raw = np.concatenate([ctx_val_raw, decomposition_context_features(pred_ctx_val, mode_k=args.mode_k)], axis=1)
        ctx_test_raw = np.concatenate([ctx_test_raw, decomposition_context_features(pred_ctx_test, mode_k=args.mode_k)], axis=1)
    if ctx_train_raw.shape[1] > args.max_critic_context_features:
        # Deterministic variance filter fitted on train only.
        var = np.var(ctx_train_raw, axis=0)
        keep = np.argsort(var)[-args.max_critic_context_features :]
        ctx_train_raw = ctx_train_raw[:, keep]
        ctx_val_raw = ctx_val_raw[:, keep]
        ctx_test_raw = ctx_test_raw[:, keep]
    ctx_train, ctx_val, ctx_test, ctx_scaler = standardize(ctx_train_raw, ctx_val_raw, ctx_test_raw)

    route_train_log = pd.DataFrame()
    route_ctx_scaler: dict[str, Any] = {}
    hybrid_budgets = resolve_hybrid_budgets(args) if args.candidate_generator == "hybrid" else {"generic": 0, "route": 0, "learned": 0}
    needs_learned_route = args.candidate_generator == "learned_route" or (
        args.candidate_generator == "hybrid" and hybrid_budgets.get("learned", 0) > 0
    )
    route_model: LearnedRouteGenerator | None = None
    route_ctx_train = route_ctx_val = route_ctx_test = None
    if needs_learned_route:
        route_blocks = closure.variant_blocks(args.learned_route_context_variant, arrays.x_train)
        route_ctx_train_raw = flatten_blocks(arrays.x_train, route_blocks)
        route_ctx_val_raw = flatten_blocks(arrays.x_val, route_blocks)
        route_ctx_test_raw = flatten_blocks(arrays.x_test, route_blocks)
        if args.learned_route_add_decomposition_context:
            pred_route_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
            pred_route_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
            pred_route_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
            route_ctx_train_raw = np.concatenate([route_ctx_train_raw, decomposition_context_features(pred_route_train, mode_k=args.mode_k)], axis=1)
            route_ctx_val_raw = np.concatenate([route_ctx_val_raw, decomposition_context_features(pred_route_val, mode_k=args.mode_k)], axis=1)
            route_ctx_test_raw = np.concatenate([route_ctx_test_raw, decomposition_context_features(pred_route_test, mode_k=args.mode_k)], axis=1)
        if route_ctx_train_raw.shape[1] > args.max_learned_route_context_features:
            var = np.var(route_ctx_train_raw, axis=0)
            keep = np.argsort(var)[-args.max_learned_route_context_features :]
            route_ctx_train_raw = route_ctx_train_raw[:, keep]
            route_ctx_val_raw = route_ctx_val_raw[:, keep]
            route_ctx_test_raw = route_ctx_test_raw[:, keep]
        route_ctx_train, route_ctx_val, route_ctx_test, route_ctx_scaler = standardize(route_ctx_train_raw, route_ctx_val_raw, route_ctx_test_raw)
        route_model, route_train_log = train_learned_route_generator(
            route_ctx_train,
            route_ctx_val,
            arrays.residual_train,
            arrays.residual_val,
            posterior.mode_soft_train,
            posterior.mode_soft_val,
            args,
            device=device,
        )
        route_model = route_model
    if args.candidate_generator == "learned_route":
        cand_train = generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_train, args, split_name="train", device=device)
        cand_val = generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_val, args, split_name="val", device=device)
        cand_test = generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_test, args, split_name="test", device=device)
    elif args.candidate_generator == "hybrid":
        cand_train = generate_hybrid_candidates(
            arrays,
            posterior,
            student,
            blocks,
            route_model,
            route_ctx_train,
            args,
            split_name="train",
            device=device,
        )
        cand_val = generate_hybrid_candidates(
            arrays,
            posterior,
            student,
            blocks,
            route_model,
            route_ctx_val,
            args,
            split_name="val",
            device=device,
        )
        cand_test = generate_hybrid_candidates(
            arrays,
            posterior,
            student,
            blocks,
            route_model,
            route_ctx_test,
            args,
            split_name="test",
            device=device,
        )
    else:
        cand_train = generate_candidates(arrays, posterior, student, blocks, args, split_name="train", device=device)
        cand_val = generate_candidates(arrays, posterior, student, blocks, args, split_name="val", device=device)
        cand_test = generate_candidates(arrays, posterior, student, blocks, args, split_name="test", device=device)
    cand_train_z, cand_val_z, cand_test_z, cand_scaler = standardize(
        cand_train.features.reshape(-1, cand_train.features.shape[-1]),
        cand_val.features.reshape(-1, cand_val.features.shape[-1]),
        cand_test.features.reshape(-1, cand_test.features.shape[-1]),
    )
    cand_train.features = cand_train_z.reshape(cand_train.features.shape)
    cand_val.features = cand_val_z.reshape(cand_val.features.shape)
    cand_test.features = cand_test_z.reshape(cand_test.features.shape)

    model, train_log = train_critic(
        ctx_train,
        ctx_val,
        cand_train,
        cand_val,
        arrays.residual_train,
        arrays.residual_val,
        posterior.mu_train,
        posterior.mode_soft_train,
        args,
        device=device,
        use_context=True,
    )
    rows, weights = evaluate_final(model, ctx_test, cand_test, arrays, args, device=device, label_prefix=f"critic_{args.generator_variant}")

    train_parts = [train_log.assign(model="context")]
    diag_rows = [
        {
            "model": "context",
            "mean_weight_entropy": float(np.mean(-np.sum(weights * np.log(weights + 1e-8), axis=1))),
            "candidate_k": args.candidate_k,
            "generator_variant": args.generator_variant,
            "context_variant": args.critic_context_variant,
        }
    ]
    if not args.skip_candidate_only:
        model_nc, train_log_nc = train_critic(
            np.zeros((len(ctx_train), 0), dtype=np.float32),
            np.zeros((len(ctx_val), 0), dtype=np.float32),
            cand_train,
            cand_val,
            arrays.residual_train,
            arrays.residual_val,
            posterior.mu_train,
            posterior.mode_soft_train,
            args,
            device=device,
            use_context=False,
        )
        rows_nc, weights_nc = evaluate_final(
            model_nc,
            np.zeros((len(ctx_test), 0), dtype=np.float32),
            cand_test,
            arrays,
            args,
            device=device,
            label_prefix=f"critic_no_context_{args.generator_variant}",
        )
        rows.extend(rows_nc)
        train_parts.append(train_log_nc.assign(model="candidate_only"))
        diag_rows.append(
            {
                "model": "candidate_only",
                "mean_weight_entropy": float(np.mean(-np.sum(weights_nc * np.log(weights_nc + 1e-8), axis=1))),
                "candidate_k": args.candidate_k,
                "generator_variant": args.generator_variant,
                "context_variant": "none",
            }
        )

    summary = pd.DataFrame(rows)
    train = pd.concat(train_parts, ignore_index=True)
    pd.DataFrame(summary).to_csv(args.out_dir / "sequence_critic_refiner_summary.csv", index=False)
    train.to_csv(args.out_dir / "sequence_critic_refiner_train_log.csv", index=False)
    if not route_train_log.empty:
        route_train_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(args.out_dir / "sequence_critic_refiner_diagnostics.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps({"context": finite_json(ctx_scaler), "route_context": finite_json(route_ctx_scaler), "candidate": finite_json(cand_scaler)}, indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, train)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "train_rows": len(train)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, train: pd.DataFrame) -> None:
    lines = ["# Sequence Critic-Refiner Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- generator_variant: `{args.generator_variant}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- critic_context_variant: `{args.critic_context_variant}`")
    lines.append(f"- add_decomposition_context: `{args.add_decomposition_context}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append("")
    lines.append("## Endpoint Metrics")
    for h in args.horizons:
        lines.append(f"### h{h}")
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        for _, row in sub.head(12).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, gain={row['gain_vs_base_pct']:.2f}%")
    lines.append("\n## Training")
    if not train.empty:
        for model, sub in train.groupby("model"):
            last = sub.sort_values("epoch").iloc[-1]
            lines.append(f"- `{model}`: val_residual_rmse={last['val_residual_rmse']:.3f}, top={last['val_top_residual_rmse']:.3f}, entropy={last['weight_entropy']:.3f}")
    lines.append("\n## Decision Notes")
    lines.append("- This gate passes only if weighted_refined beats candidate_mean and approaches oracle consistently.")
    lines.append("- If candidate_only ~= context, the critic is not using causal observability yet.")
    lines.append("- If oracle is strong but critic remains weak, next step is target-aware route pseudo-labels / teacher for the critic, not more handcrafted candidate scores.")
    (out_dir / "sequence_critic_refiner_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    parser.add_argument("--dataset", type=str, default="MDCK_Bulk")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seq", type=str, default="1,2,3,4")
    parser.add_argument("--val-seq", type=str, default="5")
    parser.add_argument("--test-seq", type=str, default="6")
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--horizons", type=str, default="1,2,4,6")
    parser.add_argument("--max-train-rows", type=int, default=18000)
    parser.add_argument("--max-val-rows", type=int, default=5000)
    parser.add_argument("--max-test-rows", type=int, default=7000)
    parser.add_argument("--max-features-per-family", type=int, default=160)
    parser.add_argument("--max-all-features", type=int, default=384)
    parser.add_argument("--history-windows", type=str, default="4,8,16,32,64")
    parser.add_argument("--history-flat-lags", type=int, default=32)
    parser.add_argument("--add-history", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--mode-k", type=int, default=12)
    parser.add_argument("--mode-temperature", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--posterior-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=16)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--posterior-beta", type=float, default=1e-3)
    parser.add_argument("--mode-loss-weight", type=float, default=0.30)
    parser.add_argument("--recon-loss-weight", type=float, default=0.20)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--oracle-k", type=str, default="8,16,32")
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument("--generator-variant", type=str, default="no_history")
    parser.add_argument("--candidate-generator", type=str, default="generic", choices=["generic", "route_conditioned", "learned_route", "hybrid"])
    parser.add_argument("--hybrid-generic-k", type=int, default=0)
    parser.add_argument("--hybrid-route-k", type=int, default=0)
    parser.add_argument("--hybrid-learned-k", type=int, default=0)
    parser.add_argument("--route-center-mix", type=float, default=0.55)
    parser.add_argument("--route-noise-scale", type=float, default=0.85)
    parser.add_argument("--route-mode-count", type=int, default=0)
    parser.add_argument("--route-include-centers", action="store_true")
    parser.add_argument("--learned-route-context-variant", type=str, default="no_history")
    parser.add_argument("--learned-route-add-decomposition-context", action="store_true")
    parser.add_argument("--learned-route-hidden", type=int, default=160)
    parser.add_argument("--learned-route-epochs", type=int, default=20)
    parser.add_argument("--learned-route-sample-scale", type=float, default=0.65)
    parser.add_argument("--learned-route-deterministic-first", action="store_true")
    parser.add_argument("--learned-route-recon-weight", type=float, default=0.60)
    parser.add_argument("--learned-route-query-weight", type=float, default=0.30)
    parser.add_argument("--learned-route-mode-weight", type=float, default=0.20)
    parser.add_argument("--learned-route-nll-weight", type=float, default=0.05)
    parser.add_argument("--learned-route-softmin-weight", type=float, default=0.15)
    parser.add_argument("--learned-route-diversity-weight", type=float, default=0.03)
    parser.add_argument("--learned-route-entropy-weight", type=float, default=0.01)
    parser.add_argument("--learned-route-softmin-temp", type=float, default=0.25)
    parser.add_argument("--learned-route-diversity-temp", type=float, default=16.0)
    parser.add_argument("--max-learned-route-context-features", type=int, default=384)
    parser.add_argument("--critic-context-variant", type=str, default="no_history")
    parser.add_argument("--add-decomposition-context", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--candidate-mode-temperature", type=float, default=1.0)
    parser.add_argument("--critic-arch", type=str, default="scalar", choices=["scalar", "route_query"])
    parser.add_argument("--critic-hidden", type=int, default=160)
    parser.add_argument("--critic-heads", type=int, default=4)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--critic-epochs", type=int, default=20)
    parser.add_argument("--critic-batch-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--correction-scale", type=float, default=1.5)
    parser.add_argument("--listwise-weight", type=float, default=0.35)
    parser.add_argument("--teacher-central", action="store_true")
    parser.add_argument("--oracle-label-weight", type=float, default=0.35)
    parser.add_argument("--teacher-latent-label-weight", type=float, default=0.45)
    parser.add_argument("--teacher-mode-label-weight", type=float, default=0.20)
    parser.add_argument("--teacher-latent-weight", type=float, default=0.20)
    parser.add_argument("--teacher-mode-weight", type=float, default=0.15)
    parser.add_argument("--route-loss-weight", type=float, default=0.20)
    parser.add_argument("--pairwise-rank-weight", type=float, default=0.10)
    parser.add_argument("--query-candidate-weight", type=float, default=0.20)
    parser.add_argument("--query-reg-weight", type=float, default=0.02)
    parser.add_argument("--query-oracle-weight", type=float, default=0.70)
    parser.add_argument("--query-mode-weight", type=float, default=0.30)
    parser.add_argument("--query-route-oracle-weight", type=float, default=0.0)
    parser.add_argument("--query-route-oracle-temperature", type=float, default=0.25)
    parser.add_argument("--teacher-latent-temperature", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--oracle-temperature", type=float, default=0.18)
    parser.add_argument("--synthetic-critic-pretrain", action="store_true")
    parser.add_argument("--synthetic-pretrain-epochs", type=int, default=3)
    parser.add_argument("--synthetic-pretrain-n", type=int, default=4096)
    parser.add_argument("--max-critic-context-features", type=int, default=320)
    parser.add_argument("--skip-candidate-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 4000)
        args.max_val_rows = min(args.max_val_rows, 1500)
        args.max_test_rows = min(args.max_test_rows, 2000)
        args.posterior_epochs = min(args.posterior_epochs, 8)
        args.student_epochs = min(args.student_epochs, 8)
        args.learned_route_epochs = min(args.learned_route_epochs, 8)
        args.critic_epochs = min(args.critic_epochs, 8)
        args.synthetic_pretrain_epochs = min(args.synthetic_pretrain_epochs, 2)
        args.synthetic_pretrain_n = min(args.synthetic_pretrain_n, 1024)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.max_all_features = min(args.max_all_features, 192)
        args.history_flat_lags = min(args.history_flat_lags, 16)
    run(args)


if __name__ == "__main__":
    main()
