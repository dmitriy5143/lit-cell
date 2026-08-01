#!/usr/bin/env python3
"""Regime/magnitude-aware h1-first sequence decoder for LaChance tracks.

This runner tests the current main failure mode of the raw-context sequence
model: direction is useful, but motion amplitude is over-smoothed.  The new
decoder keeps the existing causal feature packet, but predicts each future step
through two coupled routes:

1. residual vector route: base step + learned residual, as in the previous
   sequence decoder;
2. polar route: learned direction + learned positive speed.

A learned blend decides how much each route contributes per future step.  The
training loss adds explicit speed, direction, regime-bin, and horizon-normalized
endpoint terms.
"""

from __future__ import annotations

import argparse
import json
import math
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

import run_lachance_h1_sequence_raw_context_decoder as seq  # noqa: E402
import run_lachance_image_feature_probe as ifp  # noqa: E402


DEFAULT_FEATURES = seq.DEFAULT_FEATURES
DEFAULT_OUT = ROOT / "outputs" / "lachance_regime_magnitude_sequence_decoder_2026-06-16"
EPS = 1e-8


@dataclass
class RegimeStats:
    residual_norm: seq.Norm
    log_speed_mean: np.ndarray
    log_speed_std: np.ndarray
    speed_bins: np.ndarray
    speed_clip: float
    endpoint_scales: dict[int, float]
    target_std_scalar: float


def finite_json(value: Any) -> Any:
    return seq.finite_json(value)


def apply_train_position_norm(split: seq.SplitData) -> seq.SplitData:
    """Recompute x/y/frame normalization from train only.

    The previous sequence runner computed these three harmless but protocol
    sensitive scalars on the merged full table.  This function makes the new
    runner stricter without changing target construction.
    """

    train = split.train.copy()
    val = split.val.copy()
    test = split.test.copy()
    x_scale = max(float(train["x_px"].quantile(0.99) - train["x_px"].quantile(0.01)), 1.0)
    y_scale = max(float(train["y_px"].quantile(0.99) - train["y_px"].quantile(0.01)), 1.0)
    f_scale = max(float(train["frame"].max() - train["frame"].min()), 1.0)
    x_med = float(train["x_px"].median())
    y_med = float(train["y_px"].median())
    f_min = float(train["frame"].min())
    for part in (train, val, test):
        part["x_norm"] = (part["x_px"] - x_med) / x_scale
        part["y_norm"] = (part["y_px"] - y_med) / y_scale
        part["frame_norm"] = (part["frame"] - f_min) / f_scale
    return seq.SplitData(train=train, val=val, test=test)


def fit_regime_stats(train: seq.ArrayPack, horizons: list[int]) -> RegimeStats:
    residual_norm = seq.fit_norm(train.residual_steps, axis=(0,))
    speeds = np.linalg.norm(train.target_steps, axis=2).astype(np.float32)
    log_speed = np.log1p(speeds)
    mean = log_speed.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(log_speed.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    flat_speed = speeds.reshape(-1)
    speed_bins = np.quantile(flat_speed, [0.25, 0.5, 0.75]).astype(np.float32)
    speed_clip = float(max(np.quantile(flat_speed, 0.995) * 1.75, 5.0))
    endpoint_scales: dict[int, float] = {}
    for h in horizons:
        endpoint = train.target_steps[:, :h, :].sum(axis=1)
        endpoint_scales[int(h)] = float(max(np.std(endpoint), 1.0))
    return RegimeStats(
        residual_norm=residual_norm,
        log_speed_mean=mean,
        log_speed_std=std,
        speed_bins=speed_bins,
        speed_clip=speed_clip,
        endpoint_scales=endpoint_scales,
        target_std_scalar=float(max(np.std(train.target_steps), 1.0)),
    )


def speed_labels_np(steps: np.ndarray, bins: np.ndarray) -> np.ndarray:
    speeds = np.linalg.norm(steps, axis=2)
    return np.digitize(speeds, bins).astype(np.int64)


def log_speed_norm_np(steps: np.ndarray, stats: RegimeStats) -> np.ndarray:
    speed = np.linalg.norm(steps, axis=2).astype(np.float32)
    log_speed = np.log1p(speed)
    z = (log_speed - stats.log_speed_mean) / stats.log_speed_std
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float32)


def branch_norms(train: seq.ArrayPack) -> tuple[seq.Norm, seq.Norm, seq.Norm]:
    return seq.fit_norm(train.traj), seq.fit_norm(train.morph), seq.fit_norm(train.flow)


class Branch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(max(1, input_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            x = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return self.net(x)


class RegimeMagnitudeDecoder(nn.Module):
    def __init__(
        self,
        traj_dim: int,
        morph_dim: int,
        flow_dim: int,
        hidden_dim: int,
        max_horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.hidden_dim = int(hidden_dim)
        self.traj = Branch(traj_dim, hidden_dim, dropout)
        self.morph = Branch(morph_dim, hidden_dim, dropout)
        self.flow = Branch(flow_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.step_embed = nn.Parameter(torch.randn(max_horizon, hidden_dim) * 0.02)
        self.gru = nn.GRUCell(hidden_dim + 2, hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, 2)
        self.dir_head = nn.Linear(hidden_dim, 2)
        self.log_speed_head = nn.Linear(hidden_dim, 1)
        self.blend_head = nn.Linear(hidden_dim, 1)
        self.regime_head = nn.Linear(hidden_dim, 4)

    def encode(self, traj: torch.Tensor, morph: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ht = self.traj(traj)
        hm = self.morph(morph)
        hf = self.flow(flow)
        gates = torch.sigmoid(self.gate(torch.cat([ht, hm, hf], dim=1)))
        hm_g = hm * gates[:, 0:1]
        hf_g = hf * gates[:, 1:2]
        fused = torch.cat([ht, hm_g, hf_g, hm_g - hf_g, hm_g * hf_g], dim=1)
        return self.fuse(fused), gates

    def forward(
        self,
        traj: torch.Tensor,
        morph: torch.Tensor,
        flow: torch.Tensor,
        base_step_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h, gates = self.encode(traj, morph, flow)
        residual_norms = []
        dirs = []
        log_speeds = []
        blends = []
        regimes = []
        for step in range(self.max_horizon):
            token = self.step_embed[step].unsqueeze(0).expand(h.shape[0], -1)
            h = self.gru(torch.cat([token, base_step_norm], dim=1), h)
            residual_norms.append(self.residual_head(h))
            dirs.append(self.dir_head(h))
            log_speeds.append(self.log_speed_head(h).squeeze(1))
            blends.append(self.blend_head(h).squeeze(1))
            regimes.append(self.regime_head(h))
        return {
            "residual_norm": torch.stack(residual_norms, dim=1),
            "dir_raw": torch.stack(dirs, dim=1),
            "log_speed_norm": torch.stack(log_speeds, dim=1),
            "blend_logit": torch.stack(blends, dim=1),
            "regime_logits": torch.stack(regimes, dim=1),
            "gates": gates,
        }


def tensor_norm(mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(mean, dtype=torch.float32, device=device),
        torch.as_tensor(std, dtype=torch.float32, device=device),
    )


def decode_outputs(
    out: dict[str, torch.Tensor],
    base_step_px: torch.Tensor,
    stats: RegimeStats,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    res_mean, res_std = tensor_norm(stats.residual_norm.mean, stats.residual_norm.std, device)
    residual_px = out["residual_norm"] * res_std + res_mean
    residual_route = base_step_px[:, None, :] + residual_px

    log_mean = torch.as_tensor(stats.log_speed_mean, dtype=torch.float32, device=device)
    log_std = torch.as_tensor(stats.log_speed_std, dtype=torch.float32, device=device)
    log_speed = out["log_speed_norm"] * log_std + log_mean
    log_speed = torch.clamp(log_speed, min=0.0, max=float(math.log1p(stats.speed_clip)))
    speed = torch.expm1(log_speed)
    direction = F.normalize(out["dir_raw"], p=2, dim=2, eps=1e-6)
    polar_route = direction * speed[:, :, None]

    blend = torch.sigmoid(out["blend_logit"])[:, :, None]
    final = blend * polar_route + (1.0 - blend) * residual_route
    return {
        "pred_steps": final,
        "residual_route": residual_route,
        "polar_route": polar_route,
        "direction": direction,
        "speed": speed,
        "blend": blend,
    }


def normalize_inputs(
    pack: seq.ArrayPack,
    norms: tuple[seq.Norm, seq.Norm, seq.Norm],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traj_norm, morph_norm, flow_norm = norms
    traj = seq.apply_norm(pack.traj, traj_norm)
    morph = seq.apply_norm(pack.morph, morph_norm)
    flow = seq.apply_norm(pack.flow, flow_norm)
    base_norm = pack.base_step / max(float(np.std(pack.base_step)), 1.0)
    base_norm = np.clip(np.nan_to_num(base_norm, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float32)
    return traj, morph, flow, base_norm


def train_regime_decoder(
    train: seq.ArrayPack,
    val: seq.ArrayPack,
    *,
    max_horizon: int,
    eval_horizons: list[int],
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    device: torch.device,
    endpoint_weight: float,
    speed_weight: float,
    direction_weight: float,
    regime_weight: float,
    residual_route_weight: float,
) -> tuple[RegimeMagnitudeDecoder, tuple[seq.Norm, seq.Norm, seq.Norm], RegimeStats, dict[str, Any]]:
    torch.manual_seed(int(seed) + 8301)
    np.random.seed(int(seed) + 8303)
    input_norms = branch_norms(train)
    stats = fit_regime_stats(train, eval_horizons)

    tr_traj, tr_morph, tr_flow, tr_base_norm = normalize_inputs(train, input_norms)
    va_traj, va_morph, va_flow, va_base_norm = normalize_inputs(val, input_norms)
    tr_y = train.target_steps.astype(np.float32)
    tr_res_norm = seq.apply_norm(train.residual_steps, stats.residual_norm, clip=12.0)
    tr_log_speed = log_speed_norm_np(train.target_steps, stats)
    tr_labels = speed_labels_np(train.target_steps, stats.speed_bins)

    model = RegimeMagnitudeDecoder(
        train.traj.shape[1],
        train.morph.shape[1],
        train.flow.shape[1],
        hidden_dim,
        max_horizon,
        dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    tr_t = torch.as_tensor(tr_traj, dtype=torch.float32, device=device)
    tr_m = torch.as_tensor(tr_morph, dtype=torch.float32, device=device)
    tr_f = torch.as_tensor(tr_flow, dtype=torch.float32, device=device)
    tr_b_norm = torch.as_tensor(tr_base_norm, dtype=torch.float32, device=device)
    tr_base_px = torch.as_tensor(train.base_step, dtype=torch.float32, device=device)
    tr_y_t = torch.as_tensor(tr_y, dtype=torch.float32, device=device)
    tr_res_norm_t = torch.as_tensor(tr_res_norm, dtype=torch.float32, device=device)
    tr_log_t = torch.as_tensor(tr_log_speed, dtype=torch.float32, device=device)
    tr_label_t = torch.as_tensor(tr_labels, dtype=torch.long, device=device)

    va_t = torch.as_tensor(va_traj, dtype=torch.float32, device=device)
    va_m = torch.as_tensor(va_morph, dtype=torch.float32, device=device)
    va_f = torch.as_tensor(va_flow, dtype=torch.float32, device=device)
    va_b_norm = torch.as_tensor(va_base_norm, dtype=torch.float32, device=device)
    va_base_px = torch.as_tensor(val.base_step, dtype=torch.float32, device=device)

    step_speed = np.linalg.norm(train.target_steps, axis=2)
    sample_weight = np.ones_like(step_speed, dtype=np.float32)
    sample_weight[step_speed <= stats.speed_bins[0]] = 1.25
    sample_weight[step_speed >= stats.speed_bins[2]] = 1.45
    tr_weight_t = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    rng = np.random.default_rng(int(seed) + 8311)
    n = len(train.traj)
    for epoch in range(int(epochs)):
        order = rng.permutation(n)
        model.train()
        for start in range(0, n, int(batch_size)):
            idx_np = order[start : start + int(batch_size)]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            raw = model(tr_t[idx], tr_m[idx], tr_f[idx], tr_b_norm[idx])
            decoded = decode_outputs(raw, tr_base_px[idx], stats, device)
            pred = decoded["pred_steps"]
            step_err = F.smooth_l1_loss(
                pred / stats.target_std_scalar,
                tr_y_t[idx] / stats.target_std_scalar,
                reduction="none",
            ).sum(dim=2)
            step_loss = (step_err * tr_weight_t[idx]).mean()

            endpoint_loss = 0.0
            for h in eval_horizons:
                scale = float(stats.endpoint_scales[int(h)])
                pred_ep = pred[:, :h, :].sum(dim=1) / scale
                true_ep = tr_y_t[idx, :h, :].sum(dim=1) / scale
                endpoint_loss = endpoint_loss + F.smooth_l1_loss(pred_ep, true_ep)
            endpoint_loss = endpoint_loss / max(len(eval_horizons), 1)

            speed_loss = F.smooth_l1_loss(raw["log_speed_norm"], tr_log_t[idx])
            target_dir = F.normalize(tr_y_t[idx], p=2, dim=2, eps=1e-6)
            dir_cos = torch.sum(decoded["direction"] * target_dir, dim=2)
            dir_weight = torch.clamp(torch.linalg.norm(tr_y_t[idx], dim=2) / stats.target_std_scalar, 0.2, 3.0)
            direction_loss = ((1.0 - dir_cos) * dir_weight).mean()
            regime_loss = F.cross_entropy(
                raw["regime_logits"].reshape(-1, 4),
                tr_label_t[idx].reshape(-1),
            )
            residual_loss = F.smooth_l1_loss(raw["residual_norm"].contiguous(), tr_res_norm_t[idx].contiguous())

            loss = (
                step_loss
                + float(endpoint_weight) * endpoint_loss
                + float(speed_weight) * speed_loss
                + float(direction_weight) * direction_loss
                + float(regime_weight) * regime_loss
                + float(residual_route_weight) * residual_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            raw = model(va_t, va_m, va_f, va_b_norm)
            pred_steps = decode_outputs(raw, va_base_px, stats, device)["pred_steps"].detach().cpu().numpy()
            score = seq.validation_score(val.target_steps, pred_steps, eval_horizons)
        if score < best_val - 1e-4:
            best_val = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= 18:
            break

    model.load_state_dict(best_state)
    model.eval()
    info = {
        "best_epoch": int(best_epoch),
        "best_val_endpoint_rmse_px": float(best_val),
        "speed_clip": float(stats.speed_clip),
        "speed_bin_q25": float(stats.speed_bins[0]),
        "speed_bin_q50": float(stats.speed_bins[1]),
        "speed_bin_q75": float(stats.speed_bins[2]),
        "endpoint_weight": float(endpoint_weight),
        "speed_weight": float(speed_weight),
        "direction_weight": float(direction_weight),
        "regime_weight": float(regime_weight),
        "residual_route_weight": float(residual_route_weight),
    }
    return model, input_norms, stats, info


@torch.no_grad()
def predict_regime_decoder(
    model: RegimeMagnitudeDecoder,
    pack: seq.ArrayPack,
    input_norms: tuple[seq.Norm, seq.Norm, seq.Norm],
    stats: RegimeStats,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    traj, morph, flow, base_norm = normalize_inputs(pack, input_norms)
    x_t = torch.as_tensor(traj, dtype=torch.float32, device=device)
    x_m = torch.as_tensor(morph, dtype=torch.float32, device=device)
    x_f = torch.as_tensor(flow, dtype=torch.float32, device=device)
    x_b_norm = torch.as_tensor(base_norm, dtype=torch.float32, device=device)
    base_px = torch.as_tensor(pack.base_step, dtype=torch.float32, device=device)
    preds = []
    gates = []
    blends = []
    polar_mag = []
    residual_mag = []
    for start in range(0, len(traj), int(batch_size)):
        raw = model(
            x_t[start : start + batch_size],
            x_m[start : start + batch_size],
            x_f[start : start + batch_size],
            x_b_norm[start : start + batch_size],
        )
        decoded = decode_outputs(raw, base_px[start : start + batch_size], stats, device)
        preds.append(decoded["pred_steps"].detach().cpu().numpy())
        gates.append(raw["gates"].detach().cpu().numpy())
        blends.append(decoded["blend"].detach().cpu().numpy())
        polar_mag.append(torch.linalg.norm(decoded["polar_route"], dim=2).detach().cpu().numpy())
        residual_mag.append(torch.linalg.norm(decoded["residual_route"], dim=2).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    gate_np = np.concatenate(gates, axis=0)
    blend_np = np.concatenate(blends, axis=0)
    polar_np = np.concatenate(polar_mag, axis=0)
    residual_np = np.concatenate(residual_mag, axis=0)
    info = {
        "gate_morph_mean": float(np.mean(gate_np[:, 0])),
        "gate_flow_mean": float(np.mean(gate_np[:, 1])),
        "blend_polar_mean": float(np.mean(blend_np)),
        "polar_step_mag_mean": float(np.mean(polar_np)),
        "residual_route_step_mag_mean": float(np.mean(residual_np)),
    }
    return pred, info


def endpoint_rows(
    *,
    dataset: str,
    seed: int,
    model: str,
    feature_block: str,
    target_steps: np.ndarray,
    pred_steps: np.ndarray,
    base_step: np.ndarray,
    horizons: list[int],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    return seq.evaluate_steps(
        dataset=dataset,
        seed=seed,
        model=model,
        feature_block=feature_block,
        target_steps=target_steps,
        pred_steps=pred_steps,
        base_step=base_step,
        horizons=horizons,
        info=info,
    )


def step_metric_rows(
    *,
    dataset: str,
    seed: int,
    model: str,
    feature_block: str,
    target_steps: np.ndarray,
    pred_steps: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for i in range(target_steps.shape[1]):
        y = target_steps[:, i, :]
        pred = pred_steps[:, i, :]
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "model": model,
                "feature_block": feature_block,
                "step": int(i + 1),
                "rmse_px": ifp.vector_rmse(y, pred),
                "r2": ifp.vector_r2(y, pred),
                "cosine": ifp.mean_cosine(y, pred),
                "magnitude_ratio": ifp.magnitude_ratio(y, pred),
            }
        )
    return rows


def magnitude_bin_rows(
    *,
    dataset: str,
    seed: int,
    model: str,
    feature_block: str,
    horizon: int,
    target_steps: np.ndarray,
    pred_steps: np.ndarray,
) -> list[dict[str, Any]]:
    y = target_steps[:, :horizon, :].sum(axis=1)
    pred = pred_steps[:, :horizon, :].sum(axis=1)
    mag = np.linalg.norm(y, axis=1)
    edges = np.quantile(mag, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for b in range(4):
        if b < 3:
            mask = (mag >= edges[b]) & (mag < edges[b + 1])
        else:
            mask = (mag >= edges[b]) & (mag <= edges[b + 1])
        if not np.any(mask):
            continue
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "model": model,
                "feature_block": feature_block,
                "horizon": int(horizon),
                "bin": int(b),
                "count": int(mask.sum()),
                "target_mag_mean": float(np.mean(np.linalg.norm(y[mask], axis=1))),
                "pred_mag_mean": float(np.mean(np.linalg.norm(pred[mask], axis=1))),
                "magnitude_ratio": ifp.magnitude_ratio(y[mask], pred[mask]),
                "cosine": ifp.mean_cosine(y[mask], pred[mask]),
                "rmse_px": ifp.vector_rmse(y[mask], pred[mask]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(args.features)
    horizons = seq.parse_ints(args.eval_horizons)
    max_h = max(horizons)
    full = seq.build_sequence_table(
        features=features,
        table_root=args.table_root,
        dataset=args.dataset,
        max_horizon=max_h,
    )
    split = seq.make_split(
        full,
        seq.parse_ints(args.train_sequences),
        seq.parse_ints(args.val_sequences),
        seq.parse_ints(args.test_sequences),
        int(args.seed),
    )
    split = apply_train_position_norm(split)
    train_df = seq.sample_rows(split.train, args.max_train_rows, args.seed + 11)
    val_df = seq.sample_rows(split.val, args.max_val_rows, args.seed + 13)
    test_df = seq.sample_rows(split.test, args.max_test_rows, args.seed + 17)

    feature_blocks = seq.parse_strs(args.feature_blocks)
    models = seq.parse_strs(args.models)
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    base_pack = seq.make_pack(test_df, max_h, "trajectory_only", args.seed)
    constant_pred = np.repeat(base_pack.base_step[:, None, :], max_h, axis=1)
    rows.extend(
        endpoint_rows(
            dataset=args.dataset,
            seed=args.seed,
            model="constant_velocity",
            feature_block="none",
            target_steps=base_pack.target_steps,
            pred_steps=constant_pred,
            base_step=base_pack.base_step,
            horizons=horizons,
            info={},
        )
    )

    for block in feature_blocks:
        train_pack = seq.make_pack(train_df, max_h, block, args.seed + 101)
        val_pack = seq.make_pack(val_df, max_h, block, args.seed + 103)
        test_pack = seq.make_pack(test_df, max_h, block, args.seed + 107)
        probe_rows.append(
            {
                "dataset": args.dataset,
                "feature_block": block,
                "traj_dim": int(train_pack.traj.shape[1]),
                "morph_dim": int(train_pack.morph.shape[1]),
                "flow_dim": int(train_pack.flow.shape[1]),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
            }
        )
        for model_name in models:
            print(f"[{args.dataset} seed{args.seed}] {model_name} {block}", flush=True)
            if model_name == "baseline_torch":
                baseline_model, norms, info = seq.train_torch_decoder(
                    train_pack,
                    val_pack,
                    max_horizon=max_h,
                    eval_horizons=horizons,
                    seed=args.seed + len(rows),
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    dropout=args.dropout,
                    device=device,
                )
                pred_steps, gates = seq.predict_torch_decoder(
                    baseline_model,
                    test_pack,
                    norms,
                    batch_size=args.batch_size,
                    device=device,
                )
                info = {
                    **info,
                    "gate_morph_mean": float(np.mean(gates[:, 0])),
                    "gate_flow_mean": float(np.mean(gates[:, 1])),
                }
            elif model_name == "regime":
                model, norms, stats, info = train_regime_decoder(
                    train_pack,
                    val_pack,
                    max_horizon=max_h,
                    eval_horizons=horizons,
                    seed=args.seed + len(rows),
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    dropout=args.dropout,
                    device=device,
                    endpoint_weight=args.endpoint_weight,
                    speed_weight=args.speed_weight,
                    direction_weight=args.direction_weight,
                    regime_weight=args.regime_weight,
                    residual_route_weight=args.residual_route_weight,
                )
                pred_steps, pred_info = predict_regime_decoder(
                    model,
                    test_pack,
                    norms,
                    stats,
                    batch_size=args.batch_size,
                    device=device,
                )
                info = {**info, **pred_info}
            else:
                raise ValueError(f"Unknown model={model_name}")
            rows.extend(
                endpoint_rows(
                    dataset=args.dataset,
                    seed=args.seed,
                    model=model_name,
                    feature_block=block,
                    target_steps=test_pack.target_steps,
                    pred_steps=pred_steps,
                    base_step=test_pack.base_step,
                    horizons=horizons,
                    info=info,
                )
            )
            step_rows.extend(
                step_metric_rows(
                    dataset=args.dataset,
                    seed=args.seed,
                    model=model_name,
                    feature_block=block,
                    target_steps=test_pack.target_steps,
                    pred_steps=pred_steps,
                )
            )
            for h in horizons:
                bin_rows.extend(
                    magnitude_bin_rows(
                        dataset=args.dataset,
                        seed=args.seed,
                        model=model_name,
                        feature_block=block,
                        horizon=h,
                        target_steps=test_pack.target_steps,
                        pred_steps=pred_steps,
                    )
                )

    return pd.DataFrame(rows), pd.DataFrame(step_rows), pd.DataFrame(bin_rows), pd.DataFrame(probe_rows)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    steps: pd.DataFrame,
    bins: pd.DataFrame,
    probe: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    best = summary.sort_values(["horizon", "rmse_px"]).groupby("horizon").head(10)
    lines = [
        "# Regime/magnitude sequence decoder",
        "",
        "## Payload",
        "",
        "```json",
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Best endpoint rows",
        "",
        best.to_markdown(index=False) if not best.empty else "_No summary rows._",
        "",
        "## Step metrics",
        "",
        steps.to_markdown(index=False) if not steps.empty else "_No step rows._",
        "",
        "## h6 target-magnitude bins",
        "",
        bins[bins["horizon"].eq(max(seq.parse_ints(args.eval_horizons)))].to_markdown(index=False)
        if not bins.empty
        else "_No bin rows._",
        "",
        "## Feature probe",
        "",
        probe.to_markdown(index=False) if not probe.empty else "_No probe rows._",
        "",
        "## Interpretation checklist",
        "",
        "- Regime model should improve RMSE without worsening h1.",
        "- Large-motion bins should reduce magnitude shrink.",
        "- Small-motion bins should not be over-pushed relative to baseline.",
        "- Shuffled raw context must not reproduce full-context gain.",
    ]
    (out_dir / "regime_magnitude_sequence_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--eval-horizons", default="1,2,4,6")
    parser.add_argument("--models", default="baseline_torch,regime")
    parser.add_argument(
        "--feature-blocks",
        default="trajectory_only,trajectory_morphology_tissue_flow,trajectory_morphology_tissue_flow_shuffled_both",
    )
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-val-rows", type=int, default=20000)
    parser.add_argument("--max-test-rows", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--endpoint-weight", type=float, default=0.55)
    parser.add_argument("--speed-weight", type=float, default=0.12)
    parser.add_argument("--direction-weight", type=float, default=0.06)
    parser.add_argument("--regime-weight", type=float, default=0.04)
    parser.add_argument("--residual-route-weight", type=float, default=0.08)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, steps, bins, probe = run(args)
    summary.to_csv(args.out_dir / "regime_magnitude_sequence_summary.csv", index=False)
    steps.to_csv(args.out_dir / "regime_magnitude_sequence_step_metrics.csv", index=False)
    bins.to_csv(args.out_dir / "regime_magnitude_sequence_magnitude_bins.csv", index=False)
    probe.to_csv(args.out_dir / "regime_magnitude_sequence_feature_probe.csv", index=False)
    write_report(args.out_dir, summary, steps, bins, probe, args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
