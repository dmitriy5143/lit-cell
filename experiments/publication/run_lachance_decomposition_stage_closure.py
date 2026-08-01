#!/usr/bin/env python3
"""Closure gate for the decomposition stage.

This runner tests the missing contract between the decomposition teacher and
the next planned modules (learned trajectory generator + sequence critic):

    future-aware posterior q(z, m | residual trajectory)
        -> causal student prior p(z, m | context)
        -> validated latent interface for generator/critic

It is deliberately not a final clean-best predictor.  The goal is to decide
whether decomposition is ready to hand off a soft, causal, interpretable latent
state to the next architecture stage.
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

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402


DEFAULT_FEATURES = audit.DEFAULT_FEATURES
DEFAULT_OUT = ROOT / "outputs" / "decomposition_stage_closure_2026-06-24"
EPS = 1e-8


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)).astype(np.float32)


def entropy_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def topk_acc(y: np.ndarray, p: np.ndarray, k: int) -> float:
    order = np.argsort(-p, axis=1)[:, : int(k)]
    return float(np.mean(np.any(order == y[:, None], axis=1)))


def mode_metrics(q_soft: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    p = softmax_np(logits)
    y = np.argmax(q_soft, axis=1)
    pred = np.argmax(p, axis=1)
    kl = np.sum(q_soft * (np.log(np.clip(q_soft, 1e-8, 1.0)) - np.log(np.clip(p, 1e-8, 1.0))), axis=1)
    return {
        "mode_acc": float(np.mean(pred == y)),
        "mode_top3": topk_acc(y, p, min(3, p.shape[1])),
        "mode_kl": float(np.mean(kl)),
        "posterior_mode_entropy": float(np.mean(entropy_np(q_soft))),
        "prior_mode_entropy": float(np.mean(entropy_np(p))),
        "prior_mode_usage_entropy": float(entropy_np(p.mean(axis=0, keepdims=True))[0]),
    }


def diag_kl_q_to_p(mu_q: np.ndarray, logvar_q: np.ndarray, mu_p: np.ndarray, logvar_p: np.ndarray) -> float:
    vq = np.exp(np.clip(logvar_q, -10.0, 10.0))
    vp = np.exp(np.clip(logvar_p, -10.0, 10.0))
    kl = 0.5 * (logvar_p - logvar_q + (vq + np.square(mu_q - mu_p)) / np.maximum(vp, EPS) - 1.0)
    return float(np.mean(np.sum(kl, axis=1)))


class PosteriorVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(y)
        return self.mu(h), torch.clamp(self.logvar(h), -8.0, 5.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


class ComponentStudentPrior(nn.Module):
    def __init__(self, dims: dict[str, int], hidden: int, latent_dim: int, mode_k: int):
        super().__init__()
        self.blocks = [b for b, d in dims.items() if int(d) > 0]
        if not self.blocks:
            raise ValueError("At least one non-empty feature block is required")
        self.encoders = nn.ModuleDict(
            {
                b: nn.Sequential(
                    nn.Linear(int(dims[b]), hidden),
                    nn.SiLU(),
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                )
                for b in self.blocks
            }
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * len(self.blocks), hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.blocks)),
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.mode_logits = nn.Linear(hidden, mode_k)
        self.uncertainty = nn.Linear(hidden, 1)

    def forward(self, xs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hs = [self.encoders[b](xs[b]) for b in self.blocks]
        stack = torch.stack(hs, dim=1)
        gate_logits = self.gate(torch.cat(hs, dim=-1))
        gates = torch.softmax(gate_logits, dim=-1)
        pooled = torch.sum(stack * gates[:, :, None], dim=1)
        h = self.head(pooled)
        return (
            self.mu(h),
            torch.clamp(self.logvar(h), -7.0, 4.0),
            self.mode_logits(h),
            F.softplus(self.uncertainty(h)).squeeze(-1),
            gates,
        )


def batches(n: int, batch_size: int, seed: int, shuffle: bool = True):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def device_from_arg(text: str) -> torch.device:
    if text != "auto":
        return torch.device(text)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class PosteriorPack:
    model: PosteriorVAE
    scaler: StandardScaler
    y_train_s: np.ndarray
    y_val_s: np.ndarray
    y_test_s: np.ndarray
    mu_train: np.ndarray
    logvar_train: np.ndarray
    mu_val: np.ndarray
    logvar_val: np.ndarray
    mu_test: np.ndarray
    logvar_test: np.ndarray
    mode_soft_train: np.ndarray
    mode_soft_val: np.ndarray
    mode_soft_test: np.ndarray
    mode_labels_train: np.ndarray
    mode_labels_val: np.ndarray
    mode_labels_test: np.ndarray
    centers: np.ndarray


def decode_residual(
    model: PosteriorVAE,
    scaler: StandardScaler,
    z: np.ndarray,
    *,
    max_horizon: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for idx in batches(len(z), batch_size, 17, shuffle=False):
            y_s = model.decode(as_tensor(z[idx], device)).cpu().numpy()
            out.append(y_s)
    y_s_all = np.concatenate(out, axis=0)
    y = scaler.inverse_transform(y_s_all).astype(np.float32)
    return audit.unflatten_residual(y, max_horizon)


def train_posterior(arrays: audit.SplitArrays, args: argparse.Namespace, device: torch.device) -> PosteriorPack:
    if StandardScaler is None or KMeans is None:
        raise RuntimeError("sklearn is required")
    ytr = audit.flatten_residual(arrays.residual_train)
    yva = audit.flatten_residual(arrays.residual_val)
    yte = audit.flatten_residual(arrays.residual_test)
    scaler = StandardScaler()
    ytr_s = scaler.fit_transform(ytr).astype(np.float32)
    yva_s = scaler.transform(yva).astype(np.float32)
    yte_s = scaler.transform(yte).astype(np.float32)

    model = PosteriorVAE(ytr_s.shape[1], args.latent_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.posterior_epochs):
        model.train()
        beta = float(args.posterior_beta) * min(1.0, (epoch + 1) / max(1, int(args.kl_warmup_epochs)))
        for idx in batches(len(ytr_s), args.batch_size, args.seed + epoch):
            yb = as_tensor(ytr_s[idx], device)
            recon, mu, logvar = model(yb)
            recon_loss = F.mse_loss(recon, yb)
            kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + beta * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    def encode_all(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        mus, lvs = [], []
        with torch.no_grad():
            for idx in batches(len(y), args.batch_size, args.seed, shuffle=False):
                mu, lv = model.encode(as_tensor(y[idx], device))
                mus.append(mu.cpu().numpy())
                lvs.append(lv.cpu().numpy())
        return np.concatenate(mus, axis=0).astype(np.float32), np.concatenate(lvs, axis=0).astype(np.float32)

    mu_tr, lv_tr = encode_all(ytr_s)
    mu_va, lv_va = encode_all(yva_s)
    mu_te, lv_te = encode_all(yte_s)

    km = KMeans(n_clusters=args.mode_k, n_init=20, random_state=args.seed)
    km.fit(mu_tr)
    centers = km.cluster_centers_.astype(np.float32)

    def mode_soft(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d2 = np.mean(np.square(mu[:, None, :] - centers[None, :, :]), axis=2)
        if args.mode_temperature <= 0:
            assigned = np.min(d2, axis=1)
            temp = float(np.median(assigned) + 1e-6)
        else:
            temp = float(args.mode_temperature)
        p = softmax_np(-d2 / max(temp, 1e-5))
        return p, np.argmax(p, axis=1).astype(np.int64)

    p_tr, lab_tr = mode_soft(mu_tr)
    p_va, lab_va = mode_soft(mu_va)
    p_te, lab_te = mode_soft(mu_te)
    return PosteriorPack(
        model=model,
        scaler=scaler,
        y_train_s=ytr_s,
        y_val_s=yva_s,
        y_test_s=yte_s,
        mu_train=mu_tr,
        logvar_train=lv_tr,
        mu_val=mu_va,
        logvar_val=lv_va,
        mu_test=mu_te,
        logvar_test=lv_te,
        mode_soft_train=p_tr,
        mode_soft_val=p_va,
        mode_soft_test=p_te,
        mode_labels_train=lab_tr,
        mode_labels_val=lab_va,
        mode_labels_test=lab_te,
        centers=centers,
    )


def variant_blocks(name: str, available: dict[str, np.ndarray]) -> list[str]:
    base = [
        "self",
        "history_short",
        "history_long",
        "history_summary",
        "morphology",
        "flow",
        "raw_context",
        "observability",
        "boundary",
        "crowding",
    ]
    base = [b for b in base if b in available and available[b].shape[1] > 0]
    if name == "full":
        return base
    if name == "trajectory_only" or name == "self_only":
        return [b for b in ["self"] if b in base]
    if name == "history_only":
        return [b for b in ["history_short", "history_long", "history_summary"] if b in base]
    if name == "history_long_only":
        return [b for b in ["history_long", "history_summary"] if b in base]
    if name == "trajectory_history":
        return [b for b in ["self", "history_short", "history_long", "history_summary"] if b in base]
    if name == "no_history":
        return [b for b in base if not b.startswith("history_")]
    if name == "no_flow":
        return [b for b in base if b != "flow"]
    if name == "no_morphology":
        return [b for b in base if b != "morphology"]
    if name == "no_raw_context":
        return [b for b in base if b != "raw_context"]
    if name == "no_crowding":
        return [b for b in base if b != "crowding"]
    if name == "no_boundary":
        return [b for b in base if b != "boundary"]
    if name == "no_observability":
        return [b for b in base if b != "observability"]
    if name == "no_boundary_crowding":
        return [b for b in base if b not in {"boundary", "crowding"}]
    if name == "context_only":
        return [b for b in base if b != "self"] or base
    return base


def xs_for_blocks(xdict: dict[str, np.ndarray], blocks: list[str], idx: np.ndarray | None, device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for b in blocks:
        x = xdict[b] if idx is None else xdict[b][idx]
        out[b] = as_tensor(x, device)
    return out


def predict_student(
    model: ComponentStudentPrior,
    xdict: dict[str, np.ndarray],
    blocks: list[str],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    mus, lvs, logits, gates, unc = [], [], [], [], []
    with torch.no_grad():
        for idx in batches(len(next(iter(xdict.values()))), batch_size, 19, shuffle=False):
            mu, lv, lo, un, ga = model(xs_for_blocks(xdict, blocks, idx, device))
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
            logits.append(lo.cpu().numpy())
            gates.append(ga.cpu().numpy())
            unc.append(un.cpu().numpy())
    return {
        "mu": np.concatenate(mus, axis=0).astype(np.float32),
        "logvar": np.concatenate(lvs, axis=0).astype(np.float32),
        "logits": np.concatenate(logits, axis=0).astype(np.float32),
        "gates": np.concatenate(gates, axis=0).astype(np.float32),
        "uncertainty": np.concatenate(unc, axis=0).astype(np.float32),
    }


def train_student(
    arrays: audit.SplitArrays,
    posterior: PosteriorPack,
    args: argparse.Namespace,
    *,
    variant: str,
    device: torch.device,
    row_shuffle_train: bool = False,
) -> tuple[ComponentStudentPrior, list[str], dict[str, Any]]:
    blocks = variant_blocks(variant, arrays.x_train)
    dims = {b: arrays.x_train[b].shape[1] for b in blocks}
    model = ComponentStudentPrior(dims, args.hidden_dim, args.latent_dim, args.mode_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    perm = None
    if row_shuffle_train:
        perm = np.random.default_rng(args.seed + 7701).permutation(len(posterior.mu_train))
    for epoch in range(args.student_epochs):
        model.train()
        for idx in batches(len(posterior.mu_train), args.batch_size, args.seed + 1100 + epoch):
            x_idx = perm[idx] if perm is not None else idx
            xs = xs_for_blocks(arrays.x_train, blocks, x_idx, device)
            mu_q = as_tensor(posterior.mu_train[idx], device)
            lv_q = as_tensor(posterior.logvar_train[idx], device)
            q_mode = as_tensor(posterior.mode_soft_train[idx], device)
            y_s = as_tensor(posterior.y_train_s[idx], device)
            mu_p, lv_p, logits, unc, gates = model(xs)
            var_p = torch.exp(lv_p)
            var_q = torch.exp(lv_q)
            latent_nll = 0.5 * torch.mean(torch.sum((var_q + (mu_q - mu_p).pow(2)) / torch.clamp(var_p, min=1e-6) + lv_p, dim=1))
            mode_kl = torch.mean(torch.sum(q_mode * (torch.log(torch.clamp(q_mode, min=1e-8)) - F.log_softmax(logits, dim=-1)), dim=1))
            # Frozen decoder consistency: student latent should decode back into
            # the teacher residual trajectory, but this term is deliberately
            # weaker than latent/mode distillation.
            with torch.no_grad():
                _ = posterior.model.dec[0].weight  # keeps linter quiet; decoder is below
            recon_s = posterior.model.decode(mu_p)
            recon_loss = F.mse_loss(recon_s, y_s)
            gate_entropy = -torch.mean(torch.sum(gates * torch.log(gates + 1e-8), dim=1))
            loss = latent_nll + args.mode_loss_weight * mode_kl + args.recon_loss_weight * recon_loss - args.gate_entropy_weight * gate_entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    info = {"blocks": blocks, "row_shuffle_train": bool(row_shuffle_train)}
    return model, blocks, info


def evaluate_variant(
    arrays: audit.SplitArrays,
    posterior: PosteriorPack,
    model: ComponentStudentPrior,
    blocks: list[str],
    args: argparse.Namespace,
    *,
    variant: str,
    device: torch.device,
    test_shuffle: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    x_test = arrays.x_test
    if test_shuffle:
        rng = np.random.default_rng(args.seed + 8801)
        perm = rng.permutation(len(arrays.residual_test))
        x_test = {k: v[perm] for k, v in arrays.x_test.items()}
    pred = predict_student(model, x_test, blocks, device=device, batch_size=args.batch_size)
    residual_mean = decode_residual(
        posterior.model,
        posterior.scaler,
        pred["mu"],
        max_horizon=args.max_horizon,
        device=device,
        batch_size=args.batch_size,
    )
    label = f"student_mean_{variant}" + ("_test_shuffled" if test_shuffle else "")
    summary = audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=residual_mean,
        horizons=args.horizons,
        label=label,
        extra={"stage": "student_prior_mean", "variant": variant, "test_shuffle": bool(test_shuffle)},
    )

    # Fixed-budget sample oracle for the future generator interface.
    rng = np.random.default_rng(args.seed + 9901)
    k_values = sorted(set(int(k) for k in args.oracle_k))
    z_samples = []
    for _ in range(max(k_values)):
        eps = rng.normal(size=pred["mu"].shape).astype(np.float32)
        z_samples.append(pred["mu"] + eps * np.exp(0.5 * np.clip(pred["logvar"], -8, 5)) * float(args.sample_scale))
    best_by_k: dict[int, np.ndarray] = {}
    best_dist = np.full(len(arrays.residual_test), np.inf, dtype=np.float32)
    best_residual = np.zeros_like(arrays.residual_test)
    true_flat = audit.flatten_residual(arrays.residual_test)
    for i, z in enumerate(z_samples, start=1):
        res = decode_residual(
            posterior.model,
            posterior.scaler,
            z,
            max_horizon=args.max_horizon,
            device=device,
            batch_size=args.batch_size,
        )
        dist = np.mean(np.square(audit.flatten_residual(res) - true_flat), axis=1)
        take = dist < best_dist
        best_dist[take] = dist[take]
        best_residual[take] = res[take]
        if i in k_values:
            best_by_k[i] = best_residual.copy()
    for k, res in best_by_k.items():
        summary.extend(
            audit.endpoint_metrics(
                steps_true=arrays.steps_test,
                base=arrays.base_test,
                residual_pred=res,
                horizons=args.horizons,
                label=f"student_oracle@{k}_{variant}" + ("_test_shuffled" if test_shuffle else ""),
                extra={"stage": "student_prior_sample_oracle", "variant": variant, "oracle_k": k, "test_shuffle": bool(test_shuffle)},
            )
        )

    distill = {
        "variant": variant,
        "test_shuffle": bool(test_shuffle),
        "latent_rmse": audit.rmse(posterior.mu_test, pred["mu"]),
        "gaussian_kl_q_to_p": diag_kl_q_to_p(posterior.mu_test, posterior.logvar_test, pred["mu"], pred["logvar"]),
        "prior_logvar_mean": float(np.mean(pred["logvar"])),
        "uncertainty_mean": float(np.mean(pred["uncertainty"])),
    }
    distill.update(mode_metrics(posterior.mode_soft_test, pred["logits"]))
    distill_rows = [distill]
    gate_rows = []
    for i, b in enumerate(blocks):
        gate_rows.append(
            {
                "variant": variant,
                "test_shuffle": bool(test_shuffle),
                "block": b,
                "mean_gate": float(np.mean(pred["gates"][:, i])),
                "std_gate": float(np.std(pred["gates"][:, i])),
            }
        )
    return summary, distill_rows, gate_rows


def ridge_direct_baseline(arrays: audit.SplitArrays, posterior: PosteriorPack, args: argparse.Namespace) -> list[dict[str, Any]]:
    if Ridge is None:
        return []
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(arrays.x_train["all_context"], posterior.y_train_s)
    pred_s = ridge.predict(arrays.x_test["all_context"]).astype(np.float32)
    pred = posterior.scaler.inverse_transform(pred_s).astype(np.float32)
    residual = audit.unflatten_residual(pred, args.max_horizon)
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=residual,
        horizons=args.horizons,
        label="direct_ridge_residual_all_context",
        extra={"stage": "direct_causal_baseline", "variant": "direct_ridge"},
    )


def random_latent_oracle(arrays: audit.SplitArrays, posterior: PosteriorPack, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed + 770)
    k_values = sorted(set(int(k) for k in args.oracle_k))
    mu = posterior.mu_train.mean(axis=0, keepdims=True)
    std = posterior.mu_train.std(axis=0, keepdims=True) + 1e-6
    best_dist = np.full(len(arrays.residual_test), np.inf, dtype=np.float32)
    best_residual = np.zeros_like(arrays.residual_test)
    true_flat = audit.flatten_residual(arrays.residual_test)
    rows: list[dict[str, Any]] = []
    for i in range(1, max(k_values) + 1):
        z = mu + rng.normal(size=(len(arrays.residual_test), args.latent_dim)).astype(np.float32) * std * float(args.sample_scale)
        res = decode_residual(posterior.model, posterior.scaler, z, max_horizon=args.max_horizon, device=device, batch_size=args.batch_size)
        dist = np.mean(np.square(audit.flatten_residual(res) - true_flat), axis=1)
        take = dist < best_dist
        best_dist[take] = dist[take]
        best_residual[take] = res[take]
        if i in k_values:
            rows.extend(
                audit.endpoint_metrics(
                    steps_true=arrays.steps_test,
                    base=arrays.base_test,
                    residual_pred=best_residual,
                    horizons=args.horizons,
                    label=f"random_train_latent_oracle@{i}",
                    extra={"stage": "random_latent_control", "variant": "random_train_latent", "oracle_k": i},
                )
            )
    return rows


def posterior_metrics(arrays: audit.SplitArrays, posterior: PosteriorPack, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    residual = decode_residual(
        posterior.model,
        posterior.scaler,
        posterior.mu_test,
        max_horizon=args.max_horizon,
        device=device,
        batch_size=args.batch_size,
    )
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=residual,
        horizons=args.horizons,
        label="posterior_mean_reconstruction",
        extra={"stage": "target_aware_upper_bound", "variant": "posterior"},
    )


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, distill: pd.DataFrame, gates: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Decomposition Stage Closure Report\n")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- features: `{args.features}`")
    lines.append(f"- horizons: `{','.join(map(str, args.horizons))}`")
    lines.append("")
    lines.append("## Endpoint Metrics")
    if not summary.empty and "horizon" in summary.columns:
        focus = summary[summary["horizon"].notna()].copy()
        for h in args.horizons:
            hdf = focus[focus["horizon"].eq(h)].sort_values("rmse").head(12)
            lines.append(f"### h{h}")
            for _, row in hdf.iterrows():
                lines.append(
                    f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, "
                    f"gain={row['gain_vs_base_pct']:.2f}%"
                )
    lines.append("")
    lines.append("## Distillation Metrics")
    if not distill.empty:
        cols = ["variant", "test_shuffle", "latent_rmse", "gaussian_kl_q_to_p", "mode_acc", "mode_top3", "mode_kl", "prior_mode_usage_entropy"]
        for _, row in distill.sort_values("gaussian_kl_q_to_p").iterrows():
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"{c}={row[c]:.3f}" if isinstance(row[c], (float, np.floating)) else f"{c}={row[c]}"
                        for c in cols
                        if c in row.index
                    ]
                )
            )
    lines.append("")
    lines.append("## Component Gates")
    if not gates.empty:
        agg = gates.groupby(["variant", "block"]).agg(mean_gate=("mean_gate", "mean")).reset_index()
        for variant, sub in agg.groupby("variant"):
            lines.append(f"### {variant}")
            for _, row in sub.sort_values("mean_gate", ascending=False).iterrows():
                lines.append(f"- `{row['block']}`: {row['mean_gate']:.3f}")
    lines.append("")
    lines.append("## Decision Notes")
    lines.append("- Passing this stage means student-prior samples beat shuffled/random controls and mode distributions do not collapse.")
    lines.append("- This runner is not the final clean-best integration; it validates the interface for the future generator/critic.")
    (out_dir / "decomposition_stage_closure_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    arrays, _ = audit.prepare_data(args)
    posterior = train_posterior(arrays, args, device)

    summary_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    summary_rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=np.zeros_like(arrays.residual_test),
            horizons=args.horizons,
            label="base_self_rollout_reference",
            extra={"stage": "reference", "variant": "base"},
        )
    )
    summary_rows.extend(posterior_metrics(arrays, posterior, args, device))
    summary_rows.extend(ridge_direct_baseline(arrays, posterior, args))
    summary_rows.extend(random_latent_oracle(arrays, posterior, args, device))

    for variant in args.variants:
        model, blocks, _ = train_student(arrays, posterior, args, variant=variant, device=device, row_shuffle_train=False)
        s, d, g = evaluate_variant(arrays, posterior, model, blocks, args, variant=variant, device=device)
        summary_rows.extend(s)
        distill_rows.extend(d)
        gate_rows.extend(g)
        if variant == "full":
            s, d, g = evaluate_variant(arrays, posterior, model, blocks, args, variant="full", device=device, test_shuffle=True)
            summary_rows.extend(s)
            distill_rows.extend(d)
            gate_rows.extend(g)
            model_sh, blocks_sh, _ = train_student(arrays, posterior, args, variant="full", device=device, row_shuffle_train=True)
            s, d, g = evaluate_variant(arrays, posterior, model_sh, blocks_sh, args, variant="full_row_shuffled_train", device=device)
            summary_rows.extend(s)
            distill_rows.extend(d)
            gate_rows.extend(g)

    summary = pd.DataFrame(summary_rows)
    distill = pd.DataFrame(distill_rows)
    gates = pd.DataFrame(gate_rows)
    summary.to_csv(args.out_dir / "decomposition_stage_closure_summary.csv", index=False)
    distill.to_csv(args.out_dir / "decomposition_stage_closure_distillation.csv", index=False)
    gates.to_csv(args.out_dir / "decomposition_stage_closure_gates.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, distill, gates)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "distill_rows": len(distill), "gate_rows": len(gates)}, indent=2))


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
    parser.add_argument("--max-train-rows", type=int, default=25000)
    parser.add_argument("--max-val-rows", type=int, default=7000)
    parser.add_argument("--max-test-rows", type=int, default=8000)
    parser.add_argument("--max-features-per-family", type=int, default=160)
    parser.add_argument("--max-all-features", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--mode-k", type=int, default=12)
    parser.add_argument("--mode-temperature", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--posterior-epochs", type=int, default=28)
    parser.add_argument("--student-epochs", type=int, default=24)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--posterior-beta", type=float, default=1e-3)
    parser.add_argument("--mode-loss-weight", type=float, default=0.30)
    parser.add_argument("--recon-loss-weight", type=float, default=0.20)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--oracle-k", type=str, default="8,16,32")
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument("--variants", type=str, default="full,trajectory_only,no_flow,no_raw_context,no_boundary_crowding")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.variants = parse_strs(args.variants)
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 5000)
        args.max_val_rows = min(args.max_val_rows, 2000)
        args.max_test_rows = min(args.max_test_rows, 2500)
        args.posterior_epochs = min(args.posterior_epochs, 8)
        args.student_epochs = min(args.student_epochs, 8)
        args.max_all_features = min(args.max_all_features, 192)
        args.variants = ["full", "trajectory_only"]
        args.oracle_k = [8, 16]
    run(args)


if __name__ == "__main__":
    main()
