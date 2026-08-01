#!/usr/bin/env python3
"""Sequence/Graph Trajectory Critic-Refiner v3 for LaChance trajectories.

This runner is the first full test of the selected next architecture:

    candidate trajectory tokens
    + decomposition/component-axis compatibility
    + explicit neighbour-edge context
    + global causal context
    -> sequence/graph critic
    -> per-horizon mixture + bounded correction + uncertainty

Target/future is used only for training losses, oracle labels and metrics.
Inference features are causal: candidate trajectories, central-cell context,
component-axis predictions and current/past neighbour-derived features.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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

import run_lachance_axis_conditioned_distillation_critic as axis  # noqa: E402
import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "sequence_graph_critic_v3_2026-07-01"
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
    return axis.residual_endpoints(residual, horizons)


def candidate_endpoints(query_pred: np.ndarray, horizons: list[int]) -> np.ndarray:
    return axis.candidate_endpoints(query_pred, horizons)


def endpoint_rows_from_residual_endpoints(
    *,
    arrays: audit.SplitArrays,
    residual_endpoint_pred: np.ndarray,
    horizons: list[int],
    label: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return axis.endpoint_rows_from_residual_endpoints(
        arrays=arrays,
        residual_endpoint_pred=residual_endpoint_pred,
        horizons=horizons,
        label=label,
        extra=extra,
    )


def standardize_candidate_steps(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    flat = train.reshape(-1, train.shape[-1])
    mean = np.nanmean(flat, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(flat, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def z(x: np.ndarray) -> np.ndarray:
        out = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return z(train), z(val), z(test), {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


def select_context_block(
    arrays: audit.SplitArrays,
    block: str,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    max_features = max(1, int(max_features))
    if block not in arrays.x_train:
        return (
            np.zeros((len(arrays.residual_train), max_features), dtype=np.float32),
            np.zeros((len(arrays.residual_val), max_features), dtype=np.float32),
            np.zeros((len(arrays.residual_test), max_features), dtype=np.float32),
            {"block": block, "enabled": False, "reason": "missing"},
        )
    xtr = np.nan_to_num(arrays.x_train[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xva = np.nan_to_num(arrays.x_val[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xte = np.nan_to_num(arrays.x_test[block], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if xtr.shape[1] == 0:
        return (
            np.zeros((len(xtr), max_features), dtype=np.float32),
            np.zeros((len(xva), max_features), dtype=np.float32),
            np.zeros((len(xte), max_features), dtype=np.float32),
            {"block": block, "enabled": False, "reason": "empty"},
        )
    if xtr.shape[1] > max_features:
        var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-max_features:]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
    return xtr.astype(np.float32), xva.astype(np.float32), xte.astype(np.float32), {
        "block": block,
        "enabled": True,
        "dim": int(xtr.shape[1]),
    }


def standardize_2d(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tr, va, te, scaler = seq.standardize(train, val, test)
    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32), finite_json(scaler)


def maybe_shuffle(x: np.ndarray, seed: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.permutation(len(x))]


class SequenceGraphCriticV3(nn.Module):
    def __init__(
        self,
        *,
        step_dim: int,
        axis_dim: int,
        ctx_dim: int,
        edge_dim: int,
        n_axes: int,
        n_horizons: int,
        max_horizon: int,
        hidden: int,
        heads: int,
        layers: int,
        dropout: float,
        correction_scale: float,
    ):
        super().__init__()
        self.n_horizons = int(n_horizons)
        self.max_horizon = int(max_horizon)
        self.correction_scale = float(correction_scale)
        self.step_proj = nn.Sequential(nn.Linear(step_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.time_embed = nn.Embedding(max_horizon, hidden)
        step_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.step_encoder = nn.TransformerEncoder(step_layer, num_layers=1)

        self.axis_proj = nn.Sequential(nn.Linear(axis_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.axis_embed = nn.Embedding(max(1, int(n_axes)), hidden)
        self.ctx_proj = nn.Sequential(nn.Linear(max(ctx_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.edge_proj = nn.Sequential(nn.Linear(max(edge_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden))

        cand_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(heads)),
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(cand_layer, num_layers=max(1, int(layers)))
        self.score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_horizons))
        self.correction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.logvar_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.temp = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        cand_steps: torch.Tensor,
        axis_feat: torch.Tensor,
        ctx: torch.Tensor,
        edge: torch.Tensor,
        cand_end: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n, q, h, _ = cand_steps.shape
        step = self.step_proj(cand_steps.reshape(n * q, h, -1))
        t = self.time_embed(torch.arange(h, device=cand_steps.device))[None, :, :]
        step = self.step_encoder(step + t).mean(dim=1).reshape(n, q, -1)

        # Axis tokens are candidate-specific route/component compatibility.
        if axis_feat.shape[-1] == 0:
            axis_h = torch.zeros_like(step)
        else:
            c = axis_feat.shape[2]
            ah = self.axis_proj(axis_feat)  # n,q,c,d
            eid = self.axis_embed(torch.arange(c, device=cand_steps.device))[None, None, :, :]
            axis_h = (ah + eid).mean(dim=2)

        if ctx.shape[-1] == 0:
            ctx = torch.zeros((n, 1), dtype=cand_steps.dtype, device=cand_steps.device)
        if edge.shape[-1] == 0:
            edge = torch.zeros((n, 1), dtype=cand_steps.dtype, device=cand_steps.device)
        ctx_h = self.ctx_proj(ctx)
        edge_h = self.edge_proj(edge)
        cand_h = step + axis_h + ctx_h[:, None, :] + edge_h[:, None, :]
        cand_h = self.candidate_encoder(cand_h)
        logits = self.score_head(cand_h).permute(0, 2, 1).contiguous()  # n,hh,q
        temp = torch.clamp(F.softplus(self.temp) + 0.15, min=0.20, max=5.0)
        weights = torch.softmax(logits / temp, dim=-1)
        mixture = torch.einsum("nhq,nqhd->nhd", weights, cand_end)
        pooled = torch.einsum("nhq,nqd->nhd", weights, cand_h)
        corr = torch.tanh(self.correction_head(pooled))
        correction = corr * self.correction_scale
        pred = mixture + correction
        logvar = self.logvar_head(pooled)
        return {"pred": pred, "mixture": mixture, "weights": weights, "logits": logits, "correction": correction, "logvar": logvar}


def _diag_horizon_head(x: torch.Tensor, n_horizons: int) -> torch.Tensor:
    # Kept for readability; unused in current forward after direct diagonal.
    return x.reshape(x.shape[0], n_horizons, n_horizons, 2).diagonal(dim1=1, dim2=2).permute(0, 2, 1)


def v3_loss(
    out: dict[str, torch.Tensor],
    target_end: torch.Tensor,
    target_soft: torch.Tensor,
    target_rank: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = out["pred"].contiguous()
    target_end = target_end.contiguous()
    target_soft = target_soft.contiguous()
    target_rank = target_rank.contiguous()
    weights = torch.clamp(out["weights"].contiguous(), min=1e-8)
    logits = out["logits"].contiguous()
    correction = out["correction"].contiguous()
    logvar = torch.clamp(out["logvar"].contiguous(), min=args.v3_logvar_min, max=args.v3_logvar_max)
    reg = F.smooth_l1_loss(pred, target_end)
    listwise = -torch.mean(torch.sum(target_soft * torch.log(weights), dim=-1))
    score_z = (logits - logits.mean(dim=-1, keepdim=True)) / torch.clamp(logits.std(dim=-1, keepdim=True), min=1e-3)
    rank_z = (target_rank - target_rank.mean(dim=-1, keepdim=True)) / torch.clamp(target_rank.std(dim=-1, keepdim=True), min=1e-3)
    rank = F.smooth_l1_loss(score_z, rank_z)
    nll = 0.5 * torch.mean((target_end - pred).pow(2) * torch.exp(-logvar) + logvar)
    corr_l2 = torch.mean(correction.pow(2))
    entropy = -torch.mean(torch.sum(weights * torch.log(weights), dim=-1))
    loss = (
        args.v3_reg_weight * reg
        + args.v3_listwise_weight * listwise
        + args.v3_rank_weight * rank
        + args.v3_nll_weight * nll
        + args.v3_correction_l2_weight * corr_l2
        - args.v3_entropy_weight * entropy
    )
    return loss, {
        "reg": float(reg.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "rank": float(rank.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "corr_l2": float(corr_l2.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def predict_v3(model: SequenceGraphCriticV3, pack: dict[str, np.ndarray], split: str, args: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    out_parts: dict[str, list[np.ndarray]] = {"pred": [], "mixture": [], "weights": [], "logits": [], "correction": [], "logvar": []}
    n = len(pack[f"steps_{split}"])
    with torch.no_grad():
        for idx in closure.batches(n, args.v3_batch_size, args.seed + 81001, shuffle=False):
            out = model(
                to_tensor(pack[f"steps_{split}"][idx], device),
                to_tensor(pack[f"axis_{split}"][idx], device),
                to_tensor(pack[f"ctx_{split}"][idx], device),
                to_tensor(pack[f"edge_{split}"][idx], device),
                to_tensor(pack[f"cand_end_{split}"][idx], device),
            )
            for key in out_parts:
                out_parts[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(vals, axis=0).astype(np.float32) for key, vals in out_parts.items()}


def train_v3_variant(pack: dict[str, np.ndarray], args: argparse.Namespace, *, device: torch.device, variant: str, shuffled_labels: bool = False) -> tuple[SequenceGraphCriticV3, pd.DataFrame, float]:
    model = SequenceGraphCriticV3(
        step_dim=pack["steps_train"].shape[-1],
        axis_dim=pack["axis_train"].shape[-1],
        ctx_dim=pack["ctx_train"].shape[-1],
        edge_dim=pack["edge_train"].shape[-1],
        n_axes=pack["axis_train"].shape[2],
        n_horizons=len(args.horizons),
        max_horizon=args.max_horizon,
        hidden=args.v3_hidden,
        heads=args.v3_heads,
        layers=args.v3_layers,
        dropout=args.v3_dropout,
        correction_scale=args.v3_correction_scale,
    ).to(device)
    target_soft = pack["target_soft_train"]
    target_rank = pack["target_rank_train"]
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 82001)
        perm = rng.permutation(len(target_soft))
        target_soft = target_soft[perm]
        target_rank = target_rank[perm]
    opt = torch.optim.AdamW(model.parameters(), lr=args.v3_lr, weight_decay=args.v3_weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(pack["steps_train"])
    for epoch in range(args.v3_epochs):
        model.train()
        losses = []
        parts_acc: list[dict[str, float]] = []
        for idx in closure.batches(n, args.v3_batch_size, args.seed + 83001 + epoch):
            out = model(
                to_tensor(pack["steps_train"][idx], device),
                to_tensor(pack["axis_train"][idx], device),
                to_tensor(pack["ctx_train"][idx], device),
                to_tensor(pack["edge_train"][idx], device),
                to_tensor(pack["cand_end_train"][idx], device),
            )
            loss, parts = v3_loss(
                out,
                to_tensor(pack["target_end_train"][idx], device),
                to_tensor(target_soft[idx], device),
                to_tensor(target_rank[idx], device),
                args,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            parts_acc.append(parts)
        if epoch == args.v3_epochs - 1 or epoch % max(1, args.v3_epochs // 5) == 0:
            pred_val = predict_v3(model, pack, "val", args, device)["pred"]
            val_rmse = float(np.sqrt(np.mean(np.sum((pred_val - pack["target_end_val"]) ** 2, axis=-1))))
            row = {"variant": variant, "epoch": int(epoch), "train_loss": float(np.mean(losses)), "val_endpoint_rmse": val_rmse}
            if parts_acc:
                for key in parts_acc[0]:
                    row[f"train_{key}"] = float(np.mean([p[key] for p in parts_acc]))
            rows.append(row)
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_val


def build_v3_pack(
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
    use_context: bool = True,
    use_axis: bool = True,
    use_edge: bool = True,
    shuffle_context: bool = False,
    shuffle_edge: bool = False,
    shuffle_axis: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    steps_train, steps_val, steps_test, step_scaler = standardize_candidate_steps(q_train.query_pred, q_val.query_pred, q_test.query_pred)
    axis_train, _ = axis.axis_candidate_features(query_pred=q_train.query_pred, component_pred=component_axes.train, horizons=args.horizons)
    axis_val, _ = axis.axis_candidate_features(query_pred=q_val.query_pred, component_pred=component_axes.val, horizons=args.horizons)
    axis_test, axis_names = axis.axis_candidate_features(query_pred=q_test.query_pred, component_pred=component_axes.test, horizons=args.horizons)
    axis_train, axis_val, axis_test, axis_scaler = axis.standardize_axis_features(axis_train, axis_val, axis_test)
    if (not use_axis) or axis_train.shape[-1] == 0:
        axis_train = np.zeros_like(axis_train)
        axis_val = np.zeros_like(axis_val)
        axis_test = np.zeros_like(axis_test)
    if shuffle_axis:
        axis_train = maybe_shuffle(axis_train, args.seed + 84001, True)
        axis_val = maybe_shuffle(axis_val, args.seed + 84002, True)
        axis_test = maybe_shuffle(axis_test, args.seed + 84003, True)

    edge_train, edge_val, edge_test, edge_meta = select_context_block(arrays, args.v3_edge_block, args.v3_edge_max_features)
    edge_train, edge_val, edge_test, edge_scaler = standardize_2d(edge_train, edge_val, edge_test)
    if not use_edge:
        edge_train = np.zeros_like(edge_train)
        edge_val = np.zeros_like(edge_val)
        edge_test = np.zeros_like(edge_test)
    if shuffle_edge:
        edge_train = maybe_shuffle(edge_train, args.seed + 85001, True)
        edge_val = maybe_shuffle(edge_val, args.seed + 85002, True)
        edge_test = maybe_shuffle(edge_test, args.seed + 85003, True)

    ctx_train_use, ctx_val_use, ctx_test_use = ctx_train, ctx_val, ctx_test
    if not use_context:
        ctx_train_use = np.zeros_like(ctx_train_use)
        ctx_val_use = np.zeros_like(ctx_val_use)
        ctx_test_use = np.zeros_like(ctx_test_use)
    if shuffle_context:
        ctx_train_use = maybe_shuffle(ctx_train_use, args.seed + 86001, True)
        ctx_val_use = maybe_shuffle(ctx_val_use, args.seed + 86002, True)
        ctx_test_use = maybe_shuffle(ctx_test_use, args.seed + 86003, True)

    target_err_train = axis.target_candidate_errors(q_train.query_pred, arrays.residual_train, args.horizons)
    target_err_val = axis.target_candidate_errors(q_val.query_pred, arrays.residual_val, args.horizons)
    target_err_test = axis.target_candidate_errors(q_test.query_pred, arrays.residual_test, args.horizons)
    pack = {
        "steps_train": steps_train,
        "steps_val": steps_val,
        "steps_test": steps_test,
        "axis_train": axis_train,
        "axis_val": axis_val,
        "axis_test": axis_test,
        "ctx_train": ctx_train_use.astype(np.float32),
        "ctx_val": ctx_val_use.astype(np.float32),
        "ctx_test": ctx_test_use.astype(np.float32),
        "edge_train": edge_train,
        "edge_val": edge_val,
        "edge_test": edge_test,
        "cand_end_train": candidate_endpoints(q_train.query_pred, args.horizons),
        "cand_end_val": candidate_endpoints(q_val.query_pred, args.horizons),
        "cand_end_test": candidate_endpoints(q_test.query_pred, args.horizons),
        "target_end_train": residual_endpoints(arrays.residual_train, args.horizons),
        "target_end_val": residual_endpoints(arrays.residual_val, args.horizons),
        "target_end_test": residual_endpoints(arrays.residual_test, args.horizons),
        "target_soft_train": axis.soft_labels_from_distance_np(target_err_train, args.v3_label_temperature),
        "target_soft_val": axis.soft_labels_from_distance_np(target_err_val, args.v3_label_temperature),
        "target_rank_train": -np.log1p(target_err_train).astype(np.float32),
        "target_rank_val": -np.log1p(target_err_val).astype(np.float32),
        "target_err_test": target_err_test,
    }
    meta = {
        "step_scaler": step_scaler,
        "axis_scaler": finite_json(axis_scaler),
        "axis_feature_names": axis_names,
        "edge_meta": finite_json(edge_meta),
        "edge_scaler": finite_json(edge_scaler),
        "use_context": bool(use_context),
        "use_axis": bool(use_axis),
        "use_edge": bool(use_edge),
        "shuffle_context": bool(shuffle_context),
        "shuffle_edge": bool(shuffle_edge),
        "shuffle_axis": bool(shuffle_axis),
    }
    return pack, meta


def add_v3_variant_rows(
    *,
    rows: list[dict[str, Any]],
    logs: list[pd.DataFrame],
    diagnostics: list[dict[str, Any]],
    arrays: audit.SplitArrays,
    q_train: qrc.QueryOutputs,
    q_val: qrc.QueryOutputs,
    q_test: qrc.QueryOutputs,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    component_axes: rpr.ComponentAxisPack,
    args: argparse.Namespace,
    device: torch.device,
    name: str,
    use_context: bool = True,
    use_axis: bool = True,
    use_edge: bool = True,
    shuffle_context: bool = False,
    shuffle_edge: bool = False,
    shuffle_axis: bool = False,
    shuffled_labels: bool = False,
    scalers: dict[str, Any] | None = None,
) -> None:
    pack, meta = build_v3_pack(
        q_train=q_train,
        q_val=q_val,
        q_test=q_test,
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        component_axes=component_axes,
        args=args,
        use_context=use_context,
        use_axis=use_axis,
        use_edge=use_edge,
        shuffle_context=shuffle_context,
        shuffle_edge=shuffle_edge,
        shuffle_axis=shuffle_axis,
    )
    if scalers is not None:
        scalers[name] = finite_json(meta)
    model, log, best_val = train_v3_variant(pack, args, device=device, variant=name, shuffled_labels=shuffled_labels)
    logs.append(log.assign(variant=name))
    pred = predict_v3(model, pack, "test", args, device)
    rows.extend(
        endpoint_rows_from_residual_endpoints(
            arrays=arrays,
            residual_endpoint_pred=pred["pred"],
            horizons=args.horizons,
            label=name,
            extra={"stage": "sequence_graph_v3", "variant": name, "val_endpoint_rmse": best_val},
        )
    )
    rows.extend(
        endpoint_rows_from_residual_endpoints(
            arrays=arrays,
            residual_endpoint_pred=pred["mixture"],
            horizons=args.horizons,
            label=f"{name}_mixture",
            extra={"stage": "sequence_graph_v3_mixture", "variant": name, "val_endpoint_rmse": best_val},
        )
    )
    top_idx = np.argmax(pred["logits"].mean(axis=1), axis=1)
    top_res = q_test.query_pred[np.arange(len(top_idx)), top_idx]
    rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=top_res,
            horizons=args.horizons,
            label=f"{name}_top",
            extra={"stage": "sequence_graph_v3_top", "variant": name, "val_endpoint_rmse": best_val},
        )
    )
    err = qrc.endpoint_errors(q_test.query_pred, arrays.residual_test, args.horizons)
    scores = pred["logits"].mean(axis=1)
    diagnostics.append(
        {
            "variant": name,
            "val_endpoint_rmse": float(best_val),
            "mean_correction_norm": float(np.mean(np.linalg.norm(pred["correction"], axis=-1))),
            "mean_weight_entropy": float(axis.router_entropy_np(pred["weights"])),
            "risk_error_corr": float(qrc.risk_error_corr(-scores, err)),
        }
    )


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
    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v3 requires at least two component axes")
    component_axes.probe.to_csv(args.out_dir / "sequence_graph_v3_component_axis_probe.csv", index=False)

    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand_train = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = rpr.generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    scalers: dict[str, Any] = {"context": finite_json(ctx_scaler), "extra_feature": finite_json(extra_feature_meta)}
    rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=seq.mean_candidate_residual(cand_test),
            horizons=args.horizons,
            label="candidate_mean",
            extra={"stage": "candidate_control"},
        )
    )
    for k in args.oracle_k:
        rows.extend(
            audit.endpoint_metrics(
                steps_true=arrays.steps_test,
                base=arrays.base_test,
                residual_pred=rpr.proto.oracle_from_set(cand_test.residual[:, : int(k)], arrays.residual_test, args.horizons),
                horizons=args.horizons,
                label=f"candidate_endpoint_oracle@{k}",
                extra={"stage": "candidate_endpoint_oracle", "oracle_k": int(k)},
            )
        )

    methods = [s.strip() for s in str(args.prototype_methods).split(",") if s.strip()]
    counts = parse_ints(args.prototype_k)
    for method in methods:
        for m in counts:
            if int(m) > args.candidate_k:
                continue
            ptr, itr = rpr.build_prototype_set(cand_train, method, int(m), seed=args.seed + 87001)
            pva, iva = rpr.build_prototype_set(cand_val, method, int(m), seed=args.seed + 87002)
            pte, ite = rpr.build_prototype_set(cand_test, method, int(m), seed=args.seed + 87003)
            q_train = rpr.make_query_outputs(ptr, itr, method, args.candidate_k, arrays.residual_train, args.horizons)
            q_val = rpr.make_query_outputs(pva, iva, method, args.candidate_k, arrays.residual_val, args.horizons)
            q_test = rpr.make_query_outputs(pte, ite, method, args.candidate_k, arrays.residual_test, args.horizons)
            prefix = f"{method}{int(m)}"
            rows.extend(
                audit.endpoint_metrics(
                    steps_true=arrays.steps_test,
                    base=arrays.base_test,
                    residual_pred=q_test.query_oracle,
                    horizons=args.horizons,
                    label=f"{prefix}_oracle",
                    extra={"stage": "prototype_oracle", "prototype": method, "prototype_k": int(m)},
                )
            )
            rows.extend(
                audit.endpoint_metrics(
                    steps_true=arrays.steps_test,
                    base=arrays.base_test,
                    residual_pred=np.mean(q_test.query_pred, axis=1).astype(np.float32),
                    horizons=args.horizons,
                    label=f"{prefix}_mean",
                    extra={"stage": "prototype_mean", "prototype": method, "prototype_k": int(m)},
                )
            )
            add_v3_variant_rows(
                rows=rows,
                logs=logs,
                diagnostics=diagnostics,
                arrays=arrays,
                q_train=q_train,
                q_val=q_val,
                q_test=q_test,
                ctx_train=ctx_train,
                ctx_val=ctx_val,
                ctx_test=ctx_test,
                component_axes=component_axes,
                args=args,
                device=device,
                name=f"{prefix}_v3_full",
                scalers=scalers,
            )
            if args.include_v3_ablations:
                add_v3_variant_rows(
                    rows=rows,
                    logs=logs,
                    diagnostics=diagnostics,
                    arrays=arrays,
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    component_axes=component_axes,
                    args=args,
                    device=device,
                    name=f"{prefix}_v3_no_edge",
                    use_edge=False,
                    scalers=scalers,
                )
                add_v3_variant_rows(
                    rows=rows,
                    logs=logs,
                    diagnostics=diagnostics,
                    arrays=arrays,
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    component_axes=component_axes,
                    args=args,
                    device=device,
                    name=f"{prefix}_v3_no_axis",
                    use_axis=False,
                    scalers=scalers,
                )
                add_v3_variant_rows(
                    rows=rows,
                    logs=logs,
                    diagnostics=diagnostics,
                    arrays=arrays,
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    component_axes=component_axes,
                    args=args,
                    device=device,
                    name=f"{prefix}_v3_no_context",
                    use_context=False,
                    scalers=scalers,
                )
                if args.include_controls:
                    add_v3_variant_rows(
                        rows=rows,
                        logs=logs,
                        diagnostics=diagnostics,
                        arrays=arrays,
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        args=args,
                        device=device,
                        name=f"{prefix}_v3_shuffled_context",
                        shuffle_context=True,
                        scalers=scalers,
                    )
                    add_v3_variant_rows(
                        rows=rows,
                        logs=logs,
                        diagnostics=diagnostics,
                        arrays=arrays,
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        args=args,
                        device=device,
                        name=f"{prefix}_v3_shuffled_edge",
                        shuffle_edge=True,
                        scalers=scalers,
                    )
                    add_v3_variant_rows(
                        rows=rows,
                        logs=logs,
                        diagnostics=diagnostics,
                        arrays=arrays,
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        component_axes=component_axes,
                        args=args,
                        device=device,
                        name=f"{prefix}_v3_shuffled_labels",
                        shuffled_labels=True,
                        scalers=scalers,
                    )

    summary = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    if not summary.empty:
        summary.insert(0, "seed", int(args.seed))
        summary.insert(0, "dataset", str(args.dataset))
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "sequence_graph_critic_v3_summary.csv", index=False)
    diag.to_csv(args.out_dir / "sequence_graph_critic_v3_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "sequence_graph_critic_v3_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(scalers), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = ["# Sequence/Graph Critic-Refiner v3 Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- prototype_methods: `{args.prototype_methods}`")
    lines.append(f"- prototype_k: `{args.prototype_k}`")
    lines.append(f"- component_axis_blocks: `{args.component_axis_blocks}`")
    lines.append(f"- v3_edge_block: `{args.v3_edge_block}`")
    lines.append(f"- include_v3_ablations: `{args.include_v3_ablations}`")
    lines.append(f"- include_controls: `{args.include_controls}`")
    if getattr(args, "extra_feature_grid", None):
        lines.append(f"- extra_feature_grid: `{args.extra_feature_grid}`")
        lines.append(f"- extra_feature_block_name: `{args.extra_feature_block_name}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{int(h)}")
        sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
        for _, row in sub.head(32).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, stage={row.get('stage', '')}")
    if not diag.empty:
        lines.append("\n## Diagnostics")
        for _, row in diag.sort_values("val_endpoint_rmse").head(32).iterrows():
            lines.append(
                f"- `{row['variant']}`: val={row['val_endpoint_rmse']:.3f}, "
                f"corr={row['mean_correction_norm']:.3f}, entropy={row['mean_weight_entropy']:.3f}, "
                f"risk_corr={row['risk_error_corr']:.3f}"
            )
    lines.append("\n## Decision Notes")
    lines.append("- Pass only if full beats no_edge/no_axis/no_context and shuffled controls.")
    lines.append("- If full is near shuffled labels, the model is still exploiting candidate geometry rather than causal route information.")
    lines.append("- If oracle is strong but full stays far above oracle, the missing piece is observability or a stronger graph/video sequence encoder.")
    (out_dir / "sequence_graph_critic_v3_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.add_argument("--prototype-methods", type=str, default="fps_shape")
    parser.add_argument("--prototype-k", type=str, default="16")
    parser.add_argument("--include-v3-ablations", action="store_true")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--component-axis-blocks", type=str, default="self,flow,explicit_edge,all_context")
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
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="ef_")
    parser.add_argument("--extra-feature-block-name", type=str, default="explicit_edge")
    parser.add_argument("--extra-feature-max-cols", type=int, default=128)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")

    parser.add_argument("--v3-hidden", type=int, default=192)
    parser.add_argument("--v3-heads", type=int, default=4)
    parser.add_argument("--v3-layers", type=int, default=2)
    parser.add_argument("--v3-dropout", type=float, default=0.05)
    parser.add_argument("--v3-epochs", type=int, default=18)
    parser.add_argument("--v3-batch-size", type=int, default=384)
    parser.add_argument("--v3-lr", type=float, default=7e-4)
    parser.add_argument("--v3-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v3-correction-scale", type=float, default=0.75)
    parser.add_argument("--v3-label-temperature", type=float, default=8.0)
    parser.add_argument("--v3-reg-weight", type=float, default=1.0)
    parser.add_argument("--v3-listwise-weight", type=float, default=0.35)
    parser.add_argument("--v3-rank-weight", type=float, default=0.10)
    parser.add_argument("--v3-nll-weight", type=float, default=0.03)
    parser.add_argument("--v3-correction-l2-weight", type=float, default=0.005)
    parser.add_argument("--v3-entropy-weight", type=float, default=0.003)
    parser.add_argument("--v3-logvar-min", type=float, default=-6.0)
    parser.add_argument("--v3-logvar-max", type=float, default=6.0)
    parser.add_argument("--v3-edge-block", type=str, default="explicit_edge")
    parser.add_argument("--v3-edge-max-features", type=int, default=128)
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 3000)
        args.max_val_rows = min(args.max_val_rows, 1000)
        args.max_test_rows = min(args.max_test_rows, 1200)
        args.posterior_epochs = min(args.posterior_epochs, 5)
        args.student_epochs = min(args.student_epochs, 5)
        args.learned_route_epochs = min(args.learned_route_epochs, 4)
        args.v3_epochs = min(args.v3_epochs, 5)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.max_all_features = min(args.max_all_features, 192)
        args.max_critic_context_features = min(args.max_critic_context_features, 192)
    run(args)


if __name__ == "__main__":
    main()
