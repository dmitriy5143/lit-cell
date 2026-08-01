#!/usr/bin/env python3
"""RoutePrototypeRefiner v1 for LaChance residual candidates.

This runner follows the new bottleneck diagnosis:

    candidate cloud has strong oracle;
    current RouteQueryRefiner loses oracle during compression;
    deterministic FPS/endpoint/shape prototypes preserve oracle.

So we build route prototypes first, without target access, and only then train a
causal sequence-risk/refiner over those prototypes.
"""

from __future__ import annotations

import argparse
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
import run_lachance_route_prototype_oracle as proto  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_prototype_refiner_2026-06-27"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class ComponentAxisPack:
    train: np.ndarray  # n,c,h,2 residual-step predictions
    val: np.ndarray
    test: np.ndarray
    names: list[str]
    probe: pd.DataFrame


class ComponentAxisMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def endpoint_rows(arrays: audit.SplitArrays, pred: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def build_prototype_indices(cand: seq.CandidatePack, method: str, m: int, *, seed: int) -> np.ndarray:
    n, k = cand.residual.shape[:2]
    m = min(int(m), k)
    if method == "first":
        return np.arange(m, dtype=np.int64)[None, :].repeat(n, axis=0)
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.stack([rng.choice(k, size=m, replace=False) for _ in range(n)], axis=0).astype(np.int64)
    if method.startswith("fps_"):
        mode = method.replace("fps_", "", 1)
        x = proto.flatten_candidate(cand.residual, mode)
        return proto.fps_indices(x, m)
    raise ValueError(f"Unknown prototype method: {method}")


def build_prototype_set(cand: seq.CandidatePack, method: str, m: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = build_prototype_indices(cand, method, m, seed=seed)
    return proto.gather_candidates(cand.residual, idx), idx


def route_logits_from_indices(idx: np.ndarray, k_total: int, method: str) -> np.ndarray:
    # Earlier prototypes are more central for FPS because the first point is the
    # candidate-cloud medoid.  Give the risk model this weak rank prior, but keep
    # it mild so it cannot dominate sequence features.
    rank = np.arange(idx.shape[1], dtype=np.float32)[None, :].repeat(idx.shape[0], axis=0)
    rank = rank / max(float(idx.shape[1] - 1), 1.0)
    coverage = 1.0 - np.asarray(idx, dtype=np.float32) / max(float(k_total - 1), 1.0)
    if method == "random":
        return np.zeros_like(rank, dtype=np.float32)
    return (0.35 * coverage - 0.25 * rank).astype(np.float32)


def _safe_residual_from_flat(y: np.ndarray, max_horizon: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return audit.unflatten_residual(y, max_horizon).astype(np.float32)


def _component_endpoint_rmse(pred: np.ndarray, true: np.ndarray, h: int) -> float:
    p = np.sum(pred[:, : int(h), :], axis=1)
    t = np.sum(true[:, : int(h), :], axis=1)
    return float(np.sqrt(np.mean(np.sum((p - t) ** 2, axis=-1))))


def _fit_ridge_component_axis(
    *,
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    if qrc.Ridge is None or x_train.shape[1] == 0:
        return None
    xtr = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xva = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xte = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if xtr.shape[1] > args.component_axis_max_features:
        var = np.var(xtr, axis=0)
        keep = np.argsort(var)[-args.component_axis_max_features :]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
    model = qrc.Ridge(alpha=args.component_ridge_alpha, random_state=args.seed)
    model.fit(xtr, audit.flatten_residual(arrays.residual_train))
    pred_train = _safe_residual_from_flat(model.predict(xtr), args.max_horizon)
    pred_val = _safe_residual_from_flat(model.predict(xva), args.max_horizon)
    pred_test = _safe_residual_from_flat(model.predict(xte), args.max_horizon)
    rows: list[dict[str, Any]] = []
    for h in args.horizons:
        rows.append(
            {
                "component": name,
                "source": "ridge_axis",
                "horizon": int(h),
                "test_endpoint_rmse": _component_endpoint_rmse(pred_test, arrays.residual_test, int(h)),
                "n_features": int(xtr.shape[1]),
            }
        )
    return pred_train, pred_val, pred_test, rows


def _fit_mlp_component_axis(
    *,
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    if x_train.shape[1] == 0:
        return None
    xtr = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xva = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xte = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if xtr.shape[1] > args.component_axis_max_features:
        var = np.var(xtr, axis=0)
        keep = np.argsort(var)[-args.component_axis_max_features :]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]

    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    y_mean = np.mean(ytr, axis=0, keepdims=True)
    y_std = np.std(ytr, axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    ytr_s = ((ytr - y_mean) / y_std).astype(np.float32)
    yva_s = ((yva - y_mean) / y_std).astype(np.float32)

    model = ComponentAxisMLP(
        xtr.shape[1],
        ytr.shape[1],
        args.component_axis_hidden,
        args.component_axis_dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.component_axis_lr,
        weight_decay=args.component_axis_weight_decay,
    )
    best_state = None
    best_val = float("inf")
    for epoch in range(args.component_axis_epochs):
        model.train()
        for idx in closure.batches(len(xtr), args.batch_size, args.seed + 33000 + epoch):
            xb = torch.as_tensor(xtr[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(ytr_s[idx], dtype=torch.float32, device=device)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for idx in closure.batches(len(xva), args.batch_size, args.seed + 33100, shuffle=False):
                pred = model(torch.as_tensor(xva[idx], dtype=torch.float32, device=device))
                yb = torch.as_tensor(yva_s[idx], dtype=torch.float32, device=device)
                vals.append(float(F.smooth_l1_loss(pred, yb).detach().cpu()))
        val_loss = float(np.mean(vals)) if vals else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for idx in closure.batches(len(x), args.batch_size, args.seed + 33200, shuffle=False):
                pred = model(torch.as_tensor(x[idx], dtype=torch.float32, device=device)).cpu().numpy()
                out.append(pred)
        ys = np.concatenate(out, axis=0).astype(np.float32)
        y = ys * y_std + y_mean
        return audit.unflatten_residual(y.astype(np.float32), args.max_horizon).astype(np.float32)

    pred_train = predict(xtr)
    pred_val = predict(xva)
    pred_test = predict(xte)
    rows: list[dict[str, Any]] = []
    for h in args.horizons:
        rows.append(
            {
                "component": name,
                "source": "mlp_axis",
                "horizon": int(h),
                "test_endpoint_rmse": _component_endpoint_rmse(pred_test, arrays.residual_test, int(h)),
                "n_features": int(xtr.shape[1]),
                "val_loss": float(best_val),
            }
        )
    return pred_train, pred_val, pred_test, rows


def _fit_component_axis(
    *,
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    if args.component_axis_model == "mlp":
        return _fit_mlp_component_axis(
            name=name,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            arrays=arrays,
            args=args,
            device=device,
        )
    return _fit_ridge_component_axis(
        name=name,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        arrays=arrays,
        args=args,
    )


def build_component_axes(
    args: argparse.Namespace,
    arrays: audit.SplitArrays,
    posterior: closure.PosteriorPack,
    student,
    blocks: list[str],
    device,
) -> ComponentAxisPack | None:
    if not args.component_aware_risk:
        return None
    preds_train: list[np.ndarray] = []
    preds_val: list[np.ndarray] = []
    preds_test: list[np.ndarray] = []
    names: list[str] = []
    probe_rows: list[dict[str, Any]] = []

    for block in parse_strs(args.component_axis_blocks):
        if block not in arrays.x_train:
            continue
        fitted = _fit_component_axis(
            name=block,
            x_train=arrays.x_train[block],
            x_val=arrays.x_val[block],
            x_test=arrays.x_test[block],
            arrays=arrays,
            args=args,
            device=device,
        )
        if fitted is None:
            continue
        ptr, pva, pte, rows = fitted
        preds_train.append(ptr)
        preds_val.append(pva)
        preds_test.append(pte)
        names.append(block)
        probe_rows.extend(rows)

    if args.component_include_student_axis:
        pred_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
        pred_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
        pred_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
        res_train = closure.decode_residual(
            posterior.model,
            posterior.scaler,
            pred_train["mu"],
            max_horizon=args.max_horizon,
            device=device,
            batch_size=args.batch_size,
        )
        res_val = closure.decode_residual(
            posterior.model,
            posterior.scaler,
            pred_val["mu"],
            max_horizon=args.max_horizon,
            device=device,
            batch_size=args.batch_size,
        )
        res_test = closure.decode_residual(
            posterior.model,
            posterior.scaler,
            pred_test["mu"],
            max_horizon=args.max_horizon,
            device=device,
            batch_size=args.batch_size,
        )
        preds_train.append(res_train.astype(np.float32))
        preds_val.append(res_val.astype(np.float32))
        preds_test.append(res_test.astype(np.float32))
        names.append("decomposition_student")
        for h in args.horizons:
            probe_rows.append(
                {
                    "component": "decomposition_student",
                    "source": "student_axis",
                    "horizon": int(h),
                    "test_endpoint_rmse": _component_endpoint_rmse(res_test, arrays.residual_test, int(h)),
                    "n_features": int(seq.decomposition_context_features(pred_train, mode_k=args.mode_k).shape[1]),
                }
            )

    if len(preds_train) < 2:
        return None
    train = np.stack(preds_train, axis=1).astype(np.float32)
    val = np.stack(preds_val, axis=1).astype(np.float32)
    test = np.stack(preds_test, axis=1).astype(np.float32)
    return ComponentAxisPack(train=train, val=val, test=test, names=names, probe=pd.DataFrame(probe_rows))


def attach_extra_feature_block(
    arrays: audit.SplitArrays,
    split: audit.seq.SplitData,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Attach an external causal feature grid as a separate component block.

    This is intentionally narrow: it lets us test whether an already-built
    topology/video/edge feature table can act as a component axis without
    rewriting the core decomposition or candidate generator.
    """
    if not getattr(args, "extra_feature_grid", None):
        return {"attached": False}
    grid_path = Path(args.extra_feature_grid)
    if not grid_path.exists():
        raise FileNotFoundError(f"Extra feature grid not found: {grid_path}")
    grid = pd.read_csv(grid_path)
    key_cols = [c for c in ["dataset", "sequence", "frame", "track_id"] if c in grid.columns and c in split.train.columns]
    if len(key_cols) < 4:
        raise RuntimeError(f"Extra feature grid needs dataset/sequence/frame/track_id keys, got {key_cols}")
    prefixes = [p.strip() for p in str(args.extra_feature_prefixes).split(",") if p.strip()]
    if prefixes:
        cols = [c for c in grid.columns if any(c.startswith(p) for p in prefixes)]
    else:
        cols = [c for c in grid.columns if c not in key_cols]
    cols = [c for c in cols if c not in key_cols]
    if not cols:
        raise RuntimeError(f"No extra feature columns selected from {grid_path}")
    grid = grid[key_cols + cols].drop_duplicates(key_cols)

    def merge_matrix(df: pd.DataFrame) -> np.ndarray:
        merged = df[key_cols].merge(grid, on=key_cols, how="left")
        return audit.safe_matrix(merged, cols)

    xtr_raw = merge_matrix(split.train)
    xva_raw = merge_matrix(split.val)
    xte_raw = merge_matrix(split.test)
    if xtr_raw.shape[1] > int(args.extra_feature_max_cols):
        var = np.nan_to_num(np.var(xtr_raw, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-int(args.extra_feature_max_cols) :]
        cols = [cols[int(i)] for i in keep]
        xtr_raw, xva_raw, xte_raw = xtr_raw[:, keep], xva_raw[:, keep], xte_raw[:, keep]
    xtr, xva, xte, _ = audit.standardize_block(xtr_raw, xva_raw, xte_raw)
    block = str(args.extra_feature_block_name)
    arrays.x_train[block] = xtr.astype(np.float32)
    arrays.x_val[block] = xva.astype(np.float32)
    arrays.x_test[block] = xte.astype(np.float32)
    arrays.feature_names[block] = cols
    if parse_bool(args.extra_feature_merge_all_context):
        arrays.x_train["all_context"] = np.concatenate([arrays.x_train["all_context"], xtr], axis=1).astype(np.float32)
        arrays.x_val["all_context"] = np.concatenate([arrays.x_val["all_context"], xva], axis=1).astype(np.float32)
        arrays.x_test["all_context"] = np.concatenate([arrays.x_test["all_context"], xte], axis=1).astype(np.float32)
        arrays.feature_names["all_context"] = list(arrays.feature_names.get("all_context", [])) + [f"{block}::{c}" for c in cols]
    return {
        "attached": True,
        "path": str(grid_path),
        "block": block,
        "n_cols": int(len(cols)),
        "merge_all_context": parse_bool(args.extra_feature_merge_all_context),
        "prefixes": prefixes,
    }


def component_route_features(
    *,
    query_pred: np.ndarray,
    component_pred: np.ndarray,
    horizons: list[int],
    temperature: float,
) -> tuple[np.ndarray, list[str]]:
    """Candidate/prototype agreement with learned component axes.

    Each component axis is a causal residual predictor.  These features ask:
    which component explains this route, how strongly, and how ambiguous is the
    component assignment?
    """
    n, q, hmax, _ = query_pred.shape
    c = component_pred.shape[1]
    q_steps = query_pred[:, :, None, :, :]  # n,q,1,h,2
    comp_steps = component_pred[:, None, :, :, :]  # n,1,c,h,2
    step_delta = q_steps - comp_steps
    step_mse = np.mean(np.sum(step_delta * step_delta, axis=-1), axis=-1)  # n,q,c

    q_flat = query_pred.reshape(n, q, -1)
    comp_flat = component_pred.reshape(n, c, -1)
    dot = np.sum(q_flat[:, :, None, :] * comp_flat[:, None, :, :], axis=-1)
    q_norm = np.linalg.norm(q_flat, axis=-1)[:, :, None]
    c_norm = np.linalg.norm(comp_flat, axis=-1)[:, None, :]
    traj_cos = dot / np.maximum(q_norm * c_norm, EPS)
    traj_dist = np.linalg.norm(q_flat[:, :, None, :] - comp_flat[:, None, :, :], axis=-1)
    mag_ratio = q_norm / np.maximum(c_norm, EPS)

    endpoint_dists: list[np.ndarray] = []
    endpoint_cos: list[np.ndarray] = []
    endpoint_ratio: list[np.ndarray] = []
    endpoint_names: list[str] = []
    for h in horizons:
        qh = np.sum(query_pred[:, :, : int(h), :], axis=2)
        ch = np.sum(component_pred[:, :, : int(h), :], axis=2)
        diff = qh[:, :, None, :] - ch[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        cdot = np.sum(qh[:, :, None, :] * ch[:, None, :, :], axis=-1)
        qmag = np.linalg.norm(qh, axis=-1)[:, :, None]
        cmag = np.linalg.norm(ch, axis=-1)[:, None, :]
        endpoint_dists.append(dist)
        endpoint_cos.append(cdot / np.maximum(qmag * cmag, EPS))
        endpoint_ratio.append(qmag / np.maximum(cmag, EPS))
        endpoint_names.append(f"h{int(h)}")

    ep_dist = np.stack(endpoint_dists, axis=-1)
    ep_cos = np.stack(endpoint_cos, axis=-1)
    ep_ratio = np.stack(endpoint_ratio, axis=-1)

    dist_score = -np.mean(ep_dist, axis=-1) / max(float(temperature), 1e-4)
    cos_score = np.mean(ep_cos, axis=-1)
    attn = qrc.softmax_np(dist_score + cos_score, axis=2)
    entropy = -np.sum(attn * np.log(np.maximum(attn, EPS)), axis=2, keepdims=True)
    top_weight = np.max(attn, axis=2, keepdims=True)
    weighted_steps = np.sum(attn[:, :, :, None, None] * component_pred[:, None, :, :, :], axis=2)
    weighted_delta = query_pred - weighted_steps
    weighted_mse = np.mean(np.sum(weighted_delta * weighted_delta, axis=-1), axis=-1, keepdims=True)
    weighted_flat = weighted_steps.reshape(n, q, -1)
    weighted_cos = np.sum(q_flat * weighted_flat, axis=-1, keepdims=True) / np.maximum(
        np.linalg.norm(q_flat, axis=-1, keepdims=True) * np.linalg.norm(weighted_flat, axis=-1, keepdims=True),
        EPS,
    )

    comp_spread = np.std(comp_flat, axis=1)
    spread_norm = np.linalg.norm(comp_spread, axis=-1, keepdims=True)
    spread_rep = spread_norm[:, None, :].repeat(q, axis=1)

    pieces: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        a = arr.reshape(n, q, -1).astype(np.float32)
        pieces.append(a)
        for j in range(a.shape[-1]):
            names.append(f"{name}_{j}" if a.shape[-1] > 1 else name)

    add("comp_step_mse", step_mse)
    add("comp_traj_cos", traj_cos)
    add("comp_traj_dist", traj_dist)
    add("comp_mag_ratio", np.clip(mag_ratio, 0.0, 8.0))
    add("comp_endpoint_dist", ep_dist)
    add("comp_endpoint_cos", ep_cos)
    add("comp_endpoint_mag_ratio", np.clip(ep_ratio, 0.0, 8.0))
    add("comp_attn", attn)
    add("comp_attn_entropy", entropy)
    add("comp_attn_top_weight", top_weight)
    add("comp_weighted_mse", weighted_mse)
    add("comp_weighted_cos", weighted_cos)
    add("component_spread_norm", spread_rep)
    feat = np.concatenate(pieces, axis=-1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names


def build_refiner_features(
    *,
    q: qrc.QueryOutputs,
    base: np.ndarray,
    ctx: np.ndarray,
    args: argparse.Namespace,
    include_context: bool,
    component_pred: np.ndarray | None = None,
    include_components: bool = False,
) -> tuple[np.ndarray, list[str]]:
    feat, names = qrc.query_sequence_features(
        query_pred=q.query_pred,
        base=base,
        route_logits=q.route_logits,
        ctx=ctx,
        horizons=args.horizons,
        include_context=include_context,
        include_query_id=args.risk_include_query_id,
    )
    if include_components and component_pred is not None:
        cfeat, cnames = component_route_features(
            query_pred=q.query_pred,
            component_pred=component_pred,
            horizons=args.horizons,
            temperature=args.component_attention_temperature,
        )
        feat = np.concatenate([feat, cfeat], axis=-1).astype(np.float32)
        names = names + [f"component::{n}" for n in cnames]
    return feat, names


def make_query_outputs(prototypes: np.ndarray, idx: np.ndarray, method: str, k_total: int, true: np.ndarray, horizons: list[int]) -> qrc.QueryOutputs:
    logits = route_logits_from_indices(idx, k_total, method)
    probs = qrc.softmax_np(logits, axis=1)
    return qrc.QueryOutputs(
        query_pred=prototypes.astype(np.float32),
        route_logits=logits.astype(np.float32),
        route_probs=probs.astype(np.float32),
        weighted_pred=qrc.weighted_residual(prototypes, probs),
        top_pred=qrc.top_query_residual(prototypes, logits),
        query_oracle=qrc.query_oracle_residual(prototypes, true, horizons),
    )


def prepare_context(args, arrays, posterior, student, blocks, device):
    ctx_blocks = closure.variant_blocks(args.critic_context_variant, arrays.x_train)
    ctx_train_raw = seq.flatten_blocks(arrays.x_train, ctx_blocks)
    ctx_val_raw = seq.flatten_blocks(arrays.x_val, ctx_blocks)
    ctx_test_raw = seq.flatten_blocks(arrays.x_test, ctx_blocks)
    if args.add_decomposition_context:
        pred_ctx_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
        pred_ctx_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
        pred_ctx_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
        ctx_train_raw = np.concatenate([ctx_train_raw, seq.decomposition_context_features(pred_ctx_train, mode_k=args.mode_k)], axis=1)
        ctx_val_raw = np.concatenate([ctx_val_raw, seq.decomposition_context_features(pred_ctx_val, mode_k=args.mode_k)], axis=1)
        ctx_test_raw = np.concatenate([ctx_test_raw, seq.decomposition_context_features(pred_ctx_test, mode_k=args.mode_k)], axis=1)
    if ctx_train_raw.shape[1] > args.max_critic_context_features:
        var = np.var(ctx_train_raw, axis=0)
        keep = np.argsort(var)[-args.max_critic_context_features :]
        ctx_train_raw = ctx_train_raw[:, keep]
        ctx_val_raw = ctx_val_raw[:, keep]
        ctx_test_raw = ctx_test_raw[:, keep]
    return seq.standardize(ctx_train_raw, ctx_val_raw, ctx_test_raw)


def generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx, split_name: str, device) -> seq.CandidatePack:
    if args.candidate_generator == "learned_route":
        return seq.generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx, args, split_name=split_name, device=device)
    if args.candidate_generator == "hybrid":
        return seq.generate_hybrid_candidates(arrays, posterior, student, blocks, route_model, route_ctx, args, split_name=split_name, device=device)
    return seq.generate_candidates(arrays, posterior, student, blocks, args, split_name=split_name, device=device)


def train_route_model_if_needed(args, arrays, posterior, student, blocks, device):
    hybrid_budgets = seq.resolve_hybrid_budgets(args) if args.candidate_generator == "hybrid" else {"generic": 0, "route": 0, "learned": 0}
    needs_learned_route = args.candidate_generator == "learned_route" or (
        args.candidate_generator == "hybrid" and hybrid_budgets.get("learned", 0) > 0
    )
    if not needs_learned_route:
        return None, None, None, None, pd.DataFrame()
    route_blocks = closure.variant_blocks(args.learned_route_context_variant, arrays.x_train)
    route_ctx_train_raw = seq.flatten_blocks(arrays.x_train, route_blocks)
    route_ctx_val_raw = seq.flatten_blocks(arrays.x_val, route_blocks)
    route_ctx_test_raw = seq.flatten_blocks(arrays.x_test, route_blocks)
    if args.learned_route_add_decomposition_context:
        pred_route_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
        pred_route_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
        pred_route_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
        route_ctx_train_raw = np.concatenate([route_ctx_train_raw, seq.decomposition_context_features(pred_route_train, mode_k=args.mode_k)], axis=1)
        route_ctx_val_raw = np.concatenate([route_ctx_val_raw, seq.decomposition_context_features(pred_route_val, mode_k=args.mode_k)], axis=1)
        route_ctx_test_raw = np.concatenate([route_ctx_test_raw, seq.decomposition_context_features(pred_route_test, mode_k=args.mode_k)], axis=1)
    if route_ctx_train_raw.shape[1] > args.max_learned_route_context_features:
        var = np.var(route_ctx_train_raw, axis=0)
        keep = np.argsort(var)[-args.max_learned_route_context_features :]
        route_ctx_train_raw = route_ctx_train_raw[:, keep]
        route_ctx_val_raw = route_ctx_val_raw[:, keep]
        route_ctx_test_raw = route_ctx_test_raw[:, keep]
    route_ctx_train, route_ctx_val, route_ctx_test, _ = seq.standardize(route_ctx_train_raw, route_ctx_val_raw, route_ctx_test_raw)
    route_model, route_log = seq.train_learned_route_generator(
        route_ctx_train,
        route_ctx_val,
        arrays.residual_train,
        arrays.residual_val,
        posterior.mode_soft_train,
        posterior.mode_soft_val,
        args,
        device=device,
    )
    return route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log


def add_refiner_rows(
    *,
    rows: list[dict[str, Any]],
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    name: str,
    q_train: qrc.QueryOutputs,
    q_val: qrc.QueryOutputs,
    q_test: qrc.QueryOutputs,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    device,
    include_context: bool,
    shuffled_labels: bool,
    shuffled_context: bool,
    risk_logs: list[pd.DataFrame],
    scaler_meta: dict[str, Any],
    component_train: np.ndarray | None = None,
    component_val: np.ndarray | None = None,
    component_test: np.ndarray | None = None,
    include_components: bool = False,
    shuffled_components: bool = False,
) -> None:
    ctx_train_use, ctx_val_use, ctx_test_use = ctx_train, ctx_val, ctx_test
    if shuffled_context:
        rng = np.random.default_rng(args.seed + 70401)
        ctx_train_use = ctx_train_use[rng.permutation(len(ctx_train_use))]
        ctx_val_use = ctx_val_use[rng.permutation(len(ctx_val_use))]
        ctx_test_use = ctx_test_use[rng.permutation(len(ctx_test_use))]
    comp_train_use, comp_val_use, comp_test_use = component_train, component_val, component_test
    if shuffled_components and component_train is not None and component_val is not None and component_test is not None:
        rng = np.random.default_rng(args.seed + 71401)
        comp_train_use = component_train[rng.permutation(len(component_train))]
        comp_val_use = component_val[rng.permutation(len(component_val))]
        comp_test_use = component_test[rng.permutation(len(component_test))]
    feat_train, _ = build_refiner_features(
        q=q_train,
        base=arrays.base_train,
        ctx=ctx_train_use,
        args=args,
        include_context=include_context,
        component_pred=comp_train_use,
        include_components=include_components,
    )
    feat_val, _ = build_refiner_features(
        q=q_val,
        base=arrays.base_val,
        ctx=ctx_val_use,
        args=args,
        include_context=include_context,
        component_pred=comp_val_use,
        include_components=include_components,
    )
    feat_test, _ = build_refiner_features(
        q=q_test,
        base=arrays.base_test,
        ctx=ctx_test_use,
        args=args,
        include_context=include_context,
        component_pred=comp_test_use,
        include_components=include_components,
    )
    feat_train, feat_val, feat_test, scaler = qrc.standardize_query_features(feat_train, feat_val, feat_test)
    scaler_meta[name] = finite_json(scaler)
    model, log, best_temp = qrc.train_risk_model(
        feat_train,
        feat_val,
        q_train,
        q_val,
        arrays.residual_train,
        arrays.residual_val,
        args,
        device=device,
        shuffled_labels=shuffled_labels,
    )
    risk_logs.append(log.assign(variant=name))
    risk = qrc.predict_risk(model, feat_test, args, device=device)
    pred_w = qrc.weighted_residual(q_test.query_pred, qrc.softmax_np(-risk / best_temp, axis=1))
    pred_t = q_test.query_pred[np.arange(len(risk)), np.argmin(risk, axis=1)]
    err = qrc.endpoint_errors(q_test.query_pred, arrays.residual_test, args.horizons)
    corr = qrc.risk_error_corr(risk, err)
    rows.extend(endpoint_rows(arrays, pred_w, f"{name}_weighted", args, {"stage": "prototype_refiner", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr}))
    rows.extend(endpoint_rows(arrays, pred_t, f"{name}_top", args, {"stage": "prototype_refiner_top", "risk_variant": name, "temperature": best_temp, "risk_error_corr": corr}))


def add_sklearn_rows(
    *,
    rows: list[dict[str, Any]],
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    name: str,
    kind: str,
    q_train: qrc.QueryOutputs,
    q_val: qrc.QueryOutputs,
    q_test: qrc.QueryOutputs,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    include_context: bool,
    component_train: np.ndarray | None = None,
    component_val: np.ndarray | None = None,
    component_test: np.ndarray | None = None,
    include_components: bool = False,
    shuffled_components: bool = False,
) -> None:
    comp_train_use, comp_val_use, comp_test_use = component_train, component_val, component_test
    if shuffled_components and component_train is not None and component_val is not None and component_test is not None:
        rng = np.random.default_rng(args.seed + 72401)
        comp_train_use = component_train[rng.permutation(len(component_train))]
        comp_val_use = component_val[rng.permutation(len(component_val))]
        comp_test_use = component_test[rng.permutation(len(component_test))]
    feat_train, _ = build_refiner_features(
        q=q_train,
        base=arrays.base_train,
        ctx=ctx_train,
        args=args,
        include_context=include_context,
        component_pred=comp_train_use,
        include_components=include_components,
    )
    feat_val, _ = build_refiner_features(
        q=q_val,
        base=arrays.base_val,
        ctx=ctx_val,
        args=args,
        include_context=include_context,
        component_pred=comp_val_use,
        include_components=include_components,
    )
    feat_test, _ = build_refiner_features(
        q=q_test,
        base=arrays.base_test,
        ctx=ctx_test,
        args=args,
        include_context=include_context,
        component_pred=comp_test_use,
        include_components=include_components,
    )
    feat_train, feat_val, feat_test, _ = qrc.standardize_query_features(feat_train, feat_val, feat_test)
    risk, temp, diag = qrc.fit_predict_sklearn_risk(
        model_kind=kind,
        feat_train=feat_train,
        feat_val=feat_val,
        feat_test=feat_test,
        q_train=q_train,
        q_val=q_val,
        q_test=q_test,
        residual_train=arrays.residual_train,
        residual_val=arrays.residual_val,
        residual_test=arrays.residual_test,
        args=args,
    )
    pred_w = qrc.weighted_residual(q_test.query_pred, qrc.softmax_np(-risk / temp, axis=1))
    pred_t = q_test.query_pred[np.arange(len(risk)), np.argmin(risk, axis=1)]
    corr = float(diag.get("test_risk_error_corr", qrc.risk_error_corr(risk, qrc.endpoint_errors(q_test.query_pred, arrays.residual_test, args.horizons))))
    rows.extend(endpoint_rows(arrays, pred_w, f"{name}_weighted", args, {"stage": "prototype_refiner_sklearn", "risk_variant": name, "temperature": temp, "risk_error_corr": corr, **diag}))
    rows.extend(endpoint_rows(arrays, pred_t, f"{name}_top", args, {"stage": "prototype_refiner_sklearn_top", "risk_variant": name, "temperature": temp, "risk_error_corr": corr, **diag}))


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    extra_feature_meta = attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_scaler = prepare_context(args, arrays, posterior, student, blocks, device)
    component_axes = build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is not None:
        component_axes.probe.to_csv(args.out_dir / "route_component_axis_probe.csv", index=False)
    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = train_route_model_if_needed(args, arrays, posterior, student, blocks, device)

    cand_train = generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    cand_val = generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    cand_test = generate_candidates_for_split(args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    rows: list[dict[str, Any]] = []
    rows.extend(endpoint_rows(arrays, seq.mean_candidate_residual(cand_test), "candidate_mean", args, {"stage": "candidate_control"}))
    for k in args.oracle_k:
        rows.extend(endpoint_rows(arrays, proto.oracle_from_set(cand_test.residual[:, : int(k)], arrays.residual_test, args.horizons), f"candidate_endpoint_oracle@{k}", args, {"stage": "candidate_endpoint_oracle", "oracle_k": int(k)}))

    methods = [s.strip() for s in args.prototype_methods.split(",") if s.strip()]
    counts = parse_ints(args.prototype_k)
    risk_logs: list[pd.DataFrame] = []
    scaler_meta: dict[str, Any] = {"context": finite_json(ctx_scaler)}
    for method in methods:
        for m in counts:
            if m > args.candidate_k:
                continue
            ptr, itr = build_prototype_set(cand_train, method, m, seed=args.seed + 8001)
            pva, iva = build_prototype_set(cand_val, method, m, seed=args.seed + 9001)
            pte, ite = build_prototype_set(cand_test, method, m, seed=args.seed + 10001)
            q_train = make_query_outputs(ptr, itr, method, args.candidate_k, arrays.residual_train, args.horizons)
            q_val = make_query_outputs(pva, iva, method, args.candidate_k, arrays.residual_val, args.horizons)
            q_test = make_query_outputs(pte, ite, method, args.candidate_k, arrays.residual_test, args.horizons)
            prefix = f"{method}{m}"
            rows.extend(endpoint_rows(arrays, q_test.query_oracle, f"{prefix}_oracle", args, {"stage": "prototype_oracle", "prototype": method, "prototype_k": m}))
            rows.extend(endpoint_rows(arrays, q_test.weighted_pred, f"{prefix}_prior_weighted", args, {"stage": "prototype_prior", "prototype": method, "prototype_k": m}))
            rows.extend(endpoint_rows(arrays, np.mean(q_test.query_pred, axis=1).astype(np.float32), f"{prefix}_mean", args, {"stage": "prototype_mean", "prototype": method, "prototype_k": m}))

            add_refiner_rows(
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
            if component_axes is not None:
                add_refiner_rows(
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
                add_refiner_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_component_risk_no_context",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    device=device,
                    include_context=False,
                    shuffled_labels=False,
                    shuffled_context=False,
                    risk_logs=risk_logs,
                    scaler_meta=scaler_meta,
                    component_train=component_axes.train,
                    component_val=component_axes.val,
                    component_test=component_axes.test,
                    include_components=True,
                )
            add_refiner_rows(
                rows=rows,
                arrays=arrays,
                args=args,
                name=f"{prefix}_risk_no_context",
                q_train=q_train,
                q_val=q_val,
                q_test=q_test,
                ctx_train=ctx_train,
                ctx_val=ctx_val,
                ctx_test=ctx_test,
                device=device,
                include_context=False,
                shuffled_labels=False,
                shuffled_context=False,
                risk_logs=risk_logs,
                scaler_meta=scaler_meta,
            )
            if args.include_controls:
                add_refiner_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_risk_shuffled_labels",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    device=device,
                    include_context=True,
                    shuffled_labels=True,
                    shuffled_context=False,
                    risk_logs=risk_logs,
                    scaler_meta=scaler_meta,
                )
                add_refiner_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_risk_shuffled_context",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    device=device,
                    include_context=True,
                    shuffled_labels=False,
                    shuffled_context=True,
                    risk_logs=risk_logs,
                    scaler_meta=scaler_meta,
                )
                if component_axes is not None:
                    add_refiner_rows(
                        rows=rows,
                        arrays=arrays,
                        args=args,
                        name=f"{prefix}_component_risk_shuffled_components",
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
                        shuffled_components=True,
                    )
                    add_refiner_rows(
                        rows=rows,
                        arrays=arrays,
                        args=args,
                        name=f"{prefix}_component_risk_shuffled_labels",
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        device=device,
                        include_context=True,
                        shuffled_labels=True,
                        shuffled_context=False,
                        risk_logs=risk_logs,
                        scaler_meta=scaler_meta,
                        component_train=component_axes.train,
                        component_val=component_axes.val,
                        component_test=component_axes.test,
                        include_components=True,
                    )
            if not args.skip_sklearn_risk:
                add_sklearn_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_hgbdt_full",
                    kind="hgbdt",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    include_context=True,
                )
                if component_axes is not None:
                    add_sklearn_rows(
                        rows=rows,
                        arrays=arrays,
                        args=args,
                        name=f"{prefix}_component_hgbdt_full",
                        kind="hgbdt",
                        q_train=q_train,
                        q_val=q_val,
                        q_test=q_test,
                        ctx_train=ctx_train,
                        ctx_val=ctx_val,
                        ctx_test=ctx_test,
                        include_context=True,
                        component_train=component_axes.train,
                        component_val=component_axes.val,
                        component_test=component_axes.test,
                        include_components=True,
                    )
                    if args.include_controls:
                        add_sklearn_rows(
                            rows=rows,
                            arrays=arrays,
                            args=args,
                            name=f"{prefix}_component_hgbdt_shuffled_components",
                            kind="hgbdt",
                            q_train=q_train,
                            q_val=q_val,
                            q_test=q_test,
                            ctx_train=ctx_train,
                            ctx_val=ctx_val,
                            ctx_test=ctx_test,
                            include_context=True,
                            component_train=component_axes.train,
                            component_val=component_axes.val,
                            component_test=component_axes.test,
                            include_components=True,
                            shuffled_components=True,
                        )
                add_sklearn_rows(
                    rows=rows,
                    arrays=arrays,
                    args=args,
                    name=f"{prefix}_ridge_full",
                    kind="ridge",
                    q_train=q_train,
                    q_val=q_val,
                    q_test=q_test,
                    ctx_train=ctx_train,
                    ctx_val=ctx_val,
                    ctx_test=ctx_test,
                    include_context=True,
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "route_prototype_refiner_summary.csv", index=False)
    if risk_logs:
        pd.concat(risk_logs, ignore_index=True).to_csv(args.out_dir / "route_prototype_refiner_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "learned_route_generator_train_log.csv", index=False)
    run_meta = finite_json(vars(args))
    run_meta["extra_feature_block"] = finite_json(extra_feature_meta)
    if component_axes is not None:
        run_meta["component_axis_names"] = component_axes.names
    (args.out_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(scaler_meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = ["# Route Prototype Refiner Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- prototype_methods: `{args.prototype_methods}`")
    lines.append(f"- prototype_k: `{args.prototype_k}`")
    lines.append(f"- component_aware_risk: `{bool(args.component_aware_risk)}`")
    if args.component_aware_risk:
        lines.append(f"- component_axis_blocks: `{args.component_axis_blocks}`")
        lines.append(f"- component_axis_model: `{args.component_axis_model}`")
    if getattr(args, "extra_feature_grid", None):
        lines.append(f"- extra_feature_grid: `{args.extra_feature_grid}`")
        lines.append(f"- extra_feature_block_name: `{args.extra_feature_block_name}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{h}")
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        for _, row in sub.head(28).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, gain={row['gain_vs_base_pct']:.2f}%")
    lines.append("\n## Decision Notes")
    lines.append("- Pass if learned prototype refiner beats prototype mean/prior and is clearly better than shuffled/no-context controls.")
    lines.append("- If prototype oracle is strong but learned refiner remains near prototype mean, the missing piece is causal selection/observability, not route construction.")
    (out_dir / "route_prototype_refiner_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.add_argument("--prototype-methods", type=str, default="fps_endpoint,fps_shape,fps_full")
    parser.add_argument("--prototype-k", type=str, default="8,12,16")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--component-aware-risk", action="store_true")
    parser.add_argument(
        "--component-axis-blocks",
        type=str,
        default="self,flow,morphology,boundary,crowding,raw_context,all_context",
        help="Causal feature blocks used to train learned component axes for route selection.",
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
        args.max_train_rows = min(args.max_train_rows, 4000)
        args.max_val_rows = min(args.max_val_rows, 1500)
        args.max_test_rows = min(args.max_test_rows, 2000)
        args.posterior_epochs = min(args.posterior_epochs, 8)
        args.student_epochs = min(args.student_epochs, 8)
        args.learned_route_epochs = min(args.learned_route_epochs, 6)
        args.critic_epochs = min(args.critic_epochs, 8)
        args.risk_epochs = min(args.risk_epochs, 8)
        args.hgbdt_max_iter = min(args.hgbdt_max_iter, 80)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.max_all_features = min(args.max_all_features, 192)
    run(args)


if __name__ == "__main__":
    main()
