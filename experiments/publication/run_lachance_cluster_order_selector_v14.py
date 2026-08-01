#!/usr/bin/env python3
"""Cluster/order-aware selector over v12 route-conditioned candidates.

This runner tests the hypothesis that the selector fails because it receives a
flat, weakly ordered candidate cloud.  v14 first forms local route/trajectory
clusters and then learns a hierarchical score:

    cluster tokens -> cluster score
    candidate tokens + assigned cluster token -> candidate score

The final prediction is still made from original v12 candidates, so clustering
is only a stabilizing interface for selection, not an oracle shortcut.
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

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    KMeans = None
    HistGradientBoostingRegressor = None

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_conditioned_generator_v12 as v12  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_v13_sequence_refiner_v12_cloud as v13  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "cluster_order_selector_v14_2026-07-03"
EPS = 1e-8


@dataclass
class ClusterPack:
    assign: np.ndarray  # n,k
    features: np.ndarray  # n,c,d
    residual: np.ndarray  # n,c,h,2 representative residual
    member_oracle_dist: np.ndarray  # n,c min candidate MSE inside cluster
    rep_oracle_dist: np.ndarray  # n,c representative MSE
    size: np.ndarray  # n,c,1


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def metric_rows(arrays: audit.SplitArrays, pred: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def endpoint_signature(residual: np.ndarray, horizons: list[int]) -> np.ndarray:
    n, k, hmax, _ = residual.shape
    cum = np.cumsum(residual, axis=2)
    hs = [max(1, min(int(h), hmax)) for h in horizons]
    endpoints = np.stack([cum[:, :, h - 1, :] for h in hs], axis=2).reshape(n, k, -1)
    speed = np.linalg.norm(residual, axis=-1)
    acc = np.diff(residual, axis=2)
    acc_norm = np.linalg.norm(acc, axis=-1) if hmax > 1 else np.zeros((n, k, 1), dtype=np.float32)
    final = cum[:, :, hs[-1] - 1, :]
    final_norm = np.linalg.norm(final, axis=-1, keepdims=True)
    path = np.sum(speed, axis=2, keepdims=True)
    persistence = final_norm / np.maximum(path, EPS)
    sig = np.concatenate(
        [
            endpoints,
            np.mean(speed, axis=2, keepdims=True),
            np.std(speed, axis=2, keepdims=True),
            np.mean(acc_norm, axis=2, keepdims=True),
            path,
            final_norm,
            persistence,
        ],
        axis=-1,
    )
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def reorder_cluster_labels(labels: np.ndarray, score: np.ndarray, cluster_count: int) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int64)
    present = np.unique(labels)
    stats: list[tuple[float, int]] = []
    for lab in present:
        mask = labels == lab
        stats.append((float(np.mean(score[mask])) + 1e-3 * float(np.sum(mask)), int(lab)))
    stats.sort(reverse=True)
    mapping = {lab: min(i, cluster_count - 1) for i, (_, lab) in enumerate(stats)}
    for lab in present:
        out[labels == lab] = mapping[int(lab)]
    return out


def route_assign(pack: seq.CandidatePack, cluster_count: int) -> np.ndarray:
    n, k = pack.residual.shape[:2]
    if pack.route_mode is None:
        return np.tile(np.arange(k) % cluster_count, (n, 1)).astype(np.int64)
    assign = np.zeros((n, k), dtype=np.int64)
    route = np.asarray(pack.route_mode, dtype=np.int64)
    logp = np.squeeze(pack.logprob, axis=-1)
    for i in range(n):
        assign[i] = reorder_cluster_labels(route[i], logp[i], cluster_count)
    return assign


def kmeans_assign(pack: seq.CandidatePack, cluster_count: int, args: argparse.Namespace, *, hybrid: bool) -> np.ndarray:
    if KMeans is None:
        raise RuntimeError("sklearn KMeans is required for kmeans clustering")
    sig = endpoint_signature(pack.residual, args.horizons)
    n, k = sig.shape[:2]
    assign = np.zeros((n, k), dtype=np.int64)
    logp = np.squeeze(pack.logprob, axis=-1)
    for i in range(n):
        x = sig[i]
        if hybrid and pack.mode_prob is not None and pack.mode_prob.size:
            x = np.concatenate([x, 1.5 * pack.mode_prob[i].astype(np.float32)], axis=1)
        uniq = np.unique(np.round(x, 4), axis=0)
        c = max(1, min(cluster_count, len(uniq), k))
        if c == 1:
            lab = np.zeros(k, dtype=np.int64)
        else:
            km = KMeans(n_clusters=c, n_init=5, random_state=int(args.seed) + 21000 + i)
            lab = km.fit_predict(x).astype(np.int64)
        assign[i] = reorder_cluster_labels(lab, logp[i], cluster_count)
    return assign


def make_cluster_pack(
    pack: seq.CandidatePack,
    residual_true: np.ndarray,
    base: np.ndarray,
    args: argparse.Namespace,
    *,
    method: str,
    rep: str,
    cluster_count: int,
) -> ClusterPack:
    method = method.strip()
    if method == "route":
        assign = route_assign(pack, cluster_count)
    elif method == "kmeans":
        assign = kmeans_assign(pack, cluster_count, args, hybrid=False)
    elif method == "hybrid":
        assign = kmeans_assign(pack, cluster_count, args, hybrid=True)
    else:
        raise ValueError(f"Unknown cluster method: {method}")

    n, k, hmax, _ = pack.residual.shape
    true_flat = audit.flatten_residual(residual_true)
    sig = endpoint_signature(pack.residual, args.horizons)
    logp = np.squeeze(pack.logprob, axis=-1)
    reps = np.zeros((n, cluster_count, hmax, 2), dtype=np.float32)
    member_min = np.full((n, cluster_count), np.inf, dtype=np.float32)
    rep_dist = np.full((n, cluster_count), np.inf, dtype=np.float32)
    size = np.zeros((n, cluster_count, 1), dtype=np.float32)
    stats = np.zeros((n, cluster_count, 8 + pack.mode_prob.shape[-1]), dtype=np.float32)

    global_mean = np.mean(pack.residual, axis=1)
    for i in range(n):
        for c in range(cluster_count):
            idx = np.where(assign[i] == c)[0]
            if len(idx) == 0:
                reps[i, c] = global_mean[i]
                stats[i, c, 0] = 0.0
                continue
            cand = pack.residual[i, idx]
            cand_sig = sig[i, idx]
            center_sig = np.mean(cand_sig, axis=0, keepdims=True)
            center_res = np.mean(cand, axis=0)
            if rep == "mean":
                reps[i, c] = center_res
            elif rep == "medoid":
                med = int(np.argmin(np.mean((cand_sig - center_sig) ** 2, axis=1)))
                reps[i, c] = cand[med]
            else:
                raise ValueError(f"Unknown cluster rep: {rep}")
            member_min[i, c] = float(np.min(pack.oracle_dist[i, idx]))
            rep_dist[i, c] = float(np.mean((reps[i, c].reshape(-1) - true_flat[i]) ** 2))
            size[i, c, 0] = float(len(idx)) / float(k)
            disp = float(np.mean(np.mean((cand_sig - center_sig) ** 2, axis=1))) if len(idx) > 1 else 0.0
            lp = logp[i, idx]
            mp = np.mean(pack.mode_prob[i, idx], axis=0)
            ent = float(-np.sum(mp * np.log(np.maximum(mp, EPS))))
            stats[i, c, :8] = np.asarray(
                [
                    size[i, c, 0],
                    disp,
                    float(np.mean(lp)),
                    float(np.max(lp)),
                    float(np.std(lp)),
                    float(np.mean(np.linalg.norm(cand.reshape(len(idx), -1), axis=1))),
                    float(np.std(np.linalg.norm(cand.reshape(len(idx), -1), axis=1))),
                    ent,
                ],
                dtype=np.float32,
            )
            stats[i, c, 8:] = mp.astype(np.float32)

    rep_feat, _ = seq.build_candidate_features(
        residual=reps,
        base=base,
        z_eps=np.zeros((n, cluster_count, args.latent_dim), dtype=np.float32),
        logprob=np.zeros((n, cluster_count, 1), dtype=np.float32),
        horizons=args.horizons,
    )
    features = np.concatenate([rep_feat, stats], axis=-1).astype(np.float32)
    member_min = np.nan_to_num(member_min, nan=1e6, posinf=1e6, neginf=1e6).astype(np.float32)
    rep_dist = np.nan_to_num(rep_dist, nan=1e6, posinf=1e6, neginf=1e6).astype(np.float32)
    return ClusterPack(assign=assign, features=features, residual=reps, member_oracle_dist=member_min, rep_oracle_dist=rep_dist, size=size)


class HierarchicalClusterSelector(nn.Module):
    def __init__(
        self,
        *,
        cand_dim: int,
        cluster_dim: int,
        ctx_dim: int,
        hidden: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cand = nn.Sequential(nn.Linear(cand_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.cluster = nn.Sequential(nn.Linear(cluster_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.ctx = nn.Sequential(nn.Linear(max(ctx_dim, 1), hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.member_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cluster_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.member_score = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, cand_x: torch.Tensor, cluster_x: torch.Tensor, ctx_x: torch.Tensor, assign: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx_x.shape[1] == 0:
            ctx_x = torch.zeros((cand_x.shape[0], 1), device=cand_x.device, dtype=cand_x.dtype)
        cx = self.ctx(ctx_x)
        ctok = self.cluster_encoder(self.cluster(cluster_x) + cx[:, None, :])
        cluster_scores = self.cluster_score(ctok).squeeze(-1)
        gather = assign[:, :, None].expand(-1, -1, ctok.shape[-1])
        assigned_cluster = torch.gather(ctok, dim=1, index=gather)
        mtok = self.member_encoder(self.cand(cand_x) + assigned_cluster + cx[:, None, :])
        member_scores = self.member_score(torch.cat([mtok, assigned_cluster, cx[:, None, :].expand_as(mtok)], dim=-1)).squeeze(-1)
        member_scores = member_scores + torch.gather(cluster_scores, dim=1, index=assign)
        return member_scores, cluster_scores


def topm_prediction(scores: torch.Tensor, residual: torch.Tensor, top_m: int, temperature: float) -> torch.Tensor:
    kk = max(1, min(int(top_m), scores.shape[1]))
    vals, idx = torch.topk(scores, k=kk, dim=1)
    gather = idx[:, :, None, None].expand(-1, -1, residual.shape[2], residual.shape[3])
    cand = torch.gather(residual, dim=1, index=gather)
    w = torch.softmax(vals / max(float(temperature), 1e-6), dim=1)
    return torch.sum(w[:, :, None, None] * cand, dim=1)


def soft_prediction(scores: torch.Tensor, residual: torch.Tensor, temperature: float) -> torch.Tensor:
    w = torch.softmax(scores / max(float(temperature), 1e-6), dim=1)
    return torch.sum(w[:, :, None, None] * residual, dim=1)


def cluster_soft_labels(member_dist: torch.Tensor, assign: torch.Tensor, cluster_count: int, temperature: float) -> torch.Tensor:
    vals = []
    big = torch.full((member_dist.shape[0],), 1e6, device=member_dist.device, dtype=member_dist.dtype)
    for c in range(cluster_count):
        d = torch.where(assign == c, member_dist, big[:, None])
        vals.append(torch.min(d, dim=1).values)
    dist = torch.stack(vals, dim=1)
    return seq.soft_oracle_labels(dist, temperature)


def train_model(
    *,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cl_train: ClusterPack,
    cl_val: ClusterPack,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[HierarchicalClusterSelector, pd.DataFrame]:
    model = HierarchicalClusterSelector(
        cand_dim=cand_train.features.shape[-1],
        cluster_dim=cl_train.features.shape[-1],
        ctx_dim=ctx_train.shape[1],
        hidden=args.v14_hidden,
        heads=args.v14_heads,
        layers=args.v14_layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    rows = []
    n = len(ctx_train)
    for epoch in range(int(args.v14_epochs)):
        model.train()
        losses = []
        for idx in closure.batches(n, args.critic_batch_size, args.seed + 31000 + epoch):
            cx = seq.to_tensor(ctx_train[idx], device)
            cf = seq.to_tensor(cand_train.features[idx], device)
            cr = seq.to_tensor(cand_train.residual[idx], device)
            clx = seq.to_tensor(cl_train.features[idx], device)
            assign = torch.as_tensor(cl_train.assign[idx], dtype=torch.long, device=device)
            yt = seq.to_tensor(residual_train[idx], device)
            dist = seq.to_tensor(cand_train.oracle_dist[idx], device)
            q_member = seq.soft_oracle_labels(dist, args.oracle_temperature)
            q_cluster = cluster_soft_labels(dist, assign, args.v14_cluster_count, args.oracle_temperature)
            member_scores, cluster_scores = model(cf, clx, cx, assign)
            pred_soft = soft_prediction(member_scores, cr, args.v14_soft_temperature)
            pred_top = topm_prediction(member_scores, cr, args.v14_top_m_train, args.v14_sparse_temperature)
            reg = 0.5 * seq.endpoint_loss(pred_soft, yt, args.horizons) + seq.endpoint_loss(pred_top, yt, args.horizons)
            reg = reg + 0.5 * F.smooth_l1_loss(pred_soft.reshape(pred_soft.shape[0], -1), yt.reshape(yt.shape[0], -1))
            member_list = -torch.mean(torch.sum(q_member * F.log_softmax(member_scores, dim=1), dim=1))
            cluster_list = -torch.mean(torch.sum(q_cluster * F.log_softmax(cluster_scores, dim=1), dim=1))
            member_hard = F.cross_entropy(member_scores, torch.argmin(dist, dim=1))
            cluster_hard = F.cross_entropy(cluster_scores, torch.argmax(q_cluster, dim=1))
            pair = seq.pairwise_rank_loss(member_scores, q_member)
            ent = -torch.mean(torch.sum(torch.softmax(member_scores, dim=1) * F.log_softmax(member_scores, dim=1), dim=1))
            loss = (
                float(args.v14_reg_weight) * reg
                + float(args.v14_member_listwise_weight) * member_list
                + float(args.v14_cluster_listwise_weight) * cluster_list
                + float(args.v14_member_hard_weight) * member_hard
                + float(args.v14_cluster_hard_weight) * cluster_hard
                + float(args.v14_pairwise_weight) * pair
                + float(args.v14_entropy_weight) * ent
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pred_val = predict(model, ctx_val, cand_val, cl_val, args, device=device, top_m_values=[args.v14_top_m_train])[f"topM{args.v14_top_m_train}"]
        val_rmse = v13.residual_endpoint_rmse_np(pred_val, residual_val, args.horizons)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_residual_rmse": float(val_rmse)})
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict(
    model: HierarchicalClusterSelector,
    ctx: np.ndarray,
    cand: seq.CandidatePack,
    cl: ClusterPack,
    args: argparse.Namespace,
    *,
    device: torch.device,
    top_m_values: list[int],
) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {f"topM{m}": [] for m in top_m_values}
    outs["soft"] = []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, args.seed + 32000, shuffle=False):
            cx = seq.to_tensor(ctx[idx], device)
            cf = seq.to_tensor(cand.features[idx], device)
            cr = seq.to_tensor(cand.residual[idx], device)
            clx = seq.to_tensor(cl.features[idx], device)
            assign = torch.as_tensor(cl.assign[idx], dtype=torch.long, device=device)
            member_scores, _ = model(cf, clx, cx, assign)
            outs["soft"].append(soft_prediction(member_scores, cr, args.v14_soft_temperature).cpu().numpy())
            for m in top_m_values:
                outs[f"topM{m}"].append(topm_prediction(member_scores, cr, m, args.v14_sparse_temperature).cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def score_diagnostics(
    model: HierarchicalClusterSelector,
    ctx: np.ndarray,
    cand: seq.CandidatePack,
    cl: ClusterPack,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    pairs, ranks, top1, oracle = [], [], [], []
    cluster_pairs, cluster_ranks = [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, args.seed + 33000, shuffle=False):
            cx = seq.to_tensor(ctx[idx], device)
            cf = seq.to_tensor(cand.features[idx], device)
            clx = seq.to_tensor(cl.features[idx], device)
            assign = torch.as_tensor(cl.assign[idx], dtype=torch.long, device=device)
            ms, cs = model(cf, clx, cx, assign)
            m = ms.cpu().numpy()
            c = cs.cpu().numpy()
            dist = cand.oracle_dist[idx]
            cd = cl.member_oracle_dist[idx]
            order = np.argsort(-m, axis=1)
            best = np.argmin(dist, axis=1)
            rr = np.empty_like(order)
            rr[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
            ranks.append(rr[np.arange(len(order)), best])
            top1.append(dist[np.arange(len(order)), order[:, 0]])
            oracle.append(np.min(dist, axis=1))
            pairs.append(np.stack([m.reshape(-1), (-dist).reshape(-1)], axis=1))
            co = np.argsort(-c, axis=1)
            cbest = np.argmin(cd, axis=1)
            cr = np.empty_like(co)
            cr[np.arange(len(co))[:, None], co] = np.arange(co.shape[1])[None, :]
            cluster_ranks.append(cr[np.arange(len(co)), cbest])
            cluster_pairs.append(np.stack([c.reshape(-1), (-cd).reshape(-1)], axis=1))
    p = np.concatenate(pairs, axis=0)
    cp = np.concatenate(cluster_pairs, axis=0)
    corr = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]) if np.std(p[:, 0]) > 1e-8 and np.std(p[:, 1]) > 1e-8 else float("nan")
    ccorr = float(np.corrcoef(cp[:, 0], cp[:, 1])[0, 1]) if np.std(cp[:, 0]) > 1e-8 and np.std(cp[:, 1]) > 1e-8 else float("nan")
    return {
        "member_score_neg_error_corr": corr,
        "cluster_score_neg_error_corr": ccorr,
        "oracle_candidate_rank_mean": float(np.mean(np.concatenate(ranks))),
        "oracle_cluster_rank_mean": float(np.mean(np.concatenate(cluster_ranks))),
        "top1_mse_mean": float(np.mean(np.concatenate(top1))),
        "oracle_mse_mean": float(np.mean(np.concatenate(oracle))),
    }


def select_context_columns(train: np.ndarray, val: np.ndarray, test: np.ndarray, max_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.shape[1] <= int(max_cols):
        return train, val, test
    var = np.nan_to_num(np.var(train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.argsort(var)[-int(max_cols) :]
    return train[:, keep], val[:, keep], test[:, keep]


def member_feature_matrix(
    cand: seq.CandidatePack,
    cl: ClusterPack,
    ctx: np.ndarray,
    args: argparse.Namespace,
    *,
    include_context: bool,
) -> np.ndarray:
    n, k = cand.residual.shape[:2]
    rows = np.arange(n)[:, None]
    gathered_cluster = cl.features[rows, cl.assign]
    parts = [cand.features.astype(np.float32), gathered_cluster.astype(np.float32)]
    if include_context:
        ctx_rep = np.repeat(ctx[:, None, :], k, axis=1).astype(np.float32)
        parts.append(ctx_rep)
    out = np.concatenate(parts, axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def hgbdt_risk_selector(
    *,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    ctx_test: np.ndarray,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    cand_test: seq.CandidatePack,
    cl_train: ClusterPack,
    cl_val: ClusterPack,
    cl_test: ClusterPack,
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
    variant: str,
    include_context: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if HistGradientBoostingRegressor is None:
        raise RuntimeError("sklearn HistGradientBoostingRegressor is required for v14 HGBDT")
    ctx_train, ctx_val, ctx_test = select_context_columns(ctx_train, ctx_val, ctx_test, args.v14_hgbdt_max_context)
    xtr = member_feature_matrix(cand_train, cl_train, ctx_train, args, include_context=include_context)
    xva = member_feature_matrix(cand_val, cl_val, ctx_val, args, include_context=include_context)
    xte = member_feature_matrix(cand_test, cl_test, ctx_test, args, include_context=include_context)
    ytr = np.log1p(cand_train.oracle_dist.reshape(-1))
    model = HistGradientBoostingRegressor(
        max_iter=int(args.v14_hgbdt_iter),
        learning_rate=float(args.v14_hgbdt_lr),
        max_leaf_nodes=int(args.v14_hgbdt_leaf_nodes),
        l2_regularization=float(args.v14_hgbdt_l2),
        random_state=int(args.seed) + 35000,
    )
    model.fit(xtr.reshape(-1, xtr.shape[-1]), ytr)
    val_risk = model.predict(xva.reshape(-1, xva.shape[-1])).reshape(xva.shape[:2]).astype(np.float32)
    test_risk = model.predict(xte.reshape(-1, xte.shape[-1])).reshape(xte.shape[:2]).astype(np.float32)
    scores = -test_risk
    rows: list[dict[str, Any]] = []
    for m in [int(x) for x in parse_strs(args.v14_eval_top_m)]:
        pred = topm_prediction(
            torch.as_tensor(scores, dtype=torch.float32),
            torch.as_tensor(cand_test.residual, dtype=torch.float32),
            top_m=m,
            temperature=args.v14_sparse_temperature,
        ).numpy()
        rows.extend(metric_rows(arrays, pred, f"{variant}_hgbdt_topM{m}", args, {"stage": "v14_hgbdt_selector", "variant": variant, "top_m": int(m)}))
    pred_soft = soft_prediction(
        torch.as_tensor(scores, dtype=torch.float32),
        torch.as_tensor(cand_test.residual, dtype=torch.float32),
        temperature=args.v14_soft_temperature,
    ).numpy()
    rows.extend(metric_rows(arrays, pred_soft, f"{variant}_hgbdt_soft", args, {"stage": "v14_hgbdt_selector", "variant": variant}))

    val_corr = float(np.corrcoef(val_risk.reshape(-1), cand_val.oracle_dist.reshape(-1))[0, 1]) if np.std(val_risk) > 1e-8 else float("nan")
    test_corr = float(np.corrcoef(test_risk.reshape(-1), cand_test.oracle_dist.reshape(-1))[0, 1]) if np.std(test_risk) > 1e-8 else float("nan")
    order = np.argsort(test_risk, axis=1)
    best = np.argmin(cand_test.oracle_dist, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
    diag = {
        "variant": variant,
        "selector": "hgbdt",
        "include_context": bool(include_context),
        "val_risk_error_corr": val_corr,
        "test_risk_error_corr": test_corr,
        "oracle_candidate_rank_mean": float(np.mean(rank[np.arange(len(order)), best])),
        "top1_mse_mean": float(np.mean(cand_test.oracle_dist[np.arange(len(order)), order[:, 0]])),
        "oracle_mse_mean": float(np.mean(np.min(cand_test.oracle_dist, axis=1))),
    }
    return pd.DataFrame(rows), pd.DataFrame([diag])


def cluster_oracle_residual(cl: ClusterPack, residual_true: np.ndarray, *, use_rep: bool) -> np.ndarray:
    dist = cl.rep_oracle_dist if use_rep else cl.member_oracle_dist
    best = np.argmin(dist, axis=1)
    return cl.residual[np.arange(len(best)), best]


def build_v12_cloud(args: argparse.Namespace, device: torch.device) -> tuple[audit.SplitArrays, dict[str, seq.CandidatePack], tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame, dict[str, Any]]:
    arrays, split = audit.prepare_data(args)
    extra_meta = rpr.attach_extra_feature_block(arrays, split, args)
    velocity_blocks, velocity_names = v10.build_velocity_blocks(split, max_cols=args.v10_velocity_max_cols)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    decomp = v12.decomposition_features(student, arrays, blocks, args, device)
    labels = v12.fit_route_labels(arrays, args)
    xtr_raw, xva_raw, xte_raw, names = v12.build_route_feature_matrix(
        arrays=arrays,
        split=split,
        velocity_blocks=velocity_blocks,
        decomp=decomp,
        variant=args.v14_generator_variant,
        args=args,
    )
    prior = v12.fit_prior_model(name=args.v14_generator_variant, xtr_raw=xtr_raw, xva_raw=xva_raw, xte_raw=xte_raw, labels=labels, args=args, feature_names=names)
    bank = v12.fit_expert_bank(prior, labels, arrays, args)
    packs = {
        "train": v12.generate_expert_candidates(
            name=args.v14_generator_variant,
            prior=prior,
            bank=bank,
            probs=prior.probs_train,
            x=prior.x_train,
            residual_true=arrays.residual_train,
            arrays_base=arrays.base_train,
            args=args,
            split_name="train",
        ),
        "val": v12.generate_expert_candidates(
            name=args.v14_generator_variant,
            prior=prior,
            bank=bank,
            probs=prior.probs_val,
            x=prior.x_val,
            residual_true=arrays.residual_val,
            arrays_base=arrays.base_val,
            args=args,
            split_name="val",
        ),
        "test": v12.generate_expert_candidates(
            name=args.v14_generator_variant,
            prior=prior,
            bank=bank,
            probs=prior.probs_test,
            x=prior.x_test,
            residual_true=arrays.residual_test,
            arrays_base=arrays.base_test,
            args=args,
            split_name="test",
        ),
    }
    ctx = v13.context_matrix(arrays, prior, args)
    gate = pd.DataFrame(v12.prior_gate_rows(prior, labels))
    meta = {"extra_feature": extra_meta, "velocity_names": velocity_names, "route_k": labels.k, "expert_meta": bank.meta}
    return arrays, packs, ctx, gate, meta


def run_variant(
    *,
    method: str,
    rep: str,
    control: str,
    arrays: audit.SplitArrays,
    packs: dict[str, seq.CandidatePack],
    ctx: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx_train, ctx_val, ctx_test = ctx
    if control == "no_context":
        ctx_train = np.zeros_like(ctx_train, dtype=np.float32)
        ctx_val = np.zeros_like(ctx_val, dtype=np.float32)
        ctx_test = np.zeros_like(ctx_test, dtype=np.float32)
    elif control == "shuffled_context":
        rng = np.random.default_rng(args.seed + 34100)
        ctx_train = ctx_train[rng.permutation(len(ctx_train))]
        ctx_val = ctx_val[rng.permutation(len(ctx_val))]
        ctx_test = ctx_test[rng.permutation(len(ctx_test))]
    elif control != "full":
        raise ValueError(f"Unknown control: {control}")

    cl_train = make_cluster_pack(packs["train"], arrays.residual_train, arrays.base_train, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
    cl_val = make_cluster_pack(packs["val"], arrays.residual_val, arrays.base_val, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
    cl_test = make_cluster_pack(packs["test"], arrays.residual_test, arrays.base_test, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
    model, log = train_model(
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        cand_train=packs["train"],
        cand_val=packs["val"],
        cl_train=cl_train,
        cl_val=cl_val,
        residual_train=arrays.residual_train,
        residual_val=arrays.residual_val,
        args=args,
        device=device,
    )
    top_m = [int(x) for x in parse_strs(args.v14_eval_top_m)]
    preds = predict(model, ctx_test, packs["test"], cl_test, args, device=device, top_m_values=top_m)
    variant = f"v14_{method}_{rep}_{control}"
    rows: list[dict[str, Any]] = []
    for key, pred in preds.items():
        rows.extend(metric_rows(arrays, pred, f"{variant}_{key}", args, {"stage": "v14_hierarchical_selector", "variant": variant, "cluster_method": method, "cluster_rep": rep, "control": control}))
    diag = score_diagnostics(model, ctx_test, packs["test"], cl_test, args, device=device)
    diag.update(
        {
            "variant": variant,
            "cluster_method": method,
            "cluster_rep": rep,
            "control": control,
            "best_val_residual_rmse": float(log["val_residual_rmse"].min()) if not log.empty else float("nan"),
            "cluster_member_oracle_mse_mean": float(np.mean(np.min(cl_test.member_oracle_dist, axis=1))),
            "cluster_rep_oracle_mse_mean": float(np.mean(np.min(cl_test.rep_oracle_dist, axis=1))),
        }
    )
    log = log.copy()
    log.insert(0, "variant", variant)
    return pd.DataFrame(rows), pd.concat([pd.DataFrame([diag]), log], ignore_index=True, sort=False)


def run_hgbdt_variant(
    *,
    method: str,
    rep: str,
    control: str,
    arrays: audit.SplitArrays,
    packs: dict[str, seq.CandidatePack],
    ctx: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx_train, ctx_val, ctx_test = ctx
    include_context = True
    if control == "no_context":
        include_context = False
    elif control == "shuffled_context":
        rng = np.random.default_rng(args.seed + 36100)
        ctx_train = ctx_train[rng.permutation(len(ctx_train))]
        ctx_val = ctx_val[rng.permutation(len(ctx_val))]
        ctx_test = ctx_test[rng.permutation(len(ctx_test))]
    elif control != "full":
        raise ValueError(f"Unknown control: {control}")
    cl_train = make_cluster_pack(packs["train"], arrays.residual_train, arrays.base_train, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
    cl_val = make_cluster_pack(packs["val"], arrays.residual_val, arrays.base_val, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
    cl_test = make_cluster_pack(packs["test"], arrays.residual_test, arrays.base_test, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
    variant = f"v14_{method}_{rep}_{control}"
    return hgbdt_risk_selector(
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        cand_train=packs["train"],
        cand_val=packs["val"],
        cand_test=packs["test"],
        cl_train=cl_train,
        cl_val=cl_val,
        cl_test=cl_test,
        arrays=arrays,
        args=args,
        variant=variant,
        include_context=include_context,
    )


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, gate: pd.DataFrame) -> None:
    lines = ["# v14 Cluster/Order-Aware Selector", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- cluster_count: `{args.v14_cluster_count}`")
    lines.append(f"- generator: `{args.v14_generator_variant}`")
    lines.append("")
    if not gate.empty:
        lines.append("## Route Prior Gate")
        lines.append(gate[gate["split"].eq("test")].to_markdown(index=False))
        lines.append("")
    for h in args.horizons:
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "oracle_k"] if c in sub.columns]
        lines.append(f"## h{h}")
        lines.append(sub[cols].head(50).to_markdown(index=False))
        lines.append("")
    if not diag.empty:
        compact = diag[diag["epoch"].isna()].copy() if "epoch" in diag.columns else diag.copy()
        cols = [
            c
            for c in [
                "variant",
                "member_score_neg_error_corr",
                "cluster_score_neg_error_corr",
                "oracle_candidate_rank_mean",
                "oracle_cluster_rank_mean",
                "top1_mse_mean",
                "oracle_mse_mean",
                "cluster_member_oracle_mse_mean",
                "cluster_rep_oracle_mse_mean",
                "best_val_residual_rmse",
            ]
            if c in compact.columns
        ]
        lines.append("## Diagnostics")
        lines.append(compact[cols].to_markdown(index=False))
    (out_dir / "cluster_order_selector_v14_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.horizons = audit.parse_ints(args.horizons)
    args.oracle_k = audit.parse_ints(args.oracle_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, packs, ctx, gate, meta = build_v12_cloud(args, device)

    rows: list[dict[str, Any]] = []
    rows.extend(metric_rows(arrays, seq.mean_candidate_residual(packs["test"]), "v12_candidate_mean", args, {"stage": "candidate_mean"}))
    for k in args.oracle_k:
        kk = min(int(k), args.candidate_k)
        rows.extend(metric_rows(arrays, seq.oracle_residual(packs["test"], arrays.residual_test, kk), f"v12_oracle@{kk}", args, {"stage": "candidate_oracle", "oracle_k": kk}))

    summaries = [pd.DataFrame(rows)]
    diagnostics = []
    methods = parse_strs(args.v14_cluster_methods)
    reps = parse_strs(args.v14_cluster_reps)
    controls = parse_strs(args.v14_controls)
    for method in methods:
        for rep in reps:
            cl_test = make_cluster_pack(packs["test"], arrays.residual_test, arrays.base_test, args, method=method, rep=rep, cluster_count=args.v14_cluster_count)
            summaries.append(pd.DataFrame(metric_rows(arrays, cluster_oracle_residual(cl_test, arrays.residual_test, use_rep=True), f"v14_{method}_{rep}_cluster_rep_oracle", args, {"stage": "cluster_rep_oracle", "variant": f"v14_{method}_{rep}"})))
            for control in controls:
                if args.v14_enable_neural:
                    summary, diag = run_variant(method=method, rep=rep, control=control, arrays=arrays, packs=packs, ctx=ctx, args=args, device=device)
                    summaries.append(summary)
                    diagnostics.append(diag)
                if args.v14_enable_hgbdt:
                    summary, diag = run_hgbdt_variant(method=method, rep=rep, control=control, arrays=arrays, packs=packs, ctx=ctx, args=args)
                    summaries.append(summary)
                    diagnostics.append(diag)

    summary = pd.concat(summaries, ignore_index=True)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    diag = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))
    if not gate.empty:
        gate.insert(0, "seed", int(args.seed))
        gate.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "cluster_order_selector_v14_summary.csv", index=False)
    diag.to_csv(args.out_dir / "cluster_order_selector_v14_diagnostics.csv", index=False)
    gate.to_csv(args.out_dir / "cluster_order_selector_v14_prior_gate.csv", index=False)
    (args.out_dir / "cluster_order_selector_v14_meta.json").write_text(json.dumps(audit.finite_json(meta), indent=2), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag, gate)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diag_rows": len(diag)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(out_dir=DEFAULT_OUT)
    parser.add_argument("--extra-feature-grid", type=Path, default=v12.DEFAULT_OBJECT_GRID)
    parser.add_argument("--extra-feature-prefixes", type=str, default="oc_")
    parser.add_argument("--extra-feature-block-name", type=str, default="object_mask")
    parser.add_argument("--extra-feature-max-cols", type=int, default=256)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--v10-velocity-max-cols", type=int, default=160)
    parser.add_argument("--v12-route-k", type=int, default=12)
    parser.add_argument("--v12-min-route-cluster-size", type=int, default=40)
    parser.add_argument("--v12-prior-model", type=str, default="logistic", choices=["logistic", "hgbdt"])
    parser.add_argument("--v12-prior-max-iter", type=int, default=500)
    parser.add_argument("--v12-prior-c", type=float, default=0.35)
    parser.add_argument("--v12-hgbdt-iter", type=int, default=160)
    parser.add_argument("--v12-hgbdt-lr", type=float, default=0.05)
    parser.add_argument("--v12-hgbdt-leaf-nodes", type=int, default=31)
    parser.add_argument("--v12-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v12-max-route-features", type=int, default=768)
    parser.add_argument("--v12-include-decomposition", action="store_true")
    parser.add_argument("--v12-expert-alpha", type=float, default=300.0)
    parser.add_argument("--v12-min-expert-samples", type=int, default=80)
    parser.add_argument("--v12-error-pool-max", type=int, default=2500)
    parser.add_argument("--v12-top-route-modes", type=int, default=4)
    parser.add_argument("--v12-route-prob-power", type=float, default=1.5)
    parser.add_argument("--v12-error-noise-scale", type=float, default=0.75)
    parser.add_argument("--v12-noise-jitter", type=float, default=0.02)
    parser.add_argument("--v13-context-source", type=str, default="route_prior", choices=["route_prior", "all_context", "combined"])
    parser.add_argument("--v13-max-context-features", type=int, default=512)
    parser.add_argument("--v14-generator-variant", type=str, default="context_velocity")
    parser.add_argument("--v14-cluster-methods", type=str, default="route,hybrid")
    parser.add_argument("--v14-cluster-reps", type=str, default="medoid")
    parser.add_argument("--v14-controls", type=str, default="full,no_context,shuffled_context")
    parser.add_argument("--v14-cluster-count", type=int, default=8)
    parser.add_argument("--v14-hidden", type=int, default=192)
    parser.add_argument("--v14-heads", type=int, default=4)
    parser.add_argument("--v14-layers", type=int, default=2)
    parser.add_argument("--v14-epochs", type=int, default=10)
    parser.add_argument("--v14-top-m-train", type=int, default=8)
    parser.add_argument("--v14-eval-top-m", type=str, default="1,2,4,8,16")
    parser.add_argument("--v14-sparse-temperature", type=float, default=0.5)
    parser.add_argument("--v14-soft-temperature", type=float, default=0.75)
    parser.add_argument("--v14-reg-weight", type=float, default=1.0)
    parser.add_argument("--v14-member-listwise-weight", type=float, default=0.8)
    parser.add_argument("--v14-cluster-listwise-weight", type=float, default=0.5)
    parser.add_argument("--v14-member-hard-weight", type=float, default=0.2)
    parser.add_argument("--v14-cluster-hard-weight", type=float, default=0.15)
    parser.add_argument("--v14-pairwise-weight", type=float, default=0.15)
    parser.add_argument("--v14-entropy-weight", type=float, default=0.01)
    parser.add_argument("--v14-enable-neural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--v14-enable-hgbdt", action="store_true")
    parser.add_argument("--v14-hgbdt-iter", type=int, default=220)
    parser.add_argument("--v14-hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--v14-hgbdt-leaf-nodes", type=int, default=31)
    parser.add_argument("--v14-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v14-hgbdt-max-context", type=int, default=256)
    args = parser.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 400)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.v14_epochs = min(args.v14_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = "4,8,16"
        args.v14_cluster_methods = "route,hybrid"
        args.v14_controls = "full,no_context"
        args.v14_eval_top_m = "1,2,4,8"
    run(args)


if __name__ == "__main__":
    main()
