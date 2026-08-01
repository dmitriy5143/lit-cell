#!/usr/bin/env python3
"""Cluster-mixture generator v15 over v12 route-conditioned candidates.

v14 showed that route/trajectory clusters preserve oracle signal, but a
post-hoc selector cannot identify the correct candidate reliably.  This runner
moves cluster/route tokens into the generator objective: instead of selecting a
single candidate after sampling, it learns a calibrated mixture over generated
route clusters.
"""

from __future__ import annotations

import argparse
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

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_conditioned_generator_v12 as v12  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_cluster_order_selector_v14 as v14  # noqa: E402
import run_lachance_v13_sequence_refiner_v12_cloud as v13  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "cluster_mixture_generator_v15_2026-07-03"
EPS = 1e-8


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


def standardize_cluster_features(train: v14.ClusterPack, val: v14.ClusterPack, test: v14.ClusterPack) -> dict[str, Any]:
    shape_tr = train.features.shape
    shape_va = val.features.shape
    shape_te = test.features.shape
    tr = train.features.reshape(-1, shape_tr[-1])
    va = val.features.reshape(-1, shape_va[-1])
    te = test.features.reshape(-1, shape_te[-1])
    ztr, zva, zte, scaler = seq.standardize(tr, va, te)
    train.features = ztr.reshape(shape_tr).astype(np.float32)
    val.features = zva.reshape(shape_va).astype(np.float32)
    test.features = zte.reshape(shape_te).astype(np.float32)
    return scaler


def cluster_prior_weights(pack: seq.CandidatePack, cl: v14.ClusterPack, cluster_count: int, mode: str) -> np.ndarray:
    n, k = cl.assign.shape
    weights = np.zeros((n, cluster_count), dtype=np.float32)
    if mode == "uniform":
        weights[:] = 1.0 / float(cluster_count)
        return weights
    if mode == "size":
        weights = np.squeeze(cl.size, axis=-1).astype(np.float32)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
        return weights
    if mode == "logprob":
        prob = np.exp(np.squeeze(pack.logprob, axis=-1)).astype(np.float32)
        for c in range(cluster_count):
            weights[:, c] = np.sum(np.where(cl.assign == c, prob, 0.0), axis=1)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
        return weights
    raise ValueError(f"Unknown prior weight mode: {mode}")


def weighted_cluster_residual(cl: v14.ClusterPack, weights: np.ndarray) -> np.ndarray:
    return np.sum(weights[:, :, None, None] * cl.residual, axis=1).astype(np.float32)


class ClusterMixtureGenerator(nn.Module):
    def __init__(
        self,
        *,
        cluster_dim: int,
        ctx_dim: int,
        hidden: int,
        heads: int,
        layers: int,
        horizon: int,
        dropout: float,
        correction_scale: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.correction_scale = float(correction_scale)
        self.cluster = nn.Sequential(
            nn.Linear(cluster_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.ctx = nn.Sequential(
            nn.Linear(max(ctx_dim, 1), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.correction = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * 2))

    def forward(self, cluster_x: torch.Tensor, ctx_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx_x.shape[1] == 0:
            ctx_x = torch.zeros((cluster_x.shape[0], 1), device=cluster_x.device, dtype=cluster_x.dtype)
        cx = self.ctx(ctx_x)
        tokens = self.encoder(self.cluster(cluster_x) + cx[:, None, :])
        logits = self.score(tokens).squeeze(-1)
        corr = torch.tanh(self.correction(tokens)).view(cluster_x.shape[0], cluster_x.shape[1], self.horizon, 2)
        return logits, self.correction_scale * corr


def mixture_prediction(logits: torch.Tensor, residual: torch.Tensor, corr: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(logits / max(float(temperature), 1e-6), dim=1)
    pred = torch.sum(weights[:, :, None, None] * (residual + corr), dim=1)
    return pred, weights


def topm_prediction(logits: torch.Tensor, residual: torch.Tensor, corr: torch.Tensor, top_m: int, temperature: float) -> torch.Tensor:
    kk = max(1, min(int(top_m), logits.shape[1]))
    vals, idx = torch.topk(logits, k=kk, dim=1)
    gather = idx[:, :, None, None].expand(-1, -1, residual.shape[2], residual.shape[3])
    chosen = torch.gather(residual + corr, dim=1, index=gather)
    w = torch.softmax(vals / max(float(temperature), 1e-6), dim=1)
    return torch.sum(w[:, :, None, None] * chosen, dim=1)


def cluster_soft_labels(dist: torch.Tensor, temperature: float) -> torch.Tensor:
    return seq.soft_oracle_labels(dist, temperature)


def train_generator(
    *,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    cl_train: v14.ClusterPack,
    cl_val: v14.ClusterPack,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ClusterMixtureGenerator, pd.DataFrame]:
    model = ClusterMixtureGenerator(
        cluster_dim=cl_train.features.shape[-1],
        ctx_dim=ctx_train.shape[1],
        hidden=args.v15_hidden,
        heads=args.v15_heads,
        layers=args.v15_layers,
        horizon=args.max_horizon,
        dropout=args.dropout,
        correction_scale=args.v15_correction_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    rows = []
    n = len(ctx_train)
    for epoch in range(int(args.v15_epochs)):
        model.train()
        losses = []
        for idx in closure.batches(n, args.critic_batch_size, args.seed + 41000 + epoch):
            cx = seq.to_tensor(ctx_train[idx], device)
            clx = seq.to_tensor(cl_train.features[idx], device)
            cr = seq.to_tensor(cl_train.residual[idx], device)
            yt = seq.to_tensor(residual_train[idx], device)
            cd = seq.to_tensor(cl_train.member_oracle_dist[idx], device)
            logits, corr = model(clx, cx)
            pred, weights = mixture_prediction(logits, cr, corr, args.v15_temperature)
            pred_top = topm_prediction(logits, cr, corr, args.v15_top_m_train, args.v15_temperature)
            q = cluster_soft_labels(cd, args.oracle_temperature)
            reg = seq.endpoint_loss(pred, yt, args.horizons) + 0.5 * F.smooth_l1_loss(pred.reshape(pred.shape[0], -1), yt.reshape(yt.shape[0], -1))
            top_reg = seq.endpoint_loss(pred_top, yt, args.horizons)
            listwise = -torch.mean(torch.sum(q * F.log_softmax(logits, dim=1), dim=1))
            hard = F.cross_entropy(logits, torch.argmin(cd, dim=1))
            entropy = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, min=1e-8)), dim=1))
            corr_penalty = torch.mean(corr.pow(2))
            loss = (
                float(args.v15_reg_weight) * reg
                + float(args.v15_top_reg_weight) * top_reg
                + float(args.v15_listwise_weight) * listwise
                + float(args.v15_hard_weight) * hard
                + float(args.v15_entropy_weight) * entropy
                + 0.002 * corr_penalty
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pred_val = predict(model, ctx_val, cl_val, args, device=device, top_m_values=[args.v15_top_m_train])["mixture"]
        val_rmse = v13.residual_endpoint_rmse_np(pred_val, residual_val, args.horizons)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_residual_rmse": float(val_rmse)})
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict(
    model: ClusterMixtureGenerator,
    ctx: np.ndarray,
    cl: v14.ClusterPack,
    args: argparse.Namespace,
    *,
    device: torch.device,
    top_m_values: list[int],
) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {"mixture": []}
    for m in top_m_values:
        outs[f"topM{m}"] = []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, args.seed + 42000, shuffle=False):
            cx = seq.to_tensor(ctx[idx], device)
            clx = seq.to_tensor(cl.features[idx], device)
            cr = seq.to_tensor(cl.residual[idx], device)
            logits, corr = model(clx, cx)
            pred, _ = mixture_prediction(logits, cr, corr, args.v15_temperature)
            outs["mixture"].append(pred.cpu().numpy())
            for m in top_m_values:
                outs[f"topM{m}"].append(topm_prediction(logits, cr, corr, m, args.v15_temperature).cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def diagnostics(
    model: ClusterMixtureGenerator,
    ctx: np.ndarray,
    cl: v14.ClusterPack,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    pairs, ranks, entropies = [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, args.seed + 43000, shuffle=False):
            logits, _ = model(seq.to_tensor(cl.features[idx], device), seq.to_tensor(ctx[idx], device))
            lg = logits.cpu().numpy()
            dist = cl.member_oracle_dist[idx]
            order = np.argsort(-lg, axis=1)
            best = np.argmin(dist, axis=1)
            rr = np.empty_like(order)
            rr[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
            ranks.append(rr[np.arange(len(order)), best])
            pairs.append(np.stack([lg.reshape(-1), (-dist).reshape(-1)], axis=1))
            w = np.exp(lg - np.max(lg, axis=1, keepdims=True))
            w = w / np.maximum(np.sum(w, axis=1, keepdims=True), EPS)
            entropies.append(-np.sum(w * np.log(np.maximum(w, EPS)), axis=1))
    p = np.concatenate(pairs, axis=0)
    corr = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]) if np.std(p[:, 0]) > 1e-8 and np.std(p[:, 1]) > 1e-8 else float("nan")
    return {
        "cluster_score_neg_error_corr": corr,
        "oracle_cluster_rank_mean": float(np.mean(np.concatenate(ranks))),
        "weight_entropy_mean": float(np.mean(np.concatenate(entropies))),
        "cluster_member_oracle_mse_mean": float(np.mean(np.min(cl.member_oracle_dist, axis=1))),
        "cluster_rep_oracle_mse_mean": float(np.mean(np.min(cl.rep_oracle_dist, axis=1))),
    }


def build_cloud(args: argparse.Namespace, device: torch.device) -> tuple[audit.SplitArrays, dict[str, seq.CandidatePack], tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame, dict[str, Any]]:
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
        variant=args.v15_generator_variant,
        args=args,
    )
    prior = v12.fit_prior_model(name=args.v15_generator_variant, xtr_raw=xtr_raw, xva_raw=xva_raw, xte_raw=xte_raw, labels=labels, args=args, feature_names=names)
    bank = v12.fit_expert_bank(prior, labels, arrays, args)
    packs = {
        "train": v12.generate_expert_candidates(
            name=args.v15_generator_variant,
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
            name=args.v15_generator_variant,
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
            name=args.v15_generator_variant,
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
        rng = np.random.default_rng(args.seed + 44000)
        ctx_train = ctx_train[rng.permutation(len(ctx_train))]
        ctx_val = ctx_val[rng.permutation(len(ctx_val))]
        ctx_test = ctx_test[rng.permutation(len(ctx_test))]
    elif control != "full":
        raise ValueError(f"Unknown control: {control}")

    cl_train = v14.make_cluster_pack(packs["train"], arrays.residual_train, arrays.base_train, args, method=args.v15_cluster_method, rep=args.v15_cluster_rep, cluster_count=args.v15_cluster_count)
    cl_val = v14.make_cluster_pack(packs["val"], arrays.residual_val, arrays.base_val, args, method=args.v15_cluster_method, rep=args.v15_cluster_rep, cluster_count=args.v15_cluster_count)
    cl_test = v14.make_cluster_pack(packs["test"], arrays.residual_test, arrays.base_test, args, method=args.v15_cluster_method, rep=args.v15_cluster_rep, cluster_count=args.v15_cluster_count)
    scaler = standardize_cluster_features(cl_train, cl_val, cl_test)
    model, log = train_generator(ctx_train=ctx_train, ctx_val=ctx_val, cl_train=cl_train, cl_val=cl_val, residual_train=arrays.residual_train, residual_val=arrays.residual_val, args=args, device=device)
    preds = predict(model, ctx_test, cl_test, args, device=device, top_m_values=[int(x) for x in parse_strs(args.v15_eval_top_m)])
    variant = f"v15_{args.v15_cluster_method}_{args.v15_cluster_rep}_{control}"
    rows: list[dict[str, Any]] = []
    for key, pred in preds.items():
        rows.extend(metric_rows(arrays, pred, f"{variant}_{key}", args, {"stage": "v15_cluster_mixture_generator", "variant": variant, "control": control}))
    diag = diagnostics(model, ctx_test, cl_test, args, device=device)
    diag.update({"variant": variant, "control": control, "best_val_residual_rmse": float(log["val_residual_rmse"].min()) if not log.empty else float("nan")})
    log = log.copy()
    log.insert(0, "variant", variant)
    return pd.DataFrame(rows), pd.concat([pd.DataFrame([diag]), log], ignore_index=True, sort=False)


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, gate: pd.DataFrame) -> None:
    lines = ["# v15 Cluster-Mixture Generator", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- cluster_count: `{args.v15_cluster_count}`")
    lines.append(f"- generator: `{args.v15_generator_variant}`")
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
        cols = [c for c in ["variant", "control", "cluster_score_neg_error_corr", "oracle_cluster_rank_mean", "weight_entropy_mean", "cluster_member_oracle_mse_mean", "cluster_rep_oracle_mse_mean", "best_val_residual_rmse"] if c in compact.columns]
        lines.append("## Diagnostics")
        lines.append(compact[cols].to_markdown(index=False))
    (out_dir / "cluster_mixture_generator_v15_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.horizons = audit.parse_ints(args.horizons)
    args.oracle_k = audit.parse_ints(args.oracle_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, packs, ctx, gate, meta = build_cloud(args, device)

    summaries: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    rows.extend(metric_rows(arrays, seq.mean_candidate_residual(packs["test"]), "v12_candidate_mean", args, {"stage": "candidate_mean"}))
    for k in args.oracle_k:
        kk = min(int(k), args.candidate_k)
        rows.extend(metric_rows(arrays, seq.oracle_residual(packs["test"], arrays.residual_test, kk), f"v12_oracle@{kk}", args, {"stage": "candidate_oracle", "oracle_k": kk}))
    # Fixed cluster mixture reductions.
    cl_test_raw = v14.make_cluster_pack(packs["test"], arrays.residual_test, arrays.base_test, args, method=args.v15_cluster_method, rep=args.v15_cluster_rep, cluster_count=args.v15_cluster_count)
    rows.extend(metric_rows(arrays, v14.cluster_oracle_residual(cl_test_raw, arrays.residual_test, use_rep=True), "v15_cluster_rep_oracle", args, {"stage": "cluster_rep_oracle"}))
    for mode in ["uniform", "size", "logprob"]:
        w = cluster_prior_weights(packs["test"], cl_test_raw, args.v15_cluster_count, mode)
        rows.extend(metric_rows(arrays, weighted_cluster_residual(cl_test_raw, w), f"v15_fixed_{mode}_cluster_mix", args, {"stage": "fixed_cluster_mixture", "variant": mode}))
    summaries.append(pd.DataFrame(rows))

    diagnostics_list = []
    for control in parse_strs(args.v15_controls):
        summary, diag = run_variant(control=control, arrays=arrays, packs=packs, ctx=ctx, args=args, device=device)
        summaries.append(summary)
        diagnostics_list.append(diag)

    summary = pd.concat(summaries, ignore_index=True)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    diag = pd.concat(diagnostics_list, ignore_index=True) if diagnostics_list else pd.DataFrame()
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))
    if not gate.empty:
        gate.insert(0, "seed", int(args.seed))
        gate.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "cluster_mixture_generator_v15_summary.csv", index=False)
    diag.to_csv(args.out_dir / "cluster_mixture_generator_v15_diagnostics.csv", index=False)
    gate.to_csv(args.out_dir / "cluster_mixture_generator_v15_prior_gate.csv", index=False)
    (args.out_dir / "cluster_mixture_generator_v15_meta.json").write_text(json.dumps(audit.finite_json(meta), indent=2), encoding="utf-8")
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
    parser.add_argument("--v15-generator-variant", type=str, default="context_velocity")
    parser.add_argument("--v15-cluster-method", type=str, default="hybrid", choices=["route", "kmeans", "hybrid"])
    parser.add_argument("--v15-cluster-rep", type=str, default="medoid", choices=["medoid", "mean"])
    parser.add_argument("--v15-cluster-count", type=int, default=16)
    parser.add_argument("--v15-controls", type=str, default="full,no_context,shuffled_context")
    parser.add_argument("--v15-hidden", type=int, default=224)
    parser.add_argument("--v15-heads", type=int, default=4)
    parser.add_argument("--v15-layers", type=int, default=2)
    parser.add_argument("--v15-epochs", type=int, default=16)
    parser.add_argument("--v15-temperature", type=float, default=0.75)
    parser.add_argument("--v15-top-m-train", type=int, default=8)
    parser.add_argument("--v15-eval-top-m", type=str, default="1,2,4,8,16")
    parser.add_argument("--v15-correction-scale", type=float, default=1.0)
    parser.add_argument("--v15-reg-weight", type=float, default=1.0)
    parser.add_argument("--v15-top-reg-weight", type=float, default=0.25)
    parser.add_argument("--v15-listwise-weight", type=float, default=0.7)
    parser.add_argument("--v15-hard-weight", type=float, default=0.15)
    parser.add_argument("--v15-entropy-weight", type=float, default=-0.01)
    args = parser.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 400)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.v15_epochs = min(args.v15_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = "4,8,16"
        args.v15_controls = "full,no_context"
        args.v15_eval_top_m = "1,2,4,8"
        args.v15_cluster_count = min(args.v15_cluster_count, 8)
    run(args)


if __name__ == "__main__":
    main()
