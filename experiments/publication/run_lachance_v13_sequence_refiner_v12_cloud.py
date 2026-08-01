#!/usr/bin/env python3
"""Sequence/Graph Critic-Refiner v13 over v12 route-conditioned cloud."""

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
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "v13_sequence_refiner_v12_cloud_2026-07-03"


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


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


def context_matrix(
    arrays: audit.SplitArrays,
    prior: v12.RoutePrior,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.v13_context_source == "route_prior":
        return prior.x_train, prior.x_val, prior.x_test
    if args.v13_context_source == "all_context":
        xtr, xva, xte = arrays.x_train["all_context"], arrays.x_val["all_context"], arrays.x_test["all_context"]
    else:
        xtr = np.concatenate([arrays.x_train["all_context"], prior.x_train], axis=1)
        xva = np.concatenate([arrays.x_val["all_context"], prior.x_val], axis=1)
        xte = np.concatenate([arrays.x_test["all_context"], prior.x_test], axis=1)
    if xtr.shape[1] > int(args.v13_max_context_features):
        var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-int(args.v13_max_context_features) :]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
    return seq.standardize(xtr, xva, xte)[:3]


def make_route_teacher(labels: v12.RouteLabels, prior: v12.RoutePrior) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use the causal prior distribution as teacher; it matches candidate route
    # modes and avoids coupling this critic to the older decomposition posterior.
    return prior.probs_train, prior.probs_val, prior.probs_test


class SparseCandidateSelector(nn.Module):
    """Direct sparse selector over a strong candidate cloud.

    The route-query critic can still average route queries.  This module keeps
    the problem closer to what v12 needs: score candidate trajectories, select
    a small top-M set, and optionally apply a bounded per-candidate correction.
    """

    def __init__(
        self,
        *,
        cand_dim: int,
        ctx_dim: int,
        hidden: int,
        horizon: int,
        heads: int,
        layers: int,
        dropout: float,
        correction_scale: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.correction_scale = float(correction_scale)
        self.cand = nn.Sequential(
            nn.Linear(cand_dim, hidden),
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

    def forward(self, cand_x: torch.Tensor, ctx_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ce = self.cand(cand_x)
        if ctx_x.shape[1] == 0:
            ctx_x = torch.zeros((cand_x.shape[0], 1), device=cand_x.device, dtype=cand_x.dtype)
        cx = self.ctx(ctx_x)
        tokens = self.encoder(ce + cx[:, None, :])
        scores = self.score(tokens).squeeze(-1)
        corr = torch.tanh(self.correction(tokens)).view(cand_x.shape[0], cand_x.shape[1], self.horizon, 2)
        return scores, self.correction_scale * corr


def sparse_topm_residual(
    scores: torch.Tensor,
    residual: torch.Tensor,
    corr: torch.Tensor,
    *,
    top_m: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kk = max(1, min(int(top_m), scores.shape[1]))
    vals, idx = torch.topk(scores, k=kk, dim=1)
    gather_idx = idx[:, :, None, None].expand(-1, -1, residual.shape[2], residual.shape[3])
    cand = torch.gather(residual + corr, dim=1, index=gather_idx)
    weights = torch.softmax(vals / max(float(temperature), 1e-6), dim=1)
    pred = torch.sum(weights[:, :, None, None] * cand, dim=1)
    return pred, idx


def residual_endpoint_rmse_np(pred: np.ndarray, true: np.ndarray, horizons: list[int]) -> float:
    vals = []
    for h in horizons:
        p = np.sum(pred[:, : int(h), :], axis=1)
        y = np.sum(true[:, : int(h), :], axis=1)
        vals.append(np.mean(np.sum((p - y) ** 2, axis=1)))
    return float(np.sqrt(np.mean(vals)))


def train_sparse_selector(
    *,
    ctx_train: np.ndarray,
    ctx_val: np.ndarray,
    cand_train: seq.CandidatePack,
    cand_val: seq.CandidatePack,
    residual_train: np.ndarray,
    residual_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SparseCandidateSelector, pd.DataFrame]:
    model = SparseCandidateSelector(
        cand_dim=cand_train.features.shape[-1],
        ctx_dim=ctx_train.shape[1],
        hidden=args.critic_hidden,
        horizon=args.max_horizon,
        heads=args.critic_heads,
        layers=args.critic_layers,
        dropout=args.dropout,
        correction_scale=args.correction_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    rows: list[dict[str, Any]] = []
    n = len(ctx_train)
    for epoch in range(int(args.critic_epochs)):
        model.train()
        losses = []
        for idx in closure.batches(n, args.critic_batch_size, args.seed + 18100 + epoch):
            cx = seq.to_tensor(ctx_train[idx], device)
            cf = seq.to_tensor(cand_train.features[idx], device)
            cr = seq.to_tensor(cand_train.residual[idx], device)
            yt = seq.to_tensor(residual_train[idx], device)
            dist = seq.to_tensor(cand_train.oracle_dist[idx], device)
            q = seq.soft_oracle_labels(dist, args.oracle_temperature)
            best_idx = torch.argmin(dist, dim=1)
            scores, corr = model(cf, cx)
            pred, _ = sparse_topm_residual(scores, cr, corr, top_m=args.v13_sparse_top_m_train, temperature=args.v13_sparse_temperature)
            reg = seq.endpoint_loss(pred, yt, args.horizons) + F.smooth_l1_loss(pred.reshape(pred.shape[0], -1), yt.reshape(yt.shape[0], -1))
            listwise = -torch.mean(torch.sum(q * F.log_softmax(scores, dim=1), dim=1))
            hard_ce = F.cross_entropy(scores, best_idx)
            pairwise = seq.pairwise_rank_loss(scores, q)
            correction_penalty = torch.mean(corr.pow(2))
            loss = (
                reg
                + float(args.v13_sparse_listwise_weight) * listwise
                + float(args.v13_sparse_hard_weight) * hard_ce
                + float(args.v13_sparse_pairwise_weight) * pairwise
                + 0.002 * correction_penalty
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), getattr(args, "grad_clip", 5.0))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = evaluate_sparse_predictions(model, ctx_val, cand_val, args, device=device, top_m_values=[args.v13_sparse_top_m_train])[args.v13_sparse_top_m_train]
        val_rmse = residual_endpoint_rmse_np(val_pred, residual_val, args.horizons)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_residual_rmse": val_rmse})
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def evaluate_sparse_predictions(
    model: SparseCandidateSelector,
    ctx: np.ndarray,
    cand: seq.CandidatePack,
    args: argparse.Namespace,
    *,
    device: torch.device,
    top_m_values: list[int],
) -> dict[int, np.ndarray]:
    model.eval()
    out = {int(m): [] for m in top_m_values}
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, args.seed + 19100, shuffle=False):
            cx = seq.to_tensor(ctx[idx], device)
            cf = seq.to_tensor(cand.features[idx], device)
            cr = seq.to_tensor(cand.residual[idx], device)
            scores, corr = model(cf, cx)
            for m in out:
                pred, _ = sparse_topm_residual(scores, cr, corr, top_m=m, temperature=args.v13_sparse_temperature)
                out[m].append(pred.detach().cpu().numpy())
    return {m: np.concatenate(v, axis=0).astype(np.float32) for m, v in out.items()}


def sparse_selector_diagnostics(
    model: SparseCandidateSelector,
    ctx: np.ndarray,
    cand: seq.CandidatePack,
    residual_true: np.ndarray,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_scores, all_best_rank, all_top1_dist, all_oracle_dist = [], [], [], []
    with torch.no_grad():
        for idx in closure.batches(len(ctx), args.critic_batch_size, args.seed + 19200, shuffle=False):
            scores, _ = model(seq.to_tensor(cand.features[idx], device), seq.to_tensor(ctx[idx], device))
            sc = scores.detach().cpu().numpy()
            dist = cand.oracle_dist[idx]
            order = np.argsort(-sc, axis=1)
            best = np.argmin(dist, axis=1)
            ranks = np.empty_like(order)
            ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
            all_best_rank.append(ranks[np.arange(len(order)), best])
            all_top1_dist.append(dist[np.arange(len(order)), order[:, 0]])
            all_oracle_dist.append(np.min(dist, axis=1))
            all_scores.append(np.stack([sc.reshape(-1), (-dist).reshape(-1)], axis=1))
    pairs = np.concatenate(all_scores, axis=0)
    corr = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) > 2 and np.std(pairs[:, 0]) > 1e-8 and np.std(pairs[:, 1]) > 1e-8 else float("nan")
    top1 = np.concatenate(all_top1_dist)
    oracle = np.concatenate(all_oracle_dist)
    return {
        "score_neg_error_corr": corr,
        "oracle_candidate_rank_mean": float(np.mean(np.concatenate(all_best_rank))),
        "top1_mse_mean": float(np.mean(top1)),
        "oracle_mse_mean": float(np.mean(oracle)),
        "top1_oracle_gap_mse": float(np.mean(top1 - oracle)),
    }


def run_sparse_variant(
    *,
    name: str,
    arrays: audit.SplitArrays,
    packs: dict[str, seq.CandidatePack],
    ctx: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    use_context: bool = True,
    shuffled_context: bool = False,
    shuffled_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx_train, ctx_val, ctx_test = ctx
    if not use_context:
        ctx_train = np.zeros_like(ctx_train, dtype=np.float32)
        ctx_val = np.zeros_like(ctx_val, dtype=np.float32)
        ctx_test = np.zeros_like(ctx_test, dtype=np.float32)
    if shuffled_context:
        rng = np.random.default_rng(args.seed + 19601)
        ctx_train = ctx_train[rng.permutation(len(ctx_train))]
        ctx_val = ctx_val[rng.permutation(len(ctx_val))]
        ctx_test = ctx_test[rng.permutation(len(ctx_test))]
    residual_train = arrays.residual_train.copy()
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 19651)
        residual_train = residual_train[rng.permutation(len(residual_train))]

    model, log = train_sparse_selector(
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        cand_train=packs["train"],
        cand_val=packs["val"],
        residual_train=residual_train,
        residual_val=arrays.residual_val,
        args=args,
        device=device,
    )
    top_m_values = [int(x) for x in parse_strs(args.v13_sparse_eval_top_m)]
    preds = evaluate_sparse_predictions(model, ctx_test, packs["test"], args, device=device, top_m_values=top_m_values)
    rows: list[dict[str, Any]] = []
    for m, pred in preds.items():
        rows.extend(metric_rows(arrays, pred, f"{name}_sparse_topM{m}", args, {"stage": "sparse_selector", "variant": name, "top_m": int(m)}))
    diag = sparse_selector_diagnostics(model, ctx_test, packs["test"], arrays.residual_test, args, device=device)
    diag.update({"variant": name, "use_context": bool(use_context), "shuffled_context": bool(shuffled_context), "shuffled_labels": bool(shuffled_labels), "best_val_residual_rmse": float(log["val_residual_rmse"].min()) if not log.empty else float("nan")})
    log = log.copy()
    log.insert(0, "variant", name)
    return pd.DataFrame(rows), pd.concat([pd.DataFrame([diag]), log], ignore_index=True, sort=False)


def run_variant(
    *,
    name: str,
    arrays: audit.SplitArrays,
    packs: dict[str, seq.CandidatePack],
    ctx: tuple[np.ndarray, np.ndarray, np.ndarray],
    route_teacher: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    use_context: bool = True,
    shuffled_context: bool = False,
    shuffled_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx_train, ctx_val, ctx_test = ctx
    train_use_context = bool(use_context)
    if not use_context:
        # RouteQueryRefiner keeps a fixed context projection size.  A true
        # no-context control should remove information, not change tensor rank.
        ctx_train = np.zeros_like(ctx_train, dtype=np.float32)
        ctx_val = np.zeros_like(ctx_val, dtype=np.float32)
        ctx_test = np.zeros_like(ctx_test, dtype=np.float32)
        train_use_context = True
    if shuffled_context:
        rng = np.random.default_rng(args.seed + 13401)
        ctx_train = ctx_train[rng.permutation(len(ctx_train))]
        ctx_val = ctx_val[rng.permutation(len(ctx_val))]
        ctx_test = ctx_test[rng.permutation(len(ctx_test))]
    mode_tr, mode_va, mode_te = route_teacher
    if shuffled_labels:
        rng = np.random.default_rng(args.seed + 13451)
        mode_tr = mode_tr[rng.permutation(len(mode_tr))]
        mode_va = mode_va[rng.permutation(len(mode_va))]

    model, log = seq.train_critic(
        ctx_train,
        ctx_val,
        packs["train"],
        packs["val"],
        arrays.residual_train,
        arrays.residual_val,
        posterior_mu_train=np.zeros((len(arrays.residual_train), args.latent_dim), dtype=np.float32),
        posterior_mode_train=mode_tr.astype(np.float32),
        args=args,
        device=device,
        use_context=train_use_context,
    )
    rows, weights = seq.evaluate_final(model, ctx_test, packs["test"], arrays, args, device=device, label_prefix=name)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["variant"] = name
    diag = {
        "variant": name,
        "use_context": bool(use_context),
        "shuffled_context": bool(shuffled_context),
        "shuffled_labels": bool(shuffled_labels),
        "weight_entropy_mean": float(-np.mean(np.sum(weights * np.log(np.maximum(weights, 1e-8)), axis=1))),
        "weight_top1_mean": float(np.mean(np.max(weights, axis=1))),
    }
    if log is not None and not log.empty:
        log = log.copy()
        log.insert(0, "variant", name)
        diag["best_val_residual_rmse"] = float(np.min(log["val_residual_rmse"]))
        diag["best_val_top_residual_rmse"] = float(np.min(log["val_top_residual_rmse"]))
    else:
        diag["best_val_residual_rmse"] = float("nan")
        diag["best_val_top_residual_rmse"] = float("nan")
    return summary, pd.DataFrame([diag]) if log is None or log.empty else pd.concat([pd.DataFrame([diag]), log], ignore_index=True, sort=False)


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.horizons = audit.parse_ints(args.horizons)
    args.oracle_k = audit.parse_ints(args.oracle_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)

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
        variant=args.v13_generator_variant,
        args=args,
    )
    prior = v12.fit_prior_model(name=args.v13_generator_variant, xtr_raw=xtr_raw, xva_raw=xva_raw, xte_raw=xte_raw, labels=labels, args=args, feature_names=names)
    bank = v12.fit_expert_bank(prior, labels, arrays, args)
    packs = {
        "train": v12.generate_expert_candidates(
            name=args.v13_generator_variant,
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
            name=args.v13_generator_variant,
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
            name=args.v13_generator_variant,
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
    ctx = context_matrix(arrays, prior, args)
    route_teacher = make_route_teacher(labels, prior)

    baseline_rows: list[dict[str, Any]] = []
    baseline_rows.extend(metric_rows(arrays, seq.mean_candidate_residual(packs["test"]), "v12_candidate_mean", args, {"stage": "candidate_mean"}))
    for k in args.oracle_k:
        kk = min(int(k), args.candidate_k)
        baseline_rows.extend(metric_rows(arrays, seq.oracle_residual(packs["test"], arrays.residual_test, kk), f"v12_oracle@{kk}", args, {"stage": "candidate_oracle", "oracle_k": kk}))

    summaries = [pd.DataFrame(baseline_rows)]
    diagnostics = []
    if args.v13_enable_route_query:
        variants = {
            "v13_full": {},
            "v13_no_context": {"use_context": False},
            "v13_shuffled_context": {"shuffled_context": True},
            "v13_shuffled_labels": {"shuffled_labels": True},
        }
        requested = set(parse_strs(args.v13_variant_list))
        for name, kwargs in variants.items():
            if requested and name not in requested:
                continue
            summary, diag = run_variant(name=name, arrays=arrays, packs=packs, ctx=ctx, route_teacher=route_teacher, args=args, device=device, **kwargs)
            summaries.append(summary)
            diagnostics.append(diag)

    if args.v13_enable_sparse_selector:
        sparse_variants = {
            "v13_sparse_full": {},
            "v13_sparse_no_context": {"use_context": False},
            "v13_sparse_shuffled_context": {"shuffled_context": True},
            "v13_sparse_shuffled_labels": {"shuffled_labels": True},
        }
        requested_sparse = set(parse_strs(args.v13_sparse_variant_list))
        for name, kwargs in sparse_variants.items():
            if requested_sparse and name not in requested_sparse:
                continue
            summary, diag = run_sparse_variant(name=name, arrays=arrays, packs=packs, ctx=ctx, args=args, device=device, **kwargs)
            summaries.append(summary)
            diagnostics.append(diag)

    summary_df = pd.concat(summaries, ignore_index=True)
    if not summary_df.empty:
        summary_df.insert(0, "seed", int(args.seed))
        summary_df.insert(0, "dataset", str(args.dataset))
    diag_df = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    if not diag_df.empty:
        diag_df.insert(0, "seed", int(args.seed))
        diag_df.insert(0, "dataset", str(args.dataset))
    gate_df = pd.DataFrame(v12.prior_gate_rows(prior, labels))
    if not gate_df.empty:
        gate_df.insert(0, "seed", int(args.seed))
        gate_df.insert(0, "dataset", str(args.dataset))

    summary_df.to_csv(args.out_dir / "v13_sequence_refiner_v12_summary.csv", index=False)
    diag_df.to_csv(args.out_dir / "v13_sequence_refiner_v12_diagnostics.csv", index=False)
    gate_df.to_csv(args.out_dir / "v13_sequence_refiner_v12_prior_gate.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "v13_sequence_refiner_v12_meta.json").write_text(
        json.dumps(finite_json({"extra_feature": extra_meta, "velocity_names": velocity_names, "route_k": labels.k, "expert_meta": bank.meta}), indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir, args, summary_df, diag_df, gate_df)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary_df), "diag_rows": len(diag_df)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, gate: pd.DataFrame) -> None:
    lines = ["# v13 Sequence Refiner over v12 Cloud", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- generator variant: `{args.v13_generator_variant}`")
    lines.append(f"- critic_arch: `{args.critic_arch}`")
    lines.append("")
    if not gate.empty:
        lines.append("## Route Prior Gate")
        lines.append(gate[gate["split"].eq("test")].to_markdown(index=False))
        lines.append("")
    for h in args.horizons:
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "oracle_k"] if c in sub.columns]
        lines.append(f"## h{h}")
        lines.append(sub[cols].head(40).to_markdown(index=False))
        lines.append("")
    if not diag.empty:
        lines.append("## Diagnostics")
        cols = [c for c in ["variant", "use_context", "shuffled_context", "shuffled_labels", "weight_entropy_mean", "weight_top1_mean", "best_val_residual_rmse", "best_val_top_residual_rmse"] if c in diag.columns]
        lines.append(diag[diag["epoch"].isna() if "epoch" in diag.columns else slice(None)][cols].to_markdown(index=False))
    (out_dir / "v13_sequence_refiner_v12_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--v13-generator-variant", type=str, default="context_velocity")
    parser.add_argument("--v13-context-source", type=str, default="route_prior", choices=["route_prior", "all_context", "combined"])
    parser.add_argument("--v13-max-context-features", type=int, default=512)
    parser.add_argument("--v13-enable-route-query", action="store_true")
    parser.add_argument("--v13-enable-sparse-selector", action="store_true", default=True)
    parser.add_argument("--v13-variant-list", type=str, default="v13_full,v13_no_context,v13_shuffled_context,v13_shuffled_labels")
    parser.add_argument("--v13-sparse-variant-list", type=str, default="v13_sparse_full,v13_sparse_no_context,v13_sparse_shuffled_context,v13_sparse_shuffled_labels")
    parser.add_argument("--v13-sparse-top-m-train", type=int, default=4)
    parser.add_argument("--v13-sparse-eval-top-m", type=str, default="1,2,4,8,16")
    parser.add_argument("--v13-sparse-temperature", type=float, default=0.35)
    parser.add_argument("--v13-sparse-listwise-weight", type=float, default=0.75)
    parser.add_argument("--v13-sparse-hard-weight", type=float, default=0.25)
    parser.add_argument("--v13-sparse-pairwise-weight", type=float, default=0.15)
    args = parser.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 400)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.critic_epochs = min(args.critic_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = "4,8,16"
        args.v13_variant_list = "v13_full,v13_no_context"
        args.v13_sparse_variant_list = "v13_sparse_full,v13_sparse_no_context"
        args.v13_sparse_eval_top_m = "1,2,4,8"
    run(args)


if __name__ == "__main__":
    main()
