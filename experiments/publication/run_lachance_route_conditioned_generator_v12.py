#!/usr/bin/env python3
"""Route-conditioned candidate generator v12.

This runner tests the current diagnosis directly:

    route/velocity/object features contain observable regime signal,
    but the selector/refiner cannot pick from a broad weakly individualized cloud.

So we move the signal earlier.  A causal route-prior predicts a motion-regime
distribution, then per-route residual experts generate candidates around the
likely route modes.  The main gate is candidate oracle quality before any critic.

Target/future residuals are used only for training route labels/experts and for
metrics.  Inference features are causal.
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
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_object_centric_mask_gate as ocgate  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_sequence_joint_selector_refiner_v7 as v7  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_conditioned_generator_v12_2026-07-03"
DEFAULT_OBJECT_GRID = (
    ROOT
    / "outputs"
    / "lachance_object_centric_mask_grid_bulk_seed42_2026-07-03"
    / "object_centric_mask_feature_grid.csv"
)
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def safe_topk(y: np.ndarray, p: np.ndarray, k: int) -> float:
    if p.shape[1] <= 1:
        return float("nan")
    kk = min(int(k), p.shape[1])
    try:
        return float(top_k_accuracy_score(y, p, k=kk, labels=np.arange(p.shape[1])))
    except Exception:
        order = np.argsort(-p, axis=1)[:, :kk]
        return float(np.mean([int(y[i]) in set(order[i]) for i in range(len(y))]))


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    ez = np.exp(z)
    return (ez / np.maximum(np.sum(ez, axis=axis, keepdims=True), EPS)).astype(np.float32)


@dataclass
class RouteLabels:
    k: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    centers: np.ndarray
    scaler: StandardScaler


@dataclass
class RoutePrior:
    name: str
    model: Any
    scaler: StandardScaler
    k: int
    probs_train: np.ndarray
    probs_val: np.ndarray
    probs_test: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    feature_dim: int
    feature_names: list[str]


@dataclass
class ExpertBank:
    models: list[Ridge]
    global_model: Ridge
    error_pools: list[np.ndarray]
    global_error_pool: np.ndarray
    flat_dim: int
    meta: list[dict[str, Any]]


def fit_route_labels(arrays: audit.SplitArrays, args: argparse.Namespace) -> RouteLabels:
    sig_tr = v7.route_signature(arrays.residual_train, args.horizons)
    sig_va = v7.route_signature(arrays.residual_val, args.horizons)
    sig_te = v7.route_signature(arrays.residual_test, args.horizons)
    scaler = StandardScaler()
    ztr = scaler.fit_transform(sig_tr).astype(np.float32)
    zva = scaler.transform(sig_va).astype(np.float32)
    zte = scaler.transform(sig_te).astype(np.float32)
    k = min(int(args.v12_route_k), max(2, len(ztr) // max(10, int(args.v12_min_route_cluster_size))))
    km = KMeans(n_clusters=k, n_init=20, random_state=int(args.seed) + 12001)
    ytr = km.fit_predict(ztr).astype(np.int64)
    yva = km.predict(zva).astype(np.int64)
    yte = km.predict(zte).astype(np.int64)
    return RouteLabels(k=k, train=ytr, val=yva, test=yte, centers=km.cluster_centers_.astype(np.float32), scaler=scaler)


def maybe_object_control(x: np.ndarray, split_df: pd.DataFrame, mode: str, seed: int) -> np.ndarray:
    if mode == "real":
        return x
    if mode == "zero":
        return np.zeros_like(x)
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        return x[rng.permutation(len(x))].astype(np.float32)
    if mode == "same_frame_wrong_cell":
        return ocgate.same_frame_wrong_cell(x, split_df, seed)
    if mode == "time_shuffled":
        return ocgate.time_shuffle(x, split_df, seed)
    raise ValueError(f"Unknown object control mode: {mode}")


def decomposition_features(
    student: closure.ComponentStudentPrior | None,
    arrays: audit.SplitArrays,
    blocks: list[str] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if student is None or blocks is None or not bool(args.v12_include_decomposition):
        z0 = np.zeros((len(arrays.residual_train), 0), dtype=np.float32)
        z1 = np.zeros((len(arrays.residual_val), 0), dtype=np.float32)
        z2 = np.zeros((len(arrays.residual_test), 0), dtype=np.float32)
        return z0, z1, z2
    ptr = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
    pva = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
    pte = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
    return (
        seq.decomposition_context_features(ptr, mode_k=args.mode_k),
        seq.decomposition_context_features(pva, mode_k=args.mode_k),
        seq.decomposition_context_features(pte, mode_k=args.mode_k),
    )


def build_route_feature_matrix(
    *,
    arrays: audit.SplitArrays,
    split: audit.seq.SplitData,
    velocity_blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    decomp: tuple[np.ndarray, np.ndarray, np.ndarray],
    variant: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    variant = variant.strip()
    include_context = "context" in variant or variant == "full"
    include_velocity = "velocity" in variant or variant == "full"
    include_object = "object" in variant or variant == "full"
    include_decomp = "decomp" in variant or variant == "full"

    object_mode = "real"
    if "object_shuffled" in variant:
        include_object = True
        object_mode = "shuffled"
    elif "object_same_frame" in variant:
        include_object = True
        object_mode = "same_frame_wrong_cell"
    elif "object_zero" in variant:
        include_object = True
        object_mode = "zero"
    elif "object_time" in variant:
        include_object = True
        object_mode = "time_shuffled"

    parts_tr: list[np.ndarray] = []
    parts_va: list[np.ndarray] = []
    parts_te: list[np.ndarray] = []
    names: list[str] = []

    def add(prefix: str, mats: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        tr, va, te = mats
        if tr.shape[1] == 0:
            return
        parts_tr.append(tr.astype(np.float32))
        parts_va.append(va.astype(np.float32))
        parts_te.append(te.astype(np.float32))
        names.extend([f"{prefix}_{i}" for i in range(tr.shape[1])])

    if include_context:
        add("ctx", (arrays.x_train["all_context"], arrays.x_val["all_context"], arrays.x_test["all_context"]))
    if include_velocity:
        add("vel", velocity_blocks["all"])
    if include_object and "object_mask" in arrays.x_train:
        obj = (
            maybe_object_control(arrays.x_train["object_mask"], split.train, object_mode, args.seed + 12101),
            maybe_object_control(arrays.x_val["object_mask"], split.val, object_mode, args.seed + 12102),
            maybe_object_control(arrays.x_test["object_mask"], split.test, object_mode, args.seed + 12103),
        )
        add(f"object_{object_mode}", obj)
    if include_decomp:
        add("decomp", decomp)

    if not parts_tr:
        xtr = np.zeros((len(arrays.residual_train), 1), dtype=np.float32)
        xva = np.zeros((len(arrays.residual_val), 1), dtype=np.float32)
        xte = np.zeros((len(arrays.residual_test), 1), dtype=np.float32)
        names = ["bias"]
    else:
        xtr = np.concatenate(parts_tr, axis=1).astype(np.float32)
        xva = np.concatenate(parts_va, axis=1).astype(np.float32)
        xte = np.concatenate(parts_te, axis=1).astype(np.float32)

    if xtr.shape[1] > int(args.v12_max_route_features):
        var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-int(args.v12_max_route_features) :]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
        names = [names[int(i)] for i in keep]
    return xtr, xva, xte, names


def fit_prior_model(
    *,
    name: str,
    xtr_raw: np.ndarray,
    xva_raw: np.ndarray,
    xte_raw: np.ndarray,
    labels: RouteLabels,
    args: argparse.Namespace,
    feature_names: list[str],
) -> RoutePrior:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(np.nan_to_num(xtr_raw)).astype(np.float32)
    xva = scaler.transform(np.nan_to_num(xva_raw)).astype(np.float32)
    xte = scaler.transform(np.nan_to_num(xte_raw)).astype(np.float32)
    if args.v12_prior_model == "hgbdt":
        model = HistGradientBoostingClassifier(
            max_iter=int(args.v12_hgbdt_iter),
            learning_rate=float(args.v12_hgbdt_lr),
            max_leaf_nodes=int(args.v12_hgbdt_leaf_nodes),
            l2_regularization=float(args.v12_hgbdt_l2),
            random_state=int(args.seed) + 12201,
        )
    else:
        model = LogisticRegression(
            max_iter=int(args.v12_prior_max_iter),
            C=float(args.v12_prior_c),
            class_weight="balanced",
            random_state=int(args.seed) + 12201,
        )
    model.fit(xtr, labels.train)

    def proba(x: np.ndarray) -> np.ndarray:
        raw = model.predict_proba(x)
        out = np.full((len(x), labels.k), 1e-6, dtype=np.float32)
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = raw[:, j]
        out /= np.maximum(out.sum(axis=1, keepdims=True), EPS)
        return out.astype(np.float32)

    return RoutePrior(
        name=name,
        model=model,
        scaler=scaler,
        k=labels.k,
        probs_train=proba(xtr),
        probs_val=proba(xva),
        probs_test=proba(xte),
        x_train=xtr,
        x_val=xva,
        x_test=xte,
        feature_dim=xtr.shape[1],
        feature_names=feature_names,
    )


def prior_gate_rows(prior: RoutePrior, labels: RouteLabels) -> list[dict[str, Any]]:
    rows = []
    for split_name, y, p in [
        ("train", labels.train, prior.probs_train),
        ("val", labels.val, prior.probs_val),
        ("test", labels.test, prior.probs_test),
    ]:
        rows.append(
            {
                "variant": prior.name,
                "split": split_name,
                "feature_dim": int(prior.feature_dim),
                "route_top1": float(np.mean(np.argmax(p, axis=1) == y)),
                "route_top3": safe_topk(y, p, 3),
                "route_nll": float(log_loss(y, np.clip(p, 1e-6, 1.0), labels=np.arange(prior.k))),
                "true_route_prob": float(np.mean(p[np.arange(len(y)), y])),
                "entropy": float(-np.mean(np.sum(p * np.log(np.maximum(p, EPS)), axis=1))),
            }
        )
    return rows


def fit_expert_bank(prior: RoutePrior, labels: RouteLabels, arrays: audit.SplitArrays, args: argparse.Namespace) -> ExpertBank:
    y_flat = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    global_model = Ridge(alpha=float(args.v12_expert_alpha))
    global_model.fit(prior.x_train, y_flat)
    global_pred = global_model.predict(prior.x_train).astype(np.float32)
    global_err = (y_flat - global_pred).astype(np.float32)

    models: list[Ridge] = []
    pools: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for m in range(labels.k):
        mask = labels.train == m
        n = int(np.sum(mask))
        if n >= int(args.v12_min_expert_samples):
            model = Ridge(alpha=float(args.v12_expert_alpha))
            model.fit(prior.x_train[mask], y_flat[mask])
            pred = model.predict(prior.x_train[mask]).astype(np.float32)
            err = (y_flat[mask] - pred).astype(np.float32)
            source = "route_specific"
        else:
            model = global_model
            err = global_err
            source = "global_fallback"
        if len(err) > int(args.v12_error_pool_max):
            rng = np.random.default_rng(int(args.seed) + 12300 + m)
            take = rng.choice(len(err), size=int(args.v12_error_pool_max), replace=False)
            err = err[take]
        models.append(model)
        pools.append(err.astype(np.float32))
        meta.append({"route": m, "train_count": n, "expert_source": source, "error_pool": int(len(err))})
    return ExpertBank(models=models, global_model=global_model, error_pools=pools, global_error_pool=global_err, flat_dim=y_flat.shape[1], meta=meta)


def route_mode_schedule(probs: np.ndarray, args: argparse.Namespace, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, k_route = probs.shape
    k_cand = int(args.candidate_k)
    top_m = max(1, min(int(args.v12_top_route_modes), k_route, k_cand))
    power = max(float(args.v12_route_prob_power), 1e-6)
    modes = np.zeros((n, k_cand), dtype=np.int64)
    order = np.argsort(-probs, axis=1)
    modes[:, :top_m] = order[:, :top_m]
    for i in range(n):
        pool = order[i, :top_m]
        p = probs[i, pool].astype(np.float64)
        p = np.power(np.maximum(p, 1e-8), power)
        p = p / np.maximum(np.sum(p), EPS)
        if k_cand > top_m:
            modes[i, top_m:] = rng.choice(pool, size=k_cand - top_m, replace=True, p=p)
    return modes


def generate_expert_candidates(
    *,
    name: str,
    prior: RoutePrior,
    bank: ExpertBank,
    probs: np.ndarray,
    x: np.ndarray,
    residual_true: np.ndarray,
    arrays_base: np.ndarray,
    args: argparse.Namespace,
    split_name: str,
    shuffle_prior: bool = False,
    uniform_prior: bool = False,
) -> seq.CandidatePack:
    if uniform_prior:
        probs_use = np.full_like(probs, 1.0 / probs.shape[1])
    else:
        probs_use = probs.copy()
    if shuffle_prior:
        rng = np.random.default_rng(int(args.seed) + {"train": 12401, "val": 12402, "test": 12403}[split_name])
        probs_use = probs_use[rng.permutation(len(probs_use))]
    route_mode = route_mode_schedule(probs_use, args, seed=int(args.seed) + {"train": 12501, "val": 12502, "test": 12503}[split_name])
    n, k_cand = route_mode.shape
    flat = np.zeros((n, k_cand, bank.flat_dim), dtype=np.float32)
    rng = np.random.default_rng(int(args.seed) + {"train": 12601, "val": 12602, "test": 12603}[split_name])
    for m, model in enumerate(bank.models):
        rows, cols = np.where(route_mode == m)
        if len(rows) == 0:
            continue
        pred = model.predict(x[rows]).astype(np.float32)
        pool = bank.error_pools[m] if len(bank.error_pools[m]) else bank.global_error_pool
        take = rng.integers(0, len(pool), size=len(rows))
        noise = pool[take].astype(np.float32)
        # First deterministic copy per selected top route is left noise-free.
        deterministic = cols < min(int(args.v12_top_route_modes), k_cand)
        noise[deterministic] = 0.0
        if float(args.v12_noise_jitter) > 0:
            scale = np.std(pool, axis=0, keepdims=True).astype(np.float32)
            noise = noise + rng.normal(size=noise.shape).astype(np.float32) * scale * float(args.v12_noise_jitter)
        flat[rows, cols] = pred + float(args.v12_error_noise_scale) * noise
    residual = flat.reshape(n, k_cand, args.max_horizon, 2).astype(np.float32)
    true_flat = audit.flatten_residual(residual_true)
    oracle_dist = np.mean((flat - true_flat[:, None, :]) ** 2, axis=-1).astype(np.float32)
    features, _ = seq.build_candidate_features(
        residual=residual,
        base=arrays_base,
        z_eps=np.zeros((n, k_cand, args.latent_dim), dtype=np.float32),
        logprob=np.log(np.maximum(np.take_along_axis(probs_use, route_mode, axis=1), EPS))[:, :, None],
        horizons=args.horizons,
    )
    mode_prob = np.zeros((n, k_cand, prior.k), dtype=np.float32)
    rr = np.arange(n)[:, None].repeat(k_cand, axis=1)
    cc = np.arange(k_cand)[None, :].repeat(n, axis=0)
    mode_prob[rr, cc, route_mode] = 1.0
    route_feat = seq.route_feature_packet(
        n=n,
        k=k_cand,
        mode_k=prior.k,
        route_mode=route_mode,
        mode_prior=np.take_along_axis(probs_use, route_mode, axis=1)[:, :, None],
        route_rank=(np.argsort(np.argsort(-probs_use, axis=1), axis=1)[rr, route_mode] / max(float(prior.k - 1), 1.0))[:, :, None],
    )
    features = np.concatenate([features, route_feat], axis=-1).astype(np.float32)
    return seq.CandidatePack(
        residual=residual,
        z=np.zeros((n, k_cand, args.latent_dim), dtype=np.float32),
        z_eps=np.zeros((n, k_cand, args.latent_dim), dtype=np.float32),
        logprob=np.log(np.maximum(np.take_along_axis(probs_use, route_mode, axis=1), EPS))[:, :, None].astype(np.float32),
        mode_prob=mode_prob,
        features=features,
        oracle_dist=oracle_dist,
        route_mode=route_mode,
    )


def endpoint_rows(arrays: audit.SplitArrays, pred: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def add_candidate_metrics(
    rows: list[dict[str, Any]],
    *,
    arrays: audit.SplitArrays,
    pack: seq.CandidatePack,
    name: str,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> None:
    extra = dict(extra or {})
    rows.extend(endpoint_rows(arrays, seq.mean_candidate_residual(pack), f"{name}_mean", args, {"stage": "candidate_mean", **extra}))
    for k in args.oracle_k:
        kk = min(int(k), pack.residual.shape[1])
        rows.extend(
            endpoint_rows(
                arrays,
                seq.oracle_residual(pack, arrays.residual_test, kk),
                f"{name}_oracle@{kk}",
                args,
                {"stage": "candidate_oracle", "oracle_k": kk, **extra},
            )
        )


def candidate_diagnostics(pack: seq.CandidatePack, labels_test: np.ndarray, name: str, args: argparse.Namespace) -> dict[str, Any]:
    dist = pack.oracle_dist
    take = np.argmin(dist[:, : min(pack.residual.shape[1], max(args.oracle_k))], axis=1)
    out: dict[str, Any] = {
        "variant": name,
        "candidate_k": int(pack.residual.shape[1]),
        "oracle_candidate_mean_mse": float(np.mean(np.min(dist, axis=1))),
        "oracle_candidate_median_mse": float(np.median(np.min(dist, axis=1))),
    }
    if pack.route_mode is not None:
        chosen = pack.route_mode[np.arange(len(take)), take]
        out["oracle_candidate_route_matches_true"] = float(np.mean(chosen == labels_test))
        for m in range(int(np.max(pack.route_mode)) + 1):
            out[f"route_frac_{m}"] = float(np.mean(pack.route_mode == m))
    return out


def write_report(out_dir: Path, args: argparse.Namespace, gate: pd.DataFrame, summary: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = ["# Route-Conditioned Generator v12 Report", ""]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append(f"- route variants: `{args.v12_variants}`")
    lines.append(f"- object grid: `{args.extra_feature_grid}`")
    lines.append("")
    lines.append("## Route Prior Gate")
    lines.append("")
    view = gate[gate["split"].eq("test")].sort_values("route_top3", ascending=False)
    lines.append(view.to_markdown(index=False))
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{h}")
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "oracle_k"] if c in sub.columns]
        lines.append(sub[cols].head(40).to_markdown(index=False))
        lines.append("")
    if not diag.empty:
        lines.append("## Candidate Diagnostics")
        lines.append("")
        cols = [c for c in diag.columns if not c.startswith("route_frac_")]
        lines.append(diag[cols].to_markdown(index=False))
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- Pass if route-prior expert oracle@K beats generic/old route-conditioned oracle at fixed K and controls degrade.")
    lines.append("- If route prior gate is strong but oracle does not improve, route labels are observable but this expert generator is not expressive enough.")
    lines.append("- If oracle improves but mean/final does not, move to Sequence Critic-Refiner over the narrowed route-conditioned cloud.")
    (out_dir / "route_conditioned_generator_v12_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)

    arrays, split = audit.prepare_data(args)
    extra_meta = rpr.attach_extra_feature_block(arrays, split, args)
    velocity_blocks, velocity_names = v10.build_velocity_blocks(split, max_cols=args.v10_velocity_max_cols)

    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    decomp = decomposition_features(student, arrays, blocks, args, device)

    labels = fit_route_labels(arrays, args)
    gate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    expert_meta: dict[str, Any] = {}

    variants = parse_strs(args.v12_variants)
    priors: dict[str, RoutePrior] = {}
    banks: dict[str, ExpertBank] = {}
    for variant in variants:
        xtr, xva, xte, names = build_route_feature_matrix(
            arrays=arrays,
            split=split,
            velocity_blocks=velocity_blocks,
            decomp=decomp,
            variant=variant,
            args=args,
        )
        prior = fit_prior_model(name=variant, xtr_raw=xtr, xva_raw=xva, xte_raw=xte, labels=labels, args=args, feature_names=names)
        priors[variant] = prior
        gate_rows.extend(prior_gate_rows(prior, labels))
        bank = fit_expert_bank(prior, labels, arrays, args)
        banks[variant] = bank
        expert_meta[variant] = bank.meta

        pack = generate_expert_candidates(
            name=variant,
            prior=prior,
            bank=bank,
            probs=prior.probs_test,
            x=prior.x_test,
            residual_true=arrays.residual_test,
            arrays_base=arrays.base_test,
            args=args,
            split_name="test",
        )
        add_candidate_metrics(metric_rows, arrays=arrays, pack=pack, name=f"v12_{variant}", args=args, extra={"variant": variant, "generator": "route_prior_expert"})
        diag_rows.append(candidate_diagnostics(pack, labels.test, f"v12_{variant}", args))

        if variant == variants[0]:
            for control, kwargs in [
                ("uniform_prior", {"uniform_prior": True}),
                ("shuffled_prior", {"shuffle_prior": True}),
            ]:
                cpack = generate_expert_candidates(
                    name=f"{variant}_{control}",
                    prior=prior,
                    bank=bank,
                    probs=prior.probs_test,
                    x=prior.x_test,
                    residual_true=arrays.residual_test,
                    arrays_base=arrays.base_test,
                    args=args,
                    split_name="test",
                    **kwargs,
                )
                add_candidate_metrics(
                    metric_rows,
                    arrays=arrays,
                    pack=cpack,
                    name=f"v12_{variant}_{control}",
                    args=args,
                    extra={"variant": f"{variant}_{control}", "generator": "route_prior_expert_control"},
                )
                diag_rows.append(candidate_diagnostics(cpack, labels.test, f"v12_{variant}_{control}", args))

    # Existing candidate clouds as references.
    for gen in parse_strs(args.v12_reference_generators):
        ref_args = copy.copy(args)
        ref_args.candidate_generator = gen
        ref_args.candidate_k = int(args.candidate_k)
        if gen in {"generic", "route_conditioned"}:
            cand = seq.generate_candidates(arrays, posterior, student, blocks, ref_args, split_name="test", device=device)
        elif gen == "hybrid":
            route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(ref_args, arrays, posterior, student, blocks, device)
            if route_log is not None and not route_log.empty:
                route_log.to_csv(args.out_dir / "reference_learned_route_train_log.csv", index=False)
            cand = rpr.generate_candidates_for_split(ref_args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)
        else:
            continue
        add_candidate_metrics(metric_rows, arrays=arrays, pack=cand, name=f"reference_{gen}", args=args, extra={"variant": gen, "generator": "reference"})
        diag_rows.append(candidate_diagnostics(cand, labels.test, f"reference_{gen}", args))

    gate_df = pd.DataFrame(gate_rows)
    summary = pd.DataFrame(metric_rows)
    diag = pd.DataFrame(diag_rows)
    if not gate_df.empty:
        gate_df.insert(0, "seed", int(args.seed))
        gate_df.insert(0, "dataset", str(args.dataset))
    if not summary.empty:
        summary.insert(0, "seed", int(args.seed))
        summary.insert(0, "dataset", str(args.dataset))
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))

    gate_df.to_csv(args.out_dir / "route_conditioned_generator_v12_prior_gate.csv", index=False)
    summary.to_csv(args.out_dir / "route_conditioned_generator_v12_summary.csv", index=False)
    diag.to_csv(args.out_dir / "route_conditioned_generator_v12_diagnostics.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "route_conditioned_generator_v12_meta.json").write_text(
        json.dumps(
            finite_json(
                {
                    "extra_feature": extra_meta,
                    "velocity_names": velocity_names,
                    "route_k": labels.k,
                    "expert_meta": expert_meta,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out_dir, args, gate_df, summary, diag)
    print(json.dumps({"out_dir": str(args.out_dir), "prior_rows": len(gate_df), "summary_rows": len(summary), "diag_rows": len(diag)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(out_dir=DEFAULT_OUT)
    parser.add_argument("--extra-feature-grid", type=Path, default=DEFAULT_OBJECT_GRID)
    parser.add_argument("--extra-feature-prefixes", type=str, default="oc_")
    parser.add_argument("--extra-feature-block-name", type=str, default="object_mask")
    parser.add_argument("--extra-feature-max-cols", type=int, default=256)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--v10-velocity-max-cols", type=int, default=160)
    parser.add_argument("--v12-variants", type=str, default="full,context_velocity,context_velocity_object,context_velocity_object_shuffled,context_velocity_object_same_frame,velocity_object,object")
    parser.add_argument("--v12-reference-generators", type=str, default="generic,route_conditioned")
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
    args = parser.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 900)
        args.max_val_rows = min(args.max_val_rows, 300)
        args.max_test_rows = min(args.max_test_rows, 400)
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.learned_route_epochs = min(args.learned_route_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = "4,8,16"
        args.v12_variants = "full,context_velocity,context_velocity_object_shuffled"
        args.v12_reference_generators = "generic,route_conditioned"
    run(args)


if __name__ == "__main__":
    main()
