#!/usr/bin/env python3
"""Decomposition-module audit for the LaChance trajectory stack.

This runner is intentionally *not* a final forecasting model.  It audits whether
the planned decomposition/teacher layer can play its intended role between the
clean-best backbone and a future trajectory generator / critic:

1. split residual motion into compact route components;
2. test whether those components are observable from causal context;
3. identify which feature families carry the component signal;
4. test whether a light latent generator can sample useful residual modes.

The audit covers PCA/EigenTrajectory, DCT/Fourier, Tucker/HOSVD diagnostics,
SAE/VAE-like residual autoencoders, seq-to-seq residual encoders, contrastive
route latents, per-component encoders with adaptive routing, and an MDN in the
decomposed latent space.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

try:
    from scipy.fft import dct, idct
except Exception:  # pragma: no cover
    dct = None  # type: ignore[assignment]
    idct = None  # type: ignore[assignment]

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    log_loss = None  # type: ignore[assignment]
    top_k_accuracy_score = None  # type: ignore[assignment]

try:
    import tensorly as tl
    from tensorly.decomposition import parafac

    tl.set_backend("numpy")
except Exception:  # pragma: no cover
    tl = None  # type: ignore[assignment]
    parafac = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_h1_sequence_raw_context_decoder as seq  # noqa: E402
from run_lachance_regime_magnitude_sequence_decoder import apply_train_position_norm  # noqa: E402


DEFAULT_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_raw_context_v2_grid_bulk_full60k_2026-06-19"
    / "raw_context_v2_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "decomposition_module_audit_2026-06-24"
EPS = 1e-8
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SplitArrays:
    x_train: dict[str, np.ndarray]
    x_val: dict[str, np.ndarray]
    x_test: dict[str, np.ndarray]
    steps_train: np.ndarray
    steps_val: np.ndarray
    steps_test: np.ndarray
    base_train: np.ndarray
    base_val: np.ndarray
    base_test: np.ndarray
    residual_train: np.ndarray
    residual_val: np.ndarray
    residual_test: np.ndarray
    feature_names: dict[str, list[str]]


def finite_json(value: Any) -> Any:
    return seq.finite_json(value)


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_ints(text: str) -> list[int]:
    return seq.parse_ints(text)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    sse = float(np.sum(np.square(y_true - y_pred)))
    sst = float(np.sum(np.square(y_true - np.mean(y_true, axis=0, keepdims=True))))
    return float(1.0 - sse / max(sst, EPS))


def gain_pct(base: float, value: float) -> float:
    return float((base - value) / max(abs(base), EPS) * 100.0)


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def target_steps(df: pd.DataFrame, max_h: int) -> np.ndarray:
    out = []
    for h in range(1, int(max_h) + 1):
        out.append(df[[f"step{h}_dx", f"step{h}_dy"]].fillna(0.0).to_numpy(np.float32))
    return np.stack(out, axis=1).astype(np.float32)


def base_step(df: pd.DataFrame) -> np.ndarray:
    return df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)


def endpoint_from_steps(steps: np.ndarray, h: int) -> np.ndarray:
    return np.sum(steps[:, : int(h), :], axis=1)


def base_rollout(base: np.ndarray, h: int) -> np.ndarray:
    return float(h) * np.asarray(base, dtype=np.float32)


def endpoint_metrics(
    *,
    steps_true: np.ndarray,
    base: np.ndarray,
    residual_pred: np.ndarray,
    horizons: list[int],
    label: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_steps = base[:, None, :] + np.asarray(residual_pred, dtype=np.float32)
    for h in horizons:
        y = endpoint_from_steps(steps_true, h)
        y_base = base_rollout(base, h)
        y_hat = endpoint_from_steps(pred_steps, h)
        row: dict[str, Any] = {
            "method": label,
            "horizon": int(h),
            "rmse": rmse(y, y_hat),
            "r2": r2_score_np(y, y_hat),
            "base_rmse": rmse(y, y_base),
            "base_r2": r2_score_np(y, y_base),
        }
        row["gain_vs_base_pct"] = gain_pct(row["base_rmse"], row["rmse"])
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def standardize_block(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    if x_train.shape[1] == 0 or StandardScaler is None:
        return x_train, x_val, x_test, None
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train).astype(np.float32)
    xva = scaler.transform(x_val).astype(np.float32)
    xte = scaler.transform(x_test).astype(np.float32)
    return (
        np.clip(np.nan_to_num(xtr), -8.0, 8.0).astype(np.float32),
        np.clip(np.nan_to_num(xva), -8.0, 8.0).astype(np.float32),
        np.clip(np.nan_to_num(xte), -8.0, 8.0).astype(np.float32),
        scaler,
    )


def select_by_variance(df: pd.DataFrame, cols: list[str], max_cols: int) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return []
    if len(cols) <= max_cols:
        return cols
    x = safe_matrix(df, cols)
    var = np.nan_to_num(np.var(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-var)[: int(max_cols)]
    return [cols[int(i)] for i in order]


def build_feature_blocks(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_features_per_family: int,
    max_all_features: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    all_cols = list(train.columns)
    traj = [c for c in seq.ifp.TRAJECTORY_FEATURES if c in train.columns]
    base = [
        c
        for c in ["x_norm", "y_norm", "frame_norm", "QUALITY", "proposal_norm", "dx_px", "dy_px"]
        if c in train.columns
    ]
    self_cols = list(dict.fromkeys(traj + base))
    morph_cols = [c for c in all_cols if c.startswith("ms_")]
    flow_cols = [c for c in all_cols if c.startswith("tf_")]
    raw_cols = [c for c in all_cols if c.startswith("rc_")]
    obs_cols = [c for c in all_cols if c.startswith("obs_")]
    boundary_tokens = ("boundary", "edge", "front", "normal", "tangent")
    crowd_tokens = (
        "density",
        "degree",
        "crowd",
        "neighbor",
        "neighbour",
        "nearest",
        "closing",
        "stretch",
        "align",
        "pressure",
        "diverg",
        "shear",
    )
    boundary_cols = [c for c in all_cols if any(t in c.lower() for t in boundary_tokens)]
    crowd_cols = [c for c in all_cols if any(t in c.lower() for t in crowd_tokens)]

    raw_cols = select_by_variance(train, raw_cols, max_features_per_family)
    morph_cols = select_by_variance(train, morph_cols, max_features_per_family)
    flow_cols = select_by_variance(train, flow_cols, max_features_per_family)
    obs_cols = select_by_variance(train, obs_cols, max_features_per_family)
    boundary_cols = select_by_variance(train, boundary_cols, max_features_per_family)
    crowd_cols = select_by_variance(train, crowd_cols, max_features_per_family)

    names: dict[str, list[str]] = {
        "self": self_cols,
        "morphology": morph_cols,
        "flow": flow_cols,
        "raw_context": raw_cols,
        "observability": obs_cols,
        "boundary": boundary_cols,
        "crowding": crowd_cols,
    }
    context_cols = list(
        dict.fromkeys(self_cols + morph_cols + flow_cols + raw_cols + obs_cols + boundary_cols + crowd_cols)
    )
    names["all_context"] = select_by_variance(train, context_cols, max_all_features)

    xtr: dict[str, np.ndarray] = {}
    xva: dict[str, np.ndarray] = {}
    xte: dict[str, np.ndarray] = {}
    for block, cols in names.items():
        tr = safe_matrix(train, cols)
        va = safe_matrix(val, cols)
        te = safe_matrix(test, cols)
        xtr[block], xva[block], xte[block], _ = standardize_block(tr, va, te)
    return xtr, xva, xte, names


def prepare_data(args: argparse.Namespace) -> tuple[SplitArrays, seq.SplitData]:
    features = pd.read_csv(args.features)
    full = seq.build_sequence_table(
        features=features,
        table_root=Path(args.table_root),
        dataset=args.dataset,
        max_horizon=args.max_horizon,
    )
    split = seq.make_split(full, parse_ints(args.train_seq), parse_ints(args.val_seq), parse_ints(args.test_seq), args.seed)
    split = apply_train_position_norm(split)
    split = seq.SplitData(
        train=seq.sample_rows(split.train, args.max_train_rows, args.seed + 11),
        val=seq.sample_rows(split.val, args.max_val_rows, args.seed + 23),
        test=seq.sample_rows(split.test, args.max_test_rows, args.seed + 37),
    )
    xtr, xva, xte, names = build_feature_blocks(
        split.train,
        split.val,
        split.test,
        max_features_per_family=args.max_features_per_family,
        max_all_features=args.max_all_features,
    )
    st_tr = target_steps(split.train, args.max_horizon)
    st_va = target_steps(split.val, args.max_horizon)
    st_te = target_steps(split.test, args.max_horizon)
    bs_tr = base_step(split.train)
    bs_va = base_step(split.val)
    bs_te = base_step(split.test)
    arrays = SplitArrays(
        x_train=xtr,
        x_val=xva,
        x_test=xte,
        steps_train=st_tr,
        steps_val=st_va,
        steps_test=st_te,
        base_train=bs_tr,
        base_val=bs_va,
        base_test=bs_te,
        residual_train=st_tr - bs_tr[:, None, :],
        residual_val=st_va - bs_va[:, None, :],
        residual_test=st_te - bs_te[:, None, :],
        feature_names=names,
    )
    return arrays, split


def flatten_residual(residual: np.ndarray) -> np.ndarray:
    return residual.reshape(len(residual), -1).astype(np.float32)


def unflatten_residual(flat: np.ndarray, max_h: int) -> np.ndarray:
    return flat.reshape(len(flat), int(max_h), 2).astype(np.float32)


def fit_pca_decomposition(
    arrays: SplitArrays, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    if PCA is None or StandardScaler is None or KMeans is None:
        raise RuntimeError("sklearn is required for PCA decomposition")
    ytr = flatten_residual(arrays.residual_train)
    yte = flatten_residual(arrays.residual_test)
    scaler = StandardScaler()
    ytr_s = scaler.fit_transform(ytr).astype(np.float32)
    yte_s = scaler.transform(yte).astype(np.float32)
    max_rank = min(max(args.latent_dims), ytr_s.shape[1], max(1, len(ytr_s) - 1))
    pca = PCA(n_components=max_rank, random_state=args.seed)
    ztr_all = pca.fit_transform(ytr_s).astype(np.float32)
    zte_all = pca.transform(yte_s).astype(np.float32)

    rows: list[dict[str, Any]] = []
    for k in args.latent_dims:
        kk = min(int(k), max_rank)
        recon_s = np.zeros_like(yte_s)
        recon_s += np.dot(zte_all[:, :kk], pca.components_[:kk])
        recon_s += pca.mean_
        recon = scaler.inverse_transform(recon_s).astype(np.float32)
        rows.extend(
            endpoint_metrics(
                steps_true=arrays.steps_test,
                base=arrays.base_test,
                residual_pred=unflatten_residual(recon, args.max_horizon),
                horizons=args.horizons,
                label="pca_eigentrajectory_target_recon",
                extra={
                    "latent_dim": kk,
                    "explained_variance": float(np.sum(pca.explained_variance_ratio_[:kk])),
                    "stage": "target_aware_reconstructability",
                },
            )
        )

    route_dim = min(int(args.route_latent_dim), max_rank)
    ztr = ztr_all[:, :route_dim]
    zte = zte_all[:, :route_dim]
    km = KMeans(n_clusters=int(args.route_k), n_init=20, random_state=args.seed)
    labels_tr = km.fit_predict(ztr)
    labels_te = km.predict(zte)
    meta = {
        "scaler": scaler,
        "pca": pca,
        "route_dim": route_dim,
        "kmeans": km,
        "explained_route_variance": float(np.sum(pca.explained_variance_ratio_[:route_dim])),
    }
    latent = {"train": ztr.astype(np.float32), "test": zte.astype(np.float32), "labels_train": labels_tr, "labels_test": labels_te}
    return rows, latent, meta


def dct_decomposition(arrays: SplitArrays, args: argparse.Namespace) -> list[dict[str, Any]]:
    if dct is None or idct is None:
        return [{"method": "dct_fourier", "stage": "skipped", "reason": "scipy.fft is unavailable"}]
    rows: list[dict[str, Any]] = []
    coeff_te = dct(arrays.residual_test, axis=1, norm="ortho")
    for k in sorted(set(min(int(v), args.max_horizon) for v in args.dct_ranks)):
        masked = np.zeros_like(coeff_te)
        masked[:, :k, :] = coeff_te[:, :k, :]
        recon = idct(masked, axis=1, norm="ortho").astype(np.float32)
        energy = float(np.sum(np.square(coeff_te[:, :k, :])) / max(np.sum(np.square(coeff_te)), EPS))
        rows.extend(
            endpoint_metrics(
                steps_true=arrays.steps_test,
                base=arrays.base_test,
                residual_pred=recon,
                horizons=args.horizons,
                label="dct_fourier_target_recon",
                extra={"time_rank": int(k), "energy_ratio": energy, "stage": "target_aware_reconstructability"},
            )
        )
    return rows


def hosvd_decomposition(arrays: SplitArrays, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xtr = arrays.residual_train.astype(np.float32)
    xte = arrays.residual_test.astype(np.float32)
    mean = xtr.mean(axis=0, keepdims=True)
    xtr_c = xtr - mean
    xte_c = xte - mean
    # Time-mode basis from unfolding H x (N*2).
    time_unfold = np.transpose(xtr_c, (1, 0, 2)).reshape(args.max_horizon, -1)
    u_t, s_t, _ = np.linalg.svd(time_unfold, full_matrices=False)
    # Coordinate-mode basis from 2 x (N*H).
    coord_unfold = np.transpose(xtr_c, (2, 0, 1)).reshape(2, -1)
    u_c, s_c, _ = np.linalg.svd(coord_unfold, full_matrices=False)
    total_energy = float(np.sum(np.square(xte_c)))
    for rt in sorted(set(min(int(v), args.max_horizon) for v in args.tucker_time_ranks)):
        rc = 2
        ut = u_t[:, :rt].astype(np.float32)
        uc = u_c[:, :rc].astype(np.float32)
        core = np.einsum("nhd,ht,dc->ntc", xte_c, ut, uc, optimize=True)
        recon_c = np.einsum("ntc,ht,dc->nhd", core, ut, uc, optimize=True)
        recon = (recon_c + mean).astype(np.float32)
        rec_energy = float(np.sum(np.square(recon_c)) / max(total_energy, EPS))
        rows.extend(
            endpoint_metrics(
                steps_true=arrays.steps_test,
                base=arrays.base_test,
                residual_pred=recon,
                horizons=args.horizons,
                label="tucker_hosvd_target_recon",
                extra={
                    "time_rank": int(rt),
                    "coord_rank": int(rc),
                    "energy_ratio": rec_energy,
                    "stage": "tensor_diagnostic",
                },
            )
        )
    return rows


def cp_decomposition(arrays: SplitArrays, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Target-aware CP/parafac diagnostic over the residual trajectory tensor.

    CP over a sample-by-time-by-coordinate tensor is not used as a deployable
    model because sample factors are target-derived.  It is still useful as a
    diagnostic answer to: "is the residual trajectory field itself low-rank
    enough to support a decomposition teacher?"
    """

    if parafac is None or tl is None:
        return [{"method": "cp_parafac_target_recon", "stage": "skipped", "reason": "tensorly is unavailable"}]
    rng = np.random.default_rng(args.seed + 4400)
    n = min(len(arrays.residual_test), int(args.cp_max_rows))
    idx = rng.choice(len(arrays.residual_test), size=n, replace=False) if n < len(arrays.residual_test) else np.arange(n)
    x = arrays.residual_test[idx].astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x_centered = x - mean
    rows: list[dict[str, Any]] = []
    for rank in args.cp_ranks:
        rr = int(rank)
        try:
            cp = parafac(
                x_centered,
                rank=rr,
                n_iter_max=int(args.cp_iter_max),
                init="svd",
                tol=1e-6,
                random_state=args.seed,
                verbose=False,
            )
            recon = (tl.cp_to_tensor(cp) + mean).astype(np.float32)
            energy = float(np.sum(np.square(recon - mean)) / max(np.sum(np.square(x_centered)), EPS))
            rows.extend(
                endpoint_metrics(
                    steps_true=arrays.steps_test[idx],
                    base=arrays.base_test[idx],
                    residual_pred=recon,
                    horizons=args.horizons,
                    label="cp_parafac_target_recon",
                    extra={
                        "cp_rank": rr,
                        "sample_rows": int(n),
                        "energy_ratio": energy,
                        "stage": "tensor_diagnostic_target_aware",
                    },
                )
            )
        except Exception as exc:  # pragma: no cover
            rows.append({"method": "cp_parafac_target_recon", "stage": "failed", "cp_rank": rr, "reason": str(exc)})
    return rows


def class_prob_from_model(model: Any, x: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        out = np.full((len(x), n_classes), 1e-6, dtype=np.float32)
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = proba[:, j]
        out /= np.maximum(out.sum(axis=1, keepdims=True), EPS)
        return out
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    out = np.full((len(x), n_classes), -20.0, dtype=np.float32)
    for j, cls in enumerate(classes):
        if j < scores.shape[1]:
            out[:, int(cls)] = scores[:, j]
    out -= out.max(axis=1, keepdims=True)
    proba = np.exp(out)
    proba /= np.maximum(proba.sum(axis=1, keepdims=True), EPS)
    return proba.astype(np.float32)


def route_metrics(y_true: np.ndarray, proba: np.ndarray, prefix: str = "") -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    labels = np.arange(proba.shape[1])
    out = {
        f"{prefix}route_acc": float(accuracy_score(y_true, pred)) if accuracy_score else float(np.mean(y_true == pred)),
        f"{prefix}route_nll": float(log_loss(y_true, np.clip(proba, 1e-6, 1.0), labels=labels)) if log_loss else float("nan"),
    }
    try:
        out[f"{prefix}route_top3"] = float(top_k_accuracy_score(y_true, proba, k=min(3, proba.shape[1]), labels=labels))
    except Exception:
        out[f"{prefix}route_top3"] = float("nan")
    return out


def fit_logistic_probe(
    xtr: np.ndarray, xte: np.ndarray, labels_tr: np.ndarray, labels_te: np.ndarray, *, seed: int
) -> tuple[dict[str, float], np.ndarray]:
    if LogisticRegression is None:
        raise RuntimeError("sklearn LogisticRegression is required")
    n_classes = int(max(np.max(labels_tr), np.max(labels_te)) + 1)
    model = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=seed)
    model.fit(xtr, labels_tr)
    proba = class_prob_from_model(model, xte, model.classes_, n_classes)
    return route_metrics(labels_te, proba), proba


def causal_route_probes(
    arrays: SplitArrays, latent: dict[str, np.ndarray], args: argparse.Namespace
) -> list[dict[str, Any]]:
    labels_tr = latent["labels_train"].astype(int)
    labels_te = latent["labels_test"].astype(int)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 1001)
    for block, xtr in arrays.x_train.items():
        xte = arrays.x_test[block]
        if xtr.shape[1] == 0:
            continue
        metrics, _ = fit_logistic_probe(xtr, xte, labels_tr, labels_te, seed=args.seed)
        rows.append({"probe": "route_logistic", "feature_block": block, "control": "real", **metrics})
        perm = rng.permutation(len(xtr))
        metrics_sh, _ = fit_logistic_probe(xtr[perm], xte, labels_tr, labels_te, seed=args.seed)
        rows.append({"probe": "route_logistic", "feature_block": block, "control": "row_shuffled_train", **metrics_sh})
    return rows


def feature_family_probe(
    arrays: SplitArrays, latent: dict[str, np.ndarray], args: argparse.Namespace
) -> list[dict[str, Any]]:
    z = latent["train"].astype(np.float32)
    rows: list[dict[str, Any]] = []
    for block, x in arrays.x_train.items():
        names = arrays.feature_names.get(block, [])
        if x.shape[1] == 0 or not names:
            continue
        for k in range(z.shape[1]):
            zz = z[:, k]
            zz = (zz - zz.mean()) / max(float(zz.std()), 1e-6)
            corr = np.abs(np.mean(((x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)) * zz[:, None], axis=0))
            order = np.argsort(-np.nan_to_num(corr))[: min(args.top_feature_count, len(names))]
            for rank, idx in enumerate(order, start=1):
                rows.append(
                    {
                        "block": block,
                        "component": int(k),
                        "rank": int(rank),
                        "feature": names[int(idx)],
                        "abs_corr": float(corr[int(idx)]),
                    }
                )
    return rows


class ResidualAE(nn.Module):
    def __init__(self, dim: int, latent_dim: int, hidden: int, *, vae: bool):
        super().__init__()
        self.vae = bool(vae)
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        out_dim = 2 * latent_dim if vae else latent_dim
        self.to_latent = nn.Linear(hidden, out_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.enc(x)
        z = self.to_latent(h)
        if not self.vae:
            return z, None
        mu, logvar = z.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -8.0, 5.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z, logvar = self.encode(x)
        if self.vae and logvar is not None:
            eps = torch.randn_like(z)
            zz = z + eps * torch.exp(0.5 * logvar)
        else:
            zz = z
        return self.dec(zz), z, logvar


class SeqResidualAE(nn.Module):
    def __init__(self, max_h: int, latent_dim: int, hidden: int):
        super().__init__()
        self.max_h = int(max_h)
        self.enc = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, max_h * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.enc(x)
        return self.to_z(h[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.dec(z).reshape(len(x), self.max_h, 2), z


def make_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, seed: int, shuffle: bool = True):
    n = len(x)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if y is None:
            yield x[idx]
        else:
            yield x[idx], y[idx]


def train_autoencoder(
    arrays: SplitArrays, args: argparse.Namespace, *, kind: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ytr = flatten_residual(arrays.residual_train)
    yte = flatten_residual(arrays.residual_test)
    scaler = StandardScaler()
    ytr_s = scaler.fit_transform(ytr).astype(np.float32)
    yte_s = scaler.transform(yte).astype(np.float32)
    vae = kind == "vae"
    model = ResidualAE(ytr_s.shape[1], args.ae_latent_dim, args.hidden_dim, vae=vae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for xb in make_loader(ytr_s, None, args.batch_size, args.seed + epoch):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            pred, z, logvar = model(xt)
            loss = F.mse_loss(pred, xt)
            if kind == "sae":
                loss = loss + float(args.sae_l1) * torch.mean(torch.abs(z))
            if vae and logvar is not None:
                kl = -0.5 * torch.mean(1.0 + logvar - z.pow(2) - logvar.exp())
                loss = loss + float(args.vae_beta) * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        yte_t = torch.as_tensor(yte_s, dtype=torch.float32, device=device)
        recon_s, zte_t, _ = model(yte_t)
        ztr_t, _ = model.encode(torch.as_tensor(ytr_s, dtype=torch.float32, device=device))
    recon = scaler.inverse_transform(recon_s.cpu().numpy()).astype(np.float32)
    ztr = ztr_t.cpu().numpy().astype(np.float32)
    zte = zte_t.cpu().numpy().astype(np.float32)
    rows = endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=unflatten_residual(recon, args.max_horizon),
        horizons=args.horizons,
        label=f"{kind}_posterior_recon",
        extra={"latent_dim": int(args.ae_latent_dim), "stage": "target_aware_reconstructability"},
    )

    # Causal prior into the learned posterior latent.
    xtr = arrays.x_train["all_context"]
    xte = arrays.x_test["all_context"]
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(xtr, ztr)
    z_pred = ridge.predict(xte).astype(np.float32)
    with torch.no_grad():
        recon_prior_s = model.dec(torch.as_tensor(z_pred, dtype=torch.float32, device=device)).cpu().numpy()
    recon_prior = scaler.inverse_transform(recon_prior_s).astype(np.float32)
    rows.extend(
        endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=unflatten_residual(recon_prior, args.max_horizon),
            horizons=args.horizons,
            label=f"{kind}_causal_latent_ridge",
            extra={"latent_dim": int(args.ae_latent_dim), "stage": "posterior_to_causal_gap"},
        )
    )
    rng = np.random.default_rng(args.seed + 333)
    xtr_sh = xtr[rng.permutation(len(xtr))]
    ridge_sh = Ridge(alpha=args.ridge_alpha)
    ridge_sh.fit(xtr_sh, ztr)
    z_sh = ridge_sh.predict(xte).astype(np.float32)
    with torch.no_grad():
        recon_sh_s = model.dec(torch.as_tensor(z_sh, dtype=torch.float32, device=device)).cpu().numpy()
    recon_sh = scaler.inverse_transform(recon_sh_s).astype(np.float32)
    rows.extend(
        endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=unflatten_residual(recon_sh, args.max_horizon),
            horizons=args.horizons,
            label=f"{kind}_causal_latent_row_shuffled",
            extra={"latent_dim": int(args.ae_latent_dim), "stage": "control"},
        )
    )
    return rows, {"train": ztr, "test": zte}


def train_seq_autoencoder(arrays: SplitArrays, args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ytr = arrays.residual_train.astype(np.float32)
    yte = arrays.residual_test.astype(np.float32)
    mean = ytr.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(ytr.std(axis=(0, 1), keepdims=True), 1e-6)
    ytr_s = np.clip((ytr - mean) / std, -8, 8).astype(np.float32)
    yte_s = np.clip((yte - mean) / std, -8, 8).astype(np.float32)
    model = SeqResidualAE(args.max_horizon, args.ae_latent_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for xb in make_loader(ytr_s, None, args.batch_size, args.seed + 700 + epoch):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            pred, _ = model(xt)
            loss = F.mse_loss(pred, xt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_te_s, zte_t = model(torch.as_tensor(yte_s, dtype=torch.float32, device=device))
        ztr_t = model.encode(torch.as_tensor(ytr_s, dtype=torch.float32, device=device))
    pred_te = (pred_te_s.cpu().numpy() * std + mean).astype(np.float32)
    ztr = ztr_t.cpu().numpy().astype(np.float32)
    xtr = arrays.x_train["all_context"]
    xte = arrays.x_test["all_context"]
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(xtr, ztr)
    z_pred = ridge.predict(xte).astype(np.float32)
    with torch.no_grad():
        pred_prior_s = model.dec(torch.as_tensor(z_pred, dtype=torch.float32, device=device)).cpu().numpy()
    pred_prior = (pred_prior_s.reshape(len(xte), args.max_horizon, 2) * std + mean).astype(np.float32)
    rows = endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred_te,
        horizons=args.horizons,
        label="seq2seq_gru_posterior_recon",
        extra={"latent_dim": int(args.ae_latent_dim), "stage": "target_aware_reconstructability"},
    )
    rows.extend(
        endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=pred_prior,
            horizons=args.horizons,
            label="seq2seq_gru_causal_latent_ridge",
            extra={"latent_dim": int(args.ae_latent_dim), "stage": "posterior_to_causal_gap"},
        )
    )
    return rows


class ContrastiveRouteNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(0.05), nn.Linear(hidden, hidden), nn.SiLU())
        self.emb = nn.Linear(hidden, emb_dim)
        self.cls = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        z = F.normalize(self.emb(h), dim=-1)
        return self.cls(z), z


def supervised_contrastive(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    sim = torch.matmul(z, z.T) / float(temperature)
    sim = sim - torch.max(sim, dim=1, keepdim=True).values.detach()
    mask = labels[:, None].eq(labels[None, :]).float()
    eye = torch.eye(len(z), device=z.device)
    mask = mask * (1.0 - eye)
    exp_sim = torch.exp(sim) * (1.0 - eye)
    log_prob = sim - torch.log(torch.sum(exp_sim, dim=1, keepdim=True) + 1e-8)
    denom = torch.sum(mask, dim=1)
    valid = denom > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=z.device)
    loss = -torch.sum(mask * log_prob, dim=1) / torch.clamp(denom, min=1.0)
    return torch.mean(loss[valid])


def train_contrastive_route(
    arrays: SplitArrays, latent: dict[str, np.ndarray], args: argparse.Namespace
) -> list[dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    xtr = arrays.x_train["all_context"].astype(np.float32)
    xte = arrays.x_test["all_context"].astype(np.float32)
    ytr = latent["labels_train"].astype(np.int64)
    yte = latent["labels_test"].astype(np.int64)
    n_classes = int(max(ytr.max(), yte.max()) + 1)
    model = ContrastiveRouteNet(xtr.shape[1], args.hidden_dim, args.contrastive_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in make_loader(xtr, ytr, args.batch_size, args.seed + 900 + epoch):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yt = torch.as_tensor(yb, dtype=torch.long, device=device)
            logits, z = model(xt)
            loss = F.cross_entropy(logits, yt) + float(args.contrastive_weight) * supervised_contrastive(z, yt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        logits, z = model(torch.as_tensor(xte, dtype=torch.float32, device=device))
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        emb = z.cpu().numpy()
    metrics = route_metrics(yte, proba)
    metrics["embedding_norm_mean"] = float(np.linalg.norm(emb, axis=1).mean())
    return [{"probe": "contrastive_route_net", "feature_block": "all_context", "control": "real", **metrics}]


class BlockRouterNet(nn.Module):
    def __init__(self, dims: dict[str, int], hidden: int, route_dim: int, n_classes: int):
        super().__init__()
        self.blocks = [k for k, d in dims.items() if d > 0]
        self.encoders = nn.ModuleDict(
            {k: nn.Sequential(nn.Linear(dims[k], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()) for k in self.blocks}
        )
        self.gate = nn.Sequential(nn.Linear(hidden * len(self.blocks), hidden), nn.SiLU(), nn.Linear(hidden, len(self.blocks)))
        self.cls = nn.Linear(hidden, n_classes)
        self.reg = nn.Linear(hidden, route_dim)

    def forward(self, xs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hs = [self.encoders[k](xs[k]) for k in self.blocks]
        stack = torch.stack(hs, dim=1)
        gate_logits = self.gate(torch.cat(hs, dim=-1))
        gates = torch.softmax(gate_logits, dim=-1)
        pooled = torch.sum(stack * gates[:, :, None], dim=1)
        return self.cls(pooled), self.reg(pooled), gates


def train_block_router(
    arrays: SplitArrays, latent: dict[str, np.ndarray], args: argparse.Namespace
) -> list[dict[str, Any]]:
    blocks = ["self", "morphology", "flow", "raw_context", "observability", "boundary", "crowding"]
    blocks = [b for b in blocks if arrays.x_train[b].shape[1] > 0]
    dims = {b: int(arrays.x_train[b].shape[1]) for b in blocks}
    if not blocks:
        return []
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ytr = latent["labels_train"].astype(np.int64)
    yte = latent["labels_test"].astype(np.int64)
    ztr = latent["train"].astype(np.float32)
    zte = latent["test"].astype(np.float32)
    n_classes = int(max(ytr.max(), yte.max()) + 1)
    model = BlockRouterNet(dims, args.hidden_dim, ztr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(ytr)
    rng = np.random.default_rng(args.seed + 1111)
    for epoch in range(args.epochs):
        order = np.arange(n)
        rng.shuffle(order)
        model.train()
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xs = {b: torch.as_tensor(arrays.x_train[b][idx], dtype=torch.float32, device=device) for b in blocks}
            yt = torch.as_tensor(ytr[idx], dtype=torch.long, device=device)
            zt = torch.as_tensor(ztr[idx], dtype=torch.float32, device=device)
            logits, zhat, gates = model(xs)
            entropy = -torch.mean(torch.sum(gates * torch.log(gates + 1e-8), dim=1))
            loss = F.cross_entropy(logits, yt) + 0.20 * F.mse_loss(zhat, zt) - 0.01 * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        xs_te = {b: torch.as_tensor(arrays.x_test[b], dtype=torch.float32, device=device) for b in blocks}
        logits, zhat, gates = model(xs_te)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        zhat_np = zhat.cpu().numpy()
        gates_np = gates.cpu().numpy()
    metrics = route_metrics(yte, proba)
    metrics["coeff_rmse"] = rmse(zte, zhat_np)
    rows: list[dict[str, Any]] = [{"probe": "adaptive_block_router", "feature_block": "component_encoders", "control": "real", **metrics}]
    for i, b in enumerate(blocks):
        rows.append({"probe": "adaptive_block_router_gate", "feature_block": b, "control": "real", "mean_gate": float(gates_np[:, i].mean()), "std_gate": float(gates_np[:, i].std())})
    return rows


class LatentMDN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, components: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.components = int(components)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, components)
        self.mu = nn.Linear(hidden, components * out_dim)
        self.logstd = nn.Linear(hidden, components * out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        pi = self.pi(h)
        mu = self.mu(h).reshape(len(x), self.components, self.out_dim)
        logstd = torch.clamp(self.logstd(h).reshape(len(x), self.components, self.out_dim), -5.0, 3.0)
        return pi, mu, logstd


def mdn_nll(pi: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y[:, None, :]
    log_prob = -0.5 * torch.sum(((y - mu) / torch.exp(logstd)).pow(2) + 2.0 * logstd + math.log(2.0 * math.pi), dim=-1)
    return -torch.mean(torch.logsumexp(torch.log_softmax(pi, dim=-1) + log_prob, dim=-1))


def train_latent_mdn(
    arrays: SplitArrays, pca_meta: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    pca: PCA = pca_meta["pca"]
    scaler: StandardScaler = pca_meta["scaler"]
    route_dim = int(pca_meta["route_dim"])
    ztr = pca.transform(scaler.transform(flatten_residual(arrays.residual_train))).astype(np.float32)[:, :route_dim]
    zte_true = pca.transform(scaler.transform(flatten_residual(arrays.residual_test))).astype(np.float32)[:, :route_dim]
    xtr = arrays.x_train["all_context"].astype(np.float32)
    xte = arrays.x_test["all_context"].astype(np.float32)
    model = LatentMDN(xtr.shape[1], route_dim, args.hidden_dim, args.mdn_components).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in make_loader(xtr, ztr, args.batch_size, args.seed + 1300 + epoch):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yt = torch.as_tensor(yb, dtype=torch.float32, device=device)
            pi, mu, logstd = model(xt)
            loss = mdn_nll(pi, mu, logstd, yt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pi, mu, logstd = model(torch.as_tensor(xte, dtype=torch.float32, device=device))
        probs = torch.softmax(pi, dim=-1).cpu().numpy()
        mu_np = mu.cpu().numpy()
        std_np = np.exp(logstd.cpu().numpy())
    rng = np.random.default_rng(args.seed + 2222)
    n = len(xte)
    samples = np.zeros((n, args.mdn_oracle_k, route_dim), dtype=np.float32)
    for i in range(n):
        comp = rng.choice(args.mdn_components, size=args.mdn_oracle_k, p=probs[i] / np.maximum(probs[i].sum(), EPS))
        eps = rng.normal(size=(args.mdn_oracle_k, route_dim)).astype(np.float32)
        samples[i] = mu_np[i, comp] + eps * std_np[i, comp]
    yte_s = scaler.transform(flatten_residual(arrays.residual_test)).astype(np.float32)
    best_residual = np.zeros_like(flatten_residual(arrays.residual_test))
    best_dist = np.full(n, np.inf, dtype=np.float32)
    for k in range(args.mdn_oracle_k):
        recon_s = np.zeros_like(yte_s)
        recon_s += np.dot(samples[:, k, :], pca.components_[:route_dim])
        recon_s += pca.mean_
        recon = scaler.inverse_transform(recon_s).astype(np.float32)
        dist = np.mean(np.square(recon - flatten_residual(arrays.residual_test)), axis=1)
        take = dist < best_dist
        best_dist[take] = dist[take]
        best_residual[take] = recon[take]
    mean_z = np.sum(probs[:, :, None] * mu_np, axis=1)
    mean_s = np.zeros_like(yte_s)
    mean_s += np.dot(mean_z, pca.components_[:route_dim])
    mean_s += pca.mean_
    mean_residual = scaler.inverse_transform(mean_s).astype(np.float32)
    rows = endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=unflatten_residual(mean_residual, args.max_horizon),
        horizons=args.horizons,
        label="latent_mdn_mean",
        extra={"latent_dim": route_dim, "components": int(args.mdn_components), "stage": "decomposed_latent_generator"},
    )
    rows.extend(
        endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=unflatten_residual(best_residual, args.max_horizon),
            horizons=args.horizons,
            label="latent_mdn_oracle",
            extra={"latent_dim": route_dim, "components": int(args.mdn_components), "oracle_k": int(args.mdn_oracle_k), "stage": "decomposed_latent_generator"},
        )
    )
    rows.append(
        {
            "method": "latent_mdn_coeff_probe",
            "stage": "decomposed_latent_generator",
            "coeff_rmse_mean": rmse(zte_true, mean_z),
        }
    )
    return rows


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    probes: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Decomposition Module Audit Status\n")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- horizons: `{','.join(map(str, args.horizons))}`")
    lines.append(f"- max_horizon: `{args.max_horizon}`")
    lines.append(f"- train/val/test seq: `{args.train_seq}` / `{args.val_seq}` / `{args.test_seq}`")
    lines.append("")
    if not summary.empty:
        lines.append("## Endpoint Reconstruction / Generator Checks")
        pivot = summary[summary["horizon"].notna()].copy() if "horizon" in summary.columns else pd.DataFrame()
        if not pivot.empty:
            top = pivot.sort_values(["horizon", "rmse"]).groupby("horizon").head(8)
            for _, row in top.iterrows():
                lines.append(
                    f"- h{int(row['horizon'])} `{row['method']}` RMSE={row['rmse']:.3f}, "
                    f"R2={row['r2']:.3f}, gain_vs_base={row['gain_vs_base_pct']:.2f}%"
                )
        lines.append("")
    if not probes.empty:
        lines.append("## Causal Observability / Routing")
        real = probes[probes.get("control", "").eq("real")].copy() if "control" in probes.columns else probes.copy()
        if not real.empty and "route_acc" in real.columns:
            top = real.sort_values("route_acc", ascending=False).head(10)
            for _, row in top.iterrows():
                lines.append(
                    f"- `{row.get('probe','probe')}` / `{row.get('feature_block','block')}`: "
                    f"acc={row.get('route_acc', float('nan')):.3f}, "
                    f"top3={row.get('route_top3', float('nan')):.3f}, "
                    f"NLL={row.get('route_nll', float('nan')):.3f}"
                )
        gate_rows = probes[probes.get("probe", "").eq("adaptive_block_router_gate")] if "probe" in probes.columns else pd.DataFrame()
        if not gate_rows.empty:
            lines.append("")
            lines.append("Adaptive-router mean gates:")
            for _, row in gate_rows.sort_values("mean_gate", ascending=False).iterrows():
                lines.append(f"- `{row['feature_block']}`: {row['mean_gate']:.3f} ± {row['std_gate']:.3f}")
        lines.append("")
    if not features.empty:
        lines.append("## Top Component-Linked Features")
        top = features.sort_values("abs_corr", ascending=False).head(20)
        for _, row in top.iterrows():
            lines.append(
                f"- component {int(row['component'])}, `{row['block']}`: `{row['feature']}` "
                f"|corr|={row['abs_corr']:.3f}"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- `target_aware_reconstructability` tells whether a decomposition basis can represent residual motion if future is available."
    )
    lines.append(
        "- `posterior_to_causal_gap` tells whether that target-aware decomposition can be predicted from causal context."
    )
    lines.append(
        "- `route_logistic`, contrastive route and adaptive-router probes test whether decomposed modes are observable and ablatable."
    )
    lines.append(
        "- `latent_mdn_oracle` is the first check that a generator in decomposed latent can produce useful individualized candidate modes."
    )
    lines.append("")
    lines.append("## Explicit Scope Notes")
    lines.append(
        "- CP/Tucker: Tucker/HOSVD and CP/parafac residual-tensor diagnostics are implemented as target-aware teacher-side checks. Full static CP is not used as the final neural module because the intended architecture uses learned route/component latents."
    )
    lines.append(
        "- Autoformer/FEDformer-style seasonality is not run because LaChance horizon is 1..6 steps and we do not have a seasonal long-context signal; DCT/Fourier covers the relevant short-horizon spectral diagnostic."
    )
    (out_dir / "decomposition_module_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument("--max-val-rows", type=int, default=8000)
    parser.add_argument("--max-test-rows", type=int, default=9000)
    parser.add_argument("--max-features-per-family", type=int, default=160)
    parser.add_argument("--max-all-features", type=int, default=384)
    parser.add_argument("--latent-dims", type=str, default="2,4,8,12")
    parser.add_argument("--route-latent-dim", type=int, default=8)
    parser.add_argument("--route-k", type=int, default=16)
    parser.add_argument("--dct-ranks", type=str, default="1,2,3,4,6")
    parser.add_argument("--tucker-time-ranks", type=str, default="1,2,3,4,6")
    parser.add_argument("--cp-ranks", type=str, default="2,4,8")
    parser.add_argument("--cp-max-rows", type=int, default=5000)
    parser.add_argument("--cp-iter-max", type=int, default=80)
    parser.add_argument("--top-feature-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ae-latent-dim", type=int, default=8)
    parser.add_argument("--contrastive-dim", type=int, default=32)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--sae-l1", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=2e-3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument("--mdn-oracle-k", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-mdn", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.latent_dims = parse_ints(args.latent_dims)
    args.dct_ranks = parse_ints(args.dct_ranks)
    args.tucker_time_ranks = parse_ints(args.tucker_time_ranks)
    args.cp_ranks = parse_ints(args.cp_ranks)
    if args.smoke:
        args.epochs = min(args.epochs, 8)
        args.max_train_rows = min(args.max_train_rows, 5000)
        args.max_val_rows = min(args.max_val_rows, 2000)
        args.max_test_rows = min(args.max_test_rows, 2500)
        args.max_all_features = min(args.max_all_features, 192)
        args.cp_max_rows = min(args.cp_max_rows, 1500)
        args.cp_iter_max = min(args.cp_iter_max, 40)

    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays, split = prepare_data(args)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "feature_blocks.json").write_text(json.dumps(arrays.feature_names, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    # Clean-best base rollout reference.
    zero_residual = np.zeros_like(arrays.residual_test)
    summary_rows.extend(
        endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=zero_residual,
            horizons=args.horizons,
            label="base_self_rollout_reference",
            extra={"stage": "reference"},
        )
    )

    pca_rows, pca_latent, pca_meta = fit_pca_decomposition(arrays, args)
    summary_rows.extend(pca_rows)
    summary_rows.extend(dct_decomposition(arrays, args))
    summary_rows.extend(hosvd_decomposition(arrays, args))
    summary_rows.extend(cp_decomposition(arrays, args))
    probe_rows.extend(causal_route_probes(arrays, pca_latent, args))
    feature_rows.extend(feature_family_probe(arrays, pca_latent, args))

    if not args.skip_neural:
        for kind in ["sae", "vae"]:
            rows, _ = train_autoencoder(arrays, args, kind=kind)
            summary_rows.extend(rows)
        summary_rows.extend(train_seq_autoencoder(arrays, args))
        probe_rows.extend(train_contrastive_route(arrays, pca_latent, args))
        probe_rows.extend(train_block_router(arrays, pca_latent, args))
    if not args.skip_mdn:
        summary_rows.extend(train_latent_mdn(arrays, pca_meta, args))

    summary = pd.DataFrame(summary_rows)
    probes = pd.DataFrame(probe_rows)
    features = pd.DataFrame(feature_rows)
    summary.to_csv(args.out_dir / "decomposition_module_summary.csv", index=False)
    probes.to_csv(args.out_dir / "decomposition_module_route_probe.csv", index=False)
    features.to_csv(args.out_dir / "decomposition_module_feature_families.csv", index=False)
    write_report(args.out_dir, args, summary, probes, features)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "probe_rows": len(probes), "feature_rows": len(features)}, indent=2))


if __name__ == "__main__":
    main()
