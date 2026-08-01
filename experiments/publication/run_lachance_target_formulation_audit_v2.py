#!/usr/bin/env python3
"""Target/formulation audit v2 for LaChance forecasting.

This runner is a pre-architecture diagnostic.  It checks whether our plateau is
caused by a poor target coordinate system rather than by the backbone itself.

New checks compared with ``run_lachance_target_formulation_audit.py``:

- endpoint/residual targets in local physical frames:
  own-velocity, tissue-flow and boundary normal/tangent frames;
- overcomplete local-frame coefficients;
- local polar direction + magnitude targets;
- route-mode recoverability probes for hidden future regimes;
- stratified error analysis by speed, density, boundary, flow coherence and
  target magnitude.

No future/target-derived quantity is used as an inference feature.  Target-aware
route labels are used only as diagnostic labels for the route-recoverability
probe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
    from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    RidgeClassifier = None  # type: ignore[assignment]
    SGDClassifier = None  # type: ignore[assignment]
    SGDRegressor = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_fast_feature_triage as triage  # noqa: E402
import run_lachance_h1_sequence_raw_context_decoder as seq  # noqa: E402
import run_lachance_target_formulation_audit as base  # noqa: E402


DEFAULT_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_raw_context_v2_grid_bulk_full60k_2026-06-19"
    / "raw_context_v2_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "target_formulation_audit_v2_2026-06-22"
EPS = 1e-8
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class Basis:
    first: np.ndarray
    second: np.ndarray


@dataclass
class LocalEndpointForm:
    name: str
    horizon: int
    base_train: np.ndarray
    base_val: np.ndarray
    base_test: np.ndarray
    target_train: np.ndarray
    target_val: np.ndarray
    target_test: np.ndarray
    y_test: np.ndarray
    reconstruct: str
    basis_test: list[Basis]
    family: str


def finite_json(value: Any) -> Any:
    return seq.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return seq.parse_ints(text)


def parse_strs(text: str) -> list[str]:
    return seq.parse_strs(text)


def safe_array(x: Any, clip: float = 1e6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -float(clip), float(clip)).astype(np.float32, copy=False)


def unit(v: np.ndarray, fallback: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    v = safe_array(v)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    fb = np.asarray(fallback, dtype=np.float32)[None, :]
    out = np.where(n > 1e-6, v / np.maximum(n, 1e-6), fb)
    return safe_array(out)


def perp(u: np.ndarray) -> np.ndarray:
    return safe_array(np.column_stack([-u[:, 1], u[:, 0]]))


def basis_from_df(df: pd.DataFrame, kind: str) -> Basis:
    if kind == "self":
        first = unit(df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32))
        return Basis(first=first, second=perp(first))
    if kind.startswith("flow"):
        radius = kind.replace("flow", "") or "128"
        candidates = [
            (f"tf_r{radius}_cur_center_u", f"tf_r{radius}_cur_center_v"),
            (f"tf_r{radius}_cur_u_mean", f"tf_r{radius}_cur_v_mean"),
            (f"tf_r{radius}_cur_u_median", f"tf_r{radius}_cur_v_median"),
        ]
        vec = None
        for u_col, v_col in candidates:
            if u_col in df.columns and v_col in df.columns:
                vec = df[[u_col, v_col]].fillna(0.0).to_numpy(np.float32)
                break
        if vec is None:
            vec = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
        first = unit(vec)
        return Basis(first=first, second=perp(first))
    if kind == "boundary":
        if {"obs_boundary_normal_x", "obs_boundary_normal_y", "obs_boundary_tangent_x", "obs_boundary_tangent_y"}.issubset(df.columns):
            normal = unit(df[["obs_boundary_normal_x", "obs_boundary_normal_y"]].fillna(0.0).to_numpy(np.float32))
            tangent = unit(df[["obs_boundary_tangent_x", "obs_boundary_tangent_y"]].fillna(0.0).to_numpy(np.float32), fallback=(0.0, 1.0))
            return Basis(first=normal, second=tangent)
        first = unit(df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32))
        return Basis(first=first, second=perp(first))
    raise ValueError(f"unknown basis kind={kind}")


def project_vec(y: np.ndarray, basis: Basis) -> np.ndarray:
    y = safe_array(y)
    return safe_array(np.column_stack([np.sum(y * basis.first, axis=1), np.sum(y * basis.second, axis=1)]))


def reconstruct_coeff(coeff: np.ndarray, basis: Basis) -> np.ndarray:
    coeff = safe_array(coeff)
    return safe_array(coeff[:, :1] * basis.first + coeff[:, 1:2] * basis.second)


def polar_target(y: np.ndarray, basis: Basis) -> np.ndarray:
    coeff = project_vec(y, basis)
    mag = np.linalg.norm(coeff, axis=1, keepdims=True)
    direction = coeff / np.maximum(mag, 1e-6)
    direction = np.where(mag > 1e-6, direction, 0.0)
    return safe_array(np.concatenate([direction, np.log1p(mag)], axis=1))


def reconstruct_polar(pred: np.ndarray, basis: Basis) -> np.ndarray:
    pred = safe_array(pred)
    direction = pred[:, :2]
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    mag = np.expm1(np.clip(pred[:, 2:3], -4.0, 6.0))
    coeff = direction * mag
    return reconstruct_coeff(coeff, basis)


def endpoint(df: pd.DataFrame, h: int) -> np.ndarray:
    return df[[f"target_h{h}_dx", f"target_h{h}_dy"]].to_numpy(np.float32)


def cv_base(df: pd.DataFrame, h: int) -> np.ndarray:
    return float(h) * df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)


def accel_base(df: pd.DataFrame, h: int) -> np.ndarray:
    v = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    if {"ax_px_s2", "ay_px_s2"}.issubset(df.columns):
        a = df[["ax_px_s2", "ay_px_s2"]].fillna(0.0).to_numpy(np.float32)
    else:
        a = np.zeros_like(v)
    return float(h) * (v + 0.5 * a)


def make_forms(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, horizons: list[int], basis_kinds: list[str]) -> list[LocalEndpointForm]:
    forms: list[LocalEndpointForm] = []
    for h in horizons:
        ytr, yva, yte = endpoint(train, h), endpoint(val, h), endpoint(test, h)
        zero_tr, zero_va, zero_te = [np.zeros_like(x) for x in (ytr, yva, yte)]
        forms.append(
            LocalEndpointForm(
                name=f"h{h}_global_direct",
                horizon=h,
                base_train=zero_tr,
                base_val=zero_va,
                base_test=zero_te,
                target_train=ytr,
                target_val=yva,
                target_test=yte,
                y_test=yte,
                reconstruct="vector",
                basis_test=[],
                family="global",
            )
        )
        for base_name, btr, bva, bte in [
            ("cv", cv_base(train, h), cv_base(val, h), cv_base(test, h)),
            ("accel", accel_base(train, h), accel_base(val, h), accel_base(test, h)),
        ]:
            forms.append(
                LocalEndpointForm(
                    name=f"h{h}_global_residual_to_{base_name}",
                    horizon=h,
                    base_train=btr,
                    base_val=bva,
                    base_test=bte,
                    target_train=ytr - btr,
                    target_val=yva - bva,
                    target_test=yte - bte,
                    y_test=yte,
                    reconstruct="vector",
                    basis_test=[],
                    family="global_residual",
                )
            )
        for kind in basis_kinds:
            btr = basis_from_df(train, kind)
            bva = basis_from_df(val, kind)
            bte = basis_from_df(test, kind)
            for base_name, gtr, gva, gte in [
                ("endpoint", zero_tr, zero_va, zero_te),
                ("residual_to_cv", cv_base(train, h), cv_base(val, h), cv_base(test, h)),
            ]:
                rtr, rva, rte = ytr - gtr, yva - gva, yte - gte
                forms.append(
                    LocalEndpointForm(
                        name=f"h{h}_{kind}_frame_{base_name}",
                        horizon=h,
                        base_train=gtr,
                        base_val=gva,
                        base_test=gte,
                        target_train=project_vec(rtr, btr),
                        target_val=project_vec(rva, bva),
                        target_test=project_vec(rte, bte),
                        y_test=yte,
                        reconstruct="frame",
                        basis_test=[bte],
                        family=f"{kind}_frame",
                    )
                )
                forms.append(
                    LocalEndpointForm(
                        name=f"h{h}_{kind}_polar_{base_name}",
                        horizon=h,
                        base_train=gtr,
                        base_val=gva,
                        base_test=gte,
                        target_train=polar_target(rtr, btr),
                        target_val=polar_target(rva, bva),
                        target_test=polar_target(rte, bte),
                        y_test=yte,
                        reconstruct="polar",
                        basis_test=[bte],
                        family=f"{kind}_polar",
                    )
                )
        if all(k in basis_kinds for k in ["self", "flow128", "boundary"]):
            btr_list = [basis_from_df(train, k) for k in ["self", "flow128", "boundary"]]
            bva_list = [basis_from_df(val, k) for k in ["self", "flow128", "boundary"]]
            bte_list = [basis_from_df(test, k) for k in ["self", "flow128", "boundary"]]
            gtr, gva, gte = cv_base(train, h), cv_base(val, h), cv_base(test, h)
            rtr, rva, rte = ytr - gtr, yva - gva, yte - gte
            forms.append(
                LocalEndpointForm(
                    name=f"h{h}_overcomplete_self_flow_boundary_residual_to_cv",
                    horizon=h,
                    base_train=gtr,
                    base_val=gva,
                    base_test=gte,
                    target_train=np.concatenate([project_vec(rtr, b) for b in btr_list], axis=1),
                    target_val=np.concatenate([project_vec(rva, b) for b in bva_list], axis=1),
                    target_test=np.concatenate([project_vec(rte, b) for b in bte_list], axis=1),
                    y_test=yte,
                    reconstruct="overcomplete",
                    basis_test=bte_list,
                    family="overcomplete",
                )
            )
    return forms


def reconstruct(form: LocalEndpointForm, pred_target: np.ndarray) -> np.ndarray:
    pred_target = safe_array(pred_target)
    if form.reconstruct == "vector":
        return safe_array(form.base_test + pred_target)
    if form.reconstruct == "frame":
        return safe_array(form.base_test + reconstruct_coeff(pred_target, form.basis_test[0]))
    if form.reconstruct == "polar":
        return safe_array(form.base_test + reconstruct_polar(pred_target, form.basis_test[0]))
    if form.reconstruct == "overcomplete":
        parts = []
        for i, basis in enumerate(form.basis_test):
            parts.append(reconstruct_coeff(pred_target[:, 2 * i : 2 * i + 2], basis))
        return safe_array(form.base_test + np.mean(parts, axis=0))
    raise ValueError(form.reconstruct)


def vector_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return base.vector_rmse(y, pred)


def vector_r2(y: np.ndarray, pred: np.ndarray) -> float:
    return base.vector_r2(y, pred)


def fit_predict_v2(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "ridge":
        if Ridge is None or StandardScaler is None:
            raise RuntimeError("sklearn Ridge/StandardScaler is unavailable")
        scaler = StandardScaler()
        xtr = scaler.fit_transform(safe_array(train_x))
        xva = scaler.transform(safe_array(val_x))
        xte = scaler.transform(safe_array(test_x))
        ytr = safe_array(train_y)
        yva = safe_array(val_y)
        best: tuple[float, float, Any] | None = None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha=float(alpha), solver="cholesky")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(xtr, ytr)
                with np.errstate(all="ignore"):
                    val_pred = safe_array(model.predict(xva))
            val_rmse = vector_rmse(yva.reshape(len(yva), -1, 1), val_pred.reshape(len(val_pred), -1, 1))
            if best is None or val_rmse < best[0]:
                best = (float(val_rmse), float(alpha), model)
        assert best is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                pred = safe_array(best[2].predict(xte))
        return pred, {"alpha": best[1], "val_target_rmse": best[0]}
    if model_name == "sgd_huber":
        if SGDRegressor is None or StandardScaler is None:
            raise RuntimeError("sklearn SGDRegressor/StandardScaler is unavailable")
        scaler = StandardScaler()
        xtr = scaler.fit_transform(safe_array(train_x))
        xva = scaler.transform(safe_array(val_x))
        xte = scaler.transform(safe_array(test_x))
        ytr = safe_array(train_y)
        yva = safe_array(val_y)
        preds = []
        val_preds = []
        for dim in range(ytr.shape[1]):
            reg = SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=1e-4,
                epsilon=1.35,
                learning_rate="adaptive",
                eta0=0.01,
                max_iter=2000,
                tol=1e-4,
                random_state=int(seed) + dim * 17,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reg.fit(xtr, ytr[:, dim])
            preds.append(reg.predict(xte))
            val_preds.append(reg.predict(xva))
        pred = safe_array(np.column_stack(preds))
        val_pred = safe_array(np.column_stack(val_preds))
        val_rmse = vector_rmse(yva.reshape(len(yva), -1, 1), val_pred.reshape(len(val_pred), -1, 1))
        return pred, {"val_target_rmse": val_rmse}
    raise ValueError(f"unknown model={model_name}")


def evaluate_form(
    *,
    dataset: str,
    seed: int,
    model: str,
    block: str,
    control: str,
    form: LocalEndpointForm,
    pred_target: np.ndarray,
    info: dict[str, Any],
) -> dict[str, Any]:
    pred = reconstruct(form, pred_target)
    base_pred = form.base_test
    return {
        "dataset": dataset,
        "seed": int(seed),
        "horizon": int(form.horizon),
        "target_form": form.name,
        "target_family": form.family,
        "reconstruct": form.reconstruct,
        "model": model,
        "feature_block": block,
        "control": control,
        "rmse_px": vector_rmse(form.y_test, pred),
        "base_rmse_px": vector_rmse(form.y_test, base_pred),
        "gain_vs_base_pct": base.gain_pct(vector_rmse(form.y_test, base_pred), vector_rmse(form.y_test, pred)),
        "r2": vector_r2(form.y_test, pred),
        "cosine": base.cosine(form.y_test, pred),
        "magnitude_ratio": base.magnitude_ratio(form.y_test, pred),
        "target_space_rmse": vector_rmse(form.target_test.reshape(len(form.target_test), -1, 1), pred_target.reshape(len(pred_target), -1, 1)),
        **info,
    }


def route_repr(df: pd.DataFrame, max_h: int, kind: str) -> np.ndarray:
    step = base.steps(df, max_h)
    cv = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    residual = step - cv[:, None, :]
    if kind == "global_residual":
        return safe_array(residual.reshape(len(df), -1))
    if kind == "endpoint_residual":
        return safe_array(residual.sum(axis=1))
    if kind.endswith("_frame_residual"):
        basis_kind = kind.replace("_frame_residual", "")
        b = basis_from_df(df, basis_kind)
        parts = [project_vec(residual[:, h, :], b) for h in range(max_h)]
        return safe_array(np.concatenate(parts, axis=1))
    if kind == "shape_unit":
        end = residual.sum(axis=1)
        mag = np.linalg.norm(end, axis=1, keepdims=True)
        return safe_array(residual.reshape(len(df), -1) / np.maximum(mag, 1.0))
    raise ValueError(f"unknown route repr={kind}")


def fit_route_labels(train_y: np.ndarray, val_y: np.ndarray, test_y: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if KMeans is None or StandardScaler is None:
        raise RuntimeError("sklearn KMeans/StandardScaler is unavailable")
    scaler = StandardScaler()
    ytr = scaler.fit_transform(safe_array(train_y))
    yva = scaler.transform(safe_array(val_y))
    yte = scaler.transform(safe_array(test_y))
    km = KMeans(n_clusters=int(k), n_init=12, random_state=int(seed))
    km.fit(ytr)
    return km.predict(ytr), km.predict(yva), km.predict(yte)


def fit_router(
    train_x: np.ndarray,
    train_label: np.ndarray,
    val_x: np.ndarray,
    val_label: np.ndarray,
    test_x: np.ndarray,
    *,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if StandardScaler is None:
        raise RuntimeError("sklearn StandardScaler is unavailable")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(safe_array(train_x))
    xva = scaler.transform(safe_array(val_x))
    xte = scaler.transform(safe_array(test_x))
    labels = np.arange(int(k))
    if RidgeClassifier is not None:
        best: tuple[float, float, Any] | None = None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            clf = RidgeClassifier(alpha=float(alpha), class_weight="balanced")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(xtr, train_label)
            scores = clf.decision_function(xva)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            pva = np.full((len(val_x), int(k)), -20.0, dtype=np.float32)
            pva[:, clf.classes_.astype(int)] = safe_array(scores, clip=20.0)
            pva = np.exp(pva - np.max(pva, axis=1, keepdims=True))
            pva = pva / np.maximum(pva.sum(axis=1, keepdims=True), 1e-8)
            try:
                nll = float(log_loss(val_label, pva, labels=labels))
            except Exception:
                nll = float("inf")
            if best is None or nll < best[0]:
                best = (nll, float(alpha), clf)
        assert best is not None
        clf = best[2]
        scores = clf.decision_function(xte)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        pte = np.full((len(test_x), int(k)), -20.0, dtype=np.float32)
        pte[:, clf.classes_.astype(int)] = safe_array(scores, clip=20.0)
        pte = np.exp(pte - np.max(pte, axis=1, keepdims=True))
        pte = pte / np.maximum(pte.sum(axis=1, keepdims=True), 1e-8)
        pva_dummy = np.zeros((len(val_x), int(k)), dtype=np.float32)
        return pva_dummy, pte, {"router": "ridge_classifier", "alpha": best[1], "val_mode_nll": best[0]}

    if SGDClassifier is not None:
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=2e-4,
            max_iter=900,
            tol=1e-3,
            class_weight="balanced",
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, train_label)
        pva = np.zeros((len(val_x), int(k)), dtype=np.float32)
        pte = np.zeros((len(test_x), int(k)), dtype=np.float32)
        pva[:, clf.classes_.astype(int)] = clf.predict_proba(xva)
        pte[:, clf.classes_.astype(int)] = clf.predict_proba(xte)
        try:
            val_nll = float(log_loss(val_label, pva, labels=labels))
        except Exception:
            val_nll = float("nan")
        return pva, pte, {"router": "sgd_logistic", "val_mode_nll": val_nll}

    if LogisticRegression is None:
        raise RuntimeError("sklearn LogisticRegression/SGDClassifier is unavailable")
    best: tuple[float, float, Any] | None = None
    for c in (0.05, 0.1, 0.3, 1.0, 3.0):
        clf = LogisticRegression(
            C=float(c),
            max_iter=350,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, train_label)
        pva = np.zeros((len(val_x), int(k)), dtype=np.float32)
        pva[:, clf.classes_.astype(int)] = clf.predict_proba(xva)
        try:
            nll = float(log_loss(val_label, pva, labels=labels))
        except Exception:
            nll = float("inf")
        if best is None or nll < best[0]:
            best = (nll, float(c), clf)
    assert best is not None
    clf = best[2]
    pva = np.zeros((len(val_x), int(k)), dtype=np.float32)
    pte = np.zeros((len(test_x), int(k)), dtype=np.float32)
    pva[:, clf.classes_.astype(int)] = clf.predict_proba(xva)
    pte[:, clf.classes_.astype(int)] = clf.predict_proba(xte)
    return pva, pte, {"C": best[1], "val_mode_nll": best[0]}


def route_metrics(label: np.ndarray, proba: np.ndarray, k: int) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    labels = np.arange(int(k))
    out = {"mode_acc": float(accuracy_score(label, pred))}
    for top in (2, 3, 5):
        if int(k) >= top:
            try:
                out[f"mode_top{top}_acc"] = float(top_k_accuracy_score(label, proba, k=top, labels=labels))
            except Exception:
                out[f"mode_top{top}_acc"] = float("nan")
    try:
        out["mode_nll"] = float(log_loss(label, proba, labels=labels))
    except Exception:
        out["mode_nll"] = float("nan")
    return out


def run_route_probe(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    blocks: dict[str, list[str]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    route_reprs = parse_strs(args.route_reprs)
    ks = parse_ints(args.route_ks)
    max_h = max(parse_ints(args.horizons))
    controls = parse_strs(args.controls)
    if not controls:
        controls = ["real"]
        if args.include_controls:
            controls += ["row_shuffled"]
    for repr_name in route_reprs:
        ytr = route_repr(train, max_h, repr_name)
        yva = route_repr(val, max_h, repr_name)
        yte = route_repr(test, max_h, repr_name)
        for k in ks:
            ltr, lva, lte = fit_route_labels(ytr, yva, yte, k, int(args.seed) + k * 31)
            for block_name, cols in blocks.items():
                for control in controls:
                    if control != "real" and block_name == "trajectory_only":
                        continue
                    xtr = base.block_matrix(train, cols, mode=control, seed=int(args.seed) + 211)
                    xva = base.block_matrix(val, cols, mode=control, seed=int(args.seed) + 223)
                    xte = base.block_matrix(test, cols, mode=control, seed=int(args.seed) + 227)
                    _, pte, info = fit_router(xtr, ltr, xva, lva, xte, k=k, seed=int(args.seed) + k)
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "seed": int(args.seed),
                            "route_repr": repr_name,
                            "route_k": int(k),
                            "feature_block": block_name,
                            "control": control,
                            "feature_dim": int(xtr.shape[1]),
                            **route_metrics(lte, pte, k),
                            **info,
                        }
                    )
    return pd.DataFrame(rows)


def quantile_groups(values: np.ndarray, q: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    try:
        edges = np.unique(np.nanquantile(values, np.linspace(0.0, 1.0, q + 1)))
    except Exception:
        edges = np.array([])
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.digitize(values, edges[1:-1], right=True), 0, q - 1)


def strat_values(df: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    vals: dict[str, np.ndarray] = {
        "target_magnitude": np.linalg.norm(y, axis=1),
        "self_speed": df.get("proposal_norm", pd.Series(np.linalg.norm(df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32), axis=1))).to_numpy(),
    }
    for name, candidates in {
        "density": ["obs_density_r120", "obs_density_r80", "obs_density_r240"],
        "boundary_dist": ["obs_boundary_dist"],
        "flow_coherence": ["obs_flow_r128_coherence", "obs_flow_r64_coherence", "obs_flow_r256_coherence"],
        "quality": ["QUALITY"],
    }.items():
        for col in candidates:
            if col in df.columns:
                vals[name] = df[col].fillna(0.0).to_numpy(np.float32)
                break
    return vals


def add_stratified_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
    form: LocalEndpointForm,
    pred: np.ndarray,
    block: str,
    model: str,
    test_df: pd.DataFrame,
) -> None:
    err = np.linalg.norm(safe_array(pred) - safe_array(form.y_test), axis=1)
    for factor, values in strat_values(test_df, form.y_test).items():
        groups = quantile_groups(values, 4)
        for g in sorted(set(int(x) for x in groups)):
            idx = groups == g
            if int(idx.sum()) < 20:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "horizon": int(form.horizon),
                    "target_form": form.name,
                    "feature_block": block,
                    "model": model,
                    "factor": factor,
                    "quantile": int(g) + 1,
                    "n": int(idx.sum()),
                    "factor_mean": float(np.mean(values[idx])),
                    "rmse_px": vector_rmse(form.y_test[idx], pred[idx]),
                    "mae_endpoint_px": float(np.mean(err[idx])),
                    "r2": vector_r2(form.y_test[idx], pred[idx]),
                    "cosine": base.cosine(form.y_test[idx], pred[idx]),
                    "magnitude_ratio": base.magnitude_ratio(form.y_test[idx], pred[idx]),
                }
            )


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    route_probe: pd.DataFrame,
    stratified: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    real = summary[summary["control"].eq("real")].copy()
    top = real.sort_values(["horizon", "rmse_px"]).groupby("horizon").head(12)
    by_family = real.sort_values(["horizon", "rmse_px"]).groupby(["horizon", "target_family"]).head(1)
    route_top = pd.DataFrame()
    if len(route_probe):
        route_top = route_probe.sort_values(["route_repr", "route_k", "mode_nll"]).groupby(["route_repr", "route_k"]).head(8)
    strat_top = pd.DataFrame()
    if len(stratified):
        # Worst strata by endpoint MAE, useful for deciding target/data changes.
        strat_top = stratified.sort_values(["horizon", "mae_endpoint_px"], ascending=[True, False]).groupby("horizon").head(12)
    lines = [
        "# Target Formulation Audit v2",
        "",
        "## Payload",
        "",
        "```json",
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top Formulations",
        "",
        top.to_markdown(index=False) if len(top) else "_No rows._",
        "",
        "## Best By Target Family",
        "",
        by_family.to_markdown(index=False) if len(by_family) else "_No rows._",
        "",
        "## Route Recoverability Probe",
        "",
        route_top.to_markdown(index=False) if len(route_top) else "_Route probe disabled or empty._",
        "",
        "## Worst Error Strata",
        "",
        strat_top.to_markdown(index=False) if len(strat_top) else "_Stratification disabled or empty._",
        "",
        "## Reading Guide",
        "",
        "- If local-frame targets beat global residuals, integrate that coordinate system into the clean backbone.",
        "- If route recoverability improves in a feature block, that block is useful even before RMSE improves.",
        "- If worst strata are concentrated near boundary/front/density regimes, target/data expansion should focus there.",
        "- If all local targets stay equivalent to global, changing the target is unlikely to be the breakthrough route.",
    ]
    (out_dir / "target_formulation_v2_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(args.features)
    horizons = parse_ints(args.horizons)
    max_h = max(horizons)
    full = seq.build_sequence_table(
        features=features,
        table_root=args.table_root,
        dataset=args.dataset,
        max_horizon=max_h,
    )
    split = seq.make_split(
        full,
        parse_ints(args.train_sequences),
        parse_ints(args.val_sequences),
        parse_ints(args.test_sequences),
        int(args.seed),
    )
    train = seq.sample_rows(split.train, int(args.max_train_rows), int(args.seed) + 11).reset_index(drop=True)
    val = seq.sample_rows(split.val, int(args.max_val_rows), int(args.seed) + 13).reset_index(drop=True)
    test = seq.sample_rows(split.test, int(args.max_test_rows), int(args.seed) + 17).reset_index(drop=True)

    blocks = base.feature_blocks(full, parse_strs(args.blocks))
    forms = make_forms(train, val, test, horizons, parse_strs(args.basis_kinds))
    if args.form_regex:
        pattern = re.compile(str(args.form_regex))
        forms = [f for f in forms if pattern.search(f.name) or pattern.search(f.family)]
        if not forms:
            raise ValueError(f"--form-regex matched no forms: {args.form_regex}")
    models = parse_strs(args.models)
    controls = parse_strs(args.controls)
    if not controls:
        controls = ["real"]
        if args.include_controls:
            controls += ["row_shuffled", "time_shuffled"]

    rows: list[dict[str, Any]] = []
    strat_rows: list[dict[str, Any]] = []
    strat_forms = set(parse_strs(args.stratify_forms))
    strat_blocks = set(parse_strs(args.stratify_blocks))

    for block_name, cols in blocks.items():
        for control in controls:
            if control != "real" and block_name == "trajectory_only":
                continue
            xtr = base.block_matrix(train, cols, mode=control, seed=int(args.seed) + 101)
            xva = base.block_matrix(val, cols, mode=control, seed=int(args.seed) + 103)
            xte = base.block_matrix(test, cols, mode=control, seed=int(args.seed) + 107)
            for model in models:
                for form in forms:
                    pred_target, info = fit_predict_v2(
                        model,
                        xtr,
                        form.target_train,
                        xva,
                        form.target_val,
                        xte,
                        seed=int(args.seed) + form.horizon * 19,
                    )
                    pred = reconstruct(form, pred_target)
                    rows.append(
                        evaluate_form(
                            dataset=args.dataset,
                            seed=int(args.seed),
                            model=model,
                            block=block_name,
                            control=control,
                            form=form,
                            pred_target=pred_target,
                            info={"feature_dim": int(xtr.shape[1]), **info},
                        )
                    )
                    if (
                        args.stratify
                        and control == "real"
                        and model == "ridge"
                        and (not strat_forms or form.name in strat_forms or form.family in strat_forms)
                        and (not strat_blocks or block_name in strat_blocks)
                    ):
                        add_stratified_rows(
                            strat_rows,
                            dataset=args.dataset,
                            seed=int(args.seed),
                            form=form,
                            pred=pred,
                            block=block_name,
                            model=model,
                            test_df=test,
                        )

    summary = pd.DataFrame(rows)
    route_probe = pd.DataFrame()
    if args.route_probe:
        route_probe = run_route_probe(train=train, val=val, test=test, blocks=blocks, args=args)
    return summary, route_probe, pd.DataFrame(strat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=seq.ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--models", default="ridge")
    parser.add_argument(
        "--blocks",
        default="trajectory_only,obs_context_core,ms_shape_tf_alignment_rc_core,ms_all_tf_all_rc,rc_all",
    )
    parser.add_argument("--basis-kinds", default="self,flow64,flow128,flow256,boundary")
    parser.add_argument("--form-regex", default="")
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-val-rows", type=int, default=8000)
    parser.add_argument("--max-test-rows", type=int, default=8000)
    parser.add_argument("--controls", default="")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--route-probe", action="store_true")
    parser.add_argument("--route-reprs", default="global_residual,self_frame_residual,flow128_frame_residual,boundary_frame_residual,shape_unit")
    parser.add_argument("--route-ks", default="8,16")
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--stratify-forms", default="global_residual,self_frame,flow128_frame,boundary_frame,overcomplete")
    parser.add_argument("--stratify-blocks", default="ms_all_tf_all_rc,ms_shape_tf_alignment_rc_core")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, route_probe, stratified = run(args)
    summary.to_csv(args.out_dir / "target_formulation_v2_summary.csv", index=False)
    route_probe.to_csv(args.out_dir / "route_recoverability_probe.csv", index=False)
    stratified.to_csv(args.out_dir / "stratified_error_probe.csv", index=False)
    write_report(args.out_dir, summary, route_probe, stratified, args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
