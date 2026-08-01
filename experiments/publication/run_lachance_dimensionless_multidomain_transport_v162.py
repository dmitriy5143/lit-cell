#!/usr/bin/env python3
"""Dimensionless shared innovation transport across LaChance domains.

The base prior remains domain-specific. This runner tests whether one causal
innovation-transport kernel can be shared after train-only normalization:

* neighbour radii / median train nearest-neighbour distance;
* correction target / median train predictive Student-t scale;
* velocity and predictive-scale inputs in the same dimensionless units.

It reports per-domain, pooled, source-to-target, leave-one-domain-out and
few-shot calibration protocols. The LODO claim concerns the transport kernel,
not the domain-specific base forecaster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h  # noqa: E402
import run_lachance_semigroup_external_guards_v157i as v157i  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "lachance_dimensionless_multidomain_transport_v162"
DEFAULT_SPECS = (
    "MDCK_Bulk="
    "outputs/causal_innovation_state_space_v97_direct_h1_strict_bulk_seed42_2026-07-21/v97_direct.pt,"
    + v157i.DEFAULT_SPECS
)
EPS = 1e-8


@dataclass
class DomainData:
    name: str
    restored: v157i.RestoredRun
    payloads: dict[int, v157e.UpdatePayload]
    train_movies: list[int]
    validation_movies: list[int]
    test_movies: list[int]
    neighbour_scale: float
    predictive_scale: float
    local_scales: list[float]


@dataclass
class DimensionlessRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray


@dataclass
class AffineCalibration:
    coefficients: np.ndarray
    bound_z: float | None = None


@dataclass
class Selection:
    alpha: float
    bound_z: float
    validation_score: float
    validation_h1_gain_percent: float


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(item) for item in value]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_domain(
    restored: v157i.RestoredRun,
    scale_multipliers: list[float],
    max_scale_frames: int,
    control_seed: int,
) -> DomainData:
    neighbour_scale = v157i.train_neighbour_scale(
        restored.payloads,
        max_scale_frames,
    )
    local_scales = [
        neighbour_scale * multiplier for multiplier in scale_multipliers
    ]
    payloads = {
        movie: v157e.build_update_payload(
            split,
            payload,
            local_scales,
            control_seed + len(restored.dataset) * 1009 + movie * 17,
        )
        for movie, (split, payload) in restored.payloads.items()
    }
    train_movies = sorted(
        movie for movie, payload in payloads.items() if payload.split == "train"
    )
    validation_movies = sorted(
        movie
        for movie, payload in payloads.items()
        if payload.split == "validation"
    )
    test_movies = sorted(
        movie for movie, payload in payloads.items() if payload.split == "test"
    )
    train_scale = np.concatenate(
        [
            np.sqrt(np.mean(np.square(payloads[movie].base.scale), axis=1))
            for movie in train_movies
        ]
    )
    predictive_scale = float(np.median(train_scale))
    if not np.isfinite(predictive_scale) or predictive_scale <= 0:
        raise RuntimeError(f"Invalid predictive scale for {restored.dataset}")
    return DomainData(
        name=restored.dataset,
        restored=restored,
        payloads=payloads,
        train_movies=train_movies,
        validation_movies=validation_movies,
        test_movies=test_movies,
        neighbour_scale=neighbour_scale,
        predictive_scale=predictive_scale,
        local_scales=local_scales,
    )


def dimensionless_design(
    domain: DomainData,
    payload: v157e.UpdatePayload,
    control: str = "real",
) -> np.ndarray:
    update = np.asarray(getattr(payload, control), dtype=np.float64).copy()
    for index, name in enumerate(payload.feature_names):
        if name.endswith("effective_n"):
            update[:, index] = np.log1p(np.maximum(update[:, index], 0.0))
    uncertainty = payload.base.scale / max(domain.predictive_scale, EPS)
    velocity = payload.base.rows[["dx_px", "dy_px"]].to_numpy(np.float64)
    velocity /= max(domain.predictive_scale, EPS)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    return np.column_stack(
        [
            update,
            update * uncertainty[:, 0:1],
            update * uncertainty[:, 1:2],
            uncertainty,
            velocity,
            speed,
        ]
    )


def feature_moments(
    domains: dict[str, DomainData],
    source_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_value: np.ndarray | None = None
    sum_square: np.ndarray | None = None
    for name in source_names:
        domain = domains[name]
        for movie in domain.train_movies:
            matrix = dimensionless_design(
                domain,
                domain.payloads[movie],
            )
            if sum_value is None:
                sum_value = np.zeros(matrix.shape[1], dtype=np.float64)
                sum_square = np.zeros(matrix.shape[1], dtype=np.float64)
            sum_value += matrix.sum(axis=0)
            assert sum_square is not None
            sum_square += np.square(matrix).sum(axis=0)
            total += len(matrix)
    if total == 0 or sum_value is None or sum_square is None:
        raise RuntimeError("No source training rows")
    mean = sum_value / float(total)
    variance = sum_square / float(total) - np.square(mean)
    return mean, np.sqrt(np.maximum(variance, 1e-8))


def weighted_statistics(
    domains: dict[str, DomainData],
    source_names: list[str],
    weights_by_horizon: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = feature_moments(domains, source_names)
    feature_dim = len(mean) + 1
    gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    rhs = np.zeros((feature_dim, 2), dtype=np.float64)
    total_weight = 0.0
    total_rows = 0
    for name in source_names:
        domain = domains[name]
        for movie in domain.train_movies:
            payload = domain.payloads[movie]
            normalized = (
                dimensionless_design(domain, payload) - mean[None]
            ) / scale[None]
            per_step = np.column_stack(
                [normalized, np.ones(len(normalized), dtype=np.float64)]
            )
            residual = (
                payload.base.target - payload.base.mean
            ) / max(domain.predictive_scale, EPS)
            for horizon in v157e.HORIZONS:
                windows = v157e.consecutive_windows(
                    payload.base.rows,
                    horizon,
                )
                if not len(windows):
                    continue
                features = per_step[windows].sum(axis=1)
                target = residual[windows].sum(axis=1)
                weight = np.full(
                    len(windows),
                    weights_by_horizon[horizon] / max(len(windows), 1),
                    dtype=np.float64,
                )
                root = np.sqrt(weight)[:, None]
                weighted = features * root
                gram += weighted.T @ weighted
                rhs += weighted.T @ (target * root)
                total_weight += float(weight.sum())
                total_rows += len(weight)
    multiplier = total_rows / max(total_weight, EPS)
    gram *= multiplier
    rhs *= multiplier
    return mean, scale, gram, rhs


def solve_ridge(
    statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> DimensionlessRidge:
    mean, scale, gram, rhs = statistics
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return DimensionlessRidge(mean, scale, coefficients)


def raw_dimensionless_prediction(
    model: DimensionlessRidge,
    domain: DomainData,
    payload: v157e.UpdatePayload,
    control: str,
) -> np.ndarray:
    normalized = (
        dimensionless_design(domain, payload, control) - model.feature_mean[None]
    ) / model.feature_scale[None]
    augmented = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    return augmented @ model.coefficients


def apply_calibration(
    prediction: np.ndarray,
    calibration: AffineCalibration | None,
) -> np.ndarray:
    if calibration is None:
        return prediction
    design = np.column_stack(
        [prediction, np.ones(len(prediction), dtype=np.float64)]
    )
    return design @ calibration.coefficients


def predict(
    model: DimensionlessRidge,
    domain: DomainData,
    payload: v157e.UpdatePayload,
    control: str,
    bound_z: float,
    calibration: AffineCalibration | None = None,
) -> np.ndarray:
    raw = raw_dimensionless_prediction(model, domain, payload, control)
    calibrated = apply_calibration(raw, calibration)
    effective_bound = (
        bound_z
        if calibration is None or calibration.bound_z is None
        else calibration.bound_z
    )
    bounded = v157e.bounded_update(calibrated, effective_bound)
    return payload.base.mean + bounded * domain.predictive_scale


def validation_metrics(
    model: DimensionlessRidge,
    domains: dict[str, DomainData],
    source_names: list[str],
    bound_z: float,
) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for name in source_names:
        domain = domains[name]
        for movie in domain.validation_movies:
            payload = domain.payloads[movie]
            output.append(
                v157e.metric_rows(
                    payload,
                    predict(
                        model,
                        domain,
                        payload,
                        "real",
                        bound_z,
                    ),
                    "validation_real",
                    None,
                )
            )
    return output


def select_model(
    domains: dict[str, DomainData],
    source_names: list[str],
    objective: str,
    alphas: list[float],
    bounds_z: list[float],
) -> tuple[Selection, DimensionlessRidge, pd.DataFrame]:
    weights, h1_guard = v157h.OBJECTIVES[objective]
    statistics = weighted_statistics(domains, source_names, weights)
    records: list[dict[str, Any]] = []
    models: dict[float, DimensionlessRidge] = {}
    for alpha in alphas:
        model = solve_ridge(statistics, alpha)
        models[alpha] = model
        for bound in bounds_z:
            movie_metrics = validation_metrics(
                model,
                domains,
                source_names,
                bound,
            )
            score = float(
                np.mean(
                    [
                        v157h.score_metrics(metrics, weights)
                        for metrics in movie_metrics
                    ]
                )
            )
            h1_gain = float(
                np.mean(
                    [
                        next(
                            row["rmse_improvement_percent"]
                            for row in metrics
                            if int(row["horizon"]) == 1
                        )
                        for metrics in movie_metrics
                    ]
                )
            )
            h6_gain = float(
                np.mean(
                    [
                        next(
                            row["rmse_improvement_percent"]
                            for row in metrics
                            if int(row["horizon"]) == 6
                        )
                        for metrics in movie_metrics
                    ]
                )
            )
            records.append(
                {
                    "sources": "+".join(source_names),
                    "objective": objective,
                    "alpha": alpha,
                    "bound_z": bound,
                    "validation_score": score,
                    "validation_h1_gain_percent": h1_gain,
                    "validation_h6_gain_percent": h6_gain,
                    "validation_movies": len(movie_metrics),
                }
            )
    grid = pd.DataFrame(records)
    eligible = grid[
        grid.validation_h1_gain_percent.ge(-float(h1_guard))
    ]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        [
            "validation_score",
            "validation_h1_gain_percent",
            "bound_z",
            "alpha",
        ],
        ascending=[True, False, True, True],
    ).iloc[0]
    selection = Selection(
        alpha=float(best.alpha),
        bound_z=float(best.bound_z),
        validation_score=float(best.validation_score),
        validation_h1_gain_percent=float(best.validation_h1_gain_percent),
    )
    grid["selected"] = (
        grid.alpha.eq(selection.alpha)
        & grid.bound_z.eq(selection.bound_z)
    )
    return selection, models[selection.alpha], grid


def calibration_rows(
    domain: DomainData,
    movies: list[int],
    fraction: float,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for movie in movies:
        payload = domain.payloads[movie]
        rows = payload.base.rows.reset_index(drop=True)
        frames = np.asarray(sorted(rows["frame"].unique()), dtype=np.int64)
        count = max(1, int(math.ceil(len(frames) * fraction)))
        frame_indices = np.linspace(0, len(frames) - 1, count).round().astype(np.int64)
        keep_frames = set(int(frames[index]) for index in frame_indices)
        selected.append(
            np.flatnonzero(rows["frame"].isin(keep_frames).to_numpy())
        )
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def fit_affine_calibration(
    model: DimensionlessRidge,
    domain: DomainData,
    movies: list[int],
    fraction: float,
    ridge: float,
    objective: str,
) -> AffineCalibration:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    horizon_weights, _ = v157h.OBJECTIVES[objective]
    for movie in movies:
        payload = domain.payloads[movie]
        if fraction >= 1.0:
            indices = np.arange(len(payload.base.rows), dtype=np.int64)
        else:
            rows = payload.base.rows.reset_index(drop=True)
            frames = np.asarray(sorted(rows["frame"].unique()), dtype=np.int64)
            target_frames = max(6, int(math.ceil(len(frames) * fraction)))
            block_count = max(1, int(math.ceil(target_frames / 6.0)))
            centers = np.linspace(
                2,
                max(2, len(frames) - 4),
                block_count,
            ).round().astype(np.int64)
            keep: set[int] = set()
            for center in centers:
                start = max(0, min(int(center) - 2, len(frames) - 6))
                keep.update(int(value) for value in frames[start : start + 6])
            indices = np.flatnonzero(rows["frame"].isin(keep).to_numpy())
        raw = raw_dimensionless_prediction(model, domain, payload, "real")
        target = (
            payload.base.target - payload.base.mean
        ) / max(domain.predictive_scale, EPS)
        selected = np.zeros(len(payload.base.rows), dtype=bool)
        selected[indices] = True
        for horizon in v157e.HORIZONS:
            windows = v157e.consecutive_windows(payload.base.rows, horizon)
            if not len(windows):
                continue
            keep_windows = selected[windows].all(axis=1)
            windows = windows[keep_windows]
            if not len(windows):
                continue
            features.append(raw[windows].sum(axis=1))
            targets.append(target[windows].sum(axis=1))
            sample_weights.append(
                np.full(
                    len(windows),
                    horizon_weights[horizon] / len(windows),
                    dtype=np.float64,
                )
            )
    x = np.concatenate(features)
    y = np.concatenate(targets)
    weight = np.concatenate(sample_weights)
    weight *= len(weight) / max(float(weight.sum()), EPS)
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    root = np.sqrt(weight)[:, None]
    weighted_design = design * root
    weighted_target = y * root
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return AffineCalibration(coefficients)


def selected_frame_indices(
    payload: v157e.UpdatePayload,
    fraction: float,
) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(payload.base.rows), dtype=np.int64)
    rows = payload.base.rows.reset_index(drop=True)
    frames = np.asarray(sorted(rows["frame"].unique()), dtype=np.int64)
    target_frames = max(6, int(math.ceil(len(frames) * fraction)))
    block_count = max(1, int(math.ceil(target_frames / 6.0)))
    centers = np.linspace(
        2,
        max(2, len(frames) - 4),
        block_count,
    ).round().astype(np.int64)
    keep: set[int] = set()
    for center in centers:
        start = max(0, min(int(center) - 2, len(frames) - 6))
        keep.update(int(value) for value in frames[start : start + 6])
    return np.flatnonzero(rows["frame"].isin(keep).to_numpy())


def subset_metric_rows(
    payload: v157e.UpdatePayload,
    prediction: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, Any]]:
    selected_mask = np.zeros(len(payload.base.rows), dtype=bool)
    selected_mask[selected] = True
    records: list[dict[str, Any]] = []
    for horizon in v157e.HORIZONS:
        windows = v157e.consecutive_windows(payload.base.rows, horizon)
        windows = windows[selected_mask[windows].all(axis=1)]
        if not len(windows):
            continue
        target = payload.base.target[windows].sum(axis=1)
        candidate = prediction[windows].sum(axis=1)
        baseline = payload.base.mean[windows].sum(axis=1)
        component_rmse = float(
            np.sqrt(np.mean(np.square(target - candidate)))
        )
        baseline_rmse = float(
            np.sqrt(np.mean(np.square(target - baseline)))
        )
        records.append(
            {
                "horizon": horizon,
                "component_rmse": component_rmse,
                "baseline_component_rmse": baseline_rmse,
                "rmse_improvement_percent": (
                    100.0
                    * (baseline_rmse - component_rmse)
                    / max(baseline_rmse, EPS)
                ),
            }
        )
    return records


def select_utility_calibration(
    model: DimensionlessRidge,
    domain: DomainData,
    movies: list[int],
    fraction: float,
    objective: str,
    gains: list[float],
    bounds: list[float],
) -> tuple[AffineCalibration, pd.DataFrame]:
    weights, h1_guard = v157h.OBJECTIVES[objective]
    records: list[dict[str, Any]] = []
    raw_by_movie = {
        movie: raw_dimensionless_prediction(
            model,
            domain,
            domain.payloads[movie],
            "real",
        )
        for movie in movies
    }
    selected_by_movie = {
        movie: selected_frame_indices(domain.payloads[movie], fraction)
        for movie in movies
    }
    window_cache: dict[
        int,
        dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    for movie in movies:
        payload = domain.payloads[movie]
        selected_mask = np.zeros(len(payload.base.rows), dtype=bool)
        selected_mask[selected_by_movie[movie]] = True
        window_cache[movie] = {}
        for horizon in v157e.HORIZONS:
            windows = v157e.consecutive_windows(payload.base.rows, horizon)
            windows = windows[selected_mask[windows].all(axis=1)]
            if not len(windows):
                continue
            window_cache[movie][horizon] = (
                windows,
                payload.base.target[windows].sum(axis=1),
                payload.base.mean[windows].sum(axis=1),
            )
    for gain in gains:
        for bound in bounds:
            movie_metrics: list[list[dict[str, Any]]] = []
            for movie in movies:
                payload = domain.payloads[movie]
                correction = v157e.bounded_update(
                    raw_by_movie[movie] * float(gain),
                    float(bound),
                )
                prediction = (
                    payload.base.mean
                    + correction * domain.predictive_scale
                )
                metrics: list[dict[str, Any]] = []
                for horizon in v157e.HORIZONS:
                    cached = window_cache[movie].get(horizon)
                    if cached is None:
                        continue
                    windows, target, baseline = cached
                    candidate = prediction[windows].sum(axis=1)
                    component_rmse = float(
                        np.sqrt(np.mean(np.square(target - candidate)))
                    )
                    baseline_rmse = float(
                        np.sqrt(np.mean(np.square(target - baseline)))
                    )
                    metrics.append(
                        {
                            "horizon": horizon,
                            "component_rmse": component_rmse,
                            "baseline_component_rmse": baseline_rmse,
                            "rmse_improvement_percent": (
                                100.0
                                * (baseline_rmse - component_rmse)
                                / max(baseline_rmse, EPS)
                            ),
                        }
                    )
                if len(metrics) == len(v157e.HORIZONS):
                    movie_metrics.append(metrics)
            if not movie_metrics:
                continue
            records.append(
                {
                    "dataset": domain.name,
                    "objective": objective,
                    "fraction": fraction,
                    "gain": gain,
                    "bound_z": bound,
                    "score": float(
                        np.mean(
                            [
                                v157h.score_metrics(metrics, weights)
                                for metrics in movie_metrics
                            ]
                        )
                    ),
                    "h1_gain_percent": float(
                        np.mean(
                            [
                                next(
                                    row["rmse_improvement_percent"]
                                    for row in metrics
                                    if int(row["horizon"]) == 1
                                )
                                for metrics in movie_metrics
                            ]
                        )
                    ),
                    "h6_gain_percent": float(
                        np.mean(
                            [
                                next(
                                    row["rmse_improvement_percent"]
                                    for row in metrics
                                    if int(row["horizon"]) == 6
                                )
                                for metrics in movie_metrics
                            ]
                        )
                    ),
                    "movies": len(movie_metrics),
                }
            )
    grid = pd.DataFrame(records)
    if grid.empty:
        raise RuntimeError(f"No utility calibration rows for {domain.name}")
    calibration = calibration_from_grid(grid, float(h1_guard))
    selected_gain = float(calibration.coefficients[0, 0])
    selected_bound = float(calibration.bound_z)
    grid["selected"] = (
        grid.gain.eq(selected_gain)
        & grid.bound_z.eq(selected_bound)
    )
    return calibration, grid


def calibration_from_grid(
    grid: pd.DataFrame,
    h1_guard: float,
) -> AffineCalibration:
    eligible = grid[grid.h1_gain_percent.ge(-float(h1_guard))]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        ["score", "h1_gain_percent", "bound_z", "gain"],
        ascending=[True, False, True, True],
    ).iloc[0]
    coefficients = np.vstack(
        [
            np.eye(2, dtype=np.float64) * float(best.gain),
            np.zeros((1, 2), dtype=np.float64),
        ]
    )
    return AffineCalibration(
        coefficients=coefficients,
        bound_z=float(best.bound_z),
    )


def evaluate(
    model: DimensionlessRidge,
    selection: Selection,
    domains: dict[str, DomainData],
    target_names: list[str],
    objective: str,
    variant: str,
    controls: Iterable[str] = ("real",),
    calibrations: dict[str, AffineCalibration] | None = None,
    source_names: list[str] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name in target_names:
        domain = domains[name]
        calibration = None if calibrations is None else calibrations.get(name)
        for movie in domain.test_movies:
            payload = domain.payloads[movie]
            for control in controls:
                prediction = predict(
                    model,
                    domain,
                    payload,
                    control,
                    selection.bound_z,
                    calibration,
                )
                rows = v157e.metric_rows(
                    payload,
                    prediction,
                    control,
                    None,
                )
                for row in rows:
                    row.update(
                        {
                            "dataset": name,
                            "sequence": movie,
                            "variant": variant,
                            "control": control,
                            "objective": objective,
                            "sources": (
                                ""
                                if source_names is None
                                else "+".join(source_names)
                            ),
                            "selected_alpha": selection.alpha,
                            "selected_bound_z": selection.bound_z,
                            "predictive_scale": domain.predictive_scale,
                        }
                    )
                records.extend(rows)
    return pd.DataFrame(records)


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["objective", "variant", "control", "dataset", "horizon"],
            as_index=False,
        )
        .agg(
            sequences=("sequence", "nunique"),
            component_rmse=("component_rmse", "mean"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            gain_vs_prior_percent=("rmse_improvement_percent", "mean"),
            positive_sequences=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )


def aggregate_transfer(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            [
                "objective",
                "sources",
                "variant",
                "control",
                "dataset",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            sequences=("sequence", "nunique"),
            component_rmse=("component_rmse", "mean"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            gain_vs_prior_percent=("rmse_improvement_percent", "mean"),
            positive_sequences=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default=DEFAULT_SPECS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument("--alphas", default="1,10,100,1000,10000")
    parser.add_argument("--bounds-z", default="0.1,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--scale-multipliers", default="1,2,4,8")
    parser.add_argument("--max-scale-frames", type=int, default=300)
    parser.add_argument("--fewshot-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--calibration-ridge", type=float, default=10.0)
    parser.add_argument(
        "--calibration-gains",
        default="0,0.25,0.5,0.75,1,1.25,1.5",
    )
    parser.add_argument("--control-seed", type=int, default=162_001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = v157i.parse_specs(args.specs)
    device = v157e.device_from_cli(args.device)
    scale_multipliers = parse_floats(args.scale_multipliers)
    domains: dict[str, DomainData] = {}
    for name, checkpoint in specs.items():
        print(f"[v162] restore {name}", flush=True)
        restored = v157i.restore_run(name, checkpoint, device)
        domains[name] = prepare_domain(
            restored,
            scale_multipliers,
            int(args.max_scale_frames),
            int(args.control_seed),
        )
    domain_names = sorted(domains)
    objectives = [
        token.strip() for token in args.objectives.split(",") if token.strip()
    ]
    for objective in objectives:
        if objective not in v157h.OBJECTIVES:
            raise ValueError(f"Unknown objective {objective}")
    alphas = parse_floats(args.alphas)
    bounds = parse_floats(args.bounds_z)
    fewshot_fractions = parse_floats(args.fewshot_fractions)
    calibration_gains = parse_floats(args.calibration_gains)

    metric_frames: list[pd.DataFrame] = []
    grid_frames: list[pd.DataFrame] = []
    transfer_rows: list[pd.DataFrame] = []
    fewshot_rows: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    calibration_grids: list[pd.DataFrame] = []
    for objective in objectives:
        print(f"[v162] objective {objective}: per-domain", flush=True)
        per_domain_models: dict[str, tuple[Selection, DimensionlessRidge]] = {}
        for name in domain_names:
            selection, model, grid = select_model(
                domains,
                [name],
                objective,
                alphas,
                bounds,
            )
            grid["protocol"] = "per_domain"
            grid_frames.append(grid)
            per_domain_models[name] = (selection, model)
            frame = evaluate(
                model,
                selection,
                domains,
                [name],
                objective,
                "per_domain",
                controls=("real", "wrong_cell", "stale_time"),
                source_names=[name],
            )
            metric_frames.append(frame)
            selections.append(
                {
                    "objective": objective,
                    "protocol": "per_domain",
                    "sources": [name],
                    **selection.__dict__,
                }
            )

        print(f"[v162] objective {objective}: pooled", flush=True)
        pooled_selection, pooled_model, pooled_grid = select_model(
            domains,
            domain_names,
            objective,
            alphas,
            bounds,
        )
        pooled_grid["protocol"] = "pooled"
        grid_frames.append(pooled_grid)
        metric_frames.append(
            evaluate(
                pooled_model,
                pooled_selection,
                domains,
                domain_names,
                objective,
                "pooled_shared",
                controls=("real", "wrong_cell", "stale_time"),
                source_names=domain_names,
            )
        )
        pooled_calibrations: dict[str, AffineCalibration] = {}
        pooled_calibration_grids: dict[str, pd.DataFrame] = {}
        for name in domain_names:
            calibration, calibration_grid = select_utility_calibration(
                pooled_model,
                domains[name],
                domains[name].validation_movies,
                1.0,
                objective,
                calibration_gains,
                bounds,
            )
            pooled_calibrations[name] = calibration
            calibration_grid["protocol"] = "pooled_domain_calibrated"
            pooled_calibration_grids[name] = calibration_grid
            calibration_grids.append(calibration_grid)
        metric_frames.append(
            evaluate(
                pooled_model,
                pooled_selection,
                domains,
                domain_names,
                objective,
                "pooled_domain_calibrated",
                controls=("real",),
                calibrations=pooled_calibrations,
                source_names=domain_names,
            )
        )
        if objective == "h6_guard10":
            safe_calibrations = {
                name: calibration_from_grid(
                    pooled_calibration_grids[name],
                    0.5,
                )
                for name in domain_names
            }
            for name in domain_names:
                matching = pooled_calibration_grids[name]
                safe = safe_calibrations[name]
                matching["selected_h1_safe"] = (
                    matching.gain.eq(float(safe.coefficients[0, 0]))
                    & matching.bound_z.eq(float(safe.bound_z))
                )
            metric_frames.append(
                evaluate(
                    pooled_model,
                    pooled_selection,
                    domains,
                    domain_names,
                    objective,
                    "pooled_domain_h1safe",
                    controls=("real",),
                    calibrations=safe_calibrations,
                    source_names=domain_names,
                )
            )
        selections.append(
            {
                "objective": objective,
                "protocol": "pooled",
                "sources": domain_names,
                **pooled_selection.__dict__,
            }
        )

        print(f"[v162] objective {objective}: source-to-target", flush=True)
        for source in domain_names:
            selection, model = per_domain_models[source]
            frame = evaluate(
                model,
                selection,
                domains,
                domain_names,
                objective,
                "source_to_target",
                controls=("real",),
                source_names=[source],
            )
            transfer_rows.append(frame)

        print(f"[v162] objective {objective}: LODO", flush=True)
        for heldout in domain_names:
            sources = [name for name in domain_names if name != heldout]
            selection, model, grid = select_model(
                domains,
                sources,
                objective,
                alphas,
                bounds,
            )
            grid["protocol"] = "lodo"
            grid["heldout_domain"] = heldout
            grid_frames.append(grid)
            zero_shot = evaluate(
                model,
                selection,
                domains,
                [heldout],
                objective,
                "lodo_zero_shot",
                controls=("real", "wrong_cell", "stale_time"),
                source_names=sources,
            )
            metric_frames.append(zero_shot)
            selections.append(
                {
                    "objective": objective,
                    "protocol": "lodo",
                    "heldout_domain": heldout,
                    "sources": sources,
                    **selection.__dict__,
                }
            )
            validation_calibration, calibration_grid = select_utility_calibration(
                model,
                domains[heldout],
                domains[heldout].validation_movies,
                1.0,
                objective,
                calibration_gains,
                bounds,
            )
            calibration_grid["protocol"] = "lodo_validation_calibrated"
            calibration_grid["heldout_domain"] = heldout
            calibration_grids.append(calibration_grid)
            metric_frames.append(
                evaluate(
                    model,
                    selection,
                    domains,
                    [heldout],
                    objective,
                    "lodo_validation_calibrated",
                    controls=("real",),
                    calibrations={heldout: validation_calibration},
                    source_names=sources,
                )
            )
            if objective == "h6_guard10":
                safe_calibration = calibration_from_grid(
                    calibration_grid,
                    0.5,
                )
                calibration_grid["selected_h1_safe"] = (
                    calibration_grid.gain.eq(
                        float(safe_calibration.coefficients[0, 0])
                    )
                    & calibration_grid.bound_z.eq(
                        float(safe_calibration.bound_z)
                    )
                )
                metric_frames.append(
                    evaluate(
                        model,
                        selection,
                        domains,
                        [heldout],
                        objective,
                        "lodo_validation_h1safe",
                        controls=("real",),
                        calibrations={heldout: safe_calibration},
                        source_names=sources,
                    )
                )
            for fraction in fewshot_fractions:
                calibration, calibration_grid = select_utility_calibration(
                    model,
                    domains[heldout],
                    domains[heldout].train_movies,
                    fraction,
                    objective,
                    calibration_gains,
                    bounds,
                )
                calibration_grid["protocol"] = (
                    f"lodo_fewshot_{int(round(fraction * 100)):02d}"
                )
                calibration_grid["heldout_domain"] = heldout
                calibration_grids.append(calibration_grid)
                frame = evaluate(
                    model,
                    selection,
                    domains,
                    [heldout],
                    objective,
                    f"lodo_fewshot_{int(round(fraction * 100)):02d}",
                    controls=("real",),
                    calibrations={heldout: calibration},
                    source_names=sources,
                )
                metric_frames.append(frame)
                fewshot_rows.append(frame)
                if objective == "h6_guard10":
                    safe_calibration = calibration_from_grid(
                        calibration_grid,
                        0.5,
                    )
                    calibration_grid["selected_h1_safe"] = (
                        calibration_grid.gain.eq(
                            float(safe_calibration.coefficients[0, 0])
                        )
                        & calibration_grid.bound_z.eq(
                            float(safe_calibration.bound_z)
                        )
                    )
                    safe_variant = (
                        f"lodo_fewshot_{int(round(fraction * 100)):02d}_h1safe"
                    )
                    safe_frame = evaluate(
                        model,
                        selection,
                        domains,
                        [heldout],
                        objective,
                        safe_variant,
                        controls=("real",),
                        calibrations={heldout: safe_calibration},
                        source_names=sources,
                    )
                    metric_frames.append(safe_frame)
                    fewshot_rows.append(safe_frame)

    metrics = pd.concat(metric_frames, ignore_index=True)
    transfer = pd.concat(transfer_rows, ignore_index=True)
    fewshot = (
        pd.concat(fewshot_rows, ignore_index=True)
        if fewshot_rows
        else pd.DataFrame()
    )
    summary = aggregate(metrics)
    transfer_summary = aggregate_transfer(transfer)
    fewshot_summary = aggregate(fewshot) if not fewshot.empty else pd.DataFrame()
    controls = metrics[
        metrics.control.isin(["wrong_cell", "stale_time"])
    ].copy()

    contract_rows: list[dict[str, Any]] = []
    causal_frames: list[pd.DataFrame] = []
    for name, domain in domains.items():
        contract_rows.append(
            {
                "dataset": name,
                "base_checkpoint": str(domain.restored.checkpoint),
                "base_checkpoint_sha256": v157i.file_sha256(
                    domain.restored.checkpoint
                ),
                "train_sequences": json.dumps(domain.train_movies),
                "validation_sequences": json.dumps(domain.validation_movies),
                "test_sequences": json.dumps(domain.test_movies),
                "train_nearest_neighbour_scale": domain.neighbour_scale,
                "train_predictive_scale": domain.predictive_scale,
                "local_scale_multipliers": json.dumps(scale_multipliers),
                "space_unit": "native_track_coordinate",
                "time_unit": "frame",
                "physical_calibration_used": False,
                "transport_lodo_scope": (
                    "transport kernel only; base prior is domain-specific"
                ),
            }
        )
        if domain.validation_movies and domain.test_movies:
            causal_frames.append(
                v157e.build_causal_audit(
                    domain.payloads,
                    test_movie=domain.test_movies[0],
                    validation_movie=domain.validation_movies[0],
                    train_movies=domain.train_movies,
                ).assign(dataset=name)
            )
    causal = pd.concat(causal_frames, ignore_index=True)
    if int(causal.real_future_donor_violations.sum()) != 0:
        raise RuntimeError("Future donor found in v162")
    if int(causal.stale_future_or_nonstale_violations.sum()) != 0:
        raise RuntimeError("Invalid stale donor found in v162")
    if not bool(causal.coherent_wrong_packet.all()):
        raise RuntimeError("Wrong-cell coherence failed in v162")

    metrics.to_csv(args.out_dir / "v162_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "v162_dimensionless_summary.csv", index=False)
    controls.to_csv(args.out_dir / "v162_controls.csv", index=False)
    transfer.to_csv(args.out_dir / "v162_transfer_matrix_metrics.csv", index=False)
    transfer_summary.to_csv(
        args.out_dir / "v162_transfer_matrix.csv",
        index=False,
    )
    fewshot.to_csv(args.out_dir / "v162_fewshot_metrics.csv", index=False)
    fewshot_summary.to_csv(args.out_dir / "v162_fewshot.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(
        args.out_dir / "v162_validation_grid.csv",
        index=False,
    )
    pd.concat(calibration_grids, ignore_index=True).to_csv(
        args.out_dir / "v162_calibration_grid.csv",
        index=False,
    )
    pd.DataFrame(contract_rows).to_csv(
        args.out_dir / "v162_domain_contract.csv",
        index=False,
    )
    causal.to_csv(args.out_dir / "v162_causal_audit.csv", index=False)
    pd.DataFrame(selections).to_json(
        args.out_dir / "v162_selections.json",
        orient="records",
        indent=2,
    )

    h6 = summary[
        summary.horizon.eq(6) & summary.control.eq("real")
    ].sort_values(["objective", "dataset", "component_rmse"])
    elapsed = time.time() - started
    report = [
        "# v162 Dimensionless Multi-Domain Transport",
        "",
        "## h6 summary",
        "",
        h6.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation contract",
        "",
        "- LODO freezes the transport kernel on other domains.",
        "- Each domain still uses its own already-trained causal base prior.",
        "- Few-shot variants calibrate only a scalar gain and bound over the",
        "  frozen kernel; h1-safe rows impose a 0.5% calibration guard.",
        "- No physical-unit claim is made because the local feature contracts do",
        "  not provide independently verified pixel/time calibration for every domain.",
        "",
        f"Elapsed: `{elapsed / 3600.0:.2f} h`.",
    ]
    (args.out_dir / "v162_status_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "ok": True,
        "elapsed_sec": elapsed,
        "source_sha256": file_sha256(Path(__file__)),
        "domains": domain_names,
        "objectives": objectives,
        "selections": selections,
        "transport_contract": {
            "innovation_features": "standardized normal-score packet",
            "distance": "train nearest-neighbour scale",
            "correction": "train median predictive Student-t scale",
            "base_prior": "domain-specific and frozen",
            "lodo_scope": "shared transport kernel",
        },
    }
    (args.out_dir / "v162_run_manifest.json").write_text(
        json.dumps(v157e.finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
