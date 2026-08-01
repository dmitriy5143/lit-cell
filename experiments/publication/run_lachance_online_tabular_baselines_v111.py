#!/usr/bin/env python3
"""v111: publication-grade same-contract online tabular baselines.

This runner evaluates deliberately simple regressors under the exact streaming
contract used by v97.  At frame t, every model receives only quantities that
are observable through t and emits a prediction for the completed displacement
t -> t+1.  Rolling h2/h4/h6 results are sums of consecutive predictions that
were each issued before their corresponding next observation arrived.

Two input/target contracts are kept separate:

``raw_coordinate``
    Causal coordinate/velocity features -> direct next displacement.  A
    validation-selected eta shrinks the prediction toward constant velocity.

``v52_anchor``
    The exact v97 OOF/held-out v52 h1 anchor plus the same causal features ->
    residual to that anchor.  Eta controls only the learned correction.

Future displacements are labels only.  Feature construction never reads
``target_steps`` or any target-derived error.  Scaling is fit on train movies
only; hyperparameters and eta are selected on the validation movie only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


EPS = 1e-8
KEYS = ["sequence", "frame", "track_id"]
DEFAULT_OUT = ROOT / "outputs" / "lachance_online_tabular_baselines_v111"


@dataclass
class FeaturePack:
    values: np.ndarray
    base_dim: int
    names: list[str]
    max_source_frame: np.ndarray


@dataclass
class FittedRegressor:
    name: str
    models: list[Any]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    config: dict[str, Any]

    def predict(self, x: np.ndarray) -> np.ndarray:
        zx = clipped_transform(self.x_scaler, x)
        if self.name == "hgbdt":
            zy = np.column_stack([model.predict(zx) for model in self.models])
        else:
            zy = np.asarray(self.models[0].predict(zx), dtype=np.float64)
        if zy.ndim == 1:
            zy = zy[:, None]
        prediction = self.y_scaler.inverse_transform(zy)
        if prediction.shape != (len(x), 2) or not np.isfinite(prediction).all():
            raise FloatingPointError(
                f"{self.name} produced invalid predictions: shape={prediction.shape}, "
                f"finite={bool(np.isfinite(prediction).all())}"
            )
        return prediction.astype(np.float32)


def safe(value: Any) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def parse_ints(value: str | Iterable[int]) -> list[int]:
    return [int(part) for part in parse_strings(value)]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    return [float(part) for part in parse_strings(value)]


def parse_horizon_weights(value: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for token in parse_strings(value):
        horizon, weight = token.split(":", maxsplit=1)
        result[int(horizon)] = float(weight)
    if not result or any(weight < 0 for weight in result.values()) or sum(result.values()) <= 0:
        raise ValueError(f"Invalid validation horizon weights: {value!r}")
    return result


def parse_hidden_grid(value: str) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    for token in str(value).split(";"):
        token = token.strip()
        if token:
            result.append(tuple(int(part) for part in token.lower().split("x") if part))
    if not result or any(not hidden or any(width <= 0 for width in hidden) for hidden in result):
        raise ValueError(f"Invalid MLP hidden grid: {value!r}")
    return result


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed))


def stable_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()[:16]


def subset_bundle(bundle: v84.AnchorBundle, max_rows: int) -> v84.AnchorBundle:
    """Take movie-balanced whole-track prefixes for an explicitly marked smoke run.

    A row prefix is not a valid streaming smoke sample for dense movies: it can
    contain thousands of cells from one frame and therefore no h2/h4/h6 chains.
    Whole tracks preserve the temporal contract at every requested horizon.
    """
    if int(max_rows) <= 0 or len(bundle.rows) <= int(max_rows):
        return bundle
    movies = sorted(int(value) for value in bundle.rows.sequence.unique())
    quota = max(1, int(max_rows) // len(movies))
    selected: list[int] = []
    for movie in movies:
        movie_rows = bundle.rows[bundle.rows.sequence.eq(movie)]
        movie_selected: list[int] = []
        for _track, raw_indices in movie_rows.groupby("track_id", sort=True).groups.items():
            indices = sorted(int(index) for index in raw_indices)
            if movie_selected and len(movie_selected) + len(indices) > quota:
                continue
            movie_selected.extend(indices)
            if len(movie_selected) >= quota:
                break
        if not movie_selected:
            movie_selected.extend(movie_rows.index.to_numpy(np.int64)[:quota].tolist())
        selected.extend(movie_selected)
    remaining = int(max_rows) - len(selected)
    if remaining > 0:
        used = set(selected)
        # Fill with whole additional tracks where possible.  A small overshoot
        # is preferable to destroying the last track's temporal chain.
        for (_sequence, _track), raw_indices in bundle.rows.groupby(["sequence", "track_id"], sort=True).groups.items():
            indices = sorted(int(index) for index in raw_indices if int(index) not in used)
            if not indices:
                continue
            selected.extend(indices)
            used.update(indices)
            if len(selected) >= int(max_rows):
                break
    index = np.asarray(sorted(set(selected)), dtype=np.int64)
    return v84.AnchorBundle(
        name=f"{bundle.name}_subset{len(index)}",
        rows=bundle.rows.iloc[index].reset_index(drop=True),
        anchor_residual=safe(bundle.anchor_residual[index]),
        base=safe(bundle.base[index]),
        target_steps=safe(bundle.target_steps[index]),
        meta={**bundle.meta, "v111_subset_rows": int(len(index))},
    )


def load_bundles(args: argparse.Namespace) -> tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle]:
    bundles = v85.load_anchor_cache(args.anchor_cache)
    return (
        subset_bundle(bundles[0], int(args.max_train_rows)),
        subset_bundle(bundles[1], int(args.max_val_rows)),
        subset_bundle(bundles[2], int(args.max_test_rows)),
    )


def lookup_for(bundle: v84.AnchorBundle) -> dict[tuple[int, int, int], int]:
    return {
        (int(sequence), int(frame), int(track)): index
        for index, (sequence, frame, track) in enumerate(
            bundle.rows[KEYS].itertuples(index=False, name=None)
        )
    }


def local_flow_features(bundle: v84.AnchorBundle, k: int) -> tuple[np.ndarray, list[str]]:
    rows = bundle.rows
    velocity = rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    result = np.zeros((len(rows), 8), dtype=np.float32)
    for (_sequence, _frame), raw_indices in rows.groupby(["sequence", "frame"], sort=True).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        if len(indices) <= 1:
            continue
        xy = rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float32)
        kk = min(int(k) + 1, len(indices))
        distance, neighbours = cKDTree(xy).query(xy, k=kk)
        if neighbours.ndim == 1:
            neighbours = neighbours[:, None]
            distance = distance[:, None]
        for local, global_index in enumerate(indices):
            chosen = indices[neighbours[local, 1:]]
            if len(chosen) == 0:
                continue
            flow = velocity[chosen].mean(axis=0)
            spread = velocity[chosen].std(axis=0)
            own = velocity[global_index]
            cosine = float(np.dot(own, flow) / max(np.linalg.norm(own) * np.linalg.norm(flow), EPS))
            result[global_index] = [
                flow[0], flow[1], spread[0], spread[1], flow[0] - own[0], flow[1] - own[1],
                cosine, float(np.mean(distance[local, 1:])),
            ]
    names = [
        f"flow_k{k}_vx", f"flow_k{k}_vy", f"flow_k{k}_std_vx", f"flow_k{k}_std_vy",
        f"flow_k{k}_minus_self_x", f"flow_k{k}_minus_self_y", f"flow_k{k}_alignment",
        f"flow_k{k}_mean_distance",
    ]
    return safe(result), names


def build_features(bundle: v84.AnchorBundle, history_lags: int, flow_k: int, mode: str) -> FeaturePack:
    if mode not in {"raw_coordinate", "v52_anchor"}:
        raise ValueError(f"Unknown input mode: {mode}")
    rows = bundle.rows.reset_index(drop=True)
    lookup = lookup_for(bundle)
    velocity = rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    xy = rows[["x_px", "y_px"]].to_numpy(np.float32)
    keys = rows[KEYS].to_numpy(np.int64)
    n = len(rows)
    lags = int(history_lags)
    history = np.zeros((n, lags, 2), dtype=np.float32)
    history_mask = np.zeros((n, lags), dtype=np.float32)
    source_frame = np.full((n, lags), -1, dtype=np.int64)
    track_age = np.zeros(n, dtype=np.float32)

    for index, (sequence, frame, track) in enumerate(keys):
        contiguous_age = 0
        for lag in range(lags):
            source = lookup.get((int(sequence), int(frame) - lag, int(track)))
            if source is None:
                continue
            history[index, lag] = velocity[source]
            history_mask[index, lag] = 1.0
            source_frame[index, lag] = int(rows.iloc[source].frame)
            if lag == contiguous_age:
                contiguous_age += 1
        track_age[index] = float(contiguous_age)

    columns: list[np.ndarray] = [xy, rows[["frame"]].to_numpy(np.float32), track_age[:, None]]
    names = ["x_px", "y_px", "frame", "contiguous_observed_age"]
    for lag in range(lags):
        values = history[:, lag]
        speed = np.linalg.norm(values, axis=1, keepdims=True)
        columns.extend([values, speed, history_mask[:, lag : lag + 1]])
        names.extend([f"v_lag{lag}_x", f"v_lag{lag}_y", f"speed_lag{lag}", f"mask_lag{lag}"])

    for lag in range(max(0, lags - 1)):
        acceleration = history[:, lag] - history[:, lag + 1]
        valid = history_mask[:, lag] * history_mask[:, lag + 1]
        acceleration *= valid[:, None]
        columns.extend([acceleration, np.linalg.norm(acceleration, axis=1, keepdims=True)])
        names.extend([f"accel_lag{lag}_x", f"accel_lag{lag}_y", f"accel_lag{lag}_magnitude"])

    current = history[:, 0]
    current_norm = np.linalg.norm(current, axis=1)
    turn_cosines: list[np.ndarray] = []
    turn_sines: list[np.ndarray] = []
    for lag in range(1, lags):
        past = history[:, lag]
        denominator = np.maximum(current_norm * np.linalg.norm(past, axis=1), EPS)
        cosine = np.sum(current * past, axis=1) / denominator
        sine = (current[:, 0] * past[:, 1] - current[:, 1] * past[:, 0]) / denominator
        valid = history_mask[:, 0] * history_mask[:, lag]
        cosine *= valid
        sine *= valid
        turn_cosines.append(cosine)
        turn_sines.append(sine)
        columns.extend([cosine[:, None], sine[:, None]])
        names.extend([f"turn_cos_lag{lag}", f"turn_sin_lag{lag}"])

    mask_sum = np.maximum(history_mask.sum(axis=1, keepdims=True), 1.0)
    mean_velocity = (history * history_mask[:, :, None]).sum(axis=1) / mask_sum
    centered = (history - mean_velocity[:, None, :]) * history_mask[:, :, None]
    std_velocity = np.sqrt(np.sum(centered**2, axis=1) / mask_sum)
    speed_history = np.linalg.norm(history, axis=2)
    mean_speed = np.sum(speed_history * history_mask, axis=1, keepdims=True) / mask_sum
    persistence = np.linalg.norm(np.sum(history * history_mask[:, :, None], axis=1), axis=1, keepdims=True)
    persistence /= np.maximum(np.sum(speed_history * history_mask, axis=1, keepdims=True), EPS)
    turn_mean = np.mean(np.column_stack(turn_cosines), axis=1, keepdims=True) if turn_cosines else np.zeros((n, 1), np.float32)
    turn_variability = np.std(np.column_stack(turn_sines), axis=1, keepdims=True) if turn_sines else np.zeros((n, 1), np.float32)
    columns.extend([mean_velocity, std_velocity, mean_speed, persistence, turn_mean, turn_variability])
    names.extend([
        "history_mean_vx", "history_mean_vy", "history_std_vx", "history_std_vy",
        "history_mean_speed", "directional_persistence", "turn_cosine_mean", "turn_sine_std",
    ])

    if lags >= 3:
        accel_now = (history[:, 0] - history[:, 1]) * (history_mask[:, 0] * history_mask[:, 1])[:, None]
        accel_prev = (history[:, 1] - history[:, 2]) * (history_mask[:, 1] * history_mask[:, 2])[:, None]
        jerk = accel_now - accel_prev
    else:
        jerk = np.zeros((n, 2), dtype=np.float32)
    columns.extend([jerk, np.linalg.norm(jerk, axis=1, keepdims=True)])
    names.extend(["jerk_x", "jerk_y", "jerk_magnitude"])

    local_flow, local_names = local_flow_features(bundle, int(flow_k))
    columns.append(local_flow)
    names.extend(local_names)
    base_values = safe(np.concatenate(columns, axis=1))
    base_dim = base_values.shape[1]

    if mode == "v52_anchor":
        anchor = safe(bundle.anchor_steps[:, 0])
        anchor_speed = np.linalg.norm(anchor, axis=1, keepdims=True)
        anchor_cos = np.sum(anchor * current, axis=1, keepdims=True) / np.maximum(
            anchor_speed * np.linalg.norm(current, axis=1, keepdims=True), EPS
        )
        anchor_features = np.concatenate([anchor, anchor_speed, anchor - current, anchor_cos], axis=1)
        base_values = safe(np.concatenate([base_values, anchor_features], axis=1))
        names.extend([
            "v52_anchor_dx", "v52_anchor_dy", "v52_anchor_speed",
            "v52_anchor_minus_velocity_x", "v52_anchor_minus_velocity_y", "v52_anchor_velocity_cosine",
        ])

    max_source = np.maximum(keys[:, 1], np.max(source_frame, axis=1))
    return FeaturePack(base_values, base_dim, names, max_source.astype(np.int64))


def same_frame_wrong_cell(values: np.ndarray, bundle: v84.AnchorBundle, base_dim: int, preserve_anchor: bool) -> np.ndarray:
    out = values.copy()
    for (_sequence, _frame), raw_indices in bundle.rows.groupby(["sequence", "frame"], sort=True).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        if len(indices) <= 1:
            continue
        source = np.roll(indices, 1)
        if preserve_anchor:
            out[indices, :base_dim] = values[source, :base_dim]
        else:
            out[indices] = values[source]
    return safe(out)


def causal_time_shuffle(values: np.ndarray, bundle: v84.AnchorBundle, base_dim: int, preserve_anchor: bool, seed: int) -> np.ndarray:
    """Replace state by a random earlier state of the same track, never a future row."""
    out = values.copy()
    rng = np.random.default_rng(int(seed))
    history: dict[tuple[int, int], list[int]] = {}
    order = np.lexsort((bundle.rows.track_id, bundle.rows.frame, bundle.rows.sequence))
    for index in order:
        row = bundle.rows.iloc[int(index)]
        key = (int(row.sequence), int(row.track_id))
        previous = history.setdefault(key, [])
        if previous:
            source = previous[int(rng.integers(0, len(previous)))]
            if preserve_anchor:
                out[index, :base_dim] = values[source, :base_dim]
            else:
                out[index] = values[source]
        else:
            if preserve_anchor:
                out[index, :base_dim] = 0.0
            else:
                out[index] = 0.0
        previous.append(int(index))
    return safe(out)


def clipped_transform(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(scaler.transform(values)), -10.0, 10.0).astype(np.float32)


def target_for_mode(bundle: v84.AnchorBundle, mode: str) -> np.ndarray:
    target = safe(bundle.target_steps[:, 0])
    if mode == "v52_anchor":
        return safe(target - bundle.anchor_steps[:, 0])
    return target


def reference_for_mode(bundle: v84.AnchorBundle, mode: str) -> np.ndarray:
    if mode == "v52_anchor":
        return safe(bundle.anchor_steps[:, 0])
    return safe(bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32))


def model_grid(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    if model_name == "ridge":
        return [{"alpha": alpha} for alpha in parse_floats(args.ridge_alphas)]
    if model_name == "hgbdt":
        return [
            {"learning_rate": learning_rate, "max_leaf_nodes": leaves, "l2_regularization": l2}
            for learning_rate, leaves, l2 in itertools.product(
                parse_floats(args.hgbdt_learning_rates),
                parse_ints(args.hgbdt_max_leaf_nodes),
                parse_floats(args.hgbdt_l2),
            )
        ]
    if model_name == "mlp":
        return [
            {"hidden_layer_sizes": hidden, "alpha": alpha}
            for hidden, alpha in itertools.product(parse_hidden_grid(args.mlp_hidden_grid), parse_floats(args.mlp_alphas))
        ]
    raise ValueError(f"Unknown model: {model_name}")


def fit_regressor(
    model_name: str,
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> FittedRegressor:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    zx = clipped_transform(x_scaler, x_train)
    zy = y_scaler.transform(y_train).astype(np.float32)
    if model_name == "ridge":
        model = Ridge(alpha=float(config["alpha"]), solver="lsqr")
        model.fit(zx, zy)
        models = [model]
    elif model_name == "hgbdt":
        models = []
        for axis in range(2):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(config["learning_rate"]),
                max_iter=int(args.hgbdt_max_iter),
                max_leaf_nodes=int(config["max_leaf_nodes"]),
                l2_regularization=float(config["l2_regularization"]),
                min_samples_leaf=int(args.hgbdt_min_samples_leaf),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=12,
                random_state=int(args.seed) + axis,
            )
            model.fit(zx, zy[:, axis])
            models.append(model)
    elif model_name == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(config["alpha"]),
            batch_size=min(int(args.mlp_batch_size), max(16, len(zx))),
            learning_rate_init=float(args.mlp_learning_rate),
            max_iter=int(args.mlp_max_iter),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=int(args.mlp_patience),
            random_state=int(args.seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(zx, zy)
        models = [model]
    else:
        raise ValueError(model_name)
    return FittedRegressor(model_name, models, x_scaler, y_scaler, dict(config))


def parameter_count(fitted: FittedRegressor) -> int:
    if fitted.name == "ridge":
        model = fitted.models[0]
        return int(np.size(model.coef_) + np.size(model.intercept_))
    if fitted.name == "mlp":
        model = fitted.models[0]
        return int(sum(np.size(value) for value in model.coefs_) + sum(np.size(value) for value in model.intercepts_))
    total = 0
    for model in fitted.models:
        predictors = getattr(model, "_predictors", [])
        for stage in predictors:
            for predictor in stage:
                total += int(len(getattr(predictor, "nodes", [])))
    return total


def chronological_predict(fitted: FittedRegressor, values: np.ndarray, bundle: v84.AnchorBundle) -> tuple[np.ndarray, float]:
    prediction = np.zeros((len(bundle.rows), 2), dtype=np.float32)
    started = time.perf_counter()
    # Frame-wise replay makes the publication contract explicit: all forecasts
    # for frame t are issued together and no frame t+1 label is assimilated.
    for (_sequence, _frame), raw_indices in bundle.rows.groupby(["sequence", "frame"], sort=True).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        prediction[indices] = fitted.predict(values[indices])
    return safe(prediction), time.perf_counter() - started


def decode_prediction(bundle: v84.AnchorBundle, mode: str, model_output: np.ndarray, eta: float) -> np.ndarray:
    reference = reference_for_mode(bundle, mode)
    if mode == "v52_anchor":
        return safe(reference + float(eta) * model_output)
    return safe(reference + float(eta) * (model_output - reference))


def rolling_examples(bundle: v84.AnchorBundle, prediction: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    lookup = lookup_for(bundle)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for index, (sequence, frame, track) in enumerate(bundle.rows[KEYS].itertuples(index=False, name=None)):
        chain = [lookup.get((int(sequence), int(frame) + offset, int(track))) for offset in range(int(horizon))]
        if any(item is None for item in chain):
            continue
        indices = np.asarray(chain, dtype=np.int64)
        targets.append(bundle.target_steps[index, : int(horizon)].sum(axis=0))
        predictions.append(prediction[indices].sum(axis=0))
    return safe(targets), safe(predictions)


def metric_row(bundle: v84.AnchorBundle, prediction: np.ndarray, horizon: int) -> dict[str, Any]:
    target, pred = rolling_examples(bundle, prediction, int(horizon))
    if not len(target):
        return {
            "horizon": int(horizon), "component_rmse": np.nan, "vector_rmse": np.nan,
            "component_r2_dx": np.nan, "component_r2_dy": np.nan, "component_r2_mean": np.nan,
            "vector_r2": np.nan, "r2": np.nan, "cosine": np.nan, "magnitude_ratio": np.nan, "n_rows": 0,
        }
    axis_r2: list[float] = []
    for axis in range(2):
        denominator = float(np.sum((target[:, axis] - target[:, axis].mean()) ** 2))
        axis_r2.append(1.0 - float(np.sum((target[:, axis] - pred[:, axis]) ** 2)) / max(denominator, EPS))
    vector_r2 = v84.vector_r2(target, pred)
    return {
        "horizon": int(horizon),
        "component_rmse": v84.component_rmse(target, pred),
        "vector_rmse": v84.vector_rmse(target, pred),
        "component_r2_dx": axis_r2[0],
        "component_r2_dy": axis_r2[1],
        "component_r2_mean": float(np.mean(axis_r2)),
        "vector_r2": vector_r2,
        "r2": vector_r2,
        "cosine": v84.cosine_mean(target, pred),
        "magnitude_ratio": v84.magnitude_ratio(target, pred),
        "n_rows": int(len(target)),
    }


def metric_rows(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    horizons: list[int],
    method: str,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    return [{"method": method, "contract": "streaming_receding_h1", **metric_row(bundle, prediction, h), **extra} for h in horizons]


def weighted_validation_score(
    bundle: v84.AnchorBundle,
    prediction: np.ndarray,
    weights: dict[int, float],
) -> float:
    scores: list[float] = []
    values: list[float] = []
    for horizon, weight in weights.items():
        target, pred = rolling_examples(bundle, prediction, horizon)
        if len(target):
            scores.append(v84.component_rmse(target, pred))
            values.append(float(weight))
    return float(np.average(scores, weights=values)) if scores else float("inf")


def tune_self_flow(
    train: v84.AnchorBundle,
    val: v84.AnchorBundle,
    test: v84.AnchorBundle,
    weights: dict[int, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    del train  # The reference has no learned parameters; validation selects k/mix only.
    sweep: list[dict[str, Any]] = []
    best: tuple[float, int, float] | None = None
    for k, mix in itertools.product(parse_ints(args.self_flow_k_grid), parse_floats(args.self_flow_mix_grid)):
        val_flow = local_flow_features(val, k)[0][:, :2]
        val_velocity = val.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
        prediction = safe((1.0 - mix) * val_velocity + mix * val_flow)
        score = weighted_validation_score(val, prediction, weights)
        sweep.append({"family": "self_flow", "k": k, "mix": mix, "validation_score": score})
        if best is None or score < best[0]:
            best = (score, int(k), float(mix))
    assert best is not None
    test_flow = local_flow_features(test, best[1])[0][:, :2]
    test_velocity = test.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    prediction = safe((1.0 - best[2]) * test_velocity + best[2] * test_flow)
    return prediction, {"validation_score": best[0], "k": best[1], "mix": best[2]}, sweep


def select_model(
    model_name: str,
    mode: str,
    train_pack: FeaturePack,
    val_pack: FeaturePack,
    train: v84.AnchorBundle,
    val: v84.AnchorBundle,
    weights: dict[int, float],
    args: argparse.Namespace,
) -> tuple[FittedRegressor, float, list[dict[str, Any]], float]:
    y_train = target_for_mode(train, mode)
    eta_grid = parse_floats(args.eta_grid)
    sweep: list[dict[str, Any]] = []
    winner: tuple[float, FittedRegressor, float] | None = None
    fit_total = 0.0
    for grid_index, config in enumerate(model_grid(model_name, args)):
        started = time.perf_counter()
        fitted = fit_regressor(model_name, config, train_pack.values, y_train, args)
        fit_seconds = time.perf_counter() - started
        fit_total += fit_seconds
        output, predict_seconds = chronological_predict(fitted, val_pack.values, val)
        for eta in eta_grid:
            decoded = decode_prediction(val, mode, output, eta)
            score = weighted_validation_score(val, decoded, weights)
            row = {
                "model": model_name, "input_mode": mode, "grid_index": grid_index,
                "hyperparameters": json.dumps(finite(config), sort_keys=True), "eta": float(eta),
                "validation_weighted_component_rmse": score, "fit_seconds": fit_seconds,
                "validation_predict_seconds": predict_seconds,
            }
            sweep.append(row)
            if winner is None or score < winner[0]:
                winner = (score, fitted, float(eta))
    assert winner is not None
    return winner[1], winner[2], sweep, fit_total


def causal_audit(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    packs: dict[tuple[str, str], FeaturePack],
    args: argparse.Namespace,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    split_names = ("train", "val", "test")
    train_movies = set(int(value) for value in bundles[0].rows.sequence.unique())
    val_movies = set(int(value) for value in bundles[1].rows.sequence.unique())
    test_movies = set(int(value) for value in bundles[2].rows.sequence.unique())
    overlap = (train_movies & val_movies) | (train_movies & test_movies) | (val_movies & test_movies)
    for split_name, bundle in zip(split_names, bundles):
        frame = bundle.rows.frame.to_numpy(np.int64)
        for mode in parse_strings(args.input_modes):
            pack = packs[(split_name, mode)]
            violations = int(np.sum(pack.max_source_frame > frame))
            # Sentinel: perturb every future label.  The feature digest must be
            # unchanged because the builder is forbidden from reading labels.
            rng = np.random.default_rng(int(args.seed) + 111)
            altered = v84.AnchorBundle(
                name=f"{bundle.name}_future_sentinel",
                rows=bundle.rows.copy(),
                anchor_residual=bundle.anchor_residual.copy(),
                base=bundle.base.copy(),
                target_steps=safe(rng.normal(0.0, 1000.0, bundle.target_steps.shape)),
                meta=bundle.meta,
            )
            sentinel = build_features(altered, int(args.history_lags), int(args.flow_k), mode)
            unchanged = bool(np.array_equal(pack.values, sentinel.values))
            records.append({
                "split": split_name,
                "input_mode": mode,
                "rows": len(bundle.rows),
                "movies": ",".join(map(str, sorted(int(value) for value in bundle.rows.sequence.unique()))),
                "feature_dim": pack.values.shape[1],
                "feature_digest": stable_digest(pack.values),
                "future_label_sentinel_unchanged": unchanged,
                "causal_source_violations": violations,
                "movie_split_overlap": ",".join(map(str, sorted(overlap))),
                "latest_allowed_source": "frame_t",
                "prediction_target": "completed_transition_t_to_t+1",
                "scaler_fit": "train_movies_only",
                "train_anchor_provenance": "movie-held-out OOF exact v97 cache" if split_name == "train" else "held-out exact v97 cache",
            })
    audit = pd.DataFrame(records)
    if int(audit.causal_source_violations.sum()) != 0:
        raise RuntimeError("Causal feature-source audit failed")
    if not bool(audit.future_label_sentinel_unchanged.all()):
        raise RuntimeError("Future-label sentinel changed inference features")
    if overlap:
        raise RuntimeError(f"Movie split overlap: {sorted(overlap)}")
    return audit


def report_markdown(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    selection: pd.DataFrame,
    audit: pd.DataFrame,
    runtime: pd.DataFrame,
    args: argparse.Namespace,
    elapsed: float,
) -> str:
    hmax = max(parse_ints(args.horizons))
    main = summary[summary.horizon.eq(hmax)].sort_values("component_rmse")
    control_h = controls[controls.horizon.eq(hmax)].sort_values(["input_mode", "model", "component_rmse"])
    selected = selection[selection.selected.eq(True)].copy()
    columns = [
        "method", "input_mode", "component_rmse", "vector_rmse", "vector_r2", "cosine",
        "magnitude_ratio", "n_rows",
    ]
    lines = [
        "# v111 Online Tabular Baselines",
        "",
        "## Contract",
        "",
        "- Forecast: one causal h1 displacement issued at frame t before observing t+1.",
        "- h2/h4/h6: sums of consecutive pre-observation h1 forecasts, not fixed-origin six-step forecasts.",
        "- Train anchors: movie-held-out OOF values from the exact v97 cache.",
        "- Scaling: train movies only. Hyperparameters and eta: validation movie only.",
        "- Future/target values are labels only; the target-randomization sentinel must leave every feature unchanged.",
        "",
        f"## Test Results At h{hmax}",
        "",
        main[columns].to_markdown(index=False),
        "",
        "## Selected Validation Configurations",
        "",
        selected[["model", "input_mode", "hyperparameters", "eta", "validation_weighted_component_rmse"]].to_markdown(index=False),
        "",
        "## Causal Controls At h%d" % hmax,
        "",
        control_h[["model", "input_mode", "control", "component_rmse", "vector_r2"]].to_markdown(index=False),
        "",
        "## Audit",
        "",
        audit[["split", "input_mode", "rows", "movies", "feature_dim", "future_label_sentinel_unchanged", "causal_source_violations"]].to_markdown(index=False),
        "",
        "## Runtime And Capacity",
        "",
        runtime[["model", "input_mode", "parameter_count", "parameter_unit", "grid_fit_seconds", "test_predict_seconds"]].to_markdown(index=False),
        "",
        "## Disclosure",
        "",
        f"- Ridge alpha grid: `{args.ridge_alphas}`.",
        f"- HGBDT grid: learning_rate=`{args.hgbdt_learning_rates}`, leaves=`{args.hgbdt_max_leaf_nodes}`, l2=`{args.hgbdt_l2}`.",
        f"- MLP grid: hidden=`{args.mlp_hidden_grid}`, alpha=`{args.mlp_alphas}`.",
        f"- Eta grid: `{args.eta_grid}`; validation horizon weights: `{args.validation_horizon_weights}`.",
        f"- Smoke mode: `{bool(args.smoke)}`. Total elapsed: `{elapsed:.2f}` seconds.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    if args.smoke:
        args.max_train_rows = min_positive(int(args.max_train_rows), 1200)
        args.max_val_rows = min_positive(int(args.max_val_rows), 500)
        args.max_test_rows = min_positive(int(args.max_test_rows), 700)
        args.ridge_alphas = "10,100"
        args.hgbdt_learning_rates = "0.08"
        args.hgbdt_max_leaf_nodes = "15"
        args.hgbdt_l2 = "1"
        args.hgbdt_max_iter = min(int(args.hgbdt_max_iter), 40)
        args.mlp_hidden_grid = "64"
        args.mlp_alphas = "0.0001"
        args.mlp_max_iter = min(int(args.mlp_max_iter), 35)
        args.eta_grid = "0,0.5,1"

    horizons = parse_ints(args.horizons)
    weights = parse_horizon_weights(args.validation_horizon_weights)
    modes = parse_strings(args.input_modes)
    models = parse_strings(args.models)
    unknown_modes = sorted(set(modes) - {"raw_coordinate", "v52_anchor"})
    unknown_models = sorted(set(models) - {"ridge", "hgbdt", "mlp"})
    if unknown_modes or unknown_models:
        raise ValueError(f"Unknown modes={unknown_modes}, models={unknown_models}")
    if not set(weights).issubset(set(horizons)):
        raise ValueError("Every validation horizon must be listed in --horizons")

    train, val, test = load_bundles(args)
    bundles = (train, val, test)
    for split_name, bundle in (("val", val), ("test", test)):
        probe = np.zeros((len(bundle.rows), 2), dtype=np.float32)
        missing = [horizon for horizon in horizons if len(rolling_examples(bundle, probe, horizon)[0]) == 0]
        if missing:
            raise RuntimeError(
                f"{split_name} has no chronological rolling windows for horizons {missing}; "
                "increase the row limit or preserve whole tracks"
            )
    packs: dict[tuple[str, str], FeaturePack] = {}
    for split_name, bundle in zip(("train", "val", "test"), bundles):
        for mode in modes:
            packs[(split_name, mode)] = build_features(bundle, int(args.history_lags), int(args.flow_k), mode)
    audit = causal_audit(bundles, packs, args)

    summary_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    prediction_archive: dict[str, np.ndarray] = {}

    constant_velocity = safe(test.rows[["dx_px", "dy_px"]].to_numpy(np.float32))
    summary_rows.extend(metric_rows(test, constant_velocity, horizons, "constant_velocity", {
        "family": "reference", "model": "constant_velocity", "input_mode": "causal_velocity", "control": "real",
    }))
    prediction_archive["constant_velocity"] = constant_velocity

    self_flow, self_meta, self_sweep = tune_self_flow(train, val, test, weights, args)
    summary_rows.extend(metric_rows(test, self_flow, horizons, "self_flow", {
        "family": "reference", "model": "self_flow", "input_mode": "causal_self_plus_local_flow", "control": "real",
    }))
    for row in self_sweep:
        selection_rows.append({"model": "self_flow", "input_mode": "causal_self_plus_local_flow", "eta": np.nan,
                               "hyperparameters": json.dumps({"k": row["k"], "mix": row["mix"]}, sort_keys=True),
                               "validation_weighted_component_rmse": row["validation_score"],
                               "selected": bool(row["k"] == self_meta["k"] and row["mix"] == self_meta["mix"]),
                               "fit_seconds": 0.0, "validation_predict_seconds": np.nan})
    prediction_archive["self_flow"] = self_flow

    exact_anchor = safe(test.anchor_steps[:, 0])
    summary_rows.extend(metric_rows(test, exact_anchor, horizons, "v52_anchor", {
        "family": "reference", "model": "v52_anchor", "input_mode": "exact_v97_anchor_cache", "control": "real",
    }))
    prediction_archive["v52_anchor"] = exact_anchor

    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        train_pack = packs[("train", mode)]
        val_pack = packs[("val", mode)]
        test_pack = packs[("test", mode)]
        for model_name in models:
            print(f"[v111] selecting {model_name}/{mode}", flush=True)
            fitted, eta, sweep, grid_fit_seconds = select_model(
                model_name, mode, train_pack, val_pack, train, val, weights, args
            )
            best_score = min(row["validation_weighted_component_rmse"] for row in sweep)
            for row in sweep:
                row["selected"] = bool(
                    math.isclose(float(row["validation_weighted_component_rmse"]), float(best_score), rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(float(row["eta"]), float(eta), rel_tol=0.0, abs_tol=1e-12)
                    and row["hyperparameters"] == json.dumps(finite(fitted.config), sort_keys=True)
                )
            selection_rows.extend(sweep)

            output, predict_seconds = chronological_predict(fitted, test_pack.values, test)
            prediction = decode_prediction(test, mode, output, eta)
            method = f"{model_name}__{mode}"
            summary_rows.extend(metric_rows(test, prediction, horizons, method, {
                "family": "tabular", "model": model_name, "input_mode": mode, "control": "real", "eta": eta,
            }))
            prediction_archive[method] = prediction

            preserve_anchor = mode == "v52_anchor"
            controls = {
                "wrong_cell_features": same_frame_wrong_cell(test_pack.values, test, test_pack.base_dim, preserve_anchor),
                "time_shuffled_features": causal_time_shuffle(
                    test_pack.values, test, test_pack.base_dim, preserve_anchor, int(args.seed) + 1110
                ),
            }
            if mode == "v52_anchor":
                controls["wrong_cell_all"] = same_frame_wrong_cell(test_pack.values, test, test_pack.base_dim, False)
            for control_name, controlled_values in controls.items():
                controlled_output, _seconds = chronological_predict(fitted, controlled_values, test)
                controlled_prediction = decode_prediction(test, mode, controlled_output, eta)
                control_rows.extend(metric_rows(test, controlled_prediction, horizons, method, {
                    "family": "tabular_control", "model": model_name, "input_mode": mode,
                    "control": control_name, "eta": eta,
                }))
                prediction_archive[f"{method}__{control_name}"] = controlled_prediction

            no_feature_prediction = reference_for_mode(test, mode)
            control_rows.extend(metric_rows(test, no_feature_prediction, horizons, method, {
                "family": "tabular_control", "model": model_name, "input_mode": mode,
                "control": "no_features", "eta": 0.0,
            }))
            runtime_rows.append({
                "model": model_name, "input_mode": mode, "parameter_count": parameter_count(fitted),
                "parameter_unit": "tree_nodes" if model_name == "hgbdt" else "learned_coefficients",
                "grid_candidates": len(model_grid(model_name, args)), "grid_fit_seconds": grid_fit_seconds,
                "test_predict_seconds": predict_seconds, "test_rows_per_second": len(test.rows) / max(predict_seconds, EPS),
                "selected_eta": eta, "selected_hyperparameters": json.dumps(finite(fitted.config), sort_keys=True),
            })
            joblib.dump(
                {"fitted": fitted, "eta": eta, "feature_names": test_pack.names, "mode": mode,
                 "anchor_cache": str(args.anchor_cache), "seed": int(args.seed)},
                checkpoint_dir / f"{method}.joblib",
                compress=3,
            )

    summary = pd.DataFrame(summary_rows)
    controls = pd.DataFrame(control_rows)
    selection = pd.DataFrame(selection_rows)
    runtime = pd.DataFrame(runtime_rows)
    data_contract = pd.DataFrame([{
        "anchor_cache": str(args.anchor_cache.resolve()),
        "anchor_cache_kind": "exact v97 v52 cache: OOF train, held-out val/test",
        "train_movies": ",".join(map(str, sorted(train.rows.sequence.unique()))),
        "val_movies": ",".join(map(str, sorted(val.rows.sequence.unique()))),
        "test_movies": ",".join(map(str, sorted(test.rows.sequence.unique()))),
        "train_rows": len(train.rows), "val_rows": len(val.rows), "test_rows": len(test.rows),
        "history_lags": int(args.history_lags), "flow_k": int(args.flow_k),
        "prediction_contract": "h1 issued at t before observing t+1; h2/h4/h6 sum chronological h1 forecasts",
        "component_rmse_definition": "sqrt(mean over rows and x/y components of squared error)",
        "vector_rmse_definition": "sqrt(mean over rows of squared Euclidean endpoint error)",
        "target_usage": "training/validation labels and final metrics only",
        "feature_latest_time": "t", "scaling": "StandardScaler fit on train movies only",
        "hyperparameter_selection": "validation movie weighted rolling component RMSE only",
        "validation_horizon_weights": args.validation_horizon_weights,
        "smoke": bool(args.smoke), "seed": int(args.seed),
    }])
    parameters = {
        "args": finite(vars(args)), "python": sys.version, "platform": platform.platform(),
        "sklearn": __import__("sklearn").__version__, "numpy": np.__version__,
        "cpu_count": os.cpu_count(), "elapsed_seconds": time.perf_counter() - started,
    }

    required_numeric = [
        "component_rmse", "vector_rmse", "component_r2_dx", "component_r2_dy",
        "component_r2_mean", "vector_r2", "cosine", "magnitude_ratio",
    ]
    for table_name, table in (("summary", summary), ("controls", controls)):
        invalid = table.n_rows.le(0) | ~np.isfinite(table[required_numeric].to_numpy(np.float64)).all(axis=1)
        if bool(invalid.any()):
            raise RuntimeError(
                f"Non-finite or empty metric rows in {table_name}: "
                f"{table.loc[invalid, ['method', 'horizon', 'n_rows']].to_dict(orient='records')}"
            )

    summary.to_csv(args.out_dir / "v111_online_tabular_summary.csv", index=False)
    controls.to_csv(args.out_dir / "v111_online_tabular_controls.csv", index=False)
    selection.to_csv(args.out_dir / "v111_validation_selection.csv", index=False)
    audit.to_csv(args.out_dir / "v111_causal_audit.csv", index=False)
    runtime.to_csv(args.out_dir / "v111_runtime_parameters.csv", index=False)
    data_contract.to_csv(args.out_dir / "v111_data_contract.csv", index=False)
    np.savez_compressed(args.out_dir / "v111_test_predictions.npz", **prediction_archive)
    (args.out_dir / "run_config.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    report = report_markdown(summary, controls, selection, audit, runtime, args, time.perf_counter() - started)
    (args.out_dir / "v111_report.md").write_text(report, encoding="utf-8")
    print(report)


def min_positive(current: int, smoke_default: int) -> int:
    return smoke_default if current <= 0 else min(current, smoke_default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-modes", default="raw_coordinate,v52_anchor")
    parser.add_argument("--models", default="ridge,hgbdt,mlp")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--validation-horizon-weights", default="1:0.45,2:0.25,4:0.18,6:0.12")
    parser.add_argument("--history-lags", type=int, default=8)
    parser.add_argument("--flow-k", type=int, default=16)
    parser.add_argument("--ridge-alphas", default="1,10,100,1000")
    parser.add_argument("--hgbdt-learning-rates", default="0.04,0.08")
    parser.add_argument("--hgbdt-max-leaf-nodes", default="15,31")
    parser.add_argument("--hgbdt-l2", default="1,10")
    parser.add_argument("--hgbdt-max-iter", type=int, default=180)
    parser.add_argument("--hgbdt-min-samples-leaf", type=int, default=30)
    parser.add_argument("--mlp-hidden-grid", default="64;128x64")
    parser.add_argument("--mlp-alphas", default="0.0001,0.001")
    parser.add_argument("--mlp-learning-rate", type=float, default=8e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=512)
    parser.add_argument("--mlp-max-iter", type=int, default=220)
    parser.add_argument("--mlp-patience", type=int, default=15)
    parser.add_argument("--eta-grid", default="0,0.1,0.25,0.5,0.75,1,1.25")
    parser.add_argument("--self-flow-k-grid", default="8,16,32")
    parser.add_argument("--self-flow-mix-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    try:
        run(args)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": False, "error": repr(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.out_dir / "v111_error.json").write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
