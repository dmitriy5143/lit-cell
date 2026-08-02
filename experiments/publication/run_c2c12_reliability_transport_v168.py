#!/usr/bin/env python3
"""C2C12 reliability transfer gate with experiment-level outer rotations.

The reliability model is deliberately restricted.  It may predict a scalar
Student-t scale and modulate a bounded acceleration/update gain, but it cannot
add a free displacement vector.  All inference features are available at the
issue time: positions and transitions through t, track age, and same-frame
local flow consistency.  The transition t->t+1 is used only as a supervised
target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_ROOT = ROOT / "new_data" / "c2c12_online" / "tables"
DEFAULT_OUT = ROOT / "outputs" / "c2c12_reliability_transport_v168_2026-07-28"
KINDS = ("manual", "automatic")
ROTATIONS = (
    (1, 2, 3),
    (2, 3, 1),
    (3, 1, 2),
)
EPS = 1e-8
STUDENT_DF = 5.0

RAW_COLUMNS = [
    "sequence",
    "experiment",
    "field",
    "frame",
    "track_id",
    "x_px",
    "y_px",
    "dx_px",
    "dy_px",
    "target_dx_px",
    "target_dy_px",
    "state",
    "previous_interpolated",
    "current_interpolated",
    "target_interpolated",
]

FEATURE_COLUMNS = [
    "velocity_x",
    "velocity_y",
    "velocity_prev_x",
    "velocity_prev_y",
    "velocity_prev2_x",
    "velocity_prev2_y",
    "speed",
    "speed_prev",
    "acceleration_x",
    "acceleration_y",
    "acceleration_norm",
    "jerk_x",
    "jerk_y",
    "jerk_norm",
    "turn_cosine",
    "turn_sine",
    "recent_velocity_std",
    "track_age_log",
    "has_previous",
    "has_previous2",
    "observation_available",
    "local_flow_x",
    "local_flow_y",
    "local_flow_count_log",
    "local_flow_disagreement",
    "frame_flow_x",
    "frame_flow_y",
    "frame_flow_disagreement",
]


@dataclass
class SplitData:
    rows: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    total_rows: int
    invalid_rows_excluded: int
    row_key_sha256: str


@dataclass
class ReliabilityFit:
    model: HistGradientBoostingRegressor
    observation_model: HistGradientBoostingRegressor
    fixed_gain: float
    local_weight: float
    gain_low: float
    gain_high: float
    score_center: float
    score_scale: float
    observation_center: float
    observation_scale: float
    observation_weight: float
    calibration_factor: float
    train_error_median: float


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key_hash(rows: pd.DataFrame) -> str:
    keys = rows[["sequence", "frame", "track_id"]].to_numpy(np.int64)
    return hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()


def component_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def vector_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(target - prediction), axis=1))))


def vector_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    centered = target - target.mean(axis=0, keepdims=True)
    sse = float(np.sum(np.square(target - prediction)))
    sst = float(np.sum(np.square(centered)))
    return float(1.0 - sse / max(sst, EPS))


def cosine_mean(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.sum(target * prediction, axis=1)
    denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    valid = denominator > EPS
    return float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.0


def magnitude_ratio(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(np.linalg.norm(prediction, axis=1))
        / max(float(np.mean(np.linalg.norm(target, axis=1))), EPS)
    )


def student_nll(target: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> float:
    safe_scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-3)
    if safe_scale.ndim == 1:
        safe_scale = safe_scale[:, None]
    standardized = (target - prediction) / safe_scale
    log_prob = student_t.logpdf(standardized, df=STUDENT_DF) - np.log(safe_scale)
    return -float(np.mean(log_prob))


def uncertainty_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    safe_scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-3)
    component_error = np.abs(target - prediction)
    if safe_scale.ndim == 1:
        safe_scale = safe_scale[:, None]
    q50 = float(student_t.ppf(0.75, df=STUDENT_DF))
    q90 = float(student_t.ppf(0.95, df=STUDENT_DF))
    coverage50 = float(np.mean(component_error <= q50 * safe_scale))
    coverage90 = float(np.mean(component_error <= q90 * safe_scale))
    scalar_error = np.linalg.norm(target - prediction, axis=1)
    scalar_scale = safe_scale.mean(axis=1)
    if np.std(scalar_scale) <= EPS or np.std(scalar_error) <= EPS:
        correlation = 0.0
    else:
        correlation = spearmanr(
            scalar_scale, scalar_error, nan_policy="omit"
        ).statistic
    return {
        "student_t_nll": student_nll(target, prediction, safe_scale),
        "coverage_50": coverage50,
        "coverage_90": coverage90,
        "coverage_error": 0.5 * (abs(coverage50 - 0.50) + abs(coverage90 - 0.90)),
        "uncertainty_error_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "mean_scale": float(np.mean(scalar_scale)),
    }


def prediction_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray | None = None,
) -> dict[str, float]:
    result = {
        "component_rmse": component_rmse(target, prediction),
        "vector_rmse": vector_rmse(target, prediction),
        "r2": vector_r2(target, prediction),
        "cosine": cosine_mean(target, prediction),
        "magnitude_ratio": magnitude_ratio(target, prediction),
    }
    if scale is not None:
        result.update(uncertainty_metrics(target, prediction, scale))
    return result


def table_paths(table_root: Path, kind: str, experiment: int) -> list[Path]:
    dataset = f"C2C12_{kind.capitalize()}"
    paths = sorted((table_root / dataset).glob(f"{dataset}_{experiment}??_tracks.csv"))
    if len(paths) != 16:
        raise RuntimeError(
            f"Expected 16 {kind} tables for experiment {experiment}, found {len(paths)}"
        )
    return paths


def evenly_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    groups = list(frame.groupby("sequence", sort=True).groups.items())
    assigned = 0
    for group_index, (_sequence, indices) in enumerate(groups):
        indices = np.asarray(indices, dtype=np.int64)
        if group_index == len(groups) - 1:
            count = maximum - assigned
        else:
            count = max(1, int(round(maximum * len(indices) / len(frame))))
            count = min(count, maximum - assigned - (len(groups) - group_index - 1))
        count = min(count, len(indices))
        selected.append(np.sort(rng.choice(indices, size=count, replace=False)))
        assigned += count
    take = np.sort(np.concatenate(selected))
    return frame.iloc[take].reset_index(drop=True)


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    velocity = frame[["dx_px", "dy_px"]].to_numpy(np.float32)
    previous = frame[["lag1_dx", "lag1_dy"]].to_numpy(np.float32)
    previous2 = frame[["lag2_dx", "lag2_dy"]].to_numpy(np.float32)
    acceleration = velocity - previous
    previous_acceleration = previous - previous2
    jerk = acceleration - previous_acceleration
    speed = np.linalg.norm(velocity, axis=1)
    speed_previous = np.linalg.norm(previous, axis=1)
    denominator = np.maximum(speed * speed_previous, EPS)
    turn_cosine = np.sum(velocity * previous, axis=1) / denominator
    turn_sine = (
        previous[:, 0] * velocity[:, 1] - previous[:, 1] * velocity[:, 0]
    ) / denominator
    recent = np.stack([velocity, previous, previous2], axis=1)
    recent_std = np.sqrt(np.mean(np.var(recent, axis=1), axis=1))
    local_flow = frame[["local_flow_x", "local_flow_y"]].to_numpy(np.float32)
    frame_flow = frame[["frame_flow_x", "frame_flow_y"]].to_numpy(np.float32)
    features = pd.DataFrame(
        {
            "velocity_x": velocity[:, 0],
            "velocity_y": velocity[:, 1],
            "velocity_prev_x": previous[:, 0],
            "velocity_prev_y": previous[:, 1],
            "velocity_prev2_x": previous2[:, 0],
            "velocity_prev2_y": previous2[:, 1],
            "speed": speed,
            "speed_prev": speed_previous,
            "acceleration_x": acceleration[:, 0],
            "acceleration_y": acceleration[:, 1],
            "acceleration_norm": np.linalg.norm(acceleration, axis=1),
            "jerk_x": jerk[:, 0],
            "jerk_y": jerk[:, 1],
            "jerk_norm": np.linalg.norm(jerk, axis=1),
            "turn_cosine": np.clip(turn_cosine, -1.0, 1.0),
            "turn_sine": np.clip(turn_sine, -1.0, 1.0),
            "recent_velocity_std": recent_std,
            "track_age_log": np.log1p(frame["track_age"].to_numpy(np.float32)),
            "has_previous": frame["has_lag1"].to_numpy(np.float32),
            "has_previous2": frame["has_lag2"].to_numpy(np.float32),
            "observation_available": frame["observation_available"].to_numpy(np.float32),
            "local_flow_x": local_flow[:, 0],
            "local_flow_y": local_flow[:, 1],
            "local_flow_count_log": np.log1p(
                frame["local_flow_count"].to_numpy(np.float32)
            ),
            "local_flow_disagreement": np.linalg.norm(
                velocity - local_flow, axis=1
            ),
            "frame_flow_x": frame_flow[:, 0],
            "frame_flow_y": frame_flow[:, 1],
            "frame_flow_disagreement": np.linalg.norm(
                velocity - frame_flow, axis=1
            ),
        }
    )
    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def engineer_table(table: pd.DataFrame, local_bin_px: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    forbidden = {"target_dx_px", "target_dy_px", "target_interpolated"}
    table = table.sort_values(["sequence", "track_id", "frame"]).reset_index(drop=True)
    group = table.groupby(["sequence", "track_id"], sort=False)
    table["lag1_dx"] = group["dx_px"].shift(1)
    table["lag1_dy"] = group["dy_px"].shift(1)
    table["lag2_dx"] = group["dx_px"].shift(2)
    table["lag2_dy"] = group["dy_px"].shift(2)
    table["has_lag1"] = table["lag1_dx"].notna().astype(np.float32)
    table["has_lag2"] = table["lag2_dx"].notna().astype(np.float32)
    table["lag1_dx"] = table["lag1_dx"].fillna(table["dx_px"])
    table["lag1_dy"] = table["lag1_dy"].fillna(table["dy_px"])
    table["lag2_dx"] = table["lag2_dx"].fillna(table["lag1_dx"])
    table["lag2_dy"] = table["lag2_dy"].fillna(table["lag1_dy"])
    table["track_age"] = group.cumcount().astype(np.float32)
    table["observation_available"] = 1.0
    table["grid_x"] = np.floor(table["x_px"] / float(local_bin_px)).astype(np.int32)
    table["grid_y"] = np.floor(table["y_px"] / float(local_bin_px)).astype(np.int32)
    local_group = table.groupby(
        ["sequence", "frame", "grid_x", "grid_y"], sort=False
    )
    table["local_flow_x"] = local_group["dx_px"].transform("mean")
    table["local_flow_y"] = local_group["dy_px"].transform("mean")
    table["local_flow_count"] = local_group["dx_px"].transform("size")
    frame_group = table.groupby(["sequence", "frame"], sort=False)
    table["frame_flow_x"] = frame_group["dx_px"].transform("mean")
    table["frame_flow_y"] = frame_group["dy_px"].transform("mean")
    features = derive_features(table)
    if forbidden & set(features.columns):
        raise RuntimeError("Target/future leakage detected in inference feature packet")
    return table, features


def load_experiment(
    table_root: Path,
    kind: str,
    experiment: int,
    maximum_rows: int,
    seed: int,
    local_bin_px: float,
) -> SplitData:
    engineered_rows: list[pd.DataFrame] = []
    engineered_features: list[pd.DataFrame] = []
    total_rows = 0
    invalid_rows = 0
    for file_index, path in enumerate(table_paths(table_root, kind, experiment)):
        table = pd.read_csv(path, usecols=RAW_COLUMNS)
        total_rows += len(table)
        finite_columns = [
            "x_px",
            "y_px",
            "dx_px",
            "dy_px",
            "target_dx_px",
            "target_dy_px",
        ]
        valid = np.isfinite(table[finite_columns].to_numpy(np.float64)).all(axis=1)
        invalid_rows += int((~valid).sum())
        table = table.loc[valid].reset_index(drop=True)
        rows, features = engineer_table(table, local_bin_px)
        rows = rows.reset_index(drop=True)
        features = features.reset_index(drop=True)
        rows["_feature_index"] = np.arange(len(rows), dtype=np.int64)
        if maximum_rows > 0:
            per_file = max(1, int(math.ceil(maximum_rows / 16)))
            sampled_rows = evenly_sample(rows, per_file, seed + file_index * 1009)
            sampled_features = features.iloc[
                sampled_rows["_feature_index"].to_numpy(np.int64)
            ].reset_index(drop=True)
            sampled_rows = sampled_rows.drop(columns="_feature_index").reset_index(drop=True)
        else:
            sampled_rows = rows.drop(columns="_feature_index")
            sampled_features = features
        engineered_rows.append(sampled_rows)
        engineered_features.append(sampled_features)
    rows = pd.concat(engineered_rows, ignore_index=True)
    features = pd.concat(engineered_features, ignore_index=True)
    if maximum_rows > 0 and len(rows) > maximum_rows:
        rows["_feature_index"] = np.arange(len(rows), dtype=np.int64)
        sampled_rows = evenly_sample(rows, maximum_rows, seed + 99173)
        features = features.iloc[
            sampled_rows["_feature_index"].to_numpy(np.int64)
        ].reset_index(drop=True)
        rows = sampled_rows.drop(columns="_feature_index").reset_index(drop=True)
    target = rows[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
    return SplitData(
        rows=rows,
        features=features,
        target=target,
        total_rows=total_rows,
        invalid_rows_excluded=invalid_rows,
        row_key_sha256=row_key_hash(rows),
    )


def feature_arrays(data: SplitData) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = data.features[FEATURE_COLUMNS].to_numpy(np.float32)
    values = {column: data.features[column].to_numpy(np.float32) for column in FEATURE_COLUMNS}
    velocity = np.column_stack([values["velocity_x"], values["velocity_y"]])
    previous = np.column_stack([values["velocity_prev_x"], values["velocity_prev_y"]])
    previous2 = np.column_stack([values["velocity_prev2_x"], values["velocity_prev2_y"]])
    local = np.column_stack([values["local_flow_x"], values["local_flow_y"]])
    arrays = {
        "velocity": velocity,
        "previous": previous,
        "previous2": previous2,
        "acceleration": velocity - previous,
        "local_flow": local,
    }
    return features, arrays


def bounded_prediction(
    arrays: dict[str, np.ndarray],
    gain: np.ndarray | float,
    local_weight: float,
) -> np.ndarray:
    velocity = arrays["velocity"]
    acceleration = arrays["acceleration"]
    gain_array = np.asarray(gain, dtype=np.float32)
    if gain_array.ndim == 0:
        gain_array = np.full(len(velocity), float(gain_array), dtype=np.float32)
    individual = velocity + gain_array[:, None] * acceleration
    local = arrays["local_flow"] + gain_array[:, None] * (
        arrays["local_flow"] - arrays["previous"]
    )
    return (
        (1.0 - float(local_weight)) * individual
        + float(local_weight) * local
    ).astype(np.float32)


def best_fixed_parameters(
    validation: SplitData,
    gain_grid: list[float],
    local_weight_grid: list[float],
) -> tuple[float, float]:
    _features, arrays = feature_arrays(validation)
    best = (float("inf"), 0.0, 0.0)
    for gain in gain_grid:
        for local_weight in local_weight_grid:
            prediction = bounded_prediction(arrays, gain, local_weight)
            score = component_rmse(validation.target, prediction)
            if score < best[0]:
                best = (score, float(gain), float(local_weight))
    return best[1], best[2]


def corruption_features(
    data: SplitData,
    corruption: str,
    severity: float,
    seed: int,
) -> pd.DataFrame:
    frame = data.features.copy()
    rng = np.random.default_rng(seed)
    n = len(frame)
    current = frame[["velocity_x", "velocity_y"]].to_numpy(np.float32)
    previous = frame[["velocity_prev_x", "velocity_prev_y"]].to_numpy(np.float32)
    previous2 = frame[["velocity_prev2_x", "velocity_prev2_y"]].to_numpy(np.float32)
    if corruption == "centroid_jitter":
        current += rng.normal(0.0, math.sqrt(2.0) * severity, current.shape)
        previous += rng.normal(0.0, math.sqrt(2.0) * severity, previous.shape)
        previous2 += rng.normal(0.0, math.sqrt(2.0) * severity, previous2.shape)
    elif corruption == "missing":
        mask = rng.random(n) < severity
        current[mask] = previous[mask]
        frame.loc[mask, "observation_available"] = 0.0
    elif corruption == "delay":
        mask = rng.random(n) < severity
        current[mask] = previous[mask]
        previous[mask] = previous2[mask]
    elif corruption == "identity_swap":
        mask = rng.random(n) < severity
        for _key, indices in data.rows.groupby(["sequence", "frame"], sort=False).groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            selected = indices[mask[indices]]
            if len(selected) > 1:
                current[selected] = current[np.roll(selected, 1)]
    elif corruption == "fragmentation":
        mask = rng.random(n) < severity
        previous[mask] = current[mask]
        previous2[mask] = current[mask]
        frame.loc[mask, ["has_previous", "has_previous2", "track_age_log"]] = 0.0
    elif corruption != "clean":
        raise ValueError(corruption)
    frame[["velocity_x", "velocity_y"]] = current
    frame[["velocity_prev_x", "velocity_prev_y"]] = previous
    frame[["velocity_prev2_x", "velocity_prev2_y"]] = previous2
    acceleration = current - previous
    jerk = acceleration - (previous - previous2)
    speed = np.linalg.norm(current, axis=1)
    speed_previous = np.linalg.norm(previous, axis=1)
    denominator = np.maximum(speed * speed_previous, EPS)
    frame["speed"] = speed
    frame["speed_prev"] = speed_previous
    frame["acceleration_x"] = acceleration[:, 0]
    frame["acceleration_y"] = acceleration[:, 1]
    frame["acceleration_norm"] = np.linalg.norm(acceleration, axis=1)
    frame["jerk_x"] = jerk[:, 0]
    frame["jerk_y"] = jerk[:, 1]
    frame["jerk_norm"] = np.linalg.norm(jerk, axis=1)
    frame["turn_cosine"] = np.clip(
        np.sum(current * previous, axis=1) / denominator, -1.0, 1.0
    )
    frame["turn_sine"] = np.clip(
        (previous[:, 0] * current[:, 1] - previous[:, 1] * current[:, 0])
        / denominator,
        -1.0,
        1.0,
    )
    recent = np.stack([current, previous, previous2], axis=1)
    frame["recent_velocity_std"] = np.sqrt(
        np.mean(np.var(recent, axis=1), axis=1)
    )
    local = frame[["local_flow_x", "local_flow_y"]].to_numpy(np.float32)
    global_flow = frame[["frame_flow_x", "frame_flow_y"]].to_numpy(np.float32)
    frame["local_flow_disagreement"] = np.linalg.norm(current - local, axis=1)
    frame["frame_flow_disagreement"] = np.linalg.norm(current - global_flow, axis=1)
    return frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def arrays_from_features(features: pd.DataFrame) -> dict[str, np.ndarray]:
    velocity = features[["velocity_x", "velocity_y"]].to_numpy(np.float32)
    previous = features[["velocity_prev_x", "velocity_prev_y"]].to_numpy(np.float32)
    return {
        "velocity": velocity,
        "previous": previous,
        "previous2": features[
            ["velocity_prev2_x", "velocity_prev2_y"]
        ].to_numpy(np.float32),
        "acceleration": velocity - previous,
        "local_flow": features[["local_flow_x", "local_flow_y"]].to_numpy(np.float32),
    }


def augment_reliability_training(
    train: SplitData,
    fixed_gain: float,
    local_weight: float,
    seed: int,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    base_features = train.features[FEATURE_COLUMNS]
    base_prediction = bounded_prediction(
        arrays_from_features(base_features), fixed_gain, local_weight
    )
    feature_blocks = [base_features.to_numpy(np.float32)]
    error_blocks = [
        np.linalg.norm(train.target - base_prediction, axis=1) / math.sqrt(2.0)
    ]
    if enabled:
        specs = [
            ("centroid_jitter", 0.5),
            ("centroid_jitter", 1.5),
            ("missing", 0.20),
            ("delay", 0.40),
            ("identity_swap", 0.10),
            ("fragmentation", 0.20),
        ]
        rng = np.random.default_rng(seed + 773)
        maximum = min(len(train.rows), 120_000)
        subset = np.sort(rng.choice(len(train.rows), size=maximum, replace=False))
        subset_data = SplitData(
            rows=train.rows.iloc[subset].reset_index(drop=True),
            features=train.features.iloc[subset].reset_index(drop=True),
            target=train.target[subset],
            total_rows=maximum,
            invalid_rows_excluded=0,
            row_key_sha256="augmentation_subset",
        )
        for index, (corruption, severity) in enumerate(specs):
            corrupted = corruption_features(
                subset_data, corruption, severity, seed + index * 7919
            )
            prediction = bounded_prediction(
                arrays_from_features(corrupted), fixed_gain, local_weight
            )
            feature_blocks.append(corrupted.to_numpy(np.float32))
            error_blocks.append(
                np.linalg.norm(subset_data.target - prediction, axis=1)
                / math.sqrt(2.0)
            )
    x = np.concatenate(feature_blocks, axis=0)
    error = np.concatenate(error_blocks, axis=0)
    y = np.log(np.maximum(error, 1e-3))
    return x, y


def observation_corruption_training(
    train: SplitData,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a target-only-free observation corruption curriculum."""

    rng = np.random.default_rng(seed + 1701)
    maximum = min(len(train.rows), 150_000)
    subset = np.sort(rng.choice(len(train.rows), size=maximum, replace=False))
    base = SplitData(
        rows=train.rows.iloc[subset].reset_index(drop=True),
        features=train.features.iloc[subset].reset_index(drop=True),
        target=train.target[subset],
        total_rows=maximum,
        invalid_rows_excluded=0,
        row_key_sha256="observation_corruption_subset",
    )
    base_features = base.features[FEATURE_COLUMNS]
    base_velocity = base_features[["velocity_x", "velocity_y"]].to_numpy(
        np.float32
    )
    speed_scale = max(
        float(np.quantile(np.linalg.norm(base_velocity, axis=1), 0.75)),
        0.25,
    )
    feature_blocks = [base_features.to_numpy(np.float32)]
    target_blocks = [np.zeros(maximum, dtype=np.float32)]
    specifications = [
        ("centroid_jitter", 0.25),
        ("centroid_jitter", 0.75),
        ("centroid_jitter", 1.5),
        ("missing", 0.15),
        ("missing", 0.40),
        ("delay", 0.35),
        ("delay", 0.80),
        ("identity_swap", 0.10),
        ("identity_swap", 0.25),
        ("fragmentation", 0.20),
        ("fragmentation", 0.50),
    ]
    for index, (corruption, severity) in enumerate(specifications):
        corrupted = corruption_features(
            base, corruption, severity, seed + 3001 + index * 8191
        )
        corrupted_velocity = corrupted[
            ["velocity_x", "velocity_y"]
        ].to_numpy(np.float32)
        velocity_delta = np.linalg.norm(
            corrupted_velocity - base_velocity, axis=1
        ) / speed_scale
        missing_flag = np.maximum(
            0.0,
            base_features["observation_available"].to_numpy(np.float32)
            - corrupted["observation_available"].to_numpy(np.float32),
        )
        fragmented_flag = np.maximum(
            0.0,
            base_features["has_previous"].to_numpy(np.float32)
            - corrupted["has_previous"].to_numpy(np.float32),
        )
        corruption_strength = np.log1p(
            velocity_delta + 2.0 * missing_flag + fragmented_flag
        )
        feature_blocks.append(corrupted.to_numpy(np.float32))
        target_blocks.append(corruption_strength.astype(np.float32))
    return np.concatenate(feature_blocks), np.concatenate(target_blocks)


def normalize_score(
    log_score: np.ndarray,
    center: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = (log_score - center) / max(scale, 1e-3)
    quality = 1.0 / (1.0 + np.exp(np.clip(standardized, -20.0, 20.0)))
    return standardized, quality


def select_gain_mapping(
    validation: SplitData,
    validation_log_score: np.ndarray,
    center: float,
    score_scale: float,
    local_weight: float,
    lows: list[float],
    highs: list[float],
) -> tuple[float, float]:
    _standardized, quality = normalize_score(validation_log_score, center, score_scale)
    arrays = arrays_from_features(validation.features)
    best = (float("inf"), 0.0, 0.0)
    for low in lows:
        for high in highs:
            if high < low:
                continue
            gain = low + (high - low) * quality
            prediction = bounded_prediction(arrays, gain, local_weight)
            score = component_rmse(validation.target, prediction)
            if score < best[0]:
                best = (score, float(low), float(high))
    return best[1], best[2]


def calibrate_scale(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    factors: list[float],
) -> float:
    best = (float("inf"), 1.0)
    for factor in factors:
        score = student_nll(target, prediction, np.maximum(scale * factor, 1e-3))
        if score < best[0]:
            best = (score, float(factor))
    return best[1]


def fit_reliability(
    train: SplitData,
    validation: SplitData,
    args: argparse.Namespace,
    seed: int,
) -> ReliabilityFit:
    fixed_gain, local_weight = best_fixed_parameters(
        validation,
        parse_floats(args.gain_grid),
        parse_floats(args.local_weight_grid),
    )
    x_train, y_train = augment_reliability_training(
        train,
        fixed_gain,
        local_weight,
        seed,
        bool(args.corruption_augmentation),
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2),
        min_samples_leaf=int(args.min_samples_leaf),
        early_stopping=True,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    observation_x, observation_y = observation_corruption_training(train, seed)
    observation_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2),
        min_samples_leaf=int(args.min_samples_leaf),
        early_stopping=True,
        random_state=seed + 991,
    )
    observation_model.fit(observation_x, observation_y)
    train_features = train.features[FEATURE_COLUMNS].to_numpy(np.float32)
    train_error_score = model.predict(train_features)
    train_observation_score = observation_model.predict(train_features)
    observation_center = float(np.median(train_observation_score))
    observation_scale = float(
        np.quantile(train_observation_score, 0.75)
        - np.quantile(train_observation_score, 0.25)
    )
    normalized_observation_train = np.clip(
        (train_observation_score - observation_center)
        / max(observation_scale, 1e-3),
        -4.0,
        4.0,
    )
    train_log_score = (
        train_error_score
        + float(args.observation_score_weight) * normalized_observation_train
    )
    center = float(np.median(train_log_score))
    score_scale = float(
        np.quantile(train_log_score, 0.75) - np.quantile(train_log_score, 0.25)
    )
    validation_features = validation.features[FEATURE_COLUMNS].to_numpy(
        np.float32
    )
    validation_observation_score = observation_model.predict(validation_features)
    validation_log_score = model.predict(validation_features) + float(
        args.observation_score_weight
    ) * np.clip(
        (validation_observation_score - observation_center)
        / max(observation_scale, 1e-3),
        -4.0,
        4.0,
    )
    gain_low, gain_high = select_gain_mapping(
        validation,
        validation_log_score,
        center,
        score_scale,
        local_weight,
        parse_floats(args.gain_low_grid),
        parse_floats(args.gain_high_grid),
    )
    _z, quality = normalize_score(validation_log_score, center, score_scale)
    gain = gain_low + (gain_high - gain_low) * quality
    validation_prediction = bounded_prediction(
        arrays_from_features(validation.features), gain, local_weight
    )
    validation_scale = np.exp(validation_log_score)
    calibration = calibrate_scale(
        validation.target,
        validation_prediction,
        validation_scale,
        parse_floats(args.scale_factor_grid),
    )
    return ReliabilityFit(
        model=model,
        observation_model=observation_model,
        fixed_gain=fixed_gain,
        local_weight=local_weight,
        gain_low=gain_low,
        gain_high=gain_high,
        score_center=center,
        score_scale=max(score_scale, 1e-3),
        observation_center=observation_center,
        observation_scale=max(observation_scale, 1e-3),
        observation_weight=float(args.observation_score_weight),
        calibration_factor=calibration,
        train_error_median=float(np.median(np.exp(y_train))),
    )


def controlled_scores(
    real_log_score: np.ndarray,
    rows: pd.DataFrame,
    control: str,
    constant_log_score: float,
    seed: int,
) -> np.ndarray:
    if control == "real":
        return real_log_score.copy()
    if control in {"constant", "no_reliability"}:
        return np.full(len(real_log_score), constant_log_score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if control == "row_shuffled":
        return real_log_score[rng.permutation(len(real_log_score))]
    result = real_log_score.copy()
    if control == "time_shuffled":
        for _key, indices in rows.groupby(["sequence", "track_id"], sort=False).groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            if len(indices) > 1:
                shift = int(rng.integers(1, len(indices)))
                result[indices] = real_log_score[np.roll(indices, shift)]
        return result
    if control == "wrong_track":
        for _key, indices in rows.groupby(["sequence", "frame"], sort=False).groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            if len(indices) > 1:
                result[indices] = real_log_score[np.roll(indices, 1)]
        return result
    raise ValueError(control)


def evaluate_control(
    data: SplitData,
    fit: ReliabilityFit,
    control: str,
    seed: int,
    features: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    feature_frame = features if features is not None else data.features
    x = feature_frame[FEATURE_COLUMNS].to_numpy(np.float32)
    observation_score = fit.observation_model.predict(x)
    normalized_observation = np.clip(
        (observation_score - fit.observation_center)
        / fit.observation_scale,
        -4.0,
        4.0,
    )
    real_log_score = (
        fit.model.predict(x)
        + fit.observation_weight * normalized_observation
    )
    log_score = controlled_scores(
        real_log_score,
        data.rows,
        control,
        math.log(max(fit.train_error_median, 1e-3)),
        seed,
    )
    _z, quality = normalize_score(log_score, fit.score_center, fit.score_scale)
    gain = fit.gain_low + (fit.gain_high - fit.gain_low) * quality
    prediction = bounded_prediction(
        arrays_from_features(feature_frame), gain, fit.local_weight
    )
    scale = np.exp(log_score) * fit.calibration_factor
    metrics = prediction_metrics(data.target, prediction, scale)
    metrics.update(
        {
            "mean_gain": float(np.mean(gain)),
            "gain_std": float(np.std(gain)),
            "score_std": float(np.std(log_score)),
            "mean_observation_corruption_score": float(
                np.mean(observation_score)
            ),
        }
    )
    return prediction, scale, gain, metrics


def baseline_predictions(
    train: SplitData,
    validation: SplitData,
    test: SplitData,
    fit: ReliabilityFit,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    arrays_val = arrays_from_features(validation.features)
    arrays_test = arrays_from_features(test.features)
    predictions: dict[str, np.ndarray] = {
        "no_update_previous_velocity": arrays_test["previous"],
        "constant_velocity": arrays_test["velocity"],
        "validation_fixed_gain": bounded_prediction(
            arrays_test, fit.fixed_gain, fit.local_weight
        ),
    }
    best_alpha = min(
        parse_floats(args.kalman_alpha_grid),
        key=lambda alpha: component_rmse(
            validation.target,
            alpha * arrays_val["velocity"] + (1.0 - alpha) * arrays_val["previous"],
        ),
    )
    predictions["kalman_cv_like"] = (
        best_alpha * arrays_test["velocity"]
        + (1.0 - best_alpha) * arrays_test["previous"]
    )
    best_gamma = min(
        parse_floats(args.ca_gain_grid),
        key=lambda gamma: component_rmse(
            validation.target,
            arrays_val["velocity"] + gamma * arrays_val["acceleration"],
        ),
    )
    predictions["kalman_ca_like"] = (
        arrays_test["velocity"] + best_gamma * arrays_test["acceleration"]
    )
    acceleration_val = np.linalg.norm(arrays_val["acceleration"], axis=1)
    acceleration_test = np.linalg.norm(arrays_test["acceleration"], axis=1)
    candidates: list[tuple[float, float, float]] = []
    for threshold in parse_floats(args.imm_threshold_grid):
        for temperature in parse_floats(args.imm_temperature_grid):
            weight = 1.0 / (
                1.0
                + np.exp(
                    np.clip(
                        (acceleration_val - threshold)
                        / max(temperature, 1e-3),
                        -30.0,
                        30.0,
                    )
                )
            )
            cv = arrays_val["velocity"]
            ca = arrays_val["velocity"] + best_gamma * arrays_val["acceleration"]
            prediction = weight[:, None] * ca + (1.0 - weight[:, None]) * cv
            candidates.append(
                (component_rmse(validation.target, prediction), threshold, temperature)
            )
    _score, threshold, temperature = min(candidates)
    weight = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (acceleration_test - threshold) / max(temperature, 1e-3),
                -30.0,
                30.0,
            )
        )
    )
    ca_test = arrays_test["velocity"] + best_gamma * arrays_test["acceleration"]
    predictions["imm_cv_ca_like"] = (
        weight[:, None] * ca_test
        + (1.0 - weight[:, None]) * arrays_test["velocity"]
    )
    return {key: np.asarray(value, dtype=np.float32) for key, value in predictions.items()}


def calibration_rows(
    data: SplitData,
    prediction: np.ndarray,
    scale: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    error = np.linalg.norm(data.target - prediction, axis=1)
    scalar_scale = np.asarray(scale).reshape(-1)
    if np.unique(scalar_scale).size < 2:
        bins = np.zeros(len(scalar_scale), dtype=np.int64)
    else:
        try:
            bins = pd.qcut(
                scalar_scale, q=10, labels=False, duplicates="drop"
            )
            bins = np.asarray(pd.Series(bins).fillna(0), dtype=np.int64)
        except ValueError:
            bins = np.zeros(len(scalar_scale), dtype=np.int64)
    result = []
    for bin_value in sorted(pd.unique(bins)):
        mask = np.asarray(bins == bin_value)
        result.append(
            {
                **metadata,
                "bin": int(bin_value),
                "n_rows": int(mask.sum()),
                "mean_predicted_scale": float(np.mean(scalar_scale[mask])),
                "mean_vector_error": float(np.mean(error[mask])),
                "median_vector_error": float(np.median(error[mask])),
            }
        )
    return result


def per_field_delta(
    data: SplitData,
    real_prediction: np.ndarray,
    real_scale: np.ndarray,
    control_prediction: np.ndarray,
    control_scale: np.ndarray,
    control: str,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for field, indices in data.rows.groupby("field", sort=True).groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        real_nll = student_nll(
            data.target[indices], real_prediction[indices], real_scale[indices]
        )
        control_nll = student_nll(
            data.target[indices], control_prediction[indices], control_scale[indices]
        )
        rows.append(
            {
                "field": int(field),
                "nll_gain": control_nll - real_nll,
                "rmse_gain": component_rmse(
                    data.target[indices], control_prediction[indices]
                )
                - component_rmse(data.target[indices], real_prediction[indices]),
            }
        )
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        selected = rng.integers(0, len(frame), len(frame))
        samples.append(float(frame.nll_gain.to_numpy()[selected].mean()))
    return {
        "comparison": f"real_vs_{control}",
        "field_clusters": int(len(frame)),
        "mean_nll_gain": float(frame.nll_gain.mean()),
        "nll_gain_ci_low": float(np.quantile(samples, 0.025)),
        "nll_gain_ci_high": float(np.quantile(samples, 0.975)),
        "probability_nll_gain_positive": float(np.mean(np.asarray(samples) > 0)),
        "mean_rmse_gain": float(frame.rmse_gain.mean()),
    }


def corruption_replay(
    data: SplitData,
    fit: ReliabilityFit,
    kind: str,
    rotation: str,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    specifications = {
        "clean": [0.0],
        "centroid_jitter": parse_floats(args.jitter_levels),
        "missing": parse_floats(args.missing_levels),
        "delay": parse_floats(args.delay_levels),
        "identity_swap": parse_floats(args.swap_levels),
        "fragmentation": parse_floats(args.fragmentation_levels),
    }
    for corruption_index, (corruption, levels) in enumerate(specifications.items()):
        for level_index, severity in enumerate(levels):
            features = (
                data.features
                if corruption == "clean"
                else corruption_features(
                    data,
                    corruption,
                    severity,
                    seed + corruption_index * 10007 + level_index,
                )
            )
            for control in ("real", "constant"):
                _prediction, _scale, _gain, metrics = evaluate_control(
                    data,
                    fit,
                    control,
                    seed + 77 + level_index,
                    features=features,
                )
                records.append(
                    {
                        "annotation_kind": kind,
                        "rotation": rotation,
                        "corruption": corruption,
                        "severity": severity,
                        "control": control,
                        "n_rows": len(data.rows),
                        **metrics,
                    }
                )
    return records


def source_contract(
    table_root: Path,
    pair_summary_path: Path,
) -> pd.DataFrame:
    pair: dict[str, Any] = {}
    if pair_summary_path.is_file():
        pair = pd.read_csv(pair_summary_path).iloc[0].to_dict()
    records = []
    for kind in KINDS:
        for experiment in (1, 2, 3):
            paths = table_paths(table_root, kind, experiment)
            rows = 0
            tracks = 0
            interpolated = 0
            invalid = 0
            states: set[int] = set()
            minimum_frame = math.inf
            maximum_frame = -math.inf
            for path in paths:
                frame = pd.read_csv(
                    path,
                    usecols=[
                        "track_id",
                        "frame",
                        "state",
                        "current_interpolated",
                        "x_px",
                        "y_px",
                        "dx_px",
                        "dy_px",
                        "target_dx_px",
                        "target_dy_px",
                    ],
                )
                rows += len(frame)
                finite_columns = [
                    "x_px",
                    "y_px",
                    "dx_px",
                    "dy_px",
                    "target_dx_px",
                    "target_dy_px",
                ]
                invalid += int(
                    (
                        ~np.isfinite(
                            frame[finite_columns].to_numpy(np.float64)
                        ).all(axis=1)
                    ).sum()
                )
                tracks += frame.track_id.nunique()
                interpolated += int(frame.current_interpolated.sum())
                states.update(int(value) for value in frame.state.unique())
                minimum_frame = min(minimum_frame, int(frame.frame.min()))
                maximum_frame = max(maximum_frame, int(frame.frame.max()))
            records.append(
                {
                    "annotation_kind": kind,
                    "experiment": experiment,
                    "fields": len(paths),
                    "rows": rows,
                    "tracks": tracks,
                    "minimum_frame": minimum_frame,
                    "maximum_frame": maximum_frame,
                    "states": ",".join(map(str, sorted(states))),
                    "current_interpolated_fraction": interpolated / max(rows, 1),
                    "invalid_transition_rows": invalid,
                    "valid_transition_coverage": 1.0 - invalid / max(rows, 1),
                    "frame_cadence": "one native frame",
                    "coordinate_unit": "pixel",
                    "future_or_target_features_used": False,
                    "paired_f0009_matches": pair.get("matches", np.nan),
                    "paired_f0009_position_median_px": pair.get(
                        "position_distance_median_px", np.nan
                    ),
                    "paired_f0009_step_disagreement_median_px": pair.get(
                        "step_disagreement_median_px", np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.max_rows_train = min(int(args.max_rows_train), 30_000)
        args.max_rows_val = min(int(args.max_rows_val), 20_000)
        args.max_rows_test = min(int(args.max_rows_test), 30_000)
        args.max_iter = min(int(args.max_iter), 30)
        args.rotations = "1-2-3"
    requested_rotations = {
        tuple(int(value) for value in item.split("-"))
        for item in args.rotations.split(",")
        if item.strip()
    }
    rotations = [rotation for rotation in ROTATIONS if rotation in requested_rotations]
    if not rotations:
        raise RuntimeError("No valid rotations requested")

    contract = source_contract(args.table_root, args.pair_summary)
    contract.to_csv(args.out_dir / "c2c12_reliability_contract.csv", index=False)

    rotation_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    corruption_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []

    for kind_index, kind in enumerate(KINDS):
        for rotation_index, (train_exp, val_exp, test_exp) in enumerate(rotations):
            rotation = f"{train_exp}-{val_exp}-{test_exp}"
            seed = int(args.seed) + 100_003 * kind_index + 10_007 * rotation_index
            print(f"[v168] {kind} rotation={rotation}: loading", flush=True)
            train = load_experiment(
                args.table_root,
                kind,
                train_exp,
                int(args.max_rows_train),
                seed,
                float(args.local_bin_px),
            )
            validation = load_experiment(
                args.table_root,
                kind,
                val_exp,
                int(args.max_rows_val),
                seed + 1,
                float(args.local_bin_px),
            )
            test = load_experiment(
                args.table_root,
                kind,
                test_exp,
                int(args.max_rows_test),
                seed + 2,
                float(args.local_bin_px),
            )
            key_sets = [
                set(map(tuple, split.rows[["sequence", "frame", "track_id"]].to_numpy()))
                for split in (train, validation, test)
            ]
            if key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2]:
                raise RuntimeError("Outer-fold row-key overlap detected")
            for split_name, experiment, split in (
                ("train", train_exp, train),
                ("val", val_exp, validation),
                ("test", test_exp, test),
            ):
                fold_records.append(
                    {
                        "annotation_kind": kind,
                        "rotation": rotation,
                        "split": split_name,
                        "experiment": experiment,
                        "fields": int(split.rows.field.nunique()),
                        "sampled_rows": len(split.rows),
                        "total_rows": split.total_rows,
                        "invalid_rows_excluded": split.invalid_rows_excluded,
                        "valid_rows_before_sampling": (
                            split.total_rows - split.invalid_rows_excluded
                        ),
                        "row_key_sha256": split.row_key_sha256,
                        "target_sha256": hashlib.sha256(
                            np.ascontiguousarray(split.target).tobytes()
                        ).hexdigest(),
                        "scalers_fit": "train_only",
                        "target_leakage": False,
                    }
                )
            print(f"[v168] {kind} rotation={rotation}: fitting reliability", flush=True)
            fit = fit_reliability(train, validation, args, seed)
            baseline = baseline_predictions(train, validation, test, fit, args)
            baseline_scale = np.full(
                len(test.rows),
                max(fit.train_error_median * fit.calibration_factor, 1e-3),
                dtype=np.float32,
            )
            for method, prediction in baseline.items():
                rotation_records.append(
                    {
                        "annotation_kind": kind,
                        "rotation": rotation,
                        "train_experiment": train_exp,
                        "validation_experiment": val_exp,
                        "test_experiment": test_exp,
                        "family": "matched_baseline",
                        "method": method,
                        "n_rows": len(test.rows),
                        **prediction_metrics(test.target, prediction, baseline_scale),
                    }
                )

            evaluated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for control_index, control in enumerate(
                ("real", "constant", "row_shuffled", "wrong_track", "time_shuffled")
            ):
                prediction, scale, _gain, metrics = evaluate_control(
                    test, fit, control, seed + control_index * 1297
                )
                evaluated[control] = (prediction, scale)
                record = {
                    "annotation_kind": kind,
                    "rotation": rotation,
                    "train_experiment": train_exp,
                    "validation_experiment": val_exp,
                    "test_experiment": test_exp,
                    "family": "bounded_reliability_transport",
                    "method": f"reliability_{control}",
                    "control": control,
                    "n_rows": len(test.rows),
                    "fixed_gain": fit.fixed_gain,
                    "local_weight": fit.local_weight,
                    "gain_low": fit.gain_low,
                    "gain_high": fit.gain_high,
                    "calibration_factor": fit.calibration_factor,
                    **metrics,
                }
                control_records.append(record)
                rotation_records.append(record.copy())
                if control in {"real", "constant", "row_shuffled", "wrong_track", "time_shuffled"}:
                    calibration_records.extend(
                        calibration_rows(
                            test,
                            prediction,
                            scale,
                            {
                                "annotation_kind": kind,
                                "rotation": rotation,
                                "control": control,
                            },
                        )
                    )
            real_prediction, real_scale = evaluated["real"]
            for control in ("constant", "row_shuffled", "wrong_track", "time_shuffled"):
                control_prediction, control_scale = evaluated[control]
                bootstrap_records.append(
                    {
                        "annotation_kind": kind,
                        "rotation": rotation,
                        **per_field_delta(
                            test,
                            real_prediction,
                            real_scale,
                            control_prediction,
                            control_scale,
                            control,
                            seed + 31,
                        ),
                    }
                )
            corruption_records.extend(
                corruption_replay(test, fit, kind, rotation, args, seed + 61)
            )

    rotations_frame = pd.DataFrame(rotation_records)
    controls_frame = pd.DataFrame(control_records)
    corruptions_frame = pd.DataFrame(corruption_records)
    calibrations_frame = pd.DataFrame(calibration_records)
    folds_frame = pd.DataFrame(fold_records)
    bootstrap_frame = pd.DataFrame(bootstrap_records)
    rotations_frame.to_csv(
        args.out_dir / "c2c12_reliability_rotation_metrics.csv", index=False
    )
    controls_frame.to_csv(
        args.out_dir / "c2c12_reliability_controls.csv", index=False
    )
    corruptions_frame.to_csv(
        args.out_dir / "c2c12_corruption_replay.csv", index=False
    )
    calibrations_frame.to_csv(
        args.out_dir / "c2c12_reliability_calibration.csv", index=False
    )
    folds_frame.to_csv(
        args.out_dir / "c2c12_outer_fold_manifest.csv", index=False
    )
    bootstrap_frame.to_csv(
        args.out_dir / "c2c12_reliability_cluster_bootstrap.csv", index=False
    )

    pivot = controls_frame.pivot_table(
        index=["annotation_kind", "rotation"],
        columns="control",
        values=["student_t_nll", "coverage_error", "component_rmse"],
    )
    fold_passes: list[dict[str, Any]] = []
    for index, row in pivot.iterrows():
        kind, rotation = index
        nll_pass = all(
            row[("student_t_nll", "real")] < row[("student_t_nll", control)]
            for control in ("constant", "row_shuffled", "wrong_track", "time_shuffled")
        )
        coverage_pass = (
            row[("coverage_error", "real")]
            < row[("coverage_error", "constant")]
        )
        rmse_degradation = (
            row[("component_rmse", "real")]
            / max(row[("component_rmse", "constant")], EPS)
            - 1.0
        )
        fold_passes.append(
            {
                "annotation_kind": kind,
                "rotation": rotation,
                "nll_all_controls_pass": bool(nll_pass),
                "coverage_pass": bool(coverage_pass),
                "rmse_degradation_fraction": float(rmse_degradation),
                "fold_pass": bool(
                    nll_pass and coverage_pass and rmse_degradation <= 0.005
                ),
            }
        )
    fold_pass_frame = pd.DataFrame(fold_passes)
    monotonic_rows = []
    for (kind, rotation, corruption), group in corruptions_frame[
        (corruptions_frame.control == "real")
        & (corruptions_frame.corruption != "clean")
    ].groupby(["annotation_kind", "rotation", "corruption"]):
        ordered = group.sort_values("severity")
        monotonic = bool(
            np.all(
                np.diff(
                    ordered.mean_observation_corruption_score.to_numpy()
                )
                >= -1e-6
            )
        )
        monotonic_rows.append(
            {
                "annotation_kind": kind,
                "rotation": rotation,
                "corruption": corruption,
                "observation_score_monotonic": monotonic,
            }
        )
    monotonic_frame = pd.DataFrame(monotonic_rows)
    all_folds_pass = bool(fold_pass_frame.fold_pass.all()) if len(fold_pass_frame) else False
    corruption_pass = (
        bool(monotonic_frame.observation_score_monotonic.mean() >= 0.8)
        if len(monotonic_frame)
        else False
    )
    bootstrap_pass = bool(
        len(bootstrap_frame)
        and (bootstrap_frame.nll_gain_ci_low > 0.0).all()
    )
    constant_pair = controls_frame[
        controls_frame.control.isin(["real", "constant"])
    ].pivot_table(
        index=["annotation_kind", "rotation"],
        columns="control",
        values="component_rmse",
    )
    relative_point_gain = (
        100.0
        * (constant_pair["constant"] - constant_pair["real"])
        / np.maximum(constant_pair["constant"], EPS)
    )
    gain_by_annotation = relative_point_gain.groupby(level=0).mean()
    automatic_larger_than_manual = bool(
        {"automatic", "manual"}.issubset(gain_by_annotation.index)
        and gain_by_annotation["automatic"] > gain_by_annotation["manual"]
    )
    decision = (
        "pass"
        if (
            all_folds_pass
            and corruption_pass
            and bootstrap_pass
            and automatic_larger_than_manual
        )
        else "fail"
    )
    mean_controls = (
        controls_frame.groupby(["annotation_kind", "control"])[
            ["component_rmse", "student_t_nll", "coverage_error", "mean_scale"]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        "# C2C12 Reliability Transport v168",
        "",
        f"Decision: **{decision.upper()}**",
        "",
        "The head is a bounded reliability modifier, not an identified observation-noise variance and not a free displacement predictor.",
        "",
        "## Rotation-mean controls",
        "",
        mean_controls.to_markdown(index=False),
        "",
        "## Fold gates",
        "",
        fold_pass_frame.to_markdown(index=False),
        "",
        "## Corruption monotonicity",
        "",
        monotonic_frame.to_markdown(index=False),
        "",
        "## Cross-source and cluster gates",
        "",
        f"- Every experiment-cluster NLL confidence interval is positive: {bootstrap_pass}",
        (
            "- Mean point-RMSE gain, automatic tracks: "
            f"{gain_by_annotation.get('automatic', np.nan):.3f}%"
        ),
        (
            "- Mean point-RMSE gain, manual tracks: "
            f"{gain_by_annotation.get('manual', np.nan):.3f}%"
        ),
        (
            "- Point gain is larger on automatic/noisier tracks: "
            f"{automatic_larger_than_manual}"
        ),
        "",
        "## Interpretation",
        "",
        "- All three experiment rotations are outer held-out tests; fields are retained as bootstrap clusters.",
        "- Manual interpolation and paired manual/automatic disagreement are audit labels only and are absent from inference features.",
        "- The biological target is held fixed during corruption replay.",
        "- A failed gate keeps reliability as a diagnostic/calibration variable and forbids treating it as a particle-filter likelihood.",
    ]
    (args.out_dir / "c2c12_reliability_decision_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": decision,
        "elapsed_seconds": time.time() - started,
        "seed": int(args.seed),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": ["target_dx_px", "target_dy_px"],
        "future_or_target_features_used": False,
        "rotations": [list(item) for item in rotations],
        "experiment_cluster_ci_pass": bootstrap_pass,
        "automatic_point_gain_larger_than_manual": automatic_larger_than_manual,
        "point_gain_percent_by_annotation": {
            str(key): float(value)
            for key, value in gain_by_annotation.items()
        },
        "source_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for kind in KINDS
            for experiment in (1, 2, 3)
            for path in table_paths(args.table_root, kind, experiment)
        },
        "outputs": sorted(path.name for path in args.out_dir.iterdir()),
    }
    (args.out_dir / "c2c12_reliability_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pair-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "causal_innovation_state_space_v97_c2c12_audit_2026-07-21"
        / "c2c12_f0009_pair_summary.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotations", default="1-2-3,2-3-1,3-1-2")
    parser.add_argument("--max-rows-train", type=int, default=500_000)
    parser.add_argument("--max-rows-val", type=int, default=250_000)
    parser.add_argument("--max-rows-test", type=int, default=500_000)
    parser.add_argument("--local-bin-px", type=float, default=64.0)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--observation-score-weight", type=float, default=0.25)
    parser.add_argument("--gain-grid", default="-0.25,0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--local-weight-grid", default="0,0.1,0.25,0.5")
    parser.add_argument("--gain-low-grid", default="-0.1,0,0.1,0.25")
    parser.add_argument("--gain-high-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--scale-factor-grid", default="0.5,0.75,1,1.25,1.5,2,3")
    parser.add_argument("--kalman-alpha-grid", default="0.25,0.5,0.75,1")
    parser.add_argument("--ca-gain-grid", default="-0.25,0,0.25,0.5,0.75,1")
    parser.add_argument("--imm-threshold-grid", default="0.25,0.5,1,2,4")
    parser.add_argument("--imm-temperature-grid", default="0.1,0.25,0.5,1")
    parser.add_argument("--jitter-levels", default="0.25,0.5,1,2")
    parser.add_argument("--missing-levels", default="0.1,0.2,0.4")
    parser.add_argument("--delay-levels", default="0.25,0.5,1")
    parser.add_argument("--swap-levels", default="0.05,0.1,0.2")
    parser.add_argument("--fragmentation-levels", default="0.1,0.2,0.4")
    parser.add_argument(
        "--corruption-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
