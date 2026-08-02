#!/usr/bin/env python3
"""External C2C12 confirmation of completed-innovation transport.

The runner keeps the prediction contract used by LIT-Cell: the displacement
from t to t+1 is predicted before its target is observed, and only residuals of
transitions completed by t may update the next prediction. C2C12 experiments,
not rows or frames, form the outer train/validation/test rotations.

The conditional mean is the restricted reliability-aware C2C12 model from
v168. The external contribution tested here is the final LIT-Cell mechanism:
bounded multiscale transport of own and neighbour completed innovations. Local
radii are dimensionless multiples of the causally observed nearest-neighbour
distance, so no LaChance pixel radius is copied into C2C12.
"""

from __future__ import annotations

import argparse
import gc
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
from scipy.spatial import cKDTree
from scipy.stats import norm, t as student_t


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_c2c12_reliability_transport_v168 as v168  # noqa: E402


DEFAULT_TABLE_ROOT = ROOT / "new_data" / "c2c12_online" / "tables"
DEFAULT_OUT = ROOT / "outputs" / "c2c12_lit_cell_external_confirmation_v209_2026-08-02"
ROTATIONS = ((1, 2, 3), (2, 3, 1), (3, 1, 2))
HORIZONS = (1, 2, 4, 6)
OBJECTIVES: dict[str, tuple[dict[int, float], float]] = {
    "h1_strict": ({1: 0.80, 2: 0.10, 4: 0.06, 6: 0.04}, 0.5),
    "horizon_balanced": ({1: 0.10, 2: 0.10, 4: 0.20, 6: 0.60}, 0.5),
    "h6_utility": ({1: 0.10, 2: 0.10, 4: 0.20, 6: 0.60}, 10.0),
}
PRIMARY_OBJECTIVE = "horizon_balanced"
EPS = 1e-8
TAIL_PROBABILITY = 1e-6
NORMAL_SCORE_LIMIT = float(norm.ppf(1.0 - TAIL_PROBABILITY))
PIXEL_UM = 1.3
FRAME_MINUTES = 5.0
ROW_COLUMNS = [
    "sequence",
    "experiment",
    "field",
    "frame",
    "track_id",
    "x_px",
    "y_px",
    "state",
    "previous_interpolated",
    "current_interpolated",
    "target_interpolated",
    "target_dx_px",
    "target_dy_px",
]


@dataclass
class LoadedExperiment:
    data: v168.SplitData
    sampling: pd.DataFrame


@dataclass
class FieldState:
    annotation_kind: str
    split: str
    experiment: int
    sequence: int
    field: int
    rows: pd.DataFrame
    target: np.ndarray
    base: np.ndarray
    scale: np.ndarray
    normal_score: np.ndarray
    packet_names: list[str]
    packets: dict[str, np.ndarray]
    latest_real_donor: np.ndarray
    latest_stale_donor: np.ndarray
    frame_dnn_px: np.ndarray
    windows: dict[int, np.ndarray]
    baselines: dict[str, np.ndarray]


@dataclass
class RidgeStatistics:
    row_mean: np.ndarray
    row_scale: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray
    design_names: list[str]
    selected_windows: int


@dataclass
class RidgeModel:
    row_mean: np.ndarray
    row_scale: np.ndarray
    coefficients: np.ndarray
    design_names: list[str]


@dataclass
class EquivariantRidgeStatistics:
    vector_scale: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray
    design_names: list[str]
    selected_windows: int


@dataclass
class EquivariantRidgeModel:
    vector_scale: np.ndarray
    coefficients: np.ndarray
    design_names: list[str]


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(finite(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def checked_matmul(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    """Run Accelerate-backed matmul and reject any non-finite result.

    Apple's BLAS occasionally leaves floating-point status flags set after a
    finite multiplication and NumPy reports all three RuntimeWarning variants.
    The explicit postcondition keeps the numerical contract strict without
    flooding long experiment logs with those false-positive status warnings.
    """

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = np.matmul(left, right)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Non-finite matrix product: {label}")
    return result


def parse_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(item) for item in value]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(item) for item in value]


def constant_student_scale(
    train_target: np.ndarray,
    train_prediction: np.ndarray,
    validation_target: np.ndarray,
    validation_prediction: np.ndarray,
    factors: list[float],
) -> float:
    residual = np.abs(
        np.asarray(train_target, dtype=np.float64)
        - np.asarray(train_prediction, dtype=np.float64)
    )
    half_width = max(float(np.median(residual)), 1e-3)
    unit = half_width / float(student_t.ppf(0.75, df=v168.STUDENT_DF))
    factor = min(
        factors,
        key=lambda value: v168.student_nll(
            validation_target,
            validation_prediction,
            np.full(len(validation_target), unit * value, dtype=np.float64),
        ),
    )
    return max(unit * float(factor), 1e-3)


def select_velocity_reliability_blend(
    validation: v168.SplitData,
    reliability_prediction: np.ndarray,
    velocity_prediction: np.ndarray,
    weights: list[float],
) -> tuple[float, pd.DataFrame]:
    objective_weights = OBJECTIVES[PRIMARY_OBJECTIVE][0]
    reference_rmse: dict[int, float] = {}
    windows_by_horizon: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        windows = consecutive_windows(validation.rows, horizon)
        windows_by_horizon[horizon] = windows
        reference_rmse[horizon] = float(
            np.sqrt(
                np.mean(
                    np.square(
                        reliability_prediction[windows].sum(axis=1)
                        - validation.target[windows].sum(axis=1)
                    )
                )
            )
        )
    records: list[dict[str, float]] = []
    for reliability_weight in weights:
        prediction = (
            reliability_weight * reliability_prediction
            + (1.0 - reliability_weight) * velocity_prediction
        )
        row: dict[str, float] = {
            "reliability_weight": float(reliability_weight)
        }
        score = 0.0
        for horizon, horizon_weight in objective_weights.items():
            windows = windows_by_horizon[horizon]
            rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            prediction[windows].sum(axis=1)
                            - validation.target[windows].sum(axis=1)
                        )
                    )
                )
            )
            row[f"h{horizon}_rmse"] = rmse
            score += horizon_weight * rmse / max(reference_rmse[horizon], EPS)
        row["validation_score"] = score
        row["h1_gain_vs_reliability_percent"] = 100.0 * (
            reference_rmse[1] - row["h1_rmse"]
        ) / max(reference_rmse[1], EPS)
        records.append(row)
    grid = pd.DataFrame(records)
    eligible = grid[grid["h1_gain_vs_reliability_percent"] >= -0.5]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        ["validation_score", "h1_gain_vs_reliability_percent", "reliability_weight"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["reliability_weight"]), grid


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_block_indices(
    rows: pd.DataFrame,
    maximum_rows: int,
    blocks: int,
) -> np.ndarray:
    """Select complete, distributed contiguous frame blocks."""

    if maximum_rows <= 0 or len(rows) <= maximum_rows:
        return np.arange(len(rows), dtype=np.int64)
    counts = rows.groupby("frame", sort=True).size()
    frames = counts.index.to_numpy(np.int64)
    mean_rows = max(float(counts.mean()), 1.0)
    target_frames = max(int(maximum_rows / mean_rows), 2 * max(HORIZONS))
    target_frames = min(target_frames, len(frames))
    block_count = max(1, min(int(blocks), target_frames // max(HORIZONS)))
    width = max(max(HORIZONS) + 1, int(math.ceil(target_frames / block_count)))
    if width * block_count >= len(frames):
        chosen_frames = frames
    else:
        centers = np.linspace(width / 2, len(frames) - width / 2, block_count)
        chosen: set[int] = set()
        for center in centers:
            start = int(round(center - width / 2))
            start = max(0, min(start, len(frames) - width))
            chosen.update(int(value) for value in frames[start : start + width])
        chosen_frames = np.asarray(sorted(chosen), dtype=np.int64)
    mask = rows["frame"].isin(chosen_frames).to_numpy()
    return np.flatnonzero(mask)


def load_experiment_blocked(
    table_root: Path,
    kind: str,
    experiment: int,
    maximum_rows: int,
    blocks: int,
    local_bin_px: float,
) -> LoadedExperiment:
    row_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    sampling_records: list[dict[str, Any]] = []
    total_rows = 0
    invalid_rows = 0
    for path in v168.table_paths(table_root, kind, experiment):
        table = pd.read_csv(path, usecols=v168.RAW_COLUMNS)
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
        engineered_rows, engineered_features = v168.engineer_table(
            table, local_bin_px
        )
        per_file_limit = (
            max(1, int(math.ceil(maximum_rows / 16)))
            if maximum_rows > 0
            else 0
        )
        selected = frame_block_indices(engineered_rows, per_file_limit, blocks)
        selected_rows = engineered_rows.iloc[selected][ROW_COLUMNS].reset_index(drop=True)
        selected_features = engineered_features.iloc[selected].reset_index(drop=True)
        selected_frames = selected_rows["frame"].nunique()
        frame_values = np.sort(selected_rows["frame"].unique())
        gap_count = int(np.sum(np.diff(frame_values) > 1)) if len(frame_values) > 1 else 0
        sampling_records.append(
            {
                "annotation_kind": kind,
                "experiment": experiment,
                "sequence": int(selected_rows["sequence"].iloc[0]),
                "field": int(selected_rows["field"].iloc[0]),
                "source": str(path),
                "valid_source_rows": len(engineered_rows),
                "selected_rows": len(selected_rows),
                "selected_frames": int(selected_frames),
                "contiguous_blocks": gap_count + 1,
                "selection_fraction": len(selected_rows) / max(len(engineered_rows), 1),
            }
        )
        row_parts.append(selected_rows)
        feature_parts.append(selected_features)
    rows = pd.concat(row_parts, ignore_index=True)
    features = pd.concat(feature_parts, ignore_index=True)
    target = rows[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
    data = v168.SplitData(
        rows=rows,
        features=features,
        target=target,
        total_rows=total_rows,
        invalid_rows_excluded=invalid_rows,
        row_key_sha256=v168.row_key_hash(rows),
    )
    return LoadedExperiment(data=data, sampling=pd.DataFrame(sampling_records))


def nearest_neighbour_scale(position: np.ndarray, fallback: float) -> float:
    if len(position) < 2:
        return float(fallback)
    distances, _indices = cKDTree(position).query(position, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-6)]
    return float(np.median(nearest)) if len(nearest) else float(fallback)


def experiment_dnn_contract(data: v168.SplitData) -> tuple[float, float, float, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (sequence, frame), indices in data.rows.groupby(
        ["sequence", "frame"], sort=True
    ).groups.items():
        if int(frame) % 5 != 0:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        position = data.rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float64)
        value = nearest_neighbour_scale(position, 32.0)
        records.append(
            {
                "sequence": int(sequence),
                "frame": int(frame),
                "cells": len(indices),
                "dnn_px": value,
            }
        )
    frame = pd.DataFrame(records)
    values = frame["dnn_px"].to_numpy(np.float64)
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.05)),
        float(np.quantile(values, 0.95)),
        frame,
    )


def local_state(
    current_position: np.ndarray,
    previous_position: np.ndarray,
    previous_score: np.ndarray,
    current_tracks: np.ndarray,
    previous_tracks: np.ndarray,
    scales: list[float],
) -> dict[str, np.ndarray]:
    if not np.isfinite(current_position).all():
        raise FloatingPointError("Non-finite current positions in local state")
    if not np.isfinite(previous_position).all():
        raise FloatingPointError("Non-finite previous positions in local state")
    if not np.isfinite(previous_score).all():
        raise FloatingPointError("Non-finite completed innovations in local state")
    distance = np.linalg.norm(
        current_position[:, None, :] - previous_position[None, :, :], axis=2
    )
    same_track = current_tracks[:, None] == previous_tracks[None, :]
    output: dict[str, np.ndarray] = {}
    for multiplier, scale in scales:
        weights = np.exp(-0.5 * np.square(distance / max(scale, 1e-3)))
        weights[same_track] = 0.0
        weight_sum = weights.sum(axis=1, keepdims=True)
        normalized = weights / np.maximum(weight_sum, EPS)
        mean = checked_matmul(
            normalized, previous_score, f"local mean scale={multiplier:g}"
        )
        centered = previous_score[None, :, :] - mean[:, None, :]
        variance = np.einsum(
            "ij,ijk->ik", normalized, np.square(centered), optimize=True
        )
        effective_count = np.square(weight_sum[:, 0]) / np.maximum(
            np.sum(np.square(weights), axis=1), EPS
        )
        if not all(
            np.isfinite(value).all()
            for value in (weights, normalized, mean, variance, effective_count)
        ):
            raise FloatingPointError(
                f"Non-finite local state at dimensionless scale {multiplier:g}"
            )
        label = f"m{multiplier:g}".replace(".", "p")
        output[f"local_{label}_x"] = mean[:, 0]
        output[f"local_{label}_y"] = mean[:, 1]
        output[f"local_{label}_std_x"] = np.sqrt(np.maximum(variance[:, 0], 0.0))
        output[f"local_{label}_std_y"] = np.sqrt(np.maximum(variance[:, 1], 0.0))
        output[f"local_{label}_effective_n"] = effective_count
    return output


def wrong_cell_packet(
    real: np.ndarray, rows: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows), dtype=np.int64)
    for _frame, raw_indices in rows.groupby("frame", sort=True).groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if len(indices) > 1:
            cycle = rng.permutation(indices)
            permutation[cycle] = np.roll(cycle, 1)
            if np.any(permutation[indices] == indices):
                raise RuntimeError("Wrong-cell packet retained its receiver")
    return real[permutation].copy(), permutation


def stale_packet(
    real: np.ndarray,
    rows: pd.DataFrame,
    latest_real_donor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (int(track), int(frame)): index
        for index, (track, frame) in enumerate(
            rows[["track_id", "frame"]].itertuples(index=False)
        )
    }
    stale = np.zeros_like(real)
    donor = np.full(len(rows), -1, dtype=np.int64)
    for index, (track, frame) in enumerate(
        rows[["track_id", "frame"]].itertuples(index=False)
    ):
        source = lookup.get((int(track), int(frame) - 1), -1)
        if source >= 0:
            stale[index] = real[source]
            donor[index] = latest_real_donor[source]
    return stale, donor


def build_packet(
    rows: pd.DataFrame,
    normal_score: np.ndarray,
    multipliers: list[float],
    dnn_reference: float,
    dnn_low: float,
    dnn_high: float,
    adaptive: bool,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[str], np.ndarray, np.ndarray, np.ndarray]:
    count = len(rows)
    feature: dict[str, np.ndarray] = {
        "own_prev_x": np.zeros(count, dtype=np.float32),
        "own_prev_y": np.zeros(count, dtype=np.float32),
        "own_available": np.zeros(count, dtype=np.float32),
        "global_prev_x": np.zeros(count, dtype=np.float32),
        "global_prev_y": np.zeros(count, dtype=np.float32),
    }
    for multiplier in multipliers:
        label = f"m{multiplier:g}".replace(".", "p")
        for suffix in ("x", "y", "std_x", "std_y", "effective_n"):
            feature[f"local_{label}_{suffix}"] = np.zeros(count, dtype=np.float32)
    latest = np.full(count, -1, dtype=np.int64)
    row_dnn = np.full(count, dnn_reference, dtype=np.float32)
    frame_groups = rows.groupby("frame", sort=True).indices
    lookup = {
        (int(frame), int(track)): index
        for index, (frame, track) in enumerate(
            rows[["frame", "track_id"]].itertuples(index=False)
        )
    }
    for frame, raw_current in frame_groups.items():
        current = np.asarray(raw_current, dtype=np.int64)
        current_position = rows.iloc[current][["x_px", "y_px"]].to_numpy(np.float64)
        observed_dnn = nearest_neighbour_scale(current_position, dnn_reference)
        observed_dnn = float(np.clip(observed_dnn, dnn_low, dnn_high))
        base_scale = observed_dnn if adaptive else dnn_reference
        row_dnn[current] = observed_dnn
        previous = np.asarray(frame_groups.get(int(frame) - 1, []), dtype=np.int64)
        if not len(previous):
            continue
        latest[current] = int(frame) - 1
        previous_score = normal_score[previous]
        global_state = previous_score.mean(axis=0)
        feature["global_prev_x"][current] = global_state[0]
        feature["global_prev_y"][current] = global_state[1]
        current_tracks = rows.iloc[current]["track_id"].to_numpy(np.int64)
        previous_tracks = rows.iloc[previous]["track_id"].to_numpy(np.int64)
        own_indices = np.asarray(
            [lookup.get((int(frame) - 1, int(track)), -1) for track in current_tracks],
            dtype=np.int64,
        )
        available = own_indices >= 0
        if np.any(available):
            receivers = current[available]
            donors = own_indices[available]
            feature["own_prev_x"][receivers] = normal_score[donors, 0]
            feature["own_prev_y"][receivers] = normal_score[donors, 1]
            feature["own_available"][receivers] = 1.0
        scale_values = [(value, value * base_scale) for value in multipliers]
        local = local_state(
            current_position,
            rows.iloc[previous][["x_px", "y_px"]].to_numpy(np.float64),
            previous_score,
            current_tracks,
            previous_tracks,
            scale_values,
        )
        for name, value in local.items():
            feature[name][current] = value.astype(np.float32)
    names = list(feature)
    real = np.column_stack([feature[name] for name in names]).astype(np.float32)
    wrong, permutation = wrong_cell_packet(real, rows, seed)
    stale, stale_latest = stale_packet(real, rows, latest)
    issue_frame = rows["frame"].to_numpy(np.int64)
    if np.any((latest >= 0) & (latest > issue_frame - 1)):
        raise RuntimeError("Future donor detected in real packet")
    if np.any((stale_latest >= 0) & (stale_latest > issue_frame - 2)):
        raise RuntimeError("Stale control is not at least one extra frame old")
    packets = {"real": real, "wrong_cell": wrong, "stale_time": stale}
    return packets, names, latest, stale_latest, row_dnn


def consecutive_windows(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    if horizon == 1:
        return np.arange(len(rows), dtype=np.int64)[:, None]
    lookup = {
        (int(track), int(frame)): index
        for index, (track, frame) in enumerate(
            rows[["track_id", "frame"]].itertuples(index=False)
        )
    }
    windows: list[list[int]] = []
    for track, frame in rows[["track_id", "frame"]].itertuples(index=False):
        indices = [
            lookup.get((int(track), int(frame) + offset))
            for offset in range(horizon)
        ]
        if all(index is not None for index in indices):
            windows.append([int(index) for index in indices])
    return (
        np.asarray(windows, dtype=np.int64)
        if windows
        else np.empty((0, horizon), dtype=np.int64)
    )


def make_fields(
    data: v168.SplitData,
    annotation_kind: str,
    split: str,
    base_prediction: np.ndarray,
    base_scale: np.ndarray,
    baseline_predictions: dict[str, np.ndarray],
    multipliers: list[float],
    dnn_contract: tuple[float, float, float],
    adaptive: bool,
    seed: int,
) -> list[FieldState]:
    dnn_reference, dnn_low, dnn_high = dnn_contract
    fields: list[FieldState] = []
    for sequence, raw_indices in data.rows.groupby("sequence", sort=True).groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        subset = data.rows.iloc[indices]
        order = np.lexsort(
            (
                subset["track_id"].to_numpy(np.int64),
                subset["frame"].to_numpy(np.int64),
            )
        )
        selected = indices[order]
        rows = data.rows.iloc[selected].reset_index(drop=True)
        target = np.asarray(data.target[selected], dtype=np.float64)
        mean = np.asarray(base_prediction[selected], dtype=np.float64)
        raw_scale = np.asarray(base_scale[selected], dtype=np.float64).reshape(-1)
        if not np.isfinite(target).all() or not np.isfinite(mean).all():
            raise FloatingPointError(
                f"Non-finite target or coordinate mean in sequence {sequence}"
            )
        if not np.isfinite(raw_scale).all():
            raise FloatingPointError(
                f"Non-finite uncertainty scale in sequence {sequence}"
            )
        scale = np.repeat(np.maximum(raw_scale[:, None], 1e-3), 2, axis=1)
        standardized = np.clip((target - mean) / scale, -1e6, 1e6)
        uniform = np.clip(
            student_t.cdf(standardized, df=v168.STUDENT_DF),
            TAIL_PROBABILITY,
            1.0 - TAIL_PROBABILITY,
        )
        normal_score = np.clip(
            norm.ppf(uniform), -NORMAL_SCORE_LIMIT, NORMAL_SCORE_LIMIT
        )
        if not np.isfinite(normal_score).all():
            raise FloatingPointError(
                f"Non-finite normal-score innovations in sequence {sequence}"
            )
        packets, names, latest, stale_latest, row_dnn = build_packet(
            rows,
            normal_score,
            multipliers,
            dnn_reference,
            dnn_low,
            dnn_high,
            adaptive,
            seed + int(sequence) * 1009,
        )
        windows = {horizon: consecutive_windows(rows, horizon) for horizon in HORIZONS}
        fields.append(
            FieldState(
                annotation_kind=annotation_kind,
                split=split,
                experiment=int(rows["experiment"].iloc[0]),
                sequence=int(sequence),
                field=int(rows["field"].iloc[0]),
                rows=rows,
                target=target,
                base=mean,
                scale=scale,
                normal_score=normal_score,
                packet_names=names,
                packets=packets,
                latest_real_donor=latest,
                latest_stale_donor=stale_latest,
                frame_dnn_px=row_dnn,
                windows=windows,
                baselines={
                    name: np.asarray(prediction[selected], dtype=np.float64)
                    for name, prediction in baseline_predictions.items()
                },
            )
        )
    return fields


def raw_design(field: FieldState, control: str = "real") -> np.ndarray:
    packet = np.asarray(field.packets[control], dtype=np.float64)
    design = np.column_stack(
        [packet, packet * field.scale[:, 0:1], packet * field.scale[:, 1:2]]
    )
    if not np.isfinite(design).all():
        raise FloatingPointError(
            f"Non-finite design matrix for sequence {field.sequence}, control {control}"
        )
    return design


def design_names(field: FieldState) -> list[str]:
    return (
        [f"raw:{name}" for name in field.packet_names]
        + [f"scale_x:{name}" for name in field.packet_names]
        + [f"scale_y:{name}" for name in field.packet_names]
    )


def equivariant_vector_design(
    field: FieldState, control: str = "real"
) -> tuple[np.ndarray, list[str]]:
    packet = np.asarray(field.packets[control], dtype=np.float64)
    index = {name: position for position, name in enumerate(field.packet_names)}
    local_names = sorted(
        name[:-2]
        for name in field.packet_names
        if name.startswith("local_")
        and name.endswith("_x")
        and "_std_" not in name
        and f"{name[:-2]}_y" in index
    )
    source_names = ["own_prev", "global_prev", *local_names]
    vectors: list[np.ndarray] = []
    valid_names: list[str] = []
    for name in source_names:
        x_name = f"{name}_x"
        y_name = f"{name}_y"
        if x_name not in index or y_name not in index:
            continue
        vectors.append(packet[:, [index[x_name], index[y_name]]])
        valid_names.append(name)
    raw = np.stack(vectors, axis=1)
    uncertainty_scaled = raw * field.scale[:, None, 0:1]
    design = np.concatenate([raw, uncertainty_scaled], axis=1)
    names = [f"raw:{name}" for name in valid_names] + [
        f"scale:{name}" for name in valid_names
    ]
    if not np.isfinite(design).all():
        raise FloatingPointError(
            f"Non-finite equivariant design for sequence {field.sequence}, control {control}"
        )
    return design, names


def equivariant_family_mask(
    model: EquivariantRidgeModel, family: str
) -> np.ndarray:
    if family == "full":
        return np.ones(len(model.design_names), dtype=bool)
    base_names = [name.split(":", 1)[1] for name in model.design_names]
    if family == "own_only":
        return np.asarray([name == "own_prev" for name in base_names])
    if family == "neighbour_only":
        return np.asarray([name != "own_prev" for name in base_names])
    raise ValueError(family)


def row_normalization(fields: list[FieldState]) -> tuple[np.ndarray, np.ndarray]:
    dimension = raw_design(fields[0]).shape[1]
    total = 0
    sums = np.zeros(dimension, dtype=np.float64)
    sums_sq = np.zeros(dimension, dtype=np.float64)
    for field in fields:
        matrix = raw_design(field)
        total += len(matrix)
        sums += matrix.sum(axis=0)
        sums_sq += np.square(matrix).sum(axis=0)
    mean = sums / max(total, 1)
    variance = np.maximum(sums_sq / max(total, 1) - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), 1e-8)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise FloatingPointError("Non-finite train-only design normalization")
    return mean, scale


def capped_windows(windows: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(windows) <= maximum:
        return windows
    selected = np.linspace(0, len(windows) - 1, maximum, dtype=np.int64)
    return windows[selected]


def build_ridge_statistics(
    fields: list[FieldState],
    weights_by_horizon: dict[int, float],
    maximum_windows: int,
) -> RidgeStatistics:
    row_mean, row_scale = row_normalization(fields)
    groups: list[tuple[FieldState, int, np.ndarray]] = []
    groups_per_horizon: dict[int, int] = {horizon: 0 for horizon in HORIZONS}
    total_selected = 0
    for field in fields:
        for horizon in HORIZONS:
            windows = capped_windows(field.windows[horizon], maximum_windows)
            if not len(windows):
                continue
            groups.append((field, horizon, windows))
            groups_per_horizon[horizon] += 1
            total_selected += len(windows)
    if not groups:
        raise RuntimeError("No consecutive training windows")
    dimension = len(row_mean) + 1
    gram = np.zeros((dimension, dimension), dtype=np.float64)
    rhs = np.zeros((dimension, 2), dtype=np.float64)
    for field, horizon, windows in groups:
        normalized = (raw_design(field) - row_mean) / row_scale
        per_step = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        x = per_step[windows].sum(axis=1)
        residual = field.target - field.base
        y = residual[windows].sum(axis=1)
        group_weight = (
            total_selected
            * weights_by_horizon[horizon]
            / max(groups_per_horizon[horizon] * len(windows), 1)
        )
        gram += group_weight * checked_matmul(x.T, x, "ridge Gram")
        rhs += group_weight * checked_matmul(x.T, y, "ridge right-hand side")
    return RidgeStatistics(
        row_mean=row_mean,
        row_scale=row_scale,
        gram=gram,
        rhs=rhs,
        design_names=design_names(fields[0]),
        selected_windows=total_selected,
    )


def build_equivariant_statistics(
    fields: list[FieldState],
    weights_by_horizon: dict[int, float],
    maximum_windows: int,
) -> EquivariantRidgeStatistics:
    first_design, names = equivariant_vector_design(fields[0])
    energy = np.zeros(first_design.shape[1], dtype=np.float64)
    observations = 0
    for field in fields:
        design, field_names = equivariant_vector_design(field)
        if field_names != names:
            raise RuntimeError("Equivariant design names changed between fields")
        energy += np.square(design).sum(axis=(0, 2))
        observations += design.shape[0] * design.shape[2]
    vector_scale = np.maximum(
        np.sqrt(energy / max(observations, 1)), 1e-8
    )
    groups: list[tuple[FieldState, int, np.ndarray]] = []
    groups_per_horizon: dict[int, int] = {horizon: 0 for horizon in HORIZONS}
    total_selected = 0
    for field in fields:
        for horizon in HORIZONS:
            windows = capped_windows(field.windows[horizon], maximum_windows)
            if not len(windows):
                continue
            groups.append((field, horizon, windows))
            groups_per_horizon[horizon] += 1
            total_selected += len(windows)
    dimension = len(names)
    gram = np.zeros((dimension, dimension), dtype=np.float64)
    rhs = np.zeros(dimension, dtype=np.float64)
    for field, horizon, windows in groups:
        design, _names = equivariant_vector_design(field)
        normalized = design / vector_scale[None, :, None]
        x = normalized[windows].sum(axis=1)
        y = (field.target - field.base)[windows].sum(axis=1)
        group_weight = (
            total_selected
            * weights_by_horizon[horizon]
            / max(groups_per_horizon[horizon] * len(windows), 1)
        )
        for component in range(2):
            gram += group_weight * checked_matmul(
                x[:, :, component].T,
                x[:, :, component],
                "equivariant Ridge Gram",
            )
            rhs += group_weight * checked_matmul(
                x[:, :, component].T,
                y[:, component],
                "equivariant Ridge right-hand side",
            )
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("Non-finite equivariant Ridge statistics")
    return EquivariantRidgeStatistics(
        vector_scale=vector_scale,
        gram=gram,
        rhs=rhs,
        design_names=names,
        selected_windows=total_selected,
    )


def solve_ridge(statistics: RidgeStatistics, alpha: float) -> RidgeModel:
    penalty = np.eye(len(statistics.gram), dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(statistics.gram + penalty, statistics.rhs)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError(f"Non-finite Ridge coefficients at alpha={alpha:g}")
    return RidgeModel(
        row_mean=statistics.row_mean,
        row_scale=statistics.row_scale,
        coefficients=coefficients,
        design_names=statistics.design_names,
    )


def solve_equivariant_ridge(
    statistics: EquivariantRidgeStatistics, alpha: float
) -> EquivariantRidgeModel:
    penalty = np.eye(len(statistics.gram), dtype=np.float64) * float(alpha)
    coefficients = np.linalg.solve(statistics.gram + penalty, statistics.rhs)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError(
            f"Non-finite equivariant Ridge coefficients at alpha={alpha:g}"
        )
    return EquivariantRidgeModel(
        vector_scale=statistics.vector_scale,
        coefficients=coefficients,
        design_names=statistics.design_names,
    )


def family_mask(model: RidgeModel, family: str) -> np.ndarray:
    names = model.design_names
    if family == "full":
        return np.ones(len(names), dtype=bool)
    base_names = [name.split(":", 1)[1] for name in names]
    if family == "own_only":
        return np.asarray([name.startswith("own_") for name in base_names])
    if family == "neighbour_only":
        return np.asarray(
            [name.startswith("local_") or name.startswith("global_") for name in base_names]
        )
    raise ValueError(family)


def correction(
    model: RidgeModel,
    field: FieldState,
    control: str,
    family: str = "full",
) -> np.ndarray:
    normalized = (raw_design(field, control) - model.row_mean) / model.row_scale
    keep = family_mask(model, family)
    normalized[:, ~keep] = 0.0
    augmented = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    return checked_matmul(augmented, model.coefficients, "ridge correction")


def equivariant_correction(
    model: EquivariantRidgeModel,
    field: FieldState,
    control: str,
    family: str = "full",
) -> np.ndarray:
    design, names = equivariant_vector_design(field, control)
    if names != model.design_names:
        raise RuntimeError("Equivariant inference design does not match training")
    normalized = design / model.vector_scale[None, :, None]
    keep = equivariant_family_mask(model, family)
    coefficients = model.coefficients.copy()
    coefficients[~keep] = 0.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = np.einsum("nkd,k->nd", normalized, coefficients, optimize=True)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite equivariant correction")
    return result


def transport_statistics(
    fields: list[FieldState],
    weights: dict[int, float],
    maximum_windows: int,
    operator_kind: str,
) -> RidgeStatistics | EquivariantRidgeStatistics:
    if operator_kind == "free":
        return build_ridge_statistics(fields, weights, maximum_windows)
    if operator_kind == "equivariant":
        return build_equivariant_statistics(fields, weights, maximum_windows)
    raise ValueError(operator_kind)


def solve_transport(
    statistics: RidgeStatistics | EquivariantRidgeStatistics,
    alpha: float,
    operator_kind: str,
) -> RidgeModel | EquivariantRidgeModel:
    if operator_kind == "free":
        if not isinstance(statistics, RidgeStatistics):
            raise TypeError("Free operator received equivariant statistics")
        return solve_ridge(statistics, alpha)
    if not isinstance(statistics, EquivariantRidgeStatistics):
        raise TypeError("Equivariant operator received free statistics")
    return solve_equivariant_ridge(statistics, alpha)


def transport_correction(
    model: RidgeModel | EquivariantRidgeModel,
    field: FieldState,
    control: str,
    family: str,
    operator_kind: str,
) -> np.ndarray:
    if operator_kind == "free":
        if not isinstance(model, RidgeModel):
            raise TypeError("Free correction received equivariant model")
        return correction(model, field, control, family)
    if not isinstance(model, EquivariantRidgeModel):
        raise TypeError("Equivariant correction received free model")
    return equivariant_correction(model, field, control, family)


def bounded_update(value: np.ndarray, bound_px: float) -> np.ndarray:
    if bound_px <= 0:
        return value
    length = np.linalg.norm(value, axis=1, keepdims=True)
    bounded_length = np.tanh(length / float(bound_px)) * float(bound_px)
    return value * bounded_length / np.maximum(length, EPS)


def prediction_metrics(
    field: FieldState,
    prediction: np.ndarray,
    method: str,
    objective: str,
    control: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        windows = field.windows[horizon]
        if not len(windows):
            continue
        target = field.target[windows].sum(axis=1)
        predicted = prediction[windows].sum(axis=1)
        baseline = field.base[windows].sum(axis=1)
        error = predicted - target
        baseline_error = baseline - target
        component_rmse = float(np.sqrt(np.mean(np.square(error))))
        base_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
        vector_rmse = float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
        centered = target - target.mean(axis=0, keepdims=True)
        r2 = 1.0 - float(
            np.sum(np.square(error)) / max(np.sum(np.square(centered)), EPS)
        )
        numerator = np.sum(target * predicted, axis=1)
        denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(predicted, axis=1)
        valid = denominator > EPS
        cosine = float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.0
        row: dict[str, Any] = {
            "annotation_kind": field.annotation_kind,
            "split": field.split,
            "experiment": field.experiment,
            "sequence": field.sequence,
            "field": field.field,
            "objective": objective,
            "method": method,
            "control": control,
            "horizon": horizon,
            "windows": len(windows),
            "component_rmse": component_rmse,
            "component_rmse_um": component_rmse * PIXEL_UM,
            "vector_rmse": vector_rmse,
            "r2": r2,
            "cosine": cosine,
            "base_component_rmse": base_rmse,
            "gain_vs_base_percent": 100.0 * (base_rmse - component_rmse) / max(base_rmse, EPS),
        }
        if horizon == 1:
            uncertainty = v168.uncertainty_metrics(
                field.target, prediction, field.scale[:, 0]
            )
            row.update(uncertainty)
        output.append(row)
    return output


def validation_score(
    records: list[dict[str, Any]], weights: dict[int, float]
) -> tuple[float, float, float]:
    frame = pd.DataFrame(records)
    ratios = frame["component_rmse"] / np.maximum(frame["base_component_rmse"], EPS)
    frame = frame.assign(ratio=ratios)
    macro = frame.groupby("horizon", as_index=True).agg(
        ratio=("ratio", "mean"), gain=("gain_vs_base_percent", "mean")
    )
    score = float(sum(weights[h] * macro.loc[h, "ratio"] for h in weights))
    return score, float(macro.loc[1, "gain"]), float(macro.loc[6, "gain"])


def select_transport(
    train_fields: list[FieldState],
    validation_fields: list[FieldState],
    objective: str,
    alphas: list[float],
    bound_factors: list[float],
    maximum_windows: int,
    operator_kind: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    weights, h1_guard = OBJECTIVES[objective]
    statistics = transport_statistics(
        train_fields, weights, maximum_windows, operator_kind
    )
    residual_norm = np.concatenate(
        [np.linalg.norm(field.target - field.base, axis=1) for field in train_fields]
    )
    bound_unit = max(float(np.quantile(residual_norm, 0.75)), 0.05)
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        model = solve_transport(statistics, alpha, operator_kind)
        raw = {
            field.sequence: transport_correction(
                model, field, "real", "full", operator_kind
            )
            for field in validation_fields
        }
        for factor in bound_factors:
            bound_px = factor * bound_unit
            rows: list[dict[str, Any]] = []
            for field in validation_fields:
                prediction = field.base + bounded_update(raw[field.sequence], bound_px)
                rows.extend(prediction_metrics(field, prediction, "real", objective, "real"))
            score, h1_gain, h6_gain = validation_score(rows, weights)
            records.append(
                {
                    "objective": objective,
                    "operator_kind": operator_kind,
                    "alpha": alpha,
                    "bound_factor": factor,
                    "bound_px": bound_px,
                    "bound_unit_px": bound_unit,
                    "validation_score": score,
                    "validation_h1_gain_percent": h1_gain,
                    "validation_h6_gain_percent": h6_gain,
                    "training_windows": statistics.selected_windows,
                }
            )
    grid = pd.DataFrame(records)
    eligible = grid[grid["validation_h1_gain_percent"] >= -h1_guard]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        ["validation_score", "validation_h1_gain_percent", "bound_px", "alpha"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return {
        key: float(best[key])
        for key in best.index
        if key not in {"objective", "operator_kind"}
    }, grid


def aggregate_metrics(field_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        field_metrics.groupby(
            ["annotation_kind", "objective", "method", "control", "horizon"],
            as_index=False,
        )
        .agg(
            fields=("sequence", "nunique"),
            experiments=("experiment", "nunique"),
            component_rmse_macro=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            r2_macro=("r2", "mean"),
            cosine_macro=("cosine", "mean"),
            gain_vs_base_percent_macro=("gain_vs_base_percent", "mean"),
            fields_improved=("gain_vs_base_percent", lambda values: int((values > 0).sum())),
        )
    )


def hierarchical_bootstrap(
    field_metrics: pd.DataFrame,
    objective: str,
    left_control: str,
    right_control: str,
    annotation_kind: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    subset = field_metrics[
        (field_metrics["annotation_kind"] == annotation_kind)
        & (field_metrics["objective"] == objective)
        & (field_metrics["horizon"] == 6)
        & (field_metrics["control"].isin([left_control, right_control]))
    ]
    pivot = subset.pivot_table(
        index=["experiment", "sequence"], columns="control", values="component_rmse"
    ).dropna()
    delta = pivot[right_control] - pivot[left_control]
    frame = delta.rename("advantage_px").reset_index()
    experiments = np.sort(frame["experiment"].unique())
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_experiments = rng.choice(experiments, len(experiments), replace=True)
        values: list[float] = []
        for experiment in selected_experiments:
            group = frame[frame["experiment"] == experiment]["advantage_px"].to_numpy()
            values.extend(rng.choice(group, len(group), replace=True).tolist())
        samples[repetition] = float(np.mean(values))
    return {
        "annotation_kind": annotation_kind,
        "objective": objective,
        "horizon": 6,
        "comparison": f"{left_control}_vs_{right_control}",
        "fields": len(frame),
        "experiments": len(experiments),
        "mean_advantage_px": float(frame["advantage_px"].mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0)),
    }


def stratum_records(
    field: FieldState,
    prediction: np.ndarray,
    objective: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    density_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for horizon in (1, 6):
        windows = field.windows[horizon]
        if not len(windows):
            continue
        target = field.target[windows].sum(axis=1)
        predicted = prediction[windows].sum(axis=1)
        baseline = field.base[windows].sum(axis=1)
        first = windows[:, 0]
        dnn = field.frame_dnn_px[first]
        try:
            bins = pd.qcut(dnn, 4, labels=False, duplicates="drop")
            density_bin = np.asarray(pd.Series(bins).fillna(0), dtype=np.int64)
        except ValueError:
            density_bin = np.zeros(len(windows), dtype=np.int64)
        for bin_value in sorted(np.unique(density_bin)):
            mask = density_bin == bin_value
            real_rmse = float(np.sqrt(np.mean(np.square(predicted[mask] - target[mask]))))
            base_rmse = float(np.sqrt(np.mean(np.square(baseline[mask] - target[mask]))))
            density_rows.append(
                {
                    "annotation_kind": field.annotation_kind,
                    "objective": objective,
                    "experiment": field.experiment,
                    "sequence": field.sequence,
                    "horizon": horizon,
                    "density_quartile": int(bin_value),
                    "windows": int(mask.sum()),
                    "median_dnn_px": float(np.median(dnn[mask])),
                    "median_dnn_um": float(np.median(dnn[mask]) * PIXEL_UM),
                    "component_rmse": real_rmse,
                    "base_component_rmse": base_rmse,
                    "gain_percent": 100.0 * (base_rmse - real_rmse) / max(base_rmse, EPS),
                }
            )
        interpolation = field.rows.iloc[first][
            ["previous_interpolated", "current_interpolated", "target_interpolated"]
        ].any(axis=1).to_numpy(bool)
        for label, mask in (
            ("observed_only", ~interpolation),
            ("any_interpolated", interpolation),
        ):
            if not np.any(mask):
                continue
            real_rmse = float(np.sqrt(np.mean(np.square(predicted[mask] - target[mask]))))
            base_rmse = float(np.sqrt(np.mean(np.square(baseline[mask] - target[mask]))))
            quality_rows.append(
                {
                    "annotation_kind": field.annotation_kind,
                    "objective": objective,
                    "experiment": field.experiment,
                    "sequence": field.sequence,
                    "horizon": horizon,
                    "quality_stratum": label,
                    "windows": int(mask.sum()),
                    "component_rmse": real_rmse,
                    "base_component_rmse": base_rmse,
                    "gain_percent": 100.0 * (base_rmse - real_rmse) / max(base_rmse, EPS),
                }
            )
    return density_rows, quality_rows


def parse_rotations(value: str) -> list[tuple[int, int, int]]:
    requested = {
        tuple(int(item) for item in token.split("-"))
        for token in value.split(",")
        if token.strip()
    }
    rotations = [rotation for rotation in ROTATIONS if rotation in requested]
    if not rotations:
        raise RuntimeError("No valid C2C12 experiment rotation requested")
    return rotations


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [token.strip() for token in args.kinds.split(",") if token.strip()]
    if not set(kinds).issubset(v168.KINDS):
        raise ValueError(f"Unsupported annotation kind: {kinds}")
    rotations = parse_rotations(args.rotations)
    objectives = [token.strip() for token in args.objectives.split(",") if token.strip()]
    if not set(objectives).issubset(OBJECTIVES):
        raise ValueError(f"Unsupported objectives: {objectives}")
    if args.smoke:
        kinds = ["automatic"]
        rotations = [ROTATIONS[0]]
        objectives = ["horizon_balanced"]
        args.max_rows_automatic = min(args.max_rows_automatic, 120_000)
        args.max_iter = min(args.max_iter, 30)
        args.max_windows_per_field_horizon = min(
            args.max_windows_per_field_horizon, 1500
        )
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 500)

    multipliers = parse_floats(args.scale_multipliers)
    alphas = parse_floats(args.alpha_grid)
    bound_factors = parse_floats(args.bound_factors)
    all_sampling: list[pd.DataFrame] = []
    selection_records: list[dict[str, Any]] = []
    selection_grids: list[pd.DataFrame] = []
    field_records: list[dict[str, Any]] = []
    density_records: list[dict[str, Any]] = []
    quality_records: list[dict[str, Any]] = []
    causal_records: list[dict[str, Any]] = []
    dnn_records: list[pd.DataFrame] = []
    coordinate_blend_records: list[pd.DataFrame] = []
    coefficient_records: list[dict[str, Any]] = []

    for kind_index, kind in enumerate(kinds):
        maximum_rows = (
            int(args.max_rows_automatic)
            if kind == "automatic"
            else int(args.max_rows_manual)
        )
        loaded: dict[int, LoadedExperiment] = {}
        for experiment in (1, 2, 3):
            print(f"[v209] loading {kind} experiment={experiment}", flush=True)
            loaded[experiment] = load_experiment_blocked(
                args.table_root,
                kind,
                experiment,
                maximum_rows,
                int(args.sampling_blocks),
                float(args.local_bin_px),
            )
            all_sampling.append(loaded[experiment].sampling)

        for rotation_index, (train_exp, val_exp, test_exp) in enumerate(rotations):
            rotation = f"{train_exp}-{val_exp}-{test_exp}"
            seed = int(args.seed) + kind_index * 100_003 + rotation_index * 10_007
            print(f"[v209] {kind} rotation={rotation}: fit coordinate mean", flush=True)
            train = loaded[train_exp].data
            validation = loaded[val_exp].data
            test = loaded[test_exp].data
            split_keys = [
                set(map(tuple, data.rows[["sequence", "frame", "track_id"]].to_numpy()))
                for data in (train, validation, test)
            ]
            overlap = bool(
                split_keys[0] & split_keys[1]
                or split_keys[0] & split_keys[2]
                or split_keys[1] & split_keys[2]
            )
            if overlap:
                raise RuntimeError("Outer experiment key overlap")
            fit = v168.fit_reliability(train, validation, args, seed)
            split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for split_name, data in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            ):
                prediction, scale, _gain, _metrics = v168.evaluate_control(
                    data, fit, "real", seed + 17
                )
                split_predictions[split_name] = (prediction, scale)
            baseline_by_split = {
                split_name: v168.baseline_predictions(
                    train, validation, data, fit, args
                )
                for split_name, data in (
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                )
            }
            baseline_test = baseline_by_split["test"]
            coordinate_blend_weight = float("nan")
            if args.coordinate_base == "velocity_reliability_blend":
                coordinate_blend_weight, blend_grid = select_velocity_reliability_blend(
                    validation,
                    split_predictions["validation"][0],
                    baseline_by_split["validation"]["constant_velocity"],
                    parse_floats(args.coordinate_blend_weights),
                )
                blend_grid.insert(0, "annotation_kind", kind)
                blend_grid.insert(1, "rotation", rotation)
                coordinate_blend_records.append(blend_grid)
                blended_predictions = {
                    split_name: (
                        coordinate_blend_weight * split_predictions[split_name][0]
                        + (1.0 - coordinate_blend_weight)
                        * baseline_by_split[split_name]["constant_velocity"]
                    )
                    for split_name in ("train", "validation", "test")
                }
                scale_value = constant_student_scale(
                    train.target,
                    blended_predictions["train"],
                    validation.target,
                    blended_predictions["validation"],
                    parse_floats(args.scale_factor_grid),
                )
                split_predictions = {
                    split_name: (
                        blended_predictions[split_name],
                        np.full(len(data.rows), scale_value, dtype=np.float64),
                    )
                    for split_name, data in (
                        ("train", train),
                        ("validation", validation),
                        ("test", test),
                    )
                }
            elif args.coordinate_base != "reliability_mean":
                if args.coordinate_base not in baseline_by_split["train"]:
                    raise ValueError(
                        f"Unsupported coordinate base: {args.coordinate_base}"
                    )
                scale_value = constant_student_scale(
                    train.target,
                    baseline_by_split["train"][args.coordinate_base],
                    validation.target,
                    baseline_by_split["validation"][args.coordinate_base],
                    parse_floats(args.scale_factor_grid),
                )
                split_predictions = {
                    split_name: (
                        baseline_by_split[split_name][args.coordinate_base],
                        np.full(len(data.rows), scale_value, dtype=np.float64),
                    )
                    for split_name, data in (
                        ("train", train),
                        ("validation", validation),
                        ("test", test),
                    )
                }
            dnn_reference, dnn_low, dnn_high, dnn_frame = experiment_dnn_contract(train)
            dnn_frame.insert(0, "annotation_kind", kind)
            dnn_frame.insert(1, "rotation", rotation)
            dnn_records.append(dnn_frame)
            dnn_contract = (dnn_reference, dnn_low, dnn_high)
            print(
                f"[v209] {kind} rotation={rotation}: build causal packets "
                f"dnn={dnn_reference:.3f}px",
                flush=True,
            )
            train_fields = make_fields(
                train,
                kind,
                "train",
                *split_predictions["train"],
                {},
                multipliers,
                dnn_contract,
                bool(args.adaptive_dnn),
                seed,
            )
            validation_fields = make_fields(
                validation,
                kind,
                "validation",
                *split_predictions["validation"],
                {},
                multipliers,
                dnn_contract,
                bool(args.adaptive_dnn),
                seed + 1,
            )
            test_fields = make_fields(
                test,
                kind,
                "test",
                *split_predictions["test"],
                baseline_test,
                multipliers,
                dnn_contract,
                bool(args.adaptive_dnn),
                seed + 2,
            )
            for field in train_fields + validation_fields + test_fields:
                issue = field.rows["frame"].to_numpy(np.int64)
                causal_records.append(
                    {
                        "annotation_kind": kind,
                        "rotation": rotation,
                        "split": field.split,
                        "sequence": field.sequence,
                        "rows": len(field.rows),
                        "real_future_donor_violations": int(
                            np.sum(
                                (field.latest_real_donor >= 0)
                                & (field.latest_real_donor > issue - 1)
                            )
                        ),
                        "stale_donor_violations": int(
                            np.sum(
                                (field.latest_stale_donor >= 0)
                                & (field.latest_stale_donor > issue - 2)
                            )
                        ),
                        "split_key_overlap": overlap,
                        "target_features_used": False,
                    }
                )

            # Same-contract baselines are emitted once per rotation.
            for field in test_fields:
                field_records.extend(
                    prediction_metrics(
                        field,
                        field.base,
                        args.coordinate_base,
                        "baseline",
                        "no_update",
                    )
                )
                for name, prediction in field.baselines.items():
                    field_records.extend(
                        prediction_metrics(field, prediction, name, "baseline", name)
                    )

            for objective in objectives:
                print(
                    f"[v209] {kind} rotation={rotation}: select {objective}",
                    flush=True,
                )
                selection, grid = select_transport(
                    train_fields,
                    validation_fields,
                    objective,
                    alphas,
                    bound_factors,
                    int(args.max_windows_per_field_horizon),
                    args.operator_kind,
                )
                grid.insert(0, "annotation_kind", kind)
                grid.insert(1, "rotation", rotation)
                selection_grids.append(grid)
                selection_records.append(
                    {
                        "annotation_kind": kind,
                        "rotation": rotation,
                        "train_experiment": train_exp,
                        "validation_experiment": val_exp,
                        "test_experiment": test_exp,
                        "objective": objective,
                        "coordinate_base": args.coordinate_base,
                        "coordinate_blend_weight": coordinate_blend_weight,
                        "operator_kind": args.operator_kind,
                        "dnn_reference_px": dnn_reference,
                        "dnn_low_px": dnn_low,
                        "dnn_high_px": dnn_high,
                        "adaptive_dnn": bool(args.adaptive_dnn),
                        **selection,
                    }
                )
                statistics = transport_statistics(
                    train_fields + validation_fields,
                    OBJECTIVES[objective][0],
                    int(args.max_windows_per_field_horizon),
                    args.operator_kind,
                )
                model = solve_transport(
                    statistics, selection["alpha"], args.operator_kind
                )
                if isinstance(model, EquivariantRidgeModel):
                    for name, value in zip(
                        model.design_names, model.coefficients, strict=True
                    ):
                        coefficient_records.append(
                            {
                                "annotation_kind": kind,
                                "rotation": rotation,
                                "objective": objective,
                                "coordinate_base": args.coordinate_base,
                                "operator_kind": args.operator_kind,
                                "feature": name,
                                "output_component": "equivariant_scalar",
                                "coefficient": float(value),
                            }
                        )
                else:
                    feature_names = model.design_names + ["intercept"]
                    for feature_index, name in enumerate(feature_names):
                        for component, component_name in enumerate(("x", "y")):
                            coefficient_records.append(
                                {
                                    "annotation_kind": kind,
                                    "rotation": rotation,
                                    "objective": objective,
                                    "coordinate_base": args.coordinate_base,
                                    "operator_kind": args.operator_kind,
                                    "feature": name,
                                    "output_component": component_name,
                                    "coefficient": float(
                                        model.coefficients[feature_index, component]
                                    ),
                                }
                            )
                bound_px = selection["bound_px"]
                for field in test_fields:
                    predictions: dict[str, np.ndarray] = {
                        "real": field.base
                        + bounded_update(
                            transport_correction(
                                model, field, "real", "full", args.operator_kind
                            ),
                            bound_px,
                        ),
                        "own_only": field.base
                        + bounded_update(
                            transport_correction(
                                model,
                                field,
                                "real",
                                "own_only",
                                args.operator_kind,
                            ),
                            bound_px,
                        ),
                        "neighbour_only": field.base
                        + bounded_update(
                            transport_correction(
                                model,
                                field,
                                "real",
                                "neighbour_only",
                                args.operator_kind,
                            ),
                            bound_px,
                        ),
                        "wrong_cell": field.base
                        + bounded_update(
                            transport_correction(
                                model,
                                field,
                                "wrong_cell",
                                "full",
                                args.operator_kind,
                            ),
                            bound_px,
                        ),
                        "stale_time": field.base
                        + bounded_update(
                            transport_correction(
                                model,
                                field,
                                "stale_time",
                                "full",
                                args.operator_kind,
                            ),
                            bound_px,
                        ),
                    }
                    for control, prediction in predictions.items():
                        field_records.extend(
                            prediction_metrics(
                                field,
                                prediction,
                                f"lit_cell_{objective}",
                                objective,
                                control,
                            )
                        )
                    density, quality = stratum_records(
                        field, predictions["real"], objective
                    )
                    density_records.extend(density)
                    quality_records.extend(quality)
            del train_fields, validation_fields, test_fields, fit
            gc.collect()
        del loaded
        gc.collect()

    sampling_frame = pd.concat(all_sampling, ignore_index=True)
    selection_frame = pd.DataFrame(selection_records)
    selection_grid_frame = pd.concat(selection_grids, ignore_index=True)
    field_frame = pd.DataFrame(field_records)
    aggregate_frame = aggregate_metrics(field_frame)
    density_frame = pd.DataFrame(density_records)
    quality_frame = pd.DataFrame(quality_records)
    causal_frame = pd.DataFrame(causal_records)
    dnn_frame = pd.concat(dnn_records, ignore_index=True)
    coordinate_blend_frame = (
        pd.concat(coordinate_blend_records, ignore_index=True)
        if coordinate_blend_records
        else pd.DataFrame()
    )
    coefficient_frame = pd.DataFrame(coefficient_records)
    for frame in (
        sampling_frame,
        selection_grid_frame,
        field_frame,
        aggregate_frame,
        density_frame,
        quality_frame,
        causal_frame,
        dnn_frame,
    ):
        if "coordinate_base" not in frame.columns:
            frame.insert(0, "coordinate_base", args.coordinate_base)

    bootstrap_records: list[dict[str, Any]] = []
    for kind in kinds:
        for objective in objectives:
            for right in ("no_update", "own_only", "wrong_cell", "stale_time"):
                # no_update rows use objective='baseline'; duplicate them under the
                # objective so the paired comparison has an exact key contract.
                if right == "no_update":
                    baseline = field_frame[
                        (field_frame["annotation_kind"] == kind)
                        & (field_frame["objective"] == "baseline")
                        & (field_frame["control"] == "no_update")
                    ].copy()
                    baseline["objective"] = objective
                    comparison_frame = pd.concat([field_frame, baseline], ignore_index=True)
                else:
                    comparison_frame = field_frame
                bootstrap_records.append(
                    hierarchical_bootstrap(
                        comparison_frame,
                        objective,
                        "real",
                        right,
                        kind,
                        int(args.bootstrap_repetitions),
                        int(args.seed) + len(bootstrap_records) * 1009,
                    )
                )
    bootstrap_frame = pd.DataFrame(bootstrap_records)

    sampling_frame.to_csv(args.out_dir / "v209_data_contract.csv", index=False)
    selection_frame.to_csv(args.out_dir / "v209_selection.csv", index=False)
    selection_grid_frame.to_csv(args.out_dir / "v209_selection_grid.csv", index=False)
    field_frame.to_csv(args.out_dir / "v209_field_metrics.csv", index=False)
    aggregate_frame.to_csv(args.out_dir / "v209_aggregate_metrics.csv", index=False)
    controls = field_frame[
        field_frame["control"].isin(
            ["real", "no_update", "own_only", "neighbour_only", "wrong_cell", "stale_time"]
        )
    ]
    controls.to_csv(args.out_dir / "v209_controls.csv", index=False)
    density_frame.to_csv(args.out_dir / "v209_scale_density_strata.csv", index=False)
    quality_frame.to_csv(args.out_dir / "v209_tracking_quality_strata.csv", index=False)
    causal_frame.to_csv(args.out_dir / "v209_causal_audit.csv", index=False)
    bootstrap_frame.to_csv(args.out_dir / "v209_cluster_bootstrap.csv", index=False)
    dnn_frame.to_csv(args.out_dir / "v209_dnn_frames.csv", index=False)
    coordinate_blend_frame.to_csv(
        args.out_dir / "v209_coordinate_blend_selection.csv", index=False
    )
    coefficient_frame.to_csv(
        args.out_dir / "v209_operator_coefficients.csv", index=False
    )

    primary = field_frame[
        (field_frame["annotation_kind"] == "automatic")
        & (field_frame["objective"] == PRIMARY_OBJECTIVE)
        & (field_frame["horizon"].isin([1, 6]))
        & (field_frame["control"].isin(["real", "own_only", "wrong_cell", "stale_time"]))
    ]
    baseline = field_frame[
        (field_frame["annotation_kind"] == "automatic")
        & (field_frame["objective"] == "baseline")
        & (field_frame["control"] == "no_update")
        & (field_frame["horizon"].isin([1, 6]))
    ].copy()
    baseline["objective"] = PRIMARY_OBJECTIVE
    primary = pd.concat([primary, baseline], ignore_index=True)
    pivot = primary.pivot_table(
        index=["experiment", "sequence", "horizon"],
        columns="control",
        values="component_rmse",
    ).dropna()
    required_controls = {"no_update", "real", "own_only", "wrong_cell", "stale_time"}
    primary_evaluable = bool(
        "automatic" in kinds
        and PRIMARY_OBJECTIVE in objectives
        and len(pivot)
        and required_controls.issubset(set(pivot.columns))
        and {1, 6}.issubset(set(pivot.index.get_level_values("horizon")))
    )
    h6 = pivot.xs(6, level="horizon") if primary_evaluable else pd.DataFrame()
    h1 = pivot.xs(1, level="horizon") if primary_evaluable else pd.DataFrame()
    experiment_gain = (
        100.0
        * (h6["no_update"] - h6["real"])
        / np.maximum(h6["no_update"], EPS)
    ).groupby(level="experiment").mean() if len(h6) else pd.Series(dtype=float)
    h1_degradation = (
        100.0
        * (h1["real"] - h1["no_update"])
        / np.maximum(h1["no_update"], EPS)
    ).mean() if len(h1) else float("inf")
    boot_lookup = {
        row.comparison: row
        for row in bootstrap_frame[
            (bootstrap_frame["annotation_kind"] == "automatic")
            & (bootstrap_frame["objective"] == PRIMARY_OBJECTIVE)
        ].itertuples(index=False)
    }
    all_experiments_positive = bool(len(experiment_gain) == 3 and (experiment_gain > 0).all())
    real_beats_controls = bool(
        all(
            boot_lookup.get(f"real_vs_{control}") is not None
            and boot_lookup[f"real_vs_{control}"].mean_advantage_px > 0
            for control in ("wrong_cell", "stale_time")
        )
    )
    neighbour_increment = bool(
        boot_lookup.get("real_vs_own_only") is not None
        and boot_lookup["real_vs_own_only"].mean_advantage_px > 0
    )
    bootstrap_positive = bool(
        boot_lookup.get("real_vs_no_update") is not None
        and boot_lookup["real_vs_no_update"].ci_low > 0
    )
    passed = bool(
        all_experiments_positive
        and h1_degradation <= 0.5
        and real_beats_controls
        and neighbour_increment
        and bootstrap_positive
        and int(causal_frame["real_future_donor_violations"].sum()) == 0
        and int(causal_frame["stale_donor_violations"].sum()) == 0
    )
    decision = "pass" if passed else ("fail" if primary_evaluable else "not_evaluated")

    aggregate_excerpt = aggregate_frame[
        (aggregate_frame["horizon"].isin([1, 6]))
        & (
            aggregate_frame["control"].isin(
                ["no_update", "real", "own_only", "wrong_cell", "stale_time"]
            )
        )
    ]
    lines = [
        "# C2C12 LIT-Cell external confirmation v209",
        "",
        f"Decision: **{decision.upper()}**",
        "",
        "This is an experiment-external structural validation. C2C12 parameters are fit inside each outer rotation; LaChance pixel-valued weights are not transferred.",
        "",
        "## Primary automatic-track gates",
        "",
        f"- Predeclared primary operating point: `{PRIMARY_OBJECTIVE}`",
        f"- Positive h6 gain in all three held-out experiments: {all_experiments_positive}",
        f"- Mean h1 degradation: {h1_degradation:.4f}% (limit 0.5%)",
        f"- Real update beats wrong-cell and stale controls: {real_beats_controls}",
        f"- Full update beats own-only: {neighbour_increment}",
        f"- Hierarchical bootstrap lower bound above zero: {bootstrap_positive}",
        f"- Causal donor violations: {int(causal_frame['real_future_donor_violations'].sum() + causal_frame['stale_donor_violations'].sum())}",
        "",
        "## Experiment-level h6 gain",
        "",
        experiment_gain.rename("gain_percent").reset_index().to_markdown(index=False),
        "",
        "## Main metrics",
        "",
        aggregate_excerpt.to_markdown(index=False),
        "",
        "## Clustered comparisons",
        "",
        bootstrap_frame.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- Automatic and manual annotations are never pooled.",
        "- Manual centroids are largely interpolated and provide only a secondary observation-process audit.",
        "- A pass supports reuse of completed-innovation transport, not zero-shot transfer of learned weights.",
        "- A failure of full versus own-only means that sequential self-filtering transfers but the neighbour graph contribution does not.",
    ]
    report_path = args.out_dir / "v209_external_confirmation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "c2c12_lit_cell_external_confirmation_v209",
        "status": decision,
        "elapsed_seconds": time.time() - started,
        "table_root": str(args.table_root.resolve()),
        "annotation_kinds": kinds,
        "rotations": [list(rotation) for rotation in rotations],
        "objectives": objectives,
        "scale_multipliers": multipliers,
        "adaptive_dnn": bool(args.adaptive_dnn),
        "coordinate_base": args.coordinate_base,
        "coordinate_blend_weights": parse_floats(args.coordinate_blend_weights),
        "operator_kind": args.operator_kind,
        "future_or_target_features_used": False,
        "pixel_um": PIXEL_UM,
        "frame_minutes": FRAME_MINUTES,
        "source_tables": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for kind in kinds
            for experiment in (1, 2, 3)
            for path in v168.table_paths(args.table_root, kind, experiment)
        },
        "outputs": sorted(path.name for path in args.out_dir.iterdir()),
    }
    write_json(args.out_dir / "v209_manifest.json", manifest)
    print("\n".join(lines), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kinds", default="automatic,manual")
    parser.add_argument("--rotations", default="1-2-3,2-3-1,3-1-2")
    parser.add_argument(
        "--objectives", default="h1_strict,horizon_balanced,h6_utility"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coordinate-base",
        choices=(
            "reliability_mean",
            "constant_velocity",
            "validation_fixed_gain",
            "kalman_cv_like",
            "kalman_ca_like",
            "imm_cv_ca_like",
            "no_update_previous_velocity",
            "velocity_reliability_blend",
        ),
        default="reliability_mean",
    )
    parser.add_argument(
        "--coordinate-blend-weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
    )
    parser.add_argument(
        "--operator-kind", choices=("free", "equivariant"), default="free"
    )
    parser.add_argument("--max-rows-automatic", type=int, default=800_000)
    parser.add_argument("--max-rows-manual", type=int, default=0)
    parser.add_argument("--sampling-blocks", type=int, default=4)
    parser.add_argument("--local-bin-px", type=float, default=64.0)
    parser.add_argument("--scale-multipliers", default="0.5,1,2,4")
    parser.add_argument(
        "--adaptive-dnn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--alpha-grid", default="10,30,100,300,1000,3000,10000")
    parser.add_argument(
        "--bound-factors", default="0.025,0.05,0.075,0.1,0.15,0.25,0.5,1,2,4"
    )
    parser.add_argument("--max-windows-per-field-horizon", type=int, default=12_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)

    # Restricted reliability-aware mean inherited from v168.
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
    parser.add_argument("--corruption-augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
