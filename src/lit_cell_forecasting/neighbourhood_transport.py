"""Permutation-invariant local transport after the individual neural prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-8


def local_previous_state(
    current_position: np.ndarray,
    previous_position: np.ndarray,
    previous_score: np.ndarray,
    current_tracks: np.ndarray,
    previous_tracks: np.ndarray,
    scales: list[float],
) -> dict[str, np.ndarray]:
    distance = np.linalg.norm(
        current_position[:, None, :] - previous_position[None, :, :],
        axis=2,
    )
    same_track = current_tracks[:, None] == previous_tracks[None, :]
    output: dict[str, np.ndarray] = {}
    for scale in scales:
        weights = np.exp(-0.5 * np.square(distance / max(scale, 1e-3)))
        weights[same_track] = 0.0
        weight_sum = weights.sum(axis=1, keepdims=True)
        normalized = weights / np.maximum(weight_sum, EPS)
        mean = normalized @ previous_score
        centered = previous_score[None, :, :] - mean[:, None, :]
        variance = np.einsum(
            "ij,ijk->ik",
            normalized,
            np.square(centered),
            optimize=True,
        )
        effective_count = np.square(weight_sum[:, 0]) / np.maximum(
            np.sum(np.square(weights), axis=1),
            EPS,
        )
        label = str(int(scale))
        output[f"local_{label}_x"] = mean[:, 0]
        output[f"local_{label}_y"] = mean[:, 1]
        output[f"local_{label}_std_x"] = np.sqrt(
            np.maximum(variance[:, 0], 0.0)
        )
        output[f"local_{label}_std_y"] = np.sqrt(
            np.maximum(variance[:, 1], 0.0)
        )
        output[f"local_{label}_effective_n"] = effective_count
    return output


def coherent_wrong_cell(
    real: np.ndarray,
    rows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows), dtype=np.int64)
    for _, raw_indices in rows.groupby("frame", sort=True).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if len(indices) > 1:
            cycle = rng.permutation(indices)
            permutation[cycle] = np.roll(cycle, 1)
            if np.any(permutation[indices] == indices):
                raise RuntimeError("Wrong-cell derangement retained a donor")
    return np.asarray(real)[permutation].copy(), permutation


def coherent_stale_time(
    real: np.ndarray,
    rows: pd.DataFrame,
    real_latest_donor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (int(track), int(frame)): index
        for index, (track, frame) in enumerate(
            rows[["track_id", "frame"]].itertuples(index=False)
        )
    }
    stale = np.zeros_like(real)
    stale_donor = np.full(len(rows), -1, dtype=np.int64)
    for index, (track, frame) in enumerate(
        rows[["track_id", "frame"]].itertuples(index=False)
    ):
        source = lookup.get((int(track), int(frame) - 1), -1)
        if source >= 0:
            stale[index] = real[source]
            stale_donor[index] = real_latest_donor[source]
    return stale, stale_donor


@dataclass
class WeightedRidge:
    row_mean: np.ndarray
    row_scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        alpha: float,
    ) -> "WeightedRidge":
        x = np.nan_to_num(np.asarray(features, dtype=np.float64))
        y = np.nan_to_num(np.asarray(target, dtype=np.float64))
        row_mean = x.mean(axis=0)
        row_scale = np.maximum(x.std(axis=0), EPS)
        normalized = (x - row_mean) / row_scale
        augmented = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        weight = np.asarray(weights, dtype=np.float64)
        root = np.sqrt(weight / max(float(np.mean(weight)), EPS))[:, None]
        weighted_x = augmented * root
        gram = weighted_x.T @ weighted_x
        rhs = weighted_x.T @ (y * root)
        penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(gram + penalty, rhs)
        return cls(row_mean, row_scale, coefficients)

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = np.nan_to_num(np.asarray(features, dtype=np.float64))
        normalized = (value - self.row_mean) / self.row_scale
        augmented = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        return augmented @ self.coefficients
