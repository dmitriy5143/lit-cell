"""Bounded streaming corrections and causal rolling composition."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-8


def bounded_update(correction: np.ndarray, bound_px: float) -> np.ndarray:
    value = np.asarray(correction, dtype=np.float64)
    if bound_px <= 0:
        return value
    length = np.linalg.norm(value, axis=1, keepdims=True)
    bounded_length = np.tanh(length / float(bound_px)) * float(bound_px)
    return value * bounded_length / np.maximum(length, EPS)


def consecutive_windows(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    required = {"track_id", "frame"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(rows.columns))}")
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
            for offset in range(int(horizon))
        ]
        if all(index is not None for index in indices):
            windows.append([int(index) for index in indices])
    if not windows:
        return np.empty((0, int(horizon)), dtype=np.int64)
    return np.asarray(windows, dtype=np.int64)
