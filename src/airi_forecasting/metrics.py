"""Protocol-neutral point and proper-score metrics."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def component_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(target))))
    )


def vector_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=-1))))


def vector_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target_array = np.asarray(target)
    error = np.asarray(prediction) - target_array
    centered = target_array - target_array.mean(axis=0, keepdims=True)
    return 1.0 - float(
        np.sum(np.square(error)) / max(np.sum(np.square(centered)), EPS)
    )
