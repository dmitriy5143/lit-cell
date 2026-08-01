"""Student-t predictive utilities with explicit physical scales."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def component_nll(
    target: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: float,
) -> float:
    safe_scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)
    residual = (np.asarray(target) - np.asarray(mean)) / safe_scale
    log_density = student_t.logpdf(residual, df=float(degrees_of_freedom))
    return float(np.mean(-log_density + np.log(safe_scale)))


def marginal_coverage(
    target: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: float,
    level: float,
) -> float:
    quantile = float(
        student_t.ppf((1.0 + float(level)) / 2.0, df=degrees_of_freedom)
    )
    radius = quantile * np.maximum(np.asarray(scale), 1e-6)
    return float(np.mean(np.abs(np.asarray(target) - np.asarray(mean)) <= radius))
