"""Equivariant vector-field operators for collective-motion diagnostics.

The continuum operators in this module are E(2)-equivariant.  Their finite
difference implementation is exactly equivariant under the square-grid D4
subgroup and converges to the continuum operators as the grid is refined.

The regression layer deliberately learns one scalar coefficient per vector
term.  It never learns separate x/y coefficients or a vector intercept, which
would silently introduce a preferred laboratory direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


EPS = 1e-12


@dataclass(frozen=True)
class VectorOperatorModel:
    """A shared-coefficient linear combination of equivariant vector terms."""

    term_names: tuple[str, ...]
    coefficients: np.ndarray
    term_scales: np.ndarray
    term_clip_magnitudes: np.ndarray
    alpha: float
    threshold: float

    def predict(self, terms: Mapping[str, np.ndarray]) -> np.ndarray:
        """Predict vectors while preserving a shared x/y parameterization."""

        if not self.term_names:
            first = next(iter(terms.values()))
            return np.zeros_like(np.asarray(first, dtype=np.float64))
        prediction = None
        for name, coefficient, scale, clip_magnitude in zip(
            self.term_names,
            self.coefficients,
            self.term_scales,
            self.term_clip_magnitudes,
            strict=True,
        ):
            value = np.asarray(terms[name], dtype=np.float64)
            value = radial_clip(
                np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
                float(clip_magnitude),
            )
            contribution = float(coefficient) * value / max(float(scale), EPS)
            prediction = (
                contribution.copy()
                if prediction is None
                else prediction + contribution
            )
        assert prediction is not None
        return prediction


def _validate_vector_field(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2); observed {array.shape}")
    return array


def vector_gradient(
    field: np.ndarray,
    spacing: float | tuple[float, float],
) -> np.ndarray:
    """Return ``du_i/dx_j`` with shape ``(..., 2, 2)``.

    Array axes are ordered ``(y, x, vector_component)`` while tensor axes are
    ordered ``(vector_component, derivative_component[x, y])``.
    """

    field = _validate_vector_field(field, "field")
    if field.ndim != 3:
        raise ValueError("vector_gradient expects a two-dimensional grid")
    if np.isscalar(spacing):
        dy = dx = float(spacing)
    else:
        dy, dx = (float(item) for item in spacing)
    gradient = np.empty(field.shape[:2] + (2, 2), dtype=np.float64)
    for component in range(2):
        derivative_y, derivative_x = np.gradient(
            field[..., component],
            dy,
            dx,
            edge_order=1,
        )
        gradient[..., component, 0] = derivative_x
        gradient[..., component, 1] = derivative_y
    return gradient


def scalar_gradient(
    field: np.ndarray,
    spacing: float | tuple[float, float],
) -> np.ndarray:
    """Return a scalar-field gradient in Cartesian ``(x, y)`` order."""

    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("scalar_gradient expects a two-dimensional grid")
    if np.isscalar(spacing):
        dy = dx = float(spacing)
    else:
        dy, dx = (float(item) for item in spacing)
    derivative_y, derivative_x = np.gradient(value, dy, dx, edge_order=1)
    return np.stack([derivative_x, derivative_y], axis=-1)


def vector_laplacian(
    field: np.ndarray,
    spacing: float | tuple[float, float],
) -> np.ndarray:
    """Component-wise vector Laplacian."""

    field = _validate_vector_field(field, "field")
    if field.ndim != 3:
        raise ValueError("vector_laplacian expects a two-dimensional grid")
    if np.isscalar(spacing):
        dy = dx = float(spacing)
    else:
        dy, dx = (float(item) for item in spacing)
    result = np.empty_like(field)
    for component in range(2):
        derivative_y, derivative_x = np.gradient(
            field[..., component],
            dy,
            dx,
            edge_order=1,
        )
        second_yy = np.gradient(derivative_y, dy, axis=0, edge_order=1)
        second_xx = np.gradient(derivative_x, dx, axis=1, edge_order=1)
        result[..., component] = second_xx + second_yy
    return result


def gradient_divergence(
    field: np.ndarray,
    spacing: float | tuple[float, float],
) -> np.ndarray:
    """Return ``grad(div(field))``."""

    gradient = vector_gradient(field, spacing)
    divergence = gradient[..., 0, 0] + gradient[..., 1, 1]
    return scalar_gradient(divergence, spacing)


def directional_derivative(
    advecting: np.ndarray,
    transported: np.ndarray,
    spacing: float | tuple[float, float],
) -> np.ndarray:
    """Return ``(advecting dot grad) transported``."""

    advecting = _validate_vector_field(advecting, "advecting")
    transported = _validate_vector_field(transported, "transported")
    if advecting.shape != transported.shape:
        raise ValueError("advecting and transported fields must have equal shapes")
    gradient = vector_gradient(transported, spacing)
    return np.einsum("...j,...ij->...i", advecting, gradient)


def boundary_normal_term(
    distance: np.ndarray,
    spacing: float | tuple[float, float],
    decay_length: float,
) -> np.ndarray:
    """Inward distance-gradient vector with an exponential boundary envelope."""

    distance = np.asarray(distance, dtype=np.float64)
    gradient = scalar_gradient(distance, spacing)
    magnitude = np.linalg.norm(gradient, axis=-1, keepdims=True)
    normal = np.divide(
        gradient,
        np.maximum(magnitude, EPS),
        out=np.zeros_like(gradient),
        where=magnitude > EPS,
    )
    envelope = np.exp(-np.maximum(distance, 0.0) / max(float(decay_length), EPS))
    return normal * envelope[..., None]


def smooth_vector_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply identical spatial smoothing to both vector components."""

    field = _validate_vector_field(field, "field")
    if sigma <= 0:
        return field.copy()
    return np.stack(
        [
            gaussian_filter(field[..., component], sigma=float(sigma), mode="nearest")
            for component in range(2)
        ],
        axis=-1,
    )


def build_equivariant_library(
    previous_innovation: np.ndarray,
    previous_velocity: np.ndarray,
    boundary_distance: np.ndarray,
    *,
    spacing: float | tuple[float, float],
    smoothing_sigma: float = 0.75,
    boundary_decay: float = 50.0,
    mechanics: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build a compact E(2)-equivariant vector library on one field frame."""

    innovation = smooth_vector_field(previous_innovation, smoothing_sigma)
    velocity = smooth_vector_field(previous_velocity, smoothing_sigma)
    if innovation.shape != velocity.shape:
        raise ValueError("innovation and velocity fields must have equal shapes")
    distance = np.asarray(boundary_distance, dtype=np.float64)
    if distance.shape != innovation.shape[:2]:
        raise ValueError("boundary_distance must match the spatial field shape")

    innovation_energy = np.sum(np.square(innovation), axis=-1, keepdims=True)
    library = {
        "u_prev": innovation,
        "lap_u": vector_laplacian(innovation, spacing),
        "grad_div_u": gradient_divergence(innovation, spacing),
        "advect_v_u": directional_derivative(velocity, innovation, spacing),
        "advect_u_u": directional_derivative(innovation, innovation, spacing),
        "cubic_u": innovation_energy * innovation,
        "boundary_normal": boundary_normal_term(
            distance,
            spacing,
            boundary_decay,
        ),
    }
    if mechanics:
        for name, value in mechanics.items():
            vector = _validate_vector_field(value, name)
            if vector.shape != innovation.shape:
                raise ValueError(f"mechanics term {name} has the wrong shape")
            library[name] = vector.copy()
    return library


def radial_clip(vector: np.ndarray, maximum: float) -> np.ndarray:
    """Clip vector magnitude without introducing a preferred direction."""

    value = _validate_vector_field(vector, "vector").copy()
    if not np.isfinite(maximum) or maximum <= 0:
        return value
    magnitude = np.linalg.norm(value, axis=-1, keepdims=True)
    factor = np.minimum(1.0, float(maximum) / np.maximum(magnitude, EPS))
    return value * factor


def fit_shared_vector_ridge(
    terms: Mapping[str, np.ndarray],
    target: np.ndarray,
    term_names: Sequence[str],
    *,
    alpha: float,
    threshold: float = 0.0,
    clip_quantile: float = 0.999,
) -> VectorOperatorModel:
    """Fit one scalar coefficient per vector term.

    Scaling and clipping use rotationally invariant vector magnitudes.  A
    relative coefficient threshold is then applied to entire vector terms and
    the retained support is refit.
    """

    target = _validate_vector_field(target, "target")
    names = tuple(str(name) for name in term_names)
    if not names:
        return VectorOperatorModel(
            term_names=(),
            coefficients=np.empty(0, dtype=np.float64),
            term_scales=np.empty(0, dtype=np.float64),
            term_clip_magnitudes=np.empty(0, dtype=np.float64),
            alpha=float(alpha),
            threshold=float(threshold),
        )
    arrays = []
    scales = []
    clip_magnitudes = []
    for name in names:
        value = _validate_vector_field(terms[name], name)
        if value.shape != target.shape:
            raise ValueError(f"term {name} does not match target shape")
        clean = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(clean, axis=-1)
        finite_norm = norm[np.isfinite(norm)]
        maximum = (
            float(np.quantile(finite_norm, clip_quantile))
            if len(finite_norm)
            else 0.0
        )
        clean = radial_clip(clean, maximum)
        clip_magnitudes.append(maximum)
        scale = float(np.sqrt(np.mean(np.square(clean))))
        scales.append(max(scale, EPS))
        arrays.append(clean)
    clean_target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    target_norm = np.linalg.norm(clean_target, axis=-1)
    target_maximum = float(np.quantile(target_norm, clip_quantile))
    clean_target = radial_clip(clean_target, target_maximum)

    def solve(active: np.ndarray) -> np.ndarray:
        design = np.column_stack(
            [
                (arrays[index] / scales[index]).reshape(-1)
                for index in np.flatnonzero(active)
            ]
        )
        response = clean_target.reshape(-1)
        # NumPy 2.0 on macOS Accelerate can emit spurious floating-point
        # warnings for otherwise finite BLAS products; explicit finiteness is
        # checked immediately below.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = design.T @ design
            rhs = design.T @ response
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise FloatingPointError("Non-finite shared vector normal equations")
        regularized = gram + float(alpha) * np.eye(gram.shape[0])
        return np.linalg.solve(regularized, rhs)

    active = np.ones(len(names), dtype=bool)
    coefficients = solve(active)
    if threshold > 0 and len(coefficients) > 1:
        cutoff = float(threshold) * max(float(np.max(np.abs(coefficients))), EPS)
        active = np.abs(coefficients) >= cutoff
        if not np.any(active):
            active[int(np.argmax(np.abs(coefficients)))] = True
        coefficients = solve(active)
    selected = np.flatnonzero(active)
    return VectorOperatorModel(
        term_names=tuple(names[index] for index in selected),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        term_scales=np.asarray([scales[index] for index in selected]),
        term_clip_magnitudes=np.asarray(
            [clip_magnitudes[index] for index in selected]
        ),
        alpha=float(alpha),
        threshold=float(threshold),
    )


def vector_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Component RMSE for two-dimensional vectors."""

    prediction = _validate_vector_field(prediction, "prediction")
    target = _validate_vector_field(target, "target")
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def vector_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    """Component-wise pooled coefficient of determination."""

    prediction = _validate_vector_field(prediction, "prediction")
    target = _validate_vector_field(target, "target")
    residual = float(np.sum(np.square(prediction - target)))
    centered = target - np.mean(target, axis=0, keepdims=True)
    total = float(np.sum(np.square(centered)))
    return float(1.0 - residual / max(total, EPS))
