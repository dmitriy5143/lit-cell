"""Utilities for characterizing causal innovation fields.

The functions in this module are deliberately independent from the LaChance
data loaders. They operate on positions and two-dimensional vectors so the
estimators can be checked on synthetic fields before they are used on cell
tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


EPS = 1e-12


@dataclass(frozen=True)
class DecayScale:
    """Non-parametric and exponential decay scales for one correlation curve."""

    first_value: float
    e_folding: float
    first_zero: float
    integral_positive: float
    exponential_xi: float
    exponential_amplitude: float
    exponential_rmse: float
    n_points: int


@dataclass(frozen=True)
class LocalMomentDiagnostics:
    """Graph size and retained kernel mass for a local-moment calculation."""

    candidate_edges: int
    edges_per_scale: tuple[int, ...]
    mean_retained_mass_per_scale: tuple[float, ...]
    min_retained_mass_per_scale: tuple[float, ...]


def sha256_array(value: np.ndarray) -> str:
    """Return a shape- and dtype-aware SHA256 digest."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def nearest_neighbor_scale(position: np.ndarray) -> float:
    """Median nearest-neighbour distance for a frame."""

    position = np.asarray(position, dtype=np.float64)
    if len(position) < 2:
        return float("nan")
    distance, _ = cKDTree(position).query(position, k=2)
    nearest = np.asarray(distance[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > EPS)]
    return float(np.median(nearest)) if len(nearest) else float("nan")


def detrend_vectors(
    position: np.ndarray,
    vector: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Remove frame translation or a full affine vector field.

    ``affine`` fits each vector component to ``1, x-xbar, y-ybar``. The
    operation is diagnostic and target-aware when ``vector`` is an error field.
    """

    position = np.asarray(position, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    if position.shape != vector.shape or position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position and vector must both have shape (N, 2)")
    if mode == "none":
        return vector.copy()
    if not len(vector):
        return vector.copy()
    if mode == "translation":
        return vector - vector.mean(axis=0, keepdims=True)
    if mode != "affine":
        raise ValueError(f"Unknown detrending mode: {mode}")
    centered = position - position.mean(axis=0, keepdims=True)
    design = np.column_stack([np.ones(len(centered)), centered])
    coefficients, _, _, _ = np.linalg.lstsq(design, vector, rcond=None)
    return vector - design @ coefficients


def local_flow_direction(
    position: np.ndarray,
    velocity: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Causal local-flow direction from current/past velocities."""

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.shape != velocity.shape or position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position and velocity must both have shape (N, 2)")
    count = len(position)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    if count == 1:
        local = velocity.copy()
    else:
        actual = min(max(int(k), 1), count - 1)
        _, index = cKDTree(position).query(position, k=actual + 1)
        local = velocity[np.asarray(index[:, 1:], dtype=np.int64)].mean(axis=1)
    magnitude = np.linalg.norm(local, axis=1, keepdims=True)
    own_magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)
    use_own = magnitude[:, 0] <= EPS
    if np.any(use_own):
        local[use_own] = velocity[use_own]
        magnitude[use_own] = own_magnitude[use_own]
    direction = np.zeros_like(local)
    valid = magnitude[:, 0] > EPS
    direction[valid] = local[valid] / magnitude[valid]
    return direction


def same_frame_pairs(position: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected non-self pairs and their displacement vectors."""

    position = np.asarray(position, dtype=np.float64)
    if len(position) < 2 or not np.isfinite(radius) or radius <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 2), dtype=np.float64)
    pair = cKDTree(position).query_pairs(float(radius), output_type="ndarray")
    if not len(pair):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 2), dtype=np.float64)
    left = np.asarray(pair[:, 0], dtype=np.int64)
    right = np.asarray(pair[:, 1], dtype=np.int64)
    displacement = position[right] - position[left]
    return left, right, displacement


def cross_frame_pairs(
    target_position: np.ndarray,
    source_position: np.ndarray,
    target_track: np.ndarray,
    source_track: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed target-to-source pairs, excluding the same track."""

    target_position = np.asarray(target_position, dtype=np.float64)
    source_position = np.asarray(source_position, dtype=np.float64)
    target_track = np.asarray(target_track, dtype=np.int64)
    source_track = np.asarray(source_track, dtype=np.int64)
    if (
        not len(target_position)
        or not len(source_position)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 2), dtype=np.float64)
    neighbourhood = cKDTree(target_position).query_ball_tree(
        cKDTree(source_position),
        float(radius),
    )
    target: list[int] = []
    source: list[int] = []
    for target_index, source_indices in enumerate(neighbourhood):
        for source_index in source_indices:
            if target_track[target_index] != source_track[source_index]:
                target.append(target_index)
                source.append(source_index)
    if not target:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 2), dtype=np.float64)
    target_array = np.asarray(target, dtype=np.int64)
    source_array = np.asarray(source, dtype=np.int64)
    displacement = source_position[source_array] - target_position[target_array]
    return target_array, source_array, displacement


def assign_distance_bins(
    displacement: np.ndarray,
    neighbour_scale: float,
    bin_edges: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized distances, bin indices, and validity mask."""

    displacement = np.asarray(displacement, dtype=np.float64)
    edges = np.asarray(list(bin_edges), dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing one-dimensional array")
    distance = np.linalg.norm(displacement, axis=1)
    normalized = distance / max(float(neighbour_scale), EPS)
    index = np.searchsorted(edges, normalized, side="right") - 1
    valid = (
        np.isfinite(normalized)
        & (distance > EPS)
        & (index >= 0)
        & (index < len(edges) - 1)
    )
    return normalized, index.astype(np.int64), valid


def _safe_direction(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.linalg.norm(vector, axis=1)
    valid = magnitude > EPS
    direction = np.zeros_like(vector)
    direction[valid] = vector[valid] / magnitude[valid, None]
    return direction, valid


def pair_statistics(
    target_vector: np.ndarray,
    source_vector: np.ndarray,
    displacement: np.ndarray,
    target_flow_direction: np.ndarray | None = None,
    target_energy_reference: float | None = None,
    source_energy_reference: float | None = None,
) -> dict[str, float]:
    """Correlation, structure, tensor, and flow-harmonic statistics."""

    target_vector = np.asarray(target_vector, dtype=np.float64)
    source_vector = np.asarray(source_vector, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    if (
        target_vector.shape != source_vector.shape
        or target_vector.shape != displacement.shape
        or target_vector.ndim != 2
        or target_vector.shape[1] != 2
    ):
        raise ValueError("pair arrays must all have shape (P, 2)")
    count = len(target_vector)
    empty = {
        "n_pairs": 0,
        "target_energy": np.nan,
        "source_energy": np.nan,
        "vector_correlation": np.nan,
        "direction_correlation": np.nan,
        "structure_function": np.nan,
        "longitudinal_correlation": np.nan,
        "transverse_correlation": np.nan,
        "harmonic_c0": np.nan,
        "harmonic_c1": np.nan,
        "harmonic_c2": np.nan,
        "harmonic_condition": np.nan,
    }
    if count == 0:
        return empty

    target_pair_energy = float(np.mean(np.sum(np.square(target_vector), axis=1)))
    source_pair_energy = float(np.mean(np.sum(np.square(source_vector), axis=1)))
    target_energy = (
        target_pair_energy
        if target_energy_reference is None
        else float(target_energy_reference)
    )
    source_energy = (
        source_pair_energy
        if source_energy_reference is None
        else float(source_energy_reference)
    )
    normalizer = max(np.sqrt(target_energy * source_energy), EPS)
    dot = np.einsum("ij,ij->i", target_vector, source_vector)
    target_direction, target_valid = _safe_direction(target_vector)
    source_direction, source_valid = _safe_direction(source_vector)
    direction_valid = target_valid & source_valid
    direction_corr = (
        float(np.mean(np.einsum(
            "ij,ij->i",
            target_direction[direction_valid],
            source_direction[direction_valid],
        )))
        if np.any(direction_valid)
        else np.nan
    )

    pair_direction, pair_valid = _safe_direction(displacement)
    pair_perpendicular = np.column_stack(
        [-pair_direction[:, 1], pair_direction[:, 0]]
    )
    longitudinal = (
        np.einsum("ij,ij->i", target_vector, pair_direction)
        * np.einsum("ij,ij->i", source_vector, pair_direction)
    )
    transverse = (
        np.einsum("ij,ij->i", target_vector, pair_perpendicular)
        * np.einsum("ij,ij->i", source_vector, pair_perpendicular)
    )
    valid_tensor = pair_valid

    harmonic = {
        "harmonic_c0": np.nan,
        "harmonic_c1": np.nan,
        "harmonic_c2": np.nan,
        "harmonic_condition": np.nan,
    }
    if target_flow_direction is not None:
        flow = np.asarray(target_flow_direction, dtype=np.float64)
        if flow.shape != displacement.shape:
            raise ValueError("target_flow_direction must have shape (P, 2)")
        flow_norm = np.linalg.norm(flow, axis=1)
        valid = pair_valid & (flow_norm > EPS)
        if np.count_nonzero(valid) >= 6:
            flow_unit = flow[valid] / flow_norm[valid, None]
            cosine = np.einsum(
                "ij,ij->i",
                pair_direction[valid],
                flow_unit,
            )
            design = np.column_stack(
                [
                    np.ones(np.count_nonzero(valid)),
                    cosine,
                    2.0 * np.square(cosine) - 1.0,
                ]
            )
            response = dot[valid] / normalizer
            gram = design.T @ design
            coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
            harmonic = {
                "harmonic_c0": float(coefficients[0]),
                "harmonic_c1": float(coefficients[1]),
                "harmonic_c2": float(coefficients[2]),
                "harmonic_condition": float(np.linalg.cond(gram)),
            }

    result = {
        "n_pairs": int(count),
        "target_energy": target_energy,
        "source_energy": source_energy,
        "vector_correlation": float(np.mean(dot) / normalizer),
        "direction_correlation": direction_corr,
        "structure_function": float(
            np.mean(np.sum(np.square(target_vector - source_vector), axis=1))
            / max(target_energy + source_energy, EPS)
        ),
        "longitudinal_correlation": (
            float(np.mean(longitudinal[valid_tensor]) / normalizer)
            if np.any(valid_tensor)
            else np.nan
        ),
        "transverse_correlation": (
            float(np.mean(transverse[valid_tensor]) / normalizer)
            if np.any(valid_tensor)
            else np.nan
        ),
        **harmonic,
    }
    return result


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    for index in range(1, len(x)):
        left = y[index - 1] - level
        right = y[index] - level
        if left == 0:
            return float(x[index - 1])
        if left * right <= 0 and y[index] != y[index - 1]:
            fraction = (level - y[index - 1]) / (y[index] - y[index - 1])
            return float(x[index - 1] + fraction * (x[index] - x[index - 1]))
    return float("nan")


def estimate_decay_scale(distance: np.ndarray, correlation: np.ndarray) -> DecayScale:
    """Estimate robust and exponential scales from a binned curve."""

    distance = np.asarray(distance, dtype=np.float64)
    correlation = np.asarray(correlation, dtype=np.float64)
    valid = np.isfinite(distance) & np.isfinite(correlation)
    distance = distance[valid]
    correlation = correlation[valid]
    order = np.argsort(distance)
    distance = distance[order]
    correlation = correlation[order]
    if not len(distance):
        return DecayScale(*(float("nan"),) * 7, n_points=0)
    first = float(correlation[0])
    e_folding = (
        _crossing(distance, correlation, first / np.e)
        if first > 0
        else float("nan")
    )
    first_zero = _crossing(distance, correlation, 0.0) if first > 0 else float("nan")
    positive = np.maximum(correlation, 0.0)
    integral = (
        float(np.trapezoid(positive / first, distance))
        if first > EPS and len(distance) >= 2
        else float("nan")
    )

    exponential_xi = float("nan")
    exponential_amplitude = float("nan")
    exponential_rmse = float("nan")
    fit_valid = (correlation > 0) & (distance >= 0)
    if np.count_nonzero(fit_valid) >= 3:
        fit_x = distance[fit_valid]
        fit_y = correlation[fit_valid]

        def exponential(value: np.ndarray, amplitude: float, scale: float) -> np.ndarray:
            return amplitude * np.exp(-value / scale)

        try:
            parameters, _ = curve_fit(
                exponential,
                fit_x,
                fit_y,
                p0=(max(first, EPS), max(float(np.median(fit_x)), EPS)),
                bounds=((0.0, EPS), (np.inf, np.inf)),
                maxfev=10000,
            )
            fitted = exponential(fit_x, *parameters)
            exponential_amplitude = float(parameters[0])
            exponential_xi = float(parameters[1])
            exponential_rmse = float(np.sqrt(np.mean(np.square(fit_y - fitted))))
        except (RuntimeError, ValueError, FloatingPointError):
            pass
    return DecayScale(
        first_value=first,
        e_folding=e_folding,
        first_zero=first_zero,
        integral_positive=integral,
        exponential_xi=exponential_xi,
        exponential_amplitude=exponential_amplitude,
        exponential_rmse=exponential_rmse,
        n_points=int(len(distance)),
    )


def truncated_weighted_mean(
    distance: np.ndarray,
    value: np.ndarray,
    scale: float,
    radius: float | None = None,
) -> tuple[np.ndarray, float]:
    """Gaussian weighted mean and removed mass fraction.

    This small reference implementation is used in tests and approximation
    audits, not in the production sparse graph kernel.
    """

    distance = np.asarray(distance, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(distance / max(float(scale), EPS)))
    full_sum = float(weights.sum())
    if radius is not None:
        weights = weights * (distance <= float(radius))
    kept_sum = float(weights.sum())
    if kept_sum <= EPS:
        return np.zeros(value.shape[1], dtype=np.float64), 1.0
    mean = weights @ value / kept_sum
    removed_fraction = 1.0 - kept_sum / max(full_sum, EPS)
    return np.asarray(mean, dtype=np.float64), float(removed_fraction)


def _moment_output(
    current_count: int,
    edge_target: np.ndarray,
    edge_distance: np.ndarray,
    edge_value: np.ndarray,
    scales: list[float],
    support_radii: list[float],
    edge_direction_cosine: np.ndarray | None,
    direction_strength: float,
    dense_weight_sums: np.ndarray | None,
    kernel: str,
) -> tuple[dict[str, np.ndarray], LocalMomentDiagnostics]:
    output: dict[str, np.ndarray] = {}
    edges_per_scale: list[int] = []
    retained_mean: list[float] = []
    retained_min: list[float] = []
    for scale, support in zip(scales, support_radii):
        selected = edge_distance <= float(support)
        target = edge_target[selected]
        distance = edge_distance[selected]
        value = edge_value[selected]
        if kernel == "gaussian":
            weights = np.exp(
                -0.5 * np.square(distance / max(float(scale), EPS))
            )
        elif kernel == "wendland_c2":
            ratio = distance / max(float(support), EPS)
            weights = (
                np.power(np.maximum(1.0 - ratio, 0.0), 4)
                * (4.0 * ratio + 1.0)
            )
        else:
            raise ValueError(f"Unknown local-moment kernel: {kernel}")
        if edge_direction_cosine is not None and direction_strength != 0:
            directional = np.exp(
                np.clip(
                    float(direction_strength)
                    * edge_direction_cosine[selected],
                    -20.0,
                    20.0,
                )
            )
            weights *= directional
        weight_sum = np.bincount(
            target,
            weights=weights,
            minlength=current_count,
        )
        weight_square_sum = np.bincount(
            target,
            weights=np.square(weights),
            minlength=current_count,
        )
        weighted_x = np.bincount(
            target,
            weights=weights * value[:, 0],
            minlength=current_count,
        )
        weighted_y = np.bincount(
            target,
            weights=weights * value[:, 1],
            minlength=current_count,
        )
        weighted_x2 = np.bincount(
            target,
            weights=weights * np.square(value[:, 0]),
            minlength=current_count,
        )
        weighted_y2 = np.bincount(
            target,
            weights=weights * np.square(value[:, 1]),
            minlength=current_count,
        )
        denominator = np.maximum(weight_sum, EPS)
        mean_x = weighted_x / denominator
        mean_y = weighted_y / denominator
        variance_x = np.maximum(weighted_x2 / denominator - np.square(mean_x), 0.0)
        variance_y = np.maximum(weighted_y2 / denominator - np.square(mean_y), 0.0)
        effective_count = np.square(weight_sum) / np.maximum(
            weight_square_sum,
            EPS,
        )
        unavailable = weight_sum <= EPS
        mean_x[unavailable] = 0.0
        mean_y[unavailable] = 0.0
        variance_x[unavailable] = 0.0
        variance_y[unavailable] = 0.0
        effective_count[unavailable] = 0.0
        label = str(int(scale))
        output[f"local_{label}_x"] = mean_x
        output[f"local_{label}_y"] = mean_y
        output[f"local_{label}_std_x"] = np.sqrt(variance_x)
        output[f"local_{label}_std_y"] = np.sqrt(variance_y)
        output[f"local_{label}_effective_n"] = effective_count
        edges_per_scale.append(int(np.count_nonzero(selected)))
        if dense_weight_sums is None:
            retained = np.ones(current_count, dtype=np.float64)
        else:
            scale_index = len(edges_per_scale) - 1
            retained = weight_sum / np.maximum(
                dense_weight_sums[:, scale_index],
                EPS,
            )
        retained_mean.append(float(np.mean(retained)))
        retained_min.append(float(np.min(retained)))
    return (
        output,
        LocalMomentDiagnostics(
            candidate_edges=int(len(edge_target)),
            edges_per_scale=tuple(edges_per_scale),
            mean_retained_mass_per_scale=tuple(retained_mean),
            min_retained_mass_per_scale=tuple(retained_min),
        ),
    )


def dense_gaussian_local_moments(
    current_position: np.ndarray,
    source_position: np.ndarray,
    source_value: np.ndarray,
    current_track: np.ndarray,
    source_track: np.ndarray,
    scales: Iterable[float],
    current_flow_direction: np.ndarray | None = None,
    direction_strength: float = 0.0,
) -> tuple[dict[str, np.ndarray], LocalMomentDiagnostics]:
    """Dense all-pairs Gaussian moments, matching the historical operator."""

    current_position = np.asarray(current_position, dtype=np.float64)
    source_position = np.asarray(source_position, dtype=np.float64)
    source_value = np.asarray(source_value, dtype=np.float64)
    current_track = np.asarray(current_track, dtype=np.int64)
    source_track = np.asarray(source_track, dtype=np.int64)
    scale_list = [float(scale) for scale in scales]
    target_index = np.repeat(
        np.arange(len(current_position), dtype=np.int64),
        len(source_position),
    )
    source_index = np.tile(
        np.arange(len(source_position), dtype=np.int64),
        len(current_position),
    )
    valid = current_track[target_index] != source_track[source_index]
    target_index = target_index[valid]
    source_index = source_index[valid]
    displacement = source_position[source_index] - current_position[target_index]
    distance = np.linalg.norm(displacement, axis=1)
    direction_cosine = None
    if current_flow_direction is not None:
        flow = np.asarray(current_flow_direction, dtype=np.float64)[target_index]
        flow_norm = np.linalg.norm(flow, axis=1)
        direction_cosine = np.zeros(len(distance), dtype=np.float64)
        pair_valid = (distance > EPS) & (flow_norm > EPS)
        direction_cosine[pair_valid] = np.einsum(
            "ij,ij->i",
            displacement[pair_valid] / distance[pair_valid, None],
            flow[pair_valid] / flow_norm[pair_valid, None],
        )
    support = [float("inf")] * len(scale_list)
    return _moment_output(
        len(current_position),
        target_index,
        distance,
        source_value[source_index],
        scale_list,
        support,
        direction_cosine,
        direction_strength,
        dense_weight_sums=None,
        kernel="gaussian",
    )


def _radius_edges(
    current_position: np.ndarray,
    source_position: np.ndarray,
    current_track: np.ndarray,
    source_track: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = cKDTree(current_position).sparse_distance_matrix(
        cKDTree(source_position),
        float(radius),
        output_type="coo_matrix",
    )
    target = np.asarray(graph.row, dtype=np.int64)
    source = np.asarray(graph.col, dtype=np.int64)
    distance = np.asarray(graph.data, dtype=np.float64)
    valid = current_track[target] != source_track[source]
    return (
        target[valid],
        source[valid],
        distance[valid],
    )


def _knn_edges(
    current_position: np.ndarray,
    source_position: np.ndarray,
    current_track: np.ndarray,
    source_track: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(current_position) or not len(source_position) or k <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    query_k = min(int(k) + 1, len(source_position))
    distance, index = cKDTree(source_position).query(
        current_position,
        k=query_k,
    )
    if query_k == 1:
        distance = np.asarray(distance)[:, None]
        index = np.asarray(index)[:, None]
    target: list[int] = []
    source: list[int] = []
    selected_distance: list[float] = []
    for target_index in range(len(current_position)):
        kept = 0
        for candidate_distance, source_index in zip(
            np.asarray(distance[target_index]).reshape(-1),
            np.asarray(index[target_index]).reshape(-1),
        ):
            if not np.isfinite(candidate_distance):
                continue
            if current_track[target_index] == source_track[int(source_index)]:
                continue
            target.append(target_index)
            source.append(int(source_index))
            selected_distance.append(float(candidate_distance))
            kept += 1
            if kept >= int(k):
                break
    return (
        np.asarray(target, dtype=np.int64),
        np.asarray(source, dtype=np.int64),
        np.asarray(selected_distance, dtype=np.float64),
    )


def sparse_gaussian_local_moments(
    current_position: np.ndarray,
    source_position: np.ndarray,
    source_value: np.ndarray,
    current_track: np.ndarray,
    source_track: np.ndarray,
    scales: Iterable[float],
    support_radii: Iterable[float] | None = None,
    k: int | None = None,
    current_flow_direction: np.ndarray | None = None,
    direction_strength: float = 0.0,
    dense_weight_sums: np.ndarray | None = None,
    kernel: str = "gaussian",
) -> tuple[dict[str, np.ndarray], LocalMomentDiagnostics]:
    """Sparse Gaussian moments over a radius graph or a kNN graph."""

    current_position = np.asarray(current_position, dtype=np.float64)
    source_position = np.asarray(source_position, dtype=np.float64)
    source_value = np.asarray(source_value, dtype=np.float64)
    current_track = np.asarray(current_track, dtype=np.int64)
    source_track = np.asarray(source_track, dtype=np.int64)
    scale_list = [float(scale) for scale in scales]
    if support_radii is None:
        support = [float("inf")] * len(scale_list)
    else:
        support = [float(radius) for radius in support_radii]
    if len(support) != len(scale_list):
        raise ValueError("support_radii and scales must have equal length")
    if k is None:
        finite_support = [radius for radius in support if np.isfinite(radius)]
        if not finite_support:
            raise ValueError("A finite support radius or k is required")
        target_index, source_index, distance = _radius_edges(
            current_position,
            source_position,
            current_track,
            source_track,
            max(finite_support),
        )
    else:
        target_index, source_index, distance = _knn_edges(
            current_position,
            source_position,
            current_track,
            source_track,
            int(k),
        )
    displacement = source_position[source_index] - current_position[target_index]
    direction_cosine = None
    if current_flow_direction is not None:
        flow = np.asarray(current_flow_direction, dtype=np.float64)[target_index]
        flow_norm = np.linalg.norm(flow, axis=1)
        direction_cosine = np.zeros(len(distance), dtype=np.float64)
        valid = (distance > EPS) & (flow_norm > EPS)
        direction_cosine[valid] = np.einsum(
            "ij,ij->i",
            displacement[valid] / distance[valid, None],
            flow[valid] / flow_norm[valid, None],
        )
    return _moment_output(
        len(current_position),
        target_index,
        distance,
        source_value[source_index],
        scale_list,
        support,
        direction_cosine,
        direction_strength,
        dense_weight_sums,
        kernel,
    )
