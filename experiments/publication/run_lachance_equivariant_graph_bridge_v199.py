#!/usr/bin/env python3
"""Bridge the final LaChance streaming model to an equivariant graph field law.

The runner restores the exact outer-LOMO v97 mixtures used by v157h, reproduces
the dense winning update, and evaluates a deliberately restricted alternative:
one shared scalar coefficient per E(2)-equivariant vector term.  Every graph
term at issue frame t uses only residuals from completed frame t-1.

This is an interpretation and architecture-capacity test, not a replacement
for the frozen publication model unless it passes the same outer-fold gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "lachance_equivariant_graph_bridge_v199"
EPS = 1e-12
SCALE_FACTORS = (1.0, 2.0, 4.0)
VARIANT_TERMS: dict[str, tuple[str, ...]] = {
    "self_memory": ("self",),
    "forced_potential": (
        "self",
        "global",
        "transverse_s1",
        "longitudinal_s1",
        "transverse_s2",
        "longitudinal_s2",
        "transverse_s4",
        "longitudinal_s4",
        "cubic_damping",
    ),
    "active_advective": (
        "self",
        "global",
        "transverse_s1",
        "longitudinal_s1",
        "transverse_s2",
        "longitudinal_s2",
        "transverse_s4",
        "longitudinal_s4",
        "cubic_damping",
        "velocity_advection_s2",
        "innovation_advection_s2",
    ),
}


@dataclass
class GraphPayload:
    movie: int
    split: str
    base: Any
    names: tuple[str, ...]
    real: np.ndarray
    wrong_cell: np.ndarray
    stale_time: np.ndarray
    real_latest_donor_frame: np.ndarray
    stale_latest_donor_frame: np.ndarray
    wrong_permutation: np.ndarray
    frame_neighbour_scale: np.ndarray


@dataclass(frozen=True)
class SharedVectorModel:
    names: tuple[str, ...]
    scales: np.ndarray
    coefficients: np.ndarray
    alpha: float
    constrained: bool

    def predict(self, terms: np.ndarray) -> np.ndarray:
        indices = np.asarray([ALL_TERM_NAMES.index(name) for name in self.names])
        selected = np.asarray(terms, dtype=np.float64)[:, indices]
        return np.einsum(
            "nmc,m->nc",
            selected / self.scales[None, :, None],
            self.coefficients,
            optimize=True,
        )


@dataclass(frozen=True)
class Selection:
    alpha: float
    bound_px: float
    validation_score: float
    validation_h1_gain: float


ALL_TERM_NAMES = tuple(
    dict.fromkeys(name for names in VARIANT_TERMS.values() for name in names)
)


def parse_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument(
        "--variants",
        default="self_memory,forced_potential,active_advective",
    )
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--max-neighbours", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=199_001)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_neighbour_scale(position: np.ndarray) -> float:
    position = np.asarray(position, dtype=np.float64)
    if len(position) < 2:
        return 1.0
    distance, _ = cKDTree(position).query(position, k=2)
    values = distance[:, 1]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else 1.0


def frame_vector_terms(
    current_position: np.ndarray,
    previous_position: np.ndarray,
    previous_innovation: np.ndarray,
    current_velocity: np.ndarray,
    current_tracks: np.ndarray,
    previous_tracks: np.ndarray,
    *,
    max_neighbours: int,
) -> tuple[np.ndarray, float]:
    """Build graph differential terms for one issue frame."""

    current_position = np.asarray(current_position, dtype=np.float64)
    previous_position = np.asarray(previous_position, dtype=np.float64)
    previous_innovation = np.asarray(previous_innovation, dtype=np.float64)
    current_velocity = np.asarray(current_velocity, dtype=np.float64)
    count = len(current_position)
    output = np.zeros((count, len(ALL_TERM_NAMES), 2), dtype=np.float64)
    if not len(previous_position):
        return output, 1.0

    term_index = {name: index for index, name in enumerate(ALL_TERM_NAMES)}
    global_state = previous_innovation.mean(axis=0)
    output[:, term_index["global"]] = global_state
    previous_lookup = {
        int(track): index for index, track in enumerate(previous_tracks)
    }
    own_index = np.asarray(
        [previous_lookup.get(int(track), -1) for track in current_tracks],
        dtype=np.int64,
    )
    own_available = own_index >= 0
    own = np.zeros((count, 2), dtype=np.float64)
    own[own_available] = previous_innovation[own_index[own_available]]
    output[:, term_index["self"]] = own
    output[:, term_index["cubic_damping"]] = (
        -np.sum(np.square(own), axis=1, keepdims=True) * own
    )

    scale = nearest_neighbour_scale(current_position)
    tree = cKDTree(previous_position)
    k = min(max(int(max_neighbours) + 1, 2), len(previous_position))
    distance, neighbour = tree.query(current_position, k=k)
    if k == 1:
        distance = distance[:, None]
        neighbour = neighbour[:, None]
    neighbour = np.asarray(neighbour, dtype=np.int64)
    distance = np.asarray(distance, dtype=np.float64)

    for row in range(count):
        valid = np.isfinite(distance[row])
        selected = neighbour[row, valid]
        dist = distance[row, valid]
        if not len(selected):
            continue
        different = previous_tracks[selected] != current_tracks[row]
        selected = selected[different]
        dist = dist[different]
        if not len(selected):
            continue
        relative = previous_position[selected] - current_position[row]
        direction = relative / np.maximum(dist[:, None], EPS)
        anchor = own[row] if own_available[row] else global_state
        delta = previous_innovation[selected] - anchor
        velocity = current_velocity[row]
        velocity_norm = np.linalg.norm(velocity)
        velocity_direction = velocity / max(velocity_norm, EPS)
        innovation_norm = np.linalg.norm(anchor)
        innovation_direction = anchor / max(innovation_norm, EPS)

        for factor in SCALE_FACTORS:
            label = str(int(factor))
            radius = max(factor * scale, EPS)
            weight = np.exp(-0.5 * np.square(dist / radius))
            weight[dist > 4.0 * radius] = 0.0
            normalizer = float(weight.sum())
            if normalizer <= EPS:
                continue
            normalized = weight / normalizer
            laplacian = np.sum(normalized[:, None] * delta, axis=0)
            projection = np.sum(delta * direction, axis=1)
            longitudinal = np.sum(
                normalized[:, None] * projection[:, None] * direction,
                axis=0,
            )
            output[row, term_index[f"longitudinal_s{label}"]] = longitudinal
            output[row, term_index[f"transverse_s{label}"]] = (
                laplacian - longitudinal
            )
            if factor == 2.0:
                velocity_sign = direction @ velocity_direction
                innovation_sign = direction @ innovation_direction
                output[row, term_index["velocity_advection_s2"]] = np.sum(
                    normalized[:, None] * velocity_sign[:, None] * delta,
                    axis=0,
                )
                output[row, term_index["innovation_advection_s2"]] = np.sum(
                    normalized[:, None] * innovation_sign[:, None] * delta,
                    axis=0,
                )
    return output, scale


def coherent_wrong_terms(
    real: np.ndarray,
    rows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows), dtype=np.int64)
    for raw_indices in rows.groupby("frame", sort=True).indices.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if len(indices) > 1:
            cycle = rng.permutation(indices)
            permutation[cycle] = np.roll(cycle, 1)
    return real[permutation].copy(), permutation


def coherent_stale_terms(
    real: np.ndarray,
    rows: pd.DataFrame,
    donor: np.ndarray,
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
            stale_donor[index] = donor[source]
    return stale, stale_donor


def build_graph_payload(
    split: str,
    base: Any,
    *,
    max_neighbours: int,
    control_seed: int,
) -> GraphPayload:
    rows = base.rows.reset_index(drop=True)
    residual = np.asarray(base.target - base.mean, dtype=np.float64)
    count = len(rows)
    real = np.zeros((count, len(ALL_TERM_NAMES), 2), dtype=np.float64)
    donor = np.full(count, -1, dtype=np.int64)
    scales = np.full(count, np.nan, dtype=np.float64)
    groups = {
        int(frame): np.asarray(indices, dtype=np.int64)
        for frame, indices in rows.groupby("frame", sort=True).indices.items()
    }
    for frame, current in groups.items():
        previous = groups.get(frame - 1, np.empty(0, dtype=np.int64))
        if not len(previous):
            continue
        terms, scale = frame_vector_terms(
            rows.iloc[current][["x_px", "y_px"]].to_numpy(np.float64),
            rows.iloc[previous][["x_px", "y_px"]].to_numpy(np.float64),
            residual[previous],
            rows.iloc[current][["dx_px", "dy_px"]].to_numpy(np.float64),
            rows.iloc[current]["track_id"].to_numpy(np.int64),
            rows.iloc[previous]["track_id"].to_numpy(np.int64),
            max_neighbours=max_neighbours,
        )
        real[current] = terms
        donor[current] = frame - 1
        scales[current] = scale
    wrong, permutation = coherent_wrong_terms(
        real,
        rows,
        control_seed + int(base.movie) * 1009,
    )
    stale, stale_donor = coherent_stale_terms(real, rows, donor)
    current_frame = rows["frame"].to_numpy(np.int64)
    if np.any((donor >= 0) & (donor > current_frame - 1)):
        raise RuntimeError(f"Future donor in graph packet for movie {base.movie}")
    if np.any((stale_donor >= 0) & (stale_donor > current_frame - 2)):
        raise RuntimeError(f"Non-stale donor in graph packet for movie {base.movie}")
    return GraphPayload(
        movie=int(base.movie),
        split=split,
        base=base,
        names=ALL_TERM_NAMES,
        real=real,
        wrong_cell=wrong,
        stale_time=stale,
        real_latest_donor_frame=donor,
        stale_latest_donor_frame=stale_donor,
        wrong_permutation=permutation,
        frame_neighbour_scale=scales,
    )


def selected_terms(payload: GraphPayload, variant: str, control: str) -> np.ndarray:
    packet = np.asarray(getattr(payload, control), dtype=np.float64)
    indices = [ALL_TERM_NAMES.index(name) for name in VARIANT_TERMS[variant]]
    return packet[:, indices]


def training_arrays(
    payloads: Mapping[int, GraphPayload],
    movies: Sequence[int],
    variant: str,
    weights_by_horizon: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    for movie in movies:
        payload = payloads[int(movie)]
        terms = selected_terms(payload, variant, "real")
        residual = payload.base.target - payload.base.mean
        for horizon in v157e.HORIZONS:
            windows = v157e.consecutive_windows(payload.base.rows, horizon)
            if not len(windows):
                continue
            design_blocks.append(terms[windows].sum(axis=1))
            target_blocks.append(residual[windows].sum(axis=1))
            weight_blocks.append(
                np.full(
                    len(windows),
                    float(weights_by_horizon[horizon]) / len(windows),
                    dtype=np.float64,
                )
            )
    design = np.concatenate(design_blocks)
    target = np.concatenate(target_blocks)
    weights = np.concatenate(weight_blocks)
    weights *= len(weights) / max(float(weights.sum()), EPS)
    return design, target, weights


def fit_shared_vector_model(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
    *,
    alpha: float,
    constrained: bool,
) -> SharedVectorModel:
    clean = np.nan_to_num(np.asarray(design, dtype=np.float64))
    response = np.nan_to_num(np.asarray(target, dtype=np.float64))
    scales = np.maximum(
        np.sqrt(np.mean(np.square(clean), axis=(0, 2))),
        EPS,
    )
    matrix = (clean / scales[None, :, None]).transpose(0, 2, 1).reshape(
        -1, len(names)
    )
    vector = response.reshape(-1)
    root_weight = np.repeat(np.sqrt(weights), 2)
    matrix = matrix * root_weight[:, None]
    vector = vector * root_weight
    ridge = math.sqrt(float(alpha)) * np.eye(len(names), dtype=np.float64)
    augmented_matrix = np.vstack([matrix, ridge])
    augmented_vector = np.concatenate([vector, np.zeros(len(names))])
    if constrained:
        lower = np.full(len(names), -np.inf, dtype=np.float64)
        upper = np.full(len(names), np.inf, dtype=np.float64)
        for index, name in enumerate(names):
            if name == "self":
                upper[index] = 0.999 * scales[index]
            elif (
                name.startswith("transverse_")
                or name.startswith("longitudinal_")
                or name == "cubic_damping"
            ):
                lower[index] = 0.0
        result = lsq_linear(
            augmented_matrix,
            augmented_vector,
            bounds=(lower, upper),
            method="trf",
            lsmr_tol="auto",
            max_iter=500,
        )
        if not result.success:
            raise RuntimeError(f"Constrained graph fit failed: {result.message}")
        coefficients = result.x
    else:
        gram = matrix.T @ matrix + float(alpha) * np.eye(len(names))
        rhs = matrix.T @ vector
        coefficients = np.linalg.solve(gram, rhs)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Non-finite graph coefficients")
    return SharedVectorModel(
        names=tuple(names),
        scales=scales,
        coefficients=np.asarray(coefficients),
        alpha=float(alpha),
        constrained=bool(constrained),
    )


def fit_model(
    payloads: Mapping[int, GraphPayload],
    movies: Sequence[int],
    variant: str,
    weights: Mapping[int, float],
    alpha: float,
) -> SharedVectorModel:
    design, target, sample_weight = training_arrays(
        payloads,
        movies,
        variant,
        weights,
    )
    return fit_shared_vector_model(
        design,
        target,
        sample_weight,
        VARIANT_TERMS[variant],
        alpha=alpha,
        constrained=variant != "active_advective",
    )


def bounded_prediction(
    model: SharedVectorModel,
    payload: GraphPayload,
    variant: str,
    control: str,
    bound_px: float,
) -> np.ndarray:
    correction = model.predict(selected_terms(payload, variant, control))
    return payload.base.mean + v157e.bounded_update(correction, bound_px)


def select_model(
    payloads: Mapping[int, GraphPayload],
    train_movies: Sequence[int],
    validation_movie: int,
    variant: str,
    weights: Mapping[int, float],
    h1_guard: float,
    alphas: Sequence[float],
    bounds: Sequence[float],
) -> tuple[Selection, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    validation = payloads[int(validation_movie)]
    for alpha in alphas:
        model = fit_model(payloads, train_movies, variant, weights, alpha)
        for bound in bounds:
            prediction = bounded_prediction(
                model,
                validation,
                variant,
                "real",
                bound,
            )
            metrics = v157e.metric_rows(
                validation,
                prediction,
                "validation_real",
                None,
            )
            record: dict[str, Any] = {
                "alpha": float(alpha),
                "bound_px": float(bound),
                "validation_score": v157h.score_metrics(metrics, dict(weights)),
            }
            for row in metrics:
                horizon = int(row["horizon"])
                record[f"h{horizon}_component_rmse"] = row["component_rmse"]
                record[f"h{horizon}_gain_percent"] = row[
                    "rmse_improvement_percent"
                ]
            records.append(record)
    grid = pd.DataFrame(records)
    eligible = grid[grid["h1_gain_percent"].ge(-float(h1_guard))]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        ["validation_score", "h1_component_rmse", "bound_px", "alpha"]
    ).iloc[0]
    return (
        Selection(
            alpha=float(best["alpha"]),
            bound_px=float(best["bound_px"]),
            validation_score=float(best["validation_score"]),
            validation_h1_gain=float(best["h1_gain_percent"]),
        ),
        grid,
    )


def add_metadata(
    rows: list[dict[str, Any]],
    *,
    objective: str,
    variant: str,
    selection: Selection | v157h.Selection | None,
) -> None:
    for row in rows:
        row.update(
            {
                "objective_name": objective,
                "variant": variant,
                "selected_alpha": (
                    float(selection.alpha) if selection is not None else np.nan
                ),
                "selected_bound_px": (
                    float(selection.bound_px) if selection is not None else 0.0
                ),
                "validation_score": (
                    float(selection.validation_score)
                    if selection is not None
                    else np.nan
                ),
            }
        )


def exact_sign_flip(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = float(np.mean(values))
    outcomes = np.asarray(
        [
            np.mean(values * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    if alternative == "greater":
        return float(np.mean(outcomes >= observed - 1e-15))
    return float(np.mean(np.abs(outcomes) >= abs(observed) - 1e-15))


def bootstrap_movie_delta(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(int(repeats), len(values)))
    ].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def aggregate_metrics(
    metrics: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouping = ["objective_name", "variant", "control", "horizon"]
    for keys, rows in metrics.groupby(grouping, sort=True):
        delta = rows["component_rmse_delta"].to_numpy(np.float64)
        ci_low, ci_high = bootstrap_movie_delta(delta, bootstrap, seed)
        records.append(
            {
                **dict(zip(grouping, keys)),
                "movies": int(rows["test_movie"].nunique()),
                "component_rmse_mean": float(rows["component_rmse"].mean()),
                "component_rmse_std": float(rows["component_rmse"].std(ddof=1)),
                "vector_rmse_mean": float(rows["vector_rmse"].mean()),
                "r2_mean": float(rows["r2"].mean()),
                "gain_percent_mean": float(
                    rows["rmse_improvement_percent"].mean()
                ),
                "movies_improved": int(np.sum(delta > 0)),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "sign_flip_two_sided_p": exact_sign_flip(delta),
                "sign_flip_one_sided_p": exact_sign_flip(delta, "greater"),
            }
        )
    return pd.DataFrame(records)


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["holm_p"] = np.nan
    for (objective, variant, control), rows in output.groupby(
        ["objective_name", "variant", "control"],
        sort=False,
    ):
        indices = rows.index.to_numpy()
        p = output.loc[indices, "sign_flip_one_sided_p"].to_numpy(np.float64)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        for rank, position in enumerate(order):
            value = min(1.0, (len(p) - rank) * p[position])
            running = max(running, value)
            adjusted[position] = running
        output.loc[indices, "holm_p"] = adjusted
    return output


def projection_diagnostics(
    payload: GraphPayload,
    graph_prediction: np.ndarray,
    legacy_prediction: np.ndarray,
) -> dict[str, float]:
    graph = graph_prediction - payload.base.mean
    legacy = legacy_prediction - payload.base.mean
    denominator = np.linalg.norm(graph, axis=1) * np.linalg.norm(legacy, axis=1)
    cosine = np.divide(
        np.sum(graph * legacy, axis=1),
        np.maximum(denominator, EPS),
        out=np.zeros(len(graph)),
        where=denominator > EPS,
    )
    centered = legacy - legacy.mean(axis=0, keepdims=True)
    explained = 1.0 - float(
        np.sum(np.square(graph - legacy))
        / max(np.sum(np.square(centered)), EPS)
    )
    return {
        "correction_cosine_mean": float(np.mean(cosine)),
        "legacy_correction_explained_r2": explained,
        "graph_correction_rms": float(np.sqrt(np.mean(np.square(graph)))),
        "legacy_correction_rms": float(np.sqrt(np.mean(np.square(legacy)))),
    }


def equivariance_audit(max_neighbours: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    current = rng.normal(size=(31, 2)) * 5.0
    previous = rng.normal(size=(37, 2)) * 5.0
    innovation = rng.normal(size=(37, 2))
    velocity = rng.normal(size=(31, 2))
    current_tracks = np.arange(31)
    previous_tracks = np.arange(37)
    reference, _ = frame_vector_terms(
        current,
        previous,
        innovation,
        velocity,
        current_tracks,
        previous_tracks,
        max_neighbours=max_neighbours,
    )
    transforms = {
        "rotation_37deg": np.asarray(
            [
                [math.cos(math.radians(37)), -math.sin(math.radians(37))],
                [math.sin(math.radians(37)), math.cos(math.radians(37))],
            ]
        ),
        "reflection_x": np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
    }
    records = []
    for name, transform in transforms.items():
        transformed, _ = frame_vector_terms(
            current @ transform.T,
            previous @ transform.T,
            innovation @ transform.T,
            velocity @ transform.T,
            current_tracks,
            previous_tracks,
            max_neighbours=max_neighbours,
        )
        expected = reference @ transform.T
        records.append(
            {
                "transform": name,
                "max_abs_error": float(np.max(np.abs(transformed - expected))),
                "rms_error": float(np.sqrt(np.mean(np.square(transformed - expected)))),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = parse_ints(args.movies)
    seeds = parse_ints(args.seeds)
    objectives = parse_strings(args.objectives)
    variants = parse_strings(args.variants)
    unknown_objectives = sorted(set(objectives) - set(v157h.OBJECTIVES))
    unknown_variants = sorted(set(variants) - set(VARIANT_TERMS))
    if unknown_objectives or unknown_variants:
        raise ValueError(
            f"Unknown objectives={unknown_objectives}, variants={unknown_variants}"
        )
    alphas = parse_floats(args.alphas)
    bounds = parse_floats(args.bounds_px)
    if args.smoke:
        movies = movies[:2]
        seeds = seeds[:1]
        objectives = objectives[:1]
        alphas = alphas[:3]
        bounds = bounds[:3]
        args.bootstrap = min(args.bootstrap, 500)
        args.max_neighbours = min(args.max_neighbours, 32)
    device = v157e.device_from_cli(args.device)

    metric_records: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    projection_records: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    causal_records: list[dict[str, Any]] = []
    prediction_archive: dict[str, np.ndarray] = {}

    for test_movie in movies:
        print(f"[v199] restore outer movie={test_movie}", flush=True)
        replays = [
            v157e.restore_fold_seed(args.v102_root, test_movie, seed, device)
            for seed in seeds
        ]
        split_payloads = v157e.student_t_mixture_payloads(replays)
        validation_movies = {
            int(replay.manifest["validation_movie"]) for replay in replays
        }
        train_sets = {
            tuple(int(movie) for movie in replay.manifest["train_movies"])
            for replay in replays
        }
        if len(validation_movies) != 1 or len(train_sets) != 1:
            raise RuntimeError("Outer split mismatch across optimizer seeds")
        validation_movie = next(iter(validation_movies))
        train_movies = list(next(iter(train_sets)))
        graph_payloads = {
            movie: build_graph_payload(
                split,
                base,
                max_neighbours=args.max_neighbours,
                control_seed=args.control_seed + 100_003 * test_movie,
            )
            for movie, (split, base) in split_payloads.items()
        }
        legacy_payloads = {
            movie: v157e.build_update_payload(
                split,
                base,
                [30.0, 60.0, 120.0, 240.0],
                args.control_seed + 200_003 * test_movie,
            )
            for movie, (split, base) in split_payloads.items()
        }
        for movie, payload in graph_payloads.items():
            frame = payload.base.rows["frame"].to_numpy(np.int64)
            causal_records.append(
                {
                    "outer_test_movie": test_movie,
                    "movie": movie,
                    "split": payload.split,
                    "real_future_donor_violations": int(
                        np.sum(
                            (payload.real_latest_donor_frame >= 0)
                            & (payload.real_latest_donor_frame > frame - 1)
                        )
                    ),
                    "stale_future_or_nonstale_violations": int(
                        np.sum(
                            (payload.stale_latest_donor_frame >= 0)
                            & (payload.stale_latest_donor_frame > frame - 2)
                        )
                    ),
                    "wrong_cell_fixed_points": int(
                        np.sum(payload.wrong_permutation == np.arange(len(frame)))
                    ),
                    "median_neighbour_scale": float(
                        np.nanmedian(payload.frame_neighbour_scale)
                    ),
                }
            )

        test_graph = graph_payloads[test_movie]
        test_legacy = legacy_payloads[test_movie]
        prefix = f"movie{test_movie:02d}"
        prediction_archive[f"{prefix}__keys"] = test_graph.base.rows[
            ["sequence", "frame", "track_id"]
        ].to_numpy(np.int64)
        prediction_archive[f"{prefix}__target"] = test_graph.base.target.astype(
            np.float32
        )
        prediction_archive[f"{prefix}__base"] = test_graph.base.mean.astype(
            np.float32
        )
        prediction_archive[f"{prefix}__scale"] = test_graph.base.scale.astype(
            np.float32
        )

        for objective_name in objectives:
            weights, h1_guard = v157h.OBJECTIVES[objective_name]
            legacy_selection, legacy_grid = v157h.select_model(
                legacy_payloads,
                train_movies,
                validation_movie,
                weights,
                h1_guard,
                [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
                [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            )
            legacy_grid.insert(0, "outer_test_movie", test_movie)
            legacy_grid.insert(1, "objective_name", objective_name)
            legacy_grid.insert(2, "variant", "legacy_dense")
            selection_frames.append(legacy_grid)
            legacy_model = v157h.fit_model(
                legacy_payloads,
                train_movies + [validation_movie],
                legacy_selection.alpha,
                weights,
            )
            legacy_predictions: dict[str, np.ndarray] = {}
            for control in ("real", "wrong_cell", "stale_time"):
                prediction = v157h.predict(
                    legacy_model,
                    test_legacy,
                    control,
                    legacy_selection.bound_px,
                )
                legacy_predictions[control] = prediction
                rows = v157e.metric_rows(
                    test_legacy,
                    prediction,
                    control,
                    None,
                )
                add_metadata(
                    rows,
                    objective=objective_name,
                    variant="legacy_dense",
                    selection=legacy_selection,
                )
                metric_records.extend(rows)
            base_rows = v157e.metric_rows(
                test_legacy,
                test_legacy.base.mean,
                "no_update",
                None,
            )
            add_metadata(
                base_rows,
                objective=objective_name,
                variant="no_update",
                selection=None,
            )
            metric_records.extend(base_rows)
            prediction_archive[
                f"{prefix}__{objective_name}__legacy_dense"
            ] = legacy_predictions["real"].astype(np.float32)

            for variant in variants:
                selection, grid = select_model(
                    graph_payloads,
                    train_movies,
                    validation_movie,
                    variant,
                    weights,
                    h1_guard,
                    alphas,
                    bounds,
                )
                grid.insert(0, "outer_test_movie", test_movie)
                grid.insert(1, "objective_name", objective_name)
                grid.insert(2, "variant", variant)
                selection_frames.append(grid)
                model = fit_model(
                    graph_payloads,
                    train_movies + [validation_movie],
                    variant,
                    weights,
                    selection.alpha,
                )
                for name, coefficient, scale in zip(
                    model.names,
                    model.coefficients,
                    model.scales,
                    strict=True,
                ):
                    coefficient_records.append(
                        {
                            "outer_test_movie": test_movie,
                            "validation_movie": validation_movie,
                            "objective_name": objective_name,
                            "variant": variant,
                            "term": name,
                            "coefficient_standardized": coefficient,
                            "term_scale": scale,
                            "coefficient_physical": coefficient / max(scale, EPS),
                            "constrained": model.constrained,
                            "selected_alpha": selection.alpha,
                            "selected_bound_px": selection.bound_px,
                        }
                    )
                graph_predictions: dict[str, np.ndarray] = {}
                for control in ("real", "wrong_cell", "stale_time"):
                    prediction = bounded_prediction(
                        model,
                        test_graph,
                        variant,
                        control,
                        selection.bound_px,
                    )
                    graph_predictions[control] = prediction
                    rows = v157e.metric_rows(
                        test_graph,
                        prediction,
                        control,
                        None,
                    )
                    add_metadata(
                        rows,
                        objective=objective_name,
                        variant=variant,
                        selection=selection,
                    )
                    metric_records.extend(rows)
                projection_records.append(
                    {
                        "outer_test_movie": test_movie,
                        "objective_name": objective_name,
                        "variant": variant,
                        **projection_diagnostics(
                            test_graph,
                            graph_predictions["real"],
                            legacy_predictions["real"],
                        ),
                    }
                )
                prediction_archive[
                    f"{prefix}__{objective_name}__{variant}"
                ] = graph_predictions["real"].astype(np.float32)
                print(
                    f"[v199] test={test_movie} objective={objective_name} "
                    f"variant={variant} alpha={selection.alpha:g} "
                    f"bound={selection.bound_px:g}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_records)
    aggregate = aggregate_metrics(
        metrics,
        bootstrap=args.bootstrap,
        seed=args.control_seed,
    )
    aggregate = holm_adjust(aggregate)
    coefficients = pd.DataFrame(coefficient_records)
    projections = pd.DataFrame(projection_records)
    causal = pd.DataFrame(causal_records)
    selections = pd.concat(selection_frames, ignore_index=True)
    equivariance = equivariance_audit(args.max_neighbours, args.control_seed)
    if causal[
        ["real_future_donor_violations", "stale_future_or_nonstale_violations"]
    ].to_numpy().sum() != 0:
        raise RuntimeError("Causal donor audit failed")
    if float(equivariance["max_abs_error"].max()) > 1e-9:
        raise RuntimeError("Graph term equivariance audit failed")

    metrics.to_csv(args.out_dir / "v199_graph_bridge_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "v199_graph_bridge_aggregate.csv", index=False)
    coefficients.to_csv(
        args.out_dir / "v199_graph_bridge_coefficients.csv",
        index=False,
    )
    projections.to_csv(
        args.out_dir / "v199_graph_bridge_projection.csv",
        index=False,
    )
    causal.to_csv(args.out_dir / "v199_graph_bridge_causal_audit.csv", index=False)
    selections.to_csv(
        args.out_dir / "v199_graph_bridge_validation_grid.csv",
        index=False,
    )
    equivariance.to_csv(
        args.out_dir / "v199_graph_bridge_equivariance.csv",
        index=False,
    )
    np.savez_compressed(
        args.out_dir / "v199_graph_bridge_predictions.npz",
        **prediction_archive,
    )

    primary = aggregate[
        aggregate["control"].eq("real")
        & aggregate["horizon"].isin([1, 6])
    ].copy()
    legacy = primary[primary["variant"].eq("legacy_dense")][
        ["objective_name", "horizon", "component_rmse_mean"]
    ].rename(columns={"component_rmse_mean": "legacy_rmse"})
    baseline = aggregate[
        aggregate["variant"].eq("no_update")
        & aggregate["control"].eq("no_update")
        & aggregate["horizon"].isin([1, 6])
    ][
        ["objective_name", "horizon", "component_rmse_mean"]
    ].rename(columns={"component_rmse_mean": "baseline_rmse"})
    primary = primary.merge(
        legacy,
        on=["objective_name", "horizon"],
        how="left",
    ).merge(
        baseline,
        on=["objective_name", "horizon"],
        how="left",
    )
    legacy_gain = primary["baseline_rmse"] - primary["legacy_rmse"]
    primary["legacy_gain_retained_fraction"] = np.where(
        legacy_gain > EPS,
        (
            primary["baseline_rmse"] - primary["component_rmse_mean"]
        ) / legacy_gain,
        np.nan,
    )
    primary.to_csv(
        args.out_dir / "v199_graph_bridge_primary.csv",
        index=False,
    )
    potential_h6 = primary[
        primary["variant"].eq("forced_potential")
        & primary["objective_name"].eq("h6_guard10")
        & primary["horizon"].eq(6)
    ]
    bridge_pass = bool(
        len(potential_h6)
        and float(potential_h6.iloc[0]["legacy_gain_retained_fraction"]) >= 0.5
        and int(potential_h6.iloc[0]["movies_improved"]) == len(movies)
    )
    report = [
        "# v199 Equivariant Graph Bridge",
        "",
        f"Decision: **{'PASS' if bridge_pass else 'FAIL'}**",
        "",
        "The exact dense v157h update and the shared-coefficient graph laws were",
        "restored inside the same outer-LOMO folds. Graph inputs use only",
        "completed innovations from t-1; no held-out future donor was found.",
        "",
        "## Primary operating points",
        "",
        primary[
            [
                "objective_name",
                "variant",
                "horizon",
                "component_rmse_mean",
                "r2_mean",
                "movies_improved",
                "sign_flip_two_sided_p",
                "holm_p",
                "legacy_gain_retained_fraction",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Projection onto the frozen dense correction",
        "",
        projections.groupby(["objective_name", "variant"], as_index=False)[
            [
                "correction_cosine_mean",
                "legacy_correction_explained_r2",
                "graph_correction_rms",
                "legacy_correction_rms",
            ]
        ].mean().to_markdown(index=False, floatfmt=".6f"),
        "",
        "A PASS supports a kinematic graph-field interpretation of a substantial",
        "part of the final update. It does not identify thermodynamic energy or",
        "make the constrained graph law the publication predictor.",
    ]
    (args.out_dir / "v199_graph_bridge_decision_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__)),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "movies": movies,
        "seeds": seeds,
        "objectives": objectives,
        "variants": variants,
        "max_neighbours": args.max_neighbours,
        "future_feature_count": 0,
        "outer_test_used_for_selection": False,
        "equivariant_shared_scalar_coefficients": True,
        "bridge_pass": bridge_pass,
    }
    (args.out_dir / "v199_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
