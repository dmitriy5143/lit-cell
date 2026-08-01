#!/usr/bin/env python3
"""Joint graph copula gate over frozen v97 one-step marginals.

The conditional mean and univariate Student-t marginals are never refitted.
This runner asks whether the residual dependence found by v139 can improve a
proper joint forecast. It fits a positive-semidefinite kernel

    R = (1-g-l) I + g 11^T + l exp(-distance / length)

on outer-LOMO v97 normal-score residuals. Pairwise copula likelihood is the
primary scalable gate. Equal-budget coherent particles on deterministic
disjoint spatial blocks provide energy/variogram confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import norm, t as student_t

import run_lachance_online_spatial_innovation_audit_v139 as v139


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lachance_joint_graph_copula_v154"
VALIDATION_MAP = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}
EPS = 1e-8


@dataclass
class MoviePayload:
    movie: int
    rows: pd.DataFrame
    target: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    degrees_of_freedom: float
    normal_score: np.ndarray


@dataclass
class PairPayload:
    distance: np.ndarray
    left: np.ndarray
    right: np.ndarray
    pair_kind: np.ndarray


@dataclass
class KernelParameters:
    family: str
    global_weight: float
    local_weight: float
    length_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v139.DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--families", default="independent,global,local,global_local")
    parser.add_argument("--neighbor-k", type=int, default=8)
    parser.add_argument("--random-pairs-per-node", type=int, default=2)
    parser.add_argument("--max-pairs-per-movie", type=int, default=120_000)
    parser.add_argument("--particle-samples", type=int, default=32)
    parser.add_argument("--max-block-size", type=int, default=64)
    parser.add_argument("--max-blocks-per-frame", type=int, default=6)
    parser.add_argument("--seed", type=int, default=154)
    return parser.parse_args()


def parse_strings(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def fold_metadata(root: Path, movie: int, seed: int) -> tuple[Path, float]:
    fold = v139.fold_dir(root, movie, seed)
    metadata = pd.read_csv(fold / "v97" / "v97_model_metadata.csv")
    row = metadata[metadata["variant"].eq("v97_direct")]
    if row.empty:
        raise RuntimeError(f"Missing v97_direct metadata in {fold}")
    return fold, float(row.iloc[0]["degrees_of_freedom"])


def load_movie(root: Path, movie: int, seeds: list[int]) -> MoviePayload:
    base_rows: pd.DataFrame | None = None
    base_keys: np.ndarray | None = None
    base_target: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    dfs: list[float] = []
    for seed in seeds:
        rows, keys, target, _, _ = v139.load_seed(root, movie, seed)
        fold, degrees = fold_metadata(root, movie, seed)
        with np.load(fold / "v97" / "v97_predictions.npz", allow_pickle=False) as archive:
            prediction = np.asarray(
                archive["v97_direct__prediction"],
                dtype=np.float64,
            )
            scale = np.asarray(archive["v97_direct__scale"], dtype=np.float64)
        if base_rows is None:
            base_rows = rows
            base_keys = keys
            base_target = target
        elif not np.array_equal(keys, base_keys) or not np.array_equal(target, base_target):
            raise RuntimeError(f"Seed row/target mismatch for movie {movie}")
        predictions.append(prediction)
        scales.append(scale)
        dfs.append(degrees)
    assert base_rows is not None and base_target is not None
    stacked_prediction = np.stack(predictions)
    stacked_scale = np.stack(scales)
    mean = stacked_prediction.mean(axis=0)
    df = float(np.median(dfs))
    component_variance = np.stack(
        [
            np.square(scale) * degrees / max(degrees - 2.0, 0.1)
            for scale, degrees in zip(scales, dfs)
        ]
    )
    mixture_variance = (
        np.mean(component_variance + np.square(stacked_prediction), axis=0)
        - np.square(mean)
    )
    effective_scale = np.sqrt(
        np.maximum(mixture_variance, 1e-8) * max(df - 2.0, 0.1) / df
    )
    standardized = (base_target - mean) / effective_scale
    uniform = np.clip(student_t.cdf(standardized, df=df), 1e-6, 1.0 - 1e-6)
    normal_score = norm.ppf(uniform)
    return MoviePayload(
        movie=movie,
        rows=base_rows,
        target=base_target,
        mean=mean,
        scale=effective_scale,
        degrees_of_freedom=df,
        normal_score=normal_score,
    )


def build_pairs(
    payload: MoviePayload,
    *,
    neighbor_k: int,
    random_pairs_per_node: int,
    max_pairs: int,
    seed: int,
) -> PairPayload:
    rng = np.random.default_rng(seed + payload.movie * 1009)
    distances: list[np.ndarray] = []
    left_scores: list[np.ndarray] = []
    right_scores: list[np.ndarray] = []
    pair_kinds: list[np.ndarray] = []
    for _, raw_indices in payload.rows.groupby("frame", sort=True).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        position = payload.rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float64)
        if len(indices) < 2:
            continue
        actual_k = min(neighbor_k, len(indices) - 1)
        distance, neighbors = cKDTree(position).query(position, k=actual_k + 1)
        source = np.repeat(np.arange(len(indices), dtype=np.int64), actual_k)
        target = np.asarray(neighbors[:, 1:], dtype=np.int64).reshape(-1)
        local_distance = np.asarray(distance[:, 1:], dtype=np.float64).reshape(-1)
        keep = source < target
        source = source[keep]
        target = target[keep]
        local_distance = local_distance[keep]
        distances.append(local_distance)
        left_scores.append(payload.normal_score[indices[source]])
        right_scores.append(payload.normal_score[indices[target]])
        pair_kinds.append(np.full(len(source), "knn", dtype=object))

        random_count = len(indices) * random_pairs_per_node
        random_source = rng.integers(0, len(indices), size=random_count)
        random_target = rng.integers(0, len(indices) - 1, size=random_count)
        random_target += random_target >= random_source
        random_distance = np.linalg.norm(
            position[random_source] - position[random_target],
            axis=1,
        )
        distances.append(random_distance)
        left_scores.append(payload.normal_score[indices[random_source]])
        right_scores.append(payload.normal_score[indices[random_target]])
        pair_kinds.append(np.full(random_count, "random", dtype=object))
    result = PairPayload(
        distance=np.concatenate(distances),
        left=np.concatenate(left_scores),
        right=np.concatenate(right_scores),
        pair_kind=np.concatenate(pair_kinds),
    )
    if max_pairs > 0 and len(result.distance) > max_pairs:
        selected = np.sort(
            rng.choice(len(result.distance), size=max_pairs, replace=False)
        )
        result = PairPayload(
            result.distance[selected],
            result.left[selected],
            result.right[selected],
            result.pair_kind[selected],
        )
    return result


def concatenate_pairs(payloads: list[PairPayload]) -> PairPayload:
    return PairPayload(
        np.concatenate([payload.distance for payload in payloads]),
        np.concatenate([payload.left for payload in payloads]),
        np.concatenate([payload.right for payload in payloads]),
        np.concatenate([payload.pair_kind for payload in payloads]),
    )


def correlation(distance: np.ndarray, parameters: KernelParameters) -> np.ndarray:
    value = np.full_like(distance, parameters.global_weight, dtype=np.float64)
    if parameters.local_weight > 0:
        value += parameters.local_weight * np.exp(
            -distance / max(parameters.length_px, 1e-3)
        )
    return np.clip(value, -0.95, 0.95)


def pairwise_log_copula(
    pairs: PairPayload,
    parameters: KernelParameters,
    *,
    distance_override: np.ndarray | None = None,
    right_override: np.ndarray | None = None,
) -> np.ndarray:
    distance = pairs.distance if distance_override is None else distance_override
    right = pairs.right if right_override is None else right_override
    rho = correlation(distance, parameters)[:, None]
    denominator = np.maximum(1.0 - np.square(rho), 1e-6)
    numerator = (
        2.0 * rho * pairs.left * right
        - np.square(rho) * (np.square(pairs.left) + np.square(right))
    )
    return -0.5 * np.log(denominator) + numerator / (2.0 * denominator)


def pairwise_nll(
    pairs: PairPayload,
    parameters: KernelParameters,
    *,
    distance_override: np.ndarray | None = None,
    right_override: np.ndarray | None = None,
) -> float:
    return -float(
        np.mean(
            pairwise_log_copula(
                pairs,
                parameters,
                distance_override=distance_override,
                right_override=right_override,
            )
        )
    )


def unpack_parameters(family: str, values: np.ndarray) -> KernelParameters:
    if family == "independent":
        return KernelParameters(family, 0.0, 0.0, 100.0)
    if family == "global":
        return KernelParameters(family, float(values[0]), 0.0, 100.0)
    if family == "local":
        return KernelParameters(family, 0.0, float(values[0]), float(np.exp(values[1])))
    if family == "global_local":
        return KernelParameters(
            family,
            float(values[0]),
            float(values[1]),
            float(np.exp(values[2])),
        )
    raise KeyError(family)


def fit_kernel(
    pairs: PairPayload,
    family: str,
    *,
    shuffled_right: bool,
    seed: int,
) -> tuple[KernelParameters, float]:
    if family == "independent":
        parameters = unpack_parameters(family, np.empty(0))
        return parameters, pairwise_nll(pairs, parameters)
    if family == "global":
        initial = np.array([0.1])
        bounds = [(0.0, 0.75)]
    elif family == "local":
        initial = np.array([0.15, np.log(80.0)])
        bounds = [(0.0, 0.75), (np.log(5.0), np.log(800.0))]
    elif family == "global_local":
        initial = np.array([0.08, 0.15, np.log(80.0)])
        bounds = [
            (0.0, 0.65),
            (0.0, 0.75),
            (np.log(5.0), np.log(800.0)),
        ]
    else:
        raise KeyError(family)
    right = None
    if shuffled_right:
        rng = np.random.default_rng(seed)
        right = pairs.right[rng.permutation(len(pairs.right))]

    def objective(values: np.ndarray) -> float:
        parameters = unpack_parameters(family, values)
        if parameters.global_weight + parameters.local_weight >= 0.95:
            return 1e3 + 1e3 * (
                parameters.global_weight + parameters.local_weight - 0.95
            )
        return pairwise_nll(pairs, parameters, right_override=right)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-10},
    )
    parameters = unpack_parameters(family, result.x)
    return parameters, float(result.fun)


def recursive_spatial_blocks(
    position: np.ndarray,
    max_size: int,
) -> list[np.ndarray]:
    blocks = [np.arange(len(position), dtype=np.int64)]
    result: list[np.ndarray] = []
    while blocks:
        indices = blocks.pop()
        if len(indices) <= max_size:
            result.append(indices)
            continue
        local = position[indices]
        axis = int(np.argmax(np.ptp(local, axis=0)))
        order = indices[np.argsort(local[:, axis], kind="mergesort")]
        middle = len(order) // 2
        blocks.extend([order[:middle], order[middle:]])
    return result


def kernel_matrix(
    position: np.ndarray,
    parameters: KernelParameters,
) -> np.ndarray:
    distance = distance_matrix(position, position)
    local = np.exp(-distance / max(parameters.length_px, 1e-3))
    count = len(position)
    matrix = (
        np.eye(count)
        * max(1.0 - parameters.global_weight - parameters.local_weight, 1e-4)
        + np.ones((count, count)) * parameters.global_weight
        + local * parameters.local_weight
    )
    return (matrix + matrix.T) / 2.0


def sample_block(
    payload: MoviePayload,
    indices: np.ndarray,
    parameters: KernelParameters,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    position = payload.rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float64)
    covariance = kernel_matrix(position, parameters)
    cholesky = np.linalg.cholesky(
        covariance + np.eye(len(indices)) * 1e-6
    )
    standard = rng.normal(size=(samples, len(indices), 2))
    correlated = np.empty_like(standard)
    for axis in range(2):
        correlated[..., axis] = np.einsum(
            "ij,sj->si",
            cholesky,
            standard[..., axis],
            optimize=True,
        )
    uniform = np.clip(norm.cdf(correlated), 1e-6, 1.0 - 1e-6)
    t_noise = student_t.ppf(uniform, df=payload.degrees_of_freedom)
    return (
        payload.mean[indices][None, :, :]
        + t_noise * payload.scale[indices][None, :, :]
    )


def energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    flat_samples = samples.reshape(len(samples), -1)
    flat_target = target.reshape(-1)
    first = np.linalg.norm(flat_samples - flat_target[None, :], axis=1).mean()
    paired = np.roll(flat_samples, shift=1, axis=0)
    second = np.linalg.norm(flat_samples - paired, axis=1).mean()
    return float((first - 0.5 * second) / math.sqrt(len(flat_target)))


def variogram_score(
    samples: np.ndarray,
    target: np.ndarray,
    position: np.ndarray,
    neighbor_k: int = 8,
    power: float = 0.5,
) -> float:
    if len(target) < 2:
        return np.nan
    actual = min(neighbor_k, len(target) - 1)
    distance, neighbors = cKDTree(position).query(position, k=actual + 1)
    source = np.repeat(np.arange(len(target)), actual)
    destination = np.asarray(neighbors[:, 1:]).reshape(-1)
    pair_distance = np.asarray(distance[:, 1:]).reshape(-1)
    keep = source < destination
    source = source[keep]
    destination = destination[keep]
    pair_distance = pair_distance[keep]
    weights = np.exp(-pair_distance / 100.0)
    score = 0.0
    for axis in range(2):
        observed = np.abs(
            target[source, axis] - target[destination, axis]
        ) ** power
        expected = np.mean(
            np.abs(
                samples[:, source, axis] - samples[:, destination, axis]
            )
            ** power,
            axis=0,
        )
        score += float(
            np.sum(weights * np.square(observed - expected))
            / max(np.sum(weights), EPS)
        )
    return score


def particle_scores(
    payload: MoviePayload,
    parameters: KernelParameters,
    *,
    samples: int,
    max_block_size: int,
    max_blocks_per_frame: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for frame, raw_indices in payload.rows.groupby("frame", sort=True).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        position = payload.rows.iloc[indices][["x_px", "y_px"]].to_numpy(np.float64)
        blocks = recursive_spatial_blocks(position, max_block_size)
        blocks = sorted(blocks, key=lambda block: int(block[0]))[:max_blocks_per_frame]
        for block_index, local_indices in enumerate(blocks):
            global_indices = indices[local_indices]
            rng = np.random.default_rng(
                seed + payload.movie * 100_003 + int(frame) * 101 + block_index
            )
            draws = sample_block(
                payload,
                global_indices,
                parameters,
                samples,
                rng,
            )
            target = payload.target[global_indices]
            local_position = payload.rows.iloc[global_indices][
                ["x_px", "y_px"]
            ].to_numpy(np.float64)
            records.append(
                {
                    "movie": payload.movie,
                    "frame": int(frame),
                    "block": block_index,
                    "family": parameters.family,
                    "nodes": len(global_indices),
                    "energy_score": energy_score(draws, target),
                    "variogram_score": variogram_score(
                        draws,
                        target,
                        local_position,
                    ),
                }
            )
    return pd.DataFrame(records)


def sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    masks = np.arange(1 << len(values), dtype=np.uint64)
    bits = np.arange(len(values), dtype=np.uint64)
    signs = np.where(((masks[:, None] >> bits) & 1) > 0, 1.0, -1.0)
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float(np.mean(null >= abs(float(np.mean(values))) - 1e-12))


def main() -> None:
    args = parse_args()
    started = time.time()
    movies = v139.parse_ints(args.movies)
    seeds = v139.parse_ints(args.seeds)
    families = parse_strings(args.families)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        movie: load_movie(args.v102_root, movie, seeds)
        for movie in movies
    }
    pairs = {
        movie: build_pairs(
            payloads[movie],
            neighbor_k=args.neighbor_k,
            random_pairs_per_node=args.random_pairs_per_node,
            max_pairs=args.max_pairs_per_movie,
            seed=args.seed,
        )
        for movie in movies
    }
    parameter_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    selected_parameters: dict[int, KernelParameters] = {}
    for test_movie in movies:
        validation_movie = VALIDATION_MAP.get(
            test_movie,
            movies[(movies.index(test_movie) + 1) % len(movies)],
        )
        train_movies = [
            movie
            for movie in movies
            if movie not in {test_movie, validation_movie}
        ]
        train_pairs = concatenate_pairs([pairs[movie] for movie in train_movies])
        validation_pairs = pairs[validation_movie]
        test_pairs = pairs[test_movie]
        candidates: list[tuple[float, KernelParameters]] = []
        for family in families:
            parameters, train_nll = fit_kernel(
                train_pairs,
                family,
                shuffled_right=False,
                seed=args.seed + test_movie,
            )
            validation_nll = pairwise_nll(validation_pairs, parameters)
            test_nll = pairwise_nll(test_pairs, parameters)
            candidates.append((validation_nll, parameters))
            parameter_rows.append(
                {
                    "test_movie": test_movie,
                    "validation_movie": validation_movie,
                    "train_movies": ",".join(map(str, train_movies)),
                    "family": family,
                    "global_weight": parameters.global_weight,
                    "local_weight": parameters.local_weight,
                    "length_px": parameters.length_px,
                    "train_pair_nll": train_nll,
                    "validation_pair_nll": validation_nll,
                    "test_pair_nll": test_nll,
                    "selected": False,
                }
            )
            pair_rows.append(
                {
                    "test_movie": test_movie,
                    "control": "real_graph",
                    "family": family,
                    "pair_nll": test_nll,
                    "copula_log_gain_vs_independent": -test_nll,
                    "pairs": len(test_pairs.distance),
                }
            )
        _, selected = min(candidates, key=lambda item: item[0])
        selected_parameters[test_movie] = selected
        for row in parameter_rows:
            if row["test_movie"] == test_movie and row["family"] == selected.family:
                row["selected"] = True
        rng = np.random.default_rng(args.seed + test_movie * 1009)
        shuffled_distance = test_pairs.distance[
            rng.permutation(len(test_pairs.distance))
        ]
        pair_rows.extend(
            [
                {
                    "test_movie": test_movie,
                    "control": "selected_real_graph",
                    "family": selected.family,
                    "pair_nll": pairwise_nll(test_pairs, selected),
                    "copula_log_gain_vs_independent": -pairwise_nll(
                        test_pairs,
                        selected,
                    ),
                    "pairs": len(test_pairs.distance),
                },
                {
                    "test_movie": test_movie,
                    "control": "selected_shuffled_distance",
                    "family": selected.family,
                    "pair_nll": pairwise_nll(
                        test_pairs,
                        selected,
                        distance_override=shuffled_distance,
                    ),
                    "copula_log_gain_vs_independent": -pairwise_nll(
                        test_pairs,
                        selected,
                        distance_override=shuffled_distance,
                    ),
                    "pairs": len(test_pairs.distance),
                },
            ]
        )
        shuffled_fit, _ = fit_kernel(
            train_pairs,
            selected.family,
            shuffled_right=True,
            seed=args.seed + test_movie * 17,
        )
        shuffled_fit_nll = pairwise_nll(test_pairs, shuffled_fit)
        pair_rows.append(
            {
                "test_movie": test_movie,
                "control": "selected_shuffled_residual_fit",
                "family": selected.family,
                "pair_nll": shuffled_fit_nll,
                "copula_log_gain_vs_independent": -shuffled_fit_nll,
                "pairs": len(test_pairs.distance),
            }
        )

    parameters_frame = pd.DataFrame(parameter_rows)
    pair_frame = pd.DataFrame(pair_rows)
    particle_frames: list[pd.DataFrame] = []
    for movie in movies:
        selected = selected_parameters[movie]
        particle_frames.append(
            particle_scores(
                payloads[movie],
                KernelParameters("independent", 0.0, 0.0, 100.0),
                samples=args.particle_samples,
                max_block_size=args.max_block_size,
                max_blocks_per_frame=args.max_blocks_per_frame,
                seed=args.seed,
            )
        )
        selected_for_output = KernelParameters(
            "selected_joint",
            selected.global_weight,
            selected.local_weight,
            selected.length_px,
        )
        particle_frames.append(
            particle_scores(
                payloads[movie],
                selected_for_output,
                samples=args.particle_samples,
                max_block_size=args.max_block_size,
                max_blocks_per_frame=args.max_blocks_per_frame,
                seed=args.seed,
            )
        )
    particle_frame = pd.concat(particle_frames, ignore_index=True)
    particle_movie = (
        particle_frame.groupby(["movie", "family"], as_index=False)
        .agg(
            energy_score=("energy_score", "mean"),
            variogram_score=("variogram_score", "mean"),
            blocks=("block", "size"),
        )
    )
    particle_pivot = particle_movie.pivot(
        index="movie",
        columns="family",
        values=["energy_score", "variogram_score"],
    )
    particle_pivot.columns = [
        f"{metric}__{family}" for metric, family in particle_pivot.columns
    ]
    particle_pivot = particle_pivot.reset_index()
    particle_pivot["energy_improvement"] = (
        particle_pivot["energy_score__independent"]
        - particle_pivot["energy_score__selected_joint"]
    )
    particle_pivot["variogram_improvement"] = (
        particle_pivot["variogram_score__independent"]
        - particle_pivot["variogram_score__selected_joint"]
    )

    selected_pairs = pair_frame[
        pair_frame["control"].eq("selected_real_graph")
    ].sort_values("test_movie")
    shuffled_pairs = pair_frame[
        pair_frame["control"].eq("selected_shuffled_distance")
    ].sort_values("test_movie")
    pair_gain = selected_pairs["copula_log_gain_vs_independent"].to_numpy()
    control_margin = (
        selected_pairs["copula_log_gain_vs_independent"].to_numpy()
        - shuffled_pairs["copula_log_gain_vs_independent"].to_numpy()
    )
    energy_gain = particle_pivot["energy_improvement"].to_numpy()
    variogram_gain = particle_pivot["variogram_improvement"].to_numpy()
    pass_gate = bool(
        np.mean(pair_gain) > 0
        and np.mean(control_margin) > 0
        and np.mean(energy_gain) > 0
        and np.mean(variogram_gain) > 0
    )
    summary = pd.DataFrame(
        [
            {
                "movies": len(movies),
                "pair_log_gain_mean": float(np.mean(pair_gain)),
                "pair_log_gain_movies_positive": int(np.sum(pair_gain > 0)),
                "pair_log_gain_sign_flip_p": sign_flip_p(pair_gain),
                "shuffled_distance_margin_mean": float(np.mean(control_margin)),
                "energy_improvement_mean": float(np.mean(energy_gain)),
                "energy_improvement_movies_positive": int(
                    np.sum(energy_gain > 0)
                ),
                "variogram_improvement_mean": float(np.mean(variogram_gain)),
                "variogram_improvement_movies_positive": int(
                    np.sum(variogram_gain > 0)
                ),
                "joint_gate_pass": pass_gate,
            }
        ]
    )
    parameters_frame.to_csv(args.out_dir / "v154_kernel_parameters.csv", index=False)
    pair_frame.to_csv(args.out_dir / "v154_pairwise_scores.csv", index=False)
    particle_frame.to_csv(args.out_dir / "v154_particle_block_scores.csv", index=False)
    particle_pivot.to_csv(args.out_dir / "v154_particle_movie_scores.csv", index=False)
    summary.to_csv(args.out_dir / "v154_summary.csv", index=False)
    lines = [
        "# v154 Joint Graph Copula Gate",
        "",
        f"Decision: **{'PASS' if pass_gate else 'FAIL'}**",
        "",
        f"- pairwise copula log-score gain: `{np.mean(pair_gain):.6f}` nats/component-pair",
        f"- movies with positive pairwise gain: `{np.sum(pair_gain > 0)}/{len(movies)}`",
        f"- margin over shuffled distances: `{np.mean(control_margin):.6f}`",
        f"- equal-budget particle energy improvement: `{np.mean(energy_gain):.6f}`",
        f"- particle variogram improvement: `{np.mean(variogram_gain):.6f}`",
        "",
        "The v97 conditional mean and each Student-t marginal remain frozen.",
        "A pass validates joint dependence/coherent particles, not a lower point RMSE.",
        "Sequential posterior transport and any bounded mean update are separate later gates.",
    ]
    (args.out_dir / "v154_status_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "v154",
                "command": sys.argv,
                "elapsed_seconds": time.time() - started,
                "movies": movies,
                "seeds": seeds,
                "point_mean_refit": False,
                "marginal_distribution_refit": False,
                "joint_density": "disjoint-block particles + explicitly labelled pairwise composite likelihood",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "v154_status_report.md", flush=True)


if __name__ == "__main__":
    main()
