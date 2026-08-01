#!/usr/bin/env python3
"""Proper-score and calibration audit for v199 streaming predictions.

Scale multipliers and conformal radii are calibrated from the other outer-LOMO
movies.  The evaluated movie never participates in calibration.  Student-t and
Gaussian laws share the same mean and raw scale before family-specific
calibration, and all methods are scored on identical h1/h6 events.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, t as student_t


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402


DEFAULT_V199 = (
    ROOT
    / "outputs/lachance_equivariant_graph_bridge_v199_full_2026-07-30"
    / "v199_graph_bridge_predictions.npz"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_probabilistic_graph_closure_v201"
EPS = 1e-12


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v199-predictions", type=Path, default=DEFAULT_V199)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument(
        "--methods",
        default="no_update,legacy_dense,forced_potential,active_advective",
    )
    parser.add_argument("--horizons", default="1,6")
    parser.add_argument("--families", default="student_t,gaussian")
    parser.add_argument(
        "--scale-factors",
        default="0.5,0.65,0.8,1,1.25,1.5,2,2.5,3",
    )
    parser.add_argument("--student-df", type=float, default=4.0)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=201)
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


def movie_ids(archive: np.lib.npyio.NpzFile) -> list[int]:
    return sorted(
        {
            int(key.split("__", 1)[0].removeprefix("movie"))
            for key in archive.files
            if key.startswith("movie") and "__keys" in key
        }
    )


def event_arrays(
    archive: np.lib.npyio.NpzFile,
    movie: int,
    objective: str,
    method: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"movie{movie:02d}"
    keys = np.asarray(archive[f"{prefix}__keys"], dtype=np.int64)
    rows = pd.DataFrame(keys, columns=["sequence", "frame", "track_id"])
    target = np.asarray(archive[f"{prefix}__target"], dtype=np.float64)
    base = np.asarray(archive[f"{prefix}__base"], dtype=np.float64)
    scale = np.maximum(
        np.asarray(archive[f"{prefix}__scale"], dtype=np.float64),
        1e-5,
    )
    prediction = (
        base
        if method == "no_update"
        else np.asarray(
            archive[f"{prefix}__{objective}__{method}"],
            dtype=np.float64,
        )
    )
    windows = v157e.consecutive_windows(rows, horizon)
    return (
        target[windows].sum(axis=1),
        prediction[windows].sum(axis=1),
        np.sqrt(np.sum(np.square(scale[windows]), axis=1)),
    )


def joint_nll(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    family: str,
    df: float,
) -> float:
    safe = np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)
    residual = (target - prediction) / safe
    if family == "gaussian":
        value = (
            0.5 * math.log(2.0 * math.pi)
            + np.log(safe)
            + 0.5 * np.square(residual)
        )
    elif family == "student_t":
        constant = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * math.log(df * math.pi)
        )
        value = (
            np.log(safe)
            - constant
            + 0.5 * (df + 1.0) * np.log1p(np.square(residual) / df)
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return float(np.mean(np.sum(value, axis=1)))


def select_scale_factor(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    family: str,
    df: float,
    factors: Iterable[float],
) -> tuple[float, float]:
    scores = [
        (
            joint_nll(target, prediction, scale * factor, family, df),
            float(factor),
        )
        for factor in factors
    ]
    return min(scores)


def energy_score(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    family: str,
    df: float,
    sample_count: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    count = 0
    chunk_size = 1024
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        shape = (stop - start, sample_count, 2)
        if family == "student_t":
            first_noise = rng.standard_t(df, size=shape)
            second_noise = rng.standard_t(df, size=shape)
        else:
            first_noise = rng.normal(size=shape)
            second_noise = rng.normal(size=shape)
        first = (
            prediction[start:stop, None]
            + scale[start:stop, None] * first_noise
        )
        second = (
            prediction[start:stop, None]
            + scale[start:stop, None] * second_noise
        )
        first_term = np.mean(
            np.linalg.norm(first - target[start:stop, None], axis=2),
            axis=1,
        )
        second_term = 0.5 * np.mean(
            np.linalg.norm(first - second, axis=2),
            axis=1,
        )
        total += float(np.sum(first_term - second_term))
        count += stop - start
    return total / max(count, 1)


def calibration_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    family: str,
    df: float,
) -> tuple[float, float, float]:
    residual = np.abs((target - prediction) / np.maximum(scale, 1e-6))
    levels = np.linspace(0.1, 0.9, 9)
    errors = []
    for level in levels:
        quantile = (
            student_t.ppf((1.0 + level) / 2.0, df=df)
            if family == "student_t"
            else norm.ppf((1.0 + level) / 2.0)
        )
        empirical = float(np.mean(residual <= quantile))
        errors.append(abs(empirical - level))
    q50 = (
        student_t.ppf(0.75, df=df)
        if family == "student_t"
        else norm.ppf(0.75)
    )
    q90 = (
        student_t.ppf(0.95, df=df)
        if family == "student_t"
        else norm.ppf(0.95)
    )
    return (
        float(np.mean(errors)),
        float(np.mean(residual <= q50)),
        float(np.mean(residual <= q90)),
    )


def conformal_coverage(
    calibration_target: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_scale: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, float]:
    calibration_score = np.linalg.norm(
        (calibration_target - calibration_prediction)
        / np.maximum(calibration_scale, 1e-6),
        axis=1,
    )
    test_score = np.linalg.norm(
        (target - prediction) / np.maximum(scale, 1e-6),
        axis=1,
    )
    q50, q90 = np.quantile(calibration_score, [0.5, 0.9], method="higher")
    return float(np.mean(test_score <= q50)), float(np.mean(test_score <= q90))


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    outcomes = [
        np.mean(values * np.asarray(signs))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(outcomes) >= observed - 1e-15))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    objectives = parse_strings(args.objectives)
    methods = parse_strings(args.methods)
    horizons = [int(value) for value in parse_strings(args.horizons)]
    families = parse_strings(args.families)
    factors = parse_floats(args.scale_factors)
    records: list[dict[str, Any]] = []

    with np.load(args.v199_predictions, allow_pickle=False) as archive:
        movies = movie_ids(archive)
        cache: dict[tuple[int, str, str, int], tuple[np.ndarray, ...]] = {}
        for movie in movies:
            for objective in objectives:
                for method in methods:
                    for horizon in horizons:
                        cache[(movie, objective, method, horizon)] = event_arrays(
                            archive,
                            movie,
                            objective,
                            method,
                            horizon,
                        )
        for objective in objectives:
            for method in methods:
                for horizon in horizons:
                    for family_index, family in enumerate(families):
                        for movie in movies:
                            target, prediction, raw_scale = cache[
                                (movie, objective, method, horizon)
                            ]
                            calibration = [
                                cache[(other, objective, method, horizon)]
                                for other in movies
                                if other != movie
                            ]
                            calibration_target = np.concatenate(
                                [item[0] for item in calibration]
                            )
                            calibration_prediction = np.concatenate(
                                [item[1] for item in calibration]
                            )
                            calibration_scale = np.concatenate(
                                [item[2] for item in calibration]
                            )
                            calibration_nll, factor = select_scale_factor(
                                calibration_target,
                                calibration_prediction,
                                calibration_scale,
                                family,
                                args.student_df,
                                factors,
                            )
                            scale = raw_scale * factor
                            calibration_error, coverage50, coverage90 = (
                                calibration_metrics(
                                    target,
                                    prediction,
                                    scale,
                                    family,
                                    args.student_df,
                                )
                            )
                            conformal50, conformal90 = conformal_coverage(
                                calibration_target,
                                calibration_prediction,
                                calibration_scale * factor,
                                target,
                                prediction,
                                scale,
                            )
                            records.append(
                                {
                                    "movie": movie,
                                    "objective_name": objective,
                                    "method": method,
                                    "horizon": horizon,
                                    "family": family,
                                    "rows": len(target),
                                    "scale_factor": factor,
                                    "calibration_nll": calibration_nll,
                                    "test_joint_nll": joint_nll(
                                        target,
                                        prediction,
                                        scale,
                                        family,
                                        args.student_df,
                                    ),
                                    "energy_score": energy_score(
                                        target,
                                        prediction,
                                        scale,
                                        family,
                                        args.student_df,
                                        args.sample_count,
                                        args.seed
                                        + 1009 * movie
                                        + 17 * horizon
                                        + family_index,
                                    ),
                                    "calibration_error": calibration_error,
                                    "marginal_coverage50": coverage50,
                                    "marginal_coverage90": coverage90,
                                    "conformal_radial_coverage50": conformal50,
                                    "conformal_radial_coverage90": conformal90,
                                    "component_rmse": float(
                                        np.sqrt(
                                            np.mean(
                                                np.square(target - prediction)
                                            )
                                        )
                                    ),
                                }
                            )

    per_movie = pd.DataFrame(records)
    aggregate = (
        per_movie.groupby(
            ["objective_name", "method", "horizon", "family"],
            as_index=False,
        )
        .agg(
            movies=("movie", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            joint_nll_mean=("test_joint_nll", "mean"),
            joint_nll_std=("test_joint_nll", "std"),
            energy_score_mean=("energy_score", "mean"),
            calibration_error_mean=("calibration_error", "mean"),
            marginal_coverage50_mean=("marginal_coverage50", "mean"),
            marginal_coverage90_mean=("marginal_coverage90", "mean"),
            conformal_radial_coverage50_mean=(
                "conformal_radial_coverage50",
                "mean",
            ),
            conformal_radial_coverage90_mean=(
                "conformal_radial_coverage90",
                "mean",
            ),
        )
    )
    delta_records = []
    for (objective, horizon, family), rows in per_movie.groupby(
        ["objective_name", "horizon", "family"],
        sort=True,
    ):
        pivot = rows.pivot(index="movie", columns="method", values="test_joint_nll")
        if "no_update" not in pivot:
            continue
        for method in methods:
            if method == "no_update" or method not in pivot:
                continue
            delta = pivot["no_update"] - pivot[method]
            delta_records.append(
                {
                    "objective_name": objective,
                    "horizon": horizon,
                    "family": family,
                    "method": method,
                    "mean_nll_gain_vs_no_update": float(delta.mean()),
                    "positive_movies": int(np.sum(delta > 0)),
                    "one_sided_sign_flip_p": exact_sign_flip(
                        delta.to_numpy(np.float64)
                    ),
                }
            )
    deltas = pd.DataFrame(delta_records)
    per_movie.to_csv(
        args.out_dir / "v201_probabilistic_per_movie.csv",
        index=False,
    )
    aggregate.to_csv(
        args.out_dir / "v201_probabilistic_aggregate.csv",
        index=False,
    )
    deltas.to_csv(
        args.out_dir / "v201_probabilistic_paired_delta.csv",
        index=False,
    )
    report = [
        "# v201 Common Probabilistic Closure",
        "",
        "Every row uses outer-movie calibration. Optimizer seeds are not treated",
        "as independent biological replicates.",
        "",
        "## Aggregate proper scores and calibration",
        "",
        aggregate.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired movie-level NLL gains",
        "",
        deltas.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The h6 law is a calibrated cumulative approximation; it is not claimed",
        "to be the exact convolution of dependent Student-t innovations.",
    ]
    (args.out_dir / "v201_probabilistic_decision_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "v199_predictions": str(args.v199_predictions.resolve()),
        "movies": movies,
        "objectives": objectives,
        "methods": methods,
        "horizons": horizons,
        "families": families,
        "student_df": args.student_df,
        "sample_count": args.sample_count,
        "test_movie_used_for_calibration": False,
        "optimizer_seeds_as_independent_units": False,
    }
    (args.out_dir / "v201_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
