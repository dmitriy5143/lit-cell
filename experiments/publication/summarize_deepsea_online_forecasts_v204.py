#!/usr/bin/env python3
"""Summarize DeepSea rolling forecasts in native and normalized units.

This script performs no training and does not alter predictions. It detects
whether the source cache uses pixels or first-frame cell diameters, converts
both targets and predictions consistently, reconstructs rolling endpoints,
and reports movie-macro and family-macro summaries. The cell-diameter scale is
the video-initialization quantity frozen in the v204 data contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import spearmanr, t as student_t


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402


EPS = 1e-8
HORIZONS = (1, 2, 4, 6)


def component_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def vector_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    centered = target - np.mean(target, axis=0, keepdims=True)
    denominator = max(float(np.sum(np.square(centered))), EPS)
    return float(1.0 - np.sum(np.square(target - prediction)) / denominator)


def angular_cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.maximum(
        np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1),
        EPS,
    )
    return float(np.mean(np.sum(target * prediction, axis=1) / denominator))


def parse_runs(specifications: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected NAME=PATH, got {specification!r}")
        name, raw_path = specification.split("=", 1)
        runs[name.strip()] = Path(raw_path).expanduser().resolve()
    return runs


def prediction_arrays(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "v97_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    archive = np.load(path, allow_pickle=False)
    predictions: dict[str, np.ndarray] = {}
    for key in archive.files:
        if key.endswith("__prediction"):
            predictions[key.removesuffix("__prediction")] = archive[key].astype(np.float32)
        elif key.startswith("baseline__"):
            predictions[key.removeprefix("baseline__")] = archive[key].astype(np.float32)
    if not predictions:
        raise RuntimeError(f"No predictions found in {path}")
    return predictions


def student_t_nll(
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: float,
) -> np.ndarray:
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)
    residual = (
        np.asarray(target, dtype=np.float64)
        - np.asarray(prediction, dtype=np.float64)
    ) / scale
    df = float(degrees_of_freedom)
    constant = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi)
    )
    log_density = (
        constant
        - np.log(scale)
        - 0.5 * (df + 1.0) * np.log1p(np.square(residual) / df)
    )
    return -np.sum(log_density, axis=1)


def uncertainty_movie_metrics(
    run_dir: Path,
    rows: pd.DataFrame,
    target_h1_px: np.ndarray,
    run_name: str,
    source_unit: str,
) -> list[dict[str, Any]]:
    archive = np.load(run_dir / "v97_predictions.npz", allow_pickle=False)
    records: list[dict[str, Any]] = []
    for key in archive.files:
        if not key.endswith("__prediction"):
            continue
        method = key.removesuffix("__prediction")
        scale_key = f"{method}__scale"
        df_key = f"{method}__degrees_of_freedom"
        if scale_key not in archive.files or df_key not in archive.files:
            continue
        prediction_px = archive[key].astype(np.float32)
        scale_px = archive[scale_key].astype(np.float32)
        degrees_of_freedom = float(np.asarray(archive[df_key]).item())
        for unit in ("pixel", "cell_diameter"):
            normalization = unit_scale(rows, unit, source_unit)
            target = target_h1_px / normalization[:, None]
            prediction = prediction_px / normalization[:, None]
            scale = scale_px / normalization[:, None]
            # For the axis-aligned joint box, each marginal receives sqrt(p)
            # coverage so the nominal joint coverage is p under independence.
            threshold50 = float(
                student_t.ppf(
                    (1.0 + np.sqrt(0.50)) / 2.0,
                    degrees_of_freedom,
                )
            )
            threshold90 = float(
                student_t.ppf(
                    (1.0 + np.sqrt(0.90)) / 2.0,
                    degrees_of_freedom,
                )
            )
            absolute_standardized = np.abs(target - prediction) / np.maximum(
                scale,
                1e-6,
            )
            nll = student_t_nll(
                target,
                prediction,
                scale,
                degrees_of_freedom,
            )
            error_norm = np.linalg.norm(target - prediction, axis=1)
            scale_norm = np.linalg.norm(scale, axis=1)
            local = rows.reset_index(drop=True).copy()
            local["_nll"] = nll
            local["_coverage50"] = np.all(
                absolute_standardized <= threshold50,
                axis=1,
            )
            local["_coverage90"] = np.all(
                absolute_standardized <= threshold90,
                axis=1,
            )
            local["_error_norm"] = error_norm
            local["_scale_norm"] = scale_norm
            for sequence, movie_rows in local.groupby("sequence", sort=True):
                correlation = spearmanr(
                    movie_rows._scale_norm.to_numpy(),
                    movie_rows._error_norm.to_numpy(),
                ).statistic
                records.append(
                    {
                        "run": run_name,
                        "method": method,
                        "metric_unit": unit,
                        "sequence": int(sequence),
                        "family": str(movie_rows.family.iloc[0]),
                        "video": str(movie_rows.video.iloc[0]),
                        "degrees_of_freedom": degrees_of_freedom,
                        "joint_student_t_nll_h1": float(
                            movie_rows._nll.mean()
                        ),
                        "box_coverage50_h1": float(
                            movie_rows._coverage50.mean()
                        ),
                        "box_coverage90_h1": float(
                            movie_rows._coverage90.mean()
                        ),
                        "uncertainty_error_spearman_h1": (
                            float(correlation)
                            if np.isfinite(correlation)
                            else np.nan
                        ),
                        "rows": len(movie_rows),
                    }
                )
    return records


def unit_scale(
    rows: pd.DataFrame,
    unit: str,
    source_unit: str,
) -> np.ndarray:
    if source_unit == unit:
        return np.ones(len(rows), dtype=np.float32)
    diameter = rows.reference_diameter_px.to_numpy(np.float32)
    if not np.all(np.isfinite(diameter) & (diameter > 0)):
        raise RuntimeError("Invalid reference_diameter_px values")
    if source_unit == "pixel" and unit == "cell_diameter":
        return diameter
    if source_unit == "cell_diameter" and unit == "pixel":
        return 1.0 / diameter
    raise ValueError(f"Unsupported conversion: {source_unit} -> {unit}")


def rolling_movie_metrics(
    rows: pd.DataFrame,
    target_h1_px: np.ndarray,
    prediction_h1_px: np.ndarray,
    run: str,
    method: str,
    unit: str,
    source_unit: str,
) -> list[dict[str, Any]]:
    local = rows.reset_index(drop=True).copy()
    local["_row"] = np.arange(len(local), dtype=np.int64)
    scale = unit_scale(local, unit, source_unit)
    target_h1 = target_h1_px / scale[:, None]
    prediction_h1 = prediction_h1_px / scale[:, None]
    records: list[dict[str, Any]] = []
    for sequence, movie_rows in local.groupby("sequence", sort=True):
        movie_rows = movie_rows.sort_values(["track_id", "frame"])
        for horizon in HORIZONS:
            target_parts: list[np.ndarray] = []
            prediction_parts: list[np.ndarray] = []
            for _, track in movie_rows.groupby("track_id", sort=False):
                indices = track._row.to_numpy(np.int64)
                frames = track.frame.to_numpy(np.int64)
                for start in range(len(indices) - horizon + 1):
                    stop = start + horizon
                    if not np.all(np.diff(frames[start:stop]) == 1):
                        continue
                    selected = indices[start:stop]
                    target_parts.append(np.sum(target_h1[selected], axis=0))
                    prediction_parts.append(np.sum(prediction_h1[selected], axis=0))
            if not target_parts:
                continue
            target = np.asarray(target_parts, dtype=np.float32)
            prediction = np.asarray(prediction_parts, dtype=np.float32)
            family = str(movie_rows.family.iloc[0])
            video = str(movie_rows.video.iloc[0])
            records.append(
                {
                    "run": run,
                    "method": method,
                    "metric_unit": unit,
                    "sequence": int(sequence),
                    "family": family,
                    "video": video,
                    "frame_interval_min": float(movie_rows.frame_interval_min.iloc[0]),
                    "reference_diameter_px": float(movie_rows.reference_diameter_px.iloc[0]),
                    "horizon": horizon,
                    "component_rmse": component_rmse(target, prediction),
                    "r2": vector_r2(target, prediction),
                    "angular_cosine": angular_cosine(target, prediction),
                    "magnitude_ratio": float(
                        np.mean(np.linalg.norm(prediction, axis=1))
                        / max(float(np.mean(np.linalg.norm(target, axis=1))), EPS)
                    ),
                    "n_windows": len(target),
                }
            )
    return records


def aggregate(movie: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie_macro = (
        movie.groupby(["run", "method", "metric_unit", "horizon"], as_index=False)
        .agg(
            movie_macro_rmse=("component_rmse", "mean"),
            movie_macro_rmse_std=("component_rmse", "std"),
            movie_macro_r2=("r2", "mean"),
            movie_macro_angular_cosine=("angular_cosine", "mean"),
            movie_macro_magnitude_ratio=("magnitude_ratio", "mean"),
            movies=("sequence", "nunique"),
            windows=("n_windows", "sum"),
        )
    )
    family = (
        movie.groupby(
            ["run", "method", "metric_unit", "family", "horizon"],
            as_index=False,
        )
        .agg(
            family_movie_macro_rmse=("component_rmse", "mean"),
            family_movie_macro_r2=("r2", "mean"),
            family_movie_macro_angular_cosine=("angular_cosine", "mean"),
            movies=("sequence", "nunique"),
            windows=("n_windows", "sum"),
        )
    )
    family_macro = (
        family.groupby(["run", "method", "metric_unit", "horizon"], as_index=False)
        .agg(
            family_macro_rmse=("family_movie_macro_rmse", "mean"),
            family_macro_r2=("family_movie_macro_r2", "mean"),
            families=("family", "nunique"),
        )
    )
    movie_macro = movie_macro.merge(
        family_macro,
        on=["run", "method", "metric_unit", "horizon"],
        how="left",
        validate="one_to_one",
    )
    return movie_macro, family


def paired_bootstrap(
    movie: pd.DataFrame,
    reference_run: str,
    reference_method: str,
    candidate_run: str,
    candidate_method: str,
    unit: str,
    horizon: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    reference = movie[
        (movie.run == reference_run)
        & (movie.method == reference_method)
        & (movie.metric_unit == unit)
        & (movie.horizon == horizon)
    ][["sequence", "component_rmse"]].rename(columns={"component_rmse": "reference"})
    candidate = movie[
        (movie.run == candidate_run)
        & (movie.method == candidate_method)
        & (movie.metric_unit == unit)
        & (movie.horizon == horizon)
    ][["sequence", "component_rmse"]].rename(columns={"component_rmse": "candidate"})
    paired = reference.merge(candidate, on="sequence", validate="one_to_one")
    if len(paired) < 2:
        raise RuntimeError("At least two paired movies are required")
    reference_values = paired.reference.to_numpy(float)
    candidate_values = paired.candidate.to_numpy(float)
    observed = 100.0 * (
        np.mean(reference_values) - np.mean(candidate_values)
    ) / max(float(np.mean(reference_values)), EPS)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(paired), size=len(paired))
        reference_mean = float(np.mean(reference_values[indices]))
        candidate_mean = float(np.mean(candidate_values[indices]))
        samples[repeat] = 100.0 * (
            reference_mean - candidate_mean
        ) / max(reference_mean, EPS)
    per_movie = 100.0 * (
        reference_values - candidate_values
    ) / np.maximum(reference_values, EPS)
    return {
        "reference_run": reference_run,
        "reference_method": reference_method,
        "candidate_run": candidate_run,
        "candidate_method": candidate_method,
        "metric_unit": unit,
        "horizon": horizon,
        "movies": len(paired),
        "movie_macro_gain_pct": observed,
        "positive_movies": int(np.sum(per_movie > 0)),
        "median_movie_gain_pct": float(np.median(per_movie)),
        "bootstrap_ci_low": float(np.percentile(samples, 2.5)),
        "bootstrap_ci_high": float(np.percentile(samples, 97.5)),
    }


def paired_nll_bootstrap(
    movie: pd.DataFrame,
    reference_run: str,
    reference_method: str,
    candidate_run: str,
    candidate_method: str,
    unit: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    reference = movie[
        (movie.run == reference_run)
        & (movie.method == reference_method)
        & (movie.metric_unit == unit)
    ][["sequence", "joint_student_t_nll_h1"]].rename(
        columns={"joint_student_t_nll_h1": "reference"}
    )
    candidate = movie[
        (movie.run == candidate_run)
        & (movie.method == candidate_method)
        & (movie.metric_unit == unit)
    ][["sequence", "joint_student_t_nll_h1"]].rename(
        columns={"joint_student_t_nll_h1": "candidate"}
    )
    paired = reference.merge(candidate, on="sequence", validate="one_to_one")
    if len(paired) < 2:
        raise RuntimeError("At least two paired NLL movies are required")
    differences = (
        paired.reference.to_numpy(float)
        - paired.candidate.to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(paired), size=len(paired))
        samples[repeat] = float(np.mean(differences[indices]))
    return {
        "reference_run": reference_run,
        "reference_method": reference_method,
        "candidate_run": candidate_run,
        "candidate_method": candidate_method,
        "metric_unit": unit,
        "movies": len(paired),
        "joint_student_t_nll_reduction_h1": float(
            np.mean(differences)
        ),
        "positive_movies": int(np.sum(differences > 0)),
        "bootstrap_ci_low": float(np.percentile(samples, 2.5)),
        "bootstrap_ci_high": float(np.percentile(samples, 97.5)),
    }


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, _, test = v85.load_anchor_cache(args.cache_dir)
    cache_contract_path = args.cache_dir / "final_native" / "contract.json"
    cache_contract = json.loads(cache_contract_path.read_text(encoding="utf-8"))
    source_unit = str(cache_contract["coordinate_unit"])
    if source_unit not in {"pixel", "cell_diameter"}:
        raise RuntimeError(f"Unsupported cache coordinate unit: {source_unit}")
    rows = test.rows.reset_index(drop=True)
    target_h1 = test.target_steps[:, 0].astype(np.float32)
    run_dirs = parse_runs(args.run)
    records: list[dict[str, Any]] = []
    uncertainty_records: list[dict[str, Any]] = []
    for run_index, (run_name, run_dir) in enumerate(run_dirs.items()):
        predictions = prediction_arrays(run_dir)
        if args.include_zero_baseline and run_index == 0:
            predictions["zero_displacement"] = np.zeros_like(target_h1)
        for method, prediction in predictions.items():
            if prediction.shape != target_h1.shape:
                raise RuntimeError(
                    f"Shape mismatch for {run_name}/{method}: "
                    f"{prediction.shape} != {target_h1.shape}"
                )
            for unit in ("pixel", "cell_diameter"):
                records.extend(
                    rolling_movie_metrics(
                        rows,
                        target_h1,
                        prediction,
                        run_name,
                        method,
                        unit,
                        source_unit,
                    )
                )
        uncertainty_records.extend(
            uncertainty_movie_metrics(
                run_dir,
                rows,
                target_h1,
                run_name,
                source_unit,
            )
        )
    movie = pd.DataFrame(records)
    macro, family = aggregate(movie)
    movie.to_csv(args.out_dir / "v204_online_movie_metrics.csv", index=False)
    macro.to_csv(args.out_dir / "v204_online_macro_metrics.csv", index=False)
    family.to_csv(args.out_dir / "v204_online_family_metrics.csv", index=False)
    uncertainty_movie = pd.DataFrame(uncertainty_records)
    if not uncertainty_movie.empty:
        uncertainty_macro = (
            uncertainty_movie.groupby(
                ["run", "method", "metric_unit"],
                as_index=False,
            )
            .agg(
                movie_macro_joint_student_t_nll_h1=(
                    "joint_student_t_nll_h1",
                    "mean",
                ),
                movie_macro_box_coverage50_h1=(
                    "box_coverage50_h1",
                    "mean",
                ),
                movie_macro_box_coverage90_h1=(
                    "box_coverage90_h1",
                    "mean",
                ),
                movie_macro_uncertainty_error_spearman_h1=(
                    "uncertainty_error_spearman_h1",
                    "mean",
                ),
                movies=("sequence", "nunique"),
            )
        )
        uncertainty_movie.to_csv(
            args.out_dir / "v204_online_uncertainty_movie.csv",
            index=False,
        )
        uncertainty_macro.to_csv(
            args.out_dir / "v204_online_uncertainty_macro.csv",
            index=False,
        )

    bootstrap_rows: list[dict[str, Any]] = []
    nll_bootstrap_rows: list[dict[str, Any]] = []
    for comparison in args.compare:
        parts = comparison.split(":")
        if len(parts) != 4:
            raise ValueError(
                "--compare expects REF_RUN:REF_METHOD:CAND_RUN:CAND_METHOD"
            )
        for unit in ("pixel", "cell_diameter"):
            for horizon in HORIZONS:
                bootstrap_rows.append(
                    paired_bootstrap(
                        movie,
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3],
                        unit,
                        horizon,
                        args.bootstrap_repeats,
                        args.seed + horizon,
                    )
                )
            if not uncertainty_movie.empty:
                nll_bootstrap_rows.append(
                    paired_nll_bootstrap(
                        uncertainty_movie,
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3],
                        unit,
                        args.bootstrap_repeats,
                        args.seed + 97,
                    )
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.out_dir / "v204_online_paired_bootstrap.csv", index=False)
    if nll_bootstrap_rows:
        pd.DataFrame(nll_bootstrap_rows).to_csv(
            args.out_dir / "v204_online_nll_paired_bootstrap.csv",
            index=False,
        )

    contract = {
        "cache_dir": str(args.cache_dir.resolve()),
        "runs": {name: str(path) for name, path in run_dirs.items()},
        "test_rows": len(rows),
        "test_movies": int(rows.sequence.nunique()),
        "test_tracks": int(rows[["sequence", "track_id"]].drop_duplicates().shape[0]),
        "source_coordinate_unit": source_unit,
        "units": ["pixel", "cell_diameter"],
        "horizons": list(HORIZONS),
        "independent_unit": "movie",
        "bootstrap_estimand": "ratio_of_movie_macro_rmse_means",
        "bootstrap_repeats": args.bootstrap_repeats,
    }
    (args.out_dir / "v204_online_summary_contract.json").write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="REF_RUN:REF_METHOD:CAND_RUN:CAND_METHOD",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-zero-baseline",
        action="store_true",
        help="Add an all-zero one-step baseline to the first supplied run.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
