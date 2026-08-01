#!/usr/bin/env python3
"""Strict fold-local stress audit for the semigroup innovation update.

The runner regenerates the v97 ensemble inside every outer fold, selects and
refits the update-only semigroup correction, then degrades only its completed
innovation packet.  It measures cadence, missing updates, coherent delay,
wrong-cell updates, and TrackMate-like coordinate noise.

This is an innovation-channel stress test.  It does not replace v103, where
raw observed coordinates are corrupted before velocity/history/base-proposal
features are recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h
import run_lachance_joint_graph_copula_v154 as v154


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lachance_foldlocal_semigroup_stress_v157f"
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--cadences", default="1,2,3,6")
    parser.add_argument("--missing-rates", default="0.1,0.2,0.4")
    parser.add_argument("--noise-px", default="0.25,0.5,1.0")
    parser.add_argument("--noise-propagation-draws", type=int, default=12)
    parser.add_argument(
        "--uncertainty-scale-grid",
        default=(
            "0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,"
            "1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,"
            "1.6,1.7,1.8,2,2.25,2.5,3,4,6"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=sorted(v157h.OBJECTIVES),
        default="h1_strict",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=157_006)
    parser.add_argument("--seed", type=int, default=1576)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_packet(
    model: v157e.WeightedRidge,
    payload: v157e.UpdatePayload,
    packet: np.ndarray,
    bound_px: float,
    availability_mask: np.ndarray | None = None,
) -> np.ndarray:
    scale = payload.base.scale
    raw = np.column_stack(
        [
            packet,
            packet * scale[:, 0:1],
            packet * scale[:, 1:2],
        ]
    )
    normalized = (raw - model.row_mean) / model.row_scale
    augmented = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    correction = augmented @ model.coefficients
    if availability_mask is not None:
        available = np.asarray(availability_mask, dtype=bool)
        if available.shape != (len(correction),):
            raise ValueError(
                "availability_mask must contain one value per prediction row"
            )
        correction = correction.copy()
        correction[~available] = 0.0
    return payload.base.mean + v157e.bounded_update(correction, bound_px)


def cadence_mask(rows: pd.DataFrame, cadence: int) -> np.ndarray:
    if cadence <= 1:
        return np.ones(len(rows), dtype=bool)
    mask = np.zeros(len(rows), dtype=bool)
    for _, raw_indices in rows.groupby("track_id", sort=False).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        frames = rows.iloc[indices]["frame"].to_numpy(np.int64)
        order = np.argsort(frames)
        ordered = indices[order]
        mask[ordered[::cadence]] = True
    return mask


def missing_mask(rows: pd.DataFrame, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(rows)) >= float(rate)
    first = (
        rows.reset_index()
        .sort_values(["track_id", "frame"])
        .groupby("track_id", sort=False)["index"]
        .first()
        .to_numpy(np.int64)
    )
    mask[first] = True
    return mask


def noisy_normal_score(
    payload: v154.MoviePayload,
    noise_px: float,
    seed: int,
) -> np.ndarray:
    rows = payload.rows.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    observation_noise: dict[tuple[int, int], np.ndarray] = {}
    for track, frame in rows[["track_id", "frame"]].itertuples(index=False):
        for time_index in (int(frame), int(frame) + 1):
            key = (int(track), time_index)
            if key not in observation_noise:
                observation_noise[key] = rng.normal(scale=noise_px, size=2)
    displacement_noise = np.stack(
        [
            observation_noise[(int(track), int(frame) + 1)]
            - observation_noise[(int(track), int(frame))]
            for track, frame in rows[
                ["track_id", "frame"]
            ].itertuples(index=False)
        ]
    )
    noisy_target = payload.target + displacement_noise
    standardized = (noisy_target - payload.mean) / payload.scale
    uniform = np.clip(
        student_t.cdf(
            standardized,
            df=payload.degrees_of_freedom,
        ),
        1e-6,
        1.0 - 1e-6,
    )
    return np.asarray(norm.ppf(uniform), dtype=np.float64)


def noisy_update_packet(
    payload: v157e.UpdatePayload,
    local_scales: list[float],
    noise_px: float,
    seed: int,
) -> np.ndarray:
    changed = v154.MoviePayload(
        movie=payload.base.movie,
        rows=payload.base.rows,
        target=payload.base.target,
        mean=payload.base.mean,
        scale=payload.base.scale,
        degrees_of_freedom=payload.base.degrees_of_freedom,
        normal_score=noisy_normal_score(payload.base, noise_px, seed),
    )
    packet, names, donor = v157e.build_real_update_packet(
        changed,
        local_scales,
    )
    if names != payload.feature_names:
        raise RuntimeError("Noisy packet feature contract changed")
    frames = payload.base.rows["frame"].to_numpy(np.int64)
    if np.any((donor >= 0) & (donor > frames - 1)):
        raise RuntimeError("Noisy packet contains a future donor")
    return packet


def condition_predictions(
    args: argparse.Namespace,
    payload: v157e.UpdatePayload,
    model: v157e.WeightedRidge,
    selection: v157e.Selection,
    local_scales: list[float],
    cadences: list[int],
    missing_rates: list[float],
    noise_values: list[float],
) -> dict[str, np.ndarray]:
    predictions = {
        "no_update": payload.base.mean,
        "real_update_every_1": predict_packet(
            model,
            payload,
            payload.real,
            selection.bound_px,
        ),
        "wrong_cell": predict_packet(
            model,
            payload,
            payload.wrong_cell,
            selection.bound_px,
        ),
        "delay_1frame": predict_packet(
            model,
            payload,
            payload.stale_time,
            selection.bound_px,
        ),
    }
    for cadence in cadences:
        mask = cadence_mask(payload.base.rows, cadence)
        predictions[f"update_every_{cadence}"] = predict_packet(
            model,
            payload,
            payload.real,
            selection.bound_px,
            availability_mask=mask,
        )
    for rate in missing_rates:
        mask = missing_mask(
            payload.base.rows,
            rate,
            int(args.seed)
            + int(payload.base.movie) * 1009
            + int(rate * 10_000),
        )
        predictions[f"missing_{rate:g}"] = predict_packet(
            model,
            payload,
            payload.real,
            selection.bound_px,
            availability_mask=mask,
        )
    for noise_px in noise_values:
        packet = noisy_update_packet(
            payload,
            local_scales,
            noise_px,
            int(args.seed)
            + int(payload.base.movie) * 10_007
            + int(noise_px * 1000),
        )
        predictions[f"tracking_noise_{noise_px:g}px"] = predict_packet(
            model,
            payload,
            packet,
            selection.bound_px,
        )
    return predictions


def probabilistic_row(
    payload: v157e.UpdatePayload,
    prediction: np.ndarray,
    condition: str,
    horizon: int,
    step_scale: np.ndarray,
    calibration_mode: str,
    validation_score: float,
    update_scale_factor: float,
    prior_scale_factor: float,
) -> dict[str, Any]:
    windows = v157e.consecutive_windows(payload.base.rows, horizon)
    target = payload.base.target[windows].sum(axis=1)
    predicted = prediction[windows].sum(axis=1)
    endpoint_scale = np.sqrt(
        np.sum(np.square(step_scale[windows]), axis=1)
    )
    endpoint_scale = np.maximum(endpoint_scale, 1e-3)
    standardized = (target - predicted) / endpoint_scale
    if horizon == 1:
        distribution = student_t
        distribution_kwargs = {"df": payload.base.degrees_of_freedom}
        q50 = float(
            student_t.ppf(0.75, df=payload.base.degrees_of_freedom)
        )
        q90 = float(
            student_t.ppf(0.95, df=payload.base.degrees_of_freedom)
        )
        approximation = "student_t_exact"
    else:
        distribution = norm
        distribution_kwargs = {}
        q50 = float(norm.ppf(0.75))
        q90 = float(norm.ppf(0.95))
        approximation = "independent_moment_normal"
    nll = -float(
        np.mean(
            distribution.logpdf(standardized, **distribution_kwargs)
            - np.log(endpoint_scale)
        )
    )
    absolute = np.abs(target - predicted)
    coverage50 = float(np.mean(absolute <= q50 * endpoint_scale))
    coverage90 = float(np.mean(absolute <= q90 * endpoint_scale))
    uncertainty = endpoint_scale.mean(axis=1)
    error = np.linalg.norm(target - predicted, axis=1)
    correlation = (
        float(np.corrcoef(uncertainty, error)[0, 1])
        if np.std(uncertainty) > EPS and np.std(error) > EPS
        else 0.0
    )
    return {
        "test_movie": payload.base.movie,
        "condition": condition,
        "horizon": horizon,
        "calibration_mode": calibration_mode,
        "nll": nll,
        "coverage_50": coverage50,
        "coverage_90": coverage90,
        "calibration_error": abs(coverage50 - 0.50)
        + abs(coverage90 - 0.90),
        "uncertainty_error_corr": correlation,
        "scale_factor": float(
            np.mean(
                step_scale / np.maximum(payload.base.scale, 1e-6)
            )
        ),
        "update_scale_factor": float(update_scale_factor),
        "prior_scale_factor": float(prior_scale_factor),
        "mean_step_scale": float(np.mean(step_scale)),
        "mean_endpoint_scale": float(np.mean(endpoint_scale)),
        "validation_nll_score": float(validation_score),
        "approximation": approximation,
        "uncertainty_scope": (
            "frozen_v97_scale"
            if calibration_mode == "frozen_base_scale"
            else "frozen_clean_validation_state_aware_scale"
        ),
        "protocol": "strict_fold_local_streaming",
        "corruption_scope": "completed_innovation_packet_only",
    }


def select_uncertainty_factor(
    payload: v157e.UpdatePayload,
    prediction: np.ndarray,
    condition: str,
    factors: list[float],
    weights_by_horizon: dict[int, float],
) -> tuple[float, float]:
    best_factor = 1.0
    best_score = float("inf")
    for factor in factors:
        score = 0.0
        for horizon, weight in weights_by_horizon.items():
            row = probabilistic_row(
                payload,
                prediction,
                condition,
                horizon,
                np.maximum(
                    payload.base.scale * float(factor),
                    1e-3,
                ),
                "validation_candidate",
                np.nan,
                factor,
                factor,
            )
            score += float(weight) * float(row["nll"])
        if score < best_score:
            best_factor = float(factor)
            best_score = float(score)
    return best_factor, best_score


def condition_step_scales(
    args: argparse.Namespace,
    payload: v157e.UpdatePayload,
    model: v157e.WeightedRidge,
    selection: v157e.Selection,
    local_scales: list[float],
    update_factor: float,
    prior_factor: float,
    cadences: list[int],
    missing_rates: list[float],
    noise_values: list[float],
) -> dict[str, np.ndarray]:
    base = np.maximum(payload.base.scale, 1e-3)
    update_scale = base * float(update_factor)
    prior_scale = base * float(prior_factor)
    scales = {
        "no_update": prior_scale,
        "real_update_every_1": update_scale,
        "wrong_cell": prior_scale,
        "delay_1frame": prior_scale,
    }
    for cadence in cadences:
        mask = cadence_mask(payload.base.rows, cadence)
        scales[f"update_every_{cadence}"] = np.where(
            mask[:, None],
            update_scale,
            prior_scale,
        )
    for rate in missing_rates:
        mask = missing_mask(
            payload.base.rows,
            rate,
            int(args.seed)
            + int(payload.base.movie) * 1009
            + int(rate * 10_000),
        )
        scales[f"missing_{rate:g}"] = np.where(
            mask[:, None],
            update_scale,
            prior_scale,
        )
    for noise_px in noise_values:
        draws = []
        for draw in range(int(args.noise_propagation_draws)):
            packet = noisy_update_packet(
                payload,
                local_scales,
                noise_px,
                int(args.seed)
                + int(payload.base.movie) * 10_007
                + int(noise_px * 1000)
                + (draw + 1) * 1_000_003,
            )
            draws.append(
                predict_packet(
                    model,
                    payload,
                    packet,
                    selection.bound_px,
                )
            )
        propagated = np.std(
            np.stack(draws),
            axis=0,
            ddof=1,
        )
        scales[f"tracking_noise_{noise_px:g}px"] = np.sqrt(
            np.square(update_scale) + np.square(propagated)
        )
    return scales


def uncertainty_rows(
    payload: v157e.UpdatePayload,
    predictions: dict[str, np.ndarray],
    state_aware_scales: dict[str, np.ndarray],
    update_factor: float,
    prior_factor: float,
    update_validation_score: float,
    prior_validation_score: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition, prediction in predictions.items():
        validation_score = (
            update_validation_score
            if condition.startswith("tracking_noise_")
            or condition in {"real_update_every_1", "update_every_1"}
            else prior_validation_score
        )
        for horizon in v157e.HORIZONS:
            output.append(
                probabilistic_row(
                    payload,
                    prediction,
                    condition,
                    horizon,
                    payload.base.scale,
                    "frozen_base_scale",
                    validation_score,
                    1.0,
                    1.0,
                )
            )
            output.append(
                probabilistic_row(
                    payload,
                    prediction,
                    condition,
                    horizon,
                    state_aware_scales[condition],
                    "frozen_state_aware_scale",
                    validation_score,
                    update_factor,
                    prior_factor,
                )
            )
    return output


def append_metrics(
    output: list[dict[str, Any]],
    payload: v157e.UpdatePayload,
    prediction: np.ndarray,
    condition: str,
    selection: v157e.Selection,
) -> None:
    rows = v157e.metric_rows(
        payload,
        prediction,
        condition,
        selection,
    )
    for row in rows:
        row["condition"] = condition
        row["control"] = condition
        row["protocol"] = "strict_fold_local_streaming"
        row["corruption_scope"] = "completed_innovation_packet_only"
    output.extend(rows)


def evaluate_fold(
    args: argparse.Namespace,
    test_movie: int,
    seeds: list[int],
    alphas: list[float],
    bounds: list[float],
    local_scales: list[float],
    cadences: list[int],
    missing_rates: list[float],
    noise_values: list[float],
    uncertainty_factors: list[float],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    device = v157e.device_from_cli(args.device)
    replays = [
        v157e.restore_fold_seed(
            args.v102_root,
            test_movie,
            seed,
            device,
        )
        for seed in seeds
    ]
    split_payloads = v157e.student_t_mixture_payloads(replays)
    validation_movies = {
        int(replay.manifest["validation_movie"]) for replay in replays
    }
    train_sets = {
        tuple(replay.manifest["train_movies"]) for replay in replays
    }
    if len(validation_movies) != 1 or len(train_sets) != 1:
        raise RuntimeError(f"Fold split mismatch for movie {test_movie}")
    validation_movie = next(iter(validation_movies))
    train_movies = list(next(iter(train_sets)))
    payloads = {
        movie: v157e.build_update_payload(
            split,
            base,
            local_scales,
            int(args.control_seed) + test_movie * 100_003,
        )
        for movie, (split, base) in split_payloads.items()
    }
    weights_by_horizon, h1_guard_percent = v157h.OBJECTIVES[
        str(args.objective)
    ]
    selection, _ = v157h.select_model(
        payloads,
        train_movies,
        validation_movie,
        weights_by_horizon,
        h1_guard_percent,
        alphas,
        bounds,
    )
    model = v157h.fit_model(
        payloads,
        train_movies + [validation_movie],
        selection.alpha,
        weights_by_horizon,
    )
    calibration_model = v157h.fit_model(
        payloads,
        train_movies,
        selection.alpha,
        weights_by_horizon,
    )
    validation = payloads[validation_movie]
    test = payloads[test_movie]
    validation_predictions = condition_predictions(
        args,
        validation,
        calibration_model,
        selection,
        local_scales,
        cadences,
        missing_rates,
        noise_values,
    )
    test_predictions = condition_predictions(
        args,
        test,
        model,
        selection,
        local_scales,
        cadences,
        missing_rates,
        noise_values,
    )
    update_factor, update_validation_score = select_uncertainty_factor(
        validation,
        validation_predictions["real_update_every_1"],
        "real_update_every_1",
        uncertainty_factors,
        weights_by_horizon,
    )
    prior_factor, prior_validation_score = select_uncertainty_factor(
        validation,
        validation_predictions["no_update"],
        "no_update",
        uncertainty_factors,
        weights_by_horizon,
    )
    state_aware_scales = condition_step_scales(
        args,
        test,
        model,
        selection,
        local_scales,
        update_factor,
        prior_factor,
        cadences,
        missing_rates,
        noise_values,
    )
    output: list[dict[str, Any]] = []
    for condition, prediction in test_predictions.items():
        append_metrics(
            output,
            test,
            prediction,
            condition,
            selection,
        )
    probabilistic = uncertainty_rows(
        test,
        test_predictions,
        state_aware_scales,
        update_factor,
        prior_factor,
        update_validation_score,
        prior_validation_score,
    )
    manifest = []
    for replay in replays:
        record = dict(replay.manifest)
        record["v157f_selection"] = {
            "validation_movie": validation_movie,
            "train_movies": train_movies,
            "test_movie": test_movie,
            "alpha": selection.alpha,
            "bound_px": selection.bound_px,
            "packet": "update_only",
            "objective_name": str(args.objective),
        }
        record["v157f_uncertainty"] = {
            "update_scale_factor_selected_on_clean_validation": (
                update_factor
            ),
            "prior_scale_factor_selected_on_clean_validation": (
                prior_factor
            ),
            "corruption_specific_scale_tuning": False,
            "missingness_policy": (
                "use update scale when innovation is available and "
                "prior scale otherwise"
            ),
            "coordinate_noise_policy": (
                "Monte Carlo propagation through the frozen innovation "
                "packet and bounded correction"
            ),
        }
        manifest.append(record)
    causal = v157e.build_causal_audit(
        payloads,
        test_movie,
        validation_movie,
        train_movies,
    )
    return output, probabilistic, manifest, causal


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["condition", "horizon"], as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            vector_rmse_mean=("vector_rmse", "mean"),
            r2_mean=("r2", "mean"),
            rmse_improvement_percent_mean=(
                "rmse_improvement_percent",
                "mean",
            ),
            movies_improved=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values(["horizon", "component_rmse_mean"])
    )


def aggregate_uncertainty(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["calibration_mode", "condition", "horizon"],
            as_index=False,
        )
        .agg(
            movies=("test_movie", "nunique"),
            nll_mean=("nll", "mean"),
            coverage_50_mean=("coverage_50", "mean"),
            coverage_90_mean=("coverage_90", "mean"),
            calibration_error_mean=("calibration_error", "mean"),
            uncertainty_error_corr_mean=(
                "uncertainty_error_corr",
                "mean",
            ),
            scale_factor_mean=("scale_factor", "mean"),
            update_scale_factor_mean=("update_scale_factor", "mean"),
            prior_scale_factor_mean=("prior_scale_factor", "mean"),
            mean_step_scale_mean=("mean_step_scale", "mean"),
            mean_endpoint_scale_mean=("mean_endpoint_scale", "mean"),
        )
        .sort_values(["horizon", "condition", "calibration_mode"])
    )


def uncertainty_response_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        metrics["calibration_mode"].eq(
            "frozen_state_aware_scale"
        )
        & metrics["horizon"].eq(1)
    ].copy()
    reference = selected[
        selected["condition"].eq("real_update_every_1")
    ][["test_movie", "mean_step_scale"]].rename(
        columns={"mean_step_scale": "real_mean_step_scale"}
    )
    selected = selected.merge(
        reference,
        on="test_movie",
        how="left",
        validate="many_to_one",
    )
    selected["scale_factor_ratio_vs_real"] = (
        selected["mean_step_scale"] / selected["real_mean_step_scale"]
    )
    paired = metrics.pivot_table(
        index=["test_movie", "condition", "horizon"],
        columns="calibration_mode",
        values=["calibration_error", "coverage_90", "nll"],
        aggfunc="first",
    )
    paired.columns = [
        "_".join(column).rstrip("_")
        for column in paired.columns.to_flat_index()
    ]
    paired = paired.reset_index()
    paired["calibration_error_gain"] = (
        paired["calibration_error_frozen_base_scale"]
        - paired["calibration_error_frozen_state_aware_scale"]
    )
    paired["coverage_90_gain"] = (
        paired["coverage_90_frozen_state_aware_scale"]
        - paired["coverage_90_frozen_base_scale"]
    )
    paired["nll_gain"] = (
        paired["nll_frozen_base_scale"]
        - paired["nll_frozen_state_aware_scale"]
    )
    calibration_gain = (
        paired.groupby(["test_movie", "condition"], as_index=False)[
            ["calibration_error_gain", "coverage_90_gain", "nll_gain"]
        ]
        .mean()
        .groupby("condition", as_index=False)
        .agg(
            calibration_error_gain_mean=(
                "calibration_error_gain",
                "mean",
            ),
            coverage_90_gain_mean=("coverage_90_gain", "mean"),
            nll_gain_mean=("nll_gain", "mean"),
        )
    )
    response = (
        selected.groupby("condition", as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            scale_factor_mean=("scale_factor", "mean"),
            scale_factor_ratio_vs_real_mean=(
                "scale_factor_ratio_vs_real",
                "mean",
            ),
            movies_expanded_vs_real=(
                "scale_factor_ratio_vs_real",
                lambda values: int((values > 1.0 + 1e-12).sum()),
            ),
            mean_step_scale=("mean_step_scale", "mean"),
        )
        .merge(
            calibration_gain,
            on="condition",
            how="left",
            validate="one_to_one",
        )
        .sort_values("condition")
    )
    return response


def report_text(
    args: argparse.Namespace,
    aggregate: pd.DataFrame,
    uncertainty_response: pd.DataFrame,
    elapsed: float,
) -> tuple[str, pd.DataFrame]:
    h6 = aggregate[aggregate["horizon"].eq(6)].set_index("condition")
    real = float(
        h6.loc["real_update_every_1", "rmse_improvement_percent_mean"]
    )
    cadence6 = float(
        h6.loc["update_every_6", "rmse_improvement_percent_mean"]
    )
    missing40 = float(
        h6.loc["missing_0.4", "rmse_improvement_percent_mean"]
    )
    delayed = float(
        h6.loc["delay_1frame", "rmse_improvement_percent_mean"]
    )
    wrong = float(
        h6.loc["wrong_cell", "rmse_improvement_percent_mean"]
    )
    decision = (
        "ROBUST_PASS"
        if real >= 1.0
        and cadence6 > 0
        and missing40 > 0
        and real > delayed
        and real > wrong
        else "FRAGILE_OR_FAIL"
    )
    response = uncertainty_response.set_index("condition")
    missing_condition = "missing_0.4"
    noise_conditions = sorted(
        condition
        for condition in response.index
        if condition.startswith("tracking_noise_")
    )
    noise_condition = noise_conditions[-1] if noise_conditions else ""
    missing_ratio = (
        float(
            response.loc[
                missing_condition,
                "scale_factor_ratio_vs_real_mean",
            ]
        )
        if missing_condition in response.index
        else np.nan
    )
    noise_ratio = (
        float(
            response.loc[
                noise_condition,
                "scale_factor_ratio_vs_real_mean",
            ]
        )
        if noise_condition in response.index
        else np.nan
    )
    missing_calibration_gain = (
        float(
            response.loc[
                missing_condition,
                "calibration_error_gain_mean",
            ]
        )
        if missing_condition in response.index
        else np.nan
    )
    noise_calibration_gain = (
        float(
            response.loc[
                noise_condition,
                "calibration_error_gain_mean",
            ]
        )
        if noise_condition in response.index
        else np.nan
    )
    missing_coverage_gain = (
        float(
            response.loc[
                missing_condition,
                "coverage_90_gain_mean",
            ]
        )
        if missing_condition in response.index
        else np.nan
    )
    noise_coverage_gain = (
        float(
            response.loc[
                noise_condition,
                "coverage_90_gain_mean",
            ]
        )
        if noise_condition in response.index
        else np.nan
    )
    uncertainty_decision = (
        "FROZEN_STATE_AWARE_RESPONSE_PASS"
        if missing_ratio > 1.0
        and noise_ratio > 1.0
        and missing_coverage_gain > 0.0
        and noise_coverage_gain > 0.0
        else "UNCERTAINTY_RESPONSE_INCOMPLETE"
    )
    summary = pd.DataFrame(
        [
            {
                "decision": decision,
                "h6_real_gain_percent": real,
                "h6_update_every_6_gain_percent": cadence6,
                "h6_missing_40_gain_percent": missing40,
                "h6_delay_gain_percent": delayed,
                "h6_wrong_cell_gain_percent": wrong,
                "protocol": "strict fold-local streaming/receding h1",
                "corruption_scope": "completed innovation packet only",
                "objective_name": str(args.objective),
                "uncertainty_response_decision": uncertainty_decision,
                "missing_40_scale_ratio_vs_real": missing_ratio,
                "max_noise_condition": noise_condition,
                "max_noise_scale_ratio_vs_real": noise_ratio,
                "missing_40_calibration_error_gain": (
                    missing_calibration_gain
                ),
                "max_noise_calibration_error_gain": noise_calibration_gain,
                "missing_40_coverage_90_gain": missing_coverage_gain,
                "max_noise_coverage_90_gain": noise_coverage_gain,
            }
        ]
    )
    lines = [
        "# v157f Strict Fold-Local Semigroup Stress Audit",
        "",
        f"Decision: **{decision}**",
        "",
        f"- objective: `{args.objective}`",
        f"- real update h6 gain: `{real:+.3f}%`",
        f"- update every 6 frames h6 gain: `{cadence6:+.3f}%`",
        f"- 40% missing updates h6 gain: `{missing40:+.3f}%`",
        f"- coherent one-frame delay h6 gain: `{delayed:+.3f}%`",
        f"- coherent wrong-cell update h6 gain: `{wrong:+.3f}%`",
        f"- uncertainty response: **{uncertainty_decision}**",
        (
            "- frozen state-aware scale ratio, 40% missing vs real: "
            f"`{missing_ratio:.3f}`"
        ),
        (
            f"- frozen state-aware scale ratio, {noise_condition} vs real: "
            f"`{noise_ratio:.3f}`"
        ),
        "",
        "All base predictions were regenerated from matching outer-fold v97 checkpoints.",
        "Frozen base-scale and frozen state-aware uncertainty are both reported.",
        "Only clean real-update and clean no-update scales are selected using a",
        "train-only mean model on the validation movie. No corruption-level scale is",
        "tuned. Missing rows switch to the no-update scale; declared coordinate noise",
        "is propagated through the frozen packet/correction by Monte Carlo.",
        "The outer test movie is never used to select an uncertainty multiplier.",
        "These corruptions affect only the completed-innovation packet. Full coordinate",
        "corruption before velocity/history/base reconstruction remains the v103 result.",
        "",
        "h2/h4/h6 are streaming cumulative metrics, not open-loop forecasts.",
        "",
        f"Elapsed: `{elapsed / 60.0:.2f}` minutes.",
        f"Output: `{args.out_dir}`",
    ]
    return "\n".join(lines) + "\n", summary


def main() -> None:
    args = parse_args()
    if int(args.noise_propagation_draws) < 2:
        raise ValueError("--noise-propagation-draws must be at least 2")
    started = time.time()
    args.v102_root = args.v102_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = v157e.parse_csv_ints(args.movies)
    seeds = v157e.parse_csv_ints(args.seeds)
    alphas = v157e.parse_csv_floats(args.alphas)
    bounds = v157e.parse_csv_floats(args.bounds_px)
    local_scales = v157e.parse_csv_floats(args.local_scales_px)
    cadences = v157e.parse_csv_ints(args.cadences)
    missing_rates = v157e.parse_csv_floats(args.missing_rates)
    noise_values = v157e.parse_csv_floats(args.noise_px)
    uncertainty_factors = v157e.parse_csv_floats(
        args.uncertainty_scale_grid
    )

    all_metrics: list[dict[str, Any]] = []
    all_uncertainty: list[dict[str, Any]] = []
    all_manifests: list[dict[str, Any]] = []
    all_causal: list[pd.DataFrame] = []
    for movie in movies:
        print(f"[v157f] strict outer movie {movie}", flush=True)
        metrics, uncertainty, manifests, causal = evaluate_fold(
            args,
            movie,
            seeds,
            alphas,
            bounds,
            local_scales,
            cadences,
            missing_rates,
            noise_values,
            uncertainty_factors,
        )
        all_metrics.extend(metrics)
        all_uncertainty.extend(uncertainty)
        all_manifests.extend(manifests)
        all_causal.append(causal)
    metrics_frame = pd.DataFrame(all_metrics)
    aggregate = aggregate_metrics(metrics_frame)
    uncertainty_frame = pd.DataFrame(all_uncertainty)
    uncertainty_aggregate = aggregate_uncertainty(uncertainty_frame)
    uncertainty_response = uncertainty_response_table(uncertainty_frame)
    causal = pd.concat(all_causal, ignore_index=True)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Future donor found in real packet")
    if int(causal["stale_future_or_nonstale_violations"].sum()) != 0:
        raise RuntimeError("Invalid donor found in delayed packet")
    if not bool(causal["coherent_wrong_packet"].all()):
        raise RuntimeError("Wrong-cell packet coherence failed")

    elapsed = time.time() - started
    report, summary = report_text(
        args,
        aggregate,
        uncertainty_response,
        elapsed,
    )
    metrics_frame.to_csv(
        args.out_dir / "v157f_stress_metrics.csv",
        index=False,
    )
    aggregate.to_csv(
        args.out_dir / "v157f_stress_aggregate.csv",
        index=False,
    )
    uncertainty_frame.to_csv(
        args.out_dir / "v157f_uncertainty_metrics.csv",
        index=False,
    )
    uncertainty_aggregate.to_csv(
        args.out_dir / "v157f_uncertainty_aggregate.csv",
        index=False,
    )
    uncertainty_response.to_csv(
        args.out_dir / "v157f_uncertainty_response.csv",
        index=False,
    )
    causal.to_csv(args.out_dir / "v157f_causal_audit.csv", index=False)
    summary.to_csv(args.out_dir / "v157f_summary.csv", index=False)
    (args.out_dir / "v157f_status_report.md").write_text(
        report,
        encoding="utf-8",
    )
    (args.out_dir / "v157f_seed_replay_manifest.json").write_text(
        json.dumps(v157e.finite(all_manifests), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        **vars(args),
        "source_sha256": file_sha256(Path(__file__)),
        "protocol": "strict fold-local streaming/receding h1",
        "corruption_scope": "completed innovation packet only",
        "objective_name": str(args.objective),
        "uncertainty_policy": {
            "selection_data": "clean validation movie only",
            "selected_scales": ["real_update", "no_update_prior"],
            "corruption_specific_scale_tuning": False,
            "missingness": "rowwise update/prior scale switch",
            "coordinate_noise": (
                "Monte Carlo through frozen packet and bounded correction"
            ),
            "coordinate_noise_draws": int(args.noise_propagation_draws),
        },
        "full_observation_reference": (
            "outputs/v97_full_observation_corruption_v103_"
            "full_direct_seed42_2026-07-21"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(v157e.finite(run_manifest), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
