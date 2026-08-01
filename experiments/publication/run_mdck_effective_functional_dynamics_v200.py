#!/usr/bin/env python3
"""Finite-step and rollout audit of the v198 effective functional.

The previous audit checked only first-order alignment.  This runner fits each
operator in whole-island outer folds, reconstructs full displacement fields,
and measures actual finite-step functional changes plus 1/2/3/6-step velocity
rollouts from causal initial states.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mdck_effective_potential_audit_v198 as v198  # noqa: E402
import run_mdck_equivariant_field_law_v197 as v197  # noqa: E402

from lit_cell_forecasting.equivariant_field_law import (  # noqa: E402
    EPS,
    VectorOperatorModel,
    vector_gradient,
)


DEFAULT_DERIVED = (
    ROOT
    / "outputs/mdck_equivariant_field_law_v197_smoke_2026-07-30"
    / "v197_operator_samples.pkl.gz"
)
DEFAULT_DATA = ROOT / "data/external/mdck_force_motion/extracted"
DEFAULT_OUT = ROOT / "outputs/mdck_effective_functional_dynamics_v200"
HORIZONS = (1, 2, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-cache", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--thresholds", default="0,0.01,0.03")
    parser.add_argument("--smoothing-sigma", type=float, default=0.75)
    parser.add_argument("--boundary-decay-um", type=float, default=75.0)
    parser.add_argument("--damping-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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


def physical_coefficients(model: VectorOperatorModel) -> dict[str, float]:
    return {
        name: float(coefficient / max(scale, EPS))
        for name, coefficient, scale in zip(
            model.term_names,
            model.coefficients,
            model.term_scales,
            strict=True,
        )
    }


def potential_parameters(model: VectorOperatorModel) -> dict[str, float]:
    coefficient = physical_coefficients(model)
    return {
        "r": 1.0 - coefficient.get("u_prev", 0.0),
        "k_transverse": coefficient.get("lap_u", 0.0),
        "k_longitudinal": coefficient.get("grad_div_u", 0.0),
        "g": -coefficient.get("cubic_u", 0.0),
    }


def effective_functional(
    field: np.ndarray,
    mask: np.ndarray,
    spacing: float,
    parameters: Mapping[str, float],
) -> float:
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    gradient = vector_gradient(field, spacing)
    divergence = gradient[..., 0, 0] + gradient[..., 1, 1]
    norm2 = np.sum(np.square(field), axis=-1)
    gradient2 = np.sum(np.square(gradient), axis=(-1, -2))
    density = (
        0.5 * parameters["r"] * norm2
        + 0.25 * parameters["g"] * np.square(norm2)
        + 0.5 * parameters["k_transverse"] * gradient2
        + 0.5 * parameters["k_longitudinal"] * np.square(divergence)
    )
    valid = mask & np.isfinite(density)
    return float(np.mean(density[valid])) if np.any(valid) else np.nan


def library(
    innovation: np.ndarray,
    velocity: np.ndarray,
    boundary: np.ndarray,
    *,
    spacing: float,
    sigma: float,
    decay: float,
) -> dict[str, np.ndarray]:
    return v197.build_equivariant_library(
        innovation,
        velocity,
        boundary,
        spacing=spacing,
        smoothing_sigma=sigma,
        boundary_decay=decay,
    )


def predict_field(
    model: VectorOperatorModel,
    innovation: np.ndarray,
    velocity: np.ndarray,
    boundary: np.ndarray,
    *,
    spacing: float,
    sigma: float,
    decay: float,
) -> np.ndarray:
    terms = library(
        innovation,
        velocity,
        boundary,
        spacing=spacing,
        sigma=sigma,
        decay=decay,
    )
    return model.predict({name: terms[name] for name in model.term_names})


def fit_outer_models(
    table: pd.DataFrame,
    outer_group: str,
    *,
    alphas: Sequence[float],
    thresholds: Sequence[float],
) -> tuple[str, dict[str, VectorOperatorModel], dict[str, float]]:
    groups = table["group"].astype(str).to_numpy()
    unique = sorted(np.unique(groups))
    validation = v197.choose_validation_group(table, unique, outer_group)
    train_index = np.flatnonzero(
        (groups != outer_group) & (groups != validation)
    )
    validation_index = np.flatnonzero(groups == validation)
    target = table[["innovation_x", "innovation_y"]].to_numpy(np.float64)
    terms = v197.vector_terms(table)
    potential, _, potential_alpha = v198.tune_constrained(
        terms,
        target,
        v198.POTENTIAL_TERMS,
        train_index,
        validation_index,
        alphas,
    )
    models: dict[str, VectorOperatorModel] = {"potential": potential}
    tuning: dict[str, float] = {"potential_alpha": potential_alpha}
    for name, names in {
        "scalar_memory": ("u_prev",),
        "advective": v198.ADVECTIVE_TERMS,
    }.items():
        model, selected = v197.tune_and_fit(
            terms,
            target,
            names,
            train_index,
            validation_index,
            alphas=alphas,
            thresholds=thresholds,
        )
        models[name] = model
        tuning[f"{name}_alpha"] = selected["alpha"]
        tuning[f"{name}_threshold"] = selected["threshold"]
    return validation, models, tuning


def raw_island(
    group: str,
    islands: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    condition = group.split("/", 1)[0]
    cell_x, cell_y, target_x, target_y, domain, target_start = (
        v197.aligned_displacement_fields(condition, islands[group])
    )
    spacing = float(np.median(np.diff(cell_x[0, :]))) * v197.PIXEL_SIZE_UM
    return cell_x, cell_y, target_x, target_y, domain, target_start, spacing


def evaluate_outer(
    table: pd.DataFrame,
    outer_group: str,
    islands: Mapping[str, Path],
    models: Mapping[str, VectorOperatorModel],
    validation_group: str,
    tuning: Mapping[str, float],
    *,
    sigma: float,
    decay: float,
    damping_grid: Sequence[float],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    (
        cell_x,
        cell_y,
        target_x,
        target_y,
        domain,
        target_start,
        spacing,
    ) = raw_island(outer_group, islands)
    condition = outer_group.split("/", 1)[0]
    frame_values = sorted(
        int(value)
        for value in table.loc[table["group"].eq(outer_group), "frame"].unique()
    )
    max_frame = min(max(frame_values), target_x.shape[-1] - 1)
    potential = models["potential"]
    parameters = potential_parameters(potential)
    rng = np.random.default_rng(seed)
    shuffled_frames = rng.permutation(frame_values)
    finite_records: list[dict[str, Any]] = []
    rollout_records: list[dict[str, Any]] = []
    damping_records: list[dict[str, Any]] = []

    # Damping is selected on the held-out validation island, never on outer.
    validation_rows = table[table["group"].eq(validation_group)]
    validation_terms = v197.vector_terms(validation_rows)
    validation_target = validation_rows[
        ["innovation_x", "innovation_y"]
    ].to_numpy(np.float64)
    validation_previous = validation_terms["u_prev"]
    raw_validation = potential.predict(
        {name: validation_terms[name] for name in potential.term_names}
    )
    damping_scores = []
    for damping in damping_grid:
        candidate = validation_previous + float(damping) * (
            raw_validation - validation_previous
        )
        score = float(np.sqrt(np.mean(np.square(candidate - validation_target))))
        damping_scores.append((score, float(damping)))
        damping_records.append(
            {
                "outer_group": outer_group,
                "validation_group": validation_group,
                "damping": float(damping),
                "validation_innovation_rmse": score,
            }
        )
    selected_damping = min(damping_scores)[1]

    for index, frame in enumerate(frame_values):
        if frame < 2 or frame > max_frame:
            continue
        issue = target_start + frame
        velocity_previous = np.stack(
            [target_x[..., frame - 1], target_y[..., frame - 1]],
            axis=-1,
        )
        innovation_previous = velocity_previous - np.stack(
            [target_x[..., frame - 2], target_y[..., frame - 2]],
            axis=-1,
        )
        innovation_observed = np.stack(
            [target_x[..., frame], target_y[..., frame]],
            axis=-1,
        ) - velocity_previous
        domain_frame = np.asarray(domain[min(issue, len(domain) - 1)]) > 0
        boundary_px = distance_transform_edt(domain_frame)
        boundary = (
            v197.sample_image(boundary_px, cell_x, cell_y) * v197.PIXEL_SIZE_UM
        )
        mask = v197.sample_image(domain_frame, cell_x, cell_y) > 0
        raw_prediction = predict_field(
            potential,
            innovation_previous,
            velocity_previous,
            boundary,
            spacing=spacing,
            sigma=sigma,
            decay=decay,
        )
        damped_prediction = innovation_previous + selected_damping * (
            raw_prediction - innovation_previous
        )
        shuffled_frame = int(shuffled_frames[index % len(shuffled_frames)])
        shuffled_observed = np.stack(
            [target_x[..., shuffled_frame], target_y[..., shuffled_frame]],
            axis=-1,
        ) - np.stack(
            [
                target_x[..., max(shuffled_frame - 1, 0)],
                target_y[..., max(shuffled_frame - 1, 0)],
            ],
            axis=-1,
        )
        initial_f = effective_functional(
            innovation_previous,
            mask,
            spacing,
            parameters,
        )
        for state_name, state in {
            "observed": innovation_observed,
            "time_shuffled_observed": shuffled_observed,
            "potential_raw": raw_prediction,
            "potential_damped": damped_prediction,
        }.items():
            final_f = effective_functional(state, mask, spacing, parameters)
            finite_records.append(
                {
                    "outer_group": outer_group,
                    "condition": condition,
                    "validation_group": validation_group,
                    "frame": frame,
                    "state": state_name,
                    "selected_damping": selected_damping,
                    "functional_initial": initial_f,
                    "functional_final": final_f,
                    "functional_delta": final_f - initial_f,
                    "functional_decreased": final_f < initial_f,
                    **parameters,
                    **tuning,
                }
            )

        for model_name, model in models.items():
            for horizon in HORIZONS:
                if frame + horizon - 1 >= target_x.shape[-1]:
                    continue
                predicted_velocity = velocity_previous.copy()
                predicted_innovation = innovation_previous.copy()
                for _ in range(horizon):
                    next_innovation = predict_field(
                        model,
                        predicted_innovation,
                        predicted_velocity,
                        boundary,
                        spacing=spacing,
                        sigma=sigma,
                        decay=decay,
                    )
                    if model_name == "potential":
                        next_innovation = predicted_innovation + selected_damping * (
                            next_innovation - predicted_innovation
                        )
                    predicted_velocity = predicted_velocity + next_innovation
                    predicted_innovation = next_innovation
                target_velocity = np.stack(
                    [
                        target_x[..., frame + horizon - 1],
                        target_y[..., frame + horizon - 1],
                    ],
                    axis=-1,
                )
                error = predicted_velocity[mask] - target_velocity[mask]
                cv_error = velocity_previous[mask] - target_velocity[mask]
                rollout_records.append(
                    {
                        "outer_group": outer_group,
                        "condition": condition,
                        "validation_group": validation_group,
                        "start_frame": frame,
                        "model": model_name,
                        "horizon": horizon,
                        "selected_damping": selected_damping,
                        "component_rmse": float(
                            np.sqrt(np.mean(np.square(error)))
                        ),
                        "cv_component_rmse": float(
                            np.sqrt(np.mean(np.square(cv_error)))
                        ),
                        "gain_vs_cv_percent": 100.0
                        * (
                            np.sqrt(np.mean(np.square(cv_error)))
                            - np.sqrt(np.mean(np.square(error)))
                        )
                        / max(np.sqrt(np.mean(np.square(cv_error))), EPS),
                    }
                )
    return finite_records, rollout_records, damping_records


def summarize_finite(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("state", as_index=False)
        .agg(
            islands=("outer_group", "nunique"),
            frames=("frame", "count"),
            functional_delta_mean=("functional_delta", "mean"),
            functional_delta_median=("functional_delta", "median"),
            decrease_fraction=("functional_decreased", "mean"),
        )
        .sort_values("state")
    )


def summarize_rollout(frame: pd.DataFrame) -> pd.DataFrame:
    per_island = (
        frame.groupby(["outer_group", "model", "horizon"], as_index=False)
        .agg(
            component_rmse=("component_rmse", "mean"),
            cv_component_rmse=("cv_component_rmse", "mean"),
        )
    )
    per_island["gain_vs_cv_percent"] = 100.0 * (
        per_island["cv_component_rmse"] - per_island["component_rmse"]
    ) / per_island["cv_component_rmse"].clip(lower=EPS)
    return (
        per_island.groupby(["model", "horizon"], as_index=False)
        .agg(
            islands=("outer_group", "nunique"),
            component_rmse_macro=("component_rmse", "mean"),
            cv_component_rmse_macro=("cv_component_rmse", "mean"),
            gain_vs_cv_percent_mean=("gain_vs_cv_percent", "mean"),
            positive_islands=(
                "gain_vs_cv_percent",
                lambda value: int(np.sum(value > 0)),
            ),
        )
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.derived_cache, "rb") as handle:
        table = pickle.load(handle).reset_index(drop=True)
    groups = sorted(table["group"].unique())
    if args.smoke:
        groups = groups[:2]
    islands = v197.discover_islands(args.data_root)
    alphas = parse_floats(args.alphas)
    thresholds = parse_floats(args.thresholds)
    damping_grid = parse_floats(args.damping_grid)
    finite_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    damping_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for fold_index, outer_group in enumerate(groups):
        validation, models, tuning = fit_outer_models(
            table,
            outer_group,
            alphas=alphas,
            thresholds=thresholds,
        )
        for model_name, model in models.items():
            for name, value in physical_coefficients(model).items():
                coefficient_rows.append(
                    {
                        "outer_group": outer_group,
                        "validation_group": validation,
                        "model": model_name,
                        "term": name,
                        "coefficient_physical": value,
                    }
                )
        finite_result, rollout_result, damping_result = evaluate_outer(
            table,
            outer_group,
            islands,
            models,
            validation,
            tuning,
            sigma=args.smoothing_sigma,
            decay=args.boundary_decay_um,
            damping_grid=damping_grid,
            seed=args.seed + 1009 * fold_index,
        )
        finite_rows.extend(finite_result)
        rollout_rows.extend(rollout_result)
        damping_rows.extend(damping_result)
        print(f"[v200] outer={outer_group} complete", flush=True)

    finite_frame = pd.DataFrame(finite_rows)
    rollout_frame = pd.DataFrame(rollout_rows)
    damping_frame = pd.DataFrame(damping_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    finite_summary = summarize_finite(finite_frame)
    rollout_summary = summarize_rollout(rollout_frame)
    finite_frame.to_csv(
        args.out_dir / "v200_finite_functional_frames.csv",
        index=False,
    )
    finite_summary.to_csv(
        args.out_dir / "v200_finite_functional_summary.csv",
        index=False,
    )
    rollout_frame.to_csv(
        args.out_dir / "v200_field_rollout_frames.csv",
        index=False,
    )
    rollout_summary.to_csv(
        args.out_dir / "v200_field_rollout_summary.csv",
        index=False,
    )
    damping_frame.to_csv(
        args.out_dir / "v200_damping_validation.csv",
        index=False,
    )
    coefficients.to_csv(
        args.out_dir / "v200_outer_coefficients.csv",
        index=False,
    )

    finite_index = finite_summary.set_index("state")
    predicted_pass = bool(
        finite_index.loc["potential_damped", "decrease_fraction"] >= 0.75
    )
    control_pass = bool(
        finite_index.loc["observed", "decrease_fraction"]
        > finite_index.loc["time_shuffled_observed", "decrease_fraction"]
    )
    potential_h1 = rollout_summary[
        rollout_summary["model"].eq("potential")
        & rollout_summary["horizon"].eq(1)
    ]
    rollout_pass = bool(
        len(potential_h1)
        and potential_h1.iloc[0]["gain_vs_cv_percent_mean"] > 0
    )
    decision = predicted_pass and control_pass and rollout_pass
    report = [
        "# v200 Finite Effective-Functional Dynamics",
        "",
        f"Decision: **{'PASS' if decision else 'FAIL'}**",
        "",
        "## Finite functional changes",
        "",
        finite_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Causal full-field rollouts",
        "",
        rollout_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The functional remains a kinematic effective functional. Finite",
        "decrease and rollout skill do not turn it into measured mechanical or",
        "thermodynamic energy.",
    ]
    (args.out_dir / "v200_functional_dynamics_decision_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "elapsed_minutes": (time.time() - started) / 60.0,
        "groups": groups,
        "outer_island_count": len(groups),
        "future_feature_count": 0,
        "damping_selected_on_validation_only": True,
        "boundary_frozen_at_issue_time_during_rollout": True,
        "decision": "pass" if decision else "fail",
        "interpretation": "kinematic_effective_functional_not_physical_energy",
    }
    (args.out_dir / "v200_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
