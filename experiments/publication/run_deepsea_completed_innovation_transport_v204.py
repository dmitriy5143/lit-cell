#!/usr/bin/env python3
"""External DeepSea test of completed-innovation transport.

The base forecaster is an ensemble of independently trained v97-direct
checkpoints. The transport law sees only innovations completed by the
previous observed frame. Hyperparameters are selected on validation movies,
then the linear transport operator is refitted on train plus validation and
evaluated once on the frozen test movies.

All transport geometry and errors are expressed in per-video cell-diameter
units. Spatial coordinates are stored internally in centi-diameters so the
unchanged v157 feature builder can use integer scale labels without merging
fractional scales.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h  # noqa: E402
import run_lachance_joint_graph_copula_v154 as v154  # noqa: E402
import run_lachance_streaming_transport_confirmation_v160 as v160  # noqa: E402


EPS = 1e-8
PRIMARY_OBJECTIVE = "h1_strict"
PRIMARY_PACKET = "full"


@dataclass(frozen=True)
class Selection:
    objective: str
    packet: str
    alpha: float
    bound_cell_diameter: float
    validation_score: float
    validation_h1_gain_pct: float
    validation_h6_gain_pct: float


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_floats(value: str) -> list[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def parse_strings(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def dimensionless_payload(
    payload: v154.MoviePayload,
    source_unit: str,
) -> v154.MoviePayload:
    rows = payload.rows.reset_index(drop=True).copy()
    diameter = rows.reference_diameter_px.to_numpy(np.float64)
    if not np.all(np.isfinite(diameter) & (diameter > 0)):
        raise RuntimeError(f"Invalid cell diameter for movie {payload.movie}")
    reference = float(np.median(diameter))
    if np.max(np.abs(diameter - reference)) > max(reference * 1e-6, 1e-6):
        raise RuntimeError(
            f"Reference diameter changes within movie {payload.movie}"
        )
    if source_unit == "pixel":
        position_factor = 100.0 / reference
        displacement_factor = 1.0 / reference
    elif source_unit == "cell_diameter":
        position_factor = 100.0
        displacement_factor = 1.0
    else:
        raise ValueError(f"Unsupported source coordinate unit: {source_unit}")
    for column in ("x_px", "y_px"):
        rows[column] = rows[column].to_numpy(np.float64) * position_factor
    for column in ("dx_px", "dy_px"):
        if column in rows:
            rows[column] = (
                rows[column].to_numpy(np.float64) * displacement_factor
            )
    rows["transport_reference_diameter_px"] = reference
    rows["transport_position_unit"] = "centi_cell_diameter"
    return v154.MoviePayload(
        movie=payload.movie,
        rows=rows,
        target=np.asarray(payload.target, dtype=np.float64)
        * displacement_factor,
        mean=np.asarray(payload.mean, dtype=np.float64)
        * displacement_factor,
        scale=np.asarray(payload.scale, dtype=np.float64)
        * displacement_factor,
        degrees_of_freedom=float(payload.degrees_of_freedom),
        normal_score=np.asarray(payload.normal_score, dtype=np.float64),
    )


def split_movies(
    payloads: dict[int, v157e.UpdatePayload],
) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for movie, payload in payloads.items():
        output[payload.split].append(int(movie))
    for split in output:
        output[split].sort()
        if not output[split]:
            raise RuntimeError(f"No movies in split {split}")
    return output


def fit_model(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    alpha: float,
    weights: dict[int, float],
) -> v157e.WeightedRidge:
    return v157h.fit_model(payloads, movies, alpha, weights)


def predict(
    model: v157e.WeightedRidge,
    payload: v157e.UpdatePayload,
    control: str,
    bound: float,
) -> np.ndarray:
    correction = v157e.predict_ridge(model, payload, control)
    return payload.base.mean + v157e.bounded_update(correction, bound)


def select_configuration(
    payloads: dict[int, v157e.UpdatePayload],
    train_movies: list[int],
    validation_movies: list[int],
    objective: str,
    packet: str,
    alphas: list[float],
    bounds: list[float],
) -> tuple[Selection, pd.DataFrame]:
    weights, h1_guard = v157h.OBJECTIVES[objective]
    masked = {
        movie: v160.mask_packet(payload, packet)
        for movie, payload in payloads.items()
    }
    statistics = v160.ridge_statistics(masked, train_movies, weights)
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        model = v160.solve_ridge(statistics, alpha)
        for bound in bounds:
            movie_scores: list[float] = []
            h1_gains: list[float] = []
            h6_gains: list[float] = []
            for movie in validation_movies:
                validation = masked[movie]
                prediction = predict(model, validation, "real", bound)
                metrics = v157e.metric_rows(
                    validation,
                    prediction,
                    "validation_real",
                    None,
                )
                movie_scores.append(v157h.score_metrics(metrics, weights))
                indexed = {int(row["horizon"]): row for row in metrics}
                h1_gains.append(float(indexed[1]["rmse_improvement_percent"]))
                h6_gains.append(float(indexed[6]["rmse_improvement_percent"]))
            records.append(
                {
                    "objective": objective,
                    "packet": packet,
                    "alpha": alpha,
                    "bound_cell_diameter": bound,
                    "validation_score": float(np.mean(movie_scores)),
                    "validation_h1_gain_pct": float(np.mean(h1_gains)),
                    "validation_h6_gain_pct": float(np.mean(h6_gains)),
                    "validation_positive_h6_movies": int(
                        np.sum(np.asarray(h6_gains) > 0)
                    ),
                    "validation_movies": len(validation_movies),
                }
            )
    grid = pd.DataFrame(records)
    eligible = grid[
        grid.validation_h1_gain_pct.ge(-float(h1_guard))
    ]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        [
            "validation_score",
            "validation_h1_gain_pct",
            "bound_cell_diameter",
            "alpha",
        ],
        ascending=[True, False, True, True],
    ).iloc[0]
    selection = Selection(
        objective=objective,
        packet=packet,
        alpha=float(best.alpha),
        bound_cell_diameter=float(best.bound_cell_diameter),
        validation_score=float(best.validation_score),
        validation_h1_gain_pct=float(best.validation_h1_gain_pct),
        validation_h6_gain_pct=float(best.validation_h6_gain_pct),
    )
    grid["selected"] = (
        grid.alpha.eq(selection.alpha)
        & grid.bound_cell_diameter.eq(selection.bound_cell_diameter)
    )
    return selection, grid


def evaluate(
    payloads: dict[int, v157e.UpdatePayload],
    split: dict[str, list[int]],
    selections: list[Selection],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    fit_movies = split["train"] + split["validation"]
    for selection in selections:
        weights, _ = v157h.OBJECTIVES[selection.objective]
        masked = {
            movie: v160.mask_packet(payload, selection.packet)
            for movie, payload in payloads.items()
        }
        model = fit_model(masked, fit_movies, selection.alpha, weights)
        for movie in split["test"]:
            payload = masked[movie]
            family = str(payload.base.rows.family.iloc[0])
            video = str(payload.base.rows.video.iloc[0])
            for control in ("real", "wrong_cell", "stale_time"):
                prediction = predict(
                    model,
                    payload,
                    control,
                    selection.bound_cell_diameter,
                )
                metrics = v157e.metric_rows(payload, prediction, control, None)
                for row in metrics:
                    row.update(
                        {
                            "objective_name": selection.objective,
                            "packet_name": selection.packet,
                            "family": family,
                            "video": video,
                            "selected_alpha": selection.alpha,
                            "selected_bound_cell_diameter": (
                                selection.bound_cell_diameter
                            ),
                            "validation_score": selection.validation_score,
                            "metric_unit": "cell_diameter",
                        }
                    )
                records.extend(metrics)
            baseline = v157e.metric_rows(
                payload,
                payload.base.mean,
                "no_transport",
                None,
            )
            for row in baseline:
                row.update(
                    {
                        "objective_name": selection.objective,
                        "packet_name": selection.packet,
                        "family": family,
                        "video": video,
                        "selected_alpha": selection.alpha,
                        "selected_bound_cell_diameter": (
                            selection.bound_cell_diameter
                        ),
                        "validation_score": selection.validation_score,
                        "metric_unit": "cell_diameter",
                    }
                )
            records.extend(baseline)
    return pd.DataFrame(records)


def bootstrap_gain(
    metrics: pd.DataFrame,
    objective: str,
    packet: str,
    control: str,
    horizon: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    selected = metrics[
        metrics.objective_name.eq(objective)
        & metrics.packet_name.eq(packet)
        & metrics.control.eq(control)
        & metrics.horizon.eq(horizon)
    ].sort_values("test_movie")
    if len(selected) < 2:
        raise RuntimeError("At least two test movies are required")
    baseline = selected.baseline_component_rmse.to_numpy(float)
    candidate = selected.component_rmse.to_numpy(float)
    gain = 100.0 * (
        float(np.mean(baseline)) - float(np.mean(candidate))
    ) / max(float(np.mean(baseline)), EPS)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(selected), size=len(selected))
        baseline_mean = float(np.mean(baseline[indices]))
        candidate_mean = float(np.mean(candidate[indices]))
        samples[repeat] = 100.0 * (
            baseline_mean - candidate_mean
        ) / max(baseline_mean, EPS)
    differences = baseline - candidate
    return {
        "objective_name": objective,
        "packet_name": packet,
        "control": control,
        "horizon": horizon,
        "movies": len(selected),
        "movie_macro_baseline_rmse": float(np.mean(baseline)),
        "movie_macro_component_rmse": float(np.mean(candidate)),
        "movie_macro_gain_pct": gain,
        "positive_movies": int(np.sum(differences > 0)),
        "bootstrap_ci_low": float(np.percentile(samples, 2.5)),
        "bootstrap_ci_high": float(np.percentile(samples, 97.5)),
        "exact_sign_flip_p": v157e.exact_sign_flip_pvalue(differences),
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = v160.parse_checkpoints(args.checkpoints)
    device = v157e.device_from_cli(args.device)
    cache_contract_path = args.cache_dir / "final_native" / "contract.json"
    cache_contract = json.loads(cache_contract_path.read_text(encoding="utf-8"))
    source_unit = str(cache_contract["coordinate_unit"])
    replays = [
        v160.restore_checkpoint(checkpoints[seed], args.cache_dir, device)
        for seed in sorted(checkpoints)
    ]
    raw_payloads = v160.mixture_payloads(replays)
    normalized = {
        movie: (split, dimensionless_payload(payload, source_unit))
        for movie, (split, payload) in raw_payloads.items()
    }
    local_scales = parse_floats(args.local_scales_centi_diameter)
    payloads = {
        movie: v157e.build_update_payload(
            split,
            payload,
            local_scales,
            args.control_seed + movie * 100_003,
        )
        for movie, (split, payload) in normalized.items()
    }
    splits = split_movies(payloads)
    selections: list[Selection] = []
    grids: list[pd.DataFrame] = []
    for objective in parse_strings(args.objectives):
        if objective not in v157h.OBJECTIVES:
            raise ValueError(f"Unknown objective {objective}")
        for packet in parse_strings(args.packets):
            selection, grid = select_configuration(
                payloads,
                splits["train"],
                splits["validation"],
                objective,
                packet,
                parse_floats(args.alphas),
                parse_floats(args.bounds_cell_diameter),
            )
            selections.append(selection)
            grids.append(grid)
    validation_grid = pd.concat(grids, ignore_index=True)
    metrics = evaluate(payloads, splits, selections)
    aggregate_rows: list[dict[str, Any]] = []
    for selection in selections:
        for control in ("real", "wrong_cell", "stale_time"):
            for horizon in v157e.HORIZONS:
                aggregate_rows.append(
                    bootstrap_gain(
                        metrics,
                        selection.objective,
                        selection.packet,
                        control,
                        horizon,
                        args.bootstrap_repeats,
                        args.control_seed + horizon,
                    )
                )
    aggregate = pd.DataFrame(aggregate_rows)
    primary = aggregate[
        aggregate.objective_name.eq(PRIMARY_OBJECTIVE)
        & aggregate.packet_name.eq(PRIMARY_PACKET)
        & aggregate.control.eq("real")
    ].set_index("horizon")
    wrong = aggregate[
        aggregate.objective_name.eq(PRIMARY_OBJECTIVE)
        & aggregate.packet_name.eq(PRIMARY_PACKET)
        & aggregate.control.eq("wrong_cell")
    ].set_index("horizon")
    stale = aggregate[
        aggregate.objective_name.eq(PRIMARY_OBJECTIVE)
        & aggregate.packet_name.eq(PRIMARY_PACKET)
        & aggregate.control.eq("stale_time")
    ].set_index("horizon")
    h1_gain = float(primary.loc[1, "movie_macro_gain_pct"])
    h6_gain = float(primary.loc[6, "movie_macro_gain_pct"])
    controls_pass = bool(
        float(primary.loc[6, "movie_macro_component_rmse"])
        < float(wrong.loc[6, "movie_macro_component_rmse"])
        and float(primary.loc[6, "movie_macro_component_rmse"])
        < float(stale.loc[6, "movie_macro_component_rmse"])
    )
    decision = {
        "primary_objective": PRIMARY_OBJECTIVE,
        "primary_packet": PRIMARY_PACKET,
        "h1_gain_pct": h1_gain,
        "h6_gain_pct": h6_gain,
        "h6_positive_movies": int(primary.loc[6, "positive_movies"]),
        "h6_movies": int(primary.loc[6, "movies"]),
        "h6_bootstrap_ci_low": float(primary.loc[6, "bootstrap_ci_low"]),
        "real_beats_wrong_and_stale": controls_pass,
        "external_transport_pass": bool(
            h6_gain >= 3.0
            and h1_gain >= -0.5
            and int(primary.loc[6, "positive_movies"])
            > int(primary.loc[6, "movies"]) / 2
            and controls_pass
        ),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    pd.DataFrame([asdict(selection) for selection in selections]).to_csv(
        args.out_dir / "v204_transport_selections.csv",
        index=False,
    )
    validation_grid.to_csv(
        args.out_dir / "v204_transport_validation_grid.csv",
        index=False,
    )
    metrics.to_csv(args.out_dir / "v204_transport_movie_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "v204_transport_aggregate.csv", index=False)
    contract = {
        "cache_dir": str(args.cache_dir.resolve()),
        "source_coordinate_unit": source_unit,
        "checkpoints": {
            str(seed): str(path.resolve())
            for seed, path in sorted(checkpoints.items())
        },
        "split_movies": splits,
        "target_unit": "cell_diameter",
        "spatial_feature_unit": "centi_cell_diameter",
        "local_scales_centi_diameter": local_scales,
        "selection_uses": "train_fit_and_validation_movie_macro_only",
        "final_fit_uses": "train_plus_validation",
        "test_use": "single_frozen_evaluation",
        "target_or_future_at_issue_time": False,
        "independent_unit": "movie",
    }
    (args.out_dir / "v204_transport_contract.json").write_text(
        json.dumps(finite(contract), indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "v204_transport_decision.json").write_text(
        json.dumps(finite(decision), indent=2),
        encoding="utf-8",
    )
    report = [
        "# DeepSea Completed-Innovation Transport v204",
        "",
        "The base is a three-seed v97-direct Student-t mixture. The update uses",
        "only innovations completed by the previous observed frame.",
        "",
        f"- Primary h1 gain: `{h1_gain:.3f}%`.",
        f"- Primary h6 gain: `{h6_gain:.3f}%`.",
        (
            "- Positive h6 movies: "
            f"`{decision['h6_positive_movies']}/{decision['h6_movies']}`."
        ),
        (
            "- h6 movie-bootstrap 95% interval lower bound: "
            f"`{decision['h6_bootstrap_ci_low']:.3f}%`."
        ),
        f"- Real beats wrong-cell and stale-time: `{controls_pass}`.",
        f"- Frozen external transport gate: `{decision['external_transport_pass']}`.",
    ]
    (args.out_dir / "v204_transport_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated SEED=PATH entries.",
    )
    parser.add_argument(
        "--objectives",
        default="h1_strict,balanced_guard2,trajectory_guard5,h6_guard10",
    )
    parser.add_argument(
        "--packets",
        default="full,own_only,local_only,own_local,global_only",
    )
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument(
        "--bounds-cell-diameter",
        default="0.01,0.025,0.05,0.1,0.2,0.35",
    )
    parser.add_argument(
        "--local-scales-centi-diameter",
        default="50,100,200,400",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--control-seed", type=int, default=204_731)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
