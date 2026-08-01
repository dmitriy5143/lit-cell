#!/usr/bin/env python3
"""Strict fold-local Pareto sweep for semigroup streaming forecasts.

The primary v157e operating point assigns 80% of its objective to h1.  This
runner predefines three additional validation-selected operating points with
progressively larger h4/h6 weight and explicit validation h1 guards.  It asks
whether one causal streaming model can dominate constant velocity across
h1/h2/h4/h6 rather than trading excellent h1 for weak cumulative motion.

Only the primary h1-strict row is confirmatory.  Other objectives are labelled
exploratory Pareto operating points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lachance_foldlocal_semigroup_pareto_v157h"
DEFAULT_V102_SUMMARY = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_benchmark_v102_v97_production_2026-07-21"
    / "v102_movie_level_summary.csv"
)
EPS = 1e-8
OBJECTIVES: dict[str, tuple[dict[int, float], float]] = {
    "h1_strict": ({1: 0.80, 2: 0.10, 4: 0.06, 6: 0.04}, 0.5),
    "balanced_guard2": ({1: 0.45, 2: 0.15, 4: 0.15, 6: 0.25}, 2.0),
    "trajectory_guard5": ({1: 0.25, 2: 0.15, 4: 0.20, 6: 0.40}, 5.0),
    "h6_guard10": ({1: 0.10, 2: 0.10, 4: 0.20, 6: 0.60}, 10.0),
}


@dataclass
class Selection:
    alpha: float
    bound_px: float
    validation_score: float
    validation_h1_gain: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument(
        "--v102-summary",
        type=Path,
        default=DEFAULT_V102_SUMMARY,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument(
        "--objectives",
        default=",".join(OBJECTIVES),
    )
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=157_008)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_data(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    mean: np.ndarray,
    scale: np.ndarray,
    weights_by_horizon: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for movie in movies:
        payload = payloads[movie]
        normalized = (v157e.raw_design(payload) - mean) / scale
        per_step = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        residual = payload.base.target - payload.base.mean
        for horizon in v157e.HORIZONS:
            windows = v157e.consecutive_windows(
                payload.base.rows,
                horizon,
            )
            features.append(per_step[windows].sum(axis=1))
            targets.append(residual[windows].sum(axis=1))
            weights.append(
                np.full(
                    len(windows),
                    weights_by_horizon[horizon] / len(windows),
                    dtype=np.float64,
                )
            )
    feature_matrix = np.concatenate(features)
    target_matrix = np.concatenate(targets)
    weight_vector = np.concatenate(weights)
    weight_vector *= len(weight_vector) / max(
        float(weight_vector.sum()),
        EPS,
    )
    return feature_matrix, target_matrix, weight_vector


def fit_model(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    alpha: float,
    weights_by_horizon: dict[int, float],
) -> v157e.WeightedRidge:
    mean, scale = v157e.row_normalization(payloads, movies)
    features, targets, weights = training_data(
        payloads,
        movies,
        mean,
        scale,
        weights_by_horizon,
    )
    return v157e.fit_weighted_ridge(
        features,
        targets,
        weights,
        alpha,
        mean,
        scale,
    )


def predict(
    model: v157e.WeightedRidge,
    payload: v157e.UpdatePayload,
    control: str,
    bound_px: float,
) -> np.ndarray:
    raw = v157e.predict_ridge(model, payload, control)
    return payload.base.mean + v157e.bounded_update(raw, bound_px)


def score_metrics(
    rows: list[dict[str, Any]],
    weights_by_horizon: dict[int, float],
) -> float:
    return float(
        sum(
            weights_by_horizon[int(row["horizon"])]
            * float(row["component_rmse"])
            / max(float(row["baseline_component_rmse"]), EPS)
            for row in rows
        )
    )


def select_model(
    payloads: dict[int, v157e.UpdatePayload],
    train_movies: list[int],
    validation_movie: int,
    weights_by_horizon: dict[int, float],
    h1_guard_percent: float,
    alphas: list[float],
    bounds: list[float],
) -> tuple[Selection, pd.DataFrame]:
    validation = payloads[validation_movie]
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_model(
            payloads,
            train_movies,
            alpha,
            weights_by_horizon,
        )
        for bound in bounds:
            prediction = predict(
                model,
                validation,
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
                "validation_score": score_metrics(
                    metrics,
                    weights_by_horizon,
                ),
            }
            for row in metrics:
                horizon = int(row["horizon"])
                record[f"h{horizon}_component_rmse"] = row[
                    "component_rmse"
                ]
                record[f"h{horizon}_gain_percent"] = row[
                    "rmse_improvement_percent"
                ]
            records.append(record)
    grid = pd.DataFrame(records)
    eligible = grid[
        grid["h1_gain_percent"].ge(-float(h1_guard_percent))
    ]
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


def evaluate_fold(
    args: argparse.Namespace,
    test_movie: int,
    seeds: list[int],
    objective_names: list[str],
    alphas: list[float],
    bounds: list[float],
    local_scales: list[float],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], pd.DataFrame]:
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
    validation_movie = int(replays[0].manifest["validation_movie"])
    train_movies = list(replays[0].manifest["train_movies"])
    payloads = {
        movie: v157e.build_update_payload(
            split,
            base,
            local_scales,
            int(args.control_seed) + test_movie * 100_003,
        )
        for movie, (split, base) in split_payloads.items()
    }
    test = payloads[test_movie]
    output = v157e.metric_rows(
        test,
        test.base.mean,
        "no_update",
        None,
    )
    for row in output:
        row["objective_name"] = "no_update"
        row["variant"] = "no_update"
        row["protocol"] = "strict_fold_local_streaming"
    grids: list[pd.DataFrame] = []
    for objective_name in objective_names:
        objective_weights, h1_guard = OBJECTIVES[objective_name]
        selection, grid = select_model(
            payloads,
            train_movies,
            validation_movie,
            objective_weights,
            h1_guard,
            alphas,
            bounds,
        )
        model = fit_model(
            payloads,
            train_movies + [validation_movie],
            selection.alpha,
            objective_weights,
        )
        for control in ("real", "wrong_cell", "stale_time"):
            prediction = predict(
                model,
                test,
                control,
                selection.bound_px,
            )
            rows = v157e.metric_rows(
                test,
                prediction,
                control,
                None,
            )
            for row in rows:
                row["objective_name"] = objective_name
                row["variant"] = f"{objective_name}_{control}"
                row["selected_alpha"] = selection.alpha
                row["selected_bound_px"] = selection.bound_px
                row["validation_score"] = selection.validation_score
                row["validation_h1_gain"] = (
                    selection.validation_h1_gain
                )
                row["h1_guard_percent"] = h1_guard
                row["protocol"] = "strict_fold_local_streaming"
            output.extend(rows)
        grid.insert(0, "test_movie", test_movie)
        grid.insert(1, "validation_movie", validation_movie)
        grid.insert(2, "objective_name", objective_name)
        grid.insert(3, "h1_guard_percent", h1_guard)
        grids.append(grid)
    causal = v157e.build_causal_audit(
        payloads,
        test_movie,
        validation_movie,
        train_movies,
    )
    return output, grids, causal


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["objective_name", "variant", "horizon"],
            as_index=False,
        )
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            vector_rmse_mean=("vector_rmse", "mean"),
            r2_mean=("r2", "mean"),
            gain_percent_mean=("rmse_improvement_percent", "mean"),
            movies_improved=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )


def constant_velocity_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[
        frame["method_id"].eq("baseline/constant_velocity")
    ][
        [
            "horizon",
            "component_rmse_movie_mean",
            "r2_movie_mean",
        ]
    ].rename(
        columns={
            "component_rmse_movie_mean": "cv_component_rmse",
            "r2_movie_mean": "cv_r2",
        }
    )


def pareto_summary(
    aggregate_frame: pd.DataFrame,
    cv: pd.DataFrame,
) -> pd.DataFrame:
    real = aggregate_frame[
        aggregate_frame["variant"].str.endswith("_real")
    ].copy()
    real = real.merge(cv, on="horizon", how="left")
    real["gain_vs_cv_percent"] = 100.0 * (
        real["cv_component_rmse"] - real["component_rmse_mean"]
    ) / real["cv_component_rmse"].clip(lower=EPS)
    objective_rows: list[dict[str, Any]] = []
    for objective, group in real.groupby("objective_name"):
        ordered = group.set_index("horizon")
        objective_rows.append(
            {
                "objective_name": objective,
                "h1_component_rmse": ordered.loc[1, "component_rmse_mean"],
                "h6_component_rmse": ordered.loc[6, "component_rmse_mean"],
                "h1_gain_vs_cv_percent": ordered.loc[
                    1, "gain_vs_cv_percent"
                ],
                "h6_gain_vs_cv_percent": ordered.loc[
                    6, "gain_vs_cv_percent"
                ],
                "dominates_cv_all_horizons": bool(
                    (group["gain_vs_cv_percent"] > 0).all()
                ),
                "mean_gain_vs_cv_percent": float(
                    group["gain_vs_cv_percent"].mean()
                ),
            }
        )
    return pd.DataFrame(objective_rows).sort_values(
        ["dominates_cv_all_horizons", "mean_gain_vs_cv_percent"],
        ascending=[False, False],
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    args.v102_root = args.v102_root.resolve()
    args.v102_summary = args.v102_summary.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = v157e.parse_csv_ints(args.movies)
    seeds = v157e.parse_csv_ints(args.seeds)
    objective_names = [
        token.strip()
        for token in str(args.objectives).split(",")
        if token.strip()
    ]
    unknown = sorted(set(objective_names) - set(OBJECTIVES))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    alphas = v157e.parse_csv_floats(args.alphas)
    bounds = v157e.parse_csv_floats(args.bounds_px)
    local_scales = v157e.parse_csv_floats(args.local_scales_px)

    all_metrics: list[dict[str, Any]] = []
    all_grids: list[pd.DataFrame] = []
    all_causal: list[pd.DataFrame] = []
    for movie in movies:
        print(f"[v157h] strict outer movie {movie}", flush=True)
        metrics, grids, causal = evaluate_fold(
            args,
            movie,
            seeds,
            objective_names,
            alphas,
            bounds,
            local_scales,
        )
        all_metrics.extend(metrics)
        all_grids.extend(grids)
        all_causal.append(causal)
    metrics_frame = pd.DataFrame(all_metrics)
    aggregate_frame = aggregate(metrics_frame)
    cv = constant_velocity_reference(args.v102_summary)
    pareto = pareto_summary(aggregate_frame, cv)
    causal = pd.concat(all_causal, ignore_index=True)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Future donor found")
    if int(causal["stale_future_or_nonstale_violations"].sum()) != 0:
        raise RuntimeError("Invalid stale donor found")
    if not bool(causal["coherent_wrong_packet"].all()):
        raise RuntimeError("Wrong-cell packet coherence failed")

    metrics_frame.to_csv(
        args.out_dir / "v157h_pareto_metrics.csv",
        index=False,
    )
    aggregate_frame.to_csv(
        args.out_dir / "v157h_pareto_aggregate.csv",
        index=False,
    )
    pd.concat(all_grids, ignore_index=True).to_csv(
        args.out_dir / "v157h_validation_grid.csv",
        index=False,
    )
    pareto.to_csv(args.out_dir / "v157h_pareto_summary.csv", index=False)
    causal.to_csv(args.out_dir / "v157h_causal_audit.csv", index=False)
    report = [
        "# v157h Strict Semigroup Pareto Sweep",
        "",
        "Only `h1_strict` is the confirmatory primary operating point.",
        "Other rows are validation-selected exploratory Pareto profiles.",
        "",
        pareto.to_markdown(index=False, floatfmt=".6f"),
        "",
        "A row dominates constant velocity only when all h1/h2/h4/h6",
        "component RMSE values are lower than the frozen v102 reference.",
        "",
        f"Elapsed: `{(time.time() - started) / 60.0:.2f}` minutes.",
    ]
    (args.out_dir / "v157h_status_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **vars(args),
        "objectives": {
            name: {
                "weights": OBJECTIVES[name][0],
                "validation_h1_guard_percent": OBJECTIVES[name][1],
                "status": (
                    "confirmatory"
                    if name == "h1_strict"
                    else "exploratory"
                ),
            }
            for name in objective_names
        },
        "source_sha256": file_sha256(Path(__file__)),
        "protocol": "strict fold-local streaming/receding h1",
        "constant_velocity_source": str(args.v102_summary),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(v157e.finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "v157h_status_report.md", flush=True)


if __name__ == "__main__":
    main()
