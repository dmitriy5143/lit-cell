#!/usr/bin/env python3
"""Audit v166 final-fit access without changing its validation selection.

For every frozen outer-movie fold, this runner restores the exact three-seed
v97 prior ensemble, selects the v166 alpha/bound on the validation movie, and
then compares two final update fits:

* train_only: fit the bounded innovation update on the four train movies;
* train_plus_validation: historical v157h refit on five non-test movies.

The outer test movie is never used for selection, fitting, normalization, or
uncertainty calibration. The audit isolates whether the historical validation
refit is material to the reported v166 advantage.
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

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "outputs" / "lachance_v166_fit_scope_audit_v189_2026-07-29"
)
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=189_001)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_fold(
    args: argparse.Namespace,
    test_movie: int,
    seeds: list[int],
    objective_names: list[str],
    alphas: list[float],
    bounds: list[float],
    local_scales: list[float],
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
    test = payloads[test_movie]
    metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for objective_name in objective_names:
        weights, h1_guard = v157h.OBJECTIVES[objective_name]
        selection, _grid = v157h.select_model(
            payloads,
            train_movies,
            validation_movie,
            weights,
            h1_guard,
            alphas,
            bounds,
        )
        selections.append(
            {
                "test_movie": test_movie,
                "validation_movie": validation_movie,
                "train_movies": ",".join(map(str, train_movies)),
                "objective_name": objective_name,
                "selected_alpha": selection.alpha,
                "selected_bound_px": selection.bound_px,
                "validation_score": selection.validation_score,
                "validation_h1_gain": selection.validation_h1_gain,
                "h1_guard_percent": h1_guard,
            }
        )
        for fit_scope, fit_movies in (
            ("train_only", train_movies),
            (
                "train_plus_validation",
                train_movies + [validation_movie],
            ),
        ):
            model = v157h.fit_model(
                payloads,
                fit_movies,
                selection.alpha,
                weights,
            )
            prediction = v157h.predict(
                model,
                test,
                "real",
                selection.bound_px,
            )
            rows = v157e.metric_rows(
                test,
                prediction,
                "real",
                None,
            )
            for row in rows:
                row.update(
                    {
                        "objective_name": objective_name,
                        "fit_scope": fit_scope,
                        "fit_movies": ",".join(map(str, fit_movies)),
                        "validation_movie": validation_movie,
                        "train_movies": ",".join(map(str, train_movies)),
                        "selected_alpha": selection.alpha,
                        "selected_bound_px": selection.bound_px,
                        "validation_score": selection.validation_score,
                        "protocol": "strict_fold_local_streaming",
                        "outer_test_used_for_fit_or_selection": False,
                        "test_key_sha256": replays[0].manifest[
                            "splits"
                        ]["test"]["key_sha256"],
                        "test_target_sha256": replays[0].manifest[
                            "splits"
                        ]["test"]["target_sha256"],
                        "test_row_target_sha256": replays[0].manifest[
                            "splits"
                        ]["test"]["row_target_sha256"],
                    }
                )
            metrics.extend(rows)
    causal = v157e.build_causal_audit(
        payloads,
        test_movie,
        validation_movie,
        train_movies,
    )
    return (
        metrics,
        selections,
        [dict(replay.manifest) for replay in replays],
        causal,
    )


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["objective_name", "fit_scope", "horizon"],
            as_index=False,
        )
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            gain_vs_no_update_percent=(
                "rmse_improvement_percent",
                "mean",
            ),
            movies_improved=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values(["objective_name", "fit_scope", "horizon"])
    )


def compare_scopes(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=["test_movie", "objective_name", "horizon"],
        columns="fit_scope",
        values="component_rmse",
        aggfunc="first",
    ).reset_index()
    pivot["train_only_minus_train_plus_validation"] = (
        pivot["train_only"] - pivot["train_plus_validation"]
    )
    pivot["train_only_relative_degradation_percent"] = 100.0 * (
        pivot["train_only"] - pivot["train_plus_validation"]
    ) / pivot["train_plus_validation"].clip(lower=EPS)
    return pivot


def main() -> None:
    args = parse_args()
    started = time.time()
    args.v102_root = args.v102_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = v157e.parse_csv_ints(args.movies)
    seeds = v157e.parse_csv_ints(args.seeds)
    objectives = [
        item.strip()
        for item in str(args.objectives).split(",")
        if item.strip()
    ]
    unknown = sorted(set(objectives) - set(v157h.OBJECTIVES))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    alphas = v157e.parse_csv_floats(args.alphas)
    bounds = v157e.parse_csv_floats(args.bounds_px)
    local_scales = v157e.parse_csv_floats(args.local_scales_px)
    all_metrics: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_replays: list[dict[str, Any]] = []
    all_causal: list[pd.DataFrame] = []
    for movie in movies:
        print(f"[v189] outer movie {movie}", flush=True)
        metrics, selections, replays, causal = evaluate_fold(
            args,
            movie,
            seeds,
            objectives,
            alphas,
            bounds,
            local_scales,
        )
        all_metrics.extend(metrics)
        all_selections.extend(selections)
        all_replays.extend(replays)
        all_causal.append(causal)
    metrics = pd.DataFrame(all_metrics)
    summary = aggregate(metrics)
    comparison = compare_scopes(metrics)
    causal = pd.concat(all_causal, ignore_index=True)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Future donor found in real packet")
    if not metrics["outer_test_used_for_fit_or_selection"].eq(False).all():
        raise RuntimeError("Outer test entered fit or selection")
    if set(movies) == set(range(1, 7)):
        historical_checks = {
            ("h1_strict", "train_plus_validation", 1): 3.474374,
            ("h6_guard10", "train_plus_validation", 6): 5.500749,
        }
        for key, expected in historical_checks.items():
            objective, scope, horizon = key
            row = summary[
                summary["objective_name"].eq(objective)
                & summary["fit_scope"].eq(scope)
                & summary["horizon"].eq(horizon)
            ]
            actual = float(row.iloc[0]["component_rmse"])
            if not np.isclose(actual, expected, atol=1e-5, rtol=0.0):
                raise RuntimeError(
                    f"Historical v157h reproduction drift for {key}: "
                    f"{actual}"
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "v189_fit_scope_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "v189_fit_scope_summary.csv", index=False)
    comparison.to_csv(
        args.out_dir / "v189_fit_scope_comparison.csv",
        index=False,
    )
    pd.DataFrame(all_selections).to_csv(
        args.out_dir / "v189_validation_selection.csv",
        index=False,
    )
    causal.to_csv(args.out_dir / "v189_causal_audit.csv", index=False)
    (args.out_dir / "v189_seed_replay_manifest.json").write_text(
        json.dumps(v157e.finite(all_replays), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    elapsed = time.time() - started
    headline = summary[
        (
            summary["objective_name"].eq("h1_strict")
            & summary["horizon"].eq(1)
        )
        | (
            summary["objective_name"].eq("h6_guard10")
            & summary["horizon"].eq(6)
        )
    ]
    report = [
        "# v189 v166 Final-Fit Scope Audit",
        "",
        headline.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Hyperparameters are selected on the validation movie using a model "
        "fit only on the four train movies. The two rows differ only in the "
        "final update fit scope. The outer test movie remains untouched.",
        "",
        f"Elapsed: `{elapsed / 60.0:.2f}` minutes.",
    ]
    (args.out_dir / "v189_fit_scope_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        **vars(args),
        "source_sha256": sha256(Path(__file__)),
        "v157e_source_sha256": sha256(Path(v157e.__file__)),
        "v157h_source_sha256": sha256(Path(v157h.__file__)),
        "protocol": "strict_fold_local_streaming",
        "outer_test_used_for_fit_or_selection": False,
        "fit_scopes": ["train_only", "train_plus_validation"],
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(v157e.finite(run_manifest), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
