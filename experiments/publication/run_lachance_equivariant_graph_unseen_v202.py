#!/usr/bin/env python3
"""Frozen configuration-unseen test of the v199 equivariant graph law.

All graph-law hyperparameters are selected by leave-one-movie-out validation
inside MDCK Bulk movies 1--6.  The fitted law is then evaluated once on movies
10--16 with real, wrong-cell, and stale-time packets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_equivariant_graph_bridge_v199 as v199  # noqa: E402
import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h  # noqa: E402
import run_lachance_streaming_transport_confirmation_v160 as v160  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "lachance_equivariant_graph_unseen_v202"
EPS = 1e-12


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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-cache", type=Path, default=v160.DEFAULT_DEV_CACHE)
    parser.add_argument(
        "--confirmation-cache",
        type=Path,
        default=v160.DEFAULT_CONFIRM_CACHE,
    )
    parser.add_argument("--checkpoints", default=v160.DEFAULT_CHECKPOINTS)
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument(
        "--variants",
        default="forced_potential,active_advective",
    )
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--max-neighbours", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--control-seed", type=int, default=202_001)
    parser.add_argument("--bootstrap", type=int, default=10000)
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


def graph_payloads(
    split_payloads: Mapping[int, tuple[str, Any]],
    *,
    max_neighbours: int,
    seed: int,
) -> dict[int, v199.GraphPayload]:
    return {
        int(movie): v199.build_graph_payload(
            split,
            base,
            max_neighbours=max_neighbours,
            control_seed=seed,
        )
        for movie, (split, base) in split_payloads.items()
    }


def cross_validated_selection(
    payloads: Mapping[int, v199.GraphPayload],
    movies: Sequence[int],
    variant: str,
    weights: Mapping[int, float],
    h1_guard: float,
    alphas: Sequence[float],
    bounds: Sequence[float],
) -> tuple[v199.Selection, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        fold_models = {
            validation_movie: v199.fit_model(
                payloads,
                [movie for movie in movies if movie != validation_movie],
                variant,
                weights,
                alpha,
            )
            for validation_movie in movies
        }
        for bound in bounds:
            fold_scores = []
            h1_gains = []
            record: dict[str, Any] = {
                "alpha": float(alpha),
                "bound_px": float(bound),
            }
            for validation_movie, model in fold_models.items():
                payload = payloads[int(validation_movie)]
                prediction = v199.bounded_prediction(
                    model,
                    payload,
                    variant,
                    "real",
                    bound,
                )
                metrics = v157e.metric_rows(
                    payload,
                    prediction,
                    "development_cv_real",
                    None,
                )
                score = v157h.score_metrics(metrics, dict(weights))
                h1_gain = next(
                    float(row["rmse_improvement_percent"])
                    for row in metrics
                    if int(row["horizon"]) == 1
                )
                fold_scores.append(score)
                h1_gains.append(h1_gain)
                record[f"movie{validation_movie:02d}_score"] = score
                record[f"movie{validation_movie:02d}_h1_gain"] = h1_gain
            record["validation_score"] = float(np.mean(fold_scores))
            record["h1_gain_percent"] = float(np.mean(h1_gains))
            record["h1_worst_gain_percent"] = float(np.min(h1_gains))
            records.append(record)
    grid = pd.DataFrame(records)
    eligible = grid[grid["h1_gain_percent"].ge(-float(h1_guard))]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        ["validation_score", "h1_gain_percent", "bound_px", "alpha"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return (
        v199.Selection(
            alpha=float(best["alpha"]),
            bound_px=float(best["bound_px"]),
            validation_score=float(best["validation_score"]),
            validation_h1_gain=float(best["h1_gain_percent"]),
        ),
        grid,
    )


def aggregate(metrics: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouping = ["objective_name", "variant", "control", "horizon"]
    for keys, rows in metrics.groupby(grouping, sort=True):
        delta = rows["component_rmse_delta"].to_numpy(np.float64)
        rng = np.random.default_rng(seed)
        samples = delta[
            rng.integers(0, len(delta), size=(bootstrap, len(delta)))
        ].mean(axis=1)
        records.append(
            {
                **dict(zip(grouping, keys)),
                "movies": int(rows["test_movie"].nunique()),
                "component_rmse_mean": float(rows["component_rmse"].mean()),
                "r2_mean": float(rows["r2"].mean()),
                "gain_percent_mean": float(
                    rows["rmse_improvement_percent"].mean()
                ),
                "movies_improved": int(np.sum(delta > 0)),
                "delta_ci_low": float(np.quantile(samples, 0.025)),
                "delta_ci_high": float(np.quantile(samples, 0.975)),
                "sign_flip_two_sided_p": v199.exact_sign_flip(delta),
                "sign_flip_one_sided_p": v199.exact_sign_flip(
                    delta,
                    "greater",
                ),
            }
        )
    return pd.DataFrame(records)


def paired_control_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    focus = metrics[metrics["horizon"].isin([1, 6])]
    for (objective, variant, horizon), rows in focus.groupby(
        ["objective_name", "variant", "horizon"],
        sort=True,
    ):
        pivot = rows.pivot(
            index="test_movie",
            columns="control",
            values="component_rmse",
        )
        if "real" not in pivot:
            continue
        for control in ("wrong_cell", "stale_time", "no_update"):
            if control not in pivot:
                continue
            advantage = pivot[control] - pivot["real"]
            records.append(
                {
                    "objective_name": objective,
                    "variant": variant,
                    "horizon": horizon,
                    "comparison": f"real_vs_{control}",
                    "movies": len(advantage),
                    "rmse_advantage_mean": float(advantage.mean()),
                    "real_better_movies": int(np.sum(advantage > 0)),
                    "one_sided_sign_flip_p": v199.exact_sign_flip(
                        advantage.to_numpy(np.float64),
                        "greater",
                    ),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = v160.parse_checkpoints(args.checkpoints)
    device = v157e.device_from_cli(args.device)
    development_replays = [
        v160.restore_checkpoint(checkpoints[seed], args.development_cache, device)
        for seed in sorted(checkpoints)
    ]
    confirmation_replays = [
        v160.restore_checkpoint(checkpoints[seed], args.confirmation_cache, device)
        for seed in sorted(checkpoints)
    ]
    v160.assert_development_confirmation_identity(
        development_replays,
        confirmation_replays,
    )
    development_split = v160.mixture_payloads(development_replays)
    confirmation_split = {
        movie: payload
        for movie, payload in v160.mixture_payloads(confirmation_replays).items()
        if movie in v160.CONFIRMATION_MOVIES
    }
    development_movies = list(v160.DEVELOPMENT_MOVIES)
    confirmation_movies = list(v160.CONFIRMATION_MOVIES)
    if set(development_split) != set(development_movies):
        raise RuntimeError("Development movie contract mismatch")
    if set(confirmation_split) != set(confirmation_movies):
        raise RuntimeError("Confirmation movie contract mismatch")
    development = graph_payloads(
        development_split,
        max_neighbours=args.max_neighbours,
        seed=args.control_seed,
    )
    confirmation = graph_payloads(
        confirmation_split,
        max_neighbours=args.max_neighbours,
        seed=args.control_seed + 1_000_003,
    )
    objectives = parse_strings(args.objectives)
    variants = parse_strings(args.variants)
    alphas = parse_floats(args.alphas)
    bounds = parse_floats(args.bounds_px)
    metric_records: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    coefficient_records: list[dict[str, Any]] = []

    for objective in objectives:
        weights, h1_guard = v157h.OBJECTIVES[objective]
        for variant in variants:
            selection, grid = cross_validated_selection(
                development,
                development_movies,
                variant,
                weights,
                h1_guard,
                alphas,
                bounds,
            )
            grid.insert(0, "objective_name", objective)
            grid.insert(1, "variant", variant)
            selection_frames.append(grid)
            model = v199.fit_model(
                development,
                development_movies,
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
                        "objective_name": objective,
                        "variant": variant,
                        "term": name,
                        "coefficient_physical": coefficient / max(scale, EPS),
                        "selected_alpha": selection.alpha,
                        "selected_bound_px": selection.bound_px,
                    }
                )
            for movie in confirmation_movies:
                payload = confirmation[movie]
                for control in ("real", "wrong_cell", "stale_time"):
                    prediction = v199.bounded_prediction(
                        model,
                        payload,
                        variant,
                        control,
                        selection.bound_px,
                    )
                    rows = v157e.metric_rows(
                        payload,
                        prediction,
                        control,
                        None,
                    )
                    v199.add_metadata(
                        rows,
                        objective=objective,
                        variant=variant,
                        selection=selection,
                    )
                    metric_records.extend(rows)
                no_update = v157e.metric_rows(
                    payload,
                    payload.base.mean,
                    "no_update",
                    None,
                )
                v199.add_metadata(
                    no_update,
                    objective=objective,
                    variant=variant,
                    selection=None,
                )
                metric_records.extend(no_update)
            print(
                f"[v202] objective={objective} variant={variant} "
                f"alpha={selection.alpha:g} bound={selection.bound_px:g}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_records)
    aggregate_frame = aggregate(metrics, args.bootstrap, args.control_seed)
    controls = paired_control_statistics(metrics)
    selections = pd.concat(selection_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_records)
    metrics.to_csv(args.out_dir / "v202_unseen_graph_metrics.csv", index=False)
    aggregate_frame.to_csv(
        args.out_dir / "v202_unseen_graph_aggregate.csv",
        index=False,
    )
    controls.to_csv(
        args.out_dir / "v202_unseen_graph_controls.csv",
        index=False,
    )
    selections.to_csv(
        args.out_dir / "v202_development_cv_grid.csv",
        index=False,
    )
    coefficients.to_csv(
        args.out_dir / "v202_graph_coefficients.csv",
        index=False,
    )
    primary = aggregate_frame[
        aggregate_frame["control"].eq("real")
        & aggregate_frame["horizon"].isin([1, 6])
    ]
    h6 = primary[
        primary["objective_name"].eq("h6_guard10")
        & primary["horizon"].eq(6)
    ]
    control_h6 = controls[
        controls["objective_name"].eq("h6_guard10")
        & controls["horizon"].eq(6)
        & controls["comparison"].isin(
            ["real_vs_wrong_cell", "real_vs_stale_time"]
        )
    ]
    pass_rows = (
        h6["movies_improved"].eq(len(confirmation_movies))
        & h6["delta_ci_low"].gt(0)
    )
    control_pass = bool(
        len(control_h6)
        and control_h6["rmse_advantage_mean"].gt(0).all()
    )
    decision = bool(len(h6) and pass_rows.any() and control_pass)
    report = [
        "# v202 Frozen Unseen Equivariant Graph Law",
        "",
        f"Decision: **{'PASS' if decision else 'FAIL'}**",
        "",
        "All hyperparameters were selected by leave-one-movie-out validation",
        "inside movies 1--6. Movies 10--16 were not used for selection.",
        "",
        "## Unseen-cohort operating points",
        "",
        primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired causal controls",
        "",
        control_h6.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.out_dir / "v202_unseen_graph_decision_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    frozen_marker = (
        ROOT
        / "outputs/lachance_streaming_transport_confirmation_v160_full_2026-07-27"
        / "V160_FROZEN_BEFORE_CONFIRMATION"
    )
    manifest = {
        "elapsed_minutes": (time.time() - started) / 60.0,
        "development_movies": development_movies,
        "confirmation_movies": confirmation_movies,
        "configuration_unseen": True,
        "confirmation_used_for_selection": False,
        "frozen_marker_exists": frozen_marker.exists(),
        "future_feature_count": 0,
        "decision": "pass" if decision else "fail",
    }
    (args.out_dir / "v202_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
