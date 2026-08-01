#!/usr/bin/env python3
"""Exact sparse-operator audit for both frozen v166 operating points.

The earlier v191b audit reproduced only the h1-strict objective inherited from
v157e.  This runner applies the same sparse packet construction to the exact
v157h fold-local Pareto objectives used by the publication model, including
h6-utility.  Support radii are fixed from outer-training-movie field scales;
model selection remains validation-only and the outer movie is evaluation
only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h  # noqa: E402
import run_lachance_sparse_innovation_transport_refit_v191b as v191b  # noqa: E402


DEFAULT_OUT = (
    ROOT / "outputs" / "lachance_sparse_pareto_transport_v193_2026-07-30"
)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument("--v190-dir", type=Path, default=v191b.DEFAULT_V190)
    parser.add_argument(
        "--field-scale-unit",
        choices=["nearest_neighbour", "px"],
        default="nearest_neighbour",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument(
        "--variants",
        default="dense_start,field2_start,knn64_start",
    )
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=193_001)
    parser.add_argument("--equivalence-percent", type=float, default=0.2)
    return parser.parse_args()


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    columns = ["objective_name", "variant", "control", "horizon"]
    for keys, group in metrics.groupby(columns, sort=True):
        record = dict(zip(columns, keys))
        record.update(
            {
                "movies": int(group["test_movie"].nunique()),
                "component_rmse_mean": float(group["component_rmse"].mean()),
                "component_rmse_std": float(
                    group["component_rmse"].std(ddof=1)
                )
                if len(group) > 1
                else np.nan,
                "vector_rmse_mean": float(group["vector_rmse"].mean()),
                "r2_mean": float(group["r2"].mean()),
                "gain_vs_prior_percent_mean": float(
                    group["rmse_improvement_percent"].mean()
                ),
                "movies_improved_vs_prior": int(
                    (group["component_rmse_delta"] > 0).sum()
                ),
                "sign_flip_p": v191b.exact_sign_flip_pvalue(
                    group["component_rmse_delta"].to_numpy(np.float64)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def load_movie_scale(path: Path, unit: str) -> pd.Series:
    table = pd.read_csv(path)
    focus = table[
        table["representation"].eq("gaussian_score")
        & table["detrend"].eq("affine")
        & table["control"].eq("real")
        & table["lag"].eq(1)
        & table["geometry"].eq("endpoint")
        & table["metric"].eq("vector_correlation")
        & table["unit"].eq(unit)
    ][["movie", "exponential_xi"]]
    output = focus.set_index("movie")["exponential_xi"].sort_index()
    if output.index.duplicated().any() or output.isna().any():
        raise RuntimeError(f"Invalid v190 movie scale for unit {unit}")
    return output


def add_metric_metadata(
    rows: list[dict[str, Any]],
    *,
    objective_name: str,
    variant: v191b.Variant,
    selection: v157h.Selection | None,
    train_xi_nn: float,
) -> None:
    for row in rows:
        row.update(
            {
                "objective_name": objective_name,
                "variant": variant.name,
                "geometry": variant.geometry,
                "support_mode": variant.support_mode,
                "kernel": variant.kernel,
                "train_xi_nn": train_xi_nn,
                "selected_alpha": (
                    selection.alpha if selection is not None else np.nan
                ),
                "selected_bound_px": (
                    selection.bound_px if selection is not None else 0.0
                ),
                "validation_score": (
                    selection.validation_score
                    if selection is not None
                    else np.nan
                ),
                "validation_h1_gain": (
                    selection.validation_h1_gain
                    if selection is not None
                    else np.nan
                ),
            }
        )


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = v191b.parse_ints(args.movies)
    seeds = v191b.parse_ints(args.seeds)
    objectives = v191b.parse_strings(args.objectives)
    unknown = sorted(set(objectives) - set(v157h.OBJECTIVES))
    if unknown:
        raise SystemExit(f"Unknown v157h objectives: {unknown}")
    variants = [
        v191b.variant_from_name(name)
        for name in v191b.parse_strings(args.variants)
    ]
    if not any(variant.name == "dense_start" for variant in variants):
        raise SystemExit("variants must include dense_start")
    alphas = v191b.parse_floats(args.alphas)
    bounds = v191b.parse_floats(args.bounds_px)
    scales = v191b.parse_floats(args.local_scales_px)
    movie_xi = load_movie_scale(
        args.v190_dir / "scale_estimates.csv",
        args.field_scale_unit,
    )
    device = v157e.device_from_cli(args.device)

    metric_records: list[dict[str, Any]] = []
    grid_records: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    audit_records: list[pd.DataFrame] = []

    for test_movie in movies:
        print(f"[v193] restoring outer fold test={test_movie}", flush=True)
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
            raise RuntimeError("Fold split mismatch across optimizer seeds")
        validation_movie = next(iter(validation_movies))
        train_movies = list(next(iter(train_sets)))
        train_xi_nn = float(movie_xi.reindex(train_movies).median())

        variant_payloads: dict[str, dict[int, Any]] = {}
        for variant in variants:
            payloads: dict[int, Any] = {}
            for movie, (split, base) in split_payloads.items():
                payload, diagnostics = v191b.build_payload(
                    split,
                    base,
                    scales,
                    variant,
                    train_xi_nn,
                    args.control_seed + test_movie * 100_003,
                )
                payloads[movie] = payload
                diagnostic_records.append(
                    {
                        "outer_test_movie": test_movie,
                        "validation_movie": validation_movie,
                        "movie": movie,
                        "split": split,
                        "variant": variant.name,
                        "geometry": variant.geometry,
                        "support_mode": variant.support_mode,
                        "train_xi_nn": train_xi_nn,
                        **diagnostics,
                    }
                )
            variant_payloads[variant.name] = payloads
            audit = v157e.build_causal_audit(
                payloads,
                test_movie,
                validation_movie,
                train_movies,
            )
            audit.insert(0, "variant", variant.name)
            audit_records.append(audit)

        for objective_name in objectives:
            weights, h1_guard = v157h.OBJECTIVES[objective_name]
            for variant in variants:
                payloads = variant_payloads[variant.name]
                selection, grid = v157h.select_model(
                    payloads,
                    train_movies,
                    validation_movie,
                    weights,
                    h1_guard,
                    alphas,
                    bounds,
                )
                grid.insert(0, "outer_test_movie", test_movie)
                grid.insert(1, "variant", variant.name)
                grid.insert(2, "objective_name", objective_name)
                grid.insert(3, "train_xi_nn", train_xi_nn)
                grid_records.append(grid)
                model = v157h.fit_model(
                    payloads,
                    train_movies + [validation_movie],
                    selection.alpha,
                    weights,
                )
                test = payloads[test_movie]
                for control in ("real", "wrong_cell", "stale_time"):
                    prediction = v157h.predict(
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
                    add_metric_metadata(
                        rows,
                        objective_name=objective_name,
                        variant=variant,
                        selection=selection,
                        train_xi_nn=train_xi_nn,
                    )
                    metric_records.extend(rows)
                no_update = v157e.metric_rows(
                    test,
                    test.base.mean,
                    "no_update",
                    None,
                )
                add_metric_metadata(
                    no_update,
                    objective_name=objective_name,
                    variant=variant,
                    selection=None,
                    train_xi_nn=train_xi_nn,
                )
                metric_records.extend(no_update)
                print(
                    f"[v193] test={test_movie} objective={objective_name} "
                    f"variant={variant.name} alpha={selection.alpha:g} "
                    f"bound={selection.bound_px:g}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_records)
    aggregate = aggregate_metrics(metrics)
    diagnostics = pd.DataFrame(diagnostic_records)
    validation = pd.concat(grid_records, ignore_index=True)
    audit = pd.concat(audit_records, ignore_index=True)
    if audit["real_future_donor_violations"].sum() != 0:
        raise RuntimeError("Future donor found in v193")
    if audit["stale_future_or_nonstale_violations"].sum() != 0:
        raise RuntimeError("Invalid stale donor found in v193")

    metrics.to_csv(args.out_dir / "v193_sparse_pareto_metrics.csv", index=False)
    aggregate.to_csv(
        args.out_dir / "v193_sparse_pareto_aggregate.csv",
        index=False,
    )
    diagnostics.to_csv(
        args.out_dir / "v193_sparse_operator_diagnostics.csv",
        index=False,
    )
    validation.to_csv(
        args.out_dir / "v193_sparse_validation_grid.csv",
        index=False,
    )
    audit.to_csv(args.out_dir / "v193_sparse_causal_audit.csv", index=False)

    primary = aggregate[
        aggregate["control"].eq("real")
        & aggregate["horizon"].isin([1, 6])
    ].copy()
    dense = primary[primary["variant"].eq("dense_start")][
        ["objective_name", "horizon", "component_rmse_mean"]
    ].rename(columns={"component_rmse_mean": "dense_component_rmse"})
    primary = primary.merge(
        dense,
        on=["objective_name", "horizon"],
        how="left",
        validate="many_to_one",
    )
    primary["relative_to_dense_percent"] = 100.0 * (
        primary["component_rmse_mean"] / primary["dense_component_rmse"] - 1.0
    )
    edge = diagnostics[diagnostics["split"].eq("test")].groupby(
        "variant", as_index=False
    )["edge_fraction"].mean()
    primary = primary.merge(edge, on="variant", how="left")
    # One-sided non-inferiority: improvements are always admissible.
    primary["equivalent_to_dense"] = (
        primary["relative_to_dense_percent"]
        <= float(args.equivalence_percent)
    )
    primary.to_csv(
        args.out_dir / "v193_sparse_pareto_equivalence.csv",
        index=False,
    )

    checks: list[dict[str, Any]] = []
    for objective_name in objectives:
        for variant in variants:
            if variant.name == "dense_start":
                continue
            focus = primary[
                primary["objective_name"].eq(objective_name)
                & primary["variant"].eq(variant.name)
            ]
            controls = aggregate[
                aggregate["objective_name"].eq(objective_name)
                & aggregate["variant"].eq(variant.name)
                & aggregate["horizon"].eq(6)
            ].set_index("control")
            real = float(controls.loc["real", "component_rmse_mean"])
            checks.append(
                {
                    "objective_name": objective_name,
                    "variant": variant.name,
                    "h1_equivalent": bool(
                        focus.loc[focus["horizon"].eq(1), "equivalent_to_dense"].iloc[0]
                    ),
                    "h6_equivalent": bool(
                        focus.loc[focus["horizon"].eq(6), "equivalent_to_dense"].iloc[0]
                    ),
                    "real_beats_wrong_cell": real
                    < float(
                        controls.loc["wrong_cell", "component_rmse_mean"]
                    ),
                    "real_beats_stale_time": real
                    < float(
                        controls.loc["stale_time", "component_rmse_mean"]
                    ),
                    "positive_h6_movies": int(
                        controls.loc["real", "movies_improved_vs_prior"]
                    ),
                }
            )
    checks_frame = pd.DataFrame(checks)
    checks_frame["pass"] = checks_frame[
        [
            "h1_equivalent",
            "h6_equivalent",
            "real_beats_wrong_cell",
            "real_beats_stale_time",
        ]
    ].all(axis=1) & checks_frame["positive_h6_movies"].eq(len(movies))
    checks_frame.to_csv(
        args.out_dir / "v193_sparse_pareto_checks.csv",
        index=False,
    )

    report = [
        "# v193 Exact Sparse v166 Pareto Audit",
        "",
        "This run evaluates the sparse operator inside both exact v157h/v166",
        "objectives. Field support is derived only from outer-training movies.",
        "",
        "## Primary equivalence",
        "",
        primary[
            [
                "objective_name",
                "variant",
                "horizon",
                "component_rmse_mean",
                "r2_mean",
                "edge_fraction",
                "relative_to_dense_percent",
                "equivalent_to_dense",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Causal and equivalence checks",
        "",
        checks_frame.to_markdown(index=False),
        "",
        "A sparse final architecture is supported only when the chosen sparse",
        "variant passes both operating points; subsystem-only equivalence is",
        "not promoted to a final-model claim.",
    ]
    (args.out_dir / "v193_sparse_pareto_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "ok": True,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "movies": movies,
        "seeds": seeds,
        "objectives": objectives,
        "variants": [variant.__dict__ for variant in variants],
        "equivalence_percent": float(args.equivalence_percent),
        "strict_train_only_field_scale": True,
        "field_scale_unit": args.field_scale_unit,
        "outer_test_used_for_selection": False,
        "protocol": "exact v157h/v166 streaming Pareto replay",
    }
    (args.out_dir / "v193_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v193] complete in {manifest['elapsed_minutes']:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
