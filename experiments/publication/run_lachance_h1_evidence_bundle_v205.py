#!/usr/bin/env python3
"""Build the frozen h1--h6 evidence bundle for the streaming model.

This runner does not introduce a new forecasting architecture.  It evaluates
eleven predeclared objective profiles between the confirmatory h1 operating
point and the exploratory h6-utility point.  Hyperparameters are selected on
the inner validation movie and evaluated once on the outer movie.  The output
also reports dimensionless errors and keeps localization-reliability evidence
separate from forecasting performance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lachance_h1_evidence_bundle_v205"
DEFAULT_V102_SUMMARY = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_benchmark_v102_v97_production_2026-07-21"
    / "v102_movie_level_summary.csv"
)
DEFAULT_V102_MOVIES = DEFAULT_V102_SUMMARY.with_name(
    "v102_seed_aggregated_within_movie.csv"
)
DEFAULT_LACHANCE_RELIABILITY = (
    ROOT
    / "outputs"
    / "lachance_cellpose_reliability_lomo_v177c_current_query_2026-07-28"
    / "v177c_reliability_decision.json"
)
DEFAULT_C2C12_AUDIT = (
    ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_c2c12_audit_2026-07-21"
    / "c2c12_tracking_noise_contract.json"
)
HORIZONS = (1, 2, 4, 6)
EPS = 1e-12


def objective_profiles() -> dict[str, tuple[dict[int, float], float]]:
    h1_weights = {1: 0.80, 2: 0.10, 4: 0.06, 6: 0.04}
    h6_weights = {1: 0.10, 2: 0.10, 4: 0.20, 6: 0.60}
    profiles: dict[str, tuple[dict[int, float], float]] = {}
    for index in range(11):
        fraction = index / 10.0
        weights = {
            horizon: (1.0 - fraction) * h1_weights[horizon]
            + fraction * h6_weights[horizon]
            for horizon in HORIZONS
        }
        guard = 0.5 + 9.5 * fraction
        profiles[f"lambda_{index:02d}"] = (weights, guard)
    return profiles


PROFILES = objective_profiles()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument("--v102-summary", type=Path, default=DEFAULT_V102_SUMMARY)
    parser.add_argument("--v102-movies", type=Path, default=DEFAULT_V102_MOVIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument(
        "--alphas", default="1,10,30,100,300,1000,3000,10000"
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=205_001)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument(
        "--lachance-reliability",
        type=Path,
        default=DEFAULT_LACHANCE_RELIABILITY,
    )
    parser.add_argument("--c2c12-audit", type=Path, default=DEFAULT_C2C12_AUDIT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["objective_name", "variant", "horizon"], as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            component_r2_mean=("component_r2", "mean"),
            vector_rmse_mean=("vector_rmse", "mean"),
            vector_r2_mean=("vector_r2", "mean"),
            gain_vs_no_update_percent=("rmse_improvement_percent", "mean"),
        )
        .sort_values(["objective_name", "variant", "horizon"])
    )


def movie_normalized_metrics(
    metrics: pd.DataFrame,
    v102_movies_path: Path,
) -> pd.DataFrame:
    no_update = metrics[metrics["variant"].eq("no_update")][
        ["test_movie", "horizon", "component_rmse", "component_r2"]
    ].copy()
    no_update["target_component_sd"] = no_update["component_rmse"] / np.sqrt(
        np.clip(1.0 - no_update["component_r2"], EPS, None)
    )
    target_scale = no_update[
        ["test_movie", "horizon", "target_component_sd"]
    ]

    real = metrics[metrics["control"].eq("real")].copy()
    real = real.merge(target_scale, on=["test_movie", "horizon"], how="left")
    real["normalized_rmse"] = (
        real["component_rmse"] / real["target_component_sd"]
    )
    real["variance_explained"] = 1.0 - np.square(real["normalized_rmse"])

    cv = pd.read_csv(v102_movies_path)
    cv = cv[cv["method_id"].eq("baseline/constant_velocity")][
        ["test_movie", "horizon", "component_rmse"]
    ].rename(columns={"component_rmse": "cv_component_rmse"})
    real = real.merge(cv, on=["test_movie", "horizon"], how="left")
    real["skill_vs_cv"] = 1.0 - np.square(
        real["component_rmse"] / real["cv_component_rmse"].clip(lower=EPS)
    )
    return real


def normalized_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["objective_name", "horizon"], as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            target_component_sd_mean=("target_component_sd", "mean"),
            normalized_rmse_mean=("normalized_rmse", "mean"),
            normalized_rmse_std=("normalized_rmse", "std"),
            variance_explained_mean=("variance_explained", "mean"),
            skill_vs_cv_mean=("skill_vs_cv", "mean"),
        )
        .sort_values(["objective_name", "horizon"])
    )


def pareto_summary(
    aggregate: pd.DataFrame,
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    real = aggregate[aggregate["variant"].str.endswith("_real")].copy()
    wide_rmse = real.pivot(
        index="objective_name", columns="horizon", values="component_rmse_mean"
    )
    wide_r2 = real.pivot(
        index="objective_name", columns="horizon", values="component_r2_mean"
    )
    wide_norm = normalized.pivot(
        index="objective_name", columns="horizon", values="normalized_rmse_mean"
    )
    records: list[dict[str, Any]] = []
    for index, name in enumerate(PROFILES):
        if name not in wide_rmse.index:
            continue
        row: dict[str, Any] = {
            "objective_name": name,
            "lambda": index / 10.0,
            "h1_guard_percent": PROFILES[name][1],
            "status": (
                "confirmatory_h1"
                if index == 0
                else "frozen_later_on_unseen_movies"
                if index == 10
                else "exploratory"
            ),
        }
        for horizon in HORIZONS:
            row[f"h{horizon}_component_rmse"] = wide_rmse.loc[name, horizon]
            row[f"h{horizon}_component_r2"] = wide_r2.loc[name, horizon]
            row[f"h{horizon}_normalized_rmse"] = wide_norm.loc[name, horizon]
        records.append(row)
    result = pd.DataFrame(records)
    dominated: list[bool] = []
    for _, row in result.iterrows():
        competitor = result[
            (result["h1_component_rmse"] <= row["h1_component_rmse"])
            & (result["h6_component_rmse"] <= row["h6_component_rmse"])
            & (
                (result["h1_component_rmse"] < row["h1_component_rmse"])
                | (result["h6_component_rmse"] < row["h6_component_rmse"])
            )
        ]
        dominated.append(not competitor.empty)
    result["pareto_nondominated"] = np.logical_not(dominated)
    return result


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bootstrap_ci(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def pairwise_statistics(
    metrics: pd.DataFrame,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint_profiles = ("lambda_00", "lambda_10")
    for profile in endpoint_profiles:
        real = metrics[
            (metrics["objective_name"].eq(profile))
            & (metrics["control"].eq("real"))
        ]
        for horizon in HORIZONS:
            current = real[real["horizon"].eq(horizon)][
                ["test_movie", "component_rmse"]
            ].rename(columns={"component_rmse": "real_rmse"})
            for comparator in ("no_update", "wrong_cell", "stale_time"):
                if comparator == "no_update":
                    other = metrics[
                        (metrics["variant"].eq("no_update"))
                        & (metrics["horizon"].eq(horizon))
                    ][["test_movie", "component_rmse"]]
                else:
                    other = metrics[
                        (metrics["objective_name"].eq(profile))
                        & (metrics["control"].eq(comparator))
                        & (metrics["horizon"].eq(horizon))
                    ][["test_movie", "component_rmse"]]
                other = other.rename(columns={"component_rmse": "other_rmse"})
                paired = current.merge(other, on="test_movie", how="inner")
                differences = paired["other_rmse"].to_numpy() - paired[
                    "real_rmse"
                ].to_numpy()
                low, high = bootstrap_ci(
                    differences,
                    repeats,
                    205_000 + horizon + len(rows) * 17,
                )
                rows.append(
                    {
                        "objective_name": profile,
                        "horizon": horizon,
                        "comparator": comparator,
                        "movies": len(differences),
                        "mean_rmse_delta_comparator_minus_real": differences.mean(),
                        "movies_real_better": int((differences > 0).sum()),
                        "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def reliability_context(
    lachance_path: Path,
    c2c12_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lachance = json.loads(lachance_path.read_text(encoding="utf-8"))
    c2c12 = json.loads(c2c12_path.read_text(encoding="utf-8"))
    pair = c2c12["pair_summary"]
    rows = [
        {
            "dataset": "LaChance MDCK Bulk",
            "audit": "Cellpose current-query reliability, six movie folds",
            "quantity": "NLL gain over best hard control",
            "value": 100.0 * float(lachance["mean_gain_fraction"]),
            "unit": "%",
            "independent_localization_reference": False,
            "decision": lachance["decision"],
        },
        {
            "dataset": "C2C12 F0009",
            "audit": "manual/automatic same-frame forensic match",
            "quantity": "median one-step disagreement",
            "value": float(pair["step_disagreement_median_px"]),
            "unit": "px",
            "independent_localization_reference": True,
            "decision": "context_only",
        },
        {
            "dataset": "C2C12 F0009",
            "audit": "manual/automatic same-frame forensic match",
            "quantity": "p90 one-step disagreement",
            "value": float(pair["step_disagreement_p90_px"]),
            "unit": "px",
            "independent_localization_reference": True,
            "decision": "context_only",
        },
    ]
    conclusion = {
        "lachance_reliability_gate": lachance["decision"],
        "lachance_independent_causal_retracking_complete": bool(
            lachance["independent_causal_retracking_complete"]
        ),
        "c2c12_is_lachance_noise_floor": False,
        "claim": (
            "No LaChance-specific irreducible h1 noise floor was established. "
            "C2C12 disagreement is an external scale reference only."
        ),
    }
    return pd.DataFrame(rows), conclusion


def plot_evidence(
    pareto: pd.DataFrame,
    normalized: pd.DataFrame,
    v102_summary_path: Path,
    out_dir: Path,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)

    ax = axes[0]
    scatter = ax.scatter(
        pareto["h1_component_rmse"],
        pareto["h6_component_rmse"],
        c=pareto["lambda"],
        cmap="viridis",
        s=54,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ordered = pareto.sort_values("lambda")
    ax.plot(
        ordered["h1_component_rmse"],
        ordered["h6_component_rmse"],
        color="#6b7280",
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )
    ax.annotate(
        "строгий h1",
        tuple(ordered.iloc[0][["h1_component_rmse", "h6_component_rmse"]]),
        xytext=(8, -14),
        textcoords="offset points",
    )
    ax.annotate(
        "накопительный h6",
        tuple(ordered.iloc[-1][["h1_component_rmse", "h6_component_rmse"]]),
        xytext=(-82, 9),
        textcoords="offset points",
    )
    ax.set_xlabel("RMSE ближайшего шага h1, пикс.")
    ax.set_ylabel("накопительная RMSE h6, пикс.")
    ax.set_title("A  Компромисс рабочих режимов", loc="left", fontweight="bold")
    ax.grid(color="#e5e7eb", linewidth=0.7)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("вес перехода к цели h6")

    ax = axes[1]
    selected = {
        "lambda_00": ("строгий h1", "#2166ac", "o"),
        "lambda_05": ("сбалансированный", "#1b9e77", "s"),
        "lambda_10": ("накопительный h6", "#b2182b", "D"),
    }
    for name, (label, color, marker) in selected.items():
        group = normalized[normalized["objective_name"].eq(name)].sort_values(
            "horizon"
        )
        ax.plot(
            group["horizon"],
            group["normalized_rmse_mean"],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
    cv = pd.read_csv(v102_summary_path)
    cv = cv[cv["method_id"].eq("baseline/constant_velocity")].copy()
    target = normalized[normalized["objective_name"].eq("lambda_00")][
        ["horizon", "target_component_sd_mean"]
    ]
    cv = cv.merge(target, on="horizon", how="left")
    cv["normalized_rmse"] = (
        cv["component_rmse_movie_mean"] / cv["target_component_sd_mean"]
    )
    ax.plot(
        cv["horizon"],
        cv["normalized_rmse"],
        label="постоянная скорость",
        color="#4b5563",
        linestyle="--",
        marker="x",
        linewidth=1.5,
    )
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("накопительный горизонт")
    ax.set_ylabel("RMSE / стандартное отклонение цели")
    ax.set_title("B  Ошибка относительно масштаба цели", loc="left", fontweight="bold")
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.5)

    for suffix in ("png", "pdf"):
        fig.savefig(plot_dir / f"h1_h6_evidence.{suffix}", dpi=300)
    plt.close(fig)


def report_text(
    pareto: pd.DataFrame,
    normalized: pd.DataFrame,
    statistics: pd.DataFrame,
    reliability: pd.DataFrame,
    reliability_conclusion: dict[str, Any],
    elapsed_minutes: float,
) -> str:
    endpoints = pareto[pareto["objective_name"].isin(["lambda_00", "lambda_10"])]
    norm_endpoints = normalized[
        normalized["objective_name"].isin(["lambda_00", "lambda_10"])
    ]
    return "\n".join(
        [
            "# LaChance h1 evidence bundle v205",
            "",
            "## Decision",
            "",
            "The h1 target is difficult but not shown to be irreducible noise. "
            "The frozen model exposes a continuous h1--h6 utility trade-off; "
            "h1 remains an explicit guarded endpoint rather than being dismissed.",
            "",
            "Only `lambda_00` is the original confirmatory h1 operating point. "
            "Intermediate profiles are descriptive. `lambda_10` was frozen only "
            "before the separate movies 10--16 evaluation.",
            "",
            "## Pareto endpoints",
            "",
            endpoints.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Dimensionless error",
            "",
            norm_endpoints.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Paired controls",
            "",
            statistics.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Localization evidence",
            "",
            reliability.to_markdown(index=False, floatfmt=".6f"),
            "",
            reliability_conclusion["claim"],
            "",
            "The LaChance Cellpose reliability packet failed its movie-level hard "
            "controls and independent causal retracking was not completed. The "
            "C2C12 manual/automatic disagreement therefore supplies context, not a "
            "LaChance measurement floor and not a subtraction from h1 RMSE.",
            "",
            f"Elapsed: `{elapsed_minutes:.2f}` minutes.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    started = time.time()
    for field in (
        "v102_root",
        "v102_summary",
        "v102_movies",
        "out_dir",
        "lachance_reliability",
        "c2c12_audit",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = v157e.parse_csv_ints(args.movies)
    seeds = v157e.parse_csv_ints(args.seeds)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    alphas = v157e.parse_csv_floats(args.alphas)
    bounds = v157e.parse_csv_floats(args.bounds_px)
    local_scales = v157e.parse_csv_floats(args.local_scales_px)

    # Reuse the audited fold-local implementation while keeping a new immutable
    # objective registry and a separate output contract.
    v157h.OBJECTIVES = PROFILES
    all_metrics: list[dict[str, Any]] = []
    all_grids: list[pd.DataFrame] = []
    all_causal: list[pd.DataFrame] = []
    for movie in movies:
        print(f"[v205] strict outer movie {movie}", flush=True)
        metrics, grids, causal = v157h.evaluate_fold(
            args,
            movie,
            seeds,
            profiles,
            alphas,
            bounds,
            local_scales,
        )
        all_metrics.extend(metrics)
        all_grids.extend(grids)
        all_causal.append(causal)

    metrics = pd.DataFrame(all_metrics)
    validation = pd.concat(all_grids, ignore_index=True)
    causal = pd.concat(all_causal, ignore_index=True)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Future donor found in the real packet")
    if int(causal["stale_future_or_nonstale_violations"].sum()) != 0:
        raise RuntimeError("Invalid stale-time donor found")
    if not bool(causal["coherent_wrong_packet"].all()):
        raise RuntimeError("Wrong-cell control lost packet coherence")

    aggregate = aggregate_metrics(metrics)
    normalized_movies = movie_normalized_metrics(metrics, args.v102_movies)
    normalized = normalized_aggregate(normalized_movies)
    pareto = pareto_summary(aggregate, normalized)
    statistics = pairwise_statistics(metrics, args.bootstrap_repeats)
    reliability, reliability_conclusion = reliability_context(
        args.lachance_reliability,
        args.c2c12_audit,
    )

    metrics.to_csv(args.out_dir / "v205_h1_h6_metrics.csv", index=False)
    validation.to_csv(args.out_dir / "v205_validation_grid.csv", index=False)
    aggregate.to_csv(args.out_dir / "v205_h1_h6_aggregate.csv", index=False)
    normalized_movies.to_csv(
        args.out_dir / "v205_normalized_error_by_movie.csv", index=False
    )
    normalized.to_csv(args.out_dir / "v205_normalized_error.csv", index=False)
    pareto.to_csv(args.out_dir / "v205_pareto_points.csv", index=False)
    statistics.to_csv(args.out_dir / "v205_pairwise_statistics.csv", index=False)
    reliability.to_csv(args.out_dir / "v205_localization_context.csv", index=False)
    causal.to_csv(args.out_dir / "v205_causal_audit.csv", index=False)
    plot_evidence(pareto, normalized, args.v102_summary, args.out_dir)

    elapsed = (time.time() - started) / 60.0
    (args.out_dir / "v205_status_report.md").write_text(
        report_text(
            pareto,
            normalized,
            statistics,
            reliability,
            reliability_conclusion,
            elapsed,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__)),
        "protocol": "strict fold-local streaming/receding h1",
        "movies": movies,
        "seeds": seeds,
        "profiles": {
            name: {
                "weights": PROFILES[name][0],
                "validation_h1_guard_percent": PROFILES[name][1],
                "status": (
                    "confirmatory_h1"
                    if name == "lambda_00"
                    else "descriptive"
                ),
            }
            for name in profiles
        },
        "alphas": alphas,
        "bounds_px": bounds,
        "local_scales_px": local_scales,
        "bootstrap_repeats": args.bootstrap_repeats,
        "reliability_conclusion": reliability_conclusion,
        "elapsed_minutes": elapsed,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(v157e.finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "v205_status_report.md", flush=True)


if __name__ == "__main__":
    main()
