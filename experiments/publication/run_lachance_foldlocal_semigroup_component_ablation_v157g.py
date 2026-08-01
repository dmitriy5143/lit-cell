#!/usr/bin/env python3
"""Strict component ablation of the fold-local semigroup update.

The v157e result establishes a large streaming h6 gain, but its update packet
contains own-cell, global-frame, and multiscale local-neighbour innovation
statistics.  This runner identifies which component actually carries the
gain.  Every variant receives the same fold-local v97 replay, validation
selection, refit, h1 guard, and untouched outer test movie.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "outputs" / "lachance_foldlocal_semigroup_component_ablation_v157g"
)
EPS = 1e-8
VARIANTS = (
    "full_scaled",
    "full_raw",
    "own_scaled",
    "global_scaled",
    "local_mean_scaled",
    "local_full_scaled",
    "own_global_scaled",
    "own_local_scaled",
    "global_local_scaled",
    "full_no_local_uncertainty_scaled",
)


@dataclass
class GenericRidge:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray


@dataclass
class GenericSelection:
    alpha: float
    bound_px: float
    validation_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=v157e.DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=157_007)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_contract(
    names: list[str],
    variant: str,
) -> tuple[np.ndarray, bool]:
    own = np.asarray([name.startswith("own_") for name in names])
    global_state = np.asarray(
        [name.startswith("global_") for name in names]
    )
    local = np.asarray([name.startswith("local_") for name in names])
    local_mean = np.asarray(
        [bool(re.match(r"^local_\d+_[xy]$", name)) for name in names]
    )
    local_uncertainty = local & ~local_mean
    if variant == "full_scaled":
        mask, scaled = np.ones(len(names), dtype=bool), True
    elif variant == "full_raw":
        mask, scaled = np.ones(len(names), dtype=bool), False
    elif variant == "own_scaled":
        mask, scaled = own, True
    elif variant == "global_scaled":
        mask, scaled = global_state, True
    elif variant == "local_mean_scaled":
        mask, scaled = local_mean, True
    elif variant == "local_full_scaled":
        mask, scaled = local, True
    elif variant == "own_global_scaled":
        mask, scaled = own | global_state, True
    elif variant == "own_local_scaled":
        mask, scaled = own | local, True
    elif variant == "global_local_scaled":
        mask, scaled = global_state | local, True
    elif variant == "full_no_local_uncertainty_scaled":
        mask, scaled = ~local_uncertainty, True
    else:
        raise ValueError(f"Unknown component variant: {variant}")
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError(f"Variant {variant} selected no features")
    return indices, scaled


def design(
    payload: v157e.UpdatePayload,
    indices: np.ndarray,
    scaled: bool,
    control: str = "real",
) -> np.ndarray:
    packet = np.asarray(getattr(payload, control), dtype=np.float64)[
        :, indices
    ]
    if not scaled:
        return packet
    scale = payload.base.scale
    return np.column_stack(
        [
            packet,
            packet * scale[:, 0:1],
            packet * scale[:, 1:2],
        ]
    )


def normalization(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    indices: np.ndarray,
    scaled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate(
        [design(payloads[movie], indices, scaled) for movie in movies]
    )
    return matrix.mean(axis=0), np.maximum(matrix.std(axis=0), EPS)


def training_data(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    indices: np.ndarray,
    scaled: bool,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    for movie in movies:
        payload = payloads[movie]
        normalized = (
            design(payload, indices, scaled) - mean
        ) / scale
        per_step = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        residual = payload.base.target - payload.base.mean
        for horizon in v157e.HORIZONS:
            windows = v157e.consecutive_windows(
                payload.base.rows,
                horizon,
            )
            feature_blocks.append(per_step[windows].sum(axis=1))
            target_blocks.append(residual[windows].sum(axis=1))
            weight_blocks.append(
                np.full(
                    len(windows),
                    v157e.H1_STRICT_WEIGHTS[horizon] / len(windows),
                    dtype=np.float64,
                )
            )
    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    weights = np.concatenate(weight_blocks)
    weights *= len(weights) / max(float(weights.sum()), EPS)
    return features, targets, weights


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> GenericRidge:
    root_weight = np.sqrt(weights / max(float(weights.mean()), EPS))[
        :, None
    ]
    weighted_x = np.nan_to_num(features) * root_weight
    weighted_y = np.nan_to_num(target) * root_weight
    gram = weighted_x.T @ weighted_x
    rhs = weighted_x.T @ weighted_y
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return GenericRidge(mean, scale, coefficients)


def predict(
    model: GenericRidge,
    payload: v157e.UpdatePayload,
    indices: np.ndarray,
    scaled: bool,
    control: str,
    bound_px: float,
) -> np.ndarray:
    normalized = (
        design(payload, indices, scaled, control) - model.mean
    ) / model.scale
    augmented = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    correction = augmented @ model.coefficients
    return payload.base.mean + v157e.bounded_update(correction, bound_px)


def fit_for_movies(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    indices: np.ndarray,
    scaled: bool,
    alpha: float,
) -> GenericRidge:
    mean, scale = normalization(
        payloads,
        movies,
        indices,
        scaled,
    )
    features, target, weights = training_data(
        payloads,
        movies,
        indices,
        scaled,
        mean,
        scale,
    )
    return fit_ridge(
        features,
        target,
        weights,
        alpha,
        mean,
        scale,
    )


def select_variant(
    payloads: dict[int, v157e.UpdatePayload],
    train_movies: list[int],
    validation_movie: int,
    indices: np.ndarray,
    scaled: bool,
    alphas: list[float],
    bounds: list[float],
) -> tuple[GenericSelection, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    validation = payloads[validation_movie]
    for alpha in alphas:
        model = fit_for_movies(
            payloads,
            train_movies,
            indices,
            scaled,
            alpha,
        )
        for bound in bounds:
            prediction = predict(
                model,
                validation,
                indices,
                scaled,
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
                "validation_score": v157e.objective_score(metrics),
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
    eligible = grid[grid["h1_gain_percent"].ge(-0.5)]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        ["validation_score", "h1_component_rmse", "bound_px", "alpha"]
    ).iloc[0]
    return (
        GenericSelection(
            alpha=float(best["alpha"]),
            bound_px=float(best["bound_px"]),
            validation_score=float(best["validation_score"]),
        ),
        grid,
    )


def coefficient_rows(
    test_movie: int,
    variant: str,
    model: GenericRidge,
    selected_names: list[str],
    scaled: bool,
) -> list[dict[str, Any]]:
    labels = (
        selected_names
        if not scaled
        else selected_names
        + [f"{name}__scale_x" for name in selected_names]
        + [f"{name}__scale_y" for name in selected_names]
    )
    coefficients = model.coefficients[:-1]
    output: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        output.append(
            {
                "test_movie": test_movie,
                "variant": variant,
                "feature": label,
                "coefficient_x": coefficients[index, 0],
                "coefficient_y": coefficients[index, 1],
                "coefficient_norm": float(
                    np.linalg.norm(coefficients[index])
                ),
            }
        )
    return output


def evaluate_fold(
    args: argparse.Namespace,
    test_movie: int,
    seeds: list[int],
    variants: list[str],
    alphas: list[float],
    bounds: list[float],
    local_scales: list[float],
) -> tuple[
    list[dict[str, Any]],
    list[pd.DataFrame],
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
    metrics: list[dict[str, Any]] = []
    grids: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    baseline_rows = v157e.metric_rows(
        test,
        test.base.mean,
        "no_update",
        None,
    )
    for row in baseline_rows:
        row["variant"] = "no_update"
        row["protocol"] = "strict_fold_local_streaming"
    metrics.extend(baseline_rows)

    for variant in variants:
        indices, scaled = variant_contract(test.feature_names, variant)
        selection, grid = select_variant(
            payloads,
            train_movies,
            validation_movie,
            indices,
            scaled,
            alphas,
            bounds,
        )
        model = fit_for_movies(
            payloads,
            train_movies + [validation_movie],
            indices,
            scaled,
            selection.alpha,
        )
        prediction = predict(
            model,
            test,
            indices,
            scaled,
            "real",
            selection.bound_px,
        )
        rows = v157e.metric_rows(
            test,
            prediction,
            variant,
            None,
        )
        for row in rows:
            row["variant"] = variant
            row["selected_alpha"] = selection.alpha
            row["selected_bound_px"] = selection.bound_px
            row["validation_score"] = selection.validation_score
            row["protocol"] = "strict_fold_local_streaming"
        metrics.extend(rows)
        grid.insert(0, "test_movie", test_movie)
        grid.insert(1, "validation_movie", validation_movie)
        grid.insert(2, "variant", variant)
        grids.append(grid)
        selected_names = [
            test.feature_names[index] for index in indices
        ]
        coefficients.extend(
            coefficient_rows(
                test_movie,
                variant,
                model,
                selected_names,
                scaled,
            )
        )
        selections.append(
            {
                "test_movie": test_movie,
                "validation_movie": validation_movie,
                "train_movies": ",".join(map(str, train_movies)),
                "variant": variant,
                "features": len(indices),
                "scaled_interactions": scaled,
                "alpha": selection.alpha,
                "bound_px": selection.bound_px,
                "validation_score": selection.validation_score,
            }
        )

    full_indices, full_scaled = variant_contract(
        test.feature_names,
        "full_scaled",
    )
    full_selection_row = next(
        row for row in selections if row["variant"] == "full_scaled"
    )
    full_model = fit_for_movies(
        payloads,
        train_movies + [validation_movie],
        full_indices,
        full_scaled,
        float(full_selection_row["alpha"]),
    )
    for control in ("wrong_cell", "stale_time"):
        prediction = predict(
            full_model,
            test,
            full_indices,
            full_scaled,
            control,
            float(full_selection_row["bound_px"]),
        )
        rows = v157e.metric_rows(
            test,
            prediction,
            f"full_scaled_{control}",
            None,
        )
        for row in rows:
            row["variant"] = f"full_scaled_{control}"
            row["protocol"] = "strict_fold_local_streaming"
        metrics.extend(rows)
    causal = v157e.build_causal_audit(
        payloads,
        test_movie,
        validation_movie,
        train_movies,
    )
    return metrics, grids, coefficients, selections, causal


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["variant", "horizon"], as_index=False)
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


def decision_rows(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    h6 = aggregate_frame[
        aggregate_frame["horizon"].eq(6)
    ].set_index("variant")
    full = float(h6.loc["full_scaled", "component_rmse_mean"])
    own = float(h6.loc["own_scaled", "component_rmse_mean"])
    own_global = float(
        h6.loc["own_global_scaled", "component_rmse_mean"]
    )
    own_local = float(
        h6.loc["own_local_scaled", "component_rmse_mean"]
    )
    raw = float(h6.loc["full_raw", "component_rmse_mean"])
    no_update = float(h6.loc["no_update", "component_rmse_mean"])
    local_increment = 100.0 * (own_global - full) / max(own_global, EPS)
    global_increment = 100.0 * (own_local - full) / max(own_local, EPS)
    scaling_increment = 100.0 * (raw - full) / max(raw, EPS)
    total = 100.0 * (no_update - full) / max(no_update, EPS)
    return pd.DataFrame(
        [
            {
                "total_h6_gain_percent": total,
                "local_increment_over_own_global_percent": local_increment,
                "global_increment_over_own_local_percent": global_increment,
                "scale_interaction_increment_percent": scaling_increment,
                "local_point_gate_pass": local_increment >= 1.0,
                "global_point_gate_pass": global_increment >= 1.0,
                "scale_interaction_gate_pass": scaling_increment >= 1.0,
            }
        ]
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    args.v102_root = args.v102_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = v157e.parse_csv_ints(args.movies)
    seeds = v157e.parse_csv_ints(args.seeds)
    variants = [
        token.strip()
        for token in str(args.variants).split(",")
        if token.strip()
    ]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if "full_scaled" not in variants:
        raise ValueError("full_scaled is required for controls")
    alphas = v157e.parse_csv_floats(args.alphas)
    bounds = v157e.parse_csv_floats(args.bounds_px)
    local_scales = v157e.parse_csv_floats(args.local_scales_px)

    all_metrics: list[dict[str, Any]] = []
    all_grids: list[pd.DataFrame] = []
    all_coefficients: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_causal: list[pd.DataFrame] = []
    for movie in movies:
        print(f"[v157g] strict outer movie {movie}", flush=True)
        metrics, grids, coefficients, selections, causal = evaluate_fold(
            args,
            movie,
            seeds,
            variants,
            alphas,
            bounds,
            local_scales,
        )
        all_metrics.extend(metrics)
        all_grids.extend(grids)
        all_coefficients.extend(coefficients)
        all_selections.extend(selections)
        all_causal.append(causal)
    metrics_frame = pd.DataFrame(all_metrics)
    aggregate_frame = aggregate(metrics_frame)
    decision = decision_rows(aggregate_frame)
    causal = pd.concat(all_causal, ignore_index=True)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Future donor found")
    if int(causal["stale_future_or_nonstale_violations"].sum()) != 0:
        raise RuntimeError("Invalid stale donor found")
    if not bool(causal["coherent_wrong_packet"].all()):
        raise RuntimeError("Wrong-cell packet coherence failed")

    metrics_frame.to_csv(
        args.out_dir / "v157g_component_metrics.csv",
        index=False,
    )
    aggregate_frame.to_csv(
        args.out_dir / "v157g_component_aggregate.csv",
        index=False,
    )
    pd.concat(all_grids, ignore_index=True).to_csv(
        args.out_dir / "v157g_validation_grid.csv",
        index=False,
    )
    pd.DataFrame(all_coefficients).to_csv(
        args.out_dir / "v157g_coefficients.csv",
        index=False,
    )
    pd.DataFrame(all_selections).to_csv(
        args.out_dir / "v157g_selections.csv",
        index=False,
    )
    causal.to_csv(args.out_dir / "v157g_causal_audit.csv", index=False)
    decision.to_csv(args.out_dir / "v157g_decision.csv", index=False)
    row = decision.iloc[0]
    h6 = aggregate_frame[aggregate_frame["horizon"].eq(6)].sort_values(
        "component_rmse_mean"
    )
    report = [
        "# v157g Strict Semigroup Component Ablation",
        "",
        f"- total full update h6 gain: `{row['total_h6_gain_percent']:+.3f}%`",
        f"- local increment over own+global: `{row['local_increment_over_own_global_percent']:+.3f}%`",
        f"- global increment over own+local: `{row['global_increment_over_own_local_percent']:+.3f}%`",
        f"- uncertainty-scale interaction increment: `{row['scale_interaction_increment_percent']:+.3f}%`",
        "",
        "## h6 ranking",
        "",
        h6[
            [
                "variant",
                "component_rmse_mean",
                "r2_mean",
                "gain_percent_mean",
                "movies_improved",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "Each ablation was independently selected on the fold validation movie",
        "and refit on train+validation before the untouched outer test movie.",
        "h6 is streaming/receding-h1, not open-loop.",
        "",
        f"Elapsed: `{(time.time() - started) / 60.0:.2f}` minutes.",
    ]
    (args.out_dir / "v157g_status_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **vars(args),
        "source_sha256": file_sha256(Path(__file__)),
        "protocol": "strict fold-local streaming/receding h1",
        "target_or_future_inference_feature": False,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(v157e.finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "v157g_status_report.md", flush=True)


if __name__ == "__main__":
    main()
