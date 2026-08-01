#!/usr/bin/env python3
"""Test LifeAct/shape/contact state as a causal uncertainty observation model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_lifeact_mdck_mechanochemical_state_gate_v207 import (
    EPS,
    add_scale_normalized_columns,
    chronological_splits,
    clean_features,
    controlled_table,
    feature_packets,
    impute_train,
    leave_one_sequence_out_splits,
)


CONTROL_PACKETS = [
    "full_zero_state",
    "full_row_shuffled",
    "full_wrong_cell",
    "full_time_shuffled",
]
CHI2_RADIAL_50 = 1.3862943611198906
CHI2_RADIAL_90 = 4.605170185988092


def fit_mean_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[list[HistGradientBoostingRegressor], float]:
    best_score = math.inf
    best_models: list[HistGradientBoostingRegressor] = []
    for l2 in (1.0, 10.0, 100.0):
        models = [
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=l2,
                random_state=seed + component,
            ).fit(x_train, y_train[:, component])
            for component in range(2)
        ]
        prediction = np.column_stack([model.predict(x_val) for model in models])
        score = float(np.sqrt(np.mean(np.square(prediction - y_val))))
        if score < best_score:
            best_score = score
            best_models = models
    return best_models, best_score


def predict_pair(models: list[HistGradientBoostingRegressor], values: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(values) for model in models])


def uncertainty_metrics(error: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    variance = np.asarray(variance, dtype=float).clip(min=1e-6)
    squared_radius = np.sum(np.square(error), axis=1)
    component_nll = float(
        np.mean(0.5 * np.log(2.0 * np.pi * variance) + squared_radius / (4.0 * variance))
    )
    degrees_of_freedom = 4.0
    student_scale_squared = variance * (
        degrees_of_freedom - 2.0
    ) / degrees_of_freedom
    student_constant = (
        math.lgamma(degrees_of_freedom / 2.0)
        - math.lgamma((degrees_of_freedom + 1.0) / 2.0)
        + 0.5 * math.log(degrees_of_freedom * np.pi)
    )
    student_nll = float(
        np.mean(
            student_constant
            + 0.5 * np.log(student_scale_squared[:, None])
            + 0.5
            * (degrees_of_freedom + 1.0)
            * np.log1p(
                np.square(error)
                / (degrees_of_freedom * student_scale_squared[:, None])
            )
        )
    )
    normalized_radius = squared_radius / variance
    error_norm = np.sqrt(squared_radius)
    sigma = np.sqrt(variance)
    correlation = pd.Series(sigma).corr(pd.Series(error_norm), method="spearman")
    coverage50 = float(np.mean(normalized_radius <= CHI2_RADIAL_50))
    coverage90 = float(np.mean(normalized_radius <= CHI2_RADIAL_90))
    return {
        "gaussian_component_nll": component_nll,
        "student_t4_component_nll": student_nll,
        "coverage50": coverage50,
        "coverage90": coverage90,
        "coverage_error": 0.5 * (abs(coverage50 - 0.50) + abs(coverage90 - 0.90)),
        "uncertainty_error_spearman": float(correlation),
        "mean_sigma": float(np.mean(sigma)),
    }


def evaluate_seed(table: pd.DataFrame, seed: int, scale_normalized: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    packets = feature_packets(scale_normalized)
    splits = chronological_splits(table) + leave_one_sequence_out_splits(table)
    target_px = table[["target_dx", "target_dy"]].to_numpy(float)
    base_px = table[["vx_lag1", "vy_lag1"]].to_numpy(float)
    scale = table["frame_median_diameter"].to_numpy(float).clip(min=EPS)
    if scale_normalized:
        target_model = target_px / scale[:, None]
        base_model = base_px / scale[:, None]
    else:
        target_model = target_px
        base_model = base_px
    residual_model = target_model - base_model

    coordinate_table = add_scale_normalized_columns(table) if scale_normalized else table
    coordinate_columns = packets["coord_only"]
    coordinate_values = clean_features(coordinate_table, coordinate_columns)

    for split_index, split in enumerate(splits):
        x_train, x_val, x_test = impute_train(
            coordinate_values[split.train],
            coordinate_values[split.val],
            coordinate_values[split.test],
        )
        mean_models, mean_val_rmse = fit_mean_models(
            x_train,
            residual_model[split.train],
            x_val,
            residual_model[split.val],
            seed + split_index * 1000,
        )
        val_prediction_model = base_model[split.val] + predict_pair(mean_models, x_val)
        test_prediction_model = base_model[split.test] + predict_pair(mean_models, x_test)
        if scale_normalized:
            val_prediction_px = val_prediction_model * scale[split.val, None]
            test_prediction_px = test_prediction_model * scale[split.test, None]
        else:
            val_prediction_px = val_prediction_model
            test_prediction_px = test_prediction_model
        val_error = target_px[split.val] - val_prediction_px
        test_error = target_px[split.test] - test_prediction_px
        val_log_variance_target = np.log(
            np.mean(np.square(val_error), axis=1).clip(min=1e-4)
        )

        for packet_index, (packet, columns) in enumerate(packets.items()):
            controlled = controlled_table(
                table, packet, seed + split_index * 10000 + packet_index * 97
            )
            if scale_normalized:
                controlled = add_scale_normalized_columns(controlled)
            values = clean_features(controlled, columns)
            x_unc_val, x_unc_test, _ = impute_train(
                values[split.val], values[split.test], values[split.test]
            )
            model = HistGradientBoostingRegressor(
                max_iter=140,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=30.0,
                random_state=seed + packet_index,
            ).fit(x_unc_val, val_log_variance_target)
            val_variance = np.exp(model.predict(x_unc_val).clip(-8.0, 12.0))
            test_variance = np.exp(model.predict(x_unc_test).clip(-8.0, 12.0))
            val_mse = float(np.mean(np.square(val_error)))
            calibration = np.clip(val_mse / max(float(np.mean(val_variance)), EPS), 0.25, 4.0)
            test_variance *= calibration
            rows.append(
                {
                    "seed": seed,
                    "protocol": split.protocol,
                    "fold": split.fold,
                    "packet": packet,
                    "evaluation_scale": (
                        "current_frame_cell_diameter" if scale_normalized else "pixels"
                    ),
                    "n_val": int(split.val.sum()),
                    "n_test": int(split.test.sum()),
                    "mean_val_rmse_model_scale": mean_val_rmse,
                    **uncertainty_metrics(test_error, test_variance),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--scale-normalized", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = pd.read_parquet(args.state_path)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    metrics = pd.concat(
        [evaluate_seed(state, seed, args.scale_normalized) for seed in seeds],
        ignore_index=True,
    )
    aggregate = (
        metrics.groupby(["protocol", "packet"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            folds=("fold", "nunique"),
            nll_mean=("gaussian_component_nll", "mean"),
            nll_std=("gaussian_component_nll", "std"),
            student_t4_nll_mean=("student_t4_component_nll", "mean"),
            student_t4_nll_std=("student_t4_component_nll", "std"),
            coverage50_mean=("coverage50", "mean"),
            coverage90_mean=("coverage90", "mean"),
            coverage_error_mean=("coverage_error", "mean"),
            uncertainty_error_spearman_mean=("uncertainty_error_spearman", "mean"),
        )
    )
    baseline = aggregate[aggregate["packet"].eq("coord_only")][
        ["protocol", "nll_mean", "student_t4_nll_mean"]
    ].rename(
        columns={
            "nll_mean": "coord_nll",
            "student_t4_nll_mean": "coord_student_t4_nll",
        }
    )
    aggregate = aggregate.merge(baseline, on="protocol", how="left")
    aggregate["nll_gain_vs_coord"] = aggregate["coord_nll"] - aggregate["nll_mean"]
    aggregate["student_t4_nll_gain_vs_coord"] = (
        aggregate["coord_student_t4_nll"] - aggregate["student_t4_nll_mean"]
    )
    decisions: list[dict[str, object]] = []
    for protocol in aggregate["protocol"].unique():
        current = aggregate[aggregate["protocol"].eq(protocol)]
        real = current[current["packet"].eq("full_real")].iloc[0]
        controls = current[current["packet"].isin(CONTROL_PACKETS)]
        best_control_nll = float(controls["nll_mean"].min())
        best_control_student_nll = float(controls["student_t4_nll_mean"].min())
        decisions.append(
            {
                "protocol": protocol,
                "real_nll": float(real["nll_mean"]),
                "coord_nll": float(real["coord_nll"]),
                "best_control_nll": best_control_nll,
                "nll_gain_vs_coord": float(real["coord_nll"] - real["nll_mean"]),
                "nll_gain_vs_best_control": best_control_nll - float(real["nll_mean"]),
                "real_student_t4_nll": float(real["student_t4_nll_mean"]),
                "coord_student_t4_nll": float(real["coord_student_t4_nll"]),
                "best_control_student_t4_nll": best_control_student_nll,
                "student_t4_gain_vs_coord": float(
                    real["coord_student_t4_nll"] - real["student_t4_nll_mean"]
                ),
                "student_t4_gain_vs_best_control": best_control_student_nll
                - float(real["student_t4_nll_mean"]),
                "coverage_error": float(real["coverage_error_mean"]),
                "uncertainty_error_spearman": float(
                    real["uncertainty_error_spearman_mean"]
                ),
                "pass": bool(
                    real["nll_mean"] < real["coord_nll"]
                    and real["nll_mean"] < best_control_nll
                    and real["student_t4_nll_mean"] < real["coord_student_t4_nll"]
                    and real["student_t4_nll_mean"] < best_control_student_nll
                ),
            }
        )
    decision = pd.DataFrame(decisions)
    metrics.to_csv(args.out_dir / "v208_uncertainty_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "v208_uncertainty_aggregate.csv", index=False)
    decision.to_csv(args.out_dir / "v208_uncertainty_decision.csv", index=False)
    report = [
        "# LifeAct-MDCK causal uncertainty gate v208",
        "",
        "The coordinate mean is frozen within each split. A scale head is trained only on validation errors and causal features, then evaluated on the untouched test period or sequence.",
        "",
        "## Decision",
        "",
        decision.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Aggregate",
        "",
        aggregate.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.out_dir / "v208_uncertainty_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
