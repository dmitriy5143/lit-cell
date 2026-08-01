#!/usr/bin/env python3
"""Fit validation-selected Ridge and HGBDT coordinate baselines on DeepSea."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_deepsea_online_anchor_cache_v204 as cache204  # noqa: E402
import run_deepsea_feature_triage_v204 as triage  # noqa: E402
import run_deepsea_multimodal_validation_v204 as v204  # noqa: E402


def candidates(seed: int) -> dict[str, list[tuple[str, Any]]]:
    ridge = [
        (f"ridge_a{alpha:g}", Ridge(alpha=alpha))
        for alpha in (1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0)
    ]
    hgbdt: list[tuple[str, Any]] = []
    for leaves in (15, 31):
        for regularization in (1.0, 10.0):
            estimator = HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.055,
                max_leaf_nodes=leaves,
                l2_regularization=regularization,
                min_samples_leaf=30,
                random_state=seed,
            )
            hgbdt.append(
                (
                    f"hgbdt_l{leaves}_r{regularization:g}",
                    MultiOutputRegressor(estimator, n_jobs=2),
                )
            )
    return {"ridge": ridge, "hgbdt": hgbdt}


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(
        args.prepared_dir / "deepsea_tracks.csv",
        low_memory=False,
    )
    tracks = cache204.canonicalize_coordinates(tracks, args.coordinate_unit)
    tracks, feature_names = triage.add_history_features(
        tracks,
        args.history_lags,
    )
    tracks = tracks.loc[
        tracks[["dx_px", "dy_px", "target_dx_px", "target_dy_px"]]
        .notna()
        .all(axis=1)
    ].copy()
    splits = {
        "train": tracks.loc[tracks.split == "train"].copy(),
        "validation": tracks.loc[tracks.split == "val"].copy(),
        "test": tracks.loc[tracks.split == "test"].copy(),
    }
    scaler = StandardScaler().fit(
        triage.safe_matrix(splits["train"], feature_names)
    )
    matrices = {
        name: scaler.transform(triage.safe_matrix(table, feature_names))
        for name, table in splits.items()
    }
    targets = {
        name: table[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
        for name, table in splits.items()
    }
    selection_rows: list[dict[str, Any]] = []
    movie_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for family, family_candidates in candidates(args.seed).items():
        evaluated: list[tuple[float, str, Any]] = []
        for name, model in family_candidates:
            model.fit(matrices["train"], targets["train"])
            validation_prediction = np.asarray(
                model.predict(matrices["validation"]),
                dtype=np.float32,
            )
            h1 = v204.component_rmse(
                targets["validation"],
                validation_prediction,
            )
            h6 = triage.rolling_macro_rmse(
                splits["validation"],
                targets["validation"],
                validation_prediction,
            )
            score = h6 + 0.1 * h1
            selection_rows.append(
                {
                    "model_family": family,
                    "model": name,
                    "validation_h1_rmse": h1,
                    "validation_h6_movie_macro_rmse": h6,
                    "selection_score": score,
                }
            )
            evaluated.append((score, name, model))
        _, winner_name, winner = min(evaluated, key=lambda item: item[0])
        prediction = np.asarray(
            winner.predict(matrices["test"]),
            dtype=np.float32,
        )
        predictions[f"{family}__prediction"] = prediction
        metrics = v204.rolling_movie_metrics(
            splits["test"],
            targets["test"],
            prediction,
            method=winner_name,
            control=family,
        )
        for row in metrics:
            row["model_family"] = family
            row["metric_unit"] = args.coordinate_unit
        movie_rows.extend(metrics)
    selection = pd.DataFrame(selection_rows)
    movie = pd.DataFrame(movie_rows)
    summary = (
        movie.groupby(["model_family", "method", "horizon"], as_index=False)
        .agg(
            movie_macro_rmse=("component_rmse", "mean"),
            movie_macro_r2=("r2", "mean"),
            movies=("sequence", "nunique"),
            windows=("n_windows", "sum"),
        )
    )
    selection.to_csv(
        args.out_dir / "v204_tabular_model_selection.csv",
        index=False,
    )
    movie.to_csv(
        args.out_dir / "v204_tabular_movie_metrics.csv",
        index=False,
    )
    summary.to_csv(
        args.out_dir / "v204_tabular_summary.csv",
        index=False,
    )
    np.savez_compressed(
        args.out_dir / "v204_tabular_predictions.npz",
        **predictions,
    )
    contract = {
        "prepared_dir": str(args.prepared_dir.resolve()),
        "coordinate_unit": args.coordinate_unit,
        "feature_names": feature_names,
        "selection_split": "validation_movies",
        "test_use": "single_evaluation_after_family_specific_selection",
        "test_movies": sorted(int(item) for item in splits["test"].sequence.unique()),
    }
    (args.out_dir / "v204_tabular_contract.json").write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "deepsea_coordinate_prepared_v204_2026-07-31"
        ),
    )
    parser.add_argument(
        "--coordinate-unit",
        choices=["pixel", "cell_diameter"],
        default="cell_diameter",
    )
    parser.add_argument("--history-lags", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
