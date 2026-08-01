#!/usr/bin/env python3
"""Fast causal feature and route-observability triage for DeepSea v204."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_deepsea_multimodal_validation_v204 as v204  # noqa: E402
import build_deepsea_online_anchor_cache_v204 as cache204  # noqa: E402


KEYS = ["sequence", "frame", "track_id"]
EPS = 1e-8


def parse_strings(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def add_history_features(table: pd.DataFrame, lags: int) -> tuple[pd.DataFrame, list[str]]:
    output = table.copy()
    columns: list[str] = ["x_px", "y_px", "dx_px", "dy_px"]
    grouped = output.groupby(["sequence", "track_id"], sort=False)
    for lag in range(1, lags + 1):
        for component in ("dx_px", "dy_px"):
            name = f"coord_{component}_lag{lag}"
            output[name] = grouped[component].shift(lag)
            columns.append(name)
    output["coord_speed"] = np.sqrt(np.square(output.dx_px) + np.square(output.dy_px))
    output["coord_accel_x"] = output.dx_px - output.coord_dx_px_lag1
    output["coord_accel_y"] = output.dy_px - output.coord_dy_px_lag1
    output["coord_accel"] = np.sqrt(
        np.square(output.coord_accel_x) + np.square(output.coord_accel_y)
    )
    cross = output.coord_dx_px_lag1 * output.dy_px - output.coord_dy_px_lag1 * output.dx_px
    dot = output.coord_dx_px_lag1 * output.dx_px + output.coord_dy_px_lag1 * output.dy_px
    output["coord_turn_sin"] = cross / np.maximum(
        np.sqrt(
            (np.square(output.coord_dx_px_lag1) + np.square(output.coord_dy_px_lag1))
            * (np.square(output.dx_px) + np.square(output.dy_px))
        ),
        EPS,
    )
    output["coord_turn_cos"] = dot / np.maximum(
        np.sqrt(
            (np.square(output.coord_dx_px_lag1) + np.square(output.coord_dy_px_lag1))
            * (np.square(output.dx_px) + np.square(output.dy_px))
        ),
        EPS,
    )
    output["coord_track_age_log"] = np.log1p(output.track_age)
    output["coord_frame"] = output.frame
    columns.extend(
        [
            "coord_speed",
            "coord_accel_x",
            "coord_accel_y",
            "coord_accel",
            "coord_turn_sin",
            "coord_turn_cos",
            "coord_track_age_log",
            "coord_frame",
        ]
    )
    return output, columns


def cap_split(table: pd.DataFrame, split: str, limit: int, seed: int) -> pd.DataFrame:
    subset = table.loc[table.split == split].copy()
    if limit <= 0 or len(subset) <= limit:
        return subset
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    count = 0
    groups = list(subset.groupby(["sequence", "track_id"], sort=False))
    for index in rng.permutation(len(groups)):
        group = groups[int(index)][1]
        if count and count + len(group) > limit:
            continue
        selected.append(group)
        count += len(group)
        if count >= limit:
            break
    return pd.concat(selected, ignore_index=False).sort_index()


def safe_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.nan_to_num(
        table[columns].to_numpy(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def select_state_packet(columns: list[str], packet: str) -> list[str]:
    metadata = [column for column in columns if column.startswith("meta_")]
    state = [column for column in columns if column.startswith("ms_")]
    if packet == "full":
        selected = state
    elif packet == "shape":
        pixel_tokens = (
            "mask_available",
            "component_",
            "centroid_component",
            "centroid_to_nearest",
            "front_",
            "back_",
            "left_",
            "right_",
            "contact",
            "clearance",
            "neighbor_mask",
            "image_boundary",
            "free_space",
            "mask_area_agreement",
            "identity_reliability",
        )
        selected = [
            column for column in state if not any(token in column for token in pixel_tokens)
        ]
    elif packet == "polarity":
        tokens = ("orientation", "front_", "back_", "left_", "right_", "axis_ratio")
        selected = [column for column in state if any(token in column for token in tokens)]
    elif packet == "contact":
        tokens = (
            "contact",
            "clearance",
            "neighbor_mask_density",
            "component_count",
            "image_boundary_distance",
            "free_space",
        )
        selected = [column for column in state if any(token in column for token in tokens)]
    elif packet == "reliability":
        tokens = (
            "mask_available",
            "centroid_component",
            "centroid_to_nearest",
            "mask_area_agreement",
            "identity_reliability",
        )
        selected = [column for column in state if any(token in column for token in tokens)]
    elif packet in {"shape_polarity", "shape_contact", "shape_contact_polarity"}:
        component_packets = packet.split("_")
        selected = []
        for component in component_packets:
            selected.extend(select_state_packet(columns, component))
        selected = [column for column in selected if column.startswith("ms_")]
    else:
        raise ValueError(f"Unknown state packet: {packet}")
    return list(dict.fromkeys(metadata + selected))


def rolling_macro_rmse(rows: pd.DataFrame, target: np.ndarray, prediction: np.ndarray) -> float:
    metrics = v204.rolling_movie_metrics(
        rows.reset_index(drop=True),
        target,
        prediction,
        method="selection",
        control="validation",
        horizons=(6,),
    )
    return float(np.mean([row["component_rmse"] for row in metrics]))


def fit_regression_candidates(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_rows: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[str, Any]] = []
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        candidates.append((f"ridge_a{alpha:g}", Ridge(alpha=alpha)))
    for leaves in (15, 31):
        for l2 in (1.0, 10.0):
            estimator = HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.055,
                max_leaf_nodes=leaves,
                l2_regularization=l2,
                min_samples_leaf=30,
                random_state=seed,
            )
            candidates.append(
                (
                    f"hgbdt_l{leaves}_r{l2:g}",
                    MultiOutputRegressor(estimator, n_jobs=2),
                )
            )
    audit: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for name, model in candidates:
        model.fit(train_x, train_y)
        prediction = np.asarray(model.predict(val_x), dtype=np.float32)
        h1 = v204.component_rmse(val_y, prediction)
        h6 = rolling_macro_rmse(val_rows, val_y, prediction)
        audit.append({"model": name, "val_h1_rmse": h1, "val_h6_movie_macro_rmse": h6})
        score = h6 + 0.1 * h1
        if best is None or score < best["score"]:
            best = {"name": name, "model": model, "score": score}
    assert best is not None
    return best, audit


def route_signatures(table: pd.DataFrame) -> np.ndarray:
    target = table[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
    current = table[["dx_px", "dy_px"]].to_numpy(np.float32)
    residual = target - current
    target_speed = np.linalg.norm(target, axis=1)
    current_speed = np.linalg.norm(current, axis=1)
    cross = current[:, 0] * target[:, 1] - current[:, 1] * target[:, 0]
    dot = np.sum(current * target, axis=1)
    denominator = np.maximum(current_speed * target_speed, EPS)
    return np.column_stack(
        [
            residual,
            np.log1p(target_speed) - np.log1p(current_speed),
            cross / denominator,
            dot / denominator,
        ]
    ).astype(np.float32)


def route_probe(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_signature: np.ndarray,
    test_signature: np.ndarray,
    modes: int,
    seed: int,
) -> dict[str, float]:
    scaler = StandardScaler().fit(train_signature)
    train_state = scaler.transform(train_signature)
    test_state = scaler.transform(test_signature)
    cluster = KMeans(n_clusters=modes, n_init=20, random_state=seed).fit(train_state)
    train_label = cluster.labels_
    test_label = cluster.predict(test_state)
    classifier = HistGradientBoostingClassifier(
        max_iter=240,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=5.0,
        min_samples_leaf=30,
        random_state=seed,
    ).fit(train_x, train_label)
    probabilities = classifier.predict_proba(test_x)
    prediction = np.argmax(probabilities, axis=1)
    top3 = np.argpartition(-probabilities, kth=min(2, modes - 1), axis=1)[:, : min(3, modes)]
    return {
        "route_top1": float(np.mean(prediction == test_label)),
        "route_top3": float(np.mean(np.any(top3 == test_label[:, None], axis=1))),
        "route_macro_f1": float(f1_score(test_label, prediction, average="macro", zero_division=0)),
        "route_usage_entropy": float(
            -np.sum(
                np.bincount(prediction, minlength=modes)
                / len(prediction)
                * np.log(
                    np.maximum(np.bincount(prediction, minlength=modes) / len(prediction), EPS)
                )
            )
        ),
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(args.prepared_dir / "deepsea_tracks.csv")
    tracks = cache204.canonicalize_coordinates(tracks, args.coordinate_unit)
    tracks, coordinate_columns = add_history_features(tracks, args.history_lags)
    valid = tracks[
        ["dx_px", "dy_px", "target_dx_px", "target_dy_px"]
    ].notna().all(axis=1)
    tracks = tracks.loc[valid].copy()
    train = cap_split(tracks, "train", args.max_train_rows, args.seed)
    validation = cap_split(tracks, "val", args.max_val_rows, args.seed + 1)
    test = cap_split(tracks, "test", args.max_test_rows, args.seed + 2)
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("Frozen split contains no usable rows")

    control_dir = args.controls_dir
    if not (control_dir / "deepsea_state_real.csv").exists():
        v204.make_feature_controls(
            args.prepared_dir / "deepsea_state_features.csv",
            args.prepared_dir / "deepsea_tracks.csv",
            control_dir,
            args.seed,
        )
    controls = parse_strings(args.controls)
    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_archive: dict[str, np.ndarray] = {}
    targets = {
        "train": train[["target_dx_px", "target_dy_px"]].to_numpy(np.float32),
        "val": validation[["target_dx_px", "target_dy_px"]].to_numpy(np.float32),
        "test": test[["target_dx_px", "target_dy_px"]].to_numpy(np.float32),
    }
    coordinate_scaler = StandardScaler().fit(safe_matrix(train, coordinate_columns))

    for control in controls:
        state = pd.read_csv(control_dir / f"deepsea_state_{control}.csv")
        state_names = select_state_packet(
            v204.context_columns(state), args.state_packet
        )
        merged: dict[str, pd.DataFrame] = {}
        for split_name, split_table in (
            ("train", train),
            ("val", validation),
            ("test", test),
        ):
            base_table = split_table.drop(
                columns=[column for column in state_names if column in split_table],
                errors="ignore",
            )
            merged[split_name] = base_table.merge(
                state[KEYS + state_names],
                on=KEYS,
                how="left",
                validate="one_to_one",
            )
        state_scaler = StandardScaler().fit(safe_matrix(merged["train"], state_names))
        matrices: dict[str, np.ndarray] = {}
        for split_name in ("train", "val", "test"):
            coordinate = coordinate_scaler.transform(
                safe_matrix(merged[split_name], coordinate_columns)
            )
            state_values = state_scaler.transform(safe_matrix(merged[split_name], state_names))
            matrices[split_name] = np.concatenate([coordinate, state_values], axis=1).astype(
                np.float32
            )

        best, selection = fit_regression_candidates(
            matrices["train"],
            targets["train"],
            matrices["val"],
            targets["val"],
            merged["val"],
            args.seed,
        )
        for row in selection:
            selection_rows.append({"control": control, **row})
        prediction = np.asarray(best["model"].predict(matrices["test"]), dtype=np.float32)
        prediction_archive[f"{control}__prediction"] = prediction
        metrics = v204.rolling_movie_metrics(
            merged["test"],
            targets["test"],
            prediction,
            method=best["name"],
            control=control,
        )
        for row in metrics:
            result_rows.append(row)
        route = route_probe(
            matrices["train"],
            matrices["test"],
            route_signatures(merged["train"]),
            route_signatures(merged["test"]),
            args.route_modes,
            args.seed,
        )
        result_rows.append(
            {
                "method": "route_probe",
                "control": control,
                "sequence": -1,
                "family": "all",
                "video": "all",
                "horizon": 0,
                "component_rmse": np.nan,
                "r2": np.nan,
                "cosine": np.nan,
                "magnitude_ratio": np.nan,
                "n_windows": len(test),
                **route,
            }
        )

    movie = pd.DataFrame(result_rows)
    movie.to_csv(args.out_dir / "v204_feature_triage_movie_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        args.out_dir / "v204_feature_triage_model_selection.csv", index=False
    )
    np.savez_compressed(args.out_dir / "v204_feature_triage_predictions.npz", **prediction_archive)
    aggregate = (
        movie.loc[movie.horizon > 0]
        .groupby(["control", "horizon"], as_index=False)
        .agg(
            movie_macro_rmse=("component_rmse", "mean"),
            movie_macro_r2=("r2", "mean"),
            movies=("sequence", "nunique"),
            windows=("n_windows", "sum"),
        )
    )
    probes = movie.loc[movie.method == "route_probe"].copy()
    aggregate.to_csv(args.out_dir / "v204_feature_triage_summary.csv", index=False)
    probes.to_csv(args.out_dir / "v204_route_observability_probe.csv", index=False)

    h6 = aggregate.loc[aggregate.horizon == 6].set_index("control").movie_macro_rmse
    real_gain = (
        100.0 * (float(h6["zero"]) - float(h6["real"])) / max(float(h6["zero"]), EPS)
        if {"zero", "real"}.issubset(h6.index)
        else float("nan")
    )
    hard = [
        name
        for name in ("row_shuffled", "time_shuffled", "wrong_cell", "wrong_video")
        if name in h6.index
    ]
    real_beats_controls = bool(hard and all(float(h6["real"]) < float(h6[name]) for name in hard))
    decision = {
        "state_packet": args.state_packet,
        "real_vs_zero_h6_gain_pct": real_gain,
        "real_beats_hard_controls": real_beats_controls,
        "feature_gate": bool(real_gain >= 3.0 and real_beats_controls),
        "train_rows": len(train),
        "val_rows": len(validation),
        "test_rows": len(test),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (args.out_dir / "v204_feature_triage_decision.json").write_text(
        json.dumps(finite_json(decision), indent=2), encoding="utf-8"
    )
    print(json.dumps(finite_json(decision), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_multimodal_prepared_v204_2026-07-31",
    )
    parser.add_argument(
        "--controls-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_multimodal_validation_v204_2026-07-31/feature_controls",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_feature_triage_v204_2026-07-31",
    )
    parser.add_argument(
        "--controls",
        default="real,zero,row_shuffled,time_shuffled,wrong_cell,wrong_video,noncausal_capacity",
    )
    parser.add_argument("--history-lags", type=int, default=6)
    parser.add_argument("--route-modes", type=int, default=8)
    parser.add_argument(
        "--state-packet",
        choices=[
            "full",
            "shape",
            "polarity",
            "contact",
            "reliability",
            "shape_polarity",
            "shape_contact",
            "shape_contact_polarity",
        ],
        default="full",
    )
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coordinate-unit",
        choices=["cell_diameter", "pixel"],
        default="cell_diameter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
