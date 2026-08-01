#!/usr/bin/env python3
"""Run the preregistered DeepSea coordinate and privileged-state ladder.

This is an orchestration and evidence runner.  It invokes the unchanged v97
sequential forecaster on the frozen DeepSea cache, generates all state
controls before training, and recomputes movie-level rolling metrics directly
from saved one-step predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402


KEYS = ["sequence", "frame", "track_id"]
EPS = 1e-8


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def state_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if column.startswith("ms_") and pd.api.types.is_numeric_dtype(table[column])
    ]


def context_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if (column.startswith("meta_") or column.startswith("ms_"))
        and pd.api.types.is_numeric_dtype(table[column])
    ]


def cyclic_track_shuffle(
    table: pd.DataFrame, columns: list[str], rng: np.random.Generator
) -> np.ndarray:
    values = table[columns].to_numpy(np.float32)
    output = values.copy()
    for _, raw_indices in table.groupby(["sequence", "track_id"], sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        if len(indices) < 2:
            output[indices] = 0.0
            continue
        shift = int(rng.integers(1, len(indices)))
        output[indices] = values[np.roll(indices, shift)]
    return output


def same_frame_wrong_cell(
    table: pd.DataFrame, columns: list[str], rng: np.random.Generator
) -> np.ndarray:
    values = table[columns].to_numpy(np.float32)
    output = values.copy()
    for _, raw_indices in table.groupby(["sequence", "frame"], sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        if len(indices) < 2:
            output[indices] = 0.0
            continue
        shift = int(rng.integers(1, len(indices)))
        output[indices] = values[np.roll(indices, shift)]
    return output


def within_split_row_shuffle(
    table: pd.DataFrame, columns: list[str], rng: np.random.Generator
) -> np.ndarray:
    values = table[columns].to_numpy(np.float32)
    output = values.copy()
    for _, raw_indices in table.groupby(["split", "family"], sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        output[indices] = values[rng.permutation(indices)]
    return output


def wrong_video_state(
    table: pd.DataFrame, columns: list[str], rng: np.random.Generator
) -> np.ndarray:
    values = table[columns].to_numpy(np.float32)
    output = values.copy()
    for _, raw_indices in table.groupby(["split", "family"], sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        videos = sorted(table.loc[indices, "sequence"].astype(int).unique())
        if len(videos) < 2:
            output[indices] = values[rng.permutation(indices)]
            continue
        for source_index, source_video in enumerate(videos):
            target_video = videos[(source_index + 1) % len(videos)]
            source_rows = indices[
                table.loc[indices, "sequence"].to_numpy(np.int64) == source_video
            ]
            target_rows = indices[
                table.loc[indices, "sequence"].to_numpy(np.int64) == target_video
            ]
            sampled = rng.choice(target_rows, size=len(source_rows), replace=len(target_rows) < len(source_rows))
            output[source_rows] = values[sampled]
    return output


def make_feature_controls(
    state_path: Path,
    tracks_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    state = pd.read_csv(state_path)
    metadata = pd.read_csv(
        tracks_path,
        usecols=lambda column: column in set(KEYS + ["family", "video", "split"]),
    ).drop_duplicates(KEYS)
    state = metadata.merge(state, on=KEYS, how="left", suffixes=("", "_state"), validate="one_to_one")
    for column in ("family", "video", "split"):
        state[column] = state[column].fillna(state.get(f"{column}_state"))
    columns = state_columns(state)
    metadata_columns = [
        column
        for column in state.columns
        if column.startswith("meta_") and pd.api.types.is_numeric_dtype(state[column])
    ]
    if not columns:
        raise RuntimeError(f"No numeric ms_ state features in {state_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = state[KEYS + ["family", "video", "split"] + metadata_columns].copy()
    real_values = state[columns].to_numpy(np.float32)
    variants: dict[str, np.ndarray] = {
        "real": real_values,
        "zero": np.zeros_like(real_values),
        "row_shuffled": within_split_row_shuffle(state, columns, rng),
        "time_shuffled": cyclic_track_shuffle(state, columns, rng),
        "wrong_cell": same_frame_wrong_cell(state, columns, rng),
        "wrong_video": wrong_video_state(state, columns, rng),
    }
    # Deliberately non-causal positive control.  It is never a candidate model.
    target = pd.read_csv(
        tracks_path,
        usecols=KEYS + ["target_dx_px", "target_dy_px"],
    )
    target = base[KEYS].merge(target, on=KEYS, how="left", validate="one_to_one")
    target_xy = target[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
    target_xy = np.nan_to_num(target_xy, nan=0.0)
    capacity = np.zeros_like(real_values)
    capacity[:, : min(2, capacity.shape[1])] = target_xy[:, : min(2, capacity.shape[1])]
    if capacity.shape[1] > 2:
        capacity[:, 2:] = rng.normal(0.0, 0.01, size=capacity[:, 2:].shape)
    variants["noncausal_capacity"] = capacity

    paths: dict[str, Path] = {}
    audit_rows: list[dict[str, Any]] = []
    for name, values in variants.items():
        packet = base[KEYS + metadata_columns].copy()
        packet[columns] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        path = output_dir / f"deepsea_state_{name}.csv"
        packet.to_csv(path, index=False)
        paths[name] = path
        audit_rows.append(
            {
                "control": name,
                "rows": len(packet),
                "features": len(columns),
                "metadata_features": len(metadata_columns),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "nonzero_fraction": float(np.mean(np.abs(values) > 1e-8)),
                "explicitly_noncausal": name == "noncausal_capacity",
            }
        )
    pd.DataFrame(audit_rows).to_csv(output_dir / "v204_control_construction_audit.csv", index=False)
    (output_dir / "v204_state_feature_names.json").write_text(
        json.dumps(columns, indent=2), encoding="utf-8"
    )
    return paths


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(command)}\n{tail}")


def v97_command(
    args: argparse.Namespace,
    feature_path: Path,
    output_dir: Path,
    variants: str,
    epochs: int,
    skip_recurrent: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(ROOT / "scripts/run_lachance_causal_innovation_state_space_v97.py"),
        "--anchor-cache",
        str(args.cache_dir),
        "--features",
        str(feature_path),
        "--out-dir",
        str(output_dir),
        "--variants",
        variants,
        "--evaluation-variant",
        "auto",
        "--context-quotas",
        f"meta_:{args.max_metadata_features},ms_:{args.max_state_features}",
        "--hidden",
        str(args.hidden),
        "--history-lags",
        "6",
        "--epochs",
        str(epochs),
        "--patience",
        str(args.patience),
        "--baseline-epochs",
        str(args.baseline_epochs),
        "--bootstrap-repeats",
        str(args.bootstrap_repeats),
        "--train-coordinate-noise-px",
        str(args.train_coordinate_noise),
        "--coordinate-noise-grid",
        args.coordinate_noise_grid,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if skip_recurrent:
        command.append("--skip-recurrent-baselines")
    if args.one_step_scaler:
        command.append("--one-step-scaler")
    if args.smoke:
        command.append("--smoke")
    return command


def component_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def vector_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    sse = float(np.sum(np.square(target - prediction)))
    centered = target - np.mean(target, axis=0, keepdims=True)
    return float(1.0 - sse / max(float(np.sum(np.square(centered))), EPS))


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.maximum(
        np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1), EPS
    )
    return float(np.mean(np.sum(target * prediction, axis=1) / denominator))


def rolling_movie_metrics(
    rows: pd.DataFrame,
    target_h1: np.ndarray,
    prediction_h1: np.ndarray,
    method: str,
    control: str,
    horizons: tuple[int, ...] = (1, 2, 4, 6),
) -> list[dict[str, Any]]:
    local = rows.reset_index(drop=True).copy()
    local["_row"] = np.arange(len(local), dtype=np.int64)
    records: list[dict[str, Any]] = []
    for sequence, movie_rows in local.groupby("sequence", sort=True):
        movie_rows = movie_rows.sort_values(["track_id", "frame"])
        for horizon in horizons:
            target_parts: list[np.ndarray] = []
            prediction_parts: list[np.ndarray] = []
            for _, track in movie_rows.groupby("track_id", sort=False):
                indices = track._row.to_numpy(np.int64)
                frames = track.frame.to_numpy(np.int64)
                for start in range(0, len(indices) - horizon + 1):
                    window = slice(start, start + horizon)
                    if not np.all(np.diff(frames[window]) == 1):
                        continue
                    selected = indices[window]
                    target_parts.append(np.sum(target_h1[selected], axis=0))
                    prediction_parts.append(np.sum(prediction_h1[selected], axis=0))
            if not target_parts:
                continue
            target = np.asarray(target_parts, dtype=np.float32)
            prediction = np.asarray(prediction_parts, dtype=np.float32)
            family = str(movie_rows.family.iloc[0]) if "family" in movie_rows else "unknown"
            video = str(movie_rows.video.iloc[0]) if "video" in movie_rows else str(sequence)
            records.append(
                {
                    "method": method,
                    "control": control,
                    "sequence": int(sequence),
                    "family": family,
                    "video": video,
                    "horizon": int(horizon),
                    "component_rmse": component_rmse(target, prediction),
                    "r2": vector_r2(target, prediction),
                    "cosine": cosine(target, prediction),
                    "magnitude_ratio": float(
                        np.mean(np.linalg.norm(prediction, axis=1))
                        / max(float(np.mean(np.linalg.norm(target, axis=1))), EPS)
                    ),
                    "n_windows": len(target),
                }
            )
    return records


def collect_predictions(
    cache_dir: Path,
    run_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, test = v85.load_anchor_cache(cache_dir)
    rows = test.rows.reset_index(drop=True)
    target_h1 = test.target_steps[:, 0].astype(np.float32)
    records: list[dict[str, Any]] = []
    for control, run_dir in run_dirs.items():
        archive = np.load(run_dir / "v97_predictions.npz", allow_pickle=False)
        for key in archive.files:
            if not key.endswith("__prediction"):
                continue
            method = key.removesuffix("__prediction")
            prediction = archive[key].astype(np.float32)
            if prediction.shape != target_h1.shape:
                raise RuntimeError(f"Prediction shape mismatch for {control}/{method}: {prediction.shape}")
            records.extend(rolling_movie_metrics(rows, target_h1, prediction, method, control))
        if control == "real":
            for key in archive.files:
                if not key.startswith("baseline__"):
                    continue
                method = key.removeprefix("baseline__")
                prediction = archive[key].astype(np.float32)
                records.extend(rolling_movie_metrics(rows, target_h1, prediction, method, control))
    movie = pd.DataFrame(records)
    macro = (
        movie.groupby(["method", "control", "horizon"], as_index=False)
        .agg(
            movie_macro_rmse=("component_rmse", "mean"),
            movie_macro_rmse_std=("component_rmse", "std"),
            movie_macro_r2=("r2", "mean"),
            movie_macro_cosine=("cosine", "mean"),
            movie_macro_magnitude_ratio=("magnitude_ratio", "mean"),
            movies=("sequence", "nunique"),
            windows=("n_windows", "sum"),
        )
    )
    return movie, macro


def paired_movie_bootstrap(
    movie: pd.DataFrame,
    baseline_method: str,
    candidate_method: str,
    control: str,
    horizon: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    baseline = movie[
        (movie.method == baseline_method)
        & (movie.control == "real")
        & (movie.horizon == horizon)
    ][["sequence", "component_rmse"]].rename(columns={"component_rmse": "baseline"})
    candidate = movie[
        (movie.method == candidate_method)
        & (movie.control == control)
        & (movie.horizon == horizon)
    ][["sequence", "component_rmse"]].rename(columns={"component_rmse": "candidate"})
    paired = baseline.merge(candidate, on="sequence", validate="one_to_one")
    if len(paired) < 2:
        return {"n_movies": len(paired), "mean_gain_pct": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    baseline_values = paired.baseline.to_numpy(float)
    candidate_values = paired.candidate.to_numpy(float)
    gain = 100.0 * (baseline_values - candidate_values) / np.maximum(
        baseline_values, EPS
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(gain), size=len(gain))
        baseline_mean = float(np.mean(baseline_values[indices]))
        candidate_mean = float(np.mean(candidate_values[indices]))
        samples[repeat] = 100.0 * (
            baseline_mean - candidate_mean
        ) / max(baseline_mean, EPS)
    return {
        "n_movies": len(paired),
        "mean_gain_pct": float(
            100.0
            * (np.mean(baseline_values) - np.mean(candidate_values))
            / max(float(np.mean(baseline_values)), EPS)
        ),
        "median_gain_pct": float(np.median(gain)),
        "positive_movies": int(np.sum(gain > 0)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = make_feature_controls(
        args.prepared_dir / "deepsea_state_features.csv",
        args.prepared_dir / "deepsea_tracks.csv",
        args.out_dir / "feature_controls",
        args.seed,
    )
    controls = [name.strip() for name in args.controls.split(",") if name.strip()]
    unknown = sorted(set(controls) - set(feature_paths))
    if unknown:
        raise ValueError(f"Unknown controls: {unknown}")

    run_dirs: dict[str, Path] = {}
    for control in controls:
        run_dir = args.out_dir / f"v97_{control}"
        run_dirs[control] = run_dir
        prediction_path = run_dir / "v97_predictions.npz"
        if prediction_path.exists() and not args.force:
            continue
        if control == "real":
            variants = args.real_variants
            epochs = args.epochs
            skip_recurrent = False
        else:
            variants = args.control_variants
            epochs = args.control_epochs
            skip_recurrent = True
        command = v97_command(
            args,
            feature_paths[control],
            run_dir,
            variants,
            epochs,
            skip_recurrent,
        )
        run_command(command, args.out_dir / "logs" / f"{control}.log")

    movie, macro = collect_predictions(args.cache_dir, run_dirs)
    movie.to_csv(args.out_dir / "v204_movie_metrics.csv", index=False)
    macro.to_csv(args.out_dir / "v204_coordinate_mask_benchmark.csv", index=False)

    real_methods = set(movie.loc[movie.control == "real", "method"])
    baseline_method = args.baseline_method
    candidate_method = args.state_method
    if baseline_method not in real_methods:
        raise RuntimeError(f"Missing baseline method {baseline_method}; have {sorted(real_methods)}")
    if candidate_method not in real_methods:
        raise RuntimeError(f"Missing state method {candidate_method}; have {sorted(real_methods)}")
    bootstrap_rows: list[dict[str, Any]] = []
    for control in controls:
        if candidate_method not in set(movie.loc[movie.control == control, "method"]):
            continue
        result = paired_movie_bootstrap(
            movie,
            baseline_method,
            candidate_method,
            control,
            6,
            args.bootstrap_repeats,
            args.seed + 811,
        )
        bootstrap_rows.append({"control": control, **result})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.out_dir / "v204_mask_state_gate.csv", index=False)

    real_gate = bootstrap.loc[bootstrap.control == "real"]
    real_gain = float(real_gate.mean_gain_pct.iloc[0]) if len(real_gate) else float("nan")
    real_ci_low = float(real_gate.ci_low.iloc[0]) if len(real_gate) else float("nan")
    hard_control_gains = bootstrap.loc[
        bootstrap.control.isin(["zero", "row_shuffled", "time_shuffled", "wrong_cell", "wrong_video"]),
        "mean_gain_pct",
    ]
    controls_pass = bool(len(hard_control_gains) and real_gain > float(hard_control_gains.max()))
    privileged_pass = bool(real_gain >= 3.0 and controls_pass)
    capacity = bootstrap.loc[bootstrap.control == "noncausal_capacity"]
    capacity_pass = bool(len(capacity) and float(capacity.mean_gain_pct.iloc[0]) > real_gain)

    decision = {
        "baseline_method": baseline_method,
        "state_method": candidate_method,
        "real_h6_movie_macro_gain_pct": real_gain,
        "real_h6_gain_ci_low": real_ci_low,
        "real_beats_all_hard_controls": controls_pass,
        "positive_capacity_control_works": capacity_pass,
        "privileged_state_pass": privileged_pass,
        "next_stage": (
            "train_deployable_tracking_aligned_image_student"
            if privileged_pass
            else "close_common_channel_image_state_unless_data_audit_reveals_extraction_failure"
        ),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    (args.out_dir / "v204_manifest.json").write_text(
        json.dumps(finite_json(decision), indent=2), encoding="utf-8"
    )
    lines = [
        "# DeepSea Multimodal Validation v204",
        "",
        f"- Baseline: `{baseline_method}`.",
        f"- Privileged-state model: `{candidate_method}`.",
        f"- Movie-macro h6 gain: `{real_gain:.3f}%`.",
        f"- Movie-cluster bootstrap 95% lower bound: `{real_ci_low:.3f}%`.",
        f"- Real state beats every hard control: `{controls_pass}`.",
        f"- Positive capacity control works: `{capacity_pass}`.",
        f"- Frozen privileged-state gate: `{privileged_pass}`.",
        f"- Decision: `{decision['next_stage']}`.",
        "",
        "The noncausal capacity packet is a diagnostic only and is excluded from every model claim.",
    ]
    (args.out_dir / "v204_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "v204_decision_report.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_multimodal_prepared_v204_2026-07-31",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_online_anchor_cache_v204_2026-07-31",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_multimodal_validation_v204_2026-07-31",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv_foundation/bin/python",
    )
    parser.add_argument(
        "--controls",
        default="real,zero,row_shuffled,time_shuffled,wrong_cell,wrong_video,noncausal_capacity",
    )
    parser.add_argument(
        "--real-variants",
        default="v97_direct,v97_track_only,v97_no_context,v97_direct_context,v97_core,v97_graph",
    )
    parser.add_argument("--control-variants", default="v97_direct_context")
    parser.add_argument("--baseline-method", default="v97_direct")
    parser.add_argument("--state-method", default="v97_direct_context")
    parser.add_argument("--max-state-features", type=int, default=96)
    parser.add_argument("--max-metadata-features", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--control-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--baseline-epochs", type=int, default=28)
    parser.add_argument("--bootstrap-repeats", type=int, default=3000)
    parser.add_argument(
        "--train-coordinate-noise",
        type=float,
        default=0.012,
        help="Coordinate noise in the cache coordinate unit.",
    )
    parser.add_argument(
        "--coordinate-noise-grid",
        default="0.008,0.016,0.032",
        help="Robustness grid in the cache coordinate unit.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--one-step-scaler",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
