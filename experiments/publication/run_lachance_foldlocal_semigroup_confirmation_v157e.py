#!/usr/bin/env python3
"""Publication-grade fold-local confirmation of semigroup innovation transport.

This runner never reuses cross-fold v97 prediction payloads. For every outer
v102 fold and optimizer seed it restores that fold's ``v97_direct.pt``, rebuilds
the matching fold-local prepared data from its run configuration and anchor
cache, and performs fresh replay inference on train/validation/test.

The semigroup correction has one preregistered configuration:

* h1-strict objective over streaming horizons 1/2/4/6;
* update-only causal features (no whole-movie coordinate normalization);
* alpha and correction bound selected on the fold validation movie only;
* refit on the four training movies plus validation, then one test evaluation.

Controls preserve feature-packet coherence:

* ``wrong_cell`` uses one within-frame row permutation for all update columns;
* ``stale_time`` uses the complete packet from the same track one issue frame
  earlier, so its newest possible residual donor is t-2;
* ``no_update`` is the unmodified fold-local v97 ensemble.

The reported h2/h4/h6 metrics are streaming/receding-h1 metrics: a new causal
prediction is issued after each completed transition. They are not open-loop
multi-step forecasts from one initial frame.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_causal_innovation_state_space_v97 as v97  # noqa: E402
import run_lachance_joint_graph_copula_v154 as v154  # noqa: E402
import run_lachance_online_spatial_innovation_audit_v139 as v139  # noqa: E402


DEFAULT_V102 = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_benchmark_v102_v97_production_2026-07-21"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_foldlocal_semigroup_confirmation_v157e"
KEYS = ("sequence", "frame", "track_id")
HORIZONS = (1, 2, 4, 6)
H1_STRICT_WEIGHTS = {1: 0.80, 2: 0.10, 4: 0.06, 6: 0.04}
EPS = 1e-8


@dataclass
class SeedReplay:
    seed: int
    fold_dir: Path
    checkpoint: Path
    anchor_cache: Path
    degrees_of_freedom: float
    uncertainty_factor: float
    rows: dict[str, pd.DataFrame]
    targets: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    manifest: dict[str, Any]


@dataclass
class UpdatePayload:
    movie: int
    split: str
    base: v154.MoviePayload
    real: np.ndarray
    wrong_cell: np.ndarray
    stale_time: np.ndarray
    feature_names: list[str]
    real_latest_donor_frame: np.ndarray
    stale_latest_donor_frame: np.ndarray
    wrong_permutation: np.ndarray


@dataclass
class WeightedRidge:
    row_mean: np.ndarray
    row_scale: np.ndarray
    coefficients: np.ndarray


@dataclass
class Selection:
    alpha: float
    bound_px: float
    validation_score: float


def parse_csv_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(item) for item in value]


def parse_csv_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(item) for item in value]


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(finite(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def ordered_key_array(rows: pd.DataFrame) -> np.ndarray:
    missing = set(KEYS) - set(rows.columns)
    if missing:
        raise RuntimeError(f"Missing row identity columns: {sorted(missing)}")
    return rows[list(KEYS)].to_numpy(np.int64, copy=True)


def row_target_hash(rows: pd.DataFrame, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(ordered_key_array(rows).tobytes(order="C"))
    digest.update(np.ascontiguousarray(target, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def device_from_cli(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fold_directory(root: Path, test_movie: int, seed: int) -> Path:
    matches = sorted((root / "folds").glob(f"test{test_movie:02d}_val*_seed{seed}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one v102 fold for test={test_movie}, seed={seed}; "
            f"found {matches}"
        )
    return matches[0]


def split_name(index: int) -> str:
    return ("train", "validation", "test")[index]


def checkpoint_namespace(checkpoint_payload: dict[str, Any], device: str) -> argparse.Namespace:
    raw = dict(checkpoint_payload["args"])
    for name in ("anchor_cache", "features", "out_dir", "v88_predictions", "v96_predictions"):
        value = raw.get(name)
        if value is not None:
            raw[name] = Path(value)
    raw["device"] = device
    return argparse.Namespace(**raw)


def restore_fold_seed(
    v102_root: Path,
    test_movie: int,
    seed: int,
    device: torch.device,
) -> SeedReplay:
    fold = fold_directory(v102_root, test_movie, seed)
    checkpoint_path = fold / "v97" / "v97_direct.pt"
    config_path = fold / "v97" / "run_config.json"
    if not checkpoint_path.exists() or not config_path.exists():
        raise RuntimeError(f"Incomplete fold-local v97 artifact: {fold}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = v97.TrainVariant(**checkpoint["variant"])
    if variant.name != "v97_direct" or variant.output_mode != "direct":
        raise RuntimeError(f"Unexpected checkpoint variant in {checkpoint_path}: {variant}")
    args = checkpoint_namespace(checkpoint, str(device))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if Path(config["anchor_cache"]).resolve() != Path(args.anchor_cache).resolve():
        raise RuntimeError(f"Checkpoint/run_config anchor mismatch in {fold}")
    prep = v97.load_prepared(args, variant)
    metadata = checkpoint["metadata"]
    static_dim = int(prep.static[0].shape[1])
    first_weight = checkpoint["state_dict"]["static_encoder.0.weight"]
    if int(first_weight.shape[1]) != static_dim:
        raise RuntimeError(
            f"Prepared/checkpoint feature mismatch in {fold}: "
            f"{static_dim} != {int(first_weight.shape[1])}"
        )
    model = v97.CausalInnovationStateSpaceForecaster(
        static_dim=static_dim,
        hidden=int(args.hidden),
        history_lags=int(args.history_lags),
        correction_bound=float(args.correction_bound),
        dropout=float(args.dropout),
        use_update=bool(variant.use_update),
        use_graph=bool(variant.use_graph),
        graph_heads=int(args.graph_heads),
        output_mode=str(variant.output_mode),
        target_mean=np.asarray(metadata["target_mean"], dtype=np.float32),
        target_scale=np.asarray(metadata["target_scale"], dtype=np.float32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    eta = float(metadata["eta"])

    replay_results: dict[str, v97.ReplayResult] = {}
    for split_index in range(3):
        replay_results[split_name(split_index)] = v97.replay_inference(
            model,
            prep,
            split_index,
            device,
            eta=eta,
            control="real",
            seed=int(seed),
        )
    uncertainty_factors = v97.parse_floats(args.uncertainty_scale_grid)
    factor = v97.calibrate_uncertainty(
        prep.bundles[1],
        replay_results["validation"].prediction,
        replay_results["validation"].scale,
        float(metadata["degrees_of_freedom"]),
        uncertainty_factors,
    )

    rows: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    split_manifest: dict[str, Any] = {}
    for split_index, bundle in enumerate(prep.bundles):
        name = split_name(split_index)
        frame = bundle.rows.reset_index(drop=True).copy()
        target = np.asarray(bundle.target_steps[:, 0], dtype=np.float64)
        prediction = np.asarray(replay_results[name].prediction, dtype=np.float64)
        scale = np.maximum(
            np.asarray(replay_results[name].scale, dtype=np.float64) * factor,
            1e-3,
        )
        if not (
            len(frame) == len(target) == len(prediction) == len(scale)
            and np.isfinite(target).all()
            and np.isfinite(prediction).all()
            and np.isfinite(scale).all()
        ):
            raise RuntimeError(f"Non-finite or row-misaligned replay output in {fold}/{name}")
        if int(frame.duplicated(list(KEYS)).sum()) != 0:
            raise RuntimeError(f"Duplicate row keys in {fold}/{name}")
        rows[name] = frame
        targets[name] = target
        predictions[name] = prediction
        scales[name] = scale
        split_manifest[name] = {
            "rows": len(frame),
            "movies": sorted(int(item) for item in frame["sequence"].unique()),
            "key_sha256": sha256_array(ordered_key_array(frame)),
            "target_sha256": sha256_array(target.astype(np.float32)),
            "row_target_sha256": row_target_hash(frame, target),
        }

    anchor_cache = Path(args.anchor_cache).resolve()
    contract_files = sorted(anchor_cache.glob("**/contract.json"))
    manifest = {
        "test_movie": int(test_movie),
        "validation_movie": int(rows["validation"]["sequence"].iloc[0]),
        "train_movies": sorted(int(item) for item in rows["train"]["sequence"].unique()),
        "seed": int(seed),
        "fold_dir": str(fold.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_config": str(config_path.resolve()),
        "run_config_sha256": sha256_file(config_path),
        "anchor_cache": str(anchor_cache),
        "anchor_contract_sha256": {
            str(path.relative_to(anchor_cache)): sha256_file(path)
            for path in contract_files
        },
        "degrees_of_freedom": float(metadata["degrees_of_freedom"]),
        "eta": eta,
        "uncertainty_factor_selected_on_validation": float(factor),
        "device": str(device),
        "splits": split_manifest,
    }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return SeedReplay(
        seed=int(seed),
        fold_dir=fold,
        checkpoint=checkpoint_path,
        anchor_cache=anchor_cache,
        degrees_of_freedom=float(metadata["degrees_of_freedom"]),
        uncertainty_factor=float(factor),
        rows=rows,
        targets=targets,
        predictions=predictions,
        scales=scales,
        manifest=manifest,
    )


def assert_seed_contract(replays: list[SeedReplay], split: str) -> None:
    reference = replays[0]
    reference_keys = ordered_key_array(reference.rows[split])
    reference_target = reference.targets[split]
    for replay in replays[1:]:
        keys = ordered_key_array(replay.rows[split])
        target = replay.targets[split]
        if not np.array_equal(keys, reference_keys):
            raise RuntimeError(
                f"Strict seed key mismatch in split={split}: "
                f"seed {reference.seed} vs {replay.seed}"
            )
        if not np.array_equal(target, reference_target):
            raise RuntimeError(
                f"Strict seed target mismatch in split={split}: "
                f"seed {reference.seed} vs {replay.seed}"
            )


def student_t_mixture_payloads(
    replays: list[SeedReplay],
) -> dict[int, tuple[str, v154.MoviePayload]]:
    output: dict[int, tuple[str, v154.MoviePayload]] = {}
    for split in ("train", "validation", "test"):
        assert_seed_contract(replays, split)
        base_rows = replays[0].rows[split].reset_index(drop=True)
        target = replays[0].targets[split]
        predictions = np.stack([item.predictions[split] for item in replays])
        scales = np.stack([item.scales[split] for item in replays])
        dfs = np.asarray(
            [item.degrees_of_freedom for item in replays],
            dtype=np.float64,
        )
        mean_all = predictions.mean(axis=0)
        component_variance = np.stack(
            [
                np.square(scale) * df / max(df - 2.0, 0.1)
                for scale, df in zip(scales, dfs)
            ]
        )
        mixture_variance = (
            np.mean(component_variance + np.square(predictions), axis=0)
            - np.square(mean_all)
        )
        df = float(np.median(dfs))
        effective_scale_all = np.sqrt(
            np.maximum(mixture_variance, 1e-8) * max(df - 2.0, 0.1) / df
        )
        for movie in sorted(int(item) for item in base_rows["sequence"].unique()):
            indices = np.flatnonzero(base_rows["sequence"].to_numpy(np.int64) == movie)
            order = np.lexsort(
                (
                    base_rows.iloc[indices]["track_id"].to_numpy(np.int64),
                    base_rows.iloc[indices]["frame"].to_numpy(np.int64),
                )
            )
            selected = indices[order]
            rows = base_rows.iloc[selected].reset_index(drop=True)
            movie_target = target[selected]
            movie_mean = mean_all[selected]
            movie_scale = effective_scale_all[selected]
            standardized = (movie_target - movie_mean) / movie_scale
            uniform = np.clip(
                student_t.cdf(standardized, df=df),
                1e-6,
                1.0 - 1e-6,
            )
            normal_score = norm.ppf(uniform)
            if movie in output:
                raise RuntimeError(f"Movie {movie} appears in multiple fold-local splits")
            output[movie] = (
                split,
                v154.MoviePayload(
                    movie=movie,
                    rows=rows,
                    target=movie_target,
                    mean=movie_mean,
                    scale=movie_scale,
                    degrees_of_freedom=df,
                    normal_score=normal_score,
                ),
            )
    return output


def local_previous_state(
    current_position: np.ndarray,
    previous_position: np.ndarray,
    previous_score: np.ndarray,
    current_tracks: np.ndarray,
    previous_tracks: np.ndarray,
    scales: list[float],
) -> dict[str, np.ndarray]:
    distance = np.linalg.norm(
        current_position[:, None, :] - previous_position[None, :, :],
        axis=2,
    )
    same_track = current_tracks[:, None] == previous_tracks[None, :]
    output: dict[str, np.ndarray] = {}
    for scale in scales:
        weights = np.exp(-0.5 * np.square(distance / max(scale, 1e-3)))
        weights[same_track] = 0.0
        weight_sum = weights.sum(axis=1, keepdims=True)
        normalized = weights / np.maximum(weight_sum, EPS)
        mean = normalized @ previous_score
        centered = previous_score[None, :, :] - mean[:, None, :]
        variance = np.einsum(
            "ij,ijk->ik",
            normalized,
            np.square(centered),
            optimize=True,
        )
        effective_count = np.square(weight_sum[:, 0]) / np.maximum(
            np.sum(np.square(weights), axis=1),
            EPS,
        )
        label = str(int(scale))
        output[f"local_{label}_x"] = mean[:, 0]
        output[f"local_{label}_y"] = mean[:, 1]
        output[f"local_{label}_std_x"] = np.sqrt(np.maximum(variance[:, 0], 0.0))
        output[f"local_{label}_std_y"] = np.sqrt(np.maximum(variance[:, 1], 0.0))
        output[f"local_{label}_effective_n"] = effective_count
    return output


def build_real_update_packet(
    payload: v154.MoviePayload,
    scales: list[float],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows = payload.rows.reset_index(drop=True)
    count = len(rows)
    feature: dict[str, np.ndarray] = {
        "own_prev_x": np.zeros(count),
        "own_prev_y": np.zeros(count),
        "own_available": np.zeros(count),
        "global_prev_x": np.zeros(count),
        "global_prev_y": np.zeros(count),
    }
    for scale in scales:
        label = str(int(scale))
        for suffix in ("x", "y", "std_x", "std_y", "effective_n"):
            feature[f"local_{label}_{suffix}"] = np.zeros(count)
    latest_donor = np.full(count, -1, dtype=np.int64)
    frame_groups = rows.groupby("frame", sort=True).indices
    key_to_index = {
        (int(frame), int(track)): index
        for index, (frame, track) in enumerate(
            rows[["frame", "track_id"]].itertuples(index=False)
        )
    }
    for frame, raw_current in frame_groups.items():
        current = np.asarray(raw_current, dtype=np.int64)
        previous = np.asarray(frame_groups.get(int(frame) - 1, []), dtype=np.int64)
        if not len(previous):
            continue
        latest_donor[current] = int(frame) - 1
        previous_score = payload.normal_score[previous]
        global_state = previous_score.mean(axis=0)
        feature["global_prev_x"][current] = global_state[0]
        feature["global_prev_y"][current] = global_state[1]
        current_tracks = rows.iloc[current]["track_id"].to_numpy(np.int64)
        previous_tracks = rows.iloc[previous]["track_id"].to_numpy(np.int64)
        own_indices = np.asarray(
            [
                key_to_index.get((int(frame) - 1, int(track)), -1)
                for track in current_tracks
            ],
            dtype=np.int64,
        )
        available = own_indices >= 0
        if np.any(available):
            selected_rows = current[available]
            selected_own = own_indices[available]
            feature["own_prev_x"][selected_rows] = payload.normal_score[selected_own, 0]
            feature["own_prev_y"][selected_rows] = payload.normal_score[selected_own, 1]
            feature["own_available"][selected_rows] = 1.0
        local = local_previous_state(
            rows.iloc[current][["x_px", "y_px"]].to_numpy(np.float64),
            rows.iloc[previous][["x_px", "y_px"]].to_numpy(np.float64),
            previous_score,
            current_tracks,
            previous_tracks,
            scales,
        )
        for name, value in local.items():
            feature[name][current] = value
    names = list(feature)
    matrix = np.column_stack([feature[name] for name in names]).astype(np.float64)
    return matrix, names, latest_donor


def coherent_wrong_cell(
    real: np.ndarray,
    rows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows), dtype=np.int64)
    for _, raw_indices in rows.groupby("frame", sort=True).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if len(indices) > 1:
            cycle = rng.permutation(indices)
            permutation[cycle] = np.roll(cycle, 1)
            if np.any(permutation[indices] == indices):
                raise RuntimeError("Wrong-cell derangement retained a same-cell donor")
    wrong = real[permutation].copy()
    if not np.array_equal(wrong, real[permutation]):
        raise RuntimeError("Coherent wrong-cell construction failed")
    return wrong, permutation


def coherent_stale_time(
    real: np.ndarray,
    rows: pd.DataFrame,
    real_latest_donor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (int(track), int(frame)): index
        for index, (track, frame) in enumerate(
            rows[["track_id", "frame"]].itertuples(index=False)
        )
    }
    stale = np.zeros_like(real)
    stale_donor = np.full(len(rows), -1, dtype=np.int64)
    for index, (track, frame) in enumerate(
        rows[["track_id", "frame"]].itertuples(index=False)
    ):
        source = lookup.get((int(track), int(frame) - 1), -1)
        if source >= 0:
            stale[index] = real[source]
            stale_donor[index] = real_latest_donor[source]
    return stale, stale_donor


def build_update_payload(
    split: str,
    base: v154.MoviePayload,
    scales: list[float],
    control_seed: int,
) -> UpdatePayload:
    real, names, donor = build_real_update_packet(base, scales)
    wrong, permutation = coherent_wrong_cell(
        real,
        base.rows,
        control_seed + base.movie * 1009,
    )
    stale, stale_donor = coherent_stale_time(real, base.rows, donor)
    current_frame = base.rows["frame"].to_numpy(np.int64)
    if np.any((donor >= 0) & (donor > current_frame - 1)):
        raise RuntimeError(f"Future donor in real packet for movie {base.movie}")
    if np.any((stale_donor >= 0) & (stale_donor > current_frame - 2)):
        raise RuntimeError(f"Future/non-stale donor in stale packet for movie {base.movie}")
    return UpdatePayload(
        movie=base.movie,
        split=split,
        base=base,
        real=real,
        wrong_cell=wrong,
        stale_time=stale,
        feature_names=names,
        real_latest_donor_frame=donor,
        stale_latest_donor_frame=stale_donor,
        wrong_permutation=permutation,
    )


def raw_design(payload: UpdatePayload, control: str = "real") -> np.ndarray:
    update = np.asarray(getattr(payload, control), dtype=np.float64)
    scale = payload.base.scale
    return np.column_stack(
        [
            update,
            update * scale[:, 0:1],
            update * scale[:, 1:2],
        ]
    )


def consecutive_windows(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    lookup = {
        (int(track), int(frame)): index
        for index, (track, frame) in enumerate(
            rows[["track_id", "frame"]].itertuples(index=False)
        )
    }
    windows: list[list[int]] = []
    for track, frame in rows[["track_id", "frame"]].itertuples(index=False):
        indices = [
            lookup.get((int(track), int(frame) + offset))
            for offset in range(int(horizon))
        ]
        if all(index is not None for index in indices):
            windows.append([int(index) for index in indices])
    if not windows:
        return np.empty((0, horizon), dtype=np.int64)
    return np.asarray(windows, dtype=np.int64)


def row_normalization(
    payloads: dict[int, UpdatePayload],
    movies: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([raw_design(payloads[movie]) for movie in movies])
    return matrix.mean(axis=0), np.maximum(matrix.std(axis=0), 1e-8)


def augmented_training_data(
    payloads: dict[int, UpdatePayload],
    movies: list[int],
    row_mean: np.ndarray,
    row_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    for movie in movies:
        payload = payloads[movie]
        normalized = (raw_design(payload) - row_mean) / row_scale
        per_step = np.column_stack(
            [normalized, np.ones(len(normalized), dtype=np.float64)]
        )
        residual = payload.base.target - payload.base.mean
        for horizon in HORIZONS:
            windows = consecutive_windows(payload.base.rows, horizon)
            if not len(windows):
                continue
            feature_blocks.append(per_step[windows].sum(axis=1))
            target_blocks.append(residual[windows].sum(axis=1))
            weight_blocks.append(
                np.full(
                    len(windows),
                    H1_STRICT_WEIGHTS[horizon] / len(windows),
                    dtype=np.float64,
                )
            )
    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    weights = np.concatenate(weight_blocks)
    weights *= len(weights) / max(float(weights.sum()), EPS)
    return features, targets, weights


def fit_weighted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    row_mean: np.ndarray,
    row_scale: np.ndarray,
) -> WeightedRidge:
    x = np.nan_to_num(np.asarray(features, dtype=np.float64))
    y = np.nan_to_num(np.asarray(target, dtype=np.float64))
    normalized_weight = weights / max(float(np.mean(weights)), EPS)
    root_weight = np.sqrt(normalized_weight)[:, None]
    weighted_x = x * root_weight
    weighted_y = y * root_weight
    gram = weighted_x.T @ weighted_x
    rhs = weighted_x.T @ weighted_y
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return WeightedRidge(row_mean, row_scale, coefficients)


def predict_ridge(
    model: WeightedRidge,
    payload: UpdatePayload,
    control: str,
) -> np.ndarray:
    normalized = (raw_design(payload, control) - model.row_mean) / model.row_scale
    augmented = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    return augmented @ model.coefficients


def bounded_update(correction: np.ndarray, bound_px: float) -> np.ndarray:
    if bound_px <= 0:
        return correction
    length = np.linalg.norm(correction, axis=1, keepdims=True)
    bounded_length = np.tanh(length / float(bound_px)) * float(bound_px)
    return correction * bounded_length / np.maximum(length, EPS)


def metric_rows(
    payload: UpdatePayload,
    prediction: np.ndarray,
    control: str,
    selection: Selection | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        windows = consecutive_windows(payload.base.rows, horizon)
        target = payload.base.target[windows].sum(axis=1)
        predicted = prediction[windows].sum(axis=1)
        baseline = payload.base.mean[windows].sum(axis=1)
        error = predicted - target
        baseline_error = baseline - target
        component_rmse = float(np.sqrt(np.mean(np.square(error))))
        vector_rmse = float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
        baseline_component = float(np.sqrt(np.mean(np.square(baseline_error))))
        baseline_vector = float(
            np.sqrt(np.mean(np.sum(np.square(baseline_error), axis=1)))
        )
        centered = target - target.mean(axis=0, keepdims=True)
        vector_r2 = 1.0 - float(
            np.sum(np.square(error)) / max(np.sum(np.square(centered)), EPS)
        )
        baseline_vector_r2 = 1.0 - float(
            np.sum(np.square(baseline_error))
            / max(np.sum(np.square(centered)), EPS)
        )
        component_denominator = np.maximum(
            np.sum(np.square(centered), axis=0),
            EPS,
        )
        component_r2 = float(
            np.mean(
                1.0
                - np.sum(np.square(error), axis=0)
                / component_denominator
            )
        )
        baseline_component_r2 = float(
            np.mean(
                1.0
                - np.sum(np.square(baseline_error), axis=0)
                / component_denominator
            )
        )
        output.append(
            {
                "test_movie": payload.movie,
                "control": control,
                "objective": "h1_strict",
                "packet": "update_only" if control != "no_update" else "none",
                "horizon": horizon,
                "windows": len(windows),
                "component_rmse": component_rmse,
                "vector_rmse": vector_rmse,
                "component_r2": component_r2,
                "vector_r2": vector_r2,
                "r2": vector_r2,
                "baseline_component_rmse": baseline_component,
                "baseline_vector_rmse": baseline_vector,
                "baseline_component_r2": baseline_component_r2,
                "baseline_vector_r2": baseline_vector_r2,
                "baseline_r2": baseline_vector_r2,
                "component_rmse_delta": baseline_component - component_rmse,
                "rmse_improvement_percent": 100.0
                * (baseline_component - component_rmse)
                / max(baseline_component, EPS),
                "selected_alpha": selection.alpha if selection else np.nan,
                "selected_bound_px": selection.bound_px if selection else 0.0,
                "validation_score": (
                    selection.validation_score if selection else np.nan
                ),
                "contract": "streaming_receding_h1",
            }
        )
    return output


def objective_score(metrics: list[dict[str, Any]]) -> float:
    return float(
        sum(
            H1_STRICT_WEIGHTS[int(row["horizon"])]
            * float(row["component_rmse"])
            / max(float(row["baseline_component_rmse"]), EPS)
            for row in metrics
        )
    )


def select_on_validation(
    payloads: dict[int, UpdatePayload],
    train_movies: list[int],
    validation_movie: int,
    alphas: list[float],
    bounds: list[float],
) -> tuple[Selection, pd.DataFrame]:
    row_mean, row_scale = row_normalization(payloads, train_movies)
    features, target, weights = augmented_training_data(
        payloads,
        train_movies,
        row_mean,
        row_scale,
    )
    validation = payloads[validation_movie]
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_weighted_ridge(
            features,
            target,
            weights,
            alpha,
            row_mean,
            row_scale,
        )
        raw = predict_ridge(model, validation, "real")
        for bound in bounds:
            prediction = validation.base.mean + bounded_update(raw, bound)
            metrics = metric_rows(
                validation,
                prediction,
                "validation_real",
                None,
            )
            record: dict[str, Any] = {
                "alpha": float(alpha),
                "bound_px": float(bound),
                "validation_score": objective_score(metrics),
            }
            for row in metrics:
                horizon = int(row["horizon"])
                record[f"h{horizon}_component_rmse"] = row["component_rmse"]
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
        Selection(
            alpha=float(best["alpha"]),
            bound_px=float(best["bound_px"]),
            validation_score=float(best["validation_score"]),
        ),
        grid,
    )


def refit_model(
    payloads: dict[int, UpdatePayload],
    fit_movies: list[int],
    selection: Selection,
) -> WeightedRidge:
    row_mean, row_scale = row_normalization(payloads, fit_movies)
    features, target, weights = augmented_training_data(
        payloads,
        fit_movies,
        row_mean,
        row_scale,
    )
    return fit_weighted_ridge(
        features,
        target,
        weights,
        selection.alpha,
        row_mean,
        row_scale,
    )


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    observed = abs(float(values.mean()))
    outcomes = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        outcomes.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(outcomes) >= observed - 1e-15))


def movie_cluster_bootstrap(
    deltas: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3 or repeats <= 0:
        return {
            "bootstrap_mean_delta": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_probability_positive": float("nan"),
        }
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(repeats), len(values)), replace=True)
    means = sampled.mean(axis=1)
    return {
        "bootstrap_mean_delta": float(means.mean()),
        "bootstrap_ci_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci_high": float(np.quantile(means, 0.975)),
        "bootstrap_probability_positive": float(np.mean(means > 0)),
    }


def aggregate_metrics(
    metrics: pd.DataFrame,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (control, horizon), group in metrics.groupby(["control", "horizon"]):
        bootstrap = movie_cluster_bootstrap(
            group["component_rmse_delta"].to_numpy(np.float64),
            bootstrap_repeats,
            seed + int(horizon) * 101,
        )
        records.append(
            {
                "control": control,
                "horizon": int(horizon),
                "movies": int(group["test_movie"].nunique()),
                "component_rmse_macro_mean": float(group["component_rmse"].mean()),
                "component_rmse_macro_std": float(group["component_rmse"].std(ddof=1))
                if len(group) > 1
                else np.nan,
                "vector_rmse_macro_mean": float(group["vector_rmse"].mean()),
                "component_r2_macro_mean": float(group["component_r2"].mean()),
                "vector_r2_macro_mean": float(group["vector_r2"].mean()),
                "r2_macro_mean": float(group["vector_r2"].mean()),
                "baseline_component_rmse_macro_mean": float(
                    group["baseline_component_rmse"].mean()
                ),
                "rmse_improvement_percent_mean": float(
                    group["rmse_improvement_percent"].mean()
                ),
                "movies_improved": int((group["component_rmse_delta"] > 0).sum()),
                "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(
                    group["component_rmse_delta"].to_numpy(np.float64)
                ),
                **bootstrap,
            }
        )
    return pd.DataFrame(records)


def build_causal_audit(
    fold_payloads: dict[int, UpdatePayload],
    test_movie: int,
    validation_movie: int,
    train_movies: list[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for movie, payload in sorted(fold_payloads.items()):
        frames = payload.base.rows["frame"].to_numpy(np.int64)
        real_available = payload.real_latest_donor_frame >= 0
        stale_available = payload.stale_latest_donor_frame >= 0
        wrong_matches = np.array_equal(
            payload.wrong_cell,
            payload.real[payload.wrong_permutation],
        )
        records.append(
            {
                "outer_test_movie": test_movie,
                "validation_movie": validation_movie,
                "train_movies": ",".join(map(str, train_movies)),
                "movie": movie,
                "split": payload.split,
                "rows": len(frames),
                "duplicate_keys": int(
                    payload.base.rows.duplicated(list(KEYS)).sum()
                ),
                "real_donor_coverage": float(real_available.mean()),
                "real_future_donor_violations": int(
                    np.sum(
                        real_available
                        & (payload.real_latest_donor_frame > frames - 1)
                    )
                ),
                "stale_donor_coverage": float(stale_available.mean()),
                "stale_future_or_nonstale_violations": int(
                    np.sum(
                        stale_available
                        & (payload.stale_latest_donor_frame > frames - 2)
                    )
                ),
                "coherent_wrong_packet": bool(wrong_matches),
                "wrong_fixed_point_fraction": float(
                    np.mean(payload.wrong_permutation == np.arange(len(frames)))
                ),
                "wrong_single_permutation_sha256": sha256_array(
                    payload.wrong_permutation
                ),
                "feature_columns": len(payload.feature_names),
                "uses_whole_movie_xy_normalization": False,
                "prediction_time": "t",
                "latest_real_measurement": "completed transition t-1 to t",
                "latest_stale_measurement": "completed transition no later than t-2",
                "target_time": "t+1",
            }
        )
    return pd.DataFrame(records)


def evaluate_outer_fold(
    args: argparse.Namespace,
    test_movie: int,
    seeds: list[int],
    alphas: list[float],
    bounds: list[float],
    local_scales: list[float],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    list[dict[str, Any]],
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    seed_replays = [
        restore_fold_seed(args.v102_root, test_movie, seed, device)
        for seed in seeds
    ]
    split_payloads = student_t_mixture_payloads(seed_replays)
    validation_movies = {
        int(replay.manifest["validation_movie"]) for replay in seed_replays
    }
    train_movie_sets = {
        tuple(replay.manifest["train_movies"]) for replay in seed_replays
    }
    if len(validation_movies) != 1 or len(train_movie_sets) != 1:
        raise RuntimeError(f"Seed fold split mismatch for test movie {test_movie}")
    validation_movie = next(iter(validation_movies))
    train_movies = list(next(iter(train_movie_sets)))
    if set(train_movies) | {validation_movie, test_movie} != set(range(1, 7)):
        raise RuntimeError(f"Invalid outer fold partition for test movie {test_movie}")

    payloads = {
        movie: build_update_payload(
            split,
            base,
            local_scales,
            int(args.control_seed) + test_movie * 100_003,
        )
        for movie, (split, base) in split_payloads.items()
    }
    selection, grid = select_on_validation(
        payloads,
        train_movies,
        validation_movie,
        alphas,
        bounds,
    )
    grid.insert(0, "test_movie", test_movie)
    grid.insert(1, "validation_movie", validation_movie)
    grid.insert(2, "train_movies", ",".join(map(str, train_movies)))
    model = refit_model(
        payloads,
        train_movies + [validation_movie],
        selection,
    )
    test = payloads[test_movie]
    metric_records: list[dict[str, Any]] = []
    prediction_archive: dict[str, np.ndarray] = {
        f"movie{test_movie:02d}__target": test.base.target.astype(np.float32),
        f"movie{test_movie:02d}__no_update": test.base.mean.astype(np.float32),
        f"movie{test_movie:02d}__keys": ordered_key_array(test.base.rows),
    }
    for control in ("real", "wrong_cell", "stale_time"):
        raw = predict_ridge(model, test, control)
        prediction = test.base.mean + bounded_update(raw, selection.bound_px)
        metric_records.extend(metric_rows(test, prediction, control, selection))
        prediction_archive[f"movie{test_movie:02d}__{control}"] = prediction.astype(
            np.float32
        )
    metric_records.extend(
        metric_rows(test, test.base.mean, "no_update", None)
    )
    causal_audit = build_causal_audit(
        payloads,
        test_movie,
        validation_movie,
        train_movies,
    )
    manifests = [replay.manifest for replay in seed_replays]
    for manifest in manifests:
        manifest["strict_seed_key_target_match"] = True
        manifest["ensemble_seeds"] = seeds
        manifest["selection"] = {
            "objective": "h1_strict",
            "packet": "update_only",
            "alpha": selection.alpha,
            "bound_px": selection.bound_px,
            "validation_score": selection.validation_score,
            "selection_data": f"movie {validation_movie} only",
            "refit_data": train_movies + [validation_movie],
            "test_data": [test_movie],
        }
    return metric_records, grid, manifests, causal_audit, prediction_archive


def status_report(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    elapsed: float,
) -> str:
    real = aggregate[aggregate["control"].eq("real")]
    lines = [
        "# v157e Fold-Local Semigroup Confirmation",
        "",
        "## Protocol",
        "",
        "- Every v97 prediction was regenerated from the matching outer-fold checkpoint.",
        "- Seeds were averaged only after exact ordered key and target equality checks.",
        "- Primary objective: `h1_strict`; feature packet: `update_only`.",
        "- Hyperparameters were selected on the fold validation movie and refit on train+validation.",
        "- h2/h4/h6 are streaming/receding-h1 metrics, not open-loop forecasts.",
        "",
        "## Aggregate real update",
        "",
        "| Horizon | Movies | Component RMSE | Vector RMSE | R2 | Gain | Improved | p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in real.sort_values("horizon").itertuples(index=False):
        p_value = (
            "n/a"
            if not math.isfinite(float(row.exact_two_sided_sign_flip_p))
            else f"{float(row.exact_two_sided_sign_flip_p):.5f}"
        )
        lines.append(
            f"| {int(row.horizon)} | {int(row.movies)} | "
            f"{float(row.component_rmse_macro_mean):.6f} | "
            f"{float(row.vector_rmse_macro_mean):.6f} | "
            f"{float(row.vector_r2_macro_mean):.6f} | "
            f"{float(row.rmse_improvement_percent_mean):+.3f}% | "
            f"{int(row.movies_improved)}/{int(row.movies)} | {p_value} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "Controls are evaluated with the same model selected on real validation data.",
            "`wrong_cell` uses one within-frame permutation for every update column;",
            "`stale_time` uses a coherent same-track packet whose newest donor is t-2.",
            "",
            "## Status",
            "",
        ]
    )
    movie_count = int(metrics["test_movie"].nunique())
    if movie_count == 6:
        h1 = real[real["horizon"].eq(1)].iloc[0]
        h6 = real[real["horizon"].eq(6)].iloc[0]
        real_h6 = metrics[
            metrics["control"].eq("real") & metrics["horizon"].eq(6)
        ]["component_rmse"]
        wrong_h6 = metrics[
            metrics["control"].eq("wrong_cell") & metrics["horizon"].eq(6)
        ]["component_rmse"]
        stale_h6 = metrics[
            metrics["control"].eq("stale_time") & metrics["horizon"].eq(6)
        ]["component_rmse"]
        controls_pass = bool(
            float(real_h6.mean()) < float(wrong_h6.mean())
            and float(real_h6.mean()) < float(stale_h6.mean())
        )
        pass_gate = bool(
            float(h1.rmse_improvement_percent_mean) >= -0.5
            and float(h6.rmse_improvement_percent_mean) >= 1.0
            and int(h6.movies_improved) == 6
            and controls_pass
        )
        lines.append(
            "**PASS**" if pass_gate else "**FAIL_OR_NOT_CONFIRMED**"
        )
        lines.append(
            f" Primary gate: h1 gain {float(h1.rmse_improvement_percent_mean):+.3f}%, "
            f"h6 gain {float(h6.rmse_improvement_percent_mean):+.3f}%, "
            f"h6 improved {int(h6.movies_improved)}/6, controls_pass={controls_pass}."
        )
    else:
        lines.append(
            "**SMOKE_ONLY**: one or more folds completed; no six-movie publication claim."
        )
    lines.extend(
        [
            "",
            f"Elapsed: {elapsed / 60.0:.2f} minutes.",
            f"Output: `{args.out_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=157_005)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.v102_root = args.v102_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = parse_csv_ints(args.movies)
    seeds = parse_csv_ints(args.seeds)
    alphas = parse_csv_floats(args.alphas)
    bounds = parse_csv_floats(args.bounds_px)
    local_scales = parse_csv_floats(args.local_scales_px)
    if not movies or not seeds or not alphas or not bounds or not local_scales:
        raise ValueError("Movies, seeds, alphas, bounds and local scales must be non-empty")
    if any(movie not in range(1, 7) for movie in movies):
        raise ValueError(f"MDCK_Bulk outer movies must be 1..6, got {movies}")
    device = device_from_cli(args.device)

    all_metrics: list[dict[str, Any]] = []
    all_grids: list[pd.DataFrame] = []
    all_manifests: list[dict[str, Any]] = []
    all_audits: list[pd.DataFrame] = []
    prediction_archive: dict[str, np.ndarray] = {}
    for test_movie in movies:
        print(
            f"[v157e] test movie {test_movie}: restoring fold-local seeds {seeds}",
            flush=True,
        )
        metrics, grid, manifests, audit, predictions = evaluate_outer_fold(
            args,
            test_movie,
            seeds,
            alphas,
            bounds,
            local_scales,
            device,
        )
        all_metrics.extend(metrics)
        all_grids.append(grid)
        all_manifests.extend(manifests)
        all_audits.append(audit)
        prediction_archive.update(predictions)
        selected = manifests[0]["selection"]
        print(
            f"[v157e] test movie {test_movie}: selection "
            f"alpha={selected['alpha']}, "
            f"bound={selected['bound_px']}",
            flush=True,
        )

    metrics_frame = pd.DataFrame(all_metrics)
    validation_grid = pd.concat(all_grids, ignore_index=True)
    causal_audit = pd.concat(all_audits, ignore_index=True)
    aggregate = aggregate_metrics(
        metrics_frame,
        int(args.bootstrap_repeats),
        int(args.control_seed),
    )
    if int(causal_audit["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Causal audit found future donors in real update")
    if int(causal_audit["stale_future_or_nonstale_violations"].sum()) != 0:
        raise RuntimeError("Causal audit found non-stale/future donors in stale control")
    if not bool(causal_audit["coherent_wrong_packet"].all()):
        raise RuntimeError("Wrong-cell packet coherence audit failed")

    metrics_frame.to_csv(args.out_dir / "v157e_fold_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "v157e_aggregate.csv", index=False)
    validation_grid.to_csv(
        args.out_dir / "v157e_validation_grid.csv",
        index=False,
    )
    pd.DataFrame(all_manifests).to_csv(
        args.out_dir / "v157e_seed_replay_manifest.csv",
        index=False,
    )
    write_json(
        args.out_dir / "v157e_seed_replay_manifest.json",
        all_manifests,
    )
    causal_audit.to_csv(args.out_dir / "v157e_causal_audit.csv", index=False)
    np.savez_compressed(
        args.out_dir / "v157e_predictions.npz",
        **prediction_archive,
    )
    run_config = {
        **vars(args),
        "device_resolved": str(device),
        "objective": "h1_strict",
        "objective_weights": H1_STRICT_WEIGHTS,
        "packet": "update_only",
        "controls": ["wrong_cell", "stale_time", "no_update"],
        "protocol": "fold-local v97 replay; streaming/receding h1",
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.out_dir / "run_config.json", run_config)
    elapsed = time.time() - started
    report = status_report(args, metrics_frame, aggregate, elapsed)
    (args.out_dir / "v157e_status_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
