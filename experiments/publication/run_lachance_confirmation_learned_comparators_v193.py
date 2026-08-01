#!/usr/bin/env python3
"""Frozen learned-comparator confirmation on MDCK Bulk movies 10--16.

The development contract is fixed:

* fit on movies 1--4;
* select representation, hyperparameters, and early stopping on movie 5;
* evaluate once on movies 10--16;
* never refit or select from a confirmation metric.

The runner trains track-native HGBDT and GRU comparators directly and invokes
the source-faithful KalmanNet v98 adaptation under the same cache contract.  It
then ensembles the three optimizer seeds before computing movie-macro metrics,
matching the three-checkpoint evaluation used by v160.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_causal_innovation_state_space_v97 as v97  # noqa: E402
import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402
import run_lachance_online_tabular_baselines_v111 as v111  # noqa: E402


KEYS = ("sequence", "frame", "track_id")
TRAIN_MOVIES = (1, 2, 3, 4)
VALIDATION_MOVIES = (5,)
CONFIRMATION_MOVIES = (10, 11, 12, 13, 14, 15, 16)
DEFAULT_CONFIRMATION_CACHE = (
    ROOT / "outputs" / "lachance_track_native_v160_confirmation_cache"
)
DEFAULT_DEVELOPMENT_CACHE = (
    ROOT / "outputs" / "lachance_track_native_v160_development_cache"
)
DEFAULT_V160 = (
    ROOT
    / "outputs"
    / "lachance_streaming_transport_confirmation_v160_full_2026-07-27"
)
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "lachance_confirmation_learned_comparators_v193_2026-07-30"
)


def parse_strings(value: str | Iterable[Any]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def parse_ints(value: str | Iterable[Any]) -> list[int]:
    return [int(part) for part in parse_strings(value)]


def safe(value: Any) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(value, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def subset_bundle(
    bundle: v84.AnchorBundle,
    movies: Iterable[int],
    name: str,
) -> v84.AnchorBundle:
    movie_set = {int(movie) for movie in movies}
    mask = bundle.rows.sequence.astype(int).isin(movie_set).to_numpy()
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError(f"No rows for {name}: movies={sorted(movie_set)}")
    return v84.AnchorBundle(
        name=name,
        rows=bundle.rows.iloc[indices].reset_index(drop=True),
        anchor_residual=safe(bundle.anchor_residual[indices]),
        base=safe(bundle.base[indices]),
        target_steps=safe(bundle.target_steps[indices]),
        meta={**bundle.meta, "subset_movies": sorted(movie_set)},
    )


def assert_same_bundle(
    left: v84.AnchorBundle,
    right: v84.AnchorBundle,
    split: str,
) -> dict[str, Any]:
    left_keys = left.rows[list(KEYS)].to_numpy(np.int64, copy=True)
    right_keys = right.rows[list(KEYS)].to_numpy(np.int64, copy=True)
    checks = {
        "keys_equal": bool(np.array_equal(left_keys, right_keys)),
        "base_equal": bool(np.array_equal(left.base, right.base)),
        "anchor_equal": bool(np.array_equal(left.anchor_residual, right.anchor_residual)),
        "targets_equal": bool(np.array_equal(left.target_steps, right.target_steps)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Development/confirmation {split} mismatch: {checks}")
    return {
        "split": split,
        "rows": int(len(left.rows)),
        "movies": ",".join(map(str, sorted(left.rows.sequence.astype(int).unique()))),
        "key_sha256": array_sha256(left_keys),
        "h1_target_sha256": array_sha256(left.target_steps[:, 0]),
        **checks,
    }


def load_contract_bundles(
    confirmation_cache: Path,
    development_cache: Path,
) -> tuple[
    tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    pd.DataFrame,
]:
    confirmation = v85.load_anchor_cache(confirmation_cache)
    development = v85.load_anchor_cache(development_cache)
    records = [
        assert_same_bundle(development[0], confirmation[0], "train"),
        assert_same_bundle(development[1], confirmation[1], "validation"),
    ]
    actual = [
        tuple(sorted(map(int, confirmation[0].rows.sequence.unique()))),
        tuple(sorted(map(int, confirmation[1].rows.sequence.unique()))),
        tuple(sorted(map(int, confirmation[2].rows.sequence.unique()))),
    ]
    expected = [TRAIN_MOVIES, VALIDATION_MOVIES, CONFIRMATION_MOVIES]
    if actual != expected:
        raise RuntimeError(f"Frozen split mismatch: actual={actual}, expected={expected}")
    overlap = sum(
        len(set(actual[left]) & set(actual[right]))
        for left in range(3)
        for right in range(left + 1, 3)
    )
    if overlap:
        raise RuntimeError(f"Movie overlap across splits: {actual}")
    records.append(
        {
            "split": "confirmation",
            "rows": int(len(confirmation[2].rows)),
            "movies": ",".join(map(str, CONFIRMATION_MOVIES)),
            "key_sha256": array_sha256(
                confirmation[2].rows[list(KEYS)].to_numpy(np.int64, copy=True)
            ),
            "h1_target_sha256": array_sha256(confirmation[2].target_steps[:, 0]),
            "keys_equal": True,
            "base_equal": True,
            "anchor_equal": True,
            "targets_equal": True,
        }
    )
    return confirmation, pd.DataFrame(records)


def hgbdt_namespace(args: argparse.Namespace, seed: int, modes: str) -> argparse.Namespace:
    return argparse.Namespace(
        input_modes=modes,
        history_lags=int(args.history_lags),
        flow_k=int(args.flow_k),
        seed=int(seed),
        ridge_alphas="1",
        hgbdt_learning_rates=args.hgbdt_learning_rates,
        hgbdt_max_leaf_nodes=args.hgbdt_max_leaf_nodes,
        hgbdt_l2=args.hgbdt_l2,
        hgbdt_max_iter=int(args.hgbdt_max_iter),
        hgbdt_min_samples_leaf=int(args.hgbdt_min_samples_leaf),
        mlp_hidden_grid="64",
        mlp_alphas="0.001",
        mlp_batch_size=512,
        mlp_learning_rate=8e-4,
        mlp_max_iter=2,
        mlp_patience=1,
        eta_grid=args.eta_grid,
    )


def prepare_hgbdt_features(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], v111.FeaturePack], pd.DataFrame]:
    modes = ("raw_coordinate", "v52_anchor")
    packs = {
        (split, mode): v111.build_features(
            bundle,
            int(args.history_lags),
            int(args.flow_k),
            mode,
        )
        for split, bundle in zip(("train", "val", "test"), bundles)
        for mode in modes
    }
    local = hgbdt_namespace(args, 7, ",".join(modes))
    audit = v111.causal_audit(bundles, packs, local)
    if (
        not bool(audit.future_label_sentinel_unchanged.all())
        or int(audit.causal_source_violations.sum()) != 0
        or bool(audit.movie_split_overlap.astype(str).str.len().gt(0).any())
    ):
        raise RuntimeError(f"HGBDT causal audit failed:\n{audit.to_string(index=False)}")
    return packs, audit


def train_hgbdt(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    packs: dict[tuple[str, str], v111.FeaturePack],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    train, validation, test = bundles
    modes = ("raw_coordinate", "v52_anchor")
    local = hgbdt_namespace(args, seed, ",".join(modes))

    candidates: list[tuple[float, str, v111.FittedRegressor, float]] = []
    sweeps: list[pd.DataFrame] = []
    for mode in modes:
        fitted, eta, sweep, fit_seconds = v111.select_model(
            "hgbdt",
            mode,
            packs[("train", mode)],
            packs[("val", mode)],
            train,
            validation,
            v97.parse_horizon_weights(args.validation_horizon_weights),
            local,
        )
        sweep_frame = pd.DataFrame(sweep)
        sweep_frame["seed"] = int(seed)
        sweep_frame["fit_seconds_total_for_mode"] = float(fit_seconds)
        sweeps.append(sweep_frame)
        score = float(sweep_frame.validation_weighted_component_rmse.min())
        candidates.append((score, mode, fitted, float(eta)))

    candidates.sort(key=lambda item: (item[0], item[1]))
    score, mode, fitted, eta = candidates[0]
    output, predict_seconds = v111.chronological_predict(
        fitted,
        packs[("test", mode)].values,
        test,
    )
    prediction = v111.decode_prediction(test, mode, output, eta)
    metadata = {
        "seed": int(seed),
        "selected_input_mode": mode,
        "selected_eta": float(eta),
        "validation_weighted_component_rmse": float(score),
        "hyperparameters": fitted.config,
        "train_feature_dim": int(packs[("train", mode)].values.shape[1]),
        "prediction_seconds": float(predict_seconds),
        "selection_scope": "movie_5_only",
        "fit_scope": "movies_1_4_only",
    }
    return prediction, pd.concat(sweeps, ignore_index=True), metadata


def train_gru(
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    local = argparse.Namespace(
        baseline_history_lags=int(args.gru_history_lags),
        baseline_hidden=int(args.gru_hidden),
        baseline_batch_size=int(args.gru_batch_size),
        baseline_epochs=int(args.gru_epochs),
        validation_horizon_weights=args.validation_horizon_weights,
        seed=int(seed),
    )
    prediction, metadata = v97.fit_recurrent_baseline(
        "gru",
        bundles,
        local,
        device_from_name(args.device),
    )
    return prediction, {
        "seed": int(seed),
        "selection_scope": "movie_5_early_stopping_only",
        "fit_scope": "movies_1_4_only",
        **metadata,
    }


def kalmannet_command(
    args: argparse.Namespace,
    seed: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "run_lachance_kalmannet_online_v98.py"),
        "--anchor-cache",
        str(args.confirmation_cache),
        "--out-dir",
        str(output),
        "--models",
        "cv,ca",
        "--predictive-event",
        "next_observation_prior",
        "--validation-horizon-weights",
        args.validation_horizon_weights,
        "--loss",
        "huber",
        "--cumulative-weight",
        str(args.kalmannet_cumulative_weight),
        "--epochs",
        str(args.kalmannet_epochs),
        "--patience",
        str(args.kalmannet_patience),
        "--bootstrap-repeats",
        str(args.kalmannet_bootstrap_repeats),
        "--seed",
        str(seed),
        "--device",
        args.kalmannet_device,
    ]


def run_kalmannet_jobs(args: argparse.Namespace, seeds: list[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, Path, list[str]]] = []
    for seed in seeds:
        output = args.out_dir / "kalmannet" / f"seed{seed}"
        prediction_path = output / "v98_predictions.npz"
        sentinel_path = output / "v98_no_future_sentinel.json"
        if prediction_path.exists() and sentinel_path.exists() and not args.force:
            sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
            if bool(sentinel.get("pass")):
                records.append(
                    {
                        "seed": seed,
                        "status": "reused",
                        "returncode": 0,
                        "seconds": 0.0,
                        "command": json.dumps(kalmannet_command(args, seed, output)),
                    }
                )
                continue
        output.mkdir(parents=True, exist_ok=True)
        pending.append((seed, output, kalmannet_command(args, seed, output)))

    max_jobs = max(1, min(int(args.kalmannet_parallel_jobs), len(pending)))
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(args.kalmannet_threads_per_job))
    env.setdefault("MKL_NUM_THREADS", str(args.kalmannet_threads_per_job))
    env.setdefault("VECLIB_MAXIMUM_THREADS", str(args.kalmannet_threads_per_job))
    active: list[tuple[int, Path, list[str], subprocess.Popen[str], Any, float]] = []
    cursor = 0
    while cursor < len(pending) or active:
        while cursor < len(pending) and len(active) < max_jobs:
            seed, output, command = pending[cursor]
            cursor += 1
            log_path = output / "v193_kalmannet.log"
            log_stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            active.append((seed, output, command, process, log_stream, time.perf_counter()))
        time.sleep(2.0)
        still_active: list[
            tuple[int, Path, list[str], subprocess.Popen[str], Any, float]
        ] = []
        for seed, output, command, process, log_stream, started in active:
            returncode = process.poll()
            if returncode is None:
                still_active.append(
                    (seed, output, command, process, log_stream, started)
                )
                continue
            log_stream.close()
            seconds = time.perf_counter() - started
            records.append(
                {
                    "seed": seed,
                    "status": "complete" if returncode == 0 else "failed",
                    "returncode": int(returncode),
                    "seconds": float(seconds),
                    "command": json.dumps(command),
                }
            )
            if returncode != 0:
                raise RuntimeError(
                    f"KalmanNet seed {seed} failed; see "
                    f"{output / 'v193_kalmannet.log'}"
                )
        active = still_active
    return pd.DataFrame(records).sort_values("seed").reset_index(drop=True)


def load_kalmannet_prediction(
    output: Path,
    expected_keys: np.ndarray,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    archive = np.load(output / "v98_predictions.npz")
    keys = archive["contract__row_keys"].astype(np.int64)
    if not np.array_equal(keys, expected_keys):
        raise RuntimeError(f"KalmanNet row-key mismatch: {output}")
    # The state model is selected inside v98 from movie 5 only. Reading test
    # summary ranks here would silently turn the confirmation cohort into a
    # second model-selection set.
    contract = pd.read_csv(output / "v98_data_contract.csv")
    if len(contract) != 1 or "selected_model" not in contract:
        raise RuntimeError(f"Missing validation-selected KalmanNet contract: {output}")
    selected = str(contract.selected_model.iloc[0])
    key = f"{selected}__prediction"
    if key not in archive.files:
        raise RuntimeError(f"Missing selected KalmanNet prediction {key}: {output}")
    sentinel = json.loads(
        (output / "v98_no_future_sentinel.json").read_text(encoding="utf-8")
    )
    if not bool(sentinel.get("pass")):
        raise RuntimeError(f"KalmanNet no-future sentinel failed: {output}")
    return safe(archive[key]), selected, sentinel


def movie_metric_rows(
    test: v84.AnchorBundle,
    prediction: np.ndarray,
    method: str,
    horizons: list[int],
    *,
    evaluation: str,
    seed: int | str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for movie in CONFIRMATION_MOVIES:
        mask = test.rows.sequence.astype(int).eq(movie).to_numpy()
        indices = np.flatnonzero(mask)
        bundle = subset_bundle(test, [movie], f"{test.name}_movie{movie}")
        rows = v97.rolling_metric_rows(
            bundle,
            safe(prediction[indices]),
            horizons,
            method,
            {
                "test_movie": int(movie),
                "seed": seed,
                "evaluation": evaluation,
                "fit_movies": "1,2,3,4",
                "validation_movies": "5",
                "confirmation_movies": "10,11,12,13,14,15,16",
            },
        )
        records.extend(rows)
    return records


def exact_sign_flip_p(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    observed = abs(float(values.mean()))
    absolute = np.abs(values)
    outcomes = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        outcomes.append(abs(float(np.mean(absolute * np.asarray(signs)))))
    return float(np.mean(np.asarray(outcomes) >= observed - 1e-12))


def bootstrap_ci(
    deltas: np.ndarray,
    repeats: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(repeats), len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def v160_reference_rows(v160_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(v160_dir / "v160_confirmation_metrics.csv")
    variants = {
        "h1_strict_full_real": "ours_h1_strict",
        "h6_guard10_full_real": "ours_h6_utility",
    }
    records: list[dict[str, Any]] = []
    for variant, method in variants.items():
        selected = source[source.variant.eq(variant)].copy()
        if len(selected) != len(CONFIRMATION_MOVIES) * 4:
            raise RuntimeError(
                f"Unexpected v160 coverage for {variant}: {len(selected)}"
            )
        for row in selected.itertuples(index=False):
            records.append(
                {
                    "method": method,
                    "test_movie": int(row.test_movie),
                    "seed": "ensemble",
                    "evaluation": "frozen_v160",
                    "horizon": int(row.horizon),
                    "windows": int(row.windows),
                    "component_rmse": float(row.component_rmse),
                    "vector_rmse": float(row.vector_rmse),
                    "component_r2": float(row.component_r2),
                    "vector_r2": float(row.vector_r2),
                    "r2": float(row.r2),
                }
            )
    return pd.DataFrame(records)


def aggregate_movie_metrics(movie_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "component_rmse",
        "vector_rmse",
        "r2",
    ]
    grouped = (
        movie_metrics.groupby(["method", "horizon"], as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            vector_rmse_mean=("vector_rmse", "mean"),
            r2_mean=("r2", "mean"),
            min_movie_rmse=("component_rmse", "min"),
            max_movie_rmse=("component_rmse", "max"),
        )
        .sort_values(["horizon", "component_rmse_mean", "method"])
        .reset_index(drop=True)
    )
    if not set(numeric).issubset(movie_metrics):
        raise RuntimeError("Metric schema incomplete")
    return grouped


def pairwise_statistics(
    movie_metrics: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    references = ("ours_h1_strict", "ours_h6_utility")
    comparators = ("hgbdt_track", "gru_track", "kalmannet")
    for horizon in (1, 2, 4, 6):
        table = movie_metrics[movie_metrics.horizon.eq(horizon)].pivot(
            index="test_movie",
            columns="method",
            values="component_rmse",
        )
        for reference in references:
            for comparator in comparators:
                if reference not in table or comparator not in table:
                    continue
                paired = table[[reference, comparator]].dropna()
                delta = (
                    paired[comparator].to_numpy(np.float64)
                    - paired[reference].to_numpy(np.float64)
                )
                lo, hi = bootstrap_ci(
                    delta,
                    repeats=int(repeats),
                    seed=int(seed) + horizon * 101 + len(records),
                )
                records.append(
                    {
                        "reference": reference,
                        "comparator": comparator,
                        "horizon": int(horizon),
                        "movies": int(len(delta)),
                        "reference_rmse_mean": float(paired[reference].mean()),
                        "comparator_rmse_mean": float(paired[comparator].mean()),
                        "mean_comparator_minus_reference": float(delta.mean()),
                        "relative_gain_percent": float(
                            100.0
                            * (
                                paired[comparator].mean()
                                - paired[reference].mean()
                            )
                            / max(float(paired[comparator].mean()), 1e-12)
                        ),
                        "movies_reference_better": int(np.sum(delta > 0)),
                        "exact_sign_flip_p": exact_sign_flip_p(delta),
                        "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi,
                    }
                )
    return pd.DataFrame(records)


def report_text(
    contract: pd.DataFrame,
    aggregate: pd.DataFrame,
    pairwise: pd.DataFrame,
    selection: pd.DataFrame,
) -> str:
    h1 = aggregate[aggregate.horizon.eq(1)][
        ["method", "component_rmse_mean", "r2_mean"]
    ]
    h6 = aggregate[aggregate.horizon.eq(6)][
        ["method", "component_rmse_mean", "r2_mean"]
    ]
    joined = h1.merge(h6, on="method", suffixes=("_h1", "_h6"))
    joined = joined.sort_values("component_rmse_mean_h6")
    lines = [
        "# v193 Frozen Learned-Comparator Confirmation",
        "",
        "## Contract",
        "",
        "- Fit movies: 1--4.",
        "- Validation and all model selection: movie 5 only.",
        "- Frozen confirmation: movies 10--16.",
        "- Three optimizer seeds are ensembled before confirmation metrics.",
        "- No confirmation metric is used for fitting, early stopping, representation selection, or method selection.",
        "- HGBDT is track-native; the unavailable visual-v52 packet is not imputed.",
        "",
        f"Contract rows: train {int(contract.iloc[0].rows):,}, "
        f"validation {int(contract.iloc[1].rows):,}, "
        f"confirmation {int(contract.iloc[2].rows):,}.",
        "",
        "## Main Results",
        "",
        "| Method | h1 RMSE | h1 R2 | h6 RMSE | h6 R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in joined.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.component_rmse_mean_h1:.3f} | "
            f"{row.r2_mean_h1:.3f} | {row.component_rmse_mean_h6:.3f} | "
            f"{row.r2_mean_h6:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
        ]
    )
    for row in selection.itertuples(index=False):
        lines.append(
            f"- {row.method}, seed {row.seed}: validation-only selection "
            f"`{row.selection}`."
        )
    lines.extend(
        [
            "",
            "## Paired h6 Tests",
            "",
            "| Reference | Comparator | Delta, px | 95% CI | Positive movies | Exact p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise[pairwise.horizon.eq(6)].itertuples(index=False):
        lines.append(
            f"| {row.reference} | {row.comparator} | "
            f"{row.mean_comparator_minus_reference:.3f} | "
            f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}] | "
            f"{row.movies_reference_better}/{row.movies} | "
            f"{row.exact_sign_flip_p:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This table closes the learned-comparator gap for the frozen "
            "configuration test. It does not turn movies 10--16 into a fully "
            "hypothesis-naive prospective cohort because they appeared in older "
            "broad screening. The valid claim is generalization of the frozen "
            "current configuration without confirmation-time tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-cache",
        type=Path,
        default=DEFAULT_CONFIRMATION_CACHE,
    )
    parser.add_argument(
        "--development-cache",
        type=Path,
        default=DEFAULT_DEVELOPMENT_CACHE,
    )
    parser.add_argument("--v160-dir", type=Path, default=DEFAULT_V160)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument("--methods", default="hgbdt_track,gru_track,kalmannet")
    parser.add_argument(
        "--validation-horizon-weights",
        default="1:0.90,2:0.05,4:0.03,6:0.02",
    )
    parser.add_argument("--history-lags", type=int, default=8)
    parser.add_argument("--flow-k", type=int, default=16)
    parser.add_argument("--hgbdt-learning-rates", default="0.04,0.08")
    parser.add_argument("--hgbdt-max-leaf-nodes", default="15,31")
    parser.add_argument("--hgbdt-l2", default="1,10")
    parser.add_argument("--hgbdt-max-iter", type=int, default=180)
    parser.add_argument("--hgbdt-min-samples-leaf", type=int, default=30)
    parser.add_argument("--eta-grid", default="0,0.1,0.25,0.5,0.75,1,1.25")
    parser.add_argument("--gru-history-lags", type=int, default=8)
    parser.add_argument("--gru-hidden", type=int, default=96)
    parser.add_argument("--gru-epochs", type=int, default=24)
    parser.add_argument("--gru-batch-size", type=int, default=1024)
    parser.add_argument(
        "--kalmannet-device",
        choices=("auto", "cpu", "mps"),
        default="cpu",
    )
    parser.add_argument("--kalmannet-epochs", type=int, default=80)
    parser.add_argument("--kalmannet-patience", type=int, default=12)
    parser.add_argument("--kalmannet-cumulative-weight", type=float, default=0.05)
    parser.add_argument("--kalmannet-bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--kalmannet-parallel-jobs", type=int, default=3)
    parser.add_argument("--kalmannet-threads-per-job", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=193)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-kalmannet", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    horizons = parse_ints(args.horizons)
    methods = parse_strings(args.methods)
    unknown = sorted(set(methods) - {"hgbdt_track", "gru_track", "kalmannet"})
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if tuple(seeds) != (7, 42, 123):
        raise ValueError("Frozen confirmation seeds are 7,42,123")
    if tuple(horizons) != (1, 2, 4, 6):
        raise ValueError("Frozen confirmation horizons are 1,2,4,6")

    bundles, contract = load_contract_bundles(
        args.confirmation_cache,
        args.development_cache,
    )
    contract.to_csv(args.out_dir / "v193_data_contract.csv", index=False)
    expected_keys = bundles[2].rows[list(KEYS)].to_numpy(np.int64, copy=True)

    selection_rows: list[dict[str, Any]] = []
    audit_tables: list[pd.DataFrame] = []
    sweep_tables: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    hgbdt_packs: dict[tuple[str, str], v111.FeaturePack] | None = None

    if not args.skip_training and "hgbdt_track" in methods:
        hgbdt_packs, hgbdt_audit = prepare_hgbdt_features(bundles, args)
        audit_tables.append(
            hgbdt_audit.assign(method="hgbdt_track", seed="shared")
        )

    if not args.skip_training:
        for seed in seeds:
            prediction_payload: dict[str, np.ndarray] = {
                "contract__row_keys": expected_keys,
                "contract__h1_target_sha256": np.asarray(
                    array_sha256(bundles[2].target_steps[:, 0])
                ),
            }
            if "hgbdt_track" in methods:
                assert hgbdt_packs is not None
                started = time.perf_counter()
                prediction, sweep, metadata = train_hgbdt(
                    bundles,
                    hgbdt_packs,
                    args,
                    seed,
                )
                prediction_payload["hgbdt_track__prediction"] = prediction
                sweep_tables.append(sweep)
                metadata_rows.append(
                    {
                        "method": "hgbdt_track",
                        "seconds": time.perf_counter() - started,
                        **metadata,
                    }
                )
                selection_rows.append(
                    {
                        "method": "hgbdt_track",
                        "seed": int(seed),
                        "selection": (
                            f"{metadata['selected_input_mode']}; "
                            f"eta={metadata['selected_eta']}; "
                            f"config={json.dumps(metadata['hyperparameters'], sort_keys=True)}"
                        ),
                    }
                )
            if "gru_track" in methods:
                started = time.perf_counter()
                prediction, metadata = train_gru(bundles, args, seed)
                prediction_payload["gru_track__prediction"] = prediction
                metadata_rows.append(
                    {
                        "method": "gru_track",
                        "seconds": time.perf_counter() - started,
                        **metadata,
                    }
                )
                selection_rows.append(
                    {
                        "method": "gru_track",
                        "seed": int(seed),
                        "selection": (
                            f"best_epoch={metadata['best_epoch']}; "
                            f"val_score={metadata['val_weighted_rolling_rmse']:.6f}"
                        ),
                    }
                )
            np.savez_compressed(
                args.out_dir / f"v193_predictions_seed{seed}.npz",
                **prediction_payload,
            )

    kalmannet_jobs = pd.DataFrame()
    if "kalmannet" in methods and not args.skip_kalmannet:
        kalmannet_jobs = run_kalmannet_jobs(args, seeds)
        kalmannet_jobs.to_csv(
            args.out_dir / "v193_kalmannet_jobs.csv",
            index=False,
        )

    predictions_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in methods
    }
    seed_metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        direct_path = args.out_dir / f"v193_predictions_seed{seed}.npz"
        if any(method in methods for method in ("hgbdt_track", "gru_track")):
            if not direct_path.exists():
                raise FileNotFoundError(direct_path)
            archive = np.load(direct_path)
            if not np.array_equal(
                archive["contract__row_keys"].astype(np.int64),
                expected_keys,
            ):
                raise RuntimeError(f"Direct comparator key mismatch: seed {seed}")
            for method in ("hgbdt_track", "gru_track"):
                if method not in methods:
                    continue
                prediction = safe(archive[f"{method}__prediction"])
                predictions_by_method[method].append(prediction)
                seed_metric_rows.extend(
                    movie_metric_rows(
                        bundles[2],
                        prediction,
                        method,
                        horizons,
                        evaluation="individual_seed",
                        seed=int(seed),
                    )
                )
        if "kalmannet" in methods:
            output = args.out_dir / "kalmannet" / f"seed{seed}"
            prediction, selected, sentinel = load_kalmannet_prediction(
                output,
                expected_keys,
            )
            predictions_by_method["kalmannet"].append(prediction)
            seed_metric_rows.extend(
                movie_metric_rows(
                    bundles[2],
                    prediction,
                    "kalmannet",
                    horizons,
                    evaluation="individual_seed",
                    seed=int(seed),
                )
            )
            metadata_rows.append(
                {
                    "method": "kalmannet",
                    "seed": int(seed),
                    "selected_input_mode": selected,
                    "selection_scope": "movie_5_only",
                    "fit_scope": "movies_1_4_only",
                    "no_future_sentinel_pass": bool(sentinel["pass"]),
                }
            )
            selection_rows.append(
                {
                    "method": "kalmannet",
                    "seed": int(seed),
                    "selection": f"{selected}; validation movie 5 only",
                }
            )

    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_metrics.to_csv(args.out_dir / "v193_seed_metrics.csv", index=False)

    ensemble_rows: list[dict[str, Any]] = []
    ensemble_payload: dict[str, np.ndarray] = {
        "contract__row_keys": expected_keys,
        "contract__test_key_sha256": np.asarray(array_sha256(expected_keys)),
        "contract__test_h1_target_sha256": np.asarray(
            array_sha256(bundles[2].target_steps[:, 0])
        ),
    }
    for method, predictions in predictions_by_method.items():
        if len(predictions) != len(seeds):
            raise RuntimeError(
                f"{method} prediction seed coverage {len(predictions)}/{len(seeds)}"
            )
        ensemble = safe(np.mean(np.stack(predictions, axis=0), axis=0))
        ensemble_payload[f"{method}__ensemble_prediction"] = ensemble
        ensemble_rows.extend(
            movie_metric_rows(
                bundles[2],
                ensemble,
                method,
                horizons,
                evaluation="three_seed_prediction_ensemble",
                seed="ensemble",
            )
        )

    v160_rows = v160_reference_rows(args.v160_dir)
    movie_metrics = pd.concat(
        [pd.DataFrame(ensemble_rows), v160_rows],
        ignore_index=True,
        sort=False,
    )
    movie_metrics.to_csv(args.out_dir / "v193_movie_metrics.csv", index=False)
    aggregate = aggregate_movie_metrics(movie_metrics)
    aggregate.to_csv(args.out_dir / "v193_aggregate.csv", index=False)
    pairwise = pairwise_statistics(
        movie_metrics,
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    pairwise.to_csv(args.out_dir / "v193_pairwise_statistics.csv", index=False)

    np.savez_compressed(
        args.out_dir / "v193_ensemble_predictions.npz",
        **ensemble_payload,
    )
    if sweep_tables:
        pd.concat(sweep_tables, ignore_index=True).to_csv(
            args.out_dir / "v193_hgbdt_validation_sweep.csv",
            index=False,
        )
    if audit_tables:
        pd.concat(audit_tables, ignore_index=True).to_csv(
            args.out_dir / "v193_causal_audit.csv",
            index=False,
        )
    selection = pd.DataFrame(selection_rows).drop_duplicates().sort_values(
        ["method", "seed"]
    )
    selection.to_csv(args.out_dir / "v193_validation_selection.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(
        args.out_dir / "v193_model_metadata.csv",
        index=False,
    )

    article_table = (
        aggregate[aggregate.horizon.isin([1, 6])]
        .pivot(
            index="method",
            columns="horizon",
            values=["component_rmse_mean", "r2_mean"],
        )
        .reset_index()
    )
    article_table.columns = [
        "method",
        "h1_component_rmse",
        "h6_component_rmse",
        "h1_r2",
        "h6_r2",
    ]
    article_table = article_table.sort_values("h6_component_rmse")
    article_table.to_csv(args.out_dir / "v193_article_table.csv", index=False)

    report = report_text(contract, aggregate, pairwise, selection)
    (args.out_dir / "v193_decision_report.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__)),
        "confirmation_cache": str(args.confirmation_cache.resolve()),
        "development_cache": str(args.development_cache.resolve()),
        "v160_reference": str(args.v160_dir.resolve()),
        "fit_movies": list(TRAIN_MOVIES),
        "validation_movies": list(VALIDATION_MOVIES),
        "confirmation_movies": list(CONFIRMATION_MOVIES),
        "seeds": seeds,
        "horizons": horizons,
        "methods": methods,
        "confirmation_metrics_used_for_training_or_selection": False,
        "hgbdt_visual_v52_features_used": False,
        "three_seed_prediction_ensemble": True,
        "outputs": {
            path.name: file_sha256(path)
            for path in sorted(args.out_dir.iterdir())
            if path.is_file() and path.name != "v193_run_manifest.json"
        },
    }
    write_json(args.out_dir / "v193_run_manifest.json", manifest)
    print(report, flush=True)


if __name__ == "__main__":
    main()
