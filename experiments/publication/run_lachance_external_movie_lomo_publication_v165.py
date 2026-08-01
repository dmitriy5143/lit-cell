#!/usr/bin/env python3
"""Complete and freeze publication-grade external LaChance movie LOMO.

The runner reuses verified v164 pilot folds, executes every missing HUVEC and
MDA-MB-231 outer fold through disjoint resumable shards, invokes the canonical
v164 all-movie aggregator, and adds movie-level confidence intervals, exact
sign tests, causal-control comparisons, and a content-hashed provenance bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_external_movie_lomo_v164 as v164  # noqa: E402


V164_SCRIPT = SCRIPTS / "run_lachance_external_movie_lomo_v164.py"
V97_SCRIPT = SCRIPTS / "run_lachance_causal_innovation_state_space_v97.py"
V162_SCRIPT = SCRIPTS / "run_lachance_dimensionless_multidomain_transport_v162.py"
BUILD_SCRIPT = SCRIPTS / "build_lachance_online_track_anchor_cache_v97.py"
DEFAULT_PILOT = (
    ROOT
    / "outputs"
    / "lachance_external_movie_lomo_v164_pilot_2026-07-27"
)
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "lachance_external_movie_lomo_v165_publication_full_2026-07-27"
)
DOMAINS = ("HUVEC", "MDAMB231")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--huvec-device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--mda-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--huvec-workers", type=int, default=1)
    parser.add_argument("--mda-workers", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--skip-folds", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def checkpoint_for(fold: Path, dataset: str) -> Path:
    return fold / "v97" / f"{v164.VARIANTS[dataset]}.pt"


def fold_complete(fold: Path, dataset: str) -> bool:
    required = [
        fold / "anchor_cache" / "native_cache_status.json",
        checkpoint_for(fold, dataset),
        fold / "transport_h1_strict_metrics.csv",
        fold / "transport_h1_strict_selection.csv",
        fold / "transport_h6_guard10_metrics.csv",
        fold / "transport_h6_guard10_selection.csv",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def bootstrap_ci(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0
    positives = int((values > 0).sum())
    return float(binomtest(positives, len(values), 0.5).pvalue)


def copy_verified_pilot(
    pilot: Path,
    output: Path,
) -> list[dict[str, Any]]:
    reused: list[dict[str, Any]] = []
    for dataset in DOMAINS:
        source_domain = pilot / dataset
        if not source_domain.exists():
            continue
        for source in sorted(source_domain.glob("test_*")):
            if not fold_complete(source, dataset):
                continue
            destination = output / dataset / source.name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
            source_checkpoint = checkpoint_for(source, dataset)
            destination_checkpoint = checkpoint_for(destination, dataset)
            source_hash = sha256(source_checkpoint)
            destination_hash = sha256(destination_checkpoint)
            if source_hash != destination_hash:
                raise RuntimeError(
                    f"Pilot checkpoint copy mismatch: {source} -> {destination}"
                )
            reused.append(
                {
                    "dataset": dataset,
                    "fold": source.name,
                    "source": str(source),
                    "destination": str(destination),
                    "checkpoint_sha256": source_hash,
                }
            )
    reuse_path = output / "v165_reused_pilot_folds.json"
    reuse_path.write_text(
        json.dumps(reused, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reused


def missing_movies(output: Path, dataset: str) -> list[int]:
    return [
        movie
        for movie in v164.available_movies(dataset)
        if not fold_complete(output / dataset / f"test_{movie:02d}", dataset)
    ]


def shard_command(
    dataset: str,
    movies: list[int],
    device: str,
    output: Path,
    seed: int,
    shard_id: str,
) -> list[str]:
    heldouts = ",".join(str(movie) for movie in movies)
    return [
        sys.executable,
        str(V164_SCRIPT),
        "--datasets",
        dataset,
        "--heldouts",
        f"{dataset}={heldouts}",
        "--device",
        device,
        "--seed",
        str(seed),
        "--out-dir",
        str(output),
        "--folds-only",
        "--shard-id",
        shard_id,
    ]


def launch_shards(
    output: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.huvec_workers < 1 or args.mda_workers < 1:
        raise ValueError("Worker counts must be positive")
    domain_specs = [
        (
            "HUVEC",
            missing_movies(output, "HUVEC"),
            args.huvec_device,
            args.huvec_workers,
        ),
        (
            "MDAMB231",
            missing_movies(output, "MDAMB231"),
            args.mda_device,
            args.mda_workers,
        ),
    ]
    specs: list[tuple[str, list[int], str, str]] = []
    records: list[dict[str, Any]] = []
    for dataset, movies, device, requested_workers in domain_specs:
        if not movies:
            records.append(
                {
                    "dataset": dataset,
                    "device": device,
                    "movies": [],
                    "returncode": 0,
                    "status": "already_complete",
                }
            )
            continue
        workers = min(requested_workers, len(movies))
        chunks = [movies[index::workers] for index in range(workers)]
        for index, chunk in enumerate(chunks, start=1):
            shard_id = f"{dataset.lower()}_{device}_w{index:02d}"
            specs.append((dataset, chunk, device, shard_id))
    running: list[
        tuple[
            str,
            list[int],
            str,
            str,
            subprocess.Popen[str],
            Any,
            Path,
        ]
    ] = []
    for dataset, movies, device, shard_id in specs:
        command = shard_command(
            dataset,
            movies,
            device,
            output,
            args.seed,
            shard_id,
        )
        log_path = logs / f"{shard_id}.log"
        handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[variable] = str(args.cpu_threads)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        running.append(
            (
                dataset,
                movies,
                device,
                shard_id,
                process,
                handle,
                log_path,
            )
        )
        print(
            f"[v165] launched {shard_id} folds={len(movies)} "
            f"movies={movies} device={device} pid={process.pid}",
            flush=True,
        )
    try:
        while running:
            next_running = []
            failure: tuple[str, Path] | None = None
            for item in running:
                (
                    dataset,
                    movies,
                    device,
                    shard_id,
                    process,
                    handle,
                    log_path,
                ) = item
                returncode = process.poll()
                if returncode is None:
                    next_running.append(item)
                    continue
                if not handle.closed:
                    handle.close()
                records.append(
                    {
                        "dataset": dataset,
                        "device": device,
                        "shard_id": shard_id,
                        "movies": movies,
                        "returncode": int(returncode),
                        "status": "complete" if returncode == 0 else "failed",
                        "log": str(log_path),
                    }
                )
                if returncode != 0 and failure is None:
                    failure = (shard_id, log_path)
                else:
                    print(f"[v165] completed {shard_id}", flush=True)
            running = next_running
            if failure is not None:
                shard_id, log_path = failure
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8").splitlines()[-80:]
                )
                raise RuntimeError(f"{shard_id} failed:\n{tail}")
            if running:
                time.sleep(15)
    finally:
        for _, _, _, _, process, handle, _ in running:
            if process.poll() is None:
                process.terminate()
        for _, _, _, _, process, handle, _ in running:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not handle.closed:
                handle.close()
    return records


def aggregate_all(output: Path, seed: int) -> None:
    command = [
        sys.executable,
        str(V164_SCRIPT),
        "--datasets",
        ",".join(DOMAINS),
        "--all-movies",
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--out-dir",
        str(output),
        "--skip-training",
    ]
    log_path = output / "logs" / "aggregate_all.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8").splitlines()[-100:]
        )
        raise RuntimeError(f"All-movie aggregation failed:\n{tail}")


def movie_statistics(
    metrics: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["dataset", "objective", "control", "horizon"]
    for index, (key, group) in enumerate(metrics.groupby(keys, sort=True)):
        gains = group.rmse_improvement_percent.to_numpy(dtype=np.float64)
        deltas = (
            group.baseline_component_rmse - group.component_rmse
        ).to_numpy(dtype=np.float64)
        rmse_low, rmse_high = bootstrap_ci(
            group.component_rmse.to_numpy(dtype=np.float64),
            repeats,
            seed + 101 * index + 2,
        )
        gain_low, gain_high = bootstrap_ci(
            gains,
            repeats,
            seed + 101 * index,
        )
        delta_low, delta_high = bootstrap_ci(
            deltas,
            repeats,
            seed + 101 * index + 1,
        )
        records.append(
            {
                **dict(zip(keys, key)),
                "outer_folds": int(group.outer_test_movie.nunique()),
                "component_rmse_macro": float(group.component_rmse.mean()),
                "component_rmse_ci_low": rmse_low,
                "component_rmse_ci_high": rmse_high,
                "vector_rmse_macro": float(group.vector_rmse.mean()),
                "r2_macro": float(group.r2.mean()),
                "baseline_component_rmse_macro": float(
                    group.baseline_component_rmse.mean()
                ),
                "gain_percent_macro": float(gains.mean()),
                "gain_percent_ci_low": gain_low,
                "gain_percent_ci_high": gain_high,
                "rmse_delta_macro": float(deltas.mean()),
                "rmse_delta_ci_low": delta_low,
                "rmse_delta_ci_high": delta_high,
                "positive_folds": int((deltas > 0).sum()),
                "sign_test_p_two_sided": exact_sign_p(deltas),
            }
        )
    return pd.DataFrame(records)


def causal_control_statistics(
    metrics: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    index_columns = [
        "dataset",
        "objective",
        "horizon",
        "outer_test_movie",
    ]
    pivot = metrics.pivot(
        index=index_columns,
        columns="control",
        values="component_rmse",
    ).reset_index()
    records: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(
        pivot.groupby(["dataset", "objective", "horizon"], sort=True)
    ):
        for control_index, control in enumerate(("stale_time", "wrong_cell")):
            advantage = (
                group[control] - group["real"]
            ).to_numpy(dtype=np.float64)
            low, high = bootstrap_ci(
                advantage,
                repeats,
                seed + 10_000 + group_index * 17 + control_index,
            )
            records.append(
                {
                    "dataset": key[0],
                    "objective": key[1],
                    "horizon": key[2],
                    "comparison": f"real_vs_{control}",
                    "outer_folds": len(group),
                    "rmse_advantage_macro": float(advantage.mean()),
                    "rmse_advantage_ci_low": low,
                    "rmse_advantage_ci_high": high,
                    "real_better_folds": int((advantage > 0).sum()),
                    "sign_test_p_two_sided": exact_sign_p(advantage),
                }
            )
    return pd.DataFrame(records)


def content_manifest(
    output: Path,
    args: argparse.Namespace,
    reused: list[dict[str, Any]],
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    code_paths = [Path(__file__), V164_SCRIPT, V97_SCRIPT, V162_SCRIPT, BUILD_SCRIPT]
    tables = [
        path
        for dataset in DOMAINS
        for path in sorted(
            (v164.TABLE_ROOT / dataset).glob(f"{dataset}_*_tracks.csv")
        )
    ]
    checkpoints = [
        checkpoint_for(output / dataset / f"test_{movie:02d}", dataset)
        for dataset in DOMAINS
        for movie in v164.available_movies(dataset)
    ]
    training_configs = [
        output
        / dataset
        / f"test_{movie:02d}"
        / "v97"
        / "run_config.json"
        for dataset in DOMAINS
        for movie in v164.available_movies(dataset)
    ]
    fold_inventory = []
    for dataset in DOMAINS:
        for movie in v164.available_movies(dataset):
            fold = output / dataset / f"test_{movie:02d}"
            checkpoint = checkpoint_for(fold, dataset)
            config_path = fold / "v97" / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            fold_inventory.append(
                {
                    "dataset": dataset,
                    "outer_test_movie": movie,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": (
                        None if args.skip_hashes else sha256(checkpoint)
                    ),
                    "training_config": str(config_path.resolve()),
                    "training_config_sha256": (
                        None if args.skip_hashes else sha256(config_path)
                    ),
                    "training_device": config["device"],
                    "training_seed": config["seed"],
                    "variant": config["evaluation_variant"],
                }
            )
    artifact_names = [
        "run_manifest.json",
        "v165_reused_pilot_folds.json",
        "v164_nested_lomo_metrics.csv",
        "v164_nested_lomo_summary.csv",
        "v164_nested_lomo_operating_points.csv",
        "v164_nested_lomo_fold_decision.csv",
        "v164_nested_lomo_contract.csv",
        "v165_movie_level_statistics.csv",
        "v165_causal_control_statistics.csv",
        "v165_publication_summary.csv",
        "v165_publication_report.md",
    ]
    shard_paths = sorted((output / "shards").glob("*.json"))
    artifacts = [output / name for name in artifact_names]
    artifacts.extend(shard_paths)
    shard_history = [
        json.loads(path.read_text(encoding="utf-8")) for path in shard_paths
    ]
    timed_shards = [
        row
        for row in shard_history
        if row.get("elapsed_seconds") is not None
        and row.get("completed_outer_folds", 0) > 0
    ]

    def hashes(paths: list[Path]) -> dict[str, str | None]:
        return {
            str(path.resolve()): (
                None if args.skip_hashes else sha256(path)
            )
            for path in paths
        }

    return {
        "schema_version": 1,
        "protocol": "causal rolling h1; outer movie absent from train/validation",
        "datasets": {
            dataset: v164.available_movies(dataset) for dataset in DOMAINS
        },
        "folds_expected": sum(
            len(v164.available_movies(dataset)) for dataset in DOMAINS
        ),
        "seed": args.seed,
        "target_leakage": False,
        "pilot_reuse": reused,
        "shards": shards,
        "shard_history": shard_history,
        "parallel_fold_wall_seconds_estimate": (
            max(row["elapsed_seconds"] for row in timed_shards)
            if timed_shards
            else None
        ),
        "sum_shard_elapsed_seconds": (
            sum(row["elapsed_seconds"] for row in timed_shards)
            if timed_shards
            else None
        ),
        "fold_inventory": fold_inventory,
        "code_sha256": hashes(code_paths),
        "input_table_sha256": hashes(tables),
        "checkpoint_sha256": hashes(checkpoints),
        "training_config_sha256": hashes(training_configs),
        "artifact_sha256": hashes(artifacts),
        "args": finite(vars(args)),
    }


def build_report(
    output: Path,
    movie_stats: pd.DataFrame,
    control_stats: pd.DataFrame,
) -> None:
    primary = movie_stats[
        movie_stats.control.eq("real")
        & movie_stats.horizon.isin([1, 6])
    ].copy()
    h6_controls = control_stats[
        control_stats.objective.eq("h6_guard10")
        & control_stats.horizon.eq(6)
    ].copy()
    summary = primary[
        [
            "dataset",
            "objective",
            "control",
            "horizon",
            "outer_folds",
            "component_rmse_macro",
            "component_rmse_ci_low",
            "component_rmse_ci_high",
            "r2_macro",
            "gain_percent_macro",
            "gain_percent_ci_low",
            "gain_percent_ci_high",
            "positive_folds",
            "sign_test_p_two_sided",
        ]
    ]
    summary.to_csv(output / "v165_publication_summary.csv", index=False)
    expected = {
        dataset: len(v164.available_movies(dataset)) for dataset in DOMAINS
    }
    report = [
        "# v165 Full External Movie LOMO",
        "",
        "Every outer test movie is absent from training and inner validation.",
        "The v97 prior and transport calibration are refitted per outer fold.",
        "",
        "## Operating points",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## h6 causal controls",
        "",
        h6_controls.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Completion",
        "",
        f"- HUVEC folds: `{expected['HUVEC']}/{expected['HUVEC']}`.",
        f"- MDA-MB-231 folds: `{expected['MDAMB231']}/{expected['MDAMB231']}`.",
        "- Bootstrap intervals resample outer movies, not cell rows.",
        "- Exact sign tests use outer movies as independent units.",
        "- Optimizer seed is fixed at 42; seed sensitivity is documented by v162.",
        "",
        "This is protocol-specific evidence. It is not a global trajectory-SOTA "
        "claim and is not comparable to fixed-origin ADE/FDE.",
    ]
    (output / "v165_publication_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reused = copy_verified_pilot(args.pilot_dir.resolve(), output)
    shards: list[dict[str, Any]] = []
    if not args.skip_folds:
        shards = launch_shards(output, args)
    remaining = {
        dataset: missing_movies(output, dataset) for dataset in DOMAINS
    }
    if any(remaining.values()):
        raise RuntimeError(f"Incomplete folds after workers: {remaining}")
    aggregate_all(output, args.seed)
    metrics = pd.read_csv(output / "v164_nested_lomo_metrics.csv")
    expected_folds = sum(len(v164.available_movies(name)) for name in DOMAINS)
    actual_folds = metrics[
        ["dataset", "outer_test_movie"]
    ].drop_duplicates()
    if len(actual_folds) != expected_folds:
        raise RuntimeError(
            f"Expected {expected_folds} folds, found {len(actual_folds)}"
        )
    movie_stats = movie_statistics(
        metrics,
        args.bootstrap_repeats,
        args.seed + 165_000,
    )
    control_stats = causal_control_statistics(
        metrics,
        args.bootstrap_repeats,
        args.seed + 166_000,
    )
    movie_stats.to_csv(
        output / "v165_movie_level_statistics.csv",
        index=False,
    )
    control_stats.to_csv(
        output / "v165_causal_control_statistics.csv",
        index=False,
    )
    build_report(output, movie_stats, control_stats)
    manifest = content_manifest(output, args, reused, shards)
    manifest["invocation_elapsed_seconds"] = time.time() - started
    (output / "v165_publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[v165] wrote {output}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
