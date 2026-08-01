#!/usr/bin/env python3
"""Run exact KalmanNet outer-movie LOMO on frozen v102 anchor caches.

The standard v102 scheduler ties the compute device to the anchor-cache hash.
For KalmanNet this would rebuild an already frozen v52 anchor merely to change
MPS to CPU. This runner instead treats each existing anchor cache as immutable
input, hashes its contracts, and changes only the KalmanNet compute backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_REPLAY_MANIFEST = (
    ROOT
    / "outputs"
    / "lachance_foldlocal_semigroup_confirmation_v157e_full_2026-07-24"
    / "v157e_seed_replay_manifest.json"
)
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_kalmannet_v188_exact_2026-07-29"
)
V98_SCRIPT = SCRIPTS / "run_lachance_kalmannet_online_v98.py"
EXPECTED_OUTPUTS = (
    "v98_online_summary.csv",
    "v98_data_contract.csv",
    "v98_causal_audit.csv",
    "v98_no_future_sentinel.json",
    "v98_provenance.json",
    "v98_model_metadata.csv",
    "run_config.json",
)
HORIZONS = (1, 2, 4, 6)
MOVIES = (1, 2, 3, 4, 5, 6)
SEEDS = (7, 42, 123)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=DEFAULT_REPLAY_MANIFEST,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--only-movies",
        default="",
        help="Optional comma-separated outer movies for a disjoint worker shard.",
    )
    parser.add_argument(
        "--execute-only",
        action="store_true",
        help="Run the selected shard without writing global aggregate files.",
    )
    parser.add_argument("--models", default="cv,ca")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--cumulative-weight", type=float, default=0.05)
    parser.add_argument(
        "--validation-horizon-weights",
        default="1:0.90,2:0.05,4:0.03,6:0.02",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def output_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in EXPECTED_OUTPUTS)


def build_jobs(
    replay_manifest: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    replays = json.loads(replay_manifest.read_text(encoding="utf-8"))
    if not isinstance(replays, list) or len(replays) != 18:
        raise ValueError("Expected the 18-row v157e seed replay manifest")
    jobs = []
    for replay in replays:
        test = int(replay["test_movie"])
        validation = int(replay["validation_movie"])
        seed = int(replay["seed"])
        directory = (
            output
            / "folds"
            / f"test{test:02d}_val{validation:02d}_seed{seed}"
            / "v98"
        )
        command = [
            sys.executable,
            str(V98_SCRIPT),
            "--anchor-cache",
            str(Path(replay["anchor_cache"]).resolve()),
            "--out-dir",
            str(directory.resolve()),
            "--models",
            args.models,
            "--validation-horizon-weights",
            args.validation_horizon_weights,
            "--loss",
            args.loss,
            "--cumulative-weight",
            str(args.cumulative_weight),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]
        job = {
            "job_id": f"v98/test{test:02d}_val{validation:02d}_seed{seed}",
            "runner": "v98",
            "test_movie": test,
            "validation_movie": validation,
            "train_movies": ",".join(
                str(value) for value in replay["train_movies"]
            ),
            "seed": seed,
            "anchor_cache": str(Path(replay["anchor_cache"]).resolve()),
            "anchor_contract_sha256": canonical_sha256(
                replay["anchor_contract_sha256"]
            ),
            "test_key_sha256": replay["splits"]["test"]["key_sha256"],
            "test_target_sha256": replay["splits"]["test"]["target_sha256"],
            "test_row_target_sha256": replay["splits"]["test"][
                "row_target_sha256"
            ],
            "test_rows": int(replay["splits"]["test"]["rows"]),
            "output_dir": str(directory.resolve()),
            "command": command,
            "command_sha256": canonical_sha256(command),
        }
        jobs.append(job)
    keys = {
        (row["test_movie"], row["seed"]) for row in jobs
    }
    expected = {(movie, seed) for movie in MOVIES for seed in SEEDS}
    if keys != expected:
        raise ValueError("Replay manifest does not cover six movies x three seeds")
    return sorted(jobs, key=lambda row: (row["test_movie"], row["seed"]))


def run_job(job: dict[str, Any], force: bool) -> dict[str, Any]:
    directory = Path(job["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "v188_subprocess.log"
    if output_complete(directory) and not force:
        return {**job, "status": "cached", "returncode": 0}
    lock_path = directory / ".v188_job.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"KalmanNet job is active in another worker: {job['job_id']}"
        ) from error
    try:
        os.write(
            descriptor,
            f"pid={os.getpid()}\njob_id={job['job_id']}\n".encode("utf-8"),
        )
        os.close(descriptor)
        descriptor = None
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                + " ".join(job["command"])
                + "\n"
            )
            log.flush()
            process = subprocess.run(
                job["command"],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        complete = output_complete(directory)
        status = (
            "complete"
            if process.returncode == 0 and complete
            else "failed"
        )
        return {
            **job,
            "status": status,
            "returncode": int(process.returncode),
            "elapsed_seconds": time.time() - started,
        }
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def validate_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    directory = Path(job["output_dir"])
    if not output_complete(directory):
        raise FileNotFoundError(f"Incomplete KalmanNet job: {directory}")
    sentinel = json.loads(
        (directory / "v98_no_future_sentinel.json").read_text(
            encoding="utf-8"
        )
    )
    if sentinel.get("pass") is not True:
        raise ValueError(f"Future-target sentinel failed: {directory}")
    if sentinel.get("future_placeholder_read_at_inference") is not False:
        raise ValueError(f"Future placeholder was read: {directory}")
    contract = pd.read_csv(directory / "v98_data_contract.csv")
    if len(contract) != 1:
        raise ValueError(f"Invalid v98 data contract: {directory}")
    row = contract.iloc[0]
    if int(row["future_target_inference_features"]) != 0:
        raise ValueError(f"Future feature found: {directory}")
    if int(row["test_movies"]) != int(job["test_movie"]):
        raise ValueError(f"Wrong test movie in {directory}")
    if int(row["test_rows"]) != int(job["test_rows"]):
        raise ValueError(f"Test row mismatch in {directory}")
    selected_model = str(row["selected_model"])
    summary = pd.read_csv(directory / "v98_online_summary.csv")
    summary = summary[
        summary["family"].eq("kalmannet")
        & summary["control"].eq("real")
        & summary["method"].eq(selected_model)
    ].copy()
    if set(summary["horizon"].astype(int)) != set(HORIZONS):
        raise ValueError(f"Missing horizons in {directory}")
    if summary["method"].nunique() != 1:
        raise ValueError(
            f"Validation-selected model {selected_model!r} is unavailable: "
            f"{directory}"
        )
    records = []
    for metric in summary.to_dict("records"):
        records.append(
            {
                "test_movie": int(job["test_movie"]),
                "validation_movie": int(job["validation_movie"]),
                "train_movies": job["train_movies"],
                "seed": int(job["seed"]),
                "method": "kalmannet",
                "selected_state_model": metric["method"],
                "horizon": int(metric["horizon"]),
                "component_rmse": float(metric["component_rmse"]),
                "vector_rmse": float(metric["vector_rmse"]),
                "r2": float(metric["r2"]),
                "cosine": float(metric["cosine"]),
                "magnitude_ratio": float(metric["magnitude_ratio"]),
                "n_rows": int(metric["n_rows"]),
                "test_key_sha256": job["test_key_sha256"],
                "test_target_sha256": job["test_target_sha256"],
                "test_row_target_sha256": job["test_row_target_sha256"],
                "future_target_inference_features": 0,
                "no_future_sentinel_pass": True,
                "output_dir": str(directory),
            }
        )
    return records


def aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie = (
        raw.groupby(
            [
                "test_movie",
                "validation_movie",
                "train_movies",
                "method",
                "horizon",
                "test_key_sha256",
                "test_target_sha256",
                "test_row_target_sha256",
            ],
            as_index=False,
        )
        .agg(
            component_rmse=("component_rmse", "mean"),
            component_rmse_seed_sd=("component_rmse", "std"),
            vector_rmse=("vector_rmse", "mean"),
            vector_rmse_seed_sd=("vector_rmse", "std"),
            r2=("r2", "mean"),
            r2_seed_sd=("r2", "std"),
            cosine=("cosine", "mean"),
            magnitude_ratio=("magnitude_ratio", "mean"),
            n_rows=("n_rows", "max"),
            n_seeds=("seed", "nunique"),
            selected_models=("selected_state_model", lambda x: ",".join(sorted(x))),
        )
    )
    if not movie["n_seeds"].eq(3).all():
        raise ValueError("KalmanNet movie metric lacks three seeds")
    summary = (
        movie.groupby(["method", "horizon"], as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse=("component_rmse", "mean"),
            component_rmse_movie_std=("component_rmse", "std"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            cosine=("cosine", "mean"),
            magnitude_ratio=("magnitude_ratio", "mean"),
        )
    )
    return movie, summary


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay_manifest = args.replay_manifest.resolve()
    if not replay_manifest.exists():
        raise FileNotFoundError(replay_manifest)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    jobs = build_jobs(replay_manifest, output, args)
    if args.only_movies.strip():
        selected_movies = {
            int(value)
            for value in args.only_movies.split(",")
            if value.strip()
        }
        unknown = selected_movies - set(MOVIES)
        if unknown:
            raise ValueError(f"Unknown outer movies: {sorted(unknown)}")
        jobs = [
            job for job in jobs if job["test_movie"] in selected_movies
        ]
    if not jobs:
        raise ValueError("No KalmanNet jobs selected")
    manifest_path = output / "kalmannet_lomo_job_manifest.csv"
    manifest_frame = pd.DataFrame(
        [
            {
                **{key: value for key, value in job.items() if key != "command"},
                "command": json.dumps(job["command"]),
                "status": (
                    "complete"
                    if output_complete(Path(job["output_dir"]))
                    else "pending"
                ),
            }
            for job in jobs
        ]
    )
    if not args.execute_only:
        manifest_frame.to_csv(manifest_path, index=False)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_job, job, args.force): job["job_id"]
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[v188-kalmannet] {result['job_id']}: {result['status']}",
                flush=True,
            )
            if result["status"] == "failed":
                raise RuntimeError(
                    f"KalmanNet job failed: {result['job_id']}"
                )
    if args.execute_only:
        print(
            f"[v188-kalmannet] execute-only shard complete: {len(jobs)} jobs",
            flush=True,
        )
        return

    raw_records = []
    for job in jobs:
        raw_records.extend(validate_job(job))
    raw = pd.DataFrame(raw_records)
    movie, summary = aggregate(raw)
    raw.to_csv(output / "kalmannet_lomo_raw_fold_seed_metrics.csv", index=False)
    movie.to_csv(output / "kalmannet_lomo_movie_metrics.csv", index=False)
    summary.to_csv(output / "kalmannet_lomo_summary.csv", index=False)
    completed_manifest = manifest_frame.copy()
    completed_manifest["status"] = completed_manifest["output_dir"].map(
        lambda value: (
            "complete" if output_complete(Path(value)) else "pending"
        )
    )
    completed_manifest.to_csv(manifest_path, index=False)
    contract = pd.DataFrame(
        [
            {
                "protocol": "strict causal streaming/receding h1",
                "dataset": "MDCK_Bulk",
                "outer_movies": "1,2,3,4,5,6",
                "seeds": "7,42,123",
                "independent_unit": "movie",
                "seed_aggregation": "mean within movie",
                "selection": "validation movie only",
                "models": args.models,
                "loss": args.loss,
                "epochs": args.epochs,
                "patience": args.patience,
                "device": args.device,
                "anchor_device_independence": (
                    "frozen anchor arrays reused; only v98 compute backend changed"
                ),
                "target_leakage": False,
                "no_future_sentinel_all": True,
            }
        ]
    )
    contract.to_csv(output / "kalmannet_lomo_contract.csv", index=False)
    artifact_paths = [
        manifest_path,
        output / "kalmannet_lomo_raw_fold_seed_metrics.csv",
        output / "kalmannet_lomo_movie_metrics.csv",
        output / "kalmannet_lomo_summary.csv",
        output / "kalmannet_lomo_contract.csv",
    ]
    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "jobs": len(jobs),
        "workers": args.workers,
        "source_replay_manifest": str(replay_manifest),
        "source_replay_manifest_sha256": sha256(replay_manifest),
        "runner_sha256": sha256(Path(__file__)),
        "v98_runner_sha256": sha256(V98_SCRIPT),
        "artifact_sha256": {
            str(path): sha256(path) for path in artifact_paths
        },
        "elapsed_seconds": time.time() - started,
        "args": {
            "device": args.device,
            "models": args.models,
            "epochs": args.epochs,
            "patience": args.patience,
            "loss": args.loss,
            "cumulative_weight": args.cumulative_weight,
            "validation_horizon_weights": args.validation_horizon_weights,
        },
    }
    (output / "kalmannet_lomo_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# KalmanNet Exact Outer-Movie LOMO v188",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "- Prediction is issued before the next observation.",
        "- Hyperparameters/state model are selected on the validation movie.",
        "- Three optimizer seeds are averaged within each outer test movie.",
        "- All future-target sentinels passed.",
        "- Frozen v102 anchor arrays were reused without refitting.",
        "",
        f"Elapsed: `{(time.time() - started) / 60.0:.2f}` minutes.",
    ]
    (output / "kalmannet_lomo_status_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(output / "kalmannet_lomo_status_report.md", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
