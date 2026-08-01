#!/usr/bin/env python3
"""Execute disjoint jobs from an already frozen v102 LOMO manifest.

This helper changes only scheduling.  It reconstructs the exact commands and
configuration hashes written by ``run_lachance_online_lomo_benchmark_v102.py``
and delegates execution to that runner's marker-aware ``run_job`` function.
It is intended for parallel fold/seed workers with non-overlapping job ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from pathlib import Path

import run_lachance_online_lomo_benchmark_v102 as v102


def parse_csv_tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def load_jobs(path: Path) -> dict[str, v102.Job]:
    jobs: dict[str, v102.Job] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            job = v102.Job(
                job_id=row["job_id"],
                kind=row["kind"],
                runner=row["runner"],
                fold=row["fold"],
                test_movie=int(row["test_movie"]),
                validation_movie=int(row["validation_movie"]),
                train_movies=[int(value) for value in parse_csv_tokens(row["train_movies"])],
                seed=int(row["seed"]),
                command=shlex.split(row["command"]),
                output_dir=row["output_dir"],
                expected_outputs=list(json.loads(row["expected_outputs"])),
                dependencies=parse_csv_tokens(row["dependencies"]),
            ).finalize()
            if job.config_hash != row["config_hash"]:
                raise RuntimeError(
                    f"frozen manifest hash mismatch for {job.job_id}: "
                    f"computed={job.config_hash}, recorded={row['config_hash']}"
                )
            jobs[job.job_id] = job
    return jobs


def run_locked(job: v102.Job) -> None:
    if v102.job_complete(job):
        print(f"[v102-shard] cached {job.job_id}", flush=True)
        return
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".v102_parallel.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"job appears active in another worker: {job.job_id}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\njob_id={job.job_id}\n".encode("utf-8"))
        os.close(descriptor)
        v102.run_job(job)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen v102_job_manifest.csv")
    parser.add_argument(
        "--job-ids",
        required=True,
        help="Comma-separated runner job ids; dependencies are executed automatically.",
    )
    args = parser.parse_args()

    jobs = load_jobs(args.manifest.resolve())
    selected = parse_csv_tokens(args.job_ids)
    unknown = sorted(set(selected) - set(jobs))
    if unknown:
        raise ValueError(f"unknown v102 job ids: {unknown}")

    for job_id in selected:
        job = jobs[job_id]
        for dependency_id in job.dependencies:
            if dependency_id not in jobs:
                raise RuntimeError(f"missing dependency {dependency_id} for {job_id}")
            run_locked(jobs[dependency_id])
        run_locked(job)


if __name__ == "__main__":
    main()
