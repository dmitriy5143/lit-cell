#!/usr/bin/env python3
"""Publication-grade outer leave-one-movie-out benchmark for online h1 forecasting.

The outer unit is an MDCK_Bulk movie, never a cell or a sampled row.  Every one
of movies 1..6 is used as test exactly once.  A deterministic, distinct movie
is reserved for validation and the remaining four movies are used for fitting.
The v52 anchor is rebuilt inside each outer fold; its train anchors are produced
out-of-fold by v84, so the online v97/v98/v99 runners never receive an in-sample
anchor for a training movie.

Typical use::

    # Validate dependencies and freeze the complete command manifest.
    python3 experiments/publication/run_lachance_online_lomo_benchmark_v102.py --dry-run

    # Execute/resume all jobs, then analyse completed folds.
    python3 experiments/publication/run_lachance_online_lomo_benchmark_v102.py

    # Rebuild publication tables from cached runner outputs only.
    python3 experiments/publication/run_lachance_online_lomo_benchmark_v102.py --analyze-only

The script intentionally does not pool cells across movies for inference.
Seeds are averaged within each test movie first; paired tests then operate on
the six movie-level differences.  With six movies the smallest two-sided exact
sign-flip p-value is 2/64 = 0.03125, which is reported explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_FEATURES = (
    ROOT
    / "outputs/lachance_raw_context_v2_grid_dense_index_bulk_2026-07-17"
    / "raw_context_v2_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs/lachance_online_lomo_benchmark_v102_2026-07-21"
DEFAULT_TABLE_ROOT = ROOT / "new_data/lachance_epithelia/tables"
MOVIES = (1, 2, 3, 4, 5, 6)
HORIZONS = (1, 2, 4, 6)
METRICS = ("component_rmse", "vector_rmse")
EPS = 1e-12
ROW_KEYS = ("sequence", "frame", "track_id")

# The publication LOMO must not inherit future changes to v97 parser defaults.
# These are the exact h1-strict settings recorded in the three canonical runs.
FROZEN_V97_H1_STRICT_ARGS = (
    "--horizons", "1,2,4,6",
    "--cumulative-horizons", "2,4,6",
    "--hidden", "128",
    "--history-lags", "6",
    "--epochs", "32",
    "--patience", "7",
    "--min-delta", "0.0001",
    "--tbptt-frames", "8",
    "--lr", "0.0005",
    "--weight-decay", "0.0002",
    "--dropout", "0.06",
    "--correction-bound", "3.5",
    "--nll-weight", "0.12",
    "--innovation-nll-weight", "0.06",
    "--noise-regularization-weight", "0.002",
    "--gain-weight", "0.001",
    "--process-scale-prior", "0.75",
    "--observation-scale-prior", "0.35",
    "--train-missing-rate", "0.15",
    "--train-coordinate-noise-px", "0.35",
    "--reservoir-size", "4096",
    "--clip-grad", "5.0",
    "--eta-grid", "0,0.1,0.2,0.35,0.5,0.75,1,1.25",
    "--uncertainty-scale-grid", "0.5,0.75,1,1.25,1.5,2,3",
    "--kalman-q-grid", "0.1,0.5,1,4,16",
    "--kalman-r-grid", "0.1,0.5,1,4,16",
    "--imm-turn-grid", "0.08,0.15,0.25,0.4",
    "--baseline-history-lags", "8",
    "--baseline-hidden", "96",
    "--baseline-epochs", "24",
    "--baseline-batch-size", "1024",
    "--bootstrap-repeats", "1500",
)

RUNNER_SPECS: dict[str, dict[str, Any]] = {
    "v97": {
        "script": SCRIPTS / "run_lachance_causal_innovation_state_space_v97.py",
        "summary": "v97_online_summary.csv",
        "metadata": "v97_model_metadata.csv",
        "contract": "v97_data_contract.csv",
        "audit": "v97_causal_audit.csv",
    },
    "v98": {
        "script": SCRIPTS / "run_lachance_kalmannet_online_v98.py",
        "summary": "v98_online_summary.csv",
        "metadata": "v98_model_metadata.csv",
        "contract": "v98_data_contract.csv",
        "audit": "v98_causal_audit.csv",
    },
    "v99": {
        "script": SCRIPTS / "run_lachance_online_architecture_benchmark_v99.py",
        "summary": "v99_online_summary.csv",
        "metadata": "v99_model_metadata.csv",
        "contract": "v99_data_contract.csv",
        "audit": "v99_causal_audit.csv",
    },
}


@dataclass(frozen=True)
class Fold:
    test_movie: int
    validation_movie: int
    train_movies: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"test{self.test_movie:02d}_val{self.validation_movie:02d}"


@dataclass
class Job:
    job_id: str
    kind: str
    runner: str
    fold: str
    test_movie: int
    validation_movie: int
    train_movies: list[int]
    seed: int
    command: list[str]
    output_dir: str
    expected_outputs: list[str]
    dependencies: list[str]
    config_hash: str = ""

    def finalize(self) -> "Job":
        payload = asdict(self)
        payload.pop("config_hash", None)
        self.config_hash = stable_hash(payload)
        return self


def csv_tokens(value: str | Iterable[Any]) -> list[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(token) for token in value]


def parse_ints(value: str | Iterable[Any]) -> list[int]:
    return [int(token) for token in csv_tokens(value)]


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(finite(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def row_key_digest(rows: pd.DataFrame) -> str:
    missing = set(ROW_KEYS) - set(rows.columns)
    if missing:
        raise RuntimeError(f"row table is missing identity columns: {sorted(missing)}")
    keys = rows[list(ROW_KEYS)].astype(np.int64).sort_values(list(ROW_KEYS)).to_numpy(np.int64, copy=True)
    return hashlib.sha256(keys.tobytes(order="C")).hexdigest()


def split_row_audit(rows: pd.DataFrame) -> dict[str, Any]:
    missing = set(ROW_KEYS) - set(rows.columns)
    if missing:
        raise RuntimeError(f"row table is missing identity columns: {sorted(missing)}")
    return {
        "rows": int(len(rows)),
        "movies": sorted(int(value) for value in rows["sequence"].unique()),
        "duplicate_keys": int(rows.duplicated(list(ROW_KEYS)).sum()),
        "key_sha256": row_key_digest(rows),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validation_mapping(specification: str, movies: Sequence[int]) -> dict[int, int]:
    """Return a deterministic one-to-one validation rotation.

    The default maps test movie i to the next movie in sorted cyclic order.
    A custom map must remain a derangement and use every validation movie once.
    """

    ordered = tuple(sorted(int(movie) for movie in movies))
    if specification.strip():
        mapping: dict[int, int] = {}
        for token in csv_tokens(specification):
            left, right = token.split(":", maxsplit=1)
            mapping[int(left)] = int(right)
    else:
        mapping = {movie: ordered[(index + 1) % len(ordered)] for index, movie in enumerate(ordered)}
    if set(mapping) != set(ordered):
        raise ValueError(f"validation map keys must be {ordered}, got {sorted(mapping)}")
    if set(mapping.values()) != set(ordered):
        raise ValueError("validation movies must form a one-to-one rotation")
    if any(test == validation for test, validation in mapping.items()):
        raise ValueError("test and validation movies must always be distinct")
    return mapping


def make_folds(movies: Sequence[int], mapping: dict[int, int]) -> list[Fold]:
    universe = tuple(sorted(int(movie) for movie in movies))
    folds: list[Fold] = []
    for test in universe:
        validation = int(mapping[test])
        train = tuple(movie for movie in universe if movie not in (test, validation))
        if len(train) != len(universe) - 2 or set(train) & {test, validation}:
            raise RuntimeError("invalid outer fold construction")
        folds.append(Fold(test, validation, train))
    return folds


def quote_command(command: Sequence[str]) -> str:
    return shlex.join([str(token) for token in command])


def job_marker(job: Job) -> Path:
    return Path(job.output_dir) / ".v102_done.json"


def job_complete(job: Job) -> bool:
    marker = job_marker(job)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("ok")
        and payload.get("config_hash") == job.config_hash
        and all(Path(path).exists() for path in job.expected_outputs)
    )


def build_jobs(args: argparse.Namespace, folds: Sequence[Fold], seeds: Sequence[int]) -> list[Job]:
    jobs: list[Job] = []
    runners = csv_tokens(args.runners)
    unknown = sorted(set(runners) - set(RUNNER_SPECS))
    if unknown:
        raise ValueError(f"unknown runners: {unknown}")
    for fold in folds:
        for seed in seeds:
            fold_seed = f"{fold.name}_seed{seed}"
            anchor_contract = {
                "dataset": args.dataset,
                "features": str(args.features.resolve()),
                "table_root": str(args.table_root.resolve()),
                "train_movies": list(fold.train_movies),
                "validation_movie": fold.validation_movie,
                "test_movie": fold.test_movie,
                "seed": int(seed),
                "horizons": parse_ints(args.horizons),
                "max_rows": [int(args.max_train_rows), int(args.max_val_rows), int(args.max_test_rows)],
                "posterior_epochs": int(args.anchor_posterior_epochs),
                "student_epochs": int(args.anchor_student_epochs),
                "route_k": int(args.anchor_route_k),
                "device": str(args.device),
            }
            anchor_root = args.cache_root / fold_seed / f"v52_anchor_cache_{stable_hash(anchor_contract)}"
            anchor_worker = args.out_dir / "anchor_workers" / fold_seed
            anchor_spec = anchor_worker / "anchor_job_spec.json"
            anchor_id = f"anchor/{fold_seed}"
            anchor_command = [sys.executable, str(SCRIPT), "--worker-anchor-spec", str(anchor_spec)]
            anchor_job = Job(
                job_id=anchor_id,
                kind="anchor",
                runner="v84_exact_v52_anchor",
                fold=fold.name,
                test_movie=fold.test_movie,
                validation_movie=fold.validation_movie,
                train_movies=list(fold.train_movies),
                seed=int(seed),
                command=anchor_command,
                output_dir=str(anchor_worker),
                expected_outputs=[
                    str(anchor_worker / "anchor_worker_status.json"),
                    str(anchor_root / "v102_anchor_ready.json"),
                ],
                dependencies=[],
            ).finalize()
            jobs.append(anchor_job)
            anchor_payload = {
                "job_id": anchor_id,
                "job_config_hash": anchor_job.config_hash,
                "dataset": args.dataset,
                "features": str(args.features.resolve()),
                "table_root": str(args.table_root.resolve()),
                "train_movies": list(fold.train_movies),
                "validation_movie": fold.validation_movie,
                "test_movie": fold.test_movie,
                "seed": int(seed),
                "anchor_cache": str(anchor_root),
                "worker_out": str(anchor_worker),
                "max_horizon": max(parse_ints(args.horizons)),
                "horizons": parse_ints(args.horizons),
                "max_train_rows": int(args.max_train_rows),
                "max_val_rows": int(args.max_val_rows),
                "max_test_rows": int(args.max_test_rows),
                "posterior_epochs": int(args.anchor_posterior_epochs),
                "student_epochs": int(args.anchor_student_epochs),
                "route_k": int(args.anchor_route_k),
                "device": str(args.device),
                "rebuild": bool(args.force_anchor),
            }
            write_json(anchor_spec, anchor_payload)

            for runner in runners:
                spec = RUNNER_SPECS[runner]
                output_dir = args.out_dir / "folds" / fold_seed / runner
                command = [
                    sys.executable,
                    str(spec["script"]),
                    "--anchor-cache",
                    str(anchor_root),
                    "--out-dir",
                    str(output_dir),
                ]
                if runner == "v97":
                    command.extend(
                        [
                            "--features",
                            str(args.features),
                            "--variants",
                            args.v97_variants,
                            "--evaluation-variant",
                            args.v97_evaluation_variant,
                            "--validation-horizon-weights",
                            args.validation_horizon_weights,
                            "--cumulative-weight",
                            str(args.v97_cumulative_weight),
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ]
                    )
                    command.extend(FROZEN_V97_H1_STRICT_ARGS)
                    if args.v97_skip_recurrent_baselines:
                        command.append("--skip-recurrent-baselines")
                elif runner == "v98":
                    command.extend(
                        [
                            "--models",
                            args.v98_models,
                            "--validation-horizon-weights",
                            args.validation_horizon_weights,
                            "--loss",
                            args.v98_loss,
                            "--cumulative-weight",
                            str(args.v98_cumulative_weight),
                            "--epochs",
                            str(args.v98_epochs),
                            "--patience",
                            str(args.v98_patience),
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ]
                    )
                elif runner == "v99":
                    command.extend(
                        [
                            "--models",
                            args.v99_models,
                            "--input-variants",
                            args.v99_input_variants,
                            "--validation-horizon-weights",
                            args.validation_horizon_weights,
                            "--seeds",
                            str(seed),
                            "--device",
                            args.device,
                        ]
                    )
                extra = args.runner_extra.get(runner, [])
                command.extend([str(token) for token in extra])
                jobs.append(
                    Job(
                        job_id=f"{runner}/{fold_seed}",
                        kind="runner",
                        runner=runner,
                        fold=fold.name,
                        test_movie=fold.test_movie,
                        validation_movie=fold.validation_movie,
                        train_movies=list(fold.train_movies),
                        seed=int(seed),
                        command=command,
                        output_dir=str(output_dir),
                        expected_outputs=[
                            str(output_dir / spec["summary"]),
                            str(output_dir / spec["metadata"]),
                            str(output_dir / spec["contract"]),
                            str(output_dir / spec["audit"]),
                        ],
                        dependencies=[anchor_id],
                    ).finalize()
                )
    return jobs


def write_manifest(args: argparse.Namespace, folds: Sequence[Fold], seeds: Sequence[int], jobs: Sequence[Job]) -> None:
    manifest_path = args.out_dir / "v102_protocol_manifest.json"
    command_manifest_hash = stable_hash([asdict(job) for job in jobs])
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_hash = previous.get("command_manifest_hash")
        if previous_hash and previous_hash != command_manifest_hash and not bool(args.force):
            raise RuntimeError(
                "the output directory contains a different frozen v102 protocol; "
                "choose a new --out-dir or pass --force explicitly"
            )
    feature_columns = (
        pd.read_csv(args.features, nrows=0).columns.astype(str).tolist()
        if args.features.exists()
        else []
    )
    feature_schema_sha256 = hashlib.sha256(
        "\0".join(feature_columns).encode("utf-8")
    ).hexdigest()
    protocol = {
        "protocol_version": "v102-online-lomo-2",
        "created_unix": previous.get("created_unix", time.time()),
        "last_manifest_refresh_unix": time.time(),
        "dataset": args.dataset,
        "outer_unit": "movie",
        "movies": list(MOVIES),
        "folds": [asdict(fold) for fold in folds],
        "seeds": list(seeds),
        "horizons": parse_ints(args.horizons),
        "target": "causal next displacement h1",
        "evaluation": "streaming/receding h1; h2/h4/h6 sum consecutive pre-observation h1 predictions",
        "selection": "validation movie only; test movie never used for model/hyperparameter selection",
        "development_status": "frozen post-development LOMO; architecture-level search was not nested inside these six movies",
        "anchor": "v52 rebuilt per outer fold; training anchors generated movie-level OOF",
        "seed_aggregation": "arithmetic mean within test movie before statistical inference",
        "statistical_unit": "six held-out movies",
        "primary_method": args.primary_method,
        "primary_horizon": int(args.primary_horizon),
        "primary_metric": args.primary_metric,
        "multiple_testing": "Holm step-down over all reported paired comparisons",
        "exact_test": "two-sided exact paired sign-flip/randomization test",
        "bootstrap": f"movie-paired percentile bootstrap, {args.bootstrap_repeats} resamples",
        "component_rmse": "sqrt(mean over rows and x/y components of squared error)",
        "vector_rmse": "sqrt(mean over rows of squared Euclidean vector error)",
        "features": str(args.features.resolve()),
        "feature_column_count": len(feature_columns),
        "feature_schema_sha256": feature_schema_sha256,
        "frozen_v97_h1_strict_args": list(FROZEN_V97_H1_STRICT_ARGS),
        "table_root": str(args.table_root.resolve()),
        "runner_extra": args.runner_extra,
        "command_manifest_hash": command_manifest_hash,
    }
    write_json(manifest_path, protocol)
    rows = []
    for job in jobs:
        row = asdict(job)
        row["command"] = quote_command(job.command)
        row["train_movies"] = ",".join(map(str, job.train_movies))
        row["expected_outputs"] = json.dumps(job.expected_outputs)
        row["dependencies"] = ",".join(job.dependencies)
        row["complete"] = job_complete(job)
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "v102_job_manifest.csv", index=False)


def preflight(args: argparse.Namespace, folds: Sequence[Fold], jobs: Sequence[Job]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "detail": detail})

    check("six_outer_folds", len(folds) == 6, f"folds={len(folds)}")
    check("each_movie_test_once", sorted(f.test_movie for f in folds) == list(MOVIES), str([f.test_movie for f in folds]))
    check("each_movie_validation_once", sorted(f.validation_movie for f in folds) == list(MOVIES), str([f.validation_movie for f in folds]))
    for fold in folds:
        disjoint = not (set(fold.train_movies) & {fold.test_movie, fold.validation_movie})
        complete = set(fold.train_movies) | {fold.test_movie, fold.validation_movie} == set(MOVIES)
        check(f"split_{fold.name}", disjoint and complete and len(fold.train_movies) == 4, str(asdict(fold)))
    check("features_exist", args.features.exists(), str(args.features))
    if args.features.exists():
        feature_columns = pd.read_csv(args.features, nrows=0).columns.astype(str).tolist()
        suspicious = [
            column
            for column in feature_columns
            if re.search(r"(^|_)(target|future|lead|step[1-9])($|_)", column.lower())
        ]
        check(
            "feature_schema_has_no_obvious_future_columns",
            not suspicious,
            "none" if not suspicious else ",".join(suspicious[:30]),
        )
        required_keys = {"dataset", "sequence", "frame", "track_id", "x_px", "y_px"}
        prefixes = ("ms_", "tf_", "rc_")
        raw_context_schema = required_keys.issubset(feature_columns) and all(
            any(column.startswith(prefix) for column in feature_columns)
            for prefix in prefixes
        )
        check(
            "raw_context_v2_feature_contract",
            raw_context_schema,
            f"columns={len(feature_columns)}; required_keys={sorted(required_keys)}; prefixes={prefixes}",
        )
    check("table_root_exists", args.table_root.exists(), str(args.table_root))
    row_limited = any(int(value) > 0 for value in (args.max_train_rows, args.max_val_rows, args.max_test_rows))
    check(
        "row_limits_are_diagnostic_only",
        not row_limited or bool(args.allow_incomplete_analysis),
        "random row limits can destroy chronological h2/h4/h6 chains; publication runs must use 0/0/0",
    )
    for movie in MOVIES:
        table = args.table_root / args.dataset / f"{args.dataset}_{movie:02d}_tracks.csv"
        check(f"movie_{movie:02d}_table", table.exists(), str(table))
    for runner in csv_tokens(args.runners):
        path = Path(RUNNER_SPECS[runner]["script"])
        check(f"runner_{runner}", path.exists(), str(path))
    if "v97" in csv_tokens(args.runners):
        frozen_flags = set(FROZEN_V97_H1_STRICT_ARGS[::2]) | {
            "--features",
            "--variants",
            "--evaluation-variant",
            "--validation-horizon-weights",
            "--cumulative-weight",
            "--skip-recurrent-baselines",
        }
        v97_extra = args.runner_extra.get("v97", [])
        normalized_extra_flags = {
            token.split("=", maxsplit=1)[0]
            for token in v97_extra
            if token.startswith("--")
        }
        extra_overrides = sorted(normalized_extra_flags & frozen_flags)
        allowed_extra = ["--smoke"] if bool(args.allow_incomplete_analysis) else []
        unexpected_extra = [] if v97_extra == allowed_extra else list(v97_extra)
        frozen_v97 = (
            args.v97_variants == "v97_direct"
            and args.v97_evaluation_variant == "v97_direct"
            and args.validation_horizon_weights == "1:0.90,2:0.05,4:0.03,6:0.02"
            and math.isclose(float(args.v97_cumulative_weight), 0.05, rel_tol=0.0, abs_tol=1e-12)
            and bool(args.v97_skip_recurrent_baselines)
        )
        check(
            "frozen_h1_strict_v97_contract",
            frozen_v97 and not extra_overrides and not unexpected_extra,
            str(
                {
                    "features": str(args.features),
                    "variants": args.v97_variants,
                    "evaluation_variant": args.v97_evaluation_variant,
                    "validation_horizon_weights": args.validation_horizon_weights,
                    "cumulative_weight": args.v97_cumulative_weight,
                    "skip_recurrent_baselines": args.v97_skip_recurrent_baselines,
                    "runner_extra_frozen_overrides": extra_overrides,
                    "runner_extra_unexpected": unexpected_extra,
                }
            ),
        )
    exact_two_sided, exact_permutations = exact_sign_flip_pvalue(np.ones(len(MOVIES), dtype=np.float64))
    exact_one_sided, _ = exact_sign_flip_pvalue(np.ones(len(MOVIES), dtype=np.float64), alternative="greater")
    check(
        "exact_sign_flip_six_movie_sentinel",
        math.isclose(exact_two_sided, 2.0 / 64.0) and math.isclose(exact_one_sided, 1.0 / 64.0) and exact_permutations == 64,
        f"two_sided={exact_two_sided}, one_sided={exact_one_sided}, permutations={exact_permutations}",
    )
    bootstrap_a = paired_bootstrap_ci(np.arange(1.0, 7.0), 200, 0.95, 102)
    bootstrap_b = paired_bootstrap_ci(np.arange(1.0, 7.0), 200, 0.95, 102)
    check("paired_bootstrap_deterministic_sentinel", bootstrap_a == bootstrap_b, str(bootstrap_a))
    check("v84_anchor_builder", (SCRIPTS / "run_lachance_joint_innovation_field_v84.py").exists(), "exact fold-local v52/OOF anchor")
    check("nonempty_jobs", bool(jobs), f"jobs={len(jobs)}")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "v102_integrity_preflight.csv", index=False)
    if not bool(frame["passed"].all()):
        failed = frame.loc[~frame.passed, ["check", "detail"]].to_dict("records")
        raise RuntimeError(f"preflight failed: {failed}")
    return frame


def load_anchor_worker_args(spec_path: Path) -> tuple[argparse.Namespace, Any, dict[str, Any]]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(SCRIPTS))
    import run_lachance_joint_innovation_field_v84 as v84  # type: ignore

    parser_argv = [
        str(v84.__file__),
        "--dataset",
        payload["dataset"],
        "--table-root",
        payload["table_root"],
        "--features",
        payload["features"],
        "--train-seq",
        ",".join(map(str, payload["train_movies"])),
        "--val-seq",
        str(payload["validation_movie"]),
        "--test-seq",
        str(payload["test_movie"]),
        "--seed",
        str(payload["seed"]),
        "--max-horizon",
        str(payload["max_horizon"]),
        "--horizons",
        ",".join(map(str, payload["horizons"])),
        "--anchor-cache",
        payload["anchor_cache"],
        "--out-dir",
        str(Path(payload["worker_out"]) / "v84_anchor_only"),
        "--anchor-max-train-rows",
        str(payload["max_train_rows"]),
        "--anchor-max-val-rows",
        str(payload["max_val_rows"]),
        "--anchor-max-test-rows",
        str(payload["max_test_rows"]),
        "--posterior-epochs",
        str(payload["posterior_epochs"]),
        "--student-epochs",
        str(payload["student_epochs"]),
        "--v12-route-k",
        str(payload["route_k"]),
        "--device",
        str(payload.get("device", "auto")),
        "--oof-fold-limit",
        "0",
    ]
    if payload.get("rebuild"):
        parser_argv.append("--rebuild-anchors")
    original = sys.argv
    try:
        sys.argv = parser_argv
        namespace = v84.parse_args()
    finally:
        sys.argv = original
    return namespace, v84, payload


def anchor_worker(spec_path: Path) -> None:
    started = time.time()
    args, v84, payload = load_anchor_worker_args(spec_path)
    worker_out = Path(payload["worker_out"])
    worker_out.mkdir(parents=True, exist_ok=True)
    try:
        args.horizons = v84.parse_ints(args.horizons)
        train, validation, test, quality = v84.build_anchor_sets(args)
        expected_train = set(int(movie) for movie in payload["train_movies"])
        actual_train = set(int(movie) for movie in train.rows.sequence.unique())
        actual_validation = set(int(movie) for movie in validation.rows.sequence.unique())
        actual_test = set(int(movie) for movie in test.rows.sequence.unique())
        if actual_train != expected_train:
            raise RuntimeError(f"OOF train movies mismatch: expected={expected_train}, actual={actual_train}")
        if actual_validation != {int(payload["validation_movie"])}:
            raise RuntimeError(f"validation movie mismatch: {actual_validation}")
        if actual_test != {int(payload["test_movie"])}:
            raise RuntimeError(f"test movie mismatch: {actual_test}")
        anchor_root = Path(payload["anchor_cache"])
        final_dirs = [path for path in anchor_root.glob("final_*") if (path / "test/arrays.npz").exists()]
        oof_dirs = [path for path in anchor_root.glob("oof_seq*_*" ) if (path / "test/arrays.npz").exists()]
        if len(final_dirs) != 1 or len(oof_dirs) != len(expected_train):
            raise RuntimeError(f"incomplete exact anchor cache: final={final_dirs}, oof={oof_dirs}")
        split_audit = {
            "train": split_row_audit(train.rows),
            "validation": split_row_audit(validation.rows),
            "test": split_row_audit(test.rows),
        }
        if any(int(item["duplicate_keys"]) != 0 for item in split_audit.values()):
            raise RuntimeError(f"duplicate anchor prediction keys: {split_audit}")
        for bundle_name, bundle in (("train", train), ("validation", validation), ("test", test)):
            arrays = (bundle.anchor_residual, bundle.base, bundle.target_steps)
            if any(len(array) != len(bundle.rows) or not bool(np.isfinite(array).all()) for array in arrays):
                raise RuntimeError(f"non-finite or row-misaligned anchor arrays in {bundle_name}")

        final_contract = json.loads((final_dirs[0] / "contract.json").read_text(encoding="utf-8"))
        expected_validation = {int(payload["validation_movie"])}
        expected_test = {int(payload["test_movie"])}
        if set(map(int, final_contract["train_seq"])) != expected_train:
            raise RuntimeError(f"final anchor train contract leaked/missed movies: {final_contract}")
        if set(map(int, final_contract["val_seq"])) != expected_validation:
            raise RuntimeError(f"final anchor validation contract mismatch: {final_contract}")
        if set(map(int, final_contract["test_seq"])) != expected_test:
            raise RuntimeError(f"final anchor test contract mismatch: {final_contract}")

        oof_contract_audit: list[dict[str, Any]] = []
        heldout_seen: set[int] = set()
        for directory in sorted(oof_dirs):
            contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
            heldout_set = set(map(int, contract["test_seq"]))
            fold_train = set(map(int, contract["train_seq"]))
            fold_validation = set(map(int, contract["val_seq"]))
            if len(heldout_set) != 1:
                raise RuntimeError(f"OOF anchor must hold out exactly one movie: {contract}")
            heldout = next(iter(heldout_set))
            if heldout not in expected_train or heldout in heldout_seen:
                raise RuntimeError(f"invalid or repeated OOF held-out movie {heldout}: {contract}")
            heldout_seen.add(heldout)
            if fold_train & fold_validation or fold_train & heldout_set or fold_validation & heldout_set:
                raise RuntimeError(f"OOF train/validation/test overlap: {contract}")
            if fold_train | fold_validation | heldout_set != expected_train:
                raise RuntimeError(f"OOF contract does not partition outer training movies: {contract}")
            oof_rows = pd.read_csv(directory / "test/rows.csv")
            row_audit = split_row_audit(oof_rows)
            if set(row_audit["movies"]) != heldout_set or int(row_audit["duplicate_keys"]) != 0:
                raise RuntimeError(f"OOF output identity mismatch for movie {heldout}: {row_audit}")
            oof_contract_audit.append(
                {
                    "heldout_movie": heldout,
                    "train_movies": sorted(fold_train),
                    "validation_movies": sorted(fold_validation),
                    **row_audit,
                }
            )
        if heldout_seen != expected_train:
            raise RuntimeError(f"OOF held-out coverage mismatch: expected={expected_train}, actual={heldout_seen}")
        status = {
            "ok": True,
            "job_id": payload["job_id"],
            "config_hash": payload["job_config_hash"],
            "train_movies": sorted(actual_train),
            "validation_movies": sorted(actual_validation),
            "test_movies": sorted(actual_test),
            "train_rows": len(train.rows),
            "validation_rows": len(validation.rows),
            "test_rows": len(test.rows),
            "oof_anchor_folds": len(oof_dirs),
            "final_cache": str(final_dirs[0]),
            "split_audit": split_audit,
            "oof_contract_audit": oof_contract_audit,
            "quality_rows": len(quality),
            "elapsed_sec": time.time() - started,
        }
        write_json(worker_out / "anchor_worker_status.json", status)
        write_json(anchor_root / "v102_anchor_ready.json", status)
        write_json(worker_out / ".v102_done.json", status)
    except Exception as exc:
        write_json(
            worker_out / "anchor_worker_error.json",
            {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_sec": time.time() - started},
        )
        raise


def run_job(job: Job, force: bool = False) -> None:
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if job_complete(job) and not force:
        print(f"[v102] cached {job.job_id}", flush=True)
        return
    started = time.time()
    log_path = output_dir / "v102_subprocess.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {quote_command(job.command)}\n")
        log.flush()
        process = subprocess.run(job.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"job {job.job_id} failed with exit code {process.returncode}; see {log_path}")
    missing = [path for path in job.expected_outputs if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"job {job.job_id} completed but outputs are missing: {missing}")
    write_json(
        job_marker(job),
        {
            "ok": True,
            "job_id": job.job_id,
            "config_hash": job.config_hash,
            "command": job.command,
            "expected_outputs": job.expected_outputs,
            "elapsed_sec": time.time() - started,
        },
    )


def execute_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    selected = set(csv_tokens(args.only_runners)) if args.only_runners else set()
    by_id = {job.job_id: job for job in jobs}
    for job in jobs:
        if selected and job.kind == "runner" and job.runner not in selected:
            continue
        for dependency_id in job.dependencies:
            dependency = by_id[dependency_id]
            if not job_complete(dependency):
                run_job(dependency, force=bool(args.force))
        run_job(job, force=bool(args.force))


def canonical_method(runner: str, row: pd.Series) -> str:
    method = str(row.get("method", "unknown"))
    method = re.sub(r"__seed\d+$", "", method)
    if method == "constant_velocity":
        return "baseline/constant_velocity"
    if runner == "v97":
        family = str(row.get("family", ""))
        if family == "baseline":
            return f"baseline/{method}"
        return f"v97/{method}"
    if runner == "v98":
        return f"kalmannet/{method}"
    if runner == "v99":
        model = str(row.get("model", method.split("__", maxsplit=1)[0]))
        input_variant = str(row.get("input_variant", "unknown"))
        return f"v99/{model}/{input_variant}"
    return f"{runner}/{method}"


def parse_movie_set(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return set(parse_ints(str(value)))


def read_contract_splits(runner: str, output_dir: Path) -> dict[str, set[int]]:
    path = output_dir / RUNNER_SPECS[runner]["contract"]
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    if runner in {"v97", "v98"} and {"train_movies", "val_movies", "test_movies"}.issubset(frame.columns):
        return {
            "train": parse_movie_set(frame.iloc[0]["train_movies"]),
            "validation": parse_movie_set(frame.iloc[0]["val_movies"]),
            "test": parse_movie_set(frame.iloc[0]["test_movies"]),
        }
    if runner == "v99" and {"split", "sequences"}.issubset(frame.columns):
        result: dict[str, set[int]] = {}
        aliases = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
        for _, row in frame.iterrows():
            key = aliases.get(str(row["split"]).strip().lower())
            if key is None:
                continue
            if key in result:
                raise RuntimeError(f"duplicate {key} rows in {path}")
            result[key] = parse_movie_set(row["sequences"])
        return result
    return {}


def causal_audit_result(runner: str, output_dir: Path) -> tuple[bool, str]:
    path = output_dir / RUNNER_SPECS[runner]["audit"]
    if not path.exists():
        return False, f"missing {path}"
    frame = pd.read_csv(path)
    if frame.empty:
        return False, f"empty {path}"
    details: dict[str, Any] = {}
    zero_tokens = ("violation", "duplicate", "future_target")
    zero_columns = [column for column in frame.columns if any(token in column.lower() for token in zero_tokens)]
    if not zero_columns:
        return False, f"no causal/duplicate sentinel columns in {path}"
    passed = True
    for column in zero_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite_values = bool(np.isfinite(values.to_numpy(dtype=float)).all())
        total = float(values.sum()) if finite_values else float("nan")
        details[column] = total
        details[f"{column}_finite"] = finite_values
        passed = passed and finite_values and total == 0.0
    if "missing_required_columns" in frame.columns:
        missing = [
            str(value)
            for value in frame["missing_required_columns"]
            if pd.notna(value) and str(value).strip() not in {"", "[]", "none", "None"}
        ]
        details["missing_required_columns"] = missing
        passed = passed and not missing
    if "predict_before_observe" in frame.columns:
        values = frame["predict_before_observe"].astype(str).str.lower().isin({"true", "1", "yes"})
        details["predict_before_observe_all"] = bool(values.all())
        passed = passed and bool(values.all())
    if {"rows", "predicted_rows"}.issubset(frame.columns):
        rows = pd.to_numeric(frame["rows"], errors="coerce")
        predicted = pd.to_numeric(frame["predicted_rows"], errors="coerce")
        matched = bool((rows == predicted).all())
        details["all_rows_predicted"] = matched
        passed = passed and matched
    if "coverage" in frame.columns:
        coverage = pd.to_numeric(frame["coverage"], errors="coerce")
        complete = bool(np.isfinite(coverage).all() and (coverage >= 1.0 - 1e-12).all())
        details["complete_coverage"] = complete
        passed = passed and complete
    return bool(passed), json.dumps(finite(details), sort_keys=True)


def collect_metrics(args: argparse.Namespace, jobs: Sequence[Job]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[pd.DataFrame] = []
    integrity: list[dict[str, Any]] = []
    jobs_by_id = {job.job_id: job for job in jobs}
    anchor_test_hashes: dict[int, set[str]] = {}
    for anchor_job in (job for job in jobs if job.kind == "anchor"):
        status_path = Path(anchor_job.output_dir) / "anchor_worker_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        split_audit = status.get("split_audit", {})
        duplicate_count = sum(int(item.get("duplicate_keys", -1)) for item in split_audit.values()) if split_audit else -1
        test_hash = str(split_audit.get("test", {}).get("key_sha256", ""))
        if test_hash:
            anchor_test_hashes.setdefault(int(anchor_job.test_movie), set()).add(test_hash)
        exact_split = (
            set(map(int, status.get("train_movies", []))) == set(anchor_job.train_movies)
            and set(map(int, status.get("validation_movies", []))) == {int(anchor_job.validation_movie)}
            and set(map(int, status.get("test_movies", []))) == {int(anchor_job.test_movie)}
        )
        integrity.extend(
            [
                {"job_id": anchor_job.job_id, "check": "anchor_exact_movie_partition", "passed": exact_split, "detail": str({key: status.get(key) for key in ("train_movies", "validation_movies", "test_movies")})},
                {"job_id": anchor_job.job_id, "check": "anchor_unique_prediction_keys", "passed": duplicate_count == 0, "detail": str(duplicate_count)},
                {"job_id": anchor_job.job_id, "check": "anchor_complete_oof_coverage", "passed": int(status.get("oof_anchor_folds", -1)) == len(anchor_job.train_movies), "detail": str(status.get("oof_contract_audit", []))},
            ]
        )
    for movie, hashes in sorted(anchor_test_hashes.items()):
        integrity.append(
            {
                "job_id": f"test_movie_{movie:02d}",
                "check": "identical_test_prediction_keys_across_seeds",
                "passed": len(hashes) == 1,
                "detail": str(sorted(hashes)),
            }
        )
    for job in jobs:
        if job.kind != "runner":
            continue
        output_dir = Path(job.output_dir)
        summary = output_dir / RUNNER_SPECS[job.runner]["summary"]
        if not summary.exists():
            integrity.append({"job_id": job.job_id, "check": "summary_exists", "passed": False, "detail": str(summary)})
            continue
        frame = pd.read_csv(summary)
        required = {"method", "horizon", "component_rmse", "vector_rmse", "n_rows"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{summary} missing required columns {sorted(missing)}")
        frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
        frame["component_rmse"] = pd.to_numeric(frame["component_rmse"], errors="coerce")
        frame["vector_rmse"] = pd.to_numeric(frame["vector_rmse"], errors="coerce")
        frame["n_rows"] = pd.to_numeric(frame["n_rows"], errors="coerce")
        duplicate_summary = int(frame.duplicated(["method", "horizon"]).sum())
        finite_metrics = bool(
            np.isfinite(frame[["horizon", "component_rmse", "vector_rmse", "n_rows"]].to_numpy(float)).all()
            and (frame["component_rmse"] >= 0).all()
            and (frame["vector_rmse"] >= 0).all()
            and (frame["n_rows"] > 0).all()
        )
        rmse_identity = bool(
            np.allclose(
                frame["vector_rmse"].to_numpy(float),
                math.sqrt(2.0) * frame["component_rmse"].to_numpy(float),
                rtol=2e-5,
                atol=2e-5,
            )
        )
        horizon_complete = all(
            set(group["horizon"].astype(int)) == set(parse_ints(args.horizons))
            for _, group in frame.groupby("method", sort=False)
        )
        if "contract" in frame.columns:
            contracts = set(frame.contract.dropna().astype(str))
            valid_contract = contracts == {"streaming_receding_h1"}
        else:
            contracts = set()
            valid_contract = False
        actual_splits = read_contract_splits(job.runner, output_dir)
        expected_splits = {
            "train": set(job.train_movies),
            "validation": {int(job.validation_movie)},
            "test": {int(job.test_movie)},
        }
        causal_passed, causal_detail = causal_audit_result(job.runner, output_dir)
        dependency_status_ok = False
        dependency_detail = "missing anchor dependency"
        if len(job.dependencies) == 1 and job.dependencies[0] in jobs_by_id:
            dependency = jobs_by_id[job.dependencies[0]]
            status_path = Path(dependency.output_dir) / "anchor_worker_status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                dependency_status_ok = bool(
                    status.get("ok")
                    and set(map(int, status.get("train_movies", []))) == expected_splits["train"]
                    and set(map(int, status.get("validation_movies", []))) == expected_splits["validation"]
                    and set(map(int, status.get("test_movies", []))) == expected_splits["test"]
                    and int(status.get("oof_anchor_folds", -1)) == len(job.train_movies)
                )
                dependency_detail = str({key: status.get(key) for key in ("train_movies", "validation_movies", "test_movies", "oof_anchor_folds")})
        integrity.extend(
            [
                {"job_id": job.job_id, "check": "streaming_contract", "passed": valid_contract, "detail": str(sorted(contracts))},
                {"job_id": job.job_id, "check": "exact_train_validation_test_movies", "passed": actual_splits == expected_splits, "detail": str({key: sorted(value) for key, value in actual_splits.items()})},
                {"job_id": job.job_id, "check": "fold_local_oof_anchor_dependency", "passed": dependency_status_ok, "detail": dependency_detail},
                {"job_id": job.job_id, "check": "causal_and_duplicate_audit", "passed": causal_passed, "detail": causal_detail},
                {"job_id": job.job_id, "check": "unique_summary_method_horizon", "passed": duplicate_summary == 0, "detail": str(duplicate_summary)},
                {"job_id": job.job_id, "check": "finite_nonnegative_metrics", "passed": finite_metrics, "detail": f"rows={len(frame)}"},
                {"job_id": job.job_id, "check": "component_vector_rmse_identity", "passed": rmse_identity, "detail": "vector_rmse=sqrt(2)*component_rmse"},
                {"job_id": job.job_id, "check": "complete_horizon_rows_per_method", "passed": horizon_complete, "detail": str(sorted(frame.horizon.dropna().astype(int).unique()))},
            ]
        )
        frame = frame[frame.horizon.astype(int).isin(parse_ints(args.horizons))].copy()
        frame["runner"] = job.runner
        frame["job_id"] = job.job_id
        frame["fold"] = job.fold
        frame["test_movie"] = job.test_movie
        frame["validation_movie"] = job.validation_movie
        frame["train_movies"] = ",".join(map(str, job.train_movies))
        frame["seed"] = job.seed
        frame["method_id"] = frame.apply(lambda row: canonical_method(job.runner, row), axis=1)
        metrics.append(frame)
    integrity_frame = pd.DataFrame(integrity)
    if not metrics:
        return pd.DataFrame(), integrity_frame
    raw = pd.concat(metrics, ignore_index=True, sort=False)
    numeric = ["component_rmse", "vector_rmse", "r2", "cosine", "magnitude_ratio", "n_rows"]
    for column in numeric:
        if column in raw:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    # v97 and v98 both emit constant velocity.  Keep one canonical copy only if
    # the values agree on the same fold/seed; disagreement signals row mismatch.
    key = ["test_movie", "seed", "method_id", "horizon"]
    duplicates = raw.groupby(key, dropna=False).size()
    for duplicate_key in duplicates[duplicates > 1].index:
        group = raw.set_index(key).loc[duplicate_key]
        if isinstance(group, pd.Series):
            continue
        for metric in (*METRICS, "n_rows"):
            tolerance = 0.0 if metric == "n_rows" else float(args.duplicate_tolerance)
            if float(group[metric].max() - group[metric].min()) > tolerance:
                raise RuntimeError(f"canonical duplicate disagreement for {duplicate_key}/{metric}: {group[metric].tolist()}")
    priority = {"v97": 0, "v98": 1, "v99": 2}
    raw["source_priority"] = raw.runner.map(priority).fillna(99)
    raw = raw.sort_values(key + ["source_priority"]).drop_duplicates(key, keep="first").reset_index(drop=True)
    for (test_movie, seed, horizon), group in raw.groupby(["test_movie", "seed", "horizon"], sort=True):
        counts = sorted(set(int(value) for value in group.n_rows.dropna()))
        integrity.append(
            {
                "job_id": f"fold_test{int(test_movie):02d}_seed{int(seed)}",
                "check": f"matched_evaluation_rows_h{int(horizon)}",
                "passed": len(counts) == 1,
                "detail": str(counts),
            }
        )
    integrity_frame = pd.DataFrame(integrity)
    return raw, integrity_frame


def collect_compute(args: argparse.Namespace, jobs: Sequence[Job], raw_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.kind != "runner":
            continue
        output_dir = Path(job.output_dir)
        metadata_path = output_dir / RUNNER_SPECS[job.runner]["metadata"]
        marker_path = job_marker(job)
        marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.exists() else {}
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
        else:
            metadata = pd.DataFrame([{}])
        for _, item in metadata.iterrows():
            if job.runner == "v97":
                method_id = f"v97/{item.get('variant', 'unknown')}"
            elif job.runner == "v98":
                method_id = f"kalmannet/kalmannet_{item.get('state_model', 'unknown')}"
            else:
                method_id = f"v99/{item.get('model', 'unknown')}/{item.get('input_variant', 'unknown')}"
            parameter_count = pd.to_numeric(pd.Series([item.get("parameters", np.nan)]), errors="coerce").iloc[0]
            train_time = pd.to_numeric(pd.Series([item.get("train_elapsed_sec", np.nan)]), errors="coerce").iloc[0]
            inference_ms = pd.to_numeric(pd.Series([item.get("test_ms_per_cell", np.nan)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "runner": job.runner,
                    "method_id": method_id,
                    "test_movie": job.test_movie,
                    "seed": job.seed,
                    "trainable_parameters": parameter_count,
                    "frozen_anchor_parameters": np.nan,
                    "total_parameters": np.nan,
                    "model_train_elapsed_sec": train_time,
                    "whole_job_elapsed_sec": marker.get("elapsed_sec", np.nan),
                    "inference_ms_per_cell": inference_ms,
                    "peak_memory_mb": np.nan,
                    "flops_per_cell": np.nan,
                    "accounting_complete": bool(pd.notna(parameter_count) and pd.notna(train_time) and pd.notna(inference_ms)),
                    "accounting_note": "Frozen v52 route-basis/calibrator parameters, peak memory and FLOPs remain explicit placeholders until instrumented.",
                }
            )
    existing = {
        (str(row["method_id"]), int(row["test_movie"]), int(row["seed"]))
        for row in rows
    }
    if not raw_metrics.empty:
        for (method_id, test_movie, seed), group in raw_metrics.groupby(
            ["method_id", "test_movie", "seed"], sort=True
        ):
            key = (str(method_id), int(test_movie), int(seed))
            if key in existing:
                continue
            rows.append(
                {
                    "runner": str(group.iloc[0].runner),
                    "method_id": str(method_id),
                    "test_movie": int(test_movie),
                    "seed": int(seed),
                    "trainable_parameters": np.nan,
                    "frozen_anchor_parameters": np.nan,
                    "total_parameters": np.nan,
                    "model_train_elapsed_sec": np.nan,
                    "whole_job_elapsed_sec": np.nan,
                    "inference_ms_per_cell": np.nan,
                    "peak_memory_mb": np.nan,
                    "flops_per_cell": np.nan,
                    "accounting_complete": False,
                    "accounting_note": "Comparator emitted metrics but no normalized compute metadata; values must be instrumented before publication.",
                }
            )
    for job in jobs:
        if job.kind != "anchor":
            continue
        marker = job_marker(job)
        payload = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        rows.append(
            {
                "runner": job.runner,
                "method_id": "shared/frozen_v52_anchor_builder",
                "test_movie": job.test_movie,
                "seed": job.seed,
                "trainable_parameters": np.nan,
                "frozen_anchor_parameters": np.nan,
                "total_parameters": np.nan,
                "model_train_elapsed_sec": payload.get("elapsed_sec", np.nan),
                "whole_job_elapsed_sec": payload.get("elapsed_sec", np.nan),
                "inference_ms_per_cell": np.nan,
                "peak_memory_mb": np.nan,
                "flops_per_cell": np.nan,
                "accounting_complete": False,
                "accounting_note": "Shared fold-local v52 basis/OOF anchor cost; parameter and inference accounting are explicit publication placeholders.",
            }
        )
    return pd.DataFrame(rows)


def aggregate_seeds(raw: pd.DataFrame, expected_seeds: Sequence[int], allow_incomplete: bool) -> pd.DataFrame:
    keys = ["test_movie", "validation_movie", "train_movies", "method_id", "horizon"]
    values = [column for column in ("component_rmse", "vector_rmse", "r2", "cosine", "magnitude_ratio", "n_rows") if column in raw]
    rows: list[dict[str, Any]] = []
    for group_key, group in raw.groupby(keys, dropna=False, sort=True):
        seeds = sorted(int(seed) for seed in group.seed.unique())
        if not allow_incomplete and seeds != sorted(expected_seeds):
            continue
        row = dict(zip(keys, group_key))
        row.update({"seeds": ",".join(map(str, seeds)), "n_seeds": len(seeds)})
        for column in values:
            if column == "n_rows":
                counts = sorted(set(int(value) for value in group[column]))
                if len(counts) != 1:
                    raise RuntimeError(f"evaluation-row mismatch across seeds for {group_key}: {counts}")
                row[column] = int(counts[0])
                row[f"{column}_seed_sd"] = 0.0 if len(group) > 1 else np.nan
            else:
                row[column] = float(group[column].mean())
                row[f"{column}_seed_sd"] = float(group[column].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def validate_publication_coverage(
    seed_aggregated: pd.DataFrame,
    expected_seeds: Sequence[int],
    primary_method: str,
) -> None:
    if seed_aggregated.empty:
        raise RuntimeError("seed-aggregated publication table is empty")
    if primary_method not in set(seed_aggregated.method_id):
        raise RuntimeError(f"primary method {primary_method!r} is absent from complete seed aggregates")
    duplicated = seed_aggregated.duplicated(["method_id", "test_movie", "horizon"])
    if bool(duplicated.any()):
        rows = seed_aggregated.loc[duplicated, ["method_id", "test_movie", "horizon"]].to_dict("records")
        raise RuntimeError(f"duplicate movie-level aggregates: {rows[:10]}")
    expected_movies = set(MOVIES)
    expected_horizons = set(HORIZONS)
    expected_seed_count = len(expected_seeds)
    failures: list[str] = []
    for method, group in seed_aggregated.groupby("method_id", sort=True):
        movies = set(int(value) for value in group.test_movie.unique())
        horizons = set(int(value) for value in group.horizon.unique())
        per_movie_horizons = group.groupby("test_movie").horizon.apply(lambda values: set(map(int, values)))
        seed_counts = set(int(value) for value in group.n_seeds.unique())
        if movies != expected_movies or horizons != expected_horizons:
            failures.append(f"{method}: movies={sorted(movies)}, horizons={sorted(horizons)}")
        if any(value != expected_horizons for value in per_movie_horizons):
            failures.append(f"{method}: incomplete horizon rows by movie")
        if seed_counts != {expected_seed_count}:
            failures.append(f"{method}: n_seeds={sorted(seed_counts)}")
    if failures:
        raise RuntimeError("incomplete publication movie/method coverage: " + "; ".join(failures[:20]))


def exact_sign_flip_pvalue(differences: np.ndarray, alternative: str = "two-sided") -> tuple[float, int]:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return float("nan"), 0
    if not bool(np.isfinite(values).all()):
        raise ValueError("exact paired sign-flip received non-finite movie differences")
    observed = float(values.mean())
    null = np.asarray(
        [np.mean(values * np.asarray(signs, dtype=np.float64)) for signs in itertools.product((-1.0, 1.0), repeat=len(values))],
        dtype=np.float64,
    )
    tolerance = 1e-12
    if alternative == "greater":
        count = int(np.sum(null >= observed - tolerance))
    elif alternative == "less":
        count = int(np.sum(null <= observed + tolerance))
    else:
        count = int(np.sum(np.abs(null) >= abs(observed) - tolerance))
    return float(count / len(null)), int(len(null))


def paired_bootstrap_ci(differences: np.ndarray, repeats: int, confidence: float, seed: int) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    if not bool(np.isfinite(values).all()):
        raise ValueError("paired movie bootstrap received non-finite differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    probability_positive = float(np.mean(means > 0.0))
    return float(low), float(high), probability_positive


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(len(values), np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_inference(args: argparse.Namespace, seed_aggregated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seed_aggregated.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (method, horizon), group in seed_aggregated.groupby(["method_id", "horizon"], sort=True):
        row: dict[str, Any] = {"method_id": method, "horizon": int(horizon), "n_movies": int(group.test_movie.nunique())}
        for metric in ("component_rmse", "vector_rmse", "r2", "cosine", "magnitude_ratio"):
            if metric in group:
                row[f"{metric}_movie_mean"] = float(group[metric].mean())
                row[f"{metric}_movie_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        summary_rows.append(row)
    movie_summary = pd.DataFrame(summary_rows)

    inference_rows: list[dict[str, Any]] = []
    primary = seed_aggregated[seed_aggregated.method_id.eq(args.primary_method)]
    comparators = sorted(set(seed_aggregated.method_id) - {args.primary_method})
    for comparator in comparators:
        comparison = seed_aggregated[seed_aggregated.method_id.eq(comparator)]
        for horizon in parse_ints(args.horizons):
            for metric in METRICS:
                left = primary[primary.horizon.eq(horizon)][["test_movie", metric]].rename(columns={metric: "primary"})
                right = comparison[comparison.horizon.eq(horizon)][["test_movie", metric]].rename(columns={metric: "comparator"})
                paired = left.merge(right, on="test_movie", how="inner", validate="one_to_one").sort_values("test_movie")
                if paired.empty:
                    continue
                # Positive delta means the primary method has lower error.
                differences = paired.comparator.to_numpy(float) - paired.primary.to_numpy(float)
                pvalue, permutations = exact_sign_flip_pvalue(differences, alternative="two-sided")
                one_sided, _ = exact_sign_flip_pvalue(differences, alternative="greater")
                ci_low, ci_high, bootstrap_positive = paired_bootstrap_ci(
                    differences,
                    int(args.bootstrap_repeats),
                    float(args.confidence),
                    int(args.bootstrap_seed) + horizon * 1009 + int(stable_hash(comparator, 8), 16) % 1_000_000,
                )
                primary_mean = float(paired.primary.mean())
                comparator_mean = float(paired.comparator.mean())
                inference_rows.append(
                    {
                        "primary_method": args.primary_method,
                        "comparator_method": comparator,
                        "horizon": horizon,
                        "metric": metric,
                        "n_movies": len(paired),
                        "movies": ",".join(map(str, paired.test_movie.tolist())),
                        "primary_movie_mean": primary_mean,
                        "comparator_movie_mean": comparator_mean,
                        "mean_delta_comparator_minus_primary": float(differences.mean()),
                        "median_delta": float(np.median(differences)),
                        "relative_improvement_pct": 100.0 * float(differences.mean()) / max(comparator_mean, EPS),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "bootstrap_probability_delta_positive": bootstrap_positive,
                        "exact_sign_flip_p_two_sided": pvalue,
                        "exact_sign_flip_p_primary_better": one_sided,
                        "exact_permutations": permutations,
                        "all_movies_primary_better": bool(np.all(differences > 0)),
                        "primary_endpoint": bool(horizon == int(args.primary_horizon) and metric == args.primary_metric),
                    }
                )
    inference = pd.DataFrame(inference_rows)
    if not inference.empty:
        inference["holm_p_global"] = holm_adjust(inference.exact_sign_flip_p_two_sided.to_numpy(float))
        inference["holm_reject_0_05"] = inference.holm_p_global < 0.05
        inference["holm_p_metric_horizon_family"] = np.nan
        for (_metric, _horizon), indices in inference.groupby(["metric", "horizon"]).groups.items():
            index = np.asarray(list(indices), dtype=np.int64)
            inference.loc[index, "holm_p_metric_horizon_family"] = holm_adjust(
                inference.loc[index, "exact_sign_flip_p_two_sided"].to_numpy(float)
            )
        inference["holm_p_primary_endpoint_family"] = np.nan
        primary_indices = np.flatnonzero(inference.primary_endpoint.to_numpy(bool))
        if len(primary_indices):
            inference.loc[primary_indices, "holm_p_primary_endpoint_family"] = holm_adjust(
                inference.loc[primary_indices, "exact_sign_flip_p_two_sided"].to_numpy(float)
            )
    return movie_summary, inference


def write_analysis_report(
    args: argparse.Namespace,
    raw: pd.DataFrame,
    seed_aggregated: pd.DataFrame,
    movie_summary: pd.DataFrame,
    inference: pd.DataFrame,
    integrity: pd.DataFrame,
    compute: pd.DataFrame,
) -> None:
    completed_movies = sorted(int(movie) for movie in seed_aggregated.test_movie.unique()) if not seed_aggregated.empty else []
    primary_table = movie_summary[
        movie_summary.horizon.eq(int(args.primary_horizon))
    ].sort_values(f"{args.primary_metric}_movie_mean") if not movie_summary.empty else pd.DataFrame()
    primary_tests = inference[
        inference.primary_endpoint
    ].sort_values("holm_p_global") if not inference.empty else pd.DataFrame()
    lines = [
        "# v102 Online h1 Outer Leave-One-Movie-Out Benchmark",
        "",
        "## Protocol",
        "",
        "- Outer statistical unit: one held-out MDCK_Bulk movie.",
        "- Six folds: every movie 1..6 is test once; validation is a deterministic distinct rotation; the other four movies are train.",
        "- The v52 anchor is rebuilt inside each fold and training anchors are movie-level out-of-fold.",
        "- Every model issues h1 before observing the next transition; h2/h4/h6 sum chronological h1 predictions.",
        "- Seeds are averaged within movie. Exact paired sign-flip tests and movie bootstrap never treat cells as independent replicates.",
        "- Two RMSE definitions are kept explicitly: component RMSE and Euclidean vector RMSE.",
        "- This is a frozen post-development LOMO estimate, not a claim that architecture selection was nested or that the six movies were never seen during research.",
        "",
        "## Completion",
        "",
        f"- Completed test movies in analyzable seed aggregates: `{completed_movies}`.",
        f"- Raw metric rows: `{len(raw)}`; movie/method/horizon rows: `{len(seed_aggregated)}`.",
        f"- Failed integrity checks: `{int((~integrity.passed).sum()) if not integrity.empty else 0}`.",
        "",
    ]
    if not primary_table.empty:
        columns = [
            "method_id",
            "n_movies",
            "component_rmse_movie_mean",
            "component_rmse_movie_sd",
            "vector_rmse_movie_mean",
            "vector_rmse_movie_sd",
        ]
        lines.extend([f"## Movie-Level h{args.primary_horizon}", "", primary_table[columns].to_markdown(index=False), ""])
    if not primary_tests.empty:
        columns = [
            "comparator_method",
            "n_movies",
            "mean_delta_comparator_minus_primary",
            "relative_improvement_pct",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p_two_sided",
            "holm_p_primary_endpoint_family",
            "holm_p_global",
        ]
        lines.extend(["## Paired Primary-Endpoint Inference", "", primary_tests[columns].to_markdown(index=False), ""])
    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- With six movies, the smallest possible two-sided exact sign-flip p-value is 0.03125; multiplicity correction can therefore be severe.",
            "- The movie mean is the publication estimand. A pooled-cell RMSE may be added descriptively, but cannot replace movie-level inference.",
            "- Missing compute fields are deliberate placeholders, not zeros. A final paper table must include frozen-v52 cost and equal-budget ensembles.",
            "- Any failed causal or held-out-movie contract invalidates that runner/fold instead of being silently averaged.",
            "",
            f"Compute-accounting rows: `{len(compute)}`.",
        ]
    )
    (args.out_dir / "v102_publication_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyse(args: argparse.Namespace, jobs: Sequence[Job], seeds: Sequence[int]) -> None:
    raw, integrity = collect_metrics(args, jobs)
    integrity.to_csv(args.out_dir / "v102_completed_job_integrity.csv", index=False)
    if raw.empty:
        if args.allow_incomplete_analysis:
            print("[v102] no completed runner outputs available for analysis", flush=True)
            return
        raise RuntimeError("no completed runner outputs available for analysis")
    if not args.allow_incomplete_analysis and not integrity.empty and not bool(integrity.passed.all()):
        failures = integrity[~integrity.passed].to_dict("records")
        raise RuntimeError(f"publication integrity checks failed: {failures[:10]}")
    raw.to_csv(args.out_dir / "v102_raw_fold_seed_metrics.csv", index=False)
    seed_aggregated = aggregate_seeds(raw, seeds, bool(args.allow_incomplete_analysis))
    if seed_aggregated.empty:
        raise RuntimeError("no method has complete seed coverage; use --allow-incomplete-analysis only for diagnostics")
    if not args.allow_incomplete_analysis:
        validate_publication_coverage(seed_aggregated, seeds, args.primary_method)
    seed_aggregated.to_csv(args.out_dir / "v102_seed_aggregated_within_movie.csv", index=False)
    movie_summary, inference = paired_inference(args, seed_aggregated)
    movie_summary.to_csv(args.out_dir / "v102_movie_level_summary.csv", index=False)
    inference.to_csv(args.out_dir / "v102_paired_movie_inference.csv", index=False)
    compute = collect_compute(args, jobs, raw)
    compute.to_csv(args.out_dir / "v102_compute_accounting.csv", index=False)
    write_analysis_report(args, raw, seed_aggregated, movie_summary, inference, integrity, compute)


def parse_runner_extra(value: str) -> dict[str, list[str]]:
    if not value.strip():
        return {}
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(value)
    result: dict[str, list[str]] = {}
    for runner, tokens in payload.items():
        if runner not in RUNNER_SPECS:
            raise ValueError(f"runner-extra contains unknown runner {runner!r}")
        if isinstance(tokens, str):
            result[runner] = shlex.split(tokens)
        elif isinstance(tokens, list):
            result[runner] = [str(token) for token in tokens]
        else:
            raise TypeError(f"runner-extra[{runner!r}] must be a string or list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--movies", default="1,2,3,4,5,6", help="Must remain exactly the six frozen MDCK_Bulk movies.")
    parser.add_argument("--validation-map", default="", help="Optional derangement, e.g. 1:2,2:3,...,6:1.")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--runners", default="v97,v98,v99")
    parser.add_argument("--only-runners", default="", help="Execution filter; anchor dependencies are still built.")
    parser.add_argument("--runner-extra", default="", help="JSON string/path mapping runner names to extra CLI tokens.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--validation-horizon-weights", default="1:0.90,2:0.05,4:0.03,6:0.02")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Diagnostic only; publication LOMO must use 0.")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Diagnostic only; random limits may destroy rolling chains.")
    parser.add_argument("--max-test-rows", type=int, default=0, help="Diagnostic only; publication LOMO must use 0.")
    parser.add_argument("--anchor-posterior-epochs", type=int, default=20)
    parser.add_argument("--anchor-student-epochs", type=int, default=16)
    parser.add_argument("--anchor-route-k", type=int, default=12)
    parser.add_argument("--force-anchor", action="store_true")
    parser.add_argument("--v97-variants", default="v97_direct")
    parser.add_argument("--v97-evaluation-variant", default="v97_direct")
    parser.add_argument(
        "--v97-cumulative-weight",
        type=float,
        default=0.05,
        help="Frozen h1-strict v97 uses 0.05; trajectory-balanced uses 0.30.",
    )
    parser.add_argument(
        "--v97-skip-recurrent-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Frozen h1-strict v97 skipped its internal recurrent baselines; use v99 for matched learned comparators.",
    )
    parser.add_argument("--v98-models", default="cv,ca")
    parser.add_argument("--v98-loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--v98-cumulative-weight", type=float, default=0.05)
    parser.add_argument("--v98-epochs", type=int, default=50)
    parser.add_argument("--v98-patience", type=int, default=10)
    parser.add_argument(
        "--v99-models",
        default="temporal_transformer,pyg_transformerconv,social_lstm,agentformer_cell_adapter,mtr_cell_adapter,qcnet_cell_adapter",
    )
    parser.add_argument("--v99-input-variants", default="raw_coordinate,v52_anchor")
    parser.add_argument("--primary-method", default="v97/v97_direct")
    parser.add_argument("--primary-horizon", type=int, default=6)
    parser.add_argument("--primary-metric", choices=list(METRICS), default="component_rmse")
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=102)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--duplicate-tolerance", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true", help="Write/validate the complete manifest without training.")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--allow-incomplete-analysis", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-anchor-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.cache_root = (args.cache_root or (args.out_dir / "cache")).resolve()
    args.features = args.features.resolve()
    args.table_root = args.table_root.resolve()
    args.runner_extra = parse_runner_extra(args.runner_extra)
    return args


def main() -> None:
    args = parse_args()
    if args.worker_anchor_spec is not None:
        anchor_worker(args.worker_anchor_spec.resolve())
        return
    movies = tuple(parse_ints(args.movies))
    if movies != MOVIES:
        raise ValueError(f"publication protocol is frozen to movies {MOVIES}; got {movies}")
    seeds = tuple(parse_ints(args.seeds))
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be unique and nonempty")
    horizons = tuple(parse_ints(args.horizons))
    if horizons != HORIZONS:
        raise ValueError(f"publication protocol is frozen to horizons {HORIZONS}; got {horizons}")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("confidence must be in (0,1)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    mapping = validation_mapping(args.validation_map, movies)
    folds = make_folds(movies, mapping)
    jobs = build_jobs(args, folds, seeds)
    write_manifest(args, folds, seeds, jobs)
    preflight(args, folds, jobs)
    if args.dry_run:
        print(f"[v102] dry-run complete: {len(folds)} folds, {len(seeds)} seeds, {len(jobs)} cached jobs")
        print(args.out_dir / "v102_protocol_manifest.json")
        return
    if not args.analyze_only:
        execute_jobs(args, jobs)
        write_manifest(args, folds, seeds, jobs)
    analyse(args, jobs, seeds)
    print(args.out_dir / "v102_publication_benchmark_report.md")


if __name__ == "__main__":
    main()
