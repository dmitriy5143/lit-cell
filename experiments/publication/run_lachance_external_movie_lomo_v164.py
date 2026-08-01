#!/usr/bin/env python3
"""Nested movie-level LOMO confirmation for external LaChance domains.

For every outer held-out movie, one different movie is reserved for validation
and every remaining movie is used for training.  The v97 checkpoint and v162
transport are therefore both refitted without access to the outer test movie.
The default invocation is a two-domain pilot; ``--all-movies`` expands any
requested domain to its complete available cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_dimensionless_multidomain_transport_v162 as v162  # noqa: E402
import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_semigroup_external_guards_v157i as v157i  # noqa: E402
import run_lachance_v162_external_seed_confirmation as seed_runner  # noqa: E402


BUILD_SCRIPT = SCRIPTS / "build_lachance_online_track_anchor_cache_v97.py"
TRAIN_SCRIPT = SCRIPTS / "run_lachance_causal_innovation_state_space_v97.py"
DEFAULT_OUT = ROOT / "outputs" / "lachance_external_movie_lomo_v164"
TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"

SOURCE_CONFIGS = {
    "MDCK_Edge": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_edge_guard_seed42_2026-07-21"
    / "run_config.json",
    "HUVEC": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_huvec_guard_seed42_2026-07-21"
    / "run_config.json",
    "MDAMB231": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_direct_mdamb231_guard_seed42_2026-07-21"
    / "run_config.json",
}
VARIANTS = {
    "MDCK_Edge": "v97_no_context",
    "HUVEC": "v97_no_context",
    "MDAMB231": "v97_direct",
}
PILOT_HELDOUTS = {
    "MDCK_Edge": [7],
    "HUVEC": [22],
    "MDAMB231": [15],
}


def parse_strings(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def available_movies(dataset: str) -> list[int]:
    movies: list[int] = []
    for path in sorted((TABLE_ROOT / dataset).glob(f"{dataset}_*_tracks.csv")):
        movies.append(int(path.stem.split("_")[-2]))
    if len(movies) < 3:
        raise RuntimeError(f"Need at least three movies for {dataset}: {movies}")
    return movies


def parse_heldouts(value: str) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    if not value.strip():
        return output
    for token in value.split(";"):
        dataset, raw = token.split("=", 1)
        output[dataset.strip()] = parse_ints(raw)
    return output


def validation_movie(movies: list[int], heldout: int) -> int:
    index = movies.index(heldout)
    return movies[(index - 1) % len(movies)]


def cli_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
        raise RuntimeError(
            f"Command failed ({process.returncode}): {' '.join(command)}\n{tail}"
        )


def build_command(
    dataset: str,
    train_movies: list[int],
    validation: int,
    heldout: int,
    cache_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(BUILD_SCRIPT),
        "--dataset",
        dataset,
        "--table-root",
        str(TABLE_ROOT),
        "--train-seq",
        ",".join(map(str, train_movies)),
        "--val-seq",
        str(validation),
        "--test-seq",
        str(heldout),
        "--max-horizon",
        "6",
        "--max-train-rows",
        str(args.max_train_rows),
        "--max-val-rows",
        str(args.max_val_rows),
        "--max-test-rows",
        str(args.max_test_rows),
        "--max-val-rows-per-sequence",
        str(args.max_val_rows),
        "--max-test-rows-per-sequence",
        str(args.max_test_rows),
        "--seed",
        str(args.seed),
        "--out-dir",
        str(cache_dir),
    ]


def training_command(
    dataset: str,
    cache_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    config = json.loads(SOURCE_CONFIGS[dataset].read_text(encoding="utf-8"))
    config.update(
        {
            "anchor_cache": str(cache_dir),
            "features": str(cache_dir / "native_feature_index.csv"),
            "out_dir": str(model_dir),
            "seed": int(args.seed),
            "device": args.device,
            "variants": VARIANTS[dataset],
            "evaluation_variant": VARIANTS[dataset],
            "skip_recurrent_baselines": True,
            "checkpoint_only": True,
            "smoke": bool(args.smoke),
        }
    )
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for key, value in config.items():
        if value is None or value is False:
            continue
        if isinstance(value, bool):
            command.append(cli_name(key))
        else:
            command.extend([cli_name(key), str(value)])
    return command


def evaluate_fold(
    dataset: str,
    checkpoint: Path,
    objective: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    restored = v157i.restore_run(
        dataset,
        checkpoint,
        v157e.device_from_cli("cpu"),
    )
    domain = v162.prepare_domain(
        restored,
        v162.parse_floats(args.scale_multipliers),
        int(args.max_scale_frames),
        int(args.seed) + 164_000,
    )
    domains = {dataset: domain}
    selection, model, grid = v162.select_model(
        domains,
        [dataset],
        objective,
        v162.parse_floats(args.alphas),
        v162.parse_floats(args.bounds_z),
    )
    metrics = v162.evaluate(
        model,
        selection,
        domains,
        [dataset],
        objective,
        "nested_lomo",
        controls=("real", "wrong_cell", "stale_time"),
        source_names=[dataset],
    )
    return metrics, grid


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="HUVEC,MDAMB231")
    parser.add_argument(
        "--heldouts",
        default="HUVEC=22;MDAMB231=15",
        help="Semicolon-separated DATASET=movie,movie map.",
    )
    parser.add_argument("--all-movies", action="store_true")
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--max-val-rows", type=int, default=15_000)
    parser.add_argument("--max-test-rows", type=int, default=15_000)
    parser.add_argument("--alphas", default="1,10,100,1000,10000")
    parser.add_argument("--bounds-z", default="0.1,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--scale-multipliers", default="1,2,4,8")
    parser.add_argument("--max-scale-frames", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--folds-only",
        action="store_true",
        help="Write only disjoint fold artifacts and a shard manifest.",
    )
    parser.add_argument(
        "--shard-id",
        default="",
        help="Stable identifier required with --folds-only.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    if args.folds_only:
        if not args.shard_id:
            raise ValueError("--shard-id is required with --folds-only")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.shard_id):
            raise ValueError(f"Unsafe shard id: {args.shard_id!r}")
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets = parse_strings(args.datasets)
    requested = parse_heldouts(args.heldouts)
    objectives = parse_strings(args.objectives)
    metric_frames: list[pd.DataFrame] = []
    grid_frames: list[pd.DataFrame] = []
    contracts: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset not in SOURCE_CONFIGS:
            raise ValueError(f"Unsupported dataset {dataset}")
        movies = available_movies(dataset)
        heldouts = movies if args.all_movies else requested.get(
            dataset,
            PILOT_HELDOUTS[dataset],
        )
        unknown = sorted(set(heldouts) - set(movies))
        if unknown:
            raise ValueError(f"Unknown {dataset} heldouts: {unknown}")
        for heldout in heldouts:
            validation = validation_movie(movies, heldout)
            train_movies = [
                movie
                for movie in movies
                if movie not in {heldout, validation}
            ]
            fold = output / dataset / f"test_{heldout:02d}"
            cache_dir = fold / "anchor_cache"
            model_dir = fold / "v97"
            checkpoint = model_dir / f"{VARIANTS[dataset]}.pt"
            print(
                f"[v164] {dataset} test={heldout} val={validation}",
                flush=True,
            )
            if not (cache_dir / "native_cache_status.json").exists():
                run_logged(
                    build_command(
                        dataset,
                        train_movies,
                        validation,
                        heldout,
                        cache_dir,
                        args,
                    ),
                    fold / "build_cache.log",
                )
            if not checkpoint.exists() and not args.skip_training:
                run_logged(
                    training_command(dataset, cache_dir, model_dir, args),
                    fold / "train_v97.log",
                )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            for objective in objectives:
                metric_path = fold / f"transport_{objective}_metrics.csv"
                grid_path = fold / f"transport_{objective}_selection.csv"
                if metric_path.exists() and grid_path.exists():
                    metrics = pd.read_csv(metric_path)
                    grid = pd.read_csv(grid_path)
                else:
                    metrics, grid = evaluate_fold(
                        dataset,
                        checkpoint,
                        objective,
                        args,
                    )
                    metrics.to_csv(metric_path, index=False)
                    grid.to_csv(grid_path, index=False)
                metrics.insert(0, "outer_test_movie", heldout)
                metrics.insert(1, "inner_validation_movie", validation)
                metrics.insert(2, "seed", args.seed)
                grid.insert(0, "outer_test_movie", heldout)
                grid.insert(1, "inner_validation_movie", validation)
                grid.insert(2, "seed", args.seed)
                metric_frames.append(metrics)
                grid_frames.append(grid)
            contracts.append(
                {
                    "dataset": dataset,
                    "outer_test_movie": heldout,
                    "inner_validation_movie": validation,
                    "train_movies": ",".join(map(str, train_movies)),
                    "train_count": len(train_movies),
                    "seed": args.seed,
                    "variant": VARIANTS[dataset],
                    "checkpoint": str(checkpoint),
                    "outer_test_seen_during_training": False,
                    "outer_test_seen_during_selection": False,
                }
            )
    if args.folds_only:
        shard_dir = output / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_manifest = {
            "schema_version": 1,
            "shard_id": args.shard_id,
            "elapsed_seconds": time.time() - started,
            "datasets": datasets,
            "heldouts": requested,
            "seed": args.seed,
            "device": args.device,
            "completed_outer_folds": len(contracts),
            "contracts": contracts,
            "target_leakage": False,
            "args": finite(vars(args)),
        }
        shard_path = shard_dir / f"{args.shard_id}.json"
        shard_path.write_text(
            json.dumps(shard_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[v164] wrote shard {shard_path}", flush=True)
        return
    metrics = pd.concat(metric_frames, ignore_index=True)
    grids = pd.concat(grid_frames, ignore_index=True)
    contract_table = pd.DataFrame(contracts)
    metrics.to_csv(output / "v164_nested_lomo_metrics.csv", index=False)
    grids.to_csv(output / "v164_nested_lomo_selection.csv", index=False)
    contract_table.to_csv(output / "v164_nested_lomo_contract.csv", index=False)
    summary = (
        metrics.groupby(
            ["dataset", "objective", "control", "horizon"],
            as_index=False,
        )
        .agg(
            outer_folds=("outer_test_movie", "nunique"),
            component_rmse=("component_rmse", "mean"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            gain_vs_prior_percent=("rmse_improvement_percent", "mean"),
            positive_outer_folds=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    summary.to_csv(output / "v164_nested_lomo_summary.csv", index=False)
    h6 = summary[
        summary.objective.eq("h6_guard10")
        & summary.horizon.eq(6)
    ]
    operating_points = summary[
        summary.control.eq("real")
        & summary.horizon.isin([1, 6])
    ].copy()
    operating_points.to_csv(
        output / "v164_nested_lomo_operating_points.csv",
        index=False,
    )
    fold_h6 = metrics[
        metrics.objective.eq("h6_guard10")
        & metrics.horizon.eq(6)
    ].pivot(
        index=[
            "dataset",
            "outer_test_movie",
            "inner_validation_movie",
            "seed",
        ],
        columns="control",
        values=["component_rmse", "rmse_improvement_percent"],
    )
    fold_h6.columns = [
        f"{metric}_{control}" for metric, control in fold_h6.columns
    ]
    fold_h6 = fold_h6.reset_index()
    fold_h6["real_beats_stale"] = (
        fold_h6.component_rmse_real
        < fold_h6.component_rmse_stale_time
    )
    fold_h6["real_beats_wrong_cell"] = (
        fold_h6.component_rmse_real
        < fold_h6.component_rmse_wrong_cell
    )
    fold_h6["positive_vs_prior"] = (
        fold_h6.rmse_improvement_percent_real > 0
    )
    fold_h6["causal_control_pass"] = (
        fold_h6.real_beats_stale
        & fold_h6.real_beats_wrong_cell
        & fold_h6.positive_vs_prior
    )
    fold_h6.to_csv(
        output / "v164_nested_lomo_fold_decision.csv",
        index=False,
    )
    report = [
        "# v164 External Nested Movie LOMO",
        "",
        "Each outer test movie is absent from both training and validation.",
        "The immediately preceding available movie is the inner validation fold.",
        "",
        "## h6",
        "",
        h6.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Operating points",
        "",
        operating_points.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold-level causal controls",
        "",
        fold_h6.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Completed outer folds: `{len(contract_table)}`.",
        (
            "This is a pilot rather than full LOMO."
            if not args.all_movies
            else "All locally available movies were used as outer folds."
        ),
    ]
    (output / "v164_nested_lomo_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    fold_provenance = []
    for row in contracts:
        checkpoint = Path(row["checkpoint"])
        config_path = checkpoint.parent / "run_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        fold_provenance.append(
            {
                "dataset": row["dataset"],
                "outer_test_movie": row["outer_test_movie"],
                "inner_validation_movie": row["inner_validation_movie"],
                "train_movies": row["train_movies"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "training_device": config.get("device"),
                "training_seed": config.get("seed"),
                "training_config": str(config_path),
            }
        )
    previous_manifest_path = output / "run_manifest.json"
    previous_manifest = {}
    if previous_manifest_path.exists():
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
    initial_training_invocation = previous_manifest.get(
        "initial_training_invocation"
    )
    if initial_training_invocation is None:
        initial_args = finite(vars(args))
        initial_args["skip_training"] = False
        initial_training_invocation = {
            "elapsed_seconds": (
                time.time() - started if not args.skip_training else None
            ),
            "args": initial_args,
            "reconstructed_from_fold_artifacts": bool(args.skip_training),
        }
    manifest = {
        "schema_version": 2,
        "datasets": datasets,
        "all_movies": bool(args.all_movies),
        "heldouts": requested,
        "seed": args.seed,
        "nested_movie_holdout": True,
        "target_leakage": False,
        "completed_outer_folds": len(contracts),
        "fold_provenance": fold_provenance,
        "initial_training_invocation": initial_training_invocation,
        "last_aggregation_invocation": {
            "elapsed_seconds": time.time() - started,
            "args": finite(vars(args)),
        },
        "provenance_note": (
            "The first post-processing rerun predated schema v2 and overwrote "
            "its wall-clock manifest. Fold run_config files and checkpoint "
            "hashes remain authoritative for training provenance."
        ),
    }
    previous_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[v164] wrote {output}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
