#!/usr/bin/env python3
"""Freeze the protocol, row identity, and multiplicity contract for v188."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "outputs" / "lachance_publication_bundle_v188_2026-07-29"
)
DEFAULT_REPLAY_MANIFEST = (
    ROOT
    / "outputs"
    / "lachance_foldlocal_semigroup_confirmation_v157e_full_2026-07-24"
    / "v157e_seed_replay_manifest.json"
)
DEFAULT_V117_MANIFEST = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_baselines_v117_production_2026-07-21"
    / "v117_protocol_manifest.json"
)
DEFAULT_V157H_MANIFEST = (
    ROOT
    / "outputs"
    / "lachance_foldlocal_semigroup_pareto_v157h_full_2026-07-24"
    / "run_manifest.json"
)
DEFAULT_KALMANNET_MANIFEST = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_kalmannet_v188_exact_2026-07-29"
    / "kalmannet_lomo_job_manifest.csv"
)
SEEDS = (7, 42, 123)
MOVIES = (1, 2, 3, 4, 5, 6)
HORIZONS = (1, 2, 4, 6)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=DEFAULT_REPLAY_MANIFEST,
    )
    parser.add_argument(
        "--v117-manifest",
        type=Path,
        default=DEFAULT_V117_MANIFEST,
    )
    parser.add_argument(
        "--v157h-manifest",
        type=Path,
        default=DEFAULT_V157H_MANIFEST,
    )
    parser.add_argument(
        "--kalmannet-manifest",
        type=Path,
        default=DEFAULT_KALMANNET_MANIFEST,
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_protocol_rows(replays: list[dict[str, Any]]) -> pd.DataFrame:
    if len(replays) != len(MOVIES) * len(SEEDS):
        raise ValueError(
            f"Expected {len(MOVIES) * len(SEEDS)} replay rows, "
            f"found {len(replays)}"
        )
    records: list[dict[str, Any]] = []
    for movie in MOVIES:
        fold = [row for row in replays if int(row["test_movie"]) == movie]
        found_seeds = tuple(sorted(int(row["seed"]) for row in fold))
        if found_seeds != SEEDS:
            raise ValueError(
                f"Outer movie {movie} has seeds {found_seeds}, expected {SEEDS}"
            )
        if not all(bool(row["strict_seed_key_target_match"]) for row in fold):
            raise ValueError(f"Outer movie {movie} failed seed key/target match")
        validation_movies = {int(row["validation_movie"]) for row in fold}
        train_movies = {
            tuple(int(value) for value in row["train_movies"]) for row in fold
        }
        if len(validation_movies) != 1 or len(train_movies) != 1:
            raise ValueError(f"Outer movie {movie} has inconsistent folds")
        test_splits = [row["splits"]["test"] for row in fold]
        for key in (
            "rows",
            "key_sha256",
            "target_sha256",
            "row_target_sha256",
        ):
            values = {split[key] for split in test_splits}
            if len(values) != 1:
                raise ValueError(
                    f"Outer movie {movie} has seed-dependent test {key}"
                )
        test = test_splits[0]
        records.append(
            {
                "protocol_id": "mdck_bulk_online_outer_lomo_v188",
                "cohort_id": "MDCK_Bulk_movies_1_6",
                "dataset": "MDCK_Bulk",
                "outer_test_movie": movie,
                "validation_movie": next(iter(validation_movies)),
                "train_movies": ",".join(
                    str(value) for value in next(iter(train_movies))
                ),
                "optimizer_seeds": ",".join(str(value) for value in SEEDS),
                "independent_unit": "movie",
                "issue_time": "after observing frame t",
                "latest_allowed_observation": "frame t",
                "forbidden_observation": "frame t+1 or any future target",
                "one_step_target": "displacement from t to t+1",
                "rolling_composition": (
                    "h2/h4/h6 sum consecutive predictions each issued "
                    "before its next observation"
                ),
                "horizons": ",".join(str(value) for value in HORIZONS),
                "ordered_key_columns": "sequence,frame,track_id",
                "test_rows": int(test["rows"]),
                "test_key_sha256": str(test["key_sha256"]),
                "test_target_sha256": str(test["target_sha256"]),
                "test_row_target_sha256": str(test["row_target_sha256"]),
                "seed_key_target_match": True,
                "component_rmse_definition": (
                    "sqrt(mean((dx_error^2 + dy_error^2) / 2))"
                ),
                "vector_rmse_definition": (
                    "sqrt(mean(dx_error^2 + dy_error^2))"
                ),
                "hyperparameter_selection": (
                    "all hyperparameters selected without outer-test access; "
                    "selection models fit on train movies and validation movie "
                    "is used for selection or early stopping"
                ),
                "final_fit_scope_by_method": (
                    "v166 bounded update refit on train+validation after "
                    "selection; v97, v117 learned baselines and KalmanNet use "
                    "their frozen train-only/validation-selection procedures; "
                    "classical filters have no supervised final refit"
                ),
                "outer_test_access": "evaluation only",
                "operating_points": "h1_strict,h6_utility",
                "status": "frozen_complete",
            }
        )
    frame = pd.DataFrame(records).sort_values("outer_test_movie")
    if frame.test_key_sha256.duplicated().any():
        raise ValueError("Outer test movies unexpectedly share ordered key hash")
    return frame


def multiplicity_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "alpha": 0.05,
        "independent_unit": "outer test movie",
        "optimizer_seed_policy": (
            "average optimizer seeds within each movie before inference"
        ),
        "confirmatory_family": [
            {
                "hypothesis_id": "H1",
                "method": "v166_h1_strict",
                "comparator": "v97_no_update",
                "horizon": 1,
                "metric": "component_rmse",
                "direction": "lower",
            },
            {
                "hypothesis_id": "H2",
                "method": "v166_h6_utility",
                "comparator": "v97_no_update",
                "horizon": 6,
                "metric": "component_rmse",
                "direction": "lower",
            },
        ],
        "correction": "Holm step-down across H1 and H2",
        "paired_test": "exact two-sided sign-flip test over six movies",
        "effect_interval": (
            "movie bootstrap 95% CI for comparator-minus-method RMSE"
        ),
        "secondary_status": (
            "all other method/horizon contrasts are descriptive or exploratory"
        ),
        "cohort_separation": {
            "matched_outer_lomo": "MDCK Bulk movies 1-6",
            "configuration_unseen": "MDCK Bulk movies 10-16",
            "rule": "never pool these cohorts in one paired statistic",
        },
        "protocol_separation": {
            "primary": "causal rolling/receding h1",
            "supplementary": "fixed-origin and turn classification",
            "rule": "never rank methods across protocol clocks",
        },
    }


def kalmannet_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "jobs_expected": 18,
            "jobs_complete": 0,
            "status": "missing_manifest",
        }
    frame = pd.read_csv(path)
    jobs = frame[frame["runner"].eq("v98")].copy()
    expected = len(MOVIES) * len(SEEDS)
    if len(jobs) != expected:
        raise ValueError(f"Expected {expected} KalmanNet jobs, found {len(jobs)}")
    completed = 0
    for row in jobs.itertuples(index=False):
        output = Path(str(row.output_dir))
        required = [
            output / "v98_online_summary.csv",
            output / "v98_data_contract.csv",
            output / "v98_no_future_sentinel.json",
            output / "v98_provenance.json",
        ]
        if not all(item.exists() for item in required):
            continue
        sentinel = load_json(output / "v98_no_future_sentinel.json")
        contract = pd.read_csv(output / "v98_data_contract.csv")
        if len(contract) != 1:
            continue
        contract_row = contract.iloc[0]
        if (
            sentinel.get("pass") is True
            and sentinel.get("future_placeholder_read_at_inference") is False
            and int(contract_row["future_target_inference_features"]) == 0
            and int(contract_row["test_movies"]) == int(row.test_movie)
            and int(contract_row["test_rows"]) == int(row.test_rows)
        ):
            completed += 1
    return {
        "jobs_expected": expected,
        "jobs_complete": completed,
        "status": "complete" if completed == expected else "pending",
    }


def build_source_status(
    kalmannet: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "v117_matched_baselines",
                "role": "classical, robust, tree and GRU comparators",
                "status": "complete",
                "completed_units": 6,
                "expected_units": 6,
                "headline_eligible": True,
            },
            {
                "source_id": "v157h_foldlocal_transport",
                "role": "matched v166 h1-strict and h6-utility",
                "status": "complete",
                "completed_units": 6,
                "expected_units": 6,
                "headline_eligible": True,
            },
            {
                "source_id": "v188_exact_kalmannet",
                "role": "modern learned-filter comparator",
                "status": kalmannet["status"],
                "completed_units": kalmannet["jobs_complete"],
                "expected_units": kalmannet["jobs_expected"],
                "headline_eligible": kalmannet["status"] == "complete",
            },
            {
                "source_id": "trackssm",
                "role": "conditional source-faithful modern comparator",
                "status": "scoped_out_pending_fidelity",
                "completed_units": 0,
                "expected_units": 0,
                "headline_eligible": False,
            },
        ]
    )


def run(args: argparse.Namespace) -> None:
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay_path = args.replay_manifest.resolve()
    v117_path = args.v117_manifest.resolve()
    v157h_path = args.v157h_manifest.resolve()
    kalmannet_path = args.kalmannet_manifest.resolve()
    for path in (replay_path, v117_path, v157h_path):
        if not path.exists():
            raise FileNotFoundError(path)

    replays = load_json(replay_path)
    if not isinstance(replays, list):
        raise TypeError("Replay manifest must contain a list")
    protocol = build_protocol_rows(replays)
    v117 = load_json(v117_path)
    if v117.get("evaluation") != (
        "h1 issued before t+1; rolling h2/h4/h6 sum issued h1"
    ):
        raise ValueError("v117 evaluation clock does not match frozen contract")
    v157h = load_json(v157h_path)
    if v157h.get("protocol") != "strict fold-local streaming/receding h1":
        raise ValueError("v157h protocol does not match frozen contract")

    multiplicity = multiplicity_contract()
    kalmannet = kalmannet_status(kalmannet_path)
    sources = build_source_status(kalmannet)
    protocol_path = output / "v188_protocol_contract.csv"
    multiplicity_path = output / "v188_multiplicity_contract.json"
    source_path = output / "v188_source_status.csv"
    protocol.to_csv(protocol_path, index=False)
    multiplicity_path.write_text(
        json.dumps(multiplicity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sources.to_csv(source_path, index=False)

    contract_hash = canonical_sha256(
        {
            "protocol_rows": protocol.to_dict("records"),
            "multiplicity": multiplicity,
        }
    )
    manifest = {
        "schema_version": 1,
        "frozen": True,
        "contract_sha256": contract_hash,
        "input_sha256": {
            str(path): sha256(path)
            for path in (replay_path, v117_path, v157h_path)
        },
        "optional_input_sha256": (
            {str(kalmannet_path): sha256(kalmannet_path)}
            if kalmannet_path.exists()
            else {}
        ),
        "artifact_sha256": {
            str(path): sha256(path)
            for path in (protocol_path, multiplicity_path, source_path)
        },
        "global_sota_claim_allowed": False,
        "fixed_origin_oracle_primary": False,
        "builder_sha256": sha256(Path(__file__)),
    }
    (output / "v188_contract_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v188-contract] frozen {len(protocol)} folds; "
        f"KalmanNet {kalmannet['jobs_complete']}/"
        f"{kalmannet['jobs_expected']}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
