#!/usr/bin/env python3
"""Validate the frozen v166 publication evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    ROOT / "outputs" / "lachance_publication_bundle_v166_2026-07-27"
)

REQUIRED_TABLES = {
    "v166_configuration_unseen_transport.csv",
    "v166_configuration_unseen_paired.csv",
    "v166_bulk_lomo_baselines.csv",
    "v166_bulk_lomo_paired.csv",
    "v166_dimensionless_transfer.csv",
    "v166_temporal_covariance.csv",
    "v166_external_full_lomo.csv",
    "v166_external_causal_controls.csv",
    "v166_turning_context.csv",
    "v166_claim_scope.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(entries: dict[str, str], label: str) -> int:
    checked = 0
    for raw_path, expected in entries.items():
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} path is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} hash mismatch: {path}")
        checked += 1
    return checked


def nested_v165_checks(manifest: dict[str, Any]) -> dict[str, int]:
    if manifest.get("folds_expected") != 35:
        raise ValueError("Nested v165 manifest does not require 35 folds")
    if manifest.get("target_leakage") is not False:
        raise ValueError("Nested v165 manifest does not certify leakage=false")
    counts: dict[str, int] = {}
    for key in (
        "code_sha256",
        "input_table_sha256",
        "checkpoint_sha256",
        "training_config_sha256",
        "artifact_sha256",
    ):
        entries = manifest.get(key, {})
        if not entries or any(value is None for value in entries.values()):
            raise ValueError(f"Nested v165 {key} is incomplete")
        counts[key] = check_hashes(entries, f"v165 {key}")
    if counts["checkpoint_sha256"] != 35:
        raise ValueError("Nested v165 manifest does not hash 35 checkpoints")
    if counts["training_config_sha256"] != 35:
        raise ValueError("Nested v165 manifest does not hash 35 run configs")
    inventory = manifest.get("fold_inventory", [])
    inventory_keys = {
        (row.get("dataset"), row.get("outer_test_movie")) for row in inventory
    }
    if len(inventory) != 35 or len(inventory_keys) != 35:
        raise ValueError("Nested v165 fold inventory is not one row per fold")
    if any(
        row.get("checkpoint_sha256") is None
        or row.get("training_config_sha256") is None
        for row in inventory
    ):
        raise ValueError("Nested v165 fold inventory has unhashed entries")
    return counts


def check_finite(path: Path) -> int:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    if len(numeric.columns) and not np.isfinite(
        numeric.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(f"Non-finite values in {path}")
    return len(frame)


def run(args: argparse.Namespace) -> None:
    bundle = args.bundle_dir.resolve()
    manifest_path = bundle / "v166_publication_manifest.json"
    decision_path = bundle / "v166_sota_decision.json"
    if not manifest_path.exists() or not decision_path.exists():
        raise FileNotFoundError("v166 manifest or SOTA decision is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    if manifest.get("protocol_tables_separated") is not True:
        raise ValueError("Protocol separation is not certified")
    if manifest.get("global_sota_claim_allowed") is not False:
        raise ValueError("Manifest must forbid a global SOTA claim")
    if decision.get("global_sota") is not False:
        raise ValueError("Decision must not assert global SOTA")
    if decision.get("architecture_frozen") is not True:
        raise ValueError("Architecture is not marked frozen")

    missing = [
        name for name in sorted(REQUIRED_TABLES) if not (bundle / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing v166 tables: {missing}")

    table_rows = {
        name: check_finite(bundle / name) for name in sorted(REQUIRED_TABLES)
    }
    external = pd.read_csv(bundle / "v166_external_full_lomo.csv")
    expected_folds = {"HUVEC": 18, "MDAMB231": 17}
    for dataset, expected in expected_folds.items():
        subset = external[
            external.dataset.eq(dataset) & external.control.eq("real")
        ]
        if set(subset.outer_folds.astype(int)) != {expected}:
            raise ValueError(
                f"{dataset} external LOMO is not {expected}/{expected}"
            )

    scope = pd.read_csv(bundle / "v166_claim_scope.csv")
    if scope.protocol.duplicated().any():
        raise ValueError("Claim-scope protocols are not unique")
    if not scope.forbidden_claim.fillna("").str.strip().astype(bool).all():
        raise ValueError("Every protocol must state a forbidden claim")

    artifact_hashes = check_hashes(
        manifest.get("artifact_sha256", {}),
        "artifact",
    )
    input_hashes = check_hashes(
        manifest.get("input_sha256", {}),
        "input",
    )
    v165_manifest_paths = [
        Path(path)
        for path in manifest.get("input_sha256", {})
        if Path(path).name == "v165_publication_manifest.json"
    ]
    if len(v165_manifest_paths) != 1:
        raise ValueError("Exactly one nested v165 manifest is required")
    v165_manifest = json.loads(
        v165_manifest_paths[0].read_text(encoding="utf-8")
    )
    v165_hash_checks = nested_v165_checks(v165_manifest)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "bundle": str(bundle),
        "protocol_tables_separated": True,
        "global_sota_claim_allowed": False,
        "external_outer_folds": expected_folds,
        "table_rows": table_rows,
        "artifact_hashes_checked": artifact_hashes,
        "input_hashes_checked": input_hashes,
        "nested_v165_hashes_checked": v165_hash_checks,
    }
    output = bundle / "v166_validation_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[v166-validate] PASS: {output}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
