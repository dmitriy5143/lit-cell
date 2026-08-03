#!/usr/bin/env python3
"""Validate every frozen LIT-Cell fold state committed with the release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_cell_forecasting.frozen_release import (  # noqa: E402
    file_sha256,
    load_frozen_fold_state,
    load_release_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT / "models" / "lit_cell_mdck_bulk_primary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir.resolve()
    manifest = load_release_manifest(release_dir)
    entries = manifest["checkpoints"]
    expected = {(movie, seed) for movie in range(1, 7) for seed in (7, 42, 123)}
    actual = {(int(row["test_movie"]), int(row["seed"])) for row in entries}
    if actual != expected or len(entries) != 18:
        raise SystemExit("Frozen release does not contain the complete 6 x 3 state grid")

    for relative, expected_hash in manifest["contract_files"].items():
        path = release_dir / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise SystemExit(f"Contract hash mismatch: {relative}")

    loaded = 0
    parameter_counts: set[int] = set()
    for entry in entries:
        frozen = load_frozen_fold_state(
            int(entry["test_movie"]),
            int(entry["seed"]),
            release_dir=release_dir,
            verify=True,
        )
        count = sum(parameter.numel() for parameter in frozen.model.parameters())
        parameter_counts.add(count)
        loaded += 1

    if parameter_counts != {380553}:
        raise SystemExit(f"Unexpected parameter counts: {sorted(parameter_counts)}")

    manifest_hash = hashlib.sha256(
        (release_dir / "manifest.json").read_bytes()
    ).hexdigest()
    print(
        json.dumps(
            {
                "status": "PASS",
                "method": manifest["method"],
                "loaded_checkpoints": loaded,
                "parameter_count_per_checkpoint": 380553,
                "manifest_sha256": manifest_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

