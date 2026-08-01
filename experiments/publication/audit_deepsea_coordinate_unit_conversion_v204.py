#!/usr/bin/env python3
"""Verify pixel-to-cell-diameter equivalence of the DeepSea v204 caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402


def run(args: argparse.Namespace) -> None:
    pixel = v85.load_anchor_cache(args.pixel_cache)
    normalized = v85.load_anchor_cache(args.normalized_cache)
    splits: list[dict[str, object]] = []
    passed = True
    for split, pixel_bundle, normalized_bundle in zip(
        ("train", "val", "test"),
        pixel,
        normalized,
        strict=True,
    ):
        pixel_keys = pixel_bundle.rows[
            ["sequence", "frame", "track_id"]
        ].to_numpy(np.int64)
        normalized_keys = normalized_bundle.rows[
            ["sequence", "frame", "track_id"]
        ].to_numpy(np.int64)
        keys_match = bool(np.array_equal(pixel_keys, normalized_keys))
        diameter = pixel_bundle.rows["reference_diameter_px"].to_numpy(
            np.float32
        )
        target_delta = float(
            np.max(
                np.abs(
                    pixel_bundle.target_steps / diameter[:, None, None]
                    - normalized_bundle.target_steps
                )
            )
        )
        base_delta = float(
            np.max(
                np.abs(
                    pixel_bundle.base / diameter[:, None]
                    - normalized_bundle.base
                )
            )
        )
        split_passed = bool(
            keys_match
            and target_delta <= args.atol
            and base_delta <= args.atol
        )
        passed = passed and split_passed
        splits.append(
            {
                "split": split,
                "rows": int(len(pixel_keys)),
                "keys_match": keys_match,
                "target_max_abs_delta": target_delta,
                "base_max_abs_delta": base_delta,
                "passed": split_passed,
            }
        )
    payload = {
        "pixel_cache": str(args.pixel_cache.resolve()),
        "normalized_cache": str(args.normalized_cache.resolve()),
        "atol": args.atol,
        "splits": splits,
        "coordinate_unit_conversion_pass": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("DeepSea coordinate-unit conversion audit failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pixel-cache",
        type=Path,
        default=ROOT
        / "outputs/deepsea_online_h1_complete_cache_pixel_v204_2026-07-31",
    )
    parser.add_argument(
        "--normalized-cache",
        type=Path,
        default=ROOT
        / "outputs/deepsea_online_h1_complete_cache_v204_2026-07-31",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/deepsea_future_suffix_audit_dimensionless_v204_2026-07-31"
        / "v204_coordinate_unit_conversion.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
