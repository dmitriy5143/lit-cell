#!/usr/bin/env python3
"""Build chronological DeepSea caches consumed by the exact v97 runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


KEYS = ["sequence", "frame", "track_id"]


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def canonicalize_coordinates(table: pd.DataFrame, unit: str) -> pd.DataFrame:
    output = table.copy()
    if unit == "pixel":
        output["coordinate_unit"] = "pixel_per_frame"
        return output
    if unit != "cell_diameter":
        raise ValueError(f"Unsupported coordinate unit: {unit}")
    mapping = {
        "x_px": "x_cell_diam",
        "y_px": "y_cell_diam",
        "dx_px": "dx_cell_diam",
        "dy_px": "dy_cell_diam",
        "target_dx_px": "target_dx_cell_diam",
        "target_dy_px": "target_dy_cell_diam",
    }
    missing = sorted(set(mapping.values()) - set(output.columns))
    if missing:
        raise ValueError(f"Cell-diameter representation is missing {missing}")
    for canonical, source in mapping.items():
        output[f"{canonical}_native"] = output[canonical]
        output[canonical] = output[source]
    output["coordinate_unit"] = "first_frame_median_cell_diameter_per_frame"
    return output


def track_order(table: pd.DataFrame, seed: int) -> list[int]:
    tracks = table.track_id.drop_duplicates().to_numpy(np.int64)
    order = np.random.default_rng(seed).permutation(len(tracks))
    return [int(tracks[index]) for index in order]


def build_video(
    table: pd.DataFrame,
    sequence: int,
    max_horizon: int,
    max_rows: int,
    seed: int,
    row_contract: str,
) -> v84.AnchorBundle:
    table = table.sort_values(["track_id", "frame"]).reset_index(drop=True)
    groups = {
        int(track): group.reset_index(drop=True)
        for track, group in table.groupby("track_id", sort=False)
    }
    row_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    selected_tracks = 0
    selected_rows = 0
    for track in track_order(table, seed):
        group = groups[track]
        frames = group.frame.to_numpy(np.int64)
        current = group[["dx_px", "dy_px"]].to_numpy(np.float32)
        target = group[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
        positions: list[int] = []
        targets: list[np.ndarray] = []
        if row_contract == "h6_eligible":
            for position in range(1, max(1, len(group) - max_horizon)):
                stop = position + max_horizon
                if stop >= len(group):
                    break
                if not np.all(np.diff(frames[position - 1 : stop + 1]) == 1):
                    continue
                if not np.all(np.isfinite(current[position])):
                    continue
                if not np.all(np.isfinite(target[position:stop])):
                    continue
                if bool(group.iloc[position].get("is_division_parent", False)):
                    continue
                positions.append(position)
                targets.append(target[position:stop])
        elif row_contract == "h1_complete":
            for position in range(1, len(group) - 1):
                if not np.all(np.diff(frames[position - 1 : position + 2]) == 1):
                    continue
                if not np.all(np.isfinite(current[position])):
                    continue
                if not np.all(np.isfinite(target[position])):
                    continue
                if bool(group.iloc[position].get("is_division_parent", False)):
                    continue
                future = np.zeros((max_horizon, 2), dtype=np.float32)
                for horizon_index in range(max_horizon):
                    target_position = position + horizon_index
                    if target_position >= len(group) - 1:
                        break
                    if not np.all(
                        np.diff(frames[position - 1 : target_position + 2]) == 1
                    ):
                        break
                    if not np.all(np.isfinite(target[target_position])):
                        break
                    future[horizon_index] = target[target_position]
                positions.append(position)
                targets.append(future)
        else:
            raise ValueError(f"Unknown row contract: {row_contract}")
        if not positions:
            continue
        if max_rows > 0 and selected_rows and selected_rows + len(positions) > max_rows:
            continue
        row_parts.append(group.iloc[positions].copy())
        target_parts.extend(targets)
        selected_tracks += 1
        selected_rows += len(positions)
        if max_rows > 0 and selected_rows >= max_rows:
            break
    if not row_parts:
        raise RuntimeError(f"No contiguous h{max_horizon} rows for sequence {sequence}")
    rows = pd.concat(row_parts, ignore_index=True)
    rows[KEYS] = rows[KEYS].astype(np.int64)
    target_steps = np.asarray(target_parts, dtype=np.float32)
    base = rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    return v84.AnchorBundle(
        name=f"DeepSea_{sequence}",
        rows=rows,
        anchor_residual=np.zeros((len(rows), max_horizon * 2), dtype=np.float32),
        base=base,
        target_steps=target_steps,
        meta={
            "dataset": "DeepSea",
            "sequence": int(sequence),
            "family": str(rows.family.iloc[0]),
            "video": str(rows.video.iloc[0]),
            "split": str(rows.split.iloc[0]),
            "anchor_method": "constant_velocity",
            "row_contract": row_contract,
            "selected_tracks": selected_tracks,
            "available_tracks": int(table.track_id.nunique()),
            "source_rows": int(len(table)),
        },
    )


def concatenate(
    name: str, bundles: list[v84.AnchorBundle], contract: dict[str, Any]
) -> v84.AnchorBundle:
    if not bundles:
        raise RuntimeError(f"No bundles for {name}")
    merged = v84.concat_bundles(name, bundles)
    merged.meta = {**merged.meta, "contract": contract, "anchor_method": "constant_velocity"}
    return merged


def allocate_caps(table: pd.DataFrame, split: str, total_cap: int) -> dict[int, int]:
    sequences = sorted(table.loc[table.split == split, "sequence"].astype(int).unique())
    if total_cap <= 0:
        return {sequence: 0 for sequence in sequences}
    weights = (
        table.loc[table.split == split]
        .groupby("sequence")
        .size()
        .reindex(sequences)
        .to_numpy(float)
    )
    weights /= max(weights.sum(), 1.0)
    caps = np.maximum(np.floor(weights * total_cap).astype(int), 64)
    return {sequence: int(cap) for sequence, cap in zip(sequences, caps)}


def run(args: argparse.Namespace) -> None:
    table = pd.read_csv(args.tracks)
    required = set(
        KEYS
        + [
            "family",
            "video",
            "split",
            "x_px",
            "y_px",
            "dx_px",
            "dy_px",
            "target_dx_px",
            "target_dy_px",
        ]
    )
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Prepared DeepSea table is missing {missing}")
    table = canonicalize_coordinates(table, args.coordinate_unit)
    for key in KEYS:
        table[key] = table[key].astype(int)
    observed_splits = set(table.split.unique())
    if observed_splits != {"train", "val", "test"}:
        raise RuntimeError(f"Expected train/val/test videos, found {observed_splits}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_sequences = {
        split: sorted(table.loc[table.split == split, "sequence"].astype(int).unique())
        for split in ("train", "val", "test")
    }
    if set(split_sequences["train"]) & set(split_sequences["val"] + split_sequences["test"]):
        raise RuntimeError("Movie leakage across frozen splits")
    contract = {
        "dataset": "DeepSea",
        "version": "v204",
        "train_seq": split_sequences["train"],
        "val_seq": split_sequences["val"],
        "test_seq": split_sequences["test"],
        "max_horizon": int(args.max_horizon),
        "anchor_method": "constant_velocity",
        "coordinate_unit": args.coordinate_unit,
        "row_contract": args.row_contract,
        "split_contract": "within-family lexicographic video rank modulo five",
        "sampling_contract": "deterministic complete-track sampling before target materialization",
        "issue_contract": "forecast issued at t before any t+1 observation",
    }
    caps = {
        "train": allocate_caps(table, "train", int(args.max_train_rows)),
        "val": allocate_caps(table, "val", int(args.max_val_rows)),
        "test": allocate_caps(table, "test", int(args.max_test_rows)),
    }
    built: dict[str, list[v84.AnchorBundle]] = {"train": [], "val": [], "test": []}
    audit_rows: list[dict[str, Any]] = []
    offsets = {"train": 0, "val": 7001, "test": 9001}
    for split in ("train", "val", "test"):
        for sequence in split_sequences[split]:
            source = table.loc[table.sequence == sequence].copy()
            bundle = build_video(
                source,
                sequence,
                int(args.max_horizon),
                caps[split][sequence],
                int(args.seed) + sequence * 101 + offsets[split],
                args.row_contract,
            )
            bundle.meta = {**bundle.meta, "contract": contract}
            built[split].append(bundle)
            audit_rows.append(
                {
                    "split": split,
                    "sequence": sequence,
                    "family": bundle.meta["family"],
                    "video": bundle.meta["video"],
                    "rows": len(bundle.rows),
                    "tracks": int(bundle.rows.track_id.nunique()),
                    "source_rows": bundle.meta["source_rows"],
                    "available_tracks": bundle.meta["available_tracks"],
                }
            )
            if split == "train":
                oof_root = args.out_dir / f"oof_seq{sequence:03d}_native"
                v84.save_bundle(oof_root / "test", bundle)
                (oof_root / "contract.json").write_text(
                    json.dumps(finite(contract), indent=2), encoding="utf-8"
                )

    train = concatenate("deepsea_train", built["train"], contract)
    validation = concatenate("deepsea_validation", built["val"], contract)
    test = concatenate("deepsea_test", built["test"], contract)
    final_root = args.out_dir / "final_native"
    v84.save_bundle(final_root / "val", validation)
    v84.save_bundle(final_root / "test", test)
    (final_root / "contract.json").write_text(
        json.dumps(finite(contract), indent=2), encoding="utf-8"
    )

    feature_index = (
        pd.concat([train.rows[KEYS], validation.rows[KEYS], test.rows[KEYS]], ignore_index=True)
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
    )
    feature_path = args.out_dir / "native_feature_index.csv"
    feature_index.to_csv(feature_path, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out_dir / "cache_sampling_audit.csv", index=False)
    status = {
        **contract,
        "train_rows": len(train.rows),
        "val_rows": len(validation.rows),
        "test_rows": len(test.rows),
        "train_tracks": int(train.rows.groupby(["sequence", "track_id"]).ngroups),
        "val_tracks": int(validation.rows.groupby(["sequence", "track_id"]).ngroups),
        "test_tracks": int(test.rows.groupby(["sequence", "track_id"]).ngroups),
        "feature_index": str(feature_path),
        "movie_rows": audit.to_dict("records"),
    }
    (args.out_dir / "native_cache_status.json").write_text(
        json.dumps(finite(status), indent=2), encoding="utf-8"
    )
    print(json.dumps(finite(status), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=ROOT / "outputs/deepsea_multimodal_prepared_v204_2026-07-31/deepsea_tracks.csv",
    )
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coordinate-unit",
        choices=["cell_diameter", "pixel"],
        default="cell_diameter",
    )
    parser.add_argument(
        "--row-contract",
        choices=["h1_complete", "h6_eligible"],
        default="h1_complete",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_online_anchor_cache_v204_2026-07-31",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
