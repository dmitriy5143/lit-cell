#!/usr/bin/env python3
"""Build track-native chronological anchor bundles for v97 transfer guards.

The bundle uses the last observed displacement as a constant-velocity anchor.
It preserves complete contiguous track fragments so online replay never turns a
random row split into a synthetic time series.
"""

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


def parse_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in str(value).split(",") if token.strip()]


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def load_sequence(table_root: Path, dataset: str, sequence: int, max_horizon: int) -> v84.AnchorBundle:
    path = table_root / dataset / f"{dataset}_{sequence:02d}_tracks.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = [
        "dataset",
        "sequence",
        "frame",
        "track_id",
        "x_px",
        "y_px",
        "dx_px",
        "dy_px",
        "target_dx_px",
        "target_dy_px",
        "QUALITY",
    ]
    table = pd.read_csv(path, usecols=lambda column: column in usecols)
    table = table.sort_values(["track_id", "frame"]).reset_index(drop=True)
    row_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    for _track, group in table.groupby("track_id", sort=False):
        group = group.sort_values("frame").reset_index(drop=True)
        frames = group["frame"].to_numpy(np.int64)
        current = group[["dx_px", "dy_px"]].to_numpy(np.float32)
        target = group[["target_dx_px", "target_dy_px"]].to_numpy(np.float32)
        valid_positions: list[int] = []
        targets: list[np.ndarray] = []
        for position in range(1, max(1, len(group) - max_horizon)):
            stop = position + max_horizon
            if stop >= len(group):
                break
            if not np.all(np.diff(frames[position : stop + 1]) == 1):
                continue
            if not np.all(np.isfinite(current[position])) or not np.all(np.isfinite(target[position:stop])):
                continue
            valid_positions.append(position)
            targets.append(target[position:stop])
        if valid_positions:
            row_parts.append(group.iloc[valid_positions].copy())
            target_parts.extend(targets)
    if not row_parts:
        raise RuntimeError(f"No contiguous h{max_horizon} rows in {path}")
    rows = pd.concat(row_parts, ignore_index=True)
    rows["sequence"] = rows["sequence"].astype(int)
    rows["frame"] = rows["frame"].astype(int)
    rows["track_id"] = rows["track_id"].astype(int)
    target_steps = np.asarray(target_parts, dtype=np.float32)
    base = rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    anchor_residual = np.zeros((len(rows), max_horizon * 2), dtype=np.float32)
    return v84.AnchorBundle(
        name=f"{dataset}_{sequence:02d}",
        rows=rows,
        anchor_residual=anchor_residual,
        base=base,
        target_steps=target_steps,
        meta={"dataset": dataset, "sequence": sequence, "anchor_method": "constant_velocity"},
    )


def subset_complete_tracks(bundle: v84.AnchorBundle, max_rows: int, seed: int) -> v84.AnchorBundle:
    if max_rows <= 0 or len(bundle.rows) <= max_rows:
        return bundle
    # Grouping must include sequence and track identity; frame is deliberately excluded.
    groups = [
        np.asarray(list(indices), dtype=np.int64)
        for indices in bundle.rows.groupby(["sequence", "track_id"], sort=False).groups.values()
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    selected: list[np.ndarray] = []
    count = 0
    for group_index in order:
        indices = groups[int(group_index)]
        if selected and count + len(indices) > max_rows:
            continue
        selected.append(indices)
        count += len(indices)
        if count >= max_rows:
            break
    if not selected:
        selected = [groups[int(order[0])]]
    indices = np.sort(np.concatenate(selected))
    return v84.AnchorBundle(
        name=bundle.name,
        rows=bundle.rows.iloc[indices].reset_index(drop=True),
        anchor_residual=bundle.anchor_residual[indices],
        base=bundle.base[indices],
        target_steps=bundle.target_steps[indices],
        meta=dict(bundle.meta),
    )


def concatenate(name: str, bundles: list[v84.AnchorBundle], contract: dict[str, Any]) -> v84.AnchorBundle:
    merged = v84.concat_bundles(name, bundles)
    merged.meta = {**merged.meta, "contract": contract, "anchor_method": "constant_velocity"}
    return merged


def subset_per_sequence(
    bundles: dict[int, v84.AnchorBundle],
    sequences: list[int],
    max_rows_per_sequence: int,
    seed: int,
) -> list[v84.AnchorBundle]:
    output: list[v84.AnchorBundle] = []
    for sequence in sequences:
        output.append(
            subset_complete_tracks(
                bundles[sequence],
                int(max_rows_per_sequence),
                int(seed) + int(sequence) * 101,
            )
        )
    return output


def run(args: argparse.Namespace) -> None:
    train_sequences = parse_ints(args.train_seq)
    validation_sequences = parse_ints(args.val_seq)
    test_sequences = parse_ints(args.test_seq)
    all_sequences = train_sequences + validation_sequences + test_sequences
    if len(set(all_sequences)) != len(all_sequences):
        raise ValueError("Train/validation/test movies must be disjoint")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sequence = {
        sequence: load_sequence(args.table_root, args.dataset, sequence, int(args.max_horizon))
        for sequence in all_sequences
    }
    contract = {
        "dataset": args.dataset,
        "train_seq": train_sequences,
        "val_seq": validation_sequences,
        "test_seq": test_sequences,
        "max_horizon": int(args.max_horizon),
        "anchor_method": "constant_velocity",
        "split_contract": "movie-held-out; complete chronological track fragments",
        "max_val_rows_per_sequence": int(args.max_val_rows_per_sequence),
        "max_test_rows_per_sequence": int(args.max_test_rows_per_sequence),
    }

    train_cap_per_movie = int(args.max_train_rows) // max(len(train_sequences), 1)
    train_parts: list[v84.AnchorBundle] = []
    for sequence in train_sequences:
        part = subset_complete_tracks(
            per_sequence[sequence], train_cap_per_movie, int(args.seed) + sequence * 101
        )
        part.meta = {**part.meta, "contract": contract}
        train_parts.append(part)
        oof_root = args.out_dir / f"oof_seq{sequence:02d}_native"
        v84.save_bundle(oof_root / "test", part)
        (oof_root / "contract.json").write_text(json.dumps(finite(contract), indent=2), encoding="utf-8")

    validation_parts = subset_per_sequence(
        per_sequence,
        validation_sequences,
        int(args.max_val_rows_per_sequence),
        int(args.seed) + 7001,
    )
    test_parts = subset_per_sequence(
        per_sequence,
        test_sequences,
        int(args.max_test_rows_per_sequence),
        int(args.seed) + 9001,
    )
    validation = subset_complete_tracks(
        concatenate("native_validation", validation_parts, contract),
        int(args.max_val_rows),
        int(args.seed) + 7001,
    )
    test = subset_complete_tracks(
        concatenate("native_test", test_parts, contract),
        int(args.max_test_rows),
        int(args.seed) + 9001,
    )
    validation.meta = {**validation.meta, "contract": contract, "anchor_method": "constant_velocity"}
    test.meta = {**test.meta, "contract": contract, "anchor_method": "constant_velocity"}
    final_root = args.out_dir / "final_native"
    v84.save_bundle(final_root / "val", validation)
    v84.save_bundle(final_root / "test", test)
    (final_root / "contract.json").write_text(json.dumps(finite(contract), indent=2), encoding="utf-8")

    train = concatenate("native_train", train_parts, contract)
    feature_index = pd.concat(
        [train.rows[KEYS], validation.rows[KEYS], test.rows[KEYS]], ignore_index=True
    ).drop_duplicates(KEYS)
    feature_path = args.out_dir / "native_feature_index.csv"
    feature_index.to_csv(feature_path, index=False)
    status = {
        **contract,
        "train_rows": len(train.rows),
        "val_rows": len(validation.rows),
        "test_rows": len(test.rows),
        "train_tracks": int(train.rows.groupby(["sequence", "track_id"]).ngroups),
        "val_tracks": int(validation.rows.groupby(["sequence", "track_id"]).ngroups),
        "test_tracks": int(test.rows.groupby(["sequence", "track_id"]).ngroups),
        "feature_index": str(feature_path),
    }
    (args.out_dir / "native_cache_status.json").write_text(
        json.dumps(finite(status), indent=2), encoding="utf-8"
    )
    print(json.dumps(finite(status), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table-root", type=Path, default=ROOT / "new_data/lachance_epithelia/tables")
    parser.add_argument("--train-seq", required=True)
    parser.add_argument("--val-seq", required=True)
    parser.add_argument("--test-seq", required=True)
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=15000)
    parser.add_argument(
        "--max-val-rows-per-sequence",
        type=int,
        default=0,
        help="Optional complete-track cap applied to every validation movie before pooling.",
    )
    parser.add_argument(
        "--max-test-rows-per-sequence",
        type=int,
        default=0,
        help="Optional complete-track cap applied to every test movie before pooling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
