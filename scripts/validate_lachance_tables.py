#!/usr/bin/env python3
"""Validate the minimal LaChance trajectory-table contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CELL_TYPES = ("MDCK_Bulk", "MDCK_Edge", "MDAMB231", "HUVEC")


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    table = pd.read_csv(path)
    frame_col = "FRAME" if "FRAME" in table.columns else "frame"
    track_col = "TRACK_ID" if "TRACK_ID" in table.columns else "track_id"
    required = {frame_col, track_col, "x_px", "y_px"}
    missing = sorted(required - set(table.columns))
    if missing:
        return [f"missing columns {missing}"]

    values = table[[frame_col, track_col, "x_px", "y_px"]]
    if values.isna().any().any():
        errors.append("required columns contain missing values")
    if table.duplicated([track_col, frame_col]).any():
        errors.append("duplicate (track, frame) rows")
    if len(table) == 0:
        errors.append("empty table")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-root", type=Path, required=True)
    args = parser.parse_args()

    failures = 0
    files_seen = 0
    for cell_type in CELL_TYPES:
        files = sorted((args.table_root / cell_type).glob("*_tracks.csv"))
        if not files:
            print(f"[WARN] {cell_type}: no *_tracks.csv files")
            continue
        for path in files:
            files_seen += 1
            errors = validate_file(path)
            if errors:
                failures += 1
                print(f"[FAIL] {path}: {'; '.join(errors)}")
        print(f"[OK] {cell_type}: checked {len(files)} files")

    if files_seen == 0:
        raise SystemExit("No trajectory tables found")
    if failures:
        raise SystemExit(f"{failures} table(s) failed validation")
    print(f"[OK] validated {files_seen} trajectory tables")


if __name__ == "__main__":
    main()
