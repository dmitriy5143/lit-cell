#!/usr/bin/env python3
"""Export a causal source-column dictionary from a raw-context feature grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYS = {"dataset", "sequence", "frame", "track_id"}
COORDINATES = {"x_px", "y_px"}
COMPLETED_MOTION = {"dx_px", "dy_px"}
FAMILIES = {
    "ms_": ("multiscale_image", "multiscale morphology and local intensity"),
    "tf_": ("tissue_flow", "causal tissue-flow and temporal-flow summary"),
    "rc_": ("raw_multicell_context", "non-centred central and neighbour context"),
    "obs_": ("observation_state", "observation quality and tracking-derived state"),
}


def describe(column: str) -> tuple[str, str, str]:
    if column in KEYS:
        return "key", "row identity", "index only; never a learned numeric input"
    if column in COORDINATES:
        return "current_coordinate", "current centroid", "available at issue frame t"
    if column in COMPLETED_MOTION:
        return "completed_motion", "last completed displacement", "available at issue frame t"
    if column == "QUALITY":
        return "tracking_quality", "source tracking quality", "available at issue frame t"
    for prefix, (family, meaning) in FAMILIES.items():
        if column.startswith(prefix):
            return family, meaning, "constructed from frames no later than issue frame t"
    return "other", "unclassified source field", "must be reviewed before model inclusion"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_grid", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.feature_grid.open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_order",
                "column",
                "family",
                "meaning",
                "causal_availability",
                "fold_local_selection",
            ],
        )
        writer.writeheader()
        for index, column in enumerate(columns):
            family, meaning, availability = describe(column)
            writer.writerow(
                {
                    "source_order": index,
                    "column": column,
                    "family": family,
                    "meaning": meaning,
                    "causal_availability": availability,
                    "fold_local_selection": (
                        "identity only"
                        if family == "key"
                        else "eligible; variance filtering and quotas fit on training movies only"
                    ),
                }
            )


if __name__ == "__main__":
    main()
