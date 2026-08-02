#!/usr/bin/env python3
"""Prepare paired C2C12 manual/automatic tracks for the v97 online protocol.

The Scientific Data archive stores one XML annotation per movie.  Manual
annotations explicitly flag interpolated centroids with ``f=\"I\"``; computer
annotations contain dense tracker output.  This converter preserves that
provenance, splits discontinuous identities into contiguous track fragments,
and writes the same table contract used by the native LaChance cache builder.

By default experiment 1 is train, experiment 2 is validation, and experiment
3 is test.  Therefore no movie, treatment replicate, or track crosses splits.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "new_data/C2C12_tracking"
DEFAULT_TABLE_ROOT = ROOT / "new_data/c2c12_online/tables"


@dataclass(frozen=True)
class Point:
    frame: int
    x: float
    y: float
    state: int
    interpolated: bool


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    return value


def experiment_id(path: Path) -> int:
    match = re.search(r"exp(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer experiment from {path}")
    return int(match.group(1))


def field_id(path: Path) -> int:
    match = re.search(r"F(\d{4})", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer field from {path}")
    return int(match.group(1))


def annotation_files(source: Path, kind: str) -> list[Path]:
    files = sorted(source.glob(f"exp*/{kind}/*F???? Data.xml"))
    grouped: dict[tuple[int, int], list[Path]] = {}
    for path in files:
        grouped.setdefault((experiment_id(path), field_id(path)), []).append(path)
    selected: list[Path] = []
    for key, candidates in sorted(grouped.items()):
        if kind == "human":
            full = [path for path in candidates if "full annotation" in path.name.lower()]
            selected.append(full[0] if full else candidates[0])
        else:
            selected.append(candidates[0])
    return selected


def contiguous_parts(points: Iterable[Point]) -> list[list[Point]]:
    ordered = sorted(points, key=lambda point: point.frame)
    deduplicated: list[Point] = []
    for point in ordered:
        if deduplicated and point.frame == deduplicated[-1].frame:
            deduplicated[-1] = point
        else:
            deduplicated.append(point)
    parts: list[list[Point]] = []
    current: list[Point] = []
    for point in deduplicated:
        if current and point.frame != current[-1].frame + 1:
            if len(current) >= 3:
                parts.append(current)
            current = []
        current.append(point)
    if len(current) >= 3:
        parts.append(current)
    return parts


def parse_annotation(path: Path) -> list[tuple[int, int, list[Point]]]:
    """Return ``(annotation_id, segment_index, points)`` track fragments."""

    tracks: list[tuple[int, int, list[Point]]] = []
    fallback_id = 0
    for _event, element in ET.iterparse(path, events=("end",)):
        if local_name(element.tag) != "a":
            continue
        fallback_id += 1
        try:
            annotation_id = int(element.attrib.get("id", fallback_id))
        except ValueError:
            annotation_id = fallback_id
        points: list[Point] = []
        # Only immediate ss children belong to this annotation.  Descendant a
        # elements represent daughter tracks and are processed independently.
        for sequence in element:
            if local_name(sequence.tag) != "ss":
                continue
            for sample in sequence:
                if local_name(sample.tag) != "s":
                    continue
                try:
                    points.append(
                        Point(
                            frame=int(sample.attrib["i"]),
                            x=float(sample.attrib["x"]),
                            y=float(sample.attrib["y"]),
                            state=int(float(sample.attrib.get("s", 0))),
                            interpolated=str(sample.attrib.get("f", "")).upper() == "I",
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        for segment_index, part in enumerate(contiguous_parts(points)):
            tracks.append((annotation_id, segment_index, part))
        element.clear()
    return tracks


def table_from_xml(path: Path, dataset: str, annotation_kind: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    experiment = experiment_id(path)
    field = field_id(path)
    sequence = experiment * 100 + field
    records: list[dict[str, Any]] = []
    track_count = 0
    interpolated_count = 0
    point_count = 0
    for annotation_id, segment_index, points in parse_annotation(path):
        # Keep room for multiple discontinuous fragments of the same identity.
        track_id = int(annotation_id) * 1000 + int(segment_index)
        track_count += 1
        point_count += len(points)
        interpolated_count += sum(point.interpolated for point in points)
        for index in range(1, len(points) - 1):
            previous, current, following = points[index - 1], points[index], points[index + 1]
            records.append(
                {
                    "dataset": dataset,
                    "sequence": sequence,
                    "experiment": experiment,
                    "field": field,
                    "frame": current.frame,
                    "track_id": track_id,
                    "source_annotation_id": annotation_id,
                    "segment_index": segment_index,
                    "x_px": current.x,
                    "y_px": current.y,
                    "dx_px": current.x - previous.x,
                    "dy_px": current.y - previous.y,
                    "target_dx_px": following.x - current.x,
                    "target_dy_px": following.y - current.y,
                    "QUALITY": 1.0,
                    "annotation_kind": annotation_kind,
                    "state": current.state,
                    "previous_interpolated": int(previous.interpolated),
                    "current_interpolated": int(current.interpolated),
                    "target_interpolated": int(following.interpolated),
                    "transition_has_interpolation": int(current.interpolated or following.interpolated),
                }
            )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        raise RuntimeError(f"No usable tracks parsed from {path}")
    table = table.sort_values(["track_id", "frame"]).reset_index(drop=True)
    diagnostics = {
        "dataset": dataset,
        "annotation_kind": annotation_kind,
        "source": str(path),
        "experiment": experiment,
        "field": field,
        "sequence": sequence,
        "tracks": track_count,
        "points": point_count,
        "rows": len(table),
        "interpolated_points": interpolated_count,
        "interpolated_fraction": interpolated_count / max(point_count, 1),
        "median_speed_px": float(np.median(np.hypot(table.dx_px, table.dy_px))),
        "p90_speed_px": float(np.quantile(np.hypot(table.dx_px, table.dy_px), 0.9)),
        "p99_speed_px": float(np.quantile(np.hypot(table.dx_px, table.dy_px), 0.99)),
    }
    return table, diagnostics


def write_kind(source: Path, table_root: Path, kind: str, force: bool) -> list[dict[str, Any]]:
    annotation_kind = "manual" if kind == "human" else "automatic"
    dataset = f"C2C12_{annotation_kind.capitalize()}"
    out_root = table_root / dataset
    out_root.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, Any]] = []
    files = annotation_files(source, kind)
    if len(files) != 48:
        raise RuntimeError(f"Expected 48 {kind} annotations, found {len(files)}")
    for path in files:
        sequence = experiment_id(path) * 100 + field_id(path)
        output = out_root / f"{dataset}_{sequence:02d}_tracks.csv"
        if output.exists() and not force:
            table = pd.read_csv(output)
            speed = np.hypot(table.dx_px, table.dy_px)
            diagnostics.append(
                {
                    "dataset": dataset,
                    "annotation_kind": annotation_kind,
                    "source": str(path),
                    "experiment": experiment_id(path),
                    "field": field_id(path),
                    "sequence": sequence,
                    "tracks": int(table.track_id.nunique()),
                    "points": int(len(table) + 2 * table.track_id.nunique()),
                    "rows": len(table),
                    "interpolated_points": int(table.current_interpolated.sum()),
                    "interpolated_fraction": float(table.current_interpolated.mean()),
                    "median_speed_px": float(np.median(speed)),
                    "p90_speed_px": float(np.quantile(speed, 0.9)),
                    "p99_speed_px": float(np.quantile(speed, 0.99)),
                    "cached": True,
                }
            )
            continue
        table, row = table_from_xml(path, dataset, annotation_kind)
        table.to_csv(output, index=False)
        diagnostics.append(row)
        print(f"{dataset} sequence={sequence}: rows={len(table)} tracks={table.track_id.nunique()}")
    return diagnostics


def run(args: argparse.Namespace) -> None:
    args.table_root.mkdir(parents=True, exist_ok=True)
    rows = write_kind(args.source, args.table_root, "human", bool(args.force))
    rows.extend(write_kind(args.source, args.table_root, "automatic", bool(args.force)))
    diagnostics = pd.DataFrame(rows).sort_values(["annotation_kind", "sequence"])
    diagnostics.to_csv(args.table_root.parent / "c2c12_annotation_diagnostics.csv", index=False)
    split_contract = {
        "train_sequences": sorted(diagnostics.loc[diagnostics.experiment == 1, "sequence"].unique().tolist()),
        "validation_sequences": sorted(diagnostics.loc[diagnostics.experiment == 2, "sequence"].unique().tolist()),
        "test_sequences": sorted(diagnostics.loc[diagnostics.experiment == 3, "sequence"].unique().tolist()),
        "split_policy": "independent microscopy experiments: exp1 / exp2 / exp3",
        "manual_f0009_policy": "use fully annotated 780-frame XML instead of partial duplicate",
        "source": "https://osf.io/ysaq2/",
    }
    (args.table_root.parent / "c2c12_split_contract.json").write_text(
        json.dumps(finite(split_contract), indent=2), encoding="utf-8"
    )
    aggregate = (
        diagnostics.groupby("annotation_kind", as_index=False)
        .agg(
            movies=("sequence", "nunique"),
            tracks=("tracks", "sum"),
            rows=("rows", "sum"),
            interpolated_fraction=("interpolated_fraction", "mean"),
            median_movie_speed_px=("median_speed_px", "median"),
            p90_movie_speed_px=("p90_speed_px", "median"),
            p99_movie_speed_px=("p99_speed_px", "median"),
        )
    )
    aggregate.to_csv(args.table_root.parent / "c2c12_annotation_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
