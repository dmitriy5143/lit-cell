#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def parse_trackmate_particles(
    xml_path: Path,
    *,
    dataset: str,
    sequence: str,
    dt_seconds: float,
    pixel_size_um: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root_attributes: dict[str, str] = {}
    track_id = 0

    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start" and not root_attributes:
            root_attributes = dict(elem.attrib)
        if event != "end" or elem.tag != "particle":
            continue
        track_id += 1
        for detection in elem.findall("detection"):
            rows.append(
                {
                    "dataset": dataset,
                    "sequence": sequence,
                    "frame": int(float(detection.attrib["t"])),
                    "track_id": track_id,
                    "x_px": float(detection.attrib["x"]),
                    "y_px": float(detection.attrib["y"]),
                }
            )
        elem.clear()

    if not rows:
        raise ValueError(f"No particle/detection rows found in {xml_path}")

    df = pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)
    df = df.drop_duplicates(["track_id", "frame"], keep="first")
    group = df.groupby("track_id", sort=False)
    prev_frame = group["frame"].shift(1)
    consecutive_prev = prev_frame.notna() & (prev_frame.astype(float) + 1.0 == df["frame"])
    next_frame = group["frame"].shift(-1)
    consecutive_next = next_frame.notna() & (next_frame.astype(float) == df["frame"] + 1.0)

    df["dx_px"] = group["x_px"].diff().where(consecutive_prev)
    df["dy_px"] = group["y_px"].diff().where(consecutive_prev)
    df["vx_px_s"] = df["dx_px"] / float(dt_seconds)
    df["vy_px_s"] = df["dy_px"] / float(dt_seconds)
    df["speed_px_s"] = np.hypot(df["vx_px_s"], df["vy_px_s"])
    df["ax_px_s2"] = group["vx_px_s"].diff().where(consecutive_prev) / float(dt_seconds)
    df["ay_px_s2"] = group["vy_px_s"].diff().where(consecutive_prev) / float(dt_seconds)
    df["target_dx_px"] = (group["x_px"].shift(-1) - df["x_px"]).where(consecutive_next)
    df["target_dy_px"] = (group["y_px"].shift(-1) - df["y_px"]).where(consecutive_next)
    df["has_target"] = consecutive_next
    df["x_um"] = df["x_px"] * float(pixel_size_um)
    df["y_um"] = df["y_px"] * float(pixel_size_um)

    df["FRAME"] = df["frame"].astype(int)
    df["TRACK_ID"] = df["track_id"].astype(int)
    df["POSITION_X"] = df["x_px"]
    df["POSITION_Y"] = df["y_px"]
    df["POSITION_T"] = df["frame"] * float(dt_seconds)
    df["QUALITY"] = 1.0
    df["ID"] = np.arange(len(df), dtype=np.int64)
    df["SEQ_ID"] = sequence
    df["GLOBAL_TRACK_ID"] = sequence + ":" + df["track_id"].astype(str)
    df["SOURCE_XML"] = str(xml_path)

    track_lengths = df.groupby("track_id")["frame"].nunique()
    metadata = {
        "dataset": dataset,
        "sequence": sequence,
        "source_xml": xml_path,
        "root_attributes": root_attributes,
        "dt_seconds": float(dt_seconds),
        "pixel_size_um": float(pixel_size_um),
        "rows": int(len(df)),
        "tracks": int(df["track_id"].nunique()),
        "frames": int(df["frame"].nunique()),
        "frame_min": int(df["frame"].min()),
        "frame_max": int(df["frame"].max()),
        "objects_per_frame_median": float(df.groupby("frame").size().median()),
        "track_length_median": float(track_lengths.median()),
        "valid_next_step_targets": int(df["has_target"].sum()),
        "nonconsecutive_transitions": int((prev_frame.notna() & ~consecutive_prev).sum()),
    }
    return df, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert simple TrackMate particle/detection XML exports to causal track tables."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*_Tracks.xml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dt-seconds", type=float, required=True)
    parser.add_argument("--pixel-size-um", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_paths = sorted(args.input_root.glob(args.glob))
    if not xml_paths:
        raise FileNotFoundError(
            f"No XML files matching {args.glob!r} under {args.input_root}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for xml_path in xml_paths:
        sequence = xml_path.stem.removesuffix("_Tracks")
        safe_sequence = sequence.replace(" ", "_")
        out_path = args.out_dir / f"{args.dataset}_{safe_sequence}_tracks.csv"
        df, metadata = parse_trackmate_particles(
            xml_path,
            dataset=args.dataset,
            sequence=safe_sequence,
            dt_seconds=float(args.dt_seconds),
            pixel_size_um=float(args.pixel_size_um),
        )
        df.to_csv(out_path, index=False)
        metadata["output_table"] = out_path
        manifest.append(metadata)
        print(
            f"{args.dataset}/{safe_sequence}: {len(df)} rows, "
            f"{metadata['tracks']} tracks -> {out_path}",
            flush=True,
        )
    manifest_path = args.out_dir / f"{args.dataset}_trackmate_manifest.json"
    manifest_path.write_text(
        json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
