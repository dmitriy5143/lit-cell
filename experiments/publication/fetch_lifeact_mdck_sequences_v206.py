#!/usr/bin/env python3
"""Selectively fetch aligned LifeAct/phase sequences from Zenodo record 20047603."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "new_data" / "lifeact_mdck_mechanochemical_v206" / "sequences"

SOURCES = {
    "mitomycin": {
        "url": "https://zenodo.org/api/records/20047603/files/Drugs_MSD_Qt.zip/content",
        "pattern": re.compile(r"^Drugs_MSD_Qt/Mitomycin/s013t(\d{3})c([12])_ORG\.tif$"),
    },
    "y27632": {
        "url": "https://zenodo.org/api/records/20047603/files/Drugs_MSD_Qt.zip/content",
        "pattern": re.compile(r"^Drugs_MSD_Qt/Y27632/s1/s001t(\d{3})c([12])_ORG\.tif$"),
    },
    "lisa": {
        "url": "https://zenodo.org/api/records/20047603/files/LISA.zip/content",
        "pattern": re.compile(
            r"^LISA/LISA_timelapse/Images/LA MDCK Timelapse 49hrs_s02t(\d{2})z1c([12])_ORG\.tif$"
        ),
    },
}


def parse_frames(spec: str) -> set[int] | None:
    spec = spec.strip().lower()
    if spec == "all":
        return None
    values: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sequences", default="mitomycin,y27632,lisa")
    parser.add_argument("--frames", default="all")
    parser.add_argument("--channels", default="1,2")
    args = parser.parse_args()
    requested = [item.strip().lower() for item in args.sequences.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(SOURCES))
    if unknown:
        raise ValueError(f"Unknown sequences: {unknown}")
    frames = parse_frames(args.frames)
    channels = {int(item) for item in args.channels.split(",") if item.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    archives: dict[str, RemoteZip] = {}
    try:
        for sequence in requested:
            source = SOURCES[sequence]
            url = str(source["url"])
            if url not in archives:
                archives[url] = RemoteZip(url)
            archive = archives[url]
            pattern = source["pattern"]
            selected: list[tuple[object, int, int]] = []
            for info in archive.infolist():
                match = pattern.match(info.filename)
                if match is None:
                    continue
                frame, channel = int(match.group(1)), int(match.group(2))
                if frames is not None and frame not in frames:
                    continue
                if channel not in channels:
                    continue
                selected.append((info, frame, channel))
            if not selected:
                raise RuntimeError(f"No entries selected for {sequence}")
            sequence_dir = args.out_dir / sequence
            sequence_dir.mkdir(parents=True, exist_ok=True)
            for index, (info, frame, channel) in enumerate(sorted(selected, key=lambda x: (x[1], x[2])), 1):
                destination = sequence_dir / f"frame_{frame:03d}_c{channel}.tif"
                reused = destination.exists() and destination.stat().st_size == info.file_size
                if not reused:
                    destination.write_bytes(archive.read(info.filename))
                rows.append(
                    {
                        "sequence": sequence,
                        "frame": frame,
                        "channel": channel,
                        "remote_name": info.filename,
                        "destination": str(destination.resolve()),
                        "bytes": info.file_size,
                        "reused": reused,
                    }
                )
                if index % 20 == 0 or index == len(selected):
                    print(f"[v206] {sequence}: {index}/{len(selected)}", flush=True)
    finally:
        for archive in archives.values():
            archive.close()

    manifest = pd.DataFrame(rows).sort_values(["sequence", "frame", "channel"])
    manifest.to_csv(args.out_dir / "v206_sequence_download_manifest.csv", index=False)
    print(args.out_dir)


if __name__ == "__main__":
    main()
