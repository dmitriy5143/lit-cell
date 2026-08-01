#!/usr/bin/env python3
"""Audit the LifeAct-MDCK mechanochemical source before model integration.

The audit deliberately separates three questions that are easy to conflate:

1. Does an archive contain a time-resolved cell-state channel?
2. Is that channel aligned with an image suitable for cell segmentation?
3. Are masks, identities, or force measurements already aligned to the same movie?

No forecasting result is produced here.  The outputs define the data contract for
the later causal unary-state gate and prevent static traction assays from being
mistaken for frame-wise force labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lifeact_mdck_mechanochemical_v206_audit_2026-08-01"
RECORD_ID = "20047603"
ZENODO_API = f"https://zenodo.org/api/records/{RECORD_ID}"


def fetch_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def md5_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the Zenodo checksum.
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def remote_names(url: str) -> tuple[list[Any], str]:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - explicit environment diagnostic.
        return [], f"remotezip unavailable: {exc}"
    try:
        with RemoteZip(url) as archive:
            return archive.infolist(), "ok"
    except Exception as exc:  # pragma: no cover - depends on remote ZIP integrity.
        return [], f"{type(exc).__name__}: {exc}"


def classify_name(name: str) -> dict[str, bool]:
    lower = name.lower()
    suffix = Path(lower).suffix
    return {
        "is_tiff": suffix in {".tif", ".tiff"},
        "is_image": suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"},
        "is_mask": any(token in lower for token in ("mask", "label", "segmentation")),
        "is_track": any(token in lower for token in ("track", "trajectory")),
        "is_table": suffix in {".csv", ".tsv", ".xlsx", ".xls"},
        "is_xml": suffix == ".xml",
    }


FRAME_RE = re.compile(
    r"(?P<prefix>.*?)(?:_s|/s)(?P<scene>\d+)t(?P<frame>\d+)"
    r"(?:z(?P<z>\d+))?c(?P<channel>\d+)",
    flags=re.IGNORECASE,
)


def sequence_rows(archive_name: str, infos: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "frames": set(),
            "channels": set(),
            "z_planes": set(),
            "bytes": 0,
            "examples": [],
        }
    )
    for info in infos:
        name = info.filename
        match = FRAME_RE.search(name)
        if not match:
            continue
        prefix = re.sub(r"[/\\]+", "/", match.group("prefix")).rstrip("_/")
        scene = match.group("scene")
        key = (archive_name, prefix, scene)
        record = grouped[key]
        record["frames"].add(int(match.group("frame")))
        record["channels"].add(int(match.group("channel")))
        if match.group("z") is not None:
            record["z_planes"].add(int(match.group("z")))
        record["bytes"] += int(info.file_size)
        if len(record["examples"]) < 2:
            record["examples"].append(name)

    rows: list[dict[str, Any]] = []
    for (archive, prefix, scene), record in grouped.items():
        channels = sorted(record["channels"])
        frames = sorted(record["frames"])
        z_planes = sorted(record["z_planes"])
        rows.append(
            {
                "archive": archive,
                "sequence_prefix": prefix,
                "scene": scene,
                "n_frames": len(frames),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "channels": ",".join(map(str, channels)),
                "n_channels": len(channels),
                "z_planes": ",".join(map(str, z_planes)) if z_planes else "",
                "n_z_planes": max(1, len(z_planes)),
                "paired_c1_c2": int(1 in channels and 2 in channels),
                "uncompressed_gib": record["bytes"] / (1024**3),
                "example": " | ".join(record["examples"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=ROOT / "new_data" / "lifeact_mdck_mechanochemical_v206",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_json(ZENODO_API)
    archive_rows: list[dict[str, Any]] = []
    all_sequence_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for file_meta in metadata.get("files", []):
        name = str(file_meta["key"])
        url = str(file_meta["links"]["self"])
        infos, status = remote_names(url)
        counts = Counter()
        for info in infos:
            for key, value in classify_name(info.filename).items():
                counts[key] += int(value)
        local_path = args.download_dir / name
        expected_checksum = str(file_meta.get("checksum", "")).removeprefix("md5:")
        local_checksum = md5_file(local_path) if local_path.exists() else ""
        checksum_ok: bool | None = None
        if local_checksum and expected_checksum:
            checksum_ok = local_checksum == expected_checksum
        archive_rows.append(
            {
                "archive": name,
                "size_gib": int(file_meta.get("size", 0)) / (1024**3),
                "remote_zip_status": status,
                "n_entries": len(infos),
                "n_tiff": counts["is_tiff"],
                "n_images": counts["is_image"],
                "n_mask_like": counts["is_mask"],
                "n_track_like": counts["is_track"],
                "n_tables": counts["is_table"],
                "n_xml": counts["is_xml"],
                "local_download": local_path.exists(),
                "checksum_ok": checksum_ok,
                "url": url,
            }
        )
        if status != "ok":
            errors.append(f"{name}: {status}")
        all_sequence_rows.extend(sequence_rows(name, infos))

    archives = pd.DataFrame(archive_rows).sort_values("archive")
    sequences = pd.DataFrame(all_sequence_rows)
    if not sequences.empty:
        sequences = sequences.sort_values(
            ["paired_c1_c2", "n_frames", "archive"], ascending=[False, False, True]
        )

    readiness = pd.DataFrame(
        [
            {
                "candidate": "LISA timelapse s02",
                "causal_lifeact_state": True,
                "aligned_segmentation_channel": True,
                "ready_masks": False,
                "ready_tracks": False,
                "framewise_force": False,
                "recommended_role": "unary-state pilot after segmentation and tracking",
            },
            {
                "candidate": "Mitomycin s013",
                "causal_lifeact_state": True,
                "aligned_segmentation_channel": True,
                "ready_masks": False,
                "ready_tracks": False,
                "framewise_force": False,
                "recommended_role": "independent-condition validation",
            },
            {
                "candidate": "Y27632 s001",
                "causal_lifeact_state": True,
                "aligned_segmentation_channel": True,
                "ready_masks": False,
                "ready_tracks": False,
                "framewise_force": False,
                "recommended_role": "independent-condition validation",
            },
            {
                "candidate": "Traction archive",
                "causal_lifeact_state": False,
                "aligned_segmentation_channel": False,
                "ready_masks": False,
                "ready_tracks": False,
                "framewise_force": False,
                "recommended_role": "mechanistic interpretation only; not a per-step input",
            },
        ]
    )

    archives.to_csv(args.out_dir / "v206_archive_manifest.csv", index=False)
    sequences.to_csv(args.out_dir / "v206_sequence_inventory.csv", index=False)
    readiness.to_csv(args.out_dir / "v206_modality_readiness.csv", index=False)
    (args.out_dir / "v206_source_contract.json").write_text(
        json.dumps(
            {
                "record_id": RECORD_ID,
                "title": metadata.get("metadata", {}).get("title"),
                "license": metadata.get("metadata", {}).get("license", {}).get("id"),
                "publication_date": metadata.get("metadata", {}).get("publication_date"),
                "remote_zip_errors": errors,
                "critical_contract": {
                    "lifeact_and_phase_same_cells": True,
                    "masks_supplied": False,
                    "tracks_supplied": False,
                    "framewise_force_aligned_to_movies": False,
                    "forecasting_gate_requires_new_segmentation_and_tracking": True,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    paired = sequences[sequences.paired_c1_c2.eq(1)] if not sequences.empty else sequences
    report = [
        "# LifeAct-MDCK mechanochemical modality audit v206",
        "",
        "## Decision",
        "",
        "The source contains causal LifeAct state aligned to phase-contrast images for the same cells, but it does not provide ready instance masks or trajectories. The traction archive is a separate pre/post assay and cannot be used as frame-wise force supervision for these trajectories.",
        "",
        "Therefore the defensible next experiment is a unary-state gate: reliable segmentation and identity tracking first, then LifeAct/shape/contact variables and hard temporal/wrong-cell controls. Direct force-conditioned transfer is not supported by the data contract.",
        "",
        "## Archive inventory",
        "",
        archives.to_markdown(index=False),
        "",
        "## Paired temporal candidates",
        "",
        paired.head(20).to_markdown(index=False) if not paired.empty else "No paired sequences parsed.",
        "",
        "## Readiness",
        "",
        readiness.to_markdown(index=False),
        "",
        "## Integrity notes",
        "",
        *([f"- {item}" for item in errors] if errors else ["- All remote ZIP central directories were readable."]),
        "",
    ]
    (args.out_dir / "v206_decision_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
