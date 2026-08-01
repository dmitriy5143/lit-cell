#!/usr/bin/env python3
"""Resume the required DeepSea files from a frozen gdown JSON manifest."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
PRINT_LOCK = threading.Lock()


def parse_families(value: str) -> set[str]:
    return {token.strip() for token in str(value).split(",") if token.strip()}


def folder_family(path: str) -> str:
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else ""


def usercontent_url(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    file_ids = query.get("id", [])
    if len(file_ids) != 1:
        raise ValueError(f"Cannot recover Google Drive file id from {url}")
    return "https://drive.usercontent.google.com/download?" + urlencode(
        {"id": file_ids[0], "export": "download", "confirm": "t"}
    )


def stream_download(url: str, destination: Path) -> int:
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with requests.get(
        usercontent_url(url),
        stream=True,
        timeout=(20, 180),
        headers={"User-Agent": "Mozilla/5.0 DeepSea-v204-research-download"},
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError("Google Drive returned HTML instead of the requested file")
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty")
    temporary.replace(destination)
    return destination.stat().st_size


def download_one(
    item: dict[str, str],
    output_root: Path,
    retries: int,
    base_delay: float,
) -> dict[str, Any]:
    destination = output_root / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "path": item["path"],
            "status": "existing",
            "bytes": destination.stat().st_size,
            "attempts": 0,
            "error": "",
        }
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            downloaded_bytes = stream_download(item["url"], destination)
            if downloaded_bytes > 0:
                return {
                    "path": item["path"],
                    "status": "downloaded",
                    "bytes": downloaded_bytes,
                    "attempts": attempt,
                    "error": "",
                }
            last_error = "stream download returned no completed file"
        except Exception as error:  # acquisition audit must continue past individual failures
            last_error = repr(error)
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink()
        time.sleep(base_delay * attempt)
    return {
        "path": item["path"],
        "status": "failed",
        "bytes": destination.stat().st_size if destination.exists() else 0,
        "attempts": retries,
        "error": last_error,
    }


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    families = parse_families(args.folders)
    selected = [item for item in manifest if folder_family(item["path"]) in families]
    if args.limit > 0:
        selected = selected[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                item,
                args.output_root,
                args.retries,
                args.retry_delay,
            ): item
            for item in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % args.report_every == 0 or result["status"] == "failed":
                with PRINT_LOCK:
                    failed = sum(row["status"] == "failed" for row in results)
                    downloaded = sum(row["status"] == "downloaded" for row in results)
                    existing = sum(row["status"] == "existing" for row in results)
                    print(
                        f"[{completed}/{len(selected)}] downloaded={downloaded} "
                        f"existing={existing} failed={failed}",
                        flush=True,
                    )
    audit = pd.DataFrame(results).sort_values("path")
    audit.to_csv(args.audit, index=False)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "output_root": str(args.output_root.resolve()),
        "folders": sorted(families),
        "selected": len(selected),
        "downloaded": int((audit.status == "downloaded").sum()),
        "existing": int((audit.status == "existing").sum()),
        "failed": int((audit.status == "failed").sum()),
        "bytes": int(audit.bytes.sum()),
        "elapsed_hours": (time.time() - started) / 3600.0,
        "complete": bool((audit.status != "failed").all()),
    }
    args.audit.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "tmp/deepsea_v204_drive_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "new_data/deepsea_v204",
    )
    parser.add_argument("--folders", default="cell_features,cell_images,cell_masks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=1.5)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "outputs/deepsea_v204_acquisition_audit_2026-07-31.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
