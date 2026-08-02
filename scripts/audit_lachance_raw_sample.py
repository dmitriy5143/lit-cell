#!/usr/bin/env python3
"""Zenodo raw timelapse audit for the LaChance epithelial dataset.

Default mode is metadata-only and non-destructive.  Use ``--download`` to fetch
the selected ZIP for image/morphology feature development.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "lachance_raw_sample_audit_2026-06-14"
DEFAULT_RAW_DIR = ROOT / "new_data" / "lachance_epithelia" / "raw_timelapse"
ZENODO_API = "https://zenodo.org/api/records/4959169"


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def _open_url(url: str, timeout: int, *, allow_insecure_ssl: bool = False):
    context = ssl._create_unverified_context() if allow_insecure_ssl else None
    return urllib.request.urlopen(url, timeout=timeout, context=context)


def _is_ssl_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError):
        return True
    return False


def fetch_record(url: str, *, allow_insecure_ssl: bool = False) -> dict[str, Any]:
    try:
        with _open_url(url, 60, allow_insecure_ssl=allow_insecure_ssl) as response:
            return json.loads(response.read().decode("utf-8"))
    except (ssl.SSLError, urllib.error.URLError) as exc:
        if allow_insecure_ssl or not _is_ssl_error(exc):
            raise
        print("SSL verification failed; retrying Zenodo metadata fetch with an unverified SSL context.", file=sys.stderr)
        with _open_url(url, 60, allow_insecure_ssl=True) as response:
            return json.loads(response.read().decode("utf-8"))


def download_file(url: str, target: Path, *, allow_insecure_ssl: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        response_ctx = _open_url(url, 120, allow_insecure_ssl=allow_insecure_ssl)
    except (ssl.SSLError, urllib.error.URLError) as exc:
        if allow_insecure_ssl or not _is_ssl_error(exc):
            raise
        print("SSL verification failed; retrying raw ZIP download with an unverified SSL context.", file=sys.stderr)
        response_ctx = _open_url(url, 120, allow_insecure_ssl=True)
    with response_ctx as response, tmp.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(target)


def file_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in record.get("files", []):
        links = item.get("links", {})
        rows.append(
            {
                "key": item.get("key", ""),
                "size_bytes": int(item.get("size", 0)),
                "checksum": item.get("checksum", ""),
                "download": links.get("self") or links.get("download") or "",
            }
        )
    return rows


def inspect_zip(path: Path, max_members: int) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if not zipfile.is_zipfile(path):
        return {"exists": True, "is_zip": False}
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos[: int(max_members)]]
        image_like = [
            info.filename
            for info in infos
            if info.filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".avi", ".mov"))
        ]
        return {
            "exists": True,
            "is_zip": True,
            "members": len(infos),
            "image_like_members": len(image_like),
            "preview": names,
            "image_like_preview": image_like[: int(max_members)],
        }


def write_report(out_dir: Path, selected: dict[str, Any], zip_info: dict[str, Any], table_root: Path) -> None:
    local_tables = sorted(table_root.glob("MDCK_Bulk/*_tracks.csv"))
    lines = [
        "# LaChance raw timelapse sample audit",
        "",
        "Purpose: prepare the optional image/morphology branch without blocking the track-context work.",
        "",
        "## Selected Zenodo File",
        "",
        "```json",
        json.dumps(finite_json(selected), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Local ZIP Inspection",
        "",
        "```json",
        json.dumps(finite_json(zip_info), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Local Track Tables",
        "",
        f"- MDCK_Bulk track tables found: `{len(local_tables)}`",
        f"- table root: `{table_root}`",
        "",
        "## Next Manual Gate",
        "",
        "- If the ZIP is present and contains TIFF/AVI timelapses, map movie IDs to XML/table sequence IDs.",
        "- Then add a small frame/coordinate overlay check before extracting image/morphology features.",
        "- Do not use image features in model training until this alignment gate passes.",
    ]
    (out_dir / "raw_sample_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-url", default=ZENODO_API)
    parser.add_argument("--file-key", default="MDCK_Bulk_Timelapse_Data_Sample_Tissues.zip")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Use an unverified SSL context immediately. By default this is only used as a fallback after SSL verification fails.",
    )
    parser.add_argument("--max-preview-members", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    record = fetch_record(args.record_url, allow_insecure_ssl=bool(args.allow_insecure_ssl))
    rows = file_rows(record)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out_dir / "zenodo_file_manifest.csv", index=False)
    selected_rows = [row for row in rows if row["key"] == args.file_key]
    if not selected_rows:
        raise FileNotFoundError(f"{args.file_key} not found in Zenodo record")
    selected = selected_rows[0]
    target = args.raw_dir / args.file_key
    if args.download and not target.exists():
        print(f"downloading {selected['download']} -> {target}", flush=True)
        download_file(str(selected["download"]), target, allow_insecure_ssl=bool(args.allow_insecure_ssl))
    zip_info = inspect_zip(target, args.max_preview_members)
    (args.out_dir / "raw_sample_manifest.json").write_text(
        json.dumps(
            finite_json(
                {
                    "record_url": args.record_url,
                    "selected": selected,
                    "local_target": target,
                    "download_requested": bool(args.download),
                    "zip_info": zip_info,
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(args.out_dir, selected, zip_info, args.table_root)
    print(args.out_dir / "raw_sample_status_report.md", flush=True)


if __name__ == "__main__":
    main()
