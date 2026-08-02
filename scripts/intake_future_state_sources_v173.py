#!/usr/bin/env python3
"""Manifest-first intake for the post-v166 observability program.

This runner deliberately separates source discovery, causal/identity contracts,
and byte transfer.  A source is not promoted to a cell-level transfer dataset
merely because it contains microscopy: persistent identity, issue-time timing,
and a deployable channel must all be demonstrated first.

The default invocation is metadata-only.  S-BIAD365 files can then be fetched
from the frozen manifest with ``--download-sbiad``; downloads are resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "future_state_source_intake_v173_2026-07-28"
DEFAULT_DATA = ROOT / "new_data" / "future_state_sources_v173"

SBIAD_ACC = "S-BIAD365"
SBIAD_API = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{SBIAD_ACC}"
SBIAD_INFO = f"{SBIAD_API}/info"
SBIAD_ROOT = (
    "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/365/S-BIAD365/Files"
)
SBIAD_TSV = f"{SBIAD_ROOT}/File_list_Vaidziulyte2022_full.tsv"

ALLEN_BUCKET = "https://allencell.s3.amazonaws.com"
ALLEN_PREFIX = "aics/emt_timelapse_dataset"
ALLEN_SMALL_OBJECTS = [
    f"{ALLEN_PREFIX}/README.md",
    f"{ALLEN_PREFIX}/manifests/imaging_and_segmentation_data.csv",
    (
        f"{ALLEN_PREFIX}/manifests/"
        "Imaging_and_segmentation_data_column_description.csv"
    ),
    (
        f"{ALLEN_PREFIX}/manifests/"
        "Migration_timing_through_mesh_extracted_features.csv"
    ),
    (
        f"{ALLEN_PREFIX}/manifests/"
        "Migration_timing_through_mesh_extracted_features_column_description.csv"
    ),
    (
        f"{ALLEN_PREFIX}/manifests/"
        "Image_analysis_extracted_features_column_description.csv"
    ),
]
ALLEN_LARGE_FEATURE_KEY = (
    f"{ALLEN_PREFIX}/manifests/Image_analysis_extracted_features.csv"
)

GIGA_SOURCE_MANIFEST = (
    ROOT
    / "outputs"
    / "gigascience_wound_healing_intake_v167_2026-07-28"
    / "gigascience_source_file_manifest.csv"
)

SBIAD_PRIMARY_EXPERIMENT = "freely control"
SBIAD_PILOT_GROUPS = {
    # Frozen before any prediction experiment: four acquisition-day controls
    # and three optogenetic perturbation movies.
    "c012_pos2",
    "c013_pos3",
    "c014_pos12",
    "c015_pos5",
    "c027_pos1",
    "c028_pos5",
    "c029_pos2",
}
SBIAD_OPTO_UPPER_BOUND_GROUPS = {
    # Frozen before the Cdc42 target experiment.  This is the lexicographically
    # first movie from each of the first three acquisition days available for
    # each intervention family in the public angleOptoFree manifest.
    "c027_pos1",
    "c028_pos5",
    "c029_pos2",
    "c201_pos2",
    "c202_pos1",
    "c203_pos1",
    "c050_pos4",
    "c051_pos2",
    "c052_pos3",
    "c060_pos1",
    "c061_pos1",
    "c062_pos1",
}

USER_AGENT = "Airi-future-state-intake-v173/1.0"
CHUNK_SIZE = 8 << 20


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(finite(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def http_get(url: str, timeout: int, *, headers: dict[str, str] | None = None) -> requests.Response:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    response = requests.get(url, timeout=(timeout, max(timeout, 120)), headers=merged)
    response.raise_for_status()
    return response


def object_url(key: str) -> str:
    return f"{ALLEN_BUCKET}/{quote(key, safe='/')}"


def sbiad_group_name(path: str) -> str:
    name = Path(path).name
    suffixes = (
        "_Positions.mat",
        "_wTIRF_DIC.tif",
        "_wTIRF_405.tif",
        "_wTIRF_488.tif",
        "_wTIRF_561.tif",
        "_wTIRF_642.tif",
        "_wsola_GFP_Sola_bright.tif",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.rsplit(".", 1)[0]


def acquisition_day(group: str) -> str:
    match = re.match(r"(c\d+)_", group)
    return match.group(1) if match else "unknown"


def sbiad_channel(row: pd.Series) -> str:
    path = str(row["Files"])
    if path.endswith("_Positions.mat"):
        return "stage_position"
    if path.endswith("_wTIRF_DIC.tif"):
        return "dic"
    if path.endswith("_wTIRF_405.tif"):
        return "nucleus_405"
    if path.endswith("_wTIRF_488.tif"):
        return "rab6_488"
    if path.endswith("_wTIRF_561.tif"):
        return "opto_561"
    if path.endswith("_wTIRF_642.tif"):
        label = str(row.get("Label", "")).lower()
        return "membrane_642" if "membrane" in label else "rab6_642"
    if "_wsola_" in path:
        return "opto_activation"
    return "other"


def select_sbiad_groups(table: pd.DataFrame, scope: str) -> set[str]:
    groups = set(table["group"].astype(str))
    if scope == "none":
        return set()
    if scope == "pilot":
        return groups & SBIAD_PILOT_GROUPS
    if scope == "opto-upper-bound":
        return groups & SBIAD_OPTO_UPPER_BOUND_GROUPS
    if scope == "control":
        return set(
            table.loc[
                table["Experiment"].astype(str) == SBIAD_PRIMARY_EXPERIMENT,
                "group",
            ].astype(str)
        )
    if scope == "control-opto":
        return set(
            table.loc[
                table["Experiment"].astype(str).isin(
                    [SBIAD_PRIMARY_EXPERIMENT, "angleOptoFree"]
                ),
                "group",
            ].astype(str)
        )
    raise ValueError(scope)


def head_remote(row: dict[str, Any], timeout: int) -> dict[str, Any]:
    result = dict(row)
    try:
        response = requests.head(
            result["url"],
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        result.update(
            {
                "head_status": "ok",
                "expected_bytes": int(response.headers.get("Content-Length", 0)),
                "etag": response.headers.get("ETag", "").strip('"'),
                "last_modified": response.headers.get("Last-Modified", ""),
                "accept_ranges": response.headers.get("Accept-Ranges", ""),
            }
        )
    except requests.RequestException as error:
        result.update(
            {
                "head_status": f"error:{type(error).__name__}",
                "expected_bytes": 0,
                "etag": "",
                "last_modified": "",
                "accept_ranges": "",
            }
        )
    return result


def download_resumable(
    url: str,
    destination: Path,
    expected_bytes: int,
    timeout: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    if destination.is_file() and (
        expected_bytes <= 0 or destination.stat().st_size == expected_bytes
    ):
        return {
            "download_status": "already_complete",
            "local_bytes": int(destination.stat().st_size),
            "local_sha256": sha256_file(destination),
        }
    offset = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = requests.get(
        url,
        stream=True,
        timeout=(timeout, max(180, timeout)),
        headers=headers,
    )
    if offset and response.status_code == 200:
        part.unlink(missing_ok=True)
        offset = 0
    elif offset and response.status_code != 206:
        response.raise_for_status()
        raise RuntimeError(f"Cannot resume {url}: HTTP {response.status_code}")
    response.raise_for_status()
    with part.open("ab" if offset else "wb") as handle:
        for block in response.iter_content(chunk_size=CHUNK_SIZE):
            if block:
                handle.write(block)
    actual = int(part.stat().st_size)
    if expected_bytes > 0 and actual != expected_bytes:
        raise RuntimeError(
            f"Size mismatch for {destination.name}: {actual} != {expected_bytes}"
        )
    part.replace(destination)
    return {
        "download_status": "downloaded",
        "local_bytes": actual,
        "local_sha256": sha256_file(destination),
    }


def download_selected_sbiad(
    manifest: pd.DataFrame,
    data_root: Path,
    timeout: int,
    workers: int,
    reserve_free_gib: float,
) -> pd.DataFrame:
    rows = manifest.to_dict("records")
    required = sum(
        max(0, int(row["expected_bytes"]) - int(row.get("local_bytes", 0)))
        for row in rows
    )
    free = shutil.disk_usage(data_root).free
    reserve = int(reserve_free_gib * (1 << 30))
    if required > max(0, free - reserve):
        raise RuntimeError(
            f"S-BIAD selection needs {required / (1 << 30):.2f} GiB but "
            f"only {(free - reserve) / (1 << 30):.2f} GiB is available "
            "after reserve."
        )

    def fetch(row: dict[str, Any]) -> dict[str, Any]:
        destination = Path(str(row["destination"]))
        try:
            status = download_resumable(
                str(row["url"]),
                destination,
                int(row["expected_bytes"]),
                timeout,
            )
        except Exception as error:  # keep a complete failure ledger
            status = {
                "download_status": f"error:{type(error).__name__}:{error}",
                "local_bytes": int(destination.stat().st_size)
                if destination.is_file()
                else 0,
                "local_sha256": "",
            }
        return {**row, **status, "local_exists": destination.is_file()}

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, row) for row in rows]
        for future in as_completed(futures):
            completed.append(future.result())
    return pd.DataFrame(completed).sort_values(["group", "channel"])


def inspect_sbiad_local(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import tifffile
    except ImportError:
        tifffile = None
    records: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        path = Path(str(row["destination"]))
        record = {
            "group": row["group"],
            "channel": row["channel"],
            "path": str(path),
            "available": path.is_file(),
            "shape": "",
            "dtype": "",
            "frames_or_rows": 0,
            "inspection_error": "",
        }
        if not path.is_file():
            records.append(record)
            continue
        try:
            if path.suffix.lower() in {".tif", ".tiff"} and tifffile:
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    record["shape"] = "x".join(str(item) for item in series.shape)
                    record["dtype"] = str(series.dtype)
                    record["frames_or_rows"] = int(series.shape[0])
            elif path.suffix.lower() == ".mat":
                data = loadmat(path, simplify_cells=True)
                keys = [key for key in data if not key.startswith("__")]
                record["shape"] = json.dumps(
                    {
                        key: list(np.asarray(data[key]).shape)
                        for key in keys[:12]
                    },
                    sort_keys=True,
                )
                lengths = [int(np.asarray(data[key]).size) for key in keys]
                record["frames_or_rows"] = max(lengths, default=0)
                record["dtype"] = ",".join(keys[:12])
        except Exception as error:
            record["inspection_error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    frame = pd.DataFrame(records)
    groups = sorted(set(frame["group"])) if len(frame) else []
    group_summary: dict[str, Any] = {}
    for group in groups:
        part = frame[frame["group"] == group]
        channels = set(part.loc[part["available"], "channel"])
        lengths = part.loc[
            (part["available"]) & (part["frames_or_rows"] > 0),
            "frames_or_rows",
        ].astype(int)
        group_summary[group] = {
            "available_channels": sorted(channels),
            "all_selected_available": bool(part["available"].all()),
            "reported_lengths": sorted(set(lengths.tolist())),
            "synchronized_length_candidate": bool(
                len(lengths) >= 2 and len(set(lengths.tolist())) <= 2
            ),
            "identity_contract": (
                "stage_followed_central_cell_candidate"
                if "stage_position" in channels
                else "unresolved"
            ),
        }
    return frame, group_summary


def build_sbiad(
    out_dir: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    study = http_get(SBIAD_API, args.timeout_seconds).json()
    info = http_get(SBIAD_INFO, args.timeout_seconds).json()
    tsv_bytes = http_get(SBIAD_TSV, args.timeout_seconds).content
    table = pd.read_csv(io.BytesIO(tsv_bytes), sep="\t")
    table.columns = [str(column).strip() for column in table.columns]
    table["group"] = table["Files"].astype(str).map(sbiad_group_name)
    table["acquisition_day"] = table["group"].map(acquisition_day)
    table["channel"] = table.apply(sbiad_channel, axis=1)
    selected_groups = select_sbiad_groups(table, args.sbiad_scope)
    selected = table[table["group"].isin(selected_groups)].copy()
    selected["url"] = selected["Files"].map(
        lambda value: f"{SBIAD_ROOT}/{quote(str(value), safe='/')}"
    )
    selected["destination"] = selected["Files"].map(
        lambda value: str((data_root / "sbiad365" / str(value)).resolve())
    )
    selected["outer_unit_movie"] = selected["group"]
    selected["outer_cluster_day"] = selected["acquisition_day"]
    selected["issue_time_status"] = selected["channel"].map(
        lambda channel: (
            "causal_current_or_past"
            if channel
            in {
                "stage_position",
                "dic",
                "nucleus_405",
                "rab6_488",
                "rab6_642",
                "membrane_642",
                "opto_561",
                "opto_activation",
            }
            else "audit_required"
        )
    )
    head_rows = selected[
        [
            "Files",
            "Experiment",
            "Cell line",
            "Imaging",
            "Magnification",
            "Label",
            "Compound",
            "Concentration",
            "Description",
            "group",
            "acquisition_day",
            "channel",
            "url",
            "destination",
            "outer_unit_movie",
            "outer_cluster_day",
            "issue_time_status",
        ]
    ].to_dict("records")
    probed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.head_workers)) as pool:
        futures = [
            pool.submit(head_remote, row, args.timeout_seconds) for row in head_rows
        ]
        for future in as_completed(futures):
            probed.append(future.result())
    manifest = pd.DataFrame(probed).sort_values(["group", "channel"])
    manifest["local_exists"] = manifest["destination"].map(
        lambda value: Path(str(value)).is_file()
    )
    manifest["local_bytes"] = manifest["destination"].map(
        lambda value: int(Path(str(value)).stat().st_size)
        if Path(str(value)).is_file()
        else 0
    )
    manifest["local_sha256"] = manifest["destination"].map(
        lambda value: sha256_file(Path(str(value)))
        if Path(str(value)).is_file()
        else ""
    )
    manifest["download_status"] = np.where(
        manifest["local_exists"], "already_complete", "not_requested"
    )
    if args.download_sbiad and len(manifest):
        manifest = download_selected_sbiad(
            manifest,
            data_root,
            args.timeout_seconds,
            args.download_workers,
            args.reserve_free_gib,
        )
    manifest.to_csv(out_dir / "sbiad365_paired_subset_manifest.csv", index=False)
    table.to_csv(out_dir / "sbiad365_full_file_index.csv", index=False)
    local_audit, group_summary = inspect_sbiad_local(manifest)
    local_audit.to_csv(out_dir / "sbiad365_local_alignment_audit.csv", index=False)

    primary = table[table["Experiment"].astype(str) == SBIAD_PRIMARY_EXPERIMENT]
    contract = {
        "schema_version": 1,
        "accession": SBIAD_ACC,
        "title": next(
            (
                item.get("value", "")
                for item in study.get("attributes", [])
                if item.get("name") == "Title"
            ),
            "",
        ),
        "source_api": SBIAD_API,
        "source_file_index": SBIAD_TSV,
        "public": bool(info.get("isPublic")),
        "released_unix_ms": info.get("released"),
        "total_files_reported": int(info.get("files", 0)),
        "total_files_indexed": int(len(table)),
        "selection_scope": args.sbiad_scope,
        "selected_groups": sorted(selected_groups),
        "selected_acquisition_days": sorted(
            set(selected["acquisition_day"].astype(str))
        ),
        "selected_files": int(len(manifest)),
        "selected_expected_bytes": int(manifest["expected_bytes"].sum())
        if len(manifest)
        else 0,
        "primary_control_groups_available": int(primary["group"].nunique()),
        "primary_control_day_clusters": int(primary["acquisition_day"].nunique()),
        "outer_unit": "movie/group",
        "cluster_unit": "acquisition_day",
        "identity_claim": "stage-followed central-cell candidate",
        "identity_passed": False,
        "identity_requirement": (
            "manual stratified central-cell/link audit plus Positions.mat and "
            "image synchronization"
        ),
        "target": "next stage-corrected central-cell displacement/innovation",
        "privileged_packets": [
            "nucleus-Golgi axis",
            "membrane/protrusion asymmetry",
            "Rab6 trafficking",
            "Cdc42/opto state where channels align",
        ],
        "deployable_packets": ["DIC", "membrane if target support is audited"],
        "future_leakage_allowed": False,
        "biological_upper_bound_ready": all(
            item["all_selected_available"] for item in group_summary.values()
        )
        and bool(group_summary),
        "group_alignment_summary": group_summary,
    }
    write_json(out_dir / "sbiad365_data_contract.json", contract)
    pd.DataFrame(
        [
            {
                "group": group,
                "acquisition_day": acquisition_day(group),
                "sampled_frames": "",
                "central_cell_visible_fraction": "",
                "identity_link_precision": "",
                "division_event": "",
                "death_event": "",
                "gap_event": "",
                "stage_image_alignment_pass": "",
                "reviewer": "",
                "status": "pending_manual_review",
            }
            for group in sorted(selected_groups)
        ]
    ).to_csv(out_dir / "sbiad365_manual_identity_audit_template.csv", index=False)
    return manifest, contract


def list_s3(prefix: str, timeout: int, *, delimiter: str = "") -> dict[str, Any]:
    params = {"list-type": "2", "prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    response = requests.get(
        f"{ALLEN_BUCKET}/",
        params=params,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    objects = []
    for item in root.findall("s3:Contents", namespace):
        objects.append(
            {
                "key": item.findtext("s3:Key", default="", namespaces=namespace),
                "bytes": int(
                    item.findtext("s3:Size", default="0", namespaces=namespace)
                ),
                "etag": item.findtext(
                    "s3:ETag", default="", namespaces=namespace
                ).strip('"'),
                "last_modified": item.findtext(
                    "s3:LastModified", default="", namespaces=namespace
                ),
            }
        )
    prefixes = [
        item.findtext("s3:Prefix", default="", namespaces=namespace)
        for item in root.findall("s3:CommonPrefixes", namespace)
    ]
    return {
        "objects": objects,
        "prefixes": prefixes,
        "is_truncated": (
            root.findtext("s3:IsTruncated", default="false", namespaces=namespace)
            == "true"
        ),
        "next_token": root.findtext(
            "s3:NextContinuationToken", default="", namespaces=namespace
        ),
    }


def csv_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    frame = pd.read_csv(path, low_memory=False)
    columns = [str(column) for column in frame.columns]
    lower = [column.lower() for column in columns]
    identity_candidates = [
        column
        for column, normalized in zip(columns, lower)
        if any(
            token in normalized
            for token in ("track", "cell_id", "object_id", "nucleus_id", "label")
        )
    ]
    time_candidates = [
        column
        for column, normalized in zip(columns, lower)
        if any(token in normalized for token in ("time", "frame"))
    ]
    experiment_candidates = [
        column
        for column, normalized in zip(columns, lower)
        if any(
            token in normalized
            for token in ("experiment", "fov", "well", "file_id", "dataset")
        )
    ]
    return {
        "available": True,
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "columns": columns,
        "identity_candidates": identity_candidates,
        "time_candidates": time_candidates,
        "experiment_candidates": experiment_candidates,
    }


def build_allen_emt(
    out_dir: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    allen_root = data_root / "allen_emt"
    allen_root.mkdir(parents=True, exist_ok=True)
    inventory = list_s3(f"{ALLEN_PREFIX}/manifests/", args.timeout_seconds)
    root_inventory = list_s3(
        f"{ALLEN_PREFIX}/data/", args.timeout_seconds, delimiter="/"
    )
    inventory_rows = inventory["objects"]
    key_lookup = {row["key"]: row for row in inventory_rows}
    downloads: list[dict[str, Any]] = []
    for key in ALLEN_SMALL_OBJECTS:
        destination = allen_root / key.removeprefix(f"{ALLEN_PREFIX}/")
        expected = int(key_lookup.get(key, {}).get("bytes", 0))
        status = download_resumable(
            object_url(key), destination, expected, args.timeout_seconds
        )
        downloads.append(
            {
                "key": key,
                "url": object_url(key),
                "destination": str(destination.resolve()),
                "expected_bytes": expected,
                **status,
            }
        )
    pd.DataFrame(downloads).to_csv(
        out_dir / "allen_emt_metadata_downloads.csv", index=False
    )

    large = key_lookup.get(ALLEN_LARGE_FEATURE_KEY, {})
    range_response = http_get(
        object_url(ALLEN_LARGE_FEATURE_KEY),
        args.timeout_seconds,
        headers={"Range": "bytes=0-262143"},
    )
    first_line = range_response.content.splitlines()[0].decode(
        "utf-8", errors="replace"
    )
    large_columns = next(csv.reader([first_line]))
    (out_dir / "allen_emt_large_feature_columns.txt").write_text(
        "\n".join(large_columns) + "\n", encoding="utf-8"
    )
    imaging_path = (
        allen_root / "manifests" / "imaging_and_segmentation_data.csv"
    )
    migration_path = (
        allen_root
        / "manifests"
        / "Migration_timing_through_mesh_extracted_features.csv"
    )
    imaging_contract = csv_contract(imaging_path)
    migration_contract = csv_contract(migration_path)
    persistent_candidates = sorted(
        set(imaging_contract.get("identity_candidates", []))
        | set(migration_contract.get("identity_candidates", []))
    )
    contract = {
        "schema_version": 1,
        "title": (
            "EMT timelapse imaging and segmentation dataset, "
            "Allen Institute for Cell Science"
        ),
        "article": "https://www.nature.com/articles/s41592-026-03096-9",
        "bucket": ALLEN_BUCKET,
        "prefix": ALLEN_PREFIX,
        "manifest_objects": inventory_rows,
        "metadata_downloads": downloads,
        "large_feature_table": {
            "key": ALLEN_LARGE_FEATURE_KEY,
            "bytes": int(large.get("bytes", 0)),
            "downloaded": False,
            "range_header_audited": True,
            "columns": large_columns,
        },
        "zarr_prefixes_first_page": root_inventory["prefixes"],
        "zarr_prefix_count_first_page": int(len(root_inventory["prefixes"])),
        "zarr_inventory_truncated": bool(root_inventory["is_truncated"]),
        "imaging_manifest": imaging_contract,
        "migration_manifest": migration_contract,
        "identity_candidates": persistent_candidates,
        "persistent_identity_passed": False,
        "persistent_identity_requirement": (
            "verify identifier continuity over time in a predeclared 2D pilot; "
            "segmentation labels alone do not establish tracks"
        ),
        "issue_time_contract_passed": False,
        "outer_unit": "FOV/experiment, pending manifest audit",
        "target": "next nucleus/cell innovation or migration-state transition",
        "privileged_state": [
            "H2B nuclear segmentation",
            "all-cell mask",
            "collagen-IV/ECM geometry",
            "migration timing labels",
        ],
        "deployable_state": ["brightfield causal prefix"],
        "pilot_ready": bool(imaging_contract.get("available")),
    }
    write_json(out_dir / "allen_emt_pilot_contract.json", contract)
    return contract


def build_allen_endothelial(out_dir: Path, timeout: int) -> dict[str, Any]:
    paper_url = (
        "https://www.biorxiv.org/content/"
        "10.64898/2026.07.07.736803v1.full"
    )
    paper_status = "unavailable"
    repository_links: list[str] = []
    text = ""
    try:
        response = http_get(paper_url, timeout)
        text = response.text
        paper_status = "accessible"
        repository_links = sorted(
            set(
                re.findall(
                    r"https?://[^\"'<>\s]+(?:figshare|zenodo|quilt|dryad|osf)"
                    r"[^\"'<>\s]*",
                    text,
                    flags=re.IGNORECASE,
                )
            )
        )
    except requests.RequestException as error:
        paper_status = f"error:{type(error).__name__}"
    dataset_manifest_url = (
        "https://allencell.s3.amazonaws.com/aics/"
        "endo_cell_state_dynamics/all_datasets_manifest.csv"
    )
    dataset_manifest_status = "unavailable"
    dataset_manifest_rows = 0
    file_manifests: list[str] = []
    tracked_experiments = 0
    try:
        manifest_response = http_get(dataset_manifest_url, timeout)
        manifest_path = out_dir / "allen_endothelial_dataset_manifest.csv"
        manifest_path.write_bytes(manifest_response.content)
        dataset_manifest = pd.read_csv(
            io.BytesIO(manifest_response.content)
        )
        dataset_manifest_rows = int(len(dataset_manifest))
        file_manifests = sorted(
            dataset_manifest["File Manifest"].dropna().astype(str).unique()
        )
        tracked_experiments = int(
            dataset_manifest.loc[
                dataset_manifest["File Manifest"].eq(
                    "cdh5_classic_segmentation_tracking"
                ),
                "Identity",
            ].nunique()
        )
        dataset_manifest_status = "verified"
    except (requests.RequestException, ValueError, KeyError) as error:
        dataset_manifest_status = f"error:{type(error).__name__}"
    machine_readable_verified = bool(
        dataset_manifest_status == "verified"
        and tracked_experiments > 0
        and "cell_centered_features_filtered" in file_manifests
    )
    contract = {
        "schema_version": 1,
        "title": (
            "Dynamics of ML-based Morphological Features Indicate a "
            "Shear Stress-Dependent Bifurcation of hiPSC-Derived "
            "Endothelial Cell States"
        ),
        "paper": paper_url,
        "paper_status": paper_status,
        "repository_links_found": repository_links,
        "dataset_manifest_url": dataset_manifest_url,
        "dataset_manifest_status": dataset_manifest_status,
        "dataset_manifest_rows": dataset_manifest_rows,
        "file_manifests": file_manifests,
        "tracked_experiments_in_manifest": tracked_experiments,
        "public_machine_readable_corpus_verified": machine_readable_verified,
        "persistent_identity_passed": False,
        "current_role": (
            "metadata_only_candidate"
            if not machine_readable_verified
            else "cell_state_upper_bound_ready_identity_audit_pending"
        ),
        "known_modalities_from_article": [
            "brightfield",
            "VE-cadherin",
            "controlled shear 0-24 dyn/cm2",
            "morphology/orientation/density",
            "migration coherence",
        ],
        "hard_limit": (
            "Published grid/cell-centered states cannot be treated as "
            "persistent cell trajectories without an explicit identity table."
        ),
        "decision": (
            "Run a held-out next-innovation upper bound on the published "
            "tracking and cell-centered feature tables. Do not schedule "
            "LaChance transfer until persistent identity, causal feature "
            "timing, and a deployable bright-field state pass hard controls."
        ),
    }
    write_json(out_dir / "allen_endothelial_state_contract.json", contract)
    return contract


def build_giga_manifest(out_dir: Path) -> pd.DataFrame:
    if not GIGA_SOURCE_MANIFEST.is_file():
        frame = pd.DataFrame(
            [
                {
                    "status": "missing_v167_source_manifest",
                    "source": str(GIGA_SOURCE_MANIFEST),
                }
            ]
        )
    else:
        source = pd.read_csv(GIGA_SOURCE_MANIFEST)
        frame = source[source["kind"].astype(str) == "raw_movie"].copy()
        frame["resumable"] = True
        frame["field_level_only"] = True
        frame["persistent_identity_passed"] = False
        frame["next_action"] = "download raw; verify registration to processed MF/ROI"
    frame.to_csv(out_dir / "gigascience_raw_manifest.csv", index=False)
    return frame


def mechanics_audit(
    out_dir: Path,
    sbiad: dict[str, Any],
    allen_emt: dict[str, Any],
    allen_endothelial: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "source": "S-BIAD365",
            "modality": "DIC+nucleus+Golgi+Rab6+membrane/opto",
            "license_status": "public BioStudies; exact reuse terms to cite",
            "identity_contract": sbiad["identity_claim"],
            "identity_passed": sbiad["identity_passed"],
            "issue_time_contract": "candidate; alignment audit pending",
            "deployable_overlap_lachance": "DIC/phase support audit required",
            "current_role": "privileged single-cell upper bound",
            "blocking_issue": "manual identity and synchronized channel audit",
        },
        {
            "source": "GigaScience 100118",
            "modality": "DIC+ROI+motion field+wound front",
            "license_status": "CC BY 4.0/DataCite contract from v167",
            "identity_contract": "field-level only",
            "identity_passed": False,
            "issue_time_contract": "v169 field contract passed",
            "deployable_overlap_lachance": "DIC/phase field, no cell identity",
            "current_role": "field representation/pretraining",
            "blocking_issue": "no established persistent cell tracks",
        },
        {
            "source": "Allen EMT",
            "modality": "brightfield+H2B masks+all-cell masks+ECM",
            "license_status": "public S3; exact dataset terms in README",
            "identity_contract": "candidate identifiers in manifests",
            "identity_passed": allen_emt["persistent_identity_passed"],
            "issue_time_contract": "pending 2D prefix pilot",
            "deployable_overlap_lachance": "brightfield/phase requires support audit",
            "current_role": "identity/state pilot",
            "blocking_issue": "prove time-continuous identity and causal state",
        },
        {
            "source": "Allen endothelial shear-state",
            "modality": "brightfield+VE-cadherin+controlled shear",
            "license_status": "preprint accessible; data repository unverified",
            "identity_contract": "field/cell-centered, not persistent",
            "identity_passed": False,
            "issue_time_contract": "metadata only",
            "deployable_overlap_lachance": "brightfield candidate",
            "current_role": allen_endothelial["current_role"],
            "blocking_issue": "public machine-readable repository not verified",
        },
        {
            "source": "Spatiotemporal force and motion collection",
            "modality": "traction+motion fields",
            "license_status": "public Figshare collection",
            "identity_contract": "field-level unless nuclei/tracks verified",
            "identity_passed": False,
            "issue_time_contract": "pending source-level audit",
            "deployable_overlap_lachance": "mechanics privileged only",
            "current_role": "mechanical observability upper-bound candidate",
            "blocking_issue": "cell-resolved identity and pre-motion timing",
        },
        {
            "source": "Prospective synchronized acquisition",
            "modality": "phase+identity+membrane/polarity+optional traction",
            "license_status": "to be defined",
            "identity_contract": "required by design",
            "identity_passed": False,
            "issue_time_contract": "required by design",
            "deployable_overlap_lachance": "phase/brightfield required",
            "current_role": "fallback if public state gates fail",
            "blocking_issue": "new data acquisition",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "identity_mechanics_source_audit.csv", index=False)
    return frame


def storage_budget(
    out_dir: Path,
    data_root: Path,
    sbiad_manifest: pd.DataFrame,
    giga_manifest: pd.DataFrame,
    allen_emt: dict[str, Any],
) -> pd.DataFrame:
    disk = shutil.disk_usage(data_root)
    rows = [
        {
            "source": "filesystem",
            "planned_bytes": 0,
            "downloaded_bytes": 0,
            "free_bytes_now": int(disk.free),
            "role": "capacity",
        },
        {
            "source": "S-BIAD365 selected",
            "planned_bytes": int(sbiad_manifest["expected_bytes"].sum())
            if len(sbiad_manifest)
            else 0,
            "downloaded_bytes": int(sbiad_manifest["local_bytes"].sum())
            if len(sbiad_manifest)
            else 0,
            "free_bytes_now": int(disk.free),
            "role": "privileged single-cell state",
        },
        {
            "source": "GigaScience raw 31",
            "planned_bytes": int(giga_manifest.get("file_size", pd.Series(dtype=int)).sum())
            if len(giga_manifest)
            else 0,
            "downloaded_bytes": int(
                giga_manifest.get("local_bytes", pd.Series(dtype=int)).sum()
            )
            if len(giga_manifest)
            else 0,
            "free_bytes_now": int(disk.free),
            "role": "field-level image history",
        },
        {
            "source": "Allen EMT small metadata",
            "planned_bytes": int(
                sum(
                    item.get("expected_bytes", 0)
                    for item in allen_emt["metadata_downloads"]
                )
            ),
            "downloaded_bytes": int(
                sum(
                    item.get("local_bytes", 0)
                    for item in allen_emt["metadata_downloads"]
                )
            ),
            "free_bytes_now": int(disk.free),
            "role": "identity/state contract discovery",
        },
        {
            "source": "Allen EMT full image-analysis table",
            "planned_bytes": int(
                allen_emt["large_feature_table"].get("bytes", 0)
            ),
            "downloaded_bytes": 0,
            "free_bytes_now": int(disk.free),
            "role": "deferred until identity columns justify transfer",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["planned_gib"] = frame["planned_bytes"] / float(1 << 30)
    frame["downloaded_gib"] = frame["downloaded_bytes"] / float(1 << 30)
    frame["free_gib_now"] = frame["free_bytes_now"] / float(1 << 30)
    frame.to_csv(out_dir / "download_storage_budget.csv", index=False)
    return frame


def decision_report(
    out_dir: Path,
    sbiad: dict[str, Any],
    allen_emt: dict[str, Any],
    allen_endothelial: dict[str, Any],
    giga_manifest: pd.DataFrame,
) -> None:
    lines = [
        "# v173 source-intake decision",
        "",
        "## Status",
        "",
        f"- S-BIAD selected groups: `{len(sbiad['selected_groups'])}`; "
        f"biological upper-bound ready: `{sbiad['biological_upper_bound_ready']}`.",
        f"- Allen EMT metadata pilot ready: `{allen_emt['pilot_ready']}`; "
        f"persistent identity passed: `{allen_emt['persistent_identity_passed']}`.",
        "- Allen endothelial public machine-readable corpus verified: "
        f"`{allen_endothelial['public_machine_readable_corpus_verified']}`.",
        f"- Giga raw movies in resumable manifest: `{len(giga_manifest)}`; "
        "the branch remains field-level.",
        "",
        "## Honest routing",
        "",
        "1. Run the S-BIAD privileged upper bound only after every selected "
        "channel is present and the central-cell/stage synchronization audit passes.",
        "2. Use Allen EMT only after persistent identifiers are verified across "
        "time; segmentation labels are not silently treated as tracks.",
        "3. Use Giga raw images for the field-level image/field ladder. It cannot "
        "unlock a LaChance cell bridge by itself.",
        "4. The endothelial S3 corpus is machine-readable. Run the tracked "
        "cell-state upper bound, but block transfer until identity/timing and "
        "deployable bright-field controls pass.",
        "",
        "No biological transfer claim is made by this intake.",
    ]
    (out_dir / "v173_source_intake_decision_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--sbiad-scope",
        choices=[
            "none",
            "pilot",
            "control",
            "opto-upper-bound",
            "control-opto",
        ],
        default="pilot",
    )
    parser.add_argument("--download-sbiad", action="store_true")
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--head-workers", type=int, default=8)
    parser.add_argument("--reserve-free-gib", type=float, default=35.0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = args.out_dir.resolve()
    data_root = args.data_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    sbiad_manifest, sbiad_contract = build_sbiad(
        out_dir, data_root, args
    )
    allen_emt = build_allen_emt(out_dir, data_root, args)
    allen_endothelial = build_allen_endothelial(
        out_dir, args.timeout_seconds
    )
    giga_manifest = build_giga_manifest(out_dir)
    mechanics_audit(
        out_dir, sbiad_contract, allen_emt, allen_endothelial
    )
    budget = storage_budget(
        out_dir, data_root, sbiad_manifest, giga_manifest, allen_emt
    )
    decision_report(
        out_dir,
        sbiad_contract,
        allen_emt,
        allen_endothelial,
        giga_manifest,
    )
    run_manifest = {
        "schema_version": 1,
        "runner": str(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "sbiad_scope": args.sbiad_scope,
        "sbiad_download_requested": bool(args.download_sbiad),
        "sbiad_files": int(len(sbiad_manifest)),
        "sbiad_complete_files": int(
            (sbiad_manifest["download_status"].isin(
                ["downloaded", "already_complete"]
            )).sum()
        )
        if len(sbiad_manifest)
        else 0,
        "allen_emt_pilot_ready": bool(allen_emt["pilot_ready"]),
        "giga_raw_movies": int(len(giga_manifest)),
        "free_gib": float(budget.iloc[0]["free_gib_now"]),
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(out_dir.iterdir())
            if path.is_file()
        },
    }
    write_json(out_dir / "v173_run_manifest.json", run_manifest)
    print(json.dumps(finite(run_manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
