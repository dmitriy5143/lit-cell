#!/usr/bin/env python3
"""v150 forensic intake for the MDCK-II force-motion dataset.

This runner creates a reproducible data contract before any predictive model
is trained. It downloads Figshare archives with resume support, verifies the
published MD5 checksum, safely extracts files, inventories image/MAT schemas,
and writes a conservative timestamp contract.

The most important causal rule is that cell-image displacement slice ``k`` is
computed from cell frames ``k`` and ``k+1``. It is therefore a target for a
forecast issued at frame ``k`` and is only available after frame ``k+1``.
Traction and stress are privileged, offline processed measurements because
they depend on bead images, a stress-free reference, and inverse processing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat, whosmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "external" / "mdck_force_motion"
DEFAULT_OUT = ROOT / "outputs" / "mdck_force_motion_intake_v150"
FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"


@dataclass(frozen=True)
class FigshareArticle:
    slug: str
    condition_family: str
    article_id: int
    file_id: int
    expected_name: str
    expected_size: int
    expected_md5: str
    phase: int

    @property
    def api_url(self) -> str:
        return FIGSHARE_API.format(article_id=self.article_id)

    @property
    def download_url(self) -> str:
        return f"https://ndownloader.figshare.com/files/{self.file_id}"


ARTICLES: dict[str, FigshareArticle] = {
    "low_density": FigshareArticle(
        slug="low_density",
        condition_family="density",
        article_id=12158757,
        file_id=22358163,
        expected_name="low_density_islands.zip",
        expected_size=1_496_436_699,
        expected_md5="d28890e14441de9e32c289e10cb1eb96",
        phase=1,
    ),
    "high_density": FigshareArticle(
        slug="high_density",
        condition_family="density",
        article_id=12158784,
        file_id=22358556,
        expected_name="high_density_islands.zip",
        expected_size=0,
        expected_md5="",
        phase=1,
    ),
    "cytod": FigshareArticle(
        slug="cytod",
        condition_family="cytochalasin_d",
        article_id=12158970,
        file_id=22358520,
        expected_name="cytoD.zip",
        expected_size=0,
        expected_md5="",
        phase=2,
    ),
    "cn03_1_4": FigshareArticle(
        slug="cn03_1_4",
        condition_family="cn03",
        article_id=12159441,
        file_id=22360524,
        expected_name="CN03_islands_1-4.zip",
        expected_size=0,
        expected_md5="",
        phase=2,
    ),
    "cn03_5_8": FigshareArticle(
        slug="cn03_5_8",
        condition_family="cn03",
        article_id=12159024,
        file_id=22358550,
        expected_name="CN03_islands_5-8.zip",
        expected_size=0,
        expected_md5="",
        phase=2,
    ),
}


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAT_SUFFIXES = {".mat"}


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Airi-v150-intake/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        # The system Python on some macOS installations does not inherit the
        # Keychain CA bundle. curl does, so use it as a verified-TLS fallback.
        result = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                str(int(timeout)),
                "-H",
                "User-Agent: Airi-v150-intake/1.0",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        raise RuntimeError(
            f"Unable to fetch {url}: urllib={exc}; curl={result.stderr.strip()}"
        ) from exc


def resolve_article_metadata(spec: FigshareArticle) -> dict[str, Any]:
    record = fetch_json(spec.api_url)
    matches = [item for item in record.get("files", []) if int(item.get("id", -1)) == spec.file_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Figshare file {spec.file_id} in article {spec.article_id}, found {len(matches)}")
    item = matches[0]
    return {
        **asdict(spec),
        "title": record.get("title", ""),
        "doi": record.get("doi", ""),
        "license": (record.get("license") or {}).get("name", ""),
        "published_date": record.get("published_date", ""),
        "api_file_name": item.get("name", ""),
        "api_size": int(item.get("size", 0)),
        "api_md5": item.get("computed_md5") or item.get("supplied_md5") or "",
        "api_download_url": item.get("download_url") or spec.download_url,
    }


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    command = [
        "curl",
        "-L",
        "--fail",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "-C",
        "-",
        "-o",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    partial.replace(target)


def ensure_archive(metadata: dict[str, Any], downloads_dir: Path, *, download: bool, verify: bool) -> dict[str, Any]:
    target = downloads_dir / str(metadata["api_file_name"])
    expected_size = int(metadata["api_size"])
    expected_md5 = str(metadata["api_md5"]).lower()
    if not target.exists() and download:
        download_with_resume(str(metadata["api_download_url"]), target)
    result = {
        **metadata,
        "local_path": str(target),
        "exists": target.exists(),
        "local_size": target.stat().st_size if target.exists() else 0,
        "size_ok": bool(target.exists() and target.stat().st_size == expected_size),
        "local_md5": "",
        "md5_ok": False,
    }
    if target.exists() and verify:
        local_md5 = md5_file(target)
        result["local_md5"] = local_md5
        result["md5_ok"] = local_md5.lower() == expected_md5
    elif target.exists():
        result["md5_ok"] = None
    if target.exists() and not result["size_ok"]:
        raise RuntimeError(
            f"Size mismatch for {target}: expected {expected_size}, observed {target.stat().st_size}. "
            "Remove the completed file and retain/use the .part file for a resumed download."
        )
    if target.exists() and verify and not result["md5_ok"]:
        raise RuntimeError(f"MD5 mismatch for {target}: expected {expected_md5}, observed {result['local_md5']}")
    return result


def safe_extract_zip(path: Path, destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if os.path.commonpath([str(destination_resolved), str(target)]) != str(destination_resolved):
                raise RuntimeError(f"Unsafe ZIP member path: {info.filename}")
            rows.append(
                {
                    "archive": str(path),
                    "member": info.filename,
                    "compressed_size": int(info.compress_size),
                    "uncompressed_size": int(info.file_size),
                    "crc": f"{info.CRC:08x}",
                    "is_dir": bool(info.is_dir()),
                }
            )
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    return rows


def infer_island(path: Path) -> str:
    text = "/".join(path.parts)
    match = re.search(r"(?:island|isl)[ _-]*0*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"island_{int(match.group(1)):02d}"
    return ""


def infer_file_family(path: Path) -> str:
    name = path.name.lower()
    if "cell_displacement" in name or "fidicc2" in name:
        return "cell_displacement"
    if "fidic" in name and "c2" not in name:
        return "substrate_displacement"
    if "tract" in name:
        return "traction"
    if "stress" in name:
        return "stress"
    if "domain" in name or "mask" in name:
        return "island_mask"
    if name.startswith("c2") or "cell" in name:
        return "cell_image"
    if "tryp" in name or "reference" in name or re.search(r"(?:^|[_-])ref(?:[_-]|\.)", name):
        return "reference_image"
    if name.startswith("c1") or "bead" in name:
        return "bead_image"
    if path.suffix.lower() in MAT_SUFFIXES:
        return "other_mat"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "other_image"
    if path.suffix.lower() == ".txt":
        return "settings_or_text"
    return "other"


def image_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"image_pages": np.nan, "image_height": np.nan, "image_width": np.nan, "image_dtype": ""}
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            result["image_pages"] = len(tif.pages)
            if tif.pages:
                shape = tuple(int(value) for value in tif.pages[0].shape)
                if len(shape) >= 2:
                    result["image_height"], result["image_width"] = shape[-2:]
                result["image_dtype"] = str(tif.pages[0].dtype)
        return result
    except Exception as exc:
        result["image_error"] = f"{type(exc).__name__}: {exc}"
        return result


def estimate_transition_offset_from_video(
    image_path: Path,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    inferred_offset: int,
) -> dict[str, Any]:
    """Validate the count-derived offset against causal image motion.

    This is not used as a model feature. It is a forensic guard against
    silently pairing a published PIV target with the wrong raw-video frames.
    """
    try:
        import cv2
        import tifffile

        scale = 0.25
        target_frames = sorted(
            set([0, target_u.shape[-1] // 2, target_u.shape[-1] - 1])
        )
        with tifffile.TiffFile(image_path) as tif:
            candidate_count = len(tif.pages) - target_u.shape[-1]
            candidates = range(max(candidate_count, 0))
            needed = {
                offset + target_frame + delta
                for offset in candidates
                for target_frame in target_frames
                for delta in (0, 1)
                if offset + target_frame + delta < len(tif.pages)
            }
            images: dict[int, np.ndarray] = {}
            for frame in sorted(needed):
                image = tif.pages[frame].asarray()
                low, high = np.percentile(image, [1, 99])
                normalized = np.clip(
                    (image.astype(np.float32) - low) / max(high - low, 1.0),
                    0.0,
                    1.0,
                )
                images[frame] = cv2.resize(
                    (normalized * 255.0).astype(np.uint8),
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
        score_rows: list[tuple[int, float]] = []
        map_x = (cell_x * scale).astype(np.float32)
        map_y = (cell_y * scale).astype(np.float32)
        for offset in candidates:
            frame_scores: list[float] = []
            for target_frame in target_frames:
                first = images.get(offset + target_frame)
                second = images.get(offset + target_frame + 1)
                if first is None or second is None:
                    continue
                flow = cv2.calcOpticalFlowFarneback(
                    first,
                    second,
                    None,
                    0.5,
                    4,
                    31,
                    5,
                    7,
                    1.5,
                    0,
                )
                flow_x = cv2.remap(
                    flow[..., 0],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                ) / scale
                flow_y = cv2.remap(
                    flow[..., 1],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                ) / scale
                published_x = target_u[..., target_frame]
                published_y = target_v[..., target_frame]
                published_norm = np.hypot(published_x, published_y)
                flow_norm = np.hypot(flow_x, flow_y)
                finite = (
                    np.isfinite(published_x)
                    & np.isfinite(published_y)
                    & np.isfinite(flow_x)
                    & np.isfinite(flow_y)
                    & (published_norm > 1e-3)
                    & (flow_norm > 1e-3)
                )
                if not np.any(finite):
                    continue
                robust_limit = np.percentile(published_norm[finite], 99.0)
                finite &= published_norm <= robust_limit
                cosine = (
                    flow_x * published_x + flow_y * published_y
                ) / np.maximum(flow_norm * published_norm, 1e-6)
                frame_scores.append(float(np.median(cosine[finite])))
            score_rows.append(
                (offset, float(np.median(frame_scores)) if frame_scores else np.nan)
            )
        finite_scores = [item for item in score_rows if np.isfinite(item[1])]
        ranked = sorted(finite_scores, key=lambda item: item[1], reverse=True)
        best_offset, best_score = ranked[0] if ranked else (-1, np.nan)
        second_score = ranked[1][1] if len(ranked) > 1 else np.nan
        margin = (
            best_score - second_score
            if np.isfinite(best_score) and np.isfinite(second_score)
            else np.nan
        )
        return {
            "video_flow_best_offset": int(best_offset),
            "video_flow_best_cosine": float(best_score),
            "video_flow_second_cosine": float(second_score),
            "video_flow_offset_margin": float(margin),
            "video_flow_offset_matches_count_contract": bool(
                best_offset == inferred_offset
            ),
            "video_flow_offset_scores": "|".join(
                f"{offset}:{score:.6f}" for offset, score in score_rows
            ),
            "video_flow_alignment_error": "",
        }
    except Exception as exc:
        return {
            "video_flow_best_offset": -1,
            "video_flow_best_cosine": np.nan,
            "video_flow_second_cosine": np.nan,
            "video_flow_offset_margin": np.nan,
            "video_flow_offset_matches_count_contract": False,
            "video_flow_offset_scores": "",
            "video_flow_alignment_error": f"{type(exc).__name__}: {exc}",
        }


def mat_schema(path: Path) -> list[dict[str, Any]]:
    try:
        variables = whosmat(path)
        return [
            {
                "path": str(path),
                "variable": name,
                "shape": "x".join(str(int(value)) for value in shape),
                "matlab_class": matlab_class,
                "schema_reader": "scipy.io.whosmat",
                "schema_error": "",
            }
            for name, shape, matlab_class in variables
        ]
    except Exception as scipy_exc:
        try:
            import h5py

            rows: list[dict[str, Any]] = []
            with h5py.File(path, "r") as handle:
                def visitor(name: str, item: Any) -> None:
                    if isinstance(item, h5py.Dataset):
                        rows.append(
                            {
                                "path": str(path),
                                "variable": name,
                                "shape": "x".join(str(int(value)) for value in item.shape),
                                "matlab_class": str(item.dtype),
                                "schema_reader": "h5py",
                                "schema_error": "",
                            }
                        )

                handle.visititems(visitor)
            return rows
        except Exception as h5_exc:
            return [
                {
                    "path": str(path),
                    "variable": "",
                    "shape": "",
                    "matlab_class": "",
                    "schema_reader": "failed",
                    "schema_error": (
                        f"scipy={type(scipy_exc).__name__}: {scipy_exc}; "
                        f"h5py={type(h5_exc).__name__}: {h5_exc}"
                    ),
                }
            ]


def temporal_contract(file_family: str) -> dict[str, Any]:
    contracts: dict[str, dict[str, Any]] = {
        "cell_image": {
            "observation_support": "frame_k",
            "available_at_issue_time": "k",
            "causal_role": "transferable_input",
            "forecast_use": "frames<=k only",
            "leakage_note": "",
        },
        "island_mask": {
            "observation_support": "frame_k_or_static_domain",
            "available_at_issue_time": "k_if_computed_causally",
            "causal_role": "transferable_input_after_alignment_audit",
            "forecast_use": "mask at or before k",
            "leakage_note": "Offline segmentation using future frames is forbidden.",
        },
        "cell_displacement": {
            "observation_support": "cell_frames_k_to_k+1",
            "available_at_issue_time": "k+1",
            "causal_role": "next_transition_target",
            "forecast_use": "slice k forbidden at issue time k; completed slice k-1 allowed",
            "leakage_note": "Official plot_cellvel.m displays frames k and k+1 for displacement slice k.",
        },
        "substrate_displacement": {
            "observation_support": "bead_frame_k_to_stress_free_reference",
            "available_at_issue_time": "offline_after_reference_acquisition",
            "causal_role": "privileged_teacher_or_qa",
            "forecast_use": "never a transferable LaChance inference input",
            "leakage_note": "Stress-free reference is acquired after the experiment.",
        },
        "traction": {
            "observation_support": "substrate_displacement_k_plus_tfm_inverse",
            "available_at_issue_time": "offline_after_reference_and_processing",
            "causal_role": "privileged_teacher_label",
            "forecast_use": "upper-bound/teacher supervision only",
            "leakage_note": "Processed inverse estimate, not an online sensor.",
        },
        "stress": {
            "observation_support": "traction_k_plus_boundary_and_msm",
            "available_at_issue_time": "offline_after_reference_and_processing",
            "causal_role": "privileged_teacher_label",
            "forecast_use": "upper-bound/teacher supervision only",
            "leakage_note": "Derived from traction and boundary assumptions; not independent supervision.",
        },
        "bead_image": {
            "observation_support": "bead_frame_k",
            "available_at_issue_time": "k_but_not_available_in_lachance",
            "causal_role": "privileged_teacher_input",
            "forecast_use": "external upper-bound only",
            "leakage_note": "",
        },
        "reference_image": {
            "observation_support": "post_experiment",
            "available_at_issue_time": "never_during_experiment",
            "causal_role": "offline_qa_only",
            "forecast_use": "never a forecast input",
            "leakage_note": "Acquired after cell removal.",
        },
    }
    fallback = {
        "observation_support": "unknown",
        "available_at_issue_time": "unknown",
        "causal_role": "audit_required",
        "forecast_use": "forbidden_until_resolved",
        "leakage_note": "Conservative default.",
    }
    return contracts.get(file_family, fallback)


def git_provenance(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "commit": "", "remote": ""}
    if not path.exists():
        return result
    for key, command in {
        "commit": ["git", "-C", str(path), "rev-parse", "HEAD"],
        "remote": ["git", "-C", str(path), "remote", "get-url", "origin"],
    }.items():
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            result[key] = completed.stdout.strip()
    license_files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.name.lower().startswith(("license", "copying"))
    )
    result["license_files"] = [str(candidate) for candidate in license_files]
    result["license_md5"] = {candidate.name: md5_file(candidate) for candidate in license_files}
    return result


def data_contract_payload(selected: list[str]) -> dict[str, Any]:
    return {
        "version": "v150",
        "selected_articles": selected,
        "forecast_event": {
            "issue_time": "cell frame t",
            "target": "cell displacement/velocity from frame t to frame t+1",
            "available_inputs": [
                "cell-channel images at frames <=t",
                "causal masks/tracks at frames <=t",
                "completed cell displacement history ending at t",
            ],
            "forbidden_inputs": [
                "published cell-displacement slice t computed from frames t,t+1",
                "future images or masks",
                "stress-free reference acquired after the experiment",
            ],
        },
        "privileged_channel_contract": {
            "transferable_student_inputs": [
                "cell-channel video",
                "causal masks/tracks",
                "completed velocity/PIV history",
            ],
            "teacher_labels_or_qa_only": [
                "fluorescent bead images",
                "stress-free reference",
                "substrate displacement",
                "traction",
                "monolayer stress",
            ],
        },
        "statistical_unit": "island/movie",
        "forbidden_split": "adjacent frames or grid patches from one island across train/validation/test",
        "primary_split": "whole-island holdout",
        "source_code_evidence": {
            "cell_velocity_timestamp": "plot_cellvel.m displays cell frames k and k+1 for displacement slice k",
            "tfm_msm": "Cell-Traction-Stress README and processing scripts",
        },
    }


def _as_time_last(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D/3D field, observed shape {array.shape}")
    return array


def _grid_axes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_axis = np.asarray(x[0, :], dtype=np.float64)
    y_axis = np.asarray(y[:, 0], dtype=np.float64)
    if np.any(np.diff(x_axis) <= 0) or np.any(np.diff(y_axis) <= 0):
        raise ValueError("Only monotonic rectilinear grids are supported by the v150 alignment audit")
    return x_axis, y_axis


def _interpolate_field(
    source_x: np.ndarray,
    source_y: np.ndarray,
    field: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> np.ndarray:
    x_axis, y_axis = _grid_axes(source_x, source_y)
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis),
        np.asarray(field, dtype=np.float64),
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack([np.asarray(query_y).ravel(), np.asarray(query_x).ravel()])
    return interpolator(points).reshape(np.asarray(query_x).shape)


def _sample_domain(domain: np.ndarray, x: np.ndarray, y: np.ndarray, frame: int) -> np.ndarray:
    image = np.asarray(domain[min(int(frame), len(domain) - 1)])
    xi = np.clip(np.rint(x).astype(np.int64) - 1, 0, image.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(np.int64) - 1, 0, image.shape[0] - 1)
    return image[yi, xi] > 0


def _field_stats(
    slug: str,
    island: str,
    family: str,
    variable: str,
    value: np.ndarray,
    domain_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    selected = array
    if domain_mask is not None and array.ndim == 3 and domain_mask.shape == array.shape:
        selected = array[domain_mask]
    finite_mask = np.isfinite(selected)
    finite_values = selected[finite_mask]
    if finite_values.size:
        quantiles = np.percentile(finite_values, [0, 1, 50, 95, 99, 100])
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
    else:
        quantiles = np.full(6, np.nan)
        mean = std = np.nan
    return {
        "slug": slug,
        "island": island,
        "file_family": family,
        "variable": variable,
        "shape": "x".join(str(int(item)) for item in array.shape),
        "count": int(selected.size),
        "finite_fraction": float(np.mean(finite_mask)) if selected.size else np.nan,
        "mean": mean,
        "std": std,
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def audit_island_fields(
    slug: str,
    island_dir: Path,
    plots_dir: Path,
    *,
    make_plots: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell_path = island_dir / "cell_displacements.mat"
    traction_path = island_dir / "tract_results.mat"
    stress_path = island_dir / "stress_results.mat"
    domain_path = island_dir / "domain.tif"
    image_candidates = sorted(island_dir.glob("c2*.tif"))
    required = [cell_path, traction_path, stress_path, domain_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return (
            {
                "slug": slug,
                "island": island_dir.name,
                "status": "missing_required_files",
                "missing": "|".join(missing),
            },
            [],
        )

    import tifffile

    cell = loadmat(cell_path, variable_names=["x", "y", "u", "v", "c_peak", "d0", "w0"])
    traction = loadmat(traction_path, variable_names=["x", "y", "u", "v", "tx", "ty", "d0", "w0"])
    stress = loadmat(stress_path, variable_names=["x", "y", "Sxx", "Syy", "Sxy", "S1", "S2", "pangle"])
    domain = tifffile.imread(domain_path)
    raw_cell_u = _as_time_last(cell["u"])
    raw_cell_v = _as_time_last(cell["v"])
    tx = _as_time_last(traction["tx"])
    ty = _as_time_last(traction["ty"])
    sxx = _as_time_last(stress["Sxx"])
    syy = _as_time_last(stress["Syy"])
    sxy = _as_time_last(stress["Sxy"])

    cell_x = np.asarray(cell["x"], dtype=np.float64)
    cell_y = np.asarray(cell["y"], dtype=np.float64)
    mech_x = np.asarray(traction["x"], dtype=np.float64)
    mech_y = np.asarray(traction["y"], dtype=np.float64)
    cell_x_axis, cell_y_axis = _grid_axes(cell_x, cell_y)
    mech_x_axis, mech_y_axis = _grid_axes(mech_x, mech_y)

    image_pages = np.nan
    if image_candidates:
        with tifffile.TiffFile(image_candidates[0]) as tif:
            image_pages = len(tif.pages)
    timeline_frames = int(domain.shape[0])
    raw_target_start_frame = timeline_frames - int(raw_cell_u.shape[-1]) - 1
    if raw_target_start_frame < 0:
        raise ValueError(
            f"{island_dir}: {raw_cell_u.shape[-1]} transitions cannot fit in "
            f"{timeline_frames} observation frames"
        )
    canonical_issue_start = (
        6 if slug in {"cytod", "cn03_1_4", "cn03_5_8"} else 0
    )
    target_array_start = canonical_issue_start - raw_target_start_frame
    if target_array_start < 0 or target_array_start >= raw_cell_u.shape[-1]:
        raise ValueError(
            f"{island_dir}: cannot align raw target offset "
            f"{raw_target_start_frame} to canonical issue frame "
            f"{canonical_issue_start}"
        )
    cell_u = raw_cell_u[..., target_array_start:]
    cell_v = raw_cell_v[..., target_array_start:]
    target_start_frame = raw_target_start_frame + target_array_start
    masks = np.stack(
        [
            _sample_domain(domain, cell_x, cell_y, target_start_frame + frame)
            for frame in range(cell_u.shape[-1])
        ],
        axis=-1,
    )
    target_mag = np.hypot(cell_u, cell_v)
    inside_values = target_mag[masks]
    robust_limit = float(np.percentile(inside_values, 99.9)) if inside_values.size else np.nan

    frame_for_roundtrip = min(
        target_start_frame + cell_u.shape[-1] // 2,
        tx.shape[-1] - 1,
    )
    overlap = (
        (cell_x >= mech_x_axis.min())
        & (cell_x <= mech_x_axis.max())
        & (cell_y >= mech_y_axis.min())
        & (cell_y <= mech_y_axis.max())
    )
    roundtrip_errors: list[float] = []
    for field in (tx[..., frame_for_roundtrip], ty[..., frame_for_roundtrip], sxx[..., frame_for_roundtrip]):
        on_cell = _interpolate_field(mech_x, mech_y, field, cell_x, cell_y)
        back_on_mechanics = _interpolate_field(cell_x, cell_y, on_cell, mech_x, mech_y)
        valid = np.isfinite(back_on_mechanics) & np.isfinite(field)
        scale = float(np.sqrt(np.mean(np.square(field[valid])))) if np.any(valid) else np.nan
        error = float(np.sqrt(np.mean(np.square(back_on_mechanics[valid] - field[valid])))) if np.any(valid) else np.nan
        roundtrip_errors.append(error / max(scale, 1e-12) if np.isfinite(error) and np.isfinite(scale) else np.nan)

    video_offset_audit = (
        estimate_transition_offset_from_video(
            image_candidates[0],
            cell_x,
            cell_y,
            raw_cell_u,
            raw_cell_v,
            raw_target_start_frame,
        )
        if image_candidates
        else {}
    )
    alignment = {
        "slug": slug,
        "island": island_dir.name,
        "status": "ok",
        "image_pages": image_pages,
        "domain_pages": int(domain.shape[0]),
        "cell_transition_frames": int(raw_cell_u.shape[-1]),
        "raw_target_transition_start_frame": int(raw_target_start_frame),
        "analysis_target_array_start": int(target_array_start),
        "analysis_transition_frames": int(cell_u.shape[-1]),
        "target_transition_start_frame": int(target_start_frame),
        "traction_frames": int(tx.shape[-1]),
        "stress_frames": int(sxx.shape[-1]),
        "frame_contract_ok": bool(
            image_pages == timeline_frames
            and raw_cell_u.shape[-1] + raw_target_start_frame + 1 == image_pages
            and tx.shape[-1] == timeline_frames
            and sxx.shape[-1] == image_pages
        ),
        "analysis_interval_contract_ok": bool(
            target_start_frame == canonical_issue_start
            and cell_u.shape[-1] + target_start_frame + 1 == image_pages
        ),
        "cell_grid_rows": int(cell_x.shape[0]),
        "cell_grid_cols": int(cell_x.shape[1]),
        "mechanics_grid_rows": int(mech_x.shape[0]),
        "mechanics_grid_cols": int(mech_x.shape[1]),
        "cell_grid_spacing_px_x": float(np.median(np.diff(cell_x_axis))),
        "cell_grid_spacing_px_y": float(np.median(np.diff(cell_y_axis))),
        "mechanics_grid_spacing_px_x": float(np.median(np.diff(mech_x_axis))),
        "mechanics_grid_spacing_px_y": float(np.median(np.diff(mech_y_axis))),
        "mechanics_overlap_fraction_cell_grid": float(np.mean(overlap)),
        "domain_fraction_cell_grid": float(np.mean(masks)),
        "target_inside_p99_9_px_per_frame": robust_limit,
        "traction_x_roundtrip_nrmse": roundtrip_errors[0],
        "traction_y_roundtrip_nrmse": roundtrip_errors[1],
        "stress_xx_roundtrip_nrmse": roundtrip_errors[2],
        "pixel_size_um": 0.66,
        "frame_interval_min": 10.0,
        **video_offset_audit,
    }

    quality_rows: list[dict[str, Any]] = []
    quality_rows.extend(
        [
            _field_stats(slug, island_dir.name, "cell_displacement", "u", cell_u, masks),
            _field_stats(slug, island_dir.name, "cell_displacement", "v", cell_v, masks),
            _field_stats(slug, island_dir.name, "cell_displacement", "speed", target_mag, masks),
            _field_stats(slug, island_dir.name, "traction", "tx", tx),
            _field_stats(slug, island_dir.name, "traction", "ty", ty),
            _field_stats(slug, island_dir.name, "traction", "magnitude", np.hypot(tx, ty)),
            _field_stats(slug, island_dir.name, "stress", "Sxx", sxx),
            _field_stats(slug, island_dir.name, "stress", "Syy", syy),
            _field_stats(slug, island_dir.name, "stress", "Sxy", sxy),
        ]
    )
    quality_rows.append(
        {
            "slug": slug,
            "island": island_dir.name,
            "file_family": "cell_displacement",
            "variable": "inside_domain_outlier_fraction_above_train_independent_p99_9",
            "shape": "",
            "count": int(inside_values.size),
            "finite_fraction": 1.0,
            "mean": float(np.mean(inside_values > robust_limit)) if inside_values.size else np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": robust_limit,
        }
    )

    if make_plots and image_candidates:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_times = sorted(set([0, cell_u.shape[-1] // 2, cell_u.shape[-1] - 1]))
            figure, axes = plt.subplots(len(plot_times), 4, figsize=(14, 3.8 * len(plot_times)), constrained_layout=True)
            if len(plot_times) == 1:
                axes = axes[None, :]
            with tifffile.TiffFile(image_candidates[0]) as tif:
                for row, target_frame in enumerate(plot_times):
                    issue_frame = target_start_frame + target_frame
                    image = tif.pages[issue_frame].asarray()
                    lo, hi = np.percentile(image, [1, 99])
                    axes[row, 0].imshow(image, cmap="gray", vmin=lo, vmax=hi)
                    axes[row, 0].set_title(
                        f"issue image={issue_frame}, target transition={target_frame}"
                    )
                    axes[row, 1].imshow(
                        np.where(
                            masks[..., target_frame],
                            target_mag[..., target_frame],
                            np.nan,
                        ),
                        origin="lower",
                        cmap="magma",
                    )
                    axes[row, 1].set_title("next PIV magnitude")
                    axes[row, 2].imshow(
                        np.hypot(tx[..., issue_frame], ty[..., issue_frame]),
                        origin="lower",
                        cmap="magma",
                    )
                    axes[row, 2].set_title("traction magnitude")
                    axes[row, 3].imshow(
                        (sxx[..., issue_frame] + syy[..., issue_frame]) / 2.0,
                        origin="lower",
                        cmap="coolwarm",
                    )
                    axes[row, 3].set_title("mean normal stress")
                    for column in range(4):
                        axes[row, column].set_axis_off()
            plots_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(plots_dir / f"alignment_{slug}_{island_dir.name}.png", dpi=140)
            plt.close(figure)
        except Exception as exc:
            alignment["plot_error"] = f"{type(exc).__name__}: {exc}"
    return alignment, quality_rows


def audit_all_islands(
    extracted_dir: Path,
    selected: list[str],
    plots_dir: Path,
    *,
    make_plots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alignment_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for slug in selected:
        slug_root = extracted_dir / slug
        for cell_path in sorted(slug_root.rglob("cell_displacements.mat")):
            alignment, quality = audit_island_fields(
                slug,
                cell_path.parent,
                plots_dir,
                make_plots=make_plots,
            )
            alignment_rows.append(alignment)
            quality_rows.extend(quality)
    return pd.DataFrame(alignment_rows), pd.DataFrame(quality_rows)


def inventory_extracted(slug: str, extract_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    for path in sorted(item for item in extract_root.rglob("*") if item.is_file()):
        family = infer_file_family(path)
        row: dict[str, Any] = {
            "slug": slug,
            "path": str(path),
            "relative_path": str(path.relative_to(extract_root)),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
            "island": infer_island(path.relative_to(extract_root)),
            "file_family": family,
            **temporal_contract(family),
        }
        if path.suffix.lower() in IMAGE_SUFFIXES:
            row.update(image_metadata(path))
        files.append(row)
        if path.suffix.lower() in MAT_SUFFIXES:
            for schema_row in mat_schema(path):
                schemas.append({"slug": slug, "island": row["island"], "file_family": family, **schema_row})
    return files, schemas


def write_report(
    out_dir: Path,
    manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    schemas: pd.DataFrame,
    alignment: pd.DataFrame,
    selected: list[str],
) -> None:
    verified = int((manifest.get("md5_ok", pd.Series(dtype=bool)) == True).sum())  # noqa: E712
    files = len(inventory)
    mat_files = int((inventory.get("suffix", pd.Series(dtype=str)) == ".mat").sum())
    image_files = int(inventory.get("suffix", pd.Series(dtype=str)).isin(IMAGE_SUFFIXES).sum())
    unresolved = int((inventory.get("available_at_issue_time", pd.Series(dtype=str)) == "unknown").sum())
    failed_schemas = int((schemas.get("schema_reader", pd.Series(dtype=str)) == "failed").sum())
    alignment_count = len(alignment)
    frame_contract_ok = 0
    offset_contract_ok = 0
    mechanics_coverage = np.nan
    traction_roundtrip = np.nan
    stress_roundtrip = np.nan
    if not alignment.empty:
        if "frame_contract_ok" in alignment:
            frame_contract_ok = int(alignment["frame_contract_ok"].fillna(False).astype(bool).sum())
        if "video_flow_offset_matches_count_contract" in alignment:
            offset_contract_ok = int(
                alignment["video_flow_offset_matches_count_contract"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        if "mechanics_overlap_fraction_cell_grid" in alignment:
            mechanics_coverage = float(alignment["mechanics_overlap_fraction_cell_grid"].mean())
        traction_columns = [
            column
            for column in ("traction_x_roundtrip_nrmse", "traction_y_roundtrip_nrmse")
            if column in alignment
        ]
        if traction_columns:
            traction_roundtrip = float(alignment[traction_columns].mean().mean())
        if "stress_xx_roundtrip_nrmse" in alignment:
            stress_roundtrip = float(alignment["stress_xx_roundtrip_nrmse"].mean())

    def percent(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{100.0 * value:.2f}%"

    lines = [
        "# MDCK force-motion v150 intake report",
        "",
        "## Scope",
        "",
        f"- selected archives: `{', '.join(selected)}`",
        f"- verified archives: `{verified}/{len(manifest)}`",
        f"- extracted files: `{files}`",
        f"- image files: `{image_files}`",
        f"- MAT files: `{mat_files}`",
        f"- unresolved timestamp rows: `{unresolved}`",
        f"- unreadable MAT schemas: `{failed_schemas}`",
        f"- audited islands: `{alignment_count}`",
        f"- explicit frame contracts passed: `{frame_contract_ok}/{alignment_count}`",
        f"- raw-video/PIV offset checks passed: `{offset_contract_ok}/{alignment_count}`",
        f"- mean mechanics coverage of the cell grid: `{percent(mechanics_coverage)}`",
        f"- traction source-grid roundtrip NRMSE: `{percent(traction_roundtrip)}`",
        f"- stress source-grid roundtrip NRMSE: `{percent(stress_roundtrip)}`",
        "",
        "## Causal contract",
        "",
        "- Cell displacement slice `k` is computed from cell frames `k` and `k+1`.",
        "- It is the next-transition target for a forecast issued at `k`, not an input.",
        "- Only completed displacement `k-1 -> k` is causal at issue time `k`.",
        "- For cytoD/CN03, pre/post-treatment stacks are discontinuous; the transition across that gap is excluded.",
        "- The analysis issue-frame offset is stored per island and independently checked against raw-video optical flow.",
        "- Traction/stress are privileged offline labels because TFM/MSM requires bead/reference processing.",
        "- The stress-free reference and bead channel are never transferable LaChance inference inputs.",
        "",
        "## Gate",
        "",
        "v151 may start only when every selected archive passes size/MD5 verification,",
        "the MAT schema is readable, and frame/field counts support an explicit one-frame shift.",
        "Any unresolved timestamp is forbidden as a model input.",
        "",
        "Traction interpolation is intentionally reported as a separate gate: a large",
        "source -> cell-grid -> source error means that a single aligned vector discards",
        "fine-scale traction structure and cannot serve as the definitive mechanics test.",
    ]
    (out_dir / "v150_intake_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", default="low_density,high_density")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = parse_strings(args.articles)
    unknown = sorted(set(selected) - set(ARTICLES))
    if unknown:
        raise ValueError(f"Unknown article slugs: {unknown}; available={sorted(ARTICLES)}")

    args.data_root.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = args.data_root / "downloads"
    extracted_dir = args.data_root / "extracted"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    metadata_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []

    for slug in selected:
        spec = ARTICLES[slug]
        print(f"[v150] metadata {slug}: {spec.api_url}", flush=True)
        metadata = resolve_article_metadata(spec)
        metadata_rows.append(metadata)
        if args.metadata_only:
            archive_rows.append({**metadata, "local_path": "", "exists": False})
            continue
        archive = ensure_archive(metadata, downloads_dir, download=bool(args.download), verify=bool(args.verify))
        archive_rows.append(archive)
        if not archive["exists"]:
            print(f"[v150] archive absent (use --download): {archive['local_path']}", flush=True)
            continue
        if args.extract:
            destination = extracted_dir / slug
            print(f"[v150] extract {archive['local_path']} -> {destination}", flush=True)
            members = safe_extract_zip(Path(archive["local_path"]), destination)
            for member in members:
                member["slug"] = slug
            pd.DataFrame(members).to_csv(args.out_dir / f"{slug}_archive_members.csv", index=False)
        destination = extracted_dir / slug
        if destination.exists():
            inventory, schemas = inventory_extracted(slug, destination)
            file_rows.extend(inventory)
            schema_rows.extend(schemas)

    metadata_frame = pd.DataFrame(metadata_rows)
    manifest = pd.DataFrame(archive_rows)
    inventory_frame = pd.DataFrame(file_rows)
    schema_frame = pd.DataFrame(schema_rows)
    metadata_frame.to_csv(args.out_dir / "figshare_metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest.to_csv(args.out_dir / "download_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    inventory_frame.to_csv(args.out_dir / "extracted_file_inventory.csv", index=False)
    schema_frame.to_csv(args.out_dir / "mat_schema.csv", index=False)
    schema_frame.to_csv(args.out_dir / "field_schema.csv", index=False)
    if not inventory_frame.empty:
        contract_columns = [
            "slug",
            "island",
            "relative_path",
            "file_family",
            "observation_support",
            "available_at_issue_time",
            "causal_role",
            "forecast_use",
            "leakage_note",
        ]
        inventory_frame[contract_columns].to_csv(args.out_dir / "temporal_contract.csv", index=False)
        inventory_frame[contract_columns].to_csv(args.out_dir / "timestamp_contract.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.out_dir / "temporal_contract.csv", index=False)
        pd.DataFrame().to_csv(args.out_dir / "timestamp_contract.csv", index=False)

    alignment_frame, quality_frame = audit_all_islands(
        extracted_dir,
        selected,
        args.out_dir / "plots",
        make_plots=bool(args.plots),
    )
    alignment_frame.to_csv(args.out_dir / "alignment_audit.csv", index=False)
    quality_frame.to_csv(args.out_dir / "measurement_quality.csv", index=False)

    contract = data_contract_payload(selected)
    (args.out_dir / "data_contract.json").write_text(
        json.dumps(finite_json(contract), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "figshare": metadata_rows,
        "license": "CC BY 4.0 according to the Figshare article metadata",
        "official_code": {
            "plots": git_provenance(ROOT / "external" / "Cell-Traction-Stress-Velocity-Plots"),
            "processing": git_provenance(ROOT / "external" / "Cell-Traction-Stress"),
        },
        "download_manifest": archive_rows,
    }
    (args.out_dir / "provenance_and_license.json").write_text(
        json.dumps(finite_json(provenance), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.data_root / "manifest.json").write_text(
        json.dumps(
            finite_json(
                {
                    "version": "v150",
                    "articles": metadata_rows,
                    "downloads": archive_rows,
                    "data_contract": contract,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_manifest = {
        "version": "v150",
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "command": sys.argv,
        "selected_articles": selected,
        "data_root": args.data_root,
        "out_dir": args.out_dir,
        "download": bool(args.download),
        "extract": bool(args.extract),
        "verify": bool(args.verify),
        "official_code": {
            "plots": ROOT / "external" / "Cell-Traction-Stress-Velocity-Plots",
            "processing": ROOT / "external" / "Cell-Traction-Stress",
        },
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(finite_json(run_manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(args.out_dir, manifest, inventory_frame, schema_frame, alignment_frame, selected)
    shutil.copyfile(args.out_dir / "v150_intake_report.md", args.out_dir / "intake_report.md")
    print(args.out_dir / "v150_intake_report.md", flush=True)


if __name__ == "__main__":
    main()
