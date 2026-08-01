#!/usr/bin/env python3
"""Prepare a causal track, mask-state, and image manifest for DeepSea v204.

The official DeepSea annotation stores one table per frame.  Labels persist
across frames and division daughters use suffixed labels.  This runner turns
those frame tables into chronological tracks without interpolating gaps and,
when the masks are available, extracts cell-specific state using only the
current and preceding observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6
FRAME_RE = re.compile(r"(\d+)$")
STEM_FRAME_RE = re.compile(r"_z(\d+)_c\d+$", re.IGNORECASE)
FRAME_RATE_RE = re.compile(r"Frame rate\s*=\s*([0-9.]+)\s*min", re.IGNORECASE)
PIXEL_SIZE_RE = re.compile(
    r"Pixel size:\s*([0-9.]+)x([0-9.]+)\s*micron", re.IGNORECASE
)
PUBLISHED_FAMILY_INTERVAL_MIN = {
    "bronchial_epithelial": 5.0,
    "skeletal_muscle": 20.0,
}


@dataclass(frozen=True)
class VideoSpec:
    family: str
    video: str
    sequence: int
    split: str
    root: Path


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_family(path: Path) -> str:
    name = path.name.lower()
    if "bronch" in name:
        return "bronchial_epithelial"
    if "stem" in name or "esc" in name:
        return "embryonic_stem"
    if "muscle" in name or "c2c12" in name:
        return "skeletal_muscle"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def frame_from_feature(path: Path) -> tuple[str, int]:
    stem = path.name.removesuffix("_features.txt")
    stem_match = STEM_FRAME_RE.search(stem)
    match = stem_match if stem_match is not None else FRAME_RE.search(stem)
    if match is None:
        raise ValueError(f"Cannot parse frame from {path}")
    return stem, int(match.group(1))


def video_metadata(spec: VideoSpec, table: pd.DataFrame) -> dict[str, float | str]:
    info_path = spec.root / "info.txt"
    info_text = info_path.read_text(encoding="utf-8", errors="ignore") if info_path.exists() else ""
    frame_match = FRAME_RATE_RE.search(info_text)
    pixel_match = PIXEL_SIZE_RE.search(info_text)
    frame_interval = (
        float(frame_match.group(1))
        if frame_match is not None
        else PUBLISHED_FAMILY_INTERVAL_MIN.get(spec.family, float("nan"))
    )
    pixel_size = float(pixel_match.group(1)) if pixel_match is not None else float("nan")
    first_frame = int(table.frame.min())
    first_diameters = pd.to_numeric(
        table.loc[table.frame == first_frame, "ms_diameter"], errors="coerce"
    )
    first_diameters = first_diameters[np.isfinite(first_diameters) & (first_diameters > 0)]
    if first_diameters.empty:
        raise RuntimeError(f"No valid first-frame diameters in {spec.family}/{spec.video}")
    reference_diameter = float(first_diameters.median())
    return {
        "frame_interval_min": frame_interval,
        "frame_interval_source": "info.txt" if frame_match is not None else "published_family_metadata",
        "pixel_size_um": pixel_size,
        "pixel_size_source": "info.txt" if pixel_match is not None else "unavailable",
        "reference_diameter_px": reference_diameter,
        "reference_diameter_source": "first_frame_median",
    }


def discover_videos(data_root: Path, allow_partial: bool) -> list[VideoSpec]:
    raw: list[tuple[str, str, Path]] = []
    for family_root in sorted(path for path in data_root.iterdir() if path.is_dir()):
        family = canonical_family(family_root)
        for video_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            feature_dir = video_root / "cell_features"
            if not feature_dir.is_dir():
                continue
            count = len(list(feature_dir.glob("*_features.txt")))
            if count < 3:
                if allow_partial:
                    continue
                raise RuntimeError(f"Incomplete feature sequence: {video_root} ({count} frames)")
            raw.append((family, video_root.name, video_root))
    if not raw:
        raise RuntimeError(f"No DeepSea videos found under {data_root}")

    split_lookup: dict[tuple[str, str], str] = {}
    for family in sorted({item[0] for item in raw}):
        names = sorted(item[1] for item in raw if item[0] == family)
        if len(names) < 5 and not allow_partial:
            raise RuntimeError(f"Frozen split needs >=5 videos in {family}; found {len(names)}")
        for rank, name in enumerate(names):
            split_lookup[(family, name)] = "test" if rank % 5 == 0 else "val" if rank % 5 == 1 else "train"

    return [
        VideoSpec(
            family=family,
            video=video,
            sequence=index + 1,
            split=split_lookup[(family, video)],
            root=path,
        )
        for index, (family, video, path) in enumerate(sorted(raw))
    ]


def normalize_feature_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.rename(columns={column: str(column).strip() for column in table.columns})
    aliases = {
        "Label": "track_label",
        "pos_x": "x_px",
        "pos_y": "y_px",
        "Area": "ms_area",
        "MajorAxisLength": "ms_major_axis",
        "MinorAxisLength": "ms_minor_axis",
        "Eccentricity": "ms_eccentricity",
        "Orientation": "ms_orientation_deg",
        "ConvexArea": "ms_convex_area",
        "FilledArea": "ms_filled_area",
        "Diameter": "ms_diameter",
        "Solidity": "ms_solidity",
        "Extent": "ms_extent",
        "Perimeter": "ms_perimeter",
    }
    missing = [column for column in ("Label", "pos_x", "pos_y") if column not in table]
    if missing:
        raise ValueError(f"DeepSea feature table is missing {missing}")
    output = table.rename(columns={key: value for key, value in aliases.items() if key in table}).copy()
    missing_label = output["track_label"].isna() | output["track_label"].astype(str).str.strip().eq("")
    output = output.loc[~missing_label].copy()
    output["track_label"] = output["track_label"].astype(str)
    for column in output.columns:
        if column == "track_label":
            continue
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def companion_paths(video_root: Path, frame_stem: str) -> dict[str, str]:
    candidates = {
        "image_path": video_root / "cell_images" / f"{frame_stem}.png",
        "mask_path": video_root / "cell_masks" / f"{frame_stem}_cell_area_masked.png",
        "nucleus_mask_path": video_root / "nucleus_masks" / f"{frame_stem}_nucleus_area_masked.png",
        "label_path": video_root / "labels" / f"{frame_stem}_cell_pos_labels.txt",
        "labeled_image_path": video_root / "labeled_images" / f"{frame_stem}_cell_area_labeled.png",
    }
    return {key: str(path.resolve()) if path.exists() else "" for key, path in candidates.items()}


def read_video(spec: VideoSpec) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    raw_rows = 0
    dropped_untracked = 0
    frame_files = [
        (*frame_from_feature(path), path)
        for path in (spec.root / "cell_features").glob("*_features.txt")
    ]
    frame_files.sort(key=lambda item: (item[1], item[0]))
    for frame, (frame_stem, source_frame, path) in enumerate(frame_files, start=1):
        raw = pd.read_csv(path, sep="\t")
        raw_rows += len(raw)
        table = normalize_feature_columns(raw)
        dropped_untracked += len(raw) - len(table)
        table["family"] = spec.family
        table["video"] = spec.video
        table["sequence"] = spec.sequence
        table["split"] = spec.split
        table["frame"] = frame
        table["source_frame"] = source_frame
        table["frame_stem"] = frame_stem
        for key, value in companion_paths(spec.root, frame_stem).items():
            table[key] = value
        parts.append(table)
    if not parts:
        raise RuntimeError(f"No feature tables in {spec.root}")
    table = pd.concat(parts, ignore_index=True)
    table["video_raw_detections"] = raw_rows
    table["video_untracked_detections"] = dropped_untracked
    duplicate = table.duplicated(["frame", "track_label"], keep=False)
    if duplicate.any():
        # A repeated identity within one frame cannot define a unique
        # transition. Exclude every member of the ambiguous key so neither
        # coordinate is selected arbitrarily; adjacent observations naturally
        # become separate causal segments.
        ambiguous_identity_detections = int(duplicate.sum())
        table = table.loc[~duplicate].copy()
    else:
        ambiguous_identity_detections = 0
    table["video_ambiguous_identity_detections"] = ambiguous_identity_detections
    labels = sorted(table.track_label.unique())
    label_map = {label: index + 1 for index, label in enumerate(labels)}
    table["track_id"] = table.track_label.map(label_map).astype(np.int64)
    return table.sort_values(["track_id", "frame"]).reset_index(drop=True)


def add_track_dynamics(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    output["dx_px"] = np.nan
    output["dy_px"] = np.nan
    output["target_dx_px"] = np.nan
    output["target_dy_px"] = np.nan
    output["segment_id"] = -1
    output["track_age"] = 0
    output["track_remaining"] = 0
    output["is_entry"] = False
    output["is_exit"] = False
    output["is_gap_boundary"] = False
    segment_counter = 0
    for _, raw_indices in output.groupby("track_id", sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        indices = indices[np.argsort(output.loc[indices, "frame"].to_numpy())]
        frames = output.loc[indices, "frame"].to_numpy(np.int64)
        xy = output.loc[indices, ["x_px", "y_px"]].to_numpy(np.float64)
        boundaries = np.r_[True, np.diff(frames) != 1]
        segment_starts = np.flatnonzero(boundaries)
        segment_stops = np.r_[segment_starts[1:], len(indices)]
        for start, stop in zip(segment_starts, segment_stops):
            seg = indices[start:stop]
            seg_xy = xy[start:stop]
            segment_counter += 1
            output.loc[seg, "segment_id"] = segment_counter
            output.loc[seg, "track_age"] = np.arange(len(seg), dtype=np.int64)
            output.loc[seg, "track_remaining"] = np.arange(len(seg) - 1, -1, -1, dtype=np.int64)
            output.loc[seg[0], "is_entry"] = True
            output.loc[seg[-1], "is_exit"] = True
            if start > 0:
                output.loc[seg[0], "is_gap_boundary"] = True
                output.loc[indices[start - 1], "is_gap_boundary"] = True
            if len(seg) > 1:
                displacement = np.diff(seg_xy, axis=0)
                output.loc[seg[1:], ["dx_px", "dy_px"]] = displacement
                output.loc[seg[:-1], ["target_dx_px", "target_dy_px"]] = displacement

    output["current_interpolated"] = False
    output["state"] = "normal"
    output["is_division_parent"] = False
    output["is_division_daughter"] = output.track_label.str.contains(r"_\d+$", regex=True)
    keys = set(zip(output.frame.astype(int), output.track_label.astype(str)))
    for index, row in output.iterrows():
        children = (int(row.frame) + 1, f"{row.track_label}_1"), (int(row.frame) + 1, f"{row.track_label}_2")
        if children[0] in keys and children[1] in keys:
            output.at[index, "is_division_parent"] = True

    angle = np.deg2rad(pd.to_numeric(output.get("ms_orientation_deg", 0.0), errors="coerce").fillna(0.0))
    output["ms_orientation_cos2"] = np.cos(2.0 * angle)
    output["ms_orientation_sin2"] = np.sin(2.0 * angle)
    if "ms_major_axis" in output and "ms_minor_axis" in output:
        output["ms_axis_ratio"] = output.ms_major_axis / np.maximum(output.ms_minor_axis, EPS)
    if "ms_area" in output and "ms_perimeter" in output:
        output["ms_compactness"] = 4.0 * math.pi * output.ms_area / np.maximum(
            np.square(output.ms_perimeter), EPS
        )
    if "ms_area" in output and "ms_convex_area" in output:
        output["ms_convex_deficit"] = 1.0 - output.ms_area / np.maximum(output.ms_convex_area, EPS)

    dynamic_columns = [
        column
        for column in (
            "ms_area",
            "ms_perimeter",
            "ms_major_axis",
            "ms_minor_axis",
            "ms_eccentricity",
            "ms_solidity",
            "ms_extent",
            "ms_compactness",
            "ms_axis_ratio",
        )
        if column in output
    ]
    for column in dynamic_columns:
        previous = output.groupby("segment_id", sort=False)[column].shift(1)
        output[f"{column}_delta"] = output[column] - previous
        output[f"{column}_relative_delta"] = (output[column] - previous) / np.maximum(
            np.abs(previous), EPS
        )
    speed = np.sqrt(np.square(output.dx_px) + np.square(output.dy_px))
    output["ms_speed_over_diameter"] = speed / np.maximum(output.get("ms_diameter", 1.0), EPS)
    return output


def add_scale_metadata(
    table: pd.DataFrame, spec: VideoSpec
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    output = table.copy()
    metadata = video_metadata(spec, output)
    reference = float(metadata["reference_diameter_px"])
    frame_interval = float(metadata["frame_interval_min"])
    output["frame_interval_min"] = frame_interval
    output["pixel_size_um"] = float(metadata["pixel_size_um"])
    output["reference_diameter_px"] = reference
    output["x_cell_diam"] = output.x_px / reference
    output["y_cell_diam"] = output.y_px / reference
    for column in ("dx_px", "dy_px", "target_dx_px", "target_dy_px"):
        output[column.replace("_px", "_cell_diam")] = output[column] / reference
    hours = frame_interval / 60.0
    for column in ("dx_cell_diam", "dy_cell_diam", "target_dx_cell_diam", "target_dy_cell_diam"):
        output[f"{column}_per_hour"] = output[column] / max(hours, EPS)
    if np.isfinite(float(metadata["pixel_size_um"])):
        pixel_size = float(metadata["pixel_size_um"])
        output["x_um"] = output.x_px * pixel_size
        output["y_um"] = output.y_px * pixel_size
        for column in ("dx_px", "dy_px", "target_dx_px", "target_dy_px"):
            output[column.replace("_px", "_um")] = output[column] * pixel_size
    else:
        for column in ("x_um", "y_um", "dx_um", "dy_um", "target_dx_um", "target_dy_um"):
            output[column] = np.nan

    output["meta_frame_interval_min"] = frame_interval
    output["meta_reference_diameter_px"] = reference
    output["meta_pixel_size_known"] = float(np.isfinite(float(metadata["pixel_size_um"])))
    for family in ("bronchial_epithelial", "embryonic_stem", "skeletal_muscle"):
        output[f"meta_family_{family}"] = float(spec.family == family)
    if "ms_area" in output:
        output["ms_area_reference_diam2"] = output.ms_area / (reference * reference)
    if "ms_perimeter" in output:
        output["ms_perimeter_reference_diam"] = output.ms_perimeter / reference
    if "ms_major_axis" in output:
        output["ms_major_axis_reference_diam"] = output.ms_major_axis / reference
    if "ms_minor_axis" in output:
        output["ms_minor_axis_reference_diam"] = output.ms_minor_axis / reference
    return output, metadata


def component_for_centroid(mask: np.ndarray, x: float, y: float) -> tuple[np.ndarray, int, float]:
    labels, count = ndi.label(mask > 0)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0, float("inf")
    xi = int(np.clip(round(x), 0, mask.shape[1] - 1))
    yi = int(np.clip(round(y), 0, mask.shape[0] - 1))
    label = int(labels[yi, xi])
    distance = 0.0
    if label == 0:
        centers = np.asarray(ndi.center_of_mass(mask > 0, labels, range(1, count + 1)), dtype=float)
        delta = centers - np.asarray([y, x], dtype=float)
        distances = np.sqrt(np.sum(np.square(delta), axis=1))
        nearest = int(np.argmin(distances))
        label = nearest + 1
        distance = float(distances[nearest])
    return labels == label, count, distance


def ray_free_fraction(
    labels: np.ndarray,
    own_label: int,
    center_x: float,
    center_y: float,
    direction: np.ndarray,
    inner: float,
    outer: float,
    half_width: float,
) -> float:
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < EPS:
        return 1.0
    direction /= norm
    normal = np.asarray([-direction[1], direction[0]])
    distances = np.linspace(inner, outer, max(8, int(outer - inner) + 1))
    offsets = np.linspace(-half_width, half_width, max(3, int(half_width * 2) + 1))
    free_until = len(distances)
    for distance_index, distance in enumerate(distances):
        points = (
            np.asarray([center_x, center_y])[None]
            + distance * direction[None]
            + offsets[:, None] * normal[None]
        )
        xx = np.round(points[:, 0]).astype(int)
        yy = np.round(points[:, 1]).astype(int)
        inside = (xx >= 0) & (xx < labels.shape[1]) & (yy >= 0) & (yy < labels.shape[0])
        sampled = labels[yy[inside], xx[inside]]
        occupied = (sampled > 0) & (sampled != own_label)
        if not np.all(inside) or np.any(occupied):
            free_until = distance_index
            break
    return float(free_until / max(len(distances), 1))


def frame_mask_state(frame_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if frame_rows.mask_path.nunique(dropna=False) != 1:
        raise RuntimeError(
            "A mask-state frame group must contain exactly one mask path; "
            f"found {frame_rows.mask_path.nunique(dropna=False)}"
        )
    path_text = str(frame_rows.mask_path.iloc[0])
    if not path_text or not Path(path_text).exists():
        return [{"_index": int(index), "ms_mask_available": 0.0} for index in frame_rows.index]
    image = cv2.imread(path_text, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [{"_index": int(index), "ms_mask_available": 0.0} for index in frame_rows.index]
    full = image > 0
    labels, component_count = ndi.label(full)
    if component_count == 0:
        return [{"_index": int(index), "ms_mask_available": 0.0} for index in frame_rows.index]
    component_slices = ndi.find_objects(labels)
    component_centers = np.asarray(
        ndi.center_of_mass(full, labels, range(1, component_count + 1)),
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    for index, row in frame_rows.iterrows():
        # DeepSea feature centroids originate from MATLAB and are one-based;
        # OpenCV/NumPy image coordinates are zero-based.
        x_value = float(row.x_px) - 1.0
        y_value = float(row.y_px) - 1.0
        xi = int(np.clip(round(x_value), 0, labels.shape[1] - 1))
        yi = int(np.clip(round(y_value), 0, labels.shape[0] - 1))
        own_label = int(labels[yi, xi])
        centroid_distance = 0.0
        if own_label == 0:
            distances = np.sqrt(
                np.sum(
                    np.square(component_centers - np.asarray([y_value, x_value], dtype=float)),
                    axis=1,
                )
            )
            nearest = int(np.argmin(distances))
            own_label = nearest + 1
            centroid_distance = float(distances[nearest])
        component_slice = component_slices[own_label - 1]
        if component_slice is None:
            rows.append({"_index": int(index), "ms_mask_available": 0.0})
            continue
        local_labels = labels[component_slice]
        local_component = local_labels == own_label
        yy_local, xx_local = np.nonzero(local_component)
        yy = yy_local + int(component_slice[0].start)
        xx = xx_local + int(component_slice[1].start)
        center_x = float(np.mean(xx))
        center_y = float(np.mean(yy))
        dx = float(row.dx_px) if np.isfinite(row.dx_px) else 0.0
        dy = float(row.dy_px) if np.isfinite(row.dy_px) else 0.0
        velocity = np.asarray([dx, dy], dtype=float)
        orientation = float(row.get("ms_orientation_deg", 0.0))
        if not np.isfinite(orientation):
            orientation = 0.0
        if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) < EPS:
            angle = math.radians(orientation)
            velocity = np.asarray([math.cos(angle), -math.sin(angle)], dtype=float)
        velocity_norm = float(np.linalg.norm(velocity))
        if not np.isfinite(velocity_norm) or velocity_norm < EPS:
            velocity = np.asarray([1.0, 0.0], dtype=float)
            velocity_norm = 1.0
        velocity /= velocity_norm
        normal = np.asarray([-velocity[1], velocity[0]])
        relative = np.stack([xx - center_x, yy - center_y], axis=1).astype(float)
        # Elementwise projection avoids a spurious Accelerate/BLAS floating
        # warning observed for small two-column arrays on macOS.
        longitudinal = relative[:, 0] * velocity[0] + relative[:, 1] * velocity[1]
        lateral = relative[:, 0] * normal[0] + relative[:, 1] * normal[1]
        diameter = max(float(row.get("ms_diameter", 2.0)), 2.0)
        radius = max(int(math.ceil(2.5 * diameter)), 6)
        x0 = max(0, int(round(center_x)) - radius)
        x1 = min(labels.shape[1], int(round(center_x)) + radius + 1)
        y0 = max(0, int(round(center_y)) - radius)
        y1 = min(labels.shape[0], int(round(center_y)) + radius + 1)
        neighbourhood_labels = labels[y0:y1, x0:x1]
        neighbourhood_component = neighbourhood_labels == own_label
        other = (neighbourhood_labels > 0) & (neighbourhood_labels != own_label)
        dilated = (
            cv2.dilate(
                neighbourhood_component.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            > 0
        )
        contact_ring = dilated & ~neighbourhood_component
        contact = float(np.sum(contact_ring & other) / max(np.sum(contact_ring), 1))
        eroded = (
            cv2.erode(
                neighbourhood_component.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            > 0
        )
        component_boundary = neighbourhood_component & ~eroded
        if np.any(other):
            distance_to_other = cv2.distanceTransform(
                (~other).astype(np.uint8), cv2.DIST_L2, 5
            )
            clearance = distance_to_other[component_boundary]
        else:
            clearance = np.asarray([4.0 * diameter], dtype=float)
        direction_values = {
            "front": velocity,
            "back": -velocity,
            "left": normal,
            "right": -normal,
        }
        state: dict[str, Any] = {
            "_index": int(index),
            "ms_mask_available": 1.0,
            "ms_component_area_px": float(len(xx)),
            "ms_component_count": float(component_count),
            "ms_centroid_component_offset": float(
                math.hypot(center_x - x_value, center_y - y_value)
            ),
            "ms_centroid_to_nearest_component": centroid_distance,
            "ms_front_extent": float(max(np.max(longitudinal), 0.0) / diameter),
            "ms_back_extent": float(max(-np.min(longitudinal), 0.0) / diameter),
            "ms_left_extent": float(max(np.max(lateral), 0.0) / diameter),
            "ms_right_extent": float(max(-np.min(lateral), 0.0) / diameter),
            "ms_front_back_asymmetry": float(
                (np.max(longitudinal) + np.min(longitudinal)) / diameter
            ),
            "ms_left_right_asymmetry": float((np.max(lateral) + np.min(lateral)) / diameter),
            "ms_contact_fraction": contact,
            "ms_neighbor_clearance": float(np.min(clearance) / diameter),
            "ms_neighbor_clearance_q25": float(np.percentile(clearance, 25) / diameter),
            "ms_near_contact_fraction": float(np.mean(clearance <= 0.20 * diameter)),
            "ms_neighbor_mask_density": float(
                np.mean(other)
            ),
            "ms_image_boundary_distance": float(
                min(center_x, center_y, full.shape[1] - 1 - center_x, full.shape[0] - 1 - center_y)
                / diameter
            ),
        }
        for name, direction in direction_values.items():
            state[f"ms_{name}_free_space"] = ray_free_fraction(
                labels,
                own_label,
                center_x,
                center_y,
                direction,
                inner=0.45 * diameter,
                outer=2.0 * diameter,
                half_width=max(1.0, 0.18 * diameter),
            )
        state["ms_mask_area_agreement"] = float(
            1.0
            - min(
                abs(len(xx) - float(row.get("ms_area", len(xx))))
                / max(float(row.get("ms_area", len(xx))), 1.0),
                1.0,
            )
        )
        state["ms_identity_reliability"] = float(
            np.clip(
                state["ms_mask_area_agreement"]
                * math.exp(-centroid_distance / max(0.25 * diameter, 1.0))
                * (1.0 - min(contact, 0.95)),
                0.0,
                1.0,
            )
        )
        rows.append(state)
    return rows


def add_pixel_mask_state(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    records: list[dict[str, Any]] = []
    for _, frame_rows in output.groupby(["video", "frame"], sort=True):
        records.extend(frame_mask_state(frame_rows))
    if not records:
        output["ms_mask_available"] = 0.0
        return output
    state = pd.DataFrame(records).set_index("_index")
    for column in state:
        output.loc[state.index, column] = state[column]
    output["ms_mask_available"] = output.get("ms_mask_available", 0.0).fillna(0.0)
    temporal_columns = [
        column
        for column in (
            "ms_front_extent",
            "ms_back_extent",
            "ms_left_extent",
            "ms_right_extent",
            "ms_front_back_asymmetry",
            "ms_left_right_asymmetry",
            "ms_contact_fraction",
            "ms_neighbor_clearance",
            "ms_near_contact_fraction",
            "ms_neighbor_mask_density",
            "ms_front_free_space",
            "ms_back_free_space",
            "ms_left_free_space",
            "ms_right_free_space",
            "ms_identity_reliability",
        )
        if column in output
    ]
    for column in temporal_columns:
        previous = output.groupby("segment_id", sort=False)[column].shift(1)
        output[f"{column}_delta"] = output[column] - previous
    return output


def split_manifest(videos: Iterable[VideoSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": video.family,
                "video": video.video,
                "sequence": video.sequence,
                "split": video.split,
                "path": str(video.root.resolve()),
            }
            for video in videos
        ]
    )


def dataset_fingerprint(paths: list[Path]) -> str:
    records: list[str] = []
    for path in sorted(paths):
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(ROOT)
        except ValueError:
            display_path = resolved
        records.append(f"{display_path}\t{resolved.stat().st_size}")
    payload = "\n".join(records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> None:
    videos = discover_videos(args.data_root, args.allow_partial)
    if args.max_videos:
        videos = videos[: args.max_videos]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_manifest(videos)
    manifest.to_csv(args.out_dir / "v204_split_manifest.csv", index=False)

    prepared: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    source_files: list[Path] = []
    for video_index, video in enumerate(videos, start=1):
        print(
            f"[{video_index:02d}/{len(videos):02d}] "
            f"{video.family}/{video.video}: preparing tracks and state",
            flush=True,
        )
        table = add_track_dynamics(read_video(video))
        table, metadata = add_scale_metadata(table, video)
        if args.extract_pixel_state:
            table = add_pixel_mask_state(table)
        prepared.append(table)
        source_files.extend(sorted((video.root / "cell_features").glob("*_features.txt")))
        usable = (
            table[["dx_px", "dy_px", "target_dx_px", "target_dy_px"]]
            .notna()
            .all(axis=1)
        )
        audit_rows.append(
            {
                "family": video.family,
                "video": video.video,
                "sequence": video.sequence,
                "split": video.split,
                "frames": int(table.frame.nunique()),
                "observations": int(len(table)),
                "raw_detections": int(table.video_raw_detections.iloc[0]),
                "untracked_detections": int(table.video_untracked_detections.iloc[0]),
                "ambiguous_identity_detections": int(
                    table.video_ambiguous_identity_detections.iloc[0]
                ),
                "labels": int(table.track_label.nunique()),
                "segments": int(table.segment_id.nunique()),
                "usable_h1_rows": int(usable.sum()),
                "division_parent_rows": int(table.is_division_parent.sum()),
                "division_daughter_rows": int(table.is_division_daughter.sum()),
                "gap_boundary_rows": int(table.is_gap_boundary.sum()),
                "image_coverage": float((table.image_path != "").mean()),
                "mask_coverage": float((table.mask_path != "").mean()),
                "pixel_state_coverage": float(table.get("ms_mask_available", pd.Series(0.0)).mean()),
                "frame_interval_min": float(metadata["frame_interval_min"]),
                "frame_interval_source": metadata["frame_interval_source"],
                "pixel_size_um": float(metadata["pixel_size_um"]),
                "pixel_size_source": metadata["pixel_size_source"],
                "reference_diameter_px": float(metadata["reference_diameter_px"]),
                "reference_diameter_source": metadata["reference_diameter_source"],
            }
        )

    tracks = pd.concat(prepared, ignore_index=True)
    tracks = tracks.sort_values(["sequence", "track_id", "frame"]).reset_index(drop=True)
    tracks.to_csv(args.out_dir / "deepsea_tracks.csv", index=False)
    state_columns = [
        column
        for column in tracks.columns
        if column.startswith("ms_")
        and pd.api.types.is_numeric_dtype(tracks[column])
    ]
    meta_columns = [
        column
        for column in tracks.columns
        if column.startswith("meta_") and pd.api.types.is_numeric_dtype(tracks[column])
    ]
    state = tracks[
        ["sequence", "frame", "track_id", "family", "video", "split"]
        + meta_columns
        + state_columns
    ]
    state.to_csv(args.out_dir / "deepsea_state_features.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "v204_data_contract.csv", index=False)

    contract = {
        "dataset": "DeepSea",
        "version": "v204",
        "data_root": str(args.data_root.resolve()),
        "videos": len(videos),
        "families": sorted(manifest.family.unique()),
        "split_counts": manifest.groupby(["family", "split"]).size().reset_index(name="n").to_dict("records"),
        "rows": len(tracks),
        "tracks": int(tracks.groupby(["sequence", "track_id"]).ngroups),
        "state_features": state_columns,
        "metadata_features": meta_columns,
        "extract_pixel_state": bool(args.extract_pixel_state),
        "allow_partial": bool(args.allow_partial),
        "source_feature_fingerprint": dataset_fingerprint(source_files),
        "causal_contract": "all state variables are computed at or before the issue frame",
        "gap_contract": "no interpolation; every non-consecutive interval starts a new segment",
        "division_contract": "parent-to-daughter transitions excluded from ordinary tracks and stratified",
        "split_contract": "within-family lexicographic video rank modulo five, frozen before outcomes",
        "scale_contract": (
            "primary cross-family representation uses the median cell diameter in the "
            "first annotated frame; native pixels are retained and physical microns are "
            "reported only where source pixel calibration is available"
        ),
    }
    (args.out_dir / "v204_preparation_manifest.json").write_text(
        json.dumps(finite_json(contract), indent=2), encoding="utf-8"
    )
    print(json.dumps(finite_json(contract), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "new_data/deepsea_v204")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_multimodal_prepared_v204_2026-07-31",
    )
    parser.add_argument("--extract-pixel-state", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional deterministic prefix used only for smoke/preflight runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
