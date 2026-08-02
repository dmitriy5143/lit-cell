#!/usr/bin/env python3
"""Screen causal cell-state datasets and test an ERK privileged-driver upper bound.

The primary executable candidate is SSBD-248 Figure 4: aligned MDCK DIC,
CFP and FRET time series acquired every ten minutes.  CFP is used to recover
cell masks and persistent identities, FRET/CFP is a privileged current ERK
state, and the target is the next centroid innovation.  Biological acquisition
dates are outer units; positions from one date never cross folds.

The upper-bound experiment is deliberately simple.  A modality must improve a
motion/covariate baseline with Ridge or HGBDT and beat time-shuffled,
wrong-cell and wrong-experiment driver controls before a deployable image
encoder or LaChance transfer is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, t as student_t
from sklearn.ensemble import HistGradientBoostingRegressor
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import (
    closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import watershed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "new_data" / "identified_cell_v171"
DEFAULT_OUT = ROOT / "outputs" / "identified_cell_driver_upper_bound_v171_2026-07-28"
SSBD248_ROOT = (
    "https://ssbd.riken.jp/data/ssbd-000248/repos/Figure4"
)
PIXEL_SIZE_UM = 0.4544
CADENCE_MIN = 10.0
EPS = 1e-8
STUDENT_DF = 5.0


@dataclass(frozen=True)
class FovSpec:
    role: str
    date: str
    position: str

    @property
    def experiment(self) -> str:
        return f"{self.role}_{self.date}_{self.position}"

    @property
    def biological_unit(self) -> str:
        return self.date

    @property
    def relative_dir(self) -> Path:
        return (
            Path("ssbd248")
            / "Figure4"
            / f"HGF_EGFRi_{self.role}"
            / self.date
            / self.position
        )

    def url(self, channel: str) -> str:
        return (
            f"{SSBD248_ROOT}/HGF_EGFRi_{self.role}/"
            f"{self.date}/{self.position}/{channel}.tif"
        )


FOVS = (
    FovSpec("leader", "210223", "posi25"),
    FovSpec("follower", "210223", "posi29"),
    FovSpec("leader", "210224", "posi28"),
    FovSpec("follower", "210224", "posi31"),
    FovSpec("leader", "210226", "posi26"),
    FovSpec("follower", "210226", "posi29"),
    FovSpec("leader", "220224", "posi2"),
    FovSpec("leader", "220313", "posi15"),
)
CHANNELS = ("DIC", "CFP", "FRET")

MOTION_FEATURES = [
    "velocity_x",
    "velocity_y",
    "previous_velocity_x",
    "previous_velocity_y",
    "acceleration_x",
    "acceleration_y",
    "speed",
    "turn_cosine",
    "track_age_log",
]
COVARIATE_FEATURES = [
    "area_log",
    "eccentricity",
    "orientation_cos",
    "orientation_sin",
    "density_log",
    "nearest_distance_log",
    "frame_fraction",
    "role_leader",
    "dic_mean",
    "dic_std",
]
DRIVER_FEATURES = [
    "erk_ratio_mean",
    "erk_ratio_std",
    "erk_delta",
    "erk_gradient_x",
    "erk_gradient_y",
    "erk_front_back",
]
SYNTHETIC_FEATURES = [
    "synthetic_driver_x",
    "synthetic_driver_y",
]


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registry() -> pd.DataFrame:
    rows = [
        {
            "dataset": "SSBD-248 Figure4",
            "citation": "Hino et al., Developmental Cell 2022, doi:10.1016/j.devcel.2022.09.003",
            "url": "https://ssbd.riken.jp/repository/248/",
            "license": "CC BY-NC 4.0",
            "cell_type": "MDCK",
            "intervention": "HGF/EGFR inhibition; leader/follower state",
            "independent_units": "at least five acquisition dates in selected Figure4 subset",
            "channels": "DIC, CFP, FRET ERK biosensor",
            "persistent_identity": "not supplied; recoverable and auditable from CFP masks",
            "driver": "current FRET/CFP ERK state",
            "driver_precedes_target": True,
            "deployable_channels": "DIC only for later inferred-driver encoder",
            "estimated_download_gb": 2.2,
            "priority": 1,
            "screening_status": "selected_for_privileged_upper_bound",
        },
        {
            "dataset": "SSBD-77 CellDynERK",
            "citation": "Aoki et al., Developmental Cell 2017, doi:10.1016/j.devcel.2017.10.016",
            "url": "https://ssbd.riken.jp/database/77-Aoki-CellDynERK/",
            "license": "CC BY",
            "cell_type": "MDCK",
            "intervention": "ERK waves and optogenetic ERK perturbation",
            "independent_units": "14 image datasets, figure-oriented",
            "channels": "ERK FRET, bright field, MLC, optical flow",
            "persistent_identity": "not supplied consistently",
            "driver": "ERK wave field",
            "driver_precedes_target": True,
            "deployable_channels": "bright field",
            "estimated_download_gb": 1.4,
            "priority": 2,
            "screening_status": "field-level fallback; identity contract incomplete",
        },
        {
            "dataset": "BioImage Archive S-BIAD365",
            "citation": "Vaidziulyte et al., eLife 2022, doi:10.7554/eLife.69229",
            "url": "https://www.ebi.ac.uk/biostudies/studies/S-BIAD365",
            "license": "EBI terms / study metadata",
            "cell_type": "RPE1",
            "intervention": "Golgi polarity, protrusion, Cdc42 optogenetics, drugs",
            "independent_units": "multiple tracked single-cell acquisitions; exact count requires file audit",
            "channels": "DIC, nucleus, Golgi, Rab6, Cdc42/optogenetic channels",
            "persistent_identity": "tracking-aware acquisition described; labels require audit",
            "driver": "nucleus-Golgi axis / protrusion / optogenetic Cdc42",
            "driver_precedes_target": True,
            "deployable_channels": "DIC and morphology channels by declared deployment",
            "estimated_download_gb": np.nan,
            "priority": 2,
            "screening_status": "high-value non-collective follow-up",
        },
        {
            "dataset": "GigaScience wound healing",
            "citation": "Zaritsky et al., GigaScience 2015, doi:10.1186/s13742-015-0049-6",
            "url": "https://doi.org/10.1186/s13742-015-0049-6",
            "license": "source-specific open data",
            "cell_type": "MDCK and DA3",
            "intervention": "control/HGF/PHA",
            "independent_units": "31 experiments",
            "channels": "DIC, ROI and dense motion fields",
            "persistent_identity": "not available",
            "driver": "completed collective motion field, not a new causal driver",
            "driver_precedes_target": True,
            "deployable_channels": "tracks or images through a separate bridge",
            "estimated_download_gb": 7.7,
            "priority": 3,
            "screening_status": "native field benchmark; not identified-cell driver",
        },
        {
            "dataset": "v150 force-motion MDCK corpus",
            "citation": "Spatiotemporal force and motion in collective cell migration",
            "url": "https://figshare.com/collections/Spatiotemporal_force_and_motion_in_collective_cell_migration/4945206",
            "license": "source metadata",
            "cell_type": "MDCK",
            "intervention": "measured traction/stress",
            "independent_units": "six films used in completed upper bound",
            "channels": "velocity, traction, stress",
            "persistent_identity": "field-level",
            "driver": "traction/stress",
            "driver_precedes_target": "audited",
            "deployable_channels": "none of privileged force channels in LaChance",
            "estimated_download_gb": 3.0,
            "priority": 4,
            "screening_status": "closed negative: incremental gain below zero",
        },
        {
            "dataset": "C2C12 manual/automatic tracks",
            "citation": "Ker et al., Scientific Data 2018, doi:10.1038/sdata.2018.237",
            "url": "https://www.nature.com/articles/sdata2018237",
            "license": "CC BY",
            "cell_type": "C2C12",
            "intervention": "manual versus automatic measurement process",
            "independent_units": "three experiment groups / 48 fields",
            "channels": "phase contrast and paired tracks",
            "persistent_identity": "yes",
            "driver": "measurement reliability, not biological mechanics",
            "driver_precedes_target": True,
            "deployable_channels": "single track stream",
            "estimated_download_gb": np.nan,
            "priority": 3,
            "screening_status": "completed reliability gate v168",
        },
        {
            "dataset": "Allen Cell EMT timelapse",
            "citation": "Allen Institute, Nature Methods 2026 EMT timelapse dataset",
            "url": "https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/",
            "license": "dataset-specific open terms",
            "cell_type": "epithelial EMT model",
            "intervention": "EMT induction",
            "independent_units": "multiple colonies / wells",
            "channels": "OME-Zarr images, segmentations, morphology features",
            "persistent_identity": "tracked nuclei available for subsets",
            "driver": "state transition / morphology, no direct force driver",
            "driver_precedes_target": "potential",
            "deployable_channels": "raw microscopy",
            "estimated_download_gb": np.nan,
            "priority": 3,
            "screening_status": "representation pretraining candidate",
        },
        {
            "dataset": "CTMC-v1 / DeepSea / WHAD-CAMAD",
            "citation": "tracking and segmentation benchmark collections",
            "url": "https://motchallenge.net/data/CTMC-v1/",
            "license": "dataset-specific",
            "cell_type": "multiple",
            "intervention": "mixed",
            "independent_units": "many videos",
            "channels": "phase contrast, masks or bounding-box tracks",
            "persistent_identity": "yes in CTMC; variable elsewhere",
            "driver": "no explicit pre-motion causal driver",
            "driver_precedes_target": False,
            "deployable_channels": "raw microscopy",
            "estimated_download_gb": np.nan,
            "priority": 4,
            "screening_status": "tracking/segmentation pretraining only",
        },
    ]
    return pd.DataFrame(rows)


def download_file(url: str, destination: Path, timeout: int = 120) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "status": "reused",
            "sha256": sha256_file(destination),
        }
    partial = destination.with_suffix(destination.suffix + ".part")
    headers: dict[str, str] = {}
    mode = "wb"
    if partial.is_file() and partial.stat().st_size:
        headers["Range"] = f"bytes={partial.stat().st_size}-"
        mode = "ab"
    with requests.get(
        url,
        stream=True,
        timeout=timeout,
        headers=headers,
    ) as response:
        response.raise_for_status()
        if mode == "ab" and response.status_code != 206:
            mode = "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(1 << 20):
                if block:
                    handle.write(block)
    partial.replace(destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "status": "downloaded",
        "sha256": sha256_file(destination),
    }


def download_selected(data_root: Path, smoke: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    fovs = FOVS[:2] if smoke else FOVS
    for fov in fovs:
        for channel in CHANNELS:
            destination = data_root / fov.relative_dir / f"{channel}.tif"
            record = download_file(fov.url(channel), destination)
            record.update(
                {
                    "experiment": fov.experiment,
                    "biological_unit": fov.biological_unit,
                    "role": fov.role,
                    "channel": channel,
                }
            )
            records.append(record)
            print(
                f"[v171] {fov.experiment}/{channel}: {record['status']}",
                flush=True,
            )
    return pd.DataFrame(records)


def inventory_selected(data_root: Path, smoke: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    fovs = FOVS[:2] if smoke else FOVS
    for fov in fovs:
        for channel in CHANNELS:
            path = data_root / fov.relative_dir / f"{channel}.tif"
            exists = path.is_file() and path.stat().st_size > 0
            records.append(
                {
                    "url": fov.url(channel),
                    "path": str(path),
                    "bytes": path.stat().st_size if exists else 0,
                    "status": "cached" if exists else "missing",
                    "sha256": sha256_file(path) if exists else "",
                    "experiment": fov.experiment,
                    "biological_unit": fov.biological_unit,
                    "role": fov.role,
                    "channel": channel,
                }
            )
    return pd.DataFrame(records)


def robust_ratio(fret: np.ndarray, cfp: np.ndarray) -> np.ndarray:
    background = float(np.quantile(cfp, 0.05))
    denominator = np.maximum(cfp.astype(np.float32) - background, 20.0)
    numerator = np.maximum(fret.astype(np.float32) - float(np.quantile(fret, 0.05)), 0.0)
    ratio = numerator / denominator
    return np.clip(ratio, 0.0, float(np.quantile(ratio, 0.999))).astype(np.float32)


def segment_frame(image: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    value = image.astype(np.float32)
    low, high = np.quantile(value, [0.02, 0.995])
    normalized = np.clip((value - low) / max(high - low, 1.0), 0.0, 1.0)
    smooth = cv2.GaussianBlur(normalized, (0, 0), 2.0)
    threshold, _ = cv2.threshold(
        (smooth * 255).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binary = smooth > max(float(threshold) / 255.0, 0.08)
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(min_area - 1, 0))
    binary = remove_small_holes(binary, max_size=max(min_area, 500))
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    coordinates = peak_local_max(
        distance,
        min_distance=12,
        threshold_abs=3.0,
        labels=binary,
        exclude_border=False,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    for marker, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = marker
    if not np.any(markers):
        return np.zeros(binary.shape, dtype=np.int32)
    labels = watershed(-distance, markers, mask=binary)
    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for region in regionprops(labels):
        if min_area <= region.area <= max_area:
            output[labels == region.label] = next_label
            next_label += 1
    return output


def overlap_iou(previous: np.ndarray, current: np.ndarray) -> float:
    intersection = np.logical_and(previous, current).sum()
    union = np.logical_or(previous, current).sum()
    return float(intersection / max(union, 1))


def track_instances(
    label_frames: list[np.ndarray],
    max_distance: float,
) -> list[dict[int, int]]:
    frame_maps: list[dict[int, int]] = []
    next_track = 1
    previous_objects: dict[int, tuple[int, np.ndarray, np.ndarray, float]] = {}
    for frame_index, labels in enumerate(label_frames):
        objects: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for region in regionprops(labels):
            centroid = np.asarray(region.centroid[::-1], dtype=np.float64)
            mask = labels == region.label
            objects.append((region.label, centroid, mask, float(region.area)))
        mapping: dict[int, int] = {}
        if previous_objects and objects:
            previous_ids = sorted(previous_objects)
            cost = np.full((len(previous_ids), len(objects)), 1e6, dtype=np.float64)
            for left_index, track_id in enumerate(previous_ids):
                _label, old_centroid, old_mask, old_area = previous_objects[track_id]
                for right_index, (_new_label, centroid, mask, area) in enumerate(objects):
                    distance = float(np.linalg.norm(centroid - old_centroid))
                    if distance > max_distance:
                        continue
                    area_cost = abs(math.log(max(area, 1.0) / max(old_area, 1.0)))
                    iou = overlap_iou(old_mask, mask)
                    cost[left_index, right_index] = (
                        distance / max_distance + 0.25 * area_cost + 0.50 * (1.0 - iou)
                    )
            left_indices, right_indices = linear_sum_assignment(cost)
            for left_index, right_index in zip(left_indices, right_indices):
                if cost[left_index, right_index] <= 2.0:
                    mapping[objects[right_index][0]] = previous_ids[left_index]
        for label, _centroid, _mask, _area in objects:
            if label not in mapping:
                mapping[label] = next_track
                next_track += 1
        previous_objects = {
            mapping[label]: (label, centroid, mask, area)
            for label, centroid, mask, area in objects
        }
        frame_maps.append(mapping)
    return frame_maps


def object_table(
    fov: FovSpec,
    data_root: Path,
    min_area: int,
    max_area: int,
    max_distance: float,
    maximum_frames: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = data_root / fov.relative_dir
    paths = {channel: directory / f"{channel}.tif" for channel in CHANNELS}
    if not all(path.is_file() for path in paths.values()):
        return pd.DataFrame(), {
            "experiment": fov.experiment,
            "status": "missing_channels",
        }
    stacks = {
        channel: tifffile.imread(path)
        for channel, path in paths.items()
    }
    frame_count = min(len(stack) for stack in stacks.values())
    if maximum_frames > 0:
        frame_count = min(frame_count, maximum_frames)
    labels = [
        segment_frame(stacks["CFP"][frame], min_area, max_area)
        for frame in range(frame_count)
    ]
    mappings = track_instances(labels, max_distance)
    records: list[dict[str, Any]] = []
    for frame in range(frame_count):
        cfp = stacks["CFP"][frame]
        fret = stacks["FRET"][frame]
        dic = stacks["DIC"][frame]
        ratio = robust_ratio(fret, cfp)
        regions = regionprops(labels[frame])
        centroids = np.asarray(
            [region.centroid[::-1] for region in regions],
            dtype=np.float64,
        )
        for region_index, region in enumerate(regions):
            mask = labels[frame] == region.label
            y, x = np.nonzero(mask)
            centroid_x, centroid_y = region.centroid[::-1]
            if len(centroids) > 1:
                distances = np.linalg.norm(
                    centroids - np.asarray([centroid_x, centroid_y]),
                    axis=1,
                )
                nonzero = distances[distances > EPS]
                nearest = float(nonzero.min()) if len(nonzero) else max_distance
                density = float(np.sum((distances > EPS) & (distances <= 100.0)))
            else:
                nearest = max_distance
                density = 0.0
            centered_x = x.astype(np.float64) - centroid_x
            centered_y = y.astype(np.float64) - centroid_y
            design = np.column_stack(
                [np.ones(len(x)), centered_x, centered_y]
            )
            ratio_values = ratio[mask].astype(np.float64)
            coefficient, *_ = np.linalg.lstsq(design, ratio_values, rcond=None)
            records.append(
                {
                    "experiment": fov.experiment,
                    "biological_unit": fov.biological_unit,
                    "role": fov.role,
                    "frame": frame,
                    "track_id": mappings[frame][region.label],
                    "x_px": float(centroid_x),
                    "y_px": float(centroid_y),
                    "area": float(region.area),
                    "eccentricity": float(region.eccentricity),
                    "orientation": float(region.orientation),
                    "density_100px": density,
                    "nearest_distance_px": nearest,
                    "dic_mean": float(np.mean(dic[mask])),
                    "dic_std": float(np.std(dic[mask])),
                    "erk_ratio_mean": float(np.mean(ratio_values)),
                    "erk_ratio_std": float(np.std(ratio_values)),
                    "erk_gradient_x": float(coefficient[1]),
                    "erk_gradient_y": float(coefficient[2]),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame, {
            "experiment": fov.experiment,
            "status": "segmentation_empty",
        }
    track_lengths = frame.groupby("track_id").size()
    long_tracks = track_lengths[track_lengths >= 8].index
    frame = frame[frame.track_id.isin(long_tracks)].reset_index(drop=True)
    audit = {
        "experiment": fov.experiment,
        "biological_unit": fov.biological_unit,
        "role": fov.role,
        "status": "ok" if len(frame) else "no_tracks_after_filter",
        "frames": frame_count,
        "objects": len(records),
        "retained_objects": len(frame),
        "tracks": int(frame.track_id.nunique()),
        "median_track_length": (
            float(frame.groupby("track_id").size().median()) if len(frame) else 0.0
        ),
        "mean_cells_per_frame": len(records) / max(frame_count, 1),
        "source_sha256": {
            channel: sha256_file(path)
            for channel, path in paths.items()
        },
    }
    return frame, audit


def forecast_table(objects: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (_experiment, _track), group in objects.groupby(
        ["experiment", "track_id"], sort=False
    ):
        group = group.sort_values("frame").reset_index(drop=True)
        by_frame = {int(row.frame): index for index, row in group.iterrows()}
        for index, row in group.iterrows():
            previous_index = by_frame.get(int(row.frame) - 1)
            previous2_index = by_frame.get(int(row.frame) - 2)
            next_index = by_frame.get(int(row.frame) + 1)
            if previous_index is None or next_index is None:
                continue
            previous = group.iloc[previous_index]
            following = group.iloc[next_index]
            velocity = np.asarray(
                [row.x_px - previous.x_px, row.y_px - previous.y_px],
                dtype=np.float64,
            )
            if previous2_index is None:
                previous_velocity = velocity.copy()
            else:
                previous2 = group.iloc[previous2_index]
                previous_velocity = np.asarray(
                    [
                        previous.x_px - previous2.x_px,
                        previous.y_px - previous2.y_px,
                    ]
                )
            target = np.asarray(
                [following.x_px - row.x_px, following.y_px - row.y_px]
            )
            speed = float(np.linalg.norm(velocity))
            previous_speed = float(np.linalg.norm(previous_velocity))
            turn_cosine = float(
                np.dot(velocity, previous_velocity)
                / max(speed * previous_speed, EPS)
            )
            if speed > EPS:
                unit = velocity / speed
                gradient = np.asarray(
                    [row.erk_gradient_x, row.erk_gradient_y]
                )
                front_back = float(np.dot(gradient, unit))
            else:
                front_back = 0.0
            record = row.to_dict()
            record.update(
                {
                    "velocity_x": velocity[0],
                    "velocity_y": velocity[1],
                    "previous_velocity_x": previous_velocity[0],
                    "previous_velocity_y": previous_velocity[1],
                    "acceleration_x": velocity[0] - previous_velocity[0],
                    "acceleration_y": velocity[1] - previous_velocity[1],
                    "speed": speed,
                    "turn_cosine": turn_cosine,
                    "track_age_log": math.log1p(index),
                    "causal_area_log_change": abs(
                        math.log(
                            max(float(row.area), 1.0)
                            / max(float(previous.area), 1.0)
                        )
                    ),
                    "target_area_log_change": abs(
                        math.log(
                            max(float(following.area), 1.0)
                            / max(float(row.area), 1.0)
                        )
                    ),
                    "area_log": math.log1p(row.area),
                    "orientation_cos": math.cos(float(row.orientation)),
                    "orientation_sin": math.sin(float(row.orientation)),
                    "density_log": math.log1p(row.density_100px),
                    "nearest_distance_log": math.log1p(row.nearest_distance_px),
                    "frame_fraction": row.frame / max(group.frame.max(), 1),
                    "role_leader": float(row.role == "leader"),
                    "erk_delta": row.erk_ratio_mean - previous.erk_ratio_mean,
                    "erk_front_back": front_back,
                    "target_dx": target[0],
                    "target_dy": target[1],
                    "target_step_norm": float(np.linalg.norm(target)),
                    "innovation_x": target[0] - velocity[0],
                    "innovation_y": target[1] - velocity[1],
                }
            )
            records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    finite_columns = (
        MOTION_FEATURES
        + COVARIATE_FEATURES
        + DRIVER_FEATURES
        + ["target_dx", "target_dy", "innovation_x", "innovation_y"]
    )
    valid = np.isfinite(result[finite_columns].to_numpy(np.float64)).all(axis=1)
    return result.loc[valid].reset_index(drop=True)


def controlled_driver(
    frame: pd.DataFrame,
    control: str,
    seed: int,
) -> np.ndarray:
    values = frame[DRIVER_FEATURES].to_numpy(np.float64, copy=True)
    if control == "real":
        return values
    rng = np.random.default_rng(seed)
    if control == "zero":
        return np.zeros_like(values)
    if control == "time_shuffled":
        order = np.arange(len(frame))
        for (_experiment, _track), indices in frame.groupby(
            ["experiment", "track_id"]
        ).indices.items():
            indices = np.asarray(indices)
            order[indices] = np.roll(indices, max(1, len(indices) // 3))
        return values[order]
    if control == "wrong_cell":
        order = np.arange(len(frame))
        for (_experiment, _time), indices in frame.groupby(
            ["experiment", "frame"]
        ).indices.items():
            indices = np.asarray(indices)
            if len(indices) > 1:
                order[indices] = np.roll(indices, 1)
        return values[order]
    if control == "wrong_experiment":
        order = np.arange(len(frame))
        experiments = sorted(frame.experiment.unique())
        replacement = {
            name: experiments[(index + 1) % len(experiments)]
            for index, name in enumerate(experiments)
        }
        for experiment, indices in frame.groupby("experiment").indices.items():
            donor = frame.index[frame.experiment.eq(replacement[experiment])].to_numpy()
            if len(donor):
                indices = np.asarray(indices)
                order[indices] = rng.choice(donor, len(indices), replace=True)
        return values[order]
    raise ValueError(control)


class VectorRegressor:
    def __init__(self, family: str, seed: int) -> None:
        self.family = family
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.ridge_coef: np.ndarray | None = None
        self.ridge_intercept: np.ndarray | None = None
        if family == "ridge":
            self.models: Any = None
        elif family == "hgbdt":
            self.models = [
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=160,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=3.0,
                    early_stopping=True,
                    random_state=seed + component,
                )
                for component in range(2)
            ]
        else:
            raise ValueError(family)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        self.feature_mean = np.mean(x64, axis=0)
        self.feature_scale = np.std(x64, axis=0)
        self.feature_scale = np.where(
            self.feature_scale > 1e-8,
            self.feature_scale,
            1.0,
        )
        normalized = (x64 - self.feature_mean) / self.feature_scale
        if not np.isfinite(normalized).all():
            raise RuntimeError("Driver normalization produced non-finite values")
        if self.family == "ridge":
            self.ridge_intercept = np.mean(y64, axis=0)
            centered_target = y64 - self.ridge_intercept
            feature_count = normalized.shape[1]
            gram = np.empty((feature_count, feature_count), dtype=np.float64)
            for left in range(feature_count):
                for right in range(left, feature_count):
                    value = np.sum(
                        normalized[:, left] * normalized[:, right],
                        dtype=np.float64,
                    )
                    gram[left, right] = value
                    gram[right, left] = value
            rhs = np.empty((feature_count, centered_target.shape[1]), dtype=np.float64)
            for feature_index in range(feature_count):
                for component in range(centered_target.shape[1]):
                    rhs[feature_index, component] = np.sum(
                        normalized[:, feature_index]
                        * centered_target[:, component],
                        dtype=np.float64,
                    )
            self.ridge_coef = np.linalg.solve(
                gram + 100.0 * np.eye(feature_count, dtype=np.float64),
                rhs,
            )
        else:
            for component, model in enumerate(self.models):
                model.fit(normalized, y64[:, component])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Driver model has not been fitted")
        normalized = (
            np.asarray(x, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        if self.family == "ridge":
            if self.ridge_coef is None or self.ridge_intercept is None:
                raise RuntimeError("Ridge driver model has not been fitted")
            return np.column_stack(
                [
                    np.sum(
                        normalized * self.ridge_coef[:, component],
                        axis=1,
                        dtype=np.float64,
                    )
                    + self.ridge_intercept[component]
                    for component in range(self.ridge_coef.shape[1])
                ]
            )
        return np.column_stack(
            [model.predict(normalized) for model in self.models]
        )


def metrics(target: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    error = target - prediction
    component_rmse = float(np.sqrt(np.mean(np.square(error))))
    centered = target - target.mean(axis=0, keepdims=True)
    r2 = float(
        1.0
        - np.sum(np.square(error))
        / max(float(np.sum(np.square(centered))), EPS)
    )
    standardized = error / max(scale, 1e-4)
    nll = -float(
        np.mean(
            student_t.logpdf(standardized, STUDENT_DF)
            - math.log(max(scale, 1e-4))
        )
    )
    q50 = float(student_t.ppf(0.75, STUDENT_DF))
    q90 = float(student_t.ppf(0.95, STUDENT_DF))
    return {
        "component_rmse": component_rmse,
        "vector_rmse": float(
            np.sqrt(np.mean(np.sum(np.square(error), axis=1)))
        ),
        "r2": r2,
        "student_t_nll": nll,
        "coverage_50": float(np.mean(np.abs(error) <= q50 * scale)),
        "coverage_90": float(np.mean(np.abs(error) <= q90 * scale)),
    }


def feature_matrix(
    frame: pd.DataFrame,
    packet: str,
    control: str,
    seed: int,
) -> np.ndarray:
    columns = MOTION_FEATURES.copy()
    if packet in {"covariates", "driver", "synthetic_capacity"}:
        columns += COVARIATE_FEATURES
    base = frame[columns].to_numpy(np.float64)
    if packet == "driver":
        base = np.column_stack(
            [base, controlled_driver(frame, control, seed)]
        )
    elif packet == "synthetic_capacity":
        base = np.column_stack(
            [base, frame[SYNTHETIC_FEATURES].to_numpy(np.float64)]
        )
    return base


def upper_bound_loeo(
    frame: pd.DataFrame,
    seed: int,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    all_units = sorted(frame.biological_unit.unique())
    split_records: list[
        tuple[str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ] = []
    if smoke and len(all_units) < 3:
        ordered = frame.sort_values(
            ["frame", "experiment", "track_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        train_end = max(20, int(0.60 * len(ordered)))
        validation_end = max(train_end + 20, int(0.80 * len(ordered)))
        split_records.append(
            (
                "smoke_temporal_diagnostic",
                "smoke_temporal_test",
                "smoke_temporal_validation",
                ordered.iloc[:train_end].reset_index(drop=True),
                ordered.iloc[train_end:validation_end].reset_index(drop=True),
                ordered.iloc[validation_end:].reset_index(drop=True),
            )
        )
    else:
        test_units = all_units[: min(2, len(all_units))] if smoke else all_units
        for test_unit in test_units:
            validation_unit = next(
                unit for unit in all_units if unit != test_unit
            )
            split_records.append(
                (
                    "leave_acquisition_date_out",
                    str(test_unit),
                    str(validation_unit),
                    frame[
                        ~frame.biological_unit.isin(
                            [test_unit, validation_unit]
                        )
                    ].reset_index(drop=True),
                    frame[
                        frame.biological_unit.eq(validation_unit)
                    ].reset_index(drop=True),
                    frame[
                        frame.biological_unit.eq(test_unit)
                    ].reset_index(drop=True),
                )
            )
    for fold, (
        split_protocol,
        test_unit,
        validation_unit,
        train,
        validation,
        test,
    ) in enumerate(split_records):
        if min(len(train), len(validation), len(test)) < 20:
            continue
        target_train = train[["innovation_x", "innovation_y"]].to_numpy(np.float64)
        target_validation = validation[["target_dx", "target_dy"]].to_numpy(np.float64)
        target_test = test[["target_dx", "target_dy"]].to_numpy(np.float64)
        velocity_validation = validation[["velocity_x", "velocity_y"]].to_numpy(np.float64)
        velocity_test = test[["velocity_x", "velocity_y"]].to_numpy(np.float64)
        for family in ("ridge", "hgbdt"):
            for packet in (
                "motion",
                "covariates",
                "driver",
                "synthetic_capacity",
            ):
                controls = (
                    ("real",)
                    if packet != "driver"
                    else ("real", "zero", "time_shuffled", "wrong_cell", "wrong_experiment")
                )
                for control_index, control in enumerate(controls):
                    model = VectorRegressor(family, seed + fold * 101 + control_index)
                    model.fit(
                        feature_matrix(
                            train,
                            packet,
                            control,
                            seed + fold * 101 + control_index,
                        ),
                        target_train,
                    )
                    validation_prediction = velocity_validation + model.predict(
                        feature_matrix(
                            validation,
                            packet,
                            control,
                            seed + fold * 211 + control_index,
                        )
                    )
                    scale = max(
                        float(
                            np.quantile(
                                np.abs(target_validation - validation_prediction),
                                0.68,
                            )
                        ),
                        1e-3,
                    )
                    prediction = velocity_test + model.predict(
                        feature_matrix(
                            test,
                            packet,
                            control,
                            seed + fold * 307 + control_index,
                        )
                    )
                    record = {
                        "fold": fold,
                        "split_protocol": split_protocol,
                        "test_biological_unit": test_unit,
                        "validation_biological_unit": validation_unit,
                        "family": family,
                        "packet": packet,
                        "control": control,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "predictive_scale": scale,
                        "deployable": packet != "synthetic_capacity",
                        "future_or_target_feature_used": (
                            packet == "synthetic_capacity"
                        ),
                        **metrics(target_test, prediction, scale),
                    }
                    metric_records.append(record)
                    if packet == "driver":
                        control_records.append(record.copy())
        print(
            f"[v171] upper-bound fold {fold + 1}/{len(split_records)} "
            f"test={test_unit}",
            flush=True,
        )
    return pd.DataFrame(metric_records), pd.DataFrame(control_records)


def privileged_cluster_inference(
    metrics_frame: pd.DataFrame,
    seed: int,
    draws: int = 10_000,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for family in sorted(metrics_frame.family.unique()):
        family_frame = metrics_frame[metrics_frame.family.eq(family)]
        pivot = family_frame[
            family_frame.control.eq("real")
            & family_frame.packet.isin(["covariates", "driver"])
        ].pivot_table(
            index="test_biological_unit",
            columns="packet",
            values="component_rmse",
            aggfunc="first",
        ).dropna(subset=["covariates", "driver"])
        delta = (
            pivot["covariates"] - pivot["driver"]
        ).to_numpy(np.float64)
        bootstrap = np.asarray(
            [
                np.mean(rng.choice(delta, len(delta), replace=True))
                for _ in range(draws)
            ]
        )
        positive = int(np.sum(delta > 0))
        records.append(
            {
                "family": family,
                "outer_units": len(delta),
                "mean_rmse_gain": float(np.mean(delta)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "positive_units": positive,
                "exact_sign_pvalue": float(
                    binomtest(
                        positive,
                        len(delta),
                        p=0.5,
                        alternative="greater",
                    ).pvalue
                ),
            }
        )
    result = pd.DataFrame(records)
    if len(result):
        order = np.argsort(result.exact_sign_pvalue.to_numpy(np.float64))
        adjusted = np.empty(len(result), dtype=np.float64)
        running = 0.0
        for rank, index in enumerate(order):
            value = min(
                1.0,
                (len(result) - rank)
                * float(result.iloc[index].exact_sign_pvalue),
            )
            running = max(running, value)
            adjusted[index] = running
        result["holm_adjusted_pvalue"] = adjusted
    return result


def domain_support(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    source_speed = frame.speed.to_numpy(np.float64) * PIXEL_SIZE_UM / CADENCE_MIN
    lachance_rows = sorted(
        (
            ROOT / "outputs" / "lachance_track_native_v160_development_cache"
        ).glob("**/rows.csv")
    )
    target_speed_blocks = []
    seen: set[str] = set()
    for path in lachance_rows:
        key = sha256_file(path)
        if key in seen:
            continue
        seen.add(key)
        rows = pd.read_csv(path, usecols=["dx_px", "dy_px"])
        target_speed_blocks.append(
            np.linalg.norm(rows[["dx_px", "dy_px"]].to_numpy(np.float64), axis=1)
        )
    target_speed = (
        np.concatenate(target_speed_blocks)
        if target_speed_blocks
        else np.asarray([], dtype=np.float64)
    )
    source_normalized = source_speed / max(float(np.median(source_speed)), EPS)
    target_normalized = (
        target_speed / max(float(np.median(target_speed)), EPS)
        if len(target_speed)
        else target_speed
    )
    source_interval = np.quantile(source_normalized, [0.01, 0.99])
    coverage = (
        float(
            np.mean(
                (target_normalized >= source_interval[0])
                & (target_normalized <= source_interval[1])
            )
        )
        if len(target_normalized)
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "source_dataset": "SSBD-248 Figure4 selected subset",
                "target_dataset": "LaChance MDCK_Bulk track-native",
                "source_rows": len(source_speed),
                "target_rows": len(target_speed),
                "source_cadence_min": CADENCE_MIN,
                "source_pixel_size_um": PIXEL_SIZE_UM,
                "dimensionless_speed_support_low": source_interval[0],
                "dimensionless_speed_support_high": source_interval[1],
                "target_speed_coverage": coverage,
                "driver_channel_available_in_target": False,
                "zero_shot_driver_transfer_allowed": False,
            }
        ]
    )


def write_screening_report(output: Path, registry_frame: pd.DataFrame) -> None:
    lines = [
        "# Identified-Cell Causal Modality Screening",
        "",
        "The screen distinguishes a true pre-motion driver from segmentation or tracking data that only improve representation quality.",
        "",
        registry_frame[
            [
                "dataset",
                "cell_type",
                "driver",
                "persistent_identity",
                "priority",
                "screening_status",
            ]
        ].to_markdown(index=False),
        "",
        "SSBD-248 Figure 4 is the first executable privileged-driver candidate because it combines MDCK motion, current ERK FRET state and several acquisition dates. Persistent masks/tracks are reconstructed and audited; they are not assumed from file names.",
        "",
        "SSBD-77 and S-BIAD365 remain follow-ups. GigaScience has strong field supervision but no persistent identity, C2C12 identifies measurement reliability rather than a biological driver, and the completed v150 force-motion upper bound remains a negative result.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    registry_frame = registry()
    registry_frame.to_csv(
        args.out_dir / "identified_cell_modality_registry.csv",
        index=False,
    )
    write_screening_report(
        args.out_dir / "identified_cell_dataset_screening_report.md",
        registry_frame,
    )
    download_manifest = (
        download_selected(args.data_root, bool(args.smoke))
        if args.download
        else inventory_selected(args.data_root, bool(args.smoke))
    )
    download_manifest.to_csv(
        args.out_dir / "identified_cell_download_manifest.csv",
        index=False,
    )

    object_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    cache_dir = args.data_root / "ssbd248_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected_fovs = FOVS[:2] if args.smoke else FOVS
    if args.run_upper_bound:
        for fov in selected_fovs:
            cache_path = cache_dir / f"{fov.experiment}.csv"
            audit_path = cache_dir / f"{fov.experiment}.audit.json"
            if cache_path.is_file() and audit_path.is_file() and not args.force_features:
                frame = pd.read_csv(cache_path)
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["cache_status"] = "reused"
            else:
                frame, audit = object_table(
                    fov,
                    args.data_root,
                    int(args.min_area),
                    int(args.max_area),
                    float(args.max_track_distance),
                    int(args.maximum_frames),
                )
                if len(frame):
                    frame.to_csv(cache_path, index=False)
                audit_path.write_text(
                    json.dumps(finite(audit), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                audit["cache_status"] = "built"
            audit_rows.append(audit)
            if len(frame):
                object_frames.append(frame)
            print(
                f"[v171] features {fov.experiment}: {audit['status']}",
                flush=True,
            )
    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(
        args.out_dir / "identified_cell_tracking_audit.csv",
        index=False,
    )

    forecast = (
        forecast_table(pd.concat(object_frames, ignore_index=True))
        if object_frames
        else pd.DataFrame()
    )
    if len(forecast):
        rng = np.random.default_rng(int(args.seed) + 171_001)
        innovation = forecast[
            ["innovation_x", "innovation_y"]
        ].to_numpy(np.float64)
        noise_scale = np.maximum(np.std(innovation, axis=0), 1e-3) * 0.20
        synthetic = innovation + rng.normal(
            0.0,
            noise_scale,
            innovation.shape,
        )
        forecast[SYNTHETIC_FEATURES] = synthetic
        forecast.to_csv(
            args.out_dir / "identified_cell_forecast_table.csv",
            index=False,
        )
    metrics_frame, controls_frame = (
        upper_bound_loeo(forecast, int(args.seed), bool(args.smoke))
        if len(forecast)
        else (pd.DataFrame(), pd.DataFrame())
    )
    quality_frames: list[pd.DataFrame] = []
    if len(forecast) and not args.smoke:
        quality_subsets = {
            "causal_stable_tracking": forecast[
                forecast.speed.le(15.0)
                & forecast.causal_area_log_change.le(0.35)
            ],
            "privileged_association_stable": forecast[
                forecast.speed.le(15.0)
                & forecast.causal_area_log_change.le(0.35)
                & forecast.target_step_norm.le(15.0)
                & forecast.target_area_log_change.le(0.35)
            ],
        }
        for stratum, subset in quality_subsets.items():
            stratum_metrics, _ = upper_bound_loeo(
                subset.reset_index(drop=True),
                int(args.seed) + len(quality_frames) + 1,
                False,
            )
            if len(stratum_metrics):
                stratum_metrics["quality_stratum"] = stratum
                quality_frames.append(stratum_metrics)
    quality_sensitivity = (
        pd.concat(quality_frames, ignore_index=True)
        if quality_frames
        else pd.DataFrame()
    )
    quality_sensitivity.to_csv(
        args.out_dir / "privileged_driver_quality_sensitivity.csv",
        index=False,
    )
    metrics_frame.to_csv(
        args.out_dir / "privileged_driver_upper_bound.csv",
        index=False,
    )
    controls_frame.to_csv(
        args.out_dir / "privileged_driver_controls.csv",
        index=False,
    )
    inference_frame = (
        privileged_cluster_inference(metrics_frame, int(args.seed))
        if len(metrics_frame)
        else pd.DataFrame()
    )
    inference_frame.to_csv(
        args.out_dir / "privileged_driver_cluster_inference.csv",
        index=False,
    )
    capacity_frame = (
        metrics_frame[metrics_frame.packet.eq("synthetic_capacity")].copy()
        if len(metrics_frame)
        else pd.DataFrame()
    )
    capacity_frame.to_csv(
        args.out_dir / "privileged_driver_capacity_control.csv",
        index=False,
    )
    support_frame = (
        domain_support(forecast, args.data_root)
        if len(forecast)
        else pd.DataFrame(
            [
                {
                    "source_dataset": "SSBD-248 Figure4 selected subset",
                    "target_dataset": "LaChance MDCK_Bulk track-native",
                    "zero_shot_driver_transfer_allowed": False,
                    "reason": "Upper-bound data have not been prepared.",
                }
            ]
        )
    )
    support_frame.to_csv(
        args.out_dir / "privileged_driver_domain_support.csv",
        index=False,
    )

    decision = "not_run"
    best_family = "none"
    gain = np.nan
    controls_pass = False
    heldout_pass = False
    capacity_pass = False
    quality_summary = "not_available"
    if len(metrics_frame):
        real = metrics_frame[
            (metrics_frame.packet == "driver")
            & (metrics_frame.control == "real")
        ]
        covariates = metrics_frame[
            (metrics_frame.packet == "covariates")
            & (metrics_frame.control == "real")
        ]
        # Ridge is the predeclared primary upper-bound family. HGBDT remains
        # an exploratory capacity check and is never selected on outer tests.
        best_family = "ridge"
        real_best = real[real.family == best_family].set_index(
            "test_biological_unit"
        )
        cov_best = covariates[covariates.family == best_family].set_index(
            "test_biological_unit"
        )
        common = real_best.index.intersection(cov_best.index)
        gain = float(
            1.0
            - real_best.loc[common].component_rmse.mean()
            / cov_best.loc[common].component_rmse.mean()
        )
        control_means = (
            controls_frame[controls_frame.family == best_family]
            .groupby("control")
            .component_rmse.mean()
        )
        controls_pass = bool(
            "real" in control_means
            and all(
                control_means["real"] < control_means.get(control, -np.inf)
                for control in (
                    "zero",
                    "time_shuffled",
                    "wrong_cell",
                    "wrong_experiment",
                )
            )
        )
        positive = int(
            (
                cov_best.loc[common].component_rmse
                > real_best.loc[common].component_rmse
            ).sum()
        )
        heldout_pass = bool(
            len(common) >= 3
            and positive > len(common) / 2
            and binomtest(
                positive,
                len(common),
                p=0.5,
                alternative="greater",
            ).pvalue
            <= 0.10
        )
        decision = (
            "pass"
            if gain >= 0.03 and controls_pass and heldout_pass
            else "fail"
        )
        synthetic_mean = capacity_frame.groupby(
            "family"
        ).component_rmse.mean()
        covariate_mean = covariates.groupby(
            "family"
        ).component_rmse.mean()
        capacity_pass = bool(
            len(synthetic_mean)
            and all(
                synthetic_mean[family]
                < 0.90 * covariate_mean.get(family, np.inf)
                for family in synthetic_mean.index
            )
        )
        if len(quality_sensitivity):
            quality_real = quality_sensitivity[
                quality_sensitivity.packet.eq("driver")
                & quality_sensitivity.control.eq("real")
                & quality_sensitivity.family.eq(best_family)
            ]
            quality_covariates = quality_sensitivity[
                quality_sensitivity.packet.eq("covariates")
                & quality_sensitivity.control.eq("real")
                & quality_sensitivity.family.eq(best_family)
            ]
            quality_pairs = quality_real.merge(
                quality_covariates,
                on=["quality_stratum", "test_biological_unit"],
                suffixes=("_driver", "_covariates"),
            )
            quality_summary = "; ".join(
                f"{stratum}={100.0 * (1.0 - group.component_rmse_driver.mean() / group.component_rmse_covariates.mean()):.3f}%"
                for stratum, group in quality_pairs.groupby(
                    "quality_stratum"
                )
            )

    reopening = "blocked"
    lines = [
        "# Mechanical / Causal Driver Reopening Decision",
        "",
        f"Privileged-driver decision: **{decision.upper()}**",
        "",
        f"- Best simple model: `{best_family}`",
        f"- Mean gain over motion + covariates: {gain * 100.0:.3f}%",
        f"- Real driver beats all hard controls: {controls_pass}",
        f"- Effect is supported across held-out biological dates: {heldout_pass}",
        f"- Injected-signal capacity control passes: {capacity_pass}",
        f"- Tracking-quality sensitivity gains: {quality_summary}",
        "",
        "Mechanical/driver conditioning remains **BLOCKED** for LaChance even after a privileged upper-bound pass until a deployable DIC/phase-contrast encoder predicts the driver on held-out experiments and passes domain-support plus bounded-transfer gates.",
        "",
        "A privileged failure closes this ERK implementation. It does not invalidate the already completed reliability result or the GigaScience field result.",
    ]
    (args.out_dir / "mechanical_branch_reopening_decision.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": decision,
        "elapsed_seconds": time.time() - started,
        "selected_source": "SSBD-248 Figure4",
        "selected_fovs": [finite(fov.__dict__) for fov in selected_fovs],
        "outer_unit": "acquisition_date",
        "target": "next_centroid_innovation_relative_to_current_velocity",
        "predeclared_primary_model_family": "ridge",
        "exploratory_model_families": ["hgbdt"],
        "privileged_driver": "current_and_past_FRET_over_CFP_ERK_state",
        "latest_driver_time": "issue_time_t",
        "future_or_target_features_used": False,
        "privileged_driver_decision": decision,
        "injected_signal_capacity_pass": capacity_pass,
        "mechanical_branch_reopening": reopening,
        "outputs": sorted(path.name for path in args.out_dir.iterdir()),
    }
    (args.out_dir / "identified_cell_v171_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run-upper-bound", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=int, default=350)
    parser.add_argument("--max-area", type=int, default=35_000)
    parser.add_argument("--max-track-distance", type=float, default=45.0)
    parser.add_argument("--maximum-frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
