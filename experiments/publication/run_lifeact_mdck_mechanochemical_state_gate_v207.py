#!/usr/bin/env python3
"""Causal mechanochemical unary-state gate on aligned LifeAct-MDCK movies.

The runner builds masks and identities before constructing any forecasting
features.  Coordinate history is kept as a separate packet; morphology,
LifeAct polarity, contact/free-space, phase texture, and tracking reliability
are added only through controlled packets.  Future frames are used exclusively
to define the next-step target and evaluation labels.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from skimage import filters, measure


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "new_data" / "lifeact_mdck_mechanochemical_v206" / "sequences"
DEFAULT_OUT = ROOT / "outputs" / "lifeact_mdck_mechanochemical_state_gate_v207_2026-08-01"
SEQUENCE_ORDER = ("mitomycin", "y27632", "lisa")
EPS = 1e-8


@dataclass
class Split:
    protocol: str
    fold: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def robust_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.quantile(image, [0.01, 0.995])
    return np.clip((image - lo) / max(hi - lo, EPS), 0.0, 1.0)


def frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)_c[12]", path.name)
    if match is None:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(match.group(1))


def segmentation_metrics(labels: np.ndarray) -> dict[str, float]:
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    return {
        "n_instances": float(len(areas)),
        "coverage": float(np.mean(labels > 0)),
        "area_median": float(np.median(areas)) if len(areas) else 0.0,
        "area_q10": float(np.quantile(areas, 0.10)) if len(areas) else 0.0,
        "area_q90": float(np.quantile(areas, 0.90)) if len(areas) else 0.0,
    }


def nucleus_mask_path(cell_mask_path: Path) -> Path:
    return cell_mask_path.with_name(cell_mask_path.stem + "_nucleus.npz")


def load_or_segment(
    data_dir: Path,
    mask_dir: Path,
    sequences: list[str],
    model_name: str,
    diameter: float,
    cellprob_threshold: float,
    flow_threshold: float,
    batch_size: int,
    device: torch.device,
    max_frames: int,
    window_position: str,
) -> tuple[dict[str, list[tuple[int, Path, Path, Path]]], pd.DataFrame]:
    selected: dict[str, list[tuple[int, Path, Path, Path]]] = {}
    for sequence in sequences:
        c1_paths = sorted((data_dir / sequence).glob("frame_*_c1.tif"))
        if max_frames > 0 and len(c1_paths) > max_frames:
            if window_position == "start":
                offset = 0
            elif window_position == "end":
                offset = len(c1_paths) - max_frames
            else:
                offset = (len(c1_paths) - max_frames) // 2
            c1_paths = c1_paths[offset : offset + max_frames]
        records: list[tuple[int, Path, Path, Path]] = []
        for c1_path in c1_paths:
            frame = frame_number(c1_path)
            c2_path = c1_path.with_name(c1_path.name.replace("_c1.tif", "_c2.tif"))
            mask_path = mask_dir / sequence / f"frame_{frame:03d}.npz"
            if not c2_path.exists():
                raise FileNotFoundError(c2_path)
            records.append((frame, c1_path, c2_path, mask_path))
        if len(records) < 3:
            raise RuntimeError(f"Need at least three aligned frames for {sequence}")
        selected[sequence] = records

    missing = [
        record
        for records in selected.values()
        for record in records
        if not record[3].exists() or not nucleus_mask_path(record[3]).exists()
    ]
    model = None
    if missing:
        from cellpose import models as cellpose_models

        model = cellpose_models.CellposeModel(
            gpu=device.type != "cpu",
            device=device,
            pretrained_model=model_name,
            use_bfloat16=False,
        )

    rows: list[dict[str, Any]] = []
    for sequence, records in selected.items():
        for index, (frame, c1_path, c2_path, mask_path) in enumerate(records, 1):
            for role, image_path, output_path in (
                ("cell", c1_path, mask_path),
                ("nucleus", c2_path, nucleus_mask_path(mask_path)),
            ):
                started = time.perf_counter()
                reused = output_path.exists()
                if reused:
                    labels = np.load(output_path)["labels"].astype(np.int32)
                else:
                    image = tifffile.imread(image_path)
                    bsize = 256 if model_name.startswith("cpsam") else 384
                    labels, _, _ = model.eval(
                        image,
                        batch_size=batch_size,
                        diameter=diameter,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        min_size=80,
                        max_size_fraction=0.05,
                        bsize=bsize,
                        tile_overlap=0.20,
                        normalize={
                            "normalize": True,
                            "percentile": [1.0, 99.5],
                            "tile_norm_blocksize": 256,
                        },
                    )
                    labels = np.asarray(labels, dtype=np.int32)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output_path, labels=labels)
                rows.append(
                    {
                        "sequence": sequence,
                        "frame": frame,
                        "role": role,
                        "reused": reused,
                        "runtime_seconds": time.perf_counter() - started,
                        **segmentation_metrics(labels),
                    }
                )
            if index % 10 == 0 or index == len(records):
                print(f"[v207] masks {sequence}: {index}/{len(records)}", flush=True)
    if model is not None:
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    return selected, pd.DataFrame(rows)


def region_table(labels: np.ndarray) -> pd.DataFrame:
    properties = (
        "label",
        "area",
        "centroid",
        "perimeter",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
    )
    table = pd.DataFrame(measure.regionprops_table(labels, properties=properties))
    return table.rename(columns={"centroid-0": "y", "centroid-1": "x"})


def attach_nuclear_anchors(
    cells: pd.DataFrame,
    cell_labels: np.ndarray,
    nucleus_labels: np.ndarray,
) -> pd.DataFrame:
    cells = cells.copy()
    nuclei = region_table(nucleus_labels)
    n_cells, n_nuclei = int(cell_labels.max()), int(nucleus_labels.max())
    joint = np.bincount(
        cell_labels.ravel().astype(np.int64) * (n_nuclei + 1)
        + nucleus_labels.ravel().astype(np.int64),
        minlength=(n_cells + 1) * (n_nuclei + 1),
    ).reshape(n_cells + 1, n_nuclei + 1)
    overlap = joint[1:, 1:]
    nucleus_area = joint[:, 1:].sum(axis=0)
    nucleus_by_label = nuclei.set_index("label") if not nuclei.empty else nuclei
    assigned_to_cell: dict[int, list[int]] = {label: [] for label in cells["label"].astype(int)}
    if overlap.size:
        for nucleus_index in range(n_nuclei):
            cell_index = int(np.argmax(overlap[:, nucleus_index]))
            if overlap[cell_index, nucleus_index] > 0:
                assigned_to_cell.setdefault(cell_index + 1, []).append(nucleus_index)
    records: list[dict[str, float]] = []
    for cell_label in cells["label"].to_numpy(int):
        row = overlap[cell_label - 1] if overlap.size else np.zeros(0, dtype=int)
        candidates = np.asarray(assigned_to_cell.get(cell_label, []), dtype=int)
        if len(candidates) == 0:
            records.append(
                {
                    "label": cell_label,
                    "nucleus_count": 0.0,
                    "nucleus_x": np.nan,
                    "nucleus_y": np.nan,
                    "nucleus_area": np.nan,
                    "nucleus_containment": 0.0,
                }
            )
            continue
        best_index = int(candidates[np.argmax(row[candidates])])
        nucleus_label = best_index + 1
        nucleus = nucleus_by_label.loc[nucleus_label]
        records.append(
            {
                "label": cell_label,
                "nucleus_count": float(len(candidates)),
                "nucleus_x": float(nucleus["x"]),
                "nucleus_y": float(nucleus["y"]),
                "nucleus_area": float(nucleus["area"]),
                "nucleus_containment": float(
                    row[best_index] / max(float(nucleus_area[best_index]), 1.0)
                ),
            }
        )
    anchors = pd.DataFrame(records)
    cells = cells.merge(anchors, on="label", how="left", validate="one_to_one")
    radius = np.sqrt(cells["area"].to_numpy(float) / math.pi)
    cells["nucleus_offset_x"] = (
        cells["nucleus_x"] - cells["x"]
    ) / np.maximum(radius, 1.0)
    cells["nucleus_offset_y"] = (
        cells["nucleus_y"] - cells["y"]
    ) / np.maximum(radius, 1.0)
    cells["nucleus_offset_norm"] = np.hypot(
        cells["nucleus_offset_x"], cells["nucleus_offset_y"]
    )
    cells["nucleus_area_ratio"] = cells["nucleus_area"] / cells["area"].clip(lower=1.0)
    cells["nucleus_reliable"] = (
        cells["nucleus_count"].ge(1.0)
        & cells["nucleus_containment"].ge(0.50)
        & cells["nucleus_area_ratio"].between(0.05, 1.20)
        & cells["nucleus_offset_norm"].le(1.50)
    ).astype(float)
    return cells


def overlap_iou(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    a = previous.ravel().astype(np.int64)
    b = current.ravel().astype(np.int64)
    na, nb = int(a.max()), int(b.max())
    joint = np.bincount(a * (nb + 1) + b, minlength=(na + 1) * (nb + 1)).reshape(
        na + 1, nb + 1
    )
    intersection = joint[1:, 1:].astype(np.float64)
    area_a = joint[1:, :].sum(axis=1).astype(np.float64)
    area_b = joint[:, 1:].sum(axis=0).astype(np.float64)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def track_sequence(
    sequence: str,
    records: list[tuple[int, Path, Path, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    next_track = 1
    previous_labels: np.ndarray | None = None
    previous_table: pd.DataFrame | None = None
    previous_track_by_label: dict[int, int] = {}

    for frame, _, _, mask_path in records:
        labels = np.load(mask_path)["labels"].astype(np.int32)
        nucleus_labels = np.load(nucleus_mask_path(mask_path))["labels"].astype(np.int32)
        current = attach_nuclear_anchors(region_table(labels), labels, nucleus_labels)
        current["sequence"] = sequence
        current["frame"] = frame
        current["track_id"] = -1
        current["match_iou"] = np.nan
        current["match_distance_norm"] = np.nan
        if previous_labels is None or previous_table is None:
            for index in current.index:
                current.at[index, "track_id"] = next_track
                next_track += 1
        else:
            iou = overlap_iou(previous_labels, labels)
            prev_xy = previous_table[["x", "y"]].to_numpy(float)
            curr_xy = current[["x", "y"]].to_numpy(float)
            distances = np.linalg.norm(prev_xy[:, None, :] - curr_xy[None, :, :], axis=2)
            median_diameter = 2.0 * math.sqrt(
                float(np.median(previous_table["area"])) / math.pi
            )
            distance_norm = distances / max(median_diameter, EPS)
            previous_nucleus = previous_table[["nucleus_x", "nucleus_y"]].to_numpy(float)
            current_nucleus = current[["nucleus_x", "nucleus_y"]].to_numpy(float)
            nucleus_distance = np.linalg.norm(
                previous_nucleus[:, None, :] - current_nucleus[None, :, :], axis=2
            ) / max(median_diameter, EPS)
            reliable_pair = (
                previous_table["nucleus_reliable"].to_numpy(bool)[:, None]
                & current["nucleus_reliable"].to_numpy(bool)[None, :]
            )
            nucleus_distance = np.where(
                reliable_pair & np.isfinite(nucleus_distance), nucleus_distance, 2.0
            )
            score = iou - 0.08 * distance_norm - 0.08 * nucleus_distance
            valid = (iou >= 0.03) | (distance_norm <= 0.55) | (nucleus_distance <= 0.45)
            cost = np.where(valid, -score, 1e3)
            row_ind, col_ind = linear_sum_assignment(cost)
            accepted = 0
            matched_current: set[int] = set()
            matched_ious: list[float] = []
            for row, col in zip(row_ind, col_ind):
                if not valid[row, col]:
                    continue
                prev_label = int(previous_table.iloc[row]["label"])
                track_id = previous_track_by_label[prev_label]
                current.at[current.index[col], "track_id"] = track_id
                current.at[current.index[col], "match_iou"] = iou[row, col]
                current.at[current.index[col], "match_distance_norm"] = distance_norm[row, col]
                matched_current.add(col)
                matched_ious.append(float(iou[row, col]))
                accepted += 1
            for col in range(len(current)):
                if col not in matched_current:
                    current.at[current.index[col], "track_id"] = next_track
                    next_track += 1
            diagnostics.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "previous_instances": len(previous_table),
                    "current_instances": len(current),
                    "matched": accepted,
                    "retention_previous": accepted / max(len(previous_table), 1),
                    "retention_current": accepted / max(len(current), 1),
                    "median_matched_iou": float(np.median(matched_ious)) if matched_ious else 0.0,
                }
            )
        current["track_id"] = current["track_id"].astype(int)
        assignments.append(current)
        previous_labels = labels
        previous_table = current
        previous_track_by_label = dict(zip(current["label"].astype(int), current["track_id"]))
    combined = pd.concat(assignments, ignore_index=True).sort_values(
        ["sequence", "track_id", "frame"]
    )
    group = combined.groupby(["sequence", "track_id"], sort=False)
    combined["vx"] = group["x"].diff()
    combined["vy"] = group["y"].diff()
    frame_delta = group["frame"].diff()
    combined.loc[frame_delta.ne(1), ["vx", "vy"]] = np.nan
    return combined, pd.DataFrame(diagnostics)


def contact_degree(labels: np.ndarray) -> np.ndarray:
    n_labels = int(labels.max())
    neighbors: list[set[int]] = [set() for _ in range(n_labels + 1)]
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        valid = (first > 0) & (second > 0) & (first != second)
        if not valid.any():
            continue
        pairs = np.unique(np.stack([first[valid], second[valid]], axis=1), axis=0)
        for left, right in pairs:
            neighbors[int(left)].add(int(right))
            neighbors[int(right)].add(int(left))
    return np.asarray([len(item) for item in neighbors], dtype=float)


def intensity_state(
    labels: np.ndarray,
    lifeact: np.ndarray,
    phase: np.ndarray,
    table: pd.DataFrame,
) -> pd.DataFrame:
    n_labels = int(labels.max())
    flat_labels = labels.ravel().astype(np.int64)
    valid = flat_labels > 0
    life = robust_image(lifeact).ravel()
    phase_norm = robust_image(phase).ravel()
    yy, xx = np.indices(labels.shape, dtype=np.float32)
    xflat, yflat = xx.ravel(), yy.ravel()
    counts = np.bincount(flat_labels[valid], minlength=n_labels + 1).astype(float)

    def sums(values: np.ndarray) -> np.ndarray:
        return np.bincount(flat_labels[valid], weights=values[valid], minlength=n_labels + 1)

    life_sum = sums(life)
    life_sq = sums(life * life)
    phase_sum = sums(phase_norm)
    phase_sq = sums(phase_norm * phase_norm)
    life_mean_by_label = np.divide(
        life_sum,
        np.maximum(counts, 1.0),
        out=np.zeros_like(life_sum),
        where=counts > 0,
    )
    centered_life = np.maximum(life - life_mean_by_label[flat_labels], 0.0)
    weighted_x = sums(centered_life * xflat)
    weighted_y = sums(centered_life * yflat)
    centered_sum = sums(centered_life)
    high_fraction = sums((centered_life > 0).astype(np.float32)) / np.maximum(counts, 1.0)
    degree = contact_degree(labels)

    state = table[["label", "x", "y", "area"]].copy()
    label_index = state["label"].to_numpy(int)
    denominator = np.maximum(counts[label_index], 1.0)
    life_mean = life_sum[label_index] / denominator
    phase_mean = phase_sum[label_index] / denominator
    state["lifeact_mean"] = life_mean
    state["lifeact_std"] = np.sqrt(
        np.maximum(life_sq[label_index] / denominator - life_mean**2, 0.0)
    )
    state["phase_mean"] = phase_mean
    state["phase_std"] = np.sqrt(
        np.maximum(phase_sq[label_index] / denominator - phase_mean**2, 0.0)
    )
    weighted_denominator = np.maximum(centered_sum[label_index], EPS)
    equivalent_radius = np.sqrt(state["area"].to_numpy(float) / math.pi)
    state["polarity_x"] = (
        weighted_x[label_index] / weighted_denominator - state["x"].to_numpy(float)
    ) / np.maximum(equivalent_radius, 1.0)
    state["polarity_y"] = (
        weighted_y[label_index] / weighted_denominator - state["y"].to_numpy(float)
    ) / np.maximum(equivalent_radius, 1.0)
    state["polarity_magnitude"] = np.hypot(state["polarity_x"], state["polarity_y"])
    state["high_actin_fraction"] = high_fraction[label_index]
    state["contact_degree"] = degree[label_index]

    vx_by_label = np.zeros(n_labels + 1, dtype=float)
    vy_by_label = np.zeros(n_labels + 1, dtype=float)
    x_by_label = np.zeros(n_labels + 1, dtype=float)
    y_by_label = np.zeros(n_labels + 1, dtype=float)
    vx_by_label[label_index] = table["vx"].fillna(0.0).to_numpy(float)
    vy_by_label[label_index] = table["vy"].fillna(0.0).to_numpy(float)
    x_by_label[label_index] = table["x"].to_numpy(float)
    y_by_label[label_index] = table["y"].to_numpy(float)
    speed_by_label = np.hypot(vx_by_label, vy_by_label)
    ux = vx_by_label / np.maximum(speed_by_label, EPS)
    uy = vy_by_label / np.maximum(speed_by_label, EPS)
    longitudinal = (
        (xflat - x_by_label[flat_labels]) * ux[flat_labels]
        + (yflat - y_by_label[flat_labels]) * uy[flat_labels]
    )
    front = valid & (longitudinal >= 0.0) & (speed_by_label[flat_labels] > EPS)
    back = valid & (longitudinal < 0.0) & (speed_by_label[flat_labels] > EPS)

    def directional_difference(values: np.ndarray) -> np.ndarray:
        front_count = np.bincount(flat_labels[front], minlength=n_labels + 1)
        back_count = np.bincount(flat_labels[back], minlength=n_labels + 1)
        front_sum = np.bincount(
            flat_labels[front], weights=values[front], minlength=n_labels + 1
        )
        back_sum = np.bincount(
            flat_labels[back], weights=values[back], minlength=n_labels + 1
        )
        front_mean = front_sum / np.maximum(front_count, 1)
        back_mean = back_sum / np.maximum(back_count, 1)
        return front_mean - back_mean

    state["front_back_lifeact"] = directional_difference(life)[label_index]
    state["front_back_phase"] = directional_difference(phase_norm)[label_index]
    return state.drop(columns=["x", "y", "area"])


def add_neighborhood_state(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    xy = frame[["x", "y"]].to_numpy(float)
    radius = np.sqrt(frame["area"].to_numpy(float) / math.pi)
    median_diameter = 2.0 * float(np.median(radius))
    if len(frame) > 1:
        distances, indices = cKDTree(xy).query(xy, k=min(9, len(frame)))
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        neighbors = distances[:, 1:]
        neighbor_indices = indices[:, 1:]
        frame["nearest_distance_norm"] = neighbors[:, 0] / max(median_diameter, EPS)
        nearest_radius = radius[neighbor_indices[:, 0]]
        frame["nearest_gap_norm"] = (
            neighbors[:, 0] - radius - nearest_radius
        ) / max(median_diameter, EPS)
        frame["neighbor_distance_mean_norm"] = neighbors.mean(axis=1) / max(
            median_diameter, EPS
        )
        frame["neighbors_within_2d"] = np.sum(neighbors <= 2.0 * median_diameter, axis=1)
        relative = xy[:, None, :] - xy[neighbor_indices]
        crowding_vector = np.sum(
            relative / np.maximum(neighbors[..., None] ** 2, EPS), axis=1
        )
        crowding_norm = np.linalg.norm(crowding_vector, axis=1)
        frame["crowding_escape_x"] = crowding_vector[:, 0] / np.maximum(crowding_norm, EPS)
        frame["crowding_escape_y"] = crowding_vector[:, 1] / np.maximum(crowding_norm, EPS)
        frame["polarity_crowding_alignment"] = (
            frame["polarity_x"].to_numpy(float) * frame["crowding_escape_x"].to_numpy(float)
            + frame["polarity_y"].to_numpy(float) * frame["crowding_escape_y"].to_numpy(float)
        ) / np.maximum(frame["polarity_magnitude"].to_numpy(float), EPS)
        frame["neighbor_lifeact_mean"] = frame["lifeact_mean"].to_numpy(float)[
            neighbor_indices
        ].mean(axis=1)
        velocity = frame[["vx", "vy"]].fillna(0.0).to_numpy(float)
        velocity_norm = np.linalg.norm(velocity, axis=1)
        unit_velocity = velocity / np.maximum(velocity_norm[:, None], EPS)
        from_center = -relative
        longitudinal = np.sum(from_center * unit_velocity[:, None, :], axis=2)
        front_distance = np.where(longitudinal >= 0.0, neighbors, np.inf).min(axis=1)
        back_distance = np.where(longitudinal < 0.0, neighbors, np.inf).min(axis=1)
        fallback = neighbors.max(axis=1)
        front_distance = np.where(np.isfinite(front_distance), front_distance, fallback)
        back_distance = np.where(np.isfinite(back_distance), back_distance, fallback)
        frame["front_back_free_space"] = (front_distance - back_distance) / max(
            median_diameter, EPS
        )
    else:
        for column in (
            "nearest_distance_norm",
            "nearest_gap_norm",
            "neighbor_distance_mean_norm",
            "neighbors_within_2d",
            "crowding_escape_x",
            "crowding_escape_y",
            "polarity_crowding_alignment",
            "neighbor_lifeact_mean",
            "front_back_free_space",
        ):
            frame[column] = 0.0
    frame["frame_median_diameter"] = median_diameter
    return frame


def build_state_table(
    selected: dict[str, list[tuple[int, Path, Path, Path]]],
    tracks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sequence, records in selected.items():
        track_table = tracks[sequence]
        for frame, c1_path, c2_path, mask_path in records:
            labels = np.load(mask_path)["labels"].astype(np.int32)
            lifeact = tifffile.imread(c1_path)
            phase = tifffile.imread(c2_path)
            current = track_table[track_table["frame"].eq(frame)].copy()
            state = intensity_state(labels, lifeact, phase, current)
            current = current.merge(state, on="label", how="left", validate="one_to_one")
            current = add_neighborhood_state(current)
            rows.append(current)
    table = pd.concat(rows, ignore_index=True).sort_values(
        ["sequence", "track_id", "frame"]
    )
    table["area_norm"] = table["area"] / table.groupby(
        ["sequence", "frame"]
    )["area"].transform("median")
    table["axis_ratio"] = table["major_axis_length"] / table[
        "minor_axis_length"
    ].clip(lower=EPS)
    table["orientation_sin2"] = np.sin(2.0 * table["orientation"])
    table["orientation_cos2"] = np.cos(2.0 * table["orientation"])
    group = table.groupby(["sequence", "track_id"], sort=False)
    table["vx"] = group["x"].diff()
    table["vy"] = group["y"].diff()
    table["frame_delta"] = group["frame"].diff()
    table.loc[table["frame_delta"].ne(1), ["vx", "vy"]] = np.nan
    table["speed"] = np.hypot(table["vx"], table["vy"])
    for lag in (1, 2, 3):
        table[f"vx_lag{lag}"] = group["vx"].shift(lag - 1)
        table[f"vy_lag{lag}"] = group["vy"].shift(lag - 1)
    table["ax"] = table["vx_lag1"] - table["vx_lag2"]
    table["ay"] = table["vy_lag1"] - table["vy_lag2"]
    table["polarity_velocity_alignment"] = (
        table["polarity_x"] * table["vx_lag1"]
        + table["polarity_y"] * table["vy_lag1"]
    ) / (
        table["polarity_magnitude"].clip(lower=EPS)
        * np.hypot(table["vx_lag1"], table["vy_lag1"]).clip(lower=EPS)
    )
    for column in (
        "lifeact_mean",
        "lifeact_std",
        "polarity_x",
        "polarity_y",
        "polarity_magnitude",
        "front_back_lifeact",
        "front_back_phase",
        "high_actin_fraction",
        "area_norm",
        "contact_degree",
        "nearest_gap_norm",
    ):
        table[f"delta_{column}"] = group[column].diff()
    table["target_dx"] = group["x"].shift(-1) - table["x"]
    table["target_dy"] = group["y"].shift(-1) - table["y"]
    table["target_frame_delta"] = group["frame"].shift(-1) - table["frame"]
    table = table[table["target_frame_delta"].eq(1)].copy()
    history_columns = [
        "vx_lag1",
        "vy_lag1",
        "vx_lag2",
        "vy_lag2",
        "vx_lag3",
        "vy_lag3",
        "ax",
        "ay",
    ]
    return table.dropna(subset=history_columns + ["target_dx", "target_dy"]).reset_index(
        drop=True
    )


COORD_FEATURES = [
    "vx_lag1",
    "vy_lag1",
    "vx_lag2",
    "vy_lag2",
    "vx_lag3",
    "vy_lag3",
    "ax",
    "ay",
    "speed",
]
SHAPE_FEATURES = [
    "area_norm",
    "perimeter",
    "eccentricity",
    "solidity",
    "axis_ratio",
    "orientation_sin2",
    "orientation_cos2",
    "delta_area_norm",
    "nucleus_count",
    "nucleus_area_ratio",
    "nucleus_containment",
    "nucleus_offset_x",
    "nucleus_offset_y",
    "nucleus_offset_norm",
]
ACTIN_FEATURES = [
    "lifeact_mean",
    "lifeact_std",
    "polarity_x",
    "polarity_y",
    "polarity_magnitude",
    "polarity_velocity_alignment",
    "front_back_lifeact",
    "front_back_phase",
    "high_actin_fraction",
    "polarity_crowding_alignment",
    "neighbor_lifeact_mean",
    "delta_lifeact_mean",
    "delta_lifeact_std",
    "delta_polarity_x",
    "delta_polarity_y",
    "delta_polarity_magnitude",
]
CONTACT_FEATURES = [
    "contact_degree",
    "nearest_distance_norm",
    "nearest_gap_norm",
    "neighbor_distance_mean_norm",
    "neighbors_within_2d",
    "crowding_escape_x",
    "crowding_escape_y",
    "front_back_free_space",
    "delta_contact_degree",
    "delta_nearest_gap_norm",
]
RELIABILITY_FEATURES = [
    "match_iou",
    "match_distance_norm",
    "nucleus_reliable",
    "phase_mean",
    "phase_std",
]
STATE_FEATURES = SHAPE_FEATURES + ACTIN_FEATURES + CONTACT_FEATURES + RELIABILITY_FEATURES

NORMALIZED_COORD_FEATURES = [
    "vx_lag1_norm",
    "vy_lag1_norm",
    "vx_lag2_norm",
    "vy_lag2_norm",
    "vx_lag3_norm",
    "vy_lag3_norm",
    "ax_norm",
    "ay_norm",
    "speed_norm",
]
NORMALIZED_SHAPE_FEATURES = [
    "area_norm",
    "perimeter_norm",
    "eccentricity",
    "solidity",
    "axis_ratio",
    "orientation_sin2",
    "orientation_cos2",
    "delta_area_norm",
    "nucleus_count",
    "nucleus_area_ratio",
    "nucleus_containment",
    "nucleus_offset_x_norm",
    "nucleus_offset_y_norm",
    "nucleus_offset_norm",
]


def feature_packets(scale_normalized: bool = False) -> dict[str, list[str]]:
    coord = NORMALIZED_COORD_FEATURES if scale_normalized else COORD_FEATURES
    shape = NORMALIZED_SHAPE_FEATURES if scale_normalized else SHAPE_FEATURES
    state = shape + ACTIN_FEATURES + CONTACT_FEATURES + RELIABILITY_FEATURES
    return {
        "coord_only": coord,
        "coord_shape": coord + shape,
        "coord_actin": coord + ACTIN_FEATURES,
        "coord_contact": coord + CONTACT_FEATURES,
        "coord_reliability": coord + RELIABILITY_FEATURES,
        "full_real": coord + state,
        "full_zero_state": coord + state,
        "full_row_shuffled": coord + state,
        "full_wrong_cell": coord + state,
        "full_time_shuffled": coord + state,
    }


def add_scale_normalized_columns(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    scale = result["frame_median_diameter"].clip(lower=EPS)
    for source, destination in (
        ("vx_lag1", "vx_lag1_norm"),
        ("vy_lag1", "vy_lag1_norm"),
        ("vx_lag2", "vx_lag2_norm"),
        ("vy_lag2", "vy_lag2_norm"),
        ("vx_lag3", "vx_lag3_norm"),
        ("vy_lag3", "vy_lag3_norm"),
        ("ax", "ax_norm"),
        ("ay", "ay_norm"),
        ("speed", "speed_norm"),
        ("perimeter", "perimeter_norm"),
        ("nucleus_offset_x", "nucleus_offset_x_norm"),
        ("nucleus_offset_y", "nucleus_offset_y_norm"),
    ):
        result[destination] = result[source] / scale
    return result


def controlled_table(table: pd.DataFrame, packet: str, seed: int) -> pd.DataFrame:
    result = table.copy()
    if packet == "full_zero_state":
        result[STATE_FEATURES] = 0.0
    elif packet == "full_row_shuffled":
        rng = np.random.default_rng(seed)
        result[STATE_FEATURES] = result[STATE_FEATURES].iloc[
            rng.permutation(len(result))
        ].to_numpy()
    elif packet == "full_wrong_cell":
        rng = np.random.default_rng(seed)
        for _, index in result.groupby(["sequence", "frame"]).groups.items():
            positions = np.asarray(list(index), dtype=int)
            if len(positions) > 1:
                permutation = rng.permutation(len(positions))
                result.loc[positions, STATE_FEATURES] = result.loc[
                    positions[permutation], STATE_FEATURES
                ].to_numpy()
    elif packet == "full_time_shuffled":
        for (_, track_id), index in result.groupby(["sequence", "track_id"]).groups.items():
            positions = np.asarray(list(index), dtype=int)
            if len(positions) > 1:
                shift = 7 % len(positions)
                if shift == 0:
                    shift = 1
                result.loc[positions, STATE_FEATURES] = np.roll(
                    result.loc[positions, STATE_FEATURES].to_numpy(), shift, axis=0
                )
    return result


def chronological_splits(table: pd.DataFrame) -> list[Split]:
    train = np.zeros(len(table), dtype=bool)
    val = np.zeros(len(table), dtype=bool)
    test = np.zeros(len(table), dtype=bool)
    for sequence, index in table.groupby("sequence").groups.items():
        positions = np.asarray(list(index), dtype=int)
        frames = table.loc[positions, "frame"].to_numpy()
        unique = np.unique(frames)
        first_cut = unique[max(1, int(0.60 * len(unique))) - 1]
        second_cut = unique[max(2, int(0.80 * len(unique))) - 1]
        train[positions] = frames <= first_cut
        val[positions] = (frames > first_cut) & (frames <= second_cut)
        test[positions] = frames > second_cut
    return [Split("chronological", "all_sequences", train, val, test)]


def leave_one_sequence_out_splits(table: pd.DataFrame) -> list[Split]:
    sequences = [sequence for sequence in SEQUENCE_ORDER if sequence in set(table["sequence"])]
    # A leave-one-sequence-out fold needs independent train, validation, and
    # test sequences.  Single-sequence smoke runs still exercise the strict
    # chronological protocol without constructing an empty training split.
    if len(sequences) < 3:
        return []
    splits: list[Split] = []
    for index, test_sequence in enumerate(sequences):
        val_sequence = sequences[(index + 1) % len(sequences)]
        splits.append(
            Split(
                "leave_one_sequence_out",
                test_sequence,
                (~table["sequence"].isin([test_sequence, val_sequence])).to_numpy(),
                table["sequence"].eq(val_sequence).to_numpy(),
                table["sequence"].eq(test_sequence).to_numpy(),
            )
        )
    return splits


def clean_features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return values


def impute_train(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return tuple(np.where(np.isfinite(values), values, medians) for values in (train, val, test))


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    val_scaled = scaler.transform(x_val)
    test_scaled = scaler.transform(x_test)
    best: tuple[float, float, Ridge] | None = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        # NumPy 2.0 linked to Apple Accelerate can leave spurious floating
        # point flags after a finite matrix product.  Silence only those flags
        # and retain explicit finite checks below.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            model = Ridge(alpha=alpha).fit(train_scaled, y_train)
            prediction = model.predict(val_scaled)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite Ridge validation output for alpha={alpha}")
        score = float(np.sqrt(np.mean(np.square(prediction - y_val))))
        if best is None or score < best[0]:
            best = (score, alpha, model)
    assert best is not None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        prediction = best[2].predict(test_scaled)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite Ridge test output")
    return prediction, {"alpha": best[1], "val_rmse": best[0]}


def fit_predict_hgbdt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    best_score = math.inf
    best_models: list[HistGradientBoostingRegressor] = []
    best_l2 = 0.0
    for l2 in (1.0, 10.0, 100.0):
        models = [
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=l2,
                random_state=seed + component,
            ).fit(x_train, y_train[:, component])
            for component in range(2)
        ]
        prediction = np.column_stack(
            [model.predict(x_val) for model in models]
        )
        score = float(np.sqrt(np.mean(np.square(prediction - y_val))))
        if score < best_score:
            best_score, best_models, best_l2 = score, models, l2
    return np.column_stack([model.predict(x_test) for model in best_models]), {
        "l2": best_l2,
        "val_rmse": best_score,
    }


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "component_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "vector_rmse": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
        "component_r2": float(r2_score(target.reshape(-1), prediction.reshape(-1))),
        "direction_cosine": float(
            np.mean(
                np.sum(target * prediction, axis=1)
                / (
                    np.linalg.norm(target, axis=1)
                    * np.linalg.norm(prediction, axis=1)
                    + EPS
                )
            )
        ),
        "magnitude_ratio": float(
            np.mean(np.linalg.norm(prediction, axis=1))
            / (np.mean(np.linalg.norm(target, axis=1)) + EPS)
        ),
    }


def evaluate(
    table: pd.DataFrame,
    seed: int,
    scale_normalized: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    packets = feature_packets(scale_normalized)
    splits = chronological_splits(table) + leave_one_sequence_out_splits(table)
    target = table[["target_dx", "target_dy"]].to_numpy(float)
    base = table[["vx_lag1", "vy_lag1"]].to_numpy(float)
    scale = table["frame_median_diameter"].to_numpy(float).clip(min=EPS)
    if scale_normalized:
        target_model = target / scale[:, None]
        base_model = base / scale[:, None]
    else:
        target_model = target
        base_model = base
    residual = target_model - base_model
    for split in splits:
        counts = (int(split.train.sum()), int(split.val.sum()), int(split.test.sum()))
        if min(counts) == 0:
            raise RuntimeError(
                f"Empty {split.protocol}/{split.fold} split after track-history filtering: "
                f"train/val/test={counts}"
            )
        for packet, columns in packets.items():
            controlled = controlled_table(table, packet, seed + len(rows) * 19)
            if scale_normalized:
                controlled = add_scale_normalized_columns(controlled)
            values = clean_features(controlled, columns)
            x_train, x_val, x_test = impute_train(
                values[split.train], values[split.val], values[split.test]
            )
            y_train, y_val = residual[split.train], residual[split.val]
            for model_name in ("ridge", "hgbdt"):
                if model_name == "ridge":
                    residual_prediction, hyper = fit_predict_ridge(
                        x_train, y_train, x_val, y_val, x_test
                    )
                else:
                    residual_prediction, hyper = fit_predict_hgbdt(
                        x_train, y_train, x_val, y_val, x_test, seed
                    )
                prediction_model = base_model[split.test] + residual_prediction
                prediction = (
                    prediction_model * scale[split.test, None]
                    if scale_normalized
                    else prediction_model
                )
                result = {
                    "protocol": split.protocol,
                    "fold": split.fold,
                    "packet": packet,
                    "model": model_name,
                    "evaluation_scale": (
                        "current_frame_cell_diameter" if scale_normalized else "pixels"
                    ),
                    "n_train": int(split.train.sum()),
                    "n_val": int(split.val.sum()),
                    "n_test": int(split.test.sum()),
                    **metrics(target[split.test], prediction),
                    **hyper,
                }
                rows.append(result)
                details = table.loc[split.test, ["sequence", "frame", "track_id", "match_iou"]].copy()
                details["protocol"] = split.protocol
                details["fold"] = split.fold
                details["packet"] = packet
                details["model"] = model_name
                details["evaluation_scale"] = result["evaluation_scale"]
                details["target_dx"] = target[split.test, 0]
                details["target_dy"] = target[split.test, 1]
                details["prediction_dx"] = prediction[:, 0]
                details["prediction_dy"] = prediction[:, 1]
                details["error_norm"] = np.linalg.norm(prediction - target[split.test], axis=1)
                prediction_rows.append(details)
    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def gate_summary(results: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        results.groupby(["protocol", "packet", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            component_r2_mean=("component_r2", "mean"),
            direction_cosine_mean=("direction_cosine", "mean"),
        )
    )
    baseline = aggregate[aggregate.packet.eq("coord_only")][
        ["protocol", "model", "component_rmse_mean"]
    ].rename(columns={"component_rmse_mean": "coord_rmse"})
    aggregate = aggregate.merge(baseline, on=["protocol", "model"], how="left")
    aggregate["gain_vs_coord_percent"] = 100.0 * (
        aggregate["coord_rmse"] - aggregate["component_rmse_mean"]
    ) / aggregate["coord_rmse"].clip(lower=EPS)
    return aggregate.sort_values(["protocol", "model", "component_rmse_mean"])


def diagnostic_associations(table: pd.DataFrame) -> pd.DataFrame:
    target = table[["target_dx", "target_dy"]].to_numpy(float)
    velocity = table[["vx_lag1", "vy_lag1"]].to_numpy(float)
    residual = target - velocity
    polarity = table[["polarity_x", "polarity_y"]].to_numpy(float)
    crowding = table[["crowding_escape_x", "crowding_escape_y"]].to_numpy(float)

    def cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1) + EPS
        )

    alignment = cosine(target, polarity)
    residual_alignment = cosine(residual, polarity)
    crowding_alignment = cosine(target, crowding)
    residual_crowding_alignment = cosine(residual, crowding)
    shuffled = np.roll(polarity, 997 % max(len(polarity), 1), axis=0)
    shuffled_alignment = cosine(target, shuffled)
    baseline_error = np.linalg.norm(residual, axis=1)
    valid_iou = table["match_iou"].notna()
    iou_error_spearman = table.loc[valid_iou, "match_iou"].corr(
        pd.Series(baseline_error[valid_iou.to_numpy()], index=table.index[valid_iou]),
        method="spearman",
    )
    reliable = table["nucleus_reliable"].eq(1.0).to_numpy()
    reliability_error_delta = (
        float(np.mean(baseline_error[~reliable]) - np.mean(baseline_error[reliable]))
        if reliable.any() and (~reliable).any()
        else np.nan
    )
    return pd.DataFrame(
        [
            {"quantity": "target_polarity_cosine", "control": "real", "value": np.mean(alignment)},
            {"quantity": "target_polarity_cosine", "control": "row_shift", "value": np.mean(shuffled_alignment)},
            {"quantity": "residual_polarity_cosine", "control": "real", "value": np.mean(residual_alignment)},
            {"quantity": "target_crowding_escape_cosine", "control": "real", "value": np.mean(crowding_alignment)},
            {"quantity": "residual_crowding_escape_cosine", "control": "real", "value": np.mean(residual_crowding_alignment)},
            {"quantity": "tracking_match_iou", "control": "real", "value": table["match_iou"].mean()},
            {"quantity": "tracking_match_iou_p10", "control": "real", "value": table["match_iou"].quantile(0.10)},
            {"quantity": "nucleus_reliable_fraction", "control": "real", "value": table["nucleus_reliable"].mean()},
            {"quantity": "match_iou_vs_cv_error_spearman", "control": "real", "value": iou_error_spearman},
            {"quantity": "unreliable_minus_reliable_cv_error", "control": "real", "value": reliability_error_delta},
        ]
    )


def data_quality_gate(
    segmentation: pd.DataFrame,
    tracking: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence in sorted(state["sequence"].unique()):
        seg = segmentation[segmentation["sequence"].eq(sequence)]
        track = tracking[tracking["sequence"].eq(sequence)]
        current = state[state["sequence"].eq(sequence)]
        cell = seg[seg["role"].eq("cell")]
        nucleus = seg[seg["role"].eq("nucleus")]
        record = {
            "sequence": sequence,
            "frames": int(seg["frame"].nunique()),
            "cell_coverage_mean": float(cell["coverage"].mean()),
            "cell_instances_mean": float(cell["n_instances"].mean()),
            "nucleus_coverage_mean": float(nucleus["coverage"].mean()),
            "retention_current_mean": float(track["retention_current"].mean()),
            "median_matched_iou_mean": float(track["median_matched_iou"].mean()),
            "nucleus_reliable_fraction": float(current["nucleus_reliable"].mean()),
            "forecast_rows": int(len(current)),
        }
        record["spatial_pass"] = bool(
            0.45 <= record["cell_coverage_mean"] <= 0.98
            and 250.0 <= record["cell_instances_mean"] <= 1500.0
        )
        record["identity_pass"] = bool(
            record["retention_current_mean"] >= 0.80
            and record["median_matched_iou_mean"] >= 0.40
        )
        record["data_gate_pass"] = bool(
            record["spatial_pass"] and record["identity_pass"]
        )
        rows.append(record)
    return pd.DataFrame(rows)


def plot_results(gate: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    subset = gate[
        gate["packet"].isin(
            [
                "coord_only",
                "coord_shape",
                "coord_actin",
                "coord_contact",
                "full_real",
                "full_wrong_cell",
                "full_time_shuffled",
            ]
        )
    ].copy()
    for protocol, group in subset.groupby("protocol"):
        pivot = group.pivot(index="packet", columns="model", values="gain_vs_coord_percent")
        ax = pivot.plot(kind="bar", figsize=(9.2, 4.6), color=["#2166ac", "#b2182b"])
        ax.axhline(0.0, color="#374151", linewidth=0.8)
        ax.set_ylabel("изменение RMSE относительно координат, %")
        ax.set_xlabel("")
        ax.set_title(f"Mechanochemical unary-state gate: {protocol}")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        plt.xticks(rotation=28, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"v207_{protocol}_gate.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Shared segmentation cache; defaults to OUT_DIR/masks.",
    )
    parser.add_argument("--sequences", default=",".join(SEQUENCE_ORDER))
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--diameter", type=float, default=40.0)
    parser.add_argument("--cellprob-threshold", type=float, default=-0.5)
    parser.add_argument("--flow-threshold", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--window-position",
        choices=["start", "center", "end"],
        default="center",
        help="Contiguous temporal window used when --max-frames truncates a sequence.",
    )
    parser.add_argument("--seed", type=int, default=207_001)
    parser.add_argument("--segment-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.mask_dir if args.mask_dir is not None else args.out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    started = time.time()
    selected, segmentation = load_or_segment(
        args.data_dir,
        mask_dir,
        sequences,
        args.model,
        args.diameter,
        args.cellprob_threshold,
        args.flow_threshold,
        args.batch_size,
        device,
        args.max_frames,
        args.window_position,
    )
    segmentation.to_csv(args.out_dir / "v207_segmentation_quality.csv", index=False)
    if args.segment_only:
        return

    track_tables: dict[str, pd.DataFrame] = {}
    track_diagnostics: list[pd.DataFrame] = []
    for sequence, records in selected.items():
        track_table, diagnostic = track_sequence(sequence, records)
        track_tables[sequence] = track_table
        track_diagnostics.append(diagnostic)
    tracks = pd.concat(track_diagnostics, ignore_index=True)
    tracks.to_csv(args.out_dir / "v207_tracking_quality.csv", index=False)

    state = build_state_table(selected, track_tables)
    state.to_parquet(args.out_dir / "v207_cell_state.parquet", index=False)
    quality = data_quality_gate(segmentation, tracks, state)
    quality.to_csv(args.out_dir / "v207_data_quality_gate.csv", index=False)
    results, predictions = evaluate(state, args.seed)
    gate = gate_summary(results)
    associations = diagnostic_associations(state)
    results.to_csv(args.out_dir / "v207_state_gate_metrics.csv", index=False)
    predictions.to_parquet(args.out_dir / "v207_state_gate_predictions.parquet", index=False)
    gate.to_csv(args.out_dir / "v207_state_gate_aggregate.csv", index=False)
    associations.to_csv(args.out_dir / "v207_state_associations.csv", index=False)
    plot_results(gate, args.out_dir)

    real = gate[gate.packet.eq("full_real")].copy()
    controls = gate[
        gate.packet.isin(
            ["full_zero_state", "full_row_shuffled", "full_wrong_cell", "full_time_shuffled"]
        )
    ]
    comparisons: list[dict[str, Any]] = []
    for _, row in real.iterrows():
        matching = controls[
            controls["protocol"].eq(row["protocol"])
            & controls["model"].eq(row["model"])
        ]
        best_control = float(matching["component_rmse_mean"].min())
        comparisons.append(
            {
                "protocol": row["protocol"],
                "model": row["model"],
                "real_rmse": row["component_rmse_mean"],
                "coord_rmse": row["coord_rmse"],
                "best_control_rmse": best_control,
                "gain_vs_coord_percent": row["gain_vs_coord_percent"],
                "gain_vs_best_control_percent": 100.0
                * (best_control - row["component_rmse_mean"])
                / max(best_control, EPS),
                "soft_pass": bool(
                    row["gain_vs_coord_percent"] >= 1.0
                    and row["component_rmse_mean"] < best_control
                ),
                "hard_pass": bool(
                    row["gain_vs_coord_percent"] >= 3.0
                    and 100.0
                    * (best_control - row["component_rmse_mean"])
                    / max(best_control, EPS)
                    >= 1.0
                ),
            }
        )
    decision = pd.DataFrame(comparisons)
    decision.to_csv(args.out_dir / "v207_decision.csv", index=False)
    report = [
        "# LifeAct-MDCK mechanochemical unary-state gate v207",
        "",
        "## Decision table",
        "",
        decision.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Forecasting aggregate",
        "",
        gate.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## State diagnostics",
        "",
        associations.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Data quality gate",
        "",
        quality.to_markdown(index=False, floatfmt=".6f"),
        "",
        "A hard pass requires >=3% RMSE gain over coordinate history and >=1% over the best zero/row/wrong-cell/time control. A soft pass requires >=1% over coordinates and a strict win over the best control. Target/future frames are never inference features.",
        "",
        f"Elapsed: {(time.time() - started) / 3600.0:.2f} hours.",
    ]
    (args.out_dir / "v207_decision_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (args.out_dir / "v207_contract.json").write_text(
        json.dumps(
            {
                "sequences": sequences,
                "model": args.model,
                "diameter": args.diameter,
                "cellprob_threshold": args.cellprob_threshold,
                "flow_threshold": args.flow_threshold,
                "batch_size": args.batch_size,
                "device": str(device),
                "mask_dir": str(mask_dir.resolve()),
                "max_frames": args.max_frames,
                "window_position": args.window_position,
                "causal_contract": "current/past masks and tracks only",
                "target": "next centroid displacement",
                "protocols": ["chronological", "leave_one_sequence_out"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
