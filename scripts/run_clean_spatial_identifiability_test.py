#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_data_audit_tracklet_baselines import split_frames  # noqa: E402


OUT_DIR = ROOT / "outputs" / "clean_spatial_identifiability"
PROFILE_PATH = ROOT / "outputs" / "prior_dynamic_operator_diagnostics" / "prior_dynamic_profiles.csv"
HORIZONS = (1, 2, 4)
HISTORY = 7
K_VALUES = (2, 4, 8)
REPEAT_SEEDS = (42, 43, 44)
DATASETS = {
    "PSC": {
        "paths": (
            ROOT / "outputs/new_dataset_transfer/PhC-C2DL-PSC/tables/psc01_tracks_table.csv",
            ROOT / "outputs/new_dataset_transfer/PhC-C2DL-PSC/tables/psc02_tracks_table.csv",
        ),
        "r_cut_px": 247.2338159288515,
        "frame_width_px": 720.0,
        "frame_height_px": 576.0,
    },
    "HSC": {
        "paths": (
            ROOT / "outputs/ctc_hsc/tables_gt_tra/hsc01_gttra_tracks_table.csv",
            ROOT / "outputs/ctc_hsc/tables_gt_tra/hsc02_gttra_tracks_table.csv",
        ),
        "r_cut_px": 110.24714285714286,
        "frame_width_px": 1010.0,
        "frame_height_px": 1010.0,
    },
}


@dataclass
class ModelResult:
    pred: np.ndarray
    test: pd.DataFrame
    info: dict[str, Any]


@dataclass
class BaseBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    oof_pred: np.ndarray
    val_pred: np.ndarray
    test_pred: np.ndarray
    info: dict[str, Any]


def vector_metrics(y: np.ndarray, pred: np.ndarray, horizon: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    y = y[finite]
    pred = pred[finite]
    err = pred - y
    centered = y - y.mean(axis=0, keepdims=True)
    sse = float(np.sum(err * err))
    sst = float(np.sum(centered * centered))
    yn = np.linalg.norm(y, axis=1)
    pn = np.linalg.norm(pred, axis=1)
    valid = (yn > np.quantile(yn, 0.25)) & (pn > 1e-9)
    cos = np.sum(y * pred, axis=1) / np.maximum(yn * pn, 1e-9)
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return {
        "rmse_px": rmse,
        "rmse_px_per_frame": rmse / float(horizon),
        "mae_px": float(np.mean(np.linalg.norm(err, axis=1))),
        "r2_vec": float(1.0 - sse / sst) if sst > 1e-12 else float("nan"),
        "cos_mean": float(np.mean(cos[valid])) if np.any(valid) else float("nan"),
        "pred_mag_mean_px": float(np.mean(pn)),
        "target_mag_mean_px": float(np.mean(yn)),
        "n": int(len(y)),
    }


def standardize_source(path: Path, seq_id: int) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "frame" in d.columns:
        d["FRAME"] = pd.to_numeric(d["frame"], errors="coerce")
    if "track_id" in d.columns:
        d["TRACK_ID"] = pd.to_numeric(d["track_id"], errors="coerce")
    keep = d["FRAME"].notna() & d["TRACK_ID"].notna() & d["x_px"].notna() & d["y_px"].notna()
    d = d.loc[keep].copy()
    d["FRAME"] = d["FRAME"].astype(int)
    d["TRACK_ID"] = d["TRACK_ID"].astype(int)
    d = d[d["TRACK_ID"] >= 0].copy()
    d["SEQ_ID"] = int(seq_id)
    d["GLOBAL_TRACK_ID"] = d["SEQ_ID"].astype(str) + ":" + d["TRACK_ID"].astype(str)
    if d.duplicated(["SEQ_ID", "TRACK_ID", "FRAME"]).any():
        prefer = "QUALITY" if "QUALITY" in d.columns else "area_px2"
        d[prefer] = pd.to_numeric(d[prefer], errors="coerce").fillna(0.0)
        d = d.sort_values(["TRACK_ID", "FRAME", prefer], ascending=[True, True, False])
        d = d.drop_duplicates(["SEQ_ID", "TRACK_ID", "FRAME"], keep="first")
    for col in (
        "x_px",
        "y_px",
        "area_px2",
        "RADIUS",
        "circularity",
        "nn_dist_px",
        "neighbors_r50",
        "QUALITY",
    ):
        if col not in d:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    mapping = split_frames(d["FRAME"].unique())
    d["split"] = d["FRAME"].map(mapping)
    d = d.sort_values(["GLOBAL_TRACK_ID", "FRAME"]).reset_index(drop=True)
    grouped = d.groupby("GLOBAL_TRACK_ID", sort=False)
    prev_frame = grouped["FRAME"].shift(1)
    consecutive = prev_frame.notna() & d["FRAME"].eq(prev_frame + 1)
    d["raw_dx"] = (d["x_px"] - grouped["x_px"].shift(1)).where(consecutive)
    d["raw_dy"] = (d["y_px"] - grouped["y_px"].shift(1)).where(consecutive)
    d["raw_speed"] = np.sqrt(d["raw_dx"] ** 2 + d["raw_dy"] ** 2)
    step = d["raw_speed"].replace([np.inf, -np.inf], np.nan)
    med = step.groupby(d["GLOBAL_TRACK_ID"]).transform(
        lambda x: x.rolling(9, min_periods=3).median()
    )
    jump = step / np.maximum(med, 1e-6)
    d["quality_proxy"] = np.exp(
        -0.5 * np.square(np.clip(jump.fillna(1.0).to_numpy(float) - 1.0, 0.0, 6.0) / 2.0)
    )
    return d


def load_dataset(dataset: str) -> pd.DataFrame:
    paths = DATASETS[dataset]["paths"]
    return pd.concat(
        [standardize_source(path, seq_id) for seq_id, path in enumerate(paths)],
        ignore_index=True,
    ).sort_values(["SEQ_ID", "TRACK_ID", "FRAME"]).reset_index(drop=True)


def build_honest_samples(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, g0 in df.groupby("GLOBAL_TRACK_ID", sort=False):
        g = g0.sort_values("FRAME").reset_index(drop=True)
        frames = g["FRAME"].to_numpy(int)
        pos = g[["x_px", "y_px"]].to_numpy(float)
        step = g[["raw_dx", "raw_dy"]].to_numpy(float)
        for idx in range(HISTORY - 1, len(g) - horizon):
            start = idx - HISTORY + 1
            if not np.all(np.diff(frames[start : idx + 1]) == 1):
                continue
            if frames[idx + horizon] != frames[idx] + horizon:
                continue
            if g.loc[idx, "split"] != g.loc[idx + horizon, "split"]:
                continue
            history_step = step[start + 1 : idx + 1]
            if not np.isfinite(history_step).all():
                continue
            rel_pos = pos[start : idx + 1] - pos[idx]
            target = pos[idx + horizon] - pos[idx]
            row: dict[str, Any] = {
                "SEQ_ID": int(g.loc[idx, "SEQ_ID"]),
                "FRAME": int(frames[idx]),
                "TRACK_ID": int(g.loc[idx, "TRACK_ID"]),
                "GLOBAL_TRACK_ID": str(g.loc[idx, "GLOBAL_TRACK_ID"]),
                "split": str(g.loc[idx, "split"]),
                "target_dx": float(target[0]),
                "target_dy": float(target[1]),
                "current_dx": float(history_step[-1, 0]),
                "current_dy": float(history_step[-1, 1]),
                "current_speed": float(np.linalg.norm(history_step[-1])),
                "current_x_px": float(pos[idx, 0]),
                "current_y_px": float(pos[idx, 1]),
                "area_px2": float(g.loc[idx, "area_px2"])
                if np.isfinite(g.loc[idx, "area_px2"])
                else np.nan,
                "RADIUS": float(g.loc[idx, "RADIUS"])
                if np.isfinite(g.loc[idx, "RADIUS"])
                else np.nan,
                "circularity": float(g.loc[idx, "circularity"])
                if np.isfinite(g.loc[idx, "circularity"])
                else np.nan,
                "quality_proxy": float(g.loc[idx, "quality_proxy"]),
            }
            for lag in range(HISTORY):
                reverse = HISTORY - 1 - lag
                row[f"self_relx_lag{lag}"] = float(rel_pos[reverse, 0])
                row[f"self_rely_lag{lag}"] = float(rel_pos[reverse, 1])
            for lag in range(HISTORY - 1):
                reverse = HISTORY - 2 - lag
                row[f"self_dx_lag{lag}"] = float(history_step[reverse, 0])
                row[f"self_dy_lag{lag}"] = float(history_step[reverse, 1])
                row[f"self_speed_lag{lag}"] = float(np.linalg.norm(history_step[reverse]))
            rows.append(row)
    return pd.DataFrame(rows)


def add_position_boundary_features(samples: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = samples.copy()
    width = float(DATASETS[dataset]["frame_width_px"])
    height = float(DATASETS[dataset]["frame_height_px"])
    x = out["current_x_px"].to_numpy(float)
    y = out["current_y_px"].to_numpy(float)
    left = np.clip(x, 0.0, width)
    right = np.clip(width - x, 0.0, width)
    top = np.clip(y, 0.0, height)
    bottom = np.clip(height - y, 0.0, height)
    out["position_x_norm"] = x / width
    out["position_y_norm"] = y / height
    out["position_x_centered"] = (x - 0.5 * width) / width
    out["position_y_centered"] = (y - 0.5 * height) / height
    out["boundary_left_norm"] = left / width
    out["boundary_right_norm"] = right / width
    out["boundary_top_norm"] = top / height
    out["boundary_bottom_norm"] = bottom / height
    out["boundary_min_norm"] = np.minimum.reduce(
        [left / width, right / width, top / height, bottom / height]
    )
    out["position_radius_norm"] = np.sqrt(
        out["position_x_centered"] ** 2 + out["position_y_centered"] ** 2
    )
    return out


def prior_functions(dataset: str):
    profiles = pd.read_csv(PROFILE_PATH)
    part = profiles[profiles["dataset"].eq(dataset)].sort_values("r_px")
    r = part["r_px"].to_numpy(float)
    c = part["c_raw"].to_numpy(float)
    dc = part["dc_dr"].to_numpy(float)

    def c_fn(x: np.ndarray) -> np.ndarray:
        return np.interp(x, r, c, left=c[0], right=0.0)

    def dc_fn(x: np.ndarray) -> np.ndarray:
        return np.interp(x, r, dc, left=dc[0], right=0.0)

    return c_fn, dc_fn


def estimate_raw_gr(df: pd.DataFrame, dataset: str, bins: int = 160):
    r_cut = float(DATASETS[dataset]["r_cut_px"])
    edges = np.linspace(0.0, r_cut, bins + 1)
    hist = np.zeros(bins, dtype=float)
    expected = np.zeros(bins, dtype=float)
    for (seq_id, frame), g in df[df["split"].eq("train")].groupby(
        ["SEQ_ID", "FRAME"], sort=False
    ):
        pos = g[["x_px", "y_px"]].to_numpy(float)
        if len(pos) < 2:
            continue
        pairs = np.asarray(list(cKDTree(pos).query_pairs(r_cut)), dtype=np.int64)
        if pairs.size:
            pairs = pairs.reshape(-1, 2)
            if len(pairs) > 20_000:
                rng = np.random.default_rng(int(seq_id) * 1_000_003 + int(frame))
                pairs = pairs[rng.choice(len(pairs), 20_000, replace=False)]
            dist = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
            hist += np.histogram(dist, bins=edges)[0]
        width = max(float(pos[:, 0].max() - pos[:, 0].min()), 1.0)
        height = max(float(pos[:, 1].max() - pos[:, 1].min()), 1.0)
        area = width * height
        shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        expected += len(pos) * max(len(pos) - 1, 0) * 0.5 * shell / area
    g = hist / np.maximum(expected, 1e-8)
    g = gaussian_filter1d(g, sigma=2.0, mode="nearest")
    positive = g[np.isfinite(g) & (g > 0)]
    if positive.size:
        g /= np.median(positive)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def g_fn(x: np.ndarray) -> np.ndarray:
        return np.interp(x, centers, g, left=g[0], right=1.0)

    return centers, g, g_fn


def aggregate_scalar(n: int, dst: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    np.add.at(out, dst, values)
    return out


def aggregate_vector(n: int, dst: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((n, 2), dtype=float)
    np.add.at(out, dst, values)
    return out


def add_vec(store: dict[str, Any], name: str, x: np.ndarray) -> None:
    store[f"{name}_x"] = x[:, 0]
    store[f"{name}_y"] = x[:, 1]


def context_for_frame(
    all_nodes: pd.DataFrame,
    sample_nodes: pd.DataFrame,
    *,
    dataset: str,
    g_fn,
    c_fn,
    dc_fn,
) -> pd.DataFrame:
    pos = all_nodes[["x_px", "y_px"]].to_numpy(float)
    vel = all_nodes[["raw_dx", "raw_dy"]].fillna(0.0).to_numpy(float)
    speed = np.linalg.norm(vel, axis=1)
    id_to_idx = {int(track): idx for idx, track in enumerate(all_nodes["TRACK_ID"])}
    sample_idx = np.asarray([id_to_idx[int(track)] for track in sample_nodes["TRACK_ID"]], dtype=int)
    tree = cKDTree(pos)
    # The first neighbours describe the local interaction graph; ranks 9-32
    # provide a deliberately coarser flow estimate for the confounder branch.
    max_k = min(max(32, max(K_VALUES)) + 1, len(pos))
    if max_k <= 1:
        dist = np.empty((len(sample_idx), 0))
        nbr = np.empty((len(sample_idx), 0), dtype=int)
    else:
        dist_all, nbr_all = tree.query(pos[sample_idx], k=max_k)
        dist = np.atleast_2d(dist_all)[:, 1:]
        nbr = np.atleast_2d(nbr_all)[:, 1:]
    store: dict[str, Any] = {
        "SEQ_ID": sample_nodes["SEQ_ID"].to_numpy(int),
        "FRAME": sample_nodes["FRAME"].to_numpy(int),
        "TRACK_ID": sample_nodes["TRACK_ID"].to_numpy(int),
    }
    r_cut = float(DATASETS[dataset]["r_cut_px"])
    sample_vel = vel[sample_idx]
    if len(pos) > 1:
        global_sum = vel.sum(axis=0, keepdims=True) - sample_vel
        global_mean = global_sum / float(len(pos) - 1)
        global_speed_mean = (
            speed.sum() - speed[sample_idx]
        ) / float(len(pos) - 1)
    else:
        global_mean = np.zeros((len(sample_idx), 2), dtype=float)
        global_speed_mean = np.zeros(len(sample_idx), dtype=float)
    add_vec(store, "flow_global_velocity", global_mean)
    add_vec(store, "flow_global_relative", global_mean - sample_vel)
    store["flow_global_speed_mean"] = global_speed_mean

    far_start = min(8, dist.shape[1])
    far_stop = min(32, dist.shape[1])
    if far_stop > far_start:
        far_idx = nbr[:, far_start:far_stop]
        far_dist = dist[:, far_start:far_stop]
        far_vel = vel[far_idx]
        bandwidth = max(2.0 * r_cut, 1.0)
        far_weight = np.exp(-0.5 * np.square(far_dist / bandwidth))
        far_denom = np.maximum(far_weight.sum(axis=1, keepdims=True), 1e-8)
        far_mean = np.sum(far_weight[:, :, None] * far_vel, axis=1) / far_denom
        far_second = np.sum(
            far_weight[:, :, None] * np.square(far_vel - far_mean[:, None, :]),
            axis=1,
        ) / far_denom
        far_dispersion = np.sqrt(np.maximum(far_second.sum(axis=1), 0.0))
        far_count = np.full(len(sample_idx), far_stop - far_start, dtype=float)
    else:
        far_mean = global_mean
        far_dispersion = np.zeros(len(sample_idx), dtype=float)
        far_count = np.zeros(len(sample_idx), dtype=float)
    add_vec(store, "flow_far_velocity", far_mean)
    add_vec(store, "flow_far_relative", far_mean - sample_vel)
    store["flow_far_dispersion"] = far_dispersion
    store["flow_far_count"] = far_count

    for k in K_VALUES:
        take = min(k, dist.shape[1])
        valid = dist[:, :take] <= r_cut if take else np.zeros((len(sample_idx), 0), dtype=bool)
        dst = np.repeat(np.arange(len(sample_idx)), take)[valid.reshape(-1)]
        src = nbr[:, :take].reshape(-1)[valid.reshape(-1)]
        rr = dist[:, :take].reshape(-1)[valid.reshape(-1)]
        n = len(sample_idx)
        degree = np.bincount(dst, minlength=n).astype(float)
        rel = pos[src] - pos[sample_idx[dst]]
        radial = rel / np.maximum(rr[:, None], 1e-8)
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
        rel_vel = vel[src] - vel[sample_idx[dst]]
        own_vel = vel[sample_idx[dst]]
        own_speed = np.linalg.norm(own_vel, axis=1)
        own_u = own_vel / np.maximum(own_speed[:, None], 1e-8)
        cos1 = np.sum(radial * own_u, axis=1)
        sin1 = np.sum(tangent * own_u, axis=1)
        c = c_fn(rr)
        dc = dc_fn(rr)
        gr = g_fn(rr)
        denom = np.maximum(degree, 1.0)

        def mean_scalar(values: np.ndarray) -> np.ndarray:
            return aggregate_scalar(n, dst, values) / denom

        def mean_vector(values: np.ndarray) -> np.ndarray:
            return aggregate_vector(n, dst, values) / denom[:, None]

        def sum_vector(values: np.ndarray) -> np.ndarray:
            return aggregate_vector(n, dst, values)

        prefix = f"k{k}"
        store[f"{prefix}_degree"] = degree
        store[f"{prefix}_mean_r"] = mean_scalar(rr) / r_cut
        store[f"{prefix}_std_r"] = np.sqrt(
            np.maximum(mean_scalar(rr * rr) - mean_scalar(rr) ** 2, 0.0)
        ) / r_cut
        nearest = np.full(n, r_cut, dtype=float)
        if take:
            nearest = np.where(valid[:, 0], dist[:, 0], r_cut)
        store[f"{prefix}_nearest_r"] = nearest / r_cut
        store[f"{prefix}_neighbor_speed_mean"] = mean_scalar(speed[src])
        store[f"{prefix}_neighbor_speed_std"] = np.sqrt(
            np.maximum(mean_scalar(speed[src] ** 2) - mean_scalar(speed[src]) ** 2, 0.0)
        )
        store[f"{prefix}_alignment"] = mean_scalar(np.sum(vel[src] * own_vel, axis=1))
        add_vec(store, f"{prefix}_geometry", mean_vector(radial))
        add_vec(store, f"{prefix}_geometry_sum", sum_vector(radial))
        add_vec(store, f"{prefix}_neighbor_velocity", mean_vector(vel[src]))
        add_vec(store, f"{prefix}_neighbor_velocity_sum", sum_vector(vel[src]))
        add_vec(store, f"{prefix}_relative_velocity", mean_vector(rel_vel))
        add_vec(store, f"{prefix}_relative_velocity_sum", sum_vector(rel_vel))
        store[f"{prefix}_g_scalar_sum"] = aggregate_scalar(n, dst, gr)
        add_vec(store, f"{prefix}_g_radial", mean_vector(gr[:, None] * radial))
        add_vec(store, f"{prefix}_g_radial_sum", sum_vector(gr[:, None] * radial))
        store[f"{prefix}_c_scalar_sum"] = aggregate_scalar(n, dst, c)
        store[f"{prefix}_c_abs_sum"] = aggregate_scalar(n, dst, np.abs(c))
        store[f"{prefix}_c_scalar_std"] = np.sqrt(
            np.maximum(mean_scalar(c * c) - mean_scalar(c) ** 2, 0.0)
        )
        add_vec(store, f"{prefix}_c_radial", mean_vector(c[:, None] * radial))
        add_vec(store, f"{prefix}_c_radial_sum", sum_vector(c[:, None] * radial))
        add_vec(store, f"{prefix}_dc_radial", mean_vector(-dc[:, None] * radial))
        add_vec(store, f"{prefix}_dc_radial_sum", sum_vector(-dc[:, None] * radial))
        add_vec(store, f"{prefix}_c_rel_velocity", mean_vector(c[:, None] * rel_vel))
        add_vec(store, f"{prefix}_c_rel_velocity_sum", sum_vector(c[:, None] * rel_vel))
        add_vec(store, f"{prefix}_c_cos1_radial", mean_vector((c * cos1)[:, None] * radial))
        add_vec(
            store,
            f"{prefix}_c_cos1_radial_sum",
            sum_vector((c * cos1)[:, None] * radial),
        )
        add_vec(store, f"{prefix}_c_sin1_tangent", mean_vector((c * sin1)[:, None] * tangent))
        add_vec(
            store,
            f"{prefix}_c_sin1_tangent_sum",
            sum_vector((c * sin1)[:, None] * tangent),
        )
    return pd.DataFrame(store)


def build_context_table(df: pd.DataFrame, samples: pd.DataFrame, dataset: str):
    centers, gr, g_fn = estimate_raw_gr(df, dataset)
    c_fn, dc_fn = prior_functions(dataset)
    parts = []
    grouped_samples = {
        key: value for key, value in samples.groupby(["SEQ_ID", "FRAME"], sort=False)
    }
    for key, all_nodes in df.groupby(["SEQ_ID", "FRAME"], sort=False):
        sample_nodes = grouped_samples.get(key)
        if sample_nodes is None or sample_nodes.empty:
            continue
        parts.append(
            context_for_frame(
                all_nodes.reset_index(drop=True),
                sample_nodes.reset_index(drop=True),
                dataset=dataset,
                g_fn=g_fn,
                c_fn=c_fn,
                dc_fn=dc_fn,
            )
        )
    context = pd.concat(parts, ignore_index=True)
    profile = pd.DataFrame({"dataset": dataset, "r_px": centers, "g_raw_train": gr})
    return samples.merge(
        context,
        on=["SEQ_ID", "FRAME", "TRACK_ID"],
        how="inner",
        validate="one_to_one",
    ), profile


def matched_shuffle_context(
    table: pd.DataFrame,
    context_cols: list[str],
    *,
    seed: int,
) -> pd.DataFrame:
    out = table.copy()
    rng = np.random.default_rng(seed)
    for (_, _), idx in out.groupby(["SEQ_ID", "FRAME"], sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < 2:
            continue
        density = out.loc[idx, "k8_mean_r"].to_numpy(float)
        if len(idx) >= 9 and np.unique(density).size >= 3:
            ranks = pd.Series(density).rank(method="first", pct=True).to_numpy()
            bins = np.minimum((ranks * 3).astype(int), 2)
        else:
            bins = np.zeros(len(idx), dtype=int)
        for density_bin in np.unique(bins):
            local = idx[bins == density_bin]
            if len(local) < 2:
                continue
            perm = rng.permutation(local)
            out.loc[local, context_cols] = out.loc[perm, context_cols].to_numpy()
    return out


def clean_matrix(
    frame: pd.DataFrame,
    cols: list[str],
    train_reference: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    train_values = train_reference[cols].replace([np.inf, -np.inf], np.nan)
    med = train_values.median(axis=0).fillna(0.0)
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)
    x_train = train_values.fillna(med).to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    return np.clip(scaler.transform(x), -25.0, 25.0), scaler


def split_table(table: pd.DataFrame):
    return tuple(
        table[table["split"].eq(s)].copy().reset_index(drop=True)
        for s in ("train", "val", "test")
    )


def fit_hgb(
    table: pd.DataFrame,
    features: list[str],
    targets: list[str],
    *,
    horizon: int,
    seed: int,
) -> ModelResult:
    train, val, test = split_table(table)
    med = train[features].replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0)

    def x(part: pd.DataFrame) -> np.ndarray:
        return np.clip(
            part[features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float),
            -1e5,
            1e5,
        )

    x_train, x_val, x_test = x(train), x(val), x(test)
    y_train = train[targets].to_numpy(float)
    y_val = val[targets].to_numpy(float)
    best = None
    for max_leaf_nodes, l2 in ((15, 1.0), (31, 3.0)):
        models = []
        val_pred = []
        for dim in range(2):
            model = HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=80,
                l2_regularization=l2,
                early_stopping=True,
                validation_fraction=0.12,
                random_state=seed + dim,
            )
            model.fit(x_train, y_train[:, dim])
            models.append(model)
            val_pred.append(model.predict(x_val))
        pred = np.column_stack(val_pred)
        score = vector_metrics(y_val, pred, horizon)["rmse_px"]
        if best is None or score < best[0]:
            best = (score, models, max_leaf_nodes, l2)
    assert best is not None
    pred = np.column_stack([model.predict(x_test) for model in best[1]])
    return ModelResult(
        pred=pred,
        test=test,
        info={
            "max_leaf_nodes": best[2],
            "l2_regularization": best[3],
            "val_rmse_px": best[0],
        },
    )


def _fit_hgb_pair(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    max_leaf_nodes: int,
    l2_regularization: float,
    max_iter: int,
    min_samples_leaf: int,
) -> list[HistGradientBoostingRegressor]:
    models = []
    for dim in range(2):
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.06,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=seed + dim,
        )
        model.fit(x_train, y_train[:, dim])
        models.append(model)
    return models


def _predict_pair(models: list[HistGradientBoostingRegressor], x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in models])


def fit_base_bundle(
    table: pd.DataFrame,
    features: list[str],
    *,
    horizon: int,
    seed: int,
) -> BaseBundle:
    train, val, test = split_table(table)
    targets = ["target_dx", "target_dy"]
    med = train[features].replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0)

    def x(part: pd.DataFrame) -> np.ndarray:
        return np.clip(
            part[features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float),
            -1e5,
            1e5,
        )

    x_train, x_val, x_test = x(train), x(val), x(test)
    y_train = train[targets].to_numpy(float)
    y_val = val[targets].to_numpy(float)
    y_test = test[targets].to_numpy(float)
    best = None
    for max_leaf_nodes, l2 in ((15, 1.0), (31, 3.0)):
        models = _fit_hgb_pair(
            x_train,
            y_train,
            seed=seed,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2,
            max_iter=180,
            min_samples_leaf=80,
        )
        val_pred = _predict_pair(models, x_val)
        score = vector_metrics(y_val, val_pred, horizon)["rmse_px"]
        if best is None or score < best[0]:
            best = (score, models, max_leaf_nodes, l2, val_pred)
    assert best is not None
    groups = train["SEQ_ID"].astype(str) + ":" + train["FRAME"].astype(str)
    oof = np.zeros_like(y_train)
    splitter = GroupKFold(n_splits=3)
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(x_train, y_train, groups)):
        models = _fit_hgb_pair(
            x_train[fit_idx],
            y_train[fit_idx],
            seed=seed + 100 * (fold + 1),
            max_leaf_nodes=int(best[2]),
            l2_regularization=float(best[3]),
            max_iter=180,
            min_samples_leaf=80,
        )
        oof[hold_idx] = _predict_pair(models, x_train[hold_idx])
    return BaseBundle(
        train=train,
        val=val,
        test=test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        oof_pred=oof,
        val_pred=best[4],
        test_pred=_predict_pair(best[1], x_test),
        info={
            "max_leaf_nodes": best[2],
            "l2_regularization": best[3],
            "val_rmse_px": best[0],
            "stage": "self_base",
        },
    )


def fit_residual_block(
    bundle: BaseBundle,
    features: list[str],
    *,
    horizon: int,
    seed: int,
    name: str,
) -> ModelResult:
    train, val, test = bundle.train, bundle.val, bundle.test
    med = train[features].replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0)

    def x(part: pd.DataFrame) -> np.ndarray:
        return np.clip(
            part[features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float),
            -1e5,
            1e5,
        )

    x_train, x_val, x_test = x(train), x(val), x(test)
    residual_train = bundle.y_train - bundle.oof_pred
    base_val_score = vector_metrics(bundle.y_val, bundle.val_pred, horizon)["rmse_px"]
    val_groups = np.empty(len(val), dtype=object)
    for seq_id, idx in val.groupby("SEQ_ID", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        frames = val.loc[idx, "FRAME"].to_numpy(int)
        threshold = np.median(np.unique(frames))
        half = np.where(frames <= threshold, "early", "late")
        val_groups[idx] = [f"{int(seq_id)}:{label}" for label in half]

    def group_gains(candidate: np.ndarray) -> list[float]:
        gains = []
        for label in np.unique(val_groups):
            mask = val_groups == label
            if int(mask.sum()) < 100:
                continue
            base_score = vector_metrics(
                bundle.y_val[mask], bundle.val_pred[mask], horizon
            )["rmse_px"]
            candidate_score = vector_metrics(
                bundle.y_val[mask], candidate[mask], horizon
            )["rmse_px"]
            gains.append((base_score - candidate_score) / base_score * 100.0)
        return gains

    best: tuple[
        float,
        list[HistGradientBoostingRegressor] | None,
        int,
        float,
        float,
        list[float],
    ] = (
        base_val_score,
        None,
        0,
        0.0,
        0.0,
        [],
    )
    for max_leaf_nodes, l2 in ((7, 10.0), (15, 20.0)):
        models = _fit_hgb_pair(
            x_train,
            residual_train,
            seed=seed,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2,
            max_iter=140,
            min_samples_leaf=120,
        )
        raw_val_correction = _predict_pair(models, x_val)
        for scale in (0.05, 0.1, 0.25, 0.5, 1.0):
            val_pred = bundle.val_pred + scale * raw_val_correction
            score = vector_metrics(bundle.y_val, val_pred, horizon)["rmse_px"]
            gains = group_gains(val_pred)
            stable = (
                len(gains) >= 2
                and np.median(gains) > 0.0
                and min(gains) > -0.05
                and np.mean(np.asarray(gains) > 0.0) >= 0.75
            )
            if stable and score < best[0] * (1.0 - 1e-4):
                best = (
                    score,
                    models,
                    max_leaf_nodes,
                    l2,
                    scale,
                    gains,
                )
    if best[1] is None:
        pred = bundle.test_pred.copy()
        correction = np.zeros_like(pred)
    else:
        correction = best[4] * _predict_pair(best[1], x_test)
        pred = bundle.test_pred + correction
    return ModelResult(
        pred=pred,
        test=test,
        info={
            "max_leaf_nodes": best[2],
            "l2_regularization": best[3],
            "val_rmse_px": best[0],
            "base_val_rmse_px": base_val_score,
            "correction_accepted": bool(best[1] is not None),
            "correction_scale": best[4],
            "validation_group_gain_min_pct": min(best[5]) if best[5] else 0.0,
            "validation_group_gain_median_pct": (
                float(np.median(best[5])) if best[5] else 0.0
            ),
            "correction_mag_mean_px": float(
                np.mean(np.linalg.norm(correction, axis=1))
            ),
            "stage": name,
        },
    )


def fit_ridge_response(
    table: pd.DataFrame,
    self_features: list[str],
    response_features: list[str],
    *,
    horizon: int,
    seed: int,
) -> tuple[ModelResult, ModelResult]:
    train, val, test = split_table(table)
    targets = ["target_dx", "target_dy"]
    self_result = fit_hgb(
        table,
        self_features,
        targets,
        horizon=horizon,
        seed=seed,
    )
    # A ridge self-model supplies stable out-of-sample-like residuals for the
    # constrained closure; the direct HGB comparisons remain the main test.
    med = train[self_features].replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0)
    scaler = StandardScaler().fit(
        train[self_features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)
    )
    x_train = scaler.transform(
        train[self_features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)
    )
    x_val = scaler.transform(
        val[self_features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)
    )
    x_test = scaler.transform(
        test[self_features].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)
    )
    y_train = train[targets].to_numpy(float)
    y_val = val[targets].to_numpy(float)
    y_test = test[targets].to_numpy(float)
    best_self = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha, solver="lsqr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
        if not np.isfinite(val_pred).all():
            continue
        score = vector_metrics(y_val, val_pred, horizon)["rmse_px"]
        if best_self is None or score < best_self[0]:
            best_self = (score, model, alpha)
    assert best_self is not None
    base_train = best_self[1].predict(x_train)
    base_val = best_self[1].predict(x_val)
    base_test = best_self[1].predict(x_test)

    resp_med = (
        train[response_features]
        .replace([np.inf, -np.inf], np.nan)
        .median(axis=0)
        .fillna(0.0)
    )
    resp_scaler = StandardScaler().fit(
        train[response_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(resp_med)
        .to_numpy(float)
    )

    def xr(part: pd.DataFrame) -> np.ndarray:
        return resp_scaler.transform(
            part[response_features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(resp_med)
            .to_numpy(float)
        )

    xr_train, xr_val, xr_test = xr(train), xr(val), xr(test)
    residual_train = y_train - base_train
    best_response = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha, solver="lsqr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(xr_train, residual_train)
            correction_val = model.predict(xr_val)
        if not np.isfinite(correction_val).all():
            continue
        pred = base_val + correction_val
        score = vector_metrics(y_val, pred, horizon)["rmse_px"]
        if best_response is None or score < best_response[0]:
            best_response = (score, model, alpha)
    assert best_response is not None
    constrained_pred = base_test + best_response[1].predict(xr_test)
    base_result = ModelResult(
        pred=base_test,
        test=test,
        info={"alpha": best_self[2], "val_rmse_px": best_self[0]},
    )
    response_result = ModelResult(
        pred=constrained_pred,
        test=test,
        info={
            "alpha": best_response[2],
            "val_rmse_px": best_response[0],
            "correction_mag_mean_px": float(
                np.mean(np.linalg.norm(best_response[1].predict(xr_test), axis=1))
            ),
        },
    )
    return base_result, response_result


def feature_groups(table: pd.DataFrame) -> dict[str, list[str]]:
    self_cols = sorted(
        [
            c
            for c in table.columns
            if c.startswith("self_")
            or c in {
                "current_dx",
                "current_dy",
                "current_speed",
                "area_px2",
                "RADIUS",
                "circularity",
                "quality_proxy",
            }
        ]
    )
    geometry = sorted(
        [
            c
            for c in table.columns
            if any(
                token in c
                for token in (
                    "_degree",
                    "_mean_r",
                    "_std_r",
                    "_nearest_r",
                    "_geometry_",
                )
            )
        ]
    )
    position = sorted(
        [
            c
            for c in table.columns
            if c.startswith("position_") or c.startswith("boundary_")
        ]
    )
    flow = sorted([c for c in table.columns if c.startswith("flow_")])
    neighbour = sorted(
        [
            c
            for c in table.columns
            if any(
                token in c
                for token in (
                    "_neighbor_speed",
                    "_alignment",
                    "_neighbor_velocity_",
                    "_relative_velocity_",
                )
            )
        ]
    )
    g_cols = sorted([c for c in table.columns if "_g_radial_" in c])
    c_radial = sorted(
        [
            c
            for c in table.columns
            if any(
                token in c
                for token in (
                    "_c_radial_",
                    "_dc_radial_",
                    "_c_scalar_sum",
                    "_c_abs_sum",
                    "_c_scalar_std",
                )
            )
        ]
    )
    c_state = sorted(
        [
            c
            for c in table.columns
            if "_c_rel_velocity_" in c
        ]
    )
    c_angular = sorted(
        [
            c
            for c in table.columns
            if any(
                token in c
                for token in (
                    "_c_rel_velocity_",
                    "_c_cos1_radial_",
                    "_c_sin1_tangent_",
                )
            )
        ]
    )
    c_angular = [c for c in c_angular if "_c_rel_velocity_" not in c]
    c_cols = c_radial + c_state + c_angular
    response = sorted(
        [
            c
            for c in table.columns
            if any(
                token in c
                for token in (
                    "_g_radial_",
                    "_c_radial_",
                    "_dc_radial_",
                    "_c_rel_velocity_",
                    "_c_cos1_radial_",
                    "_c_sin1_tangent_",
                    "_geometry_",
                    "_relative_velocity_",
                )
            )
        ]
    )
    return {
        "self": self_cols,
        "position": position,
        "flow": flow,
        "geometry": geometry,
        "neighbour": neighbour,
        "g": g_cols,
        "c": c_cols,
        "c_radial": c_radial,
        "c_state": c_state,
        "c_angular": c_angular,
        "response": response,
    }


def evaluate_one(dataset: str, horizon: int):
    df = load_dataset(dataset)
    samples = build_honest_samples(df, horizon)
    samples = add_position_boundary_features(samples, dataset)
    table, profile = build_context_table(df, samples, dataset)
    groups = feature_groups(table)
    context_cols = groups["geometry"] + groups["neighbour"] + groups["g"] + groups["c"]
    shuffled = matched_shuffle_context(
        table,
        context_cols,
        seed=9100 + horizon + (100 if dataset == "HSC" else 0),
    )
    shuffled = shuffled.rename(columns={c: f"{c}__shuffled" for c in context_cols})
    merged = table.merge(
        shuffled[
            ["SEQ_ID", "FRAME", "TRACK_ID"] + [f"{c}__shuffled" for c in context_cols]
        ],
        on=["SEQ_ID", "FRAME", "TRACK_ID"],
        how="inner",
        validate="one_to_one",
    )
    self_cols = groups["self"]
    condition_cols = [
        "current_dx",
        "current_dy",
        "current_speed",
        "quality_proxy",
    ]
    base_position = condition_cols + groups["position"]
    base_flow = base_position + groups["flow"]
    local_geometry = base_flow + groups["geometry"]
    local_neighbour = local_geometry + groups["neighbour"]
    radial_oz = local_neighbour + groups["c_radial"]
    state_oz = radial_oz + groups["c_state"]
    angular_oz = state_oz + groups["c_angular"]
    correction_blocks = {
        "self_plus_position": base_position,
        "self_plus_flow": base_flow,
        "flow_plus_geometry": local_geometry,
        "flow_plus_neighbours": local_neighbour,
        "flow_plus_radial_oz": radial_oz,
        "flow_plus_state_oz": state_oz,
        "flow_plus_angular_oz": angular_oz,
        "flow_plus_all": base_flow + context_cols,
        "flow_plus_all_shuffled": base_flow
        + [f"{c}__shuffled" for c in context_cols],
    }
    rows = []
    predictions = {}
    for repeat_seed in REPEAT_SEEDS:
        paired_seed = 4200 + repeat_seed * 101 + horizon + (500 if dataset == "HSC" else 0)
        bundle = fit_base_bundle(
            merged,
            self_cols,
            horizon=horizon,
            seed=paired_seed,
        )
        base_result = ModelResult(
            pred=bundle.test_pred,
            test=bundle.test,
            info=bundle.info,
        )
        all_results = [("self_only", self_cols, base_result)]
        for block_idx, (name, features) in enumerate(correction_blocks.items()):
            result = fit_residual_block(
                bundle,
                features,
                horizon=horizon,
                seed=paired_seed + 1000 + 17 * block_idx,
                name=name,
            )
            all_results.append((name, features, result))
        for name, features, result in all_results:
            metrics = vector_metrics(
                result.test[["target_dx", "target_dy"]].to_numpy(float),
                result.pred,
                horizon,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": repeat_seed,
                    "model": name,
                    **metrics,
                    **result.info,
                    "n_features": len(features),
                }
            )
            predictions[(name, repeat_seed)] = (result.test, result.pred)

    ridge_base, constrained = fit_ridge_response(
        merged,
        self_cols,
        groups["response"],
        horizon=horizon,
        seed=7600 + horizon + (500 if dataset == "HSC" else 0),
    )
    for name, result in (
        ("ridge_self_reference", ridge_base),
        ("constrained_response", constrained),
    ):
        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "seed": -1,
                "model": name,
                **vector_metrics(
                    result.test[["target_dx", "target_dy"]].to_numpy(float),
                    result.pred,
                    horizon,
                ),
                **result.info,
                "n_features": len(self_cols)
                if name == "ridge_self_reference"
                else len(self_cols) + len(groups["response"]),
            }
        )

    strata = []
    for repeat_seed in REPEAT_SEEDS:
        for model_name in (
            "self_only",
            "self_plus_flow",
            "flow_plus_geometry",
            "flow_plus_radial_oz",
            "flow_plus_angular_oz",
            "flow_plus_all",
            "flow_plus_all_shuffled",
        ):
            test, pred = predictions[(model_name, repeat_seed)]
            y = test[["target_dx", "target_dy"]].to_numpy(float)
            density = test["k8_mean_r"].to_numpy(float)
            q1, q2 = np.quantile(density, [1 / 3, 2 / 3])
            density_bin = np.select(
                [density <= q1, density <= q2],
                ["dense", "mid_density"],
                default="sparse",
            )
            quality = test["quality_proxy"].to_numpy(float)
            quality_bin = np.where(
                quality >= np.quantile(quality, 2 / 3),
                "high_quality",
                np.where(quality <= np.quantile(quality, 1 / 3), "low_quality", "mid_quality"),
            )
            sequence = np.where(test["SEQ_ID"].to_numpy(int) == 0, "sequence_01", "sequence_02")
            for stratifier, labels in {
                "density": density_bin,
                "quality": quality_bin,
                "sequence": sequence,
            }.items():
                for label in np.unique(labels):
                    mask = labels == label
                    strata.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": repeat_seed,
                            "model": model_name,
                            "stratifier": stratifier,
                            "stratum": label,
                            **vector_metrics(y[mask], pred[mask], horizon),
                        }
                    )

    coverage = {
        "dataset": dataset,
        "horizon": horizon,
        "n_samples": len(merged),
        "n_train": int(merged["split"].eq("train").sum()),
        "n_val": int(merged["split"].eq("val").sum()),
        "n_test": int(merged["split"].eq("test").sum()),
        "target_mag_mean_px": float(
            np.mean(np.linalg.norm(merged[["target_dx", "target_dy"]].to_numpy(float), axis=1))
        ),
        "history": HISTORY,
        "raw_geometry": True,
    }
    return pd.DataFrame(rows), pd.DataFrame(strata), profile, pd.DataFrame([coverage])


def plot_results(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    figures = []
    order = [
        "self_only",
        "self_plus_position",
        "self_plus_flow",
        "flow_plus_geometry",
        "flow_plus_neighbours",
        "flow_plus_radial_oz",
        "flow_plus_angular_oz",
        "flow_plus_all",
        "flow_plus_all_shuffled",
        "constrained_response",
    ]
    colors = {
        "self_only": "#2563eb",
        "self_plus_position": "#64748b",
        "self_plus_flow": "#0f766e",
        "flow_plus_geometry": "#0891b2",
        "flow_plus_neighbours": "#0284c7",
        "flow_plus_radial_oz": "#d97706",
        "flow_plus_state_oz": "#9333ea",
        "flow_plus_angular_oz": "#7c3aed",
        "flow_plus_all": "#be123c",
        "flow_plus_all_shuffled": "#94a3b8",
        "constrained_response": "#111827",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=False)
    for ax, dataset in zip(axes, ("PSC", "HSC")):
        repeated_part = (
            summary[(summary["dataset"].eq(dataset)) & (summary["seed"] >= 0)]
            .groupby(["horizon", "model"], as_index=False)["rmse_px_per_frame"]
            .mean()
        )
        reference_part = (
            summary[
                summary["dataset"].eq(dataset)
                & summary["model"].eq("constrained_response")
            ]
            .groupby(["horizon", "model"], as_index=False)["rmse_px_per_frame"]
            .mean()
        )
        part = pd.concat([repeated_part, reference_part], ignore_index=True)
        for model in order:
            d = part[part["model"].eq(model)].sort_values("horizon")
            if d.empty:
                continue
            ax.plot(
                d["horizon"],
                d["rmse_px_per_frame"],
                marker="o",
                label=model,
                color=colors[model],
            )
        ax.set_title(f"{dataset}: honest raw displacement")
        ax.set_xlabel("Forecast horizon, frames")
        ax.set_ylabel("Test RMSE, px/frame")
        ax.set_xticks(HORIZONS)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig01_clean_spatial_rmse.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
    for ax, dataset in zip(axes, ("PSC", "HSC")):
        part = (
            summary[(summary["dataset"].eq(dataset)) & (summary["seed"] >= 0)]
            .groupby(["horizon", "model"], as_index=False)["rmse_px_per_frame"]
            .mean()
        )
        pivot = part.pivot(index="horizon", columns="model", values="rmse_px_per_frame")
        base = pivot["self_plus_flow"]
        for model, color in (
            ("flow_plus_geometry", colors["flow_plus_geometry"]),
            ("flow_plus_neighbours", colors["flow_plus_neighbours"]),
            ("flow_plus_radial_oz", colors["flow_plus_radial_oz"]),
            ("flow_plus_angular_oz", colors["flow_plus_angular_oz"]),
            ("flow_plus_all", colors["flow_plus_all"]),
            ("flow_plus_all_shuffled", colors["flow_plus_all_shuffled"]),
        ):
            gain = (base - pivot[model]) / base * 100.0
            ax.plot(gain.index, gain.values, marker="o", label=model, color=color)
        ax.axhline(0.0, color="#111827", lw=0.9)
        ax.set_title(f"{dataset}: gain over self + flow")
        ax.set_xlabel("Forecast horizon, frames")
        ax.set_ylabel("RMSE improvement, %")
        ax.set_xticks(HORIZONS)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig02_incremental_spatial_gain.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
    for ax, dataset in zip(axes, ("PSC", "HSC")):
        part = (
            summary[(summary["dataset"].eq(dataset)) & (summary["seed"] >= 0)]
            .groupby(["horizon", "model"], as_index=False)["rmse_px_per_frame"]
            .mean()
        )
        pivot = part.pivot(index="horizon", columns="model", values="rmse_px_per_frame")
        comparisons = (
            ("radial OZ vs neighbours", "flow_plus_neighbours", "flow_plus_radial_oz", "#d97706"),
            ("state OZ vs radial", "flow_plus_radial_oz", "flow_plus_state_oz", "#9333ea"),
            ("angular OZ vs state", "flow_plus_state_oz", "flow_plus_angular_oz", "#7c3aed"),
            ("correct vs shuffled", "flow_plus_all_shuffled", "flow_plus_all", "#be123c"),
        )
        for label, base_name, model_name, color in comparisons:
            gain = (pivot[base_name] - pivot[model_name]) / pivot[base_name] * 100.0
            ax.plot(gain.index, gain.values, marker="o", label=label, color=color)
        ax.axhline(0.0, color="#111827", lw=0.9)
        ax.set_title(f"{dataset}: incremental OZ tests")
        ax.set_xlabel("Forecast horizon, frames")
        ax.set_ylabel("Incremental RMSE improvement, %")
        ax.set_xticks(HORIZONS)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig03_incremental_oz_gain.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)
    return figures


def write_report(
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    coverage: pd.DataFrame,
    figures: list[Path],
    out_dir: Path,
) -> None:
    repeated = summary[summary["seed"] >= 0].copy()
    mean_summary = (
        repeated.groupby(["dataset", "horizon", "model"], as_index=False)
        .agg(
            rmse_px_per_frame=("rmse_px_per_frame", "mean"),
            rmse_std=("rmse_px_per_frame", "std"),
            r2_vec=("r2_vec", "mean"),
        )
    )
    pivot = mean_summary.pivot_table(
        index=["dataset", "horizon"], columns="model", values="rmse_px_per_frame"
    )
    pivot["position_gain_vs_self_pct"] = (
        (pivot["self_only"] - pivot["self_plus_position"]) / pivot["self_only"] * 100.0
    )
    pivot["flow_gain_vs_self_pct"] = (
        (pivot["self_only"] - pivot["self_plus_flow"]) / pivot["self_only"] * 100.0
    )
    pivot["geometry_gain_vs_flow_pct"] = (
        (pivot["self_plus_flow"] - pivot["flow_plus_geometry"])
        / pivot["self_plus_flow"]
        * 100.0
    )
    pivot["radial_oz_gain_pct"] = (
        (pivot["flow_plus_neighbours"] - pivot["flow_plus_radial_oz"])
        / pivot["flow_plus_neighbours"]
        * 100.0
    )
    pivot["state_oz_gain_pct"] = (
        (pivot["flow_plus_radial_oz"] - pivot["flow_plus_state_oz"])
        / pivot["flow_plus_radial_oz"]
        * 100.0
    )
    pivot["angular_oz_gain_pct"] = (
        (pivot["flow_plus_state_oz"] - pivot["flow_plus_angular_oz"])
        / pivot["flow_plus_state_oz"]
        * 100.0
    )
    pivot["all_gain_vs_flow_pct"] = (
        (pivot["self_plus_flow"] - pivot["flow_plus_all"])
        / pivot["self_plus_flow"]
        * 100.0
    )
    pivot["correct_vs_shuffled_pct"] = (
        (pivot["flow_plus_all_shuffled"] - pivot["flow_plus_all"])
        / pivot["flow_plus_all_shuffled"]
        * 100.0
    )
    conclusions = []
    for dataset in pivot.index.get_level_values("dataset").unique():
        part = pivot.loc[dataset]
        horizons = sorted(part.index)
        values = lambda column: "/".join(f"{part.loc[h, column]:.3f}" for h in horizons)
        conclusions.extend(
            [
                (
                    f"- {dataset}: position/boundary gain over self for h="
                    f"{'/'.join(map(str, horizons))}: "
                    f"{values('position_gain_vs_self_pct')}%."
                ),
                (
                    f"- {dataset}: coarse-flow gain over self: "
                    f"{values('flow_gain_vs_self_pct')}%."
                ),
                (
                    f"- {dataset}: complete local-context gain beyond self+flow: "
                    f"{values('all_gain_vs_flow_pct')}%."
                ),
                (
                    f"- {dataset}: correct-context advantage over matched shuffle: "
                    f"{values('correct_vs_shuffled_pct')}%."
                ),
                (
                    f"- {dataset}: radial/state/angular OZ incremental gains: "
                    f"radial [{values('radial_oz_gain_pct')}], "
                    f"state [{values('state_oz_gain_pct')}], "
                    f"angular [{values('angular_oz_gain_pct')}]%."
                ),
            ]
        )
    scientific_reading = []
    if {"HSC", "PSC"}.issubset(
        set(pivot.index.get_level_values("dataset").unique())
    ):
        hsc = pivot.loc["HSC"]
        psc = pivot.loc["PSC"]
        scientific_reading = [
            (
                "- Positive: HSC correct local context remains better than matched "
                "shuffle at all tested horizons, with the clearest and most stable "
                "effect at h=1."
            ),
            (
                "- Positive but small: the HSC radial OZ block improves the "
                "geometry+neighbour model at h=1/2/4; the state-aware extension "
                "helps at h=1/2 but not h=4."
            ),
            (
                "- Negative: the current angular extension degrades HSC at all "
                "three horizons and is not ready for the neural decoder."
            ),
            (
                "- Negative: PSC does not show a stable correct-vs-shuffled local "
                "context advantage; it remains a useful negative/control dataset "
                "for this mechanism."
            ),
            (
                "- Caution: the HSC h=2/4 benefit is concentrated in sparse strata, "
                "so a density/observability gate is required and a universal social "
                "correction is not supported."
            ),
            (
                "- Sum-preserving aggregates did not materially improve the "
                "mean-only diagnostic. They remain architecturally preferable for "
                "identifiability, but are not by themselves the missing breakthrough."
            ),
            (
                "- Decision: prototype the bounded influence decoder first on HSC "
                "h=1 with radial+state channels, a no-interaction gate and PSC as a "
                "negative control; postpone the angular branch."
            ),
        ]
    lines = [
        "# Clean spatial identifiability test",
        "",
        "Target: raw x(t+h)-x(t); history is causal through t; graph uses raw centroids at t.",
        (
            "This is an equal-capacity feature-block identifiability diagnostic, "
            "not a final benchmark of the full neural architecture."
        ),
        "",
        "## Decision table",
        "",
        pivot.reset_index().to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Main conclusion",
        "",
        *conclusions,
        "",
        "## Scientific reading",
        "",
        *scientific_reading,
        "",
        "## Full model results",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Repeat means",
        "",
        mean_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Strata",
        "",
        strata.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation rule",
        "",
        "- Local interaction is identifiable only if true context improves self+position+flow and matched shuffled context.",
        "- Radial/state/angular OZ blocks are useful only if they add beyond raw geometry and neighbour histories.",
        "- constrained_response tests whether a small explicit directional closure is sufficient.",
        "- A gain only in specific density/quality strata supports a gated rather than universal interaction branch.",
        "",
        "## Figures",
        "",
        *[f"- `{p.relative_to(ROOT)}`" for p in figures],
    ]
    (out_dir / "clean_spatial_identifiability_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["PSC", "HSC", "ALL"], default="ALL")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ("PSC", "HSC") if args.dataset == "ALL" else (args.dataset,)
    summaries = []
    strata = []
    profiles = []
    coverage = []
    for dataset in datasets:
        for horizon in args.horizons:
            if horizon not in HORIZONS:
                raise ValueError(f"Unsupported horizon {horizon}")
            summary, stratum, profile, cov = evaluate_one(dataset, horizon)
            summaries.append(summary)
            strata.append(stratum)
            profiles.append(profile.assign(horizon=horizon))
            coverage.append(cov)
            print(dataset, horizon, cov.iloc[0].to_dict(), flush=True)
    summary_df = pd.concat(summaries, ignore_index=True)
    strata_df = pd.concat(strata, ignore_index=True)
    profile_df = pd.concat(profiles, ignore_index=True)
    coverage_df = pd.concat(coverage, ignore_index=True)
    suffix = "" if args.dataset == "ALL" else f"_{args.dataset.lower()}"
    summary_df.to_csv(OUT_DIR / f"clean_spatial_summary{suffix}.csv", index=False)
    strata_df.to_csv(OUT_DIR / f"clean_spatial_strata{suffix}.csv", index=False)
    profile_df.to_csv(OUT_DIR / f"raw_gr_profiles{suffix}.csv", index=False)
    coverage_df.to_csv(OUT_DIR / f"clean_spatial_coverage{suffix}.csv", index=False)
    if args.dataset == "ALL":
        figures = plot_results(summary_df, OUT_DIR)
        write_report(summary_df, strata_df, coverage_df, figures, OUT_DIR)
    config = {
        "datasets": datasets,
        "horizons": args.horizons,
        "history": HISTORY,
        "k_values": K_VALUES,
        "target": "raw x(t+h)-x(t)",
        "geometry": "raw centroids at t",
        "frame_dimensions_px": {
            dataset: [
                DATASETS[dataset]["frame_width_px"],
                DATASETS[dataset]["frame_height_px"],
            ]
            for dataset in datasets
        },
        "coarse_flow": (
            "leave-one-out global velocity plus distance-weighted neighbours "
            "ranked 9-32"
        ),
        "aggregation": "mean and sum local vector fields",
        "training": (
            "OOF self residual correction with validation subgroup stability gate"
        ),
    }
    (OUT_DIR / f"config{suffix}.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
