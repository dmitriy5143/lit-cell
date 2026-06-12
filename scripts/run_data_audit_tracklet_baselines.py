from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]

COL_BLUE = "#2F6F9F"
COL_GREEN = "#5F8A5B"
COL_ORANGE = "#B8873A"
COL_RED = "#A33A2B"
COL_PURPLE = "#7A6FA8"
COL_GRAY = "#707070"

TRACKLET_BASE_FEATURES = [
    "dx_px",
    "dy_px",
    "speed_px_s",
    "ax_px_s2",
    "ay_px_s2",
    "area_px2",
    "RADIUS",
    "circularity",
    "nn_dist_px",
    "neighbors_r50",
    "node_quality",
    "centroid_jump_score",
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def split_frames(frames: np.ndarray, frame_split: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict[int, str]:
    frames = np.sort(np.asarray(frames, dtype=int))
    n = int(frames.size)
    if n == 0:
        return {}
    n_train = max(1, int(n * frame_split[0]))
    n_val = max(0, int(n * frame_split[1]))
    n_train = min(n_train, n - 1) if n > 1 else n
    n_val = min(n_val, n - n_train)
    train = set(frames[:n_train].tolist())
    val = set(frames[n_train:n_train + n_val].tolist())
    test = set(frames[n_train + n_val:].tolist())
    out: dict[int, str] = {}
    for fr in train:
        out[int(fr)] = "train"
    for fr in val:
        out[int(fr)] = "val"
    for fr in test:
        out[int(fr)] = "test"
    return out


def qsummary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def sanitize_feature_matrix(x: np.ndarray, clip: float = 1e5) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32).copy()
    z[~np.isfinite(z)] = np.nan
    finite = np.isfinite(z)
    if finite.any():
        np.clip(z, -float(clip), float(clip), out=z)
    return z


def prepare_flat_features(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    z = sanitize_feature_matrix(x).astype(np.float64)
    train = z[train_mask]
    med = np.nanmedian(train, axis=0)
    med = np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
    bad_r, bad_c = np.where(~np.isfinite(z))
    if bad_r.size:
        z[bad_r, bad_c] = med[bad_c]
    mean = z[train_mask].mean(axis=0)
    std = z[train_mask].std(axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    z = (z - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -25.0, 25.0).astype(np.float32)


def fit_numpy_ridge_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    xt = np.asarray(x[train_mask], dtype=np.float64)
    yt = np.asarray(y[train_mask], dtype=np.float64)
    if sample_weight is None:
        w = np.ones((xt.shape[0],), dtype=np.float64)
    else:
        w = np.asarray(sample_weight[train_mask], dtype=np.float64)
        w = np.clip(np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.05), 0.05, 1.0)
    w_sum = max(float(w.sum()), 1e-12)
    y_mean = (yt * w[:, None]).sum(axis=0, keepdims=True) / w_sum
    yc = yt - y_mean
    xtw = xt * np.sqrt(w)[:, None]
    ycw = yc * np.sqrt(w)[:, None]
    gram = np.einsum("ni,nj->ij", xtw, xtw, optimize=False)
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    rhs = np.einsum("ni,nk->ik", xtw, ycw, optimize=False)
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]
    pred = np.einsum("ij,jk->ik", np.asarray(x, dtype=np.float64), coef, optimize=False) + y_mean
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def robust_threshold(train_values: pd.Series, *, q: float, floor: float) -> float:
    x = pd.to_numeric(train_values, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(floor)
    return float(max(floor, np.quantile(x, q)))


def load_tables(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for seq_idx, path in enumerate(paths):
        df = pd.read_csv(path)
        if "frame" in df.columns and "FRAME" not in df.columns:
            df = df.rename(columns={"frame": "FRAME"})
        if "track_id" in df.columns and "TRACK_ID" not in df.columns:
            df = df.rename(columns={"track_id": "TRACK_ID"})
        for col in ["FRAME", "TRACK_ID", "x_px", "y_px", "target_dx_px", "target_dy_px"]:
            if col not in df.columns:
                raise ValueError(f"{path} is missing required column {col}")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.loc[df["FRAME"].notna() & df["TRACK_ID"].notna() & df["x_px"].notna() & df["y_px"].notna()].copy()
        df["FRAME"] = df["FRAME"].astype(int)
        df["TRACK_ID"] = df["TRACK_ID"].astype(int)
        df = df.loc[df["TRACK_ID"] >= 0].copy()
        df["SEQ_ID"] = int(seq_idx)
        df["SEQ_NAME"] = path.parent.name if path.parent.name else path.stem
        df["SOURCE_TABLE"] = str(path)
        df["GLOBAL_TRACK_ID"] = df["SEQ_ID"].astype(str) + ":" + df["TRACK_ID"].astype(str)
        if df.duplicated(["SEQ_ID", "TRACK_ID", "FRAME"]).any():
            prefer = "QUALITY" if "QUALITY" in df.columns else "area_px2"
            if prefer in df.columns:
                df[prefer] = pd.to_numeric(df[prefer], errors="coerce").fillna(0.0)
                df = df.sort_values(["SEQ_ID", "TRACK_ID", "FRAME", prefer], ascending=[True, True, True, False])
            df = df.drop_duplicates(["SEQ_ID", "TRACK_ID", "FRAME"], keep="first")
        parts.append(df)
    out = pd.concat(parts, axis=0, ignore_index=True)
    for col in set(TRACKLET_BASE_FEATURES + [
        "area_px2",
        "bbox_w",
        "bbox_h",
        "RADIUS",
        "nn_dist_px",
        "neighbors_r50",
        "node_quality",
        "centroid_jump_score",
        "has_target",
        "dx_px",
        "dy_px",
    ]):
        if col not in out.columns:
            out[col] = np.nan
        if col != "has_target":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "has_target" not in out.columns:
        out["has_target"] = out["target_dx_px"].notna() & out["target_dy_px"].notna()
    else:
        out["has_target"] = out["has_target"].astype(bool) & out["target_dx_px"].notna() & out["target_dy_px"].notna()
    out = out.sort_values(["SEQ_ID", "TRACK_ID", "FRAME"]).reset_index(drop=True)
    return out


def add_audit_columns(df: pd.DataFrame, min_segment_len: int) -> tuple[pd.DataFrame, dict[str, float]]:
    d = df.copy()
    split_map: dict[tuple[int, int], str] = {}
    for seq_id, g in d.groupby("SEQ_ID"):
        mapping = split_frames(g["FRAME"].unique())
        for frame, split in mapping.items():
            split_map[(int(seq_id), int(frame))] = split
    d["split"] = [split_map.get((int(s), int(f)), "unknown") for s, f in zip(d["SEQ_ID"], d["FRAME"])]

    grouped = d.groupby("GLOBAL_TRACK_ID", sort=False)
    d["prev_frame"] = grouped["FRAME"].shift(1)
    d["next_frame"] = grouped["FRAME"].shift(-1)
    d["prev_consecutive"] = d["prev_frame"].notna() & (d["prev_frame"].astype(float) == (d["FRAME"] - 1).astype(float))
    d["next_consecutive"] = d["next_frame"].notna() & (d["next_frame"].astype(float) == (d["FRAME"] + 1).astype(float))

    frame_gap = grouped["FRAME"].diff()
    new_segment = frame_gap.isna() | (frame_gap != 1)
    d["segment_local"] = new_segment.astype(int).groupby(d["GLOBAL_TRACK_ID"]).cumsum()
    d["segment_id_audit"] = d["GLOBAL_TRACK_ID"].astype(str) + ":" + d["segment_local"].astype(str)
    d["segment_len"] = d.groupby("segment_id_audit")["FRAME"].transform("size")
    d["track_len"] = d.groupby("GLOBAL_TRACK_ID")["FRAME"].transform("size")

    d["pos_dx_prev"] = d["x_px"] - grouped["x_px"].shift(1)
    d["pos_dy_prev"] = d["y_px"] - grouped["y_px"].shift(1)
    d["pos_step_norm"] = np.sqrt(d["pos_dx_prev"] ** 2 + d["pos_dy_prev"] ** 2)
    d["target_norm"] = np.sqrt(d["target_dx_px"] ** 2 + d["target_dy_px"] ** 2)
    d["motion_norm"] = np.sqrt(d["dx_px"] ** 2 + d["dy_px"] ** 2)

    for col in ["area_px2", "RADIUS", "bbox_w", "bbox_h", "nn_dist_px", "neighbors_r50"]:
        prev = grouped[col].shift(1)
        cur = pd.to_numeric(d[col], errors="coerce")
        d[f"{col}_log_change"] = np.abs(np.log((cur + 1e-6) / (prev + 1e-6)))

    cur = d[["dx_px", "dy_px"]].to_numpy(float)
    tgt = d[["target_dx_px", "target_dy_px"]].to_numpy(float)
    cur_norm = np.linalg.norm(cur, axis=1)
    tgt_norm = np.linalg.norm(tgt, axis=1)
    cos = np.sum(cur * tgt, axis=1) / np.clip(cur_norm * tgt_norm, 1e-6, np.inf)
    d["current_target_cos"] = np.clip(cos, -1.0, 1.0)

    train = d.loc[d["split"] == "train"].copy()
    thresholds = {
        "centroid_jump_q95": robust_threshold(train["centroid_jump_score"], q=0.95, floor=2.0),
        "centroid_jump_q99": robust_threshold(train["centroid_jump_score"], q=0.99, floor=3.0),
        "target_norm_q995": robust_threshold(train["target_norm"], q=0.995, floor=1e-6),
        "motion_norm_q995": robust_threshold(train["motion_norm"], q=0.995, floor=1e-6),
        "area_log_q99": robust_threshold(train["area_px2_log_change"], q=0.99, floor=math.log(2.0)),
        "radius_log_q99": robust_threshold(train["RADIUS_log_change"], q=0.99, floor=math.log(1.5)),
        "bbox_log_q99": max(
            robust_threshold(train["bbox_w_log_change"], q=0.99, floor=math.log(2.0)),
            robust_threshold(train["bbox_h_log_change"], q=0.99, floor=math.log(2.0)),
        ),
        "nn_dist_log_q99": robust_threshold(train["nn_dist_px_log_change"], q=0.99, floor=math.log(3.0)),
        "speed_median": float(np.nanmedian(train["motion_norm"].to_numpy(float))),
    }

    d["flag_short_segment"] = d["segment_len"] < int(min_segment_len)
    d["flag_no_target"] = ~d["has_target"].astype(bool)
    d["flag_centroid_jump"] = pd.to_numeric(d["centroid_jump_score"], errors="coerce").fillna(0.0) > thresholds["centroid_jump_q99"]
    d["flag_target_outlier"] = d["target_norm"] > thresholds["target_norm_q995"]
    d["flag_motion_outlier"] = d["motion_norm"] > thresholds["motion_norm_q995"]
    d["flag_area_jump"] = d["area_px2_log_change"] > thresholds["area_log_q99"]
    d["flag_radius_jump"] = d["RADIUS_log_change"] > thresholds["radius_log_q99"]
    d["flag_bbox_jump"] = (d["bbox_w_log_change"] > thresholds["bbox_log_q99"]) | (d["bbox_h_log_change"] > thresholds["bbox_log_q99"])
    d["flag_density_jump"] = d["nn_dist_px_log_change"] > thresholds["nn_dist_log_q99"]
    speed_floor = max(float(thresholds["speed_median"]), 1e-6)
    d["flag_velocity_reversal"] = (
        (d["current_target_cos"] < -0.75)
        & (d["motion_norm"] > speed_floor)
        & (d["target_norm"] > speed_floor)
    )
    flag_cols = [c for c in d.columns if c.startswith("flag_") and c != "flag_no_target"]
    d["audit_flag_count"] = d[flag_cols].astype(int).sum(axis=1)
    d["audit_clean_conservative"] = d["has_target"].astype(bool) & (d["audit_flag_count"] == 0)
    d["audit_suspicious"] = d["has_target"].astype(bool) & (
        (d["audit_flag_count"] >= 2)
        | d["flag_centroid_jump"]
        | d["flag_target_outlier"]
        | d["flag_motion_outlier"]
    )
    d["audit_quality_group"] = np.where(
        ~d["has_target"].astype(bool),
        "no_target",
        np.where(d["audit_clean_conservative"], "clean", np.where(d["audit_suspicious"], "suspicious", "medium")),
    )
    q = pd.to_numeric(d["node_quality"], errors="coerce").fillna(1.0).clip(0.05, 1.0)
    jump = pd.to_numeric(d["centroid_jump_score"], errors="coerce").fillna(1.0)
    jump_penalty = np.exp(-0.08 * np.clip(jump - thresholds["centroid_jump_q95"], 0.0, 20.0))
    flag_penalty = np.power(0.75, d["audit_flag_count"].clip(0, 6).to_numpy(float))
    d["audit_sample_weight"] = np.clip(q.to_numpy(float) * jump_penalty * flag_penalty, 0.05, 1.0)
    return d, thresholds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    if int(finite.sum()) == 0:
        return {"rmse_norm": float("nan"), "mae_norm": float("nan"), "r2_dx": float("nan"), "r2_dy": float("nan"), "mean_r2": float("nan"), "n": 0}
    y = y_true[finite]
    p = y_pred[finite]
    err = p - y
    mse = np.mean(err ** 2, axis=0)
    var = np.var(y, axis=0)
    r2_dx = float(1.0 - mse[0] / max(var[0], 1e-12))
    r2_dy = float(1.0 - mse[1] / max(var[1], 1e-12))
    return {
        "rmse_norm": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
        "mae_norm": float(np.mean(np.linalg.norm(err, axis=1))),
        "rmse_dx": float(np.sqrt(mse[0])),
        "rmse_dy": float(np.sqrt(mse[1])),
        "r2_dx": r2_dx,
        "r2_dy": r2_dy,
        "mean_r2": float(0.5 * (r2_dx + r2_dy)),
        "n": int(y.shape[0]),
    }


@dataclass
class TrackletSamples:
    x_flat: np.ndarray
    x_seq: np.ndarray
    y: np.ndarray
    split: np.ndarray
    group: np.ndarray
    sample_weight: np.ndarray
    current_motion: np.ndarray
    mean_motion: np.ndarray


def build_tracklet_samples(df: pd.DataFrame, window: int) -> TrackletSamples:
    features = list(TRACKLET_BASE_FEATURES)
    rows_flat: list[np.ndarray] = []
    rows_seq: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    splits: list[str] = []
    groups: list[str] = []
    sample_weights: list[float] = []
    current_motion: list[np.ndarray] = []
    mean_motion: list[np.ndarray] = []

    work = df.sort_values(["SEQ_ID", "TRACK_ID", "FRAME"]).copy()
    for _, g in work.groupby("GLOBAL_TRACK_ID", sort=False):
        g = g.sort_values("FRAME").reset_index(drop=True)
        values = g[features].to_numpy(float)
        pos = g[["x_px", "y_px"]].to_numpy(float)
        y = g[["target_dx_px", "target_dy_px"]].to_numpy(float)
        frames = g["FRAME"].to_numpy(int)
        split = g["split"].to_numpy(str)
        has_target = g["has_target"].to_numpy(bool)
        quality_group = g["audit_quality_group"].to_numpy(str)

        for idx in range(window - 1, len(g)):
            start = idx - window + 1
            fr = frames[start:idx + 1]
            if not np.all(np.diff(fr) == 1):
                continue
            if len(set(split[start:idx + 1].tolist())) != 1:
                continue
            if not has_target[idx] or not np.isfinite(y[idx]).all():
                continue
            seq = values[start:idx + 1].copy()
            rel_pos = pos[start:idx + 1] - pos[idx]
            seq = np.concatenate([rel_pos, seq], axis=1)
            seq = sanitize_feature_matrix(seq)
            rows_seq.append(seq.astype(np.float32))
            rows_flat.append(seq.reshape(-1).astype(np.float32))
            ys.append(y[idx].astype(np.float32))
            splits.append(split[idx])
            groups.append(quality_group[idx])
            sample_weights.append(float(g.loc[idx, "audit_sample_weight"]) if "audit_sample_weight" in g.columns else 1.0)
            cur_motion = np.nan_to_num(g.loc[idx, ["dx_px", "dy_px"]].to_numpy(float), nan=0.0)
            current_motion.append(cur_motion.astype(np.float32))
            mean_motion.append(np.nan_to_num(g.loc[start:idx, ["dx_px", "dy_px"]].to_numpy(float), nan=0.0).mean(axis=0).astype(np.float32))

    return TrackletSamples(
        x_flat=np.vstack(rows_flat).astype(np.float32),
        x_seq=np.stack(rows_seq, axis=0).astype(np.float32),
        y=np.vstack(ys).astype(np.float32),
        split=np.asarray(splits),
        group=np.asarray(groups),
        sample_weight=np.asarray(sample_weights, dtype=np.float32),
        current_motion=np.vstack(current_motion).astype(np.float32),
        mean_motion=np.vstack(mean_motion).astype(np.float32),
    )


class FlatMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, int] = (96, 48)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_flat_mlp_predict(x: np.ndarray, y: np.ndarray, split: np.ndarray, *, epochs: int, seed: int) -> np.ndarray:
    torch.manual_seed(int(seed))
    train_mask = split == "train"
    val_mask = split == "val"
    if int(train_mask.sum()) == 0:
        return np.zeros_like(y, dtype=np.float32)

    x_train = np.asarray(x[train_mask], dtype=np.float32)
    y_train = np.asarray(y[train_mask], dtype=np.float32)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy((y_train - y_mean) / y_std).float())
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    model = FlatMLP(in_dim=x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    bad = 0
    patience = 6
    x_val = torch.from_numpy(np.asarray(x[val_mask], dtype=np.float32)) if int(val_mask.sum()) else None
    y_val = np.asarray(y[val_mask], dtype=np.float32) if int(val_mask.sum()) else None

    for _epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if x_val is not None and y_val is not None and y_val.shape[0]:
                pred_val = model(x_val).numpy() * y_std + y_mean
                val_rmse = regression_metrics(y_val, pred_val)["rmse_norm"]
            else:
                pred_train = model(torch.from_numpy(x_train[: min(4096, x_train.shape[0])])).numpy() * y_std + y_mean
                val_rmse = regression_metrics(y_train[: pred_train.shape[0]], pred_train)["rmse_norm"]
        if val_rmse < best_val - 1e-7:
            best_val = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            xb = torch.from_numpy(np.asarray(x[start:start + 8192], dtype=np.float32))
            preds.append(model(xb).numpy() * y_std + y_mean)
    return np.vstack(preds).astype(np.float32)


class TinyGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def fit_gru_baseline(samples: TrackletSamples, epochs: int, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    torch.manual_seed(int(seed))
    train_mask = samples.split == "train"
    val_mask = samples.split == "val"
    test_mask = samples.split == "test"
    x_train = sanitize_feature_matrix(samples.x_seq[train_mask])
    y_train = samples.y[train_mask]
    x_val = sanitize_feature_matrix(samples.x_seq[val_mask])
    y_val = samples.y[val_mask]
    x_test = sanitize_feature_matrix(samples.x_seq[test_mask])
    y_test = samples.y[test_mask]
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        return {}, {}

    feat_mean = np.nanmean(x_train.reshape(-1, x_train.shape[-1]), axis=0)
    feat_std = np.nanstd(x_train.reshape(-1, x_train.shape[-1]), axis=0)
    feat_mean = np.nan_to_num(feat_mean, nan=0.0, posinf=0.0, neginf=0.0)
    feat_std = np.nan_to_num(feat_std, nan=1.0, posinf=1.0, neginf=1.0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    def prep(x: np.ndarray) -> np.ndarray:
        z = np.where(np.isfinite(x), x, feat_mean)
        return ((z - feat_mean) / feat_std).astype(np.float32)

    x_train = prep(x_train)
    x_val = prep(x_val) if x_val.shape[0] else x_val
    x_test = prep(x_test)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy((y_train - y_mean) / y_std).float())
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    model = TinyGRU(in_dim=x_train.shape[-1], hidden=48)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = 5
    bad = 0
    for _epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            if x_val.shape[0]:
                pred_val = model(torch.from_numpy(x_val)).numpy() * y_std + y_mean
                val_rmse = regression_metrics(y_val, pred_val)["rmse_norm"]
            else:
                pred_train = model(torch.from_numpy(x_train[: min(4096, x_train.shape[0])])).numpy() * y_std + y_mean
                val_rmse = regression_metrics(y_train[: pred_train.shape[0]], pred_train)["rmse_norm"]
        if val_rmse < best_val - 1e-7:
            best_val = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(torch.from_numpy(x_test)).numpy() * y_std + y_mean
        pred_val = model(torch.from_numpy(x_val)).numpy() * y_std + y_mean if x_val.shape[0] else np.zeros((0, 2), dtype=float)
    return regression_metrics(y_test, pred_test), regression_metrics(y_val, pred_val) if x_val.shape[0] else {}


def fit_huber_predict(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, *, epsilon: float) -> np.ndarray:
    from sklearn.linear_model import HuberRegressor

    x_all = np.asarray(x, dtype=np.float64)
    xt_raw = x_all[train_mask]
    x_mean = np.nanmean(xt_raw, axis=0, keepdims=True)
    x_std = np.nanstd(xt_raw, axis=0, keepdims=True)
    x_mean = np.where(np.isfinite(x_mean), x_mean, 0.0)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-6), x_std, 1.0)
    x_scaled = (np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0) - x_mean) / x_std
    x_scaled = np.clip(x_scaled, -20.0, 20.0)
    xt = x_scaled[train_mask]
    yt = np.asarray(y[train_mask], dtype=np.float64)
    y_mean = yt.mean(axis=0, keepdims=True)
    y_std = yt.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    yt_norm = (yt - y_mean) / y_std
    pred = np.zeros((x.shape[0], 2), dtype=np.float32)
    for dim in range(2):
        model = HuberRegressor(epsilon=float(epsilon), alpha=1e-4, max_iter=1000, tol=1e-5)
        model.fit(xt, yt_norm[:, dim])
        pred[:, dim] = (model.predict(x_scaled) * y_std[0, dim] + y_mean[0, dim]).astype(np.float32)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def select_and_eval_ridge(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    window: int,
    x_flat: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    eval_sets: dict[str, np.ndarray],
    sample_weight: np.ndarray | None = None,
) -> None:
    alphas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best_pred = None
    best_alpha = None
    best_val = float("inf")
    for alpha in alphas:
        pred_all = fit_numpy_ridge_predict(x_flat, y, train_mask, alpha=float(alpha), sample_weight=sample_weight)
        val_rmse = regression_metrics(y[val_mask], pred_all[val_mask])["rmse_norm"]
        if val_rmse < best_val:
            best_val = val_rmse
            best_alpha = alpha
            best_pred = pred_all
    if best_pred is None:
        return
    for subset, mask in eval_sets.items():
        m = regression_metrics(y[mask], best_pred[mask])
        rows.append({"window": window, "model": model_name, "subset": subset, "alpha": best_alpha, "val_rmse_selected": best_val, **m})


def run_tracklet_baselines(
    samples: TrackletSamples,
    window: int,
    include_gru: bool,
    torch_epochs: int,
    seed: int,
    *,
    include_robust: bool = False,
    skip_mlp: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train = samples.split == "train"
    val = samples.split == "val"
    test = samples.split == "test"
    x_flat = prepare_flat_features(samples.x_flat, train)

    eval_sets = {
        "test_full": test,
        "test_clean": test & (samples.group == "clean"),
        "test_medium": test & (samples.group == "medium"),
        "test_suspicious": test & (samples.group == "suspicious"),
    }

    baseline_preds = {
        "persistence_current": samples.current_motion,
        "persistence_tracklet_mean": samples.mean_motion,
        "train_mean": np.repeat(samples.y[train].mean(axis=0, keepdims=True), samples.y.shape[0], axis=0),
    }
    for model_name, pred in baseline_preds.items():
        for subset, mask in eval_sets.items():
            m = regression_metrics(samples.y[mask], pred[mask])
            rows.append({"window": window, "model": model_name, "subset": subset, **m})

    select_and_eval_ridge(
        rows,
        model_name="ridge_flat",
        window=window,
        x_flat=x_flat,
        y=samples.y,
        train_mask=train,
        val_mask=val,
        eval_sets=eval_sets,
    )

    if include_robust:
        select_and_eval_ridge(
            rows,
            model_name="ridge_weighted",
            window=window,
            x_flat=x_flat,
            y=samples.y,
            train_mask=train,
            val_mask=val,
            eval_sets=eval_sets,
            sample_weight=samples.sample_weight,
        )
        select_and_eval_ridge(
            rows,
            model_name="ridge_train_clean",
            window=window,
            x_flat=x_flat,
            y=samples.y,
            train_mask=train & (samples.group == "clean"),
            val_mask=val,
            eval_sets=eval_sets,
        )
        for eps in [1.15, 1.35, 1.75]:
            pred = fit_huber_predict(x_flat, samples.y, train, epsilon=float(eps))
            val_rmse = regression_metrics(samples.y[val], pred[val])["rmse_norm"]
            for subset, mask in eval_sets.items():
                m = regression_metrics(samples.y[mask], pred[mask])
                rows.append({"window": window, "model": f"huber_eps{eps:g}", "subset": subset, "epsilon": eps, "val_rmse_selected": val_rmse, **m})

    if not skip_mlp:
        pred = fit_flat_mlp_predict(x_flat, samples.y, samples.split, epochs=45, seed=int(seed))
        for subset, mask in eval_sets.items():
            m = regression_metrics(samples.y[mask], pred[mask])
            rows.append({"window": window, "model": "mlp_flat", "subset": subset, **m})

    if include_gru:
        test_metrics, val_metrics = fit_gru_baseline(samples, epochs=int(torch_epochs), seed=int(seed))
        if test_metrics:
            rows.append({"window": window, "model": "tiny_gru", "subset": "test_full", **test_metrics})
            rows.append({"window": window, "model": "tiny_gru", "subset": "val_full", **val_metrics})

    return pd.DataFrame(rows)


def build_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, g in df.groupby("split"):
        valid = g["has_target"].astype(bool)
        for group_name, mask in {
            "valid_targets": valid,
            "clean": valid & g["audit_clean_conservative"].astype(bool),
            "medium": valid & (g["audit_quality_group"] == "medium"),
            "suspicious": valid & (g["audit_quality_group"] == "suspicious"),
            "no_target": ~valid,
        }.items():
            rows.append(
                {
                    "split": split,
                    "group": group_name,
                    "rows": int(mask.sum()),
                    "fraction_of_split": float(mask.mean()) if len(mask) else 0.0,
                    "fraction_of_valid": float(mask.sum() / max(valid.sum(), 1)) if group_name != "no_target" else float(mask.mean()),
                }
            )
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, df: pd.DataFrame, coverage: pd.DataFrame, baselines: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

    cov_plot = coverage.loc[coverage["group"].isin(["clean", "medium", "suspicious"])].copy()
    pivot = cov_plot.pivot(index="split", columns="group", values="rows").fillna(0.0)
    pivot = pivot.reindex(["train", "val", "test"]).dropna(how="all")
    pivot.plot(kind="bar", stacked=True, ax=axes[0, 0], color=[COL_GREEN, COL_ORANGE, COL_RED])
    axes[0, 0].set_title("Target coverage by audit group")
    axes[0, 0].set_ylabel("rows")
    axes[0, 0].tick_params(axis="x", rotation=0)

    tracks = df.drop_duplicates(["GLOBAL_TRACK_ID"])["track_len"]
    axes[0, 1].hist(tracks, bins=40, color=COL_BLUE, alpha=0.85)
    axes[0, 1].set_title("Track length distribution")
    axes[0, 1].set_xlabel("track length, frames")
    axes[0, 1].set_ylabel("tracks")

    valid = df.loc[df["has_target"].astype(bool)].copy()
    axes[1, 0].hist(valid["centroid_jump_score"].replace([np.inf, -np.inf], np.nan).dropna(), bins=60, color=COL_PURPLE, alpha=0.85)
    axes[1, 0].set_title("Centroid jump score")
    axes[1, 0].set_xlabel("score")
    axes[1, 0].set_xlim(left=0)

    base = baselines.loc[baselines["subset"] == "test_full"].copy()
    if not base.empty:
        order = base.sort_values("rmse_norm")
        labels = [f"W{int(w)} {m}" for w, m in zip(order["window"], order["model"])]
        axes[1, 1].barh(labels, order["rmse_norm"], color=COL_GRAY)
        axes[1, 1].set_title("Self-only tracklet baselines")
        axes[1, 1].set_xlabel("test RMSE, px")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_data_audit_tracklet_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig_data_audit_tracklet_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    dataset: str,
    table_paths: list[Path],
    thresholds: dict[str, float],
    coverage: pd.DataFrame,
    baselines: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append(f"# Data Audit And Tracklet Baselines: {dataset}")
    lines.append("")
    lines.append("Date: 2026-06-01")
    lines.append("")
    lines.append("## Input Tables")
    lines.append("")
    for p in table_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Sequence Coverage")
    lines.append("")
    lines.append("| sequence | rows | frames | tracks | valid targets | clean targets | suspicious targets |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in source_summary.iterrows():
        lines.append(
            f"| {row['SEQ_NAME']} | {int(row['rows'])} | {int(row['frames'])} | {int(row['tracks'])} | "
            f"{int(row['valid_targets'])} | {int(row['clean_targets'])} | {int(row['suspicious_targets'])} |"
        )
    lines.append("")
    lines.append("## Audit Thresholds")
    lines.append("")
    lines.append("| threshold | value |")
    lines.append("|---|---:|")
    for k, v in thresholds.items():
        lines.append(f"| {k} | {float(v):.6g} |")
    lines.append("")
    lines.append("## Coverage By Split")
    lines.append("")
    lines.append("| split | group | rows | fraction of split | fraction of valid |")
    lines.append("|---|---|---:|---:|---:|")
    for _, row in coverage.iterrows():
        lines.append(
            f"| {row['split']} | {row['group']} | {int(row['rows'])} | "
            f"{float(row['fraction_of_split']):.4f} | {float(row['fraction_of_valid']):.4f} |"
        )
    lines.append("")
    lines.append("## Self-Only Tracklet Baselines")
    lines.append("")
    test = baselines.loc[baselines["subset"] == "test_full"].copy()
    test = test.sort_values(["rmse_norm", "window", "model"])
    lines.append("| W | model | test RMSE px | mean R2 | n |")
    lines.append("|---:|---|---:|---:|---:|")
    for _, row in test.iterrows():
        lines.append(
            f"| {int(row['window'])} | {row['model']} | {float(row['rmse_norm']):.6f} | "
            f"{float(row['mean_r2']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## Clean/Suspicious Slice Metrics")
    lines.append("")
    lines.append("| W | model | subset | RMSE px | mean R2 | n |")
    lines.append("|---:|---|---|---:|---:|---:|")
    slice_rows = baselines.loc[baselines["subset"].isin(["test_clean", "test_medium", "test_suspicious"])].copy()
    slice_rows = slice_rows.sort_values(["window", "model", "subset"])
    for _, row in slice_rows.iterrows():
        lines.append(
            f"| {int(row['window'])} | {row['model']} | {row['subset']} | "
            f"{float(row['rmse_norm']):.6f} | {float(row['mean_r2']):.6f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(f"![audit summary]({(out_dir / 'fig_data_audit_tracklet_summary.png').resolve()})")
    lines.append("")
    lines.append("## Initial Reading")
    lines.append("")
    if not test.empty:
        best = test.iloc[0]
        lines.append(
            f"The best self-only baseline in this run is W={int(best['window'])} `{best['model']}` "
            f"with test RMSE {float(best['rmse_norm']):.6f} px and mean R2 {float(best['mean_r2']):.6f}."
        )
    lines.append(
        "Use this as the temporal self-dynamics reference before adding a graph/prior correction. "
        "A future graph model should improve especially on dense, medium, or suspicious slices rather than only on already clean easy cases."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in [
        "audit_annotated_rows.csv",
        "audit_coverage_summary.csv",
        "tracklet_baselines.csv",
        "tracklet_baseline_summary.json",
        "fig_data_audit_tracklet_summary.png",
    ]:
        lines.append(f"- `{out_dir / name}`")
    (out_dir / "data_audit_tracklet_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / str(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_paths = [Path(p) for p in args.table_csv]
    df = load_tables(table_paths)
    df, thresholds = add_audit_columns(df, min_segment_len=int(args.min_segment_len))
    coverage = build_coverage_summary(df)

    source_summary = (
        df.groupby(["SEQ_ID", "SEQ_NAME"], as_index=False)
        .agg(
            rows=("FRAME", "size"),
            frames=("FRAME", "nunique"),
            tracks=("TRACK_ID", "nunique"),
            valid_targets=("has_target", "sum"),
            clean_targets=("audit_clean_conservative", "sum"),
            suspicious_targets=("audit_suspicious", "sum"),
        )
    )

    df.to_csv(out_dir / "audit_annotated_rows.csv", index=False)
    coverage.to_csv(out_dir / "audit_coverage_summary.csv", index=False)
    source_summary.to_csv(out_dir / "audit_source_summary.csv", index=False)

    clean_target = df.copy()
    clean_target["has_target_original"] = clean_target["has_target"]
    clean_target["has_target"] = clean_target["has_target"].astype(bool) & clean_target["audit_clean_conservative"].astype(bool)
    clean_target.to_csv(out_dir / "audit_clean_target_table.csv", index=False)

    baseline_parts: list[pd.DataFrame] = []
    for window in [int(x) for x in str(args.windows).split(",") if str(x).strip()]:
        samples = build_tracklet_samples(df, window=window)
        part = run_tracklet_baselines(
            samples,
            window=window,
            include_gru=bool(args.include_gru),
            torch_epochs=int(args.torch_epochs),
            seed=int(args.seed),
            include_robust=bool(args.include_robust),
            skip_mlp=bool(args.skip_mlp),
        )
        baseline_parts.append(part)
    baselines = pd.concat(baseline_parts, axis=0, ignore_index=True)
    baselines.to_csv(out_dir / "tracklet_baselines.csv", index=False)
    summary = {
        "dataset": str(args.dataset),
        "tables": [str(p) for p in table_paths],
        "thresholds": thresholds,
        "coverage": coverage.to_dict(orient="records"),
        "source_summary": source_summary.to_dict(orient="records"),
        "best_test_full": (
            baselines.loc[baselines["subset"] == "test_full"].sort_values("rmse_norm").head(1).to_dict(orient="records")
        ),
    }
    (out_dir / "tracklet_baseline_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_figures(out_dir, df, coverage, baselines)
    write_report(out_dir, str(args.dataset), table_paths, thresholds, coverage, baselines, source_summary)
    print(json.dumps(to_jsonable(summary["best_test_full"]), ensure_ascii=False, indent=2))
    print(out_dir / "data_audit_tracklet_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tracking quality and run self-only tracklet baselines.")
    parser.add_argument("--dataset", default="psc_causal_w5_h1")
    parser.add_argument("--table-csv", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "data_audit_tracklet")
    parser.add_argument("--windows", default="3,5,7")
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--include-gru", action="store_true")
    parser.add_argument("--include-robust", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--torch-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
