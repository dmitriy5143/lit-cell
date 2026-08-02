#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "collective_dataset_candidate_screen"
DEFAULT_DOWNLOAD_ROOT = ROOT / "new_data" / "ctc_candidate_screen"


CTC_CANDIDATES: dict[str, dict[str, Any]] = {
    "PhC-C2DH-U373": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip",
        "px_um_by_seq": {"01": 0.65, "02": 0.65},
        "dt_seconds_by_seq": {"01": 900.0, "02": 900.0},
        "why": "small CTC phase-contrast glioblastoma/astrocytoma dataset; plausible migration benchmark.",
    },
    "Fluo-C2DL-MSC": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip",
        "px_um_by_seq": {"01": 0.30, "02": 0.3977},
        "dt_seconds_by_seq": {"01": 1200.0, "02": 1800.0},
        "why": "small CTC mesenchymal stem-cell dataset; plausible collective migration/transfer candidate.",
    },
    "DIC-C2DH-HeLa": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",
        "px_um_by_seq": {"01": 0.19, "02": 0.19},
        "dt_seconds_by_seq": {"01": 600.0, "02": 600.0},
        "why": "small CTC DIC HeLa dataset; useful dense 2D sanity candidate if ST/TRA coverage is usable.",
    },
    "Fluo-N2DH-GOWT1": {
        "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip",
        "px_um_by_seq": {"01": 0.240, "02": 0.240},
        "dt_seconds_by_seq": {"01": 300.0, "02": 300.0},
        "why": "small nuclear CTC dataset with potentially cleaner centroid tracks.",
    },
}


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
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


def run_cmd(cmd: list[str], log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode == 0


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return
    try:
        response = urllib.request.urlopen(url, timeout=120)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise
        print(f"warning: TLS certificate verification failed for {url}; retrying with unverified SSL context", flush=True)
        response = urllib.request.urlopen(url, timeout=120, context=ssl._create_unverified_context())
    with response, out_path.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0") or "0")
        read = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                print(f"download {out_path.name}: {100.0 * read / total:5.1f}%", flush=True)


def locate_ctc_root(extract_dir: Path, dataset: str) -> Path:
    direct = extract_dir / dataset
    if (direct / "01").exists() or (direct / "01_ST").exists():
        return direct
    if (extract_dir / "01").exists() or (extract_dir / "01_ST").exists():
        return extract_dir
    candidates = [p for p in extract_dir.rglob(dataset) if p.is_dir()]
    for p in candidates:
        if (p / "01").exists() or (p / "01_ST").exists():
            return p
    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1 and ((children[0] / "01").exists() or (children[0] / "01_ST").exists()):
        return children[0]
    raise FileNotFoundError(f"Could not locate extracted CTC root for {dataset} under {extract_dir}")


def ensure_ctc_dataset(dataset: str, download_root: Path, logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    if dataset not in CTC_CANDIDATES:
        raise KeyError(f"Unknown CTC candidate {dataset}. Known: {sorted(CTC_CANDIDATES)}")
    spec = CTC_CANDIDATES[dataset]
    zip_path = download_root / "zips" / f"{dataset}.zip"
    extract_dir = download_root / "raw" / dataset
    if not extract_dir.exists():
        download_file(str(spec["url"]), zip_path)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        (logs_dir / f"{dataset}_extract.log").write_text(f"extracted {zip_path} -> {extract_dir}\n", encoding="utf-8")
    return locate_ctc_root(extract_dir, dataset)


def build_ctc_tables(
    *,
    dataset: str,
    ctc_root: Path,
    out_dir: Path,
    logs_dir: Path,
) -> list[Path]:
    spec = CTC_CANDIDATES[dataset]
    tables_dir = out_dir / "ctc_tables" / dataset
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for seq in ("01", "02"):
        if not (ctc_root / seq).exists():
            continue
        if (ctc_root / f"{seq}_ST" / "SEG").exists():
            source = "ST"
        elif (ctc_root / f"{seq}_GT" / "TRA").exists():
            source = "GT_TRA"
        else:
            continue
        out_path = tables_dir / f"{dataset}_{seq}_{source}.csv"
        if not out_path.exists():
            px = float(spec["px_um_by_seq"].get(seq, 1.0))
            dt = float(spec["dt_seconds_by_seq"].get(seq, 1.0))
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "build_ctc_tracks_table.py"),
                "--dataset-root",
                str(ctc_root),
                "--sequence",
                seq,
                "--seg-source",
                source,
                "--px-um",
                str(px),
                "--py-um",
                str(px),
                "--dt-seconds",
                str(dt),
                "--out",
                str(out_path),
            ]
            ok = run_cmd(cmd, logs_dir / f"build_{dataset}_{seq}_{source}.log")
            if not ok:
                continue
        if out_path.exists():
            paths.append(out_path)
    return paths


def standardize_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if "frame" in df.columns and "FRAME" not in df.columns:
        rename["frame"] = "FRAME"
    if "track_id" in df.columns and "TRACK_ID" not in df.columns:
        rename["track_id"] = "TRACK_ID"
    if rename:
        df = df.rename(columns=rename)
    need = {"FRAME", "TRACK_ID", "x_px", "y_px"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    for col in ["FRAME", "TRACK_ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.loc[df["FRAME"].notna() & df["TRACK_ID"].notna()].copy()
    df["FRAME"] = df["FRAME"].astype(int)
    df["TRACK_ID"] = df["TRACK_ID"].astype(int)
    df = df.loc[df["TRACK_ID"] >= 0].copy()
    for col in ["x_px", "y_px"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[df["x_px"].notna() & df["y_px"].notna()].copy()
    df = df.drop_duplicates(["TRACK_ID", "FRAME"], keep="first")
    df = df.sort_values(["TRACK_ID", "FRAME"]).reset_index(drop=True)
    g = df.groupby("TRACK_ID", sort=False)
    if "dx_px" not in df.columns:
        df["dx_px"] = g["x_px"].diff()
        df["dy_px"] = g["y_px"].diff()
    else:
        df["dx_px"] = pd.to_numeric(df["dx_px"], errors="coerce")
        df["dy_px"] = pd.to_numeric(df["dy_px"], errors="coerce")
    next_frame = g["FRAME"].shift(-1)
    next_x = g["x_px"].shift(-1)
    next_y = g["y_px"].shift(-1)
    consecutive = next_frame.notna() & (next_frame.astype(float) == df["FRAME"].astype(float) + 1.0)
    if "target_dx_px" not in df.columns or "target_dy_px" not in df.columns:
        df["target_dx_px"] = (next_x - df["x_px"]).where(consecutive)
        df["target_dy_px"] = (next_y - df["y_px"]).where(consecutive)
        df["has_target"] = consecutive
    else:
        df["target_dx_px"] = pd.to_numeric(df["target_dx_px"], errors="coerce").where(consecutive)
        df["target_dy_px"] = pd.to_numeric(df["target_dy_px"], errors="coerce").where(consecutive)
        if "has_target" in df.columns:
            df["has_target"] = df["has_target"].astype(bool) & consecutive
        else:
            df["has_target"] = consecutive
    df["source_path"] = str(path)
    return df


def causal_smooth_table(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if int(window) <= 1:
        return df.copy()
    out = df.sort_values(["TRACK_ID", "FRAME"]).copy()
    xs = []
    ys = []
    for _, g in out.groupby("TRACK_ID", sort=False):
        xs.append(g["x_px"].rolling(window=int(window), min_periods=1).mean())
        ys.append(g["y_px"].rolling(window=int(window), min_periods=1).mean())
    out["x_px"] = pd.concat(xs).sort_index()
    out["y_px"] = pd.concat(ys).sort_index()
    g = out.groupby("TRACK_ID", sort=False)
    out["dx_px"] = g["x_px"].diff()
    out["dy_px"] = g["y_px"].diff()
    next_frame = g["FRAME"].shift(-1)
    consecutive = next_frame.notna() & (next_frame.astype(float) == out["FRAME"].astype(float) + 1.0)
    out["target_dx_px"] = (g["x_px"].shift(-1) - out["x_px"]).where(consecutive)
    out["target_dy_px"] = (g["y_px"].shift(-1) - out["y_px"]).where(consecutive)
    out["has_target"] = consecutive
    return out


@dataclass
class FeaturePack:
    frame: np.ndarray
    split: np.ndarray
    target: np.ndarray
    self_features: np.ndarray
    geometry_features: np.ndarray
    neighbor_features: np.ndarray
    valid: np.ndarray
    feature_names_self: list[str]
    feature_names_geometry: list[str]
    feature_names_neighbor: list[str]
    neighbor_persistence: float


def frame_splits(frames: np.ndarray) -> dict[int, str]:
    unique = np.sort(np.unique(frames.astype(int)))
    n = int(unique.size)
    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n)) if n - n_train > 1 else max(0, n - n_train)
    train = set(unique[:n_train].tolist())
    val = set(unique[n_train:n_train + n_val].tolist())
    test = set(unique[n_train + n_val:].tolist())
    out: dict[int, str] = {}
    for f in train:
        out[int(f)] = "train"
    for f in val:
        out[int(f)] = "val"
    for f in test:
        out[int(f)] = "test"
    return out


def make_features(df: pd.DataFrame, *, neighbor_radius_px: float, k_neighbors: int) -> FeaturePack:
    d = df.sort_values(["FRAME", "TRACK_ID"]).reset_index(drop=True).copy()
    frames = d["FRAME"].to_numpy(int)
    split_map = frame_splits(frames)
    split = np.asarray([split_map.get(int(f), "unknown") for f in frames], dtype=object)
    target = d[["target_dx_px", "target_dy_px"]].to_numpy(float)
    prev = d[["dx_px", "dy_px"]].to_numpy(float)
    prev = np.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.linalg.norm(prev, axis=1)
    width = max(float(d["x_px"].max() - d["x_px"].min()), 1.0)
    height = max(float(d["y_px"].max() - d["y_px"].min()), 1.0)
    x0 = float(d["x_px"].min())
    y0 = float(d["y_px"].min())
    x_norm = (d["x_px"].to_numpy(float) - x0) / width - 0.5
    y_norm = (d["y_px"].to_numpy(float) - y0) / height - 0.5
    boundary = np.minimum.reduce([x_norm + 0.5, 0.5 - x_norm, y_norm + 0.5, 0.5 - y_norm])

    neigh_count = np.zeros(len(d), dtype=np.float32)
    nn_dist = np.full(len(d), np.nan, dtype=np.float32)
    neigh_vel = np.zeros((len(d), 2), dtype=np.float32)
    neigh_rel = np.zeros((len(d), 2), dtype=np.float32)
    inv_field = np.zeros((len(d), 2), dtype=np.float32)
    topk_by_row: dict[int, set[int]] = {}

    for _, idx0 in d.groupby("FRAME", sort=True).groups.items():
        idx = np.asarray(list(idx0), dtype=int)
        pos = d.loc[idx, ["x_px", "y_px"]].to_numpy(float)
        vel = prev[idx]
        track = d.loc[idx, "TRACK_ID"].to_numpy(int)
        n = int(len(idx))
        if n <= 1:
            continue
        tree = cKDTree(pos)
        dist, nbr = tree.query(pos, k=min(max(2, int(k_neighbors) + 1), n))
        if dist.ndim == 1:
            dist = dist[:, None]
            nbr = nbr[:, None]
        for local_i, global_i in enumerate(idx):
            keep = nbr[local_i] != local_i
            nb = nbr[local_i][keep][: int(k_neighbors)]
            ds = dist[local_i][keep][: int(k_neighbors)]
            if nb.size == 0:
                continue
            within = ds <= float(neighbor_radius_px)
            use = nb[within] if np.any(within) else nb
            ds_use = ds[within] if np.any(within) else ds
            rel = pos[use] - pos[local_i]
            neigh_count[global_i] = float(np.sum(within))
            nn_dist[global_i] = float(ds[0]) if ds.size else np.nan
            neigh_vel[global_i] = vel[use].mean(axis=0).astype(np.float32)
            neigh_rel[global_i] = rel.mean(axis=0).astype(np.float32)
            unit = rel / np.clip(np.linalg.norm(rel, axis=1, keepdims=True), 1e-6, np.inf)
            weights = 1.0 / np.clip(ds_use, 1.0, np.inf)
            inv_field[global_i] = np.sum(unit * weights[:, None], axis=0).astype(np.float32)
            topk_by_row[global_i] = set(int(track[j]) for j in nb[: int(k_neighbors)])

    persistence_vals: list[float] = []
    row_by_key = {(int(t), int(f)): int(i) for i, (t, f) in enumerate(zip(d["TRACK_ID"], d["FRAME"]))}
    for i, (track_id, frame) in enumerate(zip(d["TRACK_ID"], d["FRAME"])):
        j = row_by_key.get((int(track_id), int(frame) + 1))
        if j is None:
            continue
        a = topk_by_row.get(i)
        b = topk_by_row.get(j)
        if not a or not b:
            continue
        persistence_vals.append(len(a & b) / max(len(a | b), 1))
    neighbor_persistence = float(np.mean(persistence_vals)) if persistence_vals else float("nan")

    nn_fill = np.nanmedian(nn_dist[np.isfinite(nn_dist)]) if np.isfinite(nn_dist).any() else float(neighbor_radius_px)
    nn_dist = np.nan_to_num(nn_dist, nan=nn_fill, posinf=nn_fill, neginf=nn_fill)
    density = neigh_count / max(float(np.nanmedian(neigh_count[neigh_count > 0])) if np.any(neigh_count > 0) else 1.0, 1.0)
    self_features = np.column_stack([prev[:, 0], prev[:, 1], speed, x_norm, y_norm, boundary])
    geometry_features = np.column_stack([self_features, nn_dist, neigh_count, density, neigh_rel[:, 0], neigh_rel[:, 1], inv_field[:, 0], inv_field[:, 1]])
    neighbor_features = np.column_stack([geometry_features, neigh_vel[:, 0], neigh_vel[:, 1]])
    valid = (
        d["has_target"].astype(bool).to_numpy()
        & np.isfinite(target).all(axis=1)
        & np.asarray(split != "unknown")
    )
    return FeaturePack(
        frame=frames,
        split=split,
        target=target,
        self_features=self_features.astype(np.float32),
        geometry_features=geometry_features.astype(np.float32),
        neighbor_features=neighbor_features.astype(np.float32),
        valid=valid,
        feature_names_self=["prev_dx", "prev_dy", "prev_speed", "x_norm", "y_norm", "boundary"],
        feature_names_geometry=[
            "prev_dx",
            "prev_dy",
            "prev_speed",
            "x_norm",
            "y_norm",
            "boundary",
            "nn_dist",
            "neighbor_count",
            "density",
            "neighbor_rel_x",
            "neighbor_rel_y",
            "inv_field_x",
            "inv_field_y",
        ],
        feature_names_neighbor=[
            "prev_dx",
            "prev_dy",
            "prev_speed",
            "x_norm",
            "y_norm",
            "boundary",
            "nn_dist",
            "neighbor_count",
            "density",
            "neighbor_rel_x",
            "neighbor_rel_y",
            "inv_field_x",
            "inv_field_y",
            "neighbor_vx",
            "neighbor_vy",
        ],
        neighbor_persistence=neighbor_persistence,
    )


def standardize_features(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64).copy()
    z[~np.isfinite(z)] = np.nan
    med = np.zeros(z.shape[1], dtype=np.float64)
    train_z = z[train]
    if train_z.size:
        for col in range(z.shape[1]):
            vals = train_z[:, col]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                med[col] = float(np.median(vals))
    bad_r, bad_c = np.where(~np.isfinite(z))
    if bad_r.size:
        z[bad_r, bad_c] = med[bad_c]
    if int(train.sum()) == 0:
        return np.zeros_like(z, dtype=np.float32)
    mean = z[train].mean(axis=0)
    std = z[train].std(axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    out = (z - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=25.0, neginf=-25.0)
    return np.clip(out, -25.0, 25.0).astype(np.float32)


def ridge_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, float]:
    z = standardize_features(x, train)
    finite_rows = np.isfinite(z).all(axis=1) & np.isfinite(y).all(axis=1)
    train = train & finite_rows
    val = val & finite_rows
    best_alpha = 1.0
    best_val = float("inf")
    best_pred = None
    if int(train.sum()) == 0:
        return np.zeros_like(y, dtype=np.float32), best_alpha
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        xt = np.ascontiguousarray(z[train].astype(np.float64))
        yt = np.ascontiguousarray(y[train].astype(np.float64))
        y_mean = yt.mean(axis=0, keepdims=True)
        yc = yt - y_mean
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gram = xt.T @ xt
            rhs = xt.T @ yc
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all() or not np.isfinite(y_mean).all():
            continue
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]
        if not np.isfinite(coef).all():
            continue
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pred = z.astype(np.float64) @ coef + y_mean
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        if int(val.sum()) > 0:
            err = pred[val] - y[val]
            score = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
        else:
            score = float(alpha)
        if not np.isfinite(score):
            continue
        if score < best_val:
            best_val = score
            best_alpha = float(alpha)
            best_pred = pred.astype(np.float32)
    if best_pred is None:
        fallback = np.tile(y[train].mean(axis=0, keepdims=True), (y.shape[0], 1))
        best_pred = fallback.astype(np.float32)
    return best_pred, best_alpha


def metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    m = mask & np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    if int(m.sum()) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2_mean": float("nan"), "n": 0}
    yy = y[m]
    pp = pred[m]
    err = pp - yy
    mse = np.mean(err * err, axis=0)
    var = np.var(yy, axis=0)
    r2 = 1.0 - mse / np.clip(var, 1e-12, np.inf)
    return {
        "rmse": float(np.sqrt(np.mean(np.sum(err * err, axis=1)))),
        "mae": float(np.mean(np.linalg.norm(err, axis=1))),
        "r2_dx": float(r2[0]),
        "r2_dy": float(r2[1]),
        "r2_mean": float(np.mean(r2)),
        "n": int(m.sum()),
    }


def screen_variant(df: pd.DataFrame, *, dataset: str, table: str, variant: str, neighbor_radius_px: float, k_neighbors: int) -> dict[str, Any]:
    pack = make_features(df, neighbor_radius_px=neighbor_radius_px, k_neighbors=k_neighbors)
    valid = pack.valid
    train = valid & (pack.split == "train")
    val = valid & (pack.split == "val")
    test = valid & (pack.split == "test")
    y = pack.target
    mean_pred = np.tile(y[train].mean(axis=0, keepdims=True), (y.shape[0], 1)) if int(train.sum()) else np.zeros_like(y)
    prev_pred = pack.self_features[:, :2]
    self_pred, self_alpha = ridge_predict(pack.self_features, y, train, val)
    geom_pred, geom_alpha = ridge_predict(pack.geometry_features, y, train, val)
    neigh_pred, neigh_alpha = ridge_predict(pack.neighbor_features, y, train, val)
    rows = {
        "dataset": dataset,
        "table": table,
        "variant": variant,
        "rows": int(len(df)),
        "frames": int(df["FRAME"].nunique()),
        "tracks": int(df["TRACK_ID"].nunique()),
        "valid_targets": int(valid.sum()),
        "train_targets": int(train.sum()),
        "val_targets": int(val.sum()),
        "test_targets": int(test.sum()),
        "objects_per_frame_median": float(df.groupby("FRAME").size().median()),
        "objects_per_frame_q10": float(df.groupby("FRAME").size().quantile(0.10)),
        "objects_per_frame_q90": float(df.groupby("FRAME").size().quantile(0.90)),
        "track_len_median": float(df.groupby("TRACK_ID")["FRAME"].nunique().median()),
        "track_len_q10": float(df.groupby("TRACK_ID")["FRAME"].nunique().quantile(0.10)),
        "neighbor_persistence_jaccard": pack.neighbor_persistence,
        "self_alpha": self_alpha,
        "geometry_alpha": geom_alpha,
        "neighbor_alpha": neigh_alpha,
    }
    model_preds = {
        "mean": mean_pred,
        "previous": prev_pred,
        "self_ridge": self_pred,
        "geometry_ridge": geom_pred,
        "neighbor_flow_ridge": neigh_pred,
    }
    for name, pred in model_preds.items():
        mm = metrics(y, pred, test)
        for k, v in mm.items():
            rows[f"{name}_{k}"] = v
    rows["gain_geometry_vs_self_pct"] = 100.0 * (rows["self_ridge_rmse"] - rows["geometry_ridge_rmse"]) / max(rows["self_ridge_rmse"], 1e-12)
    rows["gain_neighbor_vs_self_pct"] = 100.0 * (rows["self_ridge_rmse"] - rows["neighbor_flow_ridge_rmse"]) / max(rows["self_ridge_rmse"], 1e-12)
    rows["gain_neighbor_vs_geometry_pct"] = 100.0 * (rows["geometry_ridge_rmse"] - rows["neighbor_flow_ridge_rmse"]) / max(rows["geometry_ridge_rmse"], 1e-12)
    return rows


def screen_table(path: Path, *, dataset: str, neighbor_radius_px: float, k_neighbors: int) -> list[dict[str, Any]]:
    raw = standardize_table(path)
    out = [
        screen_variant(raw, dataset=dataset, table=str(path), variant="raw_h1", neighbor_radius_px=neighbor_radius_px, k_neighbors=k_neighbors),
    ]
    for window in (3, 5):
        smooth = causal_smooth_table(raw, window=window)
        out.append(
            screen_variant(
                smooth,
                dataset=dataset,
                table=str(path),
                variant=f"causal_w{window}_h1",
                neighbor_radius_px=neighbor_radius_px,
                k_neighbors=k_neighbors,
            )
        )
    return out


def priority_score(row: pd.Series) -> float:
    density = np.clip(np.log1p(float(row.get("objects_per_frame_median", 0.0))) / np.log1p(120.0), 0.0, 1.0)
    length = np.clip(np.log1p(float(row.get("track_len_median", 0.0))) / np.log1p(180.0), 0.0, 1.0)
    predict = np.clip((float(row.get("self_ridge_r2_mean", 0.0)) + 0.1) / 0.8, 0.0, 1.0)
    neigh = np.clip(float(row.get("gain_neighbor_vs_self_pct", 0.0)) / 5.0, -1.0, 1.5)
    persist = np.clip(float(row.get("neighbor_persistence_jaccard", 0.0)) / 0.40, 0.0, 1.0)
    return float(0.22 * density + 0.18 * length + 0.24 * predict + 0.26 * neigh + 0.10 * persist)


def verdict(row: pd.Series) -> str:
    if float(row.get("objects_per_frame_median", 0.0)) < 12:
        return "too_sparse"
    if float(row.get("track_len_median", 0.0)) < 12:
        return "too_fragmented"
    if float(row.get("self_ridge_r2_mean", -1.0)) < 0.05:
        return "weak_forecast_signal"
    if float(row.get("gain_neighbor_vs_self_pct", 0.0)) >= 3.0 and float(row.get("gain_neighbor_vs_geometry_pct", 0.0)) >= 1.0:
        return "strong_collective_candidate"
    if float(row.get("gain_neighbor_vs_self_pct", 0.0)) >= 1.0:
        return "moderate_collective_candidate"
    return "self_motion_only_or_noise"


def gather_tables(args: argparse.Namespace, out_dir: Path) -> list[tuple[str, Path]]:
    logs = out_dir / "logs"
    gathered: list[tuple[str, Path]] = []
    for item in args.table:
        p = Path(item)
        gathered.append((p.stem, p))
    for item in args.table_dir:
        directory = Path(item)
        for path in sorted(directory.glob("*_tracks.csv")):
            gathered.append((path.stem, path))
    if args.include_existing:
        defaults = [
            ("PSC01_ST", ROOT / "outputs" / "new_dataset_transfer" / "PhC-C2DL-PSC" / "tables" / "psc01_tracks_table.csv"),
            ("PSC02_ST", ROOT / "outputs" / "new_dataset_transfer" / "PhC-C2DL-PSC" / "tables" / "psc02_tracks_table.csv"),
            ("HSC01_ST", ROOT / "outputs" / "ctc_hsc" / "tables" / "hsc01_tracks_table.csv"),
            ("HSC02_ST", ROOT / "outputs" / "ctc_hsc" / "tables" / "hsc02_tracks_table.csv"),
            ("HSC01_GTTRA", ROOT / "outputs" / "ctc_hsc" / "tables_gt_tra" / "hsc01_gttra_tracks_table.csv"),
            ("HSC02_GTTRA", ROOT / "outputs" / "ctc_hsc" / "tables_gt_tra" / "hsc02_gttra_tracks_table.csv"),
        ]
        gathered.extend((name, path) for name, path in defaults if path.exists())
    for dataset in args.download_ctc:
        ctc_root = ensure_ctc_dataset(dataset, Path(args.download_root), logs)
        built = build_ctc_tables(dataset=dataset, ctc_root=ctc_root, out_dir=out_dir, logs_dir=logs)
        gathered.extend((f"{dataset}_{p.stem.split('_')[-2]}_{p.stem.split('_')[-1]}", p) for p in built)
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for name, path in gathered:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append((name, path))
    return unique


def write_report(summary: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    best = summary.sort_values("priority_score", ascending=False).head(20)
    cols = [
        "dataset",
        "variant",
        "verdict",
        "priority_score",
        "objects_per_frame_median",
        "track_len_median",
        "self_ridge_r2_mean",
        "gain_neighbor_vs_self_pct",
        "gain_neighbor_vs_geometry_pct",
        "neighbor_persistence_jaccard",
        "neighbor_flow_ridge_rmse",
    ]
    candidate_lines = [
        "# Collective dataset candidate screen",
        "",
        "The screen ranks datasets by forecast-readiness and incremental neighbour-flow observability. A high self-motion R2 alone is not enough; the candidate should show neighbour/geometry signal beyond self history.",
        "",
        "## Top Rows",
        "",
        best[cols].to_markdown(index=False),
        "",
        "## Interpretation Rules",
        "",
        "- `strong_collective_candidate`: neighbour-flow Ridge improves self Ridge by at least 3% and geometry by at least 1%.",
        "- `moderate_collective_candidate`: neighbour-flow improves self Ridge by at least 1%.",
        "- `self_motion_only_or_noise`: forecastable motion may exist, but collective signal is weak in this table/target.",
        "- `too_sparse`, `too_fragmented`, `weak_forecast_signal`: poor candidates for our prior claim.",
        "",
        "## CTC candidates configured",
        "",
    ]
    for name, spec in CTC_CANDIDATES.items():
        candidate_lines.append(f"- `{name}`: {spec['url']} - {spec['why']}")
    candidate_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary CSV: `{out_dir / 'collective_dataset_screen_summary.csv'}`",
            f"- run config: `{out_dir / 'run_config.json'}`",
        ]
    )
    (out_dir / "collective_dataset_candidate_screen_report.md").write_text(
        "\n".join(candidate_lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--table", action="append", default=[], help="Extra track table CSV to screen.")
    p.add_argument(
        "--table-dir",
        action="append",
        default=[],
        help="Directory whose *_tracks.csv files should be screened.",
    )
    p.add_argument("--include-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--download-ctc",
        nargs="*",
        default=[],
        choices=sorted(CTC_CANDIDATES),
        help="Download/build selected small CTC training datasets before screening.",
    )
    p.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    p.add_argument("--neighbor-radius-px", type=float, default=50.0)
    p.add_argument("--k-neighbors", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(to_jsonable(vars(args)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tables = gather_tables(args, out_dir)
    manifest = [{"dataset": name, "path": str(path)} for name, path in tables]
    (out_dir / "screened_tables_manifest.json").write_text(
        json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for name, path in tables:
        print(f"screening {name}: {path}", flush=True)
        try:
            rows.extend(
                screen_table(
                    path,
                    dataset=name,
                    neighbor_radius_px=float(args.neighbor_radius_px),
                    k_neighbors=int(args.k_neighbors),
                )
            )
        except Exception as exc:
            rows.append({"dataset": name, "table": str(path), "variant": "ERROR", "error": repr(exc)})
    summary = pd.DataFrame(rows)
    if not summary.empty and "error" not in summary.columns:
        summary["priority_score"] = summary.apply(priority_score, axis=1)
        summary["verdict"] = summary.apply(verdict, axis=1)
    elif not summary.empty:
        ok_mask = ~summary.get("error", pd.Series([np.nan] * len(summary))).notna()
        if ok_mask.any():
            summary.loc[ok_mask, "priority_score"] = summary.loc[ok_mask].apply(priority_score, axis=1)
            summary.loc[ok_mask, "verdict"] = summary.loc[ok_mask].apply(verdict, axis=1)
    summary.to_csv(out_dir / "collective_dataset_screen_summary.csv", index=False)
    write_report(summary.loc[summary.get("error", pd.Series([np.nan] * len(summary))).isna()].copy(), out_dir, args)
    print(out_dir / "collective_dataset_candidate_screen_report.md")


if __name__ == "__main__":
    main()
