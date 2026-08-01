#!/usr/bin/env python3
"""Flow-first reliability/regime route generator v51.

This runner develops the post-forensic conclusion:

    movement = local tissue flow + route residual + reliability/noise regime.

It is intentionally not another scalar reranker.  It tests whether local
neighbour/tissue flow should be a first-class base signal and whether residual
route experts become more identifiable after subtracting that flow.

Core checks:

1. train-only coordinate/flow feature construction;
2. real local-flow vs no-flow vs shuffled-flow controls;
3. route residual labels over y - flow_base;
4. causal route-prior gate and fixed route residual experts;
5. reliability/error head: can the noisy part be predicted/stratified?

Target/future is used only for supervised training labels, route labels, and
diagnostics.  Inference features are causal: current/past velocity, same-frame
neighbour flow, density/topology, and known image-boundary proxies.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"
DEFAULT_OUT = ROOT / "outputs" / "flow_regime_route_generator_v51_2026-07-10"
EPS = 1e-8


DATASET_IMAGE_SIZE = {
    "MDCK_Bulk": (5632.0, 5632.0),
    "MDCK_Edge": (5632.0, 5632.0),
    "MDAMB231": (2000.0, 2000.0),
}


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FeaturePacket:
    name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    feature_names: list[str]


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Path):
        return str(value)
    return value


def parse_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(s)) for s in parse_csv(text)]


def parse_floats(text: str) -> list[float]:
    return [float(s) for s in parse_csv(text)]


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    # Some causal ratios, especially persistence when the denominator is tiny,
    # can become numerically enormous while carrying no useful metric scale.
    # Clip before scaling so linear models cannot explode on rare degenerate rows.
    return np.clip(x, -1.0e4, 1.0e4).astype(np.float32, copy=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    sse = float(np.sum(np.square(y_true - y_pred)))
    sst = float(np.sum(np.square(y_true - np.mean(y_true, axis=0, keepdims=True))))
    return float(1.0 - sse / max(sst, EPS))


def angular_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1), EPS)
    return float(np.mean(np.sum(y_true * y_pred, axis=1) / denom))


def magnitude_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(y_pred, axis=1)) / max(float(np.mean(np.linalg.norm(y_true, axis=1))), EPS))


def endpoint_slice(horizons: list[int], h: int) -> slice:
    i = horizons.index(int(h))
    return slice(2 * i, 2 * i + 2)


def metric_rows(
    *,
    y_true: np.ndarray,
    pred: np.ndarray,
    base_pred: np.ndarray,
    horizons: list[int],
    method: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in horizons:
        sl = endpoint_slice(horizons, h)
        y = y_true[:, sl]
        p = pred[:, sl]
        b = base_pred[:, sl]
        row: dict[str, Any] = {
            "method": method,
            "horizon": int(h),
            "rmse": rmse(y, p),
            "r2": r2_np(y, p),
            "angular_cosine": angular_cosine(y, p),
            "magnitude_ratio": magnitude_ratio(y, p),
            "base_rmse": rmse(y, b),
            "base_r2": r2_np(y, b),
        }
        row["gain_vs_base_pct"] = float((row["base_rmse"] - row["rmse"]) / max(abs(row["base_rmse"]), EPS) * 100.0)
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def table_path(table_root: Path, dataset: str, seq: int) -> Path:
    p = table_root / dataset / f"{dataset}_{seq:02d}_tracks.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_sequences(table_root: Path, dataset: str, sequences: list[int]) -> pd.DataFrame:
    cols = [
        "dataset",
        "sequence",
        "frame",
        "track_id",
        "x_px",
        "y_px",
        "dx_px",
        "dy_px",
        "target_dx_px",
        "target_dy_px",
        "has_target",
        "QUALITY",
    ]
    parts = []
    for seq in sequences:
        p = table_path(table_root, dataset, seq)
        header = pd.read_csv(p, nrows=0).columns
        usecols = [c for c in cols if c in header]
        df = pd.read_csv(p, usecols=usecols)
        df["dataset"] = dataset
        df["sequence"] = int(seq)
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    for col in ["sequence", "frame", "track_id"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").astype(int)
    for col in ["x_px", "y_px", "dx_px", "dy_px", "QUALITY"]:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")
    full = full.sort_values(["sequence", "track_id", "frame"]).reset_index(drop=True)
    full["_row_uid"] = np.arange(len(full), dtype=np.int64)
    return full


def add_history_and_targets(full: pd.DataFrame, dataset: str, horizons: list[int]) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = full.sort_values(["sequence", "track_id", "frame"]).copy()
    g = df.groupby(["sequence", "track_id"], sort=False)
    dx0 = df["dx_px"].fillna(0.0).astype(float)
    dy0 = df["dy_px"].fillna(0.0).astype(float)
    df["v0_dx"] = dx0
    df["v0_dy"] = dy0
    df["v0_speed"] = np.sqrt(dx0 * dx0 + dy0 * dy0)
    df["v0_log_speed"] = np.log1p(df["v0_speed"])
    df["_v0_speed_tmp"] = df["v0_speed"]
    hist_cols = ["v0_dx", "v0_dy", "v0_speed", "v0_log_speed"]

    for lag in [1, 2, 3, 4, 6, 8]:
        ldx = g["dx_px"].shift(lag).fillna(0.0).astype(float)
        ldy = g["dy_px"].shift(lag).fillna(0.0).astype(float)
        lsp = np.sqrt(ldx * ldx + ldy * ldy)
        valid = g["dx_px"].shift(lag).notna().astype(float)
        df[f"hist_lag{lag}_dx"] = ldx
        df[f"hist_lag{lag}_dy"] = ldy
        df[f"hist_lag{lag}_speed"] = lsp
        df[f"hist_lag{lag}_valid"] = valid
        hist_cols.extend([f"hist_lag{lag}_dx", f"hist_lag{lag}_dy", f"hist_lag{lag}_speed", f"hist_lag{lag}_valid"])

    l1dx = df["hist_lag1_dx"]
    l1dy = df["hist_lag1_dy"]
    denom = np.maximum(df["v0_speed"] * np.sqrt(l1dx * l1dx + l1dy * l1dy), EPS)
    df["hist_accel_dx1"] = df["v0_dx"] - l1dx
    df["hist_accel_dy1"] = df["v0_dy"] - l1dy
    df["hist_turn_cos1"] = (df["v0_dx"] * l1dx + df["v0_dy"] * l1dy) / denom
    df["hist_turn_sin1"] = (df["v0_dx"] * l1dy - df["v0_dy"] * l1dx) / denom
    hist_cols.extend(["hist_accel_dx1", "hist_accel_dy1", "hist_turn_cos1", "hist_turn_sin1"])

    for window in [3, 6, 12]:
        minp = max(1, min(2, window))
        rdx = g["dx_px"].transform(lambda s: s.rolling(window, min_periods=minp).mean()).fillna(0.0).astype(float)
        rdy = g["dy_px"].transform(lambda s: s.rolling(window, min_periods=minp).mean()).fillna(0.0).astype(float)
        rsp = np.sqrt(rdx * rdx + rdy * rdy)
        ssp = g["dx_px"].transform(lambda s: s.rolling(window, min_periods=minp).std()).fillna(0.0).astype(float)
        ssy = g["dy_px"].transform(lambda s: s.rolling(window, min_periods=minp).std()).fillna(0.0).astype(float)
        sumdx = g["dx_px"].transform(lambda s: s.rolling(window, min_periods=1).sum()).fillna(0.0).astype(float)
        sumdy = g["dy_px"].transform(lambda s: s.rolling(window, min_periods=1).sum()).fillna(0.0).astype(float)
        sumsp = g["_v0_speed_tmp"].transform(lambda s: s.rolling(window, min_periods=1).sum()).fillna(0.0).astype(float)
        persistence = np.sqrt(sumdx * sumdx + sumdy * sumdy) / np.maximum(sumsp, EPS)
        cols = [
            f"hist_roll{window}_dx_mean",
            f"hist_roll{window}_dy_mean",
            f"hist_roll{window}_speed_mean",
            f"hist_roll{window}_dx_std",
            f"hist_roll{window}_dy_std",
            f"hist_roll{window}_persistence",
        ]
        df[cols[0]] = rdx
        df[cols[1]] = rdy
        df[cols[2]] = rsp
        df[cols[3]] = ssp
        df[cols[4]] = ssy
        df[cols[5]] = persistence
        hist_cols.extend(cols)

    df["track_age"] = g.cumcount().astype(float)
    df["frame_norm"] = df["frame"].astype(float) / max(float(df["frame"].max()), 1.0)
    w, h = DATASET_IMAGE_SIZE.get(dataset, (float(df["x_px"].max() + 1.0), float(df["y_px"].max() + 1.0)))
    df["x_norm"] = df["x_px"].astype(float) / max(w, 1.0)
    df["y_norm"] = df["y_px"].astype(float) / max(h, 1.0)
    df["boundary_dist_px"] = np.minimum.reduce([df["x_px"].astype(float), df["y_px"].astype(float), w - df["x_px"].astype(float), h - df["y_px"].astype(float)])
    df["boundary_dist_norm"] = df["boundary_dist_px"] / max(min(w, h), 1.0)
    hist_cols.extend(["track_age", "frame_norm", "x_norm", "y_norm", "boundary_dist_px", "boundary_dist_norm"])

    valid = np.ones(len(df), dtype=bool)
    for hh in horizons:
        sx = g["x_px"].shift(-int(hh))
        sy = g["y_px"].shift(-int(hh))
        sf = g["frame"].shift(-int(hh))
        ok = sf.eq(df["frame"] + int(hh)).fillna(False).to_numpy(bool)
        df[f"target_h{hh}_dx"] = (sx - df["x_px"]).where(ok, np.nan)
        df[f"target_h{hh}_dy"] = (sy - df["y_px"]).where(ok, np.nan)
        valid &= ok
    df["valid_all_horizons"] = valid
    df = df.drop(columns=["_v0_speed_tmp"])
    return df.sort_values("_row_uid").reset_index(drop=True), hist_cols, [f"target_h{hh}_{xy}" for hh in horizons for xy in ["dx", "dy"]]


def sample_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy().reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=int(seed), replace=False).sort_values(["sequence", "frame", "track_id"]).reset_index(drop=True)


def make_split(full: pd.DataFrame, args: argparse.Namespace) -> SplitData:
    valid_mask = full["valid_all_horizons"].to_numpy(bool)

    def subset(seqs: list[int], max_rows: int, seed: int) -> pd.DataFrame:
        seq_mask = full["sequence"].isin(seqs).to_numpy(bool)
        idx = np.flatnonzero(valid_mask & seq_mask)
        if max_rows > 0 and len(idx) > max_rows:
            rng = np.random.default_rng(int(seed))
            idx = np.sort(rng.choice(idx, size=int(max_rows), replace=False))
        return full.iloc[idx].copy().reset_index(drop=True)

    return SplitData(
        train=subset(parse_ints(args.train_seq), args.max_train_rows, args.seed + 101),
        val=subset(parse_ints(args.val_seq), args.max_val_rows, args.seed + 102),
        test=subset(parse_ints(args.test_seq), args.max_test_rows, args.seed + 103),
    )


def add_local_flow_features(full: pd.DataFrame, query: pd.DataFrame, ks: list[int], radii: list[float]) -> tuple[pd.DataFrame, list[str]]:
    if query.empty:
        return query.copy(), []
    max_k = max(ks)
    out = query.copy()
    flow_cols: list[str] = []
    for k in ks:
        cols = [
            f"tf_k{k}_mean_dx",
            f"tf_k{k}_mean_dy",
            f"tf_k{k}_speed_mean",
            f"tf_k{k}_speed_std",
            f"tf_k{k}_vel_std_dx",
            f"tf_k{k}_vel_std_dy",
            f"tf_k{k}_dist_mean",
            f"tf_k{k}_dist_min",
            f"tf_k{k}_self_flow_cos",
            f"tf_k{k}_self_flow_resid_mag",
            f"tf_k{k}_pos_anisotropy",
            f"tf_k{k}_vel_anisotropy",
        ]
        for c in cols:
            out[c] = 0.0
        flow_cols.extend(cols)
    for r in radii:
        c = f"tf_r{int(r)}_density"
        out[c] = 0.0
        flow_cols.append(c)

    full_sorted = full.sort_values(["sequence", "frame"]).copy()
    full_sorted["_pool_pos"] = np.arange(len(full_sorted), dtype=np.int64)
    query_pos = out.index.to_numpy()
    out_pos_map = {int(uid): i for i, uid in enumerate(out["_row_uid"].to_numpy())}
    row_updates: dict[str, np.ndarray] = {c: out[c].to_numpy(np.float32, copy=True) for c in flow_cols}

    for (seq, frame), pool in full_sorted.groupby(["sequence", "frame"], sort=False):
        q = out[(out["sequence"].eq(seq)) & (out["frame"].eq(frame))]
        if q.empty or len(pool) <= 1:
            continue
        coords = pool[["x_px", "y_px"]].to_numpy(np.float32)
        vels = pool[["v0_dx", "v0_dy"]].to_numpy(np.float32)
        tracks = pool["track_id"].to_numpy(np.int64)
        tree = cKDTree(coords)
        qcoords = q[["x_px", "y_px"]].to_numpy(np.float32)
        qtracks = q["track_id"].to_numpy(np.int64)
        qspeed = q["v0_speed"].to_numpy(np.float32)
        qvel = q[["v0_dx", "v0_dy"]].to_numpy(np.float32)
        k_eff = min(max_k + 1, len(pool))
        try:
            dists, inds = tree.query(qcoords, k=k_eff, workers=-1)
        except TypeError:
            dists, inds = tree.query(qcoords, k=k_eff)
        if k_eff == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        radius_counts = {}
        for r in radii:
            try:
                cnt = tree.query_ball_point(qcoords, r=float(r), return_length=True, workers=-1)
            except TypeError:
                cnt = tree.query_ball_point(qcoords, r=float(r), return_length=True)
            radius_counts[float(r)] = np.maximum(np.asarray(cnt, dtype=np.float32) - 1.0, 0.0) / max(math.pi * float(r) * float(r), EPS)

        for local_i, uid in enumerate(q["_row_uid"].to_numpy()):
            out_i = out_pos_map[int(uid)]
            idx_all = np.asarray(inds[local_i], dtype=np.int64)
            dist_all = np.asarray(dists[local_i], dtype=np.float32)
            ok = (idx_all >= 0) & (idx_all < len(pool)) & (tracks[idx_all] != qtracks[local_i])
            idx_valid = idx_all[ok]
            dist_valid = dist_all[ok]
            for r in radii:
                row_updates[f"tf_r{int(r)}_density"][out_i] = radius_counts[float(r)][local_i]
            if idx_valid.size == 0:
                continue
            for k in ks:
                take = idx_valid[: min(k, idx_valid.size)]
                dd = dist_valid[: min(k, dist_valid.size)]
                vv = vels[take]
                mean_v = vv.mean(axis=0)
                sp = np.linalg.norm(vv, axis=1)
                qv = qvel[local_i]
                denom = max(float(np.linalg.norm(qv) * np.linalg.norm(mean_v)), EPS)
                cos = float(np.dot(qv, mean_v) / denom) if denom > EPS else 0.0
                resid = float(np.linalg.norm(qv - mean_v))
                rel_pos = coords[take] - qcoords[local_i][None, :]
                cov_pos = np.cov(rel_pos.T) if len(take) >= 3 else np.eye(2, dtype=np.float32)
                cov_vel = np.cov(vv.T) if len(take) >= 3 else np.eye(2, dtype=np.float32)
                eig_pos = np.linalg.eigvalsh(np.nan_to_num(cov_pos, nan=0.0))
                eig_vel = np.linalg.eigvalsh(np.nan_to_num(cov_vel, nan=0.0))
                row_updates[f"tf_k{k}_mean_dx"][out_i] = mean_v[0]
                row_updates[f"tf_k{k}_mean_dy"][out_i] = mean_v[1]
                row_updates[f"tf_k{k}_speed_mean"][out_i] = float(sp.mean()) if sp.size else 0.0
                row_updates[f"tf_k{k}_speed_std"][out_i] = float(sp.std()) if sp.size else 0.0
                row_updates[f"tf_k{k}_vel_std_dx"][out_i] = float(vv[:, 0].std()) if len(vv) else 0.0
                row_updates[f"tf_k{k}_vel_std_dy"][out_i] = float(vv[:, 1].std()) if len(vv) else 0.0
                row_updates[f"tf_k{k}_dist_mean"][out_i] = float(dd.mean()) if dd.size else 0.0
                row_updates[f"tf_k{k}_dist_min"][out_i] = float(dd.min()) if dd.size else 0.0
                row_updates[f"tf_k{k}_self_flow_cos"][out_i] = cos
                row_updates[f"tf_k{k}_self_flow_resid_mag"][out_i] = resid
                row_updates[f"tf_k{k}_pos_anisotropy"][out_i] = float((eig_pos[-1] - eig_pos[0]) / max(eig_pos[-1] + eig_pos[0], EPS))
                row_updates[f"tf_k{k}_vel_anisotropy"][out_i] = float((eig_vel[-1] - eig_vel[0]) / max(eig_vel[-1] + eig_vel[0], EPS))

    for c, arr in row_updates.items():
        out[c] = arr
    # Keep original query order.
    return out.loc[query_pos].reset_index(drop=True), flow_cols


def target_matrix(df: pd.DataFrame, horizons: list[int]) -> np.ndarray:
    cols = [f"target_h{h}_{xy}" for h in horizons for xy in ["dx", "dy"]]
    return safe_matrix(df, cols)


def rollout_from_step(step: np.ndarray, horizons: list[int]) -> np.ndarray:
    return np.concatenate([float(h) * step for h in horizons], axis=1).astype(np.float32)


def fit_ridge_cv(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, alphas: list[float]) -> tuple[StandardScaler, Ridge, float, float]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(xtr).astype(np.float32)
    zva = scaler.transform(xva).astype(np.float32)
    ztr = np.clip(np.nan_to_num(ztr), -8.0, 8.0)
    zva = np.clip(np.nan_to_num(zva), -8.0, 8.0)
    best: tuple[float, Ridge, float] | None = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), solver="svd")
        model.fit(ztr, ytr)
        pred = np.nan_to_num(model.predict(zva).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        score = rmse(yva, pred) if np.isfinite(pred).all() else float("inf")
        if best is None or score < best[0]:
            best = (score, model, float(alpha))
    assert best is not None
    return scaler, best[1], best[2], best[0]


def ridge_predict(scaler: StandardScaler, model: Ridge, x: np.ndarray) -> np.ndarray:
    z = scaler.transform(x).astype(np.float32)
    z = np.clip(np.nan_to_num(z), -8.0, 8.0)
    return np.nan_to_num(model.predict(z).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_hgbdt_multi(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, args: argparse.Namespace) -> tuple[list[HistGradientBoostingRegressor], float]:
    models: list[HistGradientBoostingRegressor] = []
    preds = []
    for j in range(ytr.shape[1]):
        m = HistGradientBoostingRegressor(
            max_iter=int(args.hgbdt_iter),
            learning_rate=float(args.hgbdt_lr),
            max_leaf_nodes=int(args.hgbdt_leaf_nodes),
            l2_regularization=float(args.hgbdt_l2),
            random_state=int(args.seed) + 5100 + j,
        )
        m.fit(xtr, ytr[:, j])
        models.append(m)
        preds.append(m.predict(xva).astype(np.float32))
    p = np.stack(preds, axis=1).astype(np.float32)
    return models, rmse(yva, p)


def predict_hgbdt_multi(models: list[HistGradientBoostingRegressor], x: np.ndarray) -> np.ndarray:
    return np.stack([m.predict(x).astype(np.float32) for m in models], axis=1).astype(np.float32)


def fit_route_labels(residual_train: np.ndarray, residual_val: np.ndarray, residual_test: np.ndarray, k: int, seed: int) -> tuple[KMeans, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(residual_train).astype(np.float32)
    zva = scaler.transform(residual_val).astype(np.float32)
    zte = scaler.transform(residual_test).astype(np.float32)
    k_eff = max(2, min(int(k), len(ztr) // 50))
    km = KMeans(n_clusters=k_eff, n_init=20, random_state=int(seed))
    km.fit(ztr)
    return km, km.predict(ztr).astype(np.int64), km.predict(zva).astype(np.int64), km.predict(zte).astype(np.int64), scaler.scale_.astype(np.float32)


def fit_route_prior(xtr: np.ndarray, labels: np.ndarray, xva: np.ndarray, xte: np.ndarray, args: argparse.Namespace) -> tuple[StandardScaler, LogisticRegression, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    ztr = np.clip(np.nan_to_num(scaler.fit_transform(xtr).astype(np.float32)), -8.0, 8.0)
    zva = np.clip(np.nan_to_num(scaler.transform(xva).astype(np.float32)), -8.0, 8.0)
    zte = np.clip(np.nan_to_num(scaler.transform(xte).astype(np.float32)), -8.0, 8.0)
    clf = LogisticRegression(
        C=float(args.route_prior_c),
        max_iter=int(args.route_prior_max_iter),
        class_weight="balanced",
        random_state=int(args.seed) + 6100,
    )
    clf.fit(ztr, labels)
    return scaler, clf, clf.predict_proba(ztr).astype(np.float32), clf.predict_proba(zva).astype(np.float32), clf.predict_proba(zte).astype(np.float32)


def align_probs(probs: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(probs), int(k)), dtype=np.float32)
    for j, c in enumerate(classes):
        out[:, int(c)] = probs[:, j]
    s = out.sum(axis=1, keepdims=True)
    missing = s[:, 0] <= EPS
    if np.any(missing):
        out[missing] = 1.0 / float(k)
        s = out.sum(axis=1, keepdims=True)
    return (out / np.maximum(s, EPS)).astype(np.float32)


def route_gate_metrics(probs: np.ndarray, labels: np.ndarray, split: str, packet: str) -> dict[str, Any]:
    k = probs.shape[1]
    pred = np.argmax(probs, axis=1)
    row: dict[str, Any] = {
        "packet": packet,
        "split": split,
        "route_k": int(k),
        "top1": float(accuracy_score(labels, pred)),
        "true_route_prob": float(np.mean(probs[np.arange(len(labels)), labels])) if len(labels) else float("nan"),
        "entropy": float(np.mean(-np.sum(probs * np.log(np.maximum(probs, EPS)), axis=1))),
    }
    try:
        row["top3"] = float(top_k_accuracy_score(labels, probs, k=min(3, k), labels=np.arange(k)))
    except Exception:
        row["top3"] = float("nan")
    try:
        row["nll"] = float(log_loss(labels, probs, labels=np.arange(k)))
    except Exception:
        row["nll"] = float("nan")
    return row


def fit_route_experts(xtr: np.ndarray, residual_train: np.ndarray, labels: np.ndarray, k: int, args: argparse.Namespace) -> list[tuple[StandardScaler, Ridge]]:
    experts: list[tuple[StandardScaler, Ridge]] = []
    # Global fallback is also better than failing tiny clusters.
    global_scaler, global_model, _, _ = fit_ridge_cv(xtr, residual_train, xtr, residual_train, [float(args.route_expert_alpha)])
    for c in range(int(k)):
        idx = np.flatnonzero(labels == c)
        if len(idx) < int(args.min_route_cluster_samples):
            experts.append((global_scaler, global_model))
            continue
        scaler = StandardScaler()
        z = np.clip(np.nan_to_num(scaler.fit_transform(xtr[idx]).astype(np.float32)), -8.0, 8.0)
        model = Ridge(alpha=float(args.route_expert_alpha), solver="svd")
        model.fit(z, residual_train[idx])
        experts.append((scaler, model))
    return experts


def predict_route_experts(experts: list[tuple[StandardScaler, Ridge]], x: np.ndarray) -> np.ndarray:
    preds = []
    for scaler, model in experts:
        preds.append(ridge_predict(scaler, model, x))
    return np.stack(preds, axis=1).astype(np.float32)


def route_mixture(expert_pred: np.ndarray, probs: np.ndarray, mode: str, top_m: int, power: float, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    n, k = probs.shape
    if mode == "all_uniform":
        w = np.full_like(probs, 1.0 / float(k))
    else:
        c = max(1, min(int(top_m), k))
        idx = np.argsort(-probs, axis=1)[:, :c]
        vals = np.take_along_axis(probs, idx, axis=1)
        if mode == "top_uniform":
            wv = np.full_like(vals, 1.0 / float(c))
        elif mode == "top_prob":
            raw = np.power(np.maximum(vals, EPS), float(power) / max(float(temperature), EPS))
            wv = raw / np.maximum(raw.sum(axis=1, keepdims=True), EPS)
        else:
            raise ValueError(f"Unknown mixture mode {mode}")
        w = np.zeros_like(probs)
        np.put_along_axis(w, idx, wv, axis=1)
    pred = np.sum(w[:, :, None] * expert_pred, axis=1).astype(np.float32)
    return pred, w.astype(np.float32)


def build_feature_packets(split: SplitData, hist_cols: list[str], flow_cols: list[str], args: argparse.Namespace) -> dict[str, FeaturePacket]:
    all_cols = hist_cols + flow_cols
    self_cols = hist_cols
    rng = np.random.default_rng(int(args.seed) + 8100)

    def make(name: str, cols: list[str], shuffle_flow: bool) -> FeaturePacket:
        tr = split.train.copy()
        va = split.val.copy()
        te = split.test.copy()
        if shuffle_flow and flow_cols:
            for df, off in [(tr, 1), (va, 2), (te, 3)]:
                order = np.random.default_rng(int(args.seed) + 8200 + off).permutation(len(df))
                vals = df[flow_cols].to_numpy(np.float32)
                df.loc[:, flow_cols] = vals[order]
        return FeaturePacket(
            name=name,
            x_train=safe_matrix(tr, cols),
            x_val=safe_matrix(va, cols),
            x_test=safe_matrix(te, cols),
            feature_names=cols,
        )

    packets = {
        "real_flow": make("real_flow", all_cols, False),
        "no_flow": make("no_flow", self_cols, False),
    }
    if flow_cols:
        packets["shuffled_flow"] = make("shuffled_flow", all_cols, True)
    return packets


def endpoint_error_h6(y: np.ndarray, pred: np.ndarray, horizons: list[int]) -> np.ndarray:
    sl = endpoint_slice(horizons, max(horizons))
    return np.sqrt(np.sum(np.square(y[:, sl] - pred[:, sl]), axis=1)).astype(np.float32)


def reliability_probe(packet: FeaturePacket, ytr: np.ndarray, yva: np.ndarray, yte: np.ndarray, base_tr: np.ndarray, base_va: np.ndarray, base_te: np.ndarray, horizons: list[int], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    err_tr = np.log1p(endpoint_error_h6(ytr, base_tr, horizons))
    err_va = np.log1p(endpoint_error_h6(yva, base_va, horizons))
    err_te = np.log1p(endpoint_error_h6(yte, base_te, horizons))
    model = HistGradientBoostingRegressor(
        max_iter=min(120, int(args.hgbdt_iter)),
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=int(args.seed) + 9001,
    )
    model.fit(packet.x_train, err_tr)
    pred_va = model.predict(packet.x_val).astype(np.float32)
    pred_te = model.predict(packet.x_test).astype(np.float32)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    actual_va = np.expm1(err_va)
    actual_te = np.expm1(err_te)
    rows = [
        {
            "packet": packet.name,
            "split": "val",
            "pred_error_corr": corr(np.expm1(pred_va), actual_va),
            "pred_log_error_rmse": rmse(err_va[:, None], pred_va[:, None]),
            "actual_error_mean": float(np.mean(actual_va)),
            "pred_error_mean": float(np.mean(np.expm1(pred_va))),
        },
        {
            "packet": packet.name,
            "split": "test",
            "pred_error_corr": corr(np.expm1(pred_te), actual_te),
            "pred_log_error_rmse": rmse(err_te[:, None], pred_te[:, None]),
            "actual_error_mean": float(np.mean(actual_te)),
            "pred_error_mean": float(np.mean(np.expm1(pred_te))),
        },
    ]
    q = pd.qcut(pd.Series(pred_te), q=4, labels=False, duplicates="drop")
    for b in sorted(q.dropna().unique()):
        mask = q.to_numpy() == int(b)
        rows.append(
            {
                "packet": packet.name,
                "split": "test",
                "bin": f"pred_error_q{int(b)+1}",
                "rows": int(mask.sum()),
                "actual_error_mean": float(np.mean(actual_te[mask])) if mask.any() else float("nan"),
                "pred_error_mean": float(np.mean(np.expm1(pred_te[mask]))) if mask.any() else float("nan"),
            }
        )
    return pred_va[:, None], pred_te[:, None], pd.DataFrame(rows)


def calibrate_on_val(features_val: np.ndarray, y_val: np.ndarray, features_test: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scaler = StandardScaler()
    zva = np.clip(np.nan_to_num(scaler.fit_transform(features_val).astype(np.float32)), -8.0, 8.0)
    zte = np.clip(np.nan_to_num(scaler.transform(features_test).astype(np.float32)), -8.0, 8.0)
    model = Ridge(alpha=float(args.calibrator_alpha), solver="svd")
    model.fit(zva, y_val)
    return np.nan_to_num(model.predict(zte).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def run_packet(
    packet: FeaturePacket,
    split: SplitData,
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    horizons: list[int],
    args: argparse.Namespace,
    *,
    shuffled_labels: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    alphas = parse_floats(args.ridge_alphas)
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"packet": packet.name, "shuffled_labels": bool(shuffled_labels), "feature_dim": int(packet.x_train.shape[1])}

    self_step_tr = split.train[["v0_dx", "v0_dy"]].to_numpy(np.float32)
    self_step_va = split.val[["v0_dx", "v0_dy"]].to_numpy(np.float32)
    self_step_te = split.test[["v0_dx", "v0_dy"]].to_numpy(np.float32)
    self_tr = rollout_from_step(self_step_tr, horizons)
    self_va = rollout_from_step(self_step_va, horizons)
    self_te = rollout_from_step(self_step_te, horizons)
    base_ref_te = self_te
    rows.extend(metric_rows(y_true=yte, pred=self_te, base_pred=base_ref_te, horizons=horizons, method=f"{packet.name}_self_rollout", extra={"stage": "causal_baseline", "packet": packet.name}))

    flow_cols = [c for c in packet.feature_names if c.startswith("tf_k") and c.endswith("_mean_dx")]
    flow_candidates = []
    for cdx in flow_cols:
        prefix = cdx[: -len("_mean_dx")]
        cdy = f"{prefix}_mean_dy"
        if cdy in split.train.columns:
            flow_candidates.append(prefix)
    for prefix in flow_candidates:
        step_te = split.test[[f"{prefix}_mean_dx", f"{prefix}_mean_dy"]].to_numpy(np.float32)
        pred_te = rollout_from_step(step_te, horizons)
        rows.extend(metric_rows(y_true=yte, pred=pred_te, base_pred=base_ref_te, horizons=horizons, method=f"{packet.name}_{prefix}_rollout", extra={"stage": "flow_rollout", "packet": packet.name}))

    # Direct flow/state base.
    scaler_base, ridge_base, alpha_base, val_base = fit_ridge_cv(packet.x_train, ytr, packet.x_val, yva, alphas)
    base_tr = ridge_predict(scaler_base, ridge_base, packet.x_train)
    base_va = ridge_predict(scaler_base, ridge_base, packet.x_val)
    base_te = ridge_predict(scaler_base, ridge_base, packet.x_test)
    diagnostics.update({"base_alpha": alpha_base, "base_val_rmse": val_base})
    rows.extend(metric_rows(y_true=yte, pred=base_te, base_pred=base_ref_te, horizons=horizons, method=f"{packet.name}_ridge_flow_state_base", extra={"stage": "flow_state_base", "packet": packet.name, "alpha": alpha_base, "val_rmse": val_base}))

    hgb_te = None
    if args.use_hgbdt:
        hgb_models, hgb_val = fit_hgbdt_multi(packet.x_train, ytr, packet.x_val, yva, args)
        hgb_te = predict_hgbdt_multi(hgb_models, packet.x_test)
        rows.extend(metric_rows(y_true=yte, pred=hgb_te, base_pred=base_ref_te, horizons=horizons, method=f"{packet.name}_hgbdt_flow_state_base", extra={"stage": "flow_state_base_hgbdt", "packet": packet.name, "val_rmse": hgb_val}))
        if hgb_val < val_base:
            # Keep route residual over the stronger validation base.
            base_tr = predict_hgbdt_multi(hgb_models, packet.x_train)
            base_va = predict_hgbdt_multi(hgb_models, packet.x_val)
            base_te = hgb_te
            diagnostics.update({"base_model": "hgbdt", "base_val_rmse": hgb_val})
        else:
            diagnostics.update({"base_model": "ridge"})
    else:
        diagnostics.update({"base_model": "ridge"})

    rel_va, rel_te, rel_df = reliability_probe(packet, ytr, yva, yte, base_tr, base_va, base_te, horizons, args)

    residual_tr = ytr - base_tr
    residual_va = yva - base_va
    residual_te = yte - base_te
    km, labels_tr, labels_va, labels_te, _ = fit_route_labels(residual_tr, residual_va, residual_te, int(args.route_k), int(args.seed) + 9200)
    if shuffled_labels:
        rng = np.random.default_rng(int(args.seed) + 9300)
        labels_tr = labels_tr[rng.permutation(len(labels_tr))]
    prior_scaler, prior_clf, p_tr_raw, p_va_raw, p_te_raw = fit_route_prior(packet.x_train, labels_tr, packet.x_val, packet.x_test, args)
    k = int(km.n_clusters)
    p_tr = align_probs(p_tr_raw, prior_clf.classes_, k)
    p_va = align_probs(p_va_raw, prior_clf.classes_, k)
    p_te = align_probs(p_te_raw, prior_clf.classes_, k)
    gate_rows.extend([route_gate_metrics(p_va, labels_va, "val", f"{packet.name}{'_shuflabels' if shuffled_labels else ''}"), route_gate_metrics(p_te, labels_te, "test", f"{packet.name}{'_shuflabels' if shuffled_labels else ''}")])

    experts = fit_route_experts(packet.x_train, residual_tr, labels_tr, k, args)
    exp_va = predict_route_experts(experts, packet.x_val)
    exp_te = predict_route_experts(experts, packet.x_test)
    global_scaler, global_model, global_alpha, global_val = fit_ridge_cv(packet.x_train, residual_tr, packet.x_val, residual_va, alphas)
    global_va = ridge_predict(global_scaler, global_model, packet.x_val)
    global_te = ridge_predict(global_scaler, global_model, packet.x_test)
    rows.extend(metric_rows(y_true=yte, pred=base_te + global_te, base_pred=base_ref_te, horizons=horizons, method=f"{packet.name}_global_residual_ridge", extra={"stage": "global_residual", "packet": packet.name, "alpha": global_alpha, "val_rmse": global_val}))

    for mode in parse_csv(args.mixture_modes):
        parts = mode.split(":")
        name = parts[0]
        top_m = int(parts[1]) if len(parts) > 1 else int(args.route_top_m)
        power = float(parts[2]) if len(parts) > 2 else float(args.route_prob_power)
        r_va, w_va = route_mixture(exp_va, p_va, name, top_m, power, float(args.route_temperature))
        r_te, w_te = route_mixture(exp_te, p_te, name, top_m, power, float(args.route_temperature))
        pred_va = base_va + r_va
        pred_te = base_te + r_te
        method = f"{packet.name}_route_{mode}{'_shuflabels' if shuffled_labels else ''}"
        rows.extend(metric_rows(y_true=yte, pred=pred_te, base_pred=base_ref_te, horizons=horizons, method=method, extra={"stage": "route_residual_generator", "packet": packet.name, "mixture": mode, "route_k": k}))

        cal_va = np.concatenate([base_va, pred_va, base_va + global_va, p_va, rel_va], axis=1).astype(np.float32)
        cal_te = np.concatenate([base_te, pred_te, base_te + global_te, p_te, rel_te], axis=1).astype(np.float32)
        cal_pred_te = calibrate_on_val(cal_va, yva, cal_te, args)
        rows.extend(metric_rows(y_true=yte, pred=cal_pred_te, base_pred=base_ref_te, horizons=horizons, method=f"{method}_val_calibrated", extra={"stage": "route_residual_val_calibrated", "packet": packet.name, "mixture": mode, "route_k": k}))

    diagnostics.update(
        {
            "route_k_eff": k,
            "cluster_min": int(np.bincount(labels_tr, minlength=k).min()),
            "cluster_max": int(np.bincount(labels_tr, minlength=k).max()),
            "global_residual_alpha": global_alpha,
            "global_residual_val_rmse": global_val,
        }
    )
    return rows, gate_rows, rel_df, diagnostics


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, gate: pd.DataFrame, rel: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# v51 Flow-First Reliability/Regime Route Generator")
    lines.append("")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- train/val/test seq: `{args.train_seq}` / `{args.val_seq}` / `{args.test_seq}`")
    lines.append(f"- rows: train `{args.max_train_rows}`, val `{args.max_val_rows}`, test `{args.max_test_rows}`")
    lines.append("")
    lines.append("## Decision")
    best_h6 = summary[summary["horizon"].eq(max(parse_ints(args.horizons)))].sort_values("rmse").head(12)
    if not best_h6.empty:
        best = best_h6.iloc[0]
        lines.append(f"Best h{max(parse_ints(args.horizons))}: `{best['method']}` RMSE `{best['rmse']:.4f}`, R2 `{best['r2']:.4f}`.")
    real_best = best_h6[best_h6["packet"].astype(str).str.contains("real_flow", na=False)] if "packet" in best_h6.columns else pd.DataFrame()
    ctrl_best = best_h6[~best_h6["packet"].astype(str).str.contains("real_flow", na=False)] if "packet" in best_h6.columns else pd.DataFrame()
    if not real_best.empty and not ctrl_best.empty:
        margin = float(ctrl_best["rmse"].min() - real_best["rmse"].min())
        lines.append(f"Best real-flow margin versus best non-real-flow control on h6: `{margin:.4f}` RMSE points.")
    lines.append("")
    lines.append("## h6 Leaderboard")
    cols = [c for c in ["method", "rmse", "r2", "angular_cosine", "magnitude_ratio", "stage", "packet", "mixture", "route_k", "val_rmse"] if c in best_h6.columns]
    lines.append(best_h6[cols].to_markdown(index=False))
    lines.append("")
    for h in parse_ints(args.horizons):
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse").head(15)
        lines.append(f"## h{h}")
        lines.append(sub[cols].to_markdown(index=False))
        lines.append("")
    if not gate.empty:
        lines.append("## Route Gate")
        lines.append(gate.to_markdown(index=False))
        lines.append("")
    if not rel.empty:
        lines.append("## Reliability/Error Probe")
        lines.append(rel.head(80).to_markdown(index=False))
        lines.append("")
    if not diag.empty:
        lines.append("## Diagnostics")
        lines.append(diag.to_markdown(index=False))
    (out_dir / "flow_regime_route_generator_v51_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    np.random.seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_ints(args.horizons)
    sequences = sorted(set(parse_ints(args.train_seq) + parse_ints(args.val_seq) + parse_ints(args.test_seq)))
    full_raw = load_sequences(Path(args.table_root), args.dataset, sequences)
    full, hist_cols, target_cols = add_history_and_targets(full_raw, args.dataset, horizons)
    split0 = make_split(full, args)
    ks = parse_ints(args.local_ks)
    radii = parse_floats(args.local_radii)
    train, flow_cols = add_local_flow_features(full, split0.train, ks, radii)
    val, flow_cols_va = add_local_flow_features(full, split0.val, ks, radii)
    test, flow_cols_te = add_local_flow_features(full, split0.test, ks, radii)
    flow_cols = [c for c in flow_cols if c in flow_cols_va and c in flow_cols_te]
    split = SplitData(train=train, val=val, test=test)
    ytr = target_matrix(split.train, horizons)
    yva = target_matrix(split.val, horizons)
    yte = target_matrix(split.test, horizons)
    packets = build_feature_packets(split, hist_cols, flow_cols, args)

    all_rows: list[dict[str, Any]] = []
    all_gate: list[dict[str, Any]] = []
    rel_parts: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    packet_names = parse_csv(args.packets)
    for name in packet_names:
        if name not in packets:
            continue
        rows, gate_rows, rel_df, diag = run_packet(packets[name], split, ytr, yva, yte, horizons, args, shuffled_labels=False)
        all_rows.extend(rows)
        all_gate.extend(gate_rows)
        rel_parts.append(rel_df)
        diag_rows.append(diag)
    if "real_flow" in packets and args.include_shuffled_route_labels:
        rows, gate_rows, rel_df, diag = run_packet(packets["real_flow"], split, ytr, yva, yte, horizons, args, shuffled_labels=True)
        all_rows.extend(rows)
        all_gate.extend(gate_rows)
        rel_parts.append(rel_df)
        diag["packet"] = "real_flow_shuffled_route_labels"
        diag_rows.append(diag)

    summary = pd.DataFrame(all_rows)
    if not summary.empty:
        summary.insert(0, "seed", int(args.seed))
        summary.insert(0, "dataset", str(args.dataset))
    gate = pd.DataFrame(all_gate)
    if not gate.empty:
        gate.insert(0, "seed", int(args.seed))
        gate.insert(0, "dataset", str(args.dataset))
    rel = pd.concat(rel_parts, ignore_index=True) if rel_parts else pd.DataFrame()
    if not rel.empty:
        rel.insert(0, "seed", int(args.seed))
        rel.insert(0, "dataset", str(args.dataset))
    diag = pd.DataFrame(diag_rows)
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))

    summary.to_csv(args.out_dir / "flow_regime_route_generator_v51_summary.csv", index=False)
    gate.to_csv(args.out_dir / "flow_regime_route_generator_v51_route_gate.csv", index=False)
    rel.to_csv(args.out_dir / "flow_regime_route_generator_v51_reliability.csv", index=False)
    diag.to_csv(args.out_dir / "flow_regime_route_generator_v51_diagnostics.csv", index=False)
    meta = {
        "elapsed_sec": time.time() - t0,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "rows": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "hist_cols": hist_cols,
        "flow_cols": flow_cols,
        "target_cols": target_cols,
        "args": vars(args),
    }
    (args.out_dir / "flow_regime_route_generator_v51_meta.json").write_text(json.dumps(finite_json(meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, gate, rel, diag)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "gate_rows": len(gate), "elapsed_sec": time.time() - t0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", type=str, default="MDCK_Bulk")
    parser.add_argument("--train-seq", type=str, default="1,2,3,4")
    parser.add_argument("--val-seq", type=str, default="5")
    parser.add_argument("--test-seq", type=str, default="6")
    parser.add_argument("--horizons", type=str, default="1,2,4,6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=20000)
    parser.add_argument("--local-ks", type=str, default="8,16,32")
    parser.add_argument("--local-radii", type=str, default="64,128,256")
    parser.add_argument("--packets", type=str, default="real_flow,no_flow,shuffled_flow")
    parser.add_argument("--ridge-alphas", type=str, default="0.1,0.3,1,3,10,30,100,300,1000,3000")
    parser.add_argument("--use-hgbdt", action="store_true")
    parser.add_argument("--hgbdt-iter", type=int, default=180)
    parser.add_argument("--hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--hgbdt-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgbdt-l2", type=float, default=0.03)
    parser.add_argument("--route-k", type=int, default=12)
    parser.add_argument("--route-prior-c", type=float, default=0.35)
    parser.add_argument("--route-prior-max-iter", type=int, default=500)
    parser.add_argument("--route-expert-alpha", type=float, default=300.0)
    parser.add_argument("--min-route-cluster-samples", type=int, default=80)
    parser.add_argument("--mixture-modes", type=str, default="top_uniform:4:1,top_uniform:8:1,top_prob:8:1,top_prob:8:2,all_uniform:999:1")
    parser.add_argument("--route-top-m", type=int, default=8)
    parser.add_argument("--route-prob-power", type=float, default=1.0)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--calibrator-alpha", type=float, default=3000.0)
    parser.add_argument("--include-shuffled-route-labels", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1200)
        args.max_val_rows = min(args.max_val_rows, 400)
        args.max_test_rows = min(args.max_test_rows, 600)
        args.local_ks = "8,16"
        args.local_radii = "128"
        args.route_k = min(args.route_k, 8)
        args.mixture_modes = "top_uniform:4:1,top_prob:4:1"
        args.packets = "real_flow,no_flow,shuffled_flow"
        args.include_shuffled_route_labels = True
        args.hgbdt_iter = min(args.hgbdt_iter, 40)
    run(args)


if __name__ == "__main__":
    main()
