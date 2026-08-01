#!/usr/bin/env python3
"""Causal tissue-flow/PIV-like feature extraction and probe for LaChance data.

This is the first raw-video feature gate after coordinate-only audits stalled.
For a cell at frame t, features are extracted only from frames t-1 and t:

    optical flow / local tissue motion: I[t-1] -> I[t]

The supervised target may be t -> t+h, but no future frame/image is used as an
inference feature.  The runner tests whether these tissue-flow observables add
deployable signal over trajectory-only h1-first style baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tifffile
    from skimage.registration import optical_flow_ilk
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore[assignment]
    optical_flow_ilk = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_image_feature_extraction as ife  # noqa: E402
import run_lachance_image_feature_probe as ifp  # noqa: E402

DEFAULT_TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"
DEFAULT_STACK_DIR = (
    ROOT
    / "new_data"
    / "lachance_epithelia"
    / "raw_timelapse"
    / "extracted_stacks"
    / "MDCK_Bulk_Timelapse_Data_Sample_Tissues"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_tissue_flow_feature_probe_2026-06-15"
EPS = 1e-8


def finite_json(value: Any) -> Any:
    return ifp.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text or "").split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def parse_frame_spec(text: str) -> list[int]:
    text = str(text or "").strip()
    if ":" in text:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"bad frame spec: {text}")
        return list(range(start, stop, step))
    return parse_ints(text)


def read_frame(stack_dir: Path, sequence: int, frame: int, cache: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError("tifffile is required")
    key = (int(sequence), int(frame))
    if key in cache:
        return cache[key]
    path = stack_dir / f"{int(sequence):02d}.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    image = ife.read_tiff_page(path, int(frame))
    cache[key] = image
    if len(cache) > 6:
        for old_key in list(cache.keys())[:-6]:
            cache.pop(old_key, None)
    return image


def downsample_norm(image: np.ndarray, downsample: int) -> np.ndarray:
    small = np.asarray(image)[:: int(downsample), :: int(downsample)]
    return ife.normalize_image(small).astype(np.float32, copy=False)


def compute_flow(prev: np.ndarray, cur: np.ndarray, *, downsample: int, radius: int, num_warp: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flow-like u/v fields in original-pixel units.

    skimage returns `(v, u)` in downsampled-pixel units.  The sign convention is
    intentionally not hard-coded as "physical truth"; downstream probes see both
    signed projections and can learn the usable convention from causal past
    motion.  Multiplication by downsample converts the field scale to px/frame.
    """

    if optical_flow_ilk is None:
        raise RuntimeError("skimage.registration.optical_flow_ilk is required")
    prev_s = downsample_norm(prev, downsample)
    cur_s = downsample_norm(cur, downsample)
    v, u = optical_flow_ilk(cur_s, prev_s, radius=int(radius), num_warp=int(num_warp), gaussian=True, prefilter=True)
    return (u.astype(np.float32) * float(downsample), v.astype(np.float32) * float(downsample))


def local_slice(x: float, y: float, shape: tuple[int, int], radius: int) -> tuple[slice, slice]:
    h, w = shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    x0 = max(0, xi - int(radius))
    x1 = min(w, xi + int(radius) + 1)
    y0 = max(0, yi - int(radius))
    y1 = min(h, yi + int(radius) + 1)
    return slice(y0, y1), slice(x0, x1)


def safe_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    a = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return float(a.mean()), float(a.std()), float(np.median(a)), float(np.percentile(a, 90))


def flow_patch_features(
    *,
    u: np.ndarray,
    v: np.ndarray,
    div: np.ndarray,
    curl: np.ndarray,
    shear1: np.ndarray,
    shear2: np.ndarray,
    x: float,
    y: float,
    own_dx: float,
    own_dy: float,
    radius_small: int,
    prefix: str,
) -> dict[str, float]:
    ys, xs = local_slice(x, y, u.shape, radius_small)
    up = u[ys, xs]
    vp = v[ys, xs]
    mag = np.sqrt(up * up + vp * vp)
    out: dict[str, float] = {}
    u_mean, u_std, u_med, u_p90 = safe_stats(up)
    v_mean, v_std, v_med, v_p90 = safe_stats(vp)
    mag_mean, mag_std, mag_med, mag_p90 = safe_stats(mag)
    out.update(
        {
            f"{prefix}u_mean": u_mean,
            f"{prefix}v_mean": v_mean,
            f"{prefix}u_std": u_std,
            f"{prefix}v_std": v_std,
            f"{prefix}u_median": u_med,
            f"{prefix}v_median": v_med,
            f"{prefix}u_p90": u_p90,
            f"{prefix}v_p90": v_p90,
            f"{prefix}mag_mean": mag_mean,
            f"{prefix}mag_std": mag_std,
            f"{prefix}mag_median": mag_med,
            f"{prefix}mag_p90": mag_p90,
        }
    )
    yi = int(np.clip(round(float(y)), 0, u.shape[0] - 1))
    xi = int(np.clip(round(float(x)), 0, u.shape[1] - 1))
    center_u = float(u[yi, xi])
    center_v = float(v[yi, xi])
    own = np.asarray([float(own_dx), float(own_dy)], dtype=np.float32)
    own_norm = float(np.linalg.norm(own))
    flow_vec = np.asarray([u_mean, v_mean], dtype=np.float32)
    flow_norm = float(np.linalg.norm(flow_vec))
    if own_norm > EPS:
        direction = own / own_norm
        tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    else:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
        tangent = np.asarray([0.0, 1.0], dtype=np.float32)
    flow_proj = float(np.dot(flow_vec, direction))
    flow_tangent = float(np.dot(flow_vec, tangent))
    cosine = float(np.dot(flow_vec, own) / max(flow_norm * own_norm, EPS)) if own_norm > EPS else 0.0
    out.update(
        {
            f"{prefix}center_u": center_u,
            f"{prefix}center_v": center_v,
            f"{prefix}own_minus_mean_u": float(own[0] - u_mean),
            f"{prefix}own_minus_mean_v": float(own[1] - v_mean),
            f"{prefix}own_minus_center_u": float(own[0] - center_u),
            f"{prefix}own_minus_center_v": float(own[1] - center_v),
            f"{prefix}proj_own_dir": flow_proj,
            f"{prefix}proj_tangent": flow_tangent,
            f"{prefix}cos_to_own": cosine,
        }
    )
    for name, field in (("div", div), ("curl", curl), ("shear1", shear1), ("shear2", shear2)):
        m, s, med, p90 = safe_stats(field[ys, xs])
        out[f"{prefix}{name}_mean"] = m
        out[f"{prefix}{name}_std"] = s
        out[f"{prefix}{name}_median"] = med
        out[f"{prefix}{name}_p90"] = p90

    # Front/back asymmetry along the current self-motion direction.
    h, w = up.shape
    if h > 0 and w > 0:
        yy, xx = np.indices((h, w))
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        rel_x = xx - cx
        rel_y = yy - cy
        front_mask = rel_x * direction[0] + rel_y * direction[1] >= 0.0
        back_mask = ~front_mask
        if np.any(front_mask) and np.any(back_mask):
            front_u = float(np.mean(up[front_mask]))
            front_v = float(np.mean(vp[front_mask]))
            back_u = float(np.mean(up[back_mask]))
            back_v = float(np.mean(vp[back_mask]))
            front_vec = np.asarray([front_u, front_v], dtype=np.float32)
            back_vec = np.asarray([back_u, back_v], dtype=np.float32)
            delta = front_vec - back_vec
            out[f"{prefix}front_back_u"] = float(delta[0])
            out[f"{prefix}front_back_v"] = float(delta[1])
            out[f"{prefix}front_back_proj"] = float(np.dot(delta, direction))
            out[f"{prefix}front_back_tangent"] = float(np.dot(delta, tangent))
        else:
            out[f"{prefix}front_back_u"] = 0.0
            out[f"{prefix}front_back_v"] = 0.0
            out[f"{prefix}front_back_proj"] = 0.0
            out[f"{prefix}front_back_tangent"] = 0.0
    return out


def derive_flow_gradients(u: np.ndarray, v: np.ndarray, downsample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    scale = max(float(downsample), 1.0)
    du_dx = du_dx / scale
    du_dy = du_dy / scale
    dv_dx = dv_dx / scale
    dv_dy = dv_dy / scale
    div = du_dx + dv_dy
    curl = dv_dx - du_dy
    shear1 = du_dx - dv_dy
    shear2 = du_dy + dv_dx
    return div.astype(np.float32), curl.astype(np.float32), shear1.astype(np.float32), shear2.astype(np.float32)


def extract_tissue_flow_features(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    cache: dict[tuple[int, int], np.ndarray] = {}
    frames = [f for f in parse_frame_spec(args.frames) if f >= 1]
    radii = parse_ints(args.radii)
    point_index: pd.DataFrame | None = None
    if args.point_index is not None:
        point_index = pd.read_csv(
            args.point_index,
            usecols=["dataset", "sequence", "frame", "track_id", "x_px", "y_px"],
        )
        point_index = point_index[point_index["dataset"].eq(args.dataset)].copy()
        point_index["sequence"] = point_index["sequence"].astype(int)
        point_index["frame"] = point_index["frame"].astype(int)
        point_index["track_id"] = point_index["track_id"].astype(int)
        point_index = point_index[point_index["frame"].isin(frames)].drop_duplicates(
            ["sequence", "frame", "track_id"]
        )
    for sequence in parse_ints(args.sequences):
        table_path = args.table_root / args.dataset / f"{args.dataset}_{sequence:02d}_tracks.csv"
        if not table_path.exists():
            raise FileNotFoundError(table_path)
        needed = set(frames)
        usecols = [c for c in ifp.TRACK_COLS if c in pd.read_csv(table_path, nrows=0).columns]
        table = pd.read_csv(table_path, usecols=usecols)
        table = table[table["frame"].isin(needed)].copy()
        table["dx_px"] = table.get("dx_px", 0.0).fillna(0.0)
        table["dy_px"] = table.get("dy_px", 0.0).fillna(0.0)
        print(f"[flow] seq={sequence:02d} frames={len(frames)} rows={len(table)}", flush=True)
        for frame in frames:
            pts = table[table["frame"].eq(frame)].copy()
            if pts.empty:
                continue
            if point_index is not None:
                wanted = point_index[
                    point_index["sequence"].eq(sequence) & point_index["frame"].eq(frame)
                ][["track_id", "x_px", "y_px"]].copy()
                if wanted.empty:
                    continue
                # Track identity is the contract.  Coordinates are taken from
                # the tracking table so velocity and position come from the
                # same observation; the index coordinates are retained only
                # to audit alignment.
                wanted = wanted.rename(columns={"x_px": "index_x_px", "y_px": "index_y_px"})
                pts = pts.merge(wanted, on="track_id", how="inner", validate="one_to_one")
                if pts.empty:
                    continue
                pts["index_alignment_px"] = np.sqrt(
                    np.square(pts["x_px"].to_numpy(np.float32) - pts["index_x_px"].to_numpy(np.float32))
                    + np.square(pts["y_px"].to_numpy(np.float32) - pts["index_y_px"].to_numpy(np.float32))
                )
            elif args.max_points_per_frame > 0 and len(pts) > args.max_points_per_frame:
                pts = pts.sample(n=int(args.max_points_per_frame), random_state=int(args.seed) + sequence * 1009 + frame)
            prev = read_frame(args.stack_dir, sequence, frame - 1, cache)
            cur = read_frame(args.stack_dir, sequence, frame, cache)
            u, v = compute_flow(
                prev,
                cur,
                downsample=int(args.downsample),
                radius=int(args.flow_radius),
                num_warp=int(args.flow_num_warp),
            )
            div, curl, shear1, shear2 = derive_flow_gradients(u, v, int(args.downsample))
            for _, row in pts.iterrows():
                x_cur = float(row["x_px"]) / float(args.downsample)
                y_cur = float(row["y_px"]) / float(args.downsample)
                x_prev = float(row["x_px"] - row.get("dx_px", 0.0)) / float(args.downsample)
                y_prev = float(row["y_px"] - row.get("dy_px", 0.0)) / float(args.downsample)
                own_dx = float(row.get("dx_px", 0.0))
                own_dy = float(row.get("dy_px", 0.0))
                out: dict[str, Any] = {
                    "dataset": args.dataset,
                    "sequence": int(sequence),
                    "frame": int(frame),
                    "track_id": int(row["track_id"]),
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "tf_downsample": int(args.downsample),
                    "tf_flow_radius": int(args.flow_radius),
                    "tf_index_alignment_px": float(row.get("index_alignment_px", 0.0)),
                }
                for radius in radii:
                    rs = max(1, int(round(float(radius) / float(args.downsample))))
                    out.update(
                        flow_patch_features(
                            u=u,
                            v=v,
                            div=div,
                            curl=curl,
                            shear1=shear1,
                            shear2=shear2,
                            x=x_cur,
                            y=y_cur,
                            own_dx=own_dx,
                            own_dy=own_dy,
                            radius_small=rs,
                            prefix=f"tf_r{radius}_cur_",
                        )
                    )
                    out.update(
                        flow_patch_features(
                            u=u,
                            v=v,
                            div=div,
                            curl=curl,
                            shear1=shear1,
                            shear2=shear2,
                            x=x_prev,
                            y=y_prev,
                            own_dx=own_dx,
                            own_dy=own_dy,
                            radius_small=rs,
                            prefix=f"tf_r{radius}_prev_",
                        )
                    )
                rows.append(out)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith("tf_")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    return df


def time_shuffled(df: pd.DataFrame, feature_cols: list[str], seed: int) -> np.ndarray:
    return ifp.time_shuffled_image(df, feature_cols, seed)


def cols_matching(feature_cols: list[str], *, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    cols = []
    for col in feature_cols:
        if include and not any(token in col for token in include):
            continue
        if exclude and any(token in col for token in exclude):
            continue
        cols.append(col)
    return cols


def safe_block(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[cols].to_numpy(np.float32)


def feature_blocks(df: pd.DataFrame, feature_cols: list[str], seed: int) -> dict[str, np.ndarray]:
    df = df.reset_index(drop=True)
    traj_cols = [c for c in ifp.TRAJECTORY_FEATURES if c in df.columns]
    traj = df[traj_cols].to_numpy(np.float32)
    flow = df[feature_cols].to_numpy(np.float32)
    flow_shuf = ifp.shuffled_features(flow, seed)
    flow_time = time_shuffled(df, feature_cols, seed + 1)
    cur_cols = cols_matching(feature_cols, include=("_cur_",))
    prev_cols = cols_matching(feature_cols, include=("_prev_",))
    base_cols = cols_matching(
        feature_cols,
        include=("u_mean", "v_mean", "mag_mean", "center_u", "center_v", "u_median", "v_median", "mag_median"),
        exclude=("div_", "curl_", "shear", "front_back", "own_minus", "proj_", "cos_to_own"),
    )
    gradient_cols = cols_matching(feature_cols, include=("div_", "curl_", "shear1_", "shear2_"))
    alignment_cols = cols_matching(feature_cols, include=("own_minus", "proj_own_dir", "proj_tangent", "cos_to_own"))
    frontback_cols = cols_matching(feature_cols, include=("front_back",))
    radius_blocks = {}
    for radius in (64, 128, 256):
        r_cols = cols_matching(feature_cols, include=(f"tf_r{radius}_",))
        if r_cols:
            radius_blocks[f"flow_r{radius}_only"] = safe_block(df, r_cols)
            radius_blocks[f"trajectory_flow_r{radius}"] = np.concatenate([traj, safe_block(df, r_cols)], axis=1)
    blocks = {
        "trajectory_only": traj,
        "flow_only": flow,
        "trajectory_tissue_flow": np.concatenate([traj, flow], axis=1),
        "trajectory_tissue_flow_shuffled": np.concatenate([traj, flow_shuf], axis=1),
        "trajectory_tissue_flow_time_shuffled": np.concatenate([traj, flow_time], axis=1),
        "trajectory_flow_cur": np.concatenate([traj, safe_block(df, cur_cols)], axis=1),
        "trajectory_flow_prev": np.concatenate([traj, safe_block(df, prev_cols)], axis=1),
        "trajectory_flow_base": np.concatenate([traj, safe_block(df, base_cols)], axis=1),
        "trajectory_flow_gradients": np.concatenate([traj, safe_block(df, gradient_cols)], axis=1),
        "trajectory_flow_alignment": np.concatenate([traj, safe_block(df, alignment_cols)], axis=1),
        "trajectory_flow_frontback": np.concatenate([traj, safe_block(df, frontback_cols)], axis=1),
        "flow_base_only": safe_block(df, base_cols),
        "flow_gradients_only": safe_block(df, gradient_cols),
        "flow_alignment_only": safe_block(df, alignment_cols),
        "flow_frontback_only": safe_block(df, frontback_cols),
    }
    blocks.update(radius_blocks)
    return blocks


def run_probe(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in features.columns if c.startswith("tf_")]
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for horizon in parse_ints(args.horizons):
        full = ifp.build_horizon_table(image_features=features, table_root=args.table_root, dataset=args.dataset, horizon=horizon)
        split = ifp.make_split(
            full,
            parse_ints(args.train_sequences),
            parse_ints(args.val_sequences),
            parse_ints(args.test_sequences),
            int(args.seed),
        )
        train = ifp.sample_rows(split.train, int(args.max_train_rows), int(args.seed) + horizon * 11)
        val = ifp.sample_rows(split.val, int(args.max_val_rows), int(args.seed) + horizon * 13)
        test = ifp.sample_rows(split.test, int(args.max_test_rows), int(args.seed) + horizon * 17)
        y_train = train[["target_dx", "target_dy"]].to_numpy(np.float32)
        y_val = val[["target_dx", "target_dy"]].to_numpy(np.float32)
        y_test = test[["target_dx", "target_dy"]].to_numpy(np.float32)
        proposal_train = train[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        proposal_val = val[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        proposal_test = test[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        residual_train = y_train - proposal_train
        residual_val = y_val - proposal_val
        rows.append(
            ifp.evaluate(
                dataset=args.dataset,
                horizon=horizon,
                seed=args.seed,
                model_name="proposal",
                block_name="constant_velocity",
                y=y_test,
                proposal=proposal_test,
                pred_residual=None,
            )
        )
        train_blocks = feature_blocks(train, feature_cols, args.seed + horizon)
        val_blocks = feature_blocks(val, feature_cols, args.seed + horizon + 1)
        test_blocks = feature_blocks(test, feature_cols, args.seed + horizon + 2)
        for block_name in parse_strs(args.feature_blocks):
            feature_rows.append(
                {
                    "dataset": args.dataset,
                    "horizon": int(horizon),
                    "feature_block": block_name,
                    "feature_dim": int(train_blocks[block_name].shape[1]),
                    "train_rows": int(len(train)),
                    "val_rows": int(len(val)),
                    "test_rows": int(len(test)),
                }
            )
            for model_name in parse_strs(args.models):
                pred_res, info = ifp.fit_predict_model(
                    model_name,
                    train_blocks[block_name],
                    residual_train,
                    val_blocks[block_name],
                    residual_val,
                    test_blocks[block_name],
                    int(args.seed) + horizon,
                )
                rows.append(
                    ifp.evaluate(
                        dataset=args.dataset,
                        horizon=horizon,
                        seed=args.seed,
                        model_name=model_name,
                        block_name=block_name,
                        y=y_test,
                        proposal=proposal_test,
                        pred_residual=pred_res,
                        info=info,
                    )
                )
    summary = pd.DataFrame(rows)
    feature_df = pd.DataFrame(feature_rows)
    ablation_rows = []
    for (dataset, horizon, seed, model), group in summary[summary["model"].ne("proposal")].groupby(
        ["dataset", "horizon", "seed", "model"]
    ):
        by = group.set_index("feature_block")
        if "trajectory_only" not in by.index or "trajectory_tissue_flow" not in by.index:
            continue
        traj_rmse = float(by.loc["trajectory_only", "rmse_px"])
        full_rmse = float(by.loc["trajectory_tissue_flow", "rmse_px"])
        shuf_rmse = float(by.loc["trajectory_tissue_flow_shuffled", "rmse_px"]) if "trajectory_tissue_flow_shuffled" in by.index else math.nan
        time_rmse = (
            float(by.loc["trajectory_tissue_flow_time_shuffled", "rmse_px"])
            if "trajectory_tissue_flow_time_shuffled" in by.index
            else math.nan
        )
        ablation_rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "seed": int(seed),
                "model": model,
                "trajectory_rmse_px": traj_rmse,
                "trajectory_tissue_flow_rmse_px": full_rmse,
                "shuffled_rmse_px": shuf_rmse,
                "time_shuffled_rmse_px": time_rmse,
                "gain_vs_trajectory_pct": ifp.gain_pct(traj_rmse, full_rmse),
                "gain_vs_best_control_pct": ifp.gain_pct(min(shuf_rmse, time_rmse), full_rmse)
                if np.isfinite(shuf_rmse) and np.isfinite(time_rmse)
                else math.nan,
                "beats_controls": bool(full_rmse < min(shuf_rmse, time_rmse)) if np.isfinite(shuf_rmse) and np.isfinite(time_rmse) else False,
            }
        )
    return summary, feature_df, pd.DataFrame(ablation_rows)


def plot_ablation(ablation: pd.DataFrame, out_path: Path) -> None:
    if ablation.empty:
        return
    df = ablation.copy()
    df["label"] = df["model"] + " h" + df["horizon"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = np.where(df["beats_controls"], "#047857", "#64748b")
    ax.bar(df["label"], df["gain_vs_trajectory_pct"], color=colors)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.axhline(3, color="#111827", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Gain vs trajectory-only, %")
    ax.set_title("Causal tissue-flow feature gate")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, ablation: pd.DataFrame, feature_rows: pd.DataFrame, args: argparse.Namespace) -> None:
    best = summary.sort_values("rmse_px").head(12) if not summary.empty else pd.DataFrame()
    strong = ablation[
        ablation["gain_vs_trajectory_pct"].ge(3.0)
        & ablation["gain_vs_best_control_pct"].ge(1.0)
        & ablation["beats_controls"].astype(bool)
    ] if not ablation.empty else pd.DataFrame()
    weak = ablation[
        ablation["gain_vs_trajectory_pct"].gt(0.5)
        & ablation["beats_controls"].astype(bool)
    ] if not ablation.empty else pd.DataFrame()
    lines = [
        "# LaChance Tissue-Flow Feature Probe",
        "",
        "## Decision",
        "",
    ]
    if len(strong):
        lines.append("- Strong tissue-flow hook candidate found: >=3% over trajectory-only and >=1% over controls.")
    elif len(weak):
        lines.append("- Weak positive tissue-flow signal found, but it is below the strong gate.")
    else:
        lines.append("- No deployable tissue-flow hook found in this run.")
    lines += [
        "",
        "## Best Rows",
        "",
        best.to_markdown(index=False) if len(best) else "_No rows._",
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False) if len(ablation) else "_No ablation rows._",
        "",
        "## Feature Rows",
        "",
        feature_rows.to_markdown(index=False) if len(feature_rows) else "_No feature rows._",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False, default=str),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["extract", "probe", "both"], default="both")
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--flow-features", type=Path, default=None)
    parser.add_argument(
        "--point-index",
        type=Path,
        default=None,
        help="Optional tracking-aligned row index. When set, extract exactly these cells instead of sampling each frame independently.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sequences", default="1,2,3,4,5,6")
    parser.add_argument("--frames", default="1:49:1")
    parser.add_argument("--radii", default="64,128,256")
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--flow-radius", type=int, default=5)
    parser.add_argument("--flow-num-warp", type=int, default=5)
    parser.add_argument("--max-points-per-frame", type=int, default=1024)
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--models", default="ridge,hgbdt")
    parser.add_argument(
        "--feature-blocks",
        default="trajectory_only,trajectory_tissue_flow,trajectory_tissue_flow_shuffled,trajectory_tissue_flow_time_shuffled,flow_only",
    )
    parser.add_argument("--max-train-rows", type=int, default=80000)
    parser.add_argument("--max-val-rows", type=int, default=25000)
    parser.add_argument("--max-test-rows", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.sequences = "1,2,3"
        args.frames = "1:16:4"
        args.radii = "64,128"
        args.downsample = max(int(args.downsample), 16)
        args.max_points_per_frame = min(int(args.max_points_per_frame), 128)
        args.horizons = "1"
        args.train_sequences = "1"
        args.val_sequences = "2"
        args.test_sequences = "3"
        args.models = "ridge"
        args.max_train_rows = min(int(args.max_train_rows), 2000)
        args.max_val_rows = min(int(args.max_val_rows), 1000)
        args.max_test_rows = min(int(args.max_test_rows), 1000)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    feature_path = args.flow_features or (args.out_dir / "tissue_flow_features.csv")
    if args.mode in {"extract", "both"}:
        features = extract_tissue_flow_features(args)
        features.to_csv(feature_path, index=False)
        print(f"Saved tissue-flow features to {feature_path}", flush=True)
    else:
        features = pd.read_csv(feature_path)
    if args.mode in {"probe", "both"}:
        summary, feature_rows, ablation = run_probe(features, args)
        summary.to_csv(args.out_dir / "tissue_flow_probe_summary.csv", index=False)
        feature_rows.to_csv(args.out_dir / "tissue_flow_feature_blocks.csv", index=False)
        ablation.to_csv(args.out_dir / "tissue_flow_ablation.csv", index=False)
        plot_ablation(ablation, args.out_dir / "plots" / "tissue_flow_ablation.png")
        write_report(args.out_dir / "tissue_flow_status_report.md", summary, ablation, feature_rows, args)
        print(f"Saved tissue-flow probe to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
