#!/usr/bin/env python3
"""Probe whether raw-image patch features add causal motion signal.

This is intentionally an isolated diagnostic.  It joins patch features from
``run_lachance_image_feature_extraction.py`` with the LaChance track tables and
tests residual correction over a constant-velocity proposal:

    proposal(t, h) = h * delta_position(t)
    residual = target_displacement(t -> t+h) - proposal

Only current-frame track and image features are used.  Future coordinates are
used only to form the supervised target.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_image_feature_extraction_pilot_mdck_bulk_6seq_2026-06-15"
    / "image_patch_features.csv"
)
DEFAULT_TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"
DEFAULT_OUT = ROOT / "outputs" / "lachance_image_feature_probe_2026-06-15"
EPS = 1e-8

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    MLPRegressor = None  # type: ignore[assignment]


TRACK_COLS = (
    "dataset",
    "sequence",
    "frame",
    "track_id",
    "x_px",
    "y_px",
    "dx_px",
    "dy_px",
    "vx_px_s",
    "vy_px_s",
    "speed_px_s",
    "ax_px_s2",
    "ay_px_s2",
    "QUALITY",
)

TRAJECTORY_FEATURES = (
    "frame_norm",
    "x_norm",
    "y_norm",
    "dx_px",
    "dy_px",
    "vx_px_s",
    "vy_px_s",
    "speed_px_s",
    "ax_px_s2",
    "ay_px_s2",
    "QUALITY",
    "proposal_dx",
    "proposal_dy",
    "proposal_norm",
)

IMAGE_FEATURES = (
    "img_mean",
    "img_std",
    "img_p10",
    "img_p50",
    "img_p90",
    "img_grad_mean",
    "img_grad_p90",
    "img_fg_frac",
    "img_centroid_dx",
    "img_centroid_dy",
    "img_elongation",
    "img_orient_cos",
    "img_orient_sin",
)


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text).split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def vector_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(pred - y), axis=1))))


def vector_r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum(np.square(y - pred)))
    centered = y - y.mean(axis=0, keepdims=True)
    sst = float(np.sum(np.square(centered)))
    return float(1.0 - sse / sst) if sst > EPS else float("nan")


def mean_cosine(y: np.ndarray, pred: np.ndarray) -> float:
    den = np.maximum(np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1), EPS)
    return float(np.mean(np.sum(y * pred, axis=1) / den))


def magnitude_ratio(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred, axis=1)) / max(float(np.mean(np.linalg.norm(y, axis=1))), EPS))


def gain_pct(base: float, value: float) -> float:
    return float((base - value) / max(abs(base), EPS) * 100.0)


def clean_matrix(x: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(out, -1e6, 1e6).astype(np.float32, copy=False)


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    train_x = clean_matrix(train_x)
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-6)

    def z(x: np.ndarray) -> np.ndarray:
        out = (clean_matrix(x) - mean) / std
        out = np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0)
        return np.clip(out, -8.0, 8.0).astype(np.float32, copy=False)

    return z(train_x), [z(x) for x in others], {
        "feature_dim": int(train_x.shape[1]),
        "zero_std_features": int(np.sum(std.reshape(-1) <= 1.000001e-6)),
    }


def sample_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=int(max_rows), random_state=int(seed)).sort_index()


def read_track_table(table_root: Path, dataset: str, sequence: int, needed_frames: set[int]) -> pd.DataFrame:
    path = table_root / dataset / f"{dataset}_{sequence:02d}_tracks.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in TRACK_COLS if c in header.columns]
    table = pd.read_csv(path, usecols=usecols)
    table = table[table["frame"].isin(needed_frames)].copy()
    table["sequence"] = table["sequence"].astype(int)
    table["frame"] = table["frame"].astype(int)
    table["track_id"] = table["track_id"].astype(int)
    return table


def build_horizon_table(
    *,
    image_features: pd.DataFrame,
    table_root: Path,
    dataset: str,
    horizon: int,
) -> pd.DataFrame:
    img = image_features[image_features["dataset"].eq(dataset)].copy()
    if img.empty:
        raise ValueError(f"No image features for dataset={dataset}")
    img["sequence"] = img["sequence"].astype(int)
    img["frame"] = img["frame"].astype(int)
    img["track_id"] = img["track_id"].astype(int)
    seqs = sorted(int(s) for s in img["sequence"].unique())
    cur_frames = set(int(f) for f in img["frame"].unique())
    needed_frames = set(cur_frames) | {int(f) + int(horizon) for f in cur_frames}
    tables = [read_track_table(table_root, dataset, seq, needed_frames) for seq in seqs]
    tracks = pd.concat(tables, ignore_index=True)

    current_cols = [c for c in TRACK_COLS if c in tracks.columns]
    current = tracks[current_cols].copy()
    future = tracks[["sequence", "frame", "track_id", "x_px", "y_px"]].copy()
    future["frame"] = future["frame"].astype(int) - int(horizon)
    future = future.rename(columns={"x_px": "future_x_px", "y_px": "future_y_px"})
    merged = img.merge(
        current,
        on=["dataset", "sequence", "frame", "track_id", "x_px", "y_px"],
        how="inner",
        suffixes=("", "_track"),
    )
    merged = merged.merge(future, on=["sequence", "frame", "track_id"], how="inner")
    merged["target_dx"] = merged["future_x_px"] - merged["x_px"]
    merged["target_dy"] = merged["future_y_px"] - merged["y_px"]
    merged["proposal_dx"] = float(horizon) * merged["dx_px"].fillna(0.0)
    merged["proposal_dy"] = float(horizon) * merged["dy_px"].fillna(0.0)
    merged["proposal_norm"] = np.sqrt(np.square(merged["proposal_dx"]) + np.square(merged["proposal_dy"]))

    x_scale = max(float(merged["x_px"].quantile(0.99) - merged["x_px"].quantile(0.01)), 1.0)
    y_scale = max(float(merged["y_px"].quantile(0.99) - merged["y_px"].quantile(0.01)), 1.0)
    f_scale = max(float(merged["frame"].max() - merged["frame"].min()), 1.0)
    merged["x_norm"] = (merged["x_px"] - float(merged["x_px"].median())) / x_scale
    merged["y_norm"] = (merged["y_px"] - float(merged["y_px"].median())) / y_scale
    merged["frame_norm"] = (merged["frame"] - float(merged["frame"].min())) / f_scale
    if "QUALITY" not in merged.columns:
        merged["QUALITY"] = 0.0
    return merged


def shuffled_features(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(x))
    return x[idx]


def time_shuffled_image(df: pd.DataFrame, image_cols: list[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros((len(df), len(image_cols)), dtype=np.float32)
    for _, seq_idx in df.groupby("sequence", sort=False).groups.items():
        seq_rows = np.asarray(seq_idx, dtype=np.int64)
        frames = np.asarray(sorted(df.loc[seq_rows, "frame"].unique()), dtype=np.int64)
        if len(frames) <= 1:
            out[seq_rows] = df.loc[seq_rows, image_cols].to_numpy(np.float32)
            continue
        permuted = frames.copy()
        for _ in range(10):
            rng.shuffle(permuted)
            if not np.any(permuted == frames):
                break
        frame_map = {int(src): int(dst) for src, dst in zip(frames, permuted)}
        src_by_frame = {
            int(frame): df.loc[seq_rows[df.loc[seq_rows, "frame"].to_numpy() == frame], image_cols].to_numpy(np.float32)
            for frame in frames
        }
        for frame in frames:
            dest_rows = seq_rows[df.loc[seq_rows, "frame"].to_numpy() == frame]
            pool = src_by_frame[frame_map[int(frame)]]
            if len(pool) == 0:
                out[dest_rows] = df.loc[dest_rows, image_cols].to_numpy(np.float32)
            else:
                choice = rng.integers(0, len(pool), size=len(dest_rows))
                out[dest_rows] = pool[choice]
    return out


def feature_blocks(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    df = df.reset_index(drop=True)
    traj_cols = [c for c in TRAJECTORY_FEATURES if c in df.columns]
    image_cols = [c for c in IMAGE_FEATURES if c in df.columns]
    traj = df[traj_cols].to_numpy(np.float32)
    image = df[image_cols].to_numpy(np.float32)
    image_shuf = shuffled_features(image, seed)
    image_time = time_shuffled_image(df, image_cols, seed)
    return {
        "trajectory_only": traj,
        "image_only": image,
        "trajectory_image": np.concatenate([traj, image], axis=1),
        "trajectory_image_shuffled": np.concatenate([traj, image_shuf], axis=1),
        "trajectory_image_time_shuffled": np.concatenate([traj, image_time], axis=1),
    }


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_split(df: pd.DataFrame, train_seq: list[int], val_seq: list[int], test_seq: list[int], seed: int) -> SplitData:
    train = df[df["sequence"].isin(train_seq)].copy()
    val = df[df["sequence"].isin(val_seq)].copy()
    test = df[df["sequence"].isin(test_seq)].copy()
    if train.empty or val.empty or test.empty:
        seqs = sorted(int(s) for s in df["sequence"].unique())
        if len(seqs) < 3:
            rng = np.random.default_rng(int(seed))
            key = rng.random(len(df))
            train = df[key < 0.70].copy()
            val = df[(key >= 0.70) & (key < 0.85)].copy()
            test = df[key >= 0.85].copy()
        else:
            train = df[df["sequence"].isin(seqs[:-2])].copy()
            val = df[df["sequence"].eq(seqs[-2])].copy()
            test = df[df["sequence"].eq(seqs[-1])].copy()
    return SplitData(train=train, val=val, test=test)


def fit_predict_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_z, [val_z, test_z], norm_info = standardize(train_x, val_x, test_x)
    if model_name == "ridge":
        if Ridge is None:
            raise RuntimeError("sklearn Ridge is unavailable")
        best: tuple[float, Any, float] | None = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(train_z, train_y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pred = model.predict(val_z)
            if not np.isfinite(pred).all():
                continue
            rmse = vector_rmse(val_y, pred)
            if best is None or rmse < best[0]:
                best = (rmse, model, float(alpha))
        if best is None:
            model = Ridge(alpha=1000.0, solver="svd")
            model.fit(train_z, train_y)
            best = (float("nan"), model, 1000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            test_pred = best[1].predict(test_z)
        test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(test_pred, -1e6, 1e6).astype(np.float32), {
            **norm_info,
            "model": model_name,
            "alpha": best[2],
            "val_residual_rmse": best[0],
        }
    if model_name == "hgbdt":
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("sklearn HistGradientBoostingRegressor is unavailable")
        preds = []
        for dim in range(2):
            model = HistGradientBoostingRegressor(
                max_iter=130,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.04,
                random_state=int(seed) + dim,
            )
            model.fit(train_z, train_y[:, dim])
            preds.append(model.predict(test_z))
        return np.column_stack(preds).astype(np.float32), {**norm_info, "model": model_name}
    if model_name == "mlp":
        if MLPRegressor is None:
            raise RuntimeError("sklearn MLPRegressor is unavailable")
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=120,
            early_stopping=True,
            random_state=int(seed),
        )
        model.fit(train_z, train_y)
        return model.predict(test_z).astype(np.float32), {**norm_info, "model": model_name, "n_iter": int(model.n_iter_)}
    raise ValueError(f"unknown model {model_name}")


def evaluate(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    model_name: str,
    block_name: str,
    y: np.ndarray,
    proposal: np.ndarray,
    pred_residual: np.ndarray | None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if pred_residual is None:
        pred = proposal
    else:
        pred = proposal + pred_residual
    rmse = vector_rmse(y, pred)
    proposal_rmse = vector_rmse(y, proposal)
    return {
        "dataset": dataset,
        "horizon": int(horizon),
        "seed": int(seed),
        "model": model_name,
        "feature_block": block_name,
        "n_test": int(len(y)),
        "rmse_px": rmse,
        "r2": vector_r2(y, pred),
        "cosine": mean_cosine(y, pred),
        "magnitude_ratio": magnitude_ratio(y, pred),
        "proposal_rmse_px": proposal_rmse,
        "gain_vs_proposal_pct": gain_pct(proposal_rmse, rmse),
        **(info or {}),
    }


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    df = summary[summary["model"].ne("proposal")].copy()
    df["label"] = df["model"] + "\n" + df["feature_block"]
    order = df.sort_values("rmse_px")["label"].tolist()
    ax.bar(df["label"], df["gain_vs_proposal_pct"], color="#3b6ea8")
    ax.axhline(0.0, color="#444", linewidth=1)
    ax.set_ylabel("RMSE gain vs proposal, %")
    ax.set_title("Image feature probe")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    if order:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"].tolist(), ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, ablation: pd.DataFrame, payload: dict[str, Any]) -> None:
    best = summary.sort_values("rmse_px").head(8)
    lines = [
        "# LaChance Image Feature Probe",
        "",
        "## What Was Tested",
        "",
        "- Raw MDCK Bulk sample frames were joined with current-frame trajectory rows.",
        "- Patch/intensity/morphology features were used only at frame t.",
        "- Target is future displacement t -> t+h; proposal is constant velocity h * dx(t).",
        "- Controls include shuffled image features and time-shuffled image features.",
        "",
        "## Run Payload",
        "",
        "```json",
        json.dumps(finite_json(payload), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Best Rows",
        "",
        best.to_markdown(index=False),
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False) if not ablation.empty else "No ablation rows.",
        "",
        "## Interpretation Gate",
        "",
    ]
    strong = ablation[
        ablation["delta_rmse_vs_trajectory_pct"].gt(1.0)
        & ablation["beats_shuffled_control"].astype(bool)
    ]
    weak = ablation[
        ablation["delta_rmse_vs_trajectory_pct"].gt(0.25)
        & ablation["beats_shuffled_control"].astype(bool)
    ]
    if not strong.empty:
        lines.append("- Strong image-feature gate candidate found: image features improve trajectory-only by >1% and beat shuffled controls.")
    elif not weak.empty:
        lines.append("- Weak positive image signal found: image features beat trajectory-only and shuffled controls, but the gain is below the strong >1% gate.")
    else:
        lines.append("- No deployable image-feature gate yet: improvements do not beat trajectory-only and shuffled controls.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-features", type=Path, default=DEFAULT_IMAGE_FEATURES)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--horizons", default="4,6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--models", default="ridge,hgbdt")
    parser.add_argument(
        "--feature-blocks",
        default="trajectory_only,trajectory_image,trajectory_image_shuffled,trajectory_image_time_shuffled,image_only",
    )
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=25000)
    parser.add_argument("--max-test-rows", type=int, default=25000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_features = pd.read_csv(args.image_features)
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    horizons = parse_ints(args.horizons)
    models = parse_strs(args.models)
    blocks = parse_strs(args.feature_blocks)
    train_seq = parse_ints(args.train_sequences)
    val_seq = parse_ints(args.val_sequences)
    test_seq = parse_ints(args.test_sequences)

    for horizon in horizons:
        full = build_horizon_table(
            image_features=image_features,
            table_root=args.table_root,
            dataset=args.dataset,
            horizon=int(horizon),
        )
        split = make_split(full, train_seq, val_seq, test_seq, args.seed)
        train = sample_rows(split.train, args.max_train_rows, args.seed + horizon * 11)
        val = sample_rows(split.val, args.max_val_rows, args.seed + horizon * 13)
        test = sample_rows(split.test, args.max_test_rows, args.seed + horizon * 17)
        target_train = train[["target_dx", "target_dy"]].to_numpy(np.float32)
        target_val = val[["target_dx", "target_dy"]].to_numpy(np.float32)
        target_test = test[["target_dx", "target_dy"]].to_numpy(np.float32)
        proposal_train = train[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        proposal_val = val[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        proposal_test = test[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        residual_train = target_train - proposal_train
        residual_val = target_val - proposal_val

        rows.append(
            evaluate(
                dataset=args.dataset,
                horizon=horizon,
                seed=args.seed,
                model_name="proposal",
                block_name="constant_velocity",
                y=target_test,
                proposal=proposal_test,
                pred_residual=None,
            )
        )
        train_blocks = feature_blocks(train, args.seed + horizon)
        val_blocks = feature_blocks(val, args.seed + horizon + 1)
        test_blocks = feature_blocks(test, args.seed + horizon + 2)
        for block_name in blocks:
            if block_name not in train_blocks:
                raise ValueError(f"unknown feature block {block_name}; available={sorted(train_blocks)}")
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
            for model_name in models:
                pred_res, info = fit_predict_model(
                    model_name,
                    train_blocks[block_name],
                    residual_train,
                    val_blocks[block_name],
                    residual_val,
                    test_blocks[block_name],
                    args.seed + horizon,
                )
                rows.append(
                    evaluate(
                        dataset=args.dataset,
                        horizon=horizon,
                        seed=args.seed,
                        model_name=model_name,
                        block_name=block_name,
                        y=target_test,
                        proposal=proposal_test,
                        pred_residual=pred_res,
                        info=info,
                    )
                )

    summary = pd.DataFrame(rows)
    feature_probe = pd.DataFrame(feature_rows)
    ablation_rows: list[dict[str, Any]] = []
    for (dataset, horizon, seed, model), group in summary[summary["model"].ne("proposal")].groupby(
        ["dataset", "horizon", "seed", "model"]
    ):
        by_block = group.set_index("feature_block")
        if "trajectory_only" not in by_block.index or "trajectory_image" not in by_block.index:
            continue
        traj_rmse = float(by_block.loc["trajectory_only", "rmse_px"])
        full_rmse = float(by_block.loc["trajectory_image", "rmse_px"])
        shuf_rmse = float(by_block.loc["trajectory_image_shuffled", "rmse_px"]) if "trajectory_image_shuffled" in by_block.index else math.nan
        time_rmse = float(by_block.loc["trajectory_image_time_shuffled", "rmse_px"]) if "trajectory_image_time_shuffled" in by_block.index else math.nan
        ablation_rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "seed": int(seed),
                "model": model,
                "trajectory_rmse_px": traj_rmse,
                "trajectory_image_rmse_px": full_rmse,
                "shuffled_rmse_px": shuf_rmse,
                "time_shuffled_rmse_px": time_rmse,
                "delta_rmse_vs_trajectory_pct": gain_pct(traj_rmse, full_rmse),
                "delta_rmse_vs_shuffled_pct": gain_pct(shuf_rmse, full_rmse) if np.isfinite(shuf_rmse) else math.nan,
                "delta_rmse_vs_time_shuffled_pct": gain_pct(time_rmse, full_rmse) if np.isfinite(time_rmse) else math.nan,
                "beats_shuffled_control": bool(
                    np.isfinite(shuf_rmse)
                    and np.isfinite(time_rmse)
                    and full_rmse < shuf_rmse
                    and full_rmse < time_rmse
                ),
            }
        )
    ablation = pd.DataFrame(ablation_rows)

    summary_path = args.out_dir / "image_feature_probe_summary.csv"
    ablation_path = args.out_dir / "image_feature_ablation.csv"
    feature_path = args.out_dir / "image_feature_probe_features.csv"
    summary.to_csv(summary_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    feature_probe.to_csv(feature_path, index=False)
    plot_summary(summary, args.out_dir / "plots" / "image_feature_probe_gain.png")
    payload = {
        "image_features": args.image_features,
        "table_root": args.table_root,
        "dataset": args.dataset,
        "horizons": horizons,
        "seed": int(args.seed),
        "train_sequences": train_seq,
        "val_sequences": val_seq,
        "test_sequences": test_seq,
        "models": models,
        "feature_blocks": blocks,
        "max_train_rows": int(args.max_train_rows),
        "max_val_rows": int(args.max_val_rows),
        "max_test_rows": int(args.max_test_rows),
        "summary_csv": summary_path,
        "ablation_csv": ablation_path,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(payload), indent=2), encoding="utf-8")
    write_report(args.out_dir / "image_feature_status_report.md", summary, ablation, payload)
    print(args.out_dir / "image_feature_status_report.md")


if __name__ == "__main__":
    main()
