#!/usr/bin/env python3
"""Multi-scale and temporal image-feature probe for LaChance raw frames.

This runner extracts a richer visual feature packet than the first simple
patch-statistics pilot:

- multi-scale patch morphology at several radii;
- previous-frame morphology at the previous cell position;
- temporal deltas between current and previous appearance.

Then it tests whether those features improve residual prediction over
trajectory-only baselines with shuffled/time-shuffled controls.
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

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_image_feature_extraction as ife  # noqa: E402
import run_lachance_image_feature_probe as ifp  # noqa: E402

DEFAULT_IMAGE_INDEX = (
    ROOT
    / "outputs"
    / "lachance_image_feature_extraction_pilot_mdck_bulk_6seq_2026-06-15"
    / "image_patch_features.csv"
)
DEFAULT_STACK_DIR = (
    ROOT
    / "new_data"
    / "lachance_epithelia"
    / "raw_timelapse"
    / "extracted_stacks"
    / "MDCK_Bulk_Timelapse_Data_Sample_Tissues"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_multiscale_image_feature_probe_2026-06-15"
CORE_DELTA_FEATURES = (
    "img_mean",
    "img_std",
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
    return ifp.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text).split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def prefixed(prefix: str, feats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{k.replace('img_', '')}": float(v) for k, v in feats.items() if k != "img_orientation"}


def crop_features(image: np.ndarray | None, x: float, y: float, radius: int) -> dict[str, float]:
    if image is None:
        return {}
    return ife.patch_features(ife.crop_patch(image, x, y, int(radius)))


def read_frame(stack_dir: Path, sequence: int, frame: int, cache: dict[tuple[int, int], np.ndarray]) -> np.ndarray | None:
    if frame < 0:
        return None
    key = (int(sequence), int(frame))
    if key in cache:
        return cache[key]
    path = stack_dir / f"{int(sequence):02d}.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        image = ife.read_tiff_page(path, int(frame))
    except Exception:
        return None
    cache[key] = image
    # Keep cache bounded: we only need current and previous frames around grouped iteration.
    if len(cache) > 8:
        for old_key in list(cache.keys())[:-8]:
            cache.pop(old_key, None)
    return image


def load_current_tracks(table_root: Path, dataset: str, index: pd.DataFrame) -> pd.DataFrame:
    frames_by_seq = {
        int(seq): set(int(f) for f in part["frame"].unique())
        for seq, part in index.groupby("sequence")
    }
    tables = []
    for seq, frames in frames_by_seq.items():
        path = table_root / dataset / f"{dataset}_{seq:02d}_tracks.csv"
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in ifp.TRACK_COLS if c in header.columns]
        table = pd.read_csv(path, usecols=usecols)
        table = table[table["frame"].isin(frames)].copy()
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def extract_multiscale_features(
    *,
    image_index_path: Path,
    table_root: Path,
    stack_dir: Path,
    dataset: str,
    radii: list[int],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    base = pd.read_csv(image_index_path, usecols=["dataset", "sequence", "frame", "track_id", "x_px", "y_px"])
    base = base[base["dataset"].eq(dataset)].copy()
    base["sequence"] = base["sequence"].astype(int)
    base["frame"] = base["frame"].astype(int)
    base["track_id"] = base["track_id"].astype(int)
    if max_rows > 0 and len(base) > max_rows:
        base = base.sample(n=int(max_rows), random_state=int(seed)).sort_values(["sequence", "frame", "track_id"])
    tracks = load_current_tracks(table_root, dataset, base)
    merged = base.merge(
        tracks,
        on=["dataset", "sequence", "frame", "track_id", "x_px", "y_px"],
        how="left",
        suffixes=("", "_track"),
    )
    merged["dx_px"] = merged.get("dx_px", 0.0).fillna(0.0)
    merged["dy_px"] = merged.get("dy_px", 0.0).fillna(0.0)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, int], np.ndarray] = {}
    for (sequence, frame), group in merged.groupby(["sequence", "frame"], sort=True):
        current = read_frame(stack_dir, int(sequence), int(frame), cache)
        previous = read_frame(stack_dir, int(sequence), int(frame) - 1, cache)
        for _, row in group.iterrows():
            out: dict[str, Any] = {
                "dataset": dataset,
                "sequence": int(sequence),
                "frame": int(frame),
                "track_id": int(row["track_id"]),
                "x_px": float(row["x_px"]),
                "y_px": float(row["y_px"]),
                "ms_has_prev": float(previous is not None),
            }
            prev_x = float(row["x_px"] - row.get("dx_px", 0.0))
            prev_y = float(row["y_px"] - row.get("dy_px", 0.0))
            for radius in radii:
                cur = crop_features(current, float(row["x_px"]), float(row["y_px"]), radius)
                prev = crop_features(previous, prev_x, prev_y, radius)
                out.update(prefixed(f"ms_r{radius}_cur_", cur))
                out.update(prefixed(f"ms_r{radius}_prev_", prev))
                for name in CORE_DELTA_FEATURES:
                    short = name.replace("img_", "")
                    out[f"ms_r{radius}_delta_{short}"] = float(cur.get(name, 0.0) - prev.get(name, 0.0))
            rows.append(out)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith("ms_")]
    df[feature_cols] = df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df


def image_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ms_")]


def feature_blocks(df: pd.DataFrame, image_cols: list[str], seed: int) -> dict[str, np.ndarray]:
    df = df.reset_index(drop=True)
    traj_cols = [c for c in ifp.TRAJECTORY_FEATURES if c in df.columns]
    traj = df[traj_cols].to_numpy(np.float32)
    image = df[image_cols].to_numpy(np.float32)
    image_shuf = ifp.shuffled_features(image, seed)
    image_time = ifp.time_shuffled_image(df, image_cols, seed)
    return {
        "trajectory_only": traj,
        "image_only": image,
        "trajectory_multiscale": np.concatenate([traj, image], axis=1),
        "trajectory_multiscale_shuffled": np.concatenate([traj, image_shuf], axis=1),
        "trajectory_multiscale_time_shuffled": np.concatenate([traj, image_time], axis=1),
    }


def run_probe(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    image_cols = image_feature_cols(features)
    for horizon in parse_ints(args.horizons):
        full = ifp.build_horizon_table(
            image_features=features,
            table_root=args.table_root,
            dataset=args.dataset,
            horizon=int(horizon),
        )
        split = ifp.make_split(
            full,
            parse_ints(args.train_sequences),
            parse_ints(args.val_sequences),
            parse_ints(args.test_sequences),
            int(args.seed),
        )
        train = ifp.sample_rows(split.train, args.max_train_rows, args.seed + horizon * 11)
        val = ifp.sample_rows(split.val, args.max_val_rows, args.seed + horizon * 13)
        test = ifp.sample_rows(split.test, args.max_test_rows, args.seed + horizon * 17)
        y_train = train[["target_dx", "target_dy"]].to_numpy(np.float32)
        y_val = val[["target_dx", "target_dy"]].to_numpy(np.float32)
        y_test = test[["target_dx", "target_dy"]].to_numpy(np.float32)
        proposal_train = train[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        proposal_val = val[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        proposal_test = test[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
        residual_train = y_train - proposal_train
        residual_val = y_val - proposal_val
        proposal_metrics = ifp.evaluate(
            dataset=args.dataset,
            horizon=horizon,
            seed=args.seed,
            model_name="proposal",
            block_name="constant_velocity",
            y=y_test,
            proposal=proposal_test,
            pred_residual=None,
        )
        rows.append(proposal_metrics)
        train_blocks = feature_blocks(train, image_cols, args.seed + horizon)
        val_blocks = feature_blocks(val, image_cols, args.seed + horizon + 1)
        test_blocks = feature_blocks(test, image_cols, args.seed + horizon + 2)
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
                    args.seed + horizon,
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
    feature_probe = pd.DataFrame(feature_rows)
    ablation_rows: list[dict[str, Any]] = []
    for (dataset, horizon, seed, model), group in summary[summary["model"].ne("proposal")].groupby(
        ["dataset", "horizon", "seed", "model"]
    ):
        by_block = group.set_index("feature_block")
        if "trajectory_only" not in by_block.index or "trajectory_multiscale" not in by_block.index:
            continue
        traj_rmse = float(by_block.loc["trajectory_only", "rmse_px"])
        full_rmse = float(by_block.loc["trajectory_multiscale", "rmse_px"])
        shuf_rmse = (
            float(by_block.loc["trajectory_multiscale_shuffled", "rmse_px"])
            if "trajectory_multiscale_shuffled" in by_block.index
            else math.nan
        )
        time_rmse = (
            float(by_block.loc["trajectory_multiscale_time_shuffled", "rmse_px"])
            if "trajectory_multiscale_time_shuffled" in by_block.index
            else math.nan
        )
        ablation_rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "seed": int(seed),
                "model": model,
                "trajectory_rmse_px": traj_rmse,
                "trajectory_multiscale_rmse_px": full_rmse,
                "shuffled_rmse_px": shuf_rmse,
                "time_shuffled_rmse_px": time_rmse,
                "delta_rmse_vs_trajectory_pct": ifp.gain_pct(traj_rmse, full_rmse),
                "delta_rmse_vs_shuffled_pct": ifp.gain_pct(shuf_rmse, full_rmse) if np.isfinite(shuf_rmse) else math.nan,
                "delta_rmse_vs_time_shuffled_pct": ifp.gain_pct(time_rmse, full_rmse) if np.isfinite(time_rmse) else math.nan,
                "beats_controls": bool(
                    np.isfinite(shuf_rmse)
                    and np.isfinite(time_rmse)
                    and full_rmse < shuf_rmse
                    and full_rmse < time_rmse
                ),
            }
        )
    return summary, pd.DataFrame(ablation_rows), feature_probe


def plot_ablation(ablation: pd.DataFrame, out_path: Path) -> None:
    if ablation.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    labels = ablation["horizon"].astype(str) + "\n" + ablation["model"]
    colors = ["#2f6f9f" if ok else "#9c6a5b" for ok in ablation["beats_controls"].astype(bool)]
    ax.bar(labels, ablation["delta_rmse_vs_trajectory_pct"], color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.axhline(1.0, color="#999999", linewidth=1, linestyle="--")
    ax.set_ylabel("RMSE gain vs trajectory-only, %")
    ax.set_title("Multi-scale visual feature gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, ablation: pd.DataFrame, payload: dict[str, Any]) -> None:
    lines = [
        "# LaChance Multi-Scale Image Feature Probe",
        "",
        "## Purpose",
        "",
        "Test whether multi-scale morphology and temporal appearance changes improve residual motion prediction beyond trajectory-only features.",
        "",
        "## Payload",
        "",
        "```json",
        json.dumps(finite_json(payload), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Best Rows",
        "",
        summary.sort_values(["horizon", "rmse_px"]).groupby("horizon").head(8).to_markdown(index=False),
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False) if not ablation.empty else "No ablation rows.",
        "",
        "## Gate",
        "",
    ]
    strong = ablation[ablation["beats_controls"].astype(bool) & ablation["delta_rmse_vs_trajectory_pct"].gt(2.0)]
    moderate = ablation[ablation["beats_controls"].astype(bool) & ablation["delta_rmse_vs_trajectory_pct"].gt(1.0)]
    weak = ablation[ablation["beats_controls"].astype(bool) & ablation["delta_rmse_vs_trajectory_pct"].gt(0.25)]
    if not strong.empty:
        lines.append("- Strong multi-scale image gate passed: >2% gain and controls are worse.")
    elif not moderate.empty:
        lines.append("- Moderate multi-scale image signal: >1% gain and controls are worse.")
    elif not weak.empty:
        lines.append("- Weak positive multi-scale image signal: controls are worse, but gain is below 1%.")
    else:
        lines.append("- Multi-scale image gate not passed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-index", type=Path, default=DEFAULT_IMAGE_INDEX)
    parser.add_argument("--table-root", type=Path, default=ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--radii", default="8,16,24,40")
    parser.add_argument("--horizons", default="4,6")
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--models", default="ridge,hgbdt")
    parser.add_argument(
        "--feature-blocks",
        default="trajectory_only,trajectory_multiscale,trajectory_multiscale_shuffled,trajectory_multiscale_time_shuffled,image_only",
    )
    parser.add_argument("--max-feature-rows", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=20000)
    parser.add_argument("--max-test-rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "multiscale_image_features.csv"
    if feature_path.exists():
        features = pd.read_csv(feature_path)
    else:
        features = extract_multiscale_features(
            image_index_path=args.image_index,
            table_root=args.table_root,
            stack_dir=args.stack_dir,
            dataset=args.dataset,
            radii=parse_ints(args.radii),
            max_rows=int(args.max_feature_rows),
            seed=int(args.seed),
        )
        features.to_csv(feature_path, index=False)
    summary, ablation, feature_probe = run_probe(features, args)
    summary_path = args.out_dir / "multiscale_image_probe_summary.csv"
    ablation_path = args.out_dir / "multiscale_image_ablation.csv"
    feature_probe_path = args.out_dir / "multiscale_image_feature_probe.csv"
    summary.to_csv(summary_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    feature_probe.to_csv(feature_probe_path, index=False)
    plot_ablation(ablation, args.out_dir / "plots" / "multiscale_image_gain.png")
    payload = {
        "image_index": args.image_index,
        "table_root": args.table_root,
        "stack_dir": args.stack_dir,
        "dataset": args.dataset,
        "radii": parse_ints(args.radii),
        "horizons": parse_ints(args.horizons),
        "models": parse_strs(args.models),
        "feature_blocks": parse_strs(args.feature_blocks),
        "max_feature_rows": int(args.max_feature_rows),
        "max_train_rows": int(args.max_train_rows),
        "max_val_rows": int(args.max_val_rows),
        "max_test_rows": int(args.max_test_rows),
        "feature_csv": feature_path,
        "summary_csv": summary_path,
        "ablation_csv": ablation_path,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(payload), indent=2), encoding="utf-8")
    write_report(args.out_dir / "multiscale_image_status_report.md", summary, ablation, payload)
    print(args.out_dir / "multiscale_image_status_report.md")


if __name__ == "__main__":
    main()
