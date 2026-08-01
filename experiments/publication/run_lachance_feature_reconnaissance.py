#!/usr/bin/env python3
"""Feature reconnaissance over causal LaChance observables.

This runner combines already extracted feature packets on a shared
cell/frame/track grid and tests which causal feature families add deployable
signal for h1-first residual prediction.

Feature packets:

- trajectory/self-motion from TrackMate tables;
- multi-scale raw-image morphology (`ms_*`);
- tissue-flow / PIV-like local motion (`tf_*`);
- combined visual + tissue-flow packet;
- shuffled/time-shuffled controls.

The target is still future displacement `X_t -> X_{t+h}`.  No future image,
future coordinate, or target-derived candidate label is used as an input
feature.
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

import run_lachance_image_feature_probe as ifp  # noqa: E402

DEFAULT_MS = (
    ROOT
    / "outputs"
    / "lachance_multiscale_on_tissue_flow_grid_mdck_bulk_h1h4h6_seed42_2026-06-15"
    / "multiscale_image_features.csv"
)
DEFAULT_TF = (
    ROOT
    / "outputs"
    / "lachance_tissue_flow_mdck_bulk_raw6_h1h2h4h6_seed42_2026-06-15"
    / "tissue_flow_features.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_feature_reconnaissance_2026-06-15"
KEYS = ["dataset", "sequence", "frame", "track_id"]


def finite_json(value: Any) -> Any:
    return ifp.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text or "").split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def prefixed_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def cols_containing(cols: list[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    out = []
    for col in cols:
        if include and not any(token in col for token in include):
            continue
        if exclude and any(token in col for token in exclude):
            continue
        out.append(col)
    return out


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def load_combined(ms_path: Path, tf_path: Path, dataset: str) -> pd.DataFrame:
    ms = pd.read_csv(ms_path)
    tf = pd.read_csv(tf_path)
    ms = ms[ms["dataset"].eq(dataset)].copy()
    tf = tf[tf["dataset"].eq(dataset)].copy()
    for df in (ms, tf):
        df["sequence"] = df["sequence"].astype(int)
        df["frame"] = df["frame"].astype(int)
        df["track_id"] = df["track_id"].astype(int)
    tf_cols = KEYS + ["x_px", "y_px"] + prefixed_cols(tf, "tf_")
    # Prefer coordinates from the morphology table.  Track ids are unique within
    # sequence/frame in these exports, and coordinates are only used downstream
    # for a strict merge with the track table.
    combined = ms.merge(tf[tf_cols], on=KEYS, how="inner", suffixes=("", "_tf"))
    if "x_px_tf" in combined.columns:
        dx = np.abs(combined["x_px"] - combined["x_px_tf"]).max()
        dy = np.abs(combined["y_px"] - combined["y_px_tf"]).max()
        if max(float(dx), float(dy)) > 1e-3:
            print(f"[warn] coordinate mismatch max dx={dx:.6f} dy={dy:.6f}", flush=True)
        combined = combined.drop(columns=[c for c in ("x_px_tf", "y_px_tf") if c in combined.columns])
    return combined


def time_shuffle(df: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    return ifp.time_shuffled_image(df, cols, seed)


def make_blocks(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    df = df.reset_index(drop=True)
    traj_cols = [c for c in ifp.TRAJECTORY_FEATURES if c in df.columns]
    ms_cols = prefixed_cols(df, "ms_")
    tf_cols = prefixed_cols(df, "tf_")
    traj = safe_matrix(df, traj_cols)
    ms = safe_matrix(df, ms_cols)
    tf = safe_matrix(df, tf_cols)
    ms_shuf = ifp.shuffled_features(ms, seed + 11)
    tf_shuf = ifp.shuffled_features(tf, seed + 17)
    ms_time = time_shuffle(df, ms_cols, seed + 23)
    tf_time = time_shuffle(df, tf_cols, seed + 29)

    tf_cur_cols = cols_containing(tf_cols, ("_cur_",))
    tf_base_cols = cols_containing(
        tf_cols,
        ("u_mean", "v_mean", "mag_mean", "center_u", "center_v", "u_median", "v_median", "mag_median"),
        ("div_", "curl_", "shear", "front_back", "own_minus", "proj_", "cos_to_own"),
    )
    tf_alignment_cols = cols_containing(tf_cols, ("own_minus", "proj_own_dir", "proj_tangent", "cos_to_own"))
    ms_cur_cols = cols_containing(ms_cols, ("_cur_",))
    ms_delta_cols = cols_containing(ms_cols, ("_delta_",))
    ms_shape_cols = cols_containing(ms_cols, ("elongation", "orient", "centroid"))
    ms_texture_cols = cols_containing(ms_cols, ("mean", "std", "p10", "p50", "p90", "grad", "fg_frac"))

    return {
        "trajectory_only": traj,
        "morphology_only": ms,
        "tissue_flow_only": tf,
        "trajectory_morphology": np.concatenate([traj, ms], axis=1),
        "trajectory_tissue_flow": np.concatenate([traj, tf], axis=1),
        "trajectory_morphology_tissue_flow": np.concatenate([traj, ms, tf], axis=1),
        "trajectory_morphology_shuffled": np.concatenate([traj, ms_shuf], axis=1),
        "trajectory_tissue_flow_shuffled": np.concatenate([traj, tf_shuf], axis=1),
        "trajectory_morphology_tissue_flow_shuffled_both": np.concatenate([traj, ms_shuf, tf_shuf], axis=1),
        "trajectory_morphology_tissue_flow_shuffled_ms": np.concatenate([traj, ms_shuf, tf], axis=1),
        "trajectory_morphology_tissue_flow_shuffled_tf": np.concatenate([traj, ms, tf_shuf], axis=1),
        "trajectory_morphology_tissue_flow_time_shuffled": np.concatenate([traj, ms_time, tf_time], axis=1),
        "trajectory_tf_cur": np.concatenate([traj, safe_matrix(df, tf_cur_cols)], axis=1),
        "trajectory_tf_base": np.concatenate([traj, safe_matrix(df, tf_base_cols)], axis=1),
        "trajectory_tf_alignment": np.concatenate([traj, safe_matrix(df, tf_alignment_cols)], axis=1),
        "trajectory_ms_cur": np.concatenate([traj, safe_matrix(df, ms_cur_cols)], axis=1),
        "trajectory_ms_delta": np.concatenate([traj, safe_matrix(df, ms_delta_cols)], axis=1),
        "trajectory_ms_shape": np.concatenate([traj, safe_matrix(df, ms_shape_cols)], axis=1),
        "trajectory_ms_texture": np.concatenate([traj, safe_matrix(df, ms_texture_cols)], axis=1),
    }


def evaluate_probe(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    requested = parse_strs(args.feature_blocks)
    for horizon in parse_ints(args.horizons):
        full = ifp.build_horizon_table(image_features=features, table_root=args.table_root, dataset=args.dataset, horizon=horizon)
        split = ifp.make_split(full, parse_ints(args.train_sequences), parse_ints(args.val_sequences), parse_ints(args.test_sequences), int(args.seed))
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
        train_blocks = make_blocks(train, int(args.seed) + horizon)
        val_blocks = make_blocks(val, int(args.seed) + horizon + 1)
        test_blocks = make_blocks(test, int(args.seed) + horizon + 2)
        for block in requested:
            if block not in train_blocks:
                raise ValueError(f"Unknown feature block: {block}")
            feature_rows.append(
                {
                    "dataset": args.dataset,
                    "horizon": int(horizon),
                    "feature_block": block,
                    "feature_dim": int(train_blocks[block].shape[1]),
                    "train_rows": int(len(train)),
                    "val_rows": int(len(val)),
                    "test_rows": int(len(test)),
                }
            )
            for model in parse_strs(args.models):
                pred_res, info = ifp.fit_predict_model(
                    model,
                    train_blocks[block],
                    residual_train,
                    val_blocks[block],
                    residual_val,
                    test_blocks[block],
                    int(args.seed) + horizon,
                )
                rows.append(
                    ifp.evaluate(
                        dataset=args.dataset,
                        horizon=horizon,
                        seed=args.seed,
                        model_name=model,
                        block_name=block,
                        y=y_test,
                        proposal=proposal_test,
                        pred_residual=pred_res,
                        info=info,
                    )
                )
    summary = pd.DataFrame(rows)
    feature_df = pd.DataFrame(feature_rows)
    ablation_rows = []
    for (dataset, horizon, seed, model), group in summary[summary["model"].ne("proposal")].groupby(["dataset", "horizon", "seed", "model"]):
        by = group.set_index("feature_block")
        if "trajectory_only" not in by.index:
            continue
        traj = float(by.loc["trajectory_only", "rmse_px"])
        full = float(by.loc["trajectory_morphology_tissue_flow", "rmse_px"]) if "trajectory_morphology_tissue_flow" in by.index else math.nan
        morph = float(by.loc["trajectory_morphology", "rmse_px"]) if "trajectory_morphology" in by.index else math.nan
        flow = float(by.loc["trajectory_tissue_flow", "rmse_px"]) if "trajectory_tissue_flow" in by.index else math.nan
        shuf_both = (
            float(by.loc["trajectory_morphology_tissue_flow_shuffled_both", "rmse_px"])
            if "trajectory_morphology_tissue_flow_shuffled_both" in by.index
            else math.nan
        )
        shuf_ms = (
            float(by.loc["trajectory_morphology_tissue_flow_shuffled_ms", "rmse_px"])
            if "trajectory_morphology_tissue_flow_shuffled_ms" in by.index
            else math.nan
        )
        shuf_tf = (
            float(by.loc["trajectory_morphology_tissue_flow_shuffled_tf", "rmse_px"])
            if "trajectory_morphology_tissue_flow_shuffled_tf" in by.index
            else math.nan
        )
        time = (
            float(by.loc["trajectory_morphology_tissue_flow_time_shuffled", "rmse_px"])
            if "trajectory_morphology_tissue_flow_time_shuffled" in by.index
            else math.nan
        )
        best_single = np.nanmin([morph, flow])
        best_control = np.nanmin([shuf_both, shuf_ms, shuf_tf, time])
        ablation_rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "seed": int(seed),
                "model": model,
                "trajectory_rmse_px": traj,
                "morphology_rmse_px": morph,
                "tissue_flow_rmse_px": flow,
                "combined_rmse_px": full,
                "shuffled_both_rmse_px": shuf_both,
                "shuffled_ms_rmse_px": shuf_ms,
                "shuffled_tf_rmse_px": shuf_tf,
                "time_shuffled_rmse_px": time,
                "gain_combined_vs_trajectory_pct": ifp.gain_pct(traj, full) if np.isfinite(full) else math.nan,
                "gain_combined_vs_best_single_pct": ifp.gain_pct(best_single, full) if np.isfinite(full) and np.isfinite(best_single) else math.nan,
                "gain_combined_vs_best_control_pct": ifp.gain_pct(best_control, full) if np.isfinite(full) and np.isfinite(best_control) else math.nan,
                "beats_controls": bool(np.isfinite(full) and np.isfinite(best_control) and full < best_control),
            }
        )
    return summary, feature_df, pd.DataFrame(ablation_rows)


def plot_ablation(ablation: pd.DataFrame, out_path: Path) -> None:
    if ablation.empty:
        return
    df = ablation.copy()
    df["label"] = df["model"] + " h" + df["horizon"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.bar(df["label"], df["gain_combined_vs_trajectory_pct"], color=np.where(df["beats_controls"], "#0f766e", "#94a3b8"))
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.axhline(5, color="#111827", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Combined gain vs trajectory-only, %")
    ax.set_title("Feature reconnaissance: morphology + tissue-flow")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, ablation: pd.DataFrame, feature_rows: pd.DataFrame, args: argparse.Namespace, n_combined: int) -> None:
    lines = [
        "# LaChance Feature Reconnaissance",
        "",
        "## Decision",
        "",
    ]
    strong = ablation[
        ablation["beats_controls"].astype(bool)
        & ablation["gain_combined_vs_trajectory_pct"].ge(5.0)
        & ablation["gain_combined_vs_best_single_pct"].gt(1.0)
    ] if not ablation.empty else pd.DataFrame()
    moderate = ablation[
        ablation["beats_controls"].astype(bool)
        & ablation["gain_combined_vs_trajectory_pct"].ge(3.0)
    ] if not ablation.empty else pd.DataFrame()
    if len(strong):
        lines.append("- Strong combined feature hook found.")
    elif len(moderate):
        lines.append("- Moderate combined feature hook found; continue but verify more splits.")
    else:
        lines.append("- No breakthrough combined feature hook yet; inspect components before scaling architecture.")
    lines += [
        "",
        "## Config",
        "",
        "```json",
        json.dumps(finite_json(vars(args)) | {"combined_rows": int(n_combined)}, indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Ablation",
        "",
        ablation.to_markdown(index=False) if len(ablation) else "_No ablation rows._",
        "",
        "## Best Rows",
        "",
        summary.sort_values(["horizon", "rmse_px"]).groupby("horizon").head(12).to_markdown(index=False) if len(summary) else "_No summary rows._",
        "",
        "## Feature Blocks",
        "",
        feature_rows.to_markdown(index=False) if len(feature_rows) else "_No feature rows._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--table-root", type=Path, default=ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--multiscale-features", type=Path, default=DEFAULT_MS)
    parser.add_argument("--tissue-flow-features", type=Path, default=DEFAULT_TF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizons", default="1,2,4,6")
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--models", default="ridge")
    parser.add_argument(
        "--feature-blocks",
        default=(
            "trajectory_only,trajectory_morphology,trajectory_tissue_flow,"
            "trajectory_morphology_tissue_flow,"
            "trajectory_morphology_tissue_flow_shuffled_both,"
            "trajectory_morphology_tissue_flow_shuffled_ms,"
            "trajectory_morphology_tissue_flow_shuffled_tf,"
            "trajectory_morphology_tissue_flow_time_shuffled,"
            "trajectory_tf_cur,trajectory_tf_base,trajectory_tf_alignment,"
            "trajectory_ms_cur,trajectory_ms_delta,trajectory_ms_shape,trajectory_ms_texture"
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    combined = load_combined(args.multiscale_features, args.tissue_flow_features, args.dataset)
    combined.to_csv(args.out_dir / "combined_feature_grid.csv", index=False)
    summary, feature_rows, ablation = evaluate_probe(combined, args)
    summary.to_csv(args.out_dir / "feature_reconnaissance_summary.csv", index=False)
    feature_rows.to_csv(args.out_dir / "feature_reconnaissance_blocks.csv", index=False)
    ablation.to_csv(args.out_dir / "feature_reconnaissance_ablation.csv", index=False)
    plot_ablation(ablation, args.out_dir / "plots" / "combined_feature_gain.png")
    write_report(args.out_dir / "feature_reconnaissance_status_report.md", summary, ablation, feature_rows, args, len(combined))
    print(args.out_dir / "feature_reconnaissance_status_report.md")


if __name__ == "__main__":
    main()
