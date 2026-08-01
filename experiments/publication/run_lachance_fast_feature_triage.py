#!/usr/bin/env python3
"""Fast feature triage for LaChance h1-first forecasting.

This runner is intentionally pre-architectural.  It takes an already extracted
causal feature grid and asks which feature families deserve expensive neural
training.  Each candidate block is tested with cheap residual probes and strict
negative controls:

    trajectory/self-motion + block -> residual to constant-velocity proposal

No future image, future coordinate, target-derived candidate label, or test-set
normalization is used as an inference feature.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

DEFAULT_FEATURE_GRID = (
    ROOT
    / "outputs"
    / "lachance_feature_reconnaissance_ms_tf_mdck_bulk_h1h4h6_seed42_2026-06-15"
    / "combined_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_fast_feature_triage_2026-06-16"
EPS = 1e-8


def finite_json(value: Any) -> Any:
    return ifp.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text or "").split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def clean_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def concat_blocks(*parts: np.ndarray) -> np.ndarray:
    keep = [p for p in parts if p.shape[1] > 0]
    if not keep:
        return np.zeros((parts[0].shape[0], 0), dtype=np.float32)
    return np.concatenate(keep, axis=1).astype(np.float32, copy=False)


def columns_matching(cols: list[str], *, include: tuple[str, ...] = (), exclude: tuple[str, ...] = ()) -> list[str]:
    out: list[str] = []
    for col in cols:
        if include and not any(token in col for token in include):
            continue
        if exclude and any(token in col for token in exclude):
            continue
        out.append(col)
    return out


def radius_columns(cols: list[str], prefix: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_r(\d+)_")
    for col in cols:
        match = pattern.match(col)
        if match:
            found.setdefault(f"{prefix}_r{match.group(1)}", []).append(col)
    return dict(sorted(found.items(), key=lambda kv: int(kv[0].split("r")[-1])))


def make_feature_specs(df: pd.DataFrame) -> dict[str, list[str]]:
    ms_cols = [c for c in df.columns if c.startswith("ms_")]
    tf_cols = [c for c in df.columns if c.startswith("tf_")]
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    specs: dict[str, list[str]] = {}

    # Morphology/image packets.
    specs["ms_all"] = ms_cols
    specs["ms_cur"] = columns_matching(ms_cols, include=("_cur_",))
    specs["ms_prev"] = columns_matching(ms_cols, include=("_prev_",))
    specs["ms_delta"] = columns_matching(ms_cols, include=("_delta_",))
    specs["ms_shape"] = columns_matching(ms_cols, include=("elongation", "orient", "centroid"))
    specs["ms_texture"] = columns_matching(ms_cols, include=("mean", "std", "p10", "p50", "p90", "grad", "fg_frac"))
    specs["ms_intensity"] = columns_matching(ms_cols, include=("mean", "std", "p10", "p50", "p90"))
    specs["ms_grad"] = columns_matching(ms_cols, include=("grad",))
    specs["ms_fg"] = columns_matching(ms_cols, include=("fg_frac",))
    specs.update(radius_columns(ms_cols, "ms"))

    # Tissue-flow/PIV-like packets.
    specs["tf_all"] = tf_cols
    specs["tf_cur"] = columns_matching(tf_cols, include=("_cur_",))
    specs["tf_prev"] = columns_matching(tf_cols, include=("_prev_",))
    specs["tf_base_motion"] = columns_matching(
        tf_cols,
        include=("u_mean", "v_mean", "mag_mean", "center_u", "center_v", "u_median", "v_median", "mag_median"),
        exclude=("div_", "curl_", "shear", "front_back", "own_minus", "proj_", "cos_to_own"),
    )
    specs["tf_alignment"] = columns_matching(tf_cols, include=("own_minus", "proj_own_dir", "proj_tangent", "cos_to_own"))
    specs["tf_deformation"] = columns_matching(tf_cols, include=("div_", "curl_", "shear"))
    specs["tf_front_back"] = columns_matching(tf_cols, include=("front_back",))
    specs["tf_magnitude"] = columns_matching(tf_cols, include=("mag_",))
    specs.update(radius_columns(tf_cols, "tf"))

    # Combined hypotheses.
    specs["ms_shape_tf_cur"] = sorted(set(specs["ms_shape"] + specs["tf_cur"]))
    specs["ms_texture_tf_cur"] = sorted(set(specs["ms_texture"] + specs["tf_cur"]))
    specs["ms_shape_tf_alignment"] = sorted(set(specs["ms_shape"] + specs["tf_alignment"]))
    specs["ms_shape_tf_deformation"] = sorted(set(specs["ms_shape"] + specs["tf_deformation"]))
    specs["ms_all_tf_cur"] = sorted(set(specs["ms_all"] + specs["tf_cur"]))
    specs["ms_shape_tf_all"] = sorted(set(specs["ms_shape"] + specs["tf_all"]))
    specs["ms_all_tf_all"] = sorted(set(specs["ms_all"] + specs["tf_all"]))

    # Observability packets generated by build_lachance_observability_feature_grid.py.
    # These are derived causal descriptors, kept under a separate prefix so they can
    # be audited independently from raw morphology/tissue-flow columns.
    specs["obs_all"] = obs_cols
    specs["obs_polarity"] = columns_matching(obs_cols, include=("polarity", "front_back", "orient", "centroid", "self_axis"))
    specs["obs_flow_lag"] = columns_matching(obs_cols, include=("flow", "lag", "accel", "coherence", "own_minus"))
    specs["obs_boundary_front"] = columns_matching(obs_cols, include=("boundary", "front"))
    specs["obs_density"] = columns_matching(obs_cols, include=("density", "knn", "crowd", "nearest"))
    specs["obs_shape_flow"] = columns_matching(obs_cols, include=("shape_flow", "elong_flow", "centroid_flow", "orient_flow"))
    specs["obs_context_core"] = sorted(
        set(
            specs.get("obs_polarity", [])
            + specs.get("obs_flow_lag", [])
            + specs.get("obs_boundary_front", [])
            + specs.get("obs_density", [])
            + specs.get("obs_shape_flow", [])
        )
    )
    specs["ms_shape_tf_alignment_obs"] = sorted(set(specs["ms_shape_tf_alignment"] + specs["obs_context_core"]))
    specs["ms_all_tf_all_obs"] = sorted(set(specs["ms_all_tf_all"] + specs["obs_context_core"]))

    # Richer raw-context v2 packets generated by build_lachance_raw_context_v2_grid.py.
    specs["rc_all"] = rc_cols
    specs["rc_center"] = columns_matching(rc_cols, include=("rc_c_",))
    specs["rc_noncenter"] = columns_matching(rc_cols, include=("rc_nc_",))
    specs["rc_temporal"] = columns_matching(rc_cols, include=("rc_lag",))
    specs["rc_neighbor"] = columns_matching(rc_cols, include=("rc_nei",))
    specs["rc_structure"] = columns_matching(rc_cols, include=("struct_", "elongation", "orient", "centroid"))
    specs["rc_polarity"] = columns_matching(
        rc_cols,
        include=("front_minus_back", "left_minus_right", "front_", "back_", "left_", "right_", "orient", "centroid"),
    )
    specs["rc_multicell"] = sorted(set(specs.get("rc_noncenter", []) + specs.get("rc_neighbor", [])))
    specs["ms_shape_tf_alignment_rc"] = sorted(set(specs["ms_shape_tf_alignment"] + specs.get("rc_all", [])))
    specs["ms_shape_tf_alignment_rc_core"] = sorted(
        set(specs["ms_shape_tf_alignment"] + specs.get("rc_structure", []) + specs.get("rc_temporal", []))
    )
    specs["ms_all_tf_all_rc"] = sorted(set(specs["ms_all_tf_all"] + specs.get("rc_all", [])))

    # Drop empty packets.  Keep insertion order.
    return {name: cols for name, cols in specs.items() if cols}


def trajectory_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ifp.TRAJECTORY_FEATURES if c in df.columns]
    return clean_matrix(df, cols)


def shuffled_by_row(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return x[rng.permutation(len(x))]


def residual_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df[["target_dx", "target_dy"]].to_numpy(np.float32)
    proposal = df[["proposal_dx", "proposal_dy"]].to_numpy(np.float32)
    residual = y - proposal
    return y, proposal, residual


def residual_corr_score(train_x: np.ndarray, train_residual: np.ndarray) -> float:
    """Cheap train-only scalar score: max absolute Pearson corr with residual dims/norm."""

    if train_x.shape[1] == 0 or len(train_x) < 3:
        return 0.0
    x = np.nan_to_num(train_x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1e6, 1e6)
    x = x - x.mean(axis=0, keepdims=True)
    x_std = np.maximum(x.std(axis=0, keepdims=True), EPS)
    x = x / x_std
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    res_norm = np.linalg.norm(train_residual.astype(np.float64), axis=1, keepdims=True)
    targets = np.concatenate([train_residual.astype(np.float64), res_norm], axis=1)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, -1e6, 1e6)
    targets = targets - targets.mean(axis=0, keepdims=True)
    targets = targets / np.maximum(targets.std(axis=0, keepdims=True), EPS)
    targets = np.clip(np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    with np.errstate(all="ignore"):
        corr = np.abs(np.matmul(x.T, targets) / max(len(x) - 1, 1))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.nanmax(corr)) if corr.size else 0.0


def evaluate_block(
    *,
    args: argparse.Namespace,
    dataset: str,
    horizon: int,
    seed: int,
    model: str,
    block_name: str,
    block_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    mode: str,
) -> dict[str, Any]:
    train_traj = trajectory_matrix(train)
    val_traj = trajectory_matrix(val)
    test_traj = trajectory_matrix(test)
    train_block = clean_matrix(train, block_cols)
    val_block = clean_matrix(val, block_cols)
    test_block = clean_matrix(test, block_cols)
    if mode == "real":
        pass
    elif mode == "row_shuffled":
        train_block = shuffled_by_row(train_block, seed + 101)
        val_block = shuffled_by_row(val_block, seed + 103)
        test_block = shuffled_by_row(test_block, seed + 107)
    elif mode == "time_shuffled":
        train_block = ifp.time_shuffled_image(train, block_cols, seed + 201)
        val_block = ifp.time_shuffled_image(val, block_cols, seed + 203)
        test_block = ifp.time_shuffled_image(test, block_cols, seed + 207)
    else:
        raise ValueError(f"unknown mode={mode}")

    train_x = concat_blocks(train_traj, train_block)
    val_x = concat_blocks(val_traj, val_block)
    test_x = concat_blocks(test_traj, test_block)
    y_train, _, residual_train = residual_targets(train)
    y_val, _, residual_val = residual_targets(val)
    y_test, proposal_test, _ = residual_targets(test)
    pred_residual, info = ifp.fit_predict_model(
        model,
        train_x,
        residual_train,
        val_x,
        residual_val,
        test_x,
        seed + horizon * 17,
    )
    row = ifp.evaluate(
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        model_name=model,
        block_name=f"{block_name}__{mode}",
        y=y_test,
        proposal=proposal_test,
        pred_residual=pred_residual,
        info=info,
    )
    row.update(
        {
            "candidate_block": block_name,
            "control_mode": mode,
            "candidate_dim": int(train_block.shape[1]),
            "total_dim": int(train_x.shape[1]),
            "train_corr_score": residual_corr_score(train_block, residual_train),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        }
    )
    return row


def evaluate_trajectory_baseline(
    *,
    args: argparse.Namespace,
    dataset: str,
    horizon: int,
    seed: int,
    model: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    y_train, _, residual_train = residual_targets(train)
    y_val, _, residual_val = residual_targets(val)
    y_test, proposal_test, _ = residual_targets(test)
    train_x = trajectory_matrix(train)
    val_x = trajectory_matrix(val)
    test_x = trajectory_matrix(test)
    pred_residual, info = ifp.fit_predict_model(
        model,
        train_x,
        residual_train,
        val_x,
        residual_val,
        test_x,
        seed + horizon * 11,
    )
    row = ifp.evaluate(
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        model_name=model,
        block_name="trajectory_only",
        y=y_test,
        proposal=proposal_test,
        pred_residual=pred_residual,
        info=info,
    )
    row.update(
        {
            "candidate_block": "trajectory_only",
            "control_mode": "baseline",
            "candidate_dim": 0,
            "total_dim": int(train_x.shape[1]),
            "train_corr_score": 0.0,
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        }
    )
    return row


def summarize_gates(summary: pd.DataFrame, gate_gain: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    real = summary[summary["control_mode"].eq("real")].copy()
    base = summary[summary["candidate_block"].eq("trajectory_only")].copy()
    for _, row in real.iterrows():
        key = (
            row["dataset"],
            int(row["horizon"]),
            int(row["seed"]),
            row["model"],
        )
        base_match = base[
            base["dataset"].eq(key[0])
            & base["horizon"].eq(key[1])
            & base["seed"].eq(key[2])
            & base["model"].eq(key[3])
        ]
        if base_match.empty:
            continue
        control_match = summary[
            summary["dataset"].eq(key[0])
            & summary["horizon"].eq(key[1])
            & summary["seed"].eq(key[2])
            & summary["model"].eq(key[3])
            & summary["candidate_block"].eq(row["candidate_block"])
            & summary["control_mode"].isin(["row_shuffled", "time_shuffled"])
        ]
        base_rmse = float(base_match.iloc[0]["rmse_px"])
        real_rmse = float(row["rmse_px"])
        best_control = float(control_match["rmse_px"].min()) if len(control_match) else math.nan
        rows.append(
            {
                "dataset": key[0],
                "horizon": key[1],
                "seed": key[2],
                "model": key[3],
                "candidate_block": row["candidate_block"],
                "candidate_dim": int(row["candidate_dim"]),
                "trajectory_rmse_px": base_rmse,
                "real_rmse_px": real_rmse,
                "best_control_rmse_px": best_control,
                "gain_vs_trajectory_pct": ifp.gain_pct(base_rmse, real_rmse),
                "gain_vs_best_control_pct": ifp.gain_pct(best_control, real_rmse) if np.isfinite(best_control) else math.nan,
                "train_corr_score": float(row["train_corr_score"]),
                "passes_gate": bool(
                    ifp.gain_pct(base_rmse, real_rmse) >= gate_gain
                    and np.isfinite(best_control)
                    and real_rmse < best_control
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_gate", "horizon", "gain_vs_trajectory_pct", "gain_vs_best_control_pct"],
        ascending=[False, True, False, False],
    )


def plot_gate(gate: pd.DataFrame, path: Path) -> None:
    if gate.empty:
        return
    top = gate.sort_values("gain_vs_trajectory_pct", ascending=False).head(24).copy()
    top["label"] = top["candidate_block"] + " h" + top["horizon"].astype(str)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = np.where(top["passes_gate"].astype(bool), "#0f766e", "#94a3b8")
    ax.barh(top["label"], top["gain_vs_trajectory_pct"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.axvline(5, color="#111827", linestyle="--", linewidth=0.8)
    ax.set_xlabel("RMSE gain vs trajectory-only, %")
    ax.set_title("Fast feature triage: top candidate blocks")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, gate: pd.DataFrame, specs: dict[str, list[str]], args: argparse.Namespace) -> None:
    winners = gate[gate["passes_gate"].astype(bool)].copy() if len(gate) else pd.DataFrame()
    lines = [
        "# Fast Feature Triage Report",
        "",
        "## Decision",
        "",
    ]
    if len(winners):
        lines.append("- Feature hooks found; only these blocks should move to full neural tests first.")
    else:
        lines.append("- No block passed the strict gate; do not spend full training on these variants yet.")
    lines += [
        "",
        "## Config",
        "",
        "```json",
        json.dumps(finite_json(vars(args)), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Passed / Top Gates",
        "",
        (winners.to_markdown(index=False) if len(winners) else "_No strict gate winners._"),
        "",
        "## Top Candidate Blocks",
        "",
        (gate.head(40).to_markdown(index=False) if len(gate) else "_No gate rows._"),
        "",
        "## Tested Feature Specs",
        "",
        pd.DataFrame(
            [{"candidate_block": name, "feature_dim": len(cols)} for name, cols in specs.items()]
        ).to_markdown(index=False),
        "",
        "## Best Raw Rows",
        "",
        (
            summary.sort_values(["horizon", "rmse_px"]).groupby(["horizon", "model"]).head(10).to_markdown(index=False)
            if len(summary)
            else "_No summary rows._"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--feature-grid", type=Path, default=DEFAULT_FEATURE_GRID)
    parser.add_argument("--table-root", type=Path, default=ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizons", default="1,4,6")
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--models", default="ridge")
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=15000)
    parser.add_argument("--max-test-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-gain-pct", type=float, default=3.0)
    parser.add_argument(
        "--blocks",
        default="",
        help="Comma-separated block subset. Empty means all automatically generated blocks.",
    )
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.feature_grid)
    features = features[features["dataset"].eq(args.dataset)].copy()
    if features.empty:
        raise ValueError(f"No rows for dataset={args.dataset} in {args.feature_grid}")
    specs = make_feature_specs(features)
    requested = parse_strs(args.blocks)
    if requested:
        missing = [name for name in requested if name not in specs]
        if missing:
            raise ValueError(f"Unknown blocks: {missing}. Available: {sorted(specs)}")
        specs = {name: specs[name] for name in requested}
    rows: list[dict[str, Any]] = []
    for horizon in parse_ints(args.horizons):
        full = ifp.build_horizon_table(
            image_features=features,
            table_root=args.table_root,
            dataset=args.dataset,
            horizon=horizon,
        )
        split = ifp.make_split(
            full,
            parse_ints(args.train_sequences),
            parse_ints(args.val_sequences),
            parse_ints(args.test_sequences),
            int(args.seed),
        )
        train = ifp.sample_rows(split.train, int(args.max_train_rows), int(args.seed) + horizon * 11).reset_index(drop=True)
        val = ifp.sample_rows(split.val, int(args.max_val_rows), int(args.seed) + horizon * 13).reset_index(drop=True)
        test = ifp.sample_rows(split.test, int(args.max_test_rows), int(args.seed) + horizon * 17).reset_index(drop=True)
        for model in parse_strs(args.models):
            rows.append(
                evaluate_trajectory_baseline(
                    args=args,
                    dataset=args.dataset,
                    horizon=horizon,
                    seed=int(args.seed),
                    model=model,
                    train=train,
                    val=val,
                    test=test,
                )
            )
            for name, cols in specs.items():
                rows.append(
                    evaluate_block(
                        args=args,
                        dataset=args.dataset,
                        horizon=horizon,
                        seed=int(args.seed),
                        model=model,
                        block_name=name,
                        block_cols=cols,
                        train=train,
                        val=val,
                        test=test,
                        mode="real",
                    )
                )
                if args.include_controls:
                    for mode in ("row_shuffled", "time_shuffled"):
                        rows.append(
                            evaluate_block(
                                args=args,
                                dataset=args.dataset,
                                horizon=horizon,
                                seed=int(args.seed),
                                model=model,
                                block_name=name,
                                block_cols=cols,
                                train=train,
                                val=val,
                                test=test,
                                mode=mode,
                            )
                        )
    summary = pd.DataFrame(rows)
    gate = summarize_gates(summary, float(args.gate_gain_pct))
    spec_df = pd.DataFrame([{"candidate_block": name, "feature_dim": len(cols)} for name, cols in specs.items()])
    summary.to_csv(args.out_dir / "fast_feature_triage_summary.csv", index=False)
    gate.to_csv(args.out_dir / "fast_feature_triage_gate.csv", index=False)
    spec_df.to_csv(args.out_dir / "fast_feature_triage_specs.csv", index=False)
    plot_gate(gate, args.out_dir / "plots" / "fast_feature_triage_top_blocks.png")
    write_report(args.out_dir / "fast_feature_triage_status_report.md", summary, gate, specs, args)
    print(args.out_dir / "fast_feature_triage_status_report.md")


if __name__ == "__main__":
    main()
