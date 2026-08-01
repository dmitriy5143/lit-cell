#!/usr/bin/env python3
"""Latent-history generator gate.

This runner extends the validated decomposition-stage bridge with long causal
track memory.  It tests whether a cell's full past trajectory improves the
student latent prior and therefore makes future trajectory samples more
individualized.

It is still a generator gate, not a final Sequence Critic-Refiner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402


DEFAULT_FEATURES = audit.DEFAULT_FEATURES
DEFAULT_OUT = ROOT / "outputs" / "latent_history_generator_2026-06-24"
EPS = 1e-8


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def read_tracks(table_root: Path, dataset: str, sequences: list[int]) -> pd.DataFrame:
    rows = []
    cols = [
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
    ]
    for seq in sorted(set(int(s) for s in sequences)):
        path = table_root / dataset / f"{dataset}_{seq:02d}_tracks.csv"
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in cols if c in header.columns]
        df = pd.read_csv(path, usecols=usecols)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["sequence"] = out["sequence"].astype(int)
    out["frame"] = out["frame"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    return out.sort_values(["sequence", "track_id", "frame"]).reset_index(drop=True)


def track_store(tracks: pd.DataFrame) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    store: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for (seq, tid), g in tracks.groupby(["sequence", "track_id"], sort=False):
        g = g.sort_values("frame")
        dx = g["dx_px"].fillna(0.0).to_numpy(np.float32)
        dy = g["dy_px"].fillna(0.0).to_numpy(np.float32)
        speed = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        ax = g["ax_px_s2"].fillna(0.0).to_numpy(np.float32) if "ax_px_s2" in g.columns else np.zeros_like(dx)
        ay = g["ay_px_s2"].fillna(0.0).to_numpy(np.float32) if "ay_px_s2" in g.columns else np.zeros_like(dx)
        frames = g["frame"].to_numpy(np.int64)
        step = np.column_stack([dx, dy]).astype(np.float32)
        prev = np.roll(step, 1, axis=0)
        prev[0] = 0.0
        denom = np.maximum(np.linalg.norm(step, axis=1) * np.linalg.norm(prev, axis=1), 1e-6)
        turn_cos = np.sum(step * prev, axis=1) / denom
        turn_cos[0] = 0.0
        turn_sin = (prev[:, 0] * step[:, 1] - prev[:, 1] * step[:, 0]) / denom
        turn_sin[0] = 0.0
        store[(int(seq), int(tid))] = {
            "frame": frames,
            "dx": dx,
            "dy": dy,
            "speed": speed,
            "ax": ax.astype(np.float32),
            "ay": ay.astype(np.float32),
            "turn_cos": np.nan_to_num(turn_cos).astype(np.float32),
            "turn_sin": np.nan_to_num(turn_sin).astype(np.float32),
        }
    return store


def slope_last(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    t = np.arange(len(x), dtype=np.float32)
    t = t - t.mean()
    y = x.astype(np.float32) - float(np.mean(x))
    return float(np.sum(t * y) / max(float(np.sum(t * t)), EPS))


def window_stats(data: dict[str, np.ndarray], end: int, w: int) -> list[float]:
    start = max(0, end - int(w) + 1)
    sl = slice(start, end + 1)
    dx, dy = data["dx"][sl], data["dy"][sl]
    speed = data["speed"][sl]
    ax, ay = data["ax"][sl], data["ay"][sl]
    tc, ts = data["turn_cos"][sl], data["turn_sin"][sl]
    n = len(dx)
    net = np.array([float(np.sum(dx)), float(np.sum(dy))], dtype=np.float32)
    path = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    persistence = float(np.linalg.norm(net) / max(path, EPS))
    cur = np.array([float(dx[-1]) if n else 0.0, float(dy[-1]) if n else 0.0], dtype=np.float32)
    cur_norm = float(np.linalg.norm(cur))
    net_norm = float(np.linalg.norm(net))
    drift_cos = float(np.dot(cur, net) / max(cur_norm * net_norm, EPS)) if n else 0.0
    return [
        n / float(max(w, 1)),
        float(np.mean(dx)) if n else 0.0,
        float(np.mean(dy)) if n else 0.0,
        float(np.std(dx)) if n else 0.0,
        float(np.std(dy)) if n else 0.0,
        float(np.mean(speed)) if n else 0.0,
        float(np.std(speed)) if n else 0.0,
        float(np.mean(ax)) if n else 0.0,
        float(np.mean(ay)) if n else 0.0,
        float(np.std(np.sqrt(ax * ax + ay * ay))) if n else 0.0,
        float(net[0]),
        float(net[1]),
        path,
        persistence,
        float(np.mean(tc)) if n else 0.0,
        float(np.std(tc)) if n else 0.0,
        float(np.mean(ts)) if n else 0.0,
        slope_last(speed),
        drift_cos,
    ]


def history_features(
    df: pd.DataFrame,
    store: dict[tuple[int, int], dict[str, np.ndarray]],
    *,
    windows: list[int],
    flat_lags: int,
) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    for w in windows:
        for n in [
            "count_frac",
            "mean_dx",
            "mean_dy",
            "std_dx",
            "std_dy",
            "mean_speed",
            "std_speed",
            "mean_ax",
            "mean_ay",
            "std_acc",
            "net_dx",
            "net_dy",
            "path_len",
            "persistence",
            "mean_turn_cos",
            "std_turn_cos",
            "mean_turn_sin",
            "speed_slope",
            "drift_cos_current",
        ]:
            names.append(f"hist_w{w}_{n}")
    for lag in range(flat_lags):
        for n in ["dx", "dy", "speed", "turn_cos", "valid"]:
            names.append(f"hist_lag{lag}_{n}")
    rows = np.zeros((len(df), len(names)), dtype=np.float32)
    for r, row in enumerate(df[["sequence", "track_id", "frame"]].itertuples(index=False)):
        key = (int(row.sequence), int(row.track_id))
        data = store.get(key)
        if data is None:
            continue
        frames = data["frame"]
        end = int(np.searchsorted(frames, int(row.frame), side="right") - 1)
        if end < 0:
            continue
        vals: list[float] = []
        for w in windows:
            vals.extend(window_stats(data, end, int(w)))
        for lag in range(flat_lags):
            idx = end - lag
            if idx >= 0:
                vals.extend(
                    [
                        float(data["dx"][idx]),
                        float(data["dy"][idx]),
                        float(data["speed"][idx]),
                        float(data["turn_cos"][idx]),
                        1.0,
                    ]
                )
            else:
                vals.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        rows[r] = np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return rows, names


def add_history_blocks(arrays: audit.SplitArrays, split: audit.seq.SplitData, args: argparse.Namespace) -> audit.SplitArrays:
    seqs = sorted(set(split.train["sequence"].astype(int)) | set(split.val["sequence"].astype(int)) | set(split.test["sequence"].astype(int)))
    tracks = read_tracks(Path(args.table_root), args.dataset, seqs)
    store = track_store(tracks)
    windows = parse_ints(args.history_windows)
    tr, names = history_features(split.train, store, windows=windows, flat_lags=args.history_flat_lags)
    va, _ = history_features(split.val, store, windows=windows, flat_lags=args.history_flat_lags)
    te, _ = history_features(split.test, store, windows=windows, flat_lags=args.history_flat_lags)
    tr_z, va_z, te_z, _ = audit.standardize_block(tr, va, te)

    # Split summary and flattened recent sequence so ablations can isolate them.
    summary_dim = len(windows) * 19
    x_train = dict(arrays.x_train)
    x_val = dict(arrays.x_val)
    x_test = dict(arrays.x_test)
    feature_names = dict(arrays.feature_names)

    x_train["history_summary"] = tr_z[:, :summary_dim]
    x_val["history_summary"] = va_z[:, :summary_dim]
    x_test["history_summary"] = te_z[:, :summary_dim]
    feature_names["history_summary"] = names[:summary_dim]

    x_train["history_long"] = tr_z
    x_val["history_long"] = va_z
    x_test["history_long"] = te_z
    feature_names["history_long"] = names

    short_cols = min(summary_dim + min(args.history_flat_lags, 8) * 5, tr_z.shape[1])
    x_train["history_short"] = tr_z[:, :short_cols]
    x_val["history_short"] = va_z[:, :short_cols]
    x_test["history_short"] = te_z[:, :short_cols]
    feature_names["history_short"] = names[:short_cols]

    return replace(arrays, x_train=x_train, x_val=x_val, x_test=x_test, feature_names=feature_names)


def shuffle_history_test(arrays: audit.SplitArrays, seed: int) -> audit.SplitArrays:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arrays.residual_test))
    x_test = dict(arrays.x_test)
    for b in ["history_short", "history_long", "history_summary"]:
        if b in x_test:
            x_test[b] = x_test[b][perm]
    return replace(arrays, x_test=x_test)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    arrays = add_history_blocks(arrays, split, args)
    posterior = closure.train_posterior(arrays, args, device)

    summary_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    summary_rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=np.zeros_like(arrays.residual_test),
            horizons=args.horizons,
            label="base_self_rollout_reference",
            extra={"stage": "reference", "variant": "base"},
        )
    )
    summary_rows.extend(closure.posterior_metrics(arrays, posterior, args, device))
    summary_rows.extend(closure.ridge_direct_baseline(arrays, posterior, args))
    summary_rows.extend(closure.random_latent_oracle(arrays, posterior, args, device))

    for variant in args.variants:
        model, blocks, _ = closure.train_student(arrays, posterior, args, variant=variant, device=device, row_shuffle_train=False)
        s, d, g = closure.evaluate_variant(arrays, posterior, model, blocks, args, variant=variant, device=device)
        summary_rows.extend(s)
        distill_rows.extend(d)
        gate_rows.extend(g)
        if variant == "full":
            s, d, g = closure.evaluate_variant(
                shuffle_history_test(arrays, args.seed + 909),
                posterior,
                model,
                blocks,
                args,
                variant="full_history_test_shuffled",
                device=device,
            )
            summary_rows.extend(s)
            distill_rows.extend(d)
            gate_rows.extend(g)
            model_sh, blocks_sh, _ = closure.train_student(arrays, posterior, args, variant="full", device=device, row_shuffle_train=True)
            s, d, g = closure.evaluate_variant(arrays, posterior, model_sh, blocks_sh, args, variant="full_row_shuffled_train", device=device)
            summary_rows.extend(s)
            distill_rows.extend(d)
            gate_rows.extend(g)

    summary = pd.DataFrame(summary_rows)
    distill = pd.DataFrame(distill_rows)
    gates = pd.DataFrame(gate_rows)
    summary.to_csv(args.out_dir / "latent_history_generator_summary.csv", index=False)
    distill.to_csv(args.out_dir / "latent_history_generator_distillation.csv", index=False)
    gates.to_csv(args.out_dir / "latent_history_generator_gates.csv", index=False)
    (args.out_dir / "feature_blocks.json").write_text(json.dumps(arrays.feature_names, indent=2), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, distill, gates)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "distill_rows": len(distill), "gate_rows": len(gates)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, distill: pd.DataFrame, gates: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Latent History Generator Gate Report\n")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- history_windows: `{args.history_windows}`, flat_lags: `{args.history_flat_lags}`")
    lines.append("")
    if not summary.empty:
        lines.append("## Endpoint / Oracle")
        focus = summary[summary["horizon"].notna()].copy()
        for h in args.horizons:
            lines.append(f"### h{h}")
            for _, row in focus[focus["horizon"].eq(h)].sort_values("rmse").head(14).iterrows():
                lines.append(
                    f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, gain={row['gain_vs_base_pct']:.2f}%"
                )
    if not distill.empty:
        lines.append("\n## Distillation")
        for _, row in distill.sort_values("gaussian_kl_q_to_p").iterrows():
            lines.append(
                f"- `{row['variant']}` shuffle={row['test_shuffle']}: KL={row['gaussian_kl_q_to_p']:.3f}, "
                f"latent_rmse={row['latent_rmse']:.3f}, mode_acc={row['mode_acc']:.3f}, top3={row['mode_top3']:.3f}"
            )
    if not gates.empty:
        lines.append("\n## Gates")
        for variant, sub in gates.groupby("variant"):
            lines.append(f"### {variant}")
            agg = sub.groupby("block").agg(mean_gate=("mean_gate", "mean")).reset_index()
            for _, row in agg.sort_values("mean_gate", ascending=False).iterrows():
                lines.append(f"- `{row['block']}`: {row['mean_gate']:.3f}")
    lines.append("\n## Decision Notes")
    lines.append("- History helps only if full/trajectory_history beats no_history/trajectory_only and history-shuffled controls degrade.")
    lines.append("- This is the last gate before building a full sequence critic-refiner.")
    (out_dir / "latent_history_generator_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    parser.add_argument("--dataset", type=str, default="MDCK_Bulk")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seq", type=str, default="1,2,3,4")
    parser.add_argument("--val-seq", type=str, default="5")
    parser.add_argument("--test-seq", type=str, default="6")
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--horizons", type=str, default="1,2,4,6")
    parser.add_argument("--max-train-rows", type=int, default=22000)
    parser.add_argument("--max-val-rows", type=int, default=6000)
    parser.add_argument("--max-test-rows", type=int, default=8000)
    parser.add_argument("--max-features-per-family", type=int, default=160)
    parser.add_argument("--max-all-features", type=int, default=384)
    parser.add_argument("--history-windows", type=str, default="4,8,16,32,64")
    parser.add_argument("--history-flat-lags", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--mode-k", type=int, default=12)
    parser.add_argument("--mode-temperature", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--posterior-epochs", type=int, default=24)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--posterior-beta", type=float, default=1e-3)
    parser.add_argument("--mode-loss-weight", type=float, default=0.30)
    parser.add_argument("--recon-loss-weight", type=float, default=0.20)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--oracle-k", type=str, default="8,16,32")
    parser.add_argument("--sample-scale", type=float, default=1.0)
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_history,trajectory_only,trajectory_history,history_only,no_raw_context,no_flow",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.variants = parse_strs(args.variants)
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 5000)
        args.max_val_rows = min(args.max_val_rows, 2000)
        args.max_test_rows = min(args.max_test_rows, 2500)
        args.posterior_epochs = min(args.posterior_epochs, 8)
        args.student_epochs = min(args.student_epochs, 8)
        args.max_all_features = min(args.max_all_features, 192)
        args.history_flat_lags = min(args.history_flat_lags, 16)
        args.variants = ["full", "no_history", "trajectory_only", "trajectory_history", "history_only"]
        args.oracle_k = [8, 16]
    run(args)


if __name__ == "__main__":
    main()
