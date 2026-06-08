#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import data_protocol as la  # noqa: E402

arch = la.arch

C = {
    "bg": "#0B1020",
    "panel": "#15213A",
    "line": "#31415F",
    "text": "#F4F7FB",
    "muted": "#AAB7CC",
    "cyan": "#42D9F5",
    "amber": "#F6B84A",
    "green": "#74D99F",
    "red": "#F06B6B",
    "gray": "#9AA8BA",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": C["bg"],
    "axes.facecolor": C["bg"],
    "savefig.facecolor": C["bg"],
    "axes.edgecolor": C["line"],
    "axes.labelcolor": C["text"],
    "xtick.color": C["muted"],
    "ytick.color": C["muted"],
    "text.color": C["text"],
})


def np_graph(graph):
    mask = graph.target_valid.detach().cpu().numpy().astype(bool)
    hist = graph.history.detach().cpu().numpy()
    flow = graph.flow.detach().cpu().numpy()
    y = graph.y_px.detach().cpu().numpy()
    # history is normalized; last step can still be useful for linear/MLP baselines.
    hist_flat = np.nan_to_num(hist.reshape(hist.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    hist_flat = np.clip(hist_flat, -20.0, 20.0)
    flow = np.clip(flow, -20.0, 20.0)
    last_step_norm = hist[:, -1, :2]
    return mask, hist, hist_flat, flow, y, last_step_norm


def vector_metrics(y, pred):
    return arch.vector_metrics(y, pred, 1)


def to_px_from_norm(pred_norm, norm):
    return pred_norm * norm.target_std + norm.target_mean


def fit_eval_dataset(dataset: str, table_root: Path) -> list[dict]:
    raw, meta = la.load_lachance_dataset(
        dataset,
        table_root=table_root,
        split_mode="movie",
        split_seed=42,
        max_movies=8,
        max_tracks_per_movie=0,
        frame_stride=1,
        smooth_window=1,
        crop_fraction=0.08,
        r_cut_px=50.0,
    )
    graphs, norm, _ = la.prepare_dataset(
        dataset,
        raw,
        meta,
        horizon=6,
        k=8,
        device=__import__("torch").device("cpu"),
    )
    train_mask, train_hist, train_x_hist, train_flow, train_y, train_last = np_graph(graphs["train"])
    test_mask, test_hist, test_x_hist, test_flow, test_y, test_last = np_graph(graphs["test"])
    val_mask, val_hist, val_x_hist, val_flow, val_y, val_last = np_graph(graphs["val"])

    rows = []

    # Constant-zero displacement: static cell baseline.
    pred_zero = np.zeros_like(test_y)
    rows.append({"dataset": dataset, "model": "Zero displacement", **vector_metrics(test_y[test_mask], pred_zero[test_mask])})

    # Constant velocity: repeat last observed per-frame displacement for horizon=6.
    # History is normalized, so invert normalization for dx/dy and multiply by horizon.
    last_step_px = test_last * norm.hist_std[:2] + norm.hist_mean[:2]
    pred_cv = last_step_px * 6.0
    rows.append({"dataset": dataset, "model": "Constant velocity", **vector_metrics(test_y[test_mask], pred_cv[test_mask])})

    # Ridge over own trajectory only.
    ridge_hist = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge_hist.fit(train_x_hist[train_mask], train_y[train_mask])
    pred_ridge_hist = ridge_hist.predict(test_x_hist)
    rows.append({"dataset": dataset, "model": "Ridge trajectory", **vector_metrics(test_y[test_mask], pred_ridge_hist[test_mask])})

    # Ridge over trajectory + coarse flow fields.
    train_x_flow = np.concatenate([train_x_hist, train_flow], axis=1)
    test_x_flow = np.concatenate([test_x_hist, test_flow], axis=1)
    ridge_flow = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge_flow.fit(train_x_flow[train_mask], train_y[train_mask])
    pred_ridge_flow = ridge_flow.predict(test_x_flow)
    rows.append({"dataset": dataset, "model": "Ridge trajectory + flow", **vector_metrics(test_y[test_mask], pred_ridge_flow[test_mask])})

    # Small MLP trajectory baseline. Use a deterministic subsample for speed and to avoid overclaiming.
    rng = np.random.default_rng(20260608)
    idx = np.flatnonzero(train_mask)
    if len(idx) > 60000:
        idx = rng.choice(idx, size=60000, replace=False)
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=140,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=12,
            random_state=42,
            batch_size=1024,
        ),
    )
    mlp.fit(train_x_flow[idx], train_y[idx])
    pred_mlp = mlp.predict(test_x_flow)
    rows.append({"dataset": dataset, "model": "MLP trajectory + flow", **vector_metrics(test_y[test_mask], pred_mlp[test_mask])})

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-root",
        type=Path,
        default=Path(os.environ.get("TABLE_ROOT", ROOT / "new_data" / "lachance_epithelia" / "tables")),
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ds in ["MDCK_Edge", "MDCK_Bulk"]:
        all_rows.extend(fit_eval_dataset(ds, args.table_root))

    classical = pd.DataFrame(all_rows)
    current = pd.read_csv(args.out_dir / "core_results.csv")
    ren = {
        "Self trajectory": "GRU self trajectory",
        "Self + flow": "GRU self + learned flow",
        "3-layer radial MP": "Ours: 3-layer radial MP",
    }
    current_best = current[current["model"].isin(["Self trajectory", "Self + flow", "3-layer radial MP"])].copy()
    current_best["model"] = current_best["model"].map(ren)
    current_best = current_best.rename(columns={"rmse": "rmse_px", "r2": "r2_vec"})
    current_best = current_best[["dataset", "model", "rmse_px", "r2_vec"]]

    out = pd.concat([classical[["dataset", "model", "rmse_px", "r2_vec"]], current_best], ignore_index=True)
    out["dataset"] = out["dataset"].str.replace("_", " ", regex=False)
    out.to_csv(args.out_dir / "classical_baselines.csv", index=False)

    order = [
        "Zero displacement",
        "Constant velocity",
        "Ridge trajectory",
        "Ridge trajectory + flow",
        "MLP trajectory + flow",
        "GRU self trajectory",
        "GRU self + learned flow",
        "Ours: 3-layer radial MP",
    ]
    palette = {
        "Zero displacement": C["gray"],
        "Constant velocity": C["gray"],
        "Ridge trajectory": C["amber"],
        "Ridge trajectory + flow": C["amber"],
        "MLP trajectory + flow": C["green"],
        "GRU self trajectory": C["cyan"],
        "GRU self + learned flow": C["cyan"],
        "Ours: 3-layer radial MP": C["red"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=False)
    for ax, ds in zip(axes, ["MDCK Edge", "MDCK Bulk"]):
        part = out[out.dataset.eq(ds)].copy()
        part["model"] = pd.Categorical(part["model"], categories=order, ordered=True)
        part = part.sort_values("model")
        y = np.arange(len(part))
        ax.barh(y, part["rmse_px"], color=[palette[m] for m in part["model"]])
        ax.set_yticks(y, part["model"], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(ds, fontsize=17, weight="bold")
        ax.set_xlabel("RMSE px, lower is better")
        ax.grid(axis="x", color="#25334E", alpha=0.8)
        for i, (_, row) in enumerate(part.iterrows()):
            ax.text(row.rmse_px + 0.08, i, f"{row.rmse_px:.2f}", va="center", fontsize=9, color=C["text"])
    fig.suptitle("Classical baselines vs current architecture", fontsize=20, weight="bold")
    fig.savefig(args.figure_dir / "classical_baselines_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=False)
    for ax, ds in zip(axes, ["MDCK Edge", "MDCK Bulk"]):
        part = out[out.dataset.eq(ds)].copy()
        part["model"] = pd.Categorical(part["model"], categories=order, ordered=True)
        part = part.sort_values("model")
        y = np.arange(len(part))
        ax.barh(y, part["r2_vec"], color=[palette[m] for m in part["model"]])
        ax.set_yticks(y, part["model"], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(ds, fontsize=17, weight="bold")
        ax.set_xlabel("Vector R2, higher is better")
        ax.grid(axis="x", color="#25334E", alpha=0.8)
        for i, (_, row) in enumerate(part.iterrows()):
            ax.text(row.r2_vec + 0.008, i, f"{row.r2_vec:.3f}", va="center", fontsize=9, color=C["text"])
    fig.suptitle("Classical baselines vs current architecture: explained displacement variance", fontsize=20, weight="bold")
    fig.savefig(args.figure_dir / "classical_baselines_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
