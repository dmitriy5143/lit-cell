#!/usr/bin/env python3
"""Gate object-centric pseudo-mask features against strict controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_sequence_joint_selector_refiner_v7 as v7  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "object_centric_mask_gate_2026-07-03"
DEFAULT_GRID = ROOT / "outputs" / "lachance_object_centric_mask_grid_bulk_seed42_2026-07-03" / "object_centric_mask_feature_grid.csv"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def endpoint_residual_rmse(true: np.ndarray, pred: np.ndarray, horizons: list[int]) -> float:
    errs = []
    for h in horizons:
        p = np.sum(pred[:, : int(h), :], axis=1)
        t = np.sum(true[:, : int(h), :], axis=1)
        errs.append(np.sum((p - t) ** 2, axis=-1))
    return float(np.sqrt(np.mean(np.stack(errs, axis=1))))


def endpoint_direction_cos(true: np.ndarray, pred: np.ndarray, h: int) -> float:
    p = np.sum(pred[:, : int(h), :], axis=1)
    t = np.sum(true[:, : int(h), :], axis=1)
    den = np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1), EPS)
    return float(np.mean(np.sum(p * t, axis=1) / den))


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return audit.safe_matrix(df, cols).astype(np.float32)


def merge_grid(split_df: pd.DataFrame, grid: pd.DataFrame, cols: list[str]) -> np.ndarray:
    keys = ["dataset", "sequence", "frame", "track_id"]
    merged = split_df[keys].merge(grid[keys + cols].drop_duplicates(keys), on=keys, how="left")
    return safe_matrix(merged, cols)


def route_labels(arrays: audit.SplitArrays, k: int, seed: int) -> dict[str, Any]:
    sig_tr = v7.route_signature(arrays.residual_train, [1, 2, 4, 6])
    sig_va = v7.route_signature(arrays.residual_val, [1, 2, 4, 6])
    sig_te = v7.route_signature(arrays.residual_test, [1, 2, 4, 6])
    scaler = StandardScaler()
    ztr = scaler.fit_transform(sig_tr).astype(np.float32)
    zva = scaler.transform(sig_va).astype(np.float32)
    zte = scaler.transform(sig_te).astype(np.float32)
    kk = min(int(k), max(2, len(ztr) // 25))
    km = KMeans(n_clusters=kk, n_init=20, random_state=int(seed) + 41001)
    return {
        "k": kk,
        "train": km.fit_predict(ztr).astype(np.int64),
        "val": km.predict(zva).astype(np.int64),
        "test": km.predict(zte).astype(np.int64),
    }


def same_frame_wrong_cell(x: np.ndarray, df: pd.DataFrame, seed: int) -> np.ndarray:
    out = x.copy()
    rng = np.random.default_rng(seed)
    work = df[["sequence", "frame"]].reset_index(drop=True)
    for _, idx in work.groupby(["sequence", "frame"], sort=False).groups.items():
        arr = np.asarray(list(idx), dtype=np.int64)
        if len(arr) <= 1:
            continue
        perm = rng.permutation(arr)
        if np.any(perm == arr):
            perm = np.roll(perm, 1)
        out[arr] = x[perm]
    return out.astype(np.float32)


def time_shuffle(x: np.ndarray, df: pd.DataFrame, seed: int) -> np.ndarray:
    out = x.copy()
    rng = np.random.default_rng(seed)
    work = df[["sequence", "frame"]].reset_index(drop=True)
    for _, idx in work.groupby("sequence", sort=False).groups.items():
        arr = np.asarray(list(idx), dtype=np.int64)
        order = arr[np.argsort(work.iloc[arr]["frame"].to_numpy())]
        if len(order) > 1:
            shift = int(rng.integers(1, len(order)))
            out[order] = x[np.roll(order, shift)]
    return out.astype(np.float32)


def fit_probe(
    name: str,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    labels: dict[str, Any],
    arrays: audit.SplitArrays,
    horizons: list[int],
    seed: int,
) -> dict[str, Any]:
    xtr, xva, xte, _ = v10.standardize_2d(xtr, xva, xte)
    k = int(labels["k"])
    clf = LogisticRegression(max_iter=500, C=0.35, class_weight="balanced", random_state=int(seed) + 42001)
    clf.fit(xtr, labels["train"])
    proba = clf.predict_proba(xte)
    full = np.full((len(xte), k), 1e-6, dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        full[:, int(cls)] = proba[:, j]
    full /= np.maximum(full.sum(axis=1, keepdims=True), EPS)

    ytr = arrays.residual_train.reshape(len(arrays.residual_train), -1)
    yva = arrays.residual_val.reshape(len(arrays.residual_val), -1)
    yte = arrays.residual_test.reshape(len(arrays.residual_test), -1)
    best_alpha = 0.0
    best_val = float("inf")
    best: Ridge | None = None
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        model = Ridge(alpha=float(alpha))
        model.fit(xtr, ytr)
        pred_va = model.predict(xva).reshape(arrays.residual_val.shape).astype(np.float32)
        score = endpoint_residual_rmse(arrays.residual_val, pred_va, horizons)
        if score < best_val:
            best_val = score
            best_alpha = float(alpha)
            best = model
    assert best is not None
    pred = best.predict(xte).reshape(arrays.residual_test.shape).astype(np.float32)
    return {
        "variant": name,
        "feature_dim": int(xtr.shape[1]),
        "route_top1": float(np.mean(np.argmax(full, axis=1) == labels["test"])),
        "route_top3": float(top_k_accuracy_score(labels["test"], full, k=min(3, k), labels=np.arange(k))),
        "route_nll": float(log_loss(labels["test"], np.clip(full, 1e-6, 1.0), labels=np.arange(k))),
        "residual_endpoint_rmse": endpoint_residual_rmse(arrays.residual_test, pred, horizons),
        "residual_h6_cos": endpoint_direction_cos(arrays.residual_test, pred, max(horizons)),
        "ridge_alpha": best_alpha,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MDCK_Bulk")
    ap.add_argument("--features", type=Path, default=audit.DEFAULT_FEATURES)
    ap.add_argument("--table-root", type=Path, default=audit.seq.ifp.DEFAULT_TABLE_ROOT)
    ap.add_argument("--object-grid", type=Path, default=DEFAULT_GRID)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seq", default="1,2,3,4")
    ap.add_argument("--val-seq", default="5")
    ap.add_argument("--test-seq", default="6")
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--horizons", default="1,2,4,6")
    ap.add_argument("--max-train-rows", type=int, default=3000)
    ap.add_argument("--max-val-rows", type=int, default=1000)
    ap.add_argument("--max-test-rows", type=int, default=1500)
    ap.add_argument("--max-features-per-family", type=int, default=96)
    ap.add_argument("--max-all-features", type=int, default=384)
    ap.add_argument("--route-k", type=int, default=12)
    ap.add_argument("--max-object-cols", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 600)
        args.max_val_rows = min(args.max_val_rows, 200)
        args.max_test_rows = min(args.max_test_rows, 300)
    horizons = parse_ints(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays, split = audit.prepare_data(args)
    grid = pd.read_csv(args.object_grid)
    oc_cols = [c for c in grid.columns if c.startswith("oc_")]
    xtr_obj = merge_grid(split.train, grid, oc_cols)
    xva_obj = merge_grid(split.val, grid, oc_cols)
    xte_obj = merge_grid(split.test, grid, oc_cols)
    if xtr_obj.shape[1] > int(args.max_object_cols):
        var = np.nan_to_num(np.var(xtr_obj, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-int(args.max_object_cols) :]
        oc_cols = [oc_cols[int(i)] for i in keep]
        xtr_obj, xva_obj, xte_obj = xtr_obj[:, keep], xva_obj[:, keep], xte_obj[:, keep]
    velocity_blocks, _ = v10.build_velocity_blocks(split, max_cols=96)
    labels = route_labels(arrays, args.route_k, args.seed)
    sh_obj = (
        xtr_obj[np.random.default_rng(args.seed + 1).permutation(len(xtr_obj))],
        xva_obj[np.random.default_rng(args.seed + 2).permutation(len(xva_obj))],
        xte_obj[np.random.default_rng(args.seed + 3).permutation(len(xte_obj))],
    )
    same_obj = (
        same_frame_wrong_cell(xtr_obj, split.train, args.seed + 11),
        same_frame_wrong_cell(xva_obj, split.val, args.seed + 12),
        same_frame_wrong_cell(xte_obj, split.test, args.seed + 13),
    )
    tshuf_obj = (
        time_shuffle(xtr_obj, split.train, args.seed + 21),
        time_shuffle(xva_obj, split.val, args.seed + 22),
        time_shuffle(xte_obj, split.test, args.seed + 23),
    )
    zero_obj = tuple(np.zeros_like(x) for x in (xtr_obj, xva_obj, xte_obj))
    ctx = (arrays.x_train["all_context"], arrays.x_val["all_context"], arrays.x_test["all_context"])
    vel = velocity_blocks["all"]
    variants = [
        ("context_only", ctx),
        ("velocity_only", vel),
        ("object_only", (xtr_obj, xva_obj, xte_obj)),
        ("object_zero", zero_obj),
        ("object_shuffled", sh_obj),
        ("object_same_frame_wrong_cell", same_obj),
        ("object_time_shuffled", tshuf_obj),
        ("context_plus_object", tuple(np.concatenate([a, b], axis=1) for a, b in zip(ctx, (xtr_obj, xva_obj, xte_obj), strict=False))),
        ("context_plus_object_same_frame_wrong_cell", tuple(np.concatenate([a, b], axis=1) for a, b in zip(ctx, same_obj, strict=False))),
        ("context_plus_velocity", tuple(np.concatenate([a, b], axis=1) for a, b in zip(ctx, vel, strict=False))),
        ("context_plus_velocity_object", tuple(np.concatenate([a, b, c], axis=1) for a, b, c in zip(ctx, vel, (xtr_obj, xva_obj, xte_obj), strict=False))),
        ("context_plus_velocity_object_same_frame_wrong_cell", tuple(np.concatenate([a, b, c], axis=1) for a, b, c in zip(ctx, vel, same_obj, strict=False))),
    ]
    rows = [fit_probe(name, mats[0], mats[1], mats[2], labels, arrays, horizons, args.seed) for name, mats in variants]
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "object_centric_mask_gate_summary.csv", index=False)
    gate = {
        "object_vs_zero_route_top3": float(summary.set_index("variant").loc["object_only", "route_top3"] - summary.set_index("variant").loc["object_zero", "route_top3"]),
        "object_vs_shuffled_route_top3": float(summary.set_index("variant").loc["object_only", "route_top3"] - summary.set_index("variant").loc["object_shuffled", "route_top3"]),
        "object_vs_same_frame_route_top3": float(summary.set_index("variant").loc["object_only", "route_top3"] - summary.set_index("variant").loc["object_same_frame_wrong_cell", "route_top3"]),
        "ctx_vel_obj_vs_ctx_vel_same_frame_rmse_gain": float(
            (summary.set_index("variant").loc["context_plus_velocity_object_same_frame_wrong_cell", "residual_endpoint_rmse"]
             - summary.set_index("variant").loc["context_plus_velocity_object", "residual_endpoint_rmse"])
            / max(abs(summary.set_index("variant").loc["context_plus_velocity_object_same_frame_wrong_cell", "residual_endpoint_rmse"]), EPS)
            * 100.0
        ),
    }
    gate["object_gate_pass"] = bool(gate["object_vs_same_frame_route_top3"] >= 0.03 and gate["ctx_vel_obj_vs_ctx_vel_same_frame_rmse_gain"] >= 1.0)
    pd.DataFrame([gate]).to_csv(args.out_dir / "object_centric_mask_gate.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    lines = [
        "# Object-Centric Mask Gate",
        "",
        f"- grid: `{args.object_grid}`",
        f"- rows train/val/test: `{len(split.train)}/{len(split.val)}/{len(split.test)}`",
        "",
        "## Summary",
        "",
        summary.sort_values("route_top3", ascending=False).to_markdown(index=False),
        "",
        "## Gate",
        "",
        pd.DataFrame([gate]).to_markdown(index=False),
    ]
    (args.out_dir / "object_centric_mask_gate_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "gate": finite_json(gate)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
