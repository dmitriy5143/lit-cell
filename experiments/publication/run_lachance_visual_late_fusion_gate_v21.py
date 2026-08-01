#!/usr/bin/env python3
"""Video late-fusion gate v21 for LaChance route experts.

This runner tests the video branch in the place where it is most likely to be
useful: after the coordinate route experts have produced stable route outputs.
Visual tokens are not merged into the generator/backbone.  They are distilled
into a route-prior/reliability signal, bridged to the current coordinate route
space using train-only labels, and then tested with strict controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402


DEFAULT_FEATURES = (
    ROOT
    / "outputs"
    / "seg_tracking_foundation_v20_paired_downstream_bulk_seed42_2026-07-04"
    / "raw_context_v2_paired_visual_keys.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "visual_late_fusion_gate_v21_bulk_seed42_2026-07-04"
DEFAULT_VIDEO_CACHE = ROOT / "outputs" / "video_track_tokens_paired_smoke_bulk_seed42_2026-07-04" / "video_track_tokens.npz"
DEFAULT_VISUAL_CONTROLS = ROOT / "outputs" / "v20_visual_route_prior_fusion_real_controls_bulk_seed42_2026-07-04"
KEY_COLS = ["dataset", "sequence", "frame", "track_id"]
EPS = 1e-8


def parse_strs(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(s) for s in parse_strs(text)]


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def split_keys(split: audit.seq.SplitData, name: str) -> pd.DataFrame:
    df = getattr(split, name)
    out = df[KEY_COLS].copy()
    out["dataset"] = out["dataset"].astype(str)
    out["sequence"] = out["sequence"].astype(int)
    out["frame"] = out["frame"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    return out.reset_index(drop=True)


def load_video_label_frame(cache: Path) -> pd.DataFrame:
    npz = np.load(cache, allow_pickle=True)
    rows = []
    for split in ["train", "val", "test"]:
        rows.append(
            pd.DataFrame(
                {
                    "dataset": npz[f"{split}_key_dataset"].astype(str),
                    "sequence": npz[f"{split}_key_sequence"].astype(int),
                    "frame": npz[f"{split}_key_frame"].astype(int),
                    "track_id": npz[f"{split}_key_track_id"].astype(int),
                    "split": split,
                    "video_route": npz[f"{split}_route"].astype(int),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def fit_video_to_coord_bridge(
    *,
    split: audit.seq.SplitData,
    coord_labels: Any,
    video_cache: Path,
    smooth: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    video = load_video_label_frame(video_cache)
    label_rows = []
    for name in ["train", "val", "test"]:
        keys = split_keys(split, name)
        keys["coord_route"] = getattr(coord_labels, name).astype(int)
        keys["split"] = name
        label_rows.append(keys)
    labels = pd.concat(label_rows, ignore_index=True)
    merged = labels.merge(video, on=KEY_COLS + ["split"], how="left")
    if merged["video_route"].isna().any():
        missing = int(merged["video_route"].isna().sum())
        raise RuntimeError(f"Video cache does not cover {missing} split rows")
    vk = int(merged["video_route"].max() + 1)
    ck = int(coord_labels.k)
    bridge = np.full((vk, ck), float(smooth), dtype=np.float64)
    tr = merged["split"].eq("train")
    for vr, cr in merged.loc[tr, ["video_route", "coord_route"]].to_numpy(int):
        bridge[int(vr), int(cr)] += 1.0
    bridge /= np.maximum(bridge.sum(axis=1, keepdims=True), EPS)
    train_agree = float(np.mean(merged.loc[tr, "video_route"].to_numpy(int) == merged.loc[tr, "coord_route"].to_numpy(int)))
    meta = {
        "video_route_k": int(vk),
        "coord_route_k": int(ck),
        "train_rows": int(tr.sum()),
        "raw_index_agreement_train": train_agree,
        "bridge_entropy_mean": float(-np.mean(np.sum(bridge * np.log(np.maximum(bridge, EPS)), axis=1))),
    }
    return bridge.astype(np.float32), merged, meta


def load_visual_probs(grid_path: Path, keys: pd.DataFrame, *, expected_k: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grid = pd.read_csv(grid_path)
    missing = [c for c in KEY_COLS if c not in grid.columns]
    if missing:
        raise RuntimeError(f"Visual grid {grid_path} misses key columns: {missing}")
    pcols = sorted([c for c in grid.columns if c.startswith("vroute_p")])
    if not pcols:
        raise RuntimeError(f"No vroute_p* columns in {grid_path}")
    compact_cols = [c for c in grid.columns if c.startswith("vroute_") and c not in pcols]
    use = grid[KEY_COLS + pcols + compact_cols].drop_duplicates(KEY_COLS)
    merged = keys.merge(use, on=KEY_COLS, how="left")
    raw = merged[pcols].to_numpy(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if expected_k is not None and raw.shape[1] != int(expected_k):
        if raw.shape[1] < int(expected_k):
            pad = np.zeros((len(raw), int(expected_k) - raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=1)
        else:
            raw = raw[:, : int(expected_k)]
    compact = merged[compact_cols].to_numpy(np.float32) if compact_cols else np.zeros((len(raw), 0), dtype=np.float32)
    compact = np.nan_to_num(compact, nan=0.0, posinf=0.0, neginf=0.0)
    return raw, compact, pcols


def load_visual_feature_splits(
    grid_path: Path,
    key_tr: pd.DataFrame,
    key_va: pd.DataFrame,
    key_te: pd.DataFrame,
    *,
    prefix: str,
    max_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    grid = pd.read_csv(grid_path)
    missing = [c for c in KEY_COLS if c not in grid.columns]
    if missing:
        raise RuntimeError(f"Visual feature grid {grid_path} misses key columns: {missing}")
    cols = [c for c in grid.columns if c.startswith(prefix) and c not in KEY_COLS + ["split"]]
    if not cols:
        raise RuntimeError(f"No columns with prefix {prefix!r} in {grid_path}")
    use = grid[KEY_COLS + cols].drop_duplicates(KEY_COLS)

    def merge(keys: pd.DataFrame) -> np.ndarray:
        merged = keys.merge(use, on=KEY_COLS, how="left")
        return np.nan_to_num(merged[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    xtr = merge(key_tr)
    xva = merge(key_va)
    xte = merge(key_te)
    if max_cols > 0 and xtr.shape[1] > int(max_cols):
        var = np.nan_to_num(np.var(xtr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        keep = np.argsort(var)[-int(max_cols) :]
        xtr, xva, xte = xtr[:, keep], xva[:, keep], xte[:, keep]
        cols = [cols[int(i)] for i in keep]
    return xtr.astype(np.float32), xva.astype(np.float32), xte.astype(np.float32), cols


def classifier_probs(model: Any, x: np.ndarray, k: int) -> np.ndarray:
    raw = model.predict_proba(x)
    p = np.full((len(x), int(k)), 1e-6, dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, j]
    return normalize_probs(p)


def topk_score(y: np.ndarray, p: np.ndarray, k: int) -> float:
    order = np.argsort(-p, axis=1)[:, : min(int(k), p.shape[1])]
    return float(np.mean([int(y[i]) in set(order[i]) for i in range(len(y))]))


def fit_direct_visual_prior(
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    labels: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model = HistGradientBoostingClassifier(
        max_iter=int(args.v21_direct_hgbdt_iter),
        learning_rate=float(args.v21_direct_hgbdt_lr),
        max_leaf_nodes=int(args.v21_direct_hgbdt_leaf_nodes),
        l2_regularization=float(args.v21_direct_hgbdt_l2),
        random_state=int(args.seed) + 33001,
    )
    model.fit(xtr, labels.train.astype(int))
    ptr = classifier_probs(model, xtr, labels.k)
    pva = classifier_probs(model, xva, labels.k)
    pte = classifier_probs(model, xte, labels.k)
    meta = {
        "direct_feature_dim": int(xtr.shape[1]),
        "direct_route_top1_train": float(np.mean(np.argmax(ptr, axis=1) == labels.train)),
        "direct_route_top3_train": topk_score(labels.train, ptr, 3),
        "direct_route_top1_val": float(np.mean(np.argmax(pva, axis=1) == labels.val)),
        "direct_route_top3_val": topk_score(labels.val, pva, 3),
        "direct_route_top1_test": float(np.mean(np.argmax(pte, axis=1) == labels.test)),
        "direct_route_top3_test": topk_score(labels.test, pte, 3),
    }
    return ptr, pva, pte, meta


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(p.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    s = p.sum(axis=1, keepdims=True)
    bad = s[:, 0] <= EPS
    if np.any(bad):
        p[bad] = 1.0 / float(p.shape[1])
        s = p.sum(axis=1, keepdims=True)
    return (p / np.maximum(s, EPS)).astype(np.float32)


def bridge_probs(raw_video_probs: np.ndarray, bridge: np.ndarray) -> np.ndarray:
    pv = normalize_probs(raw_video_probs)
    return normalize_probs(pv @ bridge)


def combine_geom(coord: np.ndarray, visual: np.ndarray, lam: float) -> np.ndarray:
    lam = float(lam)
    z = (1.0 - lam) * np.log(np.maximum(coord, EPS)) + lam * np.log(np.maximum(visual, EPS))
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return normalize_probs(p)


def disagreement(coord: np.ndarray, visual: np.ndarray) -> np.ndarray:
    coord = normalize_probs(coord)
    visual = normalize_probs(visual)
    diff = coord - visual
    l1 = np.sum(np.abs(diff), axis=1, keepdims=True)
    l2 = np.sqrt(np.sum(diff * diff, axis=1, keepdims=True))
    dot = np.sum(coord * visual, axis=1, keepdims=True)
    cos = dot / np.maximum(np.linalg.norm(coord, axis=1, keepdims=True) * np.linalg.norm(visual, axis=1, keepdims=True), EPS)
    kl_cv = np.sum(coord * (np.log(np.maximum(coord, EPS)) - np.log(np.maximum(visual, EPS))), axis=1, keepdims=True)
    kl_vc = np.sum(visual * (np.log(np.maximum(visual, EPS)) - np.log(np.maximum(coord, EPS))), axis=1, keepdims=True)
    ent_c = -np.sum(coord * np.log(np.maximum(coord, EPS)), axis=1, keepdims=True)
    ent_v = -np.sum(visual * np.log(np.maximum(visual, EPS)), axis=1, keepdims=True)
    return np.concatenate([np.abs(diff), l1, l2, cos, kl_cv, kl_vc, ent_c, ent_v, ent_v - ent_c], axis=1).astype(np.float32)


def endpoint_error_flat(pred: np.ndarray, y: np.ndarray, h: int, max_h: int) -> np.ndarray:
    ps = pred.reshape(len(pred), int(max_h), 2)
    ys = y.reshape(len(y), int(max_h), 2)
    pe = np.sum(ps[:, : int(h)], axis=1)
    ye = np.sum(ys[:, : int(h)], axis=1)
    return np.sqrt(np.sum((pe - ye) ** 2, axis=1)).astype(np.float32)


def fit_ridge_final(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    base_tr: np.ndarray,
    base_va: np.ndarray,
    base_te: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0))
    zva = scaler.transform(np.nan_to_num(xva, nan=0.0, posinf=0.0, neginf=0.0))
    zte = scaler.transform(np.nan_to_num(xte, nan=0.0, posinf=0.0, neginf=0.0))
    best: tuple[float, float, Ridge] | None = None
    target_tr = ytr - base_tr
    for alpha in parse_floats(args.v21_ridge_alphas):
        model = Ridge(alpha=float(alpha))
        model.fit(ztr, target_tr)
        pred_va = base_va + model.predict(zva).astype(np.float32)
        rmse = v16.endpoint_rmse_flat(pred_va, yva, args)
        if best is None or rmse < best[0]:
            best = (rmse, float(alpha), model)
    assert best is not None
    val_rmse, alpha, model = best
    ptr = base_tr + model.predict(ztr).astype(np.float32)
    pva = base_va + model.predict(zva).astype(np.float32)
    pte = base_te + model.predict(zte).astype(np.float32)
    return ptr.astype(np.float32), pva.astype(np.float32), pte.astype(np.float32), {"model": "ridge", "alpha": alpha, "val_endpoint_rmse": float(val_rmse)}


def fit_hgbdt_final(
    *,
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    base_tr: np.ndarray,
    base_va: np.ndarray,
    base_te: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    xtr = np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xva = np.nan_to_num(xva, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xte = np.nan_to_num(xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    target = ytr - base_tr
    ptr = np.zeros_like(ytr, dtype=np.float32)
    pva = np.zeros_like(yva, dtype=np.float32)
    pte = np.zeros_like(base_te, dtype=np.float32)
    for j in range(target.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=int(args.v21_hgbdt_iter),
            learning_rate=float(args.v21_hgbdt_lr),
            max_leaf_nodes=int(args.v21_hgbdt_leaf_nodes),
            l2_regularization=float(args.v21_hgbdt_l2),
            random_state=int(args.seed) + 9100 + j,
        )
        model.fit(xtr, target[:, j])
        ptr[:, j] = model.predict(xtr).astype(np.float32)
        pva[:, j] = model.predict(xva).astype(np.float32)
        pte[:, j] = model.predict(xte).astype(np.float32)
    ptr = base_tr + ptr
    pva = base_va + pva
    pte = base_te + pte
    return ptr, pva, pte, {"model": "hgbdt", "val_endpoint_rmse": float(v16.endpoint_rmse_flat(pva, yva, args))}


def metric_rows(arrays: audit.SplitArrays, pred_flat: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return v16.metric_rows(arrays, v16.flat_to_steps(pred_flat, args), label, args, extra)


def make_coord_features(route_pred: np.ndarray, probs: np.ndarray, x: np.ndarray, mix: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return v16.build_features(route_pred=route_pred, probs=probs, x=x, mix=mix, args=args, kind=args.v21_coord_feature_kind)


def make_visual_features(
    coord_features: np.ndarray,
    coord_probs: np.ndarray,
    visual_probs: np.ndarray,
    visual_raw: np.ndarray,
    visual_compact: np.ndarray,
) -> np.ndarray:
    parts = [
        coord_features.astype(np.float32),
        v16.prior_stats(visual_probs),
        disagreement(coord_probs, visual_probs),
        visual_compact.astype(np.float32),
        visual_raw.astype(np.float32),
    ]
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_gate_blend(
    *,
    coord_pred: tuple[np.ndarray, np.ndarray, np.ndarray],
    visual_pred: tuple[np.ndarray, np.ndarray, np.ndarray],
    gate_features: tuple[np.ndarray, np.ndarray, np.ndarray],
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ctr, cva, cte = coord_pred
    vtr, vva, vte = visual_pred
    ftr, fva, fte = gate_features
    h = max(args.horizons)
    e_coord = endpoint_error_flat(ctr, ytr, h, args.max_horizon)
    e_visual = endpoint_error_flat(vtr, ytr, h, args.max_horizon)
    y_gate = (e_visual + float(args.v21_gate_margin) < e_coord).astype(int)
    if len(np.unique(y_gate)) < 2:
        gtr = np.full((len(ctr), 1), float(np.mean(y_gate)), dtype=np.float32)
        gva = np.full((len(cva), 1), float(np.mean(y_gate)), dtype=np.float32)
        gte = np.full((len(cte), 1), float(np.mean(y_gate)), dtype=np.float32)
        meta = {"gate_model": "constant", "train_visual_better_rate": float(np.mean(y_gate))}
    else:
        scaler = StandardScaler()
        ztr = scaler.fit_transform(ftr)
        zva = scaler.transform(fva)
        zte = scaler.transform(fte)
        clf = LogisticRegression(max_iter=300, C=float(args.v21_gate_c), class_weight="balanced", random_state=int(args.seed) + 22001)
        clf.fit(ztr, y_gate)
        gtr = clf.predict_proba(ztr)[:, 1:2].astype(np.float32)
        gva = clf.predict_proba(zva)[:, 1:2].astype(np.float32)
        gte = clf.predict_proba(zte)[:, 1:2].astype(np.float32)
        meta = {
            "gate_model": "logistic",
            "train_visual_better_rate": float(np.mean(y_gate)),
            "gate_train_mean": float(gtr.mean()),
            "gate_val_mean": float(gva.mean()),
            "gate_test_mean": float(gte.mean()),
        }
    scale = float(args.v21_gate_max)
    gtr = np.clip(gtr * scale, 0.0, scale)
    gva = np.clip(gva * scale, 0.0, scale)
    gte = np.clip(gte * scale, 0.0, scale)
    return (
        (1.0 - gtr) * ctr + gtr * vtr,
        (1.0 - gva) * cva + gva * vva,
        (1.0 - gte) * cte + gte * vte,
        meta,
    )


def visual_specs_from_args(args: argparse.Namespace) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for item in parse_strs(args.visual_route_prior_specs):
        if "=" not in item:
            raise RuntimeError(f"Bad visual spec {item!r}; expected name=path")
        name, path = item.split("=", 1)
        specs[name.strip()] = Path(path.strip())
    return specs


def visual_feature_specs_from_args(args: argparse.Namespace) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    if not str(args.visual_feature_specs).strip():
        return specs
    for item in parse_strs(args.visual_feature_specs):
        if "=" not in item:
            raise RuntimeError(f"Bad visual feature spec {item!r}; expected name=path")
        name, path = item.split("=", 1)
        specs[name.strip()] = Path(path.strip())
    return specs


def add_v16_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="")
    parser.add_argument("--extra-feature-block-name", type=str, default="none")
    parser.add_argument("--extra-feature-max-cols", type=int, default=0)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--v10-velocity-max-cols", type=int, default=160)
    parser.add_argument("--v12-route-k", type=int, default=12)
    parser.add_argument("--v12-min-route-cluster-size", type=int, default=40)
    parser.add_argument("--v12-prior-model", type=str, default="logistic", choices=["logistic", "hgbdt"])
    parser.add_argument("--v12-prior-max-iter", type=int, default=500)
    parser.add_argument("--v12-prior-c", type=float, default=0.35)
    parser.add_argument("--v12-hgbdt-iter", type=int, default=160)
    parser.add_argument("--v12-hgbdt-lr", type=float, default=0.05)
    parser.add_argument("--v12-hgbdt-leaf-nodes", type=int, default=31)
    parser.add_argument("--v12-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v12-max-route-features", type=int, default=768)
    parser.add_argument("--v12-include-decomposition", action="store_true")
    parser.add_argument("--v12-expert-alpha", type=float, default=300.0)
    parser.add_argument("--v12-min-expert-samples", type=int, default=80)
    parser.add_argument("--v12-error-pool-max", type=int, default=2500)
    parser.add_argument("--v12-top-route-modes", type=int, default=4)
    parser.add_argument("--v12-route-prob-power", type=float, default=1.5)
    parser.add_argument("--v12-error-noise-scale", type=float, default=0.75)
    parser.add_argument("--v12-noise-jitter", type=float, default=0.02)
    parser.add_argument("--v16c-generator-variant", type=str, default="context_velocity")
    parser.add_argument("--v16c-top-c", type=int, default=8)
    parser.add_argument("--v16c-max-context-features", type=int, default=384)
    parser.add_argument("--v16c-base-mixes", type=str, default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    parser.add_argument("--v16c-calibrators", type=str, default="correction_context,stacked_top_context,stacked_context")
    parser.add_argument("--v16c-ridge-alphas", type=str, default="0.1,0.3,1,3,10,30,100,300,1000,3000")


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    args.horizons = audit.parse_ints(args.horizons) if isinstance(args.horizons, str) else list(args.horizons)
    args.oracle_k = audit.parse_ints(args.oracle_k) if isinstance(args.oracle_k, str) else list(args.oracle_k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.extra_feature_grid = None
    device = closure.device_from_arg(args.device)

    arrays, split = audit.prepare_data(args)
    arrays2, labels, prior, bank, packs, gate, meta = v16.build_route_data(args, device)
    if len(arrays.residual_test) != len(arrays2.residual_test):
        raise RuntimeError("Internal split mismatch between v21 and v16 route data")
    arrays = arrays2

    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    yte = audit.flatten_residual(arrays.residual_test).astype(np.float32)
    rtr = v16.route_outputs(bank, prior.x_train)
    rva = v16.route_outputs(bank, prior.x_val)
    rte = v16.route_outputs(bank, prior.x_test)
    xtr, xva, xte = v16.select_context(prior.x_train, prior.x_val, prior.x_test, args.v16c_max_context_features)

    bridge, bridge_labels, bridge_meta = fit_video_to_coord_bridge(split=split, coord_labels=labels, video_cache=args.video_label_cache, smooth=args.v21_bridge_smooth)
    key_tr, key_va, key_te = split_keys(split, "train"), split_keys(split, "val"), split_keys(split, "test")

    rows: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    rows.extend(v16.metric_rows(arrays, audit.flatten_residual(np.zeros_like(arrays.residual_test)).reshape(len(arrays.residual_test), args.max_horizon, 2), "base_self_flow", args, {"stage": "baseline"}))

    base_name = str(args.v21_base_mix)
    base_top = int(args.v21_base_top_c)
    base_mode = str(args.v21_base_mode)
    base_power = float(args.v21_base_power)
    mtr = v16.mix_route_outputs(rtr, prior.probs_train, base_top, base_mode, base_power)
    mva = v16.mix_route_outputs(rva, prior.probs_val, base_top, base_mode, base_power)
    mte = v16.mix_route_outputs(rte, prior.probs_test, base_top, base_mode, base_power)
    rows.extend(metric_rows(arrays, mte, f"coord_{base_name}", args, {"stage": "coord_fixed_mix"}))

    ftr_coord = make_coord_features(rtr, prior.probs_train, xtr, mtr, args)
    fva_coord = make_coord_features(rva, prior.probs_val, xva, mva, args)
    fte_coord = make_coord_features(rte, prior.probs_test, xte, mte, args)
    coord_pred: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {"fixed": (mtr, mva, mte)}

    ctr, cva, cte, cmeta = fit_ridge_final(xtr=ftr_coord, xva=fva_coord, xte=fte_coord, ytr=ytr, yva=yva, base_tr=mtr, base_va=mva, base_te=mte, args=args)
    coord_pred["ridge"] = (ctr, cva, cte)
    rows.extend(metric_rows(arrays, cte, f"coord_{base_name}_{args.v21_coord_feature_kind}_ridge", args, {"stage": "coord_late_correction", **cmeta}))
    diag.append({"variant": "coord_ridge", "visual": "none", **cmeta})
    if not args.v21_skip_hgbdt:
        htr, hva, hte, hmeta = fit_hgbdt_final(xtr=ftr_coord, xva=fva_coord, xte=fte_coord, ytr=ytr, yva=yva, base_tr=mtr, base_va=mva, base_te=mte, args=args)
        coord_pred["hgbdt"] = (htr, hva, hte)
        rows.extend(metric_rows(arrays, hte, f"coord_{base_name}_{args.v21_coord_feature_kind}_hgbdt", args, {"stage": "coord_late_correction", **hmeta}))
        diag.append({"variant": "coord_hgbdt", "visual": "none", **hmeta})

    specs = visual_specs_from_args(args)
    for visual_name, visual_path in specs.items():
        raw_tr, comp_tr, _ = load_visual_probs(visual_path, key_tr, expected_k=bridge.shape[0])
        raw_va, comp_va, _ = load_visual_probs(visual_path, key_va, expected_k=bridge.shape[0])
        raw_te, comp_te, _ = load_visual_probs(visual_path, key_te, expected_k=bridge.shape[0])
        vp_tr = bridge_probs(raw_tr, bridge)
        vp_va = bridge_probs(raw_va, bridge)
        vp_te = bridge_probs(raw_te, bridge)

        for lam in parse_floats(args.v21_blend_lambdas):
            bp_tr = combine_geom(prior.probs_train, vp_tr, lam)
            bp_va = combine_geom(prior.probs_val, vp_va, lam)
            bp_te = combine_geom(prior.probs_test, vp_te, lam)
            btr = v16.mix_route_outputs(rtr, bp_tr, base_top, base_mode, base_power)
            bva = v16.mix_route_outputs(rva, bp_va, base_top, base_mode, base_power)
            bte = v16.mix_route_outputs(rte, bp_te, base_top, base_mode, base_power)
            rows.extend(metric_rows(arrays, bte, f"v21_{visual_name}_route_prior_blend_lam{lam:g}", args, {"stage": "visual_route_blend", "visual": visual_name, "lambda": lam, "val_endpoint_rmse": v16.endpoint_rmse_flat(bva, yva, args)}))

        vftr = make_visual_features(ftr_coord, prior.probs_train, vp_tr, raw_tr, comp_tr)
        vfva = make_visual_features(fva_coord, prior.probs_val, vp_va, raw_va, comp_va)
        vfte = make_visual_features(fte_coord, prior.probs_test, vp_te, raw_te, comp_te)
        vtr, vva, vte, vmeta = fit_ridge_final(xtr=vftr, xva=vfva, xte=vfte, ytr=ytr, yva=yva, base_tr=mtr, base_va=mva, base_te=mte, args=args)
        rows.extend(metric_rows(arrays, vte, f"v21_{visual_name}_ridge_late_correction", args, {"stage": "visual_late_correction", "visual": visual_name, **vmeta}))
        diag.append({"variant": "visual_ridge", "visual": visual_name, **vmeta})
        visual_best = (vtr, vva, vte)

        if not args.v21_skip_hgbdt:
            vhtr, vhva, vhte, vhmeta = fit_hgbdt_final(xtr=vftr, xva=vfva, xte=vfte, ytr=ytr, yva=yva, base_tr=mtr, base_va=mva, base_te=mte, args=args)
            rows.extend(metric_rows(arrays, vhte, f"v21_{visual_name}_hgbdt_late_correction", args, {"stage": "visual_late_correction", "visual": visual_name, **vhmeta}))
            diag.append({"variant": "visual_hgbdt", "visual": visual_name, **vhmeta})
            if vhmeta["val_endpoint_rmse"] < vmeta["val_endpoint_rmse"]:
                visual_best = (vhtr, vhva, vhte)

        for coord_kind, cpred in coord_pred.items():
            gtr, gva, gte, gmeta = fit_gate_blend(coord_pred=cpred, visual_pred=visual_best, gate_features=(vftr, vfva, vfte), ytr=ytr, yva=yva, yte=yte, args=args)
            rows.extend(metric_rows(arrays, gte, f"v21_{visual_name}_gate_{coord_kind}_vs_visual", args, {"stage": "visual_reliability_gate", "visual": visual_name, "coord_kind": coord_kind, **gmeta}))
            gate_rows.append({"visual": visual_name, "coord_kind": coord_kind, **gmeta, "val_endpoint_rmse": float(v16.endpoint_rmse_flat(gva, yva, args))})

    direct_specs = visual_feature_specs_from_args(args)
    for visual_name, visual_path in direct_specs.items():
        dname = f"direct_{visual_name}"
        vxtr, vxva, vxte, vcols = load_visual_feature_splits(
            visual_path,
            key_tr,
            key_va,
            key_te,
            prefix=args.visual_feature_prefix,
            max_cols=args.visual_feature_max_cols,
        )
        vp_tr, vp_va, vp_te, pmeta = fit_direct_visual_prior(vxtr, vxva, vxte, labels, args)
        raw_tr, raw_va, raw_te = vp_tr, vp_va, vp_te
        comp_tr, comp_va, comp_te = v16.prior_stats(vp_tr), v16.prior_stats(vp_va), v16.prior_stats(vp_te)
        diag.append({"variant": "direct_visual_prior", "visual": dname, "model": "hgbdt_classifier", "val_endpoint_rmse": np.nan, "feature_dim": len(vcols), **pmeta})

        for lam in parse_floats(args.v21_blend_lambdas):
            bp_tr = combine_geom(prior.probs_train, vp_tr, lam)
            bp_va = combine_geom(prior.probs_val, vp_va, lam)
            bp_te = combine_geom(prior.probs_test, vp_te, lam)
            btr = v16.mix_route_outputs(rtr, bp_tr, base_top, base_mode, base_power)
            bva = v16.mix_route_outputs(rva, bp_va, base_top, base_mode, base_power)
            bte = v16.mix_route_outputs(rte, bp_te, base_top, base_mode, base_power)
            rows.extend(metric_rows(arrays, bte, f"v21_{dname}_route_prior_blend_lam{lam:g}", args, {"stage": "direct_visual_route_blend", "visual": dname, "lambda": lam, "val_endpoint_rmse": v16.endpoint_rmse_flat(bva, yva, args), **pmeta}))

        vftr = make_visual_features(ftr_coord, prior.probs_train, vp_tr, raw_tr, comp_tr)
        vfva = make_visual_features(fva_coord, prior.probs_val, vp_va, raw_va, comp_va)
        vfte = make_visual_features(fte_coord, prior.probs_test, vp_te, raw_te, comp_te)
        vtr, vva, vte, vmeta = fit_ridge_final(xtr=vftr, xva=vfva, xte=vfte, ytr=ytr, yva=yva, base_tr=mtr, base_va=mva, base_te=mte, args=args)
        rows.extend(metric_rows(arrays, vte, f"v21_{dname}_ridge_late_correction", args, {"stage": "direct_visual_late_correction", "visual": dname, **vmeta, **pmeta}))
        diag.append({"variant": "direct_visual_ridge", "visual": dname, **vmeta, **pmeta})
        visual_best = (vtr, vva, vte)

        if not args.v21_skip_hgbdt:
            vhtr, vhva, vhte, vhmeta = fit_hgbdt_final(xtr=vftr, xva=vfva, xte=vfte, ytr=ytr, yva=yva, base_tr=mtr, base_va=mva, base_te=mte, args=args)
            rows.extend(metric_rows(arrays, vhte, f"v21_{dname}_hgbdt_late_correction", args, {"stage": "direct_visual_late_correction", "visual": dname, **vhmeta, **pmeta}))
            diag.append({"variant": "direct_visual_hgbdt", "visual": dname, **vhmeta, **pmeta})
            if vhmeta["val_endpoint_rmse"] < vmeta["val_endpoint_rmse"]:
                visual_best = (vhtr, vhva, vhte)

        for coord_kind, cpred in coord_pred.items():
            gtr, gva, gte, gmeta = fit_gate_blend(coord_pred=cpred, visual_pred=visual_best, gate_features=(vftr, vfva, vfte), ytr=ytr, yva=yva, yte=yte, args=args)
            rows.extend(metric_rows(arrays, gte, f"v21_{dname}_gate_{coord_kind}_vs_visual", args, {"stage": "direct_visual_reliability_gate", "visual": dname, "coord_kind": coord_kind, **gmeta, **pmeta}))
            gate_rows.append({"visual": dname, "coord_kind": coord_kind, **gmeta, **pmeta, "val_endpoint_rmse": float(v16.endpoint_rmse_flat(gva, yva, args))})

    summary = pd.DataFrame(rows)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    diag_df = pd.DataFrame(diag)
    if not diag_df.empty:
        diag_df.insert(0, "seed", int(args.seed))
        diag_df.insert(0, "dataset", str(args.dataset))
    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        gate_df.insert(0, "seed", int(args.seed))
        gate_df.insert(0, "dataset", str(args.dataset))
    bridge_labels.to_csv(args.out_dir / "visual_late_fusion_v21_bridge_labels.csv", index=False)
    summary.to_csv(args.out_dir / "visual_late_fusion_v21_summary.csv", index=False)
    diag_df.to_csv(args.out_dir / "visual_late_fusion_v21_diagnostics.csv", index=False)
    gate_df.to_csv(args.out_dir / "visual_late_fusion_v21_gate.csv", index=False)
    meta_out = {
        "v16_meta": meta,
        "bridge": bridge_meta,
        "visual_specs": {k: str(v) for k, v in specs.items()},
        "direct_visual_feature_specs": {k: str(v) for k, v in direct_specs.items()},
    }
    (args.out_dir / "visual_late_fusion_v21_meta.json").write_text(json.dumps(finite_json(meta_out), indent=2), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag_df, gate_df, bridge_meta)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diag_rows": len(diag_df), "gate_rows": len(gate_df)}, indent=2), flush=True)


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, gate: pd.DataFrame, bridge_meta: dict[str, Any]) -> None:
    lines = ["# v21 Visual Late-Fusion Gate", ""]
    lines.append("## Bridge")
    lines.append("```json")
    lines.append(json.dumps(finite_json(bridge_meta), indent=2))
    lines.append("```")
    lines.append("")
    for h in args.horizons:
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        cols = [c for c in ["method", "rmse", "r2", "stage", "visual", "lambda", "model", "alpha", "val_endpoint_rmse", "gate_model", "gate_test_mean"] if c in sub.columns]
        lines.append(f"## h{h}")
        lines.append(sub[cols].head(80).to_markdown(index=False))
        lines.append("")
    if not diag.empty:
        lines.append("## Diagnostics")
        lines.append(diag.sort_values("val_endpoint_rmse").head(80).to_markdown(index=False))
        lines.append("")
    if not gate.empty:
        lines.append("## Reliability Gate")
        lines.append(gate.sort_values("val_endpoint_rmse").head(80).to_markdown(index=False))
        lines.append("")
    best_h6 = summary[summary["horizon"].eq(max(args.horizons))].sort_values("rmse").head(12)
    lines.append("## Interpretation")
    if not best_h6.empty:
        best = best_h6.iloc[0]
        lines.append(f"- Best h{max(args.horizons)} method: `{best['method']}` RMSE `{best['rmse']:.4f}`.")
        real = best_h6[best_h6["method"].astype(str).str.contains("real", case=False, regex=False)]
        control = best_h6[best_h6["method"].astype(str).str.contains("shuffled|wrong|zero|time", case=False, regex=True)]
        if not real.empty and not control.empty:
            lines.append("- Real/control comparison must be read from the ranked tables above; passing requires real-video to beat all strict controls, not just the coordinate baseline.")
    (out_dir / "visual_late_fusion_v21_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(features=DEFAULT_FEATURES, out_dir=DEFAULT_OUT, max_train_rows=0, max_val_rows=0, max_test_rows=0)
    add_v16_args(parser)
    parser.add_argument("--video-label-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    default_specs = ",".join(
        [
            f"real={DEFAULT_VISUAL_CONTROLS / 'vroute_prior_real_feature_grid.csv'}",
            f"zero={DEFAULT_VISUAL_CONTROLS / 'vroute_prior_zero_feature_grid.csv'}",
            f"shuffled={DEFAULT_VISUAL_CONTROLS / 'vroute_prior_shuffled_feature_grid.csv'}",
            f"wrong_cell={DEFAULT_VISUAL_CONTROLS / 'vroute_prior_wrong_cell_feature_grid.csv'}",
            f"same_frame_wrong_cell={DEFAULT_VISUAL_CONTROLS / 'vroute_prior_same_frame_wrong_cell_feature_grid.csv'}",
            f"time_shuffled={DEFAULT_VISUAL_CONTROLS / 'vroute_prior_time_shuffled_feature_grid.csv'}",
        ]
    )
    parser.add_argument("--visual-route-prior-specs", type=str, default=default_specs)
    parser.add_argument(
        "--visual-feature-specs",
        type=str,
        default="",
        help="Optional name=path specs for raw visual feature grids; a direct visual->current-route prior is trained inside this runner.",
    )
    parser.add_argument("--visual-feature-prefix", type=str, default="segf_")
    parser.add_argument("--visual-feature-max-cols", type=int, default=128)
    parser.add_argument("--v21-base-mix", type=str, default="expert_top8_uniform")
    parser.add_argument("--v21-base-top-c", type=int, default=8)
    parser.add_argument("--v21-base-mode", type=str, default="uniform", choices=["uniform", "prior", "all_uniform"])
    parser.add_argument("--v21-base-power", type=float, default=1.0)
    parser.add_argument("--v21-coord-feature-kind", type=str, default="stacked_top_context", choices=["correction_context", "stacked_top_context", "stacked_context"])
    parser.add_argument("--v21-blend-lambdas", type=str, default="0,0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--v21-ridge-alphas", type=str, default="0.1,0.3,1,3,10,30,100,300,1000,3000,10000")
    parser.add_argument("--v21-bridge-smooth", type=float, default=0.5)
    parser.add_argument("--v21-skip-hgbdt", action="store_true")
    parser.add_argument("--v21-hgbdt-iter", type=int, default=120)
    parser.add_argument("--v21-hgbdt-lr", type=float, default=0.04)
    parser.add_argument("--v21-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--v21-hgbdt-l2", type=float, default=0.05)
    parser.add_argument("--v21-direct-hgbdt-iter", type=int, default=180)
    parser.add_argument("--v21-direct-hgbdt-lr", type=float, default=0.05)
    parser.add_argument("--v21-direct-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--v21-direct-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v21-gate-c", type=float, default=0.5)
    parser.add_argument("--v21-gate-max", type=float, default=0.75)
    parser.add_argument("--v21-gate-margin", type=float, default=0.0)
    args = parser.parse_args()
    if args.smoke:
        args.posterior_epochs = min(args.posterior_epochs, 4)
        args.student_epochs = min(args.student_epochs, 4)
        args.candidate_k = min(args.candidate_k, 16)
        args.v21_blend_lambdas = "0,0.1,0.35"
        args.v21_hgbdt_iter = min(args.v21_hgbdt_iter, 60)
        args.v21_direct_hgbdt_iter = min(args.v21_direct_hgbdt_iter, 80)
    run(args)


if __name__ == "__main__":
    main()
