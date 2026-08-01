#!/usr/bin/env python3
"""v85: dense causal innovation and target-factorization sweep over v52.

The runner consumes movie-level OOF v52 anchors produced by v84.  It tests
whether the previously missing contiguous cell history contains deployable
signal for the remaining residual:

    v52 anchor
    + observed past one-step innovations of the same cell
    + past innovations of current neighbours / whole tissue
    + optional causal morphology-flow context
    -> bounded h1..h6 residual correction.

All model selection is validation-only.  Training innovations are OOF, and
future targets never enter an inference feature.  Direct, PCA/SVD, DCT and
endpoint target views are evaluated before nonlinear models are admitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_joint_innovation_field_v84 as v84  # noqa: E402


EPS = 1e-8
KEYS = ["dataset", "sequence", "frame", "track_id"]
DEFAULT_OUT = ROOT / "outputs" / "dense_innovation_sweep_v85_2026-07-17"


@dataclass
class Pack:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    columns: list[str]
    control: str = "real"


@dataclass
class TargetView:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    decode: Callable[[np.ndarray], np.ndarray]
    latent_dim: int


def finite(value: Any) -> Any:
    return v84.finite_json(value)


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def safe(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def locate_single(root: Path, pattern: str) -> Path:
    found = [p for p in root.glob(pattern) if p.is_dir()]
    valid = [p for p in found if (p / "test" / "arrays.npz").exists()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one complete {pattern} under {root}, found {valid}")
    return valid[0]


def load_anchor_cache(root: Path) -> tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle]:
    final = locate_single(root, "final_*")
    val = v84.load_bundle(final / "val", "dense_final_val")
    test = v84.load_bundle(final / "test", "dense_final_test")
    oof_parts: list[v84.AnchorBundle] = []
    for path in sorted(root.glob("oof_seq*_*")):
        if (path / "test" / "arrays.npz").exists():
            oof_parts.append(v84.load_bundle(path / "test", f"dense_{path.name}_test"))
    if not oof_parts:
        raise RuntimeError(f"No complete OOF bundles under {root}")
    train = v84.concat_bundles("dense_oof_train", oof_parts)
    contract = val.meta.get("contract", {}) if isinstance(val.meta, dict) else {}
    expected = {int(x) for x in contract.get("train_seq", [1, 2, 3, 4])}
    present = set(int(x) for x in train.rows["sequence"].unique())
    if present != expected:
        raise RuntimeError(f"OOF movies incomplete: expected {expected}, got {present}")
    return train, val, test


def lookup_for(bundle: v84.AnchorBundle) -> dict[tuple[int, int, int], int]:
    return {
        (int(s), int(f), int(t)): i
        for i, (s, f, t) in enumerate(
            bundle.rows[["sequence", "frame", "track_id"]].itertuples(index=False, name=None)
        )
    }


def temporal_history(bundle: v84.AnchorBundle, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(bundle.rows)
    hist = np.zeros((n, lags, 4), dtype=np.float32)
    mask = np.zeros((n, lags), dtype=np.float32)
    lookup = lookup_for(bundle)
    velocity = bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    one_step_error = bundle.errors[:, 0].astype(np.float32)
    keys = bundle.rows[["sequence", "frame", "track_id"]].to_numpy(np.int64)
    for i, (seq, frame, track) in enumerate(keys):
        for lag in range(1, lags + 1):
            j = lookup.get((int(seq), int(frame) - lag, int(track)))
            if j is None:
                continue
            hist[i, lag - 1, :2] = one_step_error[j]
            hist[i, lag - 1, 2:] = velocity[j]
            mask[i, lag - 1] = 1.0
    return hist, mask, velocity


def global_history(bundle: v84.AnchorBundle, hist: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, lags = mask.shape
    out = np.zeros((n, lags * 5), dtype=np.float32)
    for _, idx in bundle.rows.groupby(["sequence", "frame"], sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        for lag in range(lags):
            valid = mask[ids, lag] > 0.5
            if not np.any(valid):
                continue
            e = hist[ids[valid], lag, :2]
            speed = np.linalg.norm(e, axis=1)
            feat = np.asarray([e[:, 0].mean(), e[:, 1].mean(), e[:, 0].std(), e[:, 1].std(), speed.mean()], np.float32)
            out[ids, lag * 5 : (lag + 1) * 5] = feat
    return out


def neighbor_history(
    bundle: v84.AnchorBundle,
    hist: np.ndarray,
    mask: np.ndarray,
    ks: list[int],
) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    for k in ks:
        names.extend(
            [
                f"nei{k}_mean_ex",
                f"nei{k}_mean_ey",
                f"nei{k}_std_ex",
                f"nei{k}_std_ey",
                f"nei{k}_mean_minus_self_x",
                f"nei{k}_mean_minus_self_y",
                f"nei{k}_alignment",
                f"nei{k}_coverage",
            ]
        )
    out = np.zeros((len(bundle.rows), len(names)), dtype=np.float32)
    latest = hist[:, 0, :2]
    latest_mask = mask[:, 0]
    for _, raw_idx in bundle.rows.groupby(["sequence", "frame"], sort=False).groups.items():
        idx = np.asarray(list(raw_idx), dtype=np.int64)
        if len(idx) < 3:
            continue
        xy = bundle.rows.iloc[idx][["x_px", "y_px"]].to_numpy(np.float32)
        tree = cKDTree(xy)
        max_k = min(max(ks) + 1, len(idx))
        _, nn = tree.query(xy, k=max_k)
        if nn.ndim == 1:
            nn = nn[:, None]
        for row_local, row_global in enumerate(idx):
            offset = 0
            for k in ks:
                neighbours = idx[nn[row_local, 1 : min(k + 1, nn.shape[1])]]
                valid = neighbours[latest_mask[neighbours] > 0.5]
                if len(valid):
                    e = latest[valid]
                    mean = e.mean(axis=0)
                    std = e.std(axis=0)
                    self_e = latest[row_global] if latest_mask[row_global] > 0.5 else np.zeros(2, np.float32)
                    align = float(np.dot(mean, self_e) / max(np.linalg.norm(mean) * np.linalg.norm(self_e), EPS))
                    values = [mean[0], mean[1], std[0], std[1], mean[0] - self_e[0], mean[1] - self_e[1], align, len(valid) / max(k, 1)]
                    out[row_global, offset : offset + 8] = values
                offset += 8
    return safe(out), names


def anchor_state(
    bundle: v84.AnchorBundle,
    xy_lo: np.ndarray,
    xy_scale: np.ndarray,
    frame_scale: float,
) -> tuple[np.ndarray, list[str]]:
    anchor = bundle.anchor_steps.reshape(len(bundle.rows), -1)
    cumulative = np.cumsum(bundle.anchor_steps, axis=1).reshape(len(bundle.rows), -1)
    speed = np.linalg.norm(bundle.anchor_steps, axis=2)
    xy = bundle.rows[["x_px", "y_px"]].to_numpy(np.float32)
    xy = (xy - xy_lo) / np.maximum(xy_scale, 1.0)
    frame = bundle.rows[["frame"]].to_numpy(np.float32)
    frame = frame / max(float(frame_scale), 1.0)
    current = bundle.rows[["dx_px", "dy_px"]].to_numpy(np.float32)
    values = np.concatenate([anchor, cumulative, speed, bundle.base, current, xy, frame], axis=1)
    names = (
        [f"anchor_step_{i}" for i in range(anchor.shape[1])]
        + [f"anchor_cumulative_{i}" for i in range(cumulative.shape[1])]
        + [f"anchor_speed_{i}" for i in range(speed.shape[1])]
        + ["base_x", "base_y", "current_vx", "current_vy", "x_norm", "y_norm", "frame_norm"]
    )
    return safe(values), names


def shuffled_like(values: np.ndarray, rows: pd.DataFrame, seed: int, by_movie: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = values.copy()
    if by_movie:
        for _, idx in rows.groupby("sequence", sort=False).groups.items():
            ids = np.asarray(list(idx), dtype=np.int64)
            out[ids] = values[rng.permutation(ids)]
    else:
        out = values[rng.permutation(len(values))]
    return out


def choose_context_columns(path: Path, per_family: int) -> list[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    tokens = (
        "align",
        "orient",
        "elong",
        "centroid",
        "front",
        "back",
        "density",
        "boundary",
        "flow",
        "coherence",
        "curl",
        "diverg",
        "shear",
        "grad",
        "delta",
        "quality",
        "contact",
        "free",
        "polarity",
    )
    selected: list[str] = []
    for prefix in ("ms_", "tf_", "rc_", "obs_"):
        family = [c for c in cols if c.startswith(prefix)]
        preferred = [c for c in family if any(t in c.lower() for t in tokens)]
        fallback = [c for c in family if c not in preferred]
        selected.extend((preferred + fallback)[:per_family])
    return list(dict.fromkeys(selected))


def merge_context(
    path: Path,
    bundles: tuple[v84.AnchorBundle, v84.AnchorBundle, v84.AnchorBundle],
    per_family: int,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    candidates = choose_context_columns(path, per_family)
    usecols = KEYS + candidates
    context = pd.read_csv(path, usecols=lambda c: c in usecols)
    for c in ("sequence", "frame", "track_id"):
        context[c] = context[c].astype(int)
    context = context.drop_duplicates(KEYS)
    arrays: list[np.ndarray] = []
    for bundle in bundles:
        merged = bundle.rows[KEYS].merge(context, on=KEYS, how="left", validate="one_to_one")
        arrays.append(safe(merged[candidates].to_numpy(np.float32)))
    var = np.var(arrays[0], axis=0)
    keep = np.argsort(var)[::-1][: min(max_features, len(candidates))]
    return arrays[0][:, keep], arrays[1][:, keep], arrays[2][:, keep], [candidates[i] for i in keep]


def build_packs(
    train: v84.AnchorBundle,
    val: v84.AnchorBundle,
    test: v84.AnchorBundle,
    args: argparse.Namespace,
) -> tuple[dict[str, Pack], pd.DataFrame]:
    lags = int(args.history_lags)
    ks = parse_ints(args.neighbor_ks)
    bundles = (train, val, test)
    train_xy = train.rows[["x_px", "y_px"]].to_numpy(np.float32)
    xy_lo = np.percentile(train_xy, 1, axis=0)
    xy_hi = np.percentile(train_xy, 99, axis=0)
    states = [anchor_state(b, xy_lo, xy_hi - xy_lo, float(train.rows["frame"].max())) for b in bundles]
    histories = [temporal_history(b, lags) for b in bundles]
    globals_ = [global_history(b, h, m) for b, (h, m, _v) in zip(bundles, histories)]
    neighbours = [neighbor_history(b, h, m, ks) for b, (h, m, _v) in zip(bundles, histories)]

    state_arrays = [x[0] for x in states]
    state_names = states[0][1]
    hist_arrays = [np.concatenate([h.reshape(len(b.rows), -1), m], axis=1) for b, (h, m, _v) in zip(bundles, histories)]
    hist_names = [f"history_{i}" for i in range(hist_arrays[0].shape[1])]
    nei_arrays = [x[0] for x in neighbours]
    nei_names = neighbours[0][1]
    global_names = [f"global_history_{i}" for i in range(globals_[0].shape[1])]

    context = merge_context(args.features, bundles, int(args.context_per_family), int(args.context_max_features))
    context_arrays = list(context[:3])
    context_names = context[3]

    def make(name: str, blocks: list[list[np.ndarray]], names: list[list[str]], control: str = "real") -> Pack:
        split_values = [np.concatenate([block[s] for block in blocks], axis=1) for s in range(3)]
        return Pack(name, split_values[0], split_values[1], split_values[2], sum(names, []), control)

    packs: dict[str, Pack] = {}
    S = state_arrays
    H = hist_arrays
    N = nei_arrays
    G = globals_
    C = context_arrays
    packs["anchor_state"] = make("anchor_state", [S], [state_names])
    packs["anchor_self_history"] = make("anchor_self_history", [S, H], [state_names, hist_names])
    packs["anchor_neighbor_history"] = make("anchor_neighbor_history", [S, N], [state_names, nei_names])
    packs["anchor_global_history"] = make("anchor_global_history", [S, G], [state_names, global_names])
    packs["anchor_all_history"] = make("anchor_all_history", [S, H, N, G], [state_names, hist_names, nei_names, global_names])
    packs["anchor_all_history_context"] = make(
        "anchor_all_history_context", [S, H, N, G, C], [state_names, hist_names, nei_names, global_names, context_names]
    )

    shuffled_h = [shuffled_like(H[i], bundles[i].rows, args.seed + 100 + i) for i in range(3)]
    shuffled_n = [shuffled_like(N[i], bundles[i].rows, args.seed + 200 + i) for i in range(3)]
    shuffled_g = [shuffled_like(G[i], bundles[i].rows, args.seed + 250 + i) for i in range(3)]
    shuffled_c = [shuffled_like(C[i], bundles[i].rows, args.seed + 300 + i) for i in range(3)]
    packs["anchor_self_history_shuffled"] = make(
        "anchor_self_history_shuffled", [S, shuffled_h], [state_names, hist_names], "shuffled_self"
    )
    packs["anchor_neighbor_history_shuffled"] = make(
        "anchor_neighbor_history_shuffled", [S, shuffled_n], [state_names, nei_names], "shuffled_neighbor"
    )
    packs["anchor_all_history_shuffled"] = make(
        "anchor_all_history_shuffled",
        [S, shuffled_h, shuffled_n, shuffled_g],
        [state_names, hist_names, nei_names, global_names],
        "shuffled_all_history",
    )
    packs["anchor_all_history_context_shuffled"] = make(
        "anchor_all_history_context_shuffled",
        [S, H, N, G, shuffled_c],
        [state_names, hist_names, nei_names, global_names, context_names],
        "shuffled_context",
    )

    coverage_rows: list[dict[str, Any]] = []
    for split, bundle, (_h, mask, _v) in zip(("train", "val", "test"), bundles, histories):
        row: dict[str, Any] = {"split": split, "rows": len(bundle.rows)}
        for lag in range(lags):
            row[f"lag{lag + 1}_coverage"] = float(mask[:, lag].mean())
        coverage_rows.append(row)
    return packs, pd.DataFrame(coverage_rows)


def make_target_views(train: v84.AnchorBundle, val: v84.AnchorBundle, test: v84.AnchorBundle) -> dict[str, TargetView]:
    tr_steps, va_steps, te_steps = train.errors, val.errors, test.errors
    direct = [x.reshape(len(x), -1).astype(np.float32) for x in (tr_steps, va_steps, te_steps)]
    views: dict[str, TargetView] = {
        "direct": TargetView("direct", *direct, decode=lambda x: safe(x).reshape(len(x), 6, 2), latent_dim=12)
    }

    y_scaler = StandardScaler().fit(direct[0])
    ytr_std = y_scaler.transform(direct[0])
    for rank in (4, 8):
        pca = PCA(n_components=rank, random_state=42).fit(ytr_std)
        encoded = [pca.transform(y_scaler.transform(x)).astype(np.float32) for x in direct]

        def decode_pca(z: np.ndarray, pca: PCA = pca, scaler: StandardScaler = y_scaler) -> np.ndarray:
            flat = scaler.inverse_transform(pca.inverse_transform(z))
            return safe(flat).reshape(len(flat), 6, 2)

        views[f"pca_r{rank}"] = TargetView(f"pca_r{rank}", *encoded, decode=decode_pca, latent_dim=rank)

    for rank in (2, 4):
        encoded = [dct(x, axis=1, norm="ortho")[:, :rank, :].reshape(len(x), -1).astype(np.float32) for x in (tr_steps, va_steps, te_steps)]

        def decode_dct(z: np.ndarray, rank: int = rank) -> np.ndarray:
            coeff = np.zeros((len(z), 6, 2), dtype=np.float32)
            coeff[:, :rank] = z.reshape(len(z), rank, 2)
            return safe(idct(coeff, axis=1, norm="ortho"))

        views[f"dct_r{rank}"] = TargetView(f"dct_r{rank}", *encoded, decode=decode_dct, latent_dim=2 * rank)

    endpoint_idx = [0, 1, 3, 5]
    endpoints = [np.cumsum(x, axis=1)[:, endpoint_idx].reshape(len(x), -1).astype(np.float32) for x in (tr_steps, va_steps, te_steps)]

    def decode_endpoints(z: np.ndarray) -> np.ndarray:
        e = z.reshape(len(z), 4, 2)
        out = np.zeros((len(z), 6, 2), dtype=np.float32)
        out[:, 0] = e[:, 0]
        out[:, 1] = e[:, 1] - e[:, 0]
        out[:, 2:4] = (e[:, 2] - e[:, 1])[:, None, :] / 2.0
        out[:, 4:6] = (e[:, 3] - e[:, 2])[:, None, :] / 2.0
        return safe(out)

    views["endpoint_1246"] = TargetView("endpoint_1246", *endpoints, decode=decode_endpoints, latent_dim=8)
    return views


def standardize_x(pack: Pack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(pack.train)
    return tuple(np.clip(scaler.transform(x), -8.0, 8.0).astype(np.float32) for x in (pack.train, pack.val, pack.test))  # type: ignore[return-value]


def fit_ridge(
    pack: Pack,
    target: TargetView,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xtr, xva, xte = standardize_x(pack)
    sy = StandardScaler().fit(target.train)
    ytr = sy.transform(target.train)
    best: tuple[float, Ridge] | None = None
    best_alpha = 0.0
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(xtr, ytr)
        pred = sy.inverse_transform(model.predict(xva))
        score = float(np.sqrt(np.mean(np.square(pred - target.val))))
        if best is None or score < best[0]:
            best = (score, model)
            best_alpha = alpha
    assert best is not None
    pva = target.decode(sy.inverse_transform(best[1].predict(xva)).astype(np.float32))
    pte = target.decode(sy.inverse_transform(best[1].predict(xte)).astype(np.float32))
    return pva, pte, {"alpha": best_alpha, "latent_val_rmse": best[0]}


def fit_hgbdt(
    pack: Pack,
    target: TargetView,
    configs: list[tuple[int, int, float]],
    jobs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xtr, xva, xte = standardize_x(pack)
    best: tuple[float, MultiOutputRegressor] | None = None
    best_cfg: tuple[int, int, float] | None = None
    for max_iter, leaves, l2 in configs:
        base = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.05,
            max_leaf_nodes=leaves,
            min_samples_leaf=30,
            l2_regularization=l2,
            random_state=seed,
        )
        model = MultiOutputRegressor(base, n_jobs=jobs)
        model.fit(xtr, target.train)
        pred = model.predict(xva)
        score = float(np.sqrt(np.mean(np.square(pred - target.val))))
        if best is None or score < best[0]:
            best = (score, model)
            best_cfg = (max_iter, leaves, l2)
    assert best is not None and best_cfg is not None
    return target.decode(best[1].predict(xva)), target.decode(best[1].predict(xte)), {
        "max_iter": best_cfg[0], "leaves": best_cfg[1], "l2": best_cfg[2], "latent_val_rmse": best[0]
    }


def fit_extra_trees(
    pack: Pack,
    target: TargetView,
    trees: int,
    jobs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = ExtraTreesRegressor(
        n_estimators=trees,
        max_features=0.7,
        min_samples_leaf=12,
        max_depth=24,
        n_jobs=jobs,
        random_state=seed,
    )
    model.fit(pack.train, target.train)
    pva = model.predict(pack.val)
    pte = model.predict(pack.test)
    return target.decode(pva), target.decode(pte), {"trees": trees, "latent_val_rmse": float(np.sqrt(np.mean(np.square(pva - target.val))))}


def evaluate_prediction(
    model_name: str,
    pack: Pack,
    target: TargetView,
    val_corr: np.ndarray,
    test_corr: np.ndarray,
    val: v84.AnchorBundle,
    test: v84.AnchorBundle,
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eta, val_score = v84.tune_eta(val, val_corr, parse_ints(args.horizons), parse_floats(args.eta_grid))
    pred = test.anchor_steps + eta * test_corr
    method = f"v85_{model_name}_{pack.name}_{target.name}"
    extra = {"model": model_name, "pack": pack.name, "target_view": target.name, "control": pack.control, "eta": eta, "val_score": val_score, **meta}
    rows = v84.metric_rows(test, pred, parse_ints(args.horizons), method, extra)
    diag = {"method": method, "feature_dim": pack.train.shape[1], "latent_dim": target.latent_dim, **extra}
    return rows, diag


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_anchor_cache(args.anchor_cache)
    packs, coverage = build_packs(train, val, test, args)
    targets = make_target_views(train, val, test)
    metrics: list[dict[str, Any]] = v84.metric_rows(test, test.anchor_steps, parse_ints(args.horizons), "v85_v52_anchor", {"model": "anchor", "pack": "anchor", "target_view": "none", "control": "real"})
    diagnostics: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    nonlinear_predictions: dict[str, np.ndarray] = {}

    ridge_alphas = parse_floats(args.ridge_alphas)
    for pack in packs.values():
        for target in targets.values():
            pva, pte, meta = fit_ridge(pack, target, ridge_alphas)
            rows, diag = evaluate_prediction("ridge", pack, target, pva, pte, val, test, args, meta)
            metrics.extend(rows)
            diagnostics.append(diag)
            ranking.append({"pack": pack.name, "target": target.name, "val_score": diag["val_score"], "control": pack.control})

    rank_df = pd.DataFrame(ranking).sort_values("val_score")
    eligible = rank_df[rank_df["control"].eq("real")].drop_duplicates(["pack", "target"]).head(int(args.nonlinear_top))
    forced_rows: list[dict[str, Any]] = []
    for spec in [x.strip() for x in str(args.forced_nonlinear_pairs).split(",") if x.strip()]:
        if ":" not in spec:
            raise ValueError(f"Bad forced nonlinear pair {spec!r}; expected pack:target")
        pack_name, target_name = spec.split(":", 1)
        if pack_name not in packs or target_name not in targets:
            raise KeyError(f"Unknown forced pair {pack_name}:{target_name}")
        forced_rows.append({"pack": pack_name, "target": target_name, "val_score": math.inf, "control": packs[pack_name].control})
    if forced_rows:
        eligible = pd.concat([eligible, pd.DataFrame(forced_rows)], ignore_index=True).drop_duplicates(["pack", "target"])
    hgbdt_configs = [(int(args.hgbdt_iter), 15, 10.0), (int(args.hgbdt_iter), 31, 30.0)]
    for rank, row in eligible.reset_index(drop=True).iterrows():
        pack = packs[str(row["pack"])]
        target = targets[str(row["target"])]
        pva, pte, meta = fit_hgbdt(pack, target, hgbdt_configs, int(args.jobs), int(args.seed) + rank)
        rows, diag = evaluate_prediction("hgbdt", pack, target, pva, pte, val, test, args, meta)
        metrics.extend(rows)
        diagnostics.append(diag)
        pred_key = f"hgbdt__{pack.name}__{target.name}"
        nonlinear_predictions[pred_key + "__val"] = safe(pva)
        nonlinear_predictions[pred_key + "__test"] = safe(pte)
        if rank < int(args.extra_trees_top):
            pva, pte, meta = fit_extra_trees(pack, target, int(args.extra_trees), int(args.jobs), int(args.seed) + 100 + rank)
            rows, diag = evaluate_prediction("extra_trees", pack, target, pva, pte, val, test, args, meta)
            metrics.extend(rows)
            diagnostics.append(diag)
            pred_key = f"extra_trees__{pack.name}__{target.name}"
            nonlinear_predictions[pred_key + "__val"] = safe(pva)
            nonlinear_predictions[pred_key + "__test"] = safe(pte)

    metric_df = pd.DataFrame(metrics)
    diag_df = pd.DataFrame(diagnostics)
    metric_df.to_csv(args.out_dir / "v85_dense_innovation_metrics.csv", index=False)
    diag_df.to_csv(args.out_dir / "v85_dense_innovation_diagnostics.csv", index=False)
    coverage.to_csv(args.out_dir / "v85_history_coverage.csv", index=False)
    rank_df.to_csv(args.out_dir / "v85_linear_triage.csv", index=False)
    if nonlinear_predictions:
        np.savez_compressed(args.out_dir / "v85_nonlinear_predictions.npz", **nonlinear_predictions)
    pd.DataFrame(
        [{"train_rows": len(train.rows), "val_rows": len(val.rows), "test_rows": len(test.rows), "train_movies": sorted(train.rows.sequence.unique().tolist()), "elapsed_sec": time.time() - started}]
    ).to_csv(args.out_dir / "v85_data_contract.csv", index=False)

    hmax = max(parse_ints(args.horizons))
    h = metric_df[metric_df.horizon.eq(hmax)].sort_values("component_rmse")
    anchor = float(h[h.method.eq("v85_v52_anchor")].iloc[0].component_rmse)
    best = h.iloc[0]
    gain = (anchor - float(best.component_rmse)) / anchor * 100.0
    lines = [
        "# v85 Dense Innovation Sweep",
        "",
        "## Data Contract",
        "",
        coverage.to_markdown(index=False),
        "",
        f"OOF train rows: `{len(train.rows)}`; validation: `{len(val.rows)}`; test: `{len(test.rows)}`.",
        "",
        f"## h{hmax} Ranking",
        "",
        h[[c for c in ["method", "component_rmse", "vector_rmse", "r2", "cosine", "magnitude_ratio", "model", "pack", "target_view", "control", "eta", "val_score"] if c in h.columns]].head(35).to_markdown(index=False),
        "",
        "## Decision",
        "",
        f"Same-row v52 anchor component RMSE: `{anchor:.6f}`.",
        f"Best validation-selected test result: `{float(best.component_rmse):.6f}` (`{gain:.3f}%`).",
    ]
    real_hist = h[h.pack.astype(str).str.contains("history") & h.control.eq("real")]
    controls = h[h.control.ne("real")]
    if gain >= 3.0 and not real_hist.empty:
        lines.append("- Hard gate passed: dense causal innovation history justifies the graph-temporal probabilistic stage.")
    elif gain >= 1.0:
        lines.append("- Soft gate passed: history is useful but not yet a breakthrough; continue with graph-temporal conditioning and strict controls.")
    else:
        lines.append("- Dense history did not materially improve v52 in tabular/target-factorized models.")
    if not controls.empty:
        lines.append(f"- Best control h{hmax} component RMSE: `{float(controls.component_rmse.min()):.6f}`.")
    lines.append(f"- Elapsed: `{(time.time() - started) / 3600.0:.2f} h`.")
    (args.out_dir / "v85_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(finite(vars(args)), indent=2), encoding="utf-8")
    print((args.out_dir / "v85_decision_report.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-cache", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--horizons", default="1,2,4,6")
    p.add_argument("--history-lags", type=int, default=6)
    p.add_argument("--neighbor-ks", default="8,32")
    p.add_argument("--context-per-family", type=int, default=96)
    p.add_argument("--context-max-features", type=int, default=160)
    p.add_argument("--ridge-alphas", default="1,10,30,100,300,1000,3000,10000")
    p.add_argument("--eta-grid", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1,1.25")
    p.add_argument("--nonlinear-top", type=int, default=8)
    p.add_argument(
        "--forced-nonlinear-pairs",
        default="",
        help="Comma-separated pack:target pairs evaluated with HGBDT regardless of linear ranking.",
    )
    p.add_argument("--hgbdt-iter", type=int, default=180)
    p.add_argument("--extra-trees-top", type=int, default=3)
    p.add_argument("--extra-trees", type=int, default=192)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_sec": time.time() - started}
        (args.out_dir / "v85_error.json").write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
