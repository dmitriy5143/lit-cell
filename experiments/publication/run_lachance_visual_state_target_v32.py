#!/usr/bin/env python3
"""v32 dense visual-state + observable target reformulation sweep.

This runner tests the next serious route after v31:

1. Build explicit time-aligned visual-state variables:
   mask shape, polarity/front-back, contact/free-space, reliability and
   central-cell identity/history signals.
2. Compare real visual state against hard controls:
   zero, row-shuffled, same-frame wrong cell and time-shuffled.
3. Test target formulations that should be more observable than endpoint
   coordinates:
   direct residual steps, residual endpoints, parallel/perp velocities,
   speed/turn and flow-relative residuals.

The final metric remains coordinate h1/h2/h4/h6 RMSE/R2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, top_k_accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_dense_state_target_reformulation_sweep_v25 as v25  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_observability_gap_diagnostic_v27 as v27  # noqa: E402


DEFAULT_FEATURES = audit.DEFAULT_FEATURES
DEFAULT_OUT = ROOT / "outputs" / "visual_state_target_v32_2026-07-06"
KEY_COLS = ["dataset", "sequence", "frame", "track_id"]
EPS = 1e-8


@dataclass
class Packet:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    feature_names: list[str]
    control: str = "real"
    family: str = "unknown"
    coverage_train: float = 1.0
    coverage_val: float = 1.0
    coverage_test: float = 1.0


def parse_csv(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in parse_csv(text)]


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return audit.safe_matrix(df, cols).astype(np.float32, copy=False)


def standardize_x(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xtr.shape[1] == 0:
        return xtr, xva, xte
    sc = StandardScaler()
    ztr = sc.fit_transform(xtr)
    zva = sc.transform(xva)
    zte = sc.transform(xte)
    return (
        np.clip(np.nan_to_num(ztr), -8, 8).astype(np.float32),
        np.clip(np.nan_to_num(zva), -8, 8).astype(np.float32),
        np.clip(np.nan_to_num(zte), -8, 8).astype(np.float32),
    )


def select_by_variance(df: pd.DataFrame, cols: list[str], max_cols: int) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return []
    if max_cols <= 0 or len(cols) <= max_cols:
        return cols
    x = safe_matrix(df, cols)
    var = np.nan_to_num(np.var(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-var)[: int(max_cols)]
    return [cols[int(i)] for i in order]


def endpoint_rmse_from_residual(arrays: audit.SplitArrays, residual: np.ndarray, horizon: int) -> float:
    pred_steps = arrays.base_test[:, None, :] + residual
    y = audit.endpoint_from_steps(arrays.steps_test, horizon)
    yp = audit.endpoint_from_steps(pred_steps, horizon)
    return audit.rmse(y, yp)


def endpoint_rmse_val(arrays: audit.SplitArrays, residual: np.ndarray, horizon: int) -> float:
    pred_steps = arrays.base_val[:, None, :] + residual
    y = audit.endpoint_from_steps(arrays.steps_val, horizon)
    yp = audit.endpoint_from_steps(pred_steps, horizon)
    return audit.rmse(y, yp)


def vector_basis(base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.linalg.norm(base, axis=1, keepdims=True)
    e = base / np.maximum(speed, 1e-6)
    fallback = speed[:, 0] < 1e-6
    if np.any(fallback):
        e[fallback] = np.array([1.0, 0.0], dtype=np.float32)
    p = np.stack([-e[:, 1], e[:, 0]], axis=1).astype(np.float32)
    return e.astype(np.float32), p, speed[:, 0].astype(np.float32)


def flow_base(df: pd.DataFrame, radius: int) -> np.ndarray | None:
    cols = [f"tf_r{radius}_cur_center_u", f"tf_r{radius}_cur_center_v"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].fillna(0.0).to_numpy(np.float32)


def endpoint_target(residual: np.ndarray, base: np.ndarray, horizons: list[int]) -> np.ndarray:
    # Residual endpoint at each selected horizon.
    vals = []
    for h in horizons:
        vals.append(np.sum(residual[:, :h, :], axis=1))
    return np.concatenate(vals, axis=1).astype(np.float32)


def reconstruct_endpoint_residual(pred: np.ndarray, max_h: int, horizons: list[int]) -> np.ndarray:
    n = len(pred)
    ep = {h: pred[:, 2 * i : 2 * i + 2] for i, h in enumerate(horizons)}
    # Ensure h1/h2/h4/h6 are available; fall back to nearest known endpoint.
    def get(h: int) -> np.ndarray:
        if h in ep:
            return ep[h]
        smaller = [k for k in ep if k < h]
        larger = [k for k in ep if k > h]
        if smaller and larger:
            lo, hi = max(smaller), min(larger)
            t = (h - lo) / max(hi - lo, 1)
            return (1 - t) * ep[lo] + t * ep[hi]
        if smaller:
            return ep[max(smaller)] * (h / max(smaller))
        return ep[min(larger)] * (h / min(larger))

    e1, e2, e4, e6 = get(1), get(2), get(4), get(min(6, max_h))
    out = np.zeros((n, max_h, 2), dtype=np.float32)
    out[:, 0, :] = e1
    if max_h >= 2:
        out[:, 1, :] = e2 - e1
    if max_h >= 4:
        out[:, 2:4, :] = (e4 - e2)[:, None, :] / 2.0
    if max_h >= 6:
        out[:, 4:6, :] = (e6 - e4)[:, None, :] / 2.0
    if max_h > 6:
        out[:, 6:, :] = out[:, 5:6, :]
    return out.astype(np.float32)


def target_parallel_perp(steps: np.ndarray, base: np.ndarray) -> np.ndarray:
    e, p, _ = vector_basis(base)
    par = np.sum(steps * e[:, None, :], axis=2)
    perp = np.sum(steps * p[:, None, :], axis=2)
    return np.stack([par, perp], axis=2).reshape(len(steps), -1).astype(np.float32)


def reconstruct_parallel_perp(pred: np.ndarray, base: np.ndarray, max_h: int) -> np.ndarray:
    e, p, _ = vector_basis(base)
    comp = pred.reshape(len(pred), max_h, 2).astype(np.float32)
    steps = comp[:, :, 0:1] * e[:, None, :] + comp[:, :, 1:2] * p[:, None, :]
    return steps.astype(np.float32)


def target_speed_turn(steps: np.ndarray, base: np.ndarray) -> np.ndarray:
    e, p, _ = vector_basis(base)
    speed = np.linalg.norm(steps, axis=2)
    u = steps / np.maximum(speed[:, :, None], 1e-6)
    cos = np.sum(u * e[:, None, :], axis=2)
    sin = np.sum(u * p[:, None, :], axis=2)
    log_speed = np.log1p(speed)
    return np.stack([log_speed, cos, sin], axis=2).reshape(len(steps), -1).astype(np.float32)


def reconstruct_speed_turn(pred: np.ndarray, base: np.ndarray, max_h: int) -> np.ndarray:
    e, p, _ = vector_basis(base)
    comp = pred.reshape(len(pred), max_h, 3).astype(np.float32)
    speed = np.expm1(np.clip(comp[:, :, 0], -4.0, 5.0))
    cs = comp[:, :, 1:3]
    norm = np.maximum(np.linalg.norm(cs, axis=2, keepdims=True), 1e-6)
    cs = cs / norm
    direction = cs[:, :, 0:1] * e[:, None, :] + cs[:, :, 1:2] * p[:, None, :]
    return (np.maximum(speed, 0.0)[:, :, None] * direction).astype(np.float32)


def target_flow_residual(steps: np.ndarray, flow: np.ndarray) -> np.ndarray:
    return (steps - flow[:, None, :]).reshape(len(steps), -1).astype(np.float32)


def reconstruct_flow_residual(pred: np.ndarray, flow: np.ndarray, base: np.ndarray, max_h: int) -> np.ndarray:
    full_steps = flow[:, None, :] + pred.reshape(len(pred), max_h, 2)
    return (full_steps - base[:, None, :]).astype(np.float32)


def fit_ridge_multi(
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    *,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xtr, xva, xte = standardize_x(xtr, xva, xte)
    ysc = StandardScaler()
    ytr_s = ysc.fit_transform(ytr)
    best_alpha = alphas[0]
    best = float("inf")
    best_model: Ridge | None = None
    for a in alphas:
        model = Ridge(alpha=float(a))
        model.fit(xtr, ytr_s)
        pva = ysc.inverse_transform(model.predict(xva))
        score = float(np.sqrt(np.mean((pva - yva) ** 2)))
        if score < best:
            best = score
            best_alpha = float(a)
            best_model = model
    assert best_model is not None
    ptr = ysc.inverse_transform(best_model.predict(xtr)).astype(np.float32)
    pva = ysc.inverse_transform(best_model.predict(xva)).astype(np.float32)
    pte = ysc.inverse_transform(best_model.predict(xte)).astype(np.float32)
    return ptr, pva, pte, best_alpha, best


def fit_hgbdt_multi(
    xtr: np.ndarray,
    xva: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xtr, xva, xte = standardize_x(xtr, xva, xte)
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=int(max_iter),
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=int(seed),
        ),
        n_jobs=1,
    )
    model.fit(xtr, ytr)
    ptr = model.predict(xtr).astype(np.float32)
    pva = model.predict(xva).astype(np.float32)
    pte = model.predict(xte).astype(np.float32)
    score = float(np.sqrt(np.mean((pva - yva) ** 2)))
    return ptr, pva, pte, float("nan"), score


def identity_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    out = pd.DataFrame(index=df.index)
    speed = np.sqrt(np.square(df["dx_px"].fillna(0.0)) + np.square(df["dy_px"].fillna(0.0))).astype(float)
    out["id_speed"] = speed
    out["id_quality"] = df["QUALITY"].fillna(0.0).astype(float) if "QUALITY" in df.columns else 0.0
    for (_, _), g in df.assign(_speed=speed).groupby(["sequence", "track_id"], sort=False):
        idx = g.sort_values("frame").index
        gg = g.loc[idx]
        s = gg["_speed"].astype(float)
        out.loc[idx, "id_track_age"] = np.arange(len(idx), dtype=float)
        out.loc[idx, "id_speed_roll3_mean"] = s.rolling(3, min_periods=1).mean().to_numpy()
        out.loc[idx, "id_speed_roll3_std"] = s.rolling(3, min_periods=1).std().fillna(0.0).to_numpy()
        out.loc[idx, "id_speed_roll6_mean"] = s.rolling(6, min_periods=1).mean().to_numpy()
        out.loc[idx, "id_speed_roll6_std"] = s.rolling(6, min_periods=1).std().fillna(0.0).to_numpy()
        out.loc[idx, "id_dx_roll6_mean"] = gg["dx_px"].fillna(0.0).rolling(6, min_periods=1).mean().to_numpy()
        out.loc[idx, "id_dy_roll6_mean"] = gg["dy_px"].fillna(0.0).rolling(6, min_periods=1).mean().to_numpy()
        if "QUALITY" in gg.columns:
            out.loc[idx, "id_quality_roll6_mean"] = gg["QUALITY"].fillna(0.0).rolling(6, min_periods=1).mean().to_numpy()
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    cols = list(out.columns)
    return out.to_numpy(np.float32), cols


def merge_grid(
    grid_path: Path,
    prefix: str,
    split: audit.seq.SplitData,
    *,
    include_tokens: list[str],
    exclude_tokens: list[str],
    max_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    grid = pd.read_csv(grid_path)
    cols = [c for c in grid.columns if c.startswith(prefix) and c not in KEY_COLS + ["split"]]
    if include_tokens:
        toks = [t.lower() for t in include_tokens]
        cols = [c for c in cols if any(t in c.lower() for t in toks)]
    if exclude_tokens:
        toks = [t.lower() for t in exclude_tokens]
        cols = [c for c in cols if not any(t in c.lower() for t in toks)]
    cols = select_by_variance(grid, cols, max_cols)
    use = grid[KEY_COLS + cols].drop_duplicates(KEY_COLS)

    def one(df: pd.DataFrame) -> tuple[np.ndarray, float]:
        merged = df[KEY_COLS].reset_index(drop=True).merge(use, on=KEY_COLS, how="left", indicator=True)
        cov = float(np.mean(merged["_merge"].eq("both"))) if len(merged) else 0.0
        return safe_matrix(merged, cols), cov

    xtr, ctr = one(split.train)
    xva, cva = one(split.val)
    xte, cte = one(split.test)
    meta = {"path": str(grid_path), "prefix": prefix, "n_cols": len(cols), "coverage_train": ctr, "coverage_val": cva, "coverage_test": cte}
    return xtr, xva, xte, cols, meta


def make_controls(packet: Packet, split: audit.seq.SplitData, seed: int) -> dict[str, Packet]:
    controls = {"real": packet}
    mats = (packet.train, packet.val, packet.test)
    for ctrl in ["zero", "row_shuffled", "same_frame_wrong_cell", "time_shuffled"]:
        xtr, xva, xte = v25.make_derived_control(mats, split, ctrl, seed + 101)
        controls[ctrl] = Packet(
            name=f"{packet.name}_{ctrl}",
            train=xtr,
            val=xva,
            test=xte,
            feature_names=packet.feature_names,
            control=ctrl,
            family=packet.family,
            coverage_train=packet.coverage_train,
            coverage_val=packet.coverage_val,
            coverage_test=packet.coverage_test,
        )
    return controls


def concat_packet(name: str, parts: list[Packet], control: str = "real", family: str = "concat") -> Packet:
    return Packet(
        name=name,
        train=np.concatenate([p.train for p in parts if p.train.shape[1] > 0], axis=1).astype(np.float32),
        val=np.concatenate([p.val for p in parts if p.val.shape[1] > 0], axis=1).astype(np.float32),
        test=np.concatenate([p.test for p in parts if p.test.shape[1] > 0], axis=1).astype(np.float32),
        feature_names=sum((p.feature_names for p in parts), []),
        control=control,
        family=family,
        coverage_train=min((p.coverage_train for p in parts), default=1.0),
        coverage_val=min((p.coverage_val for p in parts), default=1.0),
        coverage_test=min((p.coverage_test for p in parts), default=1.0),
    )


def add_interactions(packet: Packet, split: audit.seq.SplitData, max_cols: int) -> Packet:
    # Directional visual state should be modulated by current speed/quality.
    n = min(packet.train.shape[1], int(max_cols))
    if n <= 0:
        return packet
    tr_id, id_cols = identity_features(split.train)
    va_id, _ = identity_features(split.val)
    te_id, _ = identity_features(split.test)
    speed_i = id_cols.index("id_speed")
    qual_i = id_cols.index("id_quality") if "id_quality" in id_cols else speed_i
    tr_mod = np.concatenate([packet.train[:, :n] * tr_id[:, speed_i : speed_i + 1], packet.train[:, :n] * tr_id[:, qual_i : qual_i + 1]], axis=1)
    va_mod = np.concatenate([packet.val[:, :n] * va_id[:, speed_i : speed_i + 1], packet.val[:, :n] * va_id[:, qual_i : qual_i + 1]], axis=1)
    te_mod = np.concatenate([packet.test[:, :n] * te_id[:, speed_i : speed_i + 1], packet.test[:, :n] * te_id[:, qual_i : qual_i + 1]], axis=1)
    names = [f"{c}*id_speed" for c in packet.feature_names[:n]] + [f"{c}*id_quality" for c in packet.feature_names[:n]]
    return Packet(
        name=f"{packet.name}_interactions",
        train=np.concatenate([packet.train, tr_mod], axis=1).astype(np.float32),
        val=np.concatenate([packet.val, va_mod], axis=1).astype(np.float32),
        test=np.concatenate([packet.test, te_mod], axis=1).astype(np.float32),
        feature_names=packet.feature_names + names,
        control=packet.control,
        family=packet.family,
        coverage_train=packet.coverage_train,
        coverage_val=packet.coverage_val,
        coverage_test=packet.coverage_test,
    )


def build_packets(args: argparse.Namespace, arrays: audit.SplitArrays, split: audit.seq.SplitData) -> dict[str, Packet]:
    packets: dict[str, Packet] = {}
    coord = Packet(
        name="coord_all_context",
        train=arrays.x_train["all_context"],
        val=arrays.x_val["all_context"],
        test=arrays.x_test["all_context"],
        feature_names=arrays.feature_names["all_context"],
        family="coordinate",
    )
    packets[coord.name] = coord
    id_tr, id_cols = identity_features(split.train)
    id_va, _ = identity_features(split.val)
    id_te, _ = identity_features(split.test)
    idp = Packet("identity_history", id_tr, id_va, id_te, id_cols, family="identity")
    packets[idp.name] = idp

    visual_parts: list[Packet] = [idp]

    semantic_tokens = parse_csv(args.visual_tokens)
    embed_tokens = []
    for name, path, prefix, toks, max_cols in [
        ("object_state", args.object_grid, "oc_", semantic_tokens, args.max_object_cols),
        ("temporal_mask_state", args.temporal_grid, "mi_", semantic_tokens, args.max_temporal_cols),
        ("multiseed_mask_state", args.multiseed_grid, "mi_", semantic_tokens, args.max_multiseed_cols),
        ("seg_foundation_state", args.seg_foundation_grid, "segf_", embed_tokens, args.max_seg_cols),
    ]:
        if not Path(path).exists():
            continue
        try:
            xtr, xva, xte, cols, meta = merge_grid(Path(path), prefix, split, include_tokens=toks, exclude_tokens=[], max_cols=int(max_cols))
        except Exception as exc:
            print(f"[warn] failed to load {name}: {exc}")
            continue
        p = Packet(
            name=name,
            train=xtr,
            val=xva,
            test=xte,
            feature_names=cols,
            family=name,
            coverage_train=meta["coverage_train"],
            coverage_val=meta["coverage_val"],
            coverage_test=meta["coverage_test"],
        )
        packets[p.name] = p
        packets[f"coord_plus_{name}"] = concat_packet(f"coord_plus_{name}", [coord, p], family=f"coord_{name}")
        p_int = add_interactions(p, split, args.max_interaction_cols)
        packets[f"{name}_interaction"] = p_int
        packets[f"coord_plus_{name}_interaction"] = concat_packet(
            f"coord_plus_{name}_interaction",
            [coord, p_int],
            family=f"coord_{name}",
        )
        if name in {"object_state", "seg_foundation_state"}:
            for ctrl, cp in make_controls(p, split, int(args.seed) + (17 if name == "object_state" else 31)).items():
                if ctrl == "real":
                    continue
                packets[f"coord_plus_{name}_{ctrl}"] = concat_packet(
                    f"coord_plus_{name}_{ctrl}",
                    [coord, cp],
                    control=ctrl,
                    family=f"coord_{name}_control",
                )
        visual_parts.append(p)

    visual = concat_packet("visual_state_explicit", visual_parts, family="visual_state")
    packets[visual.name] = visual
    packets["visual_state_interaction"] = add_interactions(visual, split, args.max_interaction_cols)
    packets["coord_plus_visual_state"] = concat_packet("coord_plus_visual_state", [coord, visual], family="coord_visual")
    packets["coord_plus_visual_interaction"] = concat_packet("coord_plus_visual_interaction", [coord, packets["visual_state_interaction"]], family="coord_visual")

    # Hard visual controls: keep coordinate context real, corrupt only visual state.
    for ctrl, vp in make_controls(visual, split, int(args.seed)).items():
        if ctrl == "real":
            continue
        packets[f"coord_plus_visual_{ctrl}"] = concat_packet(f"coord_plus_visual_{ctrl}", [coord, vp], control=ctrl, family="coord_visual_control")
    return packets


def route_probe(packet: Packet, basis: v26.RouteBasis, args: argparse.Namespace) -> dict[str, Any]:
    xtr, xva, xte = standardize_x(packet.train, packet.val, packet.test)
    k = int(basis.route_train.shape[1])
    clf = LogisticRegression(max_iter=700, C=0.45, class_weight="balanced", random_state=int(args.seed) + 3201)
    clf.fit(xtr, basis.oracle_labels_train)
    raw = clf.predict_proba(xte)
    proba = np.full((len(xte), k), 1e-7, dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        proba[:, int(cls)] = raw[:, j]
    proba /= np.maximum(proba.sum(axis=1, keepdims=True), EPS)
    try:
        nll = float(log_loss(basis.oracle_labels_test, np.clip(proba, 1e-7, 1.0), labels=np.arange(k)))
    except Exception:
        nll = float("nan")
    return {
        "packet": packet.name,
        "family": packet.family,
        "control": packet.control,
        "feature_dim": int(packet.train.shape[1]),
        "coverage_test": packet.coverage_test,
        "route_top1": float(np.mean(np.argmax(proba, axis=1) == basis.oracle_labels_test)),
        "route_top3": float(top_k_accuracy_score(basis.oracle_labels_test, proba, k=min(3, k), labels=np.arange(k))),
        "route_nll": nll,
    }


def formulation_targets(
    name: str,
    arrays: audit.SplitArrays,
    split: audit.seq.SplitData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray, str], np.ndarray]]:
    max_h = arrays.residual_train.shape[1]
    if name == "direct_residual_steps":
        ytr = arrays.residual_train.reshape(len(arrays.residual_train), -1)
        yva = arrays.residual_val.reshape(len(arrays.residual_val), -1)
        yte = arrays.residual_test.reshape(len(arrays.residual_test), -1)

        def rec(pred: np.ndarray, split_name: str) -> np.ndarray:
            return pred.reshape(len(pred), max_h, 2).astype(np.float32)

        return ytr, yva, yte, rec
    if name == "endpoint_residual_h1246":
        horizons = [1, 2, 4, 6]
        ytr = endpoint_target(arrays.residual_train, arrays.base_train, horizons)
        yva = endpoint_target(arrays.residual_val, arrays.base_val, horizons)
        yte = endpoint_target(arrays.residual_test, arrays.base_test, horizons)

        def rec(pred: np.ndarray, split_name: str) -> np.ndarray:
            return reconstruct_endpoint_residual(pred, max_h, horizons)

        return ytr, yva, yte, rec
    if name == "parallel_perp_full_steps":
        ytr = target_parallel_perp(arrays.steps_train, arrays.base_train)
        yva = target_parallel_perp(arrays.steps_val, arrays.base_val)
        yte = target_parallel_perp(arrays.steps_test, arrays.base_test)

        def rec(pred: np.ndarray, split_name: str) -> np.ndarray:
            base = {"train": arrays.base_train, "val": arrays.base_val, "test": arrays.base_test}[split_name]
            return reconstruct_parallel_perp(pred, base, max_h) - base[:, None, :]

        return ytr, yva, yte, rec
    if name == "speed_turn_full_steps":
        ytr = target_speed_turn(arrays.steps_train, arrays.base_train)
        yva = target_speed_turn(arrays.steps_val, arrays.base_val)
        yte = target_speed_turn(arrays.steps_test, arrays.base_test)

        def rec(pred: np.ndarray, split_name: str) -> np.ndarray:
            base = {"train": arrays.base_train, "val": arrays.base_val, "test": arrays.base_test}[split_name]
            return reconstruct_speed_turn(pred, base, max_h) - base[:, None, :]

        return ytr, yva, yte, rec
    if name.startswith("flow_relative_r"):
        radius = int(name.split("r")[-1])
        ftr = flow_base(split.train, radius)
        fva = flow_base(split.val, radius)
        fte = flow_base(split.test, radius)
        if ftr is None or fva is None or fte is None:
            raise ValueError(f"flow radius {radius} unavailable")
        ytr = target_flow_residual(arrays.steps_train, ftr)
        yva = target_flow_residual(arrays.steps_val, fva)
        yte = target_flow_residual(arrays.steps_test, fte)

        def rec(pred: np.ndarray, split_name: str) -> np.ndarray:
            flow = {"train": ftr, "val": fva, "test": fte}[split_name]
            base = {"train": arrays.base_train, "val": arrays.base_val, "test": arrays.base_test}[split_name]
            return reconstruct_flow_residual(pred, flow, base, max_h)

        return ytr, yva, yte, rec
    raise ValueError(name)


def evaluate_prediction(label: str, residual_test: np.ndarray, arrays: audit.SplitArrays, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=residual_test,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def run_formulation(
    packet: Packet,
    formulation: str,
    model_kind: str,
    arrays: audit.SplitArrays,
    split: audit.seq.SplitData,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    ytr, yva, _yte, rec = formulation_targets(formulation, arrays, split)
    if model_kind == "ridge":
        ptr, pva, pte, alpha, y_val_score = fit_ridge_multi(packet.train, packet.val, packet.test, ytr, yva, alphas=parse_floats(args.ridge_alphas))
    elif model_kind == "hgbdt":
        ptr, pva, pte, alpha, y_val_score = fit_hgbdt_multi(packet.train, packet.val, packet.test, ytr, yva, seed=int(args.seed) + 9001, max_iter=int(args.hgbdt_iter))
    else:
        raise ValueError(model_kind)
    rtr = rec(ptr, "train")
    rva = rec(pva, "val")
    rte = rec(pte, "test")
    hmax = max(args.horizons)
    val_endpoint = endpoint_rmse_val(arrays, rva, hmax)
    rows = evaluate_prediction(
        f"v32_{packet.name}_{formulation}_{model_kind}",
        rte,
        arrays,
        args,
        {
            "stage": "v32_target_reformulation",
            "packet": packet.name,
            "family": packet.family,
            "control": packet.control,
            "formulation": formulation,
            "model": model_kind,
            "feature_dim": int(packet.train.shape[1]),
            "coverage_test": packet.coverage_test,
            "alpha": alpha,
            "val_target_rmse": y_val_score,
            "val_endpoint_rmse": val_endpoint,
        },
    )
    preds = {"train": rtr.reshape(len(rtr), -1), "val": rva.reshape(len(rva), -1), "test": rte.reshape(len(rte), -1)}
    return rows, preds


def stacked_val_calibration(
    pred_blocks: dict[str, dict[str, np.ndarray]],
    arrays: audit.SplitArrays,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = list(pred_blocks.keys())
    if len(names) < 2:
        return [], {}
    xva = np.concatenate([pred_blocks[n]["val"] for n in names], axis=1)
    xte = np.concatenate([pred_blocks[n]["test"] for n in names], axis=1)
    yva = arrays.residual_val.reshape(len(arrays.residual_val), -1)
    best_alpha = 0.0
    best = float("inf")
    best_model: Ridge | None = None
    for a in parse_floats(args.stack_alphas):
        model = Ridge(alpha=a)
        model.fit(xva, yva)
        pva = model.predict(xva)
        rva = pva.reshape(arrays.residual_val.shape)
        score = endpoint_rmse_val(arrays, rva, max(args.horizons))
        if score < best:
            best = score
            best_alpha = a
            best_model = model
    assert best_model is not None
    rte = best_model.predict(xte).reshape(arrays.residual_test.shape).astype(np.float32)
    rows = evaluate_prediction(
        "v32_stacked_component_target_calibration",
        rte,
        arrays,
        args,
        {
            "stage": "v32_stacked_component_calibration",
            "packet": "stacked_best_components",
            "family": "stacked",
            "control": "real",
            "formulation": "+".join(names),
            "model": "val_ridge_stack",
            "feature_dim": int(xva.shape[1]),
            "coverage_test": 1.0,
            "alpha": best_alpha,
            "val_endpoint_rmse": best,
        },
    )
    return rows, {"stacked_components": names, "alpha": best_alpha, "val_endpoint_rmse": best}


def write_report(out_dir: Path, summary: pd.DataFrame, route_probe_df: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    lines = ["# v32 Dense Visual-State + Target Reformulation", ""]
    lines.append("## Best h6")
    h6 = summary[summary["horizon"].eq(max(args.horizons))].sort_values("rmse")
    cols = [c for c in ["method", "rmse", "r2", "stage", "packet", "control", "formulation", "model", "val_endpoint_rmse"] if c in h6.columns]
    lines.append(h6[cols].head(25).to_markdown(index=False))
    lines.append("")
    lines.append("## Route Probe")
    if not route_probe_df.empty:
        cols = [c for c in ["packet", "family", "control", "feature_dim", "coverage_test", "route_top3", "route_nll"] if c in route_probe_df.columns]
        lines.append(route_probe_df.sort_values("route_top3", ascending=False)[cols].head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Decision")
    lines.append("```json")
    lines.append(json.dumps(audit.finite_json(decision), indent=2, ensure_ascii=False))
    lines.append("```")
    (out_dir / "visual_state_target_v32_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.horizons = audit.parse_ints(args.horizons)
    arrays, split = audit.prepare_data(args)
    packets = build_packets(args, arrays, split)

    # Build route basis for route-observability diagnostics.
    basis = v26.build_route_basis(args, args.out_dir / "stage_route_basis")
    route_rows = []
    for name in parse_csv(args.route_probe_packets):
        if name in packets:
            route_rows.append(route_probe(packets[name], basis, args))
    route_df = pd.DataFrame(route_rows)
    route_df.to_csv(args.out_dir / "visual_state_v32_route_probe.csv", index=False)

    rows: list[dict[str, Any]] = []
    pred_pool: dict[str, dict[str, np.ndarray]] = {}
    formulations = parse_csv(args.formulations)
    packet_names = [p for p in parse_csv(args.packets) if p in packets]
    for packet_name in packet_names:
        packet = packets[packet_name]
        for formulation in formulations:
            try:
                r, preds = run_formulation(packet, formulation, "ridge", arrays, split, args)
            except Exception as exc:
                rows.append({
                    "method": f"ERROR_{packet_name}_{formulation}_ridge",
                    "horizon": max(args.horizons),
                    "rmse": float("nan"),
                    "r2": float("nan"),
                    "stage": "error",
                    "packet": packet_name,
                    "formulation": formulation,
                    "model": "ridge",
                    "error": repr(exc),
                })
                continue
            rows.extend(r)
            # Only real non-control candidate predictions go into the stacked component layer.
            if packet.control == "real" and "control" not in packet.name:
                h6 = [x for x in r if x["horizon"] == max(args.horizons)][0]
                key = f"{packet_name}:{formulation}"
                if float(h6["rmse"]) < float(args.stack_candidate_h6_max):
                    pred_pool[key] = preds

    if args.run_hgbdt:
        for packet_name in parse_csv(args.hgbdt_packets):
            if packet_name not in packets:
                continue
            for formulation in parse_csv(args.hgbdt_formulations):
                try:
                    r, preds = run_formulation(packets[packet_name], formulation, "hgbdt", arrays, split, args)
                    rows.extend(r)
                    h6 = [x for x in r if x["horizon"] == max(args.horizons)][0]
                    if float(h6["rmse"]) < float(args.stack_candidate_h6_max):
                        pred_pool[f"{packet_name}:{formulation}:hgbdt"] = preds
                except Exception as exc:
                    rows.append({
                        "method": f"ERROR_{packet_name}_{formulation}_hgbdt",
                        "horizon": max(args.horizons),
                        "rmse": float("nan"),
                        "r2": float("nan"),
                        "stage": "error",
                        "packet": packet_name,
                        "formulation": formulation,
                        "model": "hgbdt",
                        "error": repr(exc),
                    })

    stack_rows, stack_meta = stacked_val_calibration(pred_pool, arrays, args)
    rows.extend(stack_rows)

    summary = pd.DataFrame(rows)
    summary.insert(0, "seed", int(args.seed))
    summary.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "visual_state_target_v32_summary.csv", index=False)

    packet_meta = pd.DataFrame([
        {
            "packet": p.name,
            "family": p.family,
            "control": p.control,
            "feature_dim": p.train.shape[1],
            "coverage_train": p.coverage_train,
            "coverage_val": p.coverage_val,
            "coverage_test": p.coverage_test,
        }
        for p in packets.values()
    ])
    packet_meta.to_csv(args.out_dir / "visual_state_v32_packet_meta.csv", index=False)

    hmax = max(args.horizons)
    h6 = summary[summary["horizon"].eq(hmax) & summary["rmse"].notna()].sort_values("rmse")
    best = h6.iloc[0].to_dict() if not h6.empty else {}
    best_control = h6[h6["control"].astype(str).ne("real")].iloc[0].to_dict() if not h6[h6["control"].astype(str).ne("real")].empty else {}
    decision = {
        "elapsed_sec": time.time() - t0,
        "reference_h6": args.reference_h6,
        "best_h6": best,
        "best_control_h6": best_control,
        "stacked_meta": stack_meta,
        "pass_reference": bool(best and float(best.get("rmse", 999.0)) < float(args.reference_h6)),
        "real_beats_controls": bool(best and best_control and float(best.get("rmse", 999.0)) < float(best_control.get("rmse", -999.0))),
    }
    (args.out_dir / "visual_state_target_v32_decision.json").write_text(json.dumps(audit.finite_json(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.out_dir, summary, route_df, decision, args)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "route_rows": len(route_df), "best_h6": decision["best_h6"]}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--dense-features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--table-root", type=Path, default=ROOT / "new_data" / "lachance_epithelia" / "tables")
    ap.add_argument("--dataset", default="MDCK_Bulk")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seq", default="1,2,3,4")
    ap.add_argument("--val-seq", default="5")
    ap.add_argument("--test-seq", default="6")
    ap.add_argument("--horizons", default="1,2,4,6")
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--max-train-rows", type=int, default=6000)
    ap.add_argument("--max-val-rows", type=int, default=1800)
    ap.add_argument("--max-test-rows", type=int, default=2400)
    ap.add_argument("--max-features-per-family", type=int, default=160)
    ap.add_argument("--max-all-features", type=int, default=384)
    ap.add_argument("--reference-h6", type=float, default=16.745)
    ap.add_argument("--device", default="auto")

    # v26 route-basis compatibility for route probes.
    ap.add_argument("--generator-max-train-rows", type=int, default=-1)
    ap.add_argument("--generator-max-val-rows", type=int, default=-1)
    ap.add_argument("--generator-max-test-rows", type=int, default=-1)
    ap.add_argument("--generator-posterior-epochs", type=int, default=4)
    ap.add_argument("--generator-student-epochs", type=int, default=4)
    ap.add_argument("--generator-learned-route-epochs", type=int, default=3)
    ap.add_argument("--generator-candidate-k", type=int, default=32)
    ap.add_argument("--generator-oracle-k", default="8,16,32")
    ap.add_argument("--generator-variant", default="context_velocity")
    ap.add_argument("--generator-prior-model", default="logistic")
    ap.add_argument("--generator-base-mixes", default="expert_top8_uniform,expert_top4_uniform,expert_all_uniform")
    ap.add_argument("--generator-calibrators", default="correction_context,stacked_context")
    ap.add_argument("--generator-max-context-features", type=int, default=384)
    ap.add_argument("--dense-max-cols", type=int, default=256)
    ap.add_argument("--v25-velocity-max-cols", type=int, default=160)
    ap.add_argument("--v25-route-k", type=int, default=12)

    ap.add_argument("--object-grid", type=Path, default=ROOT / "outputs" / "lachance_object_centric_mask_grid_bulk_seed42_2026-07-03" / "object_centric_mask_feature_grid.csv")
    ap.add_argument("--temporal-grid", type=Path, default=ROOT / "outputs" / "temporal_mask_change_medium_bulk_seed42_2026-07-04" / "multiseed_instance_mask_feature_grid.csv")
    ap.add_argument("--multiseed-grid", type=Path, default=ROOT / "outputs" / "multiseed_instance_mask_medium_bulk_seed42_2026-07-04" / "multiseed_instance_mask_feature_grid.csv")
    ap.add_argument("--seg-foundation-grid", type=Path, default=ROOT / "outputs" / "seg_tracking_foundation_v21_medium_fusion_bulk_seed42_2026-07-04" / "seg_tracking_foundation_feature_grid.csv")
    ap.add_argument(
        "--visual-tokens",
        default="area,perimeter,eccentricity,solidity,extent,major,minor,elongation,orient,velocity,centroid,front,back,left,right,balance,intensity,grad,free,contact,boundary,neighbor,seed,center,available,quality,fallback,track_aligned",
    )
    ap.add_argument("--max-object-cols", type=int, default=220)
    ap.add_argument("--max-temporal-cols", type=int, default=220)
    ap.add_argument("--max-multiseed-cols", type=int, default=120)
    ap.add_argument("--max-seg-cols", type=int, default=160)
    ap.add_argument("--max-interaction-cols", type=int, default=120)

    ap.add_argument(
        "--packets",
        default="coord_all_context,identity_history,coord_plus_object_state,coord_plus_object_state_interaction,coord_plus_seg_foundation_state,coord_plus_seg_foundation_state_interaction,coord_plus_visual_state,coord_plus_visual_interaction,coord_plus_object_state_zero,coord_plus_object_state_row_shuffled,coord_plus_object_state_same_frame_wrong_cell,coord_plus_object_state_time_shuffled,coord_plus_seg_foundation_state_zero,coord_plus_seg_foundation_state_row_shuffled,coord_plus_seg_foundation_state_same_frame_wrong_cell,coord_plus_seg_foundation_state_time_shuffled",
    )
    ap.add_argument("--route-probe-packets", default="coord_all_context,identity_history,coord_plus_object_state,coord_plus_object_state_interaction,coord_plus_seg_foundation_state,coord_plus_seg_foundation_state_interaction,coord_plus_visual_state,coord_plus_visual_interaction,coord_plus_object_state_row_shuffled,coord_plus_object_state_same_frame_wrong_cell,coord_plus_object_state_time_shuffled,coord_plus_seg_foundation_state_row_shuffled,coord_plus_seg_foundation_state_same_frame_wrong_cell,coord_plus_seg_foundation_state_time_shuffled")
    ap.add_argument("--formulations", default="direct_residual_steps,endpoint_residual_h1246,parallel_perp_full_steps,speed_turn_full_steps,flow_relative_r64,flow_relative_r128,flow_relative_r256")
    ap.add_argument("--ridge-alphas", default="0.1,0.3,1,3,10,30,100,300,1000,3000")
    ap.add_argument("--stack-alphas", default="0.1,0.3,1,3,10,30,100,300,1000")
    ap.add_argument("--stack-candidate-h6-max", type=float, default=18.5)
    ap.add_argument("--run-hgbdt", action="store_true")
    ap.add_argument("--hgbdt-packets", default="coord_plus_visual_interaction,coord_all_context")
    ap.add_argument("--hgbdt-formulations", default="direct_residual_steps,endpoint_residual_h1246,parallel_perp_full_steps")
    ap.add_argument("--hgbdt-iter", type=int, default=80)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.generator_max_train_rows < 0:
        args.generator_max_train_rows = args.max_train_rows
    if args.generator_max_val_rows < 0:
        args.generator_max_val_rows = args.max_val_rows
    if args.generator_max_test_rows < 0:
        args.generator_max_test_rows = args.max_test_rows
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1200)
        args.max_val_rows = min(args.max_val_rows, 400)
        args.max_test_rows = min(args.max_test_rows, 600)
        args.generator_max_train_rows = args.max_train_rows
        args.generator_max_val_rows = args.max_val_rows
        args.generator_max_test_rows = args.max_test_rows
        args.max_all_features = min(args.max_all_features, 220)
        args.max_object_cols = min(args.max_object_cols, 80)
        args.max_temporal_cols = min(args.max_temporal_cols, 80)
        args.max_multiseed_cols = min(args.max_multiseed_cols, 60)
        args.max_seg_cols = min(args.max_seg_cols, 80)
        args.max_interaction_cols = min(args.max_interaction_cols, 50)
        args.hgbdt_iter = min(args.hgbdt_iter, 25)
    return args


if __name__ == "__main__":
    run(parse_args())
