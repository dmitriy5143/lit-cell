#!/usr/bin/env python3
"""h1-first multi-step decoder over morphology and tissue-flow context.

The runner trains causal models to predict step displacements
``delta x_1 ... delta x_H`` from current-frame features only.  Endpoints h1/h2/h4/h6
are then cumulative sums of the predicted steps.

This is the next diagnostic after the simple ``h * predicted_h1`` rollout:
instead of repeating one step, the decoder is allowed to learn acceleration,
turning, damping, and horizon-dependent uncertainty from the current state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_image_feature_probe as ifp  # noqa: E402

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]


DEFAULT_FEATURES = (
    ROOT
    / "outputs"
    / "lachance_feature_reconnaissance_ms_tf_mdck_bulk_h1h4h6_seed42_2026-06-15"
    / "combined_feature_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "lachance_h1_sequence_raw_context_decoder_2026-06-16"
EPS = 1e-8


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ArrayPack:
    traj: np.ndarray
    morph: np.ndarray
    flow: np.ndarray
    base_step: np.ndarray
    target_steps: np.ndarray
    residual_steps: np.ndarray


@dataclass
class Norm:
    mean: np.ndarray
    std: np.ndarray


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
    return [int(p.strip()) for p in str(text or "").split(",") if p.strip()]


def parse_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def gain_pct(base: float, value: float) -> float:
    return float((base - value) / max(abs(base), EPS) * 100.0)


def safe_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def fit_norm(x: np.ndarray, *, axis: tuple[int, ...] | int = 0) -> Norm:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=axis, keepdims=True).astype(np.float32)
    std = np.maximum(x.std(axis=axis, keepdims=True), 1e-6).astype(np.float32)
    return Norm(mean=mean, std=std)


def apply_norm(x: np.ndarray, norm: Norm, clip: float = 8.0) -> np.ndarray:
    z = (np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) - norm.mean) / norm.std
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip), -clip, clip).astype(np.float32)


def denorm_torch(x: torch.Tensor, norm: Norm, device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(norm.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(norm.std, dtype=torch.float32, device=device)
    return x * std + mean


def build_sequence_table(
    *,
    features: pd.DataFrame,
    table_root: Path,
    dataset: str,
    max_horizon: int,
) -> pd.DataFrame:
    img = features[features["dataset"].eq(dataset)].copy()
    if img.empty:
        raise ValueError(f"No feature rows for dataset={dataset}")
    img["sequence"] = img["sequence"].astype(int)
    img["frame"] = img["frame"].astype(int)
    img["track_id"] = img["track_id"].astype(int)
    seqs = sorted(int(s) for s in img["sequence"].unique())
    cur_frames = set(int(f) for f in img["frame"].unique())
    needed_frames = set(cur_frames)
    for h in range(1, int(max_horizon) + 1):
        needed_frames |= {int(f) + h for f in cur_frames}
    tables = [ifp.read_track_table(table_root, dataset, seq, needed_frames) for seq in seqs]
    tracks = pd.concat(tables, ignore_index=True)
    current_cols = [c for c in ifp.TRACK_COLS if c in tracks.columns]
    current = tracks[current_cols].copy()
    merged = img.merge(
        current,
        on=["dataset", "sequence", "frame", "track_id", "x_px", "y_px"],
        how="inner",
        suffixes=("", "_track"),
    )
    for h in range(1, int(max_horizon) + 1):
        future = tracks[["sequence", "frame", "track_id", "x_px", "y_px"]].copy()
        future["frame"] = future["frame"].astype(int) - h
        future = future.rename(columns={"x_px": f"x_plus_{h}", "y_px": f"y_plus_{h}"})
        merged = merged.merge(future, on=["sequence", "frame", "track_id"], how="inner")

    prev_x = merged["x_px"].astype(float)
    prev_y = merged["y_px"].astype(float)
    for h in range(1, int(max_horizon) + 1):
        xh = merged[f"x_plus_{h}"].astype(float)
        yh = merged[f"y_plus_{h}"].astype(float)
        merged[f"step{h}_dx"] = xh - prev_x
        merged[f"step{h}_dy"] = yh - prev_y
        merged[f"target_h{h}_dx"] = xh - merged["x_px"].astype(float)
        merged[f"target_h{h}_dy"] = yh - merged["y_px"].astype(float)
        prev_x = xh
        prev_y = yh

    merged["dx_px"] = merged["dx_px"].fillna(0.0)
    merged["dy_px"] = merged["dy_px"].fillna(0.0)
    merged["proposal_dx"] = merged["dx_px"]
    merged["proposal_dy"] = merged["dy_px"]
    merged["proposal_norm"] = np.sqrt(np.square(merged["proposal_dx"]) + np.square(merged["proposal_dy"]))
    if "QUALITY" not in merged.columns:
        merged["QUALITY"] = 0.0

    x_scale = max(float(merged["x_px"].quantile(0.99) - merged["x_px"].quantile(0.01)), 1.0)
    y_scale = max(float(merged["y_px"].quantile(0.99) - merged["y_px"].quantile(0.01)), 1.0)
    f_scale = max(float(merged["frame"].max() - merged["frame"].min()), 1.0)
    merged["x_norm"] = (merged["x_px"] - float(merged["x_px"].median())) / x_scale
    merged["y_norm"] = (merged["y_px"] - float(merged["y_px"].median())) / y_scale
    merged["frame_norm"] = (merged["frame"] - float(merged["frame"].min())) / f_scale
    return merged


def make_split(df: pd.DataFrame, train_seq: list[int], val_seq: list[int], test_seq: list[int], seed: int) -> SplitData:
    split = ifp.make_split(df, train_seq, val_seq, test_seq, seed)
    return SplitData(train=split.train, val=split.val, test=split.test)


def sample_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    return ifp.sample_rows(df, max_rows, seed)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    traj = [c for c in ifp.TRAJECTORY_FEATURES if c in df.columns]
    morph = [c for c in df.columns if c.startswith("ms_")]
    flow = [c for c in df.columns if c.startswith("tf_")]
    return traj, morph, flow


def shuffled(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return x[rng.permutation(len(x))]


def make_pack(df: pd.DataFrame, max_horizon: int, feature_block: str, seed: int) -> ArrayPack:
    traj_cols, morph_cols, flow_cols = feature_columns(df)
    traj = safe_matrix(df, traj_cols)
    morph_real = safe_matrix(df, morph_cols)
    flow_real = safe_matrix(df, flow_cols)
    morph = np.zeros_like(morph_real)
    flow = np.zeros_like(flow_real)
    if feature_block in {"trajectory_morphology", "trajectory_morphology_tissue_flow"}:
        morph = morph_real
    if feature_block in {"trajectory_tissue_flow", "trajectory_morphology_tissue_flow"}:
        flow = flow_real
    if feature_block == "trajectory_morphology_tissue_flow_shuffled_both":
        morph = shuffled(morph_real, seed + 11)
        flow = shuffled(flow_real, seed + 17)
    if feature_block == "trajectory_morphology_tissue_flow_shuffled_ms":
        morph = shuffled(morph_real, seed + 23)
        flow = flow_real
    if feature_block == "trajectory_morphology_tissue_flow_shuffled_tf":
        morph = morph_real
        flow = shuffled(flow_real, seed + 29)

    base = df[["dx_px", "dy_px"]].fillna(0.0).to_numpy(np.float32)
    steps = np.stack(
        [
            df[[f"step{h}_dx", f"step{h}_dy"]].to_numpy(np.float32)
            for h in range(1, int(max_horizon) + 1)
        ],
        axis=1,
    )
    residual = steps - base[:, None, :]
    return ArrayPack(
        traj=traj,
        morph=morph,
        flow=flow,
        base_step=base,
        target_steps=steps.astype(np.float32),
        residual_steps=residual.astype(np.float32),
    )


class Branch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max(1, input_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.input_dim = int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            x = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return self.net(x)


class TwoBranchSequenceDecoder(nn.Module):
    def __init__(
        self,
        traj_dim: int,
        morph_dim: int,
        flow_dim: int,
        hidden_dim: int,
        max_horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.traj = Branch(traj_dim, hidden_dim, dropout)
        self.morph = Branch(morph_dim, hidden_dim, dropout)
        self.flow = Branch(flow_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.max_horizon * 2),
        )

    def forward(self, traj: torch.Tensor, morph: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ht = self.traj(traj)
        hm = self.morph(morph)
        hf = self.flow(flow)
        gates = torch.sigmoid(self.gate(torch.cat([ht, hm, hf], dim=1)))
        hm_g = hm * gates[:, 0:1]
        hf_g = hf * gates[:, 1:2]
        fused = torch.cat([ht, hm_g, hf_g, hm_g - hf_g, hm_g * hf_g], dim=1)
        out = self.decoder(fused).reshape(-1, self.max_horizon, 2)
        return out, gates


def normalize_pack(
    pack: ArrayPack,
    norms: tuple[Norm, Norm, Norm, Norm],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traj_norm, morph_norm, flow_norm, target_norm = norms
    return (
        apply_norm(pack.traj, traj_norm),
        apply_norm(pack.morph, morph_norm),
        apply_norm(pack.flow, flow_norm),
        apply_norm(pack.residual_steps, target_norm, clip=12.0),
    )


def endpoint_rmse_from_steps(target_steps: np.ndarray, pred_steps: np.ndarray, horizon: int) -> float:
    y = target_steps[:, :horizon, :].sum(axis=1)
    pred = pred_steps[:, :horizon, :].sum(axis=1)
    return ifp.vector_rmse(y, pred)


def validation_score(target_steps: np.ndarray, pred_steps: np.ndarray, horizons: list[int]) -> float:
    return float(np.mean([endpoint_rmse_from_steps(target_steps, pred_steps, h) for h in horizons]))


def train_torch_decoder(
    train: ArrayPack,
    val: ArrayPack,
    *,
    max_horizon: int,
    eval_horizons: list[int],
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    device: torch.device,
) -> tuple[TwoBranchSequenceDecoder, tuple[Norm, Norm, Norm, Norm], dict[str, Any]]:
    torch.manual_seed(int(seed) + 7001)
    np.random.seed(int(seed) + 7003)
    norms = (
        fit_norm(train.traj),
        fit_norm(train.morph),
        fit_norm(train.flow),
        fit_norm(train.residual_steps, axis=(0,)),
    )
    tr_traj, tr_morph, tr_flow, tr_y = normalize_pack(train, norms)
    va_traj, va_morph, va_flow, va_y = normalize_pack(val, norms)
    model = TwoBranchSequenceDecoder(
        train.traj.shape[1],
        train.morph.shape[1],
        train.flow.shape[1],
        hidden_dim,
        max_horizon,
        dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    tr_t = torch.as_tensor(tr_traj, dtype=torch.float32, device=device)
    tr_m = torch.as_tensor(tr_morph, dtype=torch.float32, device=device)
    tr_f = torch.as_tensor(tr_flow, dtype=torch.float32, device=device)
    tr_y_t = torch.as_tensor(tr_y, dtype=torch.float32, device=device)
    tr_res_px = torch.as_tensor(train.residual_steps, dtype=torch.float32, device=device)
    va_t = torch.as_tensor(va_traj, dtype=torch.float32, device=device)
    va_m = torch.as_tensor(va_morph, dtype=torch.float32, device=device)
    va_f = torch.as_tensor(va_flow, dtype=torch.float32, device=device)

    target_std_scalar = float(np.maximum(np.std(train.residual_steps), 1.0))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    rng = np.random.default_rng(int(seed) + 7011)
    n = len(train.traj)
    for epoch in range(int(epochs)):
        order = rng.permutation(n)
        model.train()
        for start in range(0, n, int(batch_size)):
            idx_np = order[start : start + int(batch_size)]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            pred_norm, _ = model(tr_t[idx], tr_m[idx], tr_f[idx])
            pred_norm = pred_norm.contiguous()
            loss = F.smooth_l1_loss(pred_norm, tr_y_t[idx].contiguous())
            pred_px = denorm_torch(pred_norm, norms[3], device)
            endpoint_loss = 0.0
            for h in eval_horizons:
                pred_ep = pred_px[:, :h, :].sum(dim=1) / target_std_scalar
                true_ep = tr_res_px[idx, :h, :].sum(dim=1) / target_std_scalar
                endpoint_loss = endpoint_loss + F.smooth_l1_loss(pred_ep, true_ep)
            loss = loss + 0.5 * endpoint_loss / max(len(eval_horizons), 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_norm, _ = model(va_t, va_m, va_f)
            pred_res = denorm_torch(pred_norm, norms[3], device).detach().cpu().numpy()
            pred_steps = val.base_step[:, None, :] + pred_res
            score = validation_score(val.target_steps, pred_steps, eval_horizons)
        if score < best_val - 1e-4:
            best_val = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= 14:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, norms, {"best_epoch": best_epoch, "best_val_endpoint_rmse_px": best_val}


@torch.no_grad()
def predict_torch_decoder(
    model: TwoBranchSequenceDecoder,
    pack: ArrayPack,
    norms: tuple[Norm, Norm, Norm, Norm],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    traj, morph, flow, _ = normalize_pack(pack, norms)
    x_t = torch.as_tensor(traj, dtype=torch.float32, device=device)
    x_m = torch.as_tensor(morph, dtype=torch.float32, device=device)
    x_f = torch.as_tensor(flow, dtype=torch.float32, device=device)
    preds = []
    gates = []
    for start in range(0, len(traj), int(batch_size)):
        pred_norm, gate = model(x_t[start : start + batch_size], x_m[start : start + batch_size], x_f[start : start + batch_size])
        pred_res = denorm_torch(pred_norm, norms[3], device).detach().cpu().numpy()
        preds.append(pred_res)
        gates.append(gate.detach().cpu().numpy())
    pred_residual = np.concatenate(preds, axis=0).astype(np.float32)
    gate_np = np.concatenate(gates, axis=0).astype(np.float32)
    return pack.base_step[:, None, :] + pred_residual, gate_np


def fit_predict_linear(
    model_name: str,
    train: ArrayPack,
    val: ArrayPack,
    test: ArrayPack,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = np.concatenate([train.traj, train.morph, train.flow], axis=1)
    x_val = np.concatenate([val.traj, val.morph, val.flow], axis=1)
    x_test = np.concatenate([test.traj, test.morph, test.flow], axis=1)
    y_train = train.residual_steps.reshape(len(train.traj), -1)
    y_val = val.residual_steps.reshape(len(val.traj), -1)
    train_z, [val_z, test_z], norm_info = ifp.standardize(x_train, x_val, x_test)
    if model_name == "ridge":
        if Ridge is None:
            raise RuntimeError("sklearn Ridge unavailable")
        best: tuple[float, Any, float] | None = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0):
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(train_z, y_train)
            pred_val = model.predict(val_z).reshape(len(val_z), -1, 2)
            if not np.isfinite(pred_val).all():
                continue
            pred_steps_val = val.base_step[:, None, :] + pred_val
            score = validation_score(val.target_steps, pred_steps_val, [1, 2, 4, pred_val.shape[1]])
            if not np.isfinite(score):
                continue
            if best is None or score < best[0]:
                best = (score, model, float(alpha))
        if best is None:
            model = Ridge(alpha=3000.0, solver="svd")
            model.fit(train_z, y_train)
            best = (float("nan"), model, 3000.0)
        pred = best[1].predict(test_z).reshape(len(test_z), -1, 2)
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        pred = np.clip(pred, -1e4, 1e4).astype(np.float32)
        return test.base_step[:, None, :] + pred, {**norm_info, "alpha": best[2], "val_endpoint_rmse_px": best[0]}
    if model_name == "hgbdt":
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("sklearn HistGradientBoostingRegressor unavailable")
        preds = []
        for dim in range(y_train.shape[1]):
            model = HistGradientBoostingRegressor(
                max_iter=130,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.04,
                random_state=int(seed) + dim,
            )
            model.fit(train_z, y_train[:, dim])
            preds.append(model.predict(test_z))
        pred = np.column_stack(preds).reshape(len(test_z), -1, 2).astype(np.float32)
        return test.base_step[:, None, :] + pred, {**norm_info, "hgbdt_outputs": y_train.shape[1]}
    raise ValueError(model_name)


def evaluate_steps(
    *,
    dataset: str,
    seed: int,
    model: str,
    feature_block: str,
    target_steps: np.ndarray,
    pred_steps: np.ndarray,
    base_step: np.ndarray,
    horizons: list[int],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in horizons:
        y = target_steps[:, :h, :].sum(axis=1)
        pred = pred_steps[:, :h, :].sum(axis=1)
        base = float(h) * base_step
        rmse = ifp.vector_rmse(y, pred)
        base_rmse = ifp.vector_rmse(y, base)
        rows.append(
            {
                "dataset": dataset,
                "horizon": int(h),
                "seed": int(seed),
                "model": model,
                "feature_block": feature_block,
                "rmse_px": rmse,
                "constant_velocity_rmse_px": base_rmse,
                "gain_vs_constant_velocity_pct": gain_pct(base_rmse, rmse),
                "r2": ifp.vector_r2(y, pred),
                "cosine": ifp.mean_cosine(y, pred),
                "magnitude_ratio": ifp.magnitude_ratio(y, pred),
                **info,
            }
        )
    return rows


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(args.features)
    max_h = max(parse_ints(args.eval_horizons))
    full = build_sequence_table(
        features=features,
        table_root=args.table_root,
        dataset=args.dataset,
        max_horizon=max_h,
    )
    split = make_split(
        full,
        parse_ints(args.train_sequences),
        parse_ints(args.val_sequences),
        parse_ints(args.test_sequences),
        int(args.seed),
    )
    train_df = sample_rows(split.train, args.max_train_rows, args.seed + 11)
    val_df = sample_rows(split.val, args.max_val_rows, args.seed + 13)
    test_df = sample_rows(split.test, args.max_test_rows, args.seed + 17)
    horizons = parse_ints(args.eval_horizons)
    feature_blocks = parse_strs(args.feature_blocks)
    models = parse_strs(args.models)
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    base_pack = make_pack(test_df, max_h, "trajectory_only", args.seed)
    rows.extend(
        evaluate_steps(
            dataset=args.dataset,
            seed=args.seed,
            model="constant_velocity",
            feature_block="none",
            target_steps=base_pack.target_steps,
            pred_steps=np.repeat(base_pack.base_step[:, None, :], max_h, axis=1),
            base_step=base_pack.base_step,
            horizons=horizons,
            info={},
        )
    )

    cached: dict[str, tuple[ArrayPack, ArrayPack, ArrayPack]] = {}
    for block in feature_blocks:
        train_pack = make_pack(train_df, max_h, block, args.seed + 101)
        val_pack = make_pack(val_df, max_h, block, args.seed + 103)
        test_pack = make_pack(test_df, max_h, block, args.seed + 107)
        cached[block] = (train_pack, val_pack, test_pack)
        probe_rows.append(
            {
                "dataset": args.dataset,
                "feature_block": block,
                "traj_dim": int(train_pack.traj.shape[1]),
                "morph_dim": int(train_pack.morph.shape[1]),
                "flow_dim": int(train_pack.flow.shape[1]),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
            }
        )
        for model_name in models:
            print(f"[{args.dataset} seed{args.seed}] {model_name} {block}", flush=True)
            if model_name in {"ridge", "hgbdt"}:
                pred_steps, info = fit_predict_linear(
                    model_name,
                    train_pack,
                    val_pack,
                    test_pack,
                    seed=args.seed + len(rows),
                )
            elif model_name == "torch":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                model, norms, info = train_torch_decoder(
                    train_pack,
                    val_pack,
                    max_horizon=max_h,
                    eval_horizons=horizons,
                    seed=args.seed + len(rows),
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    dropout=args.dropout,
                    device=device,
                )
                pred_steps, gates = predict_torch_decoder(
                    model,
                    test_pack,
                    norms,
                    batch_size=args.batch_size,
                    device=device,
                )
                info = {
                    **info,
                    "gate_morph_mean": float(np.mean(gates[:, 0])),
                    "gate_flow_mean": float(np.mean(gates[:, 1])),
                }
            else:
                raise ValueError(f"Unknown model: {model_name}")
            rows.extend(
                evaluate_steps(
                    dataset=args.dataset,
                    seed=args.seed,
                    model=model_name,
                    feature_block=block,
                    target_steps=test_pack.target_steps,
                    pred_steps=pred_steps,
                    base_step=test_pack.base_step,
                    horizons=horizons,
                    info=info,
                )
            )

    summary = pd.DataFrame(rows)
    probe = pd.DataFrame(probe_rows)
    for (dataset, horizon, seed, model), part in summary[
        summary["model"].ne("constant_velocity")
    ].groupby(["dataset", "horizon", "seed", "model"]):
        by = part.set_index("feature_block")
        if "trajectory_only" not in by.index or "trajectory_morphology_tissue_flow" not in by.index:
            continue
        traj = float(by.loc["trajectory_only", "rmse_px"])
        full_rmse = float(by.loc["trajectory_morphology_tissue_flow", "rmse_px"])
        shuffled_rmse = (
            float(by.loc["trajectory_morphology_tissue_flow_shuffled_both", "rmse_px"])
            if "trajectory_morphology_tissue_flow_shuffled_both" in by.index
            else math.nan
        )
        gate_rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "seed": int(seed),
                "model": model,
                "trajectory_only_rmse_px": traj,
                "full_raw_context_rmse_px": full_rmse,
                "shuffled_raw_context_rmse_px": shuffled_rmse,
                "full_gain_vs_trajectory_pct": gain_pct(traj, full_rmse),
                "full_gain_vs_shuffled_pct": gain_pct(shuffled_rmse, full_rmse) if np.isfinite(shuffled_rmse) else math.nan,
                "context_gate_pass": bool(
                    np.isfinite(shuffled_rmse)
                    and gain_pct(traj, full_rmse) >= 1.0
                    and gain_pct(shuffled_rmse, full_rmse) >= 0.25
                ),
            }
        )
    return summary, probe, pd.DataFrame(gate_rows)


def write_report(out_dir: Path, summary: pd.DataFrame, probe: pd.DataFrame, gate: pd.DataFrame, args: argparse.Namespace) -> None:
    best = summary.sort_values(["horizon", "rmse_px"]).groupby("horizon").head(8)
    lines = [
        "# h1-first sequence raw-context decoder",
        "",
        "## Payload",
        "",
        "```json",
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Best rows",
        "",
        best.to_markdown(index=False) if not best.empty else "_No summary rows._",
        "",
        "## Context gate",
        "",
        gate.to_markdown(index=False) if not gate.empty else "_No gate rows._",
        "",
        "## Feature probe",
        "",
        probe.to_markdown(index=False) if not probe.empty else "_No probe rows._",
        "",
        "## Interpretation",
        "",
        "- The decoder predicts step sequence `dx1...dxH` from current-frame context only.",
        "- `trajectory_morphology_tissue_flow` must beat both `trajectory_only` and shuffled context to count as a real context effect.",
        "- If torch is weaker than Ridge/HGBDT, the signal exists but the neural conditioner/training still needs redesign.",
    ]
    (out_dir / "h1_sequence_raw_context_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--table-root", type=Path, default=ifp.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sequences", default="1,2,3,4")
    parser.add_argument("--val-sequences", default="5")
    parser.add_argument("--test-sequences", default="6")
    parser.add_argument("--eval-horizons", default="1,2,4,6")
    parser.add_argument("--models", default="ridge,hgbdt,torch")
    parser.add_argument(
        "--feature-blocks",
        default=(
            "trajectory_only,"
            "trajectory_tissue_flow,"
            "trajectory_morphology,"
            "trajectory_morphology_tissue_flow,"
            "trajectory_morphology_tissue_flow_shuffled_both"
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-val-rows", type=int, default=25000)
    parser.add_argument("--max-test-rows", type=int, default=25000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, probe, gate = run(args)
    summary.to_csv(args.out_dir / "h1_sequence_raw_context_summary.csv", index=False)
    probe.to_csv(args.out_dir / "h1_sequence_raw_context_feature_probe.csv", index=False)
    gate.to_csv(args.out_dir / "h1_sequence_raw_context_gate.csv", index=False)
    write_report(args.out_dir, summary, probe, gate, args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
