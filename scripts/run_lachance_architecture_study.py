#!/usr/bin/env python3
"""LaChance epithelial architecture-revival benchmark.

This runner reuses the latest causal temporal/flow/structural decoder from
`run_oz_full_architecture_study.py`, but feeds it external TrackMate tables from
LaChance et al. The first gate is deliberately not an OZ claim: it asks whether
the constrained graph decoder can recover the robust MDCK/HUVEC neighbour signal
found by the linear directional screens.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_oz_full_architecture_study as arch  # noqa: E402


DEFAULT_TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"
DEFAULT_OUT = ROOT / "outputs" / "lachance_architecture_study"
CELL_TYPES = ("HUVEC", "MDAMB231", "MDCK_Bulk", "MDCK_Edge")
SOCIAL_VARIANTS = ("geometry_structural", "geometry_self_structural", "oz_structural")


@dataclass
class SourceMeta:
    paths: list[Path]
    split_by_sequence: dict[str, str]
    frame_width_px: float
    frame_height_px: float
    r_cut_px: float
    selected_tracks_by_sequence: dict[str, int]
    rows_by_sequence: dict[str, int]


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


def causal_smooth_table(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if int(window) <= 1:
        return df.copy()
    out = df.sort_values(["GLOBAL_TRACK_ID", "FRAME"]).copy()
    xs: list[pd.Series] = []
    ys: list[pd.Series] = []
    for _, group in out.groupby("GLOBAL_TRACK_ID", sort=False):
        xs.append(group["x_px"].rolling(window=int(window), min_periods=1).mean())
        ys.append(group["y_px"].rolling(window=int(window), min_periods=1).mean())
    out["x_px"] = pd.concat(xs).sort_index()
    out["y_px"] = pd.concat(ys).sort_index()
    return out


def assign_movie_splits(
    sequences: list[str],
    *,
    split_mode: str,
    seed: int,
) -> dict[str, str]:
    if split_mode == "frame":
        return {seq: "frame" for seq in sequences}
    rng = random.Random(int(seed))
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        return {
            seq: "train" if idx == 0 else ("val" if idx == 1 else "test")
            for idx, seq in enumerate(shuffled)
        }
    n_train = max(1, int(round(0.60 * n)))
    n_val = max(1, int(round(0.20 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    out: dict[str, str] = {}
    for idx, seq in enumerate(shuffled):
        if idx < n_train:
            out[seq] = "train"
        elif idx < n_train + n_val:
            out[seq] = "val"
        else:
            out[seq] = "test"
    return out


def frame_split(frames: np.ndarray) -> dict[int, str]:
    unique = np.sort(np.unique(frames.astype(int)))
    n = int(len(unique))
    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n)) if n - n_train > 1 else max(0, n - n_train)
    out: dict[int, str] = {}
    for frame in unique[:n_train]:
        out[int(frame)] = "train"
    for frame in unique[n_train : n_train + n_val]:
        out[int(frame)] = "val"
    for frame in unique[n_train + n_val :]:
        out[int(frame)] = "test"
    return out


def standardize_lachance_table(
    path: Path,
    *,
    seq_id: int,
    sequence: str,
    split_label: str,
    frame_stride: int,
    smooth_window: int,
    crop_fraction: float,
    max_tracks_per_movie: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    d = pd.read_csv(path)
    rename: dict[str, str] = {}
    if "frame" in d.columns and "FRAME" not in d.columns:
        rename["frame"] = "FRAME"
    if "track_id" in d.columns and "TRACK_ID" not in d.columns:
        rename["track_id"] = "TRACK_ID"
    if rename:
        d = d.rename(columns=rename)
    required = {"FRAME", "TRACK_ID", "x_px", "y_px"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    d = d.loc[:, [c for c in d.columns if c in set(d.columns)]].copy()
    d["FRAME"] = pd.to_numeric(d["FRAME"], errors="coerce")
    d["TRACK_ID"] = pd.to_numeric(d["TRACK_ID"], errors="coerce")
    d["x_px"] = pd.to_numeric(d["x_px"], errors="coerce")
    d["y_px"] = pd.to_numeric(d["y_px"], errors="coerce")
    d = d.loc[
        d["FRAME"].notna()
        & d["TRACK_ID"].notna()
        & d["x_px"].notna()
        & d["y_px"].notna()
    ].copy()
    d["FRAME"] = d["FRAME"].astype(int)
    d["TRACK_ID"] = d["TRACK_ID"].astype(int)
    d = d.loc[d["TRACK_ID"] >= 0].copy()
    if frame_stride > 1:
        frame0 = int(d["FRAME"].min())
        d = d.loc[((d["FRAME"] - frame0) % int(frame_stride)) == 0].copy()
    if 0.0 < crop_fraction < 1.0 and not d.empty:
        side_fraction = math.sqrt(float(crop_fraction))
        track_center = (
            d.groupby("TRACK_ID", sort=False)[["x_px", "y_px"]]
            .median()
            .reset_index()
        )
        x_mid = float(track_center["x_px"].median())
        y_mid = float(track_center["y_px"].median())
        x_half = 0.5 * side_fraction * float(
            max(track_center["x_px"].max() - track_center["x_px"].min(), 1.0)
        )
        y_half = 0.5 * side_fraction * float(
            max(track_center["y_px"].max() - track_center["y_px"].min(), 1.0)
        )
        keep_tracks = track_center.loc[
            track_center["x_px"].between(x_mid - x_half, x_mid + x_half)
            & track_center["y_px"].between(y_mid - y_half, y_mid + y_half),
            "TRACK_ID",
        ].to_numpy()
        if len(keep_tracks) == 0:
            x_scale = max(float(track_center["x_px"].std()), 1.0)
            y_scale = max(float(track_center["y_px"].std()), 1.0)
            score = np.square((track_center["x_px"].to_numpy(float) - x_mid) / x_scale)
            score += np.square((track_center["y_px"].to_numpy(float) - y_mid) / y_scale)
            n_keep = max(50, int(math.ceil(float(crop_fraction) * len(track_center))))
            n_keep = min(n_keep, len(track_center))
            keep_tracks = track_center.iloc[np.argsort(score)[:n_keep]]["TRACK_ID"].to_numpy()
        d = d.loc[d["TRACK_ID"].isin(keep_tracks)].copy()
    if max_tracks_per_movie > 0:
        tracks = np.sort(d["TRACK_ID"].unique())
        if len(tracks) > max_tracks_per_movie:
            rng = np.random.default_rng(int(seed) + seq_id * 1009)
            selected = np.sort(
                rng.choice(tracks, size=int(max_tracks_per_movie), replace=False)
            )
            d = d.loc[d["TRACK_ID"].isin(selected)].copy()
    selected_tracks = int(d["TRACK_ID"].nunique())
    d = d.drop_duplicates(["TRACK_ID", "FRAME"], keep="first")
    d["SEQ_ID"] = int(seq_id)
    d["SEQ_NAME"] = sequence
    d["GLOBAL_TRACK_ID"] = d["SEQ_ID"].astype(str) + ":" + d["TRACK_ID"].astype(str)
    d = d.sort_values(["GLOBAL_TRACK_ID", "FRAME"]).reset_index(drop=True)
    d = causal_smooth_table(d, int(smooth_window))

    if split_label == "frame":
        mapping = frame_split(d["FRAME"].to_numpy(int))
        d["split"] = d["FRAME"].map(mapping)
    else:
        d["split"] = split_label

    grouped = d.groupby("GLOBAL_TRACK_ID", sort=False)
    prev_frame = grouped["FRAME"].shift(1)
    consecutive = prev_frame.notna() & d["FRAME"].eq(prev_frame + int(frame_stride))
    frame_scale = float(max(int(frame_stride), 1))
    d["raw_dx"] = ((d["x_px"] - grouped["x_px"].shift(1)) / frame_scale).where(
        consecutive
    )
    d["raw_dy"] = ((d["y_px"] - grouped["y_px"].shift(1)) / frame_scale).where(
        consecutive
    )
    d["raw_speed"] = np.sqrt(d["raw_dx"] ** 2 + d["raw_dy"] ** 2)
    step = d["raw_speed"].replace([np.inf, -np.inf], np.nan)
    med = step.groupby(d["GLOBAL_TRACK_ID"]).transform(
        lambda x: x.rolling(9, min_periods=3).median()
    )
    jump = step / np.maximum(med, 1e-6)
    d["quality_proxy"] = np.exp(
        -0.5
        * np.square(
            np.clip(jump.fillna(1.0).to_numpy(float) - 1.0, 0.0, 6.0) / 2.0
        )
    )
    return d, selected_tracks


def infer_dataset_geometry(raw: pd.DataFrame, r_cut_px: float | None) -> tuple[float, float, float]:
    width = float(max(raw["x_px"].max() + 1.0, raw["x_px"].max() - raw["x_px"].min(), 1.0))
    height = float(max(raw["y_px"].max() + 1.0, raw["y_px"].max() - raw["y_px"].min(), 1.0))
    if r_cut_px is not None and r_cut_px > 0:
        r_cut = float(r_cut_px)
    else:
        distances: list[np.ndarray] = []
        for _, idx0 in raw.groupby(["SEQ_ID", "FRAME"], sort=False).groups.items():
            idx = np.asarray(list(idx0), dtype=int)
            if len(idx) < 2:
                continue
            if len(distances) >= 48:
                break
            pos = raw.loc[idx, ["x_px", "y_px"]].to_numpy(float)
            if len(pos) > 2500:
                pos = pos[np.linspace(0, len(pos) - 1, 2500).astype(int)]
            from scipy.spatial import cKDTree

            dist, _ = cKDTree(pos).query(pos, k=min(11, len(pos)))
            if dist.ndim == 2 and dist.shape[1] > 1:
                distances.append(dist[:, 1:].reshape(-1))
        if distances:
            vals = np.concatenate(distances)
            r_cut = float(np.quantile(vals[np.isfinite(vals)], 0.90))
        else:
            r_cut = 50.0
    return width, height, max(r_cut, 5.0)


def load_lachance_dataset(
    cell_type: str,
    *,
    table_root: Path,
    split_mode: str,
    split_seed: int,
    max_movies: int,
    max_tracks_per_movie: int,
    frame_stride: int,
    smooth_window: int,
    crop_fraction: float,
    r_cut_px: float | None,
) -> tuple[pd.DataFrame, SourceMeta]:
    table_dir = table_root / cell_type
    paths = sorted(table_dir.glob("*_tracks.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_tracks.csv files under {table_dir}")
    if max_movies > 0:
        paths = paths[: int(max_movies)]
    sequences = [path.stem.removesuffix("_tracks").replace(f"{cell_type}_", "") for path in paths]
    split_by_sequence = assign_movie_splits(
        sequences, split_mode=split_mode, seed=split_seed
    )
    parts: list[pd.DataFrame] = []
    selected_tracks: dict[str, int] = {}
    rows_by_sequence: dict[str, int] = {}
    for seq_id, (path, sequence) in enumerate(zip(paths, sequences)):
        part, n_tracks = standardize_lachance_table(
            path,
            seq_id=seq_id,
            sequence=sequence,
            split_label=split_by_sequence[sequence],
            frame_stride=max(int(frame_stride), 1),
            smooth_window=int(smooth_window),
            crop_fraction=float(crop_fraction),
            max_tracks_per_movie=int(max_tracks_per_movie),
            seed=int(split_seed),
        )
        selected_tracks[sequence] = n_tracks
        rows_by_sequence[sequence] = int(len(part))
        parts.append(part)
    raw = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["SEQ_ID", "TRACK_ID", "FRAME"])
        .reset_index(drop=True)
    )
    width, height, r_cut = infer_dataset_geometry(raw, r_cut_px)
    return raw, SourceMeta(
        paths=paths,
        split_by_sequence=split_by_sequence,
        frame_width_px=width,
        frame_height_px=height,
        r_cut_px=r_cut,
        selected_tracks_by_sequence=selected_tracks,
        rows_by_sequence=rows_by_sequence,
    )


def prior_functions(_dataset: str):
    def c(distance: np.ndarray) -> np.ndarray:
        d = np.asarray(distance, dtype=float)
        return np.exp(-0.5 * np.square(d / 50.0))

    def dc(distance: np.ndarray) -> np.ndarray:
        d = np.asarray(distance, dtype=float)
        return -(d / (50.0**2)) * np.exp(-0.5 * np.square(d / 50.0))

    return c, dc


def prepare_dataset(
    dataset: str,
    raw: pd.DataFrame,
    meta: SourceMeta,
    *,
    horizon: int,
    k: int,
    device: torch.device,
) -> tuple[dict[str, arch.GraphTensors], arch.Normalizer, dict[str, Any]]:
    arch.DATASETS[dataset] = {
        "paths": tuple(meta.paths),
        "r_cut_px": float(meta.r_cut_px),
        "frame_width_px": float(meta.frame_width_px),
        "frame_height_px": float(meta.frame_height_px),
    }
    arch.prior_functions = prior_functions
    samples = arch.build_causal_samples(raw, horizon=int(horizon))
    samples = arch.add_visible_context_nodes(samples, raw)
    samples = arch.add_causal_flow_features(samples, dataset)
    parts = {
        split: samples[samples["split"].eq(split)].copy().reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    if min(len(parts["train"]), len(parts["val"]), len(parts["test"])) == 0:
        raise RuntimeError(
            f"Empty split for {dataset}: "
            f"{ {k0: len(v0) for k0, v0 in parts.items()} }"
        )
    norm = arch.fit_normalizer(parts["train"])
    arrays = {
        split: arch.build_graph_arrays(
            part,
            dataset,
            k=int(k),
            shuffle_seed=20260608 + split_id * 1009,
        )
        for split_id, (split, part) in enumerate(parts.items())
    }
    arch.fit_graph_normalization(arrays["train"], norm)
    graphs = {
        split: arch.graph_to_tensors(array, norm, device)
        for split, array in arrays.items()
    }
    coverage: dict[str, Any] = {
        "source_paths": [str(path) for path in meta.paths],
        "split_by_sequence": meta.split_by_sequence,
        "selected_tracks_by_sequence": meta.selected_tracks_by_sequence,
        "rows_by_sequence": meta.rows_by_sequence,
        "frame_width_px": float(meta.frame_width_px),
        "frame_height_px": float(meta.frame_height_px),
        "r_cut_px": float(meta.r_cut_px),
        "horizon": int(horizon),
    }
    for split in ("train", "val", "test"):
        valid = int(arrays[split].target_valid.sum())
        total = int(len(arrays[split].target_valid))
        coverage[f"{split}_causal_nodes"] = total
        coverage[f"{split}_target_nodes"] = valid
        coverage[f"{split}_context_only_nodes"] = total - valid
        coverage[f"{split}_edges"] = int(len(arrays[split].src))
    return graphs, norm, coverage


def decoder_variant(variant: str) -> str:
    if variant == "geometry_self_structural":
        return "geometry_structural"
    return variant


def uses_flow_base(variant: str) -> bool:
    return variant != "geometry_self_structural"


@torch.no_grad()
def evaluate_social_variant(
    temporal: arch.TemporalSelfEncoder,
    flow: arch.CoarseFlowEncoder,
    social: arch.StructuralInfluenceDecoder,
    graph: arch.GraphTensors,
    norm: arch.Normalizer,
    *,
    variant: str,
) -> tuple[np.ndarray, dict[str, float]]:
    self_pred, self_state, flow_pred, flow_state = arch.encode_base(
        temporal, flow, graph
    )
    if not uses_flow_base(variant):
        flow_pred = torch.zeros_like(flow_pred)
        flow_state = torch.zeros_like(flow_state)
    delta, diag = social(
        graph, self_state, flow_state, variant=decoder_variant(variant)
    )
    base = self_pred + flow_pred
    pred = base + delta
    mask = graph.target_valid.detach().cpu().numpy()
    pred_px = arch.to_px(pred, norm)
    y_px = graph.y_px.detach().cpu().numpy()
    metrics = arch.vector_metrics(y_px[mask], pred_px[mask], 1)
    delta_px = delta.detach().cpu().numpy() * norm.target_std
    residual_px = y_px - arch.to_px(base, norm)
    finite = mask & np.isfinite(residual_px).all(axis=1)
    dot = np.sum(delta_px[finite] * residual_px[finite], axis=1)
    denom = np.maximum(
        np.linalg.norm(delta_px[finite], axis=1)
        * np.linalg.norm(residual_px[finite], axis=1),
        1e-8,
    )
    metrics.update(
        {
            "social_magnitude_mean_px": float(
                np.mean(np.linalg.norm(delta_px[mask], axis=1))
            ),
            "social_magnitude_p90_px": float(
                np.quantile(np.linalg.norm(delta_px[mask], axis=1), 0.9)
            ),
            "social_residual_cosine": float(np.mean(dot / denom)),
            "node_gate_mean": float(diag["node_gate"][graph.target_valid].mean().cpu()),
            "node_gate_zero_fraction": float(
                (diag["node_gate"][graph.target_valid] == 0).float().mean().cpu()
            ),
            "node_gate_p90": float(
                torch.quantile(
                    diag["node_gate"][graph.target_valid].reshape(-1), 0.9
                ).cpu()
            ),
            "edge_gate_mean": float(diag["edge_gate"].mean().cpu()),
            "prior_strength": float(diag["prior_strength"].cpu()),
            "prior_amplitude_mean": float(diag["prior_amplitude"].mean().cpu()),
            "prior_amplitude_std": float(diag["prior_amplitude"].std().cpu()),
            "effective_degree_mean": float(
                diag["effective_degree"][graph.target_valid].mean().cpu()
            ),
            "mean_mix_mean": float(diag["mix"][graph.target_valid, 0].mean().cpu()),
            "mobility_parallel_mean": float(
                diag["mobility"][graph.target_valid, 0].mean().cpu()
            ),
            "mobility_perpendicular_mean": float(
                diag["mobility"][graph.target_valid, 1].mean().cpu()
            ),
        }
    )
    coeff = diag["coeff"].detach().cpu().numpy()
    for idx in range(coeff.shape[1]):
        metrics[f"basis_coeff_abs_mean_{idx}"] = float(np.mean(np.abs(coeff[:, idx])))
    return pred_px, metrics


def run_cell_type(
    cell_type: str,
    *,
    table_root: Path,
    split_mode: str,
    split_seed: int,
    max_movies: int,
    max_tracks_per_movie: int,
    frame_stride: int,
    smooth_window: int,
    crop_fraction: float,
    r_cut_px: float | None,
    horizon: int,
    seeds: list[int],
    variants: list[str],
    device: torch.device,
    k: int,
    temporal_epochs: int,
    flow_epochs: int,
    social_epochs: int,
    joint_epochs: int,
    batch_size: int,
    joint_threshold_pct: float,
    sequence_balanced_loss: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, meta = load_lachance_dataset(
        cell_type,
        table_root=table_root,
        split_mode=split_mode,
        split_seed=split_seed,
        max_movies=max_movies,
        max_tracks_per_movie=max_tracks_per_movie,
        frame_stride=frame_stride,
        smooth_window=smooth_window,
        crop_fraction=crop_fraction,
        r_cut_px=r_cut_px,
    )
    dataset = cell_type
    graphs, norm, coverage = prepare_dataset(
        dataset,
        raw,
        meta,
        horizon=horizon,
        k=k,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[{cell_type}] seed={seed} temporal", flush=True)
        temporal, temporal_info = arch.train_temporal(
            graphs["train"],
            graphs["val"],
            seed=seed,
            epochs=temporal_epochs,
            batch_size=batch_size,
            sequence_balanced_loss=sequence_balanced_loss,
        )
        zero_flow = arch.CoarseFlowEncoder(graphs["train"].flow.shape[1]).to(device)
        for parameter in zero_flow.parameters():
            torch.nn.init.zeros_(parameter)
        self_pred_px, self_metrics = arch.evaluate(
            temporal, zero_flow, None, graphs["test"], norm, variant="self_only"
        )
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "self_only",
                "joint_finetuned": False,
                **self_metrics,
                **arch.sequence_metric_fields(graphs["test"], self_pred_px),
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )

        with torch.no_grad():
            train_self, _ = temporal(graphs["train"].history)
            val_self, _ = temporal(graphs["val"].history)
        print(f"[{cell_type}] seed={seed} flow", flush=True)
        flow, flow_info = arch.train_flow(
            graphs["train"],
            graphs["val"],
            train_self.detach(),
            val_self.detach(),
            seed=seed,
            epochs=flow_epochs,
            batch_size=batch_size,
            sequence_balanced_loss=sequence_balanced_loss,
        )
        flow_pred_px, flow_metrics = arch.evaluate(
            temporal, flow, None, graphs["test"], norm, variant="self_flow"
        )
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "self_flow",
                "joint_finetuned": False,
                **flow_metrics,
                **arch.sequence_metric_fields(graphs["test"], flow_pred_px),
                **{f"flow_{k0}": v for k0, v in flow_info.items()},
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        encoded_flow = {
            split: arch.encode_base(temporal, flow, graph)
            for split, graph in graphs.items()
        }
        encoded_self = {
            split: arch.encode_base(temporal, zero_flow, graph)
            for split, graph in graphs.items()
        }
        for variant in variants:
            print(f"[{cell_type}] seed={seed} social {variant}", flush=True)
            encoded = encoded_flow if uses_flow_base(variant) else encoded_self
            train_base = encoded["train"][0] + encoded["train"][2]
            val_base = encoded["val"][0] + encoded["val"][2]
            social, social_info = arch.train_social(
                graphs["train"],
                graphs["val"],
                train_base.detach(),
                val_base.detach(),
                encoded["train"][1].detach(),
                encoded["val"][1].detach(),
                encoded["train"][3].detach(),
                encoded["val"][3].detach(),
                variant=decoder_variant(variant),
                seed=seed,
                epochs=social_epochs,
                sequence_balanced_loss=sequence_balanced_loss,
            )
            base_val = arch.masked_vector_mse(
                val_base,
                graphs["val"].y_norm,
                graphs["val"].target_valid,
                arch.sequence_groups(graphs["val"]) if sequence_balanced_loss else None,
            ).item()
            social_val = float(social_info["best_val_norm_mse"])
            val_gain = arch.relative_gain(base_val, social_val)
            candidate_temporal = temporal
            candidate_flow = flow
            joint_info: dict[str, float] = {}
            joint_used = val_gain >= joint_threshold_pct and joint_epochs > 0
            if joint_used:
                candidate_temporal = copy.deepcopy(temporal)
                candidate_flow = copy.deepcopy(flow)
                joint_info = arch.joint_finetune(
                    candidate_temporal,
                    candidate_flow,
                    social,
                    graphs["train"],
                    graphs["val"],
                    variant=decoder_variant(variant),
                    epochs=joint_epochs,
                    sequence_balanced_loss=sequence_balanced_loss,
                )
            eval_flow = candidate_flow if uses_flow_base(variant) else zero_flow
            pred_px, metrics = evaluate_social_variant(
                candidate_temporal,
                eval_flow,
                social,
                graphs["test"],
                norm,
                variant=variant,
            )
            rows.append(
                {
                    "dataset": cell_type,
                    "seed": seed,
                    "variant": variant,
                    "joint_finetuned": joint_used,
                    "stage_c_val_gain_pct": val_gain,
                    **metrics,
                    **arch.sequence_metric_fields(graphs["test"], pred_px),
                    **arch.paired_block_bootstrap(
                        graphs["test"],
                        flow_pred_px,
                        pred_px,
                        seed=seed + 300_003,
                    ),
                    **{f"social_{k0}": v for k0, v in social_info.items()},
                    **{f"joint_{k0}": v for k0, v in joint_info.items()},
                    **{f"flow_{k0}": v for k0, v in flow_info.items()},
                    **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
                }
            )
            print(
                f"[{cell_type}] seed={seed} {variant}: "
                f"rmse={metrics['rmse_px']:.5f}px "
                f"social={metrics['social_magnitude_mean_px']:.4f}px",
                flush=True,
            )
        if device.type == "mps":
            torch.mps.empty_cache()
    return pd.DataFrame(rows), coverage


def summarize(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, part in summary.groupby("dataset"):
        pivot = part.pivot_table(index="seed", columns="variant", values="rmse_px")
        for variant in pivot.columns:
            base = pivot["self_flow"] if "self_flow" in pivot else pivot["self_only"]
            gains = (base - pivot[variant]) / base * 100.0
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seeds": int(pivot[variant].notna().sum()),
                    "rmse_px_mean": float(pivot[variant].mean()),
                    "rmse_px_std": float(pivot[variant].std(ddof=0)),
                    "gain_vs_self_flow_pct_mean": float(gains.mean()),
                    "gain_vs_self_flow_pct_min": float(gains.min()),
                    "positive_seed_fraction": float((gains > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: dict[str, Any],
    out_dir: Path,
) -> None:
    lines = [
        "# LaChance Architecture Study",
        "",
        "This report is the first neural architecture-closing gate on the external epithelial trajectories.",
        "The claim tested here is graph/decoder usefulness, not a validated OZ law.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(finite_json(coverage), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Aggregate Test Metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Mean Diagnostics",
        "",
    ]
    diag_cols = [
        "dataset",
        "variant",
        "rmse_px",
        "r2_vec",
        "social_magnitude_mean_px",
        "social_residual_cosine",
        "node_gate_mean",
        "edge_gate_mean",
        "effective_degree_mean",
    ]
    available = [col for col in diag_cols if col in summary.columns]
    means = (
        summary[available]
        .groupby(["dataset", "variant"], as_index=False)
        .mean(numeric_only=True)
    )
    lines.append(means.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "A useful architecture candidate should beat `self_flow` with positive seed fraction near 1.0, positive movie/sequence-level gains, and nonzero but bounded social corrections.",
            "If this fails on MDCK bulk/edge, the bottleneck is the neural graph-to-motion interface rather than the static OZ prior alone.",
        ]
    )
    (out_dir / "lachance_architecture_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_gain(aggregate: pd.DataFrame, out_dir: Path) -> None:
    if aggregate.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    plot_df = aggregate[aggregate["variant"].ne("self_only")].copy()
    labels = plot_df["dataset"] + "\n" + plot_df["variant"]
    ax.bar(np.arange(len(plot_df)), plot_df["gain_vs_self_flow_pct_mean"], color="#4C7A6D")
    ax.axhline(0.0, color="#2a2a2a", linewidth=0.8)
    ax.set_xticks(np.arange(len(plot_df)), labels, rotation=25, ha="right")
    ax.set_ylabel("Gain over self + flow (%)")
    ax.set_title("LaChance neural graph/decoder gate")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "fig01_lachance_architecture_gain.png", dpi=260)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-types", nargs="+", choices=CELL_TYPES, default=["MDCK_Bulk", "MDCK_Edge", "MDAMB231"])
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-mode", choices=["movie", "frame"], default="movie")
    parser.add_argument("--split-seed", type=int, default=20260608)
    parser.add_argument("--max-movies", type=int, default=8)
    parser.add_argument("--max-tracks-per-movie", type=int, default=500)
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=0.0,
        help="Select a central spatial ROI by track median position; 0 disables cropping.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--r-cut-px", type=float, default=50.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--variants", nargs="+", choices=SOCIAL_VARIANTS, default=["geometry_structural"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temporal-epochs", type=int, default=40)
    parser.add_argument("--flow-epochs", type=int, default=30)
    parser.add_argument("--social-epochs", type=int, default=50)
    parser.add_argument("--joint-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--joint-threshold-pct", type=float, default=0.20)
    parser.add_argument("--sequence-balanced-loss", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.cell_types = args.cell_types[:1]
        args.max_movies = min(args.max_movies, 4)
        if args.crop_fraction <= 0:
            args.crop_fraction = 0.08
        if args.max_tracks_per_movie > 0:
            args.max_tracks_per_movie = min(args.max_tracks_per_movie, 120)
        args.seeds = args.seeds[:1]
        args.temporal_epochs = min(args.temporal_epochs, 3)
        args.flow_epochs = min(args.flow_epochs, 3)
        args.social_epochs = min(args.social_epochs, 3)
        args.joint_epochs = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = arch.select_device(args.device)
    print(f"device={device}", flush=True)
    run_config = vars(args).copy()
    (args.out_dir / "run_config.json").write_text(
        json.dumps(finite_json(run_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    all_rows: list[pd.DataFrame] = []
    all_coverage: dict[str, Any] = {}
    for cell_type in args.cell_types:
        rows, coverage = run_cell_type(
            cell_type,
            table_root=args.table_root,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            max_movies=args.max_movies,
            max_tracks_per_movie=args.max_tracks_per_movie,
            frame_stride=args.frame_stride,
            smooth_window=args.smooth_window,
            crop_fraction=args.crop_fraction,
            r_cut_px=args.r_cut_px,
            horizon=args.horizon,
            seeds=args.seeds,
            variants=args.variants,
            device=device,
            k=args.k,
            temporal_epochs=args.temporal_epochs,
            flow_epochs=args.flow_epochs,
            social_epochs=args.social_epochs,
            joint_epochs=args.joint_epochs,
            batch_size=args.batch_size,
            joint_threshold_pct=args.joint_threshold_pct,
            sequence_balanced_loss=args.sequence_balanced_loss,
        )
        rows.to_csv(args.out_dir / f"lachance_architecture_summary_{cell_type}.csv", index=False)
        all_rows.append(rows)
        all_coverage[cell_type] = coverage
    summary = pd.concat(all_rows, ignore_index=True)
    aggregate = summarize(summary)
    summary.to_csv(args.out_dir / "lachance_architecture_summary.csv", index=False)
    aggregate.to_csv(args.out_dir / "lachance_architecture_aggregate.csv", index=False)
    (args.out_dir / "coverage.json").write_text(
        json.dumps(finite_json(all_coverage), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_gain(aggregate, args.out_dir)
    write_report(summary, aggregate, all_coverage, args.out_dir)
    print(aggregate.to_string(index=False), flush=True)
    print(args.out_dir / "lachance_architecture_report.md", flush=True)


if __name__ == "__main__":
    main()
