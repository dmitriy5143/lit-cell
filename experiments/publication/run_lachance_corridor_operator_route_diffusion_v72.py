#!/usr/bin/env python3
"""Corridor-operator route-state diffusion v72.

This runner keeps the v71 route-state diffusion shell but replaces its weakest
part: the coarse route-conditioned concept packet.  It adds dense candidate
corridor / feasibility operators from v58:

    frozen v52 clean-best anchor
    + fixed v16/v12 route experts
    + route-conditioned concept variables
    + swept-corridor occupancy / contact / free-space feasibility
    + physics-guided residual flow-matching denoiser
    + route energy / sparse mixture

The target/future is used only for training the residual explanation and route
energy labels. Inference features remain causal: v52 anchor, current/past
coordinate context, local tissue-flow features, fixed route hypotheses, and
candidate-conditioned concept variables.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_raw_state_route_architecture_sweep_v26 as v26  # noqa: E402
import run_lachance_route_balanced_calibrator_v16 as v16  # noqa: E402
import run_lachance_route_conditioned_generator_v12 as v12  # noqa: E402
import run_lachance_v52_anchor_route_search_v70 as v70  # noqa: E402
import run_lachance_flow_state_cleanbest_integration_v52 as v52  # noqa: E402
import run_lachance_video_velocity_route_selector_gate_v10 as v10  # noqa: E402
import run_lachance_dense_route_feasibility_v58 as v58  # noqa: E402


DEFAULT_STATE_GRID = (
    ROOT
    / "outputs"
    / "high_coverage_raw_state_v59_aligned_full_bulk_seed42_2026-07-13"
    / "v59_high_coverage_state_grid.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "corridor_operator_route_diffusion_v72_2026-07-15"
EPS = 1e-8


@dataclass
class ConceptPack:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    names: list[str]
    groups: dict[str, list[int]]
    meta: dict[str, Any] | None = None


@dataclass
class ContextPack:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    names: list[str]
    flow_train: np.ndarray
    flow_val: np.ndarray
    flow_test: np.ndarray
    flow_names: list[str]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    context: bool = True
    groups: tuple[str, ...] = ("route", "prior", "self", "flow", "crowding", "smooth", "v52")
    shuffle_concepts: bool = False
    no_route_queries: bool = False
    energy_only: bool = False


class PhysicsGuidedRouteDiffusion(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        concept_dim: int,
        residual_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        basis_count: int,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.concept_dim = int(concept_dim)
        self.residual_dim = int(residual_dim)
        self.basis_count = int(basis_count)
        self.context_proj = nn.Sequential(
            nn.Linear(max(1, self.context_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.concept_proj = nn.Sequential(
            nn.Linear(max(1, self.concept_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.xt_proj = nn.Sequential(
            nn.Linear(self.residual_dim + 5, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        blocks: list[nn.Module] = []
        d = hidden * 3
        for _ in range(max(1, int(layers))):
            blocks.extend([nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(float(dropout))])
            d = hidden
        self.trunk = nn.Sequential(*blocks)
        self.velocity_head = nn.Linear(hidden, self.basis_count * self.residual_dim)
        self.energy_head = nn.Linear(hidden, self.basis_count)
        self.logvar_head = nn.Linear(hidden, 1)
        self.gate_head = nn.Linear(hidden, self.basis_count)

    def forward(
        self,
        context: torch.Tensor,
        concept: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k, _ = concept.shape
        if self.context_dim == 0:
            c0 = torch.zeros((n, 1), dtype=concept.dtype, device=concept.device)
        else:
            c0 = context
        if self.concept_dim == 0:
            p0 = torch.zeros((n, k, 1), dtype=concept.dtype, device=concept.device)
        else:
            p0 = concept
        ctx = self.context_proj(c0).unsqueeze(1).expand(-1, k, -1)
        con = self.concept_proj(p0.reshape(n * k, -1)).reshape(n, k, -1)
        time_feat = torch.cat(
            [
                t,
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
                torch.sin(2.0 * math.pi * t),
                torch.cos(2.0 * math.pi * t),
            ],
            dim=-1,
        )
        xtf = self.xt_proj(torch.cat([xt, time_feat], dim=-1).reshape(n * k, -1)).reshape(n, k, -1)
        h = self.trunk(torch.cat([ctx, con, xtf], dim=-1).reshape(n * k, -1))
        gate_logits = self.gate_head(h).reshape(n, k, self.basis_count)
        gate_p = torch.softmax(gate_logits, dim=-1)
        basis_vel = self.velocity_head(h).reshape(n, k, self.basis_count, self.residual_dim)
        basis_energy = self.energy_head(h).reshape(n, k, self.basis_count)
        vel = torch.sum(gate_p[:, :, :, None] * basis_vel, dim=2)
        energy = torch.sum(gate_p * basis_energy, dim=2)
        logvar = self.logvar_head(h).reshape(n, k)
        return vel, energy, logvar, gate_logits


def parse_strs(text: str | list[str]) -> list[str]:
    if isinstance(text, list):
        return [str(x) for x in text]
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_ints(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return [int(float(s)) for s in parse_strs(text)]


def parse_floats(text: str | list[float]) -> list[float]:
    if isinstance(text, list):
        return [float(x) for x in text]
    return [float(s) for s in parse_strs(text)]


def safe(x: Any) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def flat_to_steps(x: np.ndarray, max_horizon: int) -> np.ndarray:
    return safe(x).reshape(len(x), int(max_horizon), 2)


def endpoint_rmse_flat(pred: np.ndarray, true: np.ndarray, args: argparse.Namespace) -> float:
    return v16.endpoint_rmse_flat(safe(pred), safe(true), args)


def horizon_rmse_flat(pred: np.ndarray, true: np.ndarray, horizon: int, args: argparse.Namespace) -> float:
    p = safe(pred).reshape(len(pred), int(args.max_horizon), 2)[:, : int(horizon), :].sum(axis=1)
    y = safe(true).reshape(len(true), int(args.max_horizon), 2)[:, : int(horizon), :].sum(axis=1)
    return float(np.sqrt(np.mean(np.sum((p - y) ** 2, axis=1))))


def objective_rmse_flat(pred: np.ndarray, true: np.ndarray, args: argparse.Namespace) -> float:
    objective = str(getattr(args, "v72_tune_objective", "endpoint")).lower()
    if objective in {"hmax", "h6", "max_horizon"}:
        return horizon_rmse_flat(pred, true, max(args.horizons), args)
    return endpoint_rmse_flat(pred, true, args)


def residual_endpoint_loss(pred_flat: torch.Tensor, true_flat: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    pred = pred_flat.reshape(pred_flat.shape[0], int(args.max_horizon), 2)
    true = true_flat.reshape(true_flat.shape[0], int(args.max_horizon), 2)
    vals = []
    for h in args.horizons:
        p = pred[:, : int(h)].sum(dim=1)
        y = true[:, : int(h)].sum(dim=1)
        vals.append(F.smooth_l1_loss(p, y))
    vals.append(0.35 * F.smooth_l1_loss(pred_flat, true_flat))
    return torch.stack(vals).mean()


def residual_endpoint_error_matrix(route: np.ndarray, true_flat: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    n, k, _ = route.shape
    pred = route.reshape(n, k, int(args.max_horizon), 2)
    true = true_flat.reshape(n, int(args.max_horizon), 2)
    err = np.zeros((n, k), dtype=np.float32)
    for h in args.horizons:
        h = int(h)
        p = pred[:, :, :h].sum(axis=2)
        y = true[:, :h].sum(axis=1)[:, None, :]
        err += np.sum((p - y) ** 2, axis=-1).astype(np.float32)
    return np.sqrt(err / max(len(args.horizons), 1)).astype(np.float32)


def build_route_basis(args: argparse.Namespace, out_dir: Path) -> tuple[v26.RouteBasis, dict[str, Any], pd.DataFrame]:
    device = closure.device_from_arg(args.device)
    arrays, labels, prior, bank, _packs, gate, meta = v16.build_route_data(args, device)
    split_arrays, split = audit.prepare_data(args)
    if len(split_arrays.residual_train) != len(arrays.residual_train):
        raise RuntimeError("prepare_data mismatch while building v72 route basis")
    ytr = audit.flatten_residual(arrays.residual_train).astype(np.float32)
    yva = audit.flatten_residual(arrays.residual_val).astype(np.float32)
    yte = audit.flatten_residual(arrays.residual_test).astype(np.float32)
    rtr = v16.route_outputs(bank, prior.x_train)
    rva = v16.route_outputs(bank, prior.x_val)
    rte = v16.route_outputs(bank, prior.x_test)
    oracle_tr = v26.route_oracle_labels(rtr, ytr, args)
    oracle_va = v26.route_oracle_labels(rva, yva, args)
    oracle_te = v26.route_oracle_labels(rte, yte, args)
    route_meta = {
        "v16_gate": gate.to_dict(orient="records") if isinstance(gate, pd.DataFrame) else [],
        "v16_meta": meta,
        "route_count": int(rtr.shape[1]),
        "route_dim": int(rtr.shape[2]),
        "feature_dim": int(prior.x_train.shape[1]),
    }
    (out_dir / "v72_route_basis_meta.json").write_text(json.dumps(audit.finite_json(route_meta), indent=2), encoding="utf-8")
    basis = v26.RouteBasis(
        arrays=arrays,
        split=split,
        labels=labels,
        prior=prior,
        bank=bank,
        route_train=rtr,
        route_val=rva,
        route_test=rte,
        y_train=ytr,
        y_val=yva,
        y_test=yte,
        oracle_labels_train=oracle_tr,
        oracle_labels_val=oracle_va,
        oracle_labels_test=oracle_te,
        route_data_meta=route_meta,
    )
    return basis, route_meta, gate


def select_context(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray, max_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return v52.select_context(safe(xtr), safe(xva), safe(xte), max_cols)


def prepare_context_pack(args: argparse.Namespace, basis: v26.RouteBasis) -> ContextPack:
    xtr0, xva0, xte0 = select_context(
        basis.prior.x_train,
        basis.prior.x_val,
        basis.prior.x_test,
        int(args.v16c_max_context_features),
    )
    split_flow, flow_cols = v52.prepare_sampled_split_with_flow(args)
    packets = v52.build_flow_packets(split_flow, flow_cols, args)
    real = packets["clean_best_real_flow"]
    xtr = np.concatenate([xtr0, real.train], axis=1).astype(np.float32)
    xva = np.concatenate([xva0, real.val], axis=1).astype(np.float32)
    xte = np.concatenate([xte0, real.test], axis=1).astype(np.float32)
    names = [f"context_{i}" for i in range(xtr0.shape[1])] + real.feature_names
    return ContextPack(
        train=xtr,
        val=xva,
        test=xte,
        names=names,
        flow_train=real.train,
        flow_val=real.val,
        flow_test=real.test,
        flow_names=real.feature_names,
    )


def cos2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), EPS)
    return np.sum(a * b, axis=-1) / denom


def add_feat(parts: list[np.ndarray], names: list[str], groups: dict[str, list[int]], group: str, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    start = len(names)
    parts.append(arr.astype(np.float32))
    for j in range(arr.shape[2]):
        names.append(f"{group}_{name}" if arr.shape[2] == 1 else f"{group}_{name}_{j}")
    groups.setdefault(group, []).extend(range(start, start + arr.shape[2]))


def build_concepts_for_split(
    *,
    route: np.ndarray,
    anchor: np.ndarray,
    probs: np.ndarray,
    base: np.ndarray,
    flow: np.ndarray,
    flow_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    n, k, d = route.shape
    h = int(args.max_horizon)
    route_steps = route.reshape(n, k, h, 2)
    anchor_steps = anchor.reshape(n, h, 2)
    route_delta = route - anchor[:, None, :]
    delta_steps = route_delta.reshape(n, k, h, 2)
    full_steps = route_steps + base[:, None, None, :]
    mean_step = full_steps.mean(axis=2)
    endpoint = full_steps.sum(axis=2)
    anchor_full = anchor_steps + base[:, None, :]
    anchor_endpoint = anchor_full.sum(axis=1)
    endpoint_norm = np.linalg.norm(endpoint, axis=2)
    step_norm = np.linalg.norm(full_steps, axis=3)
    base_vec = base[:, None, :]
    base_speed = np.linalg.norm(base, axis=1)[:, None]

    parts: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[int]] = {}

    rid = np.tile((np.arange(k, dtype=np.float32) / max(k - 1, 1))[None, :, None], (n, 1, 1))
    rank = np.argsort(np.argsort(-probs, axis=1), axis=1).astype(np.float32) / max(k - 1, 1)
    ent = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=1, keepdims=True)
    add_feat(parts, names, groups, "prior", "prob", probs)
    add_feat(parts, names, groups, "prior", "rank", rank)
    add_feat(parts, names, groups, "prior", "route_id", rid)
    add_feat(parts, names, groups, "prior", "entropy", np.repeat(ent[:, None, :], k, axis=1))

    add_feat(parts, names, groups, "self", "cos", cos2(mean_step, np.repeat(base_vec, k, axis=1)))
    add_feat(parts, names, groups, "self", "mismatch", np.linalg.norm(mean_step - base_vec, axis=2))
    add_feat(parts, names, groups, "self", "base_speed", np.repeat(base_speed[:, :, None], k, axis=1))
    add_feat(parts, names, groups, "self", "candidate_step_speed", np.mean(step_norm, axis=2))

    delta_endpoint = endpoint - anchor_endpoint[:, None, :]
    add_feat(parts, names, groups, "v52", "delta_endpoint_norm", np.linalg.norm(delta_endpoint, axis=2))
    add_feat(parts, names, groups, "v52", "endpoint_cos_to_anchor", cos2(endpoint, np.repeat(anchor_endpoint[:, None, :], k, axis=1)))
    add_feat(parts, names, groups, "route", "delta_flat_norm", np.linalg.norm(route_delta, axis=2))
    add_feat(parts, names, groups, "route", "delta_endpoint_x", delta_endpoint[:, :, 0])
    add_feat(parts, names, groups, "route", "delta_endpoint_y", delta_endpoint[:, :, 1])

    pf = v70.path_features(route, base, args)
    for j, nm in enumerate(["path_norm", "endpoint_norm", "step_max", "accel_mean", "turn_mean", "jump_excess", "efficiency"]):
        add_feat(parts, names, groups, "smooth", nm, pf[:, :, j])

    idx = {name: i for i, name in enumerate(flow_names)}
    for kk in parse_ints(args.v71_flow_ks):
        px = f"v52tf_k{kk}_mean_dx"
        py = f"v52tf_k{kk}_mean_dy"
        if px in idx and py in idx:
            fv = flow[:, [idx[px], idx[py]]].astype(np.float32)
            fv_rep = np.repeat(fv[:, None, :], k, axis=1)
            add_feat(parts, names, groups, "flow", f"k{kk}_cos", cos2(mean_step, fv_rep))
            add_feat(parts, names, groups, "flow", f"k{kk}_mismatch", np.linalg.norm(mean_step - fv_rep, axis=2))
            add_feat(parts, names, groups, "flow", f"k{kk}_speed", np.repeat(np.linalg.norm(fv, axis=1)[:, None, None], k, axis=1))
        for suffix in ["dist_min", "dist_mean", "pos_anisotropy", "vel_anisotropy"]:
            cname = f"v52tf_k{kk}_{suffix}"
            if cname in idx:
                scalar = flow[:, idx[cname]][:, None]
                rep = np.repeat(scalar[:, None, :], k, axis=1)
                add_feat(parts, names, groups, "crowding", f"k{kk}_{suffix}", rep)
                if suffix == "dist_min":
                    pressure = endpoint_norm / np.maximum(np.repeat(scalar, k, axis=1), 1.0)
                    add_feat(parts, names, groups, "crowding", f"k{kk}_endpoint_over_distmin", pressure)
    for rr in parse_ints(args.v71_density_radii):
        cname = f"v52tf_r{rr}_density"
        if cname in idx:
            den = flow[:, idx[cname]][:, None]
            rep = np.repeat(den[:, None, :], k, axis=1)
            add_feat(parts, names, groups, "crowding", f"r{rr}_density", rep)
            add_feat(parts, names, groups, "crowding", f"r{rr}_density_x_endpoint", np.repeat(den, k, axis=1) * endpoint_norm)

    out = np.concatenate(parts, axis=2).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), names, groups


def append_group_features(
    base: np.ndarray,
    names: list[str],
    groups: dict[str, list[int]],
    group: str,
    feature_names: list[str],
    arr: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[:2] != base.shape[:2]:
        raise RuntimeError(f"cannot append {group}: shape {arr.shape} incompatible with {base.shape}")
    start = len(names)
    out_names = list(names)
    out_groups = {k: list(v) for k, v in groups.items()}
    for j, fname in enumerate(feature_names):
        out_names.append(f"{group}_{fname}")
    out_groups.setdefault(group, []).extend(range(start, start + arr.shape[2]))
    out = np.concatenate([base, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)], axis=2).astype(np.float32)
    return out, out_names, out_groups


def build_corridor_operator_triplet(
    args: argparse.Namespace,
    basis: v26.RouteBasis,
    route_train: np.ndarray,
    route_val: np.ndarray,
    route_test: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[str], dict[str, Any]]:
    """Dense candidate-corridor feasibility operators for v72.

    This reuses v58's field construction, but treats the output as a concept
    packet inside the v71-style diffusion/router rather than as a standalone
    scalar route filter.  Only causal same-frame state is used.
    """

    if not bool(getattr(args, "v72_use_corridor_operators", True)):
        ztr = np.zeros((len(route_train), route_train.shape[1], 0), dtype=np.float32)
        zva = np.zeros((len(route_val), route_val.shape[1], 0), dtype=np.float32)
        zte = np.zeros((len(route_test), route_test.shape[1], 0), dtype=np.float32)
        return (ztr, zva, zte), [], {"corridor_enabled": False}

    state_paths = [Path(p).expanduser() for p in parse_strs(str(args.v72_state_grid))]
    existing = [p for p in state_paths if p.exists()]
    if not existing:
        # v58 can still build adaptive tracking-derived radii with an empty
        # state grid, but record that this is no longer a true visual-state run.
        state_spec = ""
    else:
        state_spec = ",".join(str(p) for p in existing)

    pos = v58.v57.load_position_table(args)
    state = v58.v57.load_state_grids(state_spec)
    field = v58.build_dense_field(pos, state, args)
    split_state_train, cov_tr = v58.split_state(basis.split.train, field)
    split_state_val, cov_va = v58.split_state(basis.split.val, field)
    split_state_test, cov_te = v58.split_state(basis.split.test, field)

    mode_parts_train: list[np.ndarray] = []
    mode_parts_val: list[np.ndarray] = []
    mode_parts_test: list[np.ndarray] = []
    feature_names: list[str] = []
    stats_rows: list[dict[str, Any]] = []
    for mode in parse_strs(args.v72_corridor_occupancy_modes):
        ev_tr = v58.compute_dense_evidence_split(
            split_df=basis.split.train,
            split_state_df=split_state_train,
            field=field,
            route_residual=route_train,
            base=basis.arrays.base_train,
            args=args,
            control="real",
            occupancy_mode=mode,
            seed=int(args.seed) + 72011,
        )
        ev_va = v58.compute_dense_evidence_split(
            split_df=basis.split.val,
            split_state_df=split_state_val,
            field=field,
            route_residual=route_val,
            base=basis.arrays.base_val,
            args=args,
            control="real",
            occupancy_mode=mode,
            seed=int(args.seed) + 72012,
        )
        ev_te = v58.compute_dense_evidence_split(
            split_df=basis.split.test,
            split_state_df=split_state_test,
            field=field,
            route_residual=route_test,
            base=basis.arrays.base_test,
            args=args,
            control="real",
            occupancy_mode=mode,
            seed=int(args.seed) + 72013,
        )
        raw_names = [f"{mode}_{n}" for n in ev_tr.names]
        aux_names = [
            f"{mode}_rule_risk",
            f"{mode}_hard_violation",
            f"{mode}_coverage",
        ]
        mode_parts_train.append(
            np.concatenate(
                [
                    ev_tr.raw,
                    ev_tr.rule_risk[:, :, None],
                    ev_tr.hard_violation.astype(np.float32)[:, :, None],
                    ev_tr.coverage[:, None, None].repeat(ev_tr.raw.shape[1], axis=1),
                ],
                axis=2,
            )
        )
        mode_parts_val.append(
            np.concatenate(
                [
                    ev_va.raw,
                    ev_va.rule_risk[:, :, None],
                    ev_va.hard_violation.astype(np.float32)[:, :, None],
                    ev_va.coverage[:, None, None].repeat(ev_va.raw.shape[1], axis=1),
                ],
                axis=2,
            )
        )
        mode_parts_test.append(
            np.concatenate(
                [
                    ev_te.raw,
                    ev_te.rule_risk[:, :, None],
                    ev_te.hard_violation.astype(np.float32)[:, :, None],
                    ev_te.coverage[:, None, None].repeat(ev_te.raw.shape[1], axis=1),
                ],
                axis=2,
            )
        )
        feature_names.extend(raw_names + aux_names)
        stats_rows.append(
            {
                "occupancy_mode": mode,
                "train_feature_dim": int(ev_tr.raw.shape[2]),
                "test_rule_risk_mean": float(np.mean(ev_te.rule_risk)),
                "test_hard_violation_frac": float(np.mean(ev_te.hard_violation)),
                "test_corridor_collision_mean": float(np.mean(ev_te.raw[:, :, ev_te.names.index("corridor_collision_frac")])),
                "test_corridor_free_mean": float(np.mean(ev_te.raw[:, :, ev_te.names.index("corridor_free_frac")])),
                "test_swept_min_margin_mean": float(np.mean(ev_te.raw[:, :, ev_te.names.index("swept_min_margin")])),
            }
        )

    tr = np.concatenate(mode_parts_train, axis=2).astype(np.float32)
    va = np.concatenate(mode_parts_val, axis=2).astype(np.float32)
    te = np.concatenate(mode_parts_test, axis=2).astype(np.float32)
    meta = {
        "corridor_enabled": True,
        "corridor_state_paths": [str(p) for p in state_paths],
        "corridor_existing_state_paths": [str(p) for p in existing],
        "corridor_state_coverage_full": float(field.stats.get("mask_state_coverage", 0.0)),
        "corridor_state_coverage_train": float(cov_tr),
        "corridor_state_coverage_val": float(cov_va),
        "corridor_state_coverage_test": float(cov_te),
        "corridor_field_rows": int(field.stats.get("dense_rows", 0)),
        "corridor_feature_dim": int(tr.shape[2]),
        "corridor_modes": parse_strs(args.v72_corridor_occupancy_modes),
    }
    pd.DataFrame([field.stats]).to_csv(args.out_dir / "v72_corridor_field_stats.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(args.out_dir / "v72_corridor_operator_stats.csv", index=False)
    return (tr, va, te), feature_names, meta


def build_concepts(
    args: argparse.Namespace,
    basis: v26.RouteBasis,
    anchor: dict[str, np.ndarray],
    context: ContextPack,
    route_train: np.ndarray,
    route_val: np.ndarray,
    route_test: np.ndarray,
    probs_train: np.ndarray,
    probs_val: np.ndarray,
    probs_test: np.ndarray,
) -> ConceptPack:
    tr, names, groups = build_concepts_for_split(
        route=route_train,
        anchor=anchor["train"],
        probs=probs_train,
        base=basis.arrays.base_train,
        flow=context.flow_train,
        flow_names=context.flow_names,
        args=args,
    )
    va, _, _ = build_concepts_for_split(
        route=route_val,
        anchor=anchor["val"],
        probs=probs_val,
        base=basis.arrays.base_val,
        flow=context.flow_val,
        flow_names=context.flow_names,
        args=args,
    )
    te, _, _ = build_concepts_for_split(
        route=route_test,
        anchor=anchor["test"],
        probs=probs_test,
        base=basis.arrays.base_test,
        flow=context.flow_test,
        flow_names=context.flow_names,
        args=args,
    )
    corridor_meta: dict[str, Any] = {"corridor_enabled": False}
    if bool(getattr(args, "v72_use_corridor_operators", False)):
        (cor_tr, cor_va, cor_te), cor_names, corridor_meta = build_corridor_operator_triplet(
            args,
            basis,
            route_train,
            route_val,
            route_test,
        )
        if cor_names:
            tr, names, groups = append_group_features(tr, names, groups, "corridor", cor_names, cor_tr)
            va = np.concatenate([va, np.nan_to_num(cor_va, nan=0.0, posinf=0.0, neginf=0.0)], axis=2).astype(np.float32)
            te = np.concatenate([te, np.nan_to_num(cor_te, nan=0.0, posinf=0.0, neginf=0.0)], axis=2).astype(np.float32)
    return ConceptPack(train=tr, val=va, test=te, names=names, groups=groups, meta=corridor_meta)


def variant_specs(names: list[str]) -> list[VariantSpec]:
    all_groups = ("route", "prior", "self", "flow", "crowding", "smooth", "v52", "corridor")
    specs = {
        "full": VariantSpec("full", context=True, groups=all_groups),
        "blackbox_only": VariantSpec("blackbox_only", context=True, groups=("route", "prior", "v52")),
        "corridor_only": VariantSpec("corridor_only", context=True, groups=("route", "prior", "v52", "corridor")),
        "corridor_no_context": VariantSpec("corridor_no_context", context=False, groups=("route", "prior", "v52", "corridor")),
        "corridor_flow": VariantSpec("corridor_flow", context=True, groups=("route", "prior", "v52", "corridor", "flow")),
        "corridor_smooth": VariantSpec("corridor_smooth", context=True, groups=("route", "prior", "v52", "corridor", "smooth")),
        "corridor_crowding": VariantSpec("corridor_crowding", context=True, groups=("route", "prior", "v52", "corridor", "crowding")),
        "physics_only": VariantSpec("physics_only", context=False, groups=all_groups),
        "no_flow": VariantSpec("no_flow", context=True, groups=tuple(g for g in all_groups if g != "flow")),
        "no_crowding": VariantSpec("no_crowding", context=True, groups=tuple(g for g in all_groups if g != "crowding")),
        "no_corridor": VariantSpec("no_corridor", context=True, groups=tuple(g for g in all_groups if g != "corridor")),
        "no_route_queries": VariantSpec("no_route_queries", context=True, groups=all_groups, no_route_queries=True),
        "shuffled_physics": VariantSpec("shuffled_physics", context=True, groups=all_groups, shuffle_concepts=True),
        "energy_only": VariantSpec("energy_only", context=True, groups=all_groups, energy_only=True),
    }
    out = []
    for name in names:
        if name not in specs:
            raise ValueError(f"Unknown v72 variant {name}. Known: {sorted(specs)}")
        out.append(specs[name])
    return out


def apply_variant(
    spec: VariantSpec,
    context: ContextPack,
    concepts: ConceptPack,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    if spec.context:
        xtr, xva, xte = context.train.copy(), context.val.copy(), context.test.copy()
        ctx_names = context.names
    else:
        xtr = np.zeros((len(context.train), 0), dtype=np.float32)
        xva = np.zeros((len(context.val), 0), dtype=np.float32)
        xte = np.zeros((len(context.test), 0), dtype=np.float32)
        ctx_names = []
    keep: list[int] = []
    for group in spec.groups:
        keep.extend(concepts.groups.get(group, []))
    keep = sorted(set(keep))
    ctr = concepts.train[:, :, keep].copy() if keep else np.zeros((len(concepts.train), concepts.train.shape[1], 0), dtype=np.float32)
    cva = concepts.val[:, :, keep].copy() if keep else np.zeros((len(concepts.val), concepts.val.shape[1], 0), dtype=np.float32)
    cte = concepts.test[:, :, keep].copy() if keep else np.zeros((len(concepts.test), concepts.test.shape[1], 0), dtype=np.float32)
    c_names = [concepts.names[i] for i in keep]
    if spec.no_route_queries:
        ctr = np.repeat(ctr.mean(axis=1, keepdims=True), ctr.shape[1], axis=1)
        cva = np.repeat(cva.mean(axis=1, keepdims=True), cva.shape[1], axis=1)
        cte = np.repeat(cte.mean(axis=1, keepdims=True), cte.shape[1], axis=1)
    if spec.shuffle_concepts:
        rng = np.random.default_rng(int(args.seed) + 71001)
        ctr = ctr[rng.permutation(len(ctr))]
        cva = cva[rng.permutation(len(cva))]
        cte = cte[rng.permutation(len(cte))]
    return xtr, xva, xte, ctr, cva, cte, ctx_names, c_names


def standardize_context(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xtr.shape[1] == 0:
        return xtr.astype(np.float32), xva.astype(np.float32), xte.astype(np.float32)
    sc = StandardScaler()
    ztr = np.clip(sc.fit_transform(xtr), -8.0, 8.0).astype(np.float32)
    zva = np.clip(sc.transform(xva), -8.0, 8.0).astype(np.float32)
    zte = np.clip(sc.transform(xte), -8.0, 8.0).astype(np.float32)
    return ztr, zva, zte


def standardize_concepts(ctr: np.ndarray, cva: np.ndarray, cte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ctr.shape[2] == 0:
        return ctr.astype(np.float32), cva.astype(np.float32), cte.astype(np.float32)
    sc = StandardScaler()
    n, k, f = ctr.shape
    ztr = sc.fit_transform(ctr.reshape(n * k, f)).reshape(n, k, f)
    zva = sc.transform(cva.reshape(cva.shape[0] * k, f)).reshape(cva.shape[0], k, f)
    zte = sc.transform(cte.reshape(cte.shape[0] * k, f)).reshape(cte.shape[0], k, f)
    return (
        np.clip(np.nan_to_num(ztr), -8.0, 8.0).astype(np.float32),
        np.clip(np.nan_to_num(zva), -8.0, 8.0).astype(np.float32),
        np.clip(np.nan_to_num(zte), -8.0, 8.0).astype(np.float32),
    )


def soft_oracle_labels(err: np.ndarray, temperature: float) -> np.ndarray:
    e = err - np.min(err, axis=1, keepdims=True)
    logits = -e / max(float(temperature), 1e-6)
    logits -= np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits)
    return (p / np.maximum(p.sum(axis=1, keepdims=True), EPS)).astype(np.float32)


def topm_softmax(logits: np.ndarray, top_m: int, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32) / max(float(temperature), 1e-6)
    if 0 < int(top_m) < z.shape[1]:
        order = np.argsort(-z, axis=1)[:, : int(top_m)]
        mask = np.full_like(z, -1.0e9, dtype=np.float32)
        rows = np.arange(z.shape[0])[:, None]
        mask[rows, order] = z[rows, order]
        z = mask
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return (p / np.maximum(p.sum(axis=1, keepdims=True), EPS)).astype(np.float32)


def tune_mixture(
    candidate: np.ndarray,
    energy: np.ndarray,
    anchor: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    best_pred = anchor.copy()
    best_meta: dict[str, Any] = {
        "temperature": None,
        "top_m": None,
        "anchor_blend": 0.0,
        "val_rmse": objective_rmse_flat(anchor, y_val, args),
        "val_endpoint_rmse": endpoint_rmse_flat(anchor, y_val, args),
    }
    logits = -energy
    for top_m in parse_ints(args.v71_top_m_grid):
        for temp in parse_floats(args.v71_temperature_grid):
            w = topm_softmax(logits, top_m=top_m, temperature=temp)
            mix = np.sum(w[:, :, None] * candidate, axis=1).astype(np.float32)
            for blend in parse_floats(args.v71_anchor_blend_grid):
                pred = ((1.0 - float(blend)) * anchor + float(blend) * mix).astype(np.float32)
                rmse = objective_rmse_flat(pred, y_val, args)
                if rmse < best_meta["val_rmse"]:
                    best_meta = {
                        "temperature": float(temp),
                        "top_m": int(top_m),
                        "anchor_blend": float(blend),
                        "val_rmse": float(rmse),
                        "val_endpoint_rmse": float(endpoint_rmse_flat(pred, y_val, args)),
                    }
                    best_pred = pred.astype(np.float32)
    return best_pred, best_meta


def predict_corrections(
    model: PhysicsGuidedRouteDiffusion,
    context: np.ndarray,
    concept: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    corr_out: list[np.ndarray] = []
    energy_out: list[np.ndarray] = []
    gate_out: list[np.ndarray] = []
    n, k, _ = concept.shape
    d = int(args.max_horizon) * 2
    with torch.no_grad():
        for idx in closure.batches(n, int(args.v71_batch_size), int(args.seed) + 71900, shuffle=False):
            xb = to_tensor(context[idx], device)
            cb = to_tensor(concept[idx], device)
            x = torch.zeros((len(idx), k, d), dtype=torch.float32, device=device)
            steps = max(1, int(args.v71_inference_steps))
            energy = None
            gates = None
            for s in range(steps):
                tt = torch.full((len(idx), k, 1), float(s) / float(steps), dtype=torch.float32, device=device)
                vel, energy, _logvar, gates = model(xb, cb, x, tt)
                x = x + vel / float(steps)
            assert energy is not None and gates is not None
            corr_out.append(x.cpu().numpy().astype(np.float32))
            energy_out.append(energy.cpu().numpy().astype(np.float32))
            gate_out.append(torch.softmax(gates, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(corr_out, axis=0), np.concatenate(energy_out, axis=0), np.concatenate(gate_out, axis=0)


def physics_consistency_loss(energy: torch.Tensor, concept: torch.Tensor, concept_names: list[str], args: argparse.Namespace) -> torch.Tensor:
    if concept.shape[-1] == 0 or float(args.v71_physics_consistency_weight) <= 0:
        return torch.zeros((), dtype=energy.dtype, device=energy.device)
    losses = []
    e = (energy - energy.mean(dim=1, keepdim=True)) / torch.clamp(energy.std(dim=1, keepdim=True), min=1e-3)
    for j, name in enumerate(concept_names):
        x = concept[:, :, j]
        if torch.std(x) < 1e-3:
            continue
        z = (x - x.mean(dim=1, keepdim=True)) / torch.clamp(x.std(dim=1, keepdim=True), min=1e-3)
        cov = torch.mean(e * z)
        if "flow_" in name and "cos" in name:
            losses.append(F.relu(cov))  # better flow agreement should not increase energy
        if "crowding_" in name and ("density_x_endpoint" in name or "endpoint_over_distmin" in name):
            losses.append(F.relu(-cov))  # more pressure / close passage should not lower energy
        if "corridor_" in name and any(
            key in name
            for key in [
                "overlap",
                "collision",
                "pressure",
                "close_frac",
                "hard_violation",
                "rule_risk",
                "same_track_jump_guard",
            ]
        ):
            losses.append(F.relu(-cov))  # infeasible corridors should not lower energy
        if "corridor_" in name and any(key in name for key in ["free_frac", "free_support", "clearance", "reliability"]):
            losses.append(F.relu(cov))  # better clearance/free-space should not raise energy
        if "smooth_jump_excess" in name or "smooth_accel_mean" in name or "smooth_turn_mean" in name:
            losses.append(F.relu(-cov))
    if not losses:
        return torch.zeros((), dtype=energy.dtype, device=energy.device)
    return torch.stack(losses).mean()


def train_variant(
    *,
    spec: VariantSpec,
    xtr: np.ndarray,
    xva: np.ndarray,
    ctr: np.ndarray,
    cva: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    anchor_tr: np.ndarray,
    anchor_va: np.ndarray,
    route_tr: np.ndarray,
    route_va: np.ndarray,
    concept_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PhysicsGuidedRouteDiffusion, pd.DataFrame, dict[str, Any]]:
    n, k, _ = ctr.shape
    d = int(args.max_horizon) * 2
    model = PhysicsGuidedRouteDiffusion(
        context_dim=xtr.shape[1],
        concept_dim=ctr.shape[2],
        residual_dim=d,
        hidden=int(args.v71_hidden),
        layers=int(args.v71_layers),
        dropout=float(args.v71_dropout),
        basis_count=len(parse_strs(args.v71_basis_names)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.v71_lr), weight_decay=float(args.v71_weight_decay))
    target_corr_tr = (ytr[:, None, :] - route_tr).astype(np.float32)
    target_corr_va = (yva[:, None, :] - route_va).astype(np.float32)
    err_tr = residual_endpoint_error_matrix(route_tr, ytr, args)
    err_va = residual_endpoint_error_matrix(route_va, yva, args)
    soft_tr = soft_oracle_labels(err_tr, float(args.v71_oracle_temperature))
    soft_va = soft_oracle_labels(err_va, float(args.v71_oracle_temperature))
    if spec.energy_only:
        target_corr_tr = np.zeros_like(target_corr_tr)
        target_corr_va = np.zeros_like(target_corr_va)

    rows: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(int(args.v71_epochs)):
        model.train()
        losses = []
        for idx in closure.batches(n, int(args.v71_batch_size), int(args.seed) + 7100 + epoch):
            xb = to_tensor(xtr[idx], device)
            cb = to_tensor(ctr[idx], device)
            target = to_tensor(target_corr_tr[idx], device)
            route_base = to_tensor(route_tr[idx], device)
            soft = to_tensor(soft_tr[idx], device)
            bsz = len(idx)
            z = torch.randn((bsz, k, d), dtype=torch.float32, device=device) * float(args.v71_noise_scale)
            t = torch.rand((bsz, 1, 1), dtype=torch.float32, device=device).expand(-1, k, -1)
            xt = (1.0 - t) * z + t * target
            vel_true = target - z
            vel, energy, logvar, gates = model(xb, cb, xt, t)
            fm = torch.sum(soft[:, :, None] * (vel - vel_true).pow(2)) / torch.clamp(torch.sum(soft) * d, min=1.0)
            corr_pred = xt + (1.0 - t) * vel
            route_logits = -energy
            weights = torch.softmax(route_logits / max(float(args.v71_train_temperature), 1e-3), dim=1)
            mix_corr = torch.sum(weights[:, :, None] * corr_pred, dim=1)
            mix_route = torch.sum(weights[:, :, None] * route_base, dim=1)
            final_loss = residual_endpoint_loss(mix_route + mix_corr, to_tensor(ytr[idx], device), args)
            ce = -torch.mean(torch.sum(soft * F.log_softmax(route_logits, dim=1), dim=1))
            entropy = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, min=1e-8)), dim=1))
            gate_p = torch.softmax(gates, dim=-1)
            gate_entropy = -torch.mean(torch.sum(gate_p * torch.log(torch.clamp(gate_p, min=1e-8)), dim=-1))
            phys = physics_consistency_loss(energy, cb, concept_names, args)
            nll_proxy = torch.mean(torch.exp(-logvar) * torch.mean((corr_pred - target[:, None, :]).pow(2), dim=-1) + logvar)
            loss = (
                float(args.v71_fm_weight) * fm
                + float(args.v71_final_weight) * final_loss
                + float(args.v71_energy_weight) * ce
                + float(args.v71_nll_weight) * nll_proxy
                + float(args.v71_physics_consistency_weight) * phys
                + float(args.v71_gate_sparsity_weight) * gate_entropy
                - float(args.v71_entropy_weight) * entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.v71_grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == int(args.v71_epochs) - 1 or epoch % max(1, int(args.v71_epochs) // 5) == 0:
            corr_va, energy_va, gates_va = predict_corrections(model, xva, cva, args, device)
            cand_va = route_va + corr_va
            pred_va, tune_meta = tune_mixture(cand_va, energy_va, anchor_va, yva, args)
            val_rmse = objective_rmse_flat(pred_va, yva, args)
            val_true_endpoint = endpoint_rmse_flat(pred_va, yva, args)
            energy_ce = float(-np.mean(np.sum(soft_va * np.log(np.maximum(topm_softmax(-energy_va, 0, 1.0), EPS)), axis=1)))
            rows.append(
                {
                    "variant": spec.name,
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(losses)),
                    "val_endpoint_rmse": float(val_rmse),
                    "val_objective": str(getattr(args, "v72_tune_objective", "endpoint")),
                    "val_true_endpoint_rmse": float(val_true_endpoint),
                    "val_energy_ce": energy_ce,
                    "val_best_temperature": tune_meta.get("temperature"),
                    "val_best_top_m": tune_meta.get("top_m"),
                    "val_anchor_blend": tune_meta.get("anchor_blend"),
                    "gate_entropy_mean": float(-np.mean(np.sum(gates_va * np.log(np.maximum(gates_va, EPS)), axis=-1))),
                }
            )
            if val_rmse < best_val:
                best_val = float(val_rmse)
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), {"best_val_rmse": best_val}


def evaluate_variant(
    *,
    model: PhysicsGuidedRouteDiffusion,
    spec: VariantSpec,
    xva: np.ndarray,
    xte: np.ndarray,
    cva: np.ndarray,
    cte: np.ndarray,
    anchor_va: np.ndarray,
    anchor_te: np.ndarray,
    route_va: np.ndarray,
    route_te: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    basis: v26.RouteBasis,
    concept_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corr_va, energy_va, gates_va = predict_corrections(model, xva, cva, args, device)
    cand_va = route_va + corr_va
    _pred_va, tune = tune_mixture(cand_va, energy_va, anchor_va, yva, args)
    corr_te, energy_te, gates_te = predict_corrections(model, xte, cte, args, device)
    weights = topm_softmax(-energy_te, top_m=int(tune["top_m"]), temperature=float(tune["temperature"]))
    cand_te = route_te + corr_te
    mix = np.sum(weights[:, :, None] * cand_te, axis=1).astype(np.float32)
    pred = ((1.0 - float(tune.get("anchor_blend", 1.0))) * anchor_te + float(tune.get("anchor_blend", 1.0)) * mix).astype(np.float32)
    pred_corr = pred - anchor_te
    rows = v16.metric_rows(
        basis.arrays,
        flat_to_steps(pred, args.max_horizon),
        f"v72_{spec.name}",
        args,
        {
            "stage": "v72_corridor_operator_route_diffusion",
            "variant": spec.name,
            "temperature": tune["temperature"],
            "top_m": tune["top_m"],
            "anchor_blend": tune.get("anchor_blend", 1.0),
            "val_endpoint_rmse": tune["val_rmse"],
            "val_objective": str(getattr(args, "v72_tune_objective", "endpoint")),
            "val_true_endpoint_rmse": tune.get("val_endpoint_rmse", tune["val_rmse"]),
            "context_enabled": bool(spec.context),
            "concept_groups": ",".join(spec.groups),
            "shuffle_concepts": bool(spec.shuffle_concepts),
            "no_route_queries": bool(spec.no_route_queries),
        },
    )
    diag = pd.DataFrame(
        [
            {
                "variant": spec.name,
                "val_endpoint_rmse": tune["val_rmse"],
                "val_objective": str(getattr(args, "v72_tune_objective", "endpoint")),
                "val_true_endpoint_rmse": tune.get("val_endpoint_rmse", tune["val_rmse"]),
                "temperature": tune["temperature"],
                "top_m": tune["top_m"],
                "anchor_blend": tune.get("anchor_blend", 1.0),
                "weight_entropy_mean": float(-np.mean(np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1))),
                "active_routes_mean": float(np.mean(np.sum(weights > 1e-4, axis=1))),
                "energy_mean": float(np.mean(energy_te)),
                "energy_std": float(np.std(energy_te)),
                "corr_norm_mean": float(np.mean(np.linalg.norm(pred_corr, axis=1))),
            }
        ]
    )
    basis_names = parse_strs(args.v71_basis_names)
    gate_mean = gates_te.mean(axis=(0, 1))
    gate_rows = []
    for i, value in enumerate(gate_mean):
        gate_rows.append({"variant": spec.name, "basis": basis_names[i] if i < len(basis_names) else f"basis_{i}", "gate_mean": float(value)})
    return pd.DataFrame(rows), diag, pd.DataFrame(gate_rows)


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, train_log: pd.DataFrame, gates: pd.DataFrame) -> None:
    lines = ["# v72 Corridor-Operator Route Diffusion", ""]
    lines.append("## Decision Snapshot")
    hmax = max(args.horizons)
    sub = summary[summary["horizon"].eq(hmax)].sort_values("rmse")
    cols = [c for c in ["method", "rmse", "r2", "stage", "variant", "temperature", "top_m", "anchor_blend", "val_endpoint_rmse"] if c in sub.columns]
    if not sub.empty:
        lines.append(sub[cols].head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Diagnostics")
    if not diag.empty:
        lines.append(diag.sort_values("val_endpoint_rmse").head(40).to_markdown(index=False))
    lines.append("")
    lines.append("## Train Log Tail")
    if not train_log.empty:
        lines.append(train_log.groupby("variant", group_keys=False).tail(5).to_markdown(index=False))
    lines.append("")
    lines.append("## Physics Gates")
    if not gates.empty:
        piv = gates.pivot_table(index="variant", columns="basis", values="gate_mean", aggfunc="mean").reset_index()
        lines.append(piv.to_markdown(index=False))
    lines.append("")
    lines.append("## Reading")
    lines.extend(
        [
            "- Pass requires `full` to beat frozen v52 and blackbox/no-corridor/shuffled controls on h6 without h1 degradation.",
            "- If `no_corridor` matches `full`, corridor feasibility is not validated.",
            "- If `blackbox_only` matches `full`, the learned residual channel helps but explicit operators are not validated.",
            "- If `shuffled_physics` matches `full`, route-conditioned concept construction is not causal yet.",
            "- If all v72 variants lose to v52, keep v52 as the final coordinate-side component and do not stack this diffusion layer.",
        ]
    )
    (out_dir / "v72_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.max_horizon = max(args.horizons)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = closure.device_from_arg(args.device)

    basis, route_meta, gate = build_route_basis(args, args.out_dir)
    anchor, anchor_meta = v70.build_v52_anchor(basis, args)
    context = prepare_context_pack(args, basis)

    if bool(args.v71_include_v52_candidate):
        mass = float(args.v71_v52_candidate_prior_mass)
        route_train = np.concatenate([anchor["train"][:, None, :], basis.route_train], axis=1).astype(np.float32)
        route_val = np.concatenate([anchor["val"][:, None, :], basis.route_val], axis=1).astype(np.float32)
        route_test = np.concatenate([anchor["test"][:, None, :], basis.route_test], axis=1).astype(np.float32)
        probs_train = np.concatenate(
            [np.full((len(route_train), 1), mass, dtype=np.float32), (1.0 - mass) * basis.prior.probs_train],
            axis=1,
        )
        probs_val = np.concatenate(
            [np.full((len(route_val), 1), mass, dtype=np.float32), (1.0 - mass) * basis.prior.probs_val],
            axis=1,
        )
        probs_test = np.concatenate(
            [np.full((len(route_test), 1), mass, dtype=np.float32), (1.0 - mass) * basis.prior.probs_test],
            axis=1,
        )
        probs_train /= np.maximum(probs_train.sum(axis=1, keepdims=True), EPS)
        probs_val /= np.maximum(probs_val.sum(axis=1, keepdims=True), EPS)
        probs_test /= np.maximum(probs_test.sum(axis=1, keepdims=True), EPS)
    else:
        route_train, route_val, route_test = basis.route_train, basis.route_val, basis.route_test
        probs_train, probs_val, probs_test = basis.prior.probs_train, basis.prior.probs_val, basis.prior.probs_test

    concepts = build_concepts(
        args,
        basis,
        anchor,
        context,
        route_train,
        route_val,
        route_test,
        probs_train,
        probs_val,
        probs_test,
    )

    ytr, yva, yte = basis.y_train, basis.y_val, basis.y_test
    rows: list[pd.DataFrame] = []
    diag_rows: list[pd.DataFrame] = []
    log_rows: list[pd.DataFrame] = []
    gate_rows: list[pd.DataFrame] = []

    rows.append(
        pd.DataFrame(
            v16.metric_rows(
                basis.arrays,
                flat_to_steps(anchor["test"], args.max_horizon),
                "v72_v52_frozen_anchor",
                args,
                {"stage": "v52_frozen_anchor", **anchor_meta},
            )
        )
    )
    route_err_te = residual_endpoint_error_matrix(route_test, yte, args)
    oracle_route = route_test[np.arange(len(route_test)), np.argmin(route_err_te, axis=1)]
    rows.append(
        pd.DataFrame(
            v16.metric_rows(
                basis.arrays,
                flat_to_steps(oracle_route, args.max_horizon),
                "v72_fixed_route_oracle",
                args,
                {"stage": "fixed_route_oracle"},
            )
        )
    )

    for spec in variant_specs(parse_strs(args.v71_variants)):
        xtr, xva, xte, ctr, cva, cte, ctx_names, concept_names = apply_variant(spec, context, concepts, args)
        xtr, xva, xte = standardize_context(xtr, xva, xte)
        ctr, cva, cte = standardize_concepts(ctr, cva, cte)
        model, train_log, train_meta = train_variant(
            spec=spec,
            xtr=xtr,
            xva=xva,
            ctr=ctr,
            cva=cva,
            ytr=ytr,
            yva=yva,
            anchor_tr=anchor["train"],
            anchor_va=anchor["val"],
            route_tr=route_train,
            route_va=route_val,
            concept_names=concept_names,
            args=args,
            device=device,
        )
        metric_df, diag_df, gate_df = evaluate_variant(
            model=model,
            spec=spec,
            xva=xva,
            xte=xte,
            cva=cva,
            cte=cte,
            anchor_va=anchor["val"],
            anchor_te=anchor["test"],
            route_va=route_val,
            route_te=route_test,
            yva=yva,
            yte=yte,
            basis=basis,
            concept_names=concept_names,
            args=args,
            device=device,
        )
        metric_df["context_dim"] = int(xtr.shape[1])
        metric_df["concept_dim"] = int(ctr.shape[2])
        diag_df["context_dim"] = int(xtr.shape[1])
        diag_df["concept_dim"] = int(ctr.shape[2])
        diag_df["train_best_val_rmse"] = train_meta["best_val_rmse"]
        rows.append(metric_df)
        diag_rows.append(diag_df)
        train_log["context_dim"] = int(xtr.shape[1])
        train_log["concept_dim"] = int(ctr.shape[2])
        log_rows.append(train_log)
        gate_rows.append(gate_df)

    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    diag = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame()
    train_log = pd.concat(log_rows, ignore_index=True) if log_rows else pd.DataFrame()
    concept_gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    for df in [summary, diag, train_log, concept_gates]:
        if not df.empty:
            df.insert(0, "seed", int(args.seed))
            df.insert(0, "dataset", str(args.dataset))
    if isinstance(gate, pd.DataFrame) and not gate.empty:
        gate.to_csv(args.out_dir / "v72_route_prior_gate.csv", index=False)
    contract_meta = {**route_meta, **anchor_meta}
    if concepts.meta:
        contract_meta.update(concepts.meta)
    pd.DataFrame([{"item": k, "value": v} for k, v in contract_meta.items() if not isinstance(v, (dict, list))]).to_csv(
        args.out_dir / "v72_data_contract.csv",
        index=False,
    )
    pd.DataFrame({"concept_name": concepts.names, "concept_index": np.arange(len(concepts.names))}).to_csv(
        args.out_dir / "v72_concept_names.csv",
        index=False,
    )
    summary.to_csv(args.out_dir / "v72_corridor_operator_diffusion_summary.csv", index=False)
    diag.to_csv(args.out_dir / "v72_corridor_operator_diffusion_diagnostics.csv", index=False)
    train_log.to_csv(args.out_dir / "v72_train_log.csv", index=False)
    concept_gates.to_csv(args.out_dir / "v72_concept_gates.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(audit.finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag, train_log, concept_gates)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "summary_rows": int(len(summary)),
                "variants": parse_strs(args.v71_variants),
                "elapsed_sec": time.time() - t0,
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(ap)
    ap.set_defaults(features=v52.DEFAULT_FEATURES, out_dir=DEFAULT_OUT)

    # v12/v16 route-basis compatibility.
    ap.add_argument("--extra-feature-grid", type=Path, default=v12.DEFAULT_OBJECT_GRID)
    ap.add_argument("--extra-feature-prefixes", type=str, default="oc_")
    ap.add_argument("--extra-feature-block-name", type=str, default="object_mask")
    ap.add_argument("--extra-feature-max-cols", type=int, default=256)
    ap.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    ap.add_argument("--v10-velocity-max-cols", type=int, default=160)
    ap.add_argument("--v12-route-k", type=int, default=12)
    ap.add_argument("--v12-min-route-cluster-size", type=int, default=40)
    ap.add_argument("--v12-prior-model", type=str, default="logistic", choices=["logistic", "hgbdt"])
    ap.add_argument("--v12-prior-max-iter", type=int, default=500)
    ap.add_argument("--v12-prior-c", type=float, default=0.35)
    ap.add_argument("--v12-hgbdt-iter", type=int, default=160)
    ap.add_argument("--v12-hgbdt-lr", type=float, default=0.05)
    ap.add_argument("--v12-hgbdt-leaf-nodes", type=int, default=31)
    ap.add_argument("--v12-hgbdt-l2", type=float, default=0.02)
    ap.add_argument("--v12-max-route-features", type=int, default=768)
    ap.add_argument("--v12-include-decomposition", action="store_true")
    ap.add_argument("--v12-expert-alpha", type=float, default=300.0)
    ap.add_argument("--v12-min-expert-samples", type=int, default=80)
    ap.add_argument("--v12-error-pool-max", type=int, default=2500)
    ap.add_argument("--v12-top-route-modes", type=int, default=4)
    ap.add_argument("--v12-route-prob-power", type=float, default=1.5)
    ap.add_argument("--v12-error-noise-scale", type=float, default=0.75)
    ap.add_argument("--v12-noise-jitter", type=float, default=0.02)
    ap.add_argument("--v16c-generator-variant", type=str, default="context_velocity")
    ap.add_argument("--v16c-top-c", type=int, default=8)
    ap.add_argument("--v16c-max-context-features", type=int, default=384)
    ap.add_argument("--v16c-ridge-alphas", type=str, default="0.1,0.3,1,3,10,30,100,300,1000,3000")

    # v52 anchor compatibility.
    ap.add_argument("--generator-max-train-rows", type=int, default=18000)
    ap.add_argument("--generator-max-val-rows", type=int, default=5000)
    ap.add_argument("--generator-max-test-rows", type=int, default=7000)
    ap.add_argument("--local-ks", type=str, default="8,16,32")
    ap.add_argument("--local-radii", type=str, default="64,128,256")
    ap.add_argument("--flow-packets", type=str, default="real")
    ap.add_argument("--v70-v52-base-mix", default="expert_top8_uniform", choices=["expert_top4_uniform", "expert_top8_uniform", "expert_all_uniform"])
    ap.add_argument("--v70-v52-calibrator", default="stacked_context", choices=["stacked_context", "stacked_top_context", "correction_context"])
    ap.add_argument("--v70-v52-bounded", action="store_true", default=True)
    ap.add_argument("--no-v70-v52-bounded", action="store_false", dest="v70_v52_bounded")
    ap.add_argument("--v70-bounded-quantile", type=float, default=0.95)
    ap.add_argument("--v70-bounded-scale", type=float, default=1.25)
    ap.add_argument("--v70-jump-factor", type=float, default=3.0)

    # v72 corridor/state operators.  These names mirror v58 so that the same
    # explicit feasibility machinery can be used inside the diffusion/router.
    ap.add_argument("--v72-use-corridor-operators", action="store_true", default=True)
    ap.add_argument("--no-v72-use-corridor-operators", action="store_false", dest="v72_use_corridor_operators")
    ap.add_argument("--v72-state-grid", type=str, default=str(DEFAULT_STATE_GRID))
    ap.add_argument("--v72-corridor-occupancy-modes", type=str, default="velocity,local_flow")
    ap.add_argument("--v72-tune-objective", type=str, default="endpoint", choices=["endpoint", "hmax", "h6", "max_horizon"])
    ap.add_argument("--v58-neighbor-k", type=int, default=64)
    ap.add_argument("--v58-flow-k", type=int, default=16)
    ap.add_argument("--v58-samples-per-step", type=int, default=4)
    ap.add_argument("--v58-default-radius-px", type=float, default=23.1)
    ap.add_argument("--v58-radius-min-px", type=float, default=5.0)
    ap.add_argument("--v58-radius-max-px", type=float, default=42.0)
    ap.add_argument("--v58-nn-radius-factor", type=float, default=0.34)
    ap.add_argument("--v58-adaptive-reliability", type=float, default=0.45)
    ap.add_argument("--v58-central-radius-scale", type=float, default=0.55)
    ap.add_argument("--v58-radius-scale", type=float, default=0.85)
    ap.add_argument("--v58-neighbor-radius-scale", type=float, default=0.85)
    ap.add_argument("--v58-close-margin-px", type=float, default=10.0)
    ap.add_argument("--v58-free-margin-px", type=float, default=8.0)
    ap.add_argument("--v58-pressure-scale-px", type=float, default=18.0)
    ap.add_argument("--v58-corridor-width-px", type=float, default=92.0)
    ap.add_argument("--v58-neighbor-velocity-clip", type=float, default=18.0)
    ap.add_argument("--v58-jump-factor", type=float, default=3.0)
    ap.add_argument("--v58-max-path-factor", type=float, default=2.6)
    ap.add_argument("--v58-hard-overlap", type=float, default=0.55)
    ap.add_argument("--v58-hard-collision-frac", type=float, default=0.22)
    ap.add_argument("--v58-hard-jump-excess-px", type=float, default=24.0)
    ap.add_argument("--v58-hard-path-excess-px", type=float, default=36.0)

    # v71/v72 architecture.
    ap.add_argument("--v71-variants", type=str, default="full,no_corridor,corridor_only,blackbox_only,shuffled_physics,no_route_queries")
    ap.add_argument("--v71-hidden", type=int, default=192)
    ap.add_argument("--v71-layers", type=int, default=2)
    ap.add_argument("--v71-dropout", type=float, default=0.05)
    ap.add_argument("--v71-epochs", type=int, default=18)
    ap.add_argument("--v71-batch-size", type=int, default=384)
    ap.add_argument("--v71-lr", type=float, default=8e-4)
    ap.add_argument("--v71-weight-decay", type=float, default=1e-4)
    ap.add_argument("--v71-grad-clip", type=float, default=5.0)
    ap.add_argument("--v71-noise-scale", type=float, default=1.0)
    ap.add_argument("--v71-inference-steps", type=int, default=8)
    ap.add_argument("--v71-oracle-temperature", type=float, default=8.0)
    ap.add_argument("--v71-train-temperature", type=float, default=1.0)
    ap.add_argument("--v71-temperature-grid", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--v71-top-m-grid", type=str, default="1,2,4,8,12")
    ap.add_argument("--v71-anchor-blend-grid", type=str, default="0,0.05,0.10,0.20,0.35,0.50,0.75,1.0")
    ap.add_argument("--v71-fm-weight", type=float, default=1.0)
    ap.add_argument("--v71-final-weight", type=float, default=0.60)
    ap.add_argument("--v71-energy-weight", type=float, default=0.35)
    ap.add_argument("--v71-nll-weight", type=float, default=0.03)
    ap.add_argument("--v71-entropy-weight", type=float, default=0.004)
    ap.add_argument("--v71-gate-sparsity-weight", type=float, default=0.002)
    ap.add_argument("--v71-physics-consistency-weight", type=float, default=0.02)
    ap.add_argument("--v71-basis-names", type=str, default="self,flow,crowding,corridor,smooth,prior,route,blackbox")
    ap.add_argument("--v71-flow-ks", type=str, default="8,16,32")
    ap.add_argument("--v71-density-radii", type=str, default="64,128,256")
    ap.add_argument("--v71-include-v52-candidate", action="store_true", default=True)
    ap.add_argument("--no-v71-include-v52-candidate", action="store_false", dest="v71_include_v52_candidate")
    ap.add_argument("--v71-v52-candidate-prior-mass", type=float, default=0.35)
    args = ap.parse_args()
    if args.smoke:
        args.max_train_rows = min(int(args.max_train_rows), 900)
        args.max_val_rows = min(int(args.max_val_rows), 300)
        args.max_test_rows = min(int(args.max_test_rows), 400)
        args.generator_max_train_rows = args.max_train_rows
        args.generator_max_val_rows = args.max_val_rows
        args.generator_max_test_rows = args.max_test_rows
        args.posterior_epochs = min(int(args.posterior_epochs), 3)
        args.student_epochs = min(int(args.student_epochs), 3)
        args.learned_route_epochs = min(int(args.learned_route_epochs), 3)
        args.candidate_k = min(int(args.candidate_k), 16)
        args.oracle_k = "4,8,16"
        args.v71_epochs = min(int(args.v71_epochs), 4)
        args.v71_hidden = min(int(args.v71_hidden), 96)
        args.v71_layers = 1
        args.v71_batch_size = min(int(args.v71_batch_size), 256)
        args.v71_inference_steps = min(int(args.v71_inference_steps), 4)
        args.v71_variants = "full,no_corridor,corridor_only,shuffled_physics"
        args.v71_temperature_grid = "0.5,1.0,2.0"
        args.v71_top_m_grid = "1,4,8"
        args.v71_anchor_blend_grid = "0,0.25,0.5,1.0"
        args.v72_corridor_occupancy_modes = "velocity"
        args.v58_neighbor_k = min(int(args.v58_neighbor_k), 32)
        args.v58_samples_per_step = min(int(args.v58_samples_per_step), 3)
    return args


if __name__ == "__main__":
    run(parse_args())
