#!/usr/bin/env python3
"""Route prototype oracle diagnostic for LaChance residual candidates.

This runner asks a narrow question:

    can we compress a K-candidate residual cloud into M route prototypes
    while preserving much more oracle signal than RouteQueryRefiner does?

If simple deterministic route construction preserves candidate oracle, the next
architecture should build an oracle-preserving route set before any critic.
"""

from __future__ import annotations

import argparse
import json
import sys
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
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_prototype_oracle_2026-06-27"
EPS = 1e-8


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def flatten_candidate(residual: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full":
        return residual.reshape(residual.shape[0], residual.shape[1], -1).astype(np.float32)
    if mode == "endpoint":
        return np.cumsum(residual, axis=2)[:, :, -1, :].astype(np.float32)
    if mode == "shape":
        endpoint = np.cumsum(residual, axis=2)[:, :, -1:, :]
        centered = residual - np.mean(residual, axis=2, keepdims=True)
        return np.concatenate([centered.reshape(residual.shape[0], residual.shape[1], -1), endpoint.reshape(residual.shape[0], residual.shape[1], -1)], axis=-1).astype(np.float32)
    raise ValueError(f"unknown flatten mode: {mode}")


def fps_indices(x: np.ndarray, m: int) -> np.ndarray:
    """Per-row farthest-point sampling over a small candidate set."""
    n, k, _ = x.shape
    m = min(int(m), k)
    out = np.zeros((n, m), dtype=np.int64)
    mean = np.mean(x, axis=1, keepdims=True)
    first = np.argmin(np.sum((x - mean) ** 2, axis=-1), axis=1)
    out[:, 0] = first
    min_dist = np.sum((x - x[np.arange(n), first][:, None, :]) ** 2, axis=-1)
    for j in range(1, m):
        take = np.argmax(min_dist, axis=1)
        out[:, j] = take
        d = np.sum((x - x[np.arange(n), take][:, None, :]) ** 2, axis=-1)
        min_dist = np.minimum(min_dist, d)
    return out


def gather_candidates(residual: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return residual[np.arange(len(idx))[:, None], idx].astype(np.float32)


def oracle_from_set(residual_set: np.ndarray, true: np.ndarray, horizons: list[int]) -> np.ndarray:
    err = qrc.endpoint_errors(residual_set, true, horizons)
    take = np.argmin(err, axis=1)
    return residual_set[np.arange(len(take)), take].astype(np.float32)


def mean_from_set(residual_set: np.ndarray) -> np.ndarray:
    return np.mean(residual_set, axis=1).astype(np.float32)


def endpoint_rows(arrays: audit.SplitArrays, pred: np.ndarray, label: str, args: argparse.Namespace, extra: dict[str, Any]) -> list[dict[str, Any]]:
    return audit.endpoint_metrics(
        steps_true=arrays.steps_test,
        base=arrays.base_test,
        residual_pred=pred,
        horizons=args.horizons,
        label=label,
        extra=extra,
    )


def train_route_model_if_needed(args, arrays, posterior, student, blocks, device):
    hybrid_budgets = seq.resolve_hybrid_budgets(args) if args.candidate_generator == "hybrid" else {"generic": 0, "route": 0, "learned": 0}
    needs_learned_route = args.candidate_generator == "learned_route" or (
        args.candidate_generator == "hybrid" and hybrid_budgets.get("learned", 0) > 0
    )
    if not needs_learned_route:
        return None, None, None, None
    route_blocks = closure.variant_blocks(args.learned_route_context_variant, arrays.x_train)
    route_ctx_train_raw = seq.flatten_blocks(arrays.x_train, route_blocks)
    route_ctx_val_raw = seq.flatten_blocks(arrays.x_val, route_blocks)
    route_ctx_test_raw = seq.flatten_blocks(arrays.x_test, route_blocks)
    if args.learned_route_add_decomposition_context:
        pred_route_train = closure.predict_student(student, arrays.x_train, blocks, device=device, batch_size=args.batch_size)
        pred_route_val = closure.predict_student(student, arrays.x_val, blocks, device=device, batch_size=args.batch_size)
        pred_route_test = closure.predict_student(student, arrays.x_test, blocks, device=device, batch_size=args.batch_size)
        route_ctx_train_raw = np.concatenate([route_ctx_train_raw, seq.decomposition_context_features(pred_route_train, mode_k=args.mode_k)], axis=1)
        route_ctx_val_raw = np.concatenate([route_ctx_val_raw, seq.decomposition_context_features(pred_route_val, mode_k=args.mode_k)], axis=1)
        route_ctx_test_raw = np.concatenate([route_ctx_test_raw, seq.decomposition_context_features(pred_route_test, mode_k=args.mode_k)], axis=1)
    if route_ctx_train_raw.shape[1] > args.max_learned_route_context_features:
        var = np.var(route_ctx_train_raw, axis=0)
        keep = np.argsort(var)[-args.max_learned_route_context_features :]
        route_ctx_train_raw = route_ctx_train_raw[:, keep]
        route_ctx_val_raw = route_ctx_val_raw[:, keep]
        route_ctx_test_raw = route_ctx_test_raw[:, keep]
    route_ctx_train, route_ctx_val, route_ctx_test, _ = seq.standardize(route_ctx_train_raw, route_ctx_val_raw, route_ctx_test_raw)
    route_model, route_train_log = seq.train_learned_route_generator(
        route_ctx_train,
        route_ctx_val,
        arrays.residual_train,
        arrays.residual_val,
        posterior.mode_soft_train,
        posterior.mode_soft_val,
        args,
        device=device,
    )
    return route_model, route_ctx_train, route_ctx_val, route_ctx_test


def generate_test_candidates(args, arrays, posterior, student, blocks, route_model, route_ctx_test, device) -> seq.CandidatePack:
    if args.candidate_generator == "learned_route":
        return seq.generate_learned_route_candidates(arrays, posterior, student, blocks, route_model, route_ctx_test, args, split_name="test", device=device)
    if args.candidate_generator == "hybrid":
        return seq.generate_hybrid_candidates(arrays, posterior, student, blocks, route_model, route_ctx_test, args, split_name="test", device=device)
    return seq.generate_candidates(arrays, posterior, student, blocks, args, split_name="test", device=device)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)
    arrays, split = audit.prepare_data(args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)
    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    route_model, _, _, route_ctx_test = train_route_model_if_needed(args, arrays, posterior, student, blocks, device)
    cand = generate_test_candidates(args, arrays, posterior, student, blocks, route_model, route_ctx_test, device)

    rows: list[dict[str, Any]] = []
    rows.extend(endpoint_rows(arrays, seq.mean_candidate_residual(cand), "candidate_mean", args, {"stage": "candidate_control"}))
    for k in args.oracle_k:
        rows.extend(endpoint_rows(arrays, seq.oracle_residual(cand, arrays.residual_test, int(k)), f"candidate_oracle@{k}", args, {"stage": "candidate_oracle", "oracle_k": int(k)}))
        rows.extend(
            endpoint_rows(
                arrays,
                oracle_from_set(cand.residual[:, : int(k)], arrays.residual_test, args.horizons),
                f"candidate_endpoint_oracle@{k}",
                args,
                {"stage": "candidate_endpoint_oracle", "oracle_k": int(k)},
            )
        )

    rng = np.random.default_rng(args.seed + 31001)
    prototype_k = parse_ints(args.prototype_k)
    for m in prototype_k:
        if m > cand.residual.shape[1]:
            continue
        first = cand.residual[:, :m]
        rows.extend(endpoint_rows(arrays, oracle_from_set(first, arrays.residual_test, args.horizons), f"first{m}_oracle", args, {"stage": "prototype_oracle", "prototype": "first", "prototype_k": m}))
        rows.extend(endpoint_rows(arrays, mean_from_set(first), f"first{m}_mean", args, {"stage": "prototype_mean", "prototype": "first", "prototype_k": m}))

        ridx = np.stack([rng.choice(cand.residual.shape[1], size=m, replace=False) for _ in range(len(cand.residual))], axis=0)
        rset = gather_candidates(cand.residual, ridx)
        rows.extend(endpoint_rows(arrays, oracle_from_set(rset, arrays.residual_test, args.horizons), f"random{m}_oracle", args, {"stage": "prototype_oracle", "prototype": "random", "prototype_k": m}))

        for mode in ["endpoint", "shape", "full"]:
            x = flatten_candidate(cand.residual, mode)
            idx = fps_indices(x, m)
            pset = gather_candidates(cand.residual, idx)
            rows.extend(endpoint_rows(arrays, oracle_from_set(pset, arrays.residual_test, args.horizons), f"fps_{mode}{m}_oracle", args, {"stage": "prototype_oracle", "prototype": f"fps_{mode}", "prototype_k": m}))
            rows.extend(endpoint_rows(arrays, mean_from_set(pset), f"fps_{mode}{m}_mean", args, {"stage": "prototype_mean", "prototype": f"fps_{mode}", "prototype_k": m}))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "route_prototype_oracle_summary.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = ["# Route Prototype Oracle Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- candidate_generator: `{args.candidate_generator}`")
    lines.append(f"- candidate_k: `{args.candidate_k}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{h}")
        sub = summary[summary["horizon"].eq(h)].sort_values("rmse")
        for _, row in sub.head(20).iterrows():
            lines.append(f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, gain={row['gain_vs_base_pct']:.2f}%")
    lines.append("\n## Gate")
    lines.append("- If fps/cluster prototype oracle is close to candidate oracle and much better than RouteQueryRefiner query-oracle, route construction is the bottleneck.")
    lines.append("- If prototype oracle is also weak, candidate diversity itself is not enough and generation/observability must change.")
    (out_dir / "route_prototype_oracle_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.add_argument("--prototype-k", type=str, default="4,8,12,16")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 4000)
        args.max_val_rows = min(args.max_val_rows, 1500)
        args.max_test_rows = min(args.max_test_rows, 2000)
        args.posterior_epochs = min(args.posterior_epochs, 8)
        args.student_epochs = min(args.student_epochs, 8)
        args.learned_route_epochs = min(args.learned_route_epochs, 6)
        args.candidate_k = min(args.candidate_k, 16)
        args.oracle_k = [8, min(16, args.candidate_k)]
        args.max_all_features = min(args.max_all_features, 192)
    run(args)


if __name__ == "__main__":
    main()
