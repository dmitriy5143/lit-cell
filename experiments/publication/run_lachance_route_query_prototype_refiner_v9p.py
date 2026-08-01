#!/usr/bin/env python3
"""Route-query/pruned refiner on top of strong prototype/component candidates.

This runner is the direct follow-up to v9:

    route-prototype/component candidate cloud
    + edge memory / decomposition axes / optional video teacher
    -> v9 route-query token router/pruner
    -> sparse top-M / dense prototype mixture

The goal is to test the selector/refiner on the historical strong
``fps_shape`` prototype cloud instead of the weaker hybrid K=64 cloud used by
``run_lachance_route_query_pruned_refiner_v9.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_decomposition_module_audit as audit  # noqa: E402
import run_lachance_decomposition_stage_closure as closure  # noqa: E402
import run_lachance_graph_memory_critic_v4 as v4  # noqa: E402
import run_lachance_latent_history_generator as histgen  # noqa: E402
import run_lachance_query_risk_calibrator as qrc  # noqa: E402
import run_lachance_route_prototype_refiner as rpr  # noqa: E402
import run_lachance_sequence_critic_refiner as seq  # noqa: E402
import run_lachance_sequence_joint_selector_refiner_v7 as v7  # noqa: E402
import run_lachance_agentic_sequence_refiner_v8 as v8  # noqa: E402
import run_lachance_route_query_pruned_refiner_v9 as v9  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "route_query_prototype_refiner_v9p_2026-07-02"


def parse_ints(text: str) -> list[int]:
    return audit.parse_ints(text)


def finite_json(value: Any) -> Any:
    return audit.finite_json(value)


def make_prototype_pack(
    cand: seq.CandidatePack,
    arrays: audit.SplitArrays,
    *,
    split_name: str,
    method: str,
    k: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[seq.CandidatePack, np.ndarray]:
    residual, idx = rpr.build_prototype_set(cand, method, int(k), seed=int(seed))
    n, q, h, _ = residual.shape
    route_logits = rpr.route_logits_from_indices(idx, cand.residual.shape[1], method).astype(np.float32)
    route_prob = qrc.softmax_np(route_logits, axis=1)
    z_dim = cand.z.shape[-1] if getattr(cand, "z", None) is not None and cand.z.size else 1
    mode_dim = cand.mode_prob.shape[-1] if getattr(cand, "mode_prob", None) is not None and cand.mode_prob.size else 1
    z = np.zeros((n, q, z_dim), dtype=np.float32)
    z_eps = np.zeros_like(z)
    mode_prob = np.zeros((n, q, mode_dim), dtype=np.float32)
    if mode_dim:
        mode_prob[..., 0] = 1.0
    logprob = np.log(np.clip(route_prob, 1e-8, 1.0))[..., None].astype(np.float32)
    base = getattr(arrays, f"base_{split_name}")
    features, _ = seq.build_candidate_features(
        residual=residual,
        base=base,
        z_eps=z_eps,
        logprob=logprob,
        horizons=args.horizons,
    )
    true = getattr(arrays, f"residual_{split_name}")
    oracle_dist = qrc.risk_endpoint_errors(residual, true, args)
    route_mode = idx.astype(np.int64)
    return (
        seq.CandidatePack(
            residual=residual.astype(np.float32),
            z=z,
            z_eps=z_eps,
            logprob=logprob,
            mode_prob=mode_prob,
            features=features.astype(np.float32),
            oracle_dist=oracle_dist.astype(np.float32),
            route_mode=route_mode,
        ),
        idx,
    )


def endpoint_rows(rows: list[dict[str, Any]], arrays: audit.SplitArrays, pred: np.ndarray, args: argparse.Namespace, label: str, extra: dict[str, Any]) -> None:
    rows.extend(
        audit.endpoint_metrics(
            steps_true=arrays.steps_test,
            base=arrays.base_test,
            residual_pred=pred,
            horizons=args.horizons,
            label=label,
            extra=extra,
        )
    )


def run(args: argparse.Namespace) -> None:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = closure.device_from_arg(args.device)

    arrays, split = audit.prepare_data(args)
    extra_feature_meta = rpr.attach_extra_feature_block(arrays, split, args)
    if args.add_history:
        arrays = histgen.add_history_blocks(arrays, split, args)

    edge_memory, edge_memory_meta = v4.load_edge_memory_cache(
        args.edge_sequence_cache,
        split,
        max_lags=args.v9_edge_max_lags,
        max_neighbours=args.v9_edge_max_neighbours,
        min_found_frac=args.v9_min_edge_found_frac,
    )

    posterior = closure.train_posterior(arrays, args, device)
    student, blocks, _ = closure.train_student(arrays, posterior, args, variant=args.generator_variant, device=device)
    ctx_train, ctx_val, ctx_test, ctx_meta = rpr.prepare_context(args, arrays, posterior, student, blocks, device)

    args.component_aware_risk = True
    component_axes = rpr.build_component_axes(args, arrays, posterior, student, blocks, device)
    if component_axes is None:
        raise RuntimeError("v9p requires component axes")

    # For v9p we intentionally reproduce the old strong candidate setting:
    # generic candidate cloud -> deterministic FPS/shape prototypes.
    cand_args = argparse.Namespace(**vars(args))
    cand_args.candidate_generator = args.prototype_source_candidate_generator
    cand_args.candidate_k = int(args.prototype_source_candidate_k)
    cand_args.oracle_k = [min(int(k), cand_args.candidate_k) for k in args.oracle_k]
    route_model, route_ctx_train, route_ctx_val, route_ctx_test, route_log = rpr.train_route_model_if_needed(
        cand_args, arrays, posterior, student, blocks, device
    )
    src_train = rpr.generate_candidates_for_split(cand_args, arrays, posterior, student, blocks, route_model, route_ctx_train, "train", device)
    src_val = rpr.generate_candidates_for_split(cand_args, arrays, posterior, student, blocks, route_model, route_ctx_val, "val", device)
    src_test = rpr.generate_candidates_for_split(cand_args, arrays, posterior, student, blocks, route_model, route_ctx_test, "test", device)

    proto_train, idx_train = make_prototype_pack(
        src_train, arrays, split_name="train", method=args.prototype_method, k=args.prototype_k, args=args, seed=args.seed + 8001
    )
    proto_val, idx_val = make_prototype_pack(
        src_val, arrays, split_name="val", method=args.prototype_method, k=args.prototype_k, args=args, seed=args.seed + 9001
    )
    proto_test, idx_test = make_prototype_pack(
        src_test, arrays, split_name="test", method=args.prototype_method, k=args.prototype_k, args=args, seed=args.seed + 10001
    )

    edge_ctx_train, edge_ctx_val, edge_ctx_test, edge_ctx_meta = v7.select_context_block(
        arrays, args.v9_video_edge_block, args.v9_video_edge_max_features
    )
    residual_hints, video_probe = v7.fit_video_residual_hints(
        arrays=arrays,
        ctx_train=ctx_train,
        ctx_val=ctx_val,
        ctx_test=ctx_test,
        edge_train=edge_ctx_train,
        edge_val=edge_ctx_val,
        edge_test=edge_ctx_test,
        args=args,
    )

    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "extra_feature": finite_json(extra_feature_meta),
        "context": finite_json(ctx_meta),
        "edge_memory": finite_json(edge_memory_meta),
        "video_edge": finite_json(edge_ctx_meta),
        "video_residual_teacher": finite_json(residual_hints.get("info", {})),
        "prototype": {
            "method": args.prototype_method,
            "k": int(args.prototype_k),
            "source_candidate_generator": args.prototype_source_candidate_generator,
            "source_candidate_k": int(args.prototype_source_candidate_k),
            "test_index_mean": float(np.mean(idx_test)),
            "test_index_std": float(np.std(idx_test)),
        },
    }

    endpoint_rows(rows, arrays, seq.mean_candidate_residual(src_test), args, "source_candidate_mean", {"stage": "source_candidate_control"})
    for k in args.oracle_k:
        kk = min(int(k), src_test.residual.shape[1])
        endpoint_rows(
            rows,
            arrays,
            seq.oracle_residual(src_test, arrays.residual_test, kk),
            args,
            f"source_candidate_oracle@{kk}",
            {"stage": "source_candidate_oracle", "oracle_k": kk},
        )
    endpoint_rows(rows, arrays, seq.mean_candidate_residual(proto_test), args, f"{args.prototype_method}{args.prototype_k}_mean", {"stage": "prototype_mean"})
    endpoint_rows(
        rows,
        arrays,
        qrc.query_oracle_residual(proto_test.residual, arrays.residual_test, args.horizons),
        args,
        f"{args.prototype_method}{args.prototype_k}_oracle",
        {"stage": "prototype_oracle"},
    )

    variants: list[tuple[str, dict[str, Any]]] = [
        ("v9p_full_pruned", {}),
        ("v9p_no_video", {"use_video": False}),
        ("v9p_shuffled_video", {"shuffle_video": True}),
        ("v9p_no_edge", {"use_edge": False}),
        ("v9p_no_axes", {"use_axis": False}),
        ("v9p_shuffled_axes", {"shuffle_axis": True}),
        ("v9p_no_context", {"use_context": False}),
        ("v9p_edge_axes_only", {"use_context": False, "use_video": False}),
        ("v9p_router_disabled", {"router_disabled": True}),
        ("v9p_shuffled_labels", {"shuffled_labels": True}),
    ]
    requested = {s.strip() for s in str(args.v9_variant_list).split(",") if s.strip()}
    for name, kwargs in variants:
        if requested and name not in requested:
            continue
        v9.run_variant(
            name=name,
            rows=rows,
            logs=logs,
            diagnostics=diagnostics,
            arrays=arrays,
            cand_train=proto_train,
            cand_val=proto_val,
            cand_test=proto_test,
            ctx_train=ctx_train,
            ctx_val=ctx_val,
            ctx_test=ctx_test,
            component_axes=component_axes,
            residual_hints=residual_hints,
            edge_memory=edge_memory,
            args=args,
            device=device,
            meta_out=meta,
            **kwargs,
        )

    summary = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    if not summary.empty:
        summary.insert(0, "seed", int(args.seed))
        summary.insert(0, "dataset", str(args.dataset))
    if not diag.empty:
        diag.insert(0, "seed", int(args.seed))
        diag.insert(0, "dataset", str(args.dataset))
    summary.to_csv(args.out_dir / "route_query_prototype_v9p_summary.csv", index=False)
    diag.to_csv(args.out_dir / "route_query_prototype_v9p_diagnostics.csv", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(args.out_dir / "route_query_prototype_v9p_train_log.csv", index=False)
    if not route_log.empty:
        route_log.to_csv(args.out_dir / "route_query_prototype_v9p_route_train_log.csv", index=False)
    video_probe.to_csv(args.out_dir / "route_query_prototype_v9p_video_teacher_probe.csv", index=False)
    component_axes.probe.to_csv(args.out_dir / "route_query_prototype_v9p_component_axis_probe.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(finite_json(vars(args)), indent=2), encoding="utf-8")
    (args.out_dir / "scalers.json").write_text(json.dumps(finite_json(meta), indent=2), encoding="utf-8")
    write_report(args.out_dir, args, summary, diag, video_probe)
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "diagnostic_rows": len(diag)}, indent=2))


def write_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, diag: pd.DataFrame, video_probe: pd.DataFrame) -> None:
    lines = ["# Route-Query Prototype Refiner v9p Report\n"]
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- prototype: `{args.prototype_method}{args.prototype_k}`")
    lines.append(f"- source_candidate_generator: `{args.prototype_source_candidate_generator}`")
    lines.append(f"- source_candidate_k: `{args.prototype_source_candidate_k}`")
    lines.append(f"- route_queries: `{args.v9_route_queries}`")
    lines.append(f"- router_topk: `{args.v9_router_topk}`")
    lines.append("")
    for h in args.horizons:
        lines.append(f"## h{int(h)}")
        sub = summary[summary["horizon"].eq(int(h))].sort_values("rmse")
        for _, row in sub.head(36).iterrows():
            lines.append(
                f"- `{row['method']}`: RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}, "
                f"stage={row.get('stage', '')}"
            )
    if not diag.empty:
        lines.append("\n## Diagnostics")
        for _, row in diag.sort_values("risk_error_corr", ascending=False).iterrows():
            lines.append(
                f"- `{row['variant']}`: corr={row['risk_error_corr']:.3f}, "
                f"val={row['val_selector_rmse']:.3f}, topM={int(row['best_top_m'])}, "
                f"T={row['temperature']:.3f}, edge={row.get('gate_frac_edge', float('nan')):.3f}, "
                f"axis={row.get('gate_frac_axis', float('nan')):.3f}, "
                f"video={row.get('gate_frac_video', float('nan')):.3f}, "
                f"context={row.get('gate_frac_context', float('nan')):.3f}"
            )
    if not video_probe.empty:
        lines.append("\n## Video Teacher Probe")
        lines.append(video_probe.to_markdown(index=False))
    lines.append("\n## Decision Gates")
    lines.append("- Pass if v9p beats the historical component-aware seed42 h6 RMSE `17.131`.")
    lines.append("- Strong pass if h6 <= `16.9` and controls degrade logically.")
    lines.append("- Fail if full is near/above component-aware or only shuffled controls win.")
    (out_dir / "route_query_prototype_v9p_status_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    qrc.add_common_args(parser)
    parser.set_defaults(out_dir=DEFAULT_OUT)
    parser.add_argument("--edge-sequence-cache", type=Path, required=True)
    parser.add_argument("--extra-feature-grid", type=Path, default=None)
    parser.add_argument("--extra-feature-prefixes", type=str, default="ef_")
    parser.add_argument("--extra-feature-block-name", type=str, default="explicit_edge")
    parser.add_argument("--extra-feature-max-cols", type=int, default=128)
    parser.add_argument("--extra-feature-merge-all-context", type=str, default="false")
    parser.add_argument("--prototype-source-candidate-generator", type=str, default="generic")
    parser.add_argument("--prototype-source-candidate-k", type=int, default=32)
    parser.add_argument("--prototype-method", type=str, default="fps_shape")
    parser.add_argument("--prototype-k", type=int, default=16)
    parser.add_argument("--component-axis-blocks", type=str, default="self,flow,morphology,boundary,crowding,raw_context,all_context")
    parser.add_argument("--component-include-student-axis", action="store_true")
    parser.add_argument("--component-axis-model", type=str, default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--component-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--component-axis-max-features", type=int, default=256)
    parser.add_argument("--component-axis-hidden", type=int, default=128)
    parser.add_argument("--component-axis-epochs", type=int, default=16)
    parser.add_argument("--component-axis-lr", type=float, default=8e-4)
    parser.add_argument("--component-axis-weight-decay", type=float, default=1e-4)
    parser.add_argument("--component-axis-dropout", type=float, default=0.05)
    parser.add_argument("--component-attention-temperature", type=float, default=6.0)
    parser.add_argument("--v9-video-edge-block", type=str, default="explicit_edge")
    parser.add_argument("--v9-video-edge-max-features", type=int, default=128)
    parser.add_argument("--video-residual-model", type=str, default="hgbdt", choices=["ridge", "hgbdt"])
    parser.add_argument("--video-residual-include-context", action="store_true")
    parser.add_argument("--video-residual-hgbdt-iter", type=int, default=120)
    parser.add_argument("--video-residual-hgbdt-lr", type=float, default=0.045)
    parser.add_argument("--video-residual-hgbdt-leaf-nodes", type=int, default=15)
    parser.add_argument("--video-residual-hgbdt-l2", type=float, default=0.02)
    parser.add_argument("--v9-hidden", type=int, default=192)
    parser.add_argument("--v9-heads", type=int, default=4)
    parser.add_argument("--v9-layers", type=int, default=2)
    parser.add_argument("--v9-route-queries", type=int, default=12)
    parser.add_argument("--v9-router-topk", type=int, default=24)
    parser.add_argument("--v9-dropout", type=float, default=0.05)
    parser.add_argument("--v9-epochs", type=int, default=10)
    parser.add_argument("--v9-batch-size", type=int, default=192)
    parser.add_argument("--v9-lr", type=float, default=7e-4)
    parser.add_argument("--v9-weight-decay", type=float, default=1e-4)
    parser.add_argument("--v9-label-temperature", type=float, default=8.0)
    parser.add_argument("--v9-train-temperature", type=float, default=0.75)
    parser.add_argument("--v9-listwise-weight", type=float, default=1.0)
    parser.add_argument("--v9-rank-weight", type=float, default=0.5)
    parser.add_argument("--v9-reg-weight", type=float, default=0.25)
    parser.add_argument("--v9-candidate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--v9-gate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--v9-clip-grad", type=float, default=5.0)
    parser.add_argument("--v9-topm", type=str, default="1,2,4,8,16")
    parser.add_argument("--v9-temperatures", type=str, default="0.25,0.5,0.75,1.0,1.5")
    parser.add_argument("--v9-edge-max-lags", type=int, default=0)
    parser.add_argument("--v9-edge-max-neighbours", type=int, default=0)
    parser.add_argument("--v9-min-edge-found-frac", type=float, default=0.98)
    parser.add_argument("--v9-variant-list", type=str, default="")
    args = parser.parse_args()
    args.horizons = parse_ints(args.horizons)
    args.oracle_k = parse_ints(args.oracle_k)
    args.risk_temperatures = [float(x) for x in str(args.risk_temperatures).split(",") if x.strip()]
    args.route_temperatures = [float(x) for x in str(args.route_temperatures).split(",") if x.strip()]
    args.v8_topm = args.v9_topm
    args.v8_temperatures = [float(x) for x in str(args.v9_temperatures).split(",") if x.strip()]
    args.v8_video_edge_block = args.v9_video_edge_block
    args.v8_video_edge_max_features = args.v9_video_edge_max_features
    args.v8_lr = args.v9_lr
    args.v8_weight_decay = args.v9_weight_decay
    args.v8_batch_size = args.v9_batch_size
    args.v8_epochs = args.v9_epochs
    args.v8_label_temperature = args.v9_label_temperature
    args.v8_train_temperature = args.v9_train_temperature
    args.v8_clip_grad = args.v9_clip_grad
    args.v8_rank_weight = args.v9_rank_weight
    args.v8_reg_weight = args.v9_reg_weight
    args.v8_entropy_weight = args.v9_candidate_entropy_weight
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 1200)
        args.max_val_rows = min(args.max_val_rows, 400)
        args.max_test_rows = min(args.max_test_rows, 600)
        args.posterior_epochs = min(args.posterior_epochs, 3)
        args.student_epochs = min(args.student_epochs, 3)
        args.v9_epochs = min(args.v9_epochs, 3)
        args.component_axis_epochs = min(args.component_axis_epochs, 3)
        args.video_residual_hgbdt_iter = min(args.video_residual_hgbdt_iter, 40)
        args.prototype_source_candidate_k = min(args.prototype_source_candidate_k, 16)
        args.prototype_k = min(args.prototype_k, 8)
        args.oracle_k = [8, args.prototype_source_candidate_k]
    run(args)


if __name__ == "__main__":
    main()
