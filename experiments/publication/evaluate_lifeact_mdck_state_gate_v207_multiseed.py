#!/usr/bin/env python3
"""Re-evaluate the frozen v207 cell-state table over several model seeds."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_lifeact_mdck_mechanochemical_state_gate_v207 import (
    EPS,
    evaluate,
    gate_summary,
)


CONTROL_PACKETS = [
    "full_zero_state",
    "full_row_shuffled",
    "full_wrong_cell",
    "full_time_shuffled",
]


def decision_rows(gate: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    real = gate[gate["packet"].eq("full_real")]
    controls = gate[gate["packet"].isin(CONTROL_PACKETS)]
    for _, row in real.iterrows():
        matching = controls[
            controls["protocol"].eq(row["protocol"])
            & controls["model"].eq(row["model"])
        ]
        best_control = float(matching["component_rmse_mean"].min())
        gain_control = 100.0 * (
            best_control - float(row["component_rmse_mean"])
        ) / max(best_control, EPS)
        rows.append(
            {
                "seed": seed,
                "protocol": row["protocol"],
                "model": row["model"],
                "real_rmse": float(row["component_rmse_mean"]),
                "coord_rmse": float(row["coord_rmse"]),
                "best_control_rmse": best_control,
                "gain_vs_coord_percent": float(row["gain_vs_coord_percent"]),
                "gain_vs_best_control_percent": gain_control,
                "soft_pass": bool(
                    row["gain_vs_coord_percent"] >= 1.0
                    and row["component_rmse_mean"] < best_control
                ),
                "hard_pass": bool(
                    row["gain_vs_coord_percent"] >= 3.0 and gain_control >= 1.0
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--scale-normalized", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = pd.read_parquet(args.state_path)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    gates: list[pd.DataFrame] = []
    decisions: list[dict[str, object]] = []
    for seed in seeds:
        metrics, predictions = evaluate(
            state, seed, scale_normalized=args.scale_normalized
        )
        gate = gate_summary(metrics)
        metrics.insert(0, "seed", seed)
        gate.insert(0, "seed", seed)
        metrics.to_csv(args.out_dir / f"v207_seed{seed}_metrics.csv", index=False)
        gate.to_csv(args.out_dir / f"v207_seed{seed}_gate.csv", index=False)
        prediction_subset = predictions[
            predictions["model"].eq("hgbdt")
            & predictions["packet"].isin(["coord_only", "full_real", *CONTROL_PACKETS])
        ]
        prediction_subset.to_parquet(
            args.out_dir / f"v207_seed{seed}_hgbdt_predictions.parquet", index=False
        )
        gates.append(gate)
        decisions.extend(decision_rows(gate, seed))
        del metrics, predictions, prediction_subset
        gc.collect()

    all_gates = pd.concat(gates, ignore_index=True)
    decision = pd.DataFrame(decisions)
    aggregate = (
        all_gates.groupby(["protocol", "packet", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            component_rmse_mean=("component_rmse_mean", "mean"),
            component_rmse_seed_std=("component_rmse_mean", "std"),
            component_r2_mean=("component_r2_mean", "mean"),
            gain_vs_coord_percent_mean=("gain_vs_coord_percent", "mean"),
            gain_vs_coord_percent_std=("gain_vs_coord_percent", "std"),
        )
    )
    decision_aggregate = (
        decision.groupby(["protocol", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            real_rmse_mean=("real_rmse", "mean"),
            gain_vs_coord_percent_mean=("gain_vs_coord_percent", "mean"),
            gain_vs_coord_percent_std=("gain_vs_coord_percent", "std"),
            gain_vs_best_control_percent_mean=("gain_vs_best_control_percent", "mean"),
            soft_pass_seeds=("soft_pass", "sum"),
            hard_pass_seeds=("hard_pass", "sum"),
        )
    )
    all_gates.to_csv(args.out_dir / "v207_multiseed_gate.csv", index=False)
    decision.to_csv(args.out_dir / "v207_multiseed_decision.csv", index=False)
    aggregate.to_csv(args.out_dir / "v207_multiseed_aggregate.csv", index=False)
    decision_aggregate.to_csv(
        args.out_dir / "v207_multiseed_decision_aggregate.csv", index=False
    )

    report = [
        "# LifeAct-MDCK v207 multiseed confirmation",
        "",
        f"Frozen state table: `{args.state_path.resolve()}`.",
        f"Seeds: `{','.join(map(str, seeds))}`.",
        f"Evaluation scale: `{'current-frame cell diameter' if args.scale_normalized else 'pixels'}`.",
        "",
        "## Decision",
        "",
        decision_aggregate.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full aggregate",
        "",
        aggregate.to_markdown(index=False, floatfmt=".6f"),
        "",
        "A robust pass requires the real state to beat coordinates and every hard control across seeds; seed variation alone is not treated as independent biological replication.",
    ]
    (args.out_dir / "v207_multiseed_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
