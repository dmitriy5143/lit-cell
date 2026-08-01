#!/usr/bin/env python3
"""Train missing external v97 seeds and confirm v162 across 7/42/123.

The source feature tables and split contracts are reused unchanged. Only the
model seed changes. Each seed then receives an independent v162 fit with the
same dimensionless transport search space.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPTS / "run_lachance_causal_innovation_state_space_v97.py"
V162_SCRIPT = SCRIPTS / "run_lachance_dimensionless_multidomain_transport_v162.py"
DEFAULT_OUT = ROOT / "outputs" / "lachance_v162_external_seed_confirmation"

SOURCE_CONFIGS = {
    "MDCK_Edge": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_direct_edge_guard_cpu_seed42_2026-07-21"
    / "run_config.json",
    "MDAMB231": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_direct_mdamb231_guard_seed42_2026-07-21"
    / "run_config.json",
    "HUVEC": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_huvec_guard_seed42_2026-07-21"
    / "run_config.json",
}

VARIANTS = {
    "MDCK_Edge": "v97_direct",
    "MDAMB231": "v97_direct",
    "HUVEC": "v97_no_context",
}

SEED42_CHECKPOINTS = {
    "MDCK_Edge": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_direct_edge_guard_cpu_seed42_2026-07-21"
    / "v97_direct.pt",
    "MDAMB231": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_direct_mdamb231_guard_seed42_2026-07-21"
    / "v97_direct.pt",
    "HUVEC": ROOT
    / "outputs"
    / "causal_innovation_state_space_v97_huvec_guard_seed42_2026-07-21"
    / "v97_no_context.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-v162", action="store_true")
    return parser.parse_args()


def cli_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def training_command(
    domain: str,
    seed: int,
    device: str,
    out_dir: Path,
) -> list[str]:
    config = json.loads(SOURCE_CONFIGS[domain].read_text(encoding="utf-8"))
    override = {
        "out_dir": str(out_dir),
        "seed": seed,
        "device": device,
        "variants": VARIANTS[domain],
        "evaluation_variant": VARIANTS[domain],
        "skip_recurrent_baselines": True,
        "checkpoint_only": True,
        "smoke": False,
    }
    config.update(override)
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for key, value in config.items():
        if value is None or value is False:
            continue
        if isinstance(value, bool):
            command.append(cli_name(key))
        else:
            command.extend([cli_name(key), str(value)])
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(
            f"Command failed ({process.returncode}): {' '.join(command)}\n{tail}"
        )


def checkpoint_for(
    domain: str,
    seed: int,
    root: Path,
) -> Path:
    if seed == 42:
        return SEED42_CHECKPOINTS[domain]
    return root / f"{domain}_seed{seed}" / f"{VARIANTS[domain]}.pt"


def bulk_checkpoint(seed: int) -> Path:
    return (
        ROOT
        / "outputs"
        / f"causal_innovation_state_space_v97_direct_h1_strict_bulk_seed{seed}_2026-07-21"
        / "v97_direct.pt"
    )


def specs_for(seed: int, root: Path) -> str:
    paths = {
        "MDCK_Bulk": bulk_checkpoint(seed),
        **{
            domain: checkpoint_for(domain, seed, root)
            for domain in SOURCE_CONFIGS
        },
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    return ",".join(f"{name}={path}" for name, path in paths.items())


def aggregate_seed_outputs(root: Path, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = root / f"v162_seed{seed}" / "v162_dimensionless_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No v162 seed summaries")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "v162_seed_metrics.csv", index=False)
    aggregate = (
        combined.groupby(
            ["objective", "variant", "control", "dataset", "horizon"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            r2_mean=("r2", "mean"),
            gain_vs_prior_mean=("gain_vs_prior_percent", "mean"),
            seeds_positive=(
                "gain_vs_prior_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    aggregate.to_csv(root / "v162_3seed_aggregate.csv", index=False)
    return aggregate


def aggregate_transfer_outputs(
    root: Path,
    seeds: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = root / f"v162_seed{seed}" / "v162_transfer_matrix.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No v162 seed transfer matrices")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "v162_3seed_transfer_metrics.csv", index=False)
    aggregate = (
        combined.groupby(
            [
                "objective",
                "sources",
                "variant",
                "control",
                "dataset",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            gain_vs_prior_mean=("gain_vs_prior_percent", "mean"),
            gain_vs_prior_std=("gain_vs_prior_percent", "std"),
            seeds_positive=(
                "gain_vs_prior_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    aggregate.to_csv(
        root / "v162_3seed_transfer_aggregate.csv",
        index=False,
    )
    return aggregate


def combine_causal_audits(root: Path, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = root / f"v162_seed{seed}" / "v162_causal_audit.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No v162 causal audits")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "v162_3seed_causal_audit.csv", index=False)
    return combined


def decision_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    selected = aggregate[
        aggregate.objective.eq("h6_guard10")
        & aggregate.variant.eq("lodo_zero_shot")
    ].copy()
    h1 = selected[
        selected.control.eq("real") & selected.horizon.eq(1)
    ][
        [
            "dataset",
            "gain_vs_prior_mean",
            "component_rmse_mean",
        ]
    ].rename(
        columns={
            "gain_vs_prior_mean": "h1_gain_percent",
            "component_rmse_mean": "h1_rmse",
        }
    )
    h6 = selected[selected.horizon.eq(6)][
        [
            "dataset",
            "control",
            "gain_vs_prior_mean",
            "component_rmse_mean",
            "seeds_positive",
        ]
    ]
    gains = h6.pivot(
        index="dataset",
        columns="control",
        values="gain_vs_prior_mean",
    ).add_prefix("h6_gain_")
    rmses = h6.pivot(
        index="dataset",
        columns="control",
        values="component_rmse_mean",
    ).add_prefix("h6_rmse_")
    positives = h6[h6.control.eq("real")][
        ["dataset", "seeds_positive"]
    ].set_index("dataset")
    table = (
        h1.set_index("dataset")
        .join(gains)
        .join(rmses)
        .join(positives)
        .reset_index()
    )
    table["transport_direction_pass"] = (
        table.h6_gain_real.gt(0)
        & table.seeds_positive.eq(3)
        & table.h6_rmse_real.lt(table.h6_rmse_stale_time)
        & table.h6_rmse_real.lt(table.h6_rmse_wrong_cell)
    )
    return table


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def production_wall_seconds(root: Path, seeds: list[int]) -> float:
    starts = [
        getattr(path.stat(), "st_birthtime", path.stat().st_ctime)
        for path in (root / "logs").glob("train_*.log")
    ]
    completions = [
        (root / f"v162_seed{seed}" / "v162_dimensionless_summary.csv").stat().st_mtime
        for seed in seeds
        if (root / f"v162_seed{seed}" / "v162_dimensionless_summary.csv").exists()
    ]
    if not starts or not completions:
        return float("nan")
    return max(0.0, max(completions) - min(starts))


def main() -> None:
    args = parse_args()
    started = time.time()
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = [
        int(token.strip()) for token in args.seeds.split(",") if token.strip()
    ]
    if not args.skip_training:
        for seed in seeds:
            if seed == 42:
                continue
            for domain in SOURCE_CONFIGS:
                out_dir = root / f"{domain}_seed{seed}"
                checkpoint = out_dir / f"{VARIANTS[domain]}.pt"
                if checkpoint.exists():
                    print(f"[v162-seeds] reuse {checkpoint}", flush=True)
                    continue
                print(f"[v162-seeds] train {domain} seed {seed}", flush=True)
                run_logged(
                    training_command(
                        domain,
                        seed,
                        args.device,
                        out_dir,
                    ),
                    root / "logs" / f"train_{domain}_seed{seed}.log",
                )
    if not args.skip_v162:
        for seed in seeds:
            out_dir = root / f"v162_seed{seed}"
            summary = out_dir / "v162_dimensionless_summary.csv"
            if summary.exists():
                print(f"[v162-seeds] reuse {summary}", flush=True)
                continue
            print(f"[v162-seeds] transport seed {seed}", flush=True)
            run_logged(
                [
                    sys.executable,
                    str(V162_SCRIPT),
                    "--specs",
                    specs_for(seed, root),
                    "--device",
                    "cpu",
                    "--out-dir",
                    str(out_dir),
                ],
                root / "logs" / f"v162_seed{seed}.log",
            )
    aggregate = aggregate_seed_outputs(root, seeds)
    transfer_aggregate = aggregate_transfer_outputs(root, seeds)
    causal = combine_causal_audits(root, seeds)
    decision = decision_table(aggregate)
    decision.to_csv(root / "v162_decision.csv", index=False)
    key = aggregate[
        aggregate.objective.eq("h6_guard10")
        & aggregate.control.eq("real")
        & aggregate.variant.isin(
            [
                "lodo_zero_shot",
                "lodo_validation_h1safe",
                "pooled_domain_h1safe",
            ]
        )
        & aggregate.horizon.isin([1, 6])
    ].copy()
    transfer_h6 = transfer_aggregate[
        transfer_aggregate.objective.eq("h6_guard10")
        & transfer_aggregate.control.eq("real")
        & transfer_aggregate.horizon.eq(6)
    ].pivot(
        index="sources",
        columns="dataset",
        values="gain_vs_prior_mean",
    )
    causal_ok = (
        int(causal.real_future_donor_violations.sum()) == 0
        and int(causal.stale_future_or_nonstale_violations.sum()) == 0
        and bool(causal.coherent_wrong_packet.all())
        and not bool(causal.uses_whole_movie_xy_normalization.any())
    )
    production_elapsed = production_wall_seconds(root, seeds)
    report = [
        "# v162 External Seed Confirmation",
        "",
        "## Primary h1/h6 table",
        "",
        key.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Source-to-target h6 gain (%)",
        "",
        transfer_h6.to_markdown(floatfmt=".3f"),
        "",
        "## Decision table",
        "",
        decision.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Causal audit passed: `{causal_ok}`.",
        "",
        "The seed unit changes the domain-specific base prior and refits the",
        "dimensionless transport. LODO remains a transport-kernel claim, not",
        "zero-shot transfer of the entire base forecaster.",
        "",
        (
            "Production wall interval: "
            f"`{production_elapsed / 3600.0:.2f} h` "
            "(includes the interrupted legacy postprocessing pass)."
        ),
    ]
    (root / "v162_3seed_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            finite(
                {
                    "seeds": seeds,
                    "device": args.device,
                    "postprocess_elapsed_sec": time.time() - started,
                    "production_wall_sec": production_elapsed,
                    "source_configs": SOURCE_CONFIGS,
                    "variants": VARIANTS,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(root / "v162_3seed_report.md", flush=True)


if __name__ == "__main__":
    main()
