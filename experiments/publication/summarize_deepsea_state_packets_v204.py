#!/usr/bin/env python3
"""Summarize DeepSea state-family ablations at the independent movie level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-8


def bootstrap_gain(
    baseline: np.ndarray, candidate: np.ndarray, repeats: int, seed: int
) -> tuple[float, float, float]:
    point = 100.0 * (float(np.mean(baseline)) - float(np.mean(candidate))) / max(
        float(np.mean(baseline)), EPS
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(baseline), len(baseline))
        sampled_baseline = float(np.mean(baseline[indices]))
        sampled_candidate = float(np.mean(candidate[indices]))
        samples[repeat] = (
            100.0
            * (sampled_baseline - sampled_candidate)
            / max(sampled_baseline, EPS)
        )
    return (
        point,
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def holm_adjust(values: list[float]) -> list[float]:
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result.tolist()
    ordered = valid[np.argsort(np.asarray(values)[valid])]
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(values[index]))
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def packet_result(
    packet: str,
    directory: Path,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    movie = pd.read_csv(directory / "v204_feature_triage_movie_metrics.csv")
    h6 = movie[(movie.horizon == 6) & movie.control.isin(["real", "zero"])]
    paired = (
        h6.pivot_table(
            index=["sequence", "family", "video"],
            columns="control",
            values="component_rmse",
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    mean_gain, ci_low, ci_high = bootstrap_gain(
        paired.zero.to_numpy(), paired.real.to_numpy(), repeats, seed
    )
    difference = paired.zero.to_numpy() - paired.real.to_numpy()
    try:
        p_value = float(
            wilcoxon(
                difference,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except ValueError:
        p_value = 1.0
    controls = movie[(movie.horizon == 6) & (movie.method != "route_probe")]
    macro = controls.groupby("control").component_rmse.mean()
    probes = movie[movie.method == "route_probe"].set_index("control")
    return {
        "packet": packet,
        "movies": len(paired),
        "real_h6_rmse": float(macro.get("real", np.nan)),
        "zero_h6_rmse": float(macro.get("zero", np.nan)),
        "mean_gain_pct": mean_gain,
        "gain_ci_low": ci_low,
        "gain_ci_high": ci_high,
        "positive_movies": int(np.sum(difference > 0)),
        "wilcoxon_p": p_value,
        "real_beats_all_controls": bool(
            "real" in macro
            and all(
                float(macro["real"]) < float(macro[name])
                for name in (
                    "row_shuffled",
                    "time_shuffled",
                    "wrong_cell",
                    "wrong_video",
                )
                if name in macro
            )
        ),
        "route_top3_real": float(probes.route_top3.get("real", np.nan)),
        "route_top3_zero": float(probes.route_top3.get("zero", np.nan)),
        "route_top3_delta": float(
            probes.route_top3.get("real", np.nan)
            - probes.route_top3.get("zero", np.nan)
        ),
    }


def run(args: argparse.Namespace) -> None:
    packet_dirs = {
        "full": args.full_dir,
        "shape": args.shape_dir,
        "polarity": args.polarity_dir,
        "contact": args.contact_dir,
        "reliability": args.reliability_dir,
        "shape_contact": args.shape_contact_dir,
        "shape_polarity": args.shape_polarity_dir,
    }
    rows = [
        packet_result(packet, directory, args.bootstrap_repeats, args.seed + index)
        for index, (packet, directory) in enumerate(packet_dirs.items())
        if (directory / "v204_feature_triage_movie_metrics.csv").exists()
    ]
    table = pd.DataFrame(rows)
    table["holm_p"] = holm_adjust(table.wilcoxon_p.tolist())
    table["passes_preregistered_gate"] = (
        (table.mean_gain_pct >= 3.0)
        & table.real_beats_all_controls
        & (table.gain_ci_low > 0.0)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "v204_state_packet_summary.csv", index=False)
    best = table.sort_values("mean_gain_pct", ascending=False).iloc[0]
    decision = {
        "best_packet": str(best.packet),
        "best_gain_pct": float(best.mean_gain_pct),
        "best_gain_ci": [float(best.gain_ci_low), float(best.gain_ci_high)],
        "best_positive_movies": int(best.positive_movies),
        "best_holm_p": float(best.holm_p),
        "any_packet_passes": bool(table.passes_preregistered_gate.any()),
    }
    (args.out_dir / "v204_state_packet_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "# DeepSea v204 State-Packet Audit",
        "",
        f"- Best packet: `{decision['best_packet']}`.",
        f"- Movie-macro h6 gain: `{decision['best_gain_pct']:.3f}%`.",
        (
            "- Movie bootstrap 95% interval: "
            f"`[{decision['best_gain_ci'][0]:.3f}, {decision['best_gain_ci'][1]:.3f}]%`."
        ),
        f"- Positive outer movies: `{decision['best_positive_movies']}`.",
        f"- Holm-adjusted paired p-value: `{decision['best_holm_p']:.6f}`.",
        f"- Any packet passes the frozen 3% gate: `{decision['any_packet_passes']}`.",
    ]
    (args.out_dir / "v204_state_packet_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_full_mask_feature_triage_v204_2026-07-31",
    )
    parser.add_argument(
        "--shape-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_shape_v204_2026-07-31",
    )
    parser.add_argument(
        "--polarity-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_polarity_v204_2026-07-31",
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_contact_v204_2026-07-31",
    )
    parser.add_argument(
        "--reliability-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_reliability_v204_2026-07-31",
    )
    parser.add_argument(
        "--shape-contact-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_shape_contact_v204_2026-07-31",
    )
    parser.add_argument(
        "--shape-polarity-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_shape_polarity_v204_2026-07-31",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/deepsea_state_packet_audit_v204_2026-07-31",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
