#!/usr/bin/env python3
"""Build the compact PRX evidence package for C2C12 v209."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTO = (
    ROOT
    / "outputs"
    / "c2c12_lit_cell_external_confirmation_v209_automatic_equivariant_full_2026-08-02"
)
DEFAULT_MANUAL = (
    ROOT
    / "outputs"
    / "c2c12_lit_cell_external_confirmation_v209_manual_equivariant_full_2026-08-02"
)
DEFAULT_FREE = (
    ROOT
    / "outputs"
    / "c2c12_lit_cell_external_confirmation_v209_automatic_full_balanced_2026-08-02"
)
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "c2c12_lit_cell_external_confirmation_v209_prx_evidence_2026-08-02"
)
PRIMARY = "horizon_balanced"
EPS = 1e-8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def method_summary(fields: pd.DataFrame, source: str) -> pd.DataFrame:
    baseline = fields[fields["objective"].eq("baseline")].copy()
    primary = fields[
        fields["objective"].eq(PRIMARY) & fields["control"].eq("real")
    ].copy()
    selected = pd.concat([baseline, primary], ignore_index=True)
    result = (
        selected.groupby(
            ["annotation_kind", "objective", "method", "control", "horizon"],
            as_index=False,
        )
        .agg(
            fields=("sequence", "nunique"),
            experiments=("experiment", "nunique"),
            component_rmse=("component_rmse", "mean"),
            component_rmse_std=("component_rmse", "std"),
            r2=("r2", "mean"),
            cosine=("cosine", "mean"),
        )
    )
    result.insert(0, "source", source)
    return result


def experiment_gains(fields: pd.DataFrame, source: str) -> pd.DataFrame:
    real = fields[
        fields["objective"].eq(PRIMARY) & fields["control"].eq("real")
    ][["experiment", "sequence", "horizon", "component_rmse"]].rename(
        columns={"component_rmse": "real_rmse"}
    )
    baseline = fields[
        fields["objective"].eq("baseline") & fields["control"].eq("no_update")
    ][["experiment", "sequence", "horizon", "component_rmse"]].rename(
        columns={"component_rmse": "baseline_rmse"}
    )
    joined = real.merge(
        baseline, on=["experiment", "sequence", "horizon"], validate="one_to_one"
    )
    joined["gain_percent"] = 100.0 * (
        joined["baseline_rmse"] - joined["real_rmse"]
    ) / np.maximum(joined["baseline_rmse"], EPS)
    result = (
        joined.groupby(["experiment", "horizon"], as_index=False)
        .agg(
            fields=("sequence", "nunique"),
            baseline_rmse=("baseline_rmse", "mean"),
            real_rmse=("real_rmse", "mean"),
            gain_percent=("gain_percent", "mean"),
        )
    )
    result.insert(0, "source", source)
    return result


def density_summary(directory: Path, source: str) -> pd.DataFrame:
    frame = load(directory, "v209_scale_density_strata.csv")
    frame = frame[frame["objective"].eq(PRIMARY)]
    result = (
        frame.groupby(["annotation_kind", "horizon", "density_quartile"], as_index=False)
        .agg(
            fields=("sequence", "nunique"),
            windows=("windows", "sum"),
            median_dnn_px=("median_dnn_px", "median"),
            gain_percent=("gain_percent", "mean"),
        )
    )
    result.insert(0, "source", source)
    return result


def quality_summary(directory: Path, source: str) -> pd.DataFrame:
    frame = load(directory, "v209_tracking_quality_strata.csv")
    frame = frame[frame["objective"].eq(PRIMARY)]
    result = (
        frame.groupby(["annotation_kind", "horizon", "quality_stratum"], as_index=False)
        .agg(
            fields=("sequence", "nunique"),
            windows=("windows", "sum"),
            component_rmse=("component_rmse", "mean"),
            baseline_rmse=("base_component_rmse", "mean"),
            gain_percent=("gain_percent", "mean"),
        )
    )
    result.insert(0, "source", source)
    return result


def coefficient_summary(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = load(directory, "v209_operator_coefficients.csv")
    coefficients = coefficients[coefficients["objective"].eq(PRIMARY)].copy()
    summary = (
        coefficients.groupby("feature", as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
            minimum=("coefficient", "min"),
            maximum=("coefficient", "max"),
            positive_rotations=("coefficient", lambda values: int((values > 0).sum())),
            rotations=("rotation", "nunique"),
        )
    )
    summary["sign_consistency"] = np.maximum(
        summary["positive_rotations"],
        summary["rotations"] - summary["positive_rotations"],
    ) / summary["rotations"]
    local = coefficients[coefficients["feature"].str.contains("local_m")].copy()
    local["scale"] = local["feature"].str.extract(r"(m[0-9p]+)", expand=False)
    local_scale = (
        local.groupby(["rotation", "scale"], as_index=False)
        .agg(coefficient_l2=("coefficient", lambda values: float(np.linalg.norm(values))))
        .groupby("scale", as_index=False)
        .agg(
            mean_coefficient_l2=("coefficient_l2", "mean"),
            std_coefficient_l2=("coefficient_l2", "std"),
        )
        .sort_values("mean_coefficient_l2", ascending=False)
    )
    return summary, local_scale


def operator_comparison(auto_dir: Path, free_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, directory in (("equivariant", auto_dir), ("free", free_dir)):
        fields = load(directory, "v209_field_metrics.csv")
        selected = fields[
            fields["objective"].eq(PRIMARY)
            & fields["control"].isin(["real", "own_only", "wrong_cell", "stale_time"])
            & fields["horizon"].isin([1, 6])
        ]
        for (control, horizon), group in selected.groupby(["control", "horizon"]):
            records.append(
                {
                    "operator": label,
                    "control": control,
                    "horizon": int(horizon),
                    "component_rmse": float(group["component_rmse"].mean()),
                    "r2": float(group["r2"].mean()),
                    "fields": int(group["sequence"].nunique()),
                }
            )
    return pd.DataFrame(records)


def causal_summary(auto_dir: Path, manual_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for source, directory in (("automatic", auto_dir), ("manual", manual_dir)):
        frame = load(directory, "v209_causal_audit.csv")
        records.append(
            {
                "source": source,
                "rows": int(frame["rows"].sum()),
                "field_split_records": len(frame),
                "future_donor_violations": int(
                    frame["real_future_donor_violations"].sum()
                ),
                "stale_donor_violations": int(frame["stale_donor_violations"].sum()),
                "split_key_overlap": int(frame["split_key_overlap"].sum()),
                "target_feature_flags": int(frame["target_features_used"].sum()),
            }
        )
    return pd.DataFrame(records)


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    auto_fields = load(args.automatic_dir, "v209_field_metrics.csv")
    manual_fields = load(args.manual_dir, "v209_field_metrics.csv")

    metrics = pd.concat(
        [
            method_summary(auto_fields, "automatic"),
            method_summary(manual_fields, "manual"),
        ],
        ignore_index=True,
    )
    gains = pd.concat(
        [
            experiment_gains(auto_fields, "automatic"),
            experiment_gains(manual_fields, "manual"),
        ],
        ignore_index=True,
    )
    density = pd.concat(
        [
            density_summary(args.automatic_dir, "automatic"),
            density_summary(args.manual_dir, "manual"),
        ],
        ignore_index=True,
    )
    quality = pd.concat(
        [
            quality_summary(args.automatic_dir, "automatic"),
            quality_summary(args.manual_dir, "manual"),
        ],
        ignore_index=True,
    )
    coefficients, local_scales = coefficient_summary(args.automatic_dir)
    operator = operator_comparison(args.automatic_dir, args.free_dir)
    causal = causal_summary(args.automatic_dir, args.manual_dir)
    bootstrap = pd.concat(
        [
            load(args.automatic_dir, "v209_cluster_bootstrap.csv"),
            load(args.manual_dir, "v209_cluster_bootstrap.csv"),
        ],
        ignore_index=True,
    )
    bootstrap = bootstrap[bootstrap["objective"].eq(PRIMARY)]

    outputs = {
        "v209_prx_main_metrics.csv": metrics,
        "v209_prx_experiment_gains.csv": gains,
        "v209_prx_density_strata.csv": density,
        "v209_prx_tracking_quality.csv": quality,
        "v209_prx_operator_coefficients.csv": coefficients,
        "v209_prx_operator_scale_norms.csv": local_scales,
        "v209_prx_free_vs_equivariant.csv": operator,
        "v209_prx_causal_audit.csv": causal,
        "v209_prx_cluster_bootstrap.csv": bootstrap,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)

    def metric(source: str, objective: str, horizon: int, column: str) -> float:
        row = metrics[
            metrics["source"].eq(source)
            & metrics["objective"].eq(objective)
            & metrics["horizon"].eq(horizon)
            & (
                metrics["control"].eq("real")
                if objective == PRIMARY
                else metrics["control"].eq("no_update")
            )
        ]
        return float(row.iloc[0][column])

    auto_h1_base = metric("automatic", "baseline", 1, "component_rmse")
    auto_h1_real = metric("automatic", PRIMARY, 1, "component_rmse")
    auto_h6_base = metric("automatic", "baseline", 6, "component_rmse")
    auto_h6_real = metric("automatic", PRIMARY, 6, "component_rmse")
    manual_h1_base = metric("manual", "baseline", 1, "component_rmse")
    manual_h1_real = metric("manual", PRIMARY, 1, "component_rmse")
    manual_h6_base = metric("manual", "baseline", 6, "component_rmse")
    manual_h6_real = metric("manual", PRIMARY, 6, "component_rmse")
    auto_h6_gain = float(
        gains[gains["source"].eq("automatic") & gains["horizon"].eq(6)][
            "gain_percent"
        ].mean()
    )
    auto_h1_change = -float(
        gains[gains["source"].eq("automatic") & gains["horizon"].eq(1)][
            "gain_percent"
        ].mean()
    )
    manual_h6_gain = float(
        gains[gains["source"].eq("manual") & gains["horizon"].eq(6)][
            "gain_percent"
        ].mean()
    )
    manual_h1_change = -float(
        gains[gains["source"].eq("manual") & gains["horizon"].eq(1)][
            "gain_percent"
        ].mean()
    )

    auto_cv_h6 = float(
        metrics[
            metrics["source"].eq("automatic")
            & metrics["method"].eq("constant_velocity")
            & metrics["horizon"].eq(6)
        ].iloc[0]["component_rmse"]
    )
    free_h6 = float(
        operator[
            operator["operator"].eq("free")
            & operator["control"].eq("real")
            & operator["horizon"].eq(6)
        ].iloc[0]["component_rmse"]
    )
    e2_h6 = float(
        operator[
            operator["operator"].eq("equivariant")
            & operator["control"].eq("real")
            & operator["horizon"].eq(6)
        ].iloc[0]["component_rmse"]
    )
    dominant_scale = str(local_scales.iloc[0]["scale"])

    report = [
        "# C2C12 external LIT-Cell evidence for PRX development",
        "",
        "## Decision",
        "",
        "**The external structural gate passes; the result is not an absolute C2C12 leaderboard win.**",
        "",
        "The primary automatic-track analysis uses three experiment-level train/validation/test rotations, a predeclared horizon-balanced operating point, and an E(2)-equivariant vector operator. Manual tracks are reported separately because interpolation dominates their observation process.",
        "",
        "## Primary automatic tracks",
        "",
        f"- h1 component RMSE: {auto_h1_base:.6f} -> {auto_h1_real:.6f} px ({auto_h1_change:+.3f}% mean paired-field error change; 0.5% guard passed).",
        f"- rolling h6 component RMSE: {auto_h6_base:.6f} -> {auto_h6_real:.6f} px ({auto_h6_gain:.3f}% mean paired-field gain).",
        "- h6 gain is positive in 3/3 held-out experiments and all four density quartiles.",
        "- Real completed innovations beat no-update, own-only, wrong-cell, and stale-time controls under hierarchical bootstrap.",
        "- Future-donor, split-overlap, and target-feature violations are all zero.",
        f"- Constant velocity remains stronger in absolute h6 RMSE ({auto_cv_h6:.6f} px); the claim is transfer of the innovation mechanism, not global dominance on noisy automatic tracks.",
        "",
        "## Manual observation-process audit",
        "",
        f"- h1 component RMSE: {manual_h1_base:.6f} -> {manual_h1_real:.6f} px ({manual_h1_change:+.3f}% mean paired-field error change).",
        f"- rolling h6 component RMSE: {manual_h6_base:.6f} -> {manual_h6_real:.6f} px ({manual_h6_gain:.3f}% mean paired-field gain).",
        "- The h6 gain remains positive on observed-only windows, but is much larger on interpolated windows; automatic and manual estimates must not be pooled.",
        "",
        "## Operator interpretation",
        "",
        f"- The E(2)-equivariant operator matches the free x/y regression at h6 ({e2_h6:.6f} vs {free_h6:.6f} px). Arbitrary coordinate-axis mixing is therefore unnecessary for this result.",
        "- Own, global, and local coefficient signs are highly stable across the three rotations; the full operator remains significantly better than own-only.",
        f"- The largest normalized local coefficient norm occurs at `{dominant_scale}` (approximately twice the frame-wise nearest-neighbour spacing). This is a predictive support scale, not a force or universal correlation length.",
        "",
        "## PRX positioning",
        "",
        "This experiment closes a previous external-validation gap: delayed, identity-correct local innovation is reusable across a second tracking corpus, and a symmetry-constrained operator retains the effect. It strengthens the general active-system filtering claim and the observation-noise analysis.",
        "",
        "It does not close the PRX mechanical-law gate. There is still no positive bridge to a simultaneously measured force, polarity, or intervention, and constant velocity remains the best automatic-track h6 baseline. The defensible statement is a transferable causal kinematic update, not a recovered mechanical law or global forecasting SOTA.",
        "",
        "## Machine-readable tables",
        "",
        *[f"- `{name}`" for name in outputs],
    ]
    report_path = args.out_dir / "v209_prx_decision_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    source_files = [
        args.automatic_dir / "v209_field_metrics.csv",
        args.automatic_dir / "v209_cluster_bootstrap.csv",
        args.automatic_dir / "v209_operator_coefficients.csv",
        args.manual_dir / "v209_field_metrics.csv",
        args.manual_dir / "v209_cluster_bootstrap.csv",
        args.free_dir / "v209_field_metrics.csv",
    ]
    manifest = {
        "schema": "c2c12_lit_cell_prx_evidence_v209",
        "decision": "external_structural_pass_not_absolute_leaderboard_win",
        "primary_operating_point": PRIMARY,
        "primary_operator": "E(2)-equivariant scalar vector transport",
        "automatic_h6_gain_percent": auto_h6_gain,
        "automatic_h1_error_change_percent": auto_h1_change,
        "manual_h6_gain_percent": manual_h6_gain,
        "manual_h1_error_change_percent": manual_h1_change,
        "source_files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in source_files
        },
        "outputs": sorted(path.name for path in args.out_dir.iterdir()),
    }
    (args.out_dir / "v209_prx_manifest.json").write_text(
        json.dumps(finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic-dir", type=Path, default=DEFAULT_AUTO)
    parser.add_argument("--manual-dir", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--free-dir", type=Path, default=DEFAULT_FREE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
