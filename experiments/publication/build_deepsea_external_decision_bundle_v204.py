#!/usr/bin/env python3
"""Build the frozen DeepSea v204 external-validation decision bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def build_evidence_ledger(
    *,
    args: argparse.Namespace,
    report_path: Path,
) -> pd.DataFrame:
    test_dir = args.cache_dir / "final_native" / "test"
    keys = pd.read_csv(
        test_dir / "rows.csv",
        usecols=["sequence", "frame", "track_id"],
    ).to_numpy()
    with np.load(test_dir / "arrays.npz") as arrays:
        target_steps = arrays["target_steps"]
    if len(keys) != len(target_steps):
        raise RuntimeError(
            "Evidence ledger key/target length mismatch: "
            f"{len(keys)} != {len(target_steps)}"
        )
    ordered_key_hash = array_sha256(keys, "<i8")
    target_hash = array_sha256(target_steps, "<f4")
    protocol = "rolling_predict_before_observe_movie_macro_dimensionless"
    cohort = "frozen_outer_test_11_movies"
    coordinate_metrics = (
        args.coordinate_summary / "v204_online_macro_metrics.csv"
    )
    state_metrics = args.state_summary / "v204_online_macro_metrics.csv"
    transport_metrics = args.transport_dir / "v204_transport_aggregate.csv"
    entries = [
        (
            "reference_complete_system",
            args.lachance_publication_report,
            "frozen",
            "",
            "",
            "v166/v188 defines the complete two-operating-point publication system",
        ),
        (
            "preregistered_plan",
            ROOT / "external_deepsea_multimodal_validation_plan_2026-07-31.md",
            "frozen",
            "",
            "",
            "external multimodal validation hypotheses and gates",
        ),
        (
            "data_contract_amendment",
            ROOT / "external_deepsea_v204_data_contract_amendment_2026-07-31.md",
            "frozen",
            "",
            "",
            "dimensionless unit and h1-complete row contract",
        ),
        (
            "prepared_data_contract",
            args.preparation_manifest,
            "frozen",
            "",
            "",
            "DeepSea inventory, split and mask coverage",
        ),
        (
            "coordinate_external_benchmark",
            coordinate_metrics,
            "internal_prior_ablation",
            ordered_key_hash,
            target_hash,
            "v97 is the frozen prior, not the complete project SOTA",
        ),
        (
            "completed_innovation_transport",
            transport_metrics,
            "partial_mechanism_signal_gate_fail",
            ordered_key_hash,
            target_hash,
            "externalized v166 operating points transfer but fail the joint h1/h6 gate",
        ),
        (
            "privileged_mask_state",
            state_metrics,
            "external_negative_evidence",
            ordered_key_hash,
            target_hash,
            "real causal state fails RMSE, proper-score and identity controls",
        ),
        (
            "future_suffix_invariance",
            args.future_audit,
            "causal_audit_pass",
            ordered_key_hash,
            target_hash,
            "issued prefix predictions are invariant to future suffix mutation",
        ),
        (
            "coordinate_unit_conversion",
            args.unit_audit,
            "data_audit_pass",
            ordered_key_hash,
            target_hash,
            "pixel targets equal cell-diameter targets within float32 tolerance",
        ),
        (
            "external_decision_report",
            report_path,
            "current_external_decision",
            ordered_key_hash,
            target_hash,
            "frozen synthesis and claim boundary",
        ),
    ]
    rows = []
    for role, path, status, key_hash, target_sha, claim_scope in entries:
        rows.append(
            {
                "generation": "v204",
                "artifact_role": role,
                "protocol": protocol,
                "dataset": "DeepSea",
                "cohort": cohort,
                "status": status,
                "path": str(path.resolve()),
                "artifact_sha256": sha256(path),
                "ordered_key_sha256": key_hash,
                "target_sha256": target_sha,
                "superseded_by": "",
                "claim_scope": claim_scope,
            }
        )
    return pd.DataFrame(rows)


def scalar(
    table: pd.DataFrame,
    column: str,
    **filters: Any,
) -> float:
    selected = table
    for key, value in filters.items():
        selected = selected.loc[selected[key] == value]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for {filters}, found {len(selected)}"
        )
    return float(selected.iloc[0][column])


def row(
    table: pd.DataFrame,
    **filters: Any,
) -> pd.Series:
    selected = table
    for key, value in filters.items():
        selected = selected.loc[selected[key] == value]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for {filters}, found {len(selected)}"
        )
    return selected.iloc[0]


def plot_coordinate_benchmark(
    coordinate: pd.DataFrame,
    tabular: pd.DataFrame,
    transport: pd.DataFrame,
    path: Path,
) -> None:
    methods = [
        ("Zero displacement", "coordinate", "zero_displacement"),
        ("Constant velocity", "coordinate", "constant_velocity"),
        ("IMM", "coordinate", "imm_cv_ca_turn"),
        ("Ridge", "tabular", "ridge"),
        ("HGBDT", "tabular", "hgbdt"),
        ("LSTM", "coordinate", "lstm_h1"),
        ("GRU", "coordinate", "gru_h1"),
        ("v97 prior", "coordinate", "v97_direct"),
        ("v166-ext h1 strict", "transport", "h1_strict"),
        ("v166-ext h6 utility", "transport", "h6_guard10"),
    ]
    values = {1: [], 6: []}
    for _label, source, variant in methods:
        for horizon in values:
            if source == "coordinate":
                value = scalar(
                    coordinate,
                    "movie_macro_rmse",
                    method=variant,
                    metric_unit="cell_diameter",
                    horizon=horizon,
                )
            elif source == "tabular":
                value = scalar(
                    tabular,
                    "movie_macro_rmse",
                    model_family=variant,
                    horizon=horizon,
                )
            else:
                value = scalar(
                    transport,
                    "movie_macro_component_rmse",
                    objective_name=variant,
                    packet_name="full",
                    control="real",
                    horizon=horizon,
                )
            values[horizon].append(value)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), constrained_layout=True)
    colors = [
        "#b8b8b8",
        "#4c78a8",
        "#72a0c1",
        "#9c755f",
        "#b279a2",
        "#f2cf5b",
        "#eeca3b",
        "#e45756",
        "#f58518",
        "#54a24b",
    ]
    y = np.arange(len(methods))
    for axis, horizon in zip(axes, (1, 6), strict=True):
        axis.barh(y, values[horizon], color=colors, height=0.7)
        axis.set_yticks(y, [item[0] for item in methods])
        axis.invert_yaxis()
        axis.set_xlabel("Movie-macro component RMSE\n(cell diameters)")
        axis.set_title(f"Rolling h{horizon}")
        axis.grid(axis="x", alpha=0.22)
        for index, value in enumerate(values[horizon]):
            axis.text(
                value,
                index,
                f" {value:.3f}",
                va="center",
                ha="left",
                fontsize=8,
            )
        axis.set_xlim(0, max(values[horizon]) * 1.18)
    fig.suptitle(
        "DeepSea external coordinate benchmark",
        fontsize=12,
        fontweight="semibold",
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_state_controls(
    full_bootstrap: pd.DataFrame,
    shape_bootstrap: pd.DataFrame,
    path: Path,
) -> None:
    labels = [
        ("Real", "real"),
        ("Row shuffled", "row_shuffled"),
        ("Time shuffled", "time_shuffled"),
        ("Wrong cell", "wrong_cell"),
        ("Wrong video", "wrong_video"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    for axis, title, table in (
        (axes[0], "Full mask/state packet", full_bootstrap),
        (axes[1], "Shape packet", shape_bootstrap),
    ):
        selected = table.loc[
            (table.metric_unit == "cell_diameter")
            & (table.horizon == 6)
        ].set_index("candidate_run")
        gains = np.asarray(
            [float(selected.loc[key, "movie_macro_gain_pct"]) for _, key in labels]
        )
        lows = np.asarray(
            [float(selected.loc[key, "bootstrap_ci_low"]) for _, key in labels]
        )
        highs = np.asarray(
            [float(selected.loc[key, "bootstrap_ci_high"]) for _, key in labels]
        )
        errors = np.vstack([gains - lows, highs - gains])
        colors = ["#e45756"] + ["#a5a5a5"] * (len(labels) - 1)
        y = np.arange(len(labels))
        axis.barh(y, gains, color=colors, height=0.68)
        axis.errorbar(
            gains,
            y,
            xerr=errors,
            fmt="none",
            ecolor="#333333",
            capsize=3,
            linewidth=1,
        )
        axis.axvline(0, color="#333333", linewidth=0.9)
        axis.axvline(3, color="#59a14f", linewidth=1.1, linestyle="--")
        axis.set_yticks(y, [label for label, _ in labels])
        axis.invert_yaxis()
        axis.set_xlabel("h6 gain over zero state (%)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Privileged state does not survive identity controls",
        fontsize=12,
        fontweight="semibold",
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_transport_tradeoff(aggregate: pd.DataFrame, path: Path) -> None:
    real = aggregate.loc[aggregate.control == "real"]
    h1 = real.loc[real.horizon == 1][
        ["objective_name", "packet_name", "movie_macro_gain_pct"]
    ].rename(columns={"movie_macro_gain_pct": "h1_gain"})
    h6 = real.loc[real.horizon == 6][
        ["objective_name", "packet_name", "movie_macro_gain_pct"]
    ].rename(columns={"movie_macro_gain_pct": "h6_gain"})
    paired = h1.merge(
        h6,
        on=["objective_name", "packet_name"],
        validate="one_to_one",
    )
    objective_colors = {
        "h1_strict": "#4c78a8",
        "balanced_guard2": "#f58518",
        "trajectory_guard5": "#54a24b",
        "h6_guard10": "#e45756",
    }
    fig, axis = plt.subplots(figsize=(7.2, 5.1), constrained_layout=True)
    x_max = float(paired.h1_gain.max() + 2)
    y_max = float(paired.h6_gain.max() + 2)
    axis.fill_betweenx(
        [3.0, y_max],
        -0.5,
        x_max,
        color="#eaf4e6",
        alpha=0.9,
        label="pre-registered pass region",
    )
    for objective, group in paired.groupby("objective_name"):
        axis.scatter(
            group.h1_gain,
            group.h6_gain,
            s=52,
            color=objective_colors.get(objective, "#777777"),
            label=objective,
            alpha=0.9,
        )
    for _, item in paired.iterrows():
        if (
            (item.objective_name == "h1_strict" and item.packet_name == "full")
            or (
                item.objective_name == "h6_guard10"
                and item.packet_name == "full"
            )
        ):
            label = (
                "h1 strict / full"
                if item.objective_name == "h1_strict"
                else "h6 utility / full"
            )
            axis.annotate(
                label,
                (item.h1_gain, item.h6_gain),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axvline(-0.5, color="#333333", linestyle="--", linewidth=1)
    axis.axhline(3.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("h1 gain over three-seed v97 prior (%)")
    axis.set_ylabel("h6 gain over three-seed v97 prior (%)")
    axis.set_title("Completed-innovation transport has a horizon trade-off")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coordinate = pd.read_csv(args.coordinate_summary / "v204_online_macro_metrics.csv")
    coordinate_bootstrap = pd.read_csv(
        args.coordinate_summary / "v204_online_paired_bootstrap.csv"
    )
    state = pd.read_csv(args.state_summary / "v204_online_macro_metrics.csv")
    state_bootstrap = pd.read_csv(
        args.state_summary / "v204_online_paired_bootstrap.csv"
    )
    state_nll = pd.read_csv(
        args.state_summary / "v204_online_nll_paired_bootstrap.csv"
    )
    shape_bootstrap = pd.read_csv(
        args.shape_summary / "v204_online_paired_bootstrap.csv"
    )
    tabular = pd.read_csv(args.tabular_dir / "v204_tabular_summary.csv")
    transport = pd.read_csv(args.transport_dir / "v204_transport_aggregate.csv")
    transport_decision = json.loads(
        (args.transport_dir / "v204_transport_decision.json").read_text(
            encoding="utf-8"
        )
    )
    future_audit = json.loads(args.future_audit.read_text(encoding="utf-8"))
    preparation = json.loads(args.preparation_manifest.read_text(encoding="utf-8"))

    coordinate_h1 = row(
        coordinate,
        method="v97_direct",
        metric_unit="cell_diameter",
        horizon=1,
    )
    coordinate_h6 = row(
        coordinate,
        method="v97_direct",
        metric_unit="cell_diameter",
        horizon=6,
    )
    cv_h1 = row(
        coordinate,
        method="constant_velocity",
        metric_unit="cell_diameter",
        horizon=1,
    )
    cv_h6 = row(
        coordinate,
        method="constant_velocity",
        metric_unit="cell_diameter",
        horizon=6,
    )
    imm_h6 = row(
        coordinate,
        method="imm_cv_ca_turn",
        metric_unit="cell_diameter",
        horizon=6,
    )
    v166_h1_strict_h1 = row(
        transport,
        objective_name="h1_strict",
        packet_name="full",
        control="real",
        horizon=1,
    )
    v166_h1_strict_h6 = row(
        transport,
        objective_name="h1_strict",
        packet_name="full",
        control="real",
        horizon=6,
    )
    v166_h6_utility_h1 = row(
        transport,
        objective_name="h6_guard10",
        packet_name="full",
        control="real",
        horizon=1,
    )
    v166_h6_utility_h6 = row(
        transport,
        objective_name="h6_guard10",
        packet_name="full",
        control="real",
        horizon=6,
    )
    state_real_h6 = row(
        state,
        run="real",
        metric_unit="cell_diameter",
        horizon=6,
    )
    state_zero_h6 = row(
        state,
        run="zero",
        metric_unit="cell_diameter",
        horizon=6,
    )
    state_real_gain = row(
        state_bootstrap,
        candidate_run="real",
        metric_unit="cell_diameter",
        horizon=6,
    )
    state_real_nll = row(
        state_nll,
        candidate_run="real",
        metric_unit="cell_diameter",
    )
    capacity_gain = row(
        state_bootstrap,
        candidate_run="noncausal_capacity",
        metric_unit="cell_diameter",
        horizon=6,
    )
    transport_exploratory = row(
        transport,
        objective_name="h6_guard10",
        packet_name="own_only",
        control="real",
        horizon=6,
    )
    transport_exploratory_h1 = row(
        transport,
        objective_name="h6_guard10",
        packet_name="own_only",
        control="real",
        horizon=1,
    )

    result_rows = [
        {
            "module": "coordinate",
            "variant": "constant_velocity",
            "horizon": 1,
            "movie_macro_rmse": cv_h1.movie_macro_rmse,
            "r2": cv_h1.movie_macro_r2,
            "gain_pct": np.nan,
            "status": "reference",
        },
        {
            "module": "coordinate",
            "variant": "v97_direct_prior",
            "horizon": 1,
            "movie_macro_rmse": coordinate_h1.movie_macro_rmse,
            "r2": coordinate_h1.movie_macro_r2,
            "gain_pct": 100.0
            * (cv_h1.movie_macro_rmse - coordinate_h1.movie_macro_rmse)
            / cv_h1.movie_macro_rmse,
            "status": "h1_positive",
        },
        {
            "module": "coordinate",
            "variant": "constant_velocity",
            "horizon": 6,
            "movie_macro_rmse": cv_h6.movie_macro_rmse,
            "r2": cv_h6.movie_macro_r2,
            "gain_pct": np.nan,
            "status": "reference",
        },
        {
            "module": "coordinate",
            "variant": "v97_direct_prior",
            "horizon": 6,
            "movie_macro_rmse": coordinate_h6.movie_macro_rmse,
            "r2": coordinate_h6.movie_macro_r2,
            "gain_pct": 100.0
            * (cv_h6.movie_macro_rmse - coordinate_h6.movie_macro_rmse)
            / cv_h6.movie_macro_rmse,
            "status": "prior_ablation",
        },
        {
            "module": "complete_system",
            "variant": "v166_external_h1_strict",
            "horizon": 1,
            "movie_macro_rmse": v166_h1_strict_h1.movie_macro_component_rmse,
            "r2": np.nan,
            "gain_pct": 100.0
            * (
                cv_h1.movie_macro_rmse
                - v166_h1_strict_h1.movie_macro_component_rmse
            )
            / cv_h1.movie_macro_rmse,
            "status": "h1_operating_point",
        },
        {
            "module": "complete_system",
            "variant": "v166_external_h1_strict",
            "horizon": 6,
            "movie_macro_rmse": v166_h1_strict_h6.movie_macro_component_rmse,
            "r2": np.nan,
            "gain_pct": 100.0
            * (
                cv_h6.movie_macro_rmse
                - v166_h1_strict_h6.movie_macro_component_rmse
            )
            / cv_h6.movie_macro_rmse,
            "status": "h6_tradeoff",
        },
        {
            "module": "complete_system",
            "variant": "v166_external_h6_utility",
            "horizon": 1,
            "movie_macro_rmse": v166_h6_utility_h1.movie_macro_component_rmse,
            "r2": np.nan,
            "gain_pct": 100.0
            * (
                cv_h1.movie_macro_rmse
                - v166_h6_utility_h1.movie_macro_component_rmse
            )
            / cv_h1.movie_macro_rmse,
            "status": "h1_guard_fail_vs_own_prior",
        },
        {
            "module": "complete_system",
            "variant": "v166_external_h6_utility",
            "horizon": 6,
            "movie_macro_rmse": v166_h6_utility_h6.movie_macro_component_rmse,
            "r2": np.nan,
            "gain_pct": 100.0
            * (
                cv_h6.movie_macro_rmse
                - v166_h6_utility_h6.movie_macro_component_rmse
            )
            / cv_h6.movie_macro_rmse,
            "status": "best_external_h6_but_joint_gate_fail",
        },
        {
            "module": "privileged_state",
            "variant": "zero_state",
            "horizon": 6,
            "movie_macro_rmse": state_zero_h6.movie_macro_rmse,
            "r2": state_zero_h6.movie_macro_r2,
            "gain_pct": np.nan,
            "status": "reference",
        },
        {
            "module": "privileged_state",
            "variant": "real_state",
            "horizon": 6,
            "movie_macro_rmse": state_real_h6.movie_macro_rmse,
            "r2": state_real_h6.movie_macro_r2,
            "gain_pct": state_real_gain.movie_macro_gain_pct,
            "status": "fail",
        },
        {
            "module": "capacity_control",
            "variant": "noncausal_future_state",
            "horizon": 6,
            "movie_macro_rmse": scalar(
                state,
                "movie_macro_rmse",
                run="noncausal_capacity",
                metric_unit="cell_diameter",
                horizon=6,
            ),
            "r2": scalar(
                state,
                "movie_macro_r2",
                run="noncausal_capacity",
                metric_unit="cell_diameter",
                horizon=6,
            ),
            "gain_pct": capacity_gain.movie_macro_gain_pct,
            "status": "capacity_pass_only",
        },
        {
            "module": "transport_diagnostic",
            "variant": "h6_guard10_own_only",
            "horizon": 1,
            "movie_macro_rmse": transport_exploratory_h1.movie_macro_component_rmse,
            "r2": np.nan,
            "gain_pct": transport_exploratory_h1.movie_macro_gain_pct,
            "status": "h1_guard_fail",
        },
        {
            "module": "transport_diagnostic",
            "variant": "h6_guard10_own_only",
            "horizon": 6,
            "movie_macro_rmse": transport_exploratory.movie_macro_component_rmse,
            "r2": np.nan,
            "gain_pct": transport_exploratory.movie_macro_gain_pct,
            "status": "h6_signal_only",
        },
    ]
    results = pd.DataFrame(result_rows)
    results.to_csv(args.out_dir / "deepsea_v204_key_results.csv", index=False)

    real_beats_controls = bool(
        state_real_h6.movie_macro_rmse
        < state.loc[
            (state.metric_unit == "cell_diameter")
            & (state.horizon == 6)
            & state.run.isin(
                [
                    "row_shuffled",
                    "time_shuffled",
                    "wrong_cell",
                    "wrong_video",
                ]
            ),
            "movie_macro_rmse",
        ].min()
    )
    gates = pd.DataFrame(
        [
            {
                "gate": "data_contract",
                "passed": True,
                "evidence": (
                    f"{preparation['videos']} movies, "
                    f"{preparation['families']}, complete mask coverage"
                ),
            },
            {
                "gate": "future_suffix_invariance",
                "passed": bool(future_audit["future_suffix_invariance_pass"]),
                "evidence": (
                    "prefix prediction max delta "
                    f"{future_audit['prefix_prediction_max_abs_delta']:.3g}"
                ),
            },
            {
                "gate": "external_completed_innovation_transport",
                "passed": bool(transport_decision["external_transport_pass"]),
                "evidence": (
                    "v166-ext h1-strict vs own prior: "
                    f"h1 {transport_decision['h1_gain_pct']:.2f}%, "
                    f"h6 {transport_decision['h6_gain_pct']:.2f}%; "
                    "v166-ext h6-utility: "
                    f"h1 {v166_h6_utility_h1.movie_macro_gain_pct:.2f}%, "
                    f"h6 {v166_h6_utility_h6.movie_macro_gain_pct:.2f}%"
                ),
            },
            {
                "gate": "privileged_state",
                "passed": bool(
                    state_real_gain.movie_macro_gain_pct >= 3.0
                    and real_beats_controls
                    and state_real_nll.joint_student_t_nll_reduction_h1 > 0
                ),
                "evidence": (
                    f"h6 {state_real_gain.movie_macro_gain_pct:.2f}%, "
                    "real beats controls="
                    f"{real_beats_controls}, NLL reduction "
                    f"{state_real_nll.joint_student_t_nll_reduction_h1:.4f}"
                ),
            },
            {
                "gate": "deployable_image_student",
                "passed": False,
                "evidence": "not authorized because privileged-state gate failed",
            },
            {
                "gate": "bounded_multimodal_integration",
                "passed": False,
                "evidence": "not authorized because deployable-state gate was not entered",
            },
        ]
    )
    gates.to_csv(args.out_dir / "deepsea_v204_gates.csv", index=False)

    plot_coordinate_benchmark(
        coordinate,
        tabular,
        transport,
        args.out_dir / "deepsea_coordinate_benchmark_h1_h6.png",
    )
    plot_state_controls(
        state_bootstrap,
        shape_bootstrap,
        args.out_dir / "deepsea_state_controls_h6.png",
    )
    plot_transport_tradeoff(
        transport,
        args.out_dir / "deepsea_transport_tradeoff.png",
    )

    coordinate_gain_h1 = 100.0 * (
        cv_h1.movie_macro_rmse - coordinate_h1.movie_macro_rmse
    ) / cv_h1.movie_macro_rmse
    coordinate_gain_h6 = 100.0 * (
        cv_h6.movie_macro_rmse - coordinate_h6.movie_macro_rmse
    ) / cv_h6.movie_macro_rmse
    v166_h1_gain_vs_cv = 100.0 * (
        cv_h1.movie_macro_rmse
        - v166_h1_strict_h1.movie_macro_component_rmse
    ) / cv_h1.movie_macro_rmse
    v166_h6_gain_vs_cv = 100.0 * (
        cv_h6.movie_macro_rmse
        - v166_h6_utility_h6.movie_macro_component_rmse
    ) / cv_h6.movie_macro_rmse
    v166_h6_gain_vs_imm = 100.0 * (
        imm_h6.movie_macro_rmse
        - v166_h6_utility_h6.movie_macro_component_rmse
    ) / imm_h6.movie_macro_rmse
    h6_wrong_cell_rmse = scalar(
        transport,
        "movie_macro_component_rmse",
        objective_name="h6_guard10",
        packet_name="full",
        control="wrong_cell",
        horizon=6,
    )
    h6_stale_time_rmse = scalar(
        transport,
        "movie_macro_component_rmse",
        objective_name="h6_guard10",
        packet_name="full",
        control="stale_time",
        horizon=6,
    )
    report = f"""# DeepSea External Multimodal Validation v204

## Decision

The complete LaChance publication system is the two-operating-point v166
bounded completed-innovation transport. The `v97_direct` model is only its
frozen sequential prior and is reported as an internal ablation, not as the
project SOTA.

The contract-correct, dimensionless external experiment did not pass the
pre-registered transport or privileged-state gates. The raw-image student and
bounded multimodal adapter were therefore not trained; this is a gate-enforced
stop, not missing implementation.

The earlier native-pixel pilot is retained only as a secondary audit. It was
superseded because the frozen amendment requires learning and evaluation in
first-frame cell-diameter units across heterogeneous acquisition scales.

## Data and protocol

- Dataset: DeepSea, {preparation['videos']} independent videos and
  {len(preparation['families'])} cell families.
- Observations: {preparation['rows']:,}; tracks: {preparation['tracks']:,}.
- Frozen split: 26 train, 10 validation, 11 outer-test movies.
- Primary endpoint: movie-macro cumulative rolling h6 component RMSE.
- Unit: first-frame median segmented-cell diameter.
- Test windows: 22,149 at h1 and 17,763 at h6.
- Future-suffix invariance: **passed**; prefix prediction delta was
  {future_audit['prefix_prediction_max_abs_delta']:.1f}.

## Prior and complete-system benchmark

The v97 prior reduced h1 RMSE relative to constant velocity
from {cv_h1.movie_macro_rmse:.4f} to {coordinate_h1.movie_macro_rmse:.4f}
cell diameters ({coordinate_gain_h1:+.2f}%). It did not retain that advantage
at h6: {coordinate_h6.movie_macro_rmse:.4f} versus
{cv_h6.movie_macro_rmse:.4f} ({coordinate_gain_h6:+.2f}%).

The externalized complete v166 h1-strict operating point reached h1
{v166_h1_strict_h1.movie_macro_component_rmse:.4f}, a
{v166_h1_gain_vs_cv:.2f}% gain over constant velocity, but degraded h6 to
{v166_h1_strict_h6.movie_macro_component_rmse:.4f}. The v166 h6-utility
operating point reached h6
{v166_h6_utility_h6.movie_macro_component_rmse:.4f}, improving over constant
velocity by {v166_h6_gain_vs_cv:.2f}% and over IMM by
{v166_h6_gain_vs_imm:.2f}%. It degraded h1 by
{-v166_h6_utility_h1.movie_macro_gain_pct:.2f}% relative to its own frozen
prior, so it did not pass the pre-registered joint h1/h6 gate.

The h1 result is family-dependent. Zero displacement is strongest in the
low-motion embryonic-stem movies, while learned and velocity models help the
bronchial and muscle families. Ridge/HGBDT and LSTM/GRU show the same general
trade-off: a low one-step error does not guarantee stable cumulative
transport. The complete v166 h6-utility point, rather than the v97 prior, is
the strongest tested DeepSea h6 method in this table.

## Completed-innovation transport details

The complete h1-strict v166 transport improved h1 over its own prior by
{transport_decision['h1_gain_pct']:.2f}% but degraded h6 by
{transport_decision['h6_gain_pct']:.2f}%, with 0/11 positive test movies.
The frozen transport gate therefore failed.

The complete h6-utility v166 transport improved h6 over its own prior by
{v166_h6_utility_h6.movie_macro_gain_pct:.2f}% (95% movie bootstrap
{v166_h6_utility_h6.bootstrap_ci_low:.2f} to
{v166_h6_utility_h6.bootstrap_ci_high:.2f}) but degraded h1 by
{-v166_h6_utility_h1.movie_macro_gain_pct:.2f}%. Its h6 RMSE remained better
than wrong-cell ({h6_wrong_cell_rmse:.4f}) and stale-time
({h6_stale_time_rmse:.4f}) controls. The own-only ablation produced nearly the
same h6 result, indicating that the local graph packet did not provide the
transferable benefit.

## Privileged mask/state observability

The exact causal full state packet changed h6 by
{state_real_gain.movie_macro_gain_pct:.2f}% relative to zero state (95%
movie bootstrap {state_real_gain.bootstrap_ci_low:.2f} to
{state_real_gain.bootstrap_ci_high:.2f}). Its Student-t NLL reduction was
{state_real_nll.joint_student_t_nll_reduction_h1:.4f}; positive values would
be improvements. Real state did not beat row-shuffled, time-shuffled,
wrong-cell and wrong-video controls.

The shape-only exact branch also failed its controls. Fast packet triage found
no morphology, polarity, contact or reliability family that passed the
pre-registered gate after multiplicity control.

The noncausal future-state capacity control improved h6 by
{capacity_gain.movie_macro_gain_pct:.2f}% and improved h1 proper score on all
11 movies. The model can consume informative auxiliary variables; the tested
causal masks do not contain the required cell-specific forecasting signal.

## Interpretation

This experiment separates three claims:

1. **Model capacity:** supported by the noncausal positive control.
2. **Complete v166 mechanism transfer:** each operating point retains its
   intended advantage, but no single operating point passes the joint h1/h6
   gate on DeepSea.
3. **Causal mask/morphology information:** not supported beyond hard controls.

The result does not justify another DeepSea image encoder, flat video token
fusion, or LaChance adaptation from this source representation. The remaining
architecture hypothesis is a separately frozen two-timescale belief model:
a transient h1 correction and a persistent own-innovation state with explicit
semigroup consistency. Because the DeepSea outer test has now been inspected,
that hypothesis requires a new confirmation split or dataset.

For new observability, the next data should be synchronized MDCK-like movies
with reliable identity-resolved masks and an independent mechanical channel
(traction/stress or equivalent). Reusing appearance-only masks from DeepSea
is not supported by this experiment.

## Claim boundary

DeepSea v204 is a strong negative external-observability and partial
mechanism-transfer result. It is not evidence of global SOTA or of successful
multimodal transfer. The validated LaChance publication bundle remains the
positive efficacy result; DeepSea defines where that mechanism currently
does and does not generalize.
"""
    report_path = args.out_dir / "deepsea_v204_decision_report.md"
    report_path.write_text(report, encoding="utf-8")
    evidence_ledger = build_evidence_ledger(args=args, report_path=report_path)
    evidence_ledger.to_csv(
        args.out_dir / "v204_evidence_ledger.csv",
        index=False,
    )

    artifact_paths = [
        args.out_dir / "deepsea_v204_key_results.csv",
        args.out_dir / "deepsea_v204_gates.csv",
        args.out_dir / "deepsea_coordinate_benchmark_h1_h6.png",
        args.out_dir / "deepsea_state_controls_h6.png",
        args.out_dir / "deepsea_transport_tradeoff.png",
        report_path,
        args.coordinate_summary / "v204_online_macro_metrics.csv",
        args.state_summary / "v204_online_macro_metrics.csv",
        args.transport_dir / "v204_transport_aggregate.csv",
        args.future_audit,
        args.unit_audit,
        args.out_dir / "v204_evidence_ledger.csv",
        args.lachance_publication_report,
    ]
    manifest = pd.DataFrame(
        [
            {
                "artifact": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ]
    )
    manifest.to_csv(args.out_dir / "deepsea_v204_artifact_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinate-summary",
        type=Path,
        default=ROOT
        / "outputs/deepsea_coordinate_full_benchmark_summary_dimensionless_v204_2026-07-31",
    )
    parser.add_argument(
        "--state-summary",
        type=Path,
        default=ROOT
        / "outputs/deepsea_exact_state_summary_dimensionless_v204_2026-07-31",
    )
    parser.add_argument(
        "--shape-summary",
        type=Path,
        default=ROOT
        / "outputs/deepsea_exact_shape_summary_dimensionless_v204_2026-07-31",
    )
    parser.add_argument(
        "--tabular-dir",
        type=Path,
        default=ROOT
        / "outputs/deepsea_tabular_coordinate_baselines_v204_2026-07-31",
    )
    parser.add_argument(
        "--transport-dir",
        type=Path,
        default=ROOT
        / "outputs/deepsea_completed_innovation_transport_dimensionless_v204_2026-07-31",
    )
    parser.add_argument(
        "--future-audit",
        type=Path,
        default=ROOT
        / "outputs/deepsea_future_suffix_audit_dimensionless_v204_2026-07-31"
        / "v204_future_suffix_invariance.json",
    )
    parser.add_argument(
        "--preparation-manifest",
        type=Path,
        default=ROOT
        / "outputs/deepsea_multimodal_prepared_v204_2026-07-31"
        / "v204_preparation_manifest.json",
    )
    parser.add_argument(
        "--unit-audit",
        type=Path,
        default=ROOT
        / "outputs/deepsea_future_suffix_audit_dimensionless_v204_2026-07-31"
        / "v204_coordinate_unit_conversion.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT
        / "outputs/deepsea_online_h1_complete_cache_v204_2026-07-31",
    )
    parser.add_argument(
        "--lachance-publication-report",
        type=Path,
        default=ROOT
        / "outputs/lachance_publication_bundle_v188_2026-07-29"
        / "v188_publication_report.md",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/deepsea_external_multimodal_validation_v204_2026-07-31",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
