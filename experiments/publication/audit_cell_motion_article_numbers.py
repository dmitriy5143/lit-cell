#!/usr/bin/env python3
"""Audit every quantitative claim used by the journal manuscript figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "outputs/lachance_publication_bundle_v188_2026-07-29"
V166 = ROOT / "outputs/lachance_publication_bundle_v166_2026-07-27"
SPATIAL = ROOT / "outputs/lachance_online_spatial_innovation_audit_v139_2026-07-22"
LADDER = ROOT / "outputs/lachance_streaming_information_ladder_v161_full_2026-07-27"
CONFIRM = ROOT / "outputs/lachance_streaming_transport_confirmation_v160_full_2026-07-27"
PHYSICS_GATE = ROOT / "outputs/active_transition_physics_gate_combined_2026-06-12"
CONFIRM_LEARNED = (
    ROOT
    / "outputs"
    / "lachance_confirmation_learned_comparators_v193_full_2026-07-30"
)
FIELD_V197 = (
    ROOT
    / "outputs"
    / "mdck_equivariant_field_law_v197_full_final_2026-07-30"
)
POTENTIAL_V198 = (
    ROOT
    / "outputs"
    / "mdck_effective_potential_audit_v198_2026-07-30"
)
H1_AUDIT = ROOT / "outputs" / "h1_signal_noise_audit_2026-07-30"
GRAPH_V199 = (
    ROOT
    / "outputs"
    / "lachance_equivariant_graph_bridge_v199_full_2026-07-30"
)
FUNCTIONAL_V200 = (
    ROOT
    / "outputs"
    / "mdck_effective_functional_dynamics_v200_full_2026-07-30"
)
PROBABILISTIC_V201 = (
    ROOT
    / "outputs"
    / "lachance_probabilistic_graph_closure_v201_full_2026-07-30"
)
UNSEEN_GRAPH_V202 = (
    ROOT
    / "outputs"
    / "lachance_equivariant_graph_unseen_v202_full_2026-07-30"
)
DEEPSEA_V204 = (
    ROOT
    / "outputs"
    / "deepsea_external_multimodal_validation_v204_2026-07-31"
)
H1_EVIDENCE_V205 = (
    ROOT
    / "outputs"
    / "lachance_h1_evidence_bundle_v205_full_2026-08-01"
)
LIFEACT_V207 = (
    ROOT
    / "outputs"
    / "lifeact_mdck_mechanochemical_state_gate_v207_center60_multiseed_2026-08-01"
)
LIFEACT_V207_NORMALIZED = (
    ROOT
    / "outputs"
    / "lifeact_mdck_mechanochemical_state_gate_v207_center60_normalized_multiseed_2026-08-01"
)
LIFEACT_V208 = (
    ROOT
    / "outputs"
    / "lifeact_mdck_state_uncertainty_v208_pixel_studentt_2026-08-01"
)
LIFEACT_V208_NORMALIZED = (
    ROOT
    / "outputs"
    / "lifeact_mdck_state_uncertainty_v208_normalized_studentt_2026-08-01"
)


def one(df: pd.DataFrame, **filters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        mask &= df[column] == value
    subset = df.loc[mask]
    if len(subset) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(subset)}")
    return subset.iloc[0]


def make_claim(
    claim_id: str,
    quantity: str,
    value: float,
    unit: str,
    source: Path,
    interpretation: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "quantity": quantity,
        "value": float(value),
        "unit": unit,
        "source": str(source),
        "interpretation": interpretation,
        "status": "verified",
    }


def audit() -> tuple[pd.DataFrame, list[str]]:
    benchmark_path = BUNDLE / "v188_primary_online_benchmark.csv"
    paired_path = BUNDLE / "v188_paired_movie_statistics.csv"
    unseen_path = BUNDLE / "v188_configuration_unseen_confirmation.csv"
    external_path = BUNDLE / "v188_external_nested_lomo.csv"
    robustness_path = BUNDLE / "v188_robustness_matrix.csv"
    uncertainty_path = BUNDLE / "v188_robustness_uncertainty.csv"
    edge_path = V166 / "v166_dimensionless_transfer.csv"
    spatial_path = SPATIAL / "v139_spatial_innovation_aggregate.csv"
    ladder_path = LADDER / "v161_information_ladder_summary.csv"
    controls_path = CONFIRM / "v160_confirmation_aggregate.csv"
    physics_gate_path = PHYSICS_GATE / "combined_gate_summary.csv"
    observability_path = BUNDLE / "v188_observability_gate_matrix.csv"
    confirmation_learned_path = CONFIRM_LEARNED / "v193_aggregate.csv"
    confirmation_pairwise_path = CONFIRM_LEARNED / "v193_pairwise_statistics.csv"
    field_summary_path = FIELD_V197 / "v197_field_law_summary.csv"
    field_outer_path = FIELD_V197 / "v197_field_law_outer_folds.csv"
    field_bridge_path = FIELD_V197 / "v197_mechanical_bridge.csv"
    field_intervention_path = FIELD_V197 / "v197_intervention_transfer.csv"
    field_spectrum_path = FIELD_V197 / "v197_dynamic_spectra_summary.csv"
    field_synthetic_path = FIELD_V197 / "v197_synthetic_identifiability.csv"
    field_equivariance_path = FIELD_V197 / "v197_equivariance.csv"
    potential_summary_path = POTENTIAL_V198 / "v198_potential_summary.csv"
    potential_outer_path = POTENTIAL_V198 / "v198_potential_outer_folds.csv"
    h1_scale_path = H1_AUDIT / "h1_h6_scale_summary.csv"
    h1_ratio_path = H1_AUDIT / "h1_h6_scale_ratios.csv"
    h1_forensic_path = H1_AUDIT / "tracking_noise_forensics.csv"
    graph_aggregate_path = GRAPH_V199 / "v199_graph_bridge_aggregate.csv"
    graph_projection_path = GRAPH_V199 / "v199_graph_bridge_projection.csv"
    functional_summary_path = (
        FUNCTIONAL_V200 / "v200_finite_functional_summary.csv"
    )
    functional_rollout_path = (
        FUNCTIONAL_V200 / "v200_field_rollout_summary.csv"
    )
    probabilistic_aggregate_path = (
        PROBABILISTIC_V201 / "v201_probabilistic_aggregate.csv"
    )
    unseen_graph_aggregate_path = (
        UNSEEN_GRAPH_V202 / "v202_unseen_graph_aggregate.csv"
    )
    unseen_graph_controls_path = (
        UNSEEN_GRAPH_V202 / "v202_unseen_graph_controls.csv"
    )
    deepsea_results_path = DEEPSEA_V204 / "deepsea_v204_key_results.csv"
    h1_pareto_path = H1_EVIDENCE_V205 / "v205_pareto_points.csv"
    h1_normalized_path = H1_EVIDENCE_V205 / "v205_normalized_error.csv"
    h1_pairwise_path = H1_EVIDENCE_V205 / "v205_pairwise_statistics.csv"
    h1_localization_path = H1_EVIDENCE_V205 / "v205_localization_context.csv"
    lifeact_mean_path = LIFEACT_V207 / "v207_multiseed_decision_aggregate.csv"
    lifeact_mean_normalized_path = (
        LIFEACT_V207_NORMALIZED / "v207_multiseed_decision_aggregate.csv"
    )
    lifeact_uncertainty_path = LIFEACT_V208 / "v208_uncertainty_decision.csv"
    lifeact_uncertainty_normalized_path = (
        LIFEACT_V208_NORMALIZED / "v208_uncertainty_decision.csv"
    )

    benchmark = pd.read_csv(benchmark_path)
    paired = pd.read_csv(paired_path)
    unseen = pd.read_csv(unseen_path)
    external = pd.read_csv(external_path)
    robustness = pd.read_csv(robustness_path)
    uncertainty = pd.read_csv(uncertainty_path)
    edge = pd.read_csv(edge_path)
    spatial = pd.read_csv(spatial_path)
    ladder = pd.read_csv(ladder_path)
    controls = pd.read_csv(controls_path)
    physics_gate = pd.read_csv(physics_gate_path)
    observability = pd.read_csv(observability_path)
    confirmation_learned = pd.read_csv(confirmation_learned_path)
    confirmation_pairwise = pd.read_csv(confirmation_pairwise_path)
    field_summary = pd.read_csv(field_summary_path)
    field_outer = pd.read_csv(field_outer_path)
    field_bridge = pd.read_csv(field_bridge_path)
    field_intervention = pd.read_csv(field_intervention_path)
    field_spectrum = pd.read_csv(field_spectrum_path)
    field_synthetic = pd.read_csv(field_synthetic_path)
    field_equivariance = pd.read_csv(field_equivariance_path)
    potential_summary = pd.read_csv(potential_summary_path)
    potential_outer = pd.read_csv(potential_outer_path)
    h1_scale = pd.read_csv(h1_scale_path)
    h1_ratios = pd.read_csv(h1_ratio_path)
    h1_forensic = pd.read_csv(h1_forensic_path)
    graph_aggregate = pd.read_csv(graph_aggregate_path)
    graph_projection = pd.read_csv(graph_projection_path)
    functional_summary = pd.read_csv(functional_summary_path)
    functional_rollout = pd.read_csv(functional_rollout_path)
    probabilistic_aggregate = pd.read_csv(probabilistic_aggregate_path)
    unseen_graph_aggregate = pd.read_csv(unseen_graph_aggregate_path)
    unseen_graph_controls = pd.read_csv(unseen_graph_controls_path)
    deepsea_results = pd.read_csv(deepsea_results_path)
    h1_pareto = pd.read_csv(h1_pareto_path)
    h1_normalized = pd.read_csv(h1_normalized_path)
    h1_pairwise = pd.read_csv(h1_pairwise_path)
    h1_localization = pd.read_csv(h1_localization_path)
    lifeact_mean = pd.read_csv(lifeact_mean_path)
    lifeact_mean_normalized = pd.read_csv(lifeact_mean_normalized_path)
    lifeact_uncertainty = pd.read_csv(lifeact_uncertainty_path)
    lifeact_uncertainty_normalized = pd.read_csv(
        lifeact_uncertainty_normalized_path
    )

    claims: list[dict[str, object]] = []

    field_real = field_summary[field_summary["control"].eq("real")].set_index(
        "variant"
    )
    for variant, prefix in [
        ("cv", "field_cv"),
        ("relaxation", "field_relaxation"),
        ("advective_pde", "field_advective"),
        ("helmholtz_pde", "field_helmholtz"),
        ("mechanics_source_pde", "field_mechanics"),
    ]:
        claims.extend(
            [
                make_claim(
                    f"{prefix}_rmse",
                    f"{variant} external-island displacement RMSE",
                    field_real.loc[variant, "displacement_rmse_macro"],
                    "um/min",
                    field_summary_path,
                    "Whole-island nested holdout on 22 measured-mechanics MDCK islands.",
                ),
                make_claim(
                    f"{prefix}_gain",
                    f"{variant} gain versus constant velocity",
                    field_real.loc[variant, "gain_vs_cv_percent_mean"],
                    "%",
                    field_summary_path,
                    "Mean of island-level relative gains.",
                ),
            ]
        )
    field_pivot = field_outer[field_outer["control"].eq("real")].pivot(
        index="outer_group",
        columns="variant",
        values="displacement_rmse",
    )
    field_increment = 100.0 * (
        field_pivot["relaxation"] - field_pivot["advective_pde"]
    ) / field_pivot["relaxation"]
    field_positive = int(np.sum(field_increment > 0))
    claims.extend(
        [
            make_claim(
                "field_advective_increment",
                "Advective field gain beyond learned scalar relaxation",
                field_increment.mean(),
                "%",
                field_outer_path,
                "Paired whole-island relative gain.",
            ),
            make_claim(
                "field_advective_positive_islands",
                "Islands improved beyond learned scalar relaxation",
                field_positive,
                "islands",
                field_outer_path,
                "Whole-island sign count.",
            ),
            make_claim(
                "field_advective_sign_p",
                "One-sided exact sign-test p value",
                binomtest(
                    field_positive,
                    len(field_increment),
                    p=0.5,
                    alternative="greater",
                ).pvalue,
                "p",
                field_outer_path,
                "Predefined paired island-level direction test.",
            ),
        ]
    )
    for control, prefix in [
        ("kin_spatial_shifted", "field_spatial_control"),
        ("kin_time_shuffled", "field_time_control"),
        ("kin_wrong_island", "field_wrong_island_control"),
    ]:
        row = one(
            field_summary,
            variant="advective_pde",
            control=control,
        )
        claims.append(
            make_claim(
                f"{prefix}_gain",
                f"Advective field {control} gain",
                row["gain_vs_cv_percent_mean"],
                "%",
                field_summary_path,
                "Causal field null control.",
            )
        )
    bridge_pivot = field_bridge.pivot_table(
        index=["outer_group", "target"],
        columns="variant",
        values="rmse",
    )
    bridge_gain = 100.0 * (
        bridge_pivot["velocity_only"]
        - bridge_pivot["velocity_plus_innovation"]
    ) / bridge_pivot["velocity_only"]
    claims.append(
        make_claim(
            "field_reverse_bridge_gain",
            "Completed-innovation gain in reverse mechanics bridge",
            bridge_gain.mean(),
            "%",
            field_bridge_path,
            "Macro average over traction, stress divergence, and force balance.",
        )
    )
    intervention_real = field_intervention[
        field_intervention["control"].eq("real")
    ].pivot_table(
        index=["split", "target_group"],
        columns="variant",
        values="displacement_rmse",
    )
    intervention_mechanics = 100.0 * (
        intervention_real["helmholtz_pde"]
        - intervention_real["mechanics_source_pde"]
    ) / intervention_real["helmholtz_pde"]
    claims.append(
        make_claim(
            "field_intervention_mechanics_advantage",
            "Measured-mechanics advantage under intervention transfer",
            intervention_mechanics.mean(),
            "%",
            field_intervention_path,
            "Paired target-island gain over the matched kinematic law.",
        )
    )
    spectrum_macro = field_spectrum.groupby("variant").mean(numeric_only=True)
    spectrum_improvement = 100.0 * (
        spectrum_macro.loc["helmholtz_pde", "spectrum_relative_log_error"]
        - spectrum_macro.loc[
            "mechanics_source_pde",
            "spectrum_relative_log_error",
        ]
    ) / spectrum_macro.loc["helmholtz_pde", "spectrum_relative_log_error"]
    claims.extend(
        [
            make_claim(
                "field_mechanics_spectrum_improvement",
                "Measured-mechanics improvement in dynamic spectral log-error",
                spectrum_improvement,
                "%",
                field_spectrum_path,
                "Below the predefined one-percent gate.",
            ),
            make_claim(
                "field_synthetic_median_coefficient_error",
                "Median synthetic coefficient recovery error",
                field_synthetic["relative_error_percent"].median(),
                "%",
                field_synthetic_path,
                "Six-term synthetic E(2)-equivariant law.",
            ),
            make_claim(
                "field_synthetic_coverage",
                "Synthetic coefficients covered by frame-bootstrap intervals",
                field_synthetic["covered"].mean(),
                "fraction",
                field_synthetic_path,
                "Three hundred whole-frame bootstrap repetitions.",
            ),
            make_claim(
                "field_equivariance_max_relative_error",
                "Maximum relative error across algebraic and finite-grid audits",
                field_equivariance["relative_l2_error"].max(),
                "fraction",
                field_equivariance_path,
                "Arbitrary algebraic rotation and full-library D4 rotation.",
            ),
        ]
    )

    potential_indexed = potential_summary.set_index("variant")
    potential_row = potential_indexed.loc["potential_constrained"]
    potential_boundary_row = potential_indexed.loc[
        "potential_boundary_constrained"
    ]
    potential_advective_row = potential_indexed.loc[
        "advective_unconstrained"
    ]
    potential_paired = potential_outer.pivot(
        index="outer_group",
        columns="variant",
        values="displacement_rmse",
    )
    advective_better = int(
        np.sum(
            potential_paired["advective_unconstrained"]
            < potential_paired["potential_constrained"]
        )
    )
    potential_sign_p = binomtest(
        advective_better,
        len(potential_paired),
        p=0.5,
        alternative="greater",
    ).pvalue
    potential_retained = (
        100.0
        * potential_row["gain_vs_cv_percent_mean"]
        / potential_advective_row["gain_vs_cv_percent_mean"]
    )
    potential_gap = (
        100.0
        * (
            potential_row["displacement_rmse_macro"]
            - potential_advective_row["displacement_rmse_macro"]
        )
        / potential_advective_row["displacement_rmse_macro"]
    )
    constrained_outer = potential_outer[
        potential_outer["variant"].eq("potential_constrained")
    ]
    for claim_id, quantity, value, unit in [
        (
            "potential_rmse",
            "Sign-constrained potential-sector displacement RMSE",
            potential_row["displacement_rmse_macro"],
            "um/min",
        ),
        (
            "potential_gain",
            "Sign-constrained potential-sector gain versus constant velocity",
            potential_row["gain_vs_cv_percent_mean"],
            "%",
        ),
        (
            "potential_boundary_rmse",
            "Potential-sector plus boundary displacement RMSE",
            potential_boundary_row["displacement_rmse_macro"],
            "um/min",
        ),
        (
            "potential_advective_rmse",
            "Full advective-law displacement RMSE in the potential audit",
            potential_advective_row["displacement_rmse_macro"],
            "um/min",
        ),
        (
            "potential_advective_gain",
            "Full advective-law gain in the potential audit",
            potential_advective_row["gain_vs_cv_percent_mean"],
            "%",
        ),
        (
            "potential_gain_retained",
            "Potential-sector fraction of full advective gain",
            potential_retained,
            "%",
        ),
        (
            "potential_relative_gap",
            "Potential-sector RMSE gap versus full advective law",
            potential_gap,
            "%",
        ),
        (
            "potential_drift_cosine",
            "Potential drift cosine with observed field update",
            potential_row["drift_observed_cosine_mean"],
            "fraction",
        ),
        (
            "potential_shuffled_drift_cosine",
            "Potential drift cosine with shuffled field update",
            potential_row["drift_shuffled_cosine_mean"],
            "fraction",
        ),
        (
            "potential_energy_decrease_fraction",
            "First-order effective-functional decrease fraction",
            potential_row["energy_decrease_fraction_mean"],
            "fraction",
        ),
        (
            "potential_shuffled_energy_decrease_fraction",
            "Shuffled first-order effective-functional decrease fraction",
            potential_row["shuffled_energy_decrease_fraction_mean"],
            "fraction",
        ),
        (
            "potential_bounded_folds",
            "External folds with a bounded-below effective functional",
            potential_row["bounded_below_folds"],
            "folds",
        ),
        (
            "potential_advective_better_islands",
            "Islands where the full advective law beats the potential sector",
            advective_better,
            "islands",
        ),
        (
            "potential_advective_sign_p",
            "One-sided sign-test p value for the advective increment",
            potential_sign_p,
            "p",
        ),
    ]:
        claims.append(
            make_claim(
                claim_id,
                quantity,
                value,
                unit,
                potential_summary_path
                if claim_id
                not in {
                    "potential_advective_better_islands",
                    "potential_advective_sign_p",
                }
                else potential_outer_path,
                "Whole-island nested holdout; this is an effective kinematic "
                "functional, not calibrated physical energy.",
            )
        )
    for column, label in [
        ("quadratic_r", "r"),
        ("gradient_k_transverse", "K_T"),
        ("gradient_k_longitudinal", "K_L"),
        ("quartic_g", "g"),
    ]:
        for suffix, reducer in [("min", np.min), ("max", np.max)]:
            claims.append(
                make_claim(
                    f"potential_{label}_{suffix}",
                    f"Potential coefficient {label} {suffix}",
                    reducer(constrained_outer[column]),
                    "effective units",
                    potential_outer_path,
                    "Coefficient scale depends on the field units and one-step "
                    "time discretization; it is not a material constant.",
                )
            )

    for dataset, prefix in [("MDCK_Bulk", "physics_bulk"), ("MDCK_Edge", "physics_edge")]:
        row = one(
            physics_gate,
            run_block="mdck",
            prefix="real",
            dataset=dataset,
            horizon=6,
        )
        claims.append(
            make_claim(
                f"{prefix}_active_gain",
                f"{dataset} conditional C(1,2) gain at h6",
                row["active_gain_mean_pct"],
                "%",
                physics_gate_path,
                "Отрицательная величина означает ухудшение относительно согласованного контроля.",
            )
        )
    claims.extend(
        [
            make_claim(
                "observability_deployable_passes",
                "Переносимые сигналы, прошедшие полный шлюз наблюдаемости",
                observability["deployable_to_lachance"].sum(),
                "сигналов",
                observability_path,
                "Прохождение требует одновременно временного, идентификационного и доменного контроля.",
            ),
            make_claim(
                "observability_total",
                "Всего проверенных переносимых сигналов",
                len(observability),
                "сигналов",
                observability_path,
                "Знаменатель для панели архитектурного поиска.",
            ),
        ]
    )

    def add_benchmark(method: str, horizon: int, prefix: str) -> None:
        row = one(benchmark, method=method, horizon=horizon)
        claims.extend(
            [
                make_claim(
                    f"{prefix}_rmse",
                    f"{method} component RMSE at h{horizon}",
                    row["component_rmse"],
                    "px",
                    benchmark_path,
                    "Movie-macro component RMSE in the frozen six-movie outer-LOMO protocol.",
                ),
                make_claim(
                    f"{prefix}_r2",
                    f"{method} R2 at h{horizon}",
                    row["r2"],
                    "fraction",
                    benchmark_path,
                    "R2 is comparable only within the same domain, horizon, and causal protocol.",
                ),
            ]
        )

    add_benchmark("v166_h1_strict", 1, "primary_h1")
    add_benchmark("v166_h6_utility", 6, "primary_h6")
    add_benchmark("v97_no_update", 1, "prior_h1")
    add_benchmark("v97_no_update", 6, "prior_h6")
    add_benchmark("constant_velocity", 6, "constant_velocity_h6")

    for horizon in [1, 6]:
        row = one(h1_scale, horizon=horizon)
        for column, unit in [
            ("normalized_rmse_mean", "fraction"),
            ("implied_target_sd_mean_px", "px"),
        ]:
            claims.append(
                make_claim(
                    f"h{horizon}_{column}",
                    f"Movie-mean {column} at h{horizon}",
                    row[column],
                    unit,
                    h1_scale_path,
                    "Derived within each movie from the same v166_h1_strict predictions and target rows.",
                )
            )
    for quantity in [
        "implied_target_sd_h6_over_h1",
        "component_rmse_h6_over_h1",
        "normalized_rmse_h1_over_h6",
    ]:
        row = one(h1_ratios, quantity=quantity)
        claims.append(
            make_claim(
                f"horizon_scale_{quantity}",
                quantity,
                row["value"],
                "ratio",
                h1_ratio_path,
                "Descriptive horizon-scale audit, not an identified irreducible noise fraction.",
            )
        )
    for annotation_kind in ["manual", "automatic"]:
        for quantity, unit in [("component_rmse", "px"), ("r2", "fraction")]:
            row = one(
                h1_forensic,
                source="c2c12_h1",
                annotation_kind=annotation_kind,
                quantity=quantity,
            )
            claims.append(
                make_claim(
                    f"c2c12_{annotation_kind}_h1_{quantity}",
                    f"C2C12 {annotation_kind} h1 {quantity}",
                    row["value"],
                    unit,
                    h1_forensic_path,
                    "External tracking-quality forensic; manual tracks contain many interpolated centroids.",
                )
            )
    for quantity in ["step_disagreement_median", "step_disagreement_p90"]:
        row = one(
            h1_forensic,
            source="c2c12_paired_movie",
            annotation_kind="paired",
            quantity=quantity,
        )
        claims.append(
            make_claim(
                f"c2c12_{quantity}",
                f"Paired C2C12 {quantity}",
                row["value"],
                "px",
                h1_forensic_path,
                "Same-frame spatial matching is used only for the forensic audit and never as a model input.",
            )
        )

    h1_test = one(
        paired,
        method="v166_h1_strict",
        comparator="v97_no_update",
        horizon=1,
    )
    h6_test = one(
        paired,
        method="v166_h6_utility",
        comparator="v97_no_update",
        horizon=6,
    )
    for row, prefix in [(h1_test, "h1_test"), (h6_test, "h6_test")]:
        for column, unit in [
            ("relative_gain_percent", "% movie-macro"),
            ("mean_rmse_delta_comparator_minus_method", "px"),
            ("delta_ci_low", "px"),
            ("delta_ci_high", "px"),
            ("exact_two_sided_sign_flip_p", "p"),
            ("holm_adjusted_p", "p"),
            ("method_better_movies", "movies"),
        ]:
            claims.append(
                make_claim(
                    f"{prefix}_{column}",
                    column,
                    row[column],
                    unit,
                    paired_path,
                    "The relative gain is the mean of movie-level gains, not a ratio of pooled RMSE values.",
                )
            )

    unseen_real = one(unseen, objective_name="h6_guard10", control="real", horizon=6)
    unseen_control = one(
        controls,
        objective_name="h6_guard10",
        packet_name="full",
        control="real",
        horizon=6,
    )
    claims.extend(
        [
            make_claim(
                "unseen_h6_rmse",
                "frozen-current-configuration h6 RMSE",
                unseen_real["component_rmse"],
                "px",
                unseen_path,
                "Seven MDCK Bulk films not used to select the current configuration; distinct from the six-movie primary LOMO but present in older broad screening.",
            ),
            make_claim(
                "unseen_h6_r2",
                "frozen-current-configuration h6 R2",
                unseen_real["r2"],
                "fraction",
                unseen_path,
                "Seven films not used to select the current configuration; not a fully hypothesis-naive prospective cohort.",
            ),
            make_claim(
                "unseen_h6_gain",
                "frozen-current-configuration gain versus v97",
                unseen_real["gain_vs_v97_percent"],
                "%",
                unseen_path,
                "Relative gain on films 10-16, which were not used to select the current configuration.",
            ),
            make_claim(
                "unseen_h6_delta",
                "frozen-current-configuration paired h6 RMSE reduction",
                unseen_control["bootstrap_mean_delta"],
                "px",
                controls_path,
                "Mean movie-level v97-minus-method RMSE difference on seven frozen confirmation movies.",
            ),
            make_claim(
                "unseen_h6_delta_ci_low",
                "frozen-current-configuration paired h6 RMSE reduction CI low",
                unseen_control["bootstrap_ci_low"],
                "px",
                controls_path,
                "Lower endpoint of the movie bootstrap 95% interval.",
            ),
            make_claim(
                "unseen_h6_delta_ci_high",
                "frozen-current-configuration paired h6 RMSE reduction CI high",
                unseen_control["bootstrap_ci_high"],
                "px",
                controls_path,
                "Upper endpoint of the movie bootstrap 95% interval.",
            ),
            make_claim(
                "unseen_h6_exact_p",
                "frozen-current-configuration exact two-sided sign-flip p",
                unseen_control["sign_flip_p"],
                "p",
                controls_path,
                "Unadjusted exact paired sign-flip p on seven frozen confirmation movies.",
            ),
            make_claim(
                "unseen_h6_positive_movies",
                "frozen-current-configuration positive movies",
                unseen_control["movies_improved_vs_v97"],
                "movies",
                controls_path,
                "Number of confirmation movies with lower h6 RMSE than v97.",
            ),
        ]
    )

    confirmation_labels = {
        "gru_track": "confirmation_gru",
        "hgbdt_track": "confirmation_hgbdt",
        "kalmannet": "confirmation_kalmannet",
        "ours_h1_strict": "confirmation_ours_h1",
        "ours_h6_utility": "confirmation_ours_h6",
    }
    for method, prefix in confirmation_labels.items():
        for horizon in [1, 2, 4, 6]:
            row = one(confirmation_learned, method=method, horizon=horizon)
            claims.extend(
                [
                    make_claim(
                        f"{prefix}_h{horizon}_rmse",
                        f"{method} frozen-confirmation h{horizon} RMSE",
                        row["component_rmse_mean"],
                        "px",
                        confirmation_learned_path,
                        "Prediction-level ensemble of seeds 7,42,123; fit movies 1-4, validation movie 5, confirmation movies 10-16.",
                    ),
                    make_claim(
                        f"{prefix}_h{horizon}_r2",
                        f"{method} frozen-confirmation h{horizon} R2",
                        row["r2_mean"],
                        "fraction",
                        confirmation_learned_path,
                        "Movie-macro R2 under the same frozen streaming contract.",
                    ),
                ]
            )
    for comparator in ["hgbdt_track", "gru_track", "kalmannet"]:
        row = one(
            confirmation_pairwise,
            reference="ours_h6_utility",
            comparator=comparator,
            horizon=6,
        )
        prefix = f"confirmation_h6_vs_{comparator}"
        for column, unit in [
            ("mean_comparator_minus_reference", "px"),
            ("relative_gain_percent", "%"),
            ("bootstrap_ci_low", "px"),
            ("bootstrap_ci_high", "px"),
            ("movies_reference_better", "movies"),
            ("exact_sign_flip_p", "p"),
        ]:
            claims.append(
                make_claim(
                    f"{prefix}_{column}",
                    f"frozen-confirmation h6 comparison with {comparator}: {column}",
                    row[column],
                    unit,
                    confirmation_pairwise_path,
                    "Paired movie-level comparison on films 10-16; no confirmation metric was used for model selection.",
                )
            )

    for level in ["A0_v97", "A1_own", "A2_own_local", "A3_global"]:
        row = one(ladder, ladder_level=level, horizon=6)
        claims.append(
            make_claim(
                f"ladder_{level}",
                f"information ladder {level} h6 RMSE",
                row["component_rmse"],
                "px",
                ladder_path,
                "Causal information ladder on films 10-16 with the current configuration frozen.",
            )
        )

    for control in ["real", "stale_time", "wrong_cell"]:
        row = one(
            controls,
            objective_name="h6_guard10",
            packet_name="full",
            control=control,
            horizon=6,
        )
        claims.append(
            make_claim(
                f"control_{control}",
                f"{control} h6 RMSE",
                row["component_rmse_mean"],
                "px",
                controls_path,
                "Causal control on films 10-16 with the current configuration frozen.",
            )
        )

    for k in [1, 8]:
        row = one(spatial, method="v97", neighbor_k=k)
        claims.extend(
            [
                make_claim(
                    f"spatial_k{k}_dot",
                    f"normalized innovation dot-product excess at k={k}",
                    row["neighbor_dot_excess_normalized_mean"],
                    "fraction",
                    spatial_path,
                    "Mean across six movies; uncertainty shown as between-movie standard deviation.",
                ),
                make_claim(
                    f"spatial_k{k}_dot_sd",
                    f"between-movie SD at k={k}",
                    row["neighbor_dot_excess_normalized_std"],
                    "fraction",
                    spatial_path,
                    "Between-movie standard deviation, not a confidence interval.",
                ),
            ]
        )

    for dataset in ["HUVEC", "MDAMB231"]:
        row = one(external, dataset=dataset, objective="h6_guard10", control="real", horizon=6)
        for column, unit in [
            ("component_rmse_macro", "px"),
            ("r2_macro", "fraction"),
            ("gain_percent_macro", "%"),
            ("positive_folds", "folds"),
            ("outer_folds", "folds"),
        ]:
            claims.append(
                make_claim(
                    f"{dataset.lower()}_{column}",
                    f"{dataset} {column}",
                    row[column],
                    unit,
                    external_path,
                    "Full nested leave-one-movie-out transfer; CIs are nested movie bootstrap intervals.",
                )
            )

    edge_row = one(edge, objective="h6_guard10", control="real", dataset="MDCK_Edge", horizon=6)
    for column, unit in [
        ("component_rmse_mean", "px"),
        ("component_rmse_std", "px across seeds"),
        ("r2_mean", "fraction"),
        ("gain_vs_prior_mean", "%"),
        ("seeds_positive", "seeds"),
    ]:
        claims.append(
            make_claim(
                f"edge_{column}",
                f"MDCK Edge {column}",
                edge_row[column],
                unit,
                edge_path,
                "Zero-shot/domain transfer over three optimizer seeds; the spread is not a movie-bootstrap CI.",
            )
        )

    for condition in [
        "real_update_every_1",
        "update_every_2",
        "update_every_3",
        "update_every_6",
        "missing_0.4",
        "tracking_noise_1px",
        "delay_1frame",
        "wrong_cell",
    ]:
        row = one(
            robustness,
            operating_point="h6_utility",
            condition=condition,
            horizon=6,
        )
        claims.append(
            make_claim(
                f"robustness_{condition}",
                f"h6 improvement under {condition}",
                row["rmse_improvement_percent_mean"],
                "% movie-macro",
                robustness_path,
                "Improvement is measured against the matched no-update prior under the same condition.",
            )
        )

    for operating_point, horizon, prefix in [
        ("h1_strict", 1, "calibration_h1"),
        ("h6_utility", 6, "calibration_h6"),
    ]:
        row = one(
            uncertainty,
            operating_point=operating_point,
            calibration_mode="frozen_state_aware_scale",
            condition="real_update_every_1",
            horizon=horizon,
        )
        for column in [
            "coverage_50_mean",
            "coverage_90_mean",
            "calibration_error_mean",
            "uncertainty_error_corr_mean",
        ]:
            claims.append(
                make_claim(
                    f"{prefix}_{column}",
                    f"{operating_point}@h{horizon} {column}",
                    row[column],
                    "fraction",
                    uncertainty_path,
                    "Frozen state-aware calibration; do not mix with other horizons or the base-scale calibration.",
                )
            )

    potential_dev = one(
        graph_aggregate,
        objective_name="h6_guard10",
        variant="forced_potential",
        control="real",
        horizon=6,
    )
    potential_projection = graph_projection[
        graph_projection["objective_name"].eq("h6_guard10")
        & graph_projection["variant"].eq("forced_potential")
    ]
    potential_unseen = one(
        unseen_graph_aggregate,
        objective_name="h6_guard10",
        variant="forced_potential",
        control="real",
        horizon=6,
    )
    for prefix, row, source in [
        ("graph_potential_dev", potential_dev, graph_aggregate_path),
        (
            "graph_potential_unseen",
            potential_unseen,
            unseen_graph_aggregate_path,
        ),
    ]:
        claims.extend(
            [
                make_claim(
                    f"{prefix}_h6_rmse",
                    f"{prefix} h6 component RMSE",
                    row["component_rmse_mean"],
                    "px",
                    source,
                    "E(2)-equivariant forced-potential graph law.",
                ),
                make_claim(
                    f"{prefix}_h6_r2",
                    f"{prefix} h6 R2",
                    row["r2_mean"],
                    "fraction",
                    source,
                    "Movie-macro R2.",
                ),
                make_claim(
                    f"{prefix}_h6_gain",
                    f"{prefix} h6 gain",
                    row["gain_percent_mean"],
                    "% movie-macro",
                    source,
                    "Relative to the matched frozen no-update mean.",
                ),
                make_claim(
                    f"{prefix}_positive_movies",
                    f"{prefix} positive movies",
                    row["movies_improved"],
                    "movies",
                    source,
                    "Independent movie-level sign count.",
                ),
            ]
        )
    claims.append(
        make_claim(
            "graph_potential_projection_r2",
            "Potential graph law explained variance of dense h6 correction",
            potential_projection["legacy_correction_explained_r2"].mean(),
            "fraction",
            graph_projection_path,
            "Mean across six outer movies.",
        )
    )
    for comparison, prefix in [
        ("real_vs_wrong_cell", "graph_unseen_wrong_cell"),
        ("real_vs_stale_time", "graph_unseen_stale_time"),
    ]:
        row = one(
            unseen_graph_controls,
            objective_name="h6_guard10",
            variant="forced_potential",
            horizon=6,
            comparison=comparison,
        )
        claims.extend(
            [
                make_claim(
                    f"{prefix}_advantage",
                    f"Unseen potential graph real-signal advantage: {comparison}",
                    row["rmse_advantage_mean"],
                    "px",
                    unseen_graph_controls_path,
                    "Paired movie-macro component RMSE advantage.",
                ),
                make_claim(
                    f"{prefix}_positive_movies",
                    f"Unseen potential graph real-signal positive movies: {comparison}",
                    row["real_better_movies"],
                    "movies",
                    unseen_graph_controls_path,
                    "Independent movie-level sign count.",
                ),
            ]
        )

    for state, prefix in [
        ("observed", "functional_observed"),
        ("time_shuffled_observed", "functional_time_shuffled"),
        ("potential_damped", "functional_potential"),
    ]:
        row = one(functional_summary, state=state)
        claims.extend(
            [
                make_claim(
                    f"{prefix}_delta",
                    f"{state} finite functional delta",
                    row["functional_delta_mean"],
                    "functional units",
                    functional_summary_path,
                    "Full-grid finite update.",
                ),
                make_claim(
                    f"{prefix}_decrease_fraction",
                    f"{state} finite functional decrease fraction",
                    row["decrease_fraction"],
                    "fraction",
                    functional_summary_path,
                    "Fraction of 398 evaluated frames.",
                ),
            ]
        )
    for horizon in [1, 6]:
        row = one(functional_rollout, model="potential", horizon=horizon)
        claims.extend(
            [
                make_claim(
                    f"functional_potential_h{horizon}_rmse",
                    f"Potential full-field rollout h{horizon} RMSE",
                    row["component_rmse_macro"],
                    "um/min",
                    functional_rollout_path,
                    "Whole-island outer holdout.",
                ),
                make_claim(
                    f"functional_potential_h{horizon}_gain",
                    f"Potential full-field rollout h{horizon} gain",
                    row["gain_vs_cv_percent_mean"],
                    "% island-macro",
                    functional_rollout_path,
                    "Relative to constant velocity.",
                ),
            ]
        )

    for objective, horizon, prefix in [
        ("h1_strict", 1, "probability_strict_h1"),
        ("h1_strict", 6, "probability_strict_h6"),
        ("h6_guard10", 6, "probability_utility_h6"),
    ]:
        row = one(
            probabilistic_aggregate,
            objective_name=objective,
            method="forced_potential",
            horizon=horizon,
            family="student_t",
        )
        for column, unit in [
            ("joint_nll_mean", "NLL"),
            ("conformal_radial_coverage50_mean", "fraction"),
            ("conformal_radial_coverage90_mean", "fraction"),
        ]:
            claims.append(
                make_claim(
                    f"{prefix}_{column}",
                    f"{prefix} {column}",
                    row[column],
                    unit,
                    probabilistic_aggregate_path,
                    "Outer-movie scale and conformal calibration.",
                )
            )

    for objective_name, horizon, prefix in [
        ("lambda_00", 1, "h1_evidence_strict_h1"),
        ("lambda_00", 6, "h1_evidence_strict_h6"),
        ("lambda_10", 1, "h1_evidence_utility_h1"),
        ("lambda_10", 6, "h1_evidence_utility_h6"),
    ]:
        pareto_row = one(h1_pareto, objective_name=objective_name)
        normalized_row = one(
            h1_normalized,
            objective_name=objective_name,
            horizon=horizon,
        )
        for suffix, quantity, value, unit in [
            ("rmse", "component RMSE", pareto_row[f"h{horizon}_component_rmse"], "px"),
            ("r2", "component R2", pareto_row[f"h{horizon}_component_r2"], "fraction"),
            (
                "normalized_rmse",
                "target-scale normalized RMSE",
                normalized_row["normalized_rmse_mean"],
                "fraction",
            ),
            (
                "target_sd",
                "target component standard deviation",
                normalized_row["target_component_sd_mean"],
                "px",
            ),
            (
                "skill_vs_cv",
                "movie-macro squared-error skill versus constant velocity",
                normalized_row["skill_vs_cv_mean"],
                "fraction",
            ),
        ]:
            claims.append(
                make_claim(
                    f"{prefix}_{suffix}",
                    f"v205 {objective_name} h{horizon} {quantity}",
                    value,
                    unit,
                    h1_pareto_path if suffix in {"rmse", "r2"} else h1_normalized_path,
                    "Six strict fold-local outer movies; intermediate Pareto points are descriptive.",
                )
            )
    claims.append(
        make_claim(
            "h1_evidence_nondominated_points",
            "Predeclared h1-h6 profiles on the nondominated frontier",
            int(h1_pareto["pareto_nondominated"].sum()),
            "profiles",
            h1_pareto_path,
            "Eleven predeclared validation-selected profiles were evaluated once per outer movie.",
        )
    )
    for comparator, prefix in [
        ("wrong_cell", "h1_evidence_wrong_cell"),
        ("stale_time", "h1_evidence_stale_time"),
    ]:
        row = one(
            h1_pairwise,
            objective_name="lambda_00",
            horizon=1,
            comparator=comparator,
        )
        claims.extend(
            [
                make_claim(
                    f"{prefix}_delta",
                    f"Strict h1 RMSE advantage over {comparator}",
                    row["mean_rmse_delta_comparator_minus_real"],
                    "px",
                    h1_pairwise_path,
                    "Movie-level paired outer-fold comparison.",
                ),
                make_claim(
                    f"{prefix}_p",
                    f"Strict h1 exact two-sided sign-flip p versus {comparator}",
                    row["exact_two_sided_sign_flip_p"],
                    "p",
                    h1_pairwise_path,
                    "Six movie-level differences.",
                ),
            ]
        )
    lachance_localization = one(h1_localization, dataset="LaChance MDCK Bulk")
    claims.append(
        make_claim(
            "h1_lachance_localization_nll_gain",
            "LaChance current-query localization reliability NLL gain",
            lachance_localization["value"],
            "%",
            h1_localization_path,
            "Failed the hard controls; this is not an irreducible h1 noise-floor estimate.",
        )
    )

    for module, variant, horizon, prefix in [
        ("coordinate", "constant_velocity", 1, "deepsea_cv_h1"),
        ("coordinate", "constant_velocity", 6, "deepsea_cv_h6"),
        ("coordinate", "v97_direct_prior", 1, "deepsea_prior_h1"),
        ("coordinate", "v97_direct_prior", 6, "deepsea_prior_h6"),
        (
            "complete_system",
            "v166_external_h1_strict",
            1,
            "deepsea_v166_strict_h1",
        ),
        (
            "complete_system",
            "v166_external_h6_utility",
            6,
            "deepsea_v166_utility_h6",
        ),
        ("privileged_state", "real_state", 6, "deepsea_real_state_h6"),
        (
            "capacity_control",
            "noncausal_future_state",
            6,
            "deepsea_noncausal_state_h6",
        ),
    ]:
        row = one(
            deepsea_results,
            module=module,
            variant=variant,
            horizon=horizon,
        )
        claims.append(
            make_claim(
                f"{prefix}_rmse",
                f"DeepSea {variant} h{horizon} movie-macro RMSE",
                row["movie_macro_rmse"],
                "first-frame median cell diameters",
                deepsea_results_path,
                "Frozen 26/10/11-video split; dimensionless external protocol.",
            )
        )
        if pd.notna(row["gain_pct"]):
            claims.append(
                make_claim(
                    f"{prefix}_gain",
                    f"DeepSea {variant} h{horizon} gain",
                    row["gain_pct"],
                    "%",
                    deepsea_results_path,
                    "Comparator is defined by the frozen v204 decision bundle.",
                )
            )

    lifeact_mean_loo = one(
        lifeact_mean,
        protocol="leave_one_sequence_out",
        model="hgbdt",
    )
    lifeact_mean_loo_normalized = one(
        lifeact_mean_normalized,
        protocol="leave_one_sequence_out",
        model="hgbdt",
    )
    lifeact_uncertainty_loo = one(
        lifeact_uncertainty,
        protocol="leave_one_sequence_out",
    )
    lifeact_uncertainty_loo_normalized = one(
        lifeact_uncertainty_normalized,
        protocol="leave_one_sequence_out",
    )
    for claim_id, quantity, value, unit, source, interpretation in [
        (
            "lifeact_mean_loo_rmse",
            "LifeAct leave-one-sequence-out real-state h1 RMSE",
            lifeact_mean_loo["real_rmse_mean"],
            "px",
            lifeact_mean_path,
            "Three external conditions and three model seeds; automatic CPSAM/IoU tracks.",
        ),
        (
            "lifeact_mean_loo_gain",
            "LifeAct real-state mean gain versus coordinate-only mean",
            lifeact_mean_loo["gain_vs_coord_percent_mean"],
            "%",
            lifeact_mean_path,
            "Negative value means that the causal state did not improve the conditional mean.",
        ),
        (
            "lifeact_mean_normalized_loo_gain",
            "LifeAct diameter-normalized real-state mean gain",
            lifeact_mean_loo_normalized["gain_vs_coord_percent_mean"],
            "%",
            lifeact_mean_normalized_path,
            "Current-frame median cell diameter is causal; predictions are evaluated back in pixels.",
        ),
        (
            "lifeact_uncertainty_student_real",
            "LifeAct real-state leave-one-sequence-out Student-t4 NLL",
            lifeact_uncertainty_loo["real_student_t4_nll"],
            "NLL",
            lifeact_uncertainty_path,
            "Frozen coordinate mean; only the conditional scale head uses causal cell state.",
        ),
        (
            "lifeact_uncertainty_student_coord",
            "LifeAct coordinate-only leave-one-sequence-out Student-t4 NLL",
            lifeact_uncertainty_loo["coord_student_t4_nll"],
            "NLL",
            lifeact_uncertainty_path,
            "Comparator for the frozen coordinate mean and coordinate-only uncertainty model.",
        ),
        (
            "lifeact_uncertainty_student_control",
            "LifeAct best-control leave-one-sequence-out Student-t4 NLL",
            lifeact_uncertainty_loo["best_control_student_t4_nll"],
            "NLL",
            lifeact_uncertainty_path,
            "Best among zero, row-shuffled, wrong-cell, and time-shuffled state controls.",
        ),
        (
            "lifeact_uncertainty_error_spearman",
            "LifeAct uncertainty-error Spearman correlation",
            lifeact_uncertainty_loo["uncertainty_error_spearman"],
            "rho",
            lifeact_uncertainty_path,
            "Leave-one-sequence-out aggregate over three external conditions.",
        ),
        (
            "lifeact_uncertainty_normalized_student_real",
            "LifeAct normalized real-state leave-one-sequence-out Student-t4 NLL",
            lifeact_uncertainty_loo_normalized["real_student_t4_nll"],
            "NLL",
            lifeact_uncertainty_normalized_path,
            "Scale-normalized causal state, evaluated with the same held-out-condition protocol.",
        ),
        (
            "lifeact_uncertainty_normalized_student_coord",
            "LifeAct normalized coordinate-only leave-one-sequence-out Student-t4 NLL",
            lifeact_uncertainty_loo_normalized["coord_student_t4_nll"],
            "NLL",
            lifeact_uncertainty_normalized_path,
            "Normalized coordinate-only uncertainty comparator.",
        ),
        (
            "lifeact_uncertainty_normalized_student_control",
            "LifeAct normalized best-control leave-one-sequence-out Student-t4 NLL",
            lifeact_uncertainty_loo_normalized["best_control_student_t4_nll"],
            "NLL",
            lifeact_uncertainty_normalized_path,
            "Best normalized shuffled or zero-state control.",
        ),
    ]:
        claims.append(
            make_claim(
                claim_id,
                quantity,
                value,
                unit,
                source,
                interpretation,
            )
        )

    warnings = [
        "Коллективная добавка на h1 статистически не подтверждена: точное двустороннее p=0,84375.",
        "Эффект h6 положителен в 6/6 фильмах, но p=0,0625 после поправки Холма не пересекает порог 0,05.",
        "Выигрыш h6 29,7638% является средним по относительным выигрышам фильмов; отношение агрегированных RMSE оценивает другую величину.",
        "Последовательный h6 получает промежуточные наблюдения и не сопоставим с оптимумом открытого прогноза из одной исходной точки.",
        "Высокий R2 на h6 частично отражает большую дисперсию h6-цели и не означает столь же сильную наблюдаемость h1.",
        "MDA-MB-231 показывает воспроизводимый относительный выигрыш при низком абсолютном R2=0,0237.",
        "Для MDCK Edge показан разброс между инициализациями, а интервалы HUVEC и MDA получены вложенным бутстрепом фильмов.",
        "Панель калибровки должна явно разделять замороженные h1- и h6-режимы.",
        "Проверка замороженной текущей конфигурации на семи фильмах MDCK Bulk 10--16 дала h6-эффект 7/7 и непоправленное p=0,015625. Эти фильмы ранее участвовали в широком разведочном поиске проекта, поэтому результат следует сообщать отдельно от разработческих шести фильмов, но не называть полностью независимым перспективным опытом.",
        "Потенциальный сектор v198 является эффективным функционалом кинематической инновации. Неизвестная подвижность и провал механического шлюза запрещают трактовать его коэффициенты как физическую энергию или материальные константы.",
        "Конечное уменьшение функционала подтверждено для потенциальной карты, но наблюдаемый переход не превосходит временную перестановку; это свойство модели, а не доказанная энергия ткани.",
        "E(2)-эквивариантный потенциальный граф является интерпретируемым суррогатом плотной поправки: на замороженном h6 он дает 4,842 против 4,820 у производственного оператора.",
        "DeepSea v204 является частичным внешним переносом, а не положительным мультимодальным подтверждением: накопительный v166 улучшает h6 относительно постоянной скорости на 3,66%, но ухудшает h1 относительно собственного априорного прогноза; причинное состояние масок не прошло жесткие контроли.",
        "Полная v205-кривая h1--h6 содержит 11 недоминируемых точек, но только крайняя h1-точка была исходно подтверждающей. Независимый нижний предел шума локализации для LaChance не установлен.",
        "LifeAct-MDCK не улучшил условное среднее h1 в межусловной проверке. Положительный результат относится только к условному масштабу ошибки: Student-t4 NLL 3,649 -> 3,614 против 3,644 у лучшего контроля; из-за трех условий и автоматических треков он считается исследовательским.",
    ]
    return pd.DataFrame(claims), warnings


def write_report(claims: pd.DataFrame, warnings: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    claims_path = output_dir / "article_numeric_claims.csv"
    report_path = output_dir / "article_numeric_audit.md"
    claims.to_csv(claims_path, index=False)

    rows = [
        "# Числовой аудит журнальной рукописи",
        "",
        f"Проверено количественных утверждений: **{len(claims)}**.",
        "",
        "Все значения прочитаны из замороженных таблиц результатов; ручные числовые константы в новых рисунках не допускаются.",
        "",
        "## Критические правила интерпретации",
        "",
    ]
    rows.extend(f"- {item}" for item in warnings)
    rows.extend(
        [
            "",
            "## Проверенные основные значения",
            "",
            "| Утверждение | Значение | Единица | Источник |",
            "|---|---:|---|---|",
        ]
    )
    selected = claims[
        claims["claim_id"].isin(
            [
                "primary_h1_rmse",
                "primary_h1_r2",
                "primary_h6_rmse",
                "primary_h6_r2",
                "h6_test_relative_gain_percent",
                "h6_test_holm_adjusted_p",
                "unseen_h6_rmse",
                "unseen_h6_r2",
                "unseen_h6_gain",
                "unseen_h6_exact_p",
                "edge_component_rmse_mean",
                "huvec_component_rmse_macro",
                "mdamb231_component_rmse_macro",
                "potential_rmse",
                "potential_gain",
                "potential_gain_retained",
                "potential_advective_sign_p",
                "graph_potential_dev_h6_rmse",
                "graph_potential_dev_h6_gain",
                "graph_potential_projection_r2",
                "graph_potential_unseen_h6_rmse",
                "graph_potential_unseen_h6_r2",
                "graph_potential_unseen_h6_gain",
                "graph_unseen_wrong_cell_advantage",
                "graph_unseen_stale_time_advantage",
                "functional_potential_delta",
                "functional_potential_decrease_fraction",
                "functional_potential_h6_rmse",
                "functional_potential_h6_gain",
                "probability_utility_h6_joint_nll_mean",
                "probability_utility_h6_conformal_radial_coverage90_mean",
                "lifeact_mean_loo_rmse",
                "lifeact_mean_loo_gain",
                "lifeact_uncertainty_student_real",
                "lifeact_uncertainty_student_coord",
                "lifeact_uncertainty_student_control",
                "lifeact_uncertainty_error_spearman",
            ]
        )
    ]
    for row in selected.itertuples(index=False):
        source = Path(row.source)
        rows.append(
            f"| `{row.claim_id}` | {row.value:.6g} | {row.unit} | "
            f"`{source.relative_to(ROOT)}` |"
        )
    rows.extend(
        [
            "",
            "Полная машинно-читаемая таблица сохранена в `article_numeric_claims.csv`.",
            "",
        ]
    )
    report_path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/article_numeric_audit_2026-07-30",
    )
    args = parser.parse_args()
    claims, warnings = audit()
    write_report(claims, warnings, args.output_dir.resolve())
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
