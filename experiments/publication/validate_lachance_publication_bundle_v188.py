#!/usr/bin/env python3
"""Validate the protocol-safe v188 publication evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_lachance_publication_bundle_v188 as bundle_v188


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = bundle_v188.DEFAULT_OUT
REQUIRED_FILES = {
    "v188_protocol_contract.csv",
    "v188_multiplicity_contract.json",
    "v188_contract_manifest.json",
    "v188_source_status.csv",
    "v188_primary_online_benchmark.csv",
    "v188_primary_online_movie_metrics.csv",
    "v188_paired_movie_statistics.csv",
    "v188_configuration_unseen_confirmation.csv",
    "v188_external_nested_lomo.csv",
    "v188_external_causal_controls.csv",
    "v188_probabilistic_metrics.csv",
    "v188_robustness_matrix.csv",
    "v188_robustness_uncertainty.csv",
    "v188_robustness_uncertainty_response.csv",
    "v188_fit_scope_sensitivity.csv",
    "v188_fit_scope_movie_metrics.csv",
    "v188_observability_gate_matrix.csv",
    "v188_claim_scope.csv",
    "v188_comparator_status.csv",
    "v188_pending_tasks.csv",
    "v188_publication_report.md",
    "v188_artifact_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--require-publication-ready",
        action="store_true",
        help="Fail unless all evidence and release blockers are closed.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(entries: dict[str, str], label: str) -> int:
    checked = 0
    for raw_path, expected in entries.items():
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} hash mismatch: {path}")
        checked += 1
    return checked


def require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head().to_dict("records")
        raise ValueError(f"{label} duplicate keys {columns}: {examples}")


def require_finite(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")


def validate_protocol(bundle: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(bundle / "v188_protocol_contract.csv")
    if len(frame) != 6:
        raise ValueError("Protocol contract must contain six outer folds")
    if set(frame["outer_test_movie"].astype(int)) != set(
        bundle_v188.MOVIES
    ):
        raise ValueError("Protocol contract has the wrong outer movies")
    require_unique(frame, ["outer_test_movie"], "protocol contract")
    if not frame["seed_key_target_match"].eq(True).all():
        raise ValueError("Protocol contract failed seed key/target equality")
    if not frame["status"].eq("frozen_complete").all():
        raise ValueError("Protocol contract is not frozen complete")
    if (
        frame["final_fit_scope_by_method"]
        .str.contains("v166 bounded update refit on train\\+validation")
        .ne(True)
        .any()
    ):
        raise ValueError("Protocol contract hides method-specific fit scope")
    for column in (
        "test_key_sha256",
        "test_target_sha256",
        "test_row_target_sha256",
    ):
        if frame[column].str.fullmatch(r"[0-9a-f]{64}").ne(True).any():
            raise ValueError(f"Invalid {column}")
    multiplicity = json.loads(
        (bundle / "v188_multiplicity_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if multiplicity.get("independent_unit") != "outer test movie":
        raise ValueError("Multiplicity contract uses the wrong unit")
    family = multiplicity.get("confirmatory_family", [])
    if [row.get("hypothesis_id") for row in family] != ["H1", "H2"]:
        raise ValueError("Confirmatory family must be exactly H1 and H2")
    if multiplicity.get("correction") != (
        "Holm step-down across H1 and H2"
    ):
        raise ValueError("Unexpected multiplicity correction")
    return frame, bundle_v188.protocol_key_set_hash(frame)


def validate_benchmark(bundle: Path, key_hash: str) -> dict[str, Any]:
    summary = pd.read_csv(bundle / "v188_primary_online_benchmark.csv")
    movies = pd.read_csv(
        bundle / "v188_primary_online_movie_metrics.csv"
    )
    required_methods = set(bundle_v188.METHOD_METADATA)
    if not required_methods.issubset(set(summary["method"])):
        missing = sorted(required_methods - set(summary["method"]))
        raise ValueError(f"Primary benchmark misses methods: {missing}")
    require_unique(summary, ["method", "horizon"], "benchmark summary")
    require_unique(
        movies,
        ["test_movie", "method", "horizon"],
        "benchmark movie metrics",
    )
    require_finite(
        summary,
        ["component_rmse", "vector_rmse", "r2"],
        "benchmark summary",
    )
    if not summary["movies"].eq(6).all():
        raise ValueError("A benchmark row does not cover six movies")
    if not summary["ordered_key_set_sha256"].eq(key_hash).all():
        raise ValueError("Benchmark ordered-key set does not match contract")
    for method, group in movies.groupby("method"):
        if set(group["test_movie"].astype(int)) != set(bundle_v188.MOVIES):
            raise ValueError(f"{method} has incomplete movie coverage")
    expected = {
        ("v166_h1_strict", 1): 3.474374,
        ("v166_h6_utility", 6): 5.500749,
        ("kalmannet", 1): 3.926258,
        ("kalmannet", 6): 8.794405,
    }
    for (method, horizon), value in expected.items():
        row = summary[
            summary["method"].eq(method)
            & summary["horizon"].eq(horizon)
        ]
        if len(row) != 1:
            raise ValueError(f"Missing frozen metric {method} h{horizon}")
        actual = float(row.iloc[0]["component_rmse"])
        if not np.isclose(actual, value, atol=1e-5, rtol=0.0):
            raise ValueError(
                f"Frozen metric drift for {method} h{horizon}: {actual}"
            )
    return {
        "methods": int(summary["method"].nunique()),
        "summary_rows": len(summary),
        "movie_rows": len(movies),
    }


def validate_statistics(bundle: Path) -> dict[str, Any]:
    frame = pd.read_csv(bundle / "v188_paired_movie_statistics.csv")
    require_unique(
        frame,
        ["method", "comparator", "horizon"],
        "paired statistics",
    )
    primary = frame[frame["confirmatory"].eq(True)].sort_values(
        "hypothesis_id"
    )
    if primary["hypothesis_id"].tolist() != ["H1", "H2"]:
        raise ValueError("Paired table does not contain exactly H1 and H2")
    if not primary["movies"].eq(6).all():
        raise ValueError("Confirmatory inference is not movie-level n=6")
    require_finite(
        primary,
        [
            "mean_rmse_delta_comparator_minus_method",
            "delta_ci_low",
            "delta_ci_high",
            "exact_two_sided_sign_flip_p",
            "holm_adjusted_p",
        ],
        "confirmatory paired statistics",
    )
    if not (
        primary["holm_adjusted_p"]
        >= primary["exact_two_sided_sign_flip_p"]
    ).all():
        raise ValueError("Holm adjustment is smaller than raw p-value")
    return {
        "confirmatory_hypotheses": len(primary),
        "holm_rejections": int(primary["reject_at_0_05"].sum()),
    }


def validate_external(bundle: Path) -> dict[str, int]:
    frame = pd.read_csv(bundle / "v188_external_nested_lomo.csv")
    expected = {"HUVEC": 18, "MDAMB231": 17}
    for dataset, folds in expected.items():
        subset = frame[
            frame["dataset"].eq(dataset) & frame["control"].eq("real")
        ]
        if set(subset["outer_folds"].astype(int)) != {folds}:
            raise ValueError(
                f"{dataset} does not certify {folds} nested outer folds"
            )
    return expected


def validate_robustness(bundle: Path) -> dict[str, Any]:
    frame = pd.read_csv(bundle / "v188_robustness_matrix.csv")
    require_unique(
        frame,
        ["operating_point", "condition", "horizon"],
        "robustness matrix",
    )
    conditions = set(frame["condition"])
    required = {
        "update_every_1",
        "update_every_2",
        "update_every_3",
        "update_every_6",
        "no_update",
        "missing_0.1",
        "missing_0.2",
        "missing_0.4",
        "tracking_noise_0.5px",
        "tracking_noise_1px",
        "tracking_noise_2px",
        "delay_1frame",
        "wrong_cell",
    }
    missing = sorted(required - conditions)
    if missing:
        raise ValueError(f"Robustness matrix misses conditions: {missing}")
    require_finite(
        frame,
        ["component_rmse_mean", "vector_rmse_mean", "r2_mean"],
        "robustness matrix",
    )
    uncertainty = pd.read_csv(
        bundle / "v188_robustness_uncertainty.csv"
    )
    response = pd.read_csv(
        bundle / "v188_robustness_uncertainty_response.csv"
    )
    require_unique(
        uncertainty,
        [
            "operating_point",
            "calibration_mode",
            "condition",
            "horizon",
        ],
        "robustness uncertainty",
    )
    require_unique(
        response,
        ["operating_point", "condition"],
        "robustness uncertainty response",
    )
    expected_modes = {
        "frozen_base_scale",
        "frozen_state_aware_scale",
    }
    if set(uncertainty["calibration_mode"]) != expected_modes:
        raise ValueError("Robustness uncertainty has wrong calibration modes")
    if set(uncertainty["operating_point"]) != {
        "h1_strict",
        "h6_utility",
    }:
        raise ValueError("Robustness uncertainty misses an operating point")
    require_finite(
        uncertainty,
        [
            "nll_mean",
            "coverage_50_mean",
            "coverage_90_mean",
            "calibration_error_mean",
            "scale_factor_mean",
            "update_scale_factor_mean",
            "prior_scale_factor_mean",
            "mean_step_scale_mean",
            "mean_endpoint_scale_mean",
        ],
        "robustness uncertainty",
    )
    require_finite(
        response,
        [
            "scale_factor_mean",
            "scale_factor_ratio_vs_real_mean",
            "calibration_error_gain_mean",
            "coverage_90_gain_mean",
            "nll_gain_mean",
        ],
        "robustness uncertainty response",
    )
    stressed = response[
        response["condition"].isin(
            ["missing_0.4", "tracking_noise_2px"]
        )
    ]
    if len(stressed) != 4:
        raise ValueError("Robustness uncertainty misses stressed responses")
    if not stressed["scale_factor_ratio_vs_real_mean"].gt(1.0).all():
        raise ValueError(
            "Validation-calibrated uncertainty does not expand under stress"
        )
    if not stressed["coverage_90_gain_mean"].gt(0.0).all():
        raise ValueError(
            "Frozen state-aware uncertainty does not increase stress coverage"
        )
    missing_expanded = stressed[
        stressed["condition"].eq("missing_0.4")
    ]["movies_expanded_vs_real"]
    noise_expanded = stressed[
        stressed["condition"].eq("tracking_noise_2px")
    ]["movies_expanded_vs_real"]
    if not missing_expanded.ge(5).all():
        raise ValueError(
            "40% missingness expands uncertainty on fewer than 5/6 movies"
        )
    if not noise_expanded.eq(6).all():
        raise ValueError(
            "2 px tracking noise does not expand uncertainty on 6/6 movies"
        )
    return {
        "rows": len(frame),
        "operating_points": sorted(frame["operating_point"].unique()),
        "has_tracking_noise_2px": True,
        "uncertainty_rows": len(uncertainty),
        "uncertainty_response_rows": len(response),
        "stress_uncertainty_expands": True,
    }


def validate_fit_scope(bundle: Path) -> dict[str, Any]:
    summary = pd.read_csv(bundle / "v188_fit_scope_sensitivity.csv")
    metrics = pd.read_csv(
        bundle / "v188_fit_scope_movie_metrics.csv"
    )
    require_unique(
        summary,
        ["objective_name", "fit_scope", "horizon"],
        "fit-scope summary",
    )
    require_unique(
        metrics,
        ["test_movie", "objective_name", "fit_scope", "horizon"],
        "fit-scope movie metrics",
    )
    require_finite(
        summary,
        ["component_rmse", "vector_rmse", "r2"],
        "fit-scope summary",
    )
    if not summary["movies"].eq(6).all():
        raise ValueError("Fit-scope summary does not cover six movies")
    if not metrics["outer_test_used_for_fit_or_selection"].eq(False).all():
        raise ValueError("Outer test entered the fit-scope audit")
    checks = {
        ("h1_strict", "train_plus_validation", 1): 3.474374,
        ("h6_guard10", "train_plus_validation", 6): 5.500749,
    }
    for (objective, scope, horizon), expected in checks.items():
        row = summary[
            summary["objective_name"].eq(objective)
            & summary["fit_scope"].eq(scope)
            & summary["horizon"].eq(horizon)
        ]
        actual = float(row.iloc[0]["component_rmse"])
        if not np.isclose(actual, expected, atol=1e-5, rtol=0.0):
            raise ValueError(
                f"Fit-scope historical reproduction drift: {actual}"
            )
    headline = summary[
        (
            summary["objective_name"].eq("h6_guard10")
            & summary["horizon"].eq(6)
        )
    ].set_index("fit_scope")
    degradation = 100.0 * (
        float(headline.loc["train_only", "component_rmse"])
        - float(
            headline.loc[
                "train_plus_validation",
                "component_rmse",
            ]
        )
    ) / float(
        headline.loc["train_plus_validation", "component_rmse"]
    )
    if degradation >= 1.0:
        raise ValueError(
            f"Train-only v166 h6 degrades by {degradation:.3f}%"
        )
    if int(headline.loc["train_only", "movies_improved"]) != 6:
        raise ValueError("Train-only v166 h6 does not improve on 6/6 movies")
    return {
        "rows": len(summary),
        "movie_rows": len(metrics),
        "train_only_h6_relative_degradation_percent": degradation,
        "train_only_h6_movies_improved": 6,
    }


def validate_observability(bundle: Path) -> dict[str, Any]:
    frame = pd.read_csv(bundle / "v188_observability_gate_matrix.csv")
    passed_deployable = frame[
        frame["passed"].eq(True)
        & frame["deployable_to_lachance"].eq(True)
    ]
    if len(passed_deployable):
        raise ValueError(
            "A deployable observability state unexpectedly passed; "
            "claim scope must be manually reviewed"
        )
    return {
        "gates": len(frame),
        "deployable_passes": 0,
    }


def run(args: argparse.Namespace) -> None:
    bundle = args.bundle_dir.resolve()
    missing = [
        name for name in sorted(REQUIRED_FILES) if not (bundle / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing v188 artifacts: {missing}")
    manifest = json.loads(
        (bundle / "v188_artifact_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("architecture_frozen") is not True:
        raise ValueError("Architecture is not frozen")
    if manifest.get("protocol_tables_separated") is not True:
        raise ValueError("Protocol tables are not separated")
    if manifest.get("global_sota_claim_allowed") is not False:
        raise ValueError("Global SOTA must remain forbidden")
    if manifest.get("fixed_origin_oracle_primary") is not False:
        raise ValueError("Fixed-origin oracle cannot be primary")
    if set(manifest.get("matched_outer_movies", [])) & set(
        manifest.get("configuration_unseen_movies", [])
    ):
        raise ValueError("Matched and unseen cohorts overlap")

    artifact_hashes = check_hashes(
        manifest.get("artifact_sha256", {}),
        "artifact",
    )
    input_hashes = check_hashes(
        manifest.get("input_sha256", {}),
        "input",
    )
    protocol, key_hash = validate_protocol(bundle)
    benchmark = validate_benchmark(bundle, key_hash)
    statistics = validate_statistics(bundle)
    external = validate_external(bundle)
    robustness = validate_robustness(bundle)
    fit_scope = validate_fit_scope(bundle)
    observability = validate_observability(bundle)

    scope = pd.read_csv(bundle / "v188_claim_scope.csv")
    require_unique(scope, ["claim_id"], "claim scope")
    if not scope["forbidden_claim"].fillna("").str.strip().astype(bool).all():
        raise ValueError("Every claim must state a forbidden interpretation")
    comparator = pd.read_csv(bundle / "v188_comparator_status.csv")
    if len(comparator) != 18:
        raise ValueError("KalmanNet comparator table must contain 18 fold-seed jobs")
    if not comparator["no_future_sentinel_pass"].eq(True).all():
        raise ValueError("A KalmanNet no-future sentinel failed")
    if not comparator["contract_match"].eq(True).all():
        raise ValueError("A KalmanNet fold does not match the frozen contract")
    kalmannet_complete = bool(
        len(comparator)
        and comparator["status"].eq("complete").all()
    )
    evidence_complete = bool(manifest.get("evidence_complete"))
    if evidence_complete != kalmannet_complete:
        raise ValueError("Manifest/comparator completion status mismatch")
    pending = pd.read_csv(bundle / "v188_pending_tasks.csv")
    blockers = pending[
        pending["blocks_publication_ready"].eq(True)
        & ~pending["status"].eq("complete")
    ]["task_id"].tolist()
    publication_ready = not blockers
    if bool(manifest.get("publication_ready")) != publication_ready:
        raise ValueError("Publication-ready state does not match blockers")
    if args.require_publication_ready and not publication_ready:
        raise RuntimeError(
            "v188 is structurally valid but publication blockers remain: "
            + ", ".join(blockers)
        )

    status = "pass" if publication_ready else "pass_with_blockers"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "bundle": str(bundle),
        "architecture_frozen": True,
        "protocol_tables_separated": True,
        "global_sota_claim_allowed": False,
        "publication_ready": publication_ready,
        "publication_blockers": blockers,
        "evidence_complete": evidence_complete,
        "kalmannet_complete": kalmannet_complete,
        "artifact_hashes_checked": artifact_hashes,
        "input_hashes_checked": input_hashes,
        "protocol_folds": len(protocol),
        "ordered_key_set_sha256": key_hash,
        "benchmark": benchmark,
        "statistics": statistics,
        "external_outer_folds": external,
        "robustness": robustness,
        "fit_scope_sensitivity": fit_scope,
        "observability": observability,
    }
    output = bundle / "v188_validation_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v188-validate] {status.upper()}: {output}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
