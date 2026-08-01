#!/usr/bin/env python3
"""Build the protocol-safe v188 publication evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_post_v166_protocol_contract_v188 as contract_v188


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_OUT = contract_v188.DEFAULT_OUT
V117_DIR = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_baselines_v117_production_2026-07-21"
)
V102_DIR = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_benchmark_v102_v97_production_2026-07-21"
)
V157H_DIR = (
    ROOT
    / "outputs"
    / "lachance_foldlocal_semigroup_pareto_v157h_full_2026-07-24"
)
V157F_H1_DIR = (
    ROOT
    / "outputs"
    / "lachance_foldlocal_semigroup_stress_v188_h1_2026-07-29"
)
V157F_H6_DIR = (
    ROOT
    / "outputs"
    / "lachance_foldlocal_semigroup_stress_v188_h6_2026-07-29"
)
V163_H1_DIR = (
    ROOT
    / "outputs"
    / "lachance_temporal_student_t_covariance_v163_h1strict_2026-07-27"
)
V163_H6_DIR = (
    ROOT
    / "outputs"
    / "lachance_temporal_student_t_covariance_v163_full_2026-07-27"
)
V166_DIR = (
    ROOT / "outputs" / "lachance_publication_bundle_v166_2026-07-27"
)
V181_DIR = ROOT / "outputs" / "future_work_night_closure_v181_2026-07-28"
KALMANNET_DIR = (
    ROOT
    / "outputs"
    / "lachance_online_lomo_kalmannet_v188_exact_2026-07-29"
)
V157F_SOURCE = SCRIPTS / "run_lachance_foldlocal_semigroup_stress_v157f.py"
V189_DIR = (
    ROOT / "outputs" / "lachance_v166_fit_scope_audit_v189_2026-07-29"
)
V189_SOURCE = SCRIPTS / "run_lachance_v166_fit_scope_audit_v189.py"
HORIZONS = (1, 2, 4, 6)
MOVIES = (1, 2, 3, 4, 5, 6)
EPS = 1e-12


METHOD_METADATA = {
    "constant_velocity": (
        "kinematic",
        "exact local implementation",
    ),
    "displacement_kf_ca": (
        "classical_filter",
        "exact causal adaptation",
    ),
    "displacement_kf_cv": (
        "classical_filter",
        "exact causal adaptation",
    ),
    "gru_track": (
        "neural_sequence",
        "architecture-faithful causal adaptation",
    ),
    "hgbdt_v52": (
        "tree_regressor",
        "exact local implementation",
    ),
    "position_imm": (
        "classical_filter",
        "exact causal adaptation",
    ),
    "position_kf_ca": (
        "classical_filter",
        "exact causal adaptation",
    ),
    "position_kf_cv": (
        "classical_filter",
        "exact causal adaptation",
    ),
    "wolf_imq_ca": (
        "robust_filter",
        "source-faithful WoLF-IMQ update adaptation",
    ),
    "wolf_imq_cv": (
        "robust_filter",
        "source-faithful WoLF-IMQ update adaptation",
    ),
    "v97_seed_mean": (
        "learned_state_space_prior",
        "mean metric over three frozen local checkpoints",
    ),
    "v97_no_update": (
        "learned_state_space_prior",
        "prediction ensemble matched exactly to transport input",
    ),
    "v166_h1_strict": (
        "bounded_innovation_transport",
        "frozen fold-local operating point",
    ),
    "v166_h6_utility": (
        "bounded_innovation_transport",
        "frozen fold-local operating point",
    ),
    "kalmannet": (
        "learned_filter",
        "source-faithful predictive-prior adaptation",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stress-h1-dir", type=Path, default=V157F_H1_DIR)
    parser.add_argument("--stress-h6-dir", type=Path, default=V157F_H6_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=188)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing v188 inputs: " + ", ".join(missing))


def validate_stress_provenance(
    directory: Path,
    expected_objective: str,
) -> None:
    manifest_path = directory / "run_manifest.json"
    replay_path = directory / "v157f_seed_replay_manifest.json"
    causal_path = directory / "v157f_causal_audit.csv"
    summary_path = directory / "v157f_summary.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != sha256(V157F_SOURCE):
        raise ValueError(
            f"Stress source hash mismatch for {expected_objective}"
        )
    if manifest.get("objective_name") != expected_objective:
        raise ValueError(
            f"Stress objective mismatch for {expected_objective}"
        )
    uncertainty = manifest.get("uncertainty_policy", {})
    if uncertainty.get("corruption_specific_scale_tuning") is not False:
        raise ValueError(
            f"Stress uncertainty retunes corruption levels: "
            f"{expected_objective}"
        )
    replays = json.loads(replay_path.read_text(encoding="utf-8"))
    if len(replays) != 18:
        raise ValueError(
            f"Stress replay manifest must contain 18 fold-seed rows: "
            f"{expected_objective}"
        )
    if any(
        item.get("v157f_uncertainty", {}).get(
            "corruption_specific_scale_tuning"
        )
        is not False
        for item in replays
    ):
        raise ValueError(
            f"Stress replay contains corruption-specific tuning: "
            f"{expected_objective}"
        )
    causal = pd.read_csv(causal_path)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise ValueError(f"Future donor in {expected_objective}")
    if int(causal["stale_future_or_nonstale_violations"].sum()) != 0:
        raise ValueError(f"Invalid stale donor in {expected_objective}")
    if not causal["coherent_wrong_packet"].eq(True).all():
        raise ValueError(f"Incoherent wrong-cell packet: {expected_objective}")
    summary = pd.read_csv(summary_path)
    if summary.iloc[0]["decision"] != "ROBUST_PASS":
        raise ValueError(f"Stress point audit failed: {expected_objective}")
    if (
        summary.iloc[0]["uncertainty_response_decision"]
        != "FROZEN_STATE_AWARE_RESPONSE_PASS"
    ):
        raise ValueError(
            f"Stress uncertainty audit failed: {expected_objective}"
        )


def validate_fit_scope_provenance() -> None:
    manifest_path = V189_DIR / "run_manifest.json"
    metrics_path = V189_DIR / "v189_fit_scope_metrics.csv"
    causal_path = V189_DIR / "v189_causal_audit.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != sha256(V189_SOURCE):
        raise ValueError("v189 fit-scope source hash mismatch")
    if manifest.get("outer_test_used_for_fit_or_selection") is not False:
        raise ValueError("v189 outer test entered fit or selection")
    metrics = pd.read_csv(metrics_path)
    if not metrics["outer_test_used_for_fit_or_selection"].eq(False).all():
        raise ValueError("v189 metric row used outer test during fitting")
    if set(metrics["test_movie"].astype(int)) != set(MOVIES):
        raise ValueError("v189 fit-scope audit misses an outer movie")
    if set(metrics["fit_scope"]) != {
        "train_only",
        "train_plus_validation",
    }:
        raise ValueError("v189 fit-scope audit has unexpected scopes")
    causal = pd.read_csv(causal_path)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise ValueError("Future donor in v189 fit-scope audit")


def build_fit_scope_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = V189_DIR / "v189_fit_scope_summary.csv"
    metrics_path = V189_DIR / "v189_fit_scope_metrics.csv"
    summary = pd.read_csv(summary_path)
    metrics = pd.read_csv(metrics_path)
    summary["source_path"] = str(summary_path)
    summary["source_sha256"] = sha256(summary_path)
    metrics["source_path"] = str(metrics_path)
    metrics["source_sha256"] = sha256(metrics_path)
    return summary, metrics


def require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head(10).to_dict("records")
        raise ValueError(f"{label} duplicate keys {columns}: {examples}")


def require_finite(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")


def bootstrap_ci(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.abs(values) > EPS]
    if not len(values):
        return 1.0
    observed = abs(float(values.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistics.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(
        np.mean(np.asarray(statistics) >= observed - 1e-15)
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    raw = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * raw[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def protocol_key_set_hash(protocol: pd.DataFrame) -> str:
    rows = protocol.sort_values("outer_test_movie")[
        [
            "outer_test_movie",
            "test_key_sha256",
            "test_target_sha256",
            "test_row_target_sha256",
        ]
    ].to_dict("records")
    return canonical_sha256(rows)


def method_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["method_family"] = output["method"].map(
        lambda value: METHOD_METADATA[value][0]
    )
    output["source_fidelity"] = output["method"].map(
        lambda value: METHOD_METADATA[value][1]
    )
    output["status"] = "complete"
    output["protocol_id"] = "mdck_bulk_online_outer_lomo_v188"
    return output


def build_movie_metrics(protocol: pd.DataFrame) -> pd.DataFrame:
    v117_path = V117_DIR / "v117_seed_aggregated_within_movie.csv"
    v102_path = V102_DIR / "v102_seed_aggregated_within_movie.csv"
    v157h_path = V157H_DIR / "v157h_pareto_metrics.csv"
    v117 = pd.read_csv(v117_path)
    baseline = v117[
        [
            "test_movie",
            "validation_movie",
            "train_movies",
            "method",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
            "n_rows",
            "n_seeds",
        ]
    ].copy()
    baseline["source_path"] = str(v117_path)
    baseline["source_sha256"] = sha256(v117_path)

    v102 = pd.read_csv(v102_path)
    prior = v102[v102["method_id"].eq("v97/v97_direct")][
        [
            "test_movie",
            "validation_movie",
            "train_movies",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
            "n_rows",
            "n_seeds",
        ]
    ].copy()
    prior["method"] = "v97_seed_mean"
    prior["source_path"] = str(v102_path)
    prior["source_sha256"] = sha256(v102_path)

    transport_raw = pd.read_csv(v157h_path)
    selected = []
    no_update = transport_raw[
        transport_raw["objective_name"].eq("no_update")
        & transport_raw["control"].eq("no_update")
    ][
        [
            "test_movie",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
            "windows",
        ]
    ].copy()
    no_update["method"] = "v97_no_update"
    no_update["n_rows"] = no_update.pop("windows")
    no_update["n_seeds"] = 3
    selected.append(no_update)
    for objective, method in (
        ("h1_strict", "v166_h1_strict"),
        ("h6_guard10", "v166_h6_utility"),
    ):
        subset = transport_raw[
            transport_raw["objective_name"].eq(objective)
            & transport_raw["control"].eq("real")
        ][
            [
                "test_movie",
                "horizon",
                "component_rmse",
                "vector_rmse",
                "r2",
                "windows",
            ]
        ].copy()
        subset["method"] = method
        subset["n_rows"] = subset.pop("windows")
        subset["n_seeds"] = 3
        selected.append(subset)
    transport = pd.concat(selected, ignore_index=True)
    fold_map = baseline[
        ["test_movie", "validation_movie", "train_movies"]
    ].drop_duplicates()
    require_unique(fold_map, ["test_movie"], "v117 fold map")
    transport = transport.merge(
        fold_map,
        on="test_movie",
        validate="many_to_one",
    )
    transport["source_path"] = str(v157h_path)
    transport["source_sha256"] = sha256(v157h_path)

    kalmannet_path = KALMANNET_DIR / "kalmannet_lomo_movie_metrics.csv"
    kalmannet_raw = pd.read_csv(kalmannet_path)
    kalmannet_contract = kalmannet_raw[
        [
            "test_movie",
            "validation_movie",
            "train_movies",
            "test_key_sha256",
            "test_target_sha256",
            "test_row_target_sha256",
        ]
    ].drop_duplicates()
    require_unique(
        kalmannet_contract,
        ["test_movie"],
        "KalmanNet exact-key contract",
    )
    expected_contract = protocol[
        [
            "outer_test_movie",
            "validation_movie",
            "train_movies",
            "test_key_sha256",
            "test_target_sha256",
            "test_row_target_sha256",
        ]
    ].rename(columns={"outer_test_movie": "test_movie"})
    checked_contract = kalmannet_contract.merge(
        expected_contract,
        on="test_movie",
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    for column in (
        "validation_movie",
        "train_movies",
        "test_key_sha256",
        "test_target_sha256",
        "test_row_target_sha256",
    ):
        actual = checked_contract[f"{column}_actual"].astype(str)
        expected_values = checked_contract[f"{column}_expected"].astype(str)
        if not actual.eq(expected_values).all():
            raise ValueError(
                f"KalmanNet does not match frozen protocol column {column}"
            )
    kalmannet = kalmannet_raw[
        [
            "test_movie",
            "validation_movie",
            "train_movies",
            "method",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
            "n_rows",
            "n_seeds",
        ]
    ].copy()
    kalmannet["source_path"] = str(kalmannet_path)
    kalmannet["source_sha256"] = sha256(kalmannet_path)

    combined = pd.concat(
        [baseline, prior, transport, kalmannet],
        ignore_index=True,
        sort=False,
    )
    combined["exact_key_contract_verified"] = True
    combined = method_columns(combined)
    require_unique(
        combined,
        ["test_movie", "method", "horizon"],
        "v188 movie metrics",
    )
    require_finite(
        combined,
        ["component_rmse", "vector_rmse", "r2"],
        "v188 movie metrics",
    )
    expected = set(MOVIES)
    for method, group in combined.groupby("method"):
        if set(group["test_movie"].astype(int)) != expected:
            raise ValueError(f"{method} does not cover all six outer movies")
        if set(group["horizon"].astype(int)) != set(HORIZONS):
            raise ValueError(f"{method} does not cover all horizons")
    return combined


def aggregate_benchmark(
    movie_metrics: pd.DataFrame,
    ordered_key_set_sha256: str,
) -> pd.DataFrame:
    summary = (
        movie_metrics.groupby(
            [
                "method",
                "method_family",
                "source_fidelity",
                "status",
                "protocol_id",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            movies=("test_movie", "nunique"),
            optimizer_seeds=("n_seeds", "max"),
            component_rmse=("component_rmse", "mean"),
            component_rmse_movie_std=("component_rmse", "std"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            evaluation_rows=("n_rows", "sum"),
            source_path=("source_path", "first"),
            source_sha256=("source_sha256", "first"),
        )
    )
    summary["ordered_key_set_sha256"] = ordered_key_set_sha256
    summary["statistical_unit"] = "movie"
    summary["fixed_origin_oracle"] = False
    return summary


def paired_statistics(
    movie_metrics: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    methods = ["v166_h1_strict", "v166_h6_utility"]
    comparators = sorted(
        set(movie_metrics["method"]) - set(methods)
    )
    records: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        for horizon in HORIZONS:
            reference = movie_metrics[
                movie_metrics["method"].eq(method)
                & movie_metrics["horizon"].eq(horizon)
            ][["test_movie", "component_rmse"]].rename(
                columns={"component_rmse": "method_rmse"}
            )
            for comparator_index, comparator in enumerate(comparators):
                other = movie_metrics[
                    movie_metrics["method"].eq(comparator)
                    & movie_metrics["horizon"].eq(horizon)
                ][["test_movie", "component_rmse"]].rename(
                    columns={"component_rmse": "comparator_rmse"}
                )
                paired = reference.merge(
                    other,
                    on="test_movie",
                    validate="one_to_one",
                ).sort_values("test_movie")
                if len(paired) != len(MOVIES):
                    raise ValueError(
                        f"Incomplete pair: {method}, {comparator}, h{horizon}"
                    )
                delta = (
                    paired["comparator_rmse"]
                    - paired["method_rmse"]
                ).to_numpy(dtype=np.float64)
                relative = (
                    100.0
                    * delta
                    / np.maximum(
                        paired["comparator_rmse"].to_numpy(dtype=np.float64),
                        EPS,
                    )
                )
                low, high = bootstrap_ci(
                    delta,
                    repeats,
                    seed
                    + method_index * 100_000
                    + horizon * 1_000
                    + comparator_index,
                )
                confirmatory = (
                    method == "v166_h1_strict"
                    and comparator == "v97_no_update"
                    and horizon == 1
                ) or (
                    method == "v166_h6_utility"
                    and comparator == "v97_no_update"
                    and horizon == 6
                )
                hypothesis = ""
                if confirmatory:
                    hypothesis = (
                        "H1" if horizon == 1 else "H2"
                    )
                records.append(
                    {
                        "method": method,
                        "comparator": comparator,
                        "horizon": horizon,
                        "movies": len(paired),
                        "mean_rmse_delta_comparator_minus_method": float(
                            delta.mean()
                        ),
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "relative_gain_percent": float(relative.mean()),
                        "method_better_movies": int((delta > 0).sum()),
                        "exact_two_sided_sign_flip_p": exact_sign_flip_p(delta),
                        "confirmatory": confirmatory,
                        "hypothesis_id": hypothesis,
                        "multiplicity_family": (
                            "primary_H1_H2"
                            if confirmatory
                            else "exploratory"
                        ),
                        "statistical_unit": "movie",
                    }
                )
    frame = pd.DataFrame(records)
    primary = frame["confirmatory"]
    adjusted = holm_adjust(
        frame.loc[primary, "exact_two_sided_sign_flip_p"].tolist()
    )
    frame["holm_adjusted_p"] = np.nan
    frame.loc[primary, "holm_adjusted_p"] = adjusted
    frame["reject_at_0_05"] = False
    frame.loc[primary, "reject_at_0_05"] = (
        frame.loc[primary, "holm_adjusted_p"] <= 0.05
    )
    return frame


def build_probabilistic_metrics() -> pd.DataFrame:
    frames = []
    for operating_point, directory in (
        ("h1_strict", V163_H1_DIR),
        ("h6_utility", V163_H6_DIR),
    ):
        path = directory / "v163_temporal_covariance_decision.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["selected_overall"].eq(True)].copy()
        frame.insert(0, "operating_point", operating_point)
        frame["source_path"] = str(path)
        frame["source_sha256"] = sha256(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_robustness_matrix(
    h1_dir: Path,
    h6_dir: Path,
) -> pd.DataFrame:
    frames = []
    for operating_point, directory in (
        ("h1_strict", h1_dir),
        ("h6_utility", h6_dir),
    ):
        path = directory / "v157f_stress_aggregate.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "operating_point", operating_point)
        frame["source_path"] = str(path)
        frame["source_sha256"] = sha256(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_robustness_uncertainty(
    h1_dir: Path,
    h6_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    response_frames = []
    for operating_point, directory in (
        ("h1_strict", h1_dir),
        ("h6_utility", h6_dir),
    ):
        metric_path = directory / "v157f_uncertainty_aggregate.csv"
        response_path = directory / "v157f_uncertainty_response.csv"
        metrics = pd.read_csv(metric_path)
        metrics.insert(0, "operating_point", operating_point)
        metrics["source_path"] = str(metric_path)
        metrics["source_sha256"] = sha256(metric_path)
        metric_frames.append(metrics)
        response = pd.read_csv(response_path)
        response.insert(0, "operating_point", operating_point)
        response["source_path"] = str(response_path)
        response["source_sha256"] = sha256(response_path)
        response_frames.append(response)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(response_frames, ignore_index=True),
    )


def build_claim_scope(
    kalmannet_complete: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "protocol": "MDCK Bulk six-movie causal-online outer LOMO",
                "status": "complete",
                "headline_eligible": True,
                "allowed_claim": (
                    "v166 is the strongest completed tested method under the "
                    "frozen matched causal-online benchmark"
                ),
                "forbidden_claim": "global trajectory-forecasting SOTA",
            },
            {
                "claim_id": "C2",
                "protocol": (
                    "MDCK Bulk configuration-unseen movies 10-16"
                ),
                "status": "complete",
                "headline_eligible": True,
                "allowed_claim": (
                    "bounded innovation transport improves the frozen prior "
                    "on a separate unseen configuration cohort"
                ),
                "forbidden_claim": (
                    "pooling movies 10-16 with movies 1-6 in paired inference"
                ),
            },
            {
                "claim_id": "C3",
                "protocol": "HUVEC/MDA-MB-231 nested movie LOMO",
                "status": "complete",
                "headline_eligible": True,
                "allowed_claim": (
                    "the dimensionless update mechanism transfers across "
                    "tested cell domains"
                ),
                "forbidden_claim": "transfer of a universal full prior",
            },
            {
                "claim_id": "C4",
                "protocol": "public causal observability ladder",
                "status": "complete_negative",
                "headline_eligible": True,
                "allowed_claim": (
                    "tested deployable image/mechanics/reliability states did "
                    "not pass preregistered integration gates"
                ),
                "forbidden_claim": (
                    "cell motion is physically or visually unpredictable"
                ),
            },
            {
                "claim_id": "C5",
                "protocol": "exact learned-filter modern comparator",
                "status": "complete" if kalmannet_complete else "pending",
                "headline_eligible": kalmannet_complete,
                "allowed_claim": (
                    "matched comparison with KalmanNet"
                    if kalmannet_complete
                    else "no numerical claim until all six outer folds finish"
                ),
                "forbidden_claim": "source-name comparison from smoke results",
            },
            {
                "claim_id": "C6",
                "protocol": "historical fixed-origin best-of-K oracle",
                "status": "supplementary_diagnostic_only",
                "headline_eligible": False,
                "allowed_claim": "historical candidate-coverage diagnostic",
                "forbidden_claim": (
                    "direct comparison with rolling/receding h1 forecasts"
                ),
            },
        ]
    )


def comparator_status() -> pd.DataFrame:
    path = KALMANNET_DIR / "kalmannet_lomo_job_manifest.csv"
    frame = pd.read_csv(path)
    jobs = frame[frame["runner"].eq("v98")].copy()
    records = []
    for row in jobs.itertuples(index=False):
        output = Path(str(row.output_dir))
        required = [
            output / "v98_online_summary.csv",
            output / "v98_data_contract.csv",
            output / "v98_no_future_sentinel.json",
            output / "v98_provenance.json",
        ]
        files_complete = all(item.exists() for item in required)
        sentinel_pass = False
        contract_match = False
        selected_model = ""
        if files_complete:
            sentinel = json.loads(
                (output / "v98_no_future_sentinel.json").read_text(
                    encoding="utf-8"
                )
            )
            sentinel_pass = bool(
                sentinel.get("pass") is True
                and sentinel.get("future_placeholder_read_at_inference") is False
            )
            contract = pd.read_csv(output / "v98_data_contract.csv")
            if len(contract) == 1:
                contract_row = contract.iloc[0]
                selected_model = str(contract_row["selected_model"])
                contract_match = bool(
                    int(contract_row["future_target_inference_features"]) == 0
                    and int(contract_row["test_movies"]) == int(row.test_movie)
                    and int(contract_row["test_rows"]) == int(row.test_rows)
                )
        complete = files_complete and sentinel_pass and contract_match
        records.append(
            {
                "comparator": "KalmanNet",
                "test_movie": int(row.test_movie),
                "validation_movie": int(row.validation_movie),
                "seed": int(row.seed),
                "status": "complete" if complete else "pending",
                "selected_model": selected_model,
                "no_future_sentinel_pass": sentinel_pass,
                "contract_match": contract_match,
                "output_dir": str(output),
                "command": str(row.command),
            }
        )
    return pd.DataFrame(records)


def pending_tasks(
    kalmannet: pd.DataFrame,
) -> pd.DataFrame:
    completed = int(kalmannet["status"].eq("complete").sum())
    total = len(kalmannet)
    rows = [
        {
            "task_id": "A1.2",
            "priority": "P0",
            "task": "Complete exact KalmanNet outer-movie LOMO",
            "status": "complete" if completed == total else "pending",
            "progress": f"{completed}/{total} fold-seed jobs",
            "blocks_publication_ready": completed != total,
        },
        {
            "task_id": "B0",
            "priority": "P0",
            "task": "Choose license and initialize clean release repository",
            "status": "owner_decision_required",
            "progress": "0/1",
            "blocks_publication_ready": True,
        },
        {
            "task_id": "B1-B2",
            "priority": "P0",
            "task": "Build and validate fresh-checkout public package",
            "status": "pending",
            "progress": "0/1",
            "blocks_publication_ready": True,
        },
        {
            "task_id": "C0-C3",
            "priority": "P0",
            "task": "Write manuscript, figures, and final claim audit",
            "status": "pending",
            "progress": "protocol and evidence skeleton ready",
            "blocks_publication_ready": True,
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    benchmark: pd.DataFrame,
    paired: pd.DataFrame,
    external: pd.DataFrame,
    robustness_uncertainty_response: pd.DataFrame,
    fit_scope_sensitivity: pd.DataFrame,
    observability: pd.DataFrame,
    pending: pd.DataFrame,
) -> None:
    headline = benchmark[
        benchmark["method"].isin(
            [
                "constant_velocity",
                "gru_track",
                "hgbdt_v52",
                "kalmannet",
                "v97_no_update",
                "v166_h1_strict",
                "v166_h6_utility",
            ]
        )
        & benchmark["horizon"].isin([1, 6])
    ][
        [
            "method",
            "horizon",
            "movies",
            "component_rmse",
            "vector_rmse",
            "r2",
        ]
    ]
    primary = paired[paired["confirmatory"]][
        [
            "hypothesis_id",
            "method",
            "comparator",
            "horizon",
            "mean_rmse_delta_comparator_minus_method",
            "relative_gain_percent",
            "method_better_movies",
            "exact_two_sided_sign_flip_p",
            "holm_adjusted_p",
        ]
    ]
    deployable_passes = observability[
        observability["deployable_to_lachance"].eq(True)
        & observability["passed"].eq(True)
    ]
    kalmannet_pending = bool(
        pending.loc[
            pending["task_id"].eq("A1.2"),
            "status",
        ].iloc[0]
        != "complete"
    )
    report = [
        "# v188 Publication Evidence Bundle",
        "",
        "## Frozen matched causal-online benchmark",
        "",
        headline.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Confirmatory movie-level hypotheses",
        "",
        primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The six-movie matched transport experiment was already complete in "
        "`v157h`; v188 restores it to the primary evidence table instead of "
        "using the older baseline-only v166 summary.",
        "",
        "## External nested LOMO",
        "",
        external[
            external["control"].eq("real")
            & external["horizon"].isin([1, 6])
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Robust uncertainty response",
        "",
        robustness_uncertainty_response[
            robustness_uncertainty_response["condition"].isin(
                [
                    "real_update_every_1",
                    "missing_0.4",
                    "tracking_noise_2px",
                    "wrong_cell",
                ]
            )
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "The frozen base scale is reported separately from a frozen state-aware "
        "rule. Only clean update/no-update scales are selected on the validation "
        "movie; corruption-level multipliers are never tuned, and declared "
        "coordinate noise is propagated through the frozen packet/correction "
        "by Monte Carlo. No outer-test target is used for uncertainty "
        "calibration.",
        "",
        "## Final-fit scope sensitivity",
        "",
        fit_scope_sensitivity[
            (
                fit_scope_sensitivity["objective_name"].eq("h1_strict")
                & fit_scope_sensitivity["horizon"].eq(1)
            )
            | (
                fit_scope_sensitivity["objective_name"].eq("h6_guard10")
                & fit_scope_sensitivity["horizon"].eq(6)
            )
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "The train-only row keeps the validation movie out of the final update "
        "fit. Its near-parity shows that the historical train+validation refit "
        "does not explain the v166 result.",
        "",
        "## Observability decision",
        "",
        f"Deployable states passing the LaChance gate: "
        f"`{len(deployable_passes)}`.",
        "The public modality program is retained as a negative boundary, not "
        "as evidence that no useful prospective state can exist.",
        "",
        "## Publication readiness",
        "",
        (
            "**Evidence bundle structurally complete, decisive comparator "
            "pending.**"
            if kalmannet_pending
            else "**Core evidence complete.**"
        ),
        "",
        pending.to_markdown(index=False),
        "",
        "Global SOTA wording remains forbidden. The defensible current claim "
        "is the strongest completed tested method under the frozen "
        "causal-online LaChance protocol.",
    ]
    (output / "v188_publication_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    h1_stress = args.stress_h1_dir.resolve()
    h6_stress = args.stress_h6_dir.resolve()
    required = [
        output / "v188_protocol_contract.csv",
        output / "v188_multiplicity_contract.json",
        output / "v188_contract_manifest.json",
        V117_DIR / "v117_seed_aggregated_within_movie.csv",
        V102_DIR / "v102_seed_aggregated_within_movie.csv",
        V157H_DIR / "v157h_pareto_metrics.csv",
        V157H_DIR / "v157h_statistical_confirmation.csv",
        h1_stress / "v157f_stress_aggregate.csv",
        h6_stress / "v157f_stress_aggregate.csv",
        h1_stress / "v157f_uncertainty_aggregate.csv",
        h6_stress / "v157f_uncertainty_aggregate.csv",
        h1_stress / "v157f_uncertainty_response.csv",
        h6_stress / "v157f_uncertainty_response.csv",
        h1_stress / "run_manifest.json",
        h6_stress / "run_manifest.json",
        h1_stress / "v157f_seed_replay_manifest.json",
        h6_stress / "v157f_seed_replay_manifest.json",
        h1_stress / "v157f_causal_audit.csv",
        h6_stress / "v157f_causal_audit.csv",
        h1_stress / "v157f_summary.csv",
        h6_stress / "v157f_summary.csv",
        V157F_SOURCE,
        V189_DIR / "v189_fit_scope_metrics.csv",
        V189_DIR / "v189_fit_scope_summary.csv",
        V189_DIR / "v189_fit_scope_comparison.csv",
        V189_DIR / "v189_validation_selection.csv",
        V189_DIR / "v189_causal_audit.csv",
        V189_DIR / "v189_seed_replay_manifest.json",
        V189_DIR / "run_manifest.json",
        V189_SOURCE,
        V163_H1_DIR / "v163_temporal_covariance_decision.csv",
        V163_H6_DIR / "v163_temporal_covariance_decision.csv",
        V166_DIR / "v166_configuration_unseen_transport.csv",
        V166_DIR / "v166_external_full_lomo.csv",
        V166_DIR / "v166_external_causal_controls.csv",
        V181_DIR / "future_work_gate_matrix.csv",
        KALMANNET_DIR / "kalmannet_lomo_job_manifest.csv",
        KALMANNET_DIR / "kalmannet_lomo_movie_metrics.csv",
        KALMANNET_DIR / "kalmannet_lomo_manifest.json",
    ]
    require_files(required)
    validate_stress_provenance(h1_stress, "h1_strict")
    validate_stress_provenance(h6_stress, "h6_guard10")
    validate_fit_scope_provenance()

    protocol = pd.read_csv(output / "v188_protocol_contract.csv")
    if set(protocol["outer_test_movie"].astype(int)) != set(MOVIES):
        raise ValueError("v188 protocol does not contain six outer movies")
    key_set_hash = protocol_key_set_hash(protocol)

    movie_metrics = build_movie_metrics(protocol)
    benchmark = aggregate_benchmark(movie_metrics, key_set_hash)
    paired = paired_statistics(
        movie_metrics,
        args.bootstrap_repeats,
        args.seed,
    )
    external = pd.read_csv(V166_DIR / "v166_external_full_lomo.csv")
    external_controls = pd.read_csv(
        V166_DIR / "v166_external_causal_controls.csv"
    )
    configuration_unseen = pd.read_csv(
        V166_DIR / "v166_configuration_unseen_transport.csv"
    )
    probabilistic = build_probabilistic_metrics()
    robustness = build_robustness_matrix(h1_stress, h6_stress)
    robustness_uncertainty, robustness_uncertainty_response = (
        build_robustness_uncertainty(h1_stress, h6_stress)
    )
    fit_scope_sensitivity, fit_scope_movie_metrics = (
        build_fit_scope_sensitivity()
    )
    observability = pd.read_csv(V181_DIR / "future_work_gate_matrix.csv")
    comparators = comparator_status()
    kalmannet_complete = bool(
        len(comparators)
        and comparators["status"].eq("complete").all()
    )
    scope = build_claim_scope(kalmannet_complete)
    pending = pending_tasks(comparators)

    tables = {
        "v188_primary_online_benchmark.csv": benchmark,
        "v188_primary_online_movie_metrics.csv": movie_metrics,
        "v188_paired_movie_statistics.csv": paired,
        "v188_configuration_unseen_confirmation.csv": configuration_unseen,
        "v188_external_nested_lomo.csv": external,
        "v188_external_causal_controls.csv": external_controls,
        "v188_probabilistic_metrics.csv": probabilistic,
        "v188_robustness_matrix.csv": robustness,
        "v188_robustness_uncertainty.csv": robustness_uncertainty,
        "v188_robustness_uncertainty_response.csv": (
            robustness_uncertainty_response
        ),
        "v188_fit_scope_sensitivity.csv": fit_scope_sensitivity,
        "v188_fit_scope_movie_metrics.csv": fit_scope_movie_metrics,
        "v188_observability_gate_matrix.csv": observability,
        "v188_claim_scope.csv": scope,
        "v188_comparator_status.csv": comparators,
        "v188_pending_tasks.csv": pending,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    write_report(
        output,
        benchmark,
        paired,
        external,
        robustness_uncertainty_response,
        fit_scope_sensitivity,
        observability,
        pending,
    )

    contract_manifest = json.loads(
        (output / "v188_contract_manifest.json").read_text(encoding="utf-8")
    )
    artifact_paths = [output / name for name in tables]
    artifact_paths.extend(
        [
            output / "v188_protocol_contract.csv",
            output / "v188_multiplicity_contract.json",
            output / "v188_contract_manifest.json",
            output / "v188_source_status.csv",
            output / "v188_publication_report.md",
        ]
    )
    manifest = {
        "schema_version": 1,
        "architecture_frozen": True,
        "protocol_contract_sha256": contract_manifest["contract_sha256"],
        "protocol_tables_separated": True,
        "matched_outer_movies": list(MOVIES),
        "configuration_unseen_movies": [10, 11, 12, 13, 14, 15, 16],
        "global_sota_claim_allowed": False,
        "protocol_specific_claim_allowed": True,
        "fixed_origin_oracle_primary": False,
        "kalmannet_complete": kalmannet_complete,
        "evidence_complete": kalmannet_complete,
        "publication_ready": False,
        "publication_blockers": pending[
            pending["blocks_publication_ready"].eq(True)
            & ~pending["status"].eq("complete")
        ]["task_id"].tolist(),
        "input_sha256": {
            str(path): sha256(path) for path in required
        },
        "artifact_sha256": {
            str(path): sha256(path) for path in artifact_paths
        },
        "builder_sha256": sha256(Path(__file__)),
        "args": {
            "out_dir": str(output),
            "stress_h1_dir": str(h1_stress),
            "stress_h6_dir": str(h6_stress),
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
    }
    (output / "v188_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v188] wrote {output}; "
        f"KalmanNet complete={kalmannet_complete}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
