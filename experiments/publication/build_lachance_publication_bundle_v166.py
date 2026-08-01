#!/usr/bin/env python3
"""Build a frozen, protocol-separated publication evidence bundle.

Rolling causal forecasts, fixed-origin trajectories, and turn classification
are intentionally emitted as separate tables. The script never constructs one
cross-protocol ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "outputs" / "lachance_publication_bundle_v166_2026-07-27"
)
PATHS = {
    "v160": ROOT
    / "outputs"
    / "lachance_streaming_transport_confirmation_v160_full_2026-07-27"
    / "v160_confirmation_metrics.csv",
    "v117": ROOT
    / "outputs"
    / "lachance_online_lomo_baselines_v117_production_2026-07-21"
    / "v117_seed_aggregated_within_movie.csv",
    "v102": ROOT
    / "outputs"
    / "lachance_online_lomo_benchmark_v102_v97_production_2026-07-21"
    / "v102_seed_aggregated_within_movie.csv",
    "v162": ROOT
    / "outputs"
    / "lachance_v162_external_seed_confirmation_full_2026-07-27"
    / "v162_3seed_aggregate.csv",
    "v163_h6": ROOT
    / "outputs"
    / "lachance_temporal_student_t_covariance_v163_full_2026-07-27"
    / "v163_temporal_covariance_decision.csv",
    "v163_h1": ROOT
    / "outputs"
    / "lachance_temporal_student_t_covariance_v163_h1strict_2026-07-27"
    / "v163_temporal_covariance_decision.csv",
    "v100": ROOT
    / "outputs"
    / "lachance_original_turning_v100_full_bulk_seed42_2026-07-21"
    / "v100_turn_metrics.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0
    return float(
        binomtest(int((values > 0).sum()), len(values), 0.5).pvalue
    )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate keys {columns}: {examples}")


def require_finite(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite numeric results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v165-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "lachance_external_movie_lomo_v165_publication_full_2026-07-27"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=166)
    return parser.parse_args()


def summarize_rows(
    frame: pd.DataFrame,
    method_column: str,
    movie_column: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, ((method, horizon), group) in enumerate(
        frame.groupby([method_column, "horizon"], sort=True)
    ):
        values = group.component_rmse.to_numpy(dtype=np.float64)
        low, high = bootstrap_ci(values, repeats, seed + index)
        records.append(
            {
                "method": method,
                "horizon": int(horizon),
                "movies": int(group[movie_column].nunique()),
                "component_rmse": float(group.component_rmse.mean()),
                "component_rmse_ci_low": low,
                "component_rmse_ci_high": high,
                "vector_rmse": float(group.vector_rmse.mean()),
                "r2": float(group.r2.mean()),
            }
        )
    return pd.DataFrame(records)


def paired_table(
    frame: pd.DataFrame,
    method_column: str,
    movie_column: str,
    reference: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    require_unique(
        frame,
        [movie_column, "horizon", method_column],
        "paired comparison input",
    )
    pivot = frame.pivot(
        index=[movie_column, "horizon"],
        columns=method_column,
        values="component_rmse",
    ).reset_index()
    records: list[dict[str, Any]] = []
    methods = [
        method
        for method in sorted(frame[method_column].unique())
        if method != reference
    ]
    for horizon in sorted(frame.horizon.unique()):
        horizon_frame = pivot[pivot.horizon.eq(horizon)]
        if reference not in horizon_frame:
            continue
        for method_index, method in enumerate(methods):
            if method not in horizon_frame:
                continue
            valid = horizon_frame[[reference, method]].dropna()
            if not len(valid):
                continue
            advantage = (
                100.0
                * (valid[method] - valid[reference])
                / np.maximum(valid[method].abs(), 1e-12)
            ).to_numpy(dtype=np.float64)
            low, high = bootstrap_ci(
                advantage,
                repeats,
                seed + int(horizon) * 100 + method_index,
            )
            records.append(
                {
                    "reference": reference,
                    "comparator": method,
                    "horizon": int(horizon),
                    "movies": len(valid),
                    "reference_gain_percent": float(advantage.mean()),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                    "reference_better_movies": int((advantage > 0).sum()),
                    "sign_test_p_two_sided": sign_p(advantage),
                }
            )
    return pd.DataFrame(records)


def build_v160(
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(PATHS["v160"])
    require_columns(
        frame,
        [
            "objective_name",
            "packet_name",
            "control",
            "horizon",
            "test_movie",
            "component_rmse",
            "vector_rmse",
            "r2",
            "rmse_improvement_percent",
        ],
        "v160",
    )
    controls = [
        "real",
        "v97_no_update",
        "constant_velocity",
        "kalman_cv",
        "kalman_ca",
        "imm_cv_ca_turn",
        "stale_time",
        "wrong_cell",
    ]
    # v160 contains four transport packets. The frozen publication operating
    # point is the preselected full packet; averaging packets would silently
    # mix ablations into the headline estimate.
    frame = frame[
        frame.control.isin(controls) & frame.packet_name.eq("full")
    ].copy()
    require_unique(
        frame,
        ["objective_name", "control", "horizon", "test_movie"],
        "v160 frozen full packet",
    )
    require_finite(
        frame,
        ["component_rmse", "vector_rmse", "r2"],
        "v160 frozen full packet",
    )
    summary = (
        frame.groupby(["objective_name", "control", "horizon"], as_index=False)
        .agg(
            movies=("test_movie", "nunique"),
            component_rmse=("component_rmse", "mean"),
            vector_rmse=("vector_rmse", "mean"),
            r2=("r2", "mean"),
            gain_vs_v97_percent=("rmse_improvement_percent", "mean"),
        )
    )
    comparisons: list[pd.DataFrame] = []
    for objective_index, (objective, group) in enumerate(
        frame.groupby("objective_name", sort=True)
    ):
        table = paired_table(
            group,
            "control",
            "test_movie",
            "real",
            repeats,
            seed + objective_index * 10_000,
        )
        table.insert(0, "objective", objective)
        comparisons.append(table)
    return summary, pd.concat(comparisons, ignore_index=True)


def build_bulk_lomo(
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(PATHS["v117"])
    require_columns(
        baseline,
        [
            "test_movie",
            "method",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
        ],
        "v117",
    )
    baseline = baseline[
        [
            "test_movie",
            "method",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
        ]
    ].copy()
    v102 = pd.read_csv(PATHS["v102"])
    require_columns(
        v102,
        [
            "test_movie",
            "method_id",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
        ],
        "v102",
    )
    mapping = {
        "v97/v97_direct": "v97",
        "baseline/v52_rolling": "v52_rolling",
    }
    v102 = v102[v102.method_id.isin(mapping)].copy()
    v102["method"] = v102.method_id.map(mapping)
    v102 = v102[
        [
            "test_movie",
            "method",
            "horizon",
            "component_rmse",
            "vector_rmse",
            "r2",
        ]
    ]
    combined = pd.concat([baseline, v102], ignore_index=True)
    duplicates = combined.duplicated(
        ["test_movie", "method", "horizon"],
        keep=False,
    )
    if duplicates.any():
        raise RuntimeError("Duplicate Bulk LOMO method/movie/horizon rows")
    require_finite(
        combined,
        ["component_rmse", "vector_rmse", "r2"],
        "Bulk LOMO baselines",
    )
    summary = summarize_rows(
        combined,
        "method",
        "test_movie",
        repeats,
        seed,
    )
    paired = paired_table(
        combined,
        "method",
        "test_movie",
        "v97",
        repeats,
        seed + 20_000,
    )
    return summary, paired


def build_transfer() -> pd.DataFrame:
    frame = pd.read_csv(PATHS["v162"])
    return frame[
        frame.objective.eq("h6_guard10")
        & frame.variant.eq("lodo_zero_shot")
        & frame.control.eq("real")
        & frame.horizon.isin([1, 6])
    ].copy()


def build_temporal() -> pd.DataFrame:
    frames = []
    for operating_point, key in (
        ("h6_utility", "v163_h6"),
        ("h1_strict", "v163_h1"),
    ):
        frame = pd.read_csv(PATHS[key])
        frame = frame[
            frame.protocol.eq("lodo_zero_shot")
            & frame.selected_overall.eq(True)
            & frame.horizon.eq(6)
        ].copy()
        frame.insert(0, "operating_point", operating_point)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_turning() -> pd.DataFrame:
    frame = pd.read_csv(PATHS["v100"])
    return frame[
        frame.method.isin(
            [
                "majority_train",
                "original_asocial",
                "original_pairwise_attention",
                "original_pairwise_attention_shuffled_social",
                "v97_h1_direction",
            ]
        )
        & frame.scope.isin(
            [
                "all_turns",
                "large_turns_20_160",
                "v97_matched_all_turns",
                "v97_matched_large_turns_20_160",
            ]
        )
    ].copy()


def claim_scope() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol": "causal rolling h1, Bulk six-movie LOMO",
                "status": "complete",
                "headline_eligible": True,
                "allowed_claim": (
                    "strong matched causal-online baseline comparison"
                ),
                "forbidden_claim": "global trajectory SOTA",
            },
            {
                "protocol": "configuration-unseen rolling transport, movies 10-16",
                "status": "complete",
                "headline_eligible": True,
                "allowed_claim": (
                    "bounded innovation transport gain over frozen prior and "
                    "classical online filters"
                ),
                "forbidden_claim": "same cohort as six-movie neural LOMO",
            },
            {
                "protocol": "external HUVEC/MDA all-movie nested LOMO",
                "status": "complete",
                "headline_eligible": True,
                "allowed_claim": "external movie-level mechanism transfer",
                "forbidden_claim": "zero-shot transfer of the complete prior",
            },
            {
                "protocol": "LaChance future-turn classification",
                "status": "complete single split",
                "headline_eligible": False,
                "allowed_claim": "separate domain-positioning experiment",
                "forbidden_claim": "coordinate RMSE superiority",
            },
            {
                "protocol": "fixed-origin 8-to-6 AgentFormer/DLow/SGAN",
                "status": "official-source smoke; production incomplete",
                "headline_eligible": False,
                "allowed_claim": "architecture/source-fidelity appendix",
                "forbidden_claim": "numeric SOTA superiority",
            },
            {
                "protocol": "MTR/QCNet/QCNeXt native map-rich task",
                "status": "not protocol-equivalent",
                "headline_eligible": False,
                "allowed_claim": "related work only",
                "forbidden_claim": "native leaderboard comparison",
            },
        ]
    )


def write_report(
    output: Path,
    v160_summary: pd.DataFrame,
    bulk_summary: pd.DataFrame,
    transfer: pd.DataFrame,
    external: pd.DataFrame,
    external_controls: pd.DataFrame,
    temporal: pd.DataFrame,
    turning: pd.DataFrame,
    scope: pd.DataFrame,
) -> None:
    v160_primary = v160_summary[
        v160_summary.control.eq("real")
        & v160_summary.horizon.isin([1, 6])
    ]
    bulk_primary = bulk_summary[bulk_summary.horizon.isin([1, 6])]
    external_primary = external[
        external.control.eq("real")
        & external.horizon.isin([1, 6])
    ]
    transfer_primary = transfer[transfer.horizon.eq(6)]
    temporal_primary = temporal[
        [
            "operating_point",
            "dataset",
            "family_selected",
            "component_rmse",
            "nll_gain",
            "energy_gain_percent",
            "strong_temporal_gate",
        ]
    ]
    turning_primary = turning[
        [
            "method",
            "scope",
            "n_rows",
            "balanced_accuracy",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
        ]
    ]
    report = [
        "# v166 Frozen Publication Evidence Bundle",
        "",
        "## Configuration-unseen rolling transport",
        "",
        v160_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Six-movie matched Bulk LOMO",
        "",
        bulk_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Three-seed dimensionless transfer",
        "",
        transfer_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full external nested LOMO",
        "",
        external_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## External causal controls",
        "",
        external_controls[
            external_controls.horizon.eq(6)
            & external_controls.objective.eq("h6_guard10")
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Temporal covariance diagnostic",
        "",
        temporal_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Separate turn-classification context",
        "",
        turning_primary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Claim boundary",
        "",
        scope.to_markdown(index=False),
        "",
        "## Decision",
        "",
        "The evidence supports a protocol-specific causal-online benchmark and "
        "a strong bounded innovation-transport method. It does not support a "
        "global trajectory-forecasting SOTA claim. Rolling, fixed-origin, and "
        "turn-classification tables must remain separate.",
    ]
    (output / "v166_publication_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v165_dir = args.v165_dir.resolve()
    v165_summary_path = v165_dir / "v165_publication_summary.csv"
    v165_manifest_path = v165_dir / "v165_publication_manifest.json"
    v165_controls_path = v165_dir / "v165_causal_control_statistics.csv"
    v165_movie_stats_path = v165_dir / "v165_movie_level_statistics.csv"
    required = list(PATHS.values()) + [
        v165_summary_path,
        v165_controls_path,
        v165_movie_stats_path,
        v165_manifest_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing publication inputs: " + ", ".join(missing))

    v160_summary, v160_paired = build_v160(
        args.bootstrap_repeats,
        args.seed,
    )
    bulk_summary, bulk_paired = build_bulk_lomo(
        args.bootstrap_repeats,
        args.seed + 30_000,
    )
    transfer = build_transfer()
    temporal = build_temporal()
    turning = build_turning()
    external = pd.read_csv(v165_summary_path)
    external_controls = pd.read_csv(v165_controls_path)
    external_movie_stats = pd.read_csv(v165_movie_stats_path)
    external_manifest = json.loads(v165_manifest_path.read_text(encoding="utf-8"))
    expected_folds = {"HUVEC": 18, "MDAMB231": 17}
    if external_manifest.get("folds_expected") != sum(expected_folds.values()):
        raise ValueError("v165 manifest does not describe all 35 external folds")
    if external_manifest.get("target_leakage") is not False:
        raise ValueError("v165 manifest does not certify target_leakage=false")
    for dataset, folds in expected_folds.items():
        subset = external[
            external.dataset.eq(dataset) & external.control.eq("real")
        ]
        if set(subset.outer_folds.astype(int)) != {folds}:
            raise ValueError(
                f"v165 {dataset} does not contain exactly {folds} outer folds"
            )
    require_finite(
        external,
        ["component_rmse_macro", "r2_macro", "gain_percent_macro"],
        "v165 publication summary",
    )
    require_finite(
        external_controls,
        ["rmse_advantage_macro", "sign_test_p_two_sided"],
        "v165 causal controls",
    )
    require_unique(
        external_movie_stats,
        ["dataset", "objective", "control", "horizon"],
        "v165 movie-level statistics",
    )
    scope = claim_scope()

    tables = {
        "v166_configuration_unseen_transport.csv": v160_summary,
        "v166_configuration_unseen_paired.csv": v160_paired,
        "v166_bulk_lomo_baselines.csv": bulk_summary,
        "v166_bulk_lomo_paired.csv": bulk_paired,
        "v166_dimensionless_transfer.csv": transfer,
        "v166_temporal_covariance.csv": temporal,
        "v166_external_full_lomo.csv": external,
        "v166_external_causal_controls.csv": external_controls,
        "v166_turning_context.csv": turning,
        "v166_claim_scope.csv": scope,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    write_report(
        output,
        v160_summary,
        bulk_summary,
        transfer,
        external,
        external_controls,
        temporal,
        turning,
        scope,
    )

    artifact_paths = [output / name for name in tables]
    artifact_paths.append(output / "v166_publication_report.md")
    input_paths = required
    decision = {
        "architecture_frozen": True,
        "protocol_specific_sota_candidate": True,
        "global_sota": False,
        "reason": (
            "The method is the strongest tested model under the frozen "
            "causal-online protocol, but no published benchmark matches its "
            "predict-before-observe clock, movie split, rolling composition, "
            "and metric convention."
        ),
    }
    decision_path = output / "v166_sota_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(decision_path)
    manifest = {
        "schema_version": 1,
        "protocol_tables_separated": True,
        "global_sota_claim_allowed": False,
        "protocol_specific_claim_allowed": True,
        "input_sha256": {
            str(path): sha256(path) for path in input_paths
        },
        "artifact_sha256": {
            str(path): sha256(path) for path in artifact_paths
        },
        "builder_sha256": sha256(Path(__file__)),
        "args": {
            "v165_dir": str(v165_dir),
            "out_dir": str(output),
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
    }
    (output / "v166_publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[v166] wrote {output}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
