#!/usr/bin/env python3
"""External fixed-split guards for the semigroup innovation transport.

The runner restores existing v97 checkpoints for MDCK_Edge, MDA-MB-231, and
HUVEC, regenerates train/validation/test predictions, derives neighbour scales
from train-only nearest-neighbour distances, and fits the update-only semigroup
correction.  Validation and test sequences remain separate statistical units;
they are never concatenated into one artificial track namespace.

These are one-seed fixed-split transfer guards, not leave-one-movie-out
publication estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import norm, t as student_t

import run_lachance_causal_innovation_state_space_v97 as v97
import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h
import run_lachance_joint_graph_copula_v154 as v154


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lachance_semigroup_external_guards_v157i"
DEFAULT_SPECS = (
    "MDCK_Edge="
    "outputs/causal_innovation_state_space_v97_direct_edge_guard_cpu_seed42_2026-07-21/v97_direct.pt,"
    "MDAMB231="
    "outputs/causal_innovation_state_space_v97_direct_mdamb231_guard_seed42_2026-07-21/v97_direct.pt,"
    "HUVEC="
    "outputs/causal_innovation_state_space_v97_huvec_guard_seed42_2026-07-21/v97_no_context.pt"
)
EPS = 1e-8


@dataclass
class RestoredRun:
    dataset: str
    checkpoint: Path
    variant: str
    payloads: dict[int, tuple[str, v154.MoviePayload]]
    manifest: dict[str, Any]


@dataclass
class Selection:
    alpha: float
    bound_px: float
    validation_score: float
    validation_h1_gain: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default=DEFAULT_SPECS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--alphas",
        default="1,10,30,100,300,1000,3000,10000",
    )
    parser.add_argument("--bounds-px", default="0.25,0.5,1,1.5,2,3,4,6")
    parser.add_argument("--scale-multipliers", default="1,2,4,8")
    parser.add_argument("--max-scale-frames", type=int, default=300)
    parser.add_argument(
        "--objective",
        choices=sorted(v157h.OBJECTIVES),
        default="h1_strict",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=157_009)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_specs(value: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for token in value.split(","):
        name, raw_path = token.strip().split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        output[name] = path.resolve()
    return output


def restore_run(
    dataset: str,
    checkpoint_path: Path,
    device: torch.device,
) -> RestoredRun:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    variant = v97.TrainVariant(**checkpoint["variant"])
    args = v157e.checkpoint_namespace(checkpoint, str(device))
    prep = v97.load_prepared(args, variant)
    metadata = checkpoint["metadata"]
    static_dim = int(prep.static[0].shape[1])
    model = v97.CausalInnovationStateSpaceForecaster(
        static_dim=static_dim,
        hidden=int(args.hidden),
        history_lags=int(args.history_lags),
        correction_bound=float(args.correction_bound),
        dropout=float(args.dropout),
        use_update=bool(variant.use_update),
        use_graph=bool(variant.use_graph),
        graph_heads=int(args.graph_heads),
        output_mode=str(variant.output_mode),
        target_mean=np.asarray(metadata["target_mean"], dtype=np.float32),
        target_scale=np.asarray(metadata["target_scale"], dtype=np.float32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    eta = float(metadata["eta"])
    replay = [
        v97.replay_inference(
            model,
            prep,
            split_index,
            device,
            eta=eta,
            control="real",
            seed=int(args.seed),
        )
        for split_index in range(3)
    ]
    factor = v97.calibrate_uncertainty(
        prep.bundles[1],
        replay[1].prediction,
        replay[1].scale,
        float(metadata["degrees_of_freedom"]),
        v97.parse_floats(args.uncertainty_scale_grid),
    )
    payloads: dict[int, tuple[str, v154.MoviePayload]] = {}
    split_rows: dict[str, Any] = {}
    for split_index, bundle in enumerate(prep.bundles):
        split = v157e.split_name(split_index)
        rows = bundle.rows.reset_index(drop=True)
        target = np.asarray(bundle.target_steps[:, 0], dtype=np.float64)
        prediction = np.asarray(
            replay[split_index].prediction,
            dtype=np.float64,
        )
        scale = np.maximum(
            np.asarray(replay[split_index].scale, dtype=np.float64)
            * factor,
            1e-4,
        )
        split_rows[split] = {
            "rows": len(rows),
            "sequences": sorted(
                int(value) for value in rows["sequence"].unique()
            ),
        }
        for sequence in split_rows[split]["sequences"]:
            raw_indices = np.flatnonzero(
                rows["sequence"].to_numpy(np.int64) == sequence
            )
            order = np.lexsort(
                (
                    rows.iloc[raw_indices]["track_id"].to_numpy(np.int64),
                    rows.iloc[raw_indices]["frame"].to_numpy(np.int64),
                )
            )
            indices = raw_indices[order]
            selected_rows = rows.iloc[indices].reset_index(drop=True)
            selected_target = target[indices]
            selected_prediction = prediction[indices]
            selected_scale = scale[indices]
            standardized = (
                selected_target - selected_prediction
            ) / selected_scale
            uniform = np.clip(
                student_t.cdf(
                    standardized,
                    df=float(metadata["degrees_of_freedom"]),
                ),
                1e-6,
                1.0 - 1e-6,
            )
            if sequence in payloads:
                raise RuntimeError(
                    f"Sequence {sequence} occurs in multiple splits"
                )
            payloads[sequence] = (
                split,
                v154.MoviePayload(
                    movie=sequence,
                    rows=selected_rows,
                    target=selected_target,
                    mean=selected_prediction,
                    scale=selected_scale,
                    degrees_of_freedom=float(
                        metadata["degrees_of_freedom"]
                    ),
                    normal_score=np.asarray(
                        norm.ppf(uniform),
                        dtype=np.float64,
                    ),
                ),
            )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return RestoredRun(
        dataset=dataset,
        checkpoint=checkpoint_path,
        variant=variant.name,
        payloads=payloads,
        manifest={
            "dataset": dataset,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "variant": variant.name,
            "device": str(device),
            "uncertainty_factor": float(factor),
            "splits": split_rows,
        },
    )


def train_neighbour_scale(
    payloads: dict[int, tuple[str, v154.MoviePayload]],
    max_frames: int,
) -> float:
    distances: list[np.ndarray] = []
    for _, (split, payload) in sorted(payloads.items()):
        if split != "train":
            continue
        frames = np.asarray(
            sorted(payload.rows["frame"].unique()),
            dtype=np.int64,
        )
        if len(frames) > max_frames:
            indices = np.linspace(
                0,
                len(frames) - 1,
                max_frames,
            ).round().astype(np.int64)
            frames = frames[indices]
        for frame in frames:
            positions = payload.rows.loc[
                payload.rows["frame"].eq(frame),
                ["x_px", "y_px"],
            ].to_numpy(np.float64)
            if len(positions) < 2:
                continue
            nearest = cKDTree(positions).query(positions, k=2)[0][:, 1]
            distances.append(nearest[np.isfinite(nearest)])
    if not distances:
        raise RuntimeError("Cannot estimate train neighbour scale")
    return float(np.median(np.concatenate(distances)))


def fit_model(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    alpha: float,
    weights_by_horizon: dict[int, float],
) -> v157e.WeightedRidge:
    mean, scale = v157e.row_normalization(payloads, movies)
    features, target, weights = v157h.training_data(
        payloads,
        movies,
        mean,
        scale,
        weights_by_horizon,
    )
    return v157e.fit_weighted_ridge(
        features,
        target,
        weights,
        alpha,
        mean,
        scale,
    )


def predict(
    model: v157e.WeightedRidge,
    payload: v157e.UpdatePayload,
    control: str,
    bound_px: float,
) -> np.ndarray:
    raw = v157e.predict_ridge(model, payload, control)
    return payload.base.mean + v157e.bounded_update(raw, bound_px)


def select_model(
    payloads: dict[int, v157e.UpdatePayload],
    train_movies: list[int],
    validation_movies: list[int],
    alphas: list[float],
    bounds: list[float],
    weights_by_horizon: dict[int, float],
    h1_guard_percent: float,
) -> tuple[Selection, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_model(
            payloads,
            train_movies,
            alpha,
            weights_by_horizon,
        )
        for bound in bounds:
            movie_metrics = [
                v157e.metric_rows(
                    payloads[movie],
                    predict(
                        model,
                        payloads[movie],
                        "real",
                        bound,
                    ),
                    "validation_real",
                    None,
                )
                for movie in validation_movies
            ]
            score = float(
                np.mean(
                    [
                        v157h.score_metrics(
                            metrics,
                            weights_by_horizon,
                        )
                        for metrics in movie_metrics
                    ]
                )
            )
            h1_gain = float(
                np.mean(
                    [
                        next(
                            row["rmse_improvement_percent"]
                            for row in metrics
                            if int(row["horizon"]) == 1
                        )
                        for metrics in movie_metrics
                    ]
                )
            )
            records.append(
                {
                    "alpha": float(alpha),
                    "bound_px": float(bound),
                    "validation_score": score,
                    "validation_h1_gain_percent": h1_gain,
                }
            )
    grid = pd.DataFrame(records)
    eligible = grid[
        grid["validation_h1_gain_percent"].ge(
            -float(h1_guard_percent)
        )
    ]
    if eligible.empty:
        eligible = grid
    best = eligible.sort_values(
        [
            "validation_score",
            "validation_h1_gain_percent",
            "bound_px",
            "alpha",
        ],
        ascending=[True, False, True, True],
    ).iloc[0]
    return (
        Selection(
            alpha=float(best["alpha"]),
            bound_px=float(best["bound_px"]),
            validation_score=float(best["validation_score"]),
            validation_h1_gain=float(
                best["validation_h1_gain_percent"]
            ),
        ),
        grid,
    )


def evaluate_dataset(
    args: argparse.Namespace,
    restored: RestoredRun,
    alphas: list[float],
    bounds: list[float],
    scale_multipliers: list[float],
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    base_scale = train_neighbour_scale(
        restored.payloads,
        int(args.max_scale_frames),
    )
    local_scales = [
        base_scale * multiplier for multiplier in scale_multipliers
    ]
    payloads = {
        movie: v157e.build_update_payload(
            split,
            payload,
            local_scales,
            int(args.control_seed) + len(restored.dataset) * 1009,
        )
        for movie, (split, payload) in restored.payloads.items()
    }
    train_movies = sorted(
        movie for movie, payload in payloads.items()
        if payload.split == "train"
    )
    validation_movies = sorted(
        movie for movie, payload in payloads.items()
        if payload.split == "validation"
    )
    test_movies = sorted(
        movie for movie, payload in payloads.items()
        if payload.split == "test"
    )
    weights_by_horizon, h1_guard_percent = v157h.OBJECTIVES[
        str(args.objective)
    ]
    selection, grid = select_model(
        payloads,
        train_movies,
        validation_movies,
        alphas,
        bounds,
        weights_by_horizon,
        h1_guard_percent,
    )
    model = fit_model(
        payloads,
        train_movies + validation_movies,
        selection.alpha,
        weights_by_horizon,
    )
    metric_rows: list[dict[str, Any]] = []
    for movie in test_movies:
        payload = payloads[movie]
        baseline = v157e.metric_rows(
            payload,
            payload.base.mean,
            "no_update",
            None,
        )
        for row in baseline:
            row["dataset"] = restored.dataset
            row["variant"] = "no_update"
            row["sequence"] = movie
            row["protocol"] = "fixed_split_external_guard"
            row["objective_name"] = str(args.objective)
        metric_rows.extend(baseline)
        for control in ("real", "wrong_cell", "stale_time"):
            prediction = predict(
                model,
                payload,
                control,
                selection.bound_px,
            )
            rows = v157e.metric_rows(
                payload,
                prediction,
                control,
                None,
            )
            for row in rows:
                row["dataset"] = restored.dataset
                row["variant"] = control
                row["sequence"] = movie
                row["selected_alpha"] = selection.alpha
                row["selected_bound_px"] = selection.bound_px
                row["protocol"] = "fixed_split_external_guard"
                row["objective_name"] = str(args.objective)
            metric_rows.extend(rows)
    causal = pd.concat(
        [
            v157e.build_causal_audit(
                payloads,
                test_movie=movie,
                validation_movie=validation_movies[0],
                train_movies=train_movies,
            )
            for movie in test_movies
        ],
        ignore_index=True,
    )
    grid.insert(0, "dataset", restored.dataset)
    metadata = {
        **restored.manifest,
        "train_nearest_neighbour_median_px": base_scale,
        "local_scales_px": local_scales,
        "train_sequences": train_movies,
        "validation_sequences": validation_movies,
        "test_sequences": test_movies,
        "selection": {
            "objective_name": str(args.objective),
            "weights": weights_by_horizon,
            "h1_guard_percent": h1_guard_percent,
            "alpha": selection.alpha,
            "bound_px": selection.bound_px,
            "validation_score": selection.validation_score,
            "validation_h1_gain": selection.validation_h1_gain,
        },
    }
    return metric_rows, grid, causal, metadata


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["dataset", "variant", "horizon"],
            as_index=False,
        )
        .agg(
            sequences=("sequence", "nunique"),
            component_rmse_mean=("component_rmse", "mean"),
            vector_rmse_mean=("vector_rmse", "mean"),
            r2_mean=("r2", "mean"),
            gain_percent_mean=("rmse_improvement_percent", "mean"),
            sequences_improved=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )


def decision_table(
    aggregate: pd.DataFrame,
    objective_name: str,
    h1_guard_percent: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, group in aggregate.groupby("dataset"):
        h1 = group[group["horizon"].eq(1)].set_index("variant")
        h6 = group[group["horizon"].eq(6)].set_index("variant")
        real_h1 = float(h1.loc["real", "gain_percent_mean"])
        real_h6 = float(h6.loc["real", "gain_percent_mean"])
        wrong_h6 = float(h6.loc["wrong_cell", "component_rmse_mean"])
        stale_h6 = float(h6.loc["stale_time", "component_rmse_mean"])
        real_h6_rmse = float(h6.loc["real", "component_rmse_mean"])
        passed = bool(
            real_h1 >= -float(h1_guard_percent)
            and real_h6 >= 1.0
            and real_h6_rmse < wrong_h6
            and real_h6_rmse < stale_h6
        )
        records.append(
            {
                "dataset": dataset,
                "objective_name": objective_name,
                "decision": "PASS" if passed else "FAIL_OR_NOT_CONFIRMED",
                "h1_guard_percent": float(h1_guard_percent),
                "h1_gain_percent": real_h1,
                "h6_gain_percent": real_h6,
                "real_h6_component_rmse": real_h6_rmse,
                "wrong_h6_component_rmse": wrong_h6,
                "stale_h6_component_rmse": stale_h6,
                "fixed_split_one_seed": True,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_specs(args.specs)
    alphas = v157e.parse_csv_floats(args.alphas)
    bounds = v157e.parse_csv_floats(args.bounds_px)
    scale_multipliers = v157e.parse_csv_floats(
        args.scale_multipliers
    )
    device = v157e.device_from_cli(args.device)

    all_metrics: list[dict[str, Any]] = []
    all_grids: list[pd.DataFrame] = []
    all_causal: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for dataset, checkpoint in specs.items():
        print(f"[v157i] restore {dataset}", flush=True)
        restored = restore_run(dataset, checkpoint, device)
        metrics, grid, causal, dataset_metadata = evaluate_dataset(
            args,
            restored,
            alphas,
            bounds,
            scale_multipliers,
        )
        all_metrics.extend(metrics)
        all_grids.append(grid)
        all_causal.append(causal.assign(dataset=dataset))
        metadata.append(dataset_metadata)
    metrics_frame = pd.DataFrame(all_metrics)
    aggregate = aggregate_metrics(metrics_frame)
    _, h1_guard_percent = v157h.OBJECTIVES[str(args.objective)]
    decision = decision_table(
        aggregate,
        str(args.objective),
        h1_guard_percent,
    )
    causal = pd.concat(all_causal, ignore_index=True)
    if int(causal["real_future_donor_violations"].sum()) != 0:
        raise RuntimeError("Future donor found")
    if int(causal["stale_future_or_nonstale_violations"].sum()) != 0:
        raise RuntimeError("Invalid stale donor found")
    if not bool(causal["coherent_wrong_packet"].all()):
        raise RuntimeError("Wrong-cell packet coherence failed")

    metrics_frame.to_csv(
        args.out_dir / "v157i_guard_metrics.csv",
        index=False,
    )
    aggregate.to_csv(
        args.out_dir / "v157i_guard_aggregate.csv",
        index=False,
    )
    decision.to_csv(
        args.out_dir / "v157i_guard_decision.csv",
        index=False,
    )
    pd.concat(all_grids, ignore_index=True).to_csv(
        args.out_dir / "v157i_validation_grid.csv",
        index=False,
    )
    causal.to_csv(args.out_dir / "v157i_causal_audit.csv", index=False)
    (args.out_dir / "v157i_dataset_contract.json").write_text(
        json.dumps(v157e.finite(metadata), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# v157i External Semigroup Guards",
        "",
        decision.to_markdown(index=False, floatfmt=".6f"),
        "",
        "These are fixed-split, one-seed transfer guards. They test mechanism",
        "portability but do not replace a movie-level outer-fold estimate.",
        "Neighbour radii are derived from train-only median nearest-neighbour",
        "distance in each dataset.",
        "",
        f"Elapsed: `{(time.time() - started) / 60.0:.2f}` minutes.",
    ]
    (args.out_dir / "v157i_status_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **vars(args),
        "source_sha256": file_sha256(Path(__file__)),
        "protocol": "fixed-split external guard",
        "device_resolved": str(device),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(v157e.finite(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "v157i_status_report.md", flush=True)


if __name__ == "__main__":
    main()
