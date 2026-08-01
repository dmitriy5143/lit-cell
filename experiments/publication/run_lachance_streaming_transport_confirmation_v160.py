#!/usr/bin/env python3
"""Frozen track-native confirmation of causal innovation transport.

The public LaChance archive contains raw MDCK Bulk movies only for the six
sample tissues.  Movies 10--16 therefore cannot reproduce the visual v52
anchor used by the primary six-movie result.  This runner performs the
strongest valid confirmation available on those movies:

* train the unchanged v97-direct architecture on a track-native anchor cache;
* choose transport hyperparameters only on development movies 1--6;
* freeze a signed contract before loading confirmation predictions;
* evaluate movies 10--16 once, with coherent wrong-cell and stale controls.

This confirms the sequential innovation-transport mechanism, not the exact
visual-v52 operating point.  The distinction is written into every artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, t as student_t


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_causal_innovation_state_space_v97 as v97  # noqa: E402
import run_lachance_dense_innovation_sweep_v85 as v85  # noqa: E402
import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402
import run_lachance_foldlocal_semigroup_pareto_v157h as v157h  # noqa: E402
import run_lachance_joint_graph_copula_v154 as v154  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "lachance_streaming_transport_confirmation_v160"
DEFAULT_DEV_CACHE = (
    ROOT / "outputs" / "lachance_track_native_v160_development_cache"
)
DEFAULT_CONFIRM_CACHE = (
    ROOT / "outputs" / "lachance_track_native_v160_confirmation_cache"
)
DEFAULT_CHECKPOINTS = ",".join(
    f"{seed}=outputs/lachance_track_native_v160_seed{seed}/v97_direct.pt"
    for seed in (7, 42, 123)
)
DEVELOPMENT_MOVIES = (1, 2, 3, 4, 5, 6)
CONFIRMATION_MOVIES = (10, 11, 12, 13, 14, 15, 16)
HORIZONS = (1, 2, 4, 6)
EPS = 1e-8


@dataclass
class FrozenSelection:
    objective: str
    packet: str
    alpha: float
    bound_px: float
    development_score: float
    development_h1_gain_percent: float


def finite(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(item) for item in value]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(item) for item in value]


def parse_checkpoints(value: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for token in value.split(","):
        seed_text, path_text = token.strip().split("=", 1)
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        output[int(seed_text)] = path.resolve()
    if not output:
        raise ValueError("At least one checkpoint is required")
    return output


def cache_feature_index(cache: Path) -> Path:
    path = cache / "native_feature_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def restore_checkpoint(
    checkpoint_path: Path,
    cache: Path,
    device: torch.device,
) -> v157e.SeedReplay:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    variant = v97.TrainVariant(**checkpoint["variant"])
    if variant.name != "v97_direct" or variant.output_mode != "direct":
        raise RuntimeError(f"Expected v97_direct checkpoint, got {variant}")
    args = v157e.checkpoint_namespace(checkpoint, str(device))
    args.anchor_cache = cache.resolve()
    args.features = cache_feature_index(cache).resolve()
    prep = v97.load_prepared(args, variant)
    metadata = checkpoint["metadata"]
    static_dim = int(prep.static[0].shape[1])
    expected_dim = int(
        checkpoint["state_dict"]["static_encoder.0.weight"].shape[1]
    )
    if static_dim != expected_dim:
        raise RuntimeError(
            f"Checkpoint/cache feature mismatch: {static_dim} != {expected_dim}"
        )
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

    rows: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    split_manifest: dict[str, Any] = {}
    for split_index, bundle in enumerate(prep.bundles):
        split = v157e.split_name(split_index)
        frame = bundle.rows.reset_index(drop=True).copy()
        target = np.asarray(bundle.target_steps[:, 0], dtype=np.float64)
        prediction = np.asarray(replay[split_index].prediction, dtype=np.float64)
        scale = np.maximum(
            np.asarray(replay[split_index].scale, dtype=np.float64) * factor,
            1e-4,
        )
        if not (
            len(frame) == len(target) == len(prediction) == len(scale)
            and np.isfinite(target).all()
            and np.isfinite(prediction).all()
            and np.isfinite(scale).all()
        ):
            raise RuntimeError(f"Invalid replay output for {checkpoint_path}/{split}")
        rows[split] = frame
        targets[split] = target
        predictions[split] = prediction
        scales[split] = scale
        split_manifest[split] = {
            "rows": len(frame),
            "movies": sorted(int(item) for item in frame.sequence.unique()),
            "key_sha256": v157e.sha256_array(v157e.ordered_key_array(frame)),
            "target_sha256": v157e.sha256_array(target.astype(np.float32)),
        }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return v157e.SeedReplay(
        seed=int(args.seed),
        fold_dir=checkpoint_path.parent,
        checkpoint=checkpoint_path,
        anchor_cache=cache.resolve(),
        degrees_of_freedom=float(metadata["degrees_of_freedom"]),
        uncertainty_factor=float(factor),
        rows=rows,
        targets=targets,
        predictions=predictions,
        scales=scales,
        manifest={
            "seed": int(args.seed),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "cache": str(cache.resolve()),
            "variant": variant.name,
            "eta": eta,
            "uncertainty_factor": float(factor),
            "splits": split_manifest,
        },
    )


def assert_development_confirmation_identity(
    development: list[v157e.SeedReplay],
    confirmation: list[v157e.SeedReplay],
) -> None:
    if [item.seed for item in development] != [item.seed for item in confirmation]:
        raise RuntimeError("Seed order mismatch between caches")
    for left, right in zip(development, confirmation):
        for split in ("train", "validation"):
            if not np.array_equal(
                v157e.ordered_key_array(left.rows[split]),
                v157e.ordered_key_array(right.rows[split]),
            ):
                raise RuntimeError(
                    f"Development/confirmation {split} key mismatch for seed {left.seed}"
                )
            if not np.array_equal(left.targets[split], right.targets[split]):
                raise RuntimeError(
                    f"Development/confirmation {split} target mismatch for seed {left.seed}"
                )
            if not np.allclose(
                left.predictions[split],
                right.predictions[split],
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"Development/confirmation {split} prediction mismatch for seed {left.seed}"
                )


def mixture_payloads(
    replays: list[v157e.SeedReplay],
) -> dict[int, tuple[str, v154.MoviePayload]]:
    return v157e.student_t_mixture_payloads(replays)


def mask_packet(
    payload: v157e.UpdatePayload,
    packet: str,
) -> v157e.UpdatePayload:
    if packet == "full":
        return payload
    names = payload.feature_names
    if packet == "own_only":
        keep = np.asarray(
            [
                name.startswith("own_prev_") or name == "own_available"
                for name in names
            ],
            dtype=bool,
        )
    elif packet == "local_only":
        keep = np.asarray(
            [name.startswith("local_") for name in names],
            dtype=bool,
        )
    elif packet == "own_local":
        keep = np.asarray(
            [
                name.startswith("own_prev_")
                or name == "own_available"
                or name.startswith("local_")
                for name in names
            ],
            dtype=bool,
        )
    elif packet == "global_only":
        keep = np.asarray(
            [name.startswith("global_prev_") for name in names],
            dtype=bool,
        )
    else:
        raise ValueError(f"Unknown packet {packet!r}")

    def masked(value: np.ndarray) -> np.ndarray:
        output = np.zeros_like(value)
        output[:, keep] = value[:, keep]
        return output

    return replace(
        payload,
        real=masked(payload.real),
        wrong_cell=masked(payload.wrong_cell),
        stale_time=masked(payload.stale_time),
    )


def build_payloads(
    split_payloads: dict[int, tuple[str, v154.MoviePayload]],
    scales: list[float],
    seed: int,
    packet: str = "full",
) -> dict[int, v157e.UpdatePayload]:
    return {
        movie: mask_packet(
            v157e.build_update_payload(
                split,
                base,
                scales,
                int(seed) + int(movie) * 100_003,
            ),
            packet,
        )
        for movie, (split, base) in split_payloads.items()
    }


def ridge_statistics(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    weights: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_mean, row_scale = v157e.row_normalization(payloads, movies)
    features, target, sample_weight = v157h.training_data(
        payloads,
        movies,
        row_mean,
        row_scale,
        weights,
    )
    root = np.sqrt(
        sample_weight / max(float(np.mean(sample_weight)), EPS)
    )[:, None]
    weighted = features * root
    gram = weighted.T @ weighted
    rhs = weighted.T @ (target * root)
    return row_mean, row_scale, gram, rhs


def solve_ridge(
    statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> v157e.WeightedRidge:
    row_mean, row_scale, gram, rhs = statistics
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return v157e.WeightedRidge(row_mean, row_scale, coefficients)


def select_development_configuration(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    objective: str,
    packet: str,
    alphas: list[float],
    bounds: list[float],
) -> tuple[FrozenSelection, pd.DataFrame]:
    weights, h1_guard = v157h.OBJECTIVES[objective]
    records: list[dict[str, Any]] = []
    for holdout in movies:
        fit_movies = [movie for movie in movies if movie != holdout]
        statistics = ridge_statistics(payloads, fit_movies, weights)
        validation = payloads[holdout]
        for alpha in alphas:
            model = solve_ridge(statistics, alpha)
            raw = v157e.predict_ridge(model, validation, "real")
            for bound in bounds:
                prediction = validation.base.mean + v157e.bounded_update(
                    raw,
                    bound,
                )
                metrics = v157e.metric_rows(
                    validation,
                    prediction,
                    "development_real",
                    None,
                )
                record: dict[str, Any] = {
                    "objective": objective,
                    "packet": packet,
                    "holdout_movie": holdout,
                    "alpha": float(alpha),
                    "bound_px": float(bound),
                    "score": v157h.score_metrics(metrics, weights),
                }
                for row in metrics:
                    horizon = int(row["horizon"])
                    record[f"h{horizon}_rmse"] = float(row["component_rmse"])
                    record[f"h{horizon}_gain_percent"] = float(
                        row["rmse_improvement_percent"]
                    )
                records.append(record)
    grid = pd.DataFrame(records)
    grouped = (
        grid.groupby(
            ["objective", "packet", "alpha", "bound_px"],
            as_index=False,
        )
        .agg(
            development_score=("score", "mean"),
            development_h1_gain_percent=("h1_gain_percent", "mean"),
            development_h6_gain_percent=("h6_gain_percent", "mean"),
            positive_h6_movies=(
                "h6_gain_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    eligible = grouped[
        grouped.development_h1_gain_percent.ge(-float(h1_guard))
    ]
    if eligible.empty:
        eligible = grouped
    best = eligible.sort_values(
        [
            "development_score",
            "development_h1_gain_percent",
            "bound_px",
            "alpha",
        ],
        ascending=[True, False, True, True],
    ).iloc[0]
    selection = FrozenSelection(
        objective=objective,
        packet=packet,
        alpha=float(best.alpha),
        bound_px=float(best.bound_px),
        development_score=float(best.development_score),
        development_h1_gain_percent=float(
            best.development_h1_gain_percent
        ),
    )
    grid = grid.merge(
        grouped,
        on=["objective", "packet", "alpha", "bound_px"],
        how="left",
        validate="many_to_one",
    )
    grid["selected"] = (
        grid.alpha.eq(selection.alpha)
        & grid.bound_px.eq(selection.bound_px)
    )
    return selection, grid


def fit_final_model(
    payloads: dict[int, v157e.UpdatePayload],
    movies: list[int],
    selection: FrozenSelection,
) -> v157e.WeightedRidge:
    weights, _ = v157h.OBJECTIVES[selection.objective]
    return solve_ridge(
        ridge_statistics(payloads, movies, weights),
        selection.alpha,
    )


def evaluate_confirmation(
    payloads: dict[int, v157e.UpdatePayload],
    model: v157e.WeightedRidge,
    selection: FrozenSelection,
    classical_predictions: dict[str, dict[int, np.ndarray]] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for movie in sorted(payloads):
        payload = payloads[movie]
        for control in ("real", "wrong_cell", "stale_time"):
            raw = v157e.predict_ridge(model, payload, control)
            prediction = payload.base.mean + v157e.bounded_update(
                raw,
                selection.bound_px,
            )
            rows = v157e.metric_rows(payload, prediction, control, None)
            for row in rows:
                row.update(
                    {
                        "objective_name": selection.objective,
                        "packet_name": selection.packet,
                        "variant": f"{selection.objective}_{selection.packet}_{control}",
                        "selected_alpha": selection.alpha,
                        "selected_bound_px": selection.bound_px,
                    }
                )
            records.extend(rows)
        for control, prediction in (
            ("v97_no_update", payload.base.mean),
            (
                "constant_velocity",
                payload.base.rows[["dx_px", "dy_px"]].to_numpy(np.float64),
            ),
        ):
            rows = v157e.metric_rows(payload, prediction, control, None)
            for row in rows:
                row.update(
                    {
                        "objective_name": selection.objective,
                        "packet_name": selection.packet,
                        "variant": control,
                        "selected_alpha": selection.alpha,
                        "selected_bound_px": selection.bound_px,
                    }
                )
            records.extend(rows)
        for control, predictions_by_movie in sorted(
            (classical_predictions or {}).items()
        ):
            prediction = predictions_by_movie[movie]
            rows = v157e.metric_rows(payload, prediction, control, None)
            for row in rows:
                row.update(
                    {
                        "objective_name": selection.objective,
                        "packet_name": selection.packet,
                        "variant": control,
                        "selected_alpha": selection.alpha,
                        "selected_bound_px": selection.bound_px,
                    }
                )
            records.extend(rows)
    return pd.DataFrame(records)


def align_bundle_prediction(
    bundle: Any,
    prediction: np.ndarray,
    payloads: dict[int, v157e.UpdatePayload],
) -> dict[int, np.ndarray]:
    keys = bundle.rows[list(v157e.KEYS)].to_numpy(np.int64)
    lookup = {
        tuple(int(item) for item in key): index
        for index, key in enumerate(keys)
    }
    output: dict[int, np.ndarray] = {}
    for movie, payload in sorted(payloads.items()):
        selected: list[int] = []
        for key in payload.base.rows[list(v157e.KEYS)].itertuples(
            index=False,
            name=None,
        ):
            index = lookup.get(tuple(int(item) for item in key))
            if index is None:
                raise RuntimeError(
                    f"Classical prediction key missing for movie {movie}: {key}"
                )
            selected.append(index)
        output[movie] = np.asarray(
            prediction[np.asarray(selected, dtype=np.int64)],
            dtype=np.float64,
        )
    return output


def tune_classical_predictions(
    development_cache: Path,
    confirmation_cache: Path,
    confirmation_payloads: dict[int, v157e.UpdatePayload],
    q_grid: list[float],
    r_grid: list[float],
    turn_grid: list[float],
) -> tuple[dict[str, dict[int, np.ndarray]], pd.DataFrame]:
    development_bundles = v85.load_anchor_cache(development_cache)
    confirmation_bundles = v85.load_anchor_cache(confirmation_cache)
    development_validation = development_bundles[1]
    confirmation_test = confirmation_bundles[2]
    validation_weights = {1: 0.90, 2: 0.05, 4: 0.03, 6: 0.02}
    diagnostics: list[dict[str, Any]] = []
    output: dict[str, dict[int, np.ndarray]] = {}
    for kind in ("cv", "ca"):
        best: tuple[float, float, float] | None = None
        for q in q_grid:
            for r in r_grid:
                prediction = v97.kalman_predictions(
                    development_validation,
                    kind,
                    q,
                    r,
                )
                score = v97.weighted_rolling_score(
                    development_validation,
                    prediction,
                    validation_weights,
                )
                if best is None or score < best[0]:
                    best = (score, q, r)
        assert best is not None
        method = f"kalman_{kind}"
        prediction = v97.kalman_predictions(
            confirmation_test,
            kind,
            best[1],
            best[2],
        )
        output[method] = align_bundle_prediction(
            confirmation_test,
            prediction,
            confirmation_payloads,
        )
        diagnostics.append(
            {
                "method": method,
                "validation_score": best[0],
                "q": best[1],
                "r": best[2],
                "turn_rate": np.nan,
            }
        )

    best_imm: tuple[float, float, float, float] | None = None
    for q in q_grid:
        for r in r_grid:
            for turn in turn_grid:
                prediction = v97.imm_predictions(
                    development_validation,
                    q,
                    r,
                    turn,
                )
                score = v97.weighted_rolling_score(
                    development_validation,
                    prediction,
                    validation_weights,
                )
                if best_imm is None or score < best_imm[0]:
                    best_imm = (score, q, r, turn)
    assert best_imm is not None
    prediction = v97.imm_predictions(
        confirmation_test,
        best_imm[1],
        best_imm[2],
        best_imm[3],
    )
    output["imm_cv_ca_turn"] = align_bundle_prediction(
        confirmation_test,
        prediction,
        confirmation_payloads,
    )
    diagnostics.append(
        {
            "method": "imm_cv_ca_turn",
            "validation_score": best_imm[0],
            "q": best_imm[1],
            "r": best_imm[2],
            "turn_rate": best_imm[3],
        }
    )
    return output, pd.DataFrame(diagnostics)


def aggregate_confirmation(metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["objective_name", "packet_name", "control", "horizon"]
    for key, group in metrics.groupby(keys):
        objective, packet, control, horizon = key
        bootstrap = v157e.movie_cluster_bootstrap(
            group.component_rmse_delta.to_numpy(np.float64),
            20_000,
            int(seed) + int(horizon) * 1009,
        )
        records.append(
            {
                "objective_name": objective,
                "packet_name": packet,
                "control": control,
                "horizon": int(horizon),
                "movies": int(group.test_movie.nunique()),
                "component_rmse_mean": float(group.component_rmse.mean()),
                "component_rmse_std": float(group.component_rmse.std(ddof=1)),
                "vector_rmse_mean": float(group.vector_rmse.mean()),
                "r2_mean": float(group.r2.mean()),
                "v97_component_rmse_mean": float(
                    group.baseline_component_rmse.mean()
                ),
                "gain_vs_v97_percent_mean": float(
                    group.rmse_improvement_percent.mean()
                ),
                "movies_improved_vs_v97": int(
                    (group.component_rmse_delta > 0).sum()
                ),
                "sign_flip_p": v157e.exact_sign_flip_pvalue(
                    group.component_rmse_delta.to_numpy(np.float64)
                ),
                **bootstrap,
            }
        )
    return pd.DataFrame(records)


def decision_rows(
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for objective in ("h1_strict", "h6_guard10"):
        real = aggregate[
            aggregate.objective_name.eq(objective)
            & aggregate.packet_name.eq("full")
            & aggregate.control.eq("real")
        ].set_index("horizon")
        wrong = aggregate[
            aggregate.objective_name.eq(objective)
            & aggregate.packet_name.eq("full")
            & aggregate.control.eq("wrong_cell")
        ].set_index("horizon")
        stale = aggregate[
            aggregate.objective_name.eq(objective)
            & aggregate.packet_name.eq("full")
            & aggregate.control.eq("stale_time")
        ].set_index("horizon")
        cv = aggregate[
            aggregate.objective_name.eq(objective)
            & aggregate.packet_name.eq("full")
            & aggregate.control.eq("constant_velocity")
        ].set_index("horizon")
        h1_gain = float(real.loc[1, "gain_vs_v97_percent_mean"])
        h6_gain = float(real.loc[6, "gain_vs_v97_percent_mean"])
        h6_cv_gain = 100.0 * (
            float(cv.loc[6, "component_rmse_mean"])
            - float(real.loc[6, "component_rmse_mean"])
        ) / max(float(cv.loc[6, "component_rmse_mean"]), EPS)
        controls_pass = bool(
            float(real.loc[6, "component_rmse_mean"])
            < float(wrong.loc[6, "component_rmse_mean"])
            and float(real.loc[6, "component_rmse_mean"])
            < float(stale.loc[6, "component_rmse_mean"])
        )
        positive = int(real.loc[6, "movies_improved_vs_v97"])
        bootstrap_probability = float(
            real.loc[6, "bootstrap_probability_positive"]
        )
        if objective == "h1_strict":
            passed = bool(
                h1_gain >= 0.0
                and h6_gain >= 5.0
                and positive >= 5
                and controls_pass
                and bootstrap_probability >= 0.95
            )
        else:
            passed = bool(
                h6_gain >= 10.0
                and h6_cv_gain >= 5.0
                and h1_gain >= -10.0
                and positive >= 5
                and controls_pass
            )
        records.append(
            {
                "objective_name": objective,
                "track_native_confirmation_pass": passed,
                "h1_gain_vs_v97_percent": h1_gain,
                "h6_gain_vs_v97_percent": h6_gain,
                "h6_gain_vs_constant_velocity_percent": h6_cv_gain,
                "h6_movies_improved_vs_v97": positive,
                "h6_real_beats_wrong_and_stale": controls_pass,
                "h6_bootstrap_probability_positive": bootstrap_probability,
            }
        )
    return pd.DataFrame(records)


def contract_files(cache: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in cache.glob("**/*")
        if path.is_file()
        and path.name in {
            "arrays.npz",
            "contract.json",
            "native_cache_status.json",
            "meta.json",
            "native_feature_index.csv",
            "rows.csv",
        }
    )
    return {
        str(path.relative_to(cache)): sha256_file(path)
        for path in files
    }


def report(
    selections: list[FrozenSelection],
    aggregate: pd.DataFrame,
    decisions: pd.DataFrame,
    elapsed: float,
) -> str:
    lines = [
        "# v160 Frozen Track-Native Streaming Confirmation",
        "",
        "## Scope",
        "",
        "- Development movies: MDCK Bulk 01-06.",
        "- Current-configuration-unseen confirmation movies: 10-16.",
        "- Raw sample video is unavailable for movies 10-16 in the public archive.",
        "- Therefore this is a track-native confirmation of the innovation-transport mechanism, not an exact confirmation of the visual-v52 anchor.",
        "- h2/h4/h6 are streaming/receding-h1 forecasts.",
        "",
        "## Frozen selections",
        "",
        pd.DataFrame([finite(item.__dict__) for item in selections]).to_markdown(
            index=False
        ),
        "",
        "## Confirmation aggregate",
        "",
        aggregate[
            aggregate.packet_name.eq("full")
            & aggregate.control.isin(
                [
                    "real",
                    "wrong_cell",
                    "stale_time",
                    "v97_no_update",
                    "constant_velocity",
                    "kalman_cv",
                    "kalman_ca",
                    "imm_cv_ca_turn",
                ]
            )
        ][
            [
                "objective_name",
                "control",
                "horizon",
                "component_rmse_mean",
                "r2_mean",
                "gain_vs_v97_percent_mean",
                "movies_improved_vs_v97",
            ]
        ].to_markdown(index=False),
        "",
        "## Decision",
        "",
        decisions.to_markdown(index=False),
        "",
        f"Elapsed: `{elapsed / 3600.0:.2f} h`.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-cache", type=Path, default=DEFAULT_DEV_CACHE)
    parser.add_argument(
        "--confirmation-cache",
        type=Path,
        default=DEFAULT_CONFIRM_CACHE,
    )
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-movies", default="1,2,3,4,5,6")
    parser.add_argument("--confirmation-movies", default="10,11,12,13,14,15,16")
    parser.add_argument("--objectives", default="h1_strict,h6_guard10")
    parser.add_argument("--packets", default="full,own_only,local_only")
    parser.add_argument("--alphas", default="1,10,30,100,300,1000,3000,10000")
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--kalman-q-grid", default="0.1,0.5,1,4,16")
    parser.add_argument("--kalman-r-grid", default="0.1,0.5,1,4,16")
    parser.add_argument("--imm-turn-grid", default="0.08,0.15,0.25,0.4")
    parser.add_argument("--control-seed", type=int, default=160_001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    development_movies = parse_ints(args.development_movies)
    confirmation_movies = parse_ints(args.confirmation_movies)
    if tuple(development_movies) != DEVELOPMENT_MOVIES:
        raise ValueError(f"Frozen development cohort is {DEVELOPMENT_MOVIES}")
    if tuple(confirmation_movies) != CONFIRMATION_MOVIES:
        raise ValueError(f"Frozen confirmation cohort is {CONFIRMATION_MOVIES}")
    objectives = [token.strip() for token in args.objectives.split(",") if token.strip()]
    packets = [token.strip() for token in args.packets.split(",") if token.strip()]
    if "full" not in packets:
        raise ValueError("The primary full packet is required")
    unknown = sorted(set(objectives) - set(v157h.OBJECTIVES))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    checkpoints = parse_checkpoints(args.checkpoints)
    device = v157e.device_from_cli(args.device)

    development_replays = [
        restore_checkpoint(checkpoints[seed], args.development_cache, device)
        for seed in sorted(checkpoints)
    ]
    development_split = mixture_payloads(development_replays)
    if set(development_split) != set(development_movies):
        raise RuntimeError(
            f"Development movie mismatch: {sorted(development_split)}"
        )
    scales = parse_floats(args.local_scales_px)
    alphas = parse_floats(args.alphas)
    bounds = parse_floats(args.bounds_px)

    selections: list[FrozenSelection] = []
    grids: list[pd.DataFrame] = []
    final_models: dict[tuple[str, str], v157e.WeightedRidge] = {}
    for packet in packets:
        payloads = build_payloads(
            development_split,
            scales,
            int(args.control_seed),
            packet,
        )
        for objective in objectives:
            selection, grid = select_development_configuration(
                payloads,
                development_movies,
                objective,
                packet,
                alphas,
                bounds,
            )
            selections.append(selection)
            grids.append(grid)
            final_models[(objective, packet)] = fit_final_model(
                payloads,
                development_movies,
                selection,
            )

    development_grid = pd.concat(grids, ignore_index=True)
    development_grid.to_csv(
        args.out_dir / "v160_development_selection.csv",
        index=False,
    )
    selection_payload = [finite(item.__dict__) for item in selections]
    freeze_contract = {
        "protocol": "track_native_current_configuration_unseen_confirmation",
        "scientific_scope": (
            "Confirms sequential innovation transport without the visual-v52 "
            "anchor; raw video for confirmation movies is unavailable."
        ),
        "development_movies": development_movies,
        "confirmation_movies": confirmation_movies,
        "objectives": objectives,
        "packets": packets,
        "local_scales_px": scales,
        "alphas": alphas,
        "bounds_px": bounds,
        "kalman_q_grid": parse_floats(args.kalman_q_grid),
        "kalman_r_grid": parse_floats(args.kalman_r_grid),
        "imm_turn_grid": parse_floats(args.imm_turn_grid),
        "selections": selection_payload,
        "checkpoint_hashes": {
            str(seed): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for seed, path in sorted(checkpoints.items())
        },
        "development_cache": {
            "path": str(args.development_cache.resolve()),
            "files": contract_files(args.development_cache),
        },
        "confirmation_cache": {
            "path": str(args.confirmation_cache.resolve()),
            "files": contract_files(args.confirmation_cache),
        },
        "source_hashes": {
            Path(__file__).name: sha256_file(Path(__file__)),
            Path(v157e.__file__).name: sha256_file(Path(v157e.__file__)),
            Path(v157h.__file__).name: sha256_file(Path(v157h.__file__)),
            Path(v97.__file__).name: sha256_file(Path(v97.__file__)),
            Path(v85.__file__).name: sha256_file(Path(v85.__file__)),
            "build_lachance_online_track_anchor_cache_v97.py": sha256_file(
                SCRIPTS / "build_lachance_online_track_anchor_cache_v97.py"
            ),
        },
        "inference_contract": {
            "prediction_time": "t",
            "latest_real_measurement": "completed transition t-1 to t",
            "target_time": "t+1",
            "target_used_for_selection": False,
            "confirmation_metrics_used_for_tuning": False,
        },
    }
    contract_path = args.out_dir / "v160_frozen_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != finite(freeze_contract):
            raise RuntimeError("Existing frozen contract differs; refusing overwrite")
    else:
        write_json(contract_path, freeze_contract)
    (args.out_dir / "V160_FROZEN_BEFORE_CONFIRMATION").write_text(
        sha256_file(contract_path) + "\n",
        encoding="ascii",
    )

    if args.preflight_only:
        print(contract_path)
        return

    confirmation_replays = [
        restore_checkpoint(checkpoints[seed], args.confirmation_cache, device)
        for seed in sorted(checkpoints)
    ]
    assert_development_confirmation_identity(
        development_replays,
        confirmation_replays,
    )
    confirmation_split = mixture_payloads(confirmation_replays)
    confirmation_split = {
        movie: value
        for movie, value in confirmation_split.items()
        if movie in confirmation_movies
    }
    if set(confirmation_split) != set(confirmation_movies):
        raise RuntimeError(
            f"Confirmation movie mismatch: {sorted(confirmation_split)}"
        )

    full_confirmation_payloads = build_payloads(
        confirmation_split,
        scales,
        int(args.control_seed),
        "full",
    )
    classical_predictions, classical_diagnostics = tune_classical_predictions(
        args.development_cache,
        args.confirmation_cache,
        full_confirmation_payloads,
        parse_floats(args.kalman_q_grid),
        parse_floats(args.kalman_r_grid),
        parse_floats(args.imm_turn_grid),
    )
    classical_diagnostics.to_csv(
        args.out_dir / "v160_classical_filter_selection.csv",
        index=False,
    )

    metric_frames: list[pd.DataFrame] = []
    for packet in packets:
        confirmation_payloads = (
            full_confirmation_payloads
            if packet == "full"
            else build_payloads(
                confirmation_split,
                scales,
                int(args.control_seed),
                packet,
            )
        )
        for objective in objectives:
            selection = next(
                item
                for item in selections
                if item.objective == objective and item.packet == packet
            )
            metric_frames.append(
                evaluate_confirmation(
                    confirmation_payloads,
                    final_models[(objective, packet)],
                    selection,
                    classical_predictions,
                )
            )
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(
        args.out_dir / "v160_confirmation_metrics.csv",
        index=False,
    )
    aggregate = aggregate_confirmation(metrics, int(args.control_seed))
    aggregate.to_csv(
        args.out_dir / "v160_confirmation_aggregate.csv",
        index=False,
    )
    controls = aggregate[
        aggregate.control.isin(
            [
                "real",
                "wrong_cell",
                "stale_time",
                "v97_no_update",
                "constant_velocity",
                "kalman_cv",
                "kalman_ca",
                "imm_cv_ca_turn",
            ]
        )
    ].copy()
    controls.to_csv(
        args.out_dir / "v160_controls.csv",
        index=False,
    )
    decisions = decision_rows(metrics, aggregate)
    decisions.to_csv(
        args.out_dir / "v160_decisions.csv",
        index=False,
    )
    report_text = report(
        selections,
        aggregate,
        decisions,
        time.time() - started,
    )
    (args.out_dir / "v160_decision_report.md").write_text(
        report_text,
        encoding="utf-8",
    )
    write_json(
        args.out_dir / "v160_run_manifest.json",
        {
            "ok": True,
            "elapsed_sec": time.time() - started,
            "device": str(device),
            "development_replays": [item.manifest for item in development_replays],
            "confirmation_replays": [item.manifest for item in confirmation_replays],
            "frozen_contract_sha256": sha256_file(contract_path),
        },
    )
    print(report_text)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.out_dir / "v160_error.json",
            {
                "ok": False,
                "error": repr(exc),
                "elapsed_sec": None,
            },
        )
        raise


if __name__ == "__main__":
    main()
