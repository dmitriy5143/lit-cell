#!/usr/bin/env python3
"""Outer-fold refit gate for sparse innovation-transport operators."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lit_cell_forecasting.innovation_field import (  # noqa: E402
    local_flow_direction,
    nearest_neighbor_scale,
    sparse_gaussian_local_moments,
)
import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402


DEFAULT_V102 = v157e.DEFAULT_V102
DEFAULT_V190 = (
    ROOT / "outputs" / "lachance_innovation_field_v190_bulk_full_2026-07-30"
)
DEFAULT_OUT = (
    ROOT / "outputs" / "lachance_sparse_innovation_transport_refit_v191b_2026-07-30"
)
EPS = 1e-12


@dataclass(frozen=True)
class Variant:
    name: str
    geometry: str
    support_mode: str
    multiplier: float = 0.0
    k: int | None = None
    kernel: str = "gaussian"
    direction_strength: float = 0.0


def parse_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(item) for item in value]


def parse_floats(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(item) for item in value]


def parse_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=DEFAULT_V102)
    parser.add_argument("--v190-dir", type=Path, default=DEFAULT_V190)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument(
        "--variants",
        default=(
            "dense_start,cutoff3_start,cutoff4_start,"
            "field3_start,field3_endpoint,knn64_start,"
            "wendland3_endpoint,directed3_endpoint_beta025"
        ),
    )
    parser.add_argument("--alphas", default="1,10,30,100,300,1000,3000,10000")
    parser.add_argument("--bounds-px", default="0.5,1,1.5,2,3,4,6")
    parser.add_argument("--local-scales-px", default="30,60,120,240")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-seed", type=int, default=191_002)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    return parser.parse_args()


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
    path.write_text(
        json.dumps(finite(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def variant_from_name(name: str) -> Variant:
    mapping = {
        "dense_start": Variant("dense_start", "start", "dense"),
        "cutoff3_start": Variant("cutoff3_start", "start", "sigma", 3.0),
        "cutoff4_start": Variant("cutoff4_start", "start", "sigma", 4.0),
        "field0p5_start": Variant("field0p5_start", "start", "field", 0.5),
        "field1_start": Variant("field1_start", "start", "field", 1.0),
        "field1p5_start": Variant("field1p5_start", "start", "field", 1.5),
        "field2_start": Variant("field2_start", "start", "field", 2.0),
        "field2p5_start": Variant("field2p5_start", "start", "field", 2.5),
        "field2_midpoint": Variant(
            "field2_midpoint",
            "midpoint",
            "field",
            2.0,
        ),
        "field2_endpoint": Variant(
            "field2_endpoint",
            "endpoint",
            "field",
            2.0,
        ),
        "field2_flow_advected": Variant(
            "field2_flow_advected",
            "flow_advected",
            "field",
            2.0,
        ),
        "field3_start": Variant("field3_start", "start", "field", 3.0),
        "field4_start": Variant("field4_start", "start", "field", 4.0),
        "fieldpx1_start": Variant("fieldpx1_start", "start", "field_px", 1.0),
        "fieldpx1p5_start": Variant(
            "fieldpx1p5_start",
            "start",
            "field_px",
            1.5,
        ),
        "fieldpx2_start": Variant("fieldpx2_start", "start", "field_px", 2.0),
        "fieldpx3_start": Variant("fieldpx3_start", "start", "field_px", 3.0),
        "fieldpx4_start": Variant("fieldpx4_start", "start", "field_px", 4.0),
        "directedpx4_start_beta025": Variant(
            "directedpx4_start_beta025",
            "start",
            "field_px",
            4.0,
            direction_strength=0.25,
        ),
        "directedpx4_start_beta05": Variant(
            "directedpx4_start_beta05",
            "start",
            "field_px",
            4.0,
            direction_strength=0.5,
        ),
        "directedpx4_endpoint_beta025": Variant(
            "directedpx4_endpoint_beta025",
            "endpoint",
            "field_px",
            4.0,
            direction_strength=0.25,
        ),
        "directedpx4_endpoint_beta05": Variant(
            "directedpx4_endpoint_beta05",
            "endpoint",
            "field_px",
            4.0,
            direction_strength=0.5,
        ),
        "field3_endpoint": Variant("field3_endpoint", "endpoint", "field", 3.0),
        "knn32_start": Variant("knn32_start", "start", "knn", k=32),
        "knn64_start": Variant("knn64_start", "start", "knn", k=64),
        "knn128_start": Variant("knn128_start", "start", "knn", k=128),
        "wendland3_endpoint": Variant(
            "wendland3_endpoint",
            "endpoint",
            "field_multiscale",
            3.0,
            kernel="wendland_c2",
        ),
        "directed3_endpoint_beta025": Variant(
            "directed3_endpoint_beta025",
            "endpoint",
            "field",
            3.0,
            direction_strength=0.25,
        ),
        "directed3_endpoint_beta05": Variant(
            "directed3_endpoint_beta05",
            "endpoint",
            "field",
            3.0,
            direction_strength=0.5,
        ),
    }
    if name not in mapping:
        raise ValueError(f"Unknown v191b variant: {name}")
    return mapping[name]


def load_movie_xi(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    focus = table[
        table["representation"].eq("gaussian_score")
        & table["detrend"].eq("affine")
        & table["control"].eq("real")
        & table["lag"].eq(1)
        & table["geometry"].eq("endpoint")
        & table["metric"].eq("vector_correlation")
        & table["unit"].eq("nearest_neighbour")
    ][["movie", "exponential_xi"]]
    output = focus.set_index("movie")["exponential_xi"].sort_index()
    if output.index.duplicated().any() or output.isna().any():
        raise RuntimeError("Invalid v190 movie-scale table")
    return output


def support_radii(
    variant: Variant,
    scales: list[float],
    xi_px: float,
) -> list[float]:
    if variant.support_mode == "dense":
        return [float("inf")] * len(scales)
    if variant.support_mode == "sigma":
        return [variant.multiplier * scale for scale in scales]
    if variant.support_mode == "field":
        return [variant.multiplier * xi_px] * len(scales)
    if variant.support_mode == "field_px":
        return [variant.multiplier * xi_px] * len(scales)
    if variant.support_mode == "field_multiscale":
        factors = np.linspace(0.5, 1.25, len(scales))
        return [float(variant.multiplier * xi_px * factor) for factor in factors]
    if variant.support_mode == "knn":
        return [float("inf")] * len(scales)
    raise ValueError(f"Unknown support mode: {variant.support_mode}")


def sparse_real_packet(
    payload: Any,
    scales: list[float],
    variant: Variant,
    train_xi_nn: float,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, float]]:
    rows = payload.rows.reset_index(drop=True)
    count = len(rows)
    feature: dict[str, np.ndarray] = {
        "own_prev_x": np.zeros(count),
        "own_prev_y": np.zeros(count),
        "own_available": np.zeros(count),
        "global_prev_x": np.zeros(count),
        "global_prev_y": np.zeros(count),
    }
    for scale in scales:
        label = str(int(scale))
        for suffix in ("x", "y", "std_x", "std_y", "effective_n"):
            feature[f"local_{label}_{suffix}"] = np.zeros(count)
    latest_donor = np.full(count, -1, dtype=np.int64)
    frame_groups = {
        int(frame): np.asarray(indices, dtype=np.int64)
        for frame, indices in rows.groupby("frame", sort=True).indices.items()
    }
    key_to_index = {
        (int(frame), int(track)): index
        for index, (frame, track) in enumerate(
            rows[["frame", "track_id"]].itertuples(index=False)
        )
    }
    total_edges = 0
    total_dense_edges = 0
    frame_count = 0
    for frame, current in frame_groups.items():
        previous = frame_groups.get(int(frame) - 1, np.empty(0, dtype=np.int64))
        if not len(previous):
            continue
        frame_count += 1
        latest_donor[current] = int(frame) - 1
        previous_score = payload.normal_score[previous]
        global_state = previous_score.mean(axis=0)
        feature["global_prev_x"][current] = global_state[0]
        feature["global_prev_y"][current] = global_state[1]
        current_tracks = rows.iloc[current]["track_id"].to_numpy(np.int64)
        previous_tracks = rows.iloc[previous]["track_id"].to_numpy(np.int64)
        own_indices = np.asarray(
            [
                key_to_index.get((int(frame) - 1, int(track)), -1)
                for track in current_tracks
            ],
            dtype=np.int64,
        )
        available = own_indices >= 0
        if np.any(available):
            selected_rows = current[available]
            selected_own = own_indices[available]
            feature["own_prev_x"][selected_rows] = payload.normal_score[selected_own, 0]
            feature["own_prev_y"][selected_rows] = payload.normal_score[selected_own, 1]
            feature["own_available"][selected_rows] = 1.0

        current_position = rows.iloc[current][["x_px", "y_px"]].to_numpy(np.float64)
        previous_start = rows.iloc[previous][["x_px", "y_px"]].to_numpy(np.float64)
        if variant.geometry == "start":
            source_position = previous_start
        elif variant.geometry == "midpoint":
            source_position = previous_start + 0.5 * payload.target[previous]
        elif variant.geometry == "endpoint":
            source_position = previous_start + payload.target[previous]
        elif variant.geometry == "flow_advected":
            endpoint = previous_start + payload.target[previous]
            query_k = min(9, len(endpoint))
            _, neighbour_index = cKDTree(endpoint).query(
                endpoint,
                k=query_k,
            )
            if query_k == 1:
                neighbour_index = np.asarray(neighbour_index)[:, None]
            local_velocity = np.mean(
                payload.target[previous][np.asarray(neighbour_index)],
                axis=1,
            )
            source_position = endpoint + 0.5 * local_velocity
        else:
            raise ValueError(f"Unknown geometry: {variant.geometry}")
        neighbour_scale = nearest_neighbor_scale(current_position)
        xi_px = (
            train_xi_nn
            if variant.support_mode == "field_px"
            else train_xi_nn * neighbour_scale
        )
        radii = support_radii(variant, scales, xi_px)
        if variant.support_mode == "dense":
            local = v157e.local_previous_state(
                current_position,
                source_position,
                previous_score,
                current_tracks,
                previous_tracks,
                scales,
            )
            candidate_edges = int(
                np.sum(current_tracks[:, None] != previous_tracks[None, :])
            )
        else:
            flow = None
            if variant.direction_strength != 0:
                velocity = rows.iloc[current][["dx_px", "dy_px"]].to_numpy(np.float64)
                flow = local_flow_direction(current_position, velocity, k=8)
            local, diagnostic = sparse_gaussian_local_moments(
                current_position,
                source_position,
                previous_score,
                current_tracks,
                previous_tracks,
                scales,
                support_radii=radii,
                k=variant.k,
                current_flow_direction=flow,
                direction_strength=variant.direction_strength,
                kernel=variant.kernel,
            )
            candidate_edges = diagnostic.candidate_edges
        total_edges += candidate_edges
        total_dense_edges += int(
            np.sum(current_tracks[:, None] != previous_tracks[None, :])
        )
        for name, value in local.items():
            feature[name][current] = value
    names = list(feature)
    matrix = np.column_stack([feature[name] for name in names]).astype(np.float64)
    diagnostics = {
        "frames_with_update": frame_count,
        "candidate_edges": total_edges,
        "dense_edges": total_dense_edges,
        "edge_fraction": total_edges / max(total_dense_edges, 1),
        "train_xi_nn": train_xi_nn,
    }
    return matrix, names, latest_donor, diagnostics


def build_payload(
    split: str,
    base: Any,
    scales: list[float],
    variant: Variant,
    train_xi_nn: float,
    control_seed: int,
) -> tuple[Any, dict[str, float]]:
    if variant.support_mode == "dense" and variant.geometry == "start":
        payload = v157e.build_update_payload(
            split,
            base,
            scales,
            control_seed,
        )
        dense_edges = 0
        frame_groups = {
            int(frame): np.asarray(indices, dtype=np.int64)
            for frame, indices in base.rows.groupby("frame", sort=True).indices.items()
        }
        for frame, current in frame_groups.items():
            previous = frame_groups.get(frame - 1, np.empty(0, dtype=np.int64))
            if not len(previous):
                continue
            current_track = base.rows.iloc[current]["track_id"].to_numpy(np.int64)
            previous_track = base.rows.iloc[previous]["track_id"].to_numpy(np.int64)
            dense_edges += int(
                np.sum(current_track[:, None] != previous_track[None, :])
            )
        return payload, {
            "frames_with_update": len(frame_groups) - 1,
            "candidate_edges": dense_edges,
            "dense_edges": dense_edges,
            "edge_fraction": 1.0,
            "train_xi_nn": train_xi_nn,
        }
    real, names, donor, diagnostics = sparse_real_packet(
        base,
        scales,
        variant,
        train_xi_nn,
    )
    wrong, permutation = v157e.coherent_wrong_cell(
        real,
        base.rows,
        control_seed + base.movie * 1009,
    )
    stale, stale_donor = v157e.coherent_stale_time(real, base.rows, donor)
    current_frame = base.rows["frame"].to_numpy(np.int64)
    if np.any((donor >= 0) & (donor > current_frame - 1)):
        raise RuntimeError(f"Future donor in {variant.name}, movie {base.movie}")
    if np.any((stale_donor >= 0) & (stale_donor > current_frame - 2)):
        raise RuntimeError(f"Non-stale donor in {variant.name}, movie {base.movie}")
    return (
        v157e.UpdatePayload(
            movie=base.movie,
            split=split,
            base=base,
            real=real,
            wrong_cell=wrong,
            stale_time=stale,
            feature_names=names,
            real_latest_donor_frame=donor,
            stale_latest_donor_frame=stale_donor,
            wrong_permutation=permutation,
        ),
        diagnostics,
    )


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    observed = abs(float(values.mean()))
    outcomes = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(outcomes) >= observed - 1e-15))


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["variant", "fit_mode", "control", "horizon"]
    for keys, group in metrics.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "movies": int(group["test_movie"].nunique()),
                "component_rmse_mean": float(group["component_rmse"].mean()),
                "component_rmse_std": float(group["component_rmse"].std(ddof=1))
                if len(group) > 1
                else np.nan,
                "vector_rmse_mean": float(group["vector_rmse"].mean()),
                "vector_r2_mean": float(group["vector_r2"].mean()),
                "gain_percent_mean": float(
                    group["rmse_improvement_percent"].mean()
                ),
                "movies_improved": int((group["component_rmse_delta"] > 0).sum()),
                "sign_flip_p": exact_sign_flip_pvalue(
                    group["component_rmse_delta"].to_numpy(np.float64)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    movies = parse_ints(args.movies)
    seeds = parse_ints(args.seeds)
    alphas = parse_floats(args.alphas)
    bounds = parse_floats(args.bounds_px)
    scales = parse_floats(args.local_scales_px)
    variants = [variant_from_name(name) for name in parse_strings(args.variants)]
    if not any(variant.name == "dense_start" for variant in variants):
        raise SystemExit("variants must include dense_start")
    movie_xi = load_movie_xi(args.v190_dir / "scale_estimates.csv")
    device = v157e.device_from_cli(args.device)
    metric_records: list[dict[str, Any]] = []
    grids: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    audit_records: list[pd.DataFrame] = []

    for test_movie in movies:
        print(f"[v191b] restoring outer fold test={test_movie}", flush=True)
        seed_replays = [
            v157e.restore_fold_seed(args.v102_root, test_movie, seed, device)
            for seed in seeds
        ]
        split_payloads = v157e.student_t_mixture_payloads(seed_replays)
        validation_movies = {
            int(replay.manifest["validation_movie"]) for replay in seed_replays
        }
        train_sets = {
            tuple(int(movie) for movie in replay.manifest["train_movies"])
            for replay in seed_replays
        }
        if len(validation_movies) != 1 or len(train_sets) != 1:
            raise RuntimeError("Fold split mismatch across seeds")
        validation_movie = next(iter(validation_movies))
        train_movies = list(next(iter(train_sets)))
        train_xi_nn = float(movie_xi.reindex(train_movies).median())
        variant_payloads: dict[str, dict[int, Any]] = {}
        for variant in variants:
            payloads: dict[int, Any] = {}
            for movie, (split, base) in split_payloads.items():
                payload, diagnostic = build_payload(
                    split,
                    base,
                    scales,
                    variant,
                    train_xi_nn,
                    args.control_seed + test_movie * 100_003,
                )
                payloads[movie] = payload
                diagnostic_records.append(
                    {
                        "outer_test_movie": test_movie,
                        "validation_movie": validation_movie,
                        "movie": movie,
                        "split": split,
                        "variant": variant.name,
                        "geometry": variant.geometry,
                        "support_mode": variant.support_mode,
                        "kernel": variant.kernel,
                        "direction_strength": variant.direction_strength,
                        **diagnostic,
                    }
                )
            variant_payloads[variant.name] = payloads

        dense_payloads = variant_payloads["dense_start"]
        dense_selection, dense_grid = v157e.select_on_validation(
            dense_payloads,
            train_movies,
            validation_movie,
            alphas,
            bounds,
        )
        dense_model = v157e.refit_model(
            dense_payloads,
            train_movies + [validation_movie],
            dense_selection,
        )
        for variant in variants:
            payloads = variant_payloads[variant.name]
            selection, grid = v157e.select_on_validation(
                payloads,
                train_movies,
                validation_movie,
                alphas,
                bounds,
            )
            grid.insert(0, "outer_test_movie", test_movie)
            grid.insert(1, "variant", variant.name)
            grid.insert(2, "train_xi_nn", train_xi_nn)
            grids.append(grid)
            model = v157e.refit_model(
                payloads,
                train_movies + [validation_movie],
                selection,
            )
            test = payloads[test_movie]
            for control in ("real", "wrong_cell", "stale_time"):
                raw = v157e.predict_ridge(model, test, control)
                prediction = test.base.mean + v157e.bounded_update(
                    raw,
                    selection.bound_px,
                )
                rows = v157e.metric_rows(test, prediction, control, selection)
                for row in rows:
                    row.update(
                        {
                            "variant": variant.name,
                            "fit_mode": "refit",
                            "geometry": variant.geometry,
                            "support_mode": variant.support_mode,
                            "kernel": variant.kernel,
                            "direction_strength": variant.direction_strength,
                            "train_xi_nn": train_xi_nn,
                        }
                    )
                metric_records.extend(rows)
            no_update_rows = v157e.metric_rows(
                test,
                test.base.mean,
                "no_update",
                None,
            )
            for row in no_update_rows:
                row.update(
                    {
                        "variant": variant.name,
                        "fit_mode": "refit",
                        "geometry": variant.geometry,
                        "support_mode": variant.support_mode,
                        "kernel": variant.kernel,
                        "direction_strength": variant.direction_strength,
                        "train_xi_nn": train_xi_nn,
                    }
                )
            metric_records.extend(no_update_rows)

            if variant.name != "dense_start":
                frozen_raw = v157e.predict_ridge(dense_model, test, "real")
                frozen_prediction = test.base.mean + v157e.bounded_update(
                    frozen_raw,
                    dense_selection.bound_px,
                )
                frozen_rows = v157e.metric_rows(
                    test,
                    frozen_prediction,
                    "real",
                    dense_selection,
                )
                for row in frozen_rows:
                    row.update(
                        {
                            "variant": variant.name,
                            "fit_mode": "frozen_dense",
                            "geometry": variant.geometry,
                            "support_mode": variant.support_mode,
                            "kernel": variant.kernel,
                            "direction_strength": variant.direction_strength,
                            "train_xi_nn": train_xi_nn,
                        }
                    )
                metric_records.extend(frozen_rows)
            audit = v157e.build_causal_audit(
                payloads,
                test_movie,
                validation_movie,
                train_movies,
            )
            audit.insert(0, "variant", variant.name)
            audit_records.append(audit)
            print(
                f"[v191b] test={test_movie} variant={variant.name} "
                f"alpha={selection.alpha:g} bound={selection.bound_px:g}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_records)
    aggregate = aggregate_metrics(metrics)
    diagnostics = pd.DataFrame(diagnostic_records)
    validation_grid = pd.concat(grids, ignore_index=True)
    causal_audit = pd.concat(audit_records, ignore_index=True)
    if causal_audit["real_future_donor_violations"].sum() != 0:
        raise RuntimeError("Future donor found in sparse refit")
    if causal_audit["stale_future_or_nonstale_violations"].sum() != 0:
        raise RuntimeError("Stale control violation in sparse refit")
    metrics.to_csv(args.out_dir / "v191b_final_metrics.csv", index=False)
    aggregate.to_csv(args.out_dir / "v191b_final_aggregate.csv", index=False)
    diagnostics.to_csv(args.out_dir / "v191b_operator_diagnostics.csv", index=False)
    validation_grid.to_csv(args.out_dir / "v191b_validation_grid.csv", index=False)
    causal_audit.to_csv(args.out_dir / "v191b_causal_audit.csv", index=False)

    field_multiplier = {
        variant.name: variant.multiplier
        for variant in variants
        if (
            variant.support_mode == "field"
            and variant.geometry == "start"
            and variant.kernel == "gaussian"
            and variant.direction_strength == 0.0
        )
    }
    collapse = metrics[
        metrics["variant"].isin(field_multiplier)
        & metrics["fit_mode"].eq("refit")
        & metrics["control"].eq("real")
    ].copy()
    if len(collapse):
        dense_reference = metrics[
            metrics["variant"].eq("dense_start")
            & metrics["fit_mode"].eq("refit")
            & metrics["control"].eq("real")
        ][["test_movie", "horizon", "component_rmse"]].rename(
            columns={"component_rmse": "dense_component_rmse"}
        )
        edge_reference = diagnostics[
            diagnostics["variant"].isin(field_multiplier)
            & diagnostics["split"].eq("test")
        ][["variant", "outer_test_movie", "edge_fraction"]].rename(
            columns={"outer_test_movie": "test_movie"}
        )
        collapse["support_r_over_xi"] = collapse["variant"].map(
            field_multiplier
        )
        collapse = collapse.merge(
            dense_reference,
            on=["test_movie", "horizon"],
            how="left",
            validate="many_to_one",
        ).merge(
            edge_reference,
            on=["variant", "test_movie"],
            how="left",
            validate="many_to_one",
        )
        collapse["relative_to_dense_percent"] = 100.0 * (
            collapse["component_rmse"] / collapse["dense_component_rmse"] - 1.0
        )
        collapse = collapse[
            [
                "variant",
                "support_r_over_xi",
                "test_movie",
                "horizon",
                "train_xi_nn",
                "edge_fraction",
                "component_rmse",
                "dense_component_rmse",
                "relative_to_dense_percent",
                "rmse_improvement_percent",
            ]
        ].sort_values(["horizon", "support_r_over_xi", "test_movie"])
    collapse.to_csv(
        args.out_dir / "v191b_scale_performance_collapse.csv",
        index=False,
    )

    h6 = aggregate[
        aggregate["fit_mode"].eq("refit")
        & aggregate["control"].eq("real")
        & aggregate["horizon"].eq(6)
    ].sort_values("component_rmse_mean")
    dense_h6 = h6[h6["variant"].eq("dense_start")]
    if len(dense_h6):
        reference = float(dense_h6.iloc[0]["component_rmse_mean"])
        h6 = h6.copy()
        h6["relative_to_dense_percent"] = 100.0 * (
            h6["component_rmse_mean"] - reference
        ) / max(reference, EPS)
    lines = [
        "# v191b Sparse Innovation-Transport Refit Gate",
        "",
        "All supports are selected from v190 scales of the outer-fold training",
        "movies only. Every variant is evaluated both after its own fold-local",
        "refit and, where applicable, through the frozen dense calibrator.",
        "",
        "## Refit h6",
        "",
        h6[
            [
                "variant",
                "movies",
                "component_rmse_mean",
                "vector_r2_mean",
                "gain_percent_mean",
                "movies_improved",
                "relative_to_dense_percent",
            ]
        ].to_markdown(index=False)
        if len(h6)
        else "No h6 rows.",
        "",
        "A sparse operator is accepted only if the six-movie final metric, h1",
        "guard, and wrong/stale controls remain equivalent to the dense model.",
    ]
    (args.out_dir / "v191b_status_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    write_json(
        args.out_dir / "v191b_manifest.json",
        {
            "ok": True,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "movies": movies,
            "seeds": seeds,
            "variants": [variant.__dict__ for variant in variants],
            "scale_source": str(
                (args.v190_dir / "scale_estimates.csv").resolve()
            ),
            "strict_train_only_xi": True,
            "protocol": "outer-fold v97 replay; streaming/receding h1",
        },
    )


if __name__ == "__main__":
    main()
