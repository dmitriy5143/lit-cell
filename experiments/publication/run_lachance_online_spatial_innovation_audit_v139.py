#!/usr/bin/env python3
"""Audit spatial dependence left in causal one-step forecast errors.

This is a target-aware diagnostic, never an inference module. It consumes the
completed six-movie v102 outer-LOMO predictions, averages optimizer seeds
inside each held-out movie, and compares nearest-neighbour error dependence
with a deterministic same-frame random-pair control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V102 = ROOT / "outputs/lachance_online_lomo_benchmark_v102_v97_production_2026-07-21"
DEFAULT_OUT = ROOT / "outputs/lachance_online_spatial_innovation_audit_v139_2026-07-22"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v102-root", type=Path, default=DEFAULT_V102)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--movies", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--neighbor-k", default="1,4,8,16")
    parser.add_argument("--random-repeats", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=139)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fold_dir(root: Path, movie: int, seed: int) -> Path:
    matches = sorted((root / "folds").glob(f"test{movie:02d}_val*_seed{seed}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fold for movie={movie}, seed={seed}; found {matches}")
    return matches[0]


def load_seed(root: Path, movie: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    fold = fold_dir(root, movie, seed)
    ready_files = sorted((root / "cache" / fold.name).glob("v52_anchor_cache_*/v102_anchor_ready.json"))
    if len(ready_files) != 1:
        raise RuntimeError(f"Expected one anchor-ready file for {fold.name}; found {ready_files}")
    ready = json.loads(ready_files[0].read_text(encoding="utf-8"))
    cache = Path(ready["final_cache"])
    rows = pd.read_csv(cache / "test" / "rows.csv")
    with np.load(cache / "test" / "arrays.npz", allow_pickle=False) as archive:
        target = np.asarray(archive["target_steps"], dtype=np.float64)[:, 0]
    with np.load(fold / "v97" / "v97_predictions.npz", allow_pickle=False) as archive:
        predictions = {
            "v97": np.asarray(archive["v97_direct__prediction"], dtype=np.float64),
            "constant_velocity": np.asarray(archive["baseline__constant_velocity"], dtype=np.float64),
            "v52_rolling": np.asarray(archive["baseline__v52_rolling"], dtype=np.float64),
            "kalman_cv": np.asarray(archive["baseline__kalman_cv"], dtype=np.float64),
            "imm_cv_ca_turn": np.asarray(archive["baseline__imm_cv_ca_turn"], dtype=np.float64),
        }
    keys = rows[["sequence", "frame", "track_id"]].to_numpy(dtype=np.int64)
    if len(rows) != len(target) or any(len(value) != len(rows) for value in predictions.values()):
        raise RuntimeError(f"Row/array length mismatch in {fold}")
    return rows, keys, target, predictions, ready


def directed_knn_pairs(position: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(position)
    if count < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    actual = min(k, count - 1)
    distance, index = cKDTree(position).query(position, k=actual + 1)
    source = np.repeat(np.arange(count, dtype=np.int64), actual)
    target = np.asarray(index[:, 1 : actual + 1], dtype=np.int64).reshape(-1)
    distance = np.asarray(distance[:, 1 : actual + 1], dtype=np.float64).reshape(-1)
    return source, target, distance


def pair_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) == 0:
        return {"dot": np.nan, "cosine": np.nan, "component_corr": np.nan}
    dot = np.einsum("ij,ij->i", left, right)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid_cos = denom > 1e-12
    cosine = float(np.mean(dot[valid_cos] / denom[valid_cos])) if valid_cos.any() else np.nan
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    if np.std(flat_left) > 1e-12 and np.std(flat_right) > 1e-12:
        corr = float(np.corrcoef(flat_left, flat_right)[0, 1])
    else:
        corr = np.nan
    return {"dot": float(np.mean(dot)), "cosine": cosine, "component_corr": corr}


def movie_method_metrics(
    rows: pd.DataFrame,
    error: np.ndarray,
    movie: int,
    method: str,
    neighbor_ks: list[int],
    random_repeats: int,
    rng: np.random.Generator,
) -> list[dict]:
    records: list[dict] = []
    total_energy = float(np.mean(np.sum(error * error, axis=1)))
    grouped_indices = rows.groupby("frame", sort=True).indices
    frame_means = np.stack([error[np.asarray(idx)].mean(axis=0) for idx in grouped_indices.values()])
    frame_mean_fraction = float(np.mean(np.sum(frame_means * frame_means, axis=1)) / max(total_energy, 1e-12))
    for k in neighbor_ks:
        observed_left: list[np.ndarray] = []
        observed_right: list[np.ndarray] = []
        observed_distance: list[np.ndarray] = []
        random_left: list[np.ndarray] = []
        random_right: list[np.ndarray] = []
        for indices in grouped_indices.values():
            indices = np.asarray(indices, dtype=np.int64)
            position = rows.iloc[indices][["x_px", "y_px"]].to_numpy(dtype=np.float64)
            source, target, distance = directed_knn_pairs(position, k)
            if not len(source):
                continue
            local_error = error[indices]
            observed_left.append(local_error[source])
            observed_right.append(local_error[target])
            observed_distance.append(distance)
            count = len(local_error)
            for _ in range(random_repeats):
                random_target = rng.integers(0, count - 1, size=len(source), endpoint=False)
                random_target += random_target >= source
                random_left.append(local_error[source])
                random_right.append(local_error[random_target])
        if not observed_left:
            continue
        left = np.concatenate(observed_left)
        right = np.concatenate(observed_right)
        random_l = np.concatenate(random_left)
        random_r = np.concatenate(random_right)
        observed = pair_stats(left, right)
        control = pair_stats(random_l, random_r)
        records.append(
            {
                "movie": movie,
                "method": method,
                "neighbor_k": k,
                "n_rows": len(rows),
                "n_directed_neighbor_pairs": len(left),
                "mean_neighbor_distance_px": float(np.mean(np.concatenate(observed_distance))),
                "error_energy": total_energy,
                "frame_mean_energy_fraction": frame_mean_fraction,
                "neighbor_dot": observed["dot"],
                "random_same_frame_dot": control["dot"],
                "neighbor_dot_excess": observed["dot"] - control["dot"],
                "neighbor_dot_excess_normalized": (observed["dot"] - control["dot"]) / max(total_energy, 1e-12),
                "neighbor_cosine": observed["cosine"],
                "random_same_frame_cosine": control["cosine"],
                "neighbor_cosine_excess": observed["cosine"] - control["cosine"],
                "neighbor_component_corr": observed["component_corr"],
                "random_same_frame_component_corr": control["component_corr"],
                "neighbor_component_corr_excess": observed["component_corr"] - control["component_corr"],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    movies = parse_ints(args.movies)
    seeds = parse_ints(args.seeds)
    neighbor_ks = parse_ints(args.neighbor_k)
    if not movies or not seeds or not neighbor_ks:
        raise SystemExit("movies, seeds, and neighbor-k must be non-empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    records: list[dict] = []
    integrity: list[dict] = []
    for movie in movies:
        seed_payloads = [load_seed(args.v102_root, movie, seed) for seed in seeds]
        rows, keys, target, _, _ = seed_payloads[0]
        for seed, (_, other_keys, other_target, _, ready) in zip(seeds, seed_payloads):
            integrity.extend(
                [
                    {
                        "movie": movie,
                        "seed": seed,
                        "check": "same_ordered_keys_across_seeds",
                        "passed": bool(np.array_equal(keys, other_keys)),
                        "detail": sha256_array(other_keys),
                    },
                    {
                        "movie": movie,
                        "seed": seed,
                        "check": "same_h1_targets_across_seeds",
                        "passed": bool(np.array_equal(target, other_target)),
                        "detail": sha256_array(other_target),
                    },
                    {
                        "movie": movie,
                        "seed": seed,
                        "check": "heldout_movie_exact",
                        "passed": ready.get("test_movies") == [movie],
                        "detail": json.dumps(ready.get("test_movies")),
                    },
                ]
            )
        if not all(item["passed"] for item in integrity if item["movie"] == movie):
            raise RuntimeError(f"Integrity failure for movie {movie}")
        methods = seed_payloads[0][3].keys()
        for method in methods:
            prediction = np.mean([payload[3][method] for payload in seed_payloads], axis=0)
            error = target - prediction
            records.extend(
                movie_method_metrics(
                    rows,
                    error,
                    movie,
                    method,
                    neighbor_ks,
                    args.random_repeats,
                    rng,
                )
            )

    per_movie = pd.DataFrame(records).sort_values(["neighbor_k", "method", "movie"])
    numeric = [
        column
        for column in per_movie.columns
        if column not in {"movie", "method", "neighbor_k"} and pd.api.types.is_numeric_dtype(per_movie[column])
    ]
    aggregate = per_movie.groupby(["method", "neighbor_k"], as_index=False)[numeric].agg(["mean", "std"])
    aggregate.columns = ["_".join(token for token in column if token) for column in aggregate.columns.to_flat_index()]
    aggregate = aggregate.rename(columns={"method_": "method", "neighbor_k_": "neighbor_k"})
    integrity_frame = pd.DataFrame(integrity)
    per_movie.to_csv(args.out_dir / "v139_spatial_innovation_per_movie.csv", index=False)
    aggregate.to_csv(args.out_dir / "v139_spatial_innovation_aggregate.csv", index=False)
    integrity_frame.to_csv(args.out_dir / "v139_integrity.csv", index=False)

    focus = aggregate[aggregate["neighbor_k"].eq(8)].copy()
    focus = focus.sort_values("neighbor_dot_excess_normalized_mean", ascending=False)
    lines = [
        "# v139 Spatial Innovation Audit",
        "",
        "This is an offline target-aware diagnostic, not an inference feature.",
        "Optimizer seeds are averaged inside each held-out movie; the six movies are the independent units.",
        "The control pairs each cell error with random other cells from the same frame.",
        "",
        "## k=8 movie-level aggregate",
        "",
        focus[
            [
                "method",
                "neighbor_dot_excess_normalized_mean",
                "neighbor_dot_excess_normalized_std",
                "neighbor_cosine_excess_mean",
                "neighbor_component_corr_excess_mean",
                "frame_mean_energy_fraction_mean",
            ]
        ].to_markdown(index=False),
        "",
        "Positive excess means nearby cells retain more similarly directed forecast error than random cells observed in the same frame.",
        "It supports a joint local residual law only if the effect is stable across movies and remains after the same-frame control.",
    ]
    (args.out_dir / "v139_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "ok": bool(integrity_frame["passed"].all()),
        "diagnostic_only": True,
        "target_used_for_diagnostic": True,
        "movies": movies,
        "seeds": seeds,
        "neighbor_k": neighbor_ks,
        "random_repeats": args.random_repeats,
        "v102_root": str(args.v102_root.resolve()),
    }
    (args.out_dir / "v139_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
