#!/usr/bin/env python3
"""v197 equivariant innovation-field law on measured MDCK mechanics.

This runner tests a physical field-law hypothesis that was not covered by the
earlier tabular mechanics models.  It:

1. reconstructs derivatives on the full displacement grid before sampling;
2. fits one scalar coefficient per vector operator, shared by x and y;
3. uses whole-island nested holdouts;
4. evaluates real mechanics against temporal, spatial, and wrong-island nulls;
5. tests density/drug transfer, dynamic spectra, and the reverse motion-to-force
   bridge.

Future mechanics columns are forbidden from every inference feature.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lit_cell_forecasting.equivariant_field_law import (  # noqa: E402
    VectorOperatorModel,
    build_equivariant_library,
    fit_shared_vector_ridge,
    gradient_divergence,
    vector_laplacian,
    vector_r2,
    vector_rmse,
)


DEFAULT_CACHE = (
    ROOT
    / "outputs/mdck_measured_mechanics_upper_bound_v151_full22_2026-07-24"
    / "v151_field_samples.pkl.gz"
)
DEFAULT_DATA_ROOT = ROOT / "data/external/mdck_force_motion" / "extracted"
DEFAULT_OUT = ROOT / "outputs/mdck_equivariant_field_law_v197_2026-07-30"
PIXEL_SIZE_UM = 0.66
FRAME_INTERVAL_MIN = 10.0
EPS = 1e-12

KINEMATIC_TERMS = (
    "u_prev",
    "lap_u",
    "grad_div_u",
    "advect_v_u",
    "advect_u_u",
    "cubic_u",
    "boundary_normal",
)
MECHANICS_TERMS = ("traction", "stress_div", "force_balance")
VELOCITY_BRIDGE_TERMS = ("v_prev", "lap_v", "grad_div_v", "boundary_normal")
INNOVATION_BRIDGE_TERMS = ("u_prev", "lap_u", "grad_div_u", "advect_v_u")

VARIANTS: dict[str, tuple[str, ...]] = {
    "relaxation": ("u_prev",),
    "isotropic_pde": ("u_prev", "lap_u"),
    "helmholtz_pde": ("u_prev", "lap_u", "grad_div_u"),
    "advective_pde": (
        "u_prev",
        "lap_u",
        "grad_div_u",
        "advect_v_u",
        "advect_u_u",
        "cubic_u",
    ),
    "boundary_pde": (
        "u_prev",
        "lap_u",
        "grad_div_u",
        "advect_v_u",
        "advect_u_u",
        "cubic_u",
        "boundary_normal",
    ),
    "mechanics_source_pde": (
        "u_prev",
        "lap_u",
        "grad_div_u",
        "traction",
        "stress_div",
        "force_balance",
    ),
    "full_sparse_pde": KINEMATIC_TERMS + MECHANICS_TERMS,
}


def parse_float_list(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--derived-cache", type=Path, default=None)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--thresholds", default="0,0.01,0.03")
    parser.add_argument("--smoothing-sigma", type=float, default=0.75)
    parser.add_argument("--boundary-decay-um", type=float, default=75.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild-derived", action="store_true")
    return parser.parse_args()


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(table: pd.DataFrame) -> pd.Series:
    return table["condition"].astype(str) + "/" + table["island"].astype(str)


def discover_islands(data_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(data_root.rglob("cell_displacements.mat")):
        relative = path.relative_to(data_root)
        condition = relative.parts[0]
        key = f"{condition}/{path.parent.name}"
        result[key] = path.parent
    return result


def _time_last(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D/3D field, observed {array.shape}")
    return array


def sample_image(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(x).astype(np.int64) - 1, 0, image.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(np.int64) - 1, 0, image.shape[0] - 1)
    return np.asarray(image)[yi, xi]


def aligned_displacement_fields(
    condition: str,
    island_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    import tifffile

    cell = loadmat(
        island_dir / "cell_displacements.mat",
        variable_names=["x", "y", "u", "v"],
    )
    domain = tifffile.imread(island_dir / "domain.tif")
    cell_x = np.asarray(cell["x"], dtype=np.float64)
    cell_y = np.asarray(cell["y"], dtype=np.float64)
    target_x = _time_last(cell["u"]) * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN
    target_y = _time_last(cell["v"]) * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN
    timeline_frames = int(len(domain))
    raw_start = timeline_frames - int(target_x.shape[-1]) - 1
    canonical_start = (
        6 if condition in {"cytod", "cn03_1_4", "cn03_5_8"} else 0
    )
    array_start = canonical_start - raw_start
    if array_start < 0 or array_start >= target_x.shape[-1]:
        raise ValueError(f"Cannot align {condition}/{island_dir.name}")
    target_x = target_x[..., array_start:]
    target_y = target_y[..., array_start:]
    target_start = raw_start + array_start
    return cell_x, cell_y, target_x, target_y, np.asarray(domain), target_start


def _mechanics_from_rows(frame_rows: pd.DataFrame) -> dict[str, np.ndarray]:
    count = len(frame_rows)
    mechanics: dict[str, np.ndarray] = {}
    for name, columns in {
        "traction": ("traction_x", "traction_y"),
        "stress_div": ("stress_div_x", "stress_div_y"),
        "force_balance": ("force_balance_x", "force_balance_y"),
    }.items():
        value = frame_rows.loc[:, list(columns)].to_numpy(np.float64)
        if value.shape != (count, 2):
            raise ValueError(f"Malformed mechanics packet {name}")
        mechanics[name] = value
    return mechanics


def build_derived_table(
    table: pd.DataFrame,
    data_root: Path,
    *,
    smoothing_sigma: float,
    boundary_decay_um: float,
) -> pd.DataFrame:
    """Compute full-grid derivatives and sample them at cached valid points."""

    forbidden = [
        column
        for column in table.columns
        if column.startswith("future__") and column in KINEMATIC_TERMS
    ]
    if forbidden:
        raise ValueError(f"Future mechanics entered the causal library: {forbidden}")
    islands = discover_islands(data_root)
    result = table[
        [
            "condition",
            "island",
            "frame",
            "issue_frame",
            "grid_y",
            "grid_x",
            "target_x",
            "target_y",
            "base_x",
            "base_y",
            "innovation_x",
            "innovation_y",
            "vel_accel_x",
            "vel_accel_y",
            "traction_x",
            "traction_y",
            "stress_div_x",
            "stress_div_y",
            "force_balance_x",
            "force_balance_y",
            "is_treated",
        ]
    ].copy()
    result["group"] = group_key(result)
    vector_names = list(KINEMATIC_TERMS) + list(MECHANICS_TERMS) + [
        "v_prev",
        "lap_v",
        "grad_div_v",
    ]
    values = {
        name: np.full((len(result), 2), np.nan, dtype=np.float64)
        for name in vector_names
    }
    spacing_values = np.full(len(result), np.nan, dtype=np.float64)
    reconstruction_errors: list[float] = []

    for key, group_rows in result.groupby("group", sort=True):
        if key not in islands:
            raise FileNotFoundError(f"No raw island directory for {key}")
        condition = str(group_rows.iloc[0]["condition"])
        (
            cell_x,
            cell_y,
            target_x,
            target_y,
            domain,
            target_start,
        ) = aligned_displacement_fields(condition, islands[key])
        spacing_um = float(np.median(np.diff(cell_x[0, :]))) * PIXEL_SIZE_UM
        if not np.isfinite(spacing_um) or spacing_um <= 0:
            raise ValueError(f"Invalid spatial spacing for {key}: {spacing_um}")

        for frame, frame_rows in group_rows.groupby("frame", sort=True):
            frame = int(frame)
            if frame < 2 or frame >= target_x.shape[-1]:
                raise ValueError(f"Invalid target frame {frame} for {key}")
            issue_values = frame_rows["issue_frame"].unique()
            if len(issue_values) != 1:
                raise ValueError(f"Non-unique issue frame for {key}/{frame}")
            issue_frame = int(issue_values[0])
            if issue_frame != target_start + frame:
                raise ValueError(f"Timeline mismatch for {key}/{frame}")

            velocity = np.stack(
                [target_x[..., frame - 1], target_y[..., frame - 1]],
                axis=-1,
            )
            previous_velocity = np.stack(
                [target_x[..., frame - 2], target_y[..., frame - 2]],
                axis=-1,
            )
            previous_innovation = velocity - previous_velocity
            boundary_px = distance_transform_edt(np.asarray(domain[issue_frame]) > 0)
            boundary_distance = sample_image(boundary_px, cell_x, cell_y) * PIXEL_SIZE_UM
            library = build_equivariant_library(
                previous_innovation,
                velocity,
                boundary_distance,
                spacing=spacing_um,
                smoothing_sigma=smoothing_sigma,
                boundary_decay=boundary_decay_um,
            )
            library["v_prev"] = velocity
            library["lap_v"] = vector_laplacian(velocity, spacing_um)
            library["grad_div_v"] = gradient_divergence(velocity, spacing_um)

            row_index = frame_rows.index.to_numpy(np.int64)
            yy = frame_rows["grid_y"].to_numpy(np.int64)
            xx = frame_rows["grid_x"].to_numpy(np.int64)
            for name in KINEMATIC_TERMS + ("v_prev", "lap_v", "grad_div_v"):
                values[name][row_index] = library[name][yy, xx]
            for name, packet in _mechanics_from_rows(frame_rows).items():
                values[name][row_index] = packet
            spacing_values[row_index] = spacing_um
            reconstructed = previous_innovation[yy, xx]
            cached = frame_rows[["vel_accel_x", "vel_accel_y"]].to_numpy(np.float64)
            reconstruction_errors.append(float(np.max(np.abs(reconstructed - cached))))

    if not all(np.isfinite(value).all() for value in values.values()):
        broken = [name for name, value in values.items() if not np.isfinite(value).all()]
        raise FloatingPointError(f"Non-finite derived vector fields: {broken}")
    if max(reconstruction_errors, default=0.0) > 2e-5:
        raise ValueError(
            "Raw/cache alignment failed: max innovation mismatch "
            f"{max(reconstruction_errors):.6g}"
        )
    for name, value in values.items():
        result[f"{name}_x"] = value[:, 0].astype(np.float32)
        result[f"{name}_y"] = value[:, 1].astype(np.float32)
    result["grid_spacing_um"] = spacing_values.astype(np.float32)
    result["raw_cache_alignment_max_abs"] = max(reconstruction_errors, default=0.0)
    return result


def load_or_build_derived(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    derived_path = args.derived_cache
    if derived_path is None:
        derived_path = args.out_dir / "v197_operator_samples.pkl.gz"
    if derived_path.exists() and not args.rebuild_derived:
        with gzip.open(derived_path, "rb") as handle:
            table = pickle.load(handle)
        return table, derived_path
    with gzip.open(args.cache, "rb") as handle:
        base = pickle.load(handle)
    base = base.reset_index(drop=True)
    table = build_derived_table(
        base,
        args.data_root,
        smoothing_sigma=args.smoothing_sigma,
        boundary_decay_um=args.boundary_decay_um,
    )
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(derived_path, "wb") as handle:
        pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return table, derived_path


def vector_terms(table: pd.DataFrame) -> dict[str, np.ndarray]:
    names = (
        list(KINEMATIC_TERMS)
        + list(MECHANICS_TERMS)
        + ["v_prev", "lap_v", "grad_div_v"]
    )
    return {
        name: table[[f"{name}_x", f"{name}_y"]].to_numpy(np.float64)
        for name in names
    }


def make_family_control(
    terms: Mapping[str, np.ndarray],
    table: pd.DataFrame,
    control: str,
    *,
    seed: int,
    controlled_names: Sequence[str],
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in terms.items()}
    if control == "real":
        return result
    rng = np.random.default_rng(seed)
    row_groups = table["group"].astype(str).to_numpy()
    frames = table["frame"].to_numpy(np.int64)
    grid_y = table["grid_y"].to_numpy(np.int64)
    grid_x = table["grid_x"].to_numpy(np.int64)
    for name in controlled_names:
        source = np.asarray(terms[name])
        transformed = np.empty_like(source)
        if control == "row_shuffled":
            transformed[:] = source[rng.permutation(len(source))]
        elif control == "wrong_island":
            unique = sorted(np.unique(row_groups))
            donor = {group: unique[(index + 1) % len(unique)] for index, group in enumerate(unique)}
            for group in unique:
                target_index = np.flatnonzero(row_groups == group)
                donor_index = np.flatnonzero(row_groups == donor[group])
                select = np.linspace(
                    0,
                    max(len(donor_index) - 1, 0),
                    len(target_index),
                ).astype(np.int64)
                transformed[target_index] = source[donor_index[select]]
        elif control == "time_shuffled":
            transformed[:] = source
            key_frame = pd.DataFrame(
                {
                    "group": row_groups,
                    "frame": frames,
                    "grid_y": grid_y,
                    "grid_x": grid_x,
                    "row": np.arange(len(table)),
                }
            )
            for _, spatial_rows in key_frame.groupby(
                ["group", "grid_y", "grid_x"],
                sort=False,
            ):
                index = spatial_rows.sort_values("frame")["row"].to_numpy(np.int64)
                if len(index) > 1:
                    transformed[index] = source[np.roll(index, 1)]
        elif control == "spatial_shifted":
            transformed[:] = source
            key_frame = pd.DataFrame(
                {
                    "group": row_groups,
                    "frame": frames,
                    "grid_y": grid_y,
                    "grid_x": grid_x,
                    "row": np.arange(len(table)),
                }
            )
            for _, frame_rows in key_frame.groupby(["group", "frame"], sort=False):
                ordered = frame_rows.sort_values(["grid_y", "grid_x"])["row"].to_numpy(np.int64)
                if len(ordered) > 1:
                    shift = max(1, len(ordered) // 7)
                    transformed[ordered] = source[np.roll(ordered, shift)]
        else:
            raise ValueError(f"Unknown mechanics control: {control}")
        result[name] = transformed
    return result


def make_mechanics_control(
    terms: Mapping[str, np.ndarray],
    table: pd.DataFrame,
    control: str,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    return make_family_control(
        terms,
        table,
        control,
        seed=seed,
        controlled_names=MECHANICS_TERMS,
    )


def time_shift_vectors(
    value: np.ndarray,
    table: pd.DataFrame,
) -> np.ndarray:
    """Shift a vector field by one completed frame at fixed grid locations."""

    transformed = np.asarray(value).copy()
    key = pd.DataFrame(
        {
            "group": table["group"].astype(str).to_numpy(),
            "frame": table["frame"].to_numpy(),
            "grid_y": table["grid_y"].to_numpy(),
            "grid_x": table["grid_x"].to_numpy(),
            "row": np.arange(len(table)),
        }
    )
    for _, spatial_rows in key.groupby(
        ["group", "grid_y", "grid_x"],
        sort=False,
    ):
        index = spatial_rows.sort_values("frame")["row"].to_numpy(np.int64)
        if len(index) > 1:
            transformed[index] = value[np.roll(index, 1)]
    return transformed


def choose_validation_group(
    table: pd.DataFrame,
    available_groups: Sequence[str],
    outer_group: str,
) -> str:
    outer_condition = outer_group.split("/", 1)[0]
    same = sorted(
        group
        for group in available_groups
        if group != outer_group and group.startswith(f"{outer_condition}/")
    )
    candidates = same or sorted(group for group in available_groups if group != outer_group)
    if not candidates:
        raise ValueError("No validation group remains")
    digest = hashlib.sha256(outer_group.encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:4], "little") % len(candidates)]


def tune_and_fit(
    terms: Mapping[str, np.ndarray],
    target: np.ndarray,
    term_names: Sequence[str],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    alphas: Sequence[float],
    thresholds: Sequence[float],
) -> tuple[VectorOperatorModel, dict[str, float]]:
    best: tuple[float, float, float] | None = None
    names = tuple(term_names)
    for alpha in alphas:
        for threshold in thresholds:
            model = fit_shared_vector_ridge(
                {name: terms[name][train_index] for name in names},
                target[train_index],
                names,
                alpha=alpha,
                threshold=threshold,
            )
            prediction = model.predict(
                {name: terms[name][validation_index] for name in names}
            )
            score = vector_rmse(prediction, target[validation_index])
            candidate = (score, float(alpha), float(threshold))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    combined = np.concatenate([train_index, validation_index])
    model = fit_shared_vector_ridge(
        {name: terms[name][combined] for name in names},
        target[combined],
        names,
        alpha=best[1],
        threshold=best[2],
    )
    return model, {
        "validation_rmse": best[0],
        "alpha": best[1],
        "threshold": best[2],
    }


def metrics(
    correction: np.ndarray,
    target_innovation: np.ndarray,
    base: np.ndarray,
    target_displacement: np.ndarray,
) -> dict[str, float]:
    displacement = base + correction
    dot = np.sum(displacement * target_displacement, axis=1)
    denominator = np.linalg.norm(displacement, axis=1) * np.linalg.norm(
        target_displacement,
        axis=1,
    )
    cosine = np.divide(
        dot,
        np.maximum(denominator, EPS),
        out=np.zeros_like(dot),
        where=denominator > EPS,
    )
    return {
        "innovation_rmse": vector_rmse(correction, target_innovation),
        "displacement_rmse": vector_rmse(displacement, target_displacement),
        "displacement_r2": vector_r2(displacement, target_displacement),
        "angular_cosine": float(np.mean(cosine)),
        "magnitude_ratio": float(
            np.mean(np.linalg.norm(displacement, axis=1))
            / max(np.mean(np.linalg.norm(target_displacement, axis=1)), EPS)
        ),
    }


def evaluate_outer_folds(
    table: pd.DataFrame,
    terms_by_control: Mapping[str, Mapping[str, np.ndarray]],
    *,
    alphas: Sequence[float],
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    groups = table["group"].astype(str).to_numpy()
    unique_groups = sorted(np.unique(groups))
    target_innovation = table[["innovation_x", "innovation_y"]].to_numpy(np.float64)
    target_displacement = table[["target_x", "target_y"]].to_numpy(np.float64)
    base = table[["base_x", "base_y"]].to_numpy(np.float64)
    records: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    predictions = {
        variant: np.full_like(target_innovation, np.nan)
        for variant in ["cv", "persistence", *VARIANTS]
    }

    for outer_group in unique_groups:
        validation_group = choose_validation_group(table, unique_groups, outer_group)
        test_index = np.flatnonzero(groups == outer_group)
        validation_index = np.flatnonzero(groups == validation_group)
        train_index = np.flatnonzero(
            (groups != outer_group) & (groups != validation_group)
        )
        controls_for_variant = {
            "cv": ("real",),
            "persistence": ("real",),
            "relaxation": (
                "real",
                "kin_time_shuffled",
                "kin_spatial_shifted",
                "kin_wrong_island",
            ),
            "isotropic_pde": ("real",),
            "helmholtz_pde": (
                "real",
                "kin_time_shuffled",
                "kin_spatial_shifted",
                "kin_wrong_island",
            ),
            "advective_pde": (
                "real",
                "kin_time_shuffled",
                "kin_spatial_shifted",
                "kin_wrong_island",
            ),
            "boundary_pde": ("real",),
            "mechanics_source_pde": tuple(terms_by_control),
            "full_sparse_pde": tuple(terms_by_control),
        }
        base_rmse = vector_rmse(base[test_index], target_displacement[test_index])
        for variant, controls in controls_for_variant.items():
            for control in controls:
                terms = terms_by_control[control]
                tuning = {
                    "validation_rmse": math.nan,
                    "alpha": math.nan,
                    "threshold": math.nan,
                }
                model: VectorOperatorModel | None = None
                if variant == "cv":
                    correction = np.zeros((len(test_index), 2), dtype=np.float64)
                elif variant == "persistence":
                    correction = terms["u_prev"][test_index]
                else:
                    model, tuning = tune_and_fit(
                        terms,
                        target_innovation,
                        VARIANTS[variant],
                        train_index,
                        validation_index,
                        alphas=alphas,
                        thresholds=thresholds,
                    )
                    correction = model.predict(
                        {
                            name: terms[name][test_index]
                            for name in model.term_names
                        }
                    )
                result = metrics(
                    correction,
                    target_innovation[test_index],
                    base[test_index],
                    target_displacement[test_index],
                )
                records.append(
                    {
                        "outer_group": outer_group,
                        "validation_group": validation_group,
                        "variant": variant,
                        "control": control,
                        "n_test": len(test_index),
                        **tuning,
                        **result,
                        "gain_vs_cv_percent": 100.0
                        * (base_rmse - result["displacement_rmse"])
                        / max(base_rmse, EPS),
                    }
                )
                if control == "real":
                    predictions[variant][test_index] = correction
                if model is not None:
                    for name, coefficient, scale in zip(
                        model.term_names,
                        model.coefficients,
                        model.term_scales,
                        strict=True,
                    ):
                        coefficients.append(
                            {
                                "outer_group": outer_group,
                                "variant": variant,
                                "control": control,
                                "term": name,
                                "coefficient_standardized": coefficient,
                                "term_scale": scale,
                                "coefficient_physical": coefficient / max(scale, EPS),
                                **tuning,
                            }
                        )
    return pd.DataFrame(records), pd.DataFrame(coefficients), predictions


def evaluate_mechanical_bridge(
    table: pd.DataFrame,
    terms_by_control: Mapping[str, Mapping[str, np.ndarray]],
    *,
    alphas: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    groups = table["group"].astype(str).to_numpy()
    unique_groups = sorted(np.unique(groups))
    targets = {
        "traction": table[["traction_x", "traction_y"]].to_numpy(np.float64),
        "stress_div": table[["stress_div_x", "stress_div_y"]].to_numpy(np.float64),
        "force_balance": table[
            ["force_balance_x", "force_balance_y"]
        ].to_numpy(np.float64),
    }
    shifted_innovation_terms = {
        name: time_shift_vectors(terms_by_control["real"][name], table)
        for name in INNOVATION_BRIDGE_TERMS
    }
    records: list[dict[str, Any]] = []
    for outer_group in unique_groups:
        validation_group = choose_validation_group(table, unique_groups, outer_group)
        test_index = np.flatnonzero(groups == outer_group)
        validation_index = np.flatnonzero(groups == validation_group)
        train_index = np.flatnonzero(
            (groups != outer_group) & (groups != validation_group)
        )
        for target_name, target in targets.items():
            baseline_prediction = None
            for variant, names, control in [
                ("velocity_only", VELOCITY_BRIDGE_TERMS, "real"),
                (
                    "velocity_plus_innovation",
                    VELOCITY_BRIDGE_TERMS + INNOVATION_BRIDGE_TERMS,
                    "real",
                ),
                (
                    "velocity_plus_time_shuffled_innovation",
                    VELOCITY_BRIDGE_TERMS + INNOVATION_BRIDGE_TERMS,
                    "time_shuffled",
                ),
            ]:
                terms = dict(terms_by_control["real"])
                if control != "real":
                    for name in INNOVATION_BRIDGE_TERMS:
                        terms[name] = shifted_innovation_terms[name]
                model, tuning = tune_and_fit(
                    terms,
                    target,
                    names,
                    train_index,
                    validation_index,
                    alphas=alphas,
                    thresholds=thresholds,
                )
                prediction = model.predict(
                    {name: terms[name][test_index] for name in model.term_names}
                )
                rmse = vector_rmse(prediction, target[test_index])
                if variant == "velocity_only":
                    baseline_prediction = rmse
                records.append(
                    {
                        "outer_group": outer_group,
                        "target": target_name,
                        "variant": variant,
                        "n_test": len(test_index),
                        "rmse": rmse,
                        "r2": vector_r2(prediction, target[test_index]),
                        "gain_vs_velocity_percent": (
                            0.0
                            if baseline_prediction is None
                            else 100.0
                            * (baseline_prediction - rmse)
                            / max(baseline_prediction, EPS)
                        ),
                        **tuning,
                    }
                )
    return pd.DataFrame(records)


def evaluate_interventions(
    table: pd.DataFrame,
    terms_by_control: Mapping[str, Mapping[str, np.ndarray]],
    *,
    alphas: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    groups = table["group"].astype(str).to_numpy()
    condition = table["condition"].astype(str).to_numpy()
    treated = table["is_treated"].to_numpy(float) > 0.5
    target_innovation = table[["innovation_x", "innovation_y"]].to_numpy(np.float64)
    target_displacement = table[["target_x", "target_y"]].to_numpy(np.float64)
    base = table[["base_x", "base_y"]].to_numpy(np.float64)
    split_masks = {
        "low_to_high_density": (
            condition == "low_density",
            condition == "high_density",
        ),
        "high_to_low_density": (
            condition == "high_density",
            condition == "low_density",
        ),
        "untreated_to_treated": (~treated, treated),
        "cn03_phase1_to_phase2": (
            condition == "cn03_1_4",
            condition == "cn03_5_8",
        ),
        "cytod_control_to_treated": (
            (condition == "cytod") & (~treated),
            (condition == "cytod") & treated,
        ),
    }
    records: list[dict[str, Any]] = []
    for split_name, (source_mask, target_mask) in split_masks.items():
        source_groups = sorted(np.unique(groups[source_mask]))
        target_groups = sorted(np.unique(groups[target_mask]))
        if len(source_groups) < 2 or not target_groups:
            continue
        validation_group = source_groups[-1]
        train_index = np.flatnonzero(source_mask & (groups != validation_group))
        validation_index = np.flatnonzero(source_mask & (groups == validation_group))
        for target_group in target_groups:
            test_index = np.flatnonzero(target_mask & (groups == target_group))
            cv_rmse = vector_rmse(base[test_index], target_displacement[test_index])
            for variant in (
                "relaxation",
                "helmholtz_pde",
                "boundary_pde",
                "mechanics_source_pde",
            ):
                controls = (
                    ("real", "time_shuffled", "spatial_shifted", "wrong_island")
                    if variant == "mechanics_source_pde"
                    else ("real",)
                )
                for control in controls:
                    terms = terms_by_control[control]
                    model, tuning = tune_and_fit(
                        terms,
                        target_innovation,
                        VARIANTS[variant],
                        train_index,
                        validation_index,
                        alphas=alphas,
                        thresholds=thresholds,
                    )
                    correction = model.predict(
                        {
                            name: terms[name][test_index]
                            for name in model.term_names
                        }
                    )
                    result = metrics(
                        correction,
                        target_innovation[test_index],
                        base[test_index],
                        target_displacement[test_index],
                    )
                    records.append(
                        {
                            "split": split_name,
                            "source_groups": len(source_groups),
                            "target_group": target_group,
                            "variant": variant,
                            "control": control,
                            **tuning,
                            **result,
                            "gain_vs_cv_percent": 100.0
                            * (cv_rmse - result["displacement_rmse"])
                            / max(cv_rmse, EPS),
                        }
                    )
    return pd.DataFrame(records)


def dynamic_spectra(
    table: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    variants: Sequence[str],
    *,
    radial_bins: int = 6,
    temporal_bins: int = 4,
) -> pd.DataFrame:
    """Estimate held-out longitudinal/transverse dynamic power spectra."""

    rows: list[dict[str, Any]] = []
    observed_all = table[["innovation_x", "innovation_y"]].to_numpy(np.float64)
    for group, group_rows in table.groupby("group", sort=True):
        index = group_rows.index.to_numpy(np.int64)
        frame_values = np.sort(group_rows["frame"].unique())
        y_values = np.sort(group_rows["grid_y"].unique())
        x_values = np.sort(group_rows["grid_x"].unique())
        if len(frame_values) < 5 or len(y_values) < 5 or len(x_values) < 5:
            continue
        frame_map = {value: item for item, value in enumerate(frame_values)}
        y_map = {value: item for item, value in enumerate(y_values)}
        x_map = {value: item for item, value in enumerate(x_values)}
        shape = (len(frame_values), len(y_values), len(x_values), 2)
        mask = np.zeros(shape[:-1], dtype=bool)

        def field_from_rows(values: np.ndarray) -> np.ndarray:
            field = np.zeros(shape, dtype=np.float64)
            for row, global_index in zip(group_rows.itertuples(), index, strict=True):
                ti = frame_map[row.frame]
                yi = y_map[row.grid_y]
                xi = x_map[row.grid_x]
                field[ti, yi, xi] = values[global_index]
                mask[ti, yi, xi] = True
            for ti in range(shape[0]):
                valid = mask[ti]
                if np.any(valid):
                    field[ti, valid] -= np.mean(field[ti, valid], axis=0)
            return field

        observed = field_from_rows(observed_all)
        sample_stride = float(
            max(
                np.median(np.diff(x_values)) if len(x_values) > 1 else 1.0,
                np.median(np.diff(y_values)) if len(y_values) > 1 else 1.0,
            )
        )
        spacing = (
            float(np.nanmedian(group_rows["grid_spacing_um"])) * sample_stride
        )
        omega = np.fft.fftfreq(shape[0], d=1.0)
        ky = np.fft.fftfreq(shape[1], d=spacing)
        kx = np.fft.fftfreq(shape[2], d=spacing)
        ww, yy, xx = np.meshgrid(omega, ky, kx, indexing="ij")
        radius = np.hypot(xx, yy)
        radius_nonzero = radius[radius > 0]
        if not len(radius_nonzero):
            continue
        radius_edges = np.linspace(0.0, float(np.max(radius_nonzero)), radial_bins + 1)
        temporal_edges = np.linspace(0.0, 0.5, temporal_bins + 1)

        def powers(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            time_window = np.hanning(shape[0]) if shape[0] > 2 else np.ones(shape[0])
            fft = np.fft.fftn(
                field * time_window[:, None, None, None],
                axes=(0, 1, 2),
            )
            unit_x = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0)
            unit_y = np.divide(yy, radius, out=np.zeros_like(yy), where=radius > 0)
            longitudinal = fft[..., 0] * unit_x + fft[..., 1] * unit_y
            transverse = -fft[..., 0] * unit_y + fft[..., 1] * unit_x
            return np.abs(longitudinal) ** 2, np.abs(transverse) ** 2

        observed_l, observed_t = powers(observed)
        for variant in variants:
            predicted = field_from_rows(predictions[variant])
            predicted_l, predicted_t = powers(predicted)
            for radial_index in range(radial_bins):
                for temporal_index in range(temporal_bins):
                    selected = (
                        (radius >= radius_edges[radial_index])
                        & (radius < radius_edges[radial_index + 1])
                        & (np.abs(ww) >= temporal_edges[temporal_index])
                        & (np.abs(ww) < temporal_edges[temporal_index + 1])
                        & (radius > 0)
                    )
                    if not np.any(selected):
                        continue
                    rows.append(
                        {
                            "group": group,
                            "variant": variant,
                            "k_bin": radial_index,
                            "omega_bin": temporal_index,
                            "k_low_um_inv": radius_edges[radial_index],
                            "k_high_um_inv": radius_edges[radial_index + 1],
                            "omega_low_frame_inv": temporal_edges[temporal_index],
                            "omega_high_frame_inv": temporal_edges[temporal_index + 1],
                            "observed_longitudinal_power": float(
                                np.mean(observed_l[selected])
                            ),
                            "predicted_longitudinal_power": float(
                                np.mean(predicted_l[selected])
                            ),
                            "observed_transverse_power": float(
                                np.mean(observed_t[selected])
                            ),
                            "predicted_transverse_power": float(
                                np.mean(predicted_t[selected])
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def spectrum_summary(spectra: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (group, variant), rows in spectra.groupby(["group", "variant"], sort=True):
        correlations = []
        relative_errors = []
        for component in ("longitudinal", "transverse"):
            observed = np.log1p(rows[f"observed_{component}_power"].to_numpy(float))
            predicted = np.log1p(rows[f"predicted_{component}_power"].to_numpy(float))
            correlation = (
                float(np.corrcoef(observed, predicted)[0, 1])
                if np.std(observed) > EPS and np.std(predicted) > EPS
                else math.nan
            )
            error = float(
                np.mean(np.abs(predicted - observed))
                / max(np.mean(np.abs(observed)), EPS)
            )
            correlations.append(correlation)
            relative_errors.append(error)
        finite_correlations = [
            value for value in correlations if np.isfinite(value)
        ]
        records.append(
            {
                "group": group,
                "variant": variant,
                "spectrum_log_correlation": (
                    float(np.mean(finite_correlations))
                    if finite_correlations
                    else math.nan
                ),
                "spectrum_relative_log_error": float(np.mean(relative_errors)),
            }
        )
    return pd.DataFrame(records)


def bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draw = rng.choice(value, size=(int(samples), len(value)), replace=True)
    mean = np.mean(draw, axis=1)
    return float(np.quantile(mean, 0.025)), float(np.quantile(mean, 0.975))


def summarize_outer(
    outer: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (variant, control), rows in outer.groupby(["variant", "control"], sort=True):
        gain = rows["gain_vs_cv_percent"].to_numpy(float)
        low, high = bootstrap_interval(gain, samples=bootstrap, seed=seed)
        positive = int(np.sum(gain > 0))
        sign_p = float(
            binomtest(positive, len(gain), p=0.5, alternative="greater").pvalue
        )
        records.append(
            {
                "variant": variant,
                "control": control,
                "n_islands": len(rows),
                "displacement_rmse_macro": float(rows["displacement_rmse"].mean()),
                "displacement_r2_macro": float(rows["displacement_r2"].mean()),
                "gain_vs_cv_percent_mean": float(np.mean(gain)),
                "gain_vs_cv_percent_ci_low": low,
                "gain_vs_cv_percent_ci_high": high,
                "positive_islands": positive,
                "sign_test_one_sided_p": sign_p,
            }
        )
    return pd.DataFrame(records)


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (variant, control), variant_rows in coefficients.groupby(
        ["variant", "control"],
        sort=True,
    ):
        total_folds = int(variant_rows["outer_group"].nunique())
        for term, rows in variant_rows.groupby("term", sort=True):
            selected_value = rows["coefficient_physical"].to_numpy(float)
            value = np.zeros(total_folds, dtype=np.float64)
            value[: len(selected_value)] = selected_value
            nonzero = np.abs(value) > EPS
            positive = int(np.sum(value[nonzero] > 0))
            negative = int(np.sum(value[nonzero] < 0))
            records.append(
                {
                    "variant": variant,
                    "control": control,
                    "term": term,
                    "n_folds_selected": int(np.sum(nonzero)),
                    "selection_fraction": float(np.mean(nonzero)),
                    "coefficient_median": float(np.median(value)),
                    "coefficient_iqr": float(
                        np.quantile(value, 0.75) - np.quantile(value, 0.25)
                    ),
                    "positive_folds": positive,
                    "negative_folds": negative,
                    "dominant_sign_fraction": float(
                        max(positive, negative) / max(positive + negative, 1)
                    ),
                }
            )
    return pd.DataFrame(records)


def equivariance_audit(
    terms: Mapping[str, np.ndarray],
    target: np.ndarray,
) -> pd.DataFrame:
    rng = np.random.default_rng(197)
    selected = rng.choice(len(target), size=min(len(target), 10000), replace=False)
    names = VARIANTS["full_sparse_pde"]
    model = fit_shared_vector_ridge(
        {name: terms[name][selected] for name in names},
        target[selected],
        names,
        alpha=1.0,
        threshold=0.01,
    )
    angle = 0.731
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    original = model.predict({name: terms[name][selected] for name in model.term_names})
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rotated_terms = {
            name: np.einsum("ni,ji->nj", terms[name][selected], rotation)
            for name in model.term_names
        }
        expected = np.einsum("ni,ji->nj", original, rotation)
    observed = model.predict(rotated_terms)
    maximum = float(np.max(np.abs(observed - expected)))
    relative = float(
        np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), EPS)
    )
    y, x = np.mgrid[-1.0:1.0:41j, -1.0:1.0:41j]
    innovation = np.stack(
        [np.sin(1.7 * x) + 0.2 * y, np.cos(1.3 * y) - 0.1 * x],
        axis=-1,
    )
    velocity = np.stack([0.3 + x * y, -0.2 + x**2], axis=-1)
    distance = 2.0 - np.hypot(x, y)
    original_library = build_equivariant_library(
        innovation,
        velocity,
        distance,
        spacing=2.0 / 40.0,
        smoothing_sigma=0.0,
    )
    rotated_library = build_equivariant_library(
        -np.rot90(innovation, 2, axes=(0, 1)),
        -np.rot90(velocity, 2, axes=(0, 1)),
        np.rot90(distance, 2, axes=(0, 1)),
        spacing=2.0 / 40.0,
        smoothing_sigma=0.0,
    )
    d4_errors = []
    d4_reference = []
    for name in original_library:
        expected_term = -np.rot90(
            original_library[name],
            2,
            axes=(0, 1),
        )[2:-2, 2:-2]
        observed_term = rotated_library[name][2:-2, 2:-2]
        d4_errors.append(np.linalg.norm(observed_term - expected_term))
        d4_reference.append(np.linalg.norm(expected_term))
    d4_relative = float(sum(d4_errors) / max(sum(d4_reference), EPS))
    d4_maximum = float(
        max(
            np.max(
                np.abs(
                    rotated_library[name][2:-2, 2:-2]
                    + np.rot90(
                        original_library[name],
                        2,
                        axes=(0, 1),
                    )[2:-2, 2:-2]
                )
            )
            for name in original_library
        )
    )
    return pd.DataFrame(
        [
            {
                "audit": "algebraic_arbitrary_rotation",
                "angle_radians": angle,
                "max_abs_error": maximum,
                "relative_l2_error": relative,
                "passed_1e_10": bool(relative <= 1e-10),
                "scope": "shared-coefficient algebraic layer",
            },
            {
                "audit": "finite_difference_grid_symmetry",
                "angle_radians": math.pi,
                "max_abs_error": d4_maximum,
                "relative_l2_error": d4_relative,
                "passed_1e_10": bool(d4_relative <= 1e-10),
                "scope": "full finite-difference vector library",
            },
        ]
    )


def synthetic_identifiability(
    *,
    seed: int,
    bootstrap: int,
) -> pd.DataFrame:
    """Recover a known sparse vector law and bootstrap frame-level intervals."""

    rng = np.random.default_rng(seed)
    names = (
        "u_prev",
        "lap_u",
        "grad_div_u",
        "advect_v_u",
        "advect_u_u",
        "cubic_u",
    )
    truth = {
        "u_prev": -0.28,
        "lap_u": 10.0,
        "grad_div_u": 5.0,
        "advect_v_u": -2.0,
        "advect_u_u": 0.2,
        "cubic_u": -0.1,
    }
    frame_terms: list[dict[str, np.ndarray]] = []
    frame_targets: list[np.ndarray] = []
    for _ in range(36):
        innovation = np.stack(
            [
                gaussian_filter(rng.normal(size=(30, 30)), sigma=2.2),
                gaussian_filter(rng.normal(size=(30, 30)), sigma=2.2),
            ],
            axis=-1,
        )
        velocity = np.stack(
            [
                gaussian_filter(rng.normal(size=(30, 30)), sigma=3.0),
                gaussian_filter(rng.normal(size=(30, 30)), sigma=3.0),
            ],
            axis=-1,
        )
        distance = np.full((30, 30), 100.0)
        library = build_equivariant_library(
            innovation,
            velocity,
            distance,
            spacing=1.0,
            smoothing_sigma=0.0,
        )
        target = sum(truth[name] * library[name] for name in names)
        noise_scale = 0.03 * float(np.std(target))
        target = target + noise_scale * rng.standard_t(df=5, size=target.shape)
        frame_terms.append(
            {name: library[name][2:-2, 2:-2].reshape(-1, 2) for name in names}
        )
        frame_targets.append(target[2:-2, 2:-2].reshape(-1, 2))

    def fit_frame_indices(indices: np.ndarray) -> dict[str, float]:
        combined_terms = {
            name: np.concatenate([frame_terms[index][name] for index in indices])
            for name in names
        }
        combined_target = np.concatenate([frame_targets[index] for index in indices])
        model = fit_shared_vector_ridge(
            combined_terms,
            combined_target,
            names,
            alpha=1e-6,
            threshold=0.0,
            clip_quantile=1.0,
        )
        return {
            name: float(coefficient / max(scale, EPS))
            for name, coefficient, scale in zip(
                model.term_names,
                model.coefficients,
                model.term_scales,
                strict=True,
            )
        }

    full = fit_frame_indices(np.arange(len(frame_terms)))
    bootstrap_values = {name: [] for name in names}
    for _ in range(int(bootstrap)):
        indices = rng.choice(
            len(frame_terms),
            size=len(frame_terms),
            replace=True,
        )
        estimate = fit_frame_indices(indices)
        for name in names:
            bootstrap_values[name].append(estimate[name])
    rows = []
    for name in names:
        values = np.asarray(bootstrap_values[name], dtype=np.float64)
        low = float(np.quantile(values, 0.025))
        high = float(np.quantile(values, 0.975))
        true = truth[name]
        rows.append(
            {
                "term": name,
                "true_coefficient": true,
                "estimated_coefficient": full[name],
                "relative_error_percent": 100.0
                * abs(full[name] - true)
                / max(abs(true), EPS),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "covered": bool(low <= true <= high),
                "bootstrap_samples": int(bootstrap),
            }
        )
    return pd.DataFrame(rows)


def build_decision_report(
    outer: pd.DataFrame,
    summary: pd.DataFrame,
    bridge: pd.DataFrame,
    intervention: pd.DataFrame,
    spectra_summary_table: pd.DataFrame,
    equivariance: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    derived_path: Path,
) -> str:
    real = summary[summary["control"].eq("real")].set_index("variant")
    helmholtz_gain = float(real.loc["helmholtz_pde", "gain_vs_cv_percent_mean"])
    advective_gain = float(real.loc["advective_pde", "gain_vs_cv_percent_mean"])
    relaxation_gain = float(real.loc["relaxation", "gain_vs_cv_percent_mean"])
    mechanics_gain = float(real.loc["mechanics_source_pde", "gain_vs_cv_percent_mean"])
    full_gain = float(real.loc["full_sparse_pde", "gain_vs_cv_percent_mean"])
    mechanics_controls = summary[
        summary["variant"].eq("mechanics_source_pde")
    ].set_index("control")
    best_control_gain = float(
        mechanics_controls.drop(index="real", errors="ignore")[
            "gain_vs_cv_percent_mean"
        ].max()
    )
    kinematic_controls = summary[
        summary["variant"].eq("advective_pde")
        & summary["control"].ne("real")
    ]
    best_kinematic_control_gain = float(
        kinematic_controls["gain_vs_cv_percent_mean"].max()
    )
    real_outer = outer[outer["control"].eq("real")]
    paired = real_outer.pivot(
        index="outer_group",
        columns="variant",
        values="displacement_rmse",
    )
    advective_vs_relaxation = 100.0 * (
        paired["relaxation"] - paired["advective_pde"]
    ) / paired["relaxation"].clip(lower=EPS)
    advective_positive = int(np.sum(advective_vs_relaxation > 0))
    advective_sign_p = float(
        binomtest(
            advective_positive,
            len(advective_vs_relaxation),
            p=0.5,
            alternative="greater",
        ).pvalue
    )
    advective_increment = float(np.mean(advective_vs_relaxation))
    bridge_pivot = bridge.pivot_table(
        index=["outer_group", "target"],
        columns="variant",
        values="rmse",
    ).reset_index()
    bridge_pivot["gain"] = 100.0 * (
        bridge_pivot["velocity_only"] - bridge_pivot["velocity_plus_innovation"]
    ) / bridge_pivot["velocity_only"].clip(lower=EPS)
    bridge_gain = float(bridge_pivot["gain"].mean())
    if intervention.empty:
        intervention_mechanics_advantage = math.nan
        intervention_control_advantage = math.nan
    else:
        intervention_key = ["split", "target_group"]
        intervention_real = intervention[
            intervention["control"].eq("real")
        ].pivot_table(
            index=intervention_key,
            columns="variant",
            values="displacement_rmse",
        )
        intervention_mechanics_advantage = float(
            np.mean(
                100.0
                * (
                    intervention_real["helmholtz_pde"]
                    - intervention_real["mechanics_source_pde"]
                )
                / intervention_real["helmholtz_pde"].clip(lower=EPS)
            )
        )
        mechanics_intervention = intervention[
            intervention["variant"].eq("mechanics_source_pde")
        ].pivot_table(
            index=intervention_key,
            columns="control",
            values="displacement_rmse",
        )
        control_minimum = mechanics_intervention[
            [
                column
                for column in mechanics_intervention.columns
                if column != "real"
            ]
        ].min(axis=1)
        intervention_control_advantage = float(
            np.mean(
                100.0
                * (control_minimum - mechanics_intervention["real"])
                / control_minimum.clip(lower=EPS)
            )
        )
    spectrum = spectra_summary_table.groupby("variant", as_index=False).mean(
        numeric_only=True
    )
    spectrum_lookup = spectrum.set_index("variant")
    spectral_improvement_absolute = float(
        spectrum_lookup.loc["helmholtz_pde", "spectrum_relative_log_error"]
        - spectrum_lookup.loc["mechanics_source_pde", "spectrum_relative_log_error"]
    )
    spectral_improvement_percent = 100.0 * spectral_improvement_absolute / max(
        float(spectrum_lookup.loc["helmholtz_pde", "spectrum_relative_log_error"]),
        EPS,
    )
    e2_pass = bool(equivariance["passed_1e_10"].all())
    synthetic_median_error = float(
        synthetic["relative_error_percent"].median()
    )
    synthetic_coverage = float(synthetic["covered"].mean())
    synthetic_pass = synthetic_median_error <= 10.0 and synthetic_coverage >= 0.8

    predictive_pass = (
        advective_increment >= 0.5
        and advective_positive >= 18
        and advective_sign_p < 0.05
        and advective_gain >= best_kinematic_control_gain + 0.5
    )
    mechanics_pass = mechanics_gain >= helmholtz_gain + 1.0 and mechanics_gain > best_control_gain
    bridge_pass = bridge_gain >= 1.0
    intervention_pass = bool(
        np.isfinite(intervention_mechanics_advantage)
        and np.isfinite(intervention_control_advantage)
        and intervention_mechanics_advantage >= 1.0
        and intervention_control_advantage >= 1.0
    )
    spectra_pass = spectral_improvement_percent >= 1.0
    hard_pass = all(
        [
            predictive_pass,
            mechanics_pass,
            bridge_pass,
            intervention_pass,
            spectra_pass,
            e2_pass,
            synthetic_pass,
        ]
    )
    verdict = (
        "PRX field-law gate PASSED."
        if hard_pass
        else "PRX field-law gate FAILED; retain the kinematic sparse operator and do not claim a learned mechanical law."
    )
    return f"""# v197 E(2)-equivariant field-law decision report

## Verdict

**{verdict}**

The test used full-resolution displacement derivatives before sampling and
whole-island nested holdouts.  The derived cache is `{derived_path}`.  No
`future__*` mechanics field entered an inference feature.

## Primary results

| Gate | Value | Pass |
|---|---:|:---:|
| Learned scalar relaxation gain | {relaxation_gain:.3f}% | reference |
| Advective field gain over completed-velocity baseline | {advective_gain:.3f}% | reference |
| Advective field gain over learned relaxation | {advective_increment:.3f}% ({advective_positive}/{len(advective_vs_relaxation)}; p={advective_sign_p:.4g}) | {'yes' if predictive_pass else 'no'} |
| Best time/space/wrong-island kinematic control | {best_kinematic_control_gain:.3f}% over baseline | control |
| Measured-mechanics field gain | {mechanics_gain:.3f}% | {'yes' if mechanics_pass else 'no'} |
| Best shuffled/spatial/wrong-island mechanics control | {best_control_gain:.3f}% | control |
| Reverse motion-to-mechanics bridge gain | {bridge_gain:.3f}% | {'yes' if bridge_pass else 'no'} |
| Mechanical advantage over kinematic law under intervention transfer | {intervention_mechanics_advantage:.3f}% | {'yes' if intervention_pass else 'no'} |
| Real-mechanics advantage over best intervention control | {intervention_control_advantage:.3f}% | {'yes' if intervention_pass else 'no'} |
| Mechanical improvement in spectral log-error | {spectral_improvement_percent:.3f}% | {'yes' if spectra_pass else 'no'} |
| Algebraic E(2) / numerical D4 audit | relative error <= 1e-10 | {'yes' if e2_pass else 'no'} |
| Synthetic coefficient recovery | median error {synthetic_median_error:.2f}%; coverage {100.0 * synthetic_coverage:.1f}% | {'yes' if synthetic_pass else 'no'} |

The unrestricted sparse library gain was {full_gain:.3f}%.  It is not accepted
as a physical result unless the measured-mechanics and intervention controls
also pass.

## Interpretation

This experiment distinguishes two claims.  A causal innovation field may be
locally predictable without being an identifiable mechanical field.  The
continuum operator is structurally E(2)-equivariant, and its coefficients are
estimated without x/y-specific parameters.  That mathematical property alone
does not establish a force law.  A mechanical interpretation requires held-out
traction/stress bridging, intervention transfer, and superiority over
time/space/identity controls.

LaChance transfer is permitted only after all intra-domain mechanical gates
pass.  When the verdict above is a failure, skipping that transfer is the
predefined decision rather than an incomplete experiment.
"""


def main() -> None:
    args = parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_float_list(args.alphas)
    thresholds = parse_float_list(args.thresholds)
    table, derived_path = load_or_build_derived(args)
    if args.max_groups > 0 or args.smoke:
        maximum = args.max_groups or 4
        selected_groups = sorted(table["group"].unique())[:maximum]
        table = table[table["group"].isin(selected_groups)].reset_index(drop=True)
    if table["group"].nunique() < 3:
        raise RuntimeError("At least three whole-island groups are required")
    terms = vector_terms(table)
    controls = {
        control: make_mechanics_control(
            terms,
            table,
            control,
            seed=args.seed + index * 101,
        )
        for index, control in enumerate(
            ("real", "time_shuffled", "spatial_shifted", "wrong_island")
        )
    }
    kinematic_names = tuple(
        name for name in KINEMATIC_TERMS if name != "boundary_normal"
    )
    for index, control in enumerate(
        ("time_shuffled", "spatial_shifted", "wrong_island")
    ):
        controls[f"kin_{control}"] = make_family_control(
            terms,
            table,
            control,
            seed=args.seed + 1001 + index * 101,
            controlled_names=kinematic_names,
        )
    target_innovation = table[["innovation_x", "innovation_y"]].to_numpy(np.float64)

    outer, coefficients, predictions = evaluate_outer_folds(
        table,
        controls,
        alphas=alphas,
        thresholds=thresholds,
    )
    summary = summarize_outer(
        outer,
        bootstrap=1000 if args.smoke else args.bootstrap,
        seed=args.seed,
    )
    stability = coefficient_stability(coefficients)
    bridge = evaluate_mechanical_bridge(
        table,
        controls,
        alphas=alphas,
        thresholds=thresholds,
    )
    intervention = evaluate_interventions(
        table,
        controls,
        alphas=alphas,
        thresholds=thresholds,
    )
    spectra = dynamic_spectra(
        table,
        predictions,
        ["cv", "relaxation", "helmholtz_pde", "mechanics_source_pde"],
    )
    spectra_summary_table = spectrum_summary(spectra)
    equivariance = equivariance_audit(terms, target_innovation)
    synthetic = synthetic_identifiability(
        seed=args.seed + 5000,
        bootstrap=50 if args.smoke else 300,
    )

    outputs = {
        "v197_field_law_outer_folds.csv": outer,
        "v197_field_law_summary.csv": summary,
        "v197_field_law_coefficients.csv": coefficients,
        "v197_coefficient_stability.csv": stability,
        "v197_mechanical_bridge.csv": bridge,
        "v197_intervention_transfer.csv": intervention,
        "v197_dynamic_spectra.csv": spectra,
        "v197_dynamic_spectra_summary.csv": spectra_summary_table,
        "v197_equivariance.csv": equivariance,
        "v197_synthetic_identifiability.csv": synthetic,
    }
    for name, value in outputs.items():
        value.to_csv(args.out_dir / name, index=False)
    report = build_decision_report(
        outer,
        summary,
        bridge,
        intervention,
        spectra_summary_table,
        equivariance,
        synthetic,
        derived_path=derived_path,
    )
    (args.out_dir / "v197_decision_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "cache": str(args.cache.resolve()),
        "cache_sha256": sha256(args.cache),
        "derived_cache": str(derived_path.resolve()),
        "derived_cache_sha256": sha256(derived_path),
        "data_root": str(args.data_root.resolve()),
        "rows": len(table),
        "groups": sorted(table["group"].unique()),
        "alphas": alphas,
        "thresholds": thresholds,
        "smoothing_sigma": args.smoothing_sigma,
        "boundary_decay_um": args.boundary_decay_um,
        "bootstrap": 1000 if args.smoke else args.bootstrap,
        "seed": args.seed,
        "future_feature_count": 0,
        "elapsed_seconds": time.time() - start,
        "outputs": sorted(outputs),
    }
    (args.out_dir / "v197_manifest.json").write_text(
        json.dumps(finite_json(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(report)
    print(json.dumps(finite_json(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
