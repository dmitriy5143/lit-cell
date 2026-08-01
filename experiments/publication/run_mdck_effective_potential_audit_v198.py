#!/usr/bin/env python3
"""Audit an effective-potential sector of the v197 innovation-field law.

The audit separates an integrable gradient-flow sector from non-variational
advection. It does not identify thermodynamic or mechanical energy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import binomtest

import run_mdck_equivariant_field_law_v197 as v197

from lit_cell_forecasting.equivariant_field_law import (  # noqa: E402
    EPS,
    VectorOperatorModel,
    radial_clip,
    vector_rmse,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DERIVED = (
    ROOT
    / "outputs/mdck_equivariant_field_law_v197_smoke_2026-07-30"
    / "v197_operator_samples.pkl.gz"
)
DEFAULT_OUT = ROOT / "outputs/mdck_effective_potential_audit_v198_2026-07-30"

POTENTIAL_TERMS = ("u_prev", "lap_u", "grad_div_u", "cubic_u")
POTENTIAL_BOUNDARY_TERMS = POTENTIAL_TERMS + ("boundary_normal",)
ADVECTIVE_TERMS = v197.VARIANTS["advective_pde"]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-cache", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--thresholds", default="0,0.01,0.03")
    parser.add_argument("--seed", type=int, default=198)
    return parser.parse_args()


def _prepare_design(
    terms: Mapping[str, np.ndarray],
    target: np.ndarray,
    names: Sequence[str],
    *,
    clip_quantile: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    scales: list[float] = []
    clips: list[float] = []
    for name in names:
        value = np.nan_to_num(np.asarray(terms[name], dtype=np.float64))
        magnitude = np.linalg.norm(value, axis=-1)
        maximum = float(np.quantile(magnitude, clip_quantile))
        clean = radial_clip(value, maximum)
        arrays.append(clean)
        clips.append(maximum)
        scales.append(max(float(np.sqrt(np.mean(np.square(clean)))), EPS))
    clean_target = np.nan_to_num(np.asarray(target, dtype=np.float64))
    target_maximum = float(
        np.quantile(np.linalg.norm(clean_target, axis=-1), clip_quantile)
    )
    clean_target = radial_clip(clean_target, target_maximum)
    design = np.column_stack(
        [
            (array / scale).reshape(-1)
            for array, scale in zip(arrays, scales, strict=True)
        ]
    )
    return (
        design,
        clean_target.reshape(-1),
        np.asarray(scales),
        np.asarray(clips),
    )


def fit_sign_constrained_potential(
    terms: Mapping[str, np.ndarray],
    target: np.ndarray,
    names: Sequence[str],
    *,
    alpha: float,
) -> VectorOperatorModel:
    """Fit a bounded-below gradient sector in direct-map coordinates.

    For ``u_next = a*u + nu_T*lap(u) + nu_L*grad(div(u))
    - g*|u|^2*u + b*boundary``, boundedness of the associated effective
    functional requires ``a < 1``, ``nu_T >= 0``, ``nu_L >= 0``, and
    ``g >= 0``. The boundary coefficient is unconstrained.
    """

    names = tuple(names)
    design, response, scales, clips = _prepare_design(terms, target, names)
    ridge = math.sqrt(float(alpha)) * np.eye(len(names))
    augmented_design = np.vstack([design, ridge])
    augmented_response = np.concatenate([response, np.zeros(len(names))])
    lower = np.full(len(names), -np.inf, dtype=np.float64)
    upper = np.full(len(names), np.inf, dtype=np.float64)
    for index, name in enumerate(names):
        if name == "u_prev":
            upper[index] = 0.999 * scales[index]
        elif name in {"lap_u", "grad_div_u"}:
            lower[index] = 0.0
        elif name == "cubic_u":
            upper[index] = 0.0
    # Accelerate can emit transient overflow warnings while SciPy computes its
    # unconstrained initializer; the bounded solution is checked explicitly.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._lsq\.lsq_linear",
        )
        result = lsq_linear(
            augmented_design,
            augmented_response,
            bounds=(lower, upper),
            method="trf",
            lsmr_tol="auto",
            max_iter=500,
        )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Constrained potential fit failed: {result.message}")
    return VectorOperatorModel(
        term_names=names,
        coefficients=np.asarray(result.x, dtype=np.float64),
        term_scales=scales,
        term_clip_magnitudes=clips,
        alpha=float(alpha),
        threshold=0.0,
    )


def tune_constrained(
    terms: Mapping[str, np.ndarray],
    target: np.ndarray,
    names: Sequence[str],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    alphas: Sequence[float],
) -> tuple[VectorOperatorModel, float, float]:
    best: tuple[float, float] | None = None
    for alpha in alphas:
        model = fit_sign_constrained_potential(
            {name: terms[name][train_index] for name in names},
            target[train_index],
            names,
            alpha=alpha,
        )
        prediction = model.predict(
            {name: terms[name][validation_index] for name in names}
        )
        candidate = (vector_rmse(prediction, target[validation_index]), float(alpha))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    combined = np.concatenate([train_index, validation_index])
    model = fit_sign_constrained_potential(
        {name: terms[name][combined] for name in names},
        target[combined],
        names,
        alpha=best[1],
    )
    return model, best[0], best[1]


def physical_coefficients(model: VectorOperatorModel) -> dict[str, float]:
    return {
        name: float(coefficient / max(scale, EPS))
        for name, coefficient, scale in zip(
            model.term_names,
            model.coefficients,
            model.term_scales,
            strict=True,
        )
    }


def potential_parameters(model: VectorOperatorModel) -> dict[str, float]:
    coefficients = physical_coefficients(model)
    return {
        "quadratic_r": 1.0 - coefficients.get("u_prev", 0.0),
        "gradient_k_transverse": coefficients.get("lap_u", 0.0),
        "gradient_k_longitudinal": coefficients.get("grad_div_u", 0.0),
        "quartic_g": -coefficients.get("cubic_u", 0.0),
        "boundary_h": coefficients.get("boundary_normal", 0.0),
    }


def alignment_diagnostics(
    prediction: np.ndarray,
    u_previous: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    drift = prediction - u_previous
    observed = target - u_previous
    drift_norm = np.linalg.norm(drift, axis=1)
    observed_norm = np.linalg.norm(observed, axis=1)
    denominator = drift_norm * observed_norm
    cosine = np.divide(
        np.sum(drift * observed, axis=1),
        np.maximum(denominator, EPS),
        out=np.zeros_like(denominator),
        where=denominator > EPS,
    )
    rng = np.random.default_rng(seed)
    shuffled = observed[rng.permutation(len(observed))]
    shuffled_denominator = drift_norm * np.linalg.norm(shuffled, axis=1)
    shuffled_cosine = np.divide(
        np.sum(drift * shuffled, axis=1),
        np.maximum(shuffled_denominator, EPS),
        out=np.zeros_like(shuffled_denominator),
        where=shuffled_denominator > EPS,
    )
    first_order_delta = -np.sum(drift * observed, axis=1)
    shuffled_first_order_delta = -np.sum(drift * shuffled, axis=1)
    return {
        "drift_observed_cosine": float(np.mean(cosine)),
        "drift_shuffled_cosine": float(np.mean(shuffled_cosine)),
        "first_order_energy_decrease_fraction": float(
            np.mean(first_order_delta < 0)
        ),
        "shuffled_energy_decrease_fraction": float(
            np.mean(shuffled_first_order_delta < 0)
        ),
        "first_order_energy_change_mean": float(np.mean(first_order_delta)),
    }


def evaluate(
    table: pd.DataFrame,
    *,
    alphas: Sequence[float],
    thresholds: Sequence[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = v197.vector_terms(table)
    target = table[["innovation_x", "innovation_y"]].to_numpy(np.float64)
    target_displacement = table[["target_x", "target_y"]].to_numpy(np.float64)
    base = table[["base_x", "base_y"]].to_numpy(np.float64)
    u_previous = terms["u_prev"]
    groups = table["group"].astype(str).to_numpy()
    unique_groups = sorted(np.unique(groups))
    variants = {
        "helmholtz_unconstrained": ("unconstrained", v197.VARIANTS["helmholtz_pde"]),
        "potential_unconstrained": ("unconstrained", POTENTIAL_TERMS),
        "potential_constrained": ("constrained", POTENTIAL_TERMS),
        "potential_boundary_constrained": (
            "constrained",
            POTENTIAL_BOUNDARY_TERMS,
        ),
        "advective_unconstrained": ("unconstrained", ADVECTIVE_TERMS),
    }
    records: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for fold_index, outer_group in enumerate(unique_groups):
        validation_group = v197.choose_validation_group(
            table,
            unique_groups,
            outer_group,
        )
        test_index = np.flatnonzero(groups == outer_group)
        validation_index = np.flatnonzero(groups == validation_group)
        train_index = np.flatnonzero(
            (groups != outer_group) & (groups != validation_group)
        )
        base_rmse = vector_rmse(base[test_index], target_displacement[test_index])
        for variant, (fit_kind, names) in variants.items():
            if fit_kind == "constrained":
                model, validation_rmse, alpha = tune_constrained(
                    terms,
                    target,
                    names,
                    train_index,
                    validation_index,
                    alphas,
                )
                threshold = 0.0
            else:
                model, tuning = v197.tune_and_fit(
                    terms,
                    target,
                    names,
                    train_index,
                    validation_index,
                    alphas=alphas,
                    thresholds=thresholds,
                )
                validation_rmse = tuning["validation_rmse"]
                alpha = tuning["alpha"]
                threshold = tuning["threshold"]
            prediction = model.predict(
                {name: terms[name][test_index] for name in model.term_names}
            )
            metric = v197.metrics(
                prediction,
                target[test_index],
                base[test_index],
                target_displacement[test_index],
            )
            diagnostics = alignment_diagnostics(
                prediction,
                u_previous[test_index],
                target[test_index],
                seed=seed + 1009 * fold_index,
            )
            parameters = potential_parameters(model)
            bounded_below = (
                parameters["quadratic_r"] > 0
                and parameters["gradient_k_transverse"] >= 0
                and parameters["gradient_k_longitudinal"] >= 0
                and parameters["quartic_g"] >= 0
            )
            records.append(
                {
                    "outer_group": outer_group,
                    "validation_group": validation_group,
                    "variant": variant,
                    "fit_kind": fit_kind,
                    "n_test": len(test_index),
                    "validation_rmse": validation_rmse,
                    "alpha": alpha,
                    "threshold": threshold,
                    **metric,
                    "gain_vs_cv_percent": 100.0
                    * (base_rmse - metric["displacement_rmse"])
                    / max(base_rmse, EPS),
                    **diagnostics,
                    **parameters,
                    "bounded_below_effective_functional": bounded_below,
                }
            )
            for term, coefficient in physical_coefficients(model).items():
                coefficient_rows.append(
                    {
                        "outer_group": outer_group,
                        "variant": variant,
                        "fit_kind": fit_kind,
                        "term": term,
                        "coefficient_physical": coefficient,
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(coefficient_rows)


def summarize(outer: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for variant, rows in outer.groupby("variant", sort=True):
        records.append(
            {
                "variant": variant,
                "n_islands": len(rows),
                "displacement_rmse_macro": rows["displacement_rmse"].mean(),
                "displacement_r2_macro": rows["displacement_r2"].mean(),
                "gain_vs_cv_percent_mean": rows["gain_vs_cv_percent"].mean(),
                "positive_islands": int(np.sum(rows["gain_vs_cv_percent"] > 0)),
                "drift_observed_cosine_mean": rows[
                    "drift_observed_cosine"
                ].mean(),
                "drift_shuffled_cosine_mean": rows[
                    "drift_shuffled_cosine"
                ].mean(),
                "energy_decrease_fraction_mean": rows[
                    "first_order_energy_decrease_fraction"
                ].mean(),
                "shuffled_energy_decrease_fraction_mean": rows[
                    "shuffled_energy_decrease_fraction"
                ].mean(),
                "bounded_below_folds": int(
                    rows["bounded_below_effective_functional"].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def build_report(outer: pd.DataFrame, summary: pd.DataFrame) -> str:
    indexed = summary.set_index("variant")
    potential = indexed.loc["potential_constrained"]
    potential_boundary = indexed.loc["potential_boundary_constrained"]
    advective = indexed.loc["advective_unconstrained"]
    relative_gap = 100.0 * (
        potential["displacement_rmse_macro"]
        - advective["displacement_rmse_macro"]
    ) / advective["displacement_rmse_macro"]
    paired = outer.pivot(
        index="outer_group",
        columns="variant",
        values="displacement_rmse",
    )
    potential_better = int(
        np.sum(
            paired["potential_constrained"]
            < paired["advective_unconstrained"]
        )
    )
    sign_p = float(
        binomtest(
            len(paired) - potential_better,
            len(paired),
            p=0.5,
            alternative="greater",
        ).pvalue
    )
    energy_control_delta = (
        potential["energy_decrease_fraction_mean"]
        - potential["shuffled_energy_decrease_fraction_mean"]
    )
    gradient_sector_pass = (
        int(potential["bounded_below_folds"]) == len(paired)
        and potential["gain_vs_cv_percent_mean"] >= 8.0
        and energy_control_delta >= 0.05
    )
    physical_energy_pass = False
    return f"""# v198: аудит эффективного потенциального сектора

## Результаты

| Вариант | RMSE, мкм/мин | Выигрыш к CV | Косинус с наблюдаемым обновлением | Доля убывания первого порядка |
|---|---:|---:|---:|---:|
| constrained potential | {potential['displacement_rmse_macro']:.6f} | {potential['gain_vs_cv_percent_mean']:.3f}% | {potential['drift_observed_cosine_mean']:.3f} | {potential['energy_decrease_fraction_mean']:.3f} |
| constrained potential + boundary | {potential_boundary['displacement_rmse_macro']:.6f} | {potential_boundary['gain_vs_cv_percent_mean']:.3f}% | {potential_boundary['drift_observed_cosine_mean']:.3f} | {potential_boundary['energy_decrease_fraction_mean']:.3f} |
| full advective law | {advective['displacement_rmse_macro']:.6f} | {advective['gain_vs_cv_percent_mean']:.3f}% | {advective['drift_observed_cosine_mean']:.3f} | {advective['energy_decrease_fraction_mean']:.3f} |

Потенциальный сектор хуже полного адвективного закона на {relative_gap:.3f}%.
Полный закон лучше на {len(paired) - potential_better}/{len(paired)} островков;
односторонний знаковый p={sign_p:.4g}. Для потенциального сектора доля
убывания функционала первого порядка превосходит временную перестановку на
{energy_control_delta:.3f}.

Все {int(potential['bounded_below_folds'])}/{len(paired)} внешних оценок
ограниченного потенциального сектора имеют знаки, совместимые с функционалом

```text
F[u] = integral [
  r/2 |u|^2
  + g/4 |u|^4
  + K_T/2 |grad u|^2
  + K_L/2 (div u)^2
] dx.
```

Его отрицательный вариационный градиент воспроизводит релаксацию,
лапласиан, продольно-поперечную диффузию и кубическое насыщение. Адвективные
члены остаются неинтегрируемым активным переносом.

## Решение

- Эффективный градиентный сектор: **{'PASS' if gradient_sector_pass else 'FAIL'}**.
- Физическая или термодинамическая энергия: **{'PASS' if physical_energy_pass else 'FAIL'}**.

Функционал относится к полю кинематической инновации и определен с точностью
до неизвестной подвижности и масштаба времени. Провал механического шлюза
v197 запрещает называть его энергией клетки, ткани, адгезии или деформации.
Корректная формулировка: эффективный диссипативный функционал, дополненный
неравновесным адвективным потоком.
"""


def main() -> None:
    args = parse_args()
    start = time.time()
    with gzip.open(args.derived_cache, "rb") as handle:
        table = pickle.load(handle)
    table = table.reset_index(drop=True)
    outer, coefficients = evaluate(
        table,
        alphas=parse_float_list(args.alphas),
        thresholds=parse_float_list(args.thresholds),
        seed=args.seed,
    )
    summary = summarize(outer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outer.to_csv(args.out_dir / "v198_potential_outer_folds.csv", index=False)
    coefficients.to_csv(
        args.out_dir / "v198_potential_coefficients.csv",
        index=False,
    )
    summary.to_csv(args.out_dir / "v198_potential_summary.csv", index=False)
    report = build_report(outer, summary)
    (args.out_dir / "v198_potential_decision_report.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest = {
        "derived_cache": str(args.derived_cache.resolve()),
        "rows": len(table),
        "groups": int(table["group"].nunique()),
        "alphas": parse_float_list(args.alphas),
        "thresholds": parse_float_list(args.thresholds),
        "seed": args.seed,
        "future_feature_count": 0,
        "elapsed_seconds": time.time() - start,
        "interpretation": "effective_kinematic_functional_not_physical_energy",
    }
    (args.out_dir / "v198_potential_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(report)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
