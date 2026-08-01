#!/usr/bin/env python3
"""Independently verify the frozen numerical inputs used in Figures 3 and 4."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "figure_validation_report.md"
SPATIAL_DIR = ROOT / "evidence" / "figure_sources"
LADDER = SPATIAL_DIR / "v161_information_ladder_summary.csv"
CONTROLS = SPATIAL_DIR / "v160_confirmation_aggregate.csv"
BUNDLE = ROOT / "evidence" / "v188"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_close(left: float, right: float, label: str, atol: float = 1e-10) -> None:
    if not np.isclose(left, right, atol=atol, rtol=1e-9):
        raise AssertionError(f"{label}: {left} != {right}")


def audit_figure_3() -> list[str]:
    aggregate_path = SPATIAL_DIR / "v139_spatial_innovation_aggregate.csv"
    per_movie_path = SPATIAL_DIR / "v139_spatial_innovation_per_movie.csv"
    aggregate = pd.read_csv(aggregate_path)
    per_movie = pd.read_csv(per_movie_path)
    metrics = [
        "mean_neighbor_distance_px",
        "neighbor_dot_excess_normalized",
        "neighbor_cosine_excess",
        "neighbor_component_corr_excess",
    ]
    ks = [1, 4, 8, 16]
    selected = aggregate[
        aggregate["method"].eq("v97") & aggregate["neighbor_k"].isin(ks)
    ].sort_values("neighbor_k")
    if selected["neighbor_k"].tolist() != ks:
        raise AssertionError("Figure 3A does not contain exactly k=1,4,8,16")

    raw = per_movie[
        per_movie["method"].eq("v97") & per_movie["neighbor_k"].isin(ks)
    ].copy()
    movie_counts = raw.groupby("neighbor_k")["movie"].nunique()
    if not movie_counts.eq(6).all():
        raise AssertionError(f"Figure 3A movie counts are not six: {movie_counts.to_dict()}")
    recomputed = raw.groupby("neighbor_k")[metrics].agg(["mean", "std"])
    for k in ks:
        row = selected[selected["neighbor_k"].eq(k)].iloc[0]
        for metric in metrics:
            for statistic in ("mean", "std"):
                require_close(
                    float(row[f"{metric}_{statistic}"]),
                    float(recomputed.loc[k, (metric, statistic)]),
                    f"Figure 3A {metric}/{statistic}/k={k}",
                )
    distances = selected["mean_neighbor_distance_px_mean"].to_numpy(float)
    dot_excess = selected["neighbor_dot_excess_normalized_mean"].to_numpy(float)
    if not (np.all(np.diff(distances) > 0) and np.all(np.diff(dot_excess) < 0)):
        raise AssertionError("Figure 3A distance/excess ordering is not monotone")

    ladder = pd.read_csv(LADDER)
    levels = ["A0_v97", "A1_own", "A2_own_local", "A3_global"]
    ladder_rows = (
        ladder[ladder["horizon"].eq(6) & ladder["ladder_level"].isin(levels)]
        .set_index("ladder_level")
        .loc[levels]
    )
    if not (ladder_rows["causal"].all() and ladder_rows["movies"].eq(7).all()):
        raise AssertionError("Figure 3B is not the seven-movie causal ladder")
    ladder_rmse = ladder_rows["component_rmse"].to_numpy(float)
    if not np.all(np.diff(ladder_rmse) < 0):
        raise AssertionError("Figure 3B information ladder is not ordered")

    controls = pd.read_csv(CONTROLS)
    order = ["real", "stale_time", "v97_no_update", "wrong_cell"]
    control_rows = (
        controls[
            controls["objective_name"].eq("h6_guard10")
            & controls["packet_name"].eq("full")
            & controls["horizon"].eq(6)
            & controls["control"].isin(order)
        ]
        .set_index("control")
        .loc[order]
    )
    if not control_rows["movies"].eq(7).all():
        raise AssertionError("Figure 3C controls do not use seven movies")
    control_rmse = control_rows["component_rmse_mean"].to_numpy(float)
    if not np.all(np.diff(control_rmse) > 0):
        raise AssertionError("Figure 3C causal-control ordering is invalid")

    return [
        "### Рисунок 3",
        "",
        "- Панель A независимо пересчитана из строк по фильмам: 6 фильмов, "
        "k = 1, 4, 8, 16; средние и стандартные отклонения совпадают с агрегатом.",
        "- Среднее расстояние возрастает, а нормированный избыток скалярного "
        "произведения монотонно убывает: "
        + ", ".join(f"{value:.3f}" for value in dot_excess)
        + ".",
        "- Панель B использует причинную лестницу на 7 фильмах: "
        + " -> ".join(f"{value:.3f}" for value in ladder_rmse)
        + " px.",
        "- Панель C проходит ожидаемый порядок real < stale < no-update < wrong-cell: "
        + " < ".join(f"{value:.3f}" for value in control_rmse)
        + " px.",
        f"- SHA256 агрегата v139: `{file_sha256(aggregate_path)}`.",
        "",
    ]


def audit_figure_4() -> list[str]:
    benchmark_path = BUNDLE / "v188_primary_online_benchmark.csv"
    movie_path = BUNDLE / "v188_primary_online_movie_metrics.csv"
    paired_path = BUNDLE / "v188_paired_movie_statistics.csv"
    benchmark = pd.read_csv(benchmark_path)
    movie = pd.read_csv(movie_path)
    paired = pd.read_csv(paired_path)
    methods = [
        "constant_velocity",
        "v97_no_update",
        "v166_h1_strict",
        "v166_h6_utility",
    ]
    horizons = [1, 2, 4, 6]
    selected = benchmark[
        benchmark["method"].isin(methods) & benchmark["horizon"].isin(horizons)
    ]
    if len(selected) != len(methods) * len(horizons):
        raise AssertionError("Figure 4A has missing or duplicate method/horizon rows")
    if selected[["component_rmse", "r2"]].isna().any().any():
        raise AssertionError("Figure 4A contains non-finite values")

    prior = movie[
        movie["method"].eq("v97_no_update") & movie["horizon"].eq(6)
    ][["test_movie", "component_rmse"]].rename(columns={"component_rmse": "prior"})
    proposed = movie[
        movie["method"].eq("v166_h6_utility") & movie["horizon"].eq(6)
    ][["test_movie", "component_rmse"]].rename(columns={"component_rmse": "proposed"})
    joined = prior.merge(proposed, on="test_movie", validate="one_to_one").sort_values("test_movie")
    if joined["test_movie"].tolist() != [1, 2, 3, 4, 5, 6]:
        raise AssertionError("Figure 4B is not aligned on all six held-out movies")
    deltas = joined["prior"].to_numpy(float) - joined["proposed"].to_numpy(float)
    if not np.all(deltas > 0):
        raise AssertionError("Figure 4B is not positive on all six movies")

    reported = paired[
        paired["method"].eq("v166_h6_utility")
        & paired["comparator"].eq("v97_no_update")
        & paired["horizon"].eq(6)
    ]
    if len(reported) != 1:
        raise AssertionError("Figure 4B paired-statistics row is not unique")
    row = reported.iloc[0]
    require_close(
        float(deltas.mean()),
        float(row["mean_rmse_delta_comparator_minus_method"]),
        "Figure 4B mean paired delta",
    )
    exact_sign_flip = 2.0 / (2.0 ** len(deltas))
    require_close(exact_sign_flip, float(row["exact_two_sided_sign_flip_p"]), "Figure 4B exact p")
    if int(row["method_better_movies"]) != 6:
        raise AssertionError("Figure 4B reported improved-movie count is not six")

    h1 = float(
        benchmark[
            benchmark["method"].eq("v166_h1_strict") & benchmark["horizon"].eq(1)
        ]["component_rmse"].iloc[0]
    )
    h6 = float(
        benchmark[
            benchmark["method"].eq("v166_h6_utility") & benchmark["horizon"].eq(6)
        ]["component_rmse"].iloc[0]
    )
    return [
        "### Рисунок 4",
        "",
        "- Панель A содержит ровно 4 метода x 4 горизонта без пропусков и дубликатов.",
        f"- Основные точки: режим ближайшего шага h1 RMSE = {h1:.6f} px; "
        f"накопительный режим h6 RMSE = {h6:.6f} px.",
        "- Панель B объединена по идентификатору тестового фильма, а не по позиции строки; "
        "улучшение положительно на 6/6 фильмах.",
        f"- Независимо пересчитанная средняя парная разность: {deltas.mean():.6f} px; "
        f"точное двустороннее p = {exact_sign_flip:.5f}; "
        f"Holm p = {float(row['holm_adjusted_p']):.4f}.",
        f"- SHA256 основной таблицы v188: `{file_sha256(benchmark_path)}`.",
        "",
    ]


def main() -> None:
    lines = [
        "# Технический аудит рисунков 3 и 4",
        "",
        "Проверка выполнена отдельным кодом непосредственно по исходным CSV. "
        "Она не импортирует построитель рисунков и поэтому не повторяет его отбор строк.",
        "",
        *audit_figure_3(),
        *audit_figure_4(),
        "## Итог",
        "",
        "Все численные значения, единицы анализа, порядок контролей и парное "
        "сопоставление фильмов прошли проверку.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
