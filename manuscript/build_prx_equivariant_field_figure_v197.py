#!/usr/bin/env python3
"""Build the publication figure for the equivariant field-law experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "evidence/figure_sources"
OUT = ROOT / "manuscript/figures"
EPS = 1e-12


def strip_box(
    axis: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    positions = np.arange(1, len(values) + 1)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#18212b", "linewidth": 1.5},
        whiskerprops={"color": "#65717e", "linewidth": 1.0},
        capprops={"color": "#65717e", "linewidth": 1.0},
        boxprops={"edgecolor": "#65717e", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for index, (position, value, color) in enumerate(
        zip(positions, values, colors, strict=True)
    ):
        jitter = np.linspace(-0.16, 0.16, len(value))
        if index % 2:
            jitter = jitter[::-1]
        axis.scatter(
            position + jitter,
            value,
            s=15,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
    axis.set_xticks(positions, labels)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outer = pd.read_csv(RESULT / "v197_field_law_outer_folds.csv")
    coefficients = pd.read_csv(RESULT / "v197_field_law_coefficients.csv")
    real = outer[outer["control"].eq("real")].pivot(
        index="outer_group",
        columns="variant",
        values="displacement_rmse",
    )
    condition = real.index.to_series().str.split("/").str[0]

    palette = {
        "blue": "#2878B5",
        "orange": "#E07A3F",
        "green": "#3A9D78",
        "purple": "#7564B0",
        "gray": "#AAB2BB",
        "red": "#C64B4B",
    }
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.75))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.11,
        top=0.94,
        wspace=0.34,
        hspace=0.46,
    )

    # A: causal-field controls.
    labels_a = [
        ("real", "Реальное\nполе"),
        ("kin_spatial_shifted", "Сдвиг\nв пространстве"),
        ("kin_time_shuffled", "Сдвиг\nво времени"),
        ("kin_wrong_island", "Другой\nостровок"),
    ]
    values_a = [
        outer[
            outer["variant"].eq("advective_pde")
            & outer["control"].eq(control)
        ]["gain_vs_cv_percent"].to_numpy(float)
        for control, _ in labels_a
    ]
    strip_box(
        axes[0, 0],
        values_a,
        [label for _, label in labels_a],
        [
            palette["blue"],
            palette["gray"],
            palette["gray"],
            palette["gray"],
        ],
    )
    axes[0, 0].axhline(0.0, color="#303840", linewidth=0.8)
    axes[0, 0].set_ylabel("Уменьшение RMSE относительно\nпостоянной скорости, %")
    axes[0, 0].set_title("A   Причинный полевой сигнал")

    # B: added value beyond scalar relaxation.
    increment = 100.0 * (
        real["relaxation"] - real["advective_pde"]
    ) / real["relaxation"].clip(lower=EPS)
    condition_order = [
        "low_density",
        "high_density",
        "cytod",
        "cn03_1_4",
        "cn03_5_8",
    ]
    condition_labels = ["Низкая\nплотность", "Высокая\nплотность", "CytoD", "CN03\nдо", "CN03\nпосле"]
    condition_colors = [
        palette["green"],
        palette["orange"],
        palette["purple"],
        palette["blue"],
        "#4F9DB8",
    ]
    values_b = [
        increment[condition.eq(name)].to_numpy(float)
        for name in condition_order
    ]
    strip_box(axes[0, 1], values_b, condition_labels, condition_colors)
    axes[0, 1].axhline(0.0, color="#303840", linewidth=0.8)
    axes[0, 1].set_ylabel("Добавочное уменьшение RMSE\nнад скалярным затуханием, %")
    axes[0, 1].set_title("B   Пространственные члены сверх памяти")
    axes[0, 1].text(
        0.98,
        0.95,
        "20/22 островков",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#303840",
    )

    # C: measured mechanics over the matched kinematic law.
    labels_c = [
        ("real", "Реальная\nмеханика"),
        ("spatial_shifted", "Сдвиг\nв пространстве"),
        ("time_shuffled", "Сдвиг\nво времени"),
        ("wrong_island", "Другой\nостровок"),
    ]
    mechanics_values = []
    for control, _ in labels_c:
        mechanics = outer[
            outer["variant"].eq("mechanics_source_pde")
            & outer["control"].eq(control)
        ].set_index("outer_group")["displacement_rmse"]
        mechanics_values.append(
            (
                100.0
                * (real["helmholtz_pde"] - mechanics)
                / real["helmholtz_pde"].clip(lower=EPS)
            ).to_numpy(float)
        )
    strip_box(
        axes[1, 0],
        mechanics_values,
        [label for _, label in labels_c],
        [
            palette["red"],
            palette["gray"],
            palette["gray"],
            palette["gray"],
        ],
    )
    axes[1, 0].axhline(0.0, color="#303840", linewidth=0.8)
    axes[1, 0].set_ylabel("Добавочное уменьшение RMSE\nнад кинематическим законом, %")
    axes[1, 0].set_title("C   Измеренная механика не проходит контроли")

    # D: standardized coefficient stability.
    coefficient_rows = coefficients[
        coefficients["variant"].eq("advective_pde")
        & coefficients["control"].eq("real")
    ]
    term_order = [
        "u_prev",
        "lap_u",
        "grad_div_u",
        "advect_v_u",
        "advect_u_u",
        "cubic_u",
    ]
    term_labels = [
        r"$u_{t-1}$",
        r"$\nabla^2u$",
        r"$\nabla(\nabla\!\cdot u)$",
        r"$(v\!\cdot\nabla)u$",
        r"$(u\!\cdot\nabla)u$",
        r"$|u|^2u$",
    ]
    y_positions = np.arange(len(term_order))[::-1]
    for y_position, term in zip(y_positions, term_order, strict=True):
        value = coefficient_rows[
            coefficient_rows["term"].eq(term)
        ]["coefficient_standardized"].to_numpy(float)
        missing = 22 - len(value)
        if missing > 0:
            value = np.concatenate([value, np.zeros(missing)])
        median = np.median(value)
        low, high = np.quantile(value, [0.25, 0.75])
        axes[1, 1].plot(
            [low, high],
            [y_position, y_position],
            color=palette["blue"],
            linewidth=3.0,
            solid_capstyle="round",
        )
        axes[1, 1].scatter(
            [median],
            [y_position],
            s=34,
            color=palette["blue"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axes[1, 1].axvline(0.0, color="#303840", linewidth=0.8)
    axes[1, 1].set_yticks(y_positions, term_labels)
    axes[1, 1].set_xlabel("Стандартизованный коэффициент: медиана и МКР")
    axes[1, 1].set_title("D   Устойчивость членов оператора")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#DCE1E6", linewidth=0.55, alpha=0.75)
        axis.set_axisbelow(True)

    pdf = OUT / "fig7_equivariant_field_law.pdf"
    png = OUT / "fig7_equivariant_field_law.png"
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
