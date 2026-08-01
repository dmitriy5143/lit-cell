#!/usr/bin/env python3
"""Build Figure 8 directly from the frozen h1/h6 evidence tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "manuscript" / "figures"
HORIZONS = (1, 2, 4, 6)

INK = "#17212B"
GRAY = "#65727C"
LIGHT = "#D9E0E5"
BLUE = "#235F9C"
TEAL = "#0F857C"
RED = "#B84848"


def main() -> None:
    pareto = pd.read_csv(EVIDENCE / "h1_v205" / "v205_pareto_points.csv")
    normalized = pd.read_csv(EVIDENCE / "h1_v205" / "v205_normalized_error.csv")
    baseline = pd.read_csv(EVIDENCE / "figure_sources" / "v102_movie_level_summary.csv")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.edgecolor": GRAY,
            "axes.linewidth": 0.7,
            "grid.color": LIGHT,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.78),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    axis = axes[0]
    ordered = pareto.sort_values("lambda")
    scatter = axis.scatter(
        ordered["h1_component_rmse"],
        ordered["h6_component_rmse"],
        c=ordered["lambda"],
        cmap="viridis",
        s=38,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    axis.plot(
        ordered["h1_component_rmse"],
        ordered["h6_component_rmse"],
        color=GRAY,
        linewidth=1.1,
        alpha=0.75,
        zorder=2,
    )
    axis.annotate(
        "ближайший шаг",
        tuple(ordered.iloc[0][["h1_component_rmse", "h6_component_rmse"]]),
        xytext=(7, -14),
        textcoords="offset points",
        fontsize=7.0,
    )
    axis.annotate(
        "накопительный режим",
        tuple(ordered.iloc[-1][["h1_component_rmse", "h6_component_rmse"]]),
        xytext=(-84, 9),
        textcoords="offset points",
        fontsize=7.0,
    )
    axis.set_xlabel("RMSE ближайшего шага h1, px")
    axis.set_ylabel("Накопительная RMSE h6, px")
    axis.set_title("A   Компромисс рабочих режимов", loc="left", fontweight="bold")
    axis.grid(axis="both")
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.047, pad=0.03)
    colorbar.set_label("Вес цели h6")

    axis = axes[1]
    selected = {
        "lambda_00": ("ближайший шаг", BLUE, "o"),
        "lambda_05": ("сбалансированный", TEAL, "s"),
        "lambda_10": ("накопительный режим", RED, "D"),
    }
    for name, (label, color, marker) in selected.items():
        rows = normalized[normalized["objective_name"].eq(name)].sort_values("horizon")
        axis.plot(
            rows["horizon"],
            rows["normalized_rmse_mean"],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.45,
            markersize=4.2,
        )

    constant_velocity = baseline[baseline["method_id"].eq("baseline/constant_velocity")]
    target_scale = normalized[normalized["objective_name"].eq("lambda_00")][
        ["horizon", "target_component_sd_mean"]
    ]
    constant_velocity = constant_velocity.merge(target_scale, on="horizon", validate="one_to_one")
    constant_velocity["normalized_rmse"] = (
        constant_velocity["component_rmse_movie_mean"]
        / constant_velocity["target_component_sd_mean"]
    )
    axis.plot(
        constant_velocity["horizon"],
        constant_velocity["normalized_rmse"],
        label="постоянная скорость",
        color=GRAY,
        linestyle="--",
        marker="x",
        linewidth=1.3,
    )
    axis.set_xticks(HORIZONS, [f"h{horizon}" for horizon in HORIZONS])
    axis.set_xlabel("Последовательный горизонт")
    axis.set_ylabel("RMSE / стандартное отклонение цели")
    axis.set_title("B   Ошибка относительно масштаба цели", loc="left", fontweight="bold")
    axis.grid(axis="y")
    axis.legend(frameon=False, fontsize=6.8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig8_h1_pareto_evidence"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
