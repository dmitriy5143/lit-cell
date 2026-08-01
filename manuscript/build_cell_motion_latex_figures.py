#!/usr/bin/env python3
"""Build restrained, publication-style figures for the LaTeX manuscript."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/figures"

DATA_ROOT = Path(os.environ.get("LACHANCE_DATA_ROOT", ROOT / "data/lachance_epithelia"))

RAW_STACK = (
    DATA_ROOT
    / "raw_timelapse/extracted_stacks"
    / "MDCK_Bulk_Timelapse_Data_Sample_Tissues/01.tif"
)
TRACKS = DATA_ROOT / "tables/MDCK_Bulk/MDCK_Bulk_01_tracks.csv"
BUNDLE = ROOT / "evidence/v188"
FIGURE_SOURCES = ROOT / "evidence/figure_sources"
SPATIAL = FIGURE_SOURCES / "v139_spatial_innovation_aggregate.csv"
LADDER = FIGURE_SOURCES / "v161_information_ladder_summary.csv"
CONTROLS = FIGURE_SOURCES / "v160_confirmation_aggregate.csv"

INK = "#17212B"
GRAY = "#65727C"
LIGHT = "#D9E0E5"
BLUE = "#235F9C"
TEAL = "#0F857C"
ORANGE = "#D47725"
RED = "#B84848"
PURPLE = "#6C5B8C"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.linewidth": 0.7,
            "grid.color": LIGHT,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def panel_label_inside(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.018,
        0.975,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        zorder=20,
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _target_crop(
    frame: int = 24,
    size: int = 620,
    target_id: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, int, int, int]:
    tracks = pd.read_csv(TRACKS)
    by_track = {int(track_id): group for track_id, group in tracks.groupby("track_id")}
    eligible = []
    required = set(range(frame - 5, frame + 2))
    for track_id, rows in by_track.items():
        if required.issubset(set(rows["frame"].astype(int))):
            eligible.append(track_id)

    current = tracks[(tracks["frame"] == frame) & tracks["track_id"].isin(eligible)].copy()
    if target_id is None:
        current["center_distance"] = np.hypot(current["x_px"] - 2816, current["y_px"] - 2816)
        target = current.sort_values("center_distance").iloc[0]
        target_id = int(target["track_id"])
    else:
        selected = current[current["track_id"] == target_id]
        if selected.empty:
            raise RuntimeError(f"Track {target_id} is not available at frame {frame}")
        target = selected.iloc[0]
    radius = size // 2
    x0 = max(0, int(round(float(target["x_px"]) - radius)))
    y0 = max(0, int(round(float(target["y_px"]) - radius)))

    image = tifffile.imread(RAW_STACK, key=frame)[y0 : y0 + size, x0 : x0 + size].astype(float)
    low, high = np.percentile(image, [1.0, 99.6])
    image = np.clip((image - low) / max(high - low, 1e-9), 0, 1)
    image = np.power(image, 0.82)
    return image, tracks, target_id, x0, y0


def figure_real_cells_graph() -> None:
    frame = 24
    target_id_for_figure = 5908
    image_a, tracks, target_id, x0_a, y0_a = _target_crop(
        frame=frame,
        size=132,
        target_id=target_id_for_figure,
    )
    image_b, _, target_id_b, x0_b, y0_b = _target_crop(
        frame=frame,
        size=520,
        target_id=target_id_for_figure,
    )
    if target_id_b != target_id:
        raise RuntimeError("Target cell changed between figure crops")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.18),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.93, 1.07]},
    )

    # Panel A: a close view of one causal track. Every displayed point is
    # available at frame t; no t+1 coordinate is drawn.
    ax = axes[0]
    size_a = image_a.shape[0]
    ax.imshow(image_a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    history = tracks[
        (tracks["track_id"] == target_id)
        & tracks["frame"].between(frame - 6, frame)
    ].sort_values("frame")
    xs = history["x_px"].to_numpy(dtype=float) - x0_a
    ys = history["y_px"].to_numpy(dtype=float) - y0_a
    ax.plot(xs, ys, color="white", lw=6.0, alpha=0.94, solid_capstyle="round", zorder=4)
    ax.plot(xs, ys, color=ORANGE, lw=3.0, solid_capstyle="round", zorder=5)
    colors = plt.cm.Oranges(np.linspace(0.42, 0.92, len(xs)))
    ax.scatter(xs, ys, s=43, facecolor=colors, edgecolor="white", lw=1.0, zorder=7)
    arrow_segments = {1, max(1, len(xs) - 3)}
    for index in sorted(arrow_segments):
        if index >= len(xs) - 1:
            continue
        arrow = FancyArrowPatch(
            (xs[index], ys[index]),
            (xs[index + 1], ys[index + 1]),
            arrowstyle="-|>",
            mutation_scale=11.0,
            linewidth=1.9,
            color=ORANGE,
            shrinkA=2.5,
            shrinkB=2.5,
            zorder=8,
        )
        ax.add_patch(arrow)
    ax.add_patch(Circle((xs[-1], ys[-1]), 8.3, fill=False, ec="white", lw=2.1, zorder=6))
    ax.text(
        xs[0] + 5,
        ys[0] + 11,
        r"$t-6$",
        color="white",
        fontsize=7.2,
        ha="left",
        va="top",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.2},
        zorder=9,
    )
    ax.text(
        xs[-1] + 8,
        ys[-1] - 7,
        r"$t$",
        color="white",
        fontsize=7.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=9,
    )
    ax.plot([size_a - 38, size_a - 18], [size_a - 13, size_a - 13], color="white", lw=3)
    ax.text(size_a - 28, size_a - 20, "20 px", color="white", ha="center", va="bottom", fontsize=6.8)
    ax.set_title("История центральной клетки", loc="left", pad=5)
    panel_label(ax, "A")
    ax.set_xlim(0, size_a)
    ax.set_ylim(size_a, 0)

    # Panel B: a lagged graph. The target is at frame t, whereas donor
    # positions and donor innovations come from the completed previous cycle.
    ax = axes[1]
    size_b = image_b.shape[0]
    ax.imshow(np.clip(image_b * 0.56, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    target_row = tracks[(tracks["frame"] == frame) & (tracks["track_id"] == target_id)].iloc[0]
    tx = float(target_row["x_px"] - x0_b)
    ty = float(target_row["y_px"] - y0_b)
    donors = tracks[tracks["frame"] == frame - 1].copy()
    donors["x_local"] = donors["x_px"] - x0_b
    donors["y_local"] = donors["y_px"] - y0_b
    donors["distance"] = np.hypot(donors["x_local"] - tx, donors["y_local"] - ty)
    donors = donors[
        (donors["track_id"] != target_id)
        & donors["x_local"].between(0, size_b)
        & donors["y_local"].between(0, size_b)
        & donors["distance"].le(240)
    ].sort_values("distance")

    displayed_donors = donors.head(42)
    ax.scatter(
        displayed_donors["x_local"],
        displayed_donors["y_local"],
        s=17,
        facecolor="white",
        edgecolor=BLUE,
        lw=0.75,
        alpha=0.88,
        zorder=4,
    )
    # Draw the 20 strongest links for legibility; the computation uses all
    # donors through normalized Gaussian weights.
    for _, row in donors.head(20).iterrows():
        distance = float(row["distance"])
        weight = float(np.exp(-0.5 * (distance / 120.0) ** 2))
        ax.plot(
            [tx, float(row["x_local"])],
            [ty, float(row["y_local"])],
            color=BLUE,
            lw=0.95 + 2.10 * weight,
            alpha=0.40 + 0.52 * weight,
            solid_capstyle="round",
            zorder=3,
        )

    for _, row in donors.head(7).iterrows():
        transition = tracks[
            (tracks["track_id"] == int(row["track_id"]))
            & tracks["frame"].isin([frame - 2, frame - 1])
        ].sort_values("frame")
        if len(transition) != 2:
            continue
        x_prev = transition["x_px"].to_numpy(dtype=float) - x0_b
        y_prev = transition["y_px"].to_numpy(dtype=float) - y0_b
        transition_arrow = FancyArrowPatch(
            (x_prev[0], y_prev[0]),
            (x_prev[1], y_prev[1]),
            arrowstyle="-|>",
            mutation_scale=14.0,
            linewidth=2.35,
            color=TEAL,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=7,
        )
        transition_arrow.set_path_effects(
            [path_effects.Stroke(linewidth=4.2, foreground="white", alpha=0.82), path_effects.Normal()]
        )
        ax.add_patch(transition_arrow)

    for radius, alpha in [(30, 0.86), (60, 0.74), (120, 0.58), (240, 0.38)]:
        ax.add_patch(
            Circle(
                (tx, ty),
                radius,
                fill=False,
                ec=PURPLE,
                lw=1.55,
                ls=(0, (2.3, 2.0)),
                alpha=min(1.0, alpha + 0.12),
                zorder=5,
            )
        )
        if radius == 30:
            continue
        angle = np.deg2rad(-38)
        lx = tx + radius * np.cos(angle)
        ly = ty + radius * np.sin(angle)
        if 8 < lx < size_b - 8 and 8 < ly < size_b - 8:
            ax.text(
                lx,
                ly,
                str(radius),
                color="white",
                fontsize=6.6,
                ha="center",
                va="center",
                bbox={"facecolor": PURPLE, "edgecolor": "white", "linewidth": 0.4, "alpha": 0.88, "pad": 1.4},
                zorder=8,
            )
    ax.scatter([tx], [ty], s=126, facecolor=ORANGE, edgecolor="white", lw=2.2, zorder=9)
    ax.text(
        tx - 16,
        ty - 21,
        r"$x_{i,t}$",
        color="white",
        fontsize=7.5,
        fontweight="bold",
        ha="right",
        bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        zorder=10,
    )
    ax.set_title("Причинный граф завершенных невязок", loc="left", pad=5)
    panel_label(ax, "B")
    legend = [
        Line2D([0], [0], color=BLUE, lw=2.5, label=r"вес донорской невязки"),
        Line2D([0], [0], color=TEAL, lw=2.2, marker=">", markersize=5, label="завершенный переход"),
        Line2D([0], [0], color=PURPLE, lw=1.55, ls=(0, (2.3, 2.0)), label="пространственный масштаб"),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        frameon=True,
        framealpha=0.88,
        borderpad=0.45,
        handlelength=2.2,
    )
    ax.set_xlim(0, size_b)
    ax.set_ylim(size_b, 0)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, "fig1_real_cells_graph")


def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.82))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxed_text: list[tuple[FancyBboxPatch, plt.Text, str]] = []

    def box(
        xy: tuple[float, float],
        width: float,
        height: float,
        text: str,
        edge: str,
        face: str = "white",
        fontsize: float = 7.0,
        linewidth: float = 1.0,
    ) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(patch)
        artist = ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.25,
            color=INK,
            zorder=3,
        )
        boxed_text.append((patch, artist, text))
        return patch

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = INK,
        style: str = "-|>",
        connectionstyle: str = "arc3",
        linewidth: float = 1.0,
        zorder: int = 4,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=9,
                linewidth=linewidth,
                color=color,
                connectionstyle=connectionstyle,
                shrinkA=1,
                shrinkB=1,
                zorder=zorder,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.285),
            0.99,
            0.690,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.55,
            edgecolor=LIGHT,
            facecolor="#FCFDFE",
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.008),
            0.99,
            0.220,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.55,
            edgecolor=LIGHT,
            facecolor="#FAFAF8",
            zorder=0,
        )
    )

    ax.text(0.018, 0.955, "A", fontsize=9.0, fontweight="bold", color=INK, va="top")
    ax.text(
        0.046,
        0.955,
        r"Выпуск прогноза: доступны только наблюдения не позднее $t$",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(0.020, 0.888, "индивидуальный прогноз", fontsize=6.2, fontweight="bold", color=GRAY)
    ax.text(0.575, 0.888, "локальное поле", fontsize=6.2, fontweight="bold", color=GRAY)
    ax.text(0.810, 0.888, "распределение шага", fontsize=6.2, fontweight="bold", color=GRAY)

    obs = box(
        (0.020, 0.685), 0.155, 0.135,
        "Координаты, скорости\nи причинный контекст",
        edge=GRAY, fontsize=6.15, linewidth=0.85,
    )
    anchor = box(
        (0.210, 0.685), 0.175, 0.135,
        "Опорный координатный\nпрогноз и банк траекторий\n" + r"$a_{i,t}^{(1:6)}$",
        edge=BLUE, face="#F2F7FC", fontsize=5.95, linewidth=1.05,
    )
    state = box(
        (0.420, 0.685), 0.180, 0.135,
        "Причинный фильтр состояния\n" + r"$F_\theta(H_{i,t},z_{i,t-1},e_{i,t})$" + "\n" + r"$\mu^{\rm base},s,\nu$",
        edge=TEAL, face="#F0F8F7", fontsize=5.40, linewidth=1.05,
    )
    neighbor_input = box(
        (0.020, 0.445), 0.230, 0.145,
        "Завершенные невязки соседей\nправильного времени и клетки\n" + r"$z_{j,t}=\Phi^{-1}\!\circ F_{t_\nu}(e_{j,t})$",
        edge=GRAY, fontsize=5.80, linewidth=0.85,
    )
    graph = box(
        (0.285, 0.445), 0.250, 0.145,
        "Разреженное многомасштабное\nусреднение локального поля\n" + r"$\phi_{i,t}=A_{\ell}(z_{j,t}),\ \ell=30,60,120,240$ px",
        edge=PURPLE, face="#F5F2F8", fontsize=5.70, linewidth=1.05,
    )
    bounded = box(
        (0.625, 0.545), 0.170, 0.205,
        "Ограниченное\nобновление среднего\n"
        + r"$\mu^{(o)}=\mu^{\rm base}$"
        + "\n"
        + r"$+\mathcal{B}_{b_o}(B_o\phi_{i,t})$"
        + "\n"
        + r"$\|\mathcal{B}_{b_o}\|\leq b_o$",
        edge=INK, fontsize=5.45, linewidth=1.15,
    )
    output = box(
        (0.835, 0.555), 0.145, 0.185,
        "Следующее смещение\n\n"
        + r"$p(d_{i,t+1}\mid\mathcal{I}_t)$"
        + "\n"
        + r"$=t_\nu(\mu^{(o)},s)$",
        edge=ORANGE, face="#FFF7EF", fontsize=5.90, linewidth=1.15,
    )

    arrow((0.175, 0.752), (0.210, 0.752), color=BLUE, linewidth=1.15)
    arrow((0.385, 0.752), (0.420, 0.752), color=TEAL, linewidth=1.15)
    arrow((0.600, 0.752), (0.625, 0.690), color=TEAL, linewidth=1.15)
    arrow((0.250, 0.518), (0.285, 0.518), color=PURPLE, linewidth=1.15)
    arrow((0.535, 0.518), (0.625, 0.595), color=PURPLE, linewidth=1.15)
    arrow((0.795, 0.648), (0.835, 0.648), color=ORANGE, linewidth=1.25)
    ax.text(
        0.907, 0.520,
        r"$\widehat d_{i,t+1\mid t}$ фиксируется",
        ha="center", va="top", fontsize=5.8, color=RED, fontweight="bold",
    )

    boundary_y = 0.270
    ax.plot([0.01, 0.99], [boundary_y, boundary_y], color=RED, ls=(0, (4, 3)), lw=0.9)
    ax.text(
        0.980,
        boundary_y + 0.011,
        r"кадр $t+1$ недоступен до фиксации прогноза",
        ha="right",
        va="bottom",
        fontsize=6.15,
        color=RED,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    ax.text(0.018, 0.215, "B", fontsize=9.0, fontweight="bold", color=INK, va="top")
    ax.text(
        0.046,
        0.215,
        r"Обновление состояния после поступления кадра $t+1$",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        va="top",
    )
    bottom_y, bottom_h = 0.048, 0.100
    frame_x, frame_w = 0.030, 0.155
    residual_x, residual_w = 0.245, 0.270
    update_x, update_w = 0.585, 0.375
    box((frame_x, bottom_y), frame_w, bottom_h, r"наблюдение $x_{i,t+1}$", edge=GRAY, fontsize=6.35, linewidth=0.8)
    box(
        (residual_x, bottom_y),
        residual_w,
        bottom_h,
        r"$e_{i,t+1}=d_{i,t+1}-\widehat d_{i,t+1\mid t}$",
        edge=TEAL,
        face="#F0F8F7",
        fontsize=6.2,
        linewidth=0.95,
    )
    box(
        (update_x, bottom_y),
        update_w,
        bottom_h,
        "Обновляются скрытое состояние клетки\n"
        "и донорское поле следующего цикла",
        edge=PURPLE,
        face="#F5F2F8",
        fontsize=6.25,
        linewidth=0.95,
    )
    arrow((frame_x + frame_w, bottom_y + bottom_h / 2), (residual_x, bottom_y + bottom_h / 2), color=TEAL, linewidth=1.0)
    arrow((residual_x + residual_w, bottom_y + bottom_h / 2), (update_x, bottom_y + bottom_h / 2), color=PURPLE, linewidth=1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, artist, text_value in boxed_text:
        patch_bounds = patch.get_window_extent(renderer)
        text_bounds = artist.get_window_extent(renderer)
        margin = 1.0
        if (
            text_bounds.x0 < patch_bounds.x0 + margin
            or text_bounds.x1 > patch_bounds.x1 - margin
            or text_bounds.y0 < patch_bounds.y0 + margin
            or text_bounds.y1 > patch_bounds.y1 - margin
        ):
            raise RuntimeError(
                "Architecture text exceeds its box: "
                f"{text_value!r}; text={text_bounds.bounds}; box={patch_bounds.bounds}"
            )
    save(fig, "fig2_architecture")


def figure_innovation_evidence() -> None:
    spatial = pd.read_csv(SPATIAL)
    spatial = spatial[spatial["method"] == "v97"].sort_values("neighbor_k")
    ladder = pd.read_csv(LADDER)
    controls = pd.read_csv(CONTROLS)

    fig = plt.figure(figsize=(7.15, 2.72), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 1.18])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    ax = axes[0]
    metrics = [
        ("neighbor_dot_excess_normalized_mean", "neighbor_dot_excess_normalized_std", "скалярное произведение", INK, "o"),
        ("neighbor_cosine_excess_mean", "neighbor_cosine_excess_std", "косинус", TEAL, "s"),
        ("neighbor_component_corr_excess_mean", "neighbor_component_corr_excess_std", "корреляция компонент", ORANGE, "^"),
    ]
    x = spatial["mean_neighbor_distance_px_mean"].to_numpy()
    for mean_col, std_col, label, color, marker in metrics:
        ax.errorbar(
            x,
            spatial[mean_col],
            yerr=spatial[std_col],
            color=color,
            marker=marker,
            markersize=3.7,
            lw=1.25,
            capsize=2.5,
            label=label,
        )
    ax.set_xlabel("Среднее расстояние до соседей, px")
    ax.set_ylabel("Избыток над случайной парой")
    ax.set_ylim(0, 0.37)
    ax.grid(axis="y")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Пространственное затухание")
    panel_label_inside(ax, "A")

    ax = axes[1]
    levels = ["A0_v97", "A1_own", "A2_own_local", "A3_global"]
    labels = ["базовая", "+ своя", "+ соседи", "+ весь кадр"]
    rows = [
        ladder[(ladder["ladder_level"] == level) & (ladder["horizon"] == 6)].iloc[0]
        for level in levels
    ]
    values = np.array([float(row["component_rmse"]) for row in rows])
    ax.plot(range(4), values, color=BLUE, marker="o", lw=1.5, ms=4)
    for index, value in enumerate(values):
        if index == 0:
            ax.text(index + 0.08, value - 0.055, f"{value:.3f}".replace(".", ","), ha="left", va="top", fontsize=6.8)
        else:
            ax.text(index, value + 0.055, f"{value:.3f}".replace(".", ","), ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(4), labels, rotation=15, ha="right")
    ax.set_ylabel("h6 RMSE, px")
    ax.set_ylim(5.45, 6.65)
    ax.grid(axis="y")
    ax.set_title("Информационная лестница")
    panel_label_inside(ax, "B")

    ax = axes[2]
    names = [
        "без локального\nпереноса",
        "актуальная\nневязка",
        "устаревшая\nневязка",
        "другая\nклетка",
    ]
    controls_order = ["v97_no_update", "real", "stale_time", "wrong_cell"]
    values = []
    for control in controls_order:
        row = controls[
            (controls["objective_name"] == "h6_guard10")
            & (controls["packet_name"] == "full")
            & (controls["control"] == control)
            & (controls["horizon"] == 6)
        ].iloc[0]
        values.append(float(row["component_rmse_mean"]))
    ypos = np.arange(len(names))
    colors = [GRAY, TEAL, ORANGE, RED]
    for y, value, color in zip(ypos, values, colors):
        ax.plot([4.3, value], [y, y], color=color, lw=1.5)
        ax.scatter([value], [y], color=color, s=25, zorder=3)
        ax.text(value + 0.08, y, f"{value:.3f}".replace(".", ","), va="center", fontsize=6.8, color=color)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(4.3, 8.05)
    ax.set_xlabel("h6 RMSE, px")
    ax.grid(axis="x")
    ax.set_title("Контроли времени и идентичности")
    panel_label_inside(ax, "C")
    save(fig, "fig3_innovation_evidence")


def figure_primary_results() -> None:
    benchmark = pd.read_csv(BUNDLE / "v188_primary_online_benchmark.csv")
    movie = pd.read_csv(BUNDLE / "v188_primary_online_movie_metrics.csv")
    paired = pd.read_csv(BUNDLE / "v188_paired_movie_statistics.csv")
    methods = [
        ("constant_velocity", "постоянная скорость", GRAY, "--"),
        ("v97_no_update", "базовая модель", BLUE, ":"),
        ("v166_h1_strict", "ближайший шаг", TEAL, "-"),
        ("v166_h6_utility", "накопительный режим", ORANGE, "-"),
    ]
    horizons = [1, 2, 4, 6]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    for method, label, color, style in methods:
        rows = benchmark[benchmark["method"] == method].set_index("horizon").loc[horizons]
        ax.plot(horizons, rows["component_rmse"], color=color, ls=style, marker="o", ms=3.6, lw=1.5, label=label)
    ax.set_xticks(horizons, [f"h{h}" for h in horizons])
    ax.set_xlabel("Последовательный горизонт")
    ax.set_ylabel("Компонентная RMSE, px")
    ax.set_ylim(3.1, 8.15)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Накопление ошибки")
    panel_label(ax, "A")

    ax = axes[1]
    prior = movie[
        (movie["method"] == "v97_no_update") & (movie["horizon"] == 6)
    ][["test_movie", "component_rmse"]].rename(columns={"component_rmse": "prior"})
    ours = movie[
        (movie["method"] == "v166_h6_utility") & (movie["horizon"] == 6)
    ][["test_movie", "component_rmse"]].rename(columns={"component_rmse": "ours"})
    paired_movies = prior.merge(ours, on="test_movie", validate="one_to_one").sort_values("test_movie")
    for _, movie_row in paired_movies.iterrows():
        index = int(movie_row["test_movie"])
        prior_value = float(movie_row["prior"])
        ours_value = float(movie_row["ours"])
        ax.plot([0, 1], [prior_value, ours_value], color="#C9D2D9", lw=1.25, zorder=1)
        ax.scatter([0], [prior_value], color=BLUE, s=26, zorder=2)
        ax.scatter([1], [ours_value], color=ORANGE, s=26, zorder=2)
        ax.text(1.07, ours_value, str(index), va="center", fontsize=6.7, color=GRAY)
    ax.set_xlim(-0.25, 1.28)
    ax.set_xticks([0, 1], ["базовая модель", "накопительный режим"])
    ax.set_ylabel("h6 RMSE, px")
    ax.grid(axis="y")
    row = paired[
        (paired["method"] == "v166_h6_utility")
        & (paired["comparator"] == "v97_no_update")
        & (paired["horizon"] == 6)
    ].iloc[0]
    ax.text(
        0.50,
        0.97,
        (
            f"6/6 фильмов; средняя Δ={row['mean_rmse_delta_comparator_minus_method']:.3f} px\n"
            f"точное p={row['exact_two_sided_sign_flip_p']:.5f}; "
            f"Холм p={row['holm_adjusted_p']:.4f} (семейство H1/H2)"
        ).replace(".", ","),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax.set_title("Парный эффект по фильмам")
    panel_label(ax, "B")
    save(fig, "fig4_primary_results")


def figure_robustness_calibration() -> None:
    robustness = pd.read_csv(BUNDLE / "v188_robustness_matrix.csv")
    uncertainty = pd.read_csv(BUNDLE / "v188_robustness_uncertainty.csv")
    robust = robustness[
        (robustness["operating_point"] == "h6_utility") & (robustness["horizon"] == 6)
    ]

    def value(condition: str) -> float:
        return float(robust[robust["condition"] == condition]["rmse_improvement_percent_mean"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)
    ax = axes[0]
    cadence = [
        ("каждый\nкадр", value("real_update_every_1")),
        ("через\nкадр", value("update_every_2")),
        ("раз в\n3 кадра", value("update_every_3")),
        ("раз в\n6 кадров", value("update_every_6")),
        ("без локального\nпереноса", 0.0),
    ]
    xs = np.arange(len(cadence))
    ys = np.array([item[1] for item in cadence])
    ax.plot(xs, ys, color=TEAL, marker="o", lw=1.6, ms=4.3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1.25, f"{y:.1f}".replace(".", ","), ha="center", fontsize=7)
    ax.set_xticks(xs, [item[0] for item in cadence])
    ax.set_ylabel("Снижение h6 RMSE, %")
    ax.set_ylim(-1.5, 33)
    ax.grid(axis="y")
    ax.set_title("Частота локального переноса")
    panel_label(ax, "A")

    ax = axes[1]
    rows = []
    for operating_point, horizon, label, color in [
        ("h1_strict", 1, "ближайший шаг", TEAL),
        ("h6_utility", 6, "накопительный режим", ORANGE),
    ]:
        row = uncertainty[
            (uncertainty["operating_point"] == operating_point)
            & (uncertainty["calibration_mode"] == "frozen_state_aware_scale")
            & (uncertainty["condition"] == "real_update_every_1")
            & (uncertainty["horizon"] == horizon)
        ].iloc[0]
        rows.append((label, color, float(row["coverage_50_mean"]), float(row["coverage_90_mean"])))
    offsets = [-0.07, 0.07]
    for index, (label, color, cov50, cov90) in enumerate(rows):
        ax.scatter([0.5, 0.9], [cov50, cov90], color=color, s=32, label=label, zorder=3)
        ax.plot([0.5, 0.9], [cov50, cov90], color=color, lw=1.0, alpha=0.75)
        ax.text(0.5 + offsets[index], cov50 + 0.012, f"{cov50:.3f}".replace(".", ","), ha="center", fontsize=6.7, color=color)
        ax.text(0.9 + offsets[index], cov90 + 0.012, f"{cov90:.3f}".replace(".", ","), ha="center", fontsize=6.7, color=color)
    ax.plot([0.45, 0.95], [0.45, 0.95], color=GRAY, lw=0.9, ls="--", label="идеальная калибровка")
    ax.set_xlim(0.44, 1.0)
    ax.set_ylim(0.44, 0.96)
    ax.set_xticks([0.5, 0.9], ["50% интервал", "90% интервал"])
    ax.set_ylabel("Фактическое покрытие")
    ax.grid()
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Покрытие интервалов")
    panel_label(ax, "B")
    save(fig, "fig5_robustness_calibration")


def _field_curve(path: Path, control: str = "real") -> pd.DataFrame:
    table = pd.read_csv(path)
    return table[
        table["representation"].eq("gaussian_score")
        & table["detrend"].eq("affine")
        & table["control"].eq(control)
        & table["lag"].eq(1)
        & table["geometry"].eq("endpoint")
    ].sort_values("mean_distance_nn_mean")


def _pixel_scale(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    selected = table[
        table["representation"].eq("gaussian_score")
        & table["detrend"].eq("affine")
        & table["control"].eq("real")
        & table["lag"].eq(1)
        & table["geometry"].eq("endpoint")
        & table["metric"].eq("vector_correlation")
        & table["unit"].eq("px")
    ]["exponential_xi"].to_numpy(np.float64)
    return selected[np.isfinite(selected)]


def figure_field_operator() -> None:
    field_dev = FIGURE_SOURCES / "field_dev_domain_correlation_summary.csv"
    field_confirmation = FIGURE_SOURCES / "field_confirmation_domain_correlation_summary.csv"
    development = _field_curve(field_dev)
    confirmation = _field_curve(field_confirmation)
    shuffled = _field_curve(field_dev, "time_shuffle")
    scale_curve = pd.read_csv(FIGURE_SOURCES / "v191e_scale_curve_summary.csv")
    scale_curve = scale_curve[scale_curve["horizon"].eq(6)].sort_values(
        "support_r_over_xi"
    )
    runtime = pd.read_csv(FIGURE_SOURCES / "runtime_scaling.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.0),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    for table, label, color, marker in [
        (development, "фильмы 1--6", BLUE, "o"),
        (confirmation, "фильмы 10--16", TEAL, "s"),
    ]:
        x = table["mean_distance_nn_mean"].to_numpy(np.float64)
        y = table["vector_correlation_mean"].to_numpy(np.float64)
        low = table["vector_correlation_ci_low"].to_numpy(np.float64)
        high = table["vector_correlation_ci_high"].to_numpy(np.float64)
        ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, y, color=color, marker=marker, ms=3.2, lw=1.25, label=label)
    ax.plot(
        shuffled["mean_distance_nn_mean"],
        shuffled["vector_correlation_mean"],
        color=GRAY,
        ls="--",
        lw=1.0,
        label="переставленное время",
    )
    ax.axhline(0, color=LIGHT, lw=0.8)
    ax.set_xlim(0.5, 6.2)
    ax.set_xlabel(r"Расстояние $r/d_{\mathrm{nn}}$")
    ax.set_ylabel(r"$K_z(r,1)$")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Пространственная память завершенной невязки")
    panel_label_inside(ax, "A")

    ax = axes[0, 1]
    dev_scale = _pixel_scale(FIGURE_SOURCES / "field_dev_scale_estimates.csv")
    conf_scale = _pixel_scale(FIGURE_SOURCES / "field_confirmation_scale_estimates.csv")
    rng = np.random.default_rng(193)
    for index, (value, label, color) in enumerate(
        [
            (dev_scale, "1--6", BLUE),
            (conf_scale, "10--16", TEAL),
        ]
    ):
        jitter = rng.uniform(-0.06, 0.06, size=len(value))
        ax.scatter(
            np.full(len(value), index) + jitter,
            value,
            color=color,
            s=24,
            alpha=0.85,
            zorder=3,
        )
        mean = float(np.mean(value))
        std = float(np.std(value, ddof=1)) if len(value) > 1 else 0.0
        ax.errorbar(
            [index],
            [mean],
            yerr=[std],
            color=INK,
            marker="_",
            markersize=16,
            lw=1.0,
            capsize=3,
            zorder=4,
        )
        text_x = index + 0.08 if index == 0 else index - 0.08
        text_ha = "left" if index == 0 else "right"
        ax.text(
            text_x,
            mean + 1.2,
            f"{mean:.1f}".replace(".", ","),
            ha=text_ha,
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks([0, 1], ["фильмы 1--6", "фильмы 10--16"])
    ax.set_ylabel(r"Экспоненциальная длина $\xi$, px")
    ax.set_ylim(30, 92)
    ax.grid(axis="y")
    ax.set_title("Масштаб в единицах изображения")
    ax.text(
        0.5,
        0.03,
        "отношение средних 0,965",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=GRAY,
    )
    panel_label_inside(ax, "B")

    ax = axes[1, 0]
    x = scale_curve["support_r_over_xi"].to_numpy(np.float64)
    rmse = scale_curve["component_rmse"].to_numpy(np.float64)
    edge_fraction = scale_curve["edge_fraction"].to_numpy(np.float64) * 100.0
    ax.plot(x, rmse, color=ORANGE, marker="o", ms=3.8, lw=1.4)
    dense_rmse = float(
        pd.read_csv(FIGURE_SOURCES / "v191b_final_aggregate.csv")
        .query(
            "variant == 'dense_start' and fit_mode == 'refit' "
            "and control == 'real' and horizon == 6"
        )["component_rmse_mean"]
        .iloc[0]
    )
    ax.axhline(
        dense_rmse,
        color=GRAY,
        ls="--",
        lw=1.0,
        label="плотный оператор",
    )
    exact = pd.read_csv(FIGURE_SOURCES / "v193_sparse_pareto_aggregate.csv")
    exact_r4 = float(
        exact.query(
            "objective_name == 'h1_strict' and variant == 'field4_start' "
            "and control == 'real' and horizon == 6"
        )["component_rmse_mean"].iloc[0]
    )
    ax.scatter(
        [4.0],
        [exact_r4],
        marker="D",
        s=30,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        zorder=5,
        label="точная проверка, 3 запуска",
    )
    rmse_low = min(float(rmse.min()), dense_rmse)
    rmse_high = max(float(rmse.max()), dense_rmse)
    rmse_pad = max(0.006, 0.12 * (rmse_high - rmse_low))
    ax.set_ylim(rmse_low - rmse_pad, rmse_high + rmse_pad)
    ax.set_xlabel(r"Радиус опоры $R/\xi$")
    ax.set_ylabel("h6 RMSE, px", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.grid(axis="y")
    twin = ax.twinx()
    twin.plot(x, edge_fraction, color=BLUE, marker="s", ms=3.2, lw=1.0)
    twin.set_ylabel("Сохраненные ребра, %", color=BLUE)
    twin.tick_params(axis="y", labelcolor=BLUE)
    twin.set_ylim(0, max(45, float(edge_fraction.max()) * 1.18))
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Насыщение прогностического выигрыша")
    panel_label_inside(ax, "C")

    ax = axes[1, 1]
    ax.loglog(
        runtime["n"],
        runtime["sparse_total_seconds"],
        color=TEAL,
        marker="o",
        ms=3.4,
        lw=1.3,
        label="разреженный",
    )
    dense_rows = runtime[np.isfinite(runtime["dense_apply_seconds"])]
    ax.loglog(
        dense_rows["n"],
        dense_rows["dense_apply_seconds"],
        color=RED,
        marker="s",
        ms=3.4,
        lw=1.3,
        label="плотный",
    )
    at_10k = runtime[runtime["n"].eq(10000)].iloc[0]
    ax.annotate(
        f"{at_10k['measured_speedup']:.0f}×",
        xy=(10000, float(at_10k["sparse_total_seconds"])),
        xytext=(3500, float(at_10k["sparse_total_seconds"]) * 0.22),
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8},
        fontsize=7,
        color=INK,
    )
    ax.set_xlabel("Число клеток в кадре")
    ax.set_ylabel("Время на кадр, s")
    ax.grid(which="major")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Измеренная вычислительная сложность")
    panel_label_inside(ax, "D")
    save(fig, "fig6_field_sparse_operator")


def main() -> None:
    configure_style()
    figure_real_cells_graph()
    figure_architecture()
    figure_innovation_evidence()
    figure_primary_results()
    figure_robustness_calibration()
    figure_field_operator()
    print(OUT)


if __name__ == "__main__":
    main()
