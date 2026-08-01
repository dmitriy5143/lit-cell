#!/usr/bin/env python3
"""Render representative full-frame CPSAM masks for the v207 audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.segmentation import find_boundaries


REPRESENTATIVES = {
    "mitomycin": 74,
    "y27632": 74,
    "lisa": 38,
}


def normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values, [1.0, 99.5])
    return np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)


def overlay(image: np.ndarray, labels: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.repeat(normalize(image)[..., None], 3, axis=2)
    boundary = find_boundaries(labels, mode="inner")
    rgb[boundary] = color
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 3, figsize=(12.0, 12.0), constrained_layout=True)
    for row, (sequence, frame) in enumerate(REPRESENTATIVES.items()):
        c1 = tifffile.imread(args.data_dir / sequence / f"frame_{frame:03d}_c1.tif")
        c2 = tifffile.imread(args.data_dir / sequence / f"frame_{frame:03d}_c2.tif")
        cell = np.load(args.mask_dir / sequence / f"frame_{frame:03d}.npz")["labels"]
        nucleus = np.load(
            args.mask_dir / sequence / f"frame_{frame:03d}_nucleus.npz"
        )["labels"]
        axes[row, 0].imshow(normalize(c1), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].imshow(overlay(c1, cell, (1.0, 0.18, 0.12)))
        axes[row, 2].imshow(overlay(c2, nucleus, (0.0, 0.85, 1.0)))
        axes[row, 0].set_ylabel(sequence, fontsize=12, fontweight="bold")
        axes[row, 1].set_title(
            f"frame {frame}; coverage={np.mean(cell > 0):.3f}; cells={cell.max()}"
        )
        axes[row, 2].set_title(
            f"anchor coverage={np.mean(nucleus > 0):.3f}; objects={nucleus.max()}"
        )
    axes[0, 0].set_title("channel 1")
    axes[0, 1].set_title("CPSAM cell boundaries")
    axes[0, 2].set_title("channel 2 and anchor boundaries")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("LifeAct-MDCK v207 segmentation and identity-anchor audit", fontsize=15)
    figure.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.out)


if __name__ == "__main__":
    main()
