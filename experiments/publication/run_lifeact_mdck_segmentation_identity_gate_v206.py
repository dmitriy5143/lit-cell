#!/usr/bin/env python3
"""Foundation segmentation and temporal-identity gate for LifeAct-MDCK.

This runner does not infer biological value from an attractive overlay.  It
measures mask coverage, size plausibility, boundary support, cross-channel
agreement, and (when consecutive frames are supplied) one-to-one temporal IoU.
Only a configuration that passes this gate should feed the mechanochemical
forecasting experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import exposure, filters, measure, morphology, segmentation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / "new_data" / "lifeact_mdck_mechanochemical_v206" / "pilot_samples"
DEFAULT_OUT = ROOT / "outputs" / "lifeact_mdck_segmentation_identity_v206_2026-08-01"


def robust_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.quantile(image, [0.01, 0.995])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def mask_metrics(labels: np.ndarray, image: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    positive = labels > 0
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    boundary = segmentation.find_boundaries(labels, mode="inner")
    gradient = filters.sobel(filters.gaussian(robust_image(image), sigma=1.0))
    boundary_support = float(gradient[boundary].mean()) if boundary.any() else 0.0
    random_support = float(gradient.mean()) + 1e-9
    return {
        "n_instances": float(len(areas)),
        "coverage": float(positive.mean()),
        "area_median": float(np.median(areas)) if len(areas) else 0.0,
        "area_q10": float(np.quantile(areas, 0.10)) if len(areas) else 0.0,
        "area_q90": float(np.quantile(areas, 0.90)) if len(areas) else 0.0,
        "small_fraction_lt200": float(np.mean(areas < 200)) if len(areas) else 1.0,
        "large_fraction_gt30000": float(np.mean(areas > 30000)) if len(areas) else 1.0,
        "boundary_gradient_enrichment": boundary_support / random_support,
    }


def compact_markers(labels: np.ndarray) -> np.ndarray:
    markers = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for region in measure.regionprops(labels):
        cy, cx = np.rint(region.centroid).astype(int)
        cy = int(np.clip(cy, 0, labels.shape[0] - 1))
        cx = int(np.clip(cx, 0, labels.shape[1] - 1))
        markers[cy, cx] = next_id
        next_id += 1
    return morphology.dilation(markers, morphology.disk(1))


def seeded_boundary_watershed(labels: np.ndarray, boundary_image: np.ndarray) -> np.ndarray:
    markers = compact_markers(labels)
    if markers.max() == 0:
        return labels.astype(np.int32)
    normalized = robust_image(boundary_image)
    elevation = filters.sobel(filters.gaussian(normalized, sigma=1.2))
    # A confluent monolayer occupies the field; watershed recovers cell territories
    # while the LifeAct/phase gradient, rather than Euclidean distance alone, sets edges.
    return segmentation.watershed(elevation, markers=markers).astype(np.int32)


def relabel_dense(labels: np.ndarray) -> np.ndarray:
    return measure.label(labels > 0, connectivity=1).astype(np.int32)


def contingency_iou(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = labels_a.ravel().astype(np.int64)
    b = labels_b.ravel().astype(np.int64)
    na, nb = int(a.max()), int(b.max())
    joint = np.bincount(a * (nb + 1) + b, minlength=(na + 1) * (nb + 1)).reshape(
        na + 1, nb + 1
    )
    intersection = joint[1:, 1:].astype(np.float64)
    area_a = joint[1:, :].sum(axis=1).astype(np.float64)
    area_b = joint[:, 1:].sum(axis=0).astype(np.float64)
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return iou, area_a, area_b


def match_iou_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    iou, area_a, area_b = contingency_iou(labels_a, labels_b)
    if iou.size == 0:
        return {
            "matched_iou_median": 0.0,
            "retention_iou25": 0.0,
            "retention_iou50": 0.0,
            "count_ratio": 0.0,
        }
    row, col = linear_sum_assignment(-iou)
    matched = iou[row, col]
    denom = max(len(area_a), len(area_b), 1)
    return {
        "matched_iou_median": float(np.median(matched)) if len(matched) else 0.0,
        "retention_iou25": float(np.sum(matched >= 0.25) / denom),
        "retention_iou50": float(np.sum(matched >= 0.50) / denom),
        "count_ratio": float(min(len(area_a), len(area_b)) / max(len(area_a), len(area_b), 1)),
    }


def paired_channel_iou(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    values = match_iou_metrics(labels_a, labels_b)
    return {f"cross_channel_{key}": value for key, value in values.items()}


def overlay_figure(image: np.ndarray, labels: np.ndarray, title: str, output: Path) -> None:
    shown = exposure.equalize_adapthist(robust_image(image), clip_limit=0.015)
    boundary = segmentation.find_boundaries(labels, mode="outer")
    rgb = np.repeat(shown[..., None], 3, axis=2)
    rgb[boundary] = np.array([0.98, 0.20, 0.12])
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_image(path: Path) -> tuple[str, int | None, str | None]:
    name = path.name
    frame_match = re.search(r"(?:t|frame_)(\d+)", name, flags=re.IGNORECASE)
    channel_match = re.search(r"c([12])(?:_ORG)?", name, flags=re.IGNORECASE)
    stem = re.sub(r"t\d+", "tXXX", name, flags=re.IGNORECASE)
    stem = re.sub(r"frame_\d+", "frame_XXX", stem, flags=re.IGNORECASE)
    stem = re.sub(r"c[12](?:_ORG)?", "cX", stem, flags=re.IGNORECASE)
    return stem, int(frame_match.group(1)) if frame_match else None, channel_match.group(1) if channel_match else None


def run_cellpose(
    model: Any,
    model_name: str,
    image: np.ndarray,
    diameter: float,
    cellprob_threshold: float,
    flow_threshold: float,
) -> np.ndarray:
    bsize = 256 if model_name.startswith("cpsam") else 384
    masks, _, _ = model.eval(
        image,
        batch_size=1,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=80,
        max_size_fraction=0.05,
        bsize=bsize,
        tile_overlap=0.20,
        normalize={
            "normalize": True,
            "percentile": [1.0, 99.5],
            "tile_norm_blocksize": 256,
        },
    )
    return np.asarray(masks, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", default="cpsam_v2,cpdino")
    parser.add_argument("--diameters", default="40,60,80")
    parser.add_argument("--cellprob-thresholds", default="-0.5,0.0")
    parser.add_argument("--flow-threshold", type=float, default=0.6)
    parser.add_argument("--image-pattern", default="*Mitomycin*t074c[12]_ORG.tif")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--skip-watershed", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    mask_dir = args.out_dir / "masks"
    overlay_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    image_paths = sorted(args.pilot_dir.glob(args.image_pattern))
    if not image_paths:
        raise FileNotFoundError(f"No images matched {args.image_pattern!r} in {args.pilot_dir}")
    models_requested = [item.strip() for item in args.models.split(",") if item.strip()]
    diameters = [float(item) for item in args.diameters.split(",") if item.strip()]
    thresholds = [float(item) for item in args.cellprob_thresholds.split(",") if item.strip()]

    from cellpose import models as cellpose_models

    rows: list[dict[str, Any]] = []
    saved: dict[tuple[str, int | None, str, float, float, str], np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    for model_name in models_requested:
        started_model = time.perf_counter()
        model = cellpose_models.CellposeModel(
            gpu=device.type != "cpu",
            device=device,
            pretrained_model=model_name,
            use_bfloat16=False,
        )
        for image_path in image_paths:
            image = tifffile.imread(image_path)
            images[image_path.name] = image
            sequence, frame, channel = parse_image(image_path)
            for diameter in diameters:
                for threshold in thresholds:
                    started = time.perf_counter()
                    labels = run_cellpose(
                        model,
                        model_name,
                        image,
                        diameter,
                        threshold,
                        args.flow_threshold,
                    )
                    direct_name = f"{image_path.stem}_{model_name}_d{diameter:g}_p{threshold:g}_direct"
                    np.savez_compressed(mask_dir / f"{direct_name}.npz", labels=labels)
                    overlay_figure(
                        image,
                        labels,
                        f"{model_name}; channel {channel}; diameter {diameter:g}; p={threshold:g}",
                        overlay_dir / f"{direct_name}.png",
                    )
                    base = {
                        "sequence": sequence,
                        "frame": frame,
                        "channel": channel,
                        "image": image_path.name,
                        "model": model_name,
                        "diameter": diameter,
                        "cellprob_threshold": threshold,
                        "flow_threshold": args.flow_threshold,
                        "representation": "direct",
                        "runtime_seconds": time.perf_counter() - started,
                        **mask_metrics(labels, image),
                    }
                    rows.append(base)
                    saved[
                        (sequence, frame, model_name, diameter, threshold, f"c{channel}_direct")
                    ] = labels

                    if not args.skip_watershed:
                        paired_channel = "1" if channel == "2" else channel
                        boundary_path = next(
                            (
                                candidate
                                for candidate in image_paths
                                if parse_image(candidate)[0] == sequence
                                and parse_image(candidate)[1] == frame
                                and parse_image(candidate)[2] == paired_channel
                            ),
                            image_path,
                        )
                        boundary_image = tifffile.imread(boundary_path)
                        watershed_labels = seeded_boundary_watershed(labels, boundary_image)
                        watershed_name = direct_name.replace("_direct", "_watershed")
                        np.savez_compressed(
                            mask_dir / f"{watershed_name}.npz", labels=watershed_labels
                        )
                        overlay_figure(
                            boundary_image,
                            watershed_labels,
                            f"boundary watershed from {model_name}; d={diameter:g}; p={threshold:g}",
                            overlay_dir / f"{watershed_name}.png",
                        )
                        rows.append(
                            {
                                **base,
                                "representation": "boundary_watershed",
                                **mask_metrics(watershed_labels, boundary_image),
                            }
                        )
                        saved[
                            (
                                sequence,
                                frame,
                                model_name,
                                diameter,
                                threshold,
                                f"c{channel}_watershed",
                            )
                        ] = watershed_labels
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"[v206] {model_name} finished in {time.perf_counter() - started_model:.1f}s")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "v206_segmentation_metrics.csv", index=False)

    agreement_rows: list[dict[str, Any]] = []
    for model_name in models_requested:
        for diameter in diameters:
            for threshold in thresholds:
                sequence_frames = sorted(
                    {(parse_image(path)[0], parse_image(path)[1]) for path in image_paths}
                )
                for sequence, frame in sequence_frames:
                    for representation in ("direct", "watershed"):
                        key1 = (
                            sequence,
                            frame,
                            model_name,
                            diameter,
                            threshold,
                            f"c1_{representation}",
                        )
                        key2 = (
                            sequence,
                            frame,
                            model_name,
                            diameter,
                            threshold,
                            f"c2_{representation}",
                        )
                        if key1 not in saved or key2 not in saved:
                            continue
                        agreement_rows.append(
                            {
                                "sequence": sequence,
                                "frame": frame,
                                "model": model_name,
                                "diameter": diameter,
                                "cellprob_threshold": threshold,
                                "representation": representation,
                                **paired_channel_iou(saved[key1], saved[key2]),
                            }
                        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(args.out_dir / "v206_cross_channel_agreement.csv", index=False)

    temporal_rows: list[dict[str, Any]] = []
    for model_name in models_requested:
        for diameter in diameters:
            for threshold in thresholds:
                for sequence in sorted({parse_image(path)[0] for path in image_paths}):
                    available_frames = sorted(
                        {
                            parse_image(path)[1]
                            for path in image_paths
                            if parse_image(path)[0] == sequence
                            and parse_image(path)[1] is not None
                        }
                    )
                    for channel in ("1", "2"):
                        for representation in ("direct", "watershed"):
                            for frame_a, frame_b in zip(available_frames[:-1], available_frames[1:]):
                                if frame_b != frame_a + 1:
                                    continue
                                key_a = (
                                    sequence,
                                    frame_a,
                                    model_name,
                                    diameter,
                                    threshold,
                                    f"c{channel}_{representation}",
                                )
                                key_b = (
                                    sequence,
                                    frame_b,
                                    model_name,
                                    diameter,
                                    threshold,
                                    f"c{channel}_{representation}",
                                )
                                if key_a not in saved or key_b not in saved:
                                    continue
                                temporal_rows.append(
                                    {
                                        "sequence": sequence,
                                        "frame_a": frame_a,
                                        "frame_b": frame_b,
                                        "channel": channel,
                                        "model": model_name,
                                        "diameter": diameter,
                                        "cellprob_threshold": threshold,
                                        "representation": representation,
                                        **match_iou_metrics(saved[key_a], saved[key_b]),
                                    }
                                )
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(args.out_dir / "v206_temporal_identity.csv", index=False)

    # A foundation mask must describe most of a confluent monolayer, retain a
    # plausible number of instances, and place edges on image-supported boundaries.
    direct = metrics[metrics.representation.eq("direct")].copy()
    direct["plausible"] = (
        direct.coverage.between(0.45, 0.98)
        & direct.n_instances.between(250, 1500)
        & direct.area_median.between(500, 20000)
        & direct.boundary_gradient_enrichment.ge(1.15)
    )
    ranking = direct.sort_values(
        ["plausible", "boundary_gradient_enrichment", "coverage"],
        ascending=[False, False, False],
    )
    ranking.to_csv(args.out_dir / "v206_segmentation_ranking.csv", index=False)

    best = ranking.iloc[0].to_dict() if len(ranking) else {}
    spatial_passed = bool(best.get("plausible", False))
    temporal_passed: bool | None = None
    if not temporal.empty and best:
        matched_temporal = temporal[
            temporal["model"].eq(best["model"])
            & temporal["diameter"].eq(best["diameter"])
            & temporal["cellprob_threshold"].eq(best["cellprob_threshold"])
            & temporal["representation"].eq(best["representation"])
        ]
        temporal_passed = bool(
            not matched_temporal.empty
            and matched_temporal["retention_iou25"].mean() >= 0.50
            and matched_temporal["count_ratio"].mean() >= 0.80
        )
    passed = spatial_passed and temporal_passed is not False
    report = [
        "# LifeAct-MDCK segmentation and identity gate v206",
        "",
        "## Decision",
        "",
        (
            "At least one direct foundation segmentation passed the predeclared spatial gate. Consecutive-frame identity testing is still required before forecasting."
            if passed
            else "No direct foundation segmentation passed the spatial gate. A forecasting result built on these masks would test segmentation failure rather than mechanochemical observability."
        ),
        "",
        "## Best configuration",
        "",
        pd.DataFrame([best]).to_markdown(index=False) if best else "No result.",
        "",
        "## All direct configurations",
        "",
        ranking.to_markdown(index=False),
        "",
        "## Cross-channel agreement",
        "",
        agreement.to_markdown(index=False) if not agreement.empty else "Not available.",
        "",
        "## Consecutive-frame identity",
        "",
        temporal.to_markdown(index=False) if not temporal.empty else "Not available in this run.",
        "",
        "## Gate definition",
        "",
        "A direct mask passes only with 45-98% field coverage, 250-1500 instances, median area 500-20000 pixels, and boundary-gradient enrichment >=1.15. These are necessary spatial checks, not evidence that the state predicts motion. Temporal IoU and hard wrong-cell/time-shuffle forecasting controls remain mandatory.",
        "",
    ]
    (args.out_dir / "v206_segmentation_decision_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    (args.out_dir / "v206_run_contract.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "models": models_requested,
                "diameters": diameters,
                "cellprob_thresholds": thresholds,
                "flow_threshold": args.flow_threshold,
                "image_pattern": args.image_pattern,
                "n_images": len(image_paths),
                "spatial_gate_passed": spatial_passed,
                "temporal_gate_passed": temporal_passed,
                "combined_gate_passed": passed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
