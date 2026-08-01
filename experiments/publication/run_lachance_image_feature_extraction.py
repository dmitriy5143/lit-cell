#!/usr/bin/env python3
"""Raw-image audit, download, alignment and patch-feature extraction for LaChance.

This runner is deliberately staged:

1. ``audit``: verify local tables/XMLs, dependencies, disk space and raw ZIP status.
2. ``download``: fetch the selected Zenodo raw sample with resume and checksum.
3. ``inspect``: list image-like members inside the downloaded ZIP.
4. ``extract``: best-effort patch/intensity extraction once the ZIP layout is known.

The first useful model branch should not train on image features until the
alignment overlay gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "lachance_image_feature_extraction_2026-06-15"
DEFAULT_TABLE_ROOT = ROOT / "new_data" / "lachance_epithelia" / "tables"
DEFAULT_RAW_DIR = ROOT / "new_data" / "lachance_epithelia" / "raw_timelapse"
ZENODO_API = "https://zenodo.org/api/records/4959169"
DEFAULT_FILE_KEY = "MDCK_Bulk_Timelapse_Data_Sample_Tissues.zip"
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".avi", ".mov", ".mp4")
FRAME_RE = re.compile(r"(?:^|[^0-9])(?:t|frame|f)?0*([0-9]{1,5})(?:[^0-9]|$)", re.IGNORECASE)


def finite_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_list(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _open_url(url: str, *, timeout: int, headers: dict[str, str] | None = None, allow_insecure_ssl: bool = False):
    context = ssl._create_unverified_context() if allow_insecure_ssl else None
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout, context=context)


def _is_ssl_error(exc: BaseException) -> bool:
    return isinstance(exc, ssl.SSLError) or (
        isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError)
    )


def fetch_json(url: str, *, allow_insecure_ssl: bool = False) -> dict[str, Any]:
    try:
        with _open_url(url, timeout=60, allow_insecure_ssl=allow_insecure_ssl) as response:
            return json.loads(response.read().decode("utf-8"))
    except (ssl.SSLError, urllib.error.URLError) as exc:
        if allow_insecure_ssl or not _is_ssl_error(exc):
            raise
        print("SSL verification failed; retrying metadata fetch with unverified SSL.", file=sys.stderr)
        with _open_url(url, timeout=60, allow_insecure_ssl=True) as response:
            return json.loads(response.read().decode("utf-8"))


def zenodo_files(record_url: str, *, allow_insecure_ssl: bool = False) -> list[dict[str, Any]]:
    record = fetch_json(record_url, allow_insecure_ssl=allow_insecure_ssl)
    rows: list[dict[str, Any]] = []
    for item in record.get("files", []):
        links = item.get("links", {})
        rows.append(
            {
                "key": item.get("key", ""),
                "size_bytes": int(item.get("size", 0)),
                "checksum": item.get("checksum", ""),
                "download": links.get("self") or links.get("download") or "",
            }
        )
    return rows


def md5_file(path: Path, chunk_mb: int = 16) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(int(chunk_mb) * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def expected_md5(checksum: str) -> str | None:
    if not checksum:
        return None
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1].strip().lower()
    return None


def download_with_resume(
    url: str,
    target: Path,
    *,
    expected_size: int = 0,
    checksum: str = "",
    allow_insecure_ssl: bool = False,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    exp_md5 = expected_md5(checksum)
    if target.exists() and (expected_size <= 0 or target.stat().st_size == expected_size):
        status = {"downloaded": False, "reason": "target_exists", "target": target, "size_bytes": target.stat().st_size}
        if exp_md5:
            got = md5_file(target)
            status["md5"] = got
            status["checksum_ok"] = bool(got == exp_md5)
        return status

    part = target.with_suffix(target.suffix + ".part")
    resume_at = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at > 0 else {}
    mode = "ab" if resume_at > 0 else "wb"
    try:
        response_ctx = _open_url(url, timeout=120, headers=headers, allow_insecure_ssl=allow_insecure_ssl)
    except (ssl.SSLError, urllib.error.URLError) as exc:
        if allow_insecure_ssl or not _is_ssl_error(exc):
            raise
        print("SSL verification failed; retrying download with unverified SSL.", file=sys.stderr)
        response_ctx = _open_url(url, timeout=120, headers=headers, allow_insecure_ssl=True)

    with response_ctx as response, part.open(mode) as fh:
        status_code = getattr(response, "status", None)
        if resume_at > 0 and status_code != 206:
            print("Server did not honor Range request; restarting partial download.", file=sys.stderr)
            fh.close()
            part.unlink(missing_ok=True)
            return download_with_resume(
                url,
                target,
                expected_size=expected_size,
                checksum=checksum,
                allow_insecure_ssl=allow_insecure_ssl,
            )
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)

    got_size = part.stat().st_size
    if expected_size > 0 and got_size != expected_size:
        return {
            "downloaded": True,
            "complete": False,
            "target": target,
            "partial": part,
            "size_bytes": got_size,
            "expected_size_bytes": expected_size,
        }
    part.replace(target)
    status = {"downloaded": True, "complete": True, "target": target, "size_bytes": target.stat().st_size}
    if exp_md5:
        got = md5_file(target)
        status["md5"] = got
        status["checksum_ok"] = bool(got == exp_md5)
    return status


def dependency_status() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name in ["PIL", "imageio", "tifffile", "cv2", "skimage", "numpy", "pandas", "scipy"]:
        try:
            module = __import__(name)
            out[name] = {"status": "ok", "version": str(getattr(module, "__version__", ""))}
        except Exception as exc:
            out[name] = {"status": "missing", "error": f"{type(exc).__name__}: {exc}"}
    return out


def disk_status(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return {
        "path": path,
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
    }


def xml_attribute_status(raw_root: Path, datasets: Iterable[str]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for dataset in datasets:
        paths = sorted((raw_root / dataset).glob("**/*.xml"))
        attrs: set[str] = set()
        for p in paths[:3]:
            text = p.read_text(errors="ignore")[:20_000]
            attrs.update(re.findall(r"\s([A-Za-z_][A-Za-z0-9_]*)=", text))
        status[dataset] = {
            "xml_count": len(paths),
            "sample_attributes": sorted(attrs),
            "has_shape_like_attributes": bool(
                attrs.intersection(
                    {
                        "RADIUS",
                        "AREA",
                        "CIRCULARITY",
                        "ELLIPSE_MAJOR",
                        "ELLIPSE_MINOR",
                        "CONTOUR",
                        "MASK",
                    }
                )
            ),
        }
    return status


def table_status(table_root: Path, datasets: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in datasets:
        rows = []
        for p in sorted((table_root / dataset).glob("*_tracks.csv")):
            df = pd.read_csv(p, usecols=["sequence", "frame", "track_id", "x_px", "y_px"], nrows=200_000)
            rows.append(
                {
                    "file": p,
                    "rows_sampled": int(len(df)),
                    "sequence": int(df["sequence"].iloc[0]) if len(df) else None,
                    "frame_min": int(df["frame"].min()) if len(df) else None,
                    "frame_max": int(df["frame"].max()) if len(df) else None,
                    "track_sample_count": int(df["track_id"].nunique()) if len(df) else None,
                    "x_min": float(df["x_px"].min()) if len(df) else None,
                    "x_max": float(df["x_px"].max()) if len(df) else None,
                    "y_min": float(df["y_px"].min()) if len(df) else None,
                    "y_max": float(df["y_px"].max()) if len(df) else None,
                }
            )
        out[dataset] = {"table_count": len(rows), "tables": rows[:8]}
    return out


def inspect_zip(path: Path, max_members: int = 80) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": path}
    if not zipfile.is_zipfile(path):
        return {"exists": True, "is_zip": False, "path": path, "size_bytes": path.stat().st_size}
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        image_infos = [info for info in infos if info.filename.lower().endswith(IMAGE_SUFFIXES)]
        by_suffix: dict[str, int] = {}
        for info in image_infos:
            suffix = Path(info.filename).suffix.lower()
            by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        return {
            "exists": True,
            "is_zip": True,
            "path": path,
            "size_bytes": path.stat().st_size,
            "members": len(infos),
            "image_like_members": len(image_infos),
            "image_suffix_counts": by_suffix,
            "preview": [info.filename for info in infos[:max_members]],
            "image_like_preview": [info.filename for info in image_infos[:max_members]],
        }


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def patch_features(patch: np.ndarray) -> dict[str, float]:
    p = normalize_image(patch)
    if p.size == 0:
        return {}
    gy, gx = np.gradient(p)
    grad = np.sqrt(gx * gx + gy * gy)
    yy, xx = np.indices(p.shape)
    w = p - float(np.min(p))
    total = float(w.sum())
    if total > 1e-8:
        cx = float((xx * w).sum() / total)
        cy = float((yy * w).sum() / total)
        x0 = xx - cx
        y0 = yy - cy
        cov_xx = float((w * x0 * x0).sum() / total)
        cov_yy = float((w * y0 * y0).sum() / total)
        cov_xy = float((w * x0 * y0).sum() / total)
        eig = np.linalg.eigvalsh(np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float32))
        elong = float((eig[-1] + 1e-6) / (eig[0] + 1e-6))
        theta = float(0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy + 1e-8))
    else:
        cx = cy = elong = theta = 0.0
    thresh = float(np.mean(p) + 0.5 * np.std(p))
    mask = p >= thresh
    return {
        "img_mean": float(np.mean(p)),
        "img_std": float(np.std(p)),
        "img_p10": float(np.percentile(p, 10)),
        "img_p50": float(np.percentile(p, 50)),
        "img_p90": float(np.percentile(p, 90)),
        "img_grad_mean": float(np.mean(grad)),
        "img_grad_p90": float(np.percentile(grad, 90)),
        "img_fg_frac": float(np.mean(mask)),
        "img_centroid_dx": float(cx - (p.shape[1] - 1) / 2.0),
        "img_centroid_dy": float(cy - (p.shape[0] - 1) / 2.0),
        "img_elongation": elong,
        "img_orientation": theta,
        "img_orient_cos": float(math.cos(theta)),
        "img_orient_sin": float(math.sin(theta)),
    }


def find_image_members(zip_info: dict[str, Any], sequence: str, frame: int) -> list[str]:
    members = zip_info.get("_members", [])
    seq_tokens = {str(sequence), str(int(sequence)) if str(sequence).isdigit() else str(sequence)}
    seq_forms = set(seq_tokens)
    if str(sequence).isdigit():
        seq_int = int(sequence)
        seq_forms.update({str(seq_int), f"{seq_int:02d}", f"{seq_int:03d}"})
    frame_tokens = {str(frame), f"{frame:02d}", f"{frame:03d}", f"{frame:04d}"}
    hits: list[str] = []
    for name in members:
        lower = name.lower()
        if not lower.endswith(IMAGE_SUFFIXES):
            continue
        base = Path(lower).stem
        # A member named "02.tif" is a full sequence stack, not a frame-specific
        # image for frame 0/2. Treating it as a PIL image silently reads only
        # the first page and breaks alignment diagnostics.
        if lower.endswith((".tif", ".tiff")) and base in seq_forms:
            continue
        has_seq = any(re.search(rf"(^|[^0-9])0*{re.escape(tok)}([^0-9]|$)", lower) for tok in seq_tokens)
        has_frame = any(tok in base for tok in frame_tokens)
        if has_seq and has_frame:
            hits.append(name)
    return hits


def find_sequence_stack_member(members: list[str], sequence: str) -> str | None:
    seq_int = int(sequence) if str(sequence).isdigit() else None
    seq_forms = {str(sequence)}
    if seq_int is not None:
        seq_forms.update({str(seq_int), f"{seq_int:02d}", f"{seq_int:03d}"})
    candidates = []
    for name in members:
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        stem = Path(name).stem
        if stem in seq_forms:
            candidates.append(name)
    return sorted(candidates)[0] if candidates else None


def ensure_zip_member_extracted(zf: zipfile.ZipFile, member: str, target_dir: Path) -> Path:
    target = target_dir / member
    info = zf.getinfo(member)
    if target.exists() and target.stat().st_size == info.file_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with zf.open(member) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    tmp.replace(target)
    return target


def read_tiff_page(path: Path, frame: int) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError("tifffile is required for multi-page TIFF extraction")
    with tifffile.TiffFile(path) as tf:
        if frame < 0 or frame >= len(tf.pages):
            raise IndexError(f"frame {frame} out of range for {path}; pages={len(tf.pages)}")
        return tf.pages[int(frame)].asarray()


def _read_pil_image_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as fh:
        img = Image.open(fh)
        img.load()
        return np.asarray(img)


def crop_patch(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    h, w = image.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    return image[y0:y1, x0:x1]


def overlay_points(image: np.ndarray, points: pd.DataFrame, out_path: Path, radius: int = 5) -> None:
    arr = (normalize_image(image) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, row in points.iterrows():
        x, y = float(row["x_px"]), float(row["y_px"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 40, 40), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


@dataclass
class ExtractionResult:
    rows: list[dict[str, Any]]
    overlays: list[str]
    warnings: list[str]


def extract_from_zip(
    *,
    zip_path: Path,
    table_root: Path,
    dataset: str,
    sequences: list[str],
    frames: list[int],
    patch_radius: int,
    max_points_per_frame: int,
    out_dir: Path,
    extracted_stack_dir: Path,
    overlay_stride: int,
    max_overlays: int,
    seed: int,
) -> ExtractionResult:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    overlays: list[str] = []
    warnings: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        member_info = {"_members": members}
        for sequence in sequences:
            table_path = table_root / dataset / f"{dataset}_{int(sequence):02d}_tracks.csv"
            if not table_path.exists():
                warnings.append(f"missing table for {dataset} seq {sequence}: {table_path}")
                continue
            table = pd.read_csv(table_path)
            for frame in frames:
                hits = find_image_members(member_info, f"{int(sequence):02d}", frame)
                stack_member = find_sequence_stack_member(members, f"{int(sequence):02d}")
                if not hits and stack_member is None:
                    warnings.append(f"no image member/stack match for {dataset} seq {sequence} frame {frame}")
                    continue
                try:
                    if hits:
                        member = hits[0]
                        image = _read_pil_image_from_zip(zf, member)
                    else:
                        member = str(stack_member)
                        stack_path = ensure_zip_member_extracted(zf, member, extracted_stack_dir)
                        image = read_tiff_page(stack_path, frame)
                except Exception as exc:
                    warnings.append(f"failed to read {member} frame {frame}: {type(exc).__name__}: {exc}")
                    continue
                pts = table[table["frame"].eq(frame)].copy()
                pts = pts[
                    pts["x_px"].between(0, image.shape[1] - 1)
                    & pts["y_px"].between(0, image.shape[0] - 1)
                ]
                if max_points_per_frame > 0 and len(pts) > max_points_per_frame:
                    pts = pts.iloc[np.sort(rng.choice(len(pts), size=max_points_per_frame, replace=False))]
                if (overlay_stride > 0 and frame % overlay_stride == 0) and (
                    max_overlays <= 0 or len(overlays) < max_overlays
                ):
                    overlay_path = out_dir / "plots" / f"{dataset}_{int(sequence):02d}_f{frame:04d}_overlay.png"
                    overlay_points(image, pts.head(min(len(pts), 500)), overlay_path)
                    overlays.append(str(overlay_path))
                for _, row in pts.iterrows():
                    patch = crop_patch(image, row["x_px"], row["y_px"], patch_radius)
                    feats = patch_features(patch)
                    if not feats:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "sequence": int(sequence),
                            "frame": int(frame),
                            "track_id": int(row["track_id"]),
                            "x_px": float(row["x_px"]),
                            "y_px": float(row["y_px"]),
                            "image_member": member,
                            "patch_radius": int(patch_radius),
                            **feats,
                        }
                    )
    return ExtractionResult(rows=rows, overlays=overlays, warnings=warnings)


def write_audit_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LaChance Image/Morphology Extraction Audit",
        "",
        "## Decision",
        "",
    ]
    raw_exists = bool(payload.get("raw_zip", {}).get("exists"))
    xml_shape = any(v.get("has_shape_like_attributes") for v in payload.get("xml_status", {}).values())
    if not xml_shape:
        lines.append("- XML files contain TrackMate point trajectories only; no contour/shape attributes were found.")
    if raw_exists:
        lines.append("- Raw ZIP is present; run `inspect` and then `extract` after checking image member layout.")
    else:
        lines.append("- Raw ZIP is not present; run `download` before image/morphology extraction.")
    lines.extend(
        [
            "",
            "## Payload",
            "",
            "```json",
            json.dumps(finite_json(payload), indent=2, ensure_ascii=False),
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "download", "inspect", "extract"], default="audit")
    parser.add_argument("--record-url", default=ZENODO_API)
    parser.add_argument("--file-key", default=DEFAULT_FILE_KEY)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--extracted-stack-dir", type=Path, default=DEFAULT_RAW_DIR / "extracted_stacks")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="MDCK_Bulk,MDCK_Edge")
    parser.add_argument("--dataset", default="MDCK_Bulk")
    parser.add_argument("--sequences", default="01")
    parser.add_argument("--frames", default="0,12,24,36,48")
    parser.add_argument("--patch-radius", type=int, default=24)
    parser.add_argument("--max-points-per-frame", type=int, default=256)
    parser.add_argument("--overlay-stride", type=int, default=12)
    parser.add_argument("--max-overlays", type=int, default=30)
    parser.add_argument("--max-preview-members", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-insecure-ssl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_list(args.datasets)
    rows = zenodo_files(args.record_url, allow_insecure_ssl=bool(args.allow_insecure_ssl))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out_dir / "zenodo_file_manifest.csv", index=False)
    selected = next((row for row in rows if row["key"] == args.file_key), None)
    if selected is None:
        raise FileNotFoundError(f"{args.file_key} not found in Zenodo record")
    zip_path = args.raw_dir / args.file_key

    if args.mode == "download":
        status = download_with_resume(
            str(selected["download"]),
            zip_path,
            expected_size=int(selected["size_bytes"]),
            checksum=str(selected.get("checksum", "")),
            allow_insecure_ssl=bool(args.allow_insecure_ssl),
        )
        (args.out_dir / "download_status.json").write_text(
            json.dumps(finite_json(status), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(args.out_dir / "download_status.json")
        return

    zip_status = inspect_zip(zip_path, max_members=int(args.max_preview_members))
    if args.mode == "inspect":
        (args.out_dir / "raw_zip_inspection.json").write_text(
            json.dumps(finite_json(zip_status), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(args.out_dir / "raw_zip_inspection.json")
        return

    if args.mode == "extract":
        if not zip_path.exists():
            raise FileNotFoundError(f"raw ZIP not found: {zip_path}; run --mode download first")
        result = extract_from_zip(
            zip_path=zip_path,
            table_root=args.table_root,
            dataset=args.dataset,
            sequences=parse_list(args.sequences),
            frames=[int(x) for x in parse_list(args.frames)],
            patch_radius=int(args.patch_radius),
            max_points_per_frame=int(args.max_points_per_frame),
            out_dir=args.out_dir,
            extracted_stack_dir=args.extracted_stack_dir,
            overlay_stride=int(args.overlay_stride),
            max_overlays=int(args.max_overlays),
            seed=int(args.seed),
        )
        feature_path = args.out_dir / "image_patch_features.csv"
        pd.DataFrame(result.rows).to_csv(feature_path, index=False)
        status = {
            "dataset": args.dataset,
            "sequences": parse_list(args.sequences),
            "frames": [int(x) for x in parse_list(args.frames)],
            "patch_radius": int(args.patch_radius),
            "max_points_per_frame": int(args.max_points_per_frame),
            "overlay_stride": int(args.overlay_stride),
            "max_overlays": int(args.max_overlays),
            "feature_path": feature_path,
            "rows": len(result.rows),
            "overlays": result.overlays,
            "warnings": result.warnings[:200],
            "warning_count": len(result.warnings),
        }
        (args.out_dir / "image_feature_extraction_status.json").write_text(
            json.dumps(finite_json(status), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(args.out_dir / "image_feature_extraction_status.json")
        return

    payload = {
        "dependencies": dependency_status(),
        "disk": disk_status(args.raw_dir),
        "selected_zenodo_file": selected,
        "raw_zip": zip_status,
        "xml_status": xml_attribute_status(ROOT / "new_data" / "lachance_epithelia" / "raw", datasets),
        "table_status": table_status(args.table_root, datasets),
        "recommended_next_step": "download_raw_zip" if not zip_status.get("exists") else "inspect_raw_zip",
    }
    (args.out_dir / "image_extraction_audit.json").write_text(
        json.dumps(finite_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = args.out_dir / "image_extraction_audit.md"
    write_audit_report(report_path, payload)
    print(report_path)


if __name__ == "__main__":
    main()
