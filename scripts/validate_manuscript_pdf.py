#!/usr/bin/env python3
"""Render and validate the final manuscript PDF as a publication artifact."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageStat
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "output" / "pdf" / "sequential_cell_motion_forecasting_ru.pdf"
REPORT = ROOT / "evidence" / "manuscript_pdf_validation.json"


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required Poppler command is missing: {name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--minimum-pages", type=int, default=25)
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    if not pdf.is_file() or pdf.stat().st_size < 100_000:
        raise AssertionError(f"Missing or unexpectedly small PDF: {pdf}")

    pdftoppm = require_tool("pdftoppm")

    reader = PdfReader(pdf)
    pages = len(reader.pages)
    first_box = reader.pages[0].mediabox
    width_pt, height_pt = float(first_box.width), float(first_box.height)
    if pages < args.minimum_pages:
        raise AssertionError(f"Unexpectedly short manuscript: {pages} pages")
    if abs(width_pt - 595.276) > 2 or abs(height_pt - 841.89) > 2:
        raise AssertionError(f"Expected A4 pages, found {width_pt} x {height_pt} pt")

    fonts: dict[str, object] = {}
    for page in reader.pages:
        resources = page.get("/Resources")
        resources = resources.get_object() if resources else {}
        page_fonts = resources.get("/Font", {})
        page_fonts = page_fonts.get_object() if hasattr(page_fonts, "get_object") else page_fonts
        for name, reference in page_fonts.items():
            fonts[str(name)] = reference.get_object()
    if not fonts:
        raise AssertionError("No fonts found in the PDF")
    for name, font in fonts.items():
        descriptors = []
        descriptor = font.get("/FontDescriptor")
        if descriptor:
            descriptors.append(descriptor.get_object())
        for descendant in font.get("/DescendantFonts", []):
            descriptor = descendant.get_object().get("/FontDescriptor")
            if descriptor:
                descriptors.append(descriptor.get_object())
        embedded = any(
            descriptor.get(key) is not None
            for descriptor in descriptors
            for key in ("/FontFile", "/FontFile2", "/FontFile3")
        )
        if not embedded:
            raise AssertionError(f"Font is not embedded: {name}")

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = [
        "Аннотация",
        "Архитектурный поиск",
        "Последовательный перенос",
        "Методы",
        "Обсуждение",
        "Заключение",
        "Список литературы",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise AssertionError(f"Missing expected manuscript sections: {missing}")
    if "�" in text:
        raise AssertionError("Text extraction contains replacement glyphs")

    render_dir = ROOT / "tmp" / "pdf_validation_pages"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)
    prefix = render_dir / "page"
    subprocess.run([pdftoppm, "-r", "110", "-png", str(pdf), str(prefix)], check=True)
    images = sorted(render_dir.glob("page-*.png"))
    if len(images) != pages:
        raise AssertionError(f"Rendered {len(images)} of {pages} pages")

    nonblank_fraction: list[float] = []
    for image_path in images:
        with Image.open(image_path).convert("L") as image:
            if image.width < 900 or image.height < 1200:
                raise AssertionError(f"Low-resolution rendered page: {image_path.name}")
            if ImageStat.Stat(image).mean[0] > 253.8:
                raise AssertionError(f"Nearly blank rendered page: {image_path.name}")
            histogram = image.histogram()
            dark = sum(histogram[:245])
            nonblank_fraction.append(dark / (image.width * image.height))

    log = ROOT / "tmp" / "manuscript_build" / "main_ru.log"
    warnings: list[str] = []
    if log.is_file():
        log_text = log.read_text(encoding="utf-8", errors="replace")
        fatal_patterns = [
            "Undefined control sequence",
            "Missing character",
            "Overfull \\hbox",
            "Overfull \\vbox",
        ]
        hits = [pattern for pattern in fatal_patterns if pattern in log_text]
        if hits:
            raise AssertionError(f"LaTeX log contains layout/content failures: {hits}")
        warnings = re.findall(r"Underfull \\[hv]box[^\n]*", log_text)

    payload = {
        "status": "PASS",
        "pdf": str(pdf.relative_to(ROOT)),
        "bytes": pdf.stat().st_size,
        "pages": pages,
        "page_size_pt": [width_pt, height_pt],
        "embedded_fonts": len(fonts),
        "rendered_pages": len(images),
        "minimum_nonblank_fraction": min(nonblank_fraction),
        "underfull_box_warnings": len(warnings),
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
