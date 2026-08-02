#!/usr/bin/env python3
"""Build a portable SHA-256 manifest for the public LIT-Cell release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "release_manifest.json"
EXCLUDED = {
    OUTPUT,
    ROOT / "evidence" / "publication_release_validation.json",
    ROOT / "evidence" / "manuscript_pdf_validation.json",
    ROOT / "evidence" / "v188" / "v188_validation_report.json",
    ROOT / "scripts" / "build_manuscript.py",
    ROOT / "scripts" / "validate_manuscript_pdf.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    directories = [
        ROOT / "evidence",
        ROOT / "docs",
        ROOT / "src" / "lit_cell_forecasting",
        ROOT / "experiments" / "publication",
        ROOT / "scripts",
        ROOT / "tests",
    ]
    files: list[Path] = []
    for directory in directories:
        for path in directory.rglob("*"):
            if not path.is_file() or path in EXCLUDED:
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".aux", ".log", ".out"}:
                continue
            files.append(path)
    files.extend(
        path
        for path in (
            ROOT / "README.md",
            ROOT / "REPRODUCIBILITY.md",
            ROOT / "CITATION.cff",
            ROOT / "LICENSE",
            ROOT / "pyproject.toml",
            ROOT / "requirements.txt",
            ROOT / "requirements-vision.txt",
        )
        if path.is_file()
    )
    return sorted(files)


def main() -> None:
    entries = {
        str(path.relative_to(ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in selected_files()
    }
    OUTPUT.write_text(
        json.dumps(
            {
                "schema": "publication_release_manifest_v1",
                "algorithm": "sha256",
                "files": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(entries)} files")


if __name__ == "__main__":
    main()
