#!/usr/bin/env python3
"""Build the publication PDF from committed LaTeX sources and figures."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
SOURCE = MANUSCRIPT / "main_ru.tex"
BUILD_DIR = ROOT / "tmp" / "manuscript_build"
OUTPUT = ROOT / "output" / "pdf" / "sequential_cell_motion_forecasting_ru.pdf"


def executable(name: str, override: str | None = None) -> str | None:
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Executable does not exist: {candidate}")
        return str(candidate)
    return shutil.which(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tectonic",
        default=os.environ.get("TECTONIC"),
        help="Path to tectonic; defaults to TECTONIC or PATH.",
    )
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    tectonic = executable("tectonic", args.tectonic)
    if tectonic:
        command = [
            tectonic,
            "-X",
            "compile",
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(BUILD_DIR),
            SOURCE.name,
        ]
        subprocess.run(command, cwd=MANUSCRIPT, check=True)
        built = BUILD_DIR / "main_ru.pdf"
    else:
        latexmk = executable("latexmk")
        if not latexmk:
            raise RuntimeError("Install tectonic or latexmk, or set TECTONIC=/absolute/path")
        command = [
            latexmk,
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={BUILD_DIR}",
            SOURCE.name,
        ]
        subprocess.run(command, cwd=MANUSCRIPT, check=True)
        built = BUILD_DIR / "main_ru.pdf"

    if not built.is_file() or built.stat().st_size < 100_000:
        raise RuntimeError(f"Expected publication PDF was not produced: {built}")
    shutil.copy2(built, OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")

    if not args.keep_build:
        for suffix in (".aux", ".out", ".xdv"):
            (BUILD_DIR / f"main_ru{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
