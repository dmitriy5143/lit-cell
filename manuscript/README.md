# Manuscript build

The committed figures are sufficient to rebuild the manuscript PDF. Run:

```bash
python scripts/build_manuscript.py
python scripts/validate_manuscript_pdf.py
```

Set `TECTONIC=/absolute/path/to/tectonic` when Tectonic is not on `PATH`.
`latexmk` with XeLaTeX is used as a fallback. The LaTeX source prefers Times New
Roman, Arial, and STIX Two Math and falls back to their TeX Gyre equivalents.

Figures 1-6 can be regenerated from the original microscopy data and committed
evidence. Figure 1 requires the LaChance raw movie stack:

```bash
LACHANCE_DATA_ROOT=/absolute/path/to/lachance_epithelia \
  python manuscript/build_cell_motion_latex_figures.py
python manuscript/build_prx_equivariant_field_figure_v197.py
python manuscript/build_h1_pareto_figure.py
python manuscript/audit_cell_motion_latex_figures.py
```

The figure audit validates the numerical source tables. The PDF validator then
checks page dimensions, embedded fonts, expected sections, rendered page count,
blank pages, replacement glyphs, and critical LaTeX layout warnings.
