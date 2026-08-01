# Reproducibility Guide

## What Is Reproducible Here

The repository provides three levels of reproducibility:

1. **Artifact verification without raw data.** Frozen CSV/JSON evidence can be
   checked for schema, internal consistency, protocol identity, and registered
   headline numbers.
2. **Figure and manuscript reconstruction.** Publication figures and the PDF can
   be rebuilt from committed evidence, except Figure 1, which requires the raw
   LaChance movie stack for the microscopy background.
3. **Model re-execution.** Exact runners are provided, but large raw data,
   prepared feature grids, anchor caches, and model checkpoints must be supplied
   locally according to their original licenses.

The historical filename suffixes are provenance identifiers, not names used in
the scientific presentation. Their mapping is in `docs/EXPERIMENTS.md`.

## Environment

Reference environment:

```text
Python 3.11.9
NumPy 2.0.2
pandas 2.2.3
SciPy 1.17.1
scikit-learn 1.8.0
PyTorch 2.8.0
PyTorch Geometric 2.7.0
```

Installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The external segmentation/data-acquisition studies additionally require:

```bash
python -m pip install -r requirements-vision.txt
```

Apple MPS and CPU execution are supported. Exact floating-point identity across
hardware is not expected; protocol keys, causal controls, signs of movie-level
effects, and tolerance-bounded headline metrics are the reproducibility targets.

## LaChance Data Contract

Set the roots explicitly:

```bash
export LACHANCE_TABLE_ROOT=/absolute/path/to/lachance_epithelia/tables
export LACHANCE_DATA_ROOT=/absolute/path/to/lachance_epithelia
```

Frozen provenance tables use the portable placeholder `<AIRI_SOURCE_ROOT>` for
the original research workspace. It is descriptive metadata only; release
validation reads the committed evidence directly and does not require that
historical workspace.

Expected table layout:

```text
$LACHANCE_TABLE_ROOT/
  MDCK_Bulk/*.csv
  MDCK_Edge/*.csv
  MDAMB231/*.csv
  HUVEC/*.csv
```

Each row is a cell observation. Required semantic fields are movie/sequence,
frame, track identity, and x/y centroid. Individual source files use aliases
documented by their loader. A track must have unique frame indices. Every final
split is by whole movie, never by shuffled rows or frames.

Prepared artifacts used by the full model include:

- a movie-aligned route/coordinate anchor cache;
- a feature index and causal feature grid;
- fold-local checkpoints for the sequential state filter;
- completed residual packets keyed by `(sequence, frame, track_id)`.

All normalizers, feature filters, route labels, and calibration parameters are
fit without the outer test movie. The test target is used only after a prediction
has been committed.

## Verify the Frozen Release

Run:

```bash
python scripts/validate_publication_release.py
```

The validator checks:

- the six-movie protocol contract and ordered key hashes;
- the two-hypothesis multiplicity contract;
- registered primary and confirmation metrics;
- causal control direction and movie coverage;
- external-domain result schemas;
- late DeepSea, h1, and LifeAct evidence;
- numbers embedded in the manuscript;
- existence and dimensions of publication figures.

The older `v188` validator is retained as a provenance check. Its
`--require-publication-ready` flag intentionally reports historical packaging
tasks that are closed by the current repository-level release validator.

## Principal Workflows

The descriptive dispatcher forwards all remaining arguments to exact runners:

```bash
python scripts/run_sequential_cell_forecasting.py --list
python scripts/run_sequential_cell_forecasting.py online-core -- --help
python scripts/run_sequential_cell_forecasting.py fold-local-transport -- --help
python scripts/run_sequential_cell_forecasting.py frozen-confirmation -- --help
```

Main stages:

```text
online-core
  chronological training and replay of the Student-t innovation filter

fold-local-transport
  bounded multiscale transport, wrong-cell/stale-time controls, outer movies 1-6

frozen-confirmation
  configuration-frozen evaluation on MDCK Bulk movies 10-16

external-lomo
  nested whole-movie evaluation on HUVEC and MDA-MB-231

learned-comparators
  protocol-matched GRU, HGBDT, KalmanNet and related baselines

field-law / graph-bridge / field-dynamics
  E(2)-equivariant operator, graph surrogate and effective functional analyses
```

Each runner exposes its exact required caches and paths through `--help`. Long
runs are resumable only where the original runner explicitly implements shard
or fold reuse; do not infer completion from a partially populated output folder.

## Build the Manuscript

Figures 1-6:

```bash
LACHANCE_DATA_ROOT="$LACHANCE_DATA_ROOT" \
  python manuscript/build_cell_motion_latex_figures.py
```

Mechanistic field figure and the h1/h6 trade-off figure:

```bash
python manuscript/build_prx_equivariant_field_figure_v197.py
python manuscript/build_h1_pareto_figure.py
```

Figure checks:

```bash
python manuscript/audit_cell_motion_latex_figures.py
```

PDF:

```bash
python scripts/build_manuscript.py
```

The publication QA script renders every page and checks A4 dimensions, embedded
fonts, expected sections, raster dimensions, replacement glyphs, blank pages,
and critical LaTeX layout warnings:

```bash
python scripts/validate_manuscript_pdf.py \
  output/pdf/sequential_cell_motion_forecasting_ru.pdf
```

## Statistical Unit and Multiplicity

The independent unit is an outer test movie. The confirmatory family contains
exactly two hypotheses: the strict h1 operating point and the cumulative h6
operating point, each compared with the matched no-update prediction. Exact
two-sided sign-flip tests are corrected by Holm across these two hypotheses.
Other horizon-wise, one-sided, mechanistic, and external-domain contrasts are
explicitly exploratory or secondary unless stated otherwise.

Bootstrap intervals resample movies, not cell rows. The main prediction tables
use 20,000 movie-level bootstrap repetitions; diagnostic field curves use 2,000.

## Interpretation Boundary

Streaming h6 is the sum of six predictions issued sequentially, with a new
observation available before each next prediction. It is not an open-loop
six-step forecast from one initial frame and must not be compared with a
fixed-origin candidate oracle as though both solved the same information task.

The local operator transports completed innovations. Its equivariant reduction
supports a local kinematic field interpretation. It does not establish measured
force, traction, stress, or thermodynamic energy.
