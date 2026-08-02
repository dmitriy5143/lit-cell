# LIT-Cell Reproducibility Guide

## What Is Reproducible Here

The repository provides two explicitly different levels of reproducibility:

1. **Artifact verification without raw data.** Frozen CSV/JSON evidence can be
   checked for schema, internal consistency, protocol identity, and registered
   headline numbers.
2. **Raw-data and publication-scale re-execution.** A frozen preparation
   pipeline reconstructs the tracking-aligned 1,019-column causal grid from the
   licensed LaChance track tables and raw MDCK Bulk movie archive. The outer
   leave-one-movie-out orchestrator then rebuilds fold-local anchors, trains the
   causal filter, replays predictions before observations, and fits the bounded
   transport.

The included `smoke` command validates executable architecture and causal event
order. It is not presented as a numerical reproduction of the publication.
The raw-microscopy preparation and model fitting remain separate commands so a
reviewer can inspect the data contract before starting the long experiment.
The exact final grid is frozen by
`evidence/raw_context_v2_feature_contract.json`: 93,596 rows, 1,019 ordered
columns, and CSV SHA-256
`45f4b1db7949fd7fa6f791db27e7d8af6999ae7a9c0ece810fb54f1ad325de48`.
The v102 preflight rejects a grid whose ordered schema, byte size, or digest
differs from this registered input.

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
Byte-level feature hashes apply to the pinned reference environment. The
preparation runner also supports a dimension/schema audit for diagnostic ports,
but only the exact registered grid enters the frozen publication run.

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

## C2C12 External Structural Validation

The C2C12 phase-contrast corpus is a separate structural validation. Prepare 48
automatic and 48 manual track tables under:

```text
$C2C12_TABLE_ROOT/
  C2C12_Automatic/C2C12_Automatic_1??_tracks.csv
  C2C12_Automatic/C2C12_Automatic_2??_tracks.csv
  C2C12_Automatic/C2C12_Automatic_3??_tracks.csv
  C2C12_Manual/C2C12_Manual_1??_tracks.csv
  C2C12_Manual/C2C12_Manual_2??_tracks.csv
  C2C12_Manual/C2C12_Manual_3??_tracks.csv
```

Each pattern must resolve to 16 complete fields. The independent outer units
are experiments 1, 2, and 3, rotated as train/validation/test. Automatic and
manual annotations are never pooled. Run the primary E(2)-equivariant analysis:

```bash
python experiments/publication/run_c2c12_lit_cell_external_confirmation_v209.py \
  --table-root "$C2C12_TABLE_ROOT" \
  --kinds automatic \
  --objectives horizon_balanced \
  --operator-kind equivariant \
  --out-dir /absolute/path/to/c2c12_v209_automatic

python experiments/publication/run_c2c12_lit_cell_external_confirmation_v209.py \
  --table-root "$C2C12_TABLE_ROOT" \
  --kinds manual \
  --objectives horizon_balanced \
  --operator-kind equivariant \
  --out-dir /absolute/path/to/c2c12_v209_manual
```

The automatic primary gate requires positive h6 gain in all three held-out
experiments, h1 degradation no greater than 0.5%, a full-operator advantage
over own-only, wrong-cell, and stale-time controls, and zero causal donor
violations. Manual tracks are a secondary audit because most centroids are
interpolated. Compact frozen outputs are in `evidence/c2c12_v209/`.

## Rebuild the Causal Feature Grid

The raw MDCK Bulk archive is file
`MDCK_Bulk_Timelapse_Data_Sample_Tissues.zip` from Zenodo record `4959169`.
Its registered size is 9,305,087,422 bytes and its MD5 is
`e5b5add0c7526010f957374759809bb2`. Download it with the resumable acquisition
runner or obtain it directly under the dataset's license:

```bash
python experiments/publication/run_lachance_image_feature_extraction.py \
  --mode download \
  --raw-dir "$LACHANCE_DATA_ROOT/raw_timelapse" \
  --table-root "$LACHANCE_TABLE_ROOT" \
  --out-dir /absolute/path/to/acquisition_audit
```

Reconstruct the six tracking-aligned stacks, the exact central-cell index, the
multiscale image and tissue-flow packets, the observability packet, and the
final raw-context grid:

```bash
python scripts/reproduce_lit_cell.py features \
  --mode all \
  --table-root "$LACHANCE_TABLE_ROOT" \
  --raw-zip "$LACHANCE_DATA_ROOT/raw_timelapse/MDCK_Bulk_Timelapse_Data_Sample_Tissues.zip" \
  --stack-dir "$LACHANCE_DATA_ROOT/raw_timelapse/extracted_stacks/MDCK_Bulk_Timelapse_Data_Sample_Tissues" \
  --out-dir /absolute/path/to/lit_cell_features \
  --reference-check hash
```

The `features` command forwards to `scripts/prepare_lit_cell_features.py`. The
preparation is resumable. It records rows, columns, byte sizes, and hashes
for six stages in `feature_preparation_report.json`. A completed reference run
must report `matches_reference=true` for every stage. To audit an existing
preparation without rebuilding it, replace `--mode all` with `--mode verify`.
The exact historical stage dimensions and hashes are registered in
`evidence/raw_context_v2_feature_contract.json`.

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
- C2C12 experiment-external transport, controls, and observation-quality strata;
- the exact feature-preparation contract and dependency closure;
- registered comparator tiers and canonical project identity.

The older `v188` validator is retained as a provenance check. Its
`--require-publication-ready` flag intentionally reports historical packaging
tasks that are closed by the current repository-level release validator.

The same validation, package compilation, and unit tests can be run through a
single public entry point:

```bash
python scripts/reproduce_lit_cell.py verify
```

## Architecture Replay

Run a small graph-enabled, predict-before-observe sequence through the public
package and verify checkpoint round-tripping:

```bash
python scripts/reproduce_lit_cell.py smoke
```

This command uses synthetic observations solely to check event ordering,
identity state, uncertainty outputs, graph execution, and serialization.

## Full Movie-Level Re-execution

First validate the input tables, feature schema, six outer folds, frozen
hyperparameters, runner dependency closure, and exact job manifest:

```bash
python scripts/reproduce_lit_cell.py preflight \
  --table-root "$LACHANCE_TABLE_ROOT" \
  --feature-grid /absolute/path/to/raw_context_v2_feature_grid.csv \
  --out-dir /absolute/path/to/lit_cell_preflight
```

Then run the exact outer-movie causal filter followed by the fold-local
transport/Pareto stage:

```bash
python scripts/reproduce_lit_cell.py full \
  --table-root "$LACHANCE_TABLE_ROOT" \
  --feature-grid /absolute/path/to/raw_context_v2_feature_grid.csv \
  --out-dir /absolute/path/to/lit_cell_full \
  --device auto
```

The full command is intentionally long-running. Its first stage writes a
complete frozen command manifest and rebuilds the route/coordinate anchor
inside every outer fold. Its second stage restores those fold-local sequential
models and selects bounded transport parameters using only the corresponding
validation movie.

## Principal Workflows

The descriptive dispatcher forwards all remaining arguments to exact runners:

```bash
python scripts/run_sequential_cell_forecasting.py --list
python scripts/run_sequential_cell_forecasting.py online-core -- --help
python scripts/run_sequential_cell_forecasting.py fold-local-transport -- --help
python scripts/run_sequential_cell_forecasting.py frozen-confirmation -- --help
python scripts/run_sequential_cell_forecasting.py c2c12-external -- --help
```

Main stages:

```text
online-core
  chronological training and replay of the Student-t innovation filter

outer-lomo-benchmark
  exact six-fold orchestration, fold-local anchor construction and aggregation

online-neural-screen
  protocol-matched architecture-development screen; not a confirmation table

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

c2c12-external
  experiment-external C2C12 structural validation with automatic/manual tracks
```

Each runner exposes its exact required caches and paths through `--help`. Long
runs are resumable only where the original runner explicitly implements shard
or fold reuse; do not infer completion from a partially populated output folder.

Comparator tiers and the prohibition on pooling open-loop and streaming values
are documented in `docs/COMPARATORS.md` and encoded in
`evidence/comparators/comparator_protocol_matrix.csv`.

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
