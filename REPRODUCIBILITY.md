# Reproducibility Guide

## Reproducible Scope

The reportable RSCF path is self-contained in this repository:

1. table loading and movie-level splitting;
2. self-only and self-flow baselines;
3. route/state-aware radial message passing;
4. causal candidate generation and oracle diagnostics;
5. learned candidate-energy/transition-critic experiments and controls.

The raw trajectory tables are distributed separately and are not committed due
to their size and provenance. Reproduction therefore requires the same table
release, or tables converted to the schema described below.

## Reference Environment

- Python 3.11.9
- Apple Silicon arm64, macOS
- package versions pinned in `requirements.txt`
- CPU and Apple MPS are supported by the main runners

Create the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Input Data

Set:

```bash
export TABLE_ROOT=/absolute/path/to/lachance_epithelia/tables
```

Expected layout:

```text
$TABLE_ROOT/
  MDCK_Bulk/*_tracks.csv
  MDCK_Edge/*_tracks.csv
  MDAMB231/*_tracks.csv
  HUVEC/*_tracks.csv
```

Required columns:

```text
FRAME or frame
TRACK_ID or track_id
x_px
y_px
```

Rows represent cell observations. A track must contain unique frame indices.
Velocity, target and quality columns may be present, but the main LaChance
runners reconstruct their causal displacement features from coordinates.

Validate the tables before training:

```bash
python scripts/validate_lachance_tables.py --table-root "$TABLE_ROOT"
```

## Exact Defense Runs

The following command reproduces the route/state radial message-passing runs
used for the defense comparison:

```bash
TABLE_ROOT="$TABLE_ROOT" bash scripts/reproduce_defense_runs.sh
```

Reference aggregate values from the original three-seed runs:

```text
MDCK_Bulk h6:
  self_flow RMSE = 21.6278 px
  mp_gated_radial RMSE = 20.1205 px
  gain = 6.97%

MDCK_Edge h6:
  self_flow RMSE = 20.8864 px
  mp_gated_radial RMSE = 19.9183 px
  gain = 4.63%
```

Small numerical differences are possible across CPU/MPS devices, PyTorch
kernels and operating systems. The expected qualitative check is positive gain
for all three seeds.

## Candidate-Energy Path

Candidate coverage:

```bash
python scripts/run_lachance_candidate_oracle.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/candidate_oracle \
  --cell-types MDCK_Bulk,MDCK_Edge \
  --horizon 6 \
  --seeds 7,42,123 \
  --sobol-count 8 \
  --gaussian-count 8
```

Learned scorer/control sweep:

```bash
python scripts/run_lachance_oracle_signal_sweep.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/oracle_signal_sweep \
  --cell-types MDCK_Bulk,MDCK_Edge \
  --horizons 6,4 \
  --seeds 7,42,123
```

## Legacy Utilities

Files under `src/` are compact earlier baseline/backbone utilities retained for
reference. The defense reproduction path uses the `scripts/run_lachance_*.py`
runners.

`run_clean_spatial_identifiability_test.py` is a historical PSC/HSC diagnostic.
It expects previously generated prior profiles and is not part of the MDCK
defense reproduction path.
