# LIT-Cell: Local Innovation Transport for Cell Motion

**LIT-Cell** is the canonical name of the method and software described here.
This repository contains its experiment runners, frozen evidence, and reusable
implementation for causal online forecasting of collective cell motion. A
prediction for transition `t -> t+1` is committed before frame
`t+1` is observed. Once that transition is complete, its innovation and the
completed innovations of nearby cells may update the next forecast.

The reportable model has four parts:

1. a coordinate and velocity anchor with a bank of residual trajectory experts;
2. a recurrent causal state filter with a Student-t next-step distribution;
3. a bounded multiscale graph transport of completed neighbour innovations;
4. fold-external uncertainty calibration and a sparse deployment equivalent.

The graph is therefore neither a future-looking interaction prior nor an
unconstrained correction. It transfers only errors of transitions that were
fully observed before the next prediction was issued.

## Main Results

All values below use movie-level outer evaluation and the streaming/receding-h1
protocol frozen in the evidence contract.

| Evaluation | Operating point | h1 RMSE | h6 RMSE | h6 R2 | Main comparison |
|---|---|---:|---:|---:|---|
| MDCK Bulk movies 1-6 | strict h1 | 3.474 px | 6.785 px | 0.905 | 13.28% h6 gain over no-update |
| MDCK Bulk movies 1-6 | cumulative h6 | 3.808 px | 5.501 px | 0.938 | 29.76% h6 gain over no-update |
| MDCK Bulk movies 10-16 | frozen confirmation | 3.721 px | 4.820 px | 0.952 | 25.82% h6 gain, 7/7 movies |
| HUVEC | nested movie exclusion | - | 1.440 px | 0.973 | 11.02% h6 gain, 18/18 movies |
| MDCK Edge | frozen transport kernel | - | 5.261 px | 0.952 | 14.71% h6 gain, 3/3 seeds |
| MDA-MB-231 | nested movie exclusion | - | 31.337 px | 0.024 | 6.33% h6 gain, 17/17 movies |
| C2C12 automatic tracks | experiment-level structural transfer | 4.355 px | 5.088 px | 0.707 | 1.22% h6 gain, 3/3 experiments |

On the LaChance protocol, the method has the lowest h6 RMSE among the completed,
protocol-matched comparators in this study. We do **not** claim a global state
of the art across trajectory-forecasting tasks: no established benchmark uses
the same cell data, online observation schedule, split unit, and metric.

The C2C12 row validates transfer of the completed-innovation mechanism rather
than absolute benchmark dominance. Constant velocity remains stronger there at
h6 (`4.860 px`), consistent with endpoint cancellation in noisy automatic
centroids; the full real update nevertheless beats no-update, own-only,
wrong-cell, and stale-time controls.

## Repository Map

- `src/lit_cell_forecasting/`: compact reusable operators for state filtering,
  bounded transport, sparse neighbourhoods, and equivariant field laws.
- `experiments/publication/`: exact publication runners and their dependency
  closure. Historical numeric suffixes are mapped to scientific roles in
  `docs/EXPERIMENTS.md`.
- `experiments/history/`: index of rejected or diagnostic architecture branches.
- `evidence/`: frozen result tables, protocol contracts, control outcomes, and
  claim-scope ledgers. Raw microscopy data and checkpoints are not committed.
- `scripts/run_sequential_cell_forecasting.py`: descriptive dispatcher for the
  principal publication workflows.
- `scripts/prepare_lit_cell_features.py`: frozen raw-microscopy-to-feature-grid
  pipeline with stage-level reference hashes.

## Fast Verification

Create an environment with Python 3.11 and install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Validate the frozen evidence and registered claims without the raw data:

```bash
python scripts/validate_publication_release.py
```

Compile all experiment entry points:

```bash
python -m compileall -q src experiments scripts
python -m unittest discover -s tests -v
```

List the principal workflows:

```bash
python scripts/run_sequential_cell_forecasting.py --list
```

Run the architecture-level end-to-end smoke replay:

```bash
python scripts/reproduce_lit_cell.py smoke
```

The canonical naming contract is recorded in
[`docs/NAMING.md`](docs/NAMING.md). Exact full-data preflight and reproduction
commands are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Data acquisition, exact commands, and the distinction between reproduced,
frozen, and exploratory evidence are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Scientific Scope

The strongest supported conclusion is that recently completed residual motion
contains a causal, spatially local, identity-specific signal that improves
cumulative online cell-motion forecasts and transfers structurally to C2C12
under experiment-level exclusion. The effective field admits a compact
E(2)-equivariant and approximately dissipative representation, but the fitted
functional is not identified as physical energy, traction, or stress. Visual,
segmentation, and direct mechanics branches were retained only when they passed
wrong-cell and time-shuffle controls; none improved the conditional mean of the
final LaChance model.

See [`docs/CLAIM_SCOPE.md`](docs/CLAIM_SCOPE.md) before quoting results.
The boundary between protocol-matched comparisons and earlier fixed-origin
architecture screens is recorded in [`docs/COMPARATORS.md`](docs/COMPARATORS.md).

## Authors and License

- Dmitry Stanislavchuk-Abovsky: research design, implementation, experiments,
  diagnostics, and artifact preparation.
- Andrey P. Zakharov: scientific supervision, methodology, and expert feedback.

The repository license is defined in [`LICENSE`](LICENSE).
