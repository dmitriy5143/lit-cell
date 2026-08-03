# Frozen primary LIT-Cell states

This directory contains the trained recurrent states used by the registered
MDCK Bulk six-movie outer-LOMO evaluation. The release is a `6 movies x 3
optimizer seeds` grid, not one universal checkpoint. Each test movie has a
model trained without that movie; the transport objective is selected on a
separate validation movie.

The best cumulative operating point is `v166_h6_utility` (`h6_guard10` in the
transport contract): h6 component RMSE `5.500748634593833 px`, h6 R2
`0.9377208553285478`. The nearest-step operating point is `v166_h1_strict`:
h1 component RMSE `3.4743735539609095 px`.

`v188` is the evidence/comparator bundle that registers these states and their
results. It is not a separate neural model.

Validate every file and strictly instantiate all 18 networks:

```bash
python scripts/validate_frozen_model_release.py
```

Load one fold state:

```python
from lit_cell_forecasting import load_frozen_fold_state

frozen = load_frozen_fold_state(test_movie=1, seed=42)
model = frozen.model
metadata = frozen.payload["metadata"]
```

The checkpoints include the exact recurrent tensors and portable run metadata.
The coordinate anchor and bounded transport remain data-fitted components of
the full procedure. Rebuild them with `scripts/reproduce_lit_cell.py full` from
the documented feature grid and LaChance tables. Raw microscopy and the
approximately 2 GB disposable fold cache are not stored in Git.

