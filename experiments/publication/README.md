# Publication Runners

These files are the exact dependency closure of the final and diagnostic
experiments. They retain internal version suffixes so that frozen manifests and
source hashes remain interpretable.

Run them through the descriptive dispatcher when possible:

```bash
python scripts/run_sequential_cell_forecasting.py --list
```

All files in this directory calculate the repository root with
`Path(__file__).resolve().parents[2]`; sibling runner imports resolve from this
directory. Generated outputs default to `outputs/`, which is intentionally
ignored by Git.

Publication-critical runners are mapped in `docs/EXPERIMENTS.md`. Other files
preserve negative architecture branches used to justify the final design.
