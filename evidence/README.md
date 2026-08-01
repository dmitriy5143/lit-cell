# Frozen Evidence

This directory contains compact, reviewable outputs rather than raw microscopy
or checkpoints.

- `v188/`: primary six-movie protocol, benchmark, controls, robustness,
  multiplicity contract, external-domain tables, and artifact manifest.
- `deepsea_v204/`: partial transfer and morphology-state gates on DeepSea.
- `h1_v205/`: nearest-step scale, localization, Pareto, and causal-control audit.
- `lifeact_v206_v208/`: segmentation identity, cell-state, and uncertainty gates.
- `article_numeric_audit/`: machine-readable registered claims used by the
  manuscript checker.
- `figure_sources/`: compact tables required to reconstruct plots.
- `architecture_search_ledger.csv`: branch-level result and decision ledger.
- `raw_context_v2_source_dictionary.csv`: source-column dictionary for the
  1,093-column grid from which fold-local model inputs are selected.

Absolute source paths in old manifests are provenance records from the original
workspace; portable validation uses the files committed here. A successful
schema/number check does not imply that raw data or checkpoints are included.
