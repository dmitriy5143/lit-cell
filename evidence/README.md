# Frozen Evidence

This directory contains compact, reviewable outputs rather than raw microscopy
or checkpoints.

- `v188/`: primary six-movie protocol, benchmark, controls, robustness,
  multiplicity contract, external-domain tables, and artifact manifest.
- `deepsea_v204/`: partial transfer and morphology-state gates on DeepSea.
- `h1_v205/`: nearest-step scale, localization, Pareto, and causal-control audit.
- `lifeact_v206_v208/`: segmentation identity, cell-state, and uncertainty gates.
- `c2c12_v209/`: experiment-external automatic/manual structural validation,
  E(2)-equivariant operator coefficients, hard controls, density/quality strata,
  and the explicit non-leaderboard interpretation boundary.
- `article_numeric_audit/`: machine-readable registered quantitative claims and
  their source artifacts.
- `figure_sources/`: compact tables required to reconstruct plots.
- `architecture_search_ledger.csv`: branch-level result and decision ledger.
- `raw_context_v2_source_dictionary.csv`: source-column dictionary for the
  exact 1,019-column final v102 grid from which fold-local model inputs are
  selected.
- `raw_context_v2_feature_contract.json`: ordered-schema digest, byte-level
  reference digest, dimensions, feature-family counts, and frozen construction
  settings for that final grid.

Absolute source paths in old manifests are provenance records from the original
workspace; portable validation uses the files committed here. A successful
schema/number check does not imply that raw data or checkpoints are included.
