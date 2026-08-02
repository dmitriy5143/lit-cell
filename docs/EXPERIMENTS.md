# Experiment Map

Internal numeric suffixes preserve the chronology and hashes of the research
workspace. Scientific names in the manuscript are deliberately version-free.

| Scientific role | Provenance runner | Frozen evidence |
|---|---|---|
| raw microscopy and feature preparation | `prepare_lit_cell_features.py` plus the six extraction/build modules | `evidence/raw_context_v2_feature_contract.json` |
| coordinate/velocity route anchor | `run_lachance_route_balanced_calibrator_v16.py` | incorporated in fold-local anchor caches |
| exact outer-movie orchestration | `run_lachance_online_lomo_benchmark_v102.py` | `evidence/v188/v188_primary_online_benchmark.csv` |
| online neural architecture screen | `run_lachance_online_architecture_benchmark_v99.py` | `evidence/comparators/comparator_protocol_matrix.csv` |
| online Student-t innovation filter | `run_lachance_causal_innovation_state_space_v97.py` | `evidence/v188/v188_primary_online_benchmark.csv` |
| fold-local bounded transport | `run_lachance_foldlocal_semigroup_confirmation_v157e.py` | `evidence/v188/v188_primary_online_movie_metrics.csv` |
| h1/h6 operating-point curve | `run_lachance_foldlocal_semigroup_pareto_v157h.py` | `evidence/h1_v205/v205_pareto_points.csv` |
| movies 10-16 frozen confirmation | `run_lachance_streaming_transport_confirmation_v160.py` | `evidence/v188/v188_configuration_unseen_confirmation.csv` |
| HUVEC/MDA nested evaluation | `run_lachance_external_movie_lomo_publication_v165.py` | `evidence/v188/v188_external_nested_lomo.csv` |
| KalmanNet comparison | `run_lachance_kalmannet_outer_lomo_v188.py` | `evidence/v188/v188_primary_online_benchmark.csv` |
| learned confirmation comparators | `run_lachance_confirmation_learned_comparators_v193.py` | `evidence/comparators/online_confirmation_aggregate.csv` |
| sparse deployment transport | `run_lachance_sparse_pareto_transport_v193.py` | `evidence/figure_sources/v193_sparse_pareto_aggregate.csv` |
| E(2)-equivariant field law | `run_mdck_equivariant_field_law_v197.py` | `evidence/figure_sources/v197_field_law_*` |
| effective potential audit | `run_mdck_effective_potential_audit_v198.py` | manuscript supplementary tables |
| equivariant graph bridge | `run_lachance_equivariant_graph_bridge_v199.py` | manuscript and Figure 7 source tables |
| finite functional dynamics | `run_mdck_effective_functional_dynamics_v200.py` | manuscript supplementary tables |
| probabilistic field closure | `run_lachance_probabilistic_graph_closure_v201.py` | architecture-search ledger |
| unseen-movie field confirmation | `run_lachance_equivariant_graph_unseen_v202.py` | manuscript and supplementary tables |
| DeepSea partial transfer | `run_deepsea_multimodal_validation_v204.py` | `evidence/deepsea_v204/` |
| h1 localization/scale audit | `run_lachance_h1_evidence_bundle_v205.py` | `evidence/h1_v205/` |
| LifeAct segmentation/identity gate | `run_lifeact_mdck_segmentation_identity_gate_v206.py` | `evidence/lifeact_v206_v208/` |
| LifeAct state and uncertainty | `run_lifeact_mdck_mechanochemical_state_gate_v207.py` | `evidence/lifeact_v206_v208/` |
| LifeAct uncertainty closure | `evaluate_lifeact_mdck_state_uncertainty_gate_v208.py` | `evidence/lifeact_v206_v208/` |
| C2C12 reliability-aware conditional mean | `run_c2c12_reliability_transport_v168.py` | incorporated in the v209 outer rotations |
| C2C12 experiment-external innovation transport | `run_c2c12_lit_cell_external_confirmation_v209.py` | `evidence/c2c12_v209/` |

## Evidence Classes

**Frozen release accounting.** The v188 H1/H2 table on MDCK Bulk movies 1-6
uses exact two-sided movie-level sign-flip tests and Holm correction across two
hypotheses. It is a machine-readable publication-era contract, not a claim of
prospective preregistration. In the conservative research chronology, h1 was
the initial confirmatory endpoint, whereas h6 was developed on movies 1-6 and
then frozen before the separate movies 10-16 evaluation.

**Configuration-frozen.** Movies 10-16 were not used to choose the reported
configuration, but the broader project had previously inspected this data
family. They are stronger than another development split and weaker than a
fully prospective untouched cohort.

**Nested external evaluation.** HUVEC and MDA-MB-231 repeat the learning
procedure while excluding each outer movie. This demonstrates within-domain
reproducibility, not zero-shot weight transfer.

**Mechanistic/exploratory.** Field, potential, visual, mechanical, and modality
experiments test interpretation or observability. Their controls are important,
but they do not enlarge the confirmatory H1/H2 family after the fact.

**External structural validation.** C2C12 uses three microscopy experiments as
rotating train/validation/test units. Automatic tracks are primary and manual
tracks are a separate observation-process audit. This validates reuse of the
delayed local-innovation operator after domain-specific fitting; it is neither
zero-shot weight transfer nor an absolute C2C12 leaderboard claim.

## Historical Search

The complete branch-level outcome ledger is committed as
`evidence/architecture_search_ledger.csv`. Failed branches remain scientifically
useful because they separate geometric candidate coverage, train-time target
identifiability, causal route observability, and true conditional-mean gain.
