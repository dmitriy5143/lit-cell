# Dataset and Acquisition Contract

This document separates the datasets that support the reportable forecasting
claim from datasets used only to audit observation quality, morphology, or
mechanical interpretation. Raw microscopy is not redistributed by this
repository; use each source under its original terms.

## Reportable Evaluation Domains

| Dataset | Independent unit | Role | Official source | Preparation |
|---|---|---|---|---|
| LaChance MDCK Bulk | movie | primary online forecasting and frozen confirmation | [Zenodo 4959169](https://doi.org/10.5281/zenodo.4959169) | `audit_lachance_raw_sample.py`, `build_simple_trackmate_xml_tables.py`, `reproduce_lit_cell.py features` |
| LaChance MDCK Edge | movie | boundary-geometry guard | [Zenodo 4959169](https://doi.org/10.5281/zenodo.4959169) | same LaChance TrackMate conversion |
| LaChance HUVEC | movie | nested within-domain reproduction | [Zenodo 4959169](https://doi.org/10.5281/zenodo.4959169) | same LaChance TrackMate conversion |
| LaChance MDA-MB-231 | movie | nested within-domain reproduction under weak absolute predictability | [Zenodo 4959169](https://doi.org/10.5281/zenodo.4959169) | same LaChance TrackMate conversion |
| C2C12 | experiment, then field | external structural validation and tracking-quality audit | [Scientific Data record](https://doi.org/10.1038/sdata.2018.237), [OSF project](https://osf.io/ysaq2/) | `prepare_c2c12_online_tracks_v97.py` |

HUVEC and MDA-MB-231 are fitted within their own domains under outer movie
exclusion; they are not zero-shot transfers of LaChance MDCK weights. C2C12
reuses the completed-innovation operator after domain fitting. Constant
velocity remains stronger in absolute C2C12 h6 RMSE, so this dataset supports
structural transfer rather than a leaderboard claim.

## Observability and Interpretation Audits

| Dataset | Role | Official source | Acquisition runner |
|---|---|---|---|
| DeepSea | exact masks, identities, and morphology gate | [DeepSea](https://deepseas.org/datasets/) | `experiments/publication/download_deepsea_manifest_v204.py` |
| LifeAct-MDCK | actin polarity, contact, segmentation, and uncertainty gate | [Zenodo 20047603](https://doi.org/10.5281/zenodo.20047603) | `experiments/publication/fetch_lifeact_mdck_sequences_v206.py` |
| GigaScience wound healing | collective-front and field-transfer audit | [GigaDB 100118](https://doi.org/10.5524/100118) | `intake_gigascience_wound_healing_v167.py` |
| MDCK force-motion | measured traction/stress bridge and intervention audit | [Figshare collection](https://doi.org/10.6084/m9.figshare.c.4945206) | `intake_mdck_force_motion_mechanics_v150.py` |
| S-BIAD365 and Allen state data | privileged-state observability audit | source URLs frozen by the runner | `intake_future_state_sources_v173.py` |
| SSBD identified-cell data | cell-specific driver upper-bound audit | source URL frozen by the runner | `audit_identified_cell_driver_upper_bound_v171.py` |
| Cell Tracking Challenge | candidate-dataset screening | [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/) | `screen_collective_dataset_candidates.py` |

These datasets do not enlarge the main forecasting cohort. DeepSea supplied a
partial system transfer but did not validate a morphology correction to the
conditional mean. LifeAct supplied an exploratory uncertainty-scale signal.
Measured mechanics did not pass the causal transfer gate. Those negative
results are part of the claim boundary, not missing positive validations.

## LaChance Reconstruction

The TrackMate XML archives and MDCK Bulk time-lapse archive are listed by
Zenodo record 4959169. Audit an acquired archive and convert its XML files:

```bash
python scripts/audit_lachance_raw_sample.py --help
python scripts/build_simple_trackmate_xml_tables.py --help
```

Then follow `REPRODUCIBILITY.md` to reconstruct the exact 1,019-column causal
feature grid. The accepted grid has 93,596 rows and SHA-256
`45f4b1db7949fd7fa6f791db27e7d8af6999ae7a9c0ece810fb54f1ad325de48`.

## C2C12 Reconstruction

Download the annotation material from the official OSF project and arrange the
extracted XML files as:

```text
C2C12_tracking/
  exp1/{automatic,human}/
  exp2/{automatic,human}/
  exp3/{automatic,human}/
```

Each annotation kind must contain 16 fields per experiment. Convert without
mixing manual and automatic tracks:

```bash
python scripts/prepare_c2c12_online_tracks_v97.py \
  --source /absolute/path/to/C2C12_tracking \
  --table-root /absolute/path/to/c2c12_online/tables
```

The converter preserves interpolation flags, splits discontinuous identities,
and writes `c2c12_split_contract.json`. The publication runner verifies 48
complete fields for each annotation kind before fitting.

## DeepSea Frozen Manifest

The exact 19,890-object Google Drive inventory used by the DeepSea acquisition
runner is committed at
`evidence/data_manifests/deepsea_v204_drive_manifest.json` with SHA-256
`6648e246207fb298c525b8946ec94e5d0ba40cd1a6d23e7b734b80fd152dfca0`.
The downloader is resumable and defaults to this manifest.

## Migration Boundary

A clean clone is sufficient to verify all 287 manuscript quantities from 46
committed source tables. Full model fitting additionally needs the licensed raw
data or prepared tables. Platform-specific environments, extracted duplicate
archives, caches, and the original 149 GB research workspace are not required.
A custom local Cellpose checkpoint is relevant only to reproduction of one
exploratory visual branch and is not required for the final coordinate-side
conditional mean.

The machine-readable acquisition inventory is
`evidence/data_reacquisition_manifest.csv`. Availability statements in that
file distinguish verified reacquisition paths from sources that still require
manual download through the provider interface.
