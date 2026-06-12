# Route-State Candidate Forecasting for Collective Cell Migration

This repository is the clean code snapshot for the project defense:

**Interpretable GNN for collective cell migration with route/state-aware backbone and calibrated candidate-energy head.**

The repository intentionally contains code and instructions only. Raw trajectory tables, generated CSV results, plots and checkpoints are not committed.

## Research Question

Can local cell neighbours provide a stable, interpretable predictive signal for displacement forecasting, and can a model select a plausible future trajectory region without using the true future at inference time?

The final project architecture is **Route-State Candidate Forecasting (RSCF)**:

1. Encode cell history and local route/state graph.
2. Produce a strong causal backbone proposal.
3. Generate a candidate cloud of possible future displacements.
4. Score candidates with a calibrated candidate-energy head.
5. Predict a bounded residual mixture over the candidate region.

The physical prior is treated as a weak structural channel/control, not as a standalone future-configuration score.

## Repository Structure

- `REPRODUCIBILITY.md` - exact environment, data contract and defense-run protocol.
- `scripts/reproduce_defense_runs.sh` - three-seed MDCK Bulk/Edge commands used for the reported radial message-passing comparison.
- `scripts/validate_lachance_tables.py` - input-table schema and duplicate/missing-value validation.
- `scripts/run_lachance_architecture_study.py` - LaChance data loading, movie split, self-flow/proposal backbone and shared training utilities.
- `scripts/run_lachance_nextgen_message_passing.py` - route/state-aware graph/message-passing variants.
- `scripts/run_lachance_candidate_oracle.py` - causal candidate generation and oracle coverage diagnostics.
- `scripts/run_lachance_oracle_signal_sweep.py` - fast candidate-energy scorer/aggregator sweep.
- `scripts/run_lachance_transition_critic_v2.py` - offline learned transition critic with controls.
- `scripts/run_oz_full_architecture_study.py` and helper scripts - legacy/shared utilities used by the LaChance runners.
- `src/` - compact earlier baseline/backbone utilities kept for reference.

## Data

Large raw tables are not included. Pass the local LaChance table directory explicitly:

```bash
TABLE_ROOT=/path/to/lachance_epithelia/tables
```

Expected structure:

```text
$TABLE_ROOT/
  MDCK_Bulk/*.csv
  MDCK_Edge/*.csv
  MDAMB231/*.csv
  HUVEC/*.csv
```

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exact environment, input schema and defense-run commands are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Quick Smoke Runs

Candidate oracle gate:

```bash
python scripts/run_lachance_candidate_oracle.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/candidate_oracle_smoke \
  --cell-types MDCK_Edge \
  --horizon 6 \
  --seeds 42 \
  --max-movies 8 \
  --sobol-count 8 \
  --gaussian-count 8 \
  --train-reranker \
  --reranker-train-nodes 5000 \
  --reranker-val-nodes 2000
```

Oracle-signal sweep:

```bash
python scripts/run_lachance_oracle_signal_sweep.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/oracle_signal_sweep_smoke \
  --cell-types MDCK_Edge \
  --horizons 6 \
  --seeds 42 \
  --methods ridge_error,ridge_error_soft_blend \
  --feature-sets full,no_physics,dynamic_only
```

Transition critic v2:

```bash
python scripts/run_lachance_transition_critic_v2.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/critic_v2_smoke \
  --cell-types MDCK_Bulk,MDCK_Edge \
  --horizons 6,4 \
  --seeds 42 \
  --critic-v2-model mlp \
  --critic-v2-feature-set full,no_physics,dynamic_only,oz_only,shuffled_state,time_shuffled
```

## Main Experimental Protocol

Primary datasets:

- `MDCK_Bulk`
- `MDCK_Edge`

Guard datasets:

- `MDAMB231`
- `HUVEC`

Recommended settings:

```bash
python scripts/run_lachance_oracle_signal_sweep.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/oracle_signal_sweep_mdck_h4h6 \
  --cell-types MDCK_Bulk,MDCK_Edge \
  --horizons 6,4 \
  --seeds 7,42,123
```

For physics/critic controls:

```bash
python scripts/run_lachance_transition_critic_v2.py \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/critic_v2_mdck_guard \
  --cell-types MDCK_Bulk,MDCK_Edge,MDAMB231,HUVEC \
  --horizons 6,4 \
  --seeds 7,42,123
```

## Methodological Notes

- The primary split is by movie, not by frame.
- True future is used only in training losses and evaluation/oracle diagnostics.
- Candidate oracle is not a deployable model; it measures whether good future candidates exist.
- Static OZ/Henderson-style `c(r)` is included as a weak structural channel/control.
- The reportable architecture is the deployable path: backbone proposal + candidate cloud + calibrated candidate-energy selector.

## Defense Positioning

RSCF is positioned as a reproducible and interpretable forecasting pipeline for collective-cell displacement prediction:

> The method separates neighbour encoding, candidate generation and causal candidate selection. On MDCK Bulk/Edge it improves over the backbone proposal, exposes a large oracle ceiling, and provides ablation/control hooks for further scaling toward stronger selector and context-aware models.

## Team

- Dmitry Stanislavchuk-Abovsky - research question, implementation, experiments, diagnostics and artifact preparation.
- Andrey P. Zakharov - scientific supervision, methodology and expert feedback.
