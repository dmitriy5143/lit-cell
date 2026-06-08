# Interpretable Graph Model for Collective Cell Migration

This repository snapshot contains the code and result tables for the project defense:

**Regime-gated radial/crowding message passing for displacement forecasting in collective cell migration.**

The main scientific question is whether local neighbours carry an ablatable predictive signal beyond a cell's own trajectory and a coarse local tissue flow.

## Repository Structure

- `src/oz_core.py` - shared graph tensors, temporal encoder, flow encoder, metrics, normalization and legacy structural decoder utilities.
- `src/data_protocol.py` - LaChance table loading, causal sample construction, movie split and dataset preparation.
- `src/train_radial_mp.py` - current best radial/crowding message-passing branch.
- `src/train_transformerconv.py` - generic PyG TransformerConv social-branch baseline.
- `src/run_baselines.py` - classical baselines: zero displacement, constant velocity, Ridge and MLP.
- `results/` - compact CSV tables used in the project artifacts.
- `figures/` - key defense figures.

Large raw trajectory tables are not included in this snapshot. Use `--table-root` to point scripts to the local LaChance tables.

## Main Results

Held-out movie protocol, horizon 6 frames:

| Dataset | Self+flow RMSE | TransformerConv social RMSE | 3-layer radial MP RMSE |
|---|---:|---:|---:|
| MDCK Edge | 20.8864 | 21.5730 | 19.9183 |
| MDCK Bulk | 21.6278 | 20.3192 | 20.1205 |

Interpretation:

- `self + flow` is the strong neural baseline.
- `TransformerConv social` tests whether a generic graph attention layer can replace the structured decoder.
- `3-layer radial MP` is the final model: constrained radial/crowding message passing.

## Reproducing Key Runs

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the local data path:

```bash
TABLE_ROOT=/path/to/lachance_epithelia/tables
```

Run the current best model:

```bash
python src/train_radial_mp.py \
  --cell-types MDCK_Edge MDCK_Bulk \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/radial_mp_mdck \
  --split-mode movie \
  --split-seed 20260608 \
  --max-movies 8 \
  --crop-fraction 0.08 \
  --r-cut-px 50 \
  --horizon 6 \
  --seeds 7 42 123 \
  --k 10 \
  --temporal-epochs 35 \
  --flow-epochs 25 \
  --mp-epochs 110 \
  --sequence-balanced-loss \
  --layers 3 \
  --hidden-dim 72 \
  --edge-hidden-dim 56 \
  --max-delta-norm 1.2 \
  --lr 8e-4 \
  --social-l2 2e-4 \
  --flow-gate-l2 0.05
```

Run TransformerConv social baseline:

```bash
python src/train_transformerconv.py \
  --cell-types MDCK_Edge MDCK_Bulk \
  --table-root "$TABLE_ROOT" \
  --out-dir outputs/transformerconv_mdck \
  --split-mode movie \
  --split-seed 20260608 \
  --max-movies 8 \
  --crop-fraction 0.08 \
  --r-cut-px 50 \
  --horizon 6 \
  --seeds 7 42 123 \
  --k 10 \
  --temporal-epochs 35 \
  --flow-epochs 25 \
  --transformer-epochs 110 \
  --sequence-balanced-loss \
  --layers 3 \
  --hidden-dim 72 \
  --heads 4 \
  --lr 8e-4 \
  --social-l2 2e-4 \
  --flow-gate-l2 0.05
```

Run classical baselines:

```bash
python src/run_baselines.py
```

## Defense Claim

The project does not claim full SOTA over all published LaChance protocols. The supported claim is:

> On held-out MDCK displacement forecasting, a constrained radial/crowding graph decoder improves a strong self+flow neural baseline and is more stable/interpretable than a generic TransformerConv social branch.

## Team

- Dmitry Stanislavchuk-Abovsky - research question, implementation, experiments, diagnostics and artifact preparation.
- Andrey P. Zakharov - scientific supervision, methodology and feedback.

