#!/usr/bin/env bash
set -euo pipefail

: "${TABLE_ROOT:?Set TABLE_ROOT to the LaChance tables directory}"

COMMON=(
  --table-root "$TABLE_ROOT"
  --split-mode movie
  --split-seed 20260608
  --max-movies 8
  --crop-fraction 0.08
  --frame-stride 1
  --smooth-window 3
  --r-cut-px 50
  --horizon 6
  --seeds 7 42 123
  --variants mp_gated_radial
  --k 10
  --temporal-epochs 35
  --flow-epochs 25
  --mp-epochs 110
  --batch-size 2048
  --sequence-balanced-loss
  --layers 3
  --hidden-dim 72
  --edge-hidden-dim 56
  --max-delta-norm 1.2
  --lr 0.0008
  --social-l2 0.0002
  --flow-gate-l2 0.05
  --device auto
)

python scripts/run_lachance_nextgen_message_passing.py \
  "${COMMON[@]}" \
  --cell-types MDCK_Bulk \
  --out-dir outputs/defense_mdck_bulk_h6

python scripts/run_lachance_nextgen_message_passing.py \
  "${COMMON[@]}" \
  --cell-types MDCK_Edge \
  --out-dir outputs/defense_mdck_edge_h6
