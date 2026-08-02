# C2C12 external LIT-Cell evidence for PRX development

## Decision

**The external structural gate passes; the result is not an absolute C2C12 leaderboard win.**

The primary automatic-track analysis uses three experiment-level train/validation/test rotations, a predeclared horizon-balanced operating point, and an E(2)-equivariant vector operator. Manual tracks are reported separately because interpolation dominates their observation process.

## Primary automatic tracks

- h1 component RMSE: 4.338749 -> 4.355146 px (+0.395% mean paired-field error change; 0.5% guard passed).
- rolling h6 component RMSE: 5.148432 -> 5.088459 px (1.221% mean paired-field gain).
- h6 gain is positive in 3/3 held-out experiments and all four density quartiles.
- Real completed innovations beat no-update, own-only, wrong-cell, and stale-time controls under hierarchical bootstrap.
- Future-donor, split-overlap, and target-feature violations are all zero.
- Constant velocity remains stronger in absolute h6 RMSE (4.860158 px); the claim is transfer of the innovation mechanism, not global dominance on noisy automatic tracks.

## Manual observation-process audit

- h1 component RMSE: 1.084036 -> 1.085811 px (+0.110% mean paired-field error change).
- rolling h6 component RMSE: 1.500590 -> 1.334696 px (11.396% mean paired-field gain).
- The h6 gain remains positive on observed-only windows, but is much larger on interpolated windows; automatic and manual estimates must not be pooled.

## Operator interpretation

- The E(2)-equivariant operator matches the free x/y regression at h6 (5.088459 vs 5.088667 px). Arbitrary coordinate-axis mixing is therefore unnecessary for this result.
- Own, global, and local coefficient signs are highly stable across the three rotations; the full operator remains significantly better than own-only.
- The largest normalized local coefficient norm occurs at `m2` (approximately twice the frame-wise nearest-neighbour spacing). This is a predictive support scale, not a force or universal correlation length.

## PRX positioning

This experiment closes a previous external-validation gap: delayed, identity-correct local innovation is reusable across a second tracking corpus, and a symmetry-constrained operator retains the effect. It strengthens the general active-system filtering claim and the observation-noise analysis.

It does not close the PRX mechanical-law gate. There is still no positive bridge to a simultaneously measured force, polarity, or intervention, and constant velocity remains the best automatic-track h6 baseline. The defensible statement is a transferable causal kinematic update, not a recovered mechanical law or global forecasting SOTA.

## Machine-readable tables

- `v209_prx_main_metrics.csv`
- `v209_prx_experiment_gains.csv`
- `v209_prx_density_strata.csv`
- `v209_prx_tracking_quality.csv`
- `v209_prx_operator_coefficients.csv`
- `v209_prx_operator_scale_norms.csv`
- `v209_prx_free_vs_equivariant.csv`
- `v209_prx_causal_audit.csv`
- `v209_prx_cluster_bootstrap.csv`
