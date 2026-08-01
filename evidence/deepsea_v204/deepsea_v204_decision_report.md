# DeepSea External Multimodal Validation v204

## Decision

The complete LaChance publication system is the two-operating-point v166
bounded completed-innovation transport. The `v97_direct` model is only its
frozen sequential prior and is reported as an internal ablation, not as the
project SOTA.

The contract-correct, dimensionless external experiment did not pass the
pre-registered transport or privileged-state gates. The raw-image student and
bounded multimodal adapter were therefore not trained; this is a gate-enforced
stop, not missing implementation.

The earlier native-pixel pilot is retained only as a secondary audit. It was
superseded because the frozen amendment requires learning and evaluation in
first-frame cell-diameter units across heterogeneous acquisition scales.

## Data and protocol

- Dataset: DeepSea, 47 independent videos and
  3 cell families.
- Observations: 74,953; tracks: 3,010.
- Frozen split: 26 train, 10 validation, 11 outer-test movies.
- Primary endpoint: movie-macro cumulative rolling h6 component RMSE.
- Unit: first-frame median segmented-cell diameter.
- Test windows: 22,149 at h1 and 17,763 at h6.
- Future-suffix invariance: **passed**; prefix prediction delta was
  0.0.

## Prior and complete-system benchmark

The v97 prior reduced h1 RMSE relative to constant velocity
from 0.1940 to 0.1709
cell diameters (+11.95%). It did not retain that advantage
at h6: 0.2130 versus
0.2081 (-2.36%).

The externalized complete v166 h1-strict operating point reached h1
0.1553, a
19.98% gain over constant velocity, but degraded h6 to
0.2405. The v166 h6-utility
operating point reached h6
0.2004, improving over constant
velocity by 3.66% and over IMM by
3.44%. It degraded h1 by
8.28% relative to its own frozen
prior, so it did not pass the pre-registered joint h1/h6 gate.

The h1 result is family-dependent. Zero displacement is strongest in the
low-motion embryonic-stem movies, while learned and velocity models help the
bronchial and muscle families. Ridge/HGBDT and LSTM/GRU show the same general
trade-off: a low one-step error does not guarantee stable cumulative
transport. The complete v166 h6-utility point, rather than the v97 prior, is
the strongest tested DeepSea h6 method in this table.

## Completed-innovation transport details

The complete h1-strict v166 transport improved h1 over its own prior by
9.34% but degraded h6 by
-14.33%, with 0/11 positive test movies.
The frozen transport gate therefore failed.

The complete h6-utility v166 transport improved h6 over its own prior by
4.69% (95% movie bootstrap
0.29 to
8.43) but degraded h1 by
8.28%. Its h6 RMSE remained better
than wrong-cell (0.2262) and stale-time
(0.2029) controls. The own-only ablation produced nearly the
same h6 result, indicating that the local graph packet did not provide the
transferable benefit.

## Privileged mask/state observability

The exact causal full state packet changed h6 by
-1.04% relative to zero state (95%
movie bootstrap -2.16 to
-0.12). Its Student-t NLL reduction was
-0.0128; positive values would
be improvements. Real state did not beat row-shuffled, time-shuffled,
wrong-cell and wrong-video controls.

The shape-only exact branch also failed its controls. Fast packet triage found
no morphology, polarity, contact or reliability family that passed the
pre-registered gate after multiplicity control.

The noncausal future-state capacity control improved h6 by
34.66% and improved h1 proper score on all
11 movies. The model can consume informative auxiliary variables; the tested
causal masks do not contain the required cell-specific forecasting signal.

## Interpretation

This experiment separates three claims:

1. **Model capacity:** supported by the noncausal positive control.
2. **Complete v166 mechanism transfer:** each operating point retains its
   intended advantage, but no single operating point passes the joint h1/h6
   gate on DeepSea.
3. **Causal mask/morphology information:** not supported beyond hard controls.

The result does not justify another DeepSea image encoder, flat video token
fusion, or LaChance adaptation from this source representation. The remaining
architecture hypothesis is a separately frozen two-timescale belief model:
a transient h1 correction and a persistent own-innovation state with explicit
semigroup consistency. Because the DeepSea outer test has now been inspected,
that hypothesis requires a new confirmation split or dataset.

For new observability, the next data should be synchronized MDCK-like movies
with reliable identity-resolved masks and an independent mechanical channel
(traction/stress or equivalent). Reusing appearance-only masks from DeepSea
is not supported by this experiment.

## Claim boundary

DeepSea v204 is a strong negative external-observability and partial
mechanism-transfer result. It is not evidence of global SOTA or of successful
multimodal transfer. The validated LaChance publication bundle remains the
positive efficacy result; DeepSea defines where that mechanism currently
does and does not generalize.
