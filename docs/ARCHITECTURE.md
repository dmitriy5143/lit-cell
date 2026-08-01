# LIT-Cell Architecture

## Forecasting Contract

For cell `i`, let `x[i,t]` be the observed centroid and
`d[i,t] = x[i,t] - x[i,t-1]` the completed displacement. At issue time `t`, the
model estimates

```text
p(d[i,t+1] | observations available no later than t).
```

The prediction is stored before `x[i,t+1]` is revealed. Only after the next
frame arrives is the innovation

```text
e[i,t+1] = d[i,t+1] - d_hat[i,t+1|t]
```

allowed to update the cell state and the neighbourhood field used at the next
issue time. Cumulative h2/h4/h6 values are sums of consecutively issued h1
predictions with an observation update between predictions.

## 1. Coordinate and Velocity Anchor

The anchor uses six previous positions, instantaneous and multiscale velocity,
and causal tabular context. Training residual trajectories are grouped into 12
route classes. A ridge expert predicts six residual steps for each class; a
causal router estimates class probabilities; the eight most probable experts
are mixed and calibrated by a fold-local linear model.

This bank is a stable coordinate reference, not an oracle candidate selector.
Target-derived route labels are used only on training movies.

## 2. Causal Innovation State Filter

A recurrent state of width 128 consumes the coordinate anchor, six completed
past steps, the previous innovation, and an availability mask. It predicts a
bounded next-step mean together with Student-t scale and degrees of freedom.
Separate process and observation scale heads form a learned Kalman-like gain,
but are interpreted functionally: coordinates alone do not identify physical
process noise and tracking noise separately.

Training uses chronological track fragments, one-step Huber and Student-t NLL,
cumulative online losses, 15% observation dropout, and 0.35 px coordinate noise.

## 3. Local Transport of Completed Innovations

The filter residual is mapped to a normal score so that heavy-tailed marginal
scale does not dominate neighbourhood aggregation. For spatial scales
`30, 60, 120, 240 px`, the model averages completed neighbour scores with a
Gaussian distance kernel. The same track is excluded. Every donor transition
must be complete before the receiver prediction is issued.

A regularized linear operator maps own and neighbour summaries to a correction.
Its norm is bounded, and its strength and bound are selected on an inner movie.
The correction shifts the conditional mean but does not overwrite uncertainty.

The dense implementation is the reference. The sparse implementation uses a
radius search with complexity `O(N log N + E)` and is required to remain within
the registered non-inferiority tolerance of the dense result.

## 4. Probability and Calibration

The final one-step distribution factorizes into two Student-t coordinates. A
movie-external scale factor and radial conformal quantiles provide diagnostic
50% and 90% coverage. The cumulative h6 distribution is empirically calibrated
on sequential errors; it is not presented as an exact convolution of dependent
Student-t variables.

## 5. Equivariant Field Surrogate

For interpretation, the free local operator is projected onto a compact
E(2)-equivariant vector library: self and neighbour residuals, longitudinal and
transverse components, Laplacian, gradient-divergence, cubic saturation, and
advection. Shared scalar coefficients act on both coordinate axes. Rotation,
reflection, wrong-cell, and stale-time controls test the representation.

The potential sector supports an effective dissipative functional plus active
transport. This is a mathematical representation of the innovation field, not
an identification of traction, stress, or thermodynamic energy.

## What the Model Does Not Contain

- future frames or future neighbour transitions at inference;
- a static Ornstein-Zernike `c(r)` prior in the final predictor;
- a scalar candidate reranker;
- raw video, segmentation, or measured mechanics in the final conditional mean;
- a claim that h6 is an open-loop six-step forecast.
