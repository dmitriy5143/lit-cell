# Comparator Scope

## Quantitative Claim

The final ranking uses only methods that obey the same event order as LIT-Cell:

```text
issue the t -> t+1 prediction
observe frame t+1
update state from the completed transition
issue the next prediction
```

The primary six-movie table contains exact local baselines or causal
adaptations whose inputs, outer movie, seed aggregation, and h1/h2/h4/h6
definitions are recorded in `evidence/v188/v188_primary_online_benchmark.csv`.
The strongest learned-method check on movies 10-16 is frozen in
`evidence/comparators/online_confirmation_learned.csv`, with the row-key
contract and model-selection metadata in the same directory.

## What Is Not Pooled Into That Ranking

The project also evaluated LSTM, temporal Transformer, TransformerConv,
Social-LSTM, AgentFormer-style, MTR-style, and QCNet-style causal adapters. The
exact runner is public, but these experiments belong to the architecture
screen rather than the frozen comparison on movies 10-16. They are therefore
reported as tested model families, not silently promoted to confirmatory
baselines.

Earlier Ridge/MLP/GAT/GraphSAGE/TransformerConv/radial-MP and multimodal route
models used a fixed-origin open-loop target. Their compact outcome table is
retained in `evidence/comparators/fixed_origin_architecture_screen.csv`. Those
numbers answer whether an architecture helped that earlier formulation; they
cannot be compared numerically with streaming h6, which receives a new
observation before every next prediction.

Published AgentFormer, MTR, QCNet/QCNeXt, and EigenTrajectory results use other
datasets, scene semantics, split units, and ADE/FDE-style contracts. The paper
uses them as architectural context only. Neither their published numbers nor
our cell-domain adapters support a global state-of-the-art claim.

The machine-readable classification is
`evidence/comparators/comparator_protocol_matrix.csv`.
