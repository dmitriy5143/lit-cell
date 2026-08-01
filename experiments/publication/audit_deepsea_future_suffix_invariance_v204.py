#!/usr/bin/env python3
"""Verify that future state mutations cannot change earlier forecasts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_lachance_causal_innovation_state_space_v97 as v97  # noqa: E402
import run_lachance_foldlocal_semigroup_confirmation_v157e as v157e  # noqa: E402


KEYS = ["sequence", "frame", "track_id"]


def mutate_test_suffix(
    features: pd.DataFrame,
    test_rows: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = features.copy()
    numeric = [
        column
        for column in features.columns
        if (column.startswith("meta_") or column.startswith("ms_"))
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    state = [column for column in numeric if column.startswith("ms_")]
    if not state:
        raise RuntimeError("No state features available for suffix mutation")
    test_keys = test_rows[KEYS + ["family", "video"]].drop_duplicates(KEYS)
    merged = output[KEYS].merge(
        test_keys,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    audit: list[dict[str, object]] = []
    for sequence, raw_indices in merged.dropna(subset=["video"]).groupby(
        "sequence",
        sort=True,
    ).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int64)
        frames = output.loc[indices, "frame"].to_numpy(np.int64)
        cutoff = int(np.median(np.unique(frames)))
        suffix = indices[frames > cutoff]
        prefix = indices[frames <= cutoff]
        if len(suffix) > 1:
            permutation = rng.permutation(suffix)
            output.loc[suffix, state] = (
                features.loc[permutation, state].to_numpy()
            )
        row = merged.loc[indices].iloc[0]
        audit.append(
            {
                "sequence": int(sequence),
                "family": str(row.family),
                "video": str(row.video),
                "cutoff_frame": cutoff,
                "prefix_rows": len(prefix),
                "mutated_suffix_rows": len(suffix),
            }
        )
    return output, pd.DataFrame(audit)


def build_model(
    checkpoint: dict,
    prep: object,
    variant: v97.TrainVariant,
    device: torch.device,
) -> v97.CausalInnovationStateSpaceForecaster:
    args = v157e.checkpoint_namespace(checkpoint, str(device))
    metadata = checkpoint["metadata"]
    model = v97.CausalInnovationStateSpaceForecaster(
        static_dim=int(prep.static[0].shape[1]),
        hidden=int(args.hidden),
        history_lags=int(args.history_lags),
        correction_bound=float(args.correction_bound),
        dropout=float(args.dropout),
        use_update=bool(variant.use_update),
        use_graph=bool(variant.use_graph),
        graph_heads=int(args.graph_heads),
        output_mode=str(variant.output_mode),
        target_mean=np.asarray(metadata["target_mean"], dtype=np.float32),
        target_scale=np.asarray(metadata["target_scale"], dtype=np.float32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    variant = v97.TrainVariant(**checkpoint["variant"])
    if not variant.use_context:
        raise RuntimeError("Suffix invariance needs a context-enabled checkpoint")
    namespace = v157e.checkpoint_namespace(checkpoint, "cpu")
    namespace.anchor_cache = args.cache_dir.resolve()
    namespace.features = args.features.resolve()
    original = v97.load_prepared(namespace, variant)
    test_rows = original.bundles[2].rows.reset_index(drop=True)
    feature_table = pd.read_csv(args.features, low_memory=False)
    mutated_table, movie_audit = mutate_test_suffix(
        feature_table,
        test_rows,
        args.seed,
    )
    with tempfile.TemporaryDirectory(prefix="deepsea_v204_suffix_") as directory:
        mutated_path = Path(directory) / "features.csv"
        mutated_table.to_csv(mutated_path, index=False)
        namespace.features = mutated_path
        mutated = v97.load_prepared(namespace, variant)

    device = torch.device("cpu")
    model = build_model(checkpoint, original, variant, device)
    eta = float(checkpoint["metadata"]["eta"])
    original_result = v97.replay_inference(
        model,
        original,
        2,
        device,
        eta=eta,
        seed=args.seed,
    )
    mutated_result = v97.replay_inference(
        model,
        mutated,
        2,
        device,
        eta=eta,
        seed=args.seed,
    )
    cutoffs = movie_audit.set_index("sequence").cutoff_frame.to_dict()
    prefix_mask = np.asarray(
        [
            int(row.frame) <= int(cutoffs[int(row.sequence)])
            for row in test_rows.itertuples(index=False)
        ],
        dtype=bool,
    )
    suffix_mask = ~prefix_mask
    prediction_delta = np.abs(
        original_result.prediction - mutated_result.prediction
    )
    static_delta = np.abs(original.static[2] - mutated.static[2])
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "features": str(args.features.resolve()),
        "test_rows": len(test_rows),
        "prefix_rows": int(prefix_mask.sum()),
        "suffix_rows": int(suffix_mask.sum()),
        "prefix_static_max_abs_delta": float(static_delta[prefix_mask].max()),
        "prefix_prediction_max_abs_delta": float(
            prediction_delta[prefix_mask].max()
        ),
        "suffix_static_mean_abs_delta": float(
            static_delta[suffix_mask].mean()
        ),
        "suffix_prediction_mean_abs_delta": float(
            prediction_delta[suffix_mask].mean()
        ),
        "future_suffix_invariance_pass": bool(
            static_delta[prefix_mask].max() <= args.tolerance
            and prediction_delta[prefix_mask].max() <= args.tolerance
        ),
        "tolerance": args.tolerance,
    }
    movie_audit.to_csv(
        args.out_dir / "v204_future_suffix_movie_audit.csv",
        index=False,
    )
    (args.out_dir / "v204_future_suffix_invariance.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
