#!/usr/bin/env python3
"""Reproduce or verify the public LIT-Cell forecasting workflow.

The commands deliberately distinguish a fast architecture replay from the
publication-scale movie-level experiment. Raw microscopy preprocessing is not
silently substituted by committed frozen tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "experiments" / "publication"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def graph_edges(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [(source, target) for source in range(count) for target in range(count) if source != target]
    edge_index = torch.as_tensor(pairs, dtype=torch.long).T.contiguous()
    edge_attr = torch.zeros((len(pairs), 8), dtype=torch.float32)
    edge_attr[:, 0] = 1.0
    edge_attr[:, 2] = 0.5
    return edge_index, edge_attr


def synthetic_frame(frame: int, history: np.ndarray, anchor: np.ndarray):
    from lit_cell_forecasting import FrameBatch

    count = len(anchor)
    edge_index, edge_attr = graph_edges(count)
    static = np.column_stack(
        [
            np.linspace(0.2, 0.8, count),
            np.full(count, frame / 10.0),
            np.linalg.norm(anchor, axis=1),
            np.ones(count),
        ]
    ).astype(np.float32)
    return FrameBatch(
        dataset="synthetic_contract",
        movie=1,
        frame=frame,
        track_ids=np.arange(100, 100 + count, dtype=np.int64),
        static=torch.as_tensor(static),
        history=torch.as_tensor(history, dtype=torch.float32),
        anchor_normalized=torch.as_tensor(anchor, dtype=torch.float32),
        anchor_physical=torch.as_tensor(anchor, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def smoke(args: argparse.Namespace) -> None:
    from lit_cell_forecasting import (
        CausalInnovationStateSpaceForecaster,
        ObservationBatch,
        StreamingForecaster,
    )

    np.random.seed(17)
    torch.manual_seed(17)
    model = CausalInnovationStateSpaceForecaster(
        static_dim=4,
        hidden=16,
        history_lags=3,
        correction_bound=2.0,
        dropout=0.0,
        use_update=True,
        use_graph=True,
        graph_heads=4,
    )
    forecaster = StreamingForecaster(
        model,
        residual_scale=np.array([0.7, 0.9], dtype=np.float32),
    )
    observed_steps = np.asarray(
        [
            [[0.40, 0.05], [0.20, 0.12], [-0.15, 0.18]],
            [[0.45, 0.02], [0.18, 0.15], [-0.10, 0.20]],
            [[0.43, -0.03], [0.16, 0.18], [-0.05, 0.22]],
            [[0.38, -0.08], [0.12, 0.20], [0.01, 0.23]],
        ],
        dtype=np.float32,
    )
    history = np.zeros((3, 3, 5), dtype=np.float32)
    records: list[dict[str, object]] = []
    for issue_frame, observed in enumerate(observed_steps):
        anchor = history[:, 0, :2]
        prediction = forecaster.predict_before_observe(
            synthetic_frame(issue_frame, history, anchor)
        )
        records.append(
            {
                "issue_frame": issue_frame,
                "target_frame": prediction.target_frame,
                "measurement_rows": int(prediction.measurement_mask.sum()),
                "mean_scale": float(prediction.scale.mean()),
                "finite": bool(
                    np.isfinite(prediction.mean).all()
                    and np.isfinite(prediction.scale).all()
                ),
            }
        )
        forecaster.update_after_observe(
            ObservationBatch(
                dataset="synthetic_contract",
                movie=1,
                frame=issue_frame + 1,
                track_ids=np.arange(100, 103, dtype=np.int64),
                displacement=observed,
            )
        )
        token = np.column_stack(
            [
                observed,
                np.linalg.norm(observed, axis=1),
                np.zeros((3, 2), dtype=np.float32),
            ]
        )
        history = np.concatenate([token[:, None, :], history[:, :-1, :]], axis=1)

    if records[0]["measurement_rows"] != 0:
        raise RuntimeError("first forecast unexpectedly used a future innovation")
    if any(record["measurement_rows"] != 3 for record in records[1:]):
        raise RuntimeError("completed innovations were not exposed on the next step")
    if not all(record["finite"] for record in records):
        raise RuntimeError("non-finite synthetic replay output")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "lit_cell_smoke_checkpoint.pt"
    forecaster.save(checkpoint)
    restored = StreamingForecaster.load(checkpoint)
    forecaster.reset_tracks()
    initial_history = np.zeros((3, 3, 5), dtype=np.float32)
    initial_anchor = np.zeros((3, 2), dtype=np.float32)
    reference = forecaster.predict_before_observe(
        synthetic_frame(0, initial_history, initial_anchor)
    )
    replay = restored.predict_before_observe(
        synthetic_frame(0, initial_history, initial_anchor)
    )
    np.testing.assert_allclose(reference.mean, replay.mean, atol=1e-7, rtol=0.0)

    payload = {
        "status": "PASS",
        "scope": "architecture-level synthetic predict-before-observe replay",
        "frames": records,
        "checkpoint_roundtrip_max_abs_error": float(
            np.max(np.abs(reference.mean - replay.mean))
        ),
        "note": "This validates executable protocol wiring, not publication accuracy.",
    }
    output = args.out_dir / "lit_cell_smoke_report.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


def v102_command(args: argparse.Namespace, output: Path, *, dry_run: bool) -> list[str]:
    command = [
        sys.executable,
        str(PUBLICATION / "run_lachance_online_lomo_benchmark_v102.py"),
        "--table-root",
        str(args.table_root.resolve()),
        "--features",
        str(args.feature_grid.resolve()),
        "--out-dir",
        str(output.resolve()),
        "--runners",
        "v97",
        "--seeds",
        args.seeds,
        "--device",
        args.device,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def write_reproduction_contract(args: argparse.Namespace, commands: list[list[str]]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lit-cell-reproduction-v1",
        "table_root": str(args.table_root.resolve()),
        "feature_grid": str(args.feature_grid.resolve()),
        "seeds": args.seeds,
        "device": args.device,
        "commands": commands,
    }
    (args.out_dir / "reproduction_contract.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def preflight(args: argparse.Namespace) -> None:
    command = v102_command(args, args.out_dir, dry_run=True)
    write_reproduction_contract(args, [command])
    run(command)


def full(args: argparse.Namespace) -> None:
    outer = args.out_dir / "outer_lomo"
    transport = args.out_dir / "transport_pareto"
    outer_command = v102_command(args, outer, dry_run=False)
    transport_command = [
        sys.executable,
        str(PUBLICATION / "run_lachance_foldlocal_semigroup_pareto_v157h.py"),
        "--v102-root",
        str(outer.resolve()),
        "--v102-summary",
        str((outer / "v102_movie_level_summary.csv").resolve()),
        "--out-dir",
        str(transport.resolve()),
        "--seeds",
        args.seeds,
        "--device",
        args.device,
    ]
    write_reproduction_contract(args, [outer_command, transport_command])
    run(outer_command)
    run(transport_command)


def verify(_: argparse.Namespace) -> None:
    commands = [
        [sys.executable, "scripts/validate_publication_release.py"],
        [sys.executable, "-m", "compileall", "-q", "src", "experiments", "scripts"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    ]
    for command in commands:
        run(command)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--feature-grid", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run a synthetic end-to-end event-order and checkpoint replay.",
    )
    smoke_parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "reproduction_smoke",
    )
    smoke_parser.set_defaults(function=smoke)
    verify_parser = subparsers.add_parser(
        "verify",
        help="Validate frozen evidence, code compilation, and unit tests.",
    )
    verify_parser.set_defaults(function=verify)
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate full LOMO inputs and write the exact job manifest.",
    )
    add_data_arguments(preflight_parser)
    preflight_parser.set_defaults(function=preflight)
    full_parser = subparsers.add_parser(
        "full",
        help="Recompute outer LOMO anchors/filter and fold-local transport.",
    )
    add_data_arguments(full_parser)
    full_parser.set_defaults(function=full)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
