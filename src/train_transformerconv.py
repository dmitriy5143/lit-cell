#!/usr/bin/env python3
"""TransformerConv baseline for the current LaChance displacement protocol.

This is a deliberately plain neural graph baseline: it reuses the same
TemporalSelfEncoder and CoarseFlowEncoder as the current best radial/crowding
branch, but replaces the structured equivariant social decoder with generic
PyG TransformerConv layers.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import data_protocol as la  # noqa: E402

arch = la.arch
DEFAULT_OUT = ROOT / "outputs" / "lachance_transformerconv_baseline"
CELL_TYPES = la.CELL_TYPES


def finite_json(value: Any) -> Any:
    return la.finite_json(value)


class TransformerConvSocialDecoder(nn.Module):
    """Generic attention message passing used as a neural control."""

    def __init__(
        self,
        *,
        self_dim: int = 48,
        flow_dim: int = 24,
        hidden_dim: int = 72,
        heads: int = 4,
        layers: int = 2,
        edge_dim: int = 12,
        dropout: float = 0.05,
        max_delta_norm: float = 1.35,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.max_delta_norm = float(max_delta_norm)
        self.dropout = nn.Dropout(float(dropout))
        node_extra = 2 + 2 + 3
        node_in = self_dim + flow_dim + node_extra
        # edge_features + radial + rel_velocity + shear + closing + quality pair
        self.edge_dim_total = int(edge_dim) + 2 + 2 + 2 + 1 + 2
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.convs = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=self.edge_dim_total,
                    dropout=float(dropout),
                    concat=True,
                    root_weight=True,
                    beta=True,
                )
                for _ in range(int(layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(int(layers))])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(int(layers))
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(int(layers))])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        last = self.head[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)
        last.bias.data[0] = -0.35
        last.bias.data[1] = math.log(2.0)

    def _edge_attr(self, graph: arch.GraphTensors) -> torch.Tensor:
        return torch.cat(
            [
                graph.edge_features,
                graph.radial,
                graph.rel_velocity,
                graph.shear,
                graph.closing,
                graph.quality[graph.src],
                graph.quality[graph.dst],
            ],
            dim=1,
        )

    def forward(
        self,
        graph: arch.GraphTensors,
        self_state: torch.Tensor,
        flow_state: torch.Tensor,
        self_pred: torch.Tensor,
        flow_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        log_degree = torch.log1p(graph.degree) / math.log(10.0)
        x = self.node_proj(
            torch.cat(
                [
                    self_state,
                    flow_state,
                    self_pred,
                    flow_pred,
                    graph.quality,
                    graph.speed_norm,
                    log_degree,
                ],
                dim=1,
            )
        )
        edge_index = torch.stack([graph.src, graph.dst], dim=0)
        edge_attr = self._edge_attr(graph)
        for conv, norm, ffn, ffn_norm in zip(self.convs, self.norms, self.ffns, self.ffn_norms):
            h = conv(x, edge_index, edge_attr)
            x = norm(x + self.dropout(F.silu(h)))
            x = ffn_norm(x + self.dropout(ffn(x)))
        out = self.head(torch.cat([x, graph.quality, graph.speed_norm, log_degree], dim=1))
        node_gate = torch.sigmoid(out[:, 0:1])
        flow_gate = 1.5 * torch.sigmoid(out[:, 1:2])
        delta = self.max_delta_norm * node_gate * torch.tanh(out[:, 2:4])
        return delta, flow_gate, {"node_gate": node_gate, "flow_gate": flow_gate}


@torch.no_grad()
def encode_all(
    temporal: arch.TemporalSelfEncoder,
    flow: arch.CoarseFlowEncoder,
    graph: arch.GraphTensors,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return arch.encode_base(temporal, flow, graph)


def train_transformerconv(
    train: arch.GraphTensors,
    val: arch.GraphTensors,
    train_enc: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_enc: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    seed: int,
    epochs: int,
    layers: int,
    hidden_dim: int,
    heads: int,
    lr: float,
    social_l2: float,
    flow_gate_l2: float,
    sequence_balanced_loss: bool,
) -> tuple[TransformerConvSocialDecoder, dict[str, float]]:
    arch.set_seed(seed + 70_000)
    model = TransformerConvSocialDecoder(
        edge_dim=train.edge_features.shape[1],
        hidden_dim=hidden_dim,
        heads=heads,
        layers=layers,
    ).to(train.history.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=3e-4)
    best, best_val, best_epoch = copy.deepcopy(model.state_dict()), float("inf"), 0
    train_self, train_state, train_flow, train_flow_state = train_enc
    val_self, val_state, val_flow, val_flow_state = val_enc
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        delta, flow_gate, diag = model(
            train,
            train_state.detach(),
            train_flow_state.detach(),
            train_self.detach(),
            train_flow.detach(),
        )
        pred = train_self.detach() + flow_gate * train_flow.detach() + delta
        loss = arch.masked_vector_mse(
            pred,
            train.y_norm,
            train.target_valid,
            arch.sequence_groups(train) if sequence_balanced_loss else None,
        )
        active_delta = delta[train.target_valid]
        active_flow_gate = flow_gate[train.target_valid]
        loss = (
            loss
            + float(social_l2) * torch.mean(torch.sum(active_delta.square(), dim=1))
            + float(flow_gate_l2) * torch.mean((active_flow_gate - 1.0).square())
            + 2e-5 * diag["node_gate"].mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            delta, flow_gate, _ = model(
                val,
                val_state,
                val_flow_state,
                val_self,
                val_flow,
            )
            pred = val_self + flow_gate * val_flow + delta
            score = float(
                arch.masked_vector_mse(
                    pred,
                    val.y_norm,
                    val.target_valid,
                    arch.sequence_groups(val) if sequence_balanced_loss else None,
                )
            )
        if score < best_val - 1e-6:
            best, best_val, best_epoch = copy.deepcopy(model.state_dict()), score, epoch + 1
        elif epoch + 1 - best_epoch >= 22:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_norm_mse": best_val}


@torch.no_grad()
def evaluate_transformerconv(
    model: TransformerConvSocialDecoder,
    graph: arch.GraphTensors,
    enc: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    norm: arch.Normalizer,
) -> tuple[np.ndarray, dict[str, float]]:
    self_pred, self_state, flow_pred, flow_state = enc
    delta, flow_gate, diag = model(graph, self_state, flow_state, self_pred, flow_pred)
    base = self_pred + flow_gate * flow_pred
    pred = base + delta
    pred_px = arch.to_px(pred, norm)
    mask = graph.target_valid.detach().cpu().numpy()
    y_px = graph.y_px.detach().cpu().numpy()
    metrics = arch.vector_metrics(y_px[mask], pred_px[mask], 1)
    delta_px = delta.detach().cpu().numpy() * norm.target_std
    residual_px = y_px - arch.to_px(base, norm)
    finite = mask & np.isfinite(residual_px).all(axis=1)
    dot = np.sum(delta_px[finite] * residual_px[finite], axis=1)
    denom = np.maximum(
        np.linalg.norm(delta_px[finite], axis=1) * np.linalg.norm(residual_px[finite], axis=1),
        1e-8,
    )
    metrics.update(
        {
            "social_magnitude_mean_px": float(np.mean(np.linalg.norm(delta_px[mask], axis=1))),
            "social_magnitude_p90_px": float(np.quantile(np.linalg.norm(delta_px[mask], axis=1), 0.9)),
            "social_residual_cosine": float(np.mean(dot / denom)),
            "node_gate_mean": float(diag["node_gate"][graph.target_valid].mean().cpu()),
            "flow_gate_mean": float(flow_gate[graph.target_valid].mean().cpu()),
            "flow_gate_std": float(flow_gate[graph.target_valid].std().cpu()),
        }
    )
    return pred_px, metrics


def run_cell_type(cell_type: str, args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, meta = la.load_lachance_dataset(
        cell_type,
        table_root=args.table_root,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        max_movies=args.max_movies,
        max_tracks_per_movie=args.max_tracks_per_movie,
        frame_stride=args.frame_stride,
        smooth_window=args.smooth_window,
        crop_fraction=args.crop_fraction,
        r_cut_px=args.r_cut_px,
    )
    graphs, norm, coverage = la.prepare_dataset(
        cell_type, raw, meta, horizon=args.horizon, k=args.k, device=device
    )
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"[{cell_type}] seed={seed} temporal", flush=True)
        temporal, temporal_info = arch.train_temporal(
            graphs["train"],
            graphs["val"],
            seed=seed,
            epochs=args.temporal_epochs,
            batch_size=args.batch_size,
            sequence_balanced_loss=args.sequence_balanced_loss,
        )
        zero_flow = arch.CoarseFlowEncoder(graphs["train"].flow.shape[1]).to(device)
        for parameter in zero_flow.parameters():
            torch.nn.init.zeros_(parameter)
        self_pred_px, self_metrics = arch.evaluate(
            temporal, zero_flow, None, graphs["test"], norm, variant="self_only"
        )
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "self_only",
                **self_metrics,
                **arch.sequence_metric_fields(graphs["test"], self_pred_px),
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        with torch.no_grad():
            train_self, _ = temporal(graphs["train"].history)
            val_self, _ = temporal(graphs["val"].history)
        print(f"[{cell_type}] seed={seed} flow", flush=True)
        flow, flow_info = arch.train_flow(
            graphs["train"],
            graphs["val"],
            train_self.detach(),
            val_self.detach(),
            seed=seed,
            epochs=args.flow_epochs,
            batch_size=args.batch_size,
            sequence_balanced_loss=args.sequence_balanced_loss,
        )
        flow_pred_px, flow_metrics = arch.evaluate(
            temporal, flow, None, graphs["test"], norm, variant="self_flow"
        )
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "self_flow",
                **flow_metrics,
                **arch.sequence_metric_fields(graphs["test"], flow_pred_px),
                **{f"flow_{k0}": v for k0, v in flow_info.items()},
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        encoded = {split: encode_all(temporal, flow, graph) for split, graph in graphs.items()}
        print(f"[{cell_type}] seed={seed} transformerconv", flush=True)
        model, info = train_transformerconv(
            graphs["train"],
            graphs["val"],
            encoded["train"],
            encoded["val"],
            seed=seed,
            epochs=args.transformer_epochs,
            layers=args.layers,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            lr=args.lr,
            social_l2=args.social_l2,
            flow_gate_l2=args.flow_gate_l2,
            sequence_balanced_loss=args.sequence_balanced_loss,
        )
        base_val = float(
            arch.masked_vector_mse(
                encoded["val"][0] + encoded["val"][2],
                graphs["val"].y_norm,
                graphs["val"].target_valid,
                arch.sequence_groups(graphs["val"]) if args.sequence_balanced_loss else None,
            )
        )
        val_gain = arch.relative_gain(base_val, float(info["best_val_norm_mse"]))
        pred_px, metrics = evaluate_transformerconv(model, graphs["test"], encoded["test"], norm)
        rows.append(
            {
                "dataset": cell_type,
                "seed": seed,
                "variant": "transformerconv_social",
                "stage_val_gain_pct": val_gain,
                **metrics,
                **arch.sequence_metric_fields(graphs["test"], pred_px),
                **arch.paired_block_bootstrap(
                    graphs["test"], flow_pred_px, pred_px, seed=seed + 970_001
                ),
                **{f"transformer_{k0}": v for k0, v in info.items()},
                **{f"flow_{k0}": v for k0, v in flow_info.items()},
                **{f"temporal_{k0}": v for k0, v in temporal_info.items()},
            }
        )
        print(
            f"[{cell_type}] seed={seed} transformerconv: "
            f"rmse={metrics['rmse_px']:.5f}px val_gain={val_gain:.2f}%",
            flush=True,
        )
    return pd.DataFrame(rows), coverage


def summarize(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, part in summary.groupby("dataset"):
        pivot = part.pivot_table(index="seed", columns="variant", values="rmse_px")
        base = pivot["self_flow"] if "self_flow" in pivot else pivot["self_only"]
        for variant in pivot.columns:
            gains = (base - pivot[variant]) / base * 100.0
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seeds": int(pivot[variant].notna().sum()),
                    "rmse_px_mean": float(pivot[variant].mean()),
                    "rmse_px_std": float(pivot[variant].std(ddof=0)),
                    "gain_vs_self_flow_pct_mean": float(gains.mean()),
                    "gain_vs_self_flow_pct_min": float(gains.min()),
                    "positive_seed_fraction": float((gains > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_gain(aggregate: pd.DataFrame, out_dir: Path) -> None:
    if aggregate.empty:
        return
    plot_df = aggregate[aggregate["variant"].ne("self_only")].copy()
    colors = {
        "self_flow": "#42D9F5",
        "transformerconv_social": "#C084FC",
    }
    fig, ax = plt.subplots(figsize=(9.6, 4.6), constrained_layout=True)
    labels = plot_df["dataset"] + "\n" + plot_df["variant"]
    ax.bar(
        np.arange(len(plot_df)),
        plot_df["gain_vs_self_flow_pct_mean"],
        color=[colors.get(v, "#AAB7CC") for v in plot_df["variant"]],
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(np.arange(len(plot_df)), labels, rotation=20, ha="right")
    ax.set_ylabel("Gain over self + flow (%)")
    ax.set_title("TransformerConv social branch baseline")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "fig_transformerconv_gain.png", dpi=260)
    plt.close(fig)


def write_report(summary: pd.DataFrame, aggregate: pd.DataFrame, coverage: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# LaChance TransformerConv Baseline",
        "",
        "This is a neural ablation/control: the temporal and coarse-flow encoders are kept fixed, while the structured radial/crowding social decoder is replaced with generic PyG TransformerConv message passing.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(finite_json(coverage), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Aggregate Test Metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Mean Diagnostics",
        "",
    ]
    diag_cols = [
        "dataset",
        "variant",
        "rmse_px",
        "r2_vec",
        "social_magnitude_mean_px",
        "social_residual_cosine",
        "node_gate_mean",
        "flow_gate_mean",
    ]
    available = [col for col in diag_cols if col in summary.columns]
    means = summary[available].groupby(["dataset", "variant"], as_index=False).mean(numeric_only=True)
    lines.append(means.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "TransformerConv is a useful neural baseline because it checks whether a generic attention graph layer can replace the physically constrained radial/crowding message basis.",
        ]
    )
    (out_dir / "transformerconv_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-types", nargs="+", choices=CELL_TYPES, default=["MDCK_Edge"])
    parser.add_argument("--table-root", type=Path, default=la.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-mode", choices=["movie", "frame"], default="movie")
    parser.add_argument("--split-seed", type=int, default=20260608)
    parser.add_argument("--max-movies", type=int, default=8)
    parser.add_argument("--max-tracks-per-movie", type=int, default=0)
    parser.add_argument("--crop-fraction", type=float, default=0.08)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--r-cut-px", type=float, default=50.0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temporal-epochs", type=int, default=35)
    parser.add_argument("--flow-epochs", type=int, default=25)
    parser.add_argument("--transformer-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sequence-balanced-loss", action="store_true")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=72)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--social-l2", type=float, default=1e-4)
    parser.add_argument("--flow-gate-l2", type=float, default=2e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.cell_types = args.cell_types[:1]
        args.max_movies = min(args.max_movies, 4)
        args.seeds = args.seeds[:1]
        args.temporal_epochs = min(args.temporal_epochs, 3)
        args.flow_epochs = min(args.flow_epochs, 3)
        args.transformer_epochs = min(args.transformer_epochs, 3)
        args.hidden_dim = min(args.hidden_dim, 40)
        args.heads = min(args.heads, 4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = arch.select_device(args.device)
    print(f"device={device}", flush=True)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(finite_json(vars(args)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    all_rows: list[pd.DataFrame] = []
    all_coverage: dict[str, Any] = {}
    for cell_type in args.cell_types:
        rows, coverage = run_cell_type(cell_type, args, device)
        rows.to_csv(args.out_dir / f"transformerconv_summary_{cell_type}.csv", index=False)
        all_rows.append(rows)
        all_coverage[cell_type] = coverage
    summary = pd.concat(all_rows, ignore_index=True)
    aggregate = summarize(summary)
    summary.to_csv(args.out_dir / "transformerconv_summary.csv", index=False)
    aggregate.to_csv(args.out_dir / "transformerconv_aggregate.csv", index=False)
    (args.out_dir / "coverage.json").write_text(
        json.dumps(finite_json(all_coverage), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_gain(aggregate, args.out_dir)
    write_report(summary, aggregate, all_coverage, args.out_dir)
    print(aggregate.to_string(index=False), flush=True)
    print(args.out_dir / "transformerconv_baseline_report.md", flush=True)


if __name__ == "__main__":
    main()
