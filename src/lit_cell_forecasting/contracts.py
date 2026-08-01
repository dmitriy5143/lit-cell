"""Strict predict-before-observe contracts for streaming cell forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class ProtocolError(RuntimeError):
    """Raised when an event would violate the frozen causal protocol."""


@dataclass(frozen=True)
class FrameBatch:
    """All inference-time inputs available at one issue frame."""

    dataset: str
    movie: int
    frame: int
    track_ids: np.ndarray
    static: torch.Tensor
    history: torch.Tensor
    anchor_normalized: torch.Tensor
    anchor_physical: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    def validate(self) -> None:
        count = len(self.track_ids)
        if len(np.unique(self.track_ids)) != count:
            raise ProtocolError("Duplicate track IDs within an issue frame")
        for name in ("static", "history", "anchor_normalized", "anchor_physical"):
            value = getattr(self, name)
            if value.shape[0] != count:
                raise ProtocolError(
                    f"{name} row count {value.shape[0]} != track count {count}"
                )
            if not torch.isfinite(value).all():
                raise ProtocolError(f"{name} contains non-finite values")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ProtocolError("edge_index must have shape [2, E]")
        if self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ProtocolError("edge_attr and edge_index disagree on edge count")
        if self.edge_index.numel():
            lower = int(self.edge_index.min())
            upper = int(self.edge_index.max())
            if lower < 0 or upper >= count:
                raise ProtocolError("edge_index references a row outside this frame")


@dataclass(frozen=True)
class ObservationBatch:
    """Completed displacements observed after an issued prediction."""

    dataset: str
    movie: int
    frame: int
    track_ids: np.ndarray
    displacement: np.ndarray

    def validate(self) -> None:
        if self.displacement.shape != (len(self.track_ids), 2):
            raise ProtocolError("displacement must have shape [N, 2]")
        if len(np.unique(self.track_ids)) != len(self.track_ids):
            raise ProtocolError("Duplicate track IDs in observation")
        if not np.isfinite(self.displacement).all():
            raise ProtocolError("Observation contains non-finite values")


@dataclass(frozen=True)
class PredictionBatch:
    dataset: str
    movie: int
    issue_frame: int
    target_frame: int
    track_ids: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    process_scale: np.ndarray
    observation_scale: np.ndarray
    gain: np.ndarray
    measurement_mask: np.ndarray
