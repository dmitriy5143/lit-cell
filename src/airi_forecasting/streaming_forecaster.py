"""Stateful inference wrapper implementing the frozen causal event order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .contracts import (
    FrameBatch,
    ObservationBatch,
    PredictionBatch,
    ProtocolError,
)
from .innovation_state import TrackState, TrackStateStore
from .model import CausalInnovationStateSpaceForecaster


class StreamingForecaster:
    def __init__(
        self,
        model: CausalInnovationStateSpaceForecaster,
        *,
        residual_scale: np.ndarray | None = None,
        eta: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.residual_scale = np.asarray(
            np.ones(2) if residual_scale is None else residual_scale,
            dtype=np.float32,
        )
        self.eta = float(eta)
        self.state = TrackStateStore()
        self._last_observation_frame: dict[tuple[str, int], int] = {}

    def reset_tracks(self) -> None:
        self.state.reset()
        self._last_observation_frame.clear()

    def reset_track(self, dataset: str, movie: int, track_id: int) -> None:
        """Reset identity state after division, relinking, or explicit track end."""
        self.state.reset_track(dataset, movie, track_id)

    def predict_before_observe(self, batch: FrameBatch) -> PredictionBatch:
        batch.validate()
        self.state.assert_issue_frame(batch.dataset, batch.movie, batch.frame)
        count = len(batch.track_ids)
        previous_states = torch.zeros(
            (count, self.model.hidden),
            dtype=torch.float32,
            device=self.device,
        )
        innovation = torch.zeros(
            (count, 2),
            dtype=torch.float32,
            device=self.device,
        )
        measurement_mask = torch.zeros(
            count,
            dtype=torch.float32,
            device=self.device,
        )
        has_previous_state = torch.zeros(
            count,
            dtype=torch.float32,
            device=self.device,
        )
        for row, raw_track_id in enumerate(batch.track_ids):
            entry = self.state.get(
                batch.dataset,
                batch.movie,
                int(raw_track_id),
            )
            if entry is None:
                continue
            if entry.frame != batch.frame - 1:
                self.state.reset_track(
                    batch.dataset,
                    batch.movie,
                    int(raw_track_id),
                )
                continue
            previous_states[row] = entry.state.to(self.device)
            has_previous_state[row] = 1.0
            if entry.pending_innovation is not None:
                innovation[row] = (
                    entry.pending_innovation.to(self.device)
                    / torch.as_tensor(
                        self.residual_scale,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                measurement_mask[row] = 1.0

        tensors = [
            batch.static,
            batch.history,
            batch.anchor_normalized,
            batch.anchor_physical,
            batch.edge_index,
            batch.edge_attr,
        ]
        static, history, anchor_normalized, anchor_physical, edge_index, edge_attr = [
            value.to(self.device) for value in tensors
        ]
        with torch.no_grad():
            (
                posterior,
                mean,
                predictive_scale,
                process_scale,
                observation_scale,
                gain,
            ) = self.model.forward_step(
                static,
                history,
                innovation,
                measurement_mask,
                previous_states,
                has_previous_state,
                anchor_normalized,
                edge_index,
                edge_attr,
            )
            if self.model.output_mode == "direct":
                direct = self.model.denormalize_direct_target(mean)
                physical_mean = anchor_physical + self.eta * (
                    direct - anchor_physical
                )
                physical_scale = predictive_scale * self.model.target_scale
            else:
                physical_mean = (
                    anchor_physical
                    + self.eta
                    * mean
                    * torch.as_tensor(
                        self.residual_scale,
                        dtype=mean.dtype,
                        device=self.device,
                    )
                )
                physical_scale = predictive_scale * torch.as_tensor(
                    self.residual_scale,
                    dtype=mean.dtype,
                    device=self.device,
                )

        for row, raw_track_id in enumerate(batch.track_ids):
            previous = self.state.get(
                batch.dataset,
                batch.movie,
                int(raw_track_id),
            )
            self.state.put(
                batch.dataset,
                batch.movie,
                int(raw_track_id),
                TrackState(
                    frame=batch.frame,
                    state=posterior[row].detach().cpu(),
                    prediction=physical_mean[row].detach().cpu(),
                    pending_innovation=None,
                    last_innovation=(
                        None if previous is None else previous.pending_innovation
                    ),
                ),
            )

        return PredictionBatch(
            dataset=batch.dataset,
            movie=batch.movie,
            issue_frame=batch.frame,
            target_frame=batch.frame + 1,
            track_ids=np.asarray(batch.track_ids, dtype=np.int64).copy(),
            mean=physical_mean.cpu().numpy(),
            scale=physical_scale.cpu().numpy(),
            process_scale=(
                process_scale
                * torch.as_tensor(
                    self.residual_scale,
                    dtype=process_scale.dtype,
                    device=self.device,
                )
            )
            .cpu()
            .numpy(),
            observation_scale=(
                observation_scale
                * torch.as_tensor(
                    self.residual_scale,
                    dtype=observation_scale.dtype,
                    device=self.device,
                )
            )
            .cpu()
            .numpy(),
            gain=gain.mean(dim=1).cpu().numpy(),
            measurement_mask=measurement_mask.cpu().numpy(),
        )

    def update_after_observe(self, observation: ObservationBatch) -> None:
        observation.validate()
        movie_key = (observation.dataset, int(observation.movie))
        previous_observation = self._last_observation_frame.get(movie_key)
        if (
            previous_observation is not None
            and observation.frame <= previous_observation
        ):
            raise ProtocolError(
                "Observation frames must increase strictly within a movie"
            )
        latest_issue = self.state._last_issue_frame.get(movie_key)
        if latest_issue is None or observation.frame != latest_issue + 1:
            raise ProtocolError(
                "Observation must complete the most recently issued transition"
            )
        self._last_observation_frame[movie_key] = int(observation.frame)
        for row, raw_track_id in enumerate(observation.track_ids):
            entry = self.state.get(
                observation.dataset,
                observation.movie,
                int(raw_track_id),
            )
            if entry is None or entry.frame != observation.frame - 1:
                continue
            observed = torch.as_tensor(
                observation.displacement[row],
                dtype=torch.float32,
            )
            entry.pending_innovation = observed - entry.prediction
            entry.last_innovation = entry.pending_innovation.clone()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": {
                    "static_dim": self.model.static_encoder[0].in_features,
                    "hidden": self.model.hidden,
                    "history_lags": self.model.history_lags,
                    "correction_bound": self.model.correction_bound,
                    "dropout": self.model.dropout,
                    "use_update": self.model.use_update,
                    "use_graph": self.model.use_graph,
                    "graph_heads": (
                        self.model.graph_conv.heads if self.model.use_graph else 1
                    ),
                    "output_mode": self.model.output_mode,
                    "target_mean": self.model.target_mean.cpu().numpy(),
                    "target_scale": self.model.target_scale.cpu().numpy(),
                },
                "residual_scale": self.residual_scale,
                "eta": self.eta,
            },
            target,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "StreamingForecaster":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = CausalInnovationStateSpaceForecaster(**payload["model_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(
            model,
            residual_scale=payload["residual_scale"],
            eta=float(payload["eta"]),
            device=device,
        )
