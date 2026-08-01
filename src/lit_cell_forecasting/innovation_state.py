"""Track-local recurrent state and completed-innovation storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import ProtocolError


@dataclass
class TrackState:
    frame: int
    state: torch.Tensor
    prediction: torch.Tensor
    pending_innovation: torch.Tensor | None = None
    last_innovation: torch.Tensor | None = None


class TrackStateStore:
    def __init__(self) -> None:
        self._tracks: dict[tuple[str, int, int], TrackState] = {}
        self._last_issue_frame: dict[tuple[str, int], int] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._last_issue_frame.clear()

    def reset_track(self, dataset: str, movie: int, track_id: int) -> None:
        self._tracks.pop((dataset, int(movie), int(track_id)), None)

    def assert_issue_frame(self, dataset: str, movie: int, frame: int) -> None:
        key = (dataset, int(movie))
        previous = self._last_issue_frame.get(key)
        if previous is not None and int(frame) <= previous:
            raise ProtocolError(
                f"Issue frames must increase strictly: {frame} after {previous}"
            )
        self._last_issue_frame[key] = int(frame)

    def get(self, dataset: str, movie: int, track_id: int) -> TrackState | None:
        return self._tracks.get((dataset, int(movie), int(track_id)))

    def put(
        self,
        dataset: str,
        movie: int,
        track_id: int,
        value: TrackState,
    ) -> None:
        self._tracks[(dataset, int(movie), int(track_id))] = value
