from __future__ import annotations

import unittest

import numpy as np
import torch

from airi_forecasting import (
    CausalInnovationStateSpaceForecaster,
    FrameBatch,
    ObservationBatch,
    ProtocolError,
    StreamingForecaster,
)


def frame(frame_number: int) -> FrameBatch:
    count = 2
    return FrameBatch(
        dataset="synthetic",
        movie=1,
        frame=frame_number,
        track_ids=np.array([10, 20], dtype=np.int64),
        static=torch.zeros((count, 3), dtype=torch.float32),
        history=torch.zeros((count, 2, 5), dtype=torch.float32),
        anchor_normalized=torch.zeros((count, 2), dtype=torch.float32),
        anchor_physical=torch.zeros((count, 2), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 8), dtype=torch.float32),
    )


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        model = CausalInnovationStateSpaceForecaster(
            static_dim=3,
            hidden=8,
            history_lags=2,
            correction_bound=1.0,
            dropout=0.0,
            use_update=True,
            use_graph=False,
            graph_heads=1,
        )
        self.forecaster = StreamingForecaster(model)

    def test_predict_then_observe_exposes_completed_innovation_next_step(self) -> None:
        first = self.forecaster.predict_before_observe(frame(0))
        self.assertEqual(first.target_frame, 1)
        np.testing.assert_array_equal(first.measurement_mask, np.zeros(2))

        self.forecaster.update_after_observe(
            ObservationBatch(
                dataset="synthetic",
                movie=1,
                frame=1,
                track_ids=np.array([10, 20], dtype=np.int64),
                displacement=np.array([[0.3, -0.1], [0.2, 0.4]]),
            )
        )
        second = self.forecaster.predict_before_observe(frame(1))
        np.testing.assert_array_equal(second.measurement_mask, np.ones(2))
        self.assertTrue(np.isfinite(second.mean).all())
        self.assertTrue((second.scale > 0).all())

    def test_observation_cannot_arrive_before_prediction(self) -> None:
        with self.assertRaises(ProtocolError):
            self.forecaster.update_after_observe(
                ObservationBatch(
                    dataset="synthetic",
                    movie=1,
                    frame=1,
                    track_ids=np.array([10], dtype=np.int64),
                    displacement=np.zeros((1, 2)),
                )
            )

    def test_issue_frames_must_increase(self) -> None:
        self.forecaster.predict_before_observe(frame(0))
        with self.assertRaises(ProtocolError):
            self.forecaster.predict_before_observe(frame(0))

    def test_duplicate_identity_is_rejected(self) -> None:
        batch = frame(0)
        invalid = FrameBatch(
            dataset=batch.dataset,
            movie=batch.movie,
            frame=batch.frame,
            track_ids=np.array([10, 10], dtype=np.int64),
            static=batch.static,
            history=batch.history,
            anchor_normalized=batch.anchor_normalized,
            anchor_physical=batch.anchor_physical,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )
        with self.assertRaises(ProtocolError):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
