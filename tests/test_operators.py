from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from airi_forecasting.bounded_semigroup import bounded_update, consecutive_windows
from airi_forecasting.equivariant_field_law import vector_laplacian
from airi_forecasting.neighbourhood_transport import (
    coherent_wrong_cell,
    local_previous_state,
)


class OperatorTests(unittest.TestCase):
    def test_bounded_update_preserves_direction_and_limits_norm(self) -> None:
        value = np.array([[3.0, 4.0], [0.0, 0.0], [-6.0, 8.0]])
        result = bounded_update(value, bound_px=2.0)
        self.assertTrue((np.linalg.norm(result, axis=1) <= 2.0 + 1e-10).all())
        self.assertGreater(np.dot(value[0], result[0]), 0.0)
        np.testing.assert_array_equal(result[1], np.zeros(2))

    def test_consecutive_windows_never_cross_track_identity(self) -> None:
        rows = pd.DataFrame(
            {
                "track_id": [1, 1, 1, 2, 2],
                "frame": [0, 1, 2, 0, 2],
            }
        )
        windows = consecutive_windows(rows, horizon=2)
        np.testing.assert_array_equal(windows, np.array([[0, 1], [1, 2]]))

    def test_local_transport_excludes_the_receiver_track(self) -> None:
        result = local_previous_state(
            current_position=np.array([[0.0, 0.0]]),
            previous_position=np.array([[0.0, 0.0], [1.0, 0.0]]),
            previous_score=np.array([[100.0, 100.0], [2.0, -3.0]]),
            current_tracks=np.array([7]),
            previous_tracks=np.array([7, 8]),
            scales=[30.0],
        )
        np.testing.assert_allclose(
            [result["local_30_x"][0], result["local_30_y"][0]],
            [2.0, -3.0],
            atol=1e-10,
        )

    def test_wrong_cell_control_is_a_derangement_within_each_frame(self) -> None:
        rows = pd.DataFrame(
            {
                "frame": [0, 0, 0, 1, 1, 1],
                "track_id": [1, 2, 3, 1, 2, 3],
            }
        )
        value = np.arange(12, dtype=float).reshape(6, 2)
        shuffled, permutation = coherent_wrong_cell(value, rows, seed=42)
        self.assertTrue(np.all(permutation != np.arange(len(rows))))
        np.testing.assert_array_equal(shuffled, value[permutation])
        for indices in rows.groupby("frame").indices.values():
            self.assertEqual(set(permutation[indices]), set(indices))

    def test_laplacian_is_equivariant_under_half_turn(self) -> None:
        rng = np.random.default_rng(12)
        field = rng.normal(size=(9, 11, 2))
        rotated = np.flip(np.flip(-field, axis=0), axis=1)
        lhs = vector_laplacian(rotated, spacing=1.0)
        rhs = np.flip(np.flip(-vector_laplacian(field, spacing=1.0), axis=0), axis=1)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
