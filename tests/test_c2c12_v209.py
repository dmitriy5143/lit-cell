from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "experiments" / "publication"
if str(PUBLICATION) not in sys.path:
    sys.path.insert(0, str(PUBLICATION))

import run_c2c12_lit_cell_external_confirmation_v209 as v209


class C2C12ExternalOperatorTests(unittest.TestCase):
    def test_packet_uses_only_completed_previous_transition(self) -> None:
        rows = pd.DataFrame(
            {
                "frame": np.repeat(np.arange(3), 2),
                "track_id": np.tile([11, 22], 3),
                "x_px": np.tile([0.0, 10.0], 3),
                "y_px": np.zeros(6),
            }
        )
        scores = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
            dtype=np.float64,
        )
        packets, names, latest, stale_latest, _dnn = v209.build_packet(
            rows,
            scores,
            multipliers=[1.0],
            dnn_reference=10.0,
            dnn_low=5.0,
            dnn_high=20.0,
            adaptive=True,
            seed=42,
        )
        own_x = names.index("own_prev_x")
        local_x = names.index("local_m1_x")
        np.testing.assert_allclose(packets["real"][2:4, own_x], [1.0, 2.0])
        np.testing.assert_allclose(packets["real"][2:4, local_x], [2.0, 1.0])
        np.testing.assert_array_equal(latest[2:4], [0, 0])
        np.testing.assert_array_equal(stale_latest[4:6], [0, 0])

        changed = scores.copy()
        changed[4:] += 1000.0
        changed_packets, *_rest = v209.build_packet(
            rows,
            changed,
            multipliers=[1.0],
            dnn_reference=10.0,
            dnn_low=5.0,
            dnn_high=20.0,
            adaptive=True,
            seed=42,
        )
        np.testing.assert_allclose(changed_packets["real"], packets["real"])

    def test_equivariant_operator_rotates_with_vector_inputs(self) -> None:
        packet_names = [
            "own_prev_x",
            "own_prev_y",
            "global_prev_x",
            "global_prev_y",
            "local_m1_x",
            "local_m1_y",
        ]
        packet = np.asarray(
            [[1.0, 2.0, -1.0, 0.5, 0.2, -0.4], [0.5, -0.2, 0.7, 1.1, -0.3, 0.8]],
            dtype=np.float32,
        )

        def field(value: np.ndarray) -> v209.FieldState:
            return v209.FieldState(
                annotation_kind="synthetic",
                split="test",
                experiment=1,
                sequence=1,
                field=1,
                rows=pd.DataFrame({"frame": [0, 1], "track_id": [1, 1]}),
                target=np.zeros((2, 2)),
                base=np.zeros((2, 2)),
                scale=np.asarray([[1.5, 1.5], [0.75, 0.75]]),
                normal_score=np.zeros((2, 2)),
                packet_names=packet_names,
                packets={"real": value},
                latest_real_donor=np.asarray([-1, 0]),
                latest_stale_donor=np.asarray([-1, -1]),
                frame_dnn_px=np.ones(2),
                windows={horizon: np.empty((0, horizon), dtype=np.int64) for horizon in v209.HORIZONS},
                baselines={},
            )

        original = field(packet)
        _design, names = v209.equivariant_vector_design(original)
        model = v209.EquivariantRidgeModel(
            vector_scale=np.ones(len(names)),
            coefficients=np.linspace(0.1, 0.1 * len(names), len(names)),
            design_names=names,
        )
        prediction = v209.equivariant_correction(model, original, "real")

        angle = 0.73
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated_packet = packet.copy()
        for start in (0, 2, 4):
            rotated_packet[:, start : start + 2] = packet[:, start : start + 2] @ rotation.T
        rotated_prediction = v209.equivariant_correction(
            model, field(rotated_packet), "real"
        )
        np.testing.assert_allclose(
            rotated_prediction, prediction @ rotation.T, atol=1e-7, rtol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
