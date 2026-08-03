from pathlib import Path
import unittest

import torch

from lit_cell_forecasting.frozen_release import load_frozen_fold_state


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "models" / "lit_cell_mdck_bulk_primary"


class FrozenReleaseTests(unittest.TestCase):
    def test_reference_fold_loads_strictly_and_runs(self) -> None:
        frozen = load_frozen_fold_state(1, 42, release_dir=RELEASE)
        model = frozen.model
        static_dim = int(model.static_encoder[0].in_features)
        hidden = int(model.hidden)
        batch = 3
        output = model.forward_step(
            static=torch.zeros(batch, static_dim),
            history=torch.zeros(batch, model.history_lags, 5),
            innovation=torch.zeros(batch, 2),
            measurement_mask=torch.zeros(batch),
            previous_state=torch.zeros(batch, hidden),
            has_previous_state=torch.zeros(batch),
            anchor_h1=torch.zeros(batch, 2),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 8),
        )
        self.assertEqual(tuple(output[1].shape), (batch, 2))
        self.assertTrue(torch.isfinite(output[1]).all())


if __name__ == "__main__":
    unittest.main()
