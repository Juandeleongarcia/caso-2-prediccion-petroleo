"""Inner-validation selection, real checkpoint restoration and CPU determinism."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.models import RecurrentForecaster
from oil_forecast.training import predict_scaled, set_reproducible, train_with_early_stopping


class ZeroLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, values):
        return self.head(values[:, -1])


def fixture():
    return np.ones((12, 4, 2), dtype=np.float32), np.ones((12, 3), dtype=np.float32)


def config(**changes):
    return {"max_epochs": 20, "patience": 2, "min_delta": 0.0001,
            "seed": 123, "threads": 1, "batch_size": 4, "learning_rate": 0.01,
            "weight_decay": 0.0, "gradient_clip": 1.0, "optimizer": "Adam",
            "shuffle_train": True} | changes


class EarlyStoppingContracts(unittest.TestCase):
    def test_patience_uses_inner_validation_and_restores_epoch_one_not_last(self):
        x, y = fixture()
        model = ZeroLinear()
        result = train_with_early_stopping(model, x, y, x[:5], -y[:5], config())
        self.assertEqual(result["stopping_reason"], "early_stopping")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["last_epoch"], 3)
        self.assertLess(result["last_train_loss"], result["train_loss_at_best"])
        self.assertGreater(result["last_validation_loss"], result["validation_loss_at_best"])
        one_epoch = ZeroLinear()
        train_with_early_stopping(one_epoch, x, y, x[:5], -y[:5], config(max_epochs=1))
        for name, value in one_epoch.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        np.testing.assert_array_equal(predict_scaled(model, x[:5], 4), result["inner_predictions"])
        self.assertAlmostEqual(result["restored_validation_loss"], result["validation_loss_at_best"], places=12)

    def test_min_delta_controls_patience_but_retains_exact_minimum(self):
        x, y = fixture()
        result = train_with_early_stopping(ZeroLinear(), x, y, x[:5], y[:5],
                                           config(min_delta=1.0, learning_rate=0.001))
        self.assertEqual(result["last_epoch"], 3)
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["stopping_reason"], "early_stopping")
        losses = [row["validation_loss"] for row in result["history"][1:]]
        self.assertLess(losses[-1], losses[0])
        self.assertEqual(result["validation_loss_at_best"], min(losses))

    def test_epoch_zero_is_diagnostic_and_not_a_selected_untrained_network(self):
        x, y = fixture()
        result = train_with_early_stopping(ZeroLinear(), x, y, x[:5], np.zeros_like(y[:5]), config())
        self.assertEqual(result["initial_validation_loss"], 0.0)
        self.assertEqual(result["best_epoch"], 1)
        self.assertGreater(result["validation_loss_at_best"], 0.0)

    def test_maximum_epochs_reason_and_component_losses_are_recorded(self):
        x, y = fixture()
        result = train_with_early_stopping(ZeroLinear(), x, y, x[:5], y[:5],
                                           config(max_epochs=2, patience=5))
        self.assertEqual(result["stopping_reason"], "max_epochs")
        self.assertEqual([item["epoch"] for item in result["history"]], [0, 1, 2])
        self.assertEqual(result["last_epoch"], 2)
        self.assertGreater(result["training_seconds"], 0)
        self.assertGreaterEqual(result["elapsed_seconds"], result["training_seconds"])
        self.assertEqual(len(result["train_loss_components"]), 3)
        self.assertAlmostEqual(np.mean(result["validation_loss_components"]),
                               result["restored_validation_loss"], places=6)

    def test_repeated_recurrent_training_restores_identical_state_and_predictions(self):
        x, y = fixture()
        states, outcomes = [], []
        for _ in range(2):
            set_reproducible(123)
            model = RecurrentForecaster("RNN", input_size=2)
            result = train_with_early_stopping(model, x, y, x[:5], y[:5],
                                               config(max_epochs=3, patience=2))
            states.append({name: value.clone() for name, value in model.state_dict().items()})
            outcomes.append(result)
        self.assertEqual(outcomes[0]["best_epoch"], outcomes[1]["best_epoch"])
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]))
        np.testing.assert_array_equal(outcomes[0]["inner_predictions"], outcomes[1]["inner_predictions"])
        for left, right in zip(outcomes[0]["history"], outcomes[1]["history"]):
            self.assertEqual(left["train_loss"], right["train_loss"])
            self.assertEqual(left["validation_loss"], right["validation_loss"])

    def test_nonfinite_data_and_invalid_stopping_configuration_are_rejected(self):
        x, y = fixture()
        malformed = x.copy()
        malformed[0, 0, 0] = np.nan
        for options in [{"max_epochs": 31}, {"max_epochs": 0}, {"patience": 0},
                        {"min_delta": -1.0}, {"min_delta": np.inf}, {"optimizer": "SGD"}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                train_with_early_stopping(ZeroLinear(), x, y, x[:5], y[:5], config(**options))
        with self.assertRaisesRegex(ValueError, "finite"):
            train_with_early_stopping(ZeroLinear(), malformed, y, x[:5], y[:5], config())
        with self.assertRaises(ValueError):
            predict_scaled(ZeroLinear(), malformed)


if __name__ == "__main__":
    unittest.main()
