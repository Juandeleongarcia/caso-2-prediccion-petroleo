"""Optional robust optimization keeps the same MSE-based inner selection."""

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


def config(**changes):
    return {"max_epochs": 3, "patience": 2, "min_delta": 0.0001,
            "seed": 123, "threads": 1, "batch_size": 4, "learning_rate": 0.01,
            "weight_decay": 0.0001, "gradient_clip": 1.0, "optimizer": "Adam",
            "shuffle_train": False} | changes


class OptionalLossContracts(unittest.TestCase):
    def test_default_and_explicit_mse_preserve_identical_weights_losses_and_predictions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(12, 4, 2)).astype(np.float32)
        y = rng.normal(size=(12, 3)).astype(np.float32)
        for architecture in ("RNN", "LSTM", "GRU"):
            with self.subTest(architecture=architecture):
                results = []
                for objective_options in ({}, {"loss_function": "mse"}):
                    set_reproducible(123)
                    model = RecurrentForecaster(architecture, input_size=2)
                    results.append(train_with_early_stopping(
                        model, x, y, x[:5], y[:5], config(**objective_options)))
                first, second = results
                self.assertEqual(first["best_epoch"], second["best_epoch"])
                for key, value in first["state_dict"].items():
                    self.assertTrue(torch.equal(value, second["state_dict"][key]))
                np.testing.assert_array_equal(first["inner_predictions"], second["inner_predictions"])
                for left, right in zip(first["history"], second["history"]):
                    for key in ("train_loss", "validation_loss", "train_objective_loss"):
                        self.assertEqual(left[key], right[key])
                    self.assertEqual(left["train_objective_loss"], left["train_loss"])
                self.assertEqual(first["loss_function"], "mse")
                self.assertEqual(first["validation_metric"], "mse")

    def test_smooth_l1_uses_its_actual_gradients_and_records_distinct_objective(self):
        x = np.ones((4, 3, 2), dtype=np.float32)
        y = np.tile(np.array([0.2, 2.0, 20.0], dtype=np.float32), (4, 1))
        options = config(max_epochs=1, loss_function="smooth_l1", huber_beta=1.0)
        model, manual = ZeroLinear(), ZeroLinear()
        optimizer = torch.optim.Adam(manual.parameters(), lr=options["learning_rate"],
                                     weight_decay=options["weight_decay"])
        prediction = manual(torch.from_numpy(x))
        objective = nn.SmoothL1Loss(beta=1.0, reduction="mean")
        initial_objective = float(objective(prediction, torch.from_numpy(y)).detach())
        objective(prediction, torch.from_numpy(y)).backward()
        expected_norm = float(nn.utils.clip_grad_norm_(manual.parameters(), 1.0))
        optimizer.step()
        result = train_with_early_stopping(model, x, y, x, y, options)
        for key, value in manual.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertEqual(result["max_gradient_norm"], expected_norm)
        self.assertEqual(result["history"][0]["train_objective_loss"], initial_objective)
        self.assertLess(initial_objective, result["initial_train_loss"])
        self.assertEqual(result["loss_function"], "smooth_l1")
        self.assertEqual(result["huber_beta"], 1.0)
        self.assertEqual(result["validation_metric"], "mse")
        predicted = predict_scaled(model, x)
        mse = float(np.mean((predicted - y) ** 2))
        np.testing.assert_allclose(result["restored_validation_loss"], mse, rtol=1e-6, atol=1e-7)

    def test_smooth_l1_restores_minimum_inner_mse_checkpoint_not_last(self):
        x = np.ones((12, 4, 2), dtype=np.float32)
        y = np.ones((12, 3), dtype=np.float32)
        model = ZeroLinear()
        result = train_with_early_stopping(
            model, x, y, x[:5], -y[:5], config(loss_function="smooth_l1"))
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["last_epoch"], 3)
        self.assertEqual(result["stopping_reason"], "early_stopping")
        losses = [row["validation_loss"] for row in result["history"][1:]]
        self.assertEqual(result["validation_loss_at_best"], min(losses))
        self.assertAlmostEqual(result["restored_validation_loss"], min(losses), places=12)
        first_epoch = ZeroLinear()
        train_with_early_stopping(first_epoch, x, y, x[:5], -y[:5],
                                  config(max_epochs=1, loss_function="smooth_l1"))
        for key, value in first_epoch.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_unknown_objective_and_invalid_huber_beta_fail_explicitly(self):
        x = np.ones((4, 3, 2), dtype=np.float32)
        y = np.ones((4, 3), dtype=np.float32)
        for options in ({"loss_function": "mae"}, {"loss_function": None},
                        *({"loss_function": "smooth_l1", "huber_beta": beta}
                          for beta in (0, -1, float("inf"), float("nan"), None, "1", True))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                train_with_early_stopping(ZeroLinear(), x, y, x, y, config(**options))


if __name__ == "__main__":
    unittest.main()
