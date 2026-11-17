"""Final fixed refits have no held-out inputs and retain the frozen last epoch."""

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.models import RecurrentForecaster
from oil_forecast.training import predict_scaled, set_reproducible, train_fixed_epochs


class ZeroLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, values):
        return self.head(values[:, -1])


def config(**changes):
    return {"epochs": 4, "seed": 42, "threads": 1, "batch_size": 4,
            "learning_rate": 0.0003, "weight_decay": 0.0001,
            "gradient_clip": 1.0, "shuffle_train": False, "optimizer": "Adam"} | changes


class FixedTraining(unittest.TestCase):
    def test_exact_frozen_last_epoch_matches_manual_updates_and_accepts_more_than_three(self):
        inputs = np.ones((4, 3, 2), dtype=np.float32)
        targets = np.tile(np.array([0.2, 2., 20.], dtype=np.float32), (4, 1))
        for loss_function in ["mse", "smooth_l1"]:
            with self.subTest(loss_function=loss_function):
                actual, manual = ZeroLinear(), ZeroLinear()
                options = config(loss_function=loss_function)
                optimizer = torch.optim.Adam(manual.parameters(), lr=options["learning_rate"],
                                             weight_decay=options["weight_decay"])
                objective = nn.MSELoss() if loss_function == "mse" else nn.SmoothL1Loss(beta=1.)
                for _ in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    objective(manual(torch.from_numpy(inputs)), torch.from_numpy(targets)).backward()
                    nn.utils.clip_grad_norm_(manual.parameters(), options["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                result = train_fixed_epochs(actual, inputs, targets, options)
                for key, value in manual.state_dict().items():
                    self.assertTrue(torch.equal(value, result["state_dict"][key]))
                self.assertEqual(result["selected_epoch"], 4)
                self.assertEqual(result["last_epoch"], 4)
                self.assertEqual(result["stopping_reason"], "fixed_development_epochs")
                self.assertEqual([row["epoch"] for row in result["history"]], [0, 1, 2, 3, 4])
                self.assertGreater(result["training_seconds"], 0.)
                self.assertGreaterEqual(result["elapsed_seconds"], result["training_seconds"])
                self.assertEqual(result["gradient_steps"], 4)
                self.assertEqual(result["nonzero_gradient_steps"], 4)
                self.assertNotIn("validation_loss", result["history"][-1])
                self.assertFalse(actual.training)

    def test_seed_repeats_weights_history_and_predictions_for_both_objectives(self):
        rng = np.random.default_rng(42)
        inputs = rng.normal(size=(9, 4, 2)).astype(np.float32)
        targets = rng.normal(size=(9, 3)).astype(np.float32)
        for objective in ["mse", "smooth_l1"]:
            results = []
            for _ in range(2):
                set_reproducible(42)
                model = RecurrentForecaster("RNN", input_size=2)
                result = train_fixed_epochs(model, inputs, targets, config(loss_function=objective, shuffle_train=True))
                results.append((result, predict_scaled(model, inputs)))
            first, second = results
            for name, value in first[0]["state_dict"].items():
                self.assertTrue(torch.equal(value, second[0]["state_dict"][name]))
            np.testing.assert_array_equal(first[1], second[1])
            for left, right in zip(first[0]["history"], second[0]["history"]):
                self.assertEqual(left["train_loss"], right["train_loss"])
                self.assertEqual(left["train_objective_loss"], right["train_objective_loss"])

    def test_invalid_config_and_nonfinite_inputs_fail_explicitly(self):
        inputs = np.ones((4, 3, 2), dtype=np.float32)
        targets = np.ones((4, 3), dtype=np.float32)
        for changes in [{"epochs": None}, {"epochs": 0}, {"epochs": 31}, {"epochs": True},
                        {"optimizer": "SGD"}, {"learning_rate": np.inf}, {"weight_decay": -1},
                        {"gradient_clip": 0}, {"batch_size": 0}, {"shuffle_train": 1},
                        {"loss_function": "mae"}, {"loss_function": "smooth_l1", "huber_beta": 0}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train_fixed_epochs(ZeroLinear(), inputs, targets, config(**changes))
        with self.assertRaises(ValueError):
            train_fixed_epochs(ZeroLinear().double(), inputs, targets, config())
        targets[0, 0] = np.nan
        with self.assertRaises(ValueError):
            train_fixed_epochs(ZeroLinear(), inputs, targets, config())

    def test_signature_accepts_training_only_and_snapshot_is_independent(self):
        self.assertEqual(list(inspect.signature(train_fixed_epochs).parameters), ["model", "X_train", "y_train", "config"])
        inputs = np.ones((4, 3, 2), dtype=np.float32)
        targets = np.ones((4, 3), dtype=np.float32)
        model = ZeroLinear()
        result = train_fixed_epochs(model, inputs, targets, config())
        saved = {key: value.clone() for key, value in result["state_dict"].items()}
        with torch.no_grad():
            model.head.weight.add_(100)
        for key, value in saved.items():
            self.assertTrue(torch.equal(value, result["state_dict"][key]))


if __name__ == "__main__":
    unittest.main()
