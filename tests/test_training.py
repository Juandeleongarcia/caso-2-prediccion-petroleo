"""Small CPU training, validation isolation and deterministic checkpoints."""

import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.training import set_reproducible, train_short
from oil_forecast.models import RecurrentForecaster


class TinyForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 3)

    def forward(self, values):
        return self.head(values[:, -1])


def fixture():
    x = np.zeros((12, 4, 2), dtype=np.float32)
    x[:, :, 0] = np.linspace(-1, 1, len(x))[:, None]
    y = np.full((12, 3), 0.3, dtype=np.float32)
    return x, y


def config(**changes):
    return {"seed": 123, "threads": 1, "epochs": 3, "batch_size": 4,
            "learning_rate": 0.03, "optimizer": "Adam", "gradient_clip": 1.0,
            "weight_decay": 0.0, "shuffle_train": True} | changes


class ShortTrainingContracts(unittest.TestCase):
    def test_three_epochs_have_finite_gradients_predictions_and_lower_loss(self):
        x, y = fixture()
        set_reproducible(123)
        model = TinyForecaster()
        result = train_short(model, x, y, x[:5], y[:5], config())
        self.assertEqual(result["predictions"].shape, (5, 3))
        self.assertEqual([entry["epoch"] for entry in result["history"]], [0, 1, 2, 3])
        self.assertTrue(result["loss_decreased"])
        self.assertGreater(result["training_seconds"], 0.0)
        self.assertEqual(result["gradient_steps"], 9)
        self.assertEqual(result["nonzero_gradient_steps"], 9)
        self.assertTrue(np.isfinite(result["predictions"]).all())
        self.assertFalse(model.training)

    def test_same_seed_repeats_exact_weights_history_and_predictions(self):
        x, y = fixture()
        states, outcomes = [], []
        for _ in range(2):
            set_reproducible(123)
            model = TinyForecaster()
            result = train_short(model, x, y, x[:5], y[:5], config())
            states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            outcomes.append(result)
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]))
        np.testing.assert_array_equal(outcomes[0]["predictions"], outcomes[1]["predictions"])
        for left, right in zip(outcomes[0]["history"], outcomes[1]["history"]):
            self.assertEqual(left["train_loss"], right["train_loss"])
            self.assertEqual(left["validation_loss"], right["validation_loss"])

    def test_validation_targets_never_change_optimizer_updates(self):
        x, y = fixture()
        states = []
        for valid_targets in [y[:5], y[:5] + 1000]:
            set_reproducible(123)
            model = TinyForecaster()
            train_short(model, x, y, x[:5], valid_targets, config())
            states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]))

    def test_each_recurrent_architecture_repeats_exact_short_training(self):
        x, y = fixture()
        for architecture in ["RNN", "LSTM", "GRU"]:
            with self.subTest(architecture=architecture):
                states, outcomes = [], []
                for _ in range(2):
                    set_reproducible(123)
                    model = RecurrentForecaster(architecture, input_size=2,
                                                hidden_size=32, num_layers=1)
                    result = train_short(model, x, y, x[:5], y[:5], config(epochs=2))
                    states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
                    outcomes.append(result)
                for key in states[0]:
                    self.assertTrue(torch.equal(states[0][key], states[1][key]))
                np.testing.assert_array_equal(outcomes[0]["predictions"], outcomes[1]["predictions"])
                for left, right in zip(outcomes[0]["history"], outcomes[1]["history"]):
                    self.assertEqual(left["train_loss"], right["train_loss"])
                    self.assertEqual(left["validation_loss"], right["validation_loss"])

    def test_safe_state_dict_roundtrip_keeps_trained_predictions(self):
        x, y = fixture()
        set_reproducible(123)
        model = TinyForecaster()
        result = train_short(model, x, y, x[:5], y[:5], config(epochs=1))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = TinyForecaster()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        restored.eval()
        with torch.inference_mode():
            predictions = restored(torch.as_tensor(x[:5])).numpy()
        np.testing.assert_allclose(predictions, result["predictions"], rtol=1e-6, atol=1e-7)

    def test_invalid_shapes_nonfinite_values_and_extended_runs_are_rejected(self):
        x, y = fixture()
        malformed = x.copy()
        malformed[0, 0, 0] = np.nan
        for inputs, labels, options in [(malformed, y, {}), (x, y[:, :2], {}),
                                         (x, y, {"epochs": 4}),
                                         (x, y, {"optimizer": "SGD"})]:
            with self.subTest(options=options, target_shape=labels.shape):
                with self.assertRaises(ValueError):
                    train_short(TinyForecaster(), inputs, labels, x[:5], y[:5], config(**options))


if __name__ == "__main__":
    unittest.main()
