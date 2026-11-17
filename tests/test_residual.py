"""Persistence-centred output initialization in the existing residual units."""

import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.models import (RecurrentForecaster, build_model, count_parameters,
                                 initialize_persistence_head)
from oil_forecast.preprocessing import TargetTransformer
from oil_forecast.training import set_reproducible


def target_transformer():
    return TargetTransformer().fit(np.array([
        [-2.0, -5.0, -12.0], [1.0, 8.0, 20.0],
        [5.0, 17.0, 50.0], [0.0, 11.0, 35.0],
    ]))


class PersistenceInitializationContracts(unittest.TestCase):
    def setUp(self):
        set_reproducible(42)

    def test_all_architectures_start_at_persistence_after_inverse_scaling(self):
        transformer = target_transformer()
        inputs = torch.randn(4, 8, 20)
        reference = np.array([20.0, 60.0, 80.0, 100.0])[:, None]
        for architecture in ("RNN", "LSTM", "GRU"):
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture)
                initialize_persistence_head(model, transformer.mean_, transformer.scale_)
                with torch.inference_mode():
                    scaled = model(inputs).numpy()
                expected = np.broadcast_to(-transformer.mean_ / transformer.scale_, scaled.shape)
                np.testing.assert_allclose(scaled, expected, rtol=1e-7, atol=1e-7)
                changes = transformer.inverse_transform(scaled)
                np.testing.assert_allclose(changes, 0.0, rtol=0, atol=2e-6)
                np.testing.assert_allclose(reference + changes,
                                           np.broadcast_to(reference, changes.shape),
                                           rtol=0, atol=2e-6)
        # A zero standardized output is the train mean change, not persistence.
        np.testing.assert_allclose(transformer.inverse_transform(np.zeros((1, 3))),
                                   transformer.mean_[None, :], rtol=0, atol=0)
        zero_change = transformer.transform(np.zeros((4, 3)))
        np.testing.assert_allclose(transformer.inverse_transform(zero_change), 0, atol=1e-14)

    def test_initialization_preserves_encoder_rng_config_keys_and_parameter_count(self):
        transformer = target_transformer()
        for architecture in ("RNN", "LSTM", "GRU"):
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture)
                before = {key: value.clone() for key, value in model.state_dict().items()}
                parameters, config = count_parameters(model), model.config()
                rng = torch.random.get_rng_state().clone()
                initialize_persistence_head(model, transformer.mean_, transformer.scale_)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
                self.assertEqual(model.config(), config)
                self.assertEqual(count_parameters(model), parameters)
                self.assertEqual(set(before), set(model.state_dict()))
                self.assertTrue(torch.equal(model.head.weight, torch.zeros_like(model.head.weight)))
                for key, value in before.items():
                    if key.startswith("recurrent."):
                        self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_initialized_checkpoint_roundtrip_and_seed_are_exact(self):
        transformer = target_transformer()
        inputs = torch.randn(3, 7, 20)
        for architecture in ("RNN", "LSTM", "GRU"):
            with self.subTest(architecture=architecture):
                models = []
                for _ in range(2):
                    set_reproducible(123)
                    model = RecurrentForecaster(architecture)
                    initialize_persistence_head(model, transformer.mean_, transformer.scale_)
                    models.append(model)
                for key, value in models[0].state_dict().items():
                    self.assertTrue(torch.equal(value, models[1].state_dict()[key]))
                buffer = io.BytesIO()
                torch.save({"model_config": models[0].config(),
                            "state_dict": models[0].state_dict()}, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, map_location="cpu", weights_only=True)
                restored = build_model(saved["model_config"])
                restored.load_state_dict(saved["state_dict"], strict=True)
                with torch.inference_mode():
                    torch.testing.assert_close(models[0](inputs), restored(inputs), rtol=0, atol=0)

    def test_head_learns_first_and_encoder_receives_gradients_after_an_update(self):
        transformer = target_transformer()
        inputs = torch.randn(4, 6, 20)
        targets = torch.tensor([[0.4, -0.2, 0.8]]).expand(4, 3)
        for architecture in ("RNN", "LSTM", "GRU"):
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture)
                initialize_persistence_head(model, transformer.mean_, transformer.scale_)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
                torch.nn.functional.mse_loss(model(inputs), targets).backward()
                self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
                for parameter in model.recurrent.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertEqual(float(parameter.grad.abs().sum()), 0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(inputs), targets).backward()
                for parameter in model.recurrent.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_invalid_statistics_or_model_fail_before_changing_weights(self):
        cases = [([0, 0], [1, 1, 1]), ([0, 0, 0], [1, 1]),
                 ([[0, 0, 0]], [1, 1, 1]), ([0, 0, 0], [[1, 1, 1]]),
                 ([float("nan"), 0, 0], [1, 1, 1]),
                 ([float("inf"), 0, 0], [1, 1, 1]),
                 ([0, 0, 0], [1, float("nan"), 1]),
                 ([0, 0, 0], [1, float("inf"), 1]),
                 ([0, 0, 0], [0, 1, 1]), ([0, 0, 0], [-1, 1, 1]),
                 ([1e300, 0, 0], [1e-300, 1, 1]),
                 (["invalid", 0, 0], [1, 1, 1])]
        model = RecurrentForecaster()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for mean, scale in cases:
            with self.subTest(mean=mean, scale=scale), self.assertRaises(ValueError):
                initialize_persistence_head(model, mean, scale)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
        with self.assertRaises(ValueError):
            initialize_persistence_head(torch.nn.Linear(32, 3), [0, 0, 0], [1, 1, 1])
        with self.assertRaises(ValueError):
            initialize_persistence_head(RecurrentForecaster().double(), [0, 0, 0], [1, 1, 1])
        with self.assertRaises(ValueError):
            initialize_persistence_head(RecurrentForecaster().to("meta"), [0, 0, 0], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
