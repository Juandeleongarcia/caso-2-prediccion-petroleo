"""Architecture contracts on synthetic tensors; no forecasting experiments."""

import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.models import RecurrentForecaster, build_model, count_parameters


class RecurrentModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_three_architectures_share_input_and_direct_output_dimensions(self):
        inputs = torch.randn(4, 60, 20)
        for architecture in ["RNN", "LSTM", "GRU"]:
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture)
                output = model(inputs)
                self.assertEqual(tuple(output.shape), (4, 3))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(output.device.type, "cpu")
                self.assertFalse(model.recurrent.bidirectional)
                self.assertEqual(model.recurrent.dropout, 0.0)

    def test_default_parameter_counts_are_exact(self):
        for architecture, expected in [("RNN", 1827), ("LSTM", 7011), ("GRU", 5283)]:
            with self.subTest(architecture=architecture):
                self.assertEqual(count_parameters(RecurrentForecaster(architecture)), expected)

    def test_all_parameters_receive_finite_nonzero_gradients(self):
        inputs, target = torch.randn(3, 8, 20), torch.randn(3, 3)
        for architecture in ["RNN", "LSTM", "GRU"]:
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture)
                loss = torch.nn.functional.mse_loss(model(inputs), target)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_hidden_state_is_not_carried_between_calls_or_windows(self):
        inputs = torch.randn(4, 12, 20)
        for architecture in ["RNN", "LSTM", "GRU"]:
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture).eval()
                with torch.inference_mode():
                    together = model(inputs)
                    model(torch.randn(2, 12, 20))
                    repeated = model(inputs)
                    separate = torch.cat([model(window.unsqueeze(0)) for window in inputs])
                torch.testing.assert_close(together, repeated, rtol=0, atol=0)
                torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-6)

    def test_two_layers_and_64_hidden_units_preserve_three_outputs(self):
        for architecture in ["RNN", "LSTM", "GRU"]:
            with self.subTest(architecture=architecture):
                model = RecurrentForecaster(architecture, input_size=7, hidden_size=64, num_layers=2)
                self.assertEqual(tuple(model(torch.randn(2, 5, 7)).shape), (2, 3))
                self.assertEqual(model.config()["num_layers"], 2)

    def test_trusted_state_dict_roundtrip_preserves_predictions(self):
        inputs = torch.randn(2, 9, 20)
        for architecture in ["rnn", "lstm", "gru"]:
            with self.subTest(architecture=architecture):
                original = RecurrentForecaster(architecture).eval()
                stream = io.BytesIO()
                torch.save({"model_config": original.config(), "state_dict": original.state_dict()}, stream)
                stream.seek(0)
                saved = torch.load(stream, map_location="cpu", weights_only=True)
                loaded = build_model(saved["model_config"]).eval()
                loaded.load_state_dict(saved["state_dict"], strict=True)
                with torch.inference_mode():
                    torch.testing.assert_close(original(inputs), loaded(inputs), rtol=0, atol=0)

    def test_seed_reproduces_initial_weights_and_outputs(self):
        inputs = torch.randn(2, 7, 20)
        for architecture in ["RNN", "LSTM", "GRU"]:
            with self.subTest(architecture=architecture):
                torch.manual_seed(123)
                first = RecurrentForecaster(architecture).eval()
                torch.manual_seed(123)
                second = RecurrentForecaster(architecture).eval()
                with torch.inference_mode():
                    torch.testing.assert_close(first(inputs), second(inputs), rtol=0, atol=0)

    def test_invalid_architectures_and_dimensions_fail_explicitly(self):
        for config in [{"architecture": "transformer"}, {"architecture": None},
                       {"input_size": 0}, {"input_size": True}, {"hidden_size": 16},
                       {"num_layers": 0}, {"num_layers": 3}, {"output_size": 1}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                RecurrentForecaster(**config)
        model = RecurrentForecaster()
        for inputs in [torch.randn(3, 20), torch.randn(2, 4, 19), torch.empty(0, 4, 20),
                       torch.empty(2, 0, 20), torch.ones(2, 4, 20, dtype=torch.int64)]:
            with self.subTest(shape=tuple(inputs.shape)), self.assertRaises(ValueError):
                model(inputs)


if __name__ == "__main__":
    unittest.main()
