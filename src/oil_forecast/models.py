"""Small comparable recurrent networks with three direct forecast outputs."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    """Encode a window and predict its three scaled Brent price changes.

    Every call starts with the recurrent layer's zero state. No hidden state is
    carried between overlapping windows, batches or evaluation blocks.
    """

    def __init__(self, architecture: str = "RNN", input_size: int = 20,
                 hidden_size: int = 32, num_layers: int = 1,
                 output_size: int = 3) -> None:
        super().__init__()
        if not isinstance(architecture, str) or architecture.upper() not in {"RNN", "LSTM", "GRU"}:
            raise ValueError("architecture must be RNN, LSTM or GRU")
        if type(input_size) is not int or input_size < 1:
            raise ValueError("input_size must be a positive integer")
        if type(hidden_size) is not int or hidden_size not in {32, 64}:
            raise ValueError("hidden_size must be 32 or 64")
        if type(num_layers) is not int or num_layers not in {1, 2}:
            raise ValueError("num_layers must be 1 or 2")
        if type(output_size) is not int or output_size != 3:
            raise ValueError("output_size must be 3 for horizons 5, 63 and 252")
        self.architecture = architecture.upper()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        recurrent_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[self.architecture]
        options = {"input_size": input_size, "hidden_size": hidden_size,
                   "num_layers": num_layers, "batch_first": True,
                   "dropout": 0.0, "bidirectional": False}
        if self.architecture == "RNN":
            options["nonlinearity"] = "tanh"
        self.recurrent = recurrent_class(**options)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
            raise ValueError("inputs must be a tensor shaped (batch, lookback, features)")
        if inputs.shape[0] < 1 or inputs.shape[1] < 1 or inputs.shape[2] != self.input_size:
            raise ValueError("inputs have an empty batch/window or an incorrect feature count")
        if not inputs.is_floating_point():
            raise ValueError("inputs must use a floating-point dtype")
        sequence, _ = self.recurrent(inputs)
        return self.head(sequence[:, -1, :])

    def config(self) -> dict:
        return {"architecture": self.architecture, "input_size": self.input_size,
                "hidden_size": self.hidden_size, "num_layers": self.num_layers,
                "output_size": self.output_size}


def build_model(config: Mapping) -> RecurrentForecaster:
    """Reconstruct the architecture separately from its trusted state_dict."""
    return RecurrentForecaster(**dict(config))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initialize_persistence_head(
    model: RecurrentForecaster, target_mean, target_scale
) -> None:
    """Start a residual forecast at zero change in the original price units.

    The existing target is ``z = (future_price - reference - mean) / scale``.
    Therefore persistence requires ``z = -mean / scale``, rather than zero.
    Only the output head changes; recurrent weights, model configuration, state
    keys and parameter count remain intact. Statistics must come exclusively
    from the inner-training target transformer, never from validation labels.
    """
    if not isinstance(model, RecurrentForecaster):
        raise ValueError("Persistence initialization requires a RecurrentForecaster")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError("Persistence initialization requires a CPU float32 model")
    try:
        mean = torch.as_tensor(target_mean, dtype=torch.float64, device="cpu")
        scale = torch.as_tensor(target_scale, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Target statistics must be finite numeric vectors") from error
    if (mean.shape != (model.output_size,) or scale.shape != (model.output_size,)
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
            or not (scale > 0).all()):
        raise ValueError("Target mean/scale need three finite values and strictly positive scales")
    bias = (-mean / scale).to(dtype=torch.float32)
    if not torch.isfinite(bias).all():
        raise ValueError("Persistence bias must be representable in float32")
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(bias)
