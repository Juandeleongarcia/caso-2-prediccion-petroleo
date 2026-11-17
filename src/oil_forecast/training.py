"""Short, deterministic CPU checks; validation never updates model weights."""

from __future__ import annotations

import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_reproducible(seed: int, threads: int = 1) -> None:
    """Set this before constructing a model to also reproduce its initial weights.

    Determinism is scoped to the same CPU, PyTorch version and configuration;
    seeds do not promise identical results across hardware or library versions.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _data_tensors(inputs, targets, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device="cpu")
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device="cpu")
    if x.ndim != 3 or y.ndim != 2 or y.shape != (len(x), 3):
        raise ValueError(f"{name} requires inputs (N, L, F) and targets (N, 3)")
    if min(x.shape) < 1 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError(f"{name} requires nonempty finite transformed inputs and targets")
    return x.contiguous(), y.contiguous()


def _forward(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    prediction = model(inputs)
    if prediction.shape != (len(inputs), 3) or not torch.isfinite(prediction).all():
        raise ValueError("Model must return finite predictions shaped (N, 3)")
    return prediction


def _diagnostic(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int
) -> tuple[float, np.ndarray]:
    """MSE over individual components, not a mean of unequal batch means."""
    model.eval()
    squared_error = 0.0
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            output = _forward(model, inputs[start:start + batch_size])
            squared_error += torch.sum((output - targets[start:start + batch_size]).square()).item()
            predictions.append(output.cpu().numpy().copy())
    loss = squared_error / targets.numel()
    if not np.isfinite(loss):
        raise ValueError("Non-finite diagnostic loss")
    return loss, np.concatenate(predictions, axis=0)


def train_short(
    model: nn.Module,
    X_train,
    y_train,
    X_valid,
    y_valid,
    config: dict,
) -> dict:
    """Run one to three fixed epochs and retain the final, predetermined epoch.

    Targets must already be standardized changes using the training cut's
    transformer. Returned predictions use that scale; callers must invert it
    before scoring prices. ``training_seconds`` sums optimization loops, including
    loading batches and gradient checks, excluding diagnostic evaluation.
    """
    epochs = config.get("epochs", 3)
    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0)
    gradient_clip = config.get("gradient_clip", 1.0)
    shuffle = config.get("shuffle_train", True)
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("This short experiment permits only one to three epochs")
    if type(batch_size) is not int or batch_size < 1 or type(shuffle) is not bool:
        raise ValueError("batch_size must be positive and shuffle_train boolean")
    if str(config.get("optimizer", "Adam")).lower() != "adam":
        raise ValueError("The common optimizer for this experiment is Adam")
    if not np.isfinite([learning_rate, weight_decay, gradient_clip]).all():
        raise ValueError("Optimizer parameters must be finite")
    if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
        raise ValueError("Invalid optimizer parameters")
    parameters = tuple(model.parameters())
    if not parameters or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in parameters):
        raise ValueError("Short experiments require a CPU float32 model")
    train_x, train_y = _data_tensors(X_train, y_train, "Train")
    valid_x, valid_y = _data_tensors(X_valid, y_valid, "Validation")
    if train_x.shape[1:] != valid_x.shape[1:]:
        raise ValueError("Training and validation window shapes differ")
    seed = config.get("seed", 42)
    set_reproducible(seed, config.get("threads", 1))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                        shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    objective = nn.MSELoss(reduction="mean")
    initial_train, _ = _diagnostic(model, train_x, train_y, batch_size)
    initial_valid, predictions = _diagnostic(model, valid_x, valid_y, batch_size)
    history = [{"epoch": 0, "train_loss": initial_train,
                "validation_loss": initial_valid, "optimization_seconds": 0.0}]
    training_seconds = 0.0
    gradient_steps = nonzero_gradient_steps = 0
    max_gradient_norm = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        started = perf_counter()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(_forward(model, inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("Missing or non-finite model gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
            norm_value = float(norm.item())
            max_gradient_norm = max(max_gradient_norm, norm_value)
            gradient_steps += 1
            nonzero_gradient_steps += int(norm_value > 0)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("Non-finite model parameter after optimizer step")
        seconds = perf_counter() - started
        training_seconds += seconds
        train_loss, _ = _diagnostic(model, train_x, train_y, batch_size)
        validation_loss, predictions = _diagnostic(model, valid_x, valid_y, batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "optimization_seconds": seconds})
    return {"history": history, "training_seconds": training_seconds,
            "predictions": predictions, "initial_train_loss": initial_train,
            "final_train_loss": history[-1]["train_loss"],
            "initial_validation_loss": initial_valid,
            "final_validation_loss": history[-1]["validation_loss"],
            "loss_decreased": history[-1]["train_loss"] < initial_train,
            "gradient_steps": gradient_steps, "nonzero_gradient_steps": nonzero_gradient_steps,
            "max_gradient_norm": max_gradient_norm}


def predict_scaled(model: nn.Module, inputs, batch_size: int = 64) -> np.ndarray:
    """Predict scaled changes on CPU without labels or updates to model weights."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    values = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device="cpu")
    if values.ndim != 3 or min(values.shape) < 1 or not torch.isfinite(values).all():
        raise ValueError("Prediction inputs must be nonempty finite windows (N, L, F)")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError("Prediction requires a CPU float32 model")
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            _forward(model, values[start:start + batch_size]).numpy().copy()
            for start in range(0, len(values), batch_size)
        ])


def train_with_early_stopping(
    model: nn.Module,
    X_train,
    y_train,
    X_inner_valid,
    y_inner_valid,
    config: dict,
) -> dict:
    """Restore the minimum inner-validation checkpoint from completed epochs.

    No outer evaluation inputs are accepted. Epoch zero is diagnostic only.
    ``min_delta`` controls patience against the last significant improvement;
    it never prevents retaining a smaller numerical validation loss. Statistics
    must already have been fitted exclusively on the inner training subset.
    An optional SmoothL1 training objective leaves checkpoint selection and
    ``train_loss``/``validation_loss`` on the common normalized MSE scale.
    """
    elapsed_started = perf_counter()
    max_epochs = config.get("max_epochs", 20)
    patience = config.get("patience", 5)
    min_delta = config.get("min_delta", 0.0001)
    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    gradient_clip = config.get("gradient_clip", 1.0)
    shuffle = config.get("shuffle_train", True)
    loss_function = config.get("loss_function", "mse")
    huber_beta = config.get("huber_beta", 1.0)
    if loss_function not in ("mse", "smooth_l1"):
        raise ValueError("loss_function must be mse or smooth_l1")
    if loss_function == "smooth_l1" and (
            type(huber_beta) not in (int, float) or not np.isfinite(huber_beta) or huber_beta <= 0):
        raise ValueError("SmoothL1 huber_beta must be finite and strictly positive")
    if type(max_epochs) is not int or not 1 <= max_epochs <= 30:
        raise ValueError("max_epochs must be an integer between 1 and 30")
    if type(patience) is not int or patience < 1:
        raise ValueError("patience must be a positive integer")
    if type(batch_size) is not int or batch_size < 1 or type(shuffle) is not bool:
        raise ValueError("batch_size must be positive and shuffle_train boolean")
    if str(config.get("optimizer", "Adam")).lower() != "adam":
        raise ValueError("The common optimizer for this experiment is Adam")
    if not np.isfinite([learning_rate, weight_decay, gradient_clip, min_delta]).all():
        raise ValueError("Optimizer and early-stopping parameters must be finite")
    if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0 or min_delta < 0:
        raise ValueError("Invalid optimizer or early-stopping parameters")
    parameters = tuple(model.parameters())
    if not parameters or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in parameters):
        raise ValueError("Experiments require a CPU float32 model")
    train_x, train_y = _data_tensors(X_train, y_train, "Inner train")
    valid_x, valid_y = _data_tensors(X_inner_valid, y_inner_valid, "Inner validation")
    if train_x.shape[1:] != valid_x.shape[1:]:
        raise ValueError("Inner training and validation window shapes differ")
    seed = config.get("seed", 42)
    set_reproducible(seed, config.get("threads", 1))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                        shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    objective = (nn.MSELoss(reduction="mean") if loss_function == "mse"
                 else nn.SmoothL1Loss(beta=float(huber_beta), reduction="mean"))
    initial_train, train_predictions = _diagnostic(model, train_x, train_y, batch_size)
    initial_valid, _ = _diagnostic(model, valid_x, valid_y, batch_size)
    history = [{"epoch": 0, "train_loss": initial_train,
                "validation_loss": initial_valid, "optimization_seconds": 0.0}]
    history[0]["train_objective_loss"] = (
        initial_train if loss_function == "mse"
        else float(objective(torch.from_numpy(train_predictions), train_y).item()))
    best_loss = patience_reference = float("inf")
    best_epoch = bad_epochs = 0
    best_state = None
    training_seconds = 0.0
    gradient_steps = nonzero_gradient_steps = 0
    max_gradient_norm = 0.0
    stopping_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        started = perf_counter()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(_forward(model, inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("Missing or non-finite model gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
            norm_value = float(norm.item())
            max_gradient_norm = max(max_gradient_norm, norm_value)
            gradient_steps += 1
            nonzero_gradient_steps += int(norm_value > 0)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("Non-finite model parameter after optimizer step")
        seconds = perf_counter() - started
        training_seconds += seconds
        train_loss, train_predictions = _diagnostic(model, train_x, train_y, batch_size)
        validation_loss, _ = _diagnostic(model, valid_x, valid_y, batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "optimization_seconds": seconds})
        history[-1]["train_objective_loss"] = (
            train_loss if loss_function == "mse"
            else float(objective(torch.from_numpy(train_predictions), train_y).item()))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if validation_loss < patience_reference and patience_reference - validation_loss >= min_delta:
            patience_reference, bad_epochs = validation_loss, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            stopping_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("No completed epoch produced a valid inner checkpoint")
    model.load_state_dict(best_state)
    restored_train_loss, train_predictions = _diagnostic(model, train_x, train_y, batch_size)
    restored_valid_loss, valid_predictions = _diagnostic(model, valid_x, valid_y, batch_size)
    train_components = np.mean((train_predictions.astype(np.float64) - train_y.numpy()) ** 2, axis=0)
    valid_components = np.mean((valid_predictions.astype(np.float64) - valid_y.numpy()) ** 2, axis=0)
    if not np.isfinite(np.r_[train_components, valid_components]).all():
        raise ValueError("Non-finite loss component after restoring the checkpoint")
    best_row = history[best_epoch]
    return {
        "history": history, "best_epoch": best_epoch, "last_epoch": history[-1]["epoch"],
        "stopping_reason": stopping_reason, "reason": stopping_reason,
        "train_loss_at_best": best_row["train_loss"], "validation_loss_at_best": best_row["validation_loss"],
        "last_train_loss": history[-1]["train_loss"], "last_validation_loss": history[-1]["validation_loss"],
        "restored_train_loss": restored_train_loss, "restored_validation_loss": restored_valid_loss,
        "initial_train_loss": initial_train, "initial_validation_loss": initial_valid,
        "training_seconds": training_seconds, "elapsed_seconds": perf_counter() - elapsed_started,
        "predictions": valid_predictions, "inner_predictions": valid_predictions,
        "train_loss_components": train_components.tolist(), "validation_loss_components": valid_components.tolist(),
        "gradient_steps": gradient_steps, "nonzero_gradient_steps": nonzero_gradient_steps,
        "max_gradient_norm": max_gradient_norm,
        "state_dict": best_state,
        "model_config": model.config() if callable(getattr(model, "config", None)) else None,
        "training_config": dict(config),
        "loss_function": loss_function, "objective_name": loss_function,
        "huber_beta": float(huber_beta) if loss_function == "smooth_l1" else None,
        "validation_metric": "mse",
    }


def train_fixed_epochs(model: nn.Module, X_train, y_train, config: dict) -> dict:
    """Fit exactly the epoch count frozen in development, without held-out data.

    This final refit accepts no validation/calibration/test arrays and performs
    no model selection. It retains the last predetermined epoch. Targets and
    inputs must already be transformed using this refit's permitted history.
    CPU optimizer timing excludes diagnostic passes; elapsed time includes them.
    """
    elapsed_started = perf_counter()
    epochs = config.get("epochs")
    batch_size = config.get("batch_size", 64)
    learning_rate = config.get("learning_rate", 0.0003)
    weight_decay = config.get("weight_decay", 0.0001)
    gradient_clip = config.get("gradient_clip", 1.0)
    shuffle = config.get("shuffle_train", True)
    loss_function = config.get("loss_function", "mse")
    huber_beta = config.get("huber_beta", 1.0)
    if type(epochs) is not int or not 1 <= epochs <= 30:
        raise ValueError("Fixed epochs must be explicitly frozen as an integer between 1 and 30")
    if type(batch_size) is not int or batch_size < 1 or type(shuffle) is not bool:
        raise ValueError("batch_size must be positive and shuffle_train boolean")
    if str(config.get("optimizer", "Adam")).lower() != "adam":
        raise ValueError("The frozen common optimizer is Adam")
    if not np.isfinite([learning_rate, weight_decay, gradient_clip]).all():
        raise ValueError("Optimizer parameters must be finite")
    if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
        raise ValueError("Invalid optimizer parameters")
    if loss_function not in ("mse", "smooth_l1"):
        raise ValueError("loss_function must be mse or smooth_l1")
    if loss_function == "smooth_l1" and (
            type(huber_beta) not in (int, float) or not np.isfinite(huber_beta) or huber_beta <= 0):
        raise ValueError("SmoothL1 huber_beta must be finite and strictly positive")
    parameters = tuple(model.parameters())
    if not parameters or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                             for parameter in parameters):
        raise ValueError("Fixed refits require a CPU float32 model")
    train_x, train_y = _data_tensors(X_train, y_train, "Train")
    seed = config.get("seed", 42)
    set_reproducible(seed, config.get("threads", 1))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                        shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    objective = (nn.MSELoss(reduction="mean") if loss_function == "mse"
                 else nn.SmoothL1Loss(beta=float(huber_beta), reduction="mean"))
    initial_train, predicted = _diagnostic(model, train_x, train_y, batch_size)
    initial_objective = (initial_train if loss_function == "mse"
                         else float(objective(torch.from_numpy(predicted), train_y).item()))
    history = [{"epoch": 0, "train_loss": initial_train, "train_objective_loss": initial_objective,
                "optimization_seconds": 0.0}]
    training_seconds = 0.0
    gradient_steps = nonzero_gradient_steps = 0
    max_gradient_norm = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        started = perf_counter()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(_forward(model, inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite fixed-training loss")
            loss.backward()
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("Missing or non-finite model gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
            norm_value = float(norm.item())
            max_gradient_norm = max(max_gradient_norm, norm_value)
            gradient_steps += 1
            nonzero_gradient_steps += int(norm_value > 0)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("Non-finite model parameter after optimizer step")
        seconds = perf_counter() - started
        training_seconds += seconds
        train_loss, predicted = _diagnostic(model, train_x, train_y, batch_size)
        objective_loss = (train_loss if loss_function == "mse"
                          else float(objective(torch.from_numpy(predicted), train_y).item()))
        if not np.isfinite(objective_loss):
            raise ValueError("Non-finite fixed-training objective diagnostic")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_objective_loss": objective_loss, "optimization_seconds": seconds})
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {"history": history, "selected_epoch": epochs, "last_epoch": epochs,
            "stopping_reason": "fixed_development_epochs", "training_seconds": training_seconds,
            "elapsed_seconds": perf_counter() - elapsed_started,
            "initial_train_loss": initial_train, "final_train_loss": history[-1]["train_loss"],
            "gradient_steps": gradient_steps, "nonzero_gradient_steps": nonzero_gradient_steps,
            "max_gradient_norm": max_gradient_norm, "state_dict": state,
            "model_config": model.config() if callable(getattr(model, "config", None)) else None,
            "training_config": dict(config), "loss_function": loss_function,
            "huber_beta": float(huber_beta) if loss_function == "smooth_l1" else None}
