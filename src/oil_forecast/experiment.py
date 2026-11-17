"""Verified temporal folds with separate development and final entry points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from .acquisition import sha256
from .preprocessing import InputTransformer, TargetTransformer
from .supervised import SupervisedDataset
from .validation import TemporalFold, build_folds
from .validation import build_inner_split


@dataclass
class DevelopmentData:
    dataset: SupervisedDataset
    folds: list[TemporalFold]
    transformers: dict
    baseline: pd.DataFrame
    manifest: dict
    environment_changes: dict


def require_development(phase: str) -> None:
    if phase != "development":
        raise ValueError("Short experiments permit development only; no calibration or test")


def _relative_input_path(relative: str) -> PurePosixPath:
    """Interpret historical Windows separators without rewriting a manifest."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("Manifest input paths must be nonempty relative paths")
    normalized = relative.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not path.parts or path.is_absolute() or PureWindowsPath(normalized).drive
            or ".." in path.parts):
        raise ValueError("Manifest input paths cannot be absolute or escape the project")
    return path


def load_development_data(directory: Path, config: dict) -> DevelopmentData:
    """Expose development only; historical callers cannot request final phases."""
    return _load_verified_data(directory, config, ("development",))


def load_final_evaluation_data(directory: Path, config: dict) -> DevelopmentData:
    """Expose the frozen calibration/test folds, without changing old artifacts."""
    return _load_verified_data(directory, config, ("calibration", "test"))


def _load_verified_data(directory: Path, config: dict, phases: tuple[str, ...]) -> DevelopmentData:
    """Keep historical provenance intact when the explicitly requested env changes."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if config != manifest["config"]:
        raise ValueError("Data/evaluation configuration differs from the frozen supervised snapshot")
    root = directory.parents[2]
    changes = {}
    for relative, expected in manifest["input_sha256"].items():
        input_path = _relative_input_path(relative)
        actual = sha256(root.joinpath(*input_path.parts))
        if actual != expected:
            if input_path.as_posix() not in {"pyproject.toml", "poetry.lock"}:
                raise ValueError(f"Input hash mismatch: {relative}")
            changes[relative] = {"historical_sha256": expected, "current_sha256": actual,
                                 "reason": "Authorized project dependency additions; historical environment hashes remain recorded"}
    for name, expected in manifest["output_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Supervised artifact hash mismatch: {name}")
    with np.load(directory / "dataset.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    samples = pd.read_csv(directory / "samples.csv")
    for name in ["origin_date", "window_start_date", "reference_observation_date",
                 *[f"target_date_{h}" for h in [5, 63, 252]]]:
        samples[name] = pd.to_datetime(samples[name], errors="raise")
    for name in ["decision_at", "reference_available_at", *[f"label_available_at_{h}" for h in [5, 63, 252]]]:
        samples[name] = pd.to_datetime(samples[name], utc=True, errors="raise")
    names = tuple(str(name) for name in arrays["feature_names"])
    horizons = tuple(int(h) for h in arrays["horizons"])
    if horizons != (5, 63, 252) or arrays["X"].shape != (len(samples), 60, len(names)):
        raise ValueError("Supervised shape or horizon mismatch")
    if not np.array_equal(samples["sample_id"], np.arange(len(samples))):
        raise ValueError("Unstable supervised sample identifiers")
    if not np.allclose(arrays["y_changes"], samples[[f"y_change_{h}" for h in horizons]], atol=1e-12, rtol=1e-12):
        raise ValueError("Supervised targets differ between exports")
    dataset = SupervisedDataset(arrays["X"], arrays["y_changes"], samples, pd.DataFrame(),
                                pd.DataFrame(arrays["feature_values"], columns=names),
                                arrays["window_rows"], names, horizons, manifest["dataset"])
    # Reuse the single frozen protocol, then expose only its development cuts.
    folds = [fold for fold in build_folds(dataset, config) if fold.phase in phases]
    transforms = json.loads((directory / "transformers.json").read_text(encoding="utf-8"))
    for fold in folds:
        for prefix, expected in [("train", fold.train_indices), ("eval", fold.eval_indices),
                                 ("fit_rows", fold.fit_row_indices)]:
            if not np.array_equal(arrays[f"{prefix}_{fold.block_id}"], expected):
                raise ValueError(f"Stored indices differ from mature-label protocol: {fold.block_id}")
    baseline = pd.read_csv(directory / "baseline_predictions.csv")
    baseline = baseline.loc[baseline["phase"].isin(phases)].copy()
    for name in ["origin_date", "target_date", "window_start_date", "reference_observation_date"]:
        baseline[name] = pd.to_datetime(baseline[name], errors="raise")
    for name in ["decision_at", "reference_available_at", "label_available_at", "training_cutoff", "refit_at"]:
        baseline[name] = pd.to_datetime(baseline[name], utc=True, errors="raise")
    data = DevelopmentData(dataset, folds, {fold.block_id: transforms[fold.block_id] for fold in folds},
                           baseline, manifest, changes)
    for fold in folds:
        _verified_pairs(data, fold.block_id)
    return data


def baseline_pairs(data: DevelopmentData, block_id: str) -> pd.DataFrame:
    """Keep the original development API restricted to development."""
    matches = [fold for fold in data.folds if fold.block_id == block_id]
    if len(matches) != 1:
        raise ValueError("Unknown or non-development block")
    require_development(matches[0].phase)
    return _verified_pairs(data, block_id)


def _verified_pairs(data: DevelopmentData, block_id: str) -> pd.DataFrame:
    """Require exact sample/horizon/target keys and original-unit truth/reference."""
    matches = [fold for fold in data.folds if fold.block_id == block_id]
    if len(matches) != 1:
        raise ValueError("Unknown or non-development block")
    fold = matches[0]
    sample = data.dataset.samples.iloc[fold.eval_indices].set_index("sample_id")
    pairs = data.baseline.loc[data.baseline["block_id"].eq(block_id)].copy()
    if pairs.duplicated(["sample_id", "horizon"]).any() or len(pairs) != len(sample) * 3:
        raise ValueError("Baseline pair identifiers are missing or duplicated")
    if set(pairs["sample_id"]) != set(sample.index) or set(pairs["horizon"]) != {5, 63, 252}:
        raise ValueError("Baseline origin/horizon identifiers differ")
    for horizon in data.dataset.horizons:
        rows = pairs.loc[pairs["horizon"].eq(horizon)].set_index("sample_id").sort_index()
        expected = sample.sort_index()
        if not rows.index.equals(expected.index):
            raise ValueError("Baseline sample/horizon keys differ")
        for column in ["origin_date", "decision_at", "reference_observation_date", "reference_available_at"]:
            if not rows[column].equals(expected[column]):
                raise ValueError(f"Baseline metadata differs: {column}")
        if not rows["target_date"].equals(expected[f"target_date_{horizon}"]):
            raise ValueError("Baseline target dates differ")
        if not rows["label_available_at"].equals(expected[f"label_available_at_{horizon}"]):
            raise ValueError("Baseline label availability differs")
        for actual, truth in [(rows["y_true"], expected[f"y_price_{horizon}"]),
                              (rows["reference_brent"], expected["reference_brent"])]:
            if not np.allclose(actual, truth, atol=1e-12, rtol=1e-12):
                raise ValueError("Baseline truth/reference differs from supervised prices")
    return pairs.sort_values(["sample_id", "horizon"]).reset_index(drop=True)


def prepare_final_fold(data: DevelopmentData, block_id: str) -> dict:
    """Refit transforms on unique rows of mature history; return no eval labels.

    Final epochs are fixed from development before this function is used. Known
    historical prices can recur across origins in the predeclared online policy;
    no sample whose origin belongs to test is allowed into gradient training.
    """
    matches = [fold for fold in data.folds if fold.block_id == block_id]
    if len(matches) != 1 or matches[0].phase not in {"calibration", "test"}:
        raise ValueError("A final fold must be calibration or test")
    fold = matches[0]
    dataset = data.dataset
    train = dataset.samples.iloc[fold.train_indices]
    if (train["origin_date"].dt.year.eq(2024).any()
            or np.intersect1d(fold.train_indices, fold.eval_indices).size):
        raise ValueError("Test-origin samples cannot train a final model")
    for horizon in dataset.horizons:
        if not train[f"label_available_at_{horizon}"].le(fold.training_cutoff).all():
            raise ValueError("Immature final training label")
    rows = np.unique(dataset.window_rows[fold.train_indices])
    if not np.array_equal(rows, fold.fit_row_indices):
        raise ValueError("Final preprocessing rows differ from mature training windows")
    inputs = InputTransformer().fit(dataset.feature_values.iloc[rows], dataset.feature_names)
    targets = TargetTransformer().fit(dataset.y_changes[fold.train_indices])
    saved = data.transformers[block_id]
    if saved["input"] != inputs.to_dict() or saved["target"] != targets.to_dict():
        raise ValueError("Final transformers differ from the frozen full-history protocol")
    return {"fold": fold, "input_transformer": inputs, "target_transformer": targets,
            "X_train": inputs.transform(dataset.X[fold.train_indices]).astype(np.float32),
            "y_train": targets.transform(dataset.y_changes[fold.train_indices]).astype(np.float32),
            "X_outer": inputs.transform(dataset.X[fold.eval_indices]).astype(np.float32),
            "pairs": _verified_pairs(data, block_id)}


def prepare_development_fold(data: DevelopmentData, block_id: str) -> dict:
    """Load each cut's existing train-only state; never refit on validation."""
    matches = [fold for fold in data.folds if fold.block_id == block_id]
    if len(matches) != 1:
        raise ValueError("Unknown or non-development block")
    fold = matches[0]
    require_development(fold.phase)
    state = data.transformers[block_id]
    input_transformer = InputTransformer.from_dict(state["input"])
    target_transformer = TargetTransformer.from_dict(state["target"])
    if input_transformer.n_fit_rows_ != len(fold.fit_row_indices) or target_transformer.n_fit_rows_ != len(fold.train_indices):
        raise ValueError("Transformer does not belong to this training cut")
    if pd.Timestamp(state["training_cutoff"]) != fold.training_cutoff:
        raise ValueError("Transformer training cutoff differs")
    dataset = data.dataset
    expected_input = InputTransformer().fit(dataset.feature_values.iloc[fold.fit_row_indices], dataset.feature_names)
    expected_target = TargetTransformer().fit(dataset.y_changes[fold.train_indices])
    if input_transformer.to_dict() != expected_input.to_dict() or target_transformer.to_dict() != expected_target.to_dict():
        raise ValueError("Saved transformer statistics differ from this training history")
    return {"fold": fold, "input_transformer": input_transformer, "target_transformer": target_transformer,
            "X_train": input_transformer.transform(dataset.X[fold.train_indices]).astype(np.float32),
            "y_train": target_transformer.transform(dataset.y_changes[fold.train_indices]).astype(np.float32),
            "X_valid": input_transformer.transform(dataset.X[fold.eval_indices]).astype(np.float32),
            "y_valid": target_transformer.transform(dataset.y_changes[fold.eval_indices]).astype(np.float32),
            "pairs": baseline_pairs(data, block_id)}


def prepare_inner_development_fold(data: DevelopmentData, block_id: str,
                                   validation_business_days: int = 252) -> dict:
    """Fit only inner train; outer inputs are returned solely for later inference."""
    matches = [fold for fold in data.folds if fold.block_id == block_id]
    if len(matches) != 1:
        raise ValueError("Unknown or non-development block")
    fold = matches[0]
    require_development(fold.phase)
    dataset = data.dataset
    inner = build_inner_split(dataset, fold, validation_business_days)
    inputs = InputTransformer().fit(dataset.feature_values.iloc[inner.fit_row_indices], dataset.feature_names)
    targets = TargetTransformer().fit(dataset.y_changes[inner.train_indices])
    return {"fold": fold, "inner_split": inner, "input_transformer": inputs,
            "target_transformer": targets,
            "X_train": inputs.transform(dataset.X[inner.train_indices]).astype(np.float32),
            "y_train": targets.transform(dataset.y_changes[inner.train_indices]).astype(np.float32),
            "X_inner_valid": inputs.transform(dataset.X[inner.validation_indices]).astype(np.float32),
            "y_inner_valid": targets.transform(dataset.y_changes[inner.validation_indices]).astype(np.float32),
            "X_outer": inputs.transform(dataset.X[fold.eval_indices]).astype(np.float32),
            "pairs": baseline_pairs(data, block_id)}
