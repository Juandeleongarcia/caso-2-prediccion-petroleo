"""Prepare supervised windows, per-cutoff transforms and persistence scores offline."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from oil_forecast.acquisition import sha256
from oil_forecast.evaluation import persistence_predictions, score_predictions
from oil_forecast.preprocessing import InputTransformer, TargetTransformer
from oil_forecast.supervised import build_supervised
from oil_forecast.temporal import evaluation_schedule
from oil_forecast.validation import build_folds


def load_prepared(directory: Path, config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Configuration differs from the prepared snapshot; investigate before evaluation")
    for name, digest in manifest["output_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Prepared snapshot hash mismatch: {name}")
    raw_path = directory.parents[1] / "raw" / directory.name / "observations.csv"
    if sha256(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Unfilled snapshot hash mismatch")
    return pd.read_csv(directory / "features.csv"), pd.read_csv(directory / "target.csv"), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--snapshot", help="Existing prepared snapshot identifier; no acquisition")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    snapshot = args.snapshot or config["snapshot_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", snapshot):
        raise ValueError("Invalid snapshot identifier")
    if config["lookback_business_days"] != 60 or config["horizons_business_days"] != [5, 63, 252]:
        raise ValueError("The frozen protocol uses lookback 60 and horizons 5, 63, 252")
    prepared = ROOT / "data/processed" / snapshot
    raw_directory = ROOT / "data/raw" / snapshot
    protected = [ROOT / "data/petroleo_dataset.csv", ROOT / "scripts/download_data.py",
                 ROOT / "docs/assignment/ENUNCIADO.pdf", ROOT / "docs/README_profesor.md",
                 ROOT / "pyproject.toml", ROOT / "poetry.lock", args.config,
                 *prepared.glob("*"), *raw_directory.glob("*")]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected if path.is_file()}
    original = ROOT / "data/petroleo_dataset.csv"
    if original.exists() and sha256(original) != config["original_sha256"]:
        raise ValueError("The professor's original CSV differs from the audited hash")
    try:
        features, targets, source_manifest = load_prepared(prepared, args.config)
        feature_names = tuple(name for name, spec in config["sources"].items()
                              if spec["enabled"] and name in source_manifest["selected_features"])
        dataset = build_supervised(features, targets, feature_names,
                                   config["lookback_business_days"], config["horizons_business_days"])
        folds = build_folds(dataset, config)
        output = ROOT / "data/supervised" / snapshot
        output.mkdir(parents=True, exist_ok=True)
        arrays = {"X": dataset.X, "y_changes": dataset.y_changes,
                  "reference_brent": dataset.samples["reference_brent"].to_numpy(),
                  "window_rows": dataset.window_rows,
                  "feature_values": dataset.feature_values.to_numpy(),
                  "feature_names": np.asarray(feature_names),
                  "horizons": np.asarray(dataset.horizons),
                  "row_decision_dates": pd.to_datetime(features["decision_date"]).to_numpy(dtype="datetime64[ns]")}
        transformers, predictions = {}, []
        for fold in folds:
            train, evaluate = fold.train_indices, fold.eval_indices
            input_transformer = InputTransformer().fit(
                dataset.feature_values.iloc[fold.fit_row_indices], feature_names)
            target_transformer = TargetTransformer().fit(dataset.y_changes[train])
            # Exercise the real transformations, including all missing masks.
            for indices in [train, evaluate]:
                transformed = input_transformer.transform(dataset.X[indices])
                if transformed.shape != (len(indices), 60, 2 * len(feature_names)) or not np.isfinite(transformed).all():
                    raise ValueError(f"Invalid transformed inputs: {fold.block_id}")
                restored = target_transformer.inverse_transform(target_transformer.transform(dataset.y_changes[indices]))
                if not np.allclose(restored, dataset.y_changes[indices], atol=1e-12, rtol=1e-12):
                    raise ValueError("Target scaling changes original-unit values")
                del transformed
            state = {"input": input_transformer.to_dict(), "target": target_transformer.to_dict(),
                     "training_cutoff": fold.training_cutoff.isoformat(),
                     "fit_rows_rule": "distinct temporal rows used by mature training windows only"}
            transformers[fold.block_id] = state
            arrays[f"train_{fold.block_id}"] = train
            arrays[f"eval_{fold.block_id}"] = evaluate
            arrays[f"fit_rows_{fold.block_id}"] = fold.fit_row_indices
            sample = dataset.samples.iloc[evaluate]
            forecast = persistence_predictions(sample["reference_brent"], dataset.horizons)
            common = ["sample_id", "origin_date", "decision_at", "window_start_date",
                      "reference_brent", "reference_observation_date", "reference_available_at"]
            for column, horizon in enumerate(dataset.horizons):
                part = sample[common].copy()
                part["phase"], part["block_id"] = fold.phase, fold.block_id
                part["refit_at"], part["training_cutoff"] = fold.refit_at, fold.training_cutoff
                part["horizon"] = horizon
                part["target_date"] = sample[f"target_date_{horizon}"]
                part["label_available_at"] = sample[f"label_available_at_{horizon}"]
                part["availability_basis"] = sample[f"availability_basis_{horizon}"]
                part["y_true"] = sample[f"y_price_{horizon}"]
                part["y_pred"] = forecast[:, column]
                part["error"] = part["y_pred"] - part["y_true"]
                predictions.append(part)
        predictions = pd.concat(predictions, ignore_index=True).sort_values(["origin_date", "horizon"])
        if predictions.duplicated(["origin_date", "horizon"]).any():
            raise ValueError("An origin/horizon was evaluated more than once")
        metrics = score_predictions(predictions)
        totals = score_predictions(predictions, group_by=("phase", "horizon"))
        totals["block_id"] = "all"
        metrics = pd.concat([metrics, totals], ignore_index=True)
        if metrics["n_dropped"].any():
            raise ValueError("Complete-case evaluation unexpectedly dropped price pairs")
        np.savez_compressed(output / "dataset.npz", **arrays)
        frames = {"samples.csv": dataset.samples, "exclusions.csv": dataset.exclusions,
                  "folds.csv": pd.DataFrame([fold.summary for fold in folds]),
                  "baseline_predictions.csv": predictions, "baseline_metrics.csv": metrics}
        for name, frame in frames.items():
            frame.to_csv(output / name, index=False, lineterminator="\n")
        (output / "transformers.json").write_text(json.dumps(transformers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        phases = {phase: int(group[["origin_date"]].drop_duplicates().shape[0])
                  for phase, group in predictions.groupby("phase")}
        manifest = {"snapshot_id": snapshot, "config": config, "config_sha256": sha256(args.config),
                    "input_sha256": before,
                    "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                    [Path(__file__).resolve(), *sorted((ROOT / "src/oil_forecast").glob("*.py"))]},
                    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "dataset": dataset.summary, "raw_X_shape": list(dataset.X.shape),
                    "model_X_features": list(next(iter(transformers.values()))["input"]["output_feature_names"]),
                    "eligible_origins_by_phase": phases,
                    "pending_sources": [name for name, spec in source_manifest["source_coverage"].items() if spec["status"] != "ok"],
                    "baseline": {"name": "last_available_brent_persistence", "trainable_parameters": 0,
                                 "training_seconds": None, "mae_rmse_unit": "USD/barrel", "mse_unit": "USD^2/barrel^2"},
                    "historical_point_in_time": False,
                    "checks": {"all_training_labels_mature_at_each_cutoff": True,
                               "preprocessors_fit_on_unique_training_rows": True,
                               "transformed_train_and_eval_inputs_finite": True,
                               "target_transform_roundtrip_original_units": True,
                               "same_complete_cases_for_all_horizons": True},
                    "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())
                                      if path.is_file() and path.name != "manifest.json"}}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output.relative_to(ROOT)), "dataset": dataset.summary,
                          "eligible_origins_by_phase": phases}, ensure_ascii=False, indent=2))
        print(totals[["phase", "horizon", "n", "mse", "mae", "rmse"]].to_string(index=False))
    finally:
        after = {str(path.relative_to(ROOT)): sha256(path) for path in protected if path.is_file()}
        if before != after:
            raise RuntimeError("An original or input snapshot changed during supervised preparation")


if __name__ == "__main__":
    main()
