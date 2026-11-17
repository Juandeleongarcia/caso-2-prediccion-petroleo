"""Frozen semiannual final evaluation and pre-test empirical uncertainty."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from run_optimization import prediction_frame, save_progress
from oil_forecast.acquisition import sha256
from oil_forecast.evaluation import freeze_final_choices, score_final_predictions
from oil_forecast.experiment import load_final_evaluation_data, prepare_final_fold
from oil_forecast.models import build_model, count_parameters, initialize_persistence_head
from oil_forecast.training import predict_scaled, set_reproducible, train_fixed_epochs
from oil_forecast.uncertainty import apply_intervals, fit_intervals, interval_metrics


def write_new_json(path: Path, value: dict) -> None:
    """Frozen protocol and quantiles can be verified on resume, never replaced."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen JSON changed: {path.name}")
        return
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final_evaluation.json")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = {"variants": {"RNN": "D", "LSTM": "B", "GRU": "C"},
              "epochs": {"RNN": 4, "LSTM": 3, "GRU": 5}, "selected_model": "LSTM", "selected_seed": 42,
              "seeds": [42, 123, 2026], "hidden_size": 32, "num_layers": 1, "output_size": 3,
              "batch_size": 64, "optimizer": "Adam", "learning_rate": 0.0003, "weight_decay": 0.0001,
              "gradient_clip": 1.0, "shuffle_train": True, "threads": 1, "device": "cpu",
              "interval_coverage": 0.8, "calibration_deadline": "2024-01-01T08:00:00+01:00"}
    if any(config.get(name) != value for name, value in frozen.items()):
        raise ValueError("The final protocol is frozen; no new final-data tuning is permitted")
    run_id = args.run_id or config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid final run identifier")
    output = ROOT / "experiments" / run_id
    if output.exists() and not args.resume:
        raise FileExistsError("Previous final evaluation cannot be overwritten")
    if args.resume and (output / "experiment.json").exists():
        raise FileExistsError("Completed experiments are immutable")
    previous = [ROOT / config["development_reference"], ROOT / config["optimization_reference"]]
    reports = [json.loads((path / "experiment.json").read_text(encoding="utf-8")) for path in previous]
    for path, report in zip(previous, reports):
        for name, digest in report["input_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Previous input changed: {name}")
        for name, digest in report["output_sha256"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"Previous result changed: {path.name}/{name}")
    development_metrics = pd.read_csv(previous[0] / "metrics.csv")
    development_runs = pd.DataFrame([run for report in reports for run in report["runs"]])
    choices = freeze_final_choices(development_metrics, development_runs)
    if any(choices[name] != config[name] for name in ["variants", "epochs", "selected_model", "selected_seed"]):
        raise ValueError("Final choices disagree with the selected development evidence")
    data_config = json.loads((ROOT / config["data_config"]).read_text(encoding="utf-8"))
    data_dir = ROOT / "data/supervised" / data_config["snapshot_id"]
    data = load_final_evaluation_data(data_dir, data_config)
    prepared = {fold.block_id: prepare_final_fold(data, fold.block_id) for fold in data.folds}
    calibration = data.baseline.loc[data.baseline["phase"].eq("calibration")]
    boundary_ids = sorted(int(value) for value in calibration.loc[
        ~calibration["label_available_at"].lt(pd.Timestamp(config["calibration_deadline"])), "sample_id"].unique())
    raw_manifest = json.loads((ROOT / "data/raw" / data_config["snapshot_id"] / "manifest.json").read_text(encoding="utf-8"))
    protocol = config | choices | {"schema_version": 1, "feature_names": list(data.dataset.feature_names),
        "lookback": 60, "horizons": [5, 63, 252], "decision_hour": 8,
        "data_snapshot": data_config["snapshot_id"], "data_updated_at": raw_manifest["downloaded_at"],
        "availability_mode": data_config["availability_mode"],
        "calibration_point_origins": int(calibration["sample_id"].nunique()),
        "interval_excluded_boundary_sample_ids": boundary_ids,
        "interval_calibration_origins": int(calibration[~calibration["sample_id"].isin(boundary_ids)]["sample_id"].nunique()),
        "test_origins": int(data.baseline.loc[data.baseline["phase"].eq("test"), "sample_id"].nunique()),
        "model_selection_uses_final_results": False,
        "online_policy_note": "Known historical prices may recur as labels of older origins at later scheduled refits; no test-origin sample trains. This is online evaluation, not a static holdout of every price observation."}
    protected = {ROOT / name for report in reports for name in report["input_sha256"]}
    protected |= {args.config.resolve(), *[path for path in data_dir.iterdir() if path.is_file()]}
    for directory in [*previous, ROOT / "experiments/cpu-smoke-3epochs-seed42"]:
        protected |= {path for path in directory.rglob("*") if path.is_file()}
    before = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(protected)}
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/run_optimization.py",
                  *sorted((ROOT / "src/oil_forecast").glob("*.py"))]
    code = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    if args.resume:
        bundle = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        if bundle["protocol"] != protocol or bundle["input_sha256"] != before or bundle["code_sha256"] != code:
            raise ValueError("Resume needs identical frozen protocol, code and inputs")
    else:
        output.mkdir(parents=True)
        bundle = {"schema_version": 1, "protocol": protocol, "input_sha256": before, "code_sha256": code,
                  "models": {}, "interval_state": None, "reproducibility": None, "finished": False}
        save_progress(bundle, output)
    write_new_json(output / "protocol.json", protocol)

    def fit(model_name: str, seed: int, block_id: str) -> tuple[object, dict]:
        block = prepared[block_id]
        variant = protocol["variants"][model_name]
        options = config | {"epochs": protocol["epochs"][model_name], "seed": seed,
                            "loss_function": "smooth_l1" if variant == "D" else "mse", "huber_beta": 1.0}
        set_reproducible(seed, 1)
        model = build_model({"architecture": model_name, "input_size": 20, "hidden_size": 32,
                             "num_layers": 1, "output_size": 3})
        if variant in {"C", "D"}:
            targets = block["target_transformer"]
            initialize_persistence_head(model, targets.mean_, targets.scale_)
        result = train_fixed_epochs(model, block["X_train"], block["y_train"], options)
        return model, result

    def trial(model_name: str, seed: int, block_id: str) -> None:
        key = f"{model_name}_{seed}_{block_id}"
        if key in bundle["models"]:
            return
        block = prepared[block_id]
        fold = block["fold"]
        if fold.phase == "test" and bundle["interval_state"] is None:
            raise ValueError("Intervals must be frozen before any test fit or prediction")
        model, result = fit(model_name, seed, block_id)
        prediction = predict_scaled(model, block["X_outer"])
        summary = {"phase": fold.phase, "model": model_name, "variant": protocol["variants"][model_name],
            "seed": seed, "block_id": block_id, "parameters": count_parameters(model),
            "epochs": result["selected_epoch"], "stopping_reason": result["stopping_reason"],
            "training_seconds": result["training_seconds"], "elapsed_seconds": result["elapsed_seconds"],
            "initial_train_loss": result["initial_train_loss"], "final_train_loss": result["final_train_loss"],
            "gradient_steps": result["gradient_steps"], "nonzero_gradient_steps": result["nonzero_gradient_steps"],
            "max_gradient_norm": result["max_gradient_norm"], "loss_function": result["loss_function"],
            "n_train": len(fold.train_indices), "n_eval": len(fold.eval_indices),
            "training_cutoff": fold.training_cutoff.isoformat(), "refit_at": fold.refit_at.isoformat(),
            "latest_train_label_available_at": fold.summary["latest_train_label_available_at"]}
        bundle["models"][key] = summary | {"summary": summary, "model_config": model.config(),
            "training_config": result["training_config"], "state_dict": result["state_dict"],
            "history": result["history"], "outer_predictions_scaled": torch.from_numpy(prediction.copy()),
            "transformers": {"input": block["input_transformer"].to_dict(), "target": block["target_transformer"].to_dict()},
            "train_indices": torch.from_numpy(fold.train_indices.copy()),
            "fit_row_indices": torch.from_numpy(fold.fit_row_indices.copy())}
        save_progress(bundle, output)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    def predictions_for(phase: str) -> pd.DataFrame:
        scoped = replace(data, folds=[fold for fold in data.folds if fold.phase == phase],
                         baseline=data.baseline.loc[data.baseline["phase"].eq(phase)])
        models = {key: entry for key, entry in bundle["models"].items() if entry["phase"] == phase}
        return prediction_frame({"models": models}, scoped, prepared)

    try:
        for fold in data.folds:
            if fold.phase == "calibration":
                for model_name in protocol["variants"]:
                    for seed in protocol["seeds"]:
                        trial(model_name, seed, fold.block_id)
        calibration_predictions = predictions_for("calibration")
        calibration_fit = calibration_predictions.loc[~calibration_predictions["sample_id"].isin(boundary_ids)]
        state = fit_intervals(calibration_fit, protocol["interval_coverage"], protocol["calibration_deadline"])
        if bundle["interval_state"] is not None and bundle["interval_state"] != state:
            raise ValueError("Frozen calibration quantiles changed on resume")
        write_new_json(output / "intervals.json", state)
        quantile_hash_before_test = sha256(output / "intervals.json")
        bundle["interval_state"] = state
        save_progress(bundle, output)
        print("INTERVALS_FROZEN " + json.dumps({"groups": len(state["groups"]), "origins_per_horizon": protocol["interval_calibration_origins"],
                                                "excluded_boundary_ids": boundary_ids, "sha256": quantile_hash_before_test}), flush=True)
        for fold in data.folds:
            if fold.phase == "test":
                for model_name in protocol["variants"]:
                    for seed in protocol["seeds"]:
                        trial(model_name, seed, fold.block_id)
        if bundle["reproducibility"] is None:
            fold = next(fold for fold in data.folds if fold.phase == "calibration")
            key = f"LSTM_42_{fold.block_id}"
            original = bundle["models"][key]
            model, repeated = fit("LSTM", 42, fold.block_id)
            if any(not torch.equal(value, original["state_dict"][name]) for name, value in model.state_dict().items()):
                raise ValueError("Repeated fixed fit weights differ")
            if [(row["train_loss"], row["train_objective_loss"]) for row in repeated["history"]] != [
                (row["train_loss"], row["train_objective_loss"]) for row in original["history"]]:
                raise ValueError("Repeated fixed fit losses differ")
            if not np.array_equal(predict_scaled(model, prepared[fold.block_id]["X_outer"]), original["outer_predictions_scaled"].numpy()):
                raise ValueError("Repeated fixed fit predictions differ")
            bundle["reproducibility"] = {"trial": key, "weights_losses_predictions_exact": True,
                                         "training_seconds": repeated["training_seconds"]}
            save_progress(bundle, output)
        loaded = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        for entry in loaded["models"].values():
            model = build_model(entry["model_config"])
            model.load_state_dict(entry["state_dict"])
            if not np.array_equal(predict_scaled(model, prepared[entry["block_id"]]["X_outer"]), entry["outer_predictions_scaled"].numpy()):
                raise ValueError("Reloaded fixed-fit predictions differ")
        frame = apply_intervals(pd.concat([calibration_predictions, predictions_for("test")], ignore_index=True), state)
        if quantile_hash_before_test != sha256(output / "intervals.json"):
            raise ValueError("Test changed interval quantiles")
        metrics = score_final_predictions(frame)
        runs = pd.DataFrame([entry["summary"] for entry in bundle["models"].values()])
        def resources(row: dict) -> tuple[int, float]:
            if row["model"] == "persistence":
                return 0, 0.
            selected = runs.loc[runs["phase"].eq(row["phase"]) & runs["model"].eq(row["model"]) & runs["seed"].eq(row["seed"])]
            if row["block_id"] != "all":
                selected = selected.loc[selected["block_id"].eq(row["block_id"])]
            return int(selected["parameters"].iloc[0]), float(selected["training_seconds"].sum())
        used = [resources(row) for row in metrics.to_dict(orient="records")]
        metrics["parameters"] = [item[0] for item in used]
        metrics["training_seconds"] = [item[1] for item in used]
        interval_cells = interval_metrics(frame, group_by=("phase", "model", "variant", "seed", "block_id", "horizon"))
        interval_totals = interval_metrics(frame)
        interval_totals["block_id"] = "all"
        interval_scores = pd.concat([interval_cells, interval_totals], ignore_index=True)
        histories = [{"model": entry["model"], "variant": entry["variant"], "seed": entry["seed"],
                      "phase": entry["phase"], "block_id": entry["block_id"], **row}
                     for entry in bundle["models"].values() for row in entry["history"]]
        for name, table in {"predictions.csv": frame, "metrics.csv": metrics, "interval_metrics.csv": interval_scores,
                            "history.csv": pd.DataFrame(histories), "folds.csv": pd.DataFrame([fold.summary for fold in data.folds])}.items():
            table.to_csv(output / name, index=False, lineterminator="\n")
        summary_metrics = []
        for key, rows in metrics.loc[metrics["block_id"].eq("all")].groupby(["phase", "model", "variant", "horizon"], observed=True):
            record = dict(zip(["phase", "model", "variant", "horizon"], key)) | {"n_seeds": len(rows), "n_per_seed": int(rows["n"].iloc[0])}
            for metric in ["mse", "mae", "rmse", "skill_mae_pct", "bias", "training_seconds"]:
                record[f"{metric}_mean"] = float(rows[metric].mean())
                record[f"{metric}_std"] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else None
            record["parameters"] = int(rows["parameters"].iloc[0])
            summary_metrics.append(record)
        bundle["finished"] = True
        save_progress(bundle, output)
        serving = next(fold for fold in reversed(data.folds) if fold.phase == "test")
        serving_key = f"{protocol['selected_model']}_{protocol['selected_seed']}_{serving.block_id}"
        report = {"run_id": run_id, "protocol": protocol, "input_sha256": before, "code_sha256": code,
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu", "threads": 1},
            "runs": runs.to_dict(orient="records"), "summary_metrics": summary_metrics,
            "reproducibility": bundle["reproducibility"],
            "deployment": {"selected_model": protocol["selected_model"], "selected_seed": protocol["selected_seed"],
                           "checkpoint_key": serving_key, "model_refit_at": serving.refit_at.isoformat(),
                           "training_cutoff": serving.training_cutoff.isoformat(), "business_reference": "persistence",
                           "selection_basis": "development only; never changed after calibration or test",
                           "note": "Historical evaluation checkpoint, not a new production fit on the latest snapshot"},
            "checks": {"test_origin_samples_in_training": 0, "all_labels_mature_at_each_cut": True,
                       "transforms_fit_unique_allowed_train_rows_only": True,
                       "fixed_epochs_before_final_predictions": True, "selected_network_before_final_predictions": True,
                       "intervals_frozen_before_test_sha256": quantile_hash_before_test,
                       "interval_quantiles_unchanged_after_test": True,
                       "all_checkpoints_replayed_exactly": len(bundle["models"]),
                       "calibration_point_origins": protocol["calibration_point_origins"],
                       "interval_calibration_origins_per_horizon": protocol["interval_calibration_origins"],
                       "test_origins_per_horizon": protocol["test_origins"],
                       "old_results_and_data_unchanged": True},
            "limitations": [data_config["availability_mode"], protocol["online_policy_note"],
                            "Overlapping horizons and seeds are not independent observations",
                            "Nominal empirical bands have no exact coverage guarantee under serial dependence or regime change",
                            "Calibration coverage is descriptive on the residual pool, not independent validation of coverage"],
            "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())
                              if path.is_file() and path.name != "experiment.json"}}
        write_new_json(output / "experiment.json", report)
        print("FINAL_TEST_METRICS " + json.dumps([row for row in summary_metrics if row["phase"] == "test"]), flush=True)
    finally:
        if before != {str(path.relative_to(ROOT)): sha256(path) for path in sorted(protected)}:
            raise RuntimeError("Original data or historical experiment changed")


if __name__ == "__main__":
    main()
