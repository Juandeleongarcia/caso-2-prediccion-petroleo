"""One bounded persistence-initialization/robust-loss comparison on development."""

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
import torch

from run_optimization import evaluate, prediction_frame, save_progress
from oil_forecast.acquisition import sha256
from oil_forecast.evaluation import assess_development_refinement, diagnostics
from oil_forecast.experiment import load_development_data, prepare_inner_development_fold, require_development
from oil_forecast.models import build_model, count_parameters, initialize_persistence_head
from oil_forecast.training import predict_scaled, set_reproducible, train_with_early_stopping


def load_reference(directory: Path, data) -> tuple[dict, pd.DataFrame, dict]:
    """Verify the untouched historical results, then expose only B development."""
    report = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    for name, digest in report["input_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Original B input changed: {name}")
    for name, digest in report["output_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Original B output changed: {name}")
    if not report["checks"]["development_only"] or report["selected_variants_seed42"] != dict.fromkeys(["RNN", "LSTM", "GRU"], "B"):
        raise ValueError("Reference is not the completed B development experiment")
    frame = pd.read_csv(directory / "predictions.csv")
    frame = frame.loc[frame["variant"].eq("B")].copy()
    if not frame["phase"].eq("development").all():
        raise ValueError("Reference contains a forbidden evaluation phase")
    for name in ["origin_date", "target_date", "window_start_date", "reference_observation_date"]:
        frame[name] = pd.to_datetime(frame[name], errors="raise")
    for name in ["decision_at", "reference_available_at", "label_available_at", "training_cutoff", "refit_at"]:
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    expected = data.baseline.set_index(["sample_id", "horizon", "target_date"]).sort_index()
    if set(map(tuple, frame[["model", "seed"]].drop_duplicates().to_numpy())) != {
        (model, seed) for model in ("RNN", "LSTM", "GRU") for seed in (42, 123, 2026)
    }:
        raise ValueError("Reference B needs all three architectures and three original seeds")
    for _, rows in frame.groupby(["model", "seed"], observed=True):
        indexed = rows.set_index(["sample_id", "horizon", "target_date"]).sort_index()
        if not indexed.index.equals(expected.index):
            raise ValueError("B and baseline prediction pairs differ")
        for name in ["y_true", "reference_brent"]:
            if not np.allclose(indexed[name], expected[name], atol=1e-12, rtol=1e-12):
                raise ValueError("B and baseline original-unit prices differ")
        for name in ["origin_date", "label_available_at", "reference_available_at"]:
            if not indexed[name].equals(expected[name]):
                raise ValueError("B and baseline temporal metadata differ")
    if not np.allclose(frame["y_pred"], frame["reference_brent"] + frame["normalized_pred"]
                       * frame["training_target_scale"] + frame["training_target_mean"], atol=1e-10, rtol=1e-12):
        raise ValueError("Reference residual reconstruction differs")
    checkpoint = torch.load(directory / "checkpoints.pt", map_location="cpu", weights_only=True)
    return report, frame, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/refinement.json")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require_development(config["phase"])
    expected = {"architectures": ["RNN", "LSTM", "GRU"], "hidden_size": 32, "num_layers": 1,
                "output_size": 3, "batch_size": 64, "learning_rate": 0.0003, "weight_decay": 0.0001,
                "max_epochs": 20, "patience": 5, "min_delta": 0.0001, "gradient_clip": 1.0,
                "optimizer": "Adam", "threads": 1, "device": "cpu", "initial_seed": 42,
                "stability_seeds": [42, 123, 2026], "inner_validation_business_days": 252,
                "refit_on_full_outer_train": False, "validation_metric": "normalized_mse",
                "variants": {"C": {"initialization": "persistence", "loss_function": "mse"},
                             "D": {"initialization": "persistence", "loss_function": "smooth_l1", "huber_beta": 1.0}}}
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("This executor supports only the frozen, limited C/D development comparison")
    selection = config["selection"]
    if (selection["minimum_relative_score_gain"] != 0.05 or selection["minimum_improved_blocks"] != 4
            or selection["minimum_improved_horizons"] != 2):
        raise ValueError("Do not change the predeclared refinement gate after looking at results")
    run_id = args.run_id or config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run identifier")
    output = ROOT / "experiments" / run_id
    if output.exists() and not args.resume:
        raise FileExistsError("An existing experiment cannot be overwritten")
    if args.resume and (output / "experiment.json").exists():
        raise FileExistsError("Completed experiments are immutable")
    data_config = json.loads((ROOT / config["data_config"]).read_text(encoding="utf-8"))
    data_dir = ROOT / "data/supervised" / data_config["snapshot_id"]
    data = load_development_data(data_dir, data_config)
    prepared = {fold.block_id: prepare_inner_development_fold(data, fold.block_id, 252) for fold in data.folds}
    reference_dir = ROOT / config["reference_dir"]
    reference_report, reference, reference_checkpoint = load_reference(reference_dir, data)
    for fold in data.folds:
        old = reference_checkpoint["models"][f"B_LSTM_42_{fold.block_id}"]
        block = prepared[fold.block_id]
        for name, actual in [("inner_train_indices", block["inner_split"].train_indices),
                             ("inner_validation_indices", block["inner_split"].validation_indices)]:
            if not np.array_equal(old[name].numpy(), actual):
                raise ValueError("Inner split changed relative to B")
        if old["transformers"] != {"input": block["input_transformer"].to_dict(),
                                   "target": block["target_transformer"].to_dict()}:
            raise ValueError("Inner transformers changed relative to B")
    protected = {ROOT / name for name in reference_report["input_sha256"]}
    protected |= {path for path in reference_dir.iterdir() if path.is_file()}
    protected |= {args.config.resolve(), *[path for path in data_dir.iterdir() if path.is_file()]}
    before = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(protected)}
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/run_optimization.py",
                  *sorted((ROOT / "src/oil_forecast").glob("*.py"))]
    code = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    if args.resume:
        bundle = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        if bundle["config"] != config or bundle["input_sha256"] != before or bundle["code_sha256"] != code:
            raise ValueError("Resume requires identical code, configuration and inputs")
    else:
        output.mkdir(parents=True)
        bundle = {"schema_version": 1, "run_id": run_id, "config": config, "input_sha256": before,
                  "code_sha256": code, "models": {}, "stage_b": None, "selected_candidates": None,
                  "reproducibility": None, "finished": False}
        save_progress(bundle, output)

    def combined_predictions() -> pd.DataFrame:
        return pd.concat([prediction_frame(bundle, data, prepared), reference], ignore_index=True)

    def assessment(metrics: pd.DataFrame, architecture: str, variant: str, seeds: tuple[int, ...]) -> dict:
        return assess_development_refinement(metrics, architecture, variant, seeds,
            selection["minimum_relative_score_gain"], selection["minimum_improved_blocks"],
            selection["minimum_improved_horizons"])

    def fit(variant: str, architecture: str, seed: int, block_id: str) -> tuple[object, dict, float]:
        block = prepared[block_id]
        options = config | config["variants"][variant] | {"seed": seed}
        set_reproducible(seed, 1)
        model = build_model({"architecture": architecture, "input_size": 20, "hidden_size": 32,
                             "num_layers": 1, "output_size": 3})
        target = block["target_transformer"]
        initialize_persistence_head(model, target.mean_, target.scale_)
        initial_change = target.inverse_transform(predict_scaled(model, block["X_train"][:16]))
        initial_error = float(np.abs(initial_change).max())
        if initial_error > 1e-5:
            raise ValueError("The initial residual prediction is not persistence")
        result = train_with_early_stopping(model, block["X_train"], block["y_train"],
            block["X_inner_valid"], block["y_inner_valid"], options)
        return model, result, initial_error

    def trial(variant: str, architecture: str, seed: int, block_id: str) -> None:
        key = f"{variant}_{architecture}_{seed}_{block_id}"
        if key in bundle["models"]:
            return
        block = prepared[block_id]
        model, result, initial_error = fit(variant, architecture, seed, block_id)
        predictions = predict_scaled(model, block["X_outer"], 64)
        fields = ["best_epoch", "last_epoch", "stopping_reason", "train_loss_at_best", "validation_loss_at_best",
                  "last_train_loss", "last_validation_loss", "training_seconds", "elapsed_seconds",
                  "gradient_steps", "nonzero_gradient_steps", "max_gradient_norm", "loss_function",
                  "validation_metric"]
        summary = {"variant": variant, "model": architecture, "seed": seed, "block_id": block_id,
                   "parameters": count_parameters(model), "initial_max_abs_change": initial_error,
                   **{name: result[name] for name in fields}}
        summary["validation_train_gap_at_best"] = result["validation_loss_at_best"] - result["train_loss_at_best"]
        summary["validation_last_minus_best"] = result["last_validation_loss"] - result["validation_loss_at_best"]
        for index, horizon in enumerate(data.dataset.horizons):
            summary[f"target_mean_{horizon}"] = float(block["target_transformer"].mean_[index])
            summary[f"target_scale_{horizon}"] = float(block["target_transformer"].scale_[index])
            summary[f"train_component_mse_{horizon}"] = result["train_loss_components"][index]
            summary[f"validation_component_mse_{horizon}"] = result["validation_loss_components"][index]
        bundle["models"][key] = {"variant": variant, "model": architecture, "seed": seed, "block_id": block_id,
            "model_config": model.config(), "training_config": result["training_config"],
            "state_dict": result["state_dict"], "history": result["history"], "summary": summary,
            "outer_predictions_scaled": torch.from_numpy(predictions.copy()),
            "transformers": {"input": block["input_transformer"].to_dict(), "target": block["target_transformer"].to_dict()},
            "inner_split": block["inner_split"].summary,
            "inner_train_indices": torch.from_numpy(block["inner_split"].train_indices.copy()),
            "inner_validation_indices": torch.from_numpy(block["inner_split"].validation_indices.copy())}
        save_progress(bundle, output)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    try:
        for variant in ("C", "D"):
            for architecture in config["architectures"]:
                for fold in data.folds:
                    trial(variant, architecture, 42, fold.block_id)
        initial_metrics = evaluate(combined_predictions())
        initial = [assessment(initial_metrics, model, variant, (42,))
                   for model in config["architectures"] for variant in ("C", "D")]
        selected = {}
        for architecture in config["architectures"]:
            passing = [row for row in initial if row["model"] == architecture and row["passes"]]
            if passing:
                selected[architecture] = min(passing, key=lambda row: (row["candidate_score"], row["variant"]))["variant"]
        if bundle["stage_b"] is not None and (bundle["stage_b"] != initial or bundle["selected_candidates"] != selected):
            raise ValueError("Frozen stage B selection changed on resume")
        bundle["stage_b"], bundle["selected_candidates"] = initial, selected
        save_progress(bundle, output)
        print("STAGE_B " + json.dumps(initial), flush=True)
        print("STABILITY_CANDIDATES " + json.dumps(selected), flush=True)
        for architecture, variant in selected.items():
            for seed in (123, 2026):
                for fold in data.folds:
                    trial(variant, architecture, seed, fold.block_id)
        if bundle["reproducibility"] is None:
            fold = data.folds[0]
            key = f"C_RNN_42_{fold.block_id}"
            original = bundle["models"][key]
            model, repeated, _ = fit("C", "RNN", 42, fold.block_id)
            if any(not torch.equal(value, original["state_dict"][name]) for name, value in model.state_dict().items()):
                raise ValueError("Repeated refinement weights differ")
            if [(row["train_loss"], row["validation_loss"]) for row in repeated["history"]] != [
                (row["train_loss"], row["validation_loss"]) for row in original["history"]]:
                raise ValueError("Repeated refinement losses differ")
            if not np.array_equal(predict_scaled(model, prepared[fold.block_id]["X_outer"]),
                                  original["outer_predictions_scaled"].numpy()):
                raise ValueError("Repeated refinement predictions differ")
            bundle["reproducibility"] = {"trial": key, "weights_losses_predictions_exact": True,
                                         "training_seconds": repeated["training_seconds"]}
            save_progress(bundle, output)
        loaded = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        for entry in loaded["models"].values():
            model = build_model(entry["model_config"])
            model.load_state_dict(entry["state_dict"])
            if not np.array_equal(predict_scaled(model, prepared[entry["block_id"]]["X_outer"]),
                                  entry["outer_predictions_scaled"].numpy()):
                raise ValueError("Saved refinement checkpoint predictions differ")
        # Replay, without training, one original B checkpoint with the current
        # model code. Its default forward path must remain exactly compatible.
        old_key = f"B_LSTM_42_{data.folds[0].block_id}"
        old = reference_checkpoint["models"][old_key]
        model = build_model(old["model_config"])
        model.load_state_dict(old["state_dict"])
        if not np.array_equal(predict_scaled(model, prepared[old["block_id"]]["X_outer"]),
                              old["outer_predictions_scaled"].numpy()):
            raise ValueError("Original B checkpoint behavior changed")
        frame = combined_predictions()
        metrics = evaluate(frame)
        diagnostic = diagnostics(frame, horizons=(5, 63, 252))
        stability = [assessment(metrics, model, variant, (42, 123, 2026)) for model, variant in selected.items()]
        recommendations = {model: next((row["variant"] for row in stability if row["model"] == model and row["passes"]), "B")
                           for model in config["architectures"]}
        histories = [{"variant": entry["variant"], "model": entry["model"], "seed": entry["seed"],
                      "block_id": entry["block_id"], **row} for entry in bundle["models"].values() for row in entry["history"]]
        for name, table in {"predictions.csv": frame, "metrics.csv": metrics, "diagnostics.csv": diagnostic,
                            "history.csv": pd.DataFrame(histories), "inner_folds.csv": pd.DataFrame([
                                block["inner_split"].summary for block in prepared.values()])}.items():
            table.to_csv(output / name, index=False, lineterminator="\n")
        runs = [entry["summary"] for entry in bundle["models"].values()]
        summaries = []
        for key, rows in metrics.loc[metrics["block_id"].eq("all")].groupby(["variant", "model", "horizon"], observed=True):
            record = dict(zip(["variant", "model", "horizon"], key)) | {"n_seeds": len(rows)}
            for metric in ("mse", "mae", "rmse", "skill_mae_pct"):
                record[f"{metric}_mean"] = float(rows[metric].mean())
                record[f"{metric}_std"] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else None
            summaries.append(record)
        bundle["finished"] = True
        save_progress(bundle, output)
        report = {"run_id": run_id, "config": config, "input_sha256": before, "code_sha256": code,
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu", "threads": 1},
            "stage_b": initial, "selected_candidates": selected, "stability_assessments": stability,
            "recommendations": recommendations, "summary_metrics": summaries, "runs": runs,
            "reference": {"run_id": reference_report["run_id"], "retrained": False,
                          "checkpoint_replayed_exactly": old_key, "original_target_already_residual": True},
            "reproducibility": bundle["reproducibility"],
            "checks": {"development_only": True, "same_outer_and_inner_splits_as_B": True,
                       "same_inner_transformers_as_B": True, "outer_pairs_per_trial": len(data.baseline),
                       "initial_price_matches_persistence_atol": 1e-5, "epoch_zero_diagnostic_only": True,
                       "all_checkpoints_replayed_exactly": len(bundle["models"]),
                       "checkpoint_criterion": "inner validation normalized MSE for C and D", "no_outer_refit": True},
            "interpretation": "Exploratory development only; repeated selection does not establish final superiority",
            "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())
                              if path.is_file() and path.name != "experiment.json"}}
        (output / "experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("RECOMMENDATIONS " + json.dumps(recommendations), flush=True)
        print("STABILITY " + json.dumps(stability), flush=True)
    finally:
        if before != {str(path.relative_to(ROOT)): sha256(path) for path in sorted(protected)}:
            raise RuntimeError("Original inputs or previous experiments changed")


if __name__ == "__main__":
    main()
