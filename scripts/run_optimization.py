"""Bounded A/B search with purged inner early stopping, development only."""

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

from oil_forecast.acquisition import sha256
from oil_forecast.evaluation import diagnostics, score_predictions, select_development_configs
from oil_forecast.experiment import load_development_data, prepare_inner_development_fold, require_development
from oil_forecast.models import build_model, count_parameters
from oil_forecast.preprocessing import TargetTransformer
from oil_forecast.training import predict_scaled, set_reproducible, train_with_early_stopping


def save_progress(bundle: dict, output: Path) -> None:
    temporary = output / "checkpoint.tmp"
    torch.save(bundle, temporary)
    temporary.replace(output / "checkpoints.pt")


def prediction_frame(bundle: dict, data, prepared: dict) -> pd.DataFrame:
    parts = []
    jobs = [("baseline", "persistence", -1, fold.block_id, None) for fold in data.folds]
    jobs += [(entry["variant"], entry["model"], entry["seed"], entry["block_id"], entry)
             for entry in bundle["models"].values()]
    for variant, architecture, seed, block_id, entry in jobs:
        block = prepared[block_id]
        sample_ids = data.dataset.samples.iloc[block["fold"].eval_indices]["sample_id"].to_numpy()
        target = block["target_transformer"]
        scaled = None if entry is None else entry["outer_predictions_scaled"].numpy()
        changes = np.zeros((len(sample_ids), 3)) if scaled is None else target.inverse_transform(scaled)
        for index, horizon in enumerate(data.dataset.horizons):
            part = block["pairs"].loc[block["pairs"]["horizon"].eq(horizon)].copy()
            part["variant"], part["model"], part["seed"] = variant, architecture, seed
            part["predicted_change"] = part["sample_id"].map(pd.Series(changes[:, index], index=sample_ids))
            part["y_pred"] = part["reference_brent"] + part["predicted_change"]
            part["observed_change"] = part["y_true"] - part["reference_brent"]
            part["error"] = part["y_pred"] - part["y_true"]
            part["training_target_mean"] = target.mean_[index]
            part["training_target_scale"] = target.scale_[index]
            part["normalized_pred"] = (part["predicted_change"] - target.mean_[index]) / target.scale_[index]
            part["normalized_true"] = (part["observed_change"] - target.mean_[index]) / target.scale_[index]
            parts.append(part)
    frame = pd.concat(parts, ignore_index=True).sort_values(["variant", "model", "seed", "origin_date", "horizon"])
    if frame.duplicated(["variant", "model", "seed", "sample_id", "horizon"]).any():
        raise ValueError("Duplicate optimization prediction pair")
    expected = set(map(tuple, data.baseline[["sample_id", "horizon", "target_date"]].to_numpy()))
    for _, rows in frame.groupby(["variant", "model", "seed"], observed=True):
        if set(map(tuple, rows[["sample_id", "horizon", "target_date"]].to_numpy())) != expected:
            raise ValueError("A trial differs from the exact persistence pairs")
    return frame


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ("variant", "model", "seed", "phase")
    cells = score_predictions(frame, group_by=(*groups, "block_id", "horizon"))
    totals = score_predictions(frame, group_by=(*groups, "horizon"))
    totals["block_id"] = "all"
    metrics = pd.concat([cells, totals], ignore_index=True)
    if metrics["n_dropped"].any():
        raise ValueError("Non-finite optimization price pairs")
    baseline = metrics.loc[metrics["model"].eq("persistence")].set_index(["block_id", "horizon"])
    for metric in ["mae", "rmse"]:
        metrics[f"skill_{metric}_pct"] = [100 * (1 - value / baseline.loc[(block, h), metric])
            if baseline.loc[(block, h), metric] else np.nan
            for block, h, value in zip(metrics["block_id"], metrics["horizon"], metrics[metric])]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/optimization.json")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true", help="Append missing trials to an unfinished run only")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require_development(config["phase"])
    if (config["architectures"] != ["RNN", "LSTM", "GRU"] or config["hidden_size"] != 32
            or config["num_layers"] != 1 or config["batch_size"] != 64
            or config["max_epochs"] != 20 or config["patience"] != 5
            or config["variants"] != {"A": {"learning_rate": 0.001}, "B": {"learning_rate": 0.0003}}
            or config["weight_decay"] != 0.0001 or config["refit_on_full_outer_train"]
            or config["initial_seed"] != 42 or config["stability_seeds"] != [42, 123, 2026]
            or config["device"] != "cpu" or config["inner_validation_business_days"] != 252):
        raise ValueError("This executor implements only the authorized A/B budget and inner checkpoints")
    run_id = args.run_id or config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run identifier")
    output = ROOT / "experiments" / run_id
    if output.exists() and not args.resume:
        raise FileExistsError("Run exists; use a new id or explicitly resume an unfinished run")
    data_config = json.loads((ROOT / config["data_config"]).read_text(encoding="utf-8"))
    directory = ROOT / "data/supervised" / data_config["snapshot_id"]
    data = load_development_data(directory, data_config)
    prepared = {fold.block_id: prepare_inner_development_fold(data, fold.block_id,
                 config["inner_validation_business_days"]) for fold in data.folds}
    protected = [ROOT / key for key in data.manifest["input_sha256"]]
    protected += [path for path in directory.iterdir() if path.is_file()]
    protected += list((ROOT / "experiments/cpu-smoke-3epochs-seed42").glob("*"))
    protected += [args.config]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected if path.is_file()}
    code = {str(path.relative_to(ROOT)): sha256(path) for path in
            [Path(__file__).resolve(), *sorted((ROOT / "src/oil_forecast").glob("*.py"))]}
    if args.resume:
        bundle = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        if bundle["finished"] and (output / "experiment.json").exists():
            raise FileExistsError("Completed experiments are immutable")
        if bundle["config"] != config or bundle["input_sha256"] != before or bundle["code_sha256"] != code:
            raise ValueError("Cannot resume with different configuration, code or inputs")
    else:
        output.mkdir(parents=True)
        bundle = {"schema_version": 1, "run_id": run_id, "config": config, "input_sha256": before,
                  "code_sha256": code, "models": {}, "selected_variants": None,
                  "reproducibility": None, "finished": False}
        save_progress(bundle, output)

    def trial(variant: str, architecture: str, seed: int, block_id: str) -> None:
        key = f"{variant}_{architecture}_{seed}_{block_id}"
        if key in bundle["models"]:
            return
        block = prepared[block_id]
        options = config | config["variants"][variant] | {"seed": seed}
        set_reproducible(seed, config["threads"])
        model = build_model({"architecture": architecture, "input_size": block["X_train"].shape[-1],
                             "hidden_size": 32, "num_layers": 1, "output_size": 3})
        result = train_with_early_stopping(model, block["X_train"], block["y_train"],
                                           block["X_inner_valid"], block["y_inner_valid"], options)
        outer_predictions = predict_scaled(model, block["X_outer"], config["batch_size"])
        summary_keys = ["best_epoch", "last_epoch", "stopping_reason", "train_loss_at_best", "validation_loss_at_best",
                        "last_validation_loss", "training_seconds", "elapsed_seconds", "gradient_steps",
                        "nonzero_gradient_steps", "max_gradient_norm"]
        summary = {"variant": variant, "model": architecture, "seed": seed, "block_id": block_id,
                   "parameters": count_parameters(model), **{key: result[key] for key in summary_keys}}
        summary["validation_train_gap_at_best"] = result["validation_loss_at_best"] - result["train_loss_at_best"]
        summary["validation_last_minus_best"] = result["last_validation_loss"] - result["validation_loss_at_best"]
        for index, horizon in enumerate(data.dataset.horizons):
            summary[f"target_mean_{horizon}"] = float(block["target_transformer"].mean_[index])
            summary[f"target_scale_{horizon}"] = float(block["target_transformer"].scale_[index])
            summary[f"train_component_mse_{horizon}"] = result["train_loss_components"][index]
            summary[f"validation_component_mse_{horizon}"] = result["validation_loss_components"][index]
        bundle["models"][key] = {"variant": variant, "model": architecture, "seed": seed, "block_id": block_id,
            "model_config": model.config(), "state_dict": result["state_dict"], "history": result["history"],
            "summary": summary, "outer_predictions_scaled": torch.from_numpy(outer_predictions.copy()),
            "transformers": {"input": block["input_transformer"].to_dict(), "target": block["target_transformer"].to_dict()},
            "inner_split": block["inner_split"].summary,
            "inner_train_indices": torch.from_numpy(block["inner_split"].train_indices.copy()),
            "inner_validation_indices": torch.from_numpy(block["inner_split"].validation_indices.copy())}
        save_progress(bundle, output)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    try:
        for variant in config["variants"]:
            for architecture in config["architectures"]:
                for fold in data.folds:
                    trial(variant, architecture, config["initial_seed"], fold.block_id)
        initial_metrics = evaluate(prediction_frame(bundle, data, prepared))
        selected = select_development_configs(initial_metrics)
        if bundle["selected_variants"] is not None and bundle["selected_variants"] != selected:
            raise ValueError("Frozen configuration selection changed on resume")
        bundle["selected_variants"] = selected
        save_progress(bundle, output)
        print("SELECTED " + json.dumps(selected), flush=True)
        for architecture, variant in selected.items():
            for seed in config["stability_seeds"]:
                for fold in data.folds:
                    trial(variant, architecture, seed, fold.block_id)
        # Repeat one completed real trial, not the smoke or the whole search.
        if bundle["reproducibility"] is None:
            fold = data.folds[0]
            block = prepared[fold.block_id]
            key = f"A_RNN_42_{fold.block_id}"
            original = bundle["models"][key]
            set_reproducible(42, config["threads"])
            model = build_model(original["model_config"])
            result = train_with_early_stopping(model, block["X_train"], block["y_train"],
                block["X_inner_valid"], block["y_inner_valid"], config | config["variants"]["A"] | {"seed": 42})
            if any(not torch.equal(value, original["state_dict"][name]) for name, value in model.state_dict().items()):
                raise ValueError("Repeated trial weights differ")
            losses = lambda history: [(item["train_loss"], item["validation_loss"]) for item in history]
            if losses(result["history"]) != losses(original["history"]):
                raise ValueError("Repeated trial inner loss history differs")
            if not np.array_equal(predict_scaled(model, block["X_outer"], 64), original["outer_predictions_scaled"].numpy()):
                raise ValueError("Repeated trial outer predictions differ")
            bundle["reproducibility"] = {"trial": key, "weights_losses_predictions_exact": True,
                                         "training_seconds": result["training_seconds"]}
            save_progress(bundle, output)
        loaded = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        for entry in loaded["models"].values():
            model = build_model(entry["model_config"])
            model.load_state_dict(entry["state_dict"])
            if not np.array_equal(predict_scaled(model, prepared[entry["block_id"]]["X_outer"], 64),
                                  entry["outer_predictions_scaled"].numpy()):
                raise ValueError("Saved best checkpoint does not reproduce predictions")
        frame = prediction_frame(bundle, data, prepared)
        metrics = evaluate(frame)
        diagnostic = diagnostics(frame, horizons=(5, 63, 252))
        histories = [{"variant": entry["variant"], "model": entry["model"], "seed": entry["seed"],
                      "block_id": entry["block_id"], **item}
                     for entry in bundle["models"].values() for item in entry["history"]]
        runs = pd.DataFrame([entry["summary"] for entry in bundle["models"].values()])
        stability = []
        totals = metrics.loc[metrics["block_id"].eq("all")]
        for architecture, variant in selected.items():
            for horizon in data.dataset.horizons:
                rows = totals.loc[totals["model"].eq(architecture) & totals["variant"].eq(variant) & totals["horizon"].eq(horizon)]
                record = {"model": architecture, "variant": variant, "horizon": horizon, "n_seeds": len(rows)}
                for metric in ["mse", "mae", "rmse", "skill_mae_pct"]:
                    record[f"{metric}_mean"] = float(rows[metric].mean())
                    record[f"{metric}_std"] = float(rows[metric].std(ddof=1))
                stability.append(record)
        for name, table in {"history.csv": pd.DataFrame(histories), "metrics.csv": metrics,
                            "predictions.csv": frame, "diagnostics.csv": diagnostic,
                            "inner_folds.csv": pd.DataFrame([value["inner_split"].summary for value in prepared.values()])}.items():
            table.to_csv(output / name, index=False, lineterminator="\n")
        bundle["finished"] = True
        save_progress(bundle, output)
        report = {"run_id": run_id, "config": config, "input_sha256": before, "code_sha256": code,
                  "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                  "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu", "threads": 1},
                  "selected_variants_seed42": selected, "stability": stability,
                  "runs": runs.to_dict(orient="records"), "reproducibility": bundle["reproducibility"],
                  "checks": {"development_only": True, "outer_pairs_per_trial": len(data.baseline),
                             "restored_best_inner_checkpoint": True, "all_checkpoints_replayed_exactly": len(bundle["models"]),
                             "no_outer_refit": True, "train_only_inner_scaling": True},
                  "interpretation": "Development configuration selection and seed diagnostics; no final calibration/test estimate",
                  "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())
                                    if path.is_file() and path.name != "experiment.json"}}
        (output / "experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("STABILITY " + json.dumps(stability, ensure_ascii=False), flush=True)
    finally:
        if before != {str(path.relative_to(ROOT)): sha256(path) for path in protected if path.is_file()}:
            raise RuntimeError("Original inputs or the preserved smoke changed")


if __name__ == "__main__":
    main()
