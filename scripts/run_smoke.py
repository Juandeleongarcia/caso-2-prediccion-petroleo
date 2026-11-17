"""Three fixed CPU epochs for each architecture and development semester only."""

from __future__ import annotations

import argparse
import json
import os
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
from oil_forecast.evaluation import score_predictions
from oil_forecast.experiment import load_development_data, prepare_development_fold, require_development
from oil_forecast.models import build_model, count_parameters
from oil_forecast.training import set_reproducible, train_short


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.json")
    parser.add_argument("--run-id", help="New local experiment name; existing runs are immutable")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require_development(config["phase"])
    if config["epochs"] not in [2, 3] or config["device"] != "cpu":
        raise ValueError("Only two or three CPU epochs are authorized by this smoke runner")
    if config["architectures"] != ["RNN", "LSTM", "GRU"]:
        raise ValueError("The short comparison must include the same three architectures")
    run_id = args.run_id or config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid local experiment name")
    output = ROOT / "experiments" / run_id
    if output.exists():
        raise FileExistsError("Experiment already exists; use --run-id to preserve prior results")
    data_path = ROOT / config["data_config"]
    data_config = json.loads(data_path.read_text(encoding="utf-8"))
    directory = ROOT / "data/supervised" / data_config["snapshot_id"]
    data = load_development_data(directory, data_config)
    protected = [ROOT / key for key in data.manifest["input_sha256"]]
    protected += [path for path in directory.iterdir() if path.is_file()]
    protected += [args.config]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    predictions, histories, summaries = [], [], []
    checkpoints = {"schema_version": 1, "experiment_config": config, "models": {}}
    normalized_predictions = {}
    try:
        for fold in data.folds:
            prepared = prepare_development_fold(data, fold.block_id)
            x_train, y_train = prepared["X_train"], prepared["y_train"]
            x_valid, y_valid = prepared["X_valid"], prepared["y_valid"]
            for architecture in config["architectures"]:
                set_reproducible(config["seed"], config["threads"])
                model_config = {"architecture": architecture, "input_size": x_train.shape[-1],
                                "hidden_size": config["hidden_size"], "num_layers": config["num_layers"],
                                "output_size": config["output_size"]}
                model = build_model(model_config).cpu()
                result = train_short(model, x_train, y_train, x_valid, y_valid, config)
                key = f"{architecture}_{fold.block_id}"
                normalized_predictions[key] = result["predictions"]
                prices = prepared["target_transformer"].inverse_transform(result["predictions"])
                references = data.dataset.samples.iloc[fold.eval_indices]["reference_brent"].to_numpy()
                prices += references[:, None]
                for horizon_index, horizon in enumerate(data.dataset.horizons):
                    part = prepared["pairs"].loc[prepared["pairs"]["horizon"].eq(horizon)].copy()
                    # Match by stable sample id rather than relying on export row order.
                    mapping = pd.Series(prices[:, horizon_index], index=data.dataset.samples.iloc[fold.eval_indices]["sample_id"])
                    part["y_pred"] = part["sample_id"].map(mapping)
                    part["error"] = part["y_pred"] - part["y_true"]
                    part["model"] = architecture
                    predictions.append(part)
                for item in result["history"]:
                    histories.append({"model": architecture, "block_id": fold.block_id, **item})
                summary = {"model": architecture, "block_id": fold.block_id, "n_train": len(x_train),
                           "n_valid": len(x_valid), "parameters": count_parameters(model),
                           **{name: result[name] for name in ["training_seconds", "initial_train_loss", "final_train_loss",
                                                             "initial_validation_loss", "final_validation_loss", "loss_decreased",
                                                             "gradient_steps", "nonzero_gradient_steps", "max_gradient_norm"]}}
                summaries.append(summary)
                checkpoints["models"][key] = {"model_config": model.config(),
                    "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    "transformers": data.transformers[fold.block_id], "fold": fold.summary,
                    "feature_names": list(data.dataset.feature_names), "horizons": list(data.dataset.horizons)}
                print(json.dumps(summary, ensure_ascii=False), flush=True)
        baseline = data.baseline.copy()
        baseline["model"] = "persistence"
        prediction_frame = pd.concat([baseline, *predictions], ignore_index=True).sort_values(["model", "origin_date", "horizon"])
        if prediction_frame.duplicated(["model", "sample_id", "horizon"]).any():
            raise ValueError("Duplicate model/origin/horizon pairs")
        baseline_keys = set(map(tuple, baseline[["sample_id", "horizon", "target_date"]].to_numpy()))
        for architecture in config["architectures"]:
            rows = prediction_frame.loc[prediction_frame["model"].eq(architecture)]
            if set(map(tuple, rows[["sample_id", "horizon", "target_date"]].to_numpy())) != baseline_keys:
                raise ValueError("Network and persistence evaluated on different pairs")
        metrics = score_predictions(prediction_frame, group_by=("model", "phase", "block_id", "horizon"))
        totals = score_predictions(prediction_frame, group_by=("model", "phase", "horizon"))
        totals["block_id"] = "all"
        metrics = pd.concat([metrics, totals], ignore_index=True)
        base_scores = metrics.loc[metrics["model"].eq("persistence")].set_index(["block_id", "horizon"])
        for metric in ["mae", "rmse"]:
            metrics[f"skill_{metric}_pct"] = [100 * (1 - value / base_scores.loc[(block, h), metric])
                if base_scores.loc[(block, h), metric] != 0 else np.nan
                for block, h, value in zip(metrics["block_id"], metrics["horizon"], metrics[metric])]
        if metrics["n_dropped"].any():
            raise ValueError("Unexpected non-finite predictions")
        output.mkdir(parents=True)
        torch.save(checkpoints, output / "checkpoints.pt")
        loaded = torch.load(output / "checkpoints.pt", map_location="cpu", weights_only=True)
        # Reload every real development checkpoint and reproduce its predictions.
        for fold in data.folds:
            x_valid = prepare_development_fold(data, fold.block_id)["X_valid"]
            for architecture in config["architectures"]:
                key = f"{architecture}_{fold.block_id}"
                state = loaded["models"][key]
                restored = build_model(state["model_config"]).cpu()
                restored.load_state_dict(state["state_dict"])
                restored.eval()
                with torch.inference_mode():
                    replay = np.concatenate([restored(torch.from_numpy(x_valid[start:start + config["batch_size"]])).numpy()
                        for start in range(0, len(x_valid), config["batch_size"])])
                if not np.array_equal(replay, normalized_predictions[key]):
                    raise ValueError(f"Checkpoint prediction replay differs: {key}")
        for name, frame in {"history.csv": pd.DataFrame(histories), "metrics.csv": metrics,
                            "predictions.csv": prediction_frame}.items():
            frame.to_csv(output / name, index=False, lineterminator="\n")
        manifest = {"run_id": run_id, "config": config, "experiment_config_sha256": sha256(args.config),
            "data_manifest_sha256": sha256(directory / "manifest.json"), "input_sha256": before,
            "historical_environment_changes": data.environment_changes,
            "current_environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                    "numpy": np.__version__, "pandas": pd.__version__, "cuda_build": torch.version.cuda,
                                    "processor": platform.processor(), "logical_cpus": os.cpu_count(), "threads_used": config["threads"]},
            "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                            [Path(__file__).resolve(), *sorted((ROOT / "src/oil_forecast").glob("*.py"))]},
            "runs": summaries,
            "checks": {"development_only": True, "same_baseline_pairs_per_architecture": len(baseline_keys),
                       "all_18_checkpoints_replayed_exactly": True, "finite_losses_gradients_predictions": True,
                       "deterministic_cpu_algorithms": True, "last_epoch_without_validation_selection": True},
            "interpretation": "Short CPU functionality experiment, not final model selection or performance",
            "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
        (output / "experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(totals[["model", "horizon", "n", "mse", "mae", "rmse"]].to_string(index=False), flush=True)
    finally:
        if before != {str(path.relative_to(ROOT)): sha256(path) for path in protected}:
            raise RuntimeError("Original or supervised inputs changed during the short experiment")


if __name__ == "__main__":
    main()
