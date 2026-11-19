"""Additional academic scores derived from immutable saved price predictions."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .acquisition import sha256
from .artifacts import MANIFEST, _fresh, _json, _manifest, _safe_name

EXPERIMENTS = {
    "smoke": "cpu-smoke-3epochs-seed42",
    "optimization": "cpu-nested-ab-20epochs-seeds-v1",
    "refinement": "cpu-refinement-persistence-init-v1",
    "final": "cpu-final-evaluation-v1",
}
KEYS = ["experiment", "phase", "model", "variant", "seed", "block_id", "horizon"]
SCORES = ["n", "n_dropped", "mse", "mae", "rmse", "bias", "r2"]


def _scores(actual, predicted) -> dict:
    truth = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(predicted, dtype=np.float64)
    if truth.ndim != 1 or forecast.shape != truth.shape or not len(truth):
        raise ValueError("Academic scores require nonempty, aligned one-dimensional pairs")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
        raise ValueError("Academic scores do not silently drop non-finite prediction pairs")
    with np.errstate(over="raise", invalid="raise"):
        try:
            errors = forecast - truth
            squared_error = float(np.dot(errors, errors))
            centered = truth - truth.mean()
            total = float(np.dot(centered, centered))
            mse = squared_error / len(truth)
            result = {"n": len(truth), "n_dropped": 0, "mse": mse,
                      "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(mse)),
                      "bias": float(errors.mean()),
                      "r2": 1. - squared_error / total if len(truth) >= 2 and total > 0 else np.nan}
        except FloatingPointError as failure:
            raise ValueError("Prediction magnitudes overflow the academic score calculation") from failure
    if not np.isfinite([result[name] for name in ("mse", "mae", "rmse", "bias")]).all():
        raise ValueError("Academic error scores must be finite")
    return result


def derive_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score every seed/horizon/block separately, then recompute phase totals.

    R² uses 1 - sum((prediction-truth)²) / sum((truth-mean(truth))²).
    It remains negative when appropriate and undefined for fewer than two
    observations or a constant target. Seeds and horizons are never pooled.
    """
    required = set(KEYS + ["sample_id", "y_true", "y_pred"])
    if not required.issubset(predictions):
        raise ValueError(f"Missing academic prediction fields: {sorted(required - set(predictions))}")
    if predictions.empty:
        return pd.DataFrame(columns=[*KEYS, *SCORES])
    frame = predictions.copy()
    if frame[KEYS + ["sample_id"]].isna().any().any():
        raise ValueError("Academic prediction identifiers cannot be missing")
    if not frame["phase"].isin(["development", "calibration", "test"]).all():
        raise ValueError("Unknown academic evaluation phase")
    if frame["block_id"].eq("all").any():
        raise ValueError("Academic scores require individual predictions, not pooled input rows")
    if not frame["horizon"].isin([5, 63, 252]).all():
        raise ValueError("Academic predictions must use the three frozen horizons")
    if any(not str(block).startswith(str(phase) + "_")
           for block, phase in zip(frame["block_id"], frame["phase"])):
        raise ValueError("Prediction blocks must belong to their declared evaluation phase")
    total_keys = [key for key in KEYS if key != "block_id"]
    if frame.duplicated(total_keys + ["sample_id"]).any():
        raise ValueError("A prediction pair cannot occur twice within a seed, phase and horizon")
    records = []
    for key, part in frame.groupby(KEYS, sort=True, observed=True):
        records.append(dict(zip(KEYS, key)) | _scores(part["y_true"], part["y_pred"]))
    for key, part in frame.groupby(total_keys, sort=True, observed=True):
        records.append(dict(zip(total_keys, key)) | {"block_id": "all"}
                       | _scores(part["y_true"], part["y_pred"]))
    return pd.DataFrame(records, columns=[*KEYS, *SCORES]).sort_values(KEYS).reset_index(drop=True)


def enrich_metrics(metrics: pd.DataFrame, supplement: pd.DataFrame,
                   experiment: str | None = None) -> pd.DataFrame:
    """Append only R² to a copy, reconciling existing scores without replacing them."""
    original = metrics.copy(deep=True)
    frame = original.copy()
    if "experiment" not in frame:
        if experiment is None:
            raise ValueError("An experiment identifier is required for serving-only metrics")
        frame["experiment"] = experiment
    required = set(KEYS + ["n", "mae", "mse", "rmse"])
    if not required.issubset(frame) or not (required | {"r2"}).issubset(supplement):
        raise ValueError("Academic enrichment requires matching saved metric keys and scores")
    if frame[KEYS].isna().any().any() or supplement[KEYS].isna().any().any():
        raise ValueError("Missing academic metric identifiers")
    if frame.duplicated(KEYS).any() or supplement.duplicated(KEYS).any():
        raise ValueError("Academic enrichment requires unique metric keys")
    right_columns = [*KEYS, "n", "mae", "mse", "rmse", "r2"]
    if "bias" in supplement:
        right_columns.append("bias")
    merged = frame.merge(supplement[right_columns], on=KEYS, how="left", sort=False,
                         validate="one_to_one", suffixes=("", "__derived"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("A saved score has no matching academic prediction group")
    if not np.array_equal(merged["n"].to_numpy(), merged["n__derived"].to_numpy()):
        raise ValueError("Academic and saved metric pair counts differ")
    for metric in ("mae", "mse", "rmse", "bias"):
        if f"{metric}__derived" not in merged:
            continue
        saved = pd.to_numeric(merged[metric], errors="raise").to_numpy(dtype=float)
        derived = pd.to_numeric(merged[f"{metric}__derived"], errors="raise").to_numpy(dtype=float)
        known = np.isfinite(saved)
        if not np.isfinite(derived[known]).all() or not np.allclose(saved[known], derived[known], rtol=1e-9, atol=1e-9):
            raise ValueError(f"Academic predictions do not reconcile the saved {metric}")
    if "n_dropped" in frame and not frame["n_dropped"].eq(0).all():
        raise ValueError("Academic enrichment cannot hide dropped original pairs")
    derived_r2 = merged["r2__derived" if "r2" in original else "r2"].to_numpy(dtype=float)
    if np.isinf(derived_r2).any():
        raise ValueError("Undefined R² must be missing, not infinite")
    if "r2" in original and not np.allclose(original["r2"], derived_r2, rtol=1e-9, atol=1e-9, equal_nan=True):
        raise ValueError("An existing R² column differs from its saved prediction supplement")
    original["r2"] = derived_r2
    return original


def _single_selection(predictions: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["sample_id", "target_date", "reference_brent", "y_true", "y_pred"])
    if predictions.empty or not required.issubset(predictions):
        raise ValueError("Error analysis requires a nonempty saved prediction selection")
    columns = [key for key in KEYS if key != "block_id"]
    if predictions[columns].isna().any().any() or len(predictions[columns].drop_duplicates()) != 1:
        raise ValueError("Error analysis requires one experiment, phase, model, variant, seed and horizon")
    if predictions["sample_id"].duplicated().any() or predictions["block_id"].eq("all").any():
        raise ValueError("Error analysis requires unique individual prediction pairs")
    result = predictions.copy(deep=True)
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise")
    if result["target_date"].isna().any() or result["target_date"].duplicated().any():
        raise ValueError("Error analysis requires unique real target dates")
    numbers = result[["reference_brent", "y_true", "y_pred"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Error analysis cannot silently drop non-finite prices")
    return result


def detail_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return actual signed/absolute errors for one saved model and horizon."""
    result = _single_selection(predictions)
    result["error"] = result["y_pred"] - result["y_true"]
    result["absolute_error"] = result["error"].abs()
    result["naive_error"] = result["reference_brent"] - result["y_true"]
    result["absolute_naive_error"] = result["naive_error"].abs()
    return result


def error_periods(predictions: pd.DataFrame) -> pd.DataFrame:
    """Describe errors by target quarter on exactly the same naive/model pairs."""
    frame = detail_cases(predictions)
    frame["period"] = frame["target_date"].dt.to_period("Q").astype(str)
    records = []
    for period, part in frame.groupby("period", sort=True, observed=True):
        scores = _scores(part["y_true"], part["y_pred"])
        naive_mae = float(part["absolute_naive_error"].mean())
        records.append({"period": period, "n": len(part), "mae": scores["mae"],
                        "rmse": scores["rmse"], "bias": scores["bias"], "naive_mae": naive_mae,
                        "skill_mae_pct": 100 * (1 - scores["mae"] / naive_mae) if naive_mae > 0 else np.nan})
    return pd.DataFrame(records)


def _normalize(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    result = frame.copy()
    if "phase" not in result:
        result["phase"] = result["block_id"].str.split("_").str[0]
    if "seed" not in result:
        result["seed"] = 42
    if "variant" not in result:
        result["variant"] = "smoke"
    baseline = result["model"].eq("persistence")
    result.loc[baseline, "variant"] = "baseline"
    result.loc[baseline, "seed"] = -1
    result.insert(0, "experiment", experiment)
    return result


def build_academic_metrics(experiments_root: Path | str, output_dir: Path | str) -> dict:
    """Create a separate, portable supplement; never overwrite historical files."""
    source = Path(experiments_root).resolve()
    output = Path(output_dir).absolute()
    _fresh(output)
    if output == source or source in output.parents:
        raise ValueError("Academic supplements cannot be written inside historical experiments")
    protected, predictions, metrics = {}, [], []
    for experiment, identifier in EXPERIMENTS.items():
        folder = source / identifier
        report_path = folder / "experiment.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protected[report_path] = sha256(report_path)
        for name, digest in report["output_sha256"].items():
            if not _safe_name(name) or sha256(folder / name) != digest:
                raise ValueError(f"Historical output hash mismatch: {identifier}/{name}")
            protected[folder / name] = digest
        if not {"metrics.csv", "predictions.csv"}.issubset(report["output_sha256"]):
            raise ValueError("Historical reports must certify both scores and predictions")
        predictions.append(_normalize(pd.read_csv(folder / "predictions.csv"), experiment))
        metrics.append(_normalize(pd.read_csv(folder / "metrics.csv"), experiment))
    scores = derive_metrics(pd.concat(predictions, ignore_index=True))
    originals = pd.concat(metrics, ignore_index=True)
    if len(scores) != len(originals):
        raise ValueError("Academic groups do not cover exactly the saved historical metric rows")
    enrich_metrics(originals, scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".academic-", dir=output.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        scores.to_csv(stage / "metrics.csv", index=False, lineterminator="\n")
        manifest = {"schema_version": 1, "bundle_id": "brent-academic-v1",
                    "artifact_kind": "additional_scores_from_frozen_predictions",
                    "r2_formula": "1 - sum((y_pred-y_true)^2) / sum((y_true-mean(y_true))^2)",
                    "undefined_r2": "Missing when n < 2 or target variance is zero; never replaced by zero or one",
                    "aggregation": "Separate experiment, phase, model, variant, seed, horizon; phase totals recompute over pairs",
                    "original_results_modified": False, "models_trained": False,
                    "source_files": {str(path.relative_to(source)).replace("\\", "/"): digest
                                     for path, digest in sorted(protected.items())},
                    "row_counts": {"metrics.csv": len(scores)},
                    "files": {"metrics.csv": {"sha256": sha256(stage / "metrics.csv"),
                                               "size": (stage / "metrics.csv").stat().st_size}}}
        (stage / MANIFEST).write_bytes(_json(manifest))
        _manifest(stage)
        if any(sha256(path) != digest for path, digest in protected.items()):
            raise ValueError("Academic derivation changed a historical source")
        _fresh(output)
        stage.rename(output)
    return {"bundle_id": manifest["bundle_id"], "metric_rows": len(scores), "manifest": manifest}
