"""Persistence and errors in the original price units, grouped by evaluation block."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict:
    """Score finite price pairs, without broadcasting or pooling horizons.

    MAE/RMSE have the input price unit; MSE has its square. Inputs must already
    be in the original scale. Missing/non-finite pairs are counted explicitly.
    An empty eligible set has undefined errors, represented by None, not zero.
    """
    observed = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(predicted, dtype=np.float64)
    if observed.ndim != 1 or forecast.ndim != 1 or observed.shape != forecast.shape:
        raise ValueError("Actual and predicted must be one-dimensional arrays of identical shape")
    valid = np.isfinite(observed) & np.isfinite(forecast)
    n = int(valid.sum())
    result = {"n": n, "n_dropped": int(len(valid) - n), "mse": None, "mae": None, "rmse": None}
    if n:
        error = forecast[valid] - observed[valid]
        mse = float(np.mean(np.square(error)))
        result.update(mse=mse, mae=float(np.mean(np.abs(error))), rmse=math.sqrt(mse))
    return result


def persistence_predictions(
    reference: Sequence[float], horizons: Sequence[int] = (5, 63, 252)
) -> np.ndarray:
    """Repeat the last known Brent for every horizon; no parameters are fitted."""
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Reference must be a finite one-dimensional array of known prices")
    horizons = tuple(horizons)
    if not horizons or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
                           or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be distinct positive integers")
    return np.repeat(values[:, None], len(horizons), axis=1)


def score_predictions(
    predictions: pd.DataFrame,
    group_by: Sequence[str] = ("phase", "block_id", "horizon"),
) -> pd.DataFrame:
    """Score exported origin/target pairs without mixing evaluation phases.

    For phase totals call with group_by=("phase", "horizon"); the caller can
    label those rows block_id="all". Counts aggregate pairs, not block metrics.
    """
    groups = tuple(group_by)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("At least one distinct grouping column is required")
    missing = set(groups + ("y_true", "y_pred")) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    columns = [*groups, "n", "n_dropped", "mse", "mae", "rmse"]
    records = []
    for key, frame in predictions.groupby(list(groups), sort=True, dropna=False, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(groups, keys)) | regression_metrics(frame["y_true"], frame["y_pred"]))
    return pd.DataFrame(records, columns=columns)


def diagnostics(predictions: pd.DataFrame, horizons: Sequence[int] = (252,)) -> pd.DataFrame:
    """Describe original-unit prices and changes, by block and pooled development.

    A small predicted-change spread can flag shrinkage toward a typical change;
    it does not by itself prove reversion toward a historical *price* mean.
    Standard deviations use ddof=0. Undefined correlations/ratios remain missing.
    """
    groups = ["variant", "model", "seed", "block_id", "horizon"]
    required = set(groups + ["phase", "y_true", "y_pred", "reference_brent"])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    if not predictions["phase"].eq("development").all():
        raise ValueError("Diagnostics permit development only")
    if predictions["block_id"].eq("all").any():
        raise ValueError("Diagnostics require individual predictions, not pre-aggregated rows")
    horizons = tuple(horizons)
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("Diagnostic horizons must be positive integers")
    frame = predictions.loc[predictions["horizon"].isin(horizons)].copy()
    if frame[groups].isna().any().any():
        raise ValueError("Missing diagnostic group identifiers")
    numbers = frame[["y_true", "y_pred", "reference_brent"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError("Diagnostics require finite original-unit prices and references")
    if "sample_id" in frame and frame.duplicated(groups + ["sample_id"]).any():
        raise ValueError("Duplicate prediction pairs in diagnostics")

    statistics = ["n", "bias", "negative_predicted_prices"]
    for prefix in ["true_price", "pred_price"]:
        statistics += [f"{prefix}_{item}" for item in ["min", "max", "mean", "std"]]
    for prefix in ["true_delta", "pred_delta"]:
        statistics += [f"{prefix}_{item}" for item in ["mean", "std", "q05", "q50", "q95"]]
    statistics += ["delta_std_ratio", "delta_correlation"]
    records = []

    def describe(part: pd.DataFrame) -> dict:
        truth = part["y_true"].to_numpy(dtype=np.float64)
        predicted = part["y_pred"].to_numpy(dtype=np.float64)
        reference = part["reference_brent"].to_numpy(dtype=np.float64)
        actual_delta, predicted_delta = truth - reference, predicted - reference
        result = {"n": len(part), "bias": float(np.mean(predicted - truth)),
                  "negative_predicted_prices": int(np.sum(predicted < 0))}
        for prefix, values in [("true_price", truth), ("pred_price", predicted)]:
            result.update({f"{prefix}_min": float(values.min()), f"{prefix}_max": float(values.max()),
                           f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std(ddof=0))})
        for prefix, values in [("true_delta", actual_delta), ("pred_delta", predicted_delta)]:
            quantiles = np.quantile(values, [0.05, 0.5, 0.95])
            result.update({f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std(ddof=0)),
                           f"{prefix}_q05": float(quantiles[0]), f"{prefix}_q50": float(quantiles[1]),
                           f"{prefix}_q95": float(quantiles[2])})
        true_std, pred_std = result["true_delta_std"], result["pred_delta_std"]
        result["delta_std_ratio"] = pred_std / true_std if true_std > 0 else None
        result["delta_correlation"] = float(np.corrcoef(actual_delta, predicted_delta)[0, 1]) \
            if len(part) > 1 and true_std > 0 and pred_std > 0 else None
        return result

    for key, part in frame.groupby(groups, sort=True, observed=True):
        records.append(dict(zip(groups, key)) | {"phase": "development"} | describe(part))
    total_groups = ["variant", "model", "seed", "horizon"]
    for key, part in frame.groupby(total_groups, sort=True, observed=True):
        records.append(dict(zip(total_groups, key)) | {"block_id": "all", "phase": "development"} | describe(part))
    return pd.DataFrame(records, columns=[*groups, "phase", *statistics])


def select_development_configs(metrics: pd.DataFrame) -> dict[str, str]:
    """Select A/B using seed 42 and equal weight for each of six blocks × 3 h.

    The score is mean(MAE_model / MAE_persistence), derived from skill_mae_pct.
    Pooled rows and later stability seeds never affect this initial selection.
    Exact ties resolve to A. This development choice is not an unbiased estimate
    of final generalization and never consumes calibration/test results.
    """
    required = {"phase", "model", "variant", "seed", "block_id", "horizon", "skill_mae_pct"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")
    if not metrics["phase"].eq("development").all():
        raise ValueError("Configuration selection permits development only")
    candidates = metrics.loc[~metrics["model"].eq("persistence") & ~metrics["block_id"].eq("all")].copy()
    candidates = candidates.loc[pd.to_numeric(candidates["seed"], errors="raise").eq(42)].copy()
    if candidates.empty:
        raise ValueError("Selection requires seed 42 A/B development metrics")
    if set(candidates["model"]) != {"RNN", "LSTM", "GRU"} or set(candidates["variant"]) != {"A", "B"}:
        raise ValueError("Selection requires the three architectures and exactly variants A/B")
    if candidates[["model", "variant", "block_id", "horizon"]].isna().any().any():
        raise ValueError("Missing configuration-selection identifiers")
    if candidates.duplicated(["model", "variant", "block_id", "horizon"]).any():
        raise ValueError("Duplicate configuration-selection metric rows")
    blocks = set(candidates["block_id"])
    if len(blocks) != 6 or set(candidates["horizon"]) != {5, 63, 252}:
        raise ValueError("Selection requires the same six development blocks and three horizons")
    ratio = 1 - pd.to_numeric(candidates["skill_mae_pct"], errors="raise") / 100
    if not np.isfinite(ratio.to_numpy()).all() or ratio.lt(0).any():
        raise ValueError("MAE-relative selection scores must be finite and nonnegative")
    candidates["relative_mae"] = ratio
    if "n" in candidates:
        if candidates["n"].isna().any() or candidates["n"].le(0).any():
            raise ValueError("Every selection cell needs valid price pairs")
        if candidates.groupby(["block_id", "horizon"], observed=True)["n"].nunique().gt(1).any():
            raise ValueError("Configurations must use the same prediction-pair counts")
    expected = {(block, horizon) for block in blocks for horizon in (5, 63, 252)}
    selected = {}
    for model, architecture in candidates.groupby("model", sort=True, observed=True):
        scores = []
        for variant in ("A", "B"):
            rows = architecture.loc[architecture["variant"].eq(variant)]
            if set(map(tuple, rows[["block_id", "horizon"]].to_numpy())) != expected:
                raise ValueError("Incomplete block/horizon grid for a candidate configuration")
            scores.append((float(rows["relative_mae"].mean()), variant))
        selected[model] = min(scores)[1]
    return selected


def assess_development_refinement(
    metrics: pd.DataFrame, model: str, variant: str, seeds: Sequence[int] = (42,),
    minimum_gain: float = 0.05, minimum_blocks: int = 4, minimum_horizons: int = 2,
) -> dict:
    """Apply a fixed, paired development gate against the original B variant.

    The score averages MAE/persistence MAE equally across six blocks and three
    horizons, then across the requested seeds. Stability also requires a lower
    score for every seed; seeds are never selected individually. This is an
    exploratory development decision, not a final generalization estimate.
    """
    required = {"phase", "variant", "model", "seed", "block_id", "horizon",
                "mae", "skill_mae_pct", "n"}
    if not required.issubset(metrics.columns) or not metrics["phase"].eq("development").all():
        raise ValueError("Refinement comparison requires development metrics only")
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds) or variant not in {"C", "D"}
            or model not in {"RNN", "LSTM", "GRU"}):
        raise ValueError("Invalid refinement candidate or seed set")
    if not 0 <= minimum_gain < 1 or not 1 <= minimum_blocks <= 6 or not 1 <= minimum_horizons <= 3:
        raise ValueError("Invalid fixed refinement thresholds")
    rows = metrics.loc[metrics["model"].eq(model) & metrics["variant"].isin(["B", variant])
                       & metrics["seed"].isin(seeds)].copy()
    if rows.duplicated(["variant", "seed", "block_id", "horizon"]).any():
        raise ValueError("Duplicate refinement metric cells")
    blocks = {f"development_{year}H{half}" for year in (2015, 2017, 2019) for half in (1, 2)}
    expected = {(v, s, b, h) for v in ("B", variant) for s in seeds
                for b in [*blocks, "all"] for h in (5, 63, 252)}
    if set(map(tuple, rows[["variant", "seed", "block_id", "horizon"]].to_numpy())) != expected:
        raise ValueError("Refinement comparison needs the exact six blocks, three horizons and seeds")
    numbers = rows[["mae", "skill_mae_pct", "n"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numbers).all() or rows["mae"].lt(0).any() or rows["n"].le(0).any():
        raise ValueError("Refinement metrics must be finite with positive pair counts")
    if rows.groupby(["block_id", "horizon"], observed=True)["n"].nunique().gt(1).any():
        raise ValueError("Refinement and B must have equal pair counts")
    rows["relative_mae"] = 1 - rows["skill_mae_pct"] / 100
    if rows["relative_mae"].lt(0).any():
        raise ValueError("Relative MAE cannot be negative")
    cells = rows.loc[~rows["block_id"].eq("all")]
    scores = cells.groupby("variant", observed=True)["relative_mae"].mean()
    if scores["B"] <= 0:
        raise ValueError("A perfect B score cannot define a relative improvement")
    gain = 1 - scores[variant] / scores["B"]
    per_block = cells.groupby(["variant", "block_id"], observed=True)["relative_mae"].mean().unstack(0)
    totals = rows.loc[rows["block_id"].eq("all")]
    per_horizon = totals.groupby(["variant", "horizon"], observed=True)["mae"].mean().unstack(0)
    per_seed = cells.groupby(["variant", "seed"], observed=True)["relative_mae"].mean().unstack(0)
    improved_blocks = int(per_block[variant].lt(per_block["B"]).sum())
    improved_horizons = int(per_horizon[variant].lt(per_horizon["B"]).sum())
    all_seeds_improve = bool(per_seed[variant].lt(per_seed["B"]).all())
    return {"model": model, "variant": variant, "seeds": list(seeds),
            "reference_score": float(scores["B"]), "candidate_score": float(scores[variant]),
            "relative_gain_pct": float(100 * gain), "improved_blocks": improved_blocks,
            "improved_horizons": improved_horizons, "all_seeds_improve": all_seeds_improve,
            "passes": bool(gain >= minimum_gain and improved_blocks >= minimum_blocks
                           and improved_horizons >= minimum_horizons and all_seeds_improve),
            "scores_by_seed": [{"seed": int(seed), "B": float(row["B"]),
                                variant: float(row[variant])} for seed, row in per_seed.iterrows()]}


def freeze_final_choices(development_metrics: pd.DataFrame, development_runs: pd.DataFrame) -> dict:
    """Freeze app network and epoch counts using selected development runs only.

    Reuse equal block/horizon/seed relative MAE, with alphabetical tie-breaking.
    Epoch count is ceil(median(best_epoch)) across 6 blocks x 3 seeds, not the
    best seed. Operational seed 42 is a convention fixed before final results.
    """
    variants = {"RNN": "D", "LSTM": "B", "GRU": "C"}
    if ("phase" not in development_metrics or not development_metrics["phase"].eq("development").all()
            or ("phase" in development_runs and not development_runs["phase"].eq("development").all())):
        raise ValueError("Final choices must use development exclusively")
    blocks = {f"development_{year}H{half}" for year in (2015, 2017, 2019) for half in (1, 2)}
    expected_runs = {(seed, block) for seed in (42, 123, 2026) for block in blocks}
    expected_cells = {(seed, block, horizon) for seed, block in expected_runs for horizon in (5, 63, 252)}
    scores, epochs, details = {}, {}, {}
    for model, variant in variants.items():
        cells = development_metrics.loc[development_metrics["model"].eq(model)
            & development_metrics["variant"].eq(variant) & ~development_metrics["block_id"].eq("all")].copy()
        runs = development_runs.loc[development_runs["model"].eq(model)
                                    & development_runs["variant"].eq(variant)]
        if (cells.duplicated(["seed", "block_id", "horizon"]).any()
                or set(map(tuple, cells[["seed", "block_id", "horizon"]].to_numpy())) != expected_cells
                or runs.duplicated(["seed", "block_id"]).any()
                or set(map(tuple, runs[["seed", "block_id"]].to_numpy())) != expected_runs):
            raise ValueError("Selected development results need all six blocks and three seeds")
        best = runs["best_epoch"].to_numpy(dtype=float)
        relative = 1 - cells["skill_mae_pct"].to_numpy(dtype=float) / 100
        if (not np.isfinite(best).all() or not np.equal(best, np.floor(best)).all()
                or (best < 1).any() or (best > 20).any() or not np.isfinite(relative).all()
                or (relative < 0).any()):
            raise ValueError("Invalid development epochs or relative MAE")
        scores[model] = float(relative.mean())
        epochs[model] = int(math.ceil(float(np.median(best))))
        details[model] = {"variant": variant, "n_runs": len(runs), "median_best_epoch": float(np.median(best)),
                          "score": scores[model]}
    selected = min(scores, key=lambda model: (scores[model], model))
    return {"variants": variants, "epochs": epochs, "selected_model": selected, "selected_seed": 42,
            "development_selection": details,
            "epoch_rule": "ceil(median of 18 selected development best epochs per architecture)",
            "selection_rule": "lowest mean MAE/persistence MAE, equal weight for 6 development blocks x 3 horizons x 3 seeds; alphabetical tie",
            "seed_rule": "42 operational convention; 123/2026 stability only, never choose a seed from final scores"}


def score_final_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Monetary metrics by phase, model, seed, horizon and semester or pooled."""
    if predictions.empty or not predictions["phase"].isin(["calibration", "test"]).all():
        raise ValueError("Final scoring requires calibration/test predictions")
    identifiers = ["phase", "model", "variant", "seed", "block_id", "horizon"]
    if predictions.duplicated(["model", "variant", "seed", "sample_id", "horizon"]).any():
        raise ValueError("Duplicate final prediction pairs")
    cells = score_predictions(predictions, group_by=identifiers)
    pooled_keys = [name for name in identifiers if name != "block_id"]
    pooled = score_predictions(predictions, group_by=pooled_keys)
    pooled["block_id"] = "all"
    scored = pd.concat([cells, pooled], ignore_index=True)
    if scored["n_dropped"].any():
        raise ValueError("Final scoring cannot silently drop non-finite prices")
    error = predictions.assign(_error=predictions["y_pred"] - predictions["y_true"])
    bias_cells = error.groupby(identifiers, observed=True)["_error"].mean()
    bias_pooled = error.groupby(pooled_keys, observed=True)["_error"].mean()
    scored["bias"] = [float(bias_pooled.loc[tuple(row[name] for name in pooled_keys)])
                       if row["block_id"] == "all" else float(bias_cells.loc[tuple(row[name] for name in identifiers)])
                       for row in scored.to_dict(orient="records")]
    baseline = scored.loc[scored["model"].eq("persistence")].set_index(["phase", "block_id", "horizon"])
    if baseline.index.has_duplicates:
        raise ValueError("Persistence must have exactly one row per phase/block/horizon")
    scored["skill_mae_pct"] = [100 * (1 - row["mae"] / baseline.loc[(row["phase"], row["block_id"], row["horizon"]), "mae"])
                               for row in scored.to_dict(orient="records")]
    return scored
