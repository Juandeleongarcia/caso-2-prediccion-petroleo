"""Frozen empirical, horizon-specific intervals from prequential calibration.

Errors are measured in original USD/barrel prices. The finite-sample order
statistic is explicit, but overlapping financial forecasts are not exchangeable:
these are nominal empirical bands, not a guarantee of 80% future coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


GROUPS = ("model", "variant", "seed", "horizon")
HORIZONS = (5, 63, 252)
METHOD = "prequential_absolute_error_order_statistic"


def _coverage(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Interval coverage must be strictly between zero and one")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Interval coverage must be numeric") from error
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError("Interval coverage must be strictly between zero and one")
    return result


def _deadline(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Calibration deadline requires an explicit timezone")
    return result.tz_convert("UTC")


def _release_times(values: pd.Series) -> pd.Series:
    if any(pd.isna(value) or pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError("Publication and decision timestamps require explicit timezones")
    return pd.to_datetime(values, utc=True, errors="raise")


def _groups(frame: pd.DataFrame) -> None:
    if not set(GROUPS).issubset(frame.columns) or frame[list(GROUPS)].isna().any().any():
        raise ValueError("Interval groups require model, variant, seed and horizon")
    if any(not isinstance(value, str) or not value for name in ["model", "variant"] for value in frame[name]):
        raise ValueError("Interval model and variant identifiers must be nonempty strings")
    for name in ["seed", "horizon"]:
        values = frame[name]
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("Interval seed and horizon identifiers must be integers")
        numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numbers).all() or not np.equal(numbers, np.floor(numbers)).all():
            raise ValueError("Interval seed and horizon identifiers must be integers")
    if not frame["horizon"].isin(HORIZONS).all():
        raise ValueError("Intervals support the frozen horizons 5, 63 and 252")


def _prices(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Missing original-unit interval columns: {list(columns)}")
    try:
        numbers = frame[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Interval prices must be finite numeric values") from error
    if not np.isfinite(numbers).all():
        raise ValueError("Interval prices must be finite numeric values")
    return numbers


def fit_intervals(
    calibration_predictions: pd.DataFrame,
    coverage: float = 0.8,
    available_before: object = "2024-01-01T08:00:00+01:00",
) -> dict:
    """Fit separate |price error| quantiles without pooling seeds or horizons.

    Each row must be a forecast issued during calibration origins 2021/2022.
    All labels must have been available strictly before the frozen test start.
    Upstream code is responsible for generating these forecasts out of sample
    under the frozen prequential training policy; this function never trains.
    Rank is min(n, ceil((n + 1) * coverage)), one-based, without interpolation.
    """
    level = _coverage(coverage)
    deadline = _deadline(available_before)
    frame = calibration_predictions.copy()
    required = {"phase", "origin_date", "label_available_at", "sample_id", "y_true", "y_pred"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Calibration needs nonempty original-unit prediction pairs and label availability")
    if not frame["phase"].eq("calibration").all():
        raise ValueError("Intervals must be fitted on calibration only, never development or test")
    _groups(frame)
    if frame["sample_id"].isna().any() or frame.duplicated([*GROUPS, "sample_id"]).any():
        raise ValueError("Duplicate or missing calibration prediction-pair identifiers")
    origins = pd.to_datetime(frame["origin_date"], errors="raise")
    if origins.isna().any() or not origins.dt.year.isin([2021, 2022]).all():
        raise ValueError("Calibration origins must remain in the frozen years 2021 and 2022")
    available = _release_times(frame["label_available_at"])
    if available.isna().any() or not available.lt(deadline).all():
        raise ValueError("All calibration labels must mature before the frozen test deadline")
    if "decision_at" in frame:
        decisions = _release_times(frame["decision_at"])
        if decisions.isna().any() or not decisions.lt(available).all():
            raise ValueError("Calibration residual labels must be unknown at the forecast decision")
    if "target_date" in frame:
        observed = pd.to_datetime(frame["target_date"], errors="raise")
        release_dates = available.dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize()
        if observed.isna().any() or (release_dates < observed).any():
            raise ValueError("Calibration label publication cannot precede target observation")
    prices = _prices(frame, ["y_true", "y_pred"])
    with np.errstate(over="raise", invalid="raise"):
        try:
            error = prices[:, 1] - prices[:, 0]
        except FloatingPointError as failure:
            raise ValueError("Calibration price errors must be finite") from failure
    if "error" in frame and not np.allclose(_prices(frame, ["error"])[:, 0], error, rtol=1e-10, atol=1e-10):
        raise ValueError("Stored calibration errors differ from original-unit prices")
    if {"reference_brent", "predicted_change"}.issubset(frame.columns):
        reconstructed = _prices(frame, ["reference_brent", "predicted_change"]).sum(axis=1)
        if not np.isfinite(reconstructed).all() or not np.allclose(prices[:, 1], reconstructed, rtol=1e-10, atol=1e-10):
            raise ValueError("Calibration residual price reconstruction is inconsistent")
    frame["_absolute_error"] = np.abs(error)
    frame["_available_at"] = available
    states = []
    for key, part in frame.groupby(list(GROUPS), sort=True, observed=True):
        n = len(part)
        rank = min(n, math.ceil((n + 1) * level))
        q = float(np.sort(part["_absolute_error"].to_numpy(dtype=np.float64))[rank - 1])
        model, variant, seed, horizon = key
        states.append({"model": model, "variant": variant, "seed": int(seed), "horizon": int(horizon),
                       "n": n, "rank": rank, "q": q,
                       "latest_label_available_at": part["_available_at"].max().isoformat()})
    return {"schema_version": 1, "method": METHOD, "coverage": level, "unit": "USD/barrel",
            "calibration_years": [2021, 2022], "available_before": deadline.isoformat(),
            "quantile_rule": "min(n, ceil((n+1)*coverage)), one-based sorted absolute price errors",
            "limitations": "Temporal dependence and regime changes prevent an exact coverage guarantee; seeds are not independent observations",
            "groups": states}


def _state_lookup(state: dict) -> tuple[float, dict]:
    if (state.get("schema_version") != 1 or state.get("method") != METHOD
            or state.get("unit") != "USD/barrel" or state.get("calibration_years") != [2021, 2022]):
        raise ValueError("Unrecognized frozen interval state")
    level = _coverage(state.get("coverage"))
    deadline = _deadline(state.get("available_before"))
    rows = state.get("groups")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Interval state needs nonempty calibration groups")
    table = pd.DataFrame(rows)
    _groups(table)
    if not {"n", "rank", "q", "latest_label_available_at"}.issubset(table.columns):
        raise ValueError("Incomplete interval calibration state")
    if table.duplicated(list(GROUPS)).any():
        raise ValueError("Duplicate groups in interval state")
    lookup = {}
    for row in rows:
        n, rank, q = row["n"], row["rank"], row["q"]
        if (type(n) is not int or n < 1 or type(rank) is not int
                or rank != min(n, math.ceil((n + 1) * level))
                or isinstance(q, bool) or not isinstance(q, (int, float)) or not np.isfinite(q) or q < 0):
            raise ValueError("Invalid interval quantile, rank or calibration sample count")
        if _deadline(row["latest_label_available_at"]) >= deadline:
            raise ValueError("Interval state contains a label unavailable before test")
        key = (row["model"], row["variant"], int(row["seed"]), int(row["horizon"]))
        lookup[key] = float(q)
    return level, lookup


def apply_intervals(predictions: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Attach frozen symmetric bands using point predictions only, never truth.

    Negative lower limits are retained: clipping would change the calibrated
    method. Updating observed test prices cannot alter a band or its state.
    """
    level, lookup = _state_lookup(state)
    _groups(predictions)
    if "phase" not in predictions or not predictions["phase"].isin(["calibration", "test", "live"]).all():
        raise ValueError("Frozen intervals apply only to calibration, test or live predictions")
    point = _prices(predictions, ["y_pred"])[:, 0]
    quantiles = []
    for model, variant, seed, horizon in predictions[list(GROUPS)].itertuples(index=False, name=None):
        key = (model, variant, int(seed), int(horizon))
        if key not in lookup:
            raise ValueError(f"No frozen calibration quantile for {key}")
        quantiles.append(lookup[key])
    q = np.asarray(quantiles, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            lower, upper = point - q, point + q
        except FloatingPointError as failure:
            raise ValueError("Interval endpoints must be finite") from failure
    if not np.isfinite(np.r_[lower, upper]).all() or (lower > point).any() or (point > upper).any():
        raise ValueError("Incoherent or non-finite interval endpoints")
    result = predictions.copy()
    result["lower"], result["upper"], result["interval_level"] = lower, upper, level
    return result


def interval_metrics(
    predictions_with_intervals: pd.DataFrame,
    group_by: Sequence[str] = ("phase", "model", "variant", "seed", "horizon"),
) -> pd.DataFrame:
    """Measure empirical coverage, width and Winkler score without refitting."""
    frame = predictions_with_intervals
    groups = tuple(group_by)
    if not groups or len(set(groups)) != len(groups) or not set(groups).issubset(frame.columns):
        raise ValueError("Interval metrics need distinct existing grouping columns")
    numbers = _prices(frame, ["y_true", "y_pred", "lower", "upper", "interval_level"])
    truth, point, lower, upper, levels = numbers.T
    if ((lower > point).any() or (point > upper).any() or (levels <= 0).any() or (levels >= 1).any()):
        raise ValueError("Interval metrics require coherent endpoints and nominal coverage")
    columns = [*groups, "n", "coverage", "nominal_coverage", "mean_width", "median_width", "interval_score"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if "phase" in frame and frame["phase"].nunique() > 1 and "phase" not in groups:
        raise ValueError("Interval metrics must not mix calibration and test phases")
    records = []
    for key, part in frame.groupby(list(groups), sort=True, dropna=False, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        values = _prices(part, ["y_true", "lower", "upper", "interval_level"])
        observed, lo, hi, nominal = values.T
        if not np.equal(nominal, nominal[0]).all():
            raise ValueError("Each metric group must have one frozen nominal interval level")
        alpha = 1 - nominal[0]
        width = hi - lo
        score = width + 2 / alpha * np.maximum(lo - observed, 0) + 2 / alpha * np.maximum(observed - hi, 0)
        if not np.isfinite(np.r_[width, score]).all():
            raise ValueError("Non-finite interval width or score")
        records.append(dict(zip(groups, keys)) | {"n": len(part),
                       "coverage": float(np.mean((lo <= observed) & (observed <= hi))),
                       "nominal_coverage": float(nominal[0]), "mean_width": float(width.mean()),
                       "median_width": float(np.median(width)), "interval_score": float(score.mean())})
    return pd.DataFrame(records, columns=columns)
