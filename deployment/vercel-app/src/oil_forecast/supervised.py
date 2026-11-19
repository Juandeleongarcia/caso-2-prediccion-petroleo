"""Supervised windows with exact calendar labels and an explicit common sample set."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .temporal import HORIZONS, TIMEZONE, target_dates


@dataclass
class SupervisedDataset:
    X: np.ndarray
    y_changes: np.ndarray
    samples: pd.DataFrame
    exclusions: pd.DataFrame
    feature_values: pd.DataFrame
    window_rows: np.ndarray
    feature_names: tuple[str, ...]
    horizons: tuple[int, ...]
    summary: dict


def _dates(values: pd.Series, name: str, allow_missing: bool = False) -> pd.Series:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if dates.tz is not None or (not allow_missing and dates.hasnans):
        raise ValueError(f"{name} must contain local dates without timezone")
    if not dates.equals(dates.normalize()):
        raise ValueError(f"{name} must contain dates without time")
    return pd.Series(dates.as_unit("ns"), index=values.index, name=name)


def _instants(values: pd.Series, name: str) -> pd.Series:
    for value in values.dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError(f"{name} requires an explicit timezone")
    return pd.to_datetime(values, errors="raise", utc=True, format="mixed").dt.as_unit("ns")


def _booleans(values: pd.Series, name: str) -> pd.Series:
    result = values.map({True: True, False: False, "True": True, "False": False})
    if result.isna().any():
        raise ValueError(f"{name} must be a boolean mask without missing values")
    return result.astype(bool)


def build_supervised(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: Sequence[str],
    lookback: int = 60,
    horizons: Sequence[int] = HORIZONS,
) -> SupervisedDataset:
    """Keep an origin only when its reference and every original target exist.

    Inputs remain unscaled and may contain missing features. A future fold must
    fit its imputer/scaler on the distinct rows used by its training windows.
    Label availability is metadata, not an input to the recurrent networks.
    """
    names, horizons = tuple(feature_names), tuple(horizons)
    if type(lookback) is not int or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if not names or len(names) != len(set(names)) or "brent" not in names:
        raise ValueError("Unique feature names including brent are required")
    if not horizons or len(horizons) != len(set(horizons)):
        raise ValueError("Distinct positive horizons are required")
    if features.empty:
        raise ValueError("No decision calendar is available")
    source = features.copy().reset_index(drop=True)
    dates = _dates(source["decision_date"], "decision_date")
    expected_dates = pd.bdate_range(dates.iloc[0], dates.iloc[-1]).as_unit("ns")
    if not pd.DatetimeIndex(dates).equals(expected_dates):
        raise ValueError("Decision dates must be ordered, unique and a complete Mon-Fri calendar")
    decisions = _instants(source["decision_at"], "decision_at")
    if decisions.isna().any() or not decisions.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize().equals(dates):
        raise ValueError("Decision timestamps must belong to their local decision dates")
    # Reuse the agreed business-day contract rather than shifting filtered prices.
    future_dates = [target_dates(date, horizons) for date in dates]
    values = pd.DataFrame({name: pd.to_numeric(source[name], errors="raise") for name in names})
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("Infinite input values are not permitted")
    observed_dates, available_at = {}, {}
    for name in names:
        observed_dates[name] = _dates(source[f"{name}__observation_date"], f"{name}__observation_date", True)
        available_at[name] = _instants(source[f"{name}__available_at"], f"{name}__available_at")
        known = _booleans(source[f"{name}__is_available"], f"{name}__is_available")
        stale = _booleans(source[f"{name}__is_stale"], f"{name}__is_stale")
        if not values[name].notna().equals(known) or (known & stale).any():
            raise ValueError(f"Input availability/staleness masks disagree with {name}")
        if (available_at[name] > decisions).any() or (observed_dates[name] > dates).any():
            raise ValueError(f"Future information in input {name}")
        if (known & (available_at[name].isna() | observed_dates[name].isna())).any():
            raise ValueError(f"Missing provenance for available input {name}")

    target = targets.copy().reset_index(drop=True)
    target["observation_date"] = _dates(target["observation_date"], "target observation_date")
    if target["observation_date"].duplicated().any() or not target["observation_date"].is_monotonic_increasing:
        raise ValueError("Target dates must be ordered and unique")
    if target["observation_date"].dt.day_of_week.ge(5).any():
        raise ValueError("Target calendar contains a weekend")
    target["y_brent"] = pd.to_numeric(target["y_brent"], errors="raise")
    target["observed"] = _booleans(target["observed"], "target observed")
    if np.isinf(target["y_brent"].to_numpy(dtype=float)).any() or not target["observed"].equals(target["y_brent"].notna()):
        raise ValueError("Original target values and observed mask disagree")
    target["label_available_at"] = _instants(target["label_available_at"], "label_available_at")
    published_dates = target["label_available_at"].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    if (published_dates < target["observation_date"]).any():
        raise ValueError("Target availability precedes its observation")
    indexed_target = target.set_index("observation_date")

    rows = np.arange(len(source), dtype=np.int64)
    metadata = pd.DataFrame({
        "origin_date": dates, "decision_at": decisions,
        "window_start_date": dates.shift(lookback - 1),
        "window_start_row": rows - lookback + 1, "window_end_row": rows,
        "reference_brent": values["brent"],
        "reference_observation_date": observed_dates["brent"],
        "reference_available_at": available_at["brent"],
    })
    missing_labels, missing_availability = {}, {}
    for horizon in horizons:
        dates_h = pd.Series([item[horizon] for item in future_dates])
        prices_h = dates_h.map(indexed_target["y_brent"])
        label_observed = dates_h.map(indexed_target["observed"]).fillna(False).astype(bool)
        metadata[f"target_date_{horizon}"] = dates_h
        metadata[f"y_price_{horizon}"] = prices_h
        metadata[f"y_change_{horizon}"] = prices_h - metadata["reference_brent"]
        metadata[f"label_available_at_{horizon}"] = dates_h.map(indexed_target["label_available_at"])
        metadata[f"availability_basis_{horizon}"] = dates_h.map(indexed_target["availability_basis"])
        missing_labels[horizon] = ~label_observed | prices_h.isna()
        missing_availability[horizon] = label_observed & prices_h.notna() & metadata[f"label_available_at_{horizon}"].isna()

    reasons = []
    for row in rows:
        if row < lookback - 1:
            reasons.append("warmup")
            continue
        rejected = []
        if pd.isna(metadata.loc[row, "reference_brent"]):
            rejected.append("missing_reference")
        for horizon in horizons:
            if missing_labels[horizon].iloc[row]:
                rejected.append(f"missing_label_{horizon}")
            elif missing_availability[horizon].iloc[row]:
                rejected.append(f"missing_label_availability_{horizon}")
        reasons.append(";".join(rejected))
    reason = pd.Series(reasons, name="reason")
    accepted = reason.eq("")
    samples = metadata.loc[accepted].reset_index(drop=True)
    samples.insert(0, "sample_id", np.arange(len(samples), dtype=np.int64))
    exclusions = pd.concat([metadata.loc[~accepted].reset_index(drop=True), reason.loc[~accepted].reset_index(drop=True)], axis=1)
    end_rows = samples["window_end_row"].to_numpy(dtype=np.int64)
    window_rows = end_rows[:, None] - np.arange(lookback - 1, -1, -1, dtype=np.int64)[None, :]
    X = values.to_numpy(dtype=float)[window_rows]
    y_changes = samples[[f"y_change_{horizon}" for horizon in horizons]].to_numpy(dtype=float)
    candidate = rows >= lookback - 1
    summary = {
        "calendar_rows": len(source), "lookback": lookback, "horizons": list(horizons),
        "candidate_windows": int(candidate.sum()), "warmup_excluded": int((~candidate).sum()),
        "valid_examples": len(samples), "excluded_candidates": int(candidate.sum()) - len(samples),
        "missing_reference": int((candidate & metadata["reference_brent"].isna()).sum()),
        "missing_labels_by_horizon": {str(h): int((candidate & mask).sum()) for h, mask in missing_labels.items()},
        "missing_label_availability_by_horizon": {str(h): int((candidate & mask).sum()) for h, mask in missing_availability.items()},
        "exclusion_patterns": {str(key): int(value) for key, value in reason.loc[~accepted].value_counts().items()},
        "exclusion_policy": "complete-case: all original target prices and availability metadata required; input NaNs kept",
    }
    return SupervisedDataset(X, y_changes, samples, exclusions, values, window_rows, names, horizons, summary)
