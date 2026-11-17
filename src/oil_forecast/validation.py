"""One expanding, semiannual protocol shared by persistence and future networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .supervised import SupervisedDataset
from .temporal import TIMEZONE, decision_times, eligible_training_labels, evaluation_schedule


@dataclass
class TemporalFold:
    block_id: str
    phase: str
    refit_at: pd.Timestamp
    training_cutoff: pd.Timestamp
    train_indices: np.ndarray
    eval_indices: np.ndarray
    fit_row_indices: np.ndarray
    summary: dict


def mature_samples(dataset: SupervisedDataset, cutoff: object) -> np.ndarray:
    """All three original labels must have become available by this exact instant."""
    samples = dataset.samples
    result = np.ones(len(samples), dtype=bool)
    for horizon in dataset.horizons:
        label = pd.DataFrame({"observed": True, "y_brent": samples[f"y_price_{horizon}"],
                              "label_available_at": samples[f"label_available_at_{horizon}"]})
        result &= eligible_training_labels(label, cutoff).to_numpy()
    return result


def build_folds(dataset: SupervisedDataset, config: dict) -> list[TemporalFold]:
    """Expand train and refit at every configured semester, never a static split."""
    samples = dataset.samples
    if samples.empty:
        raise ValueError("No complete supervised samples")
    schedule = evaluation_schedule(config).sort_values("origin_start")
    protocol = config["evaluation"]
    folds = []
    for row in schedule.itertuples(index=False):
        phase = row.phase
        half = 1 if row.origin_start.month == 1 else 2
        block_id = f"{phase}_{row.origin_start.year}H{half}"
        cutoff = pd.Timestamp(row.training_cutoff)
        if cutoff >= row.refit_at:
            raise ValueError("Training cutoff must precede refit")
        past_origins = samples["decision_at"].le(cutoff).to_numpy()
        matured = mature_samples(dataset, cutoff)
        train_indices = np.flatnonzero(past_origins & matured)
        in_block = samples["origin_date"].between(row.origin_start, row.origin_end).to_numpy()
        # Selection/calibration may only consume outcomes already known at their
        # frozen deadline, even if this snapshot contains later observations.
        if phase == "development":
            allowed_evaluation = mature_samples(dataset, protocol["selection_frozen_before"])
        elif phase == "calibration":
            allowed_evaluation = mature_samples(dataset, protocol["calibration_errors_available_before"])
        else:
            allowed_evaluation = np.ones(len(samples), dtype=bool)
            for horizon in dataset.horizons:
                allowed_evaluation &= samples[f"target_date_{horizon}"].le(
                    pd.Timestamp(protocol["test_outcomes_end"])).to_numpy()
        eval_indices = np.flatnonzero(in_block & allowed_evaluation)
        if not len(train_indices) or not len(eval_indices):
            raise ValueError(f"Empty training/evaluation block: {block_id}")
        if np.intersect1d(train_indices, eval_indices).size:
            raise ValueError("Train and evaluation origins overlap")
        fit_rows = np.unique(dataset.window_rows[train_indices])
        last_fit_decision_row = dataset.samples.iloc[train_indices]["window_end_row"].max()
        if fit_rows.max() > last_fit_decision_row:
            raise ValueError("Preprocessing rows extend beyond training windows")
        latest_label = max(samples.iloc[train_indices][f"label_available_at_{h}"].max()
                           for h in dataset.horizons)
        if latest_label > cutoff:
            raise ValueError("An immature label entered a training block")
        summary = {
            "block_id": block_id, "phase": phase,
            "training_cutoff": cutoff.isoformat(), "refit_at": row.refit_at.isoformat(),
            "origin_start": row.origin_start.strftime("%Y-%m-%d"),
            "origin_end": row.origin_end.strftime("%Y-%m-%d"),
            "n_calendar_origins": len(pd.bdate_range(row.origin_start, row.origin_end)),
            "n_train": len(train_indices), "n_fit_rows": len(fit_rows),
            "n_purged_immature": int((past_origins & ~matured).sum()),
            "n_eval_complete_case": int(in_block.sum()), "n_eval": len(eval_indices),
            "n_eval_deadline_excluded": int((in_block & ~allowed_evaluation).sum()),
            "first_train_origin": samples.iloc[train_indices]["origin_date"].min().strftime("%Y-%m-%d"),
            "last_train_origin": samples.iloc[train_indices]["origin_date"].max().strftime("%Y-%m-%d"),
            "latest_train_label_available_at": latest_label.isoformat(),
        }
        folds.append(TemporalFold(block_id, phase, row.refit_at, cutoff, train_indices,
                                  eval_indices, fit_rows, summary))
    return folds


@dataclass
class InnerSplit:
    """Training/validation indices available before one unchanged outer refit."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    fit_row_indices: np.ndarray
    training_cutoff: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    outer_training_cutoff: pd.Timestamp
    summary: dict


def build_inner_split(
    dataset: SupervisedDataset,
    outer_fold: TemporalFold,
    validation_business_days: int = 252,
) -> InnerSplit:
    """Take the latest mature origin year and purge by all three label releases.

    The inner validation tail is measured on the full Mon--Fri calendar, not on
    the filtered complete-case rows. Its labels must be known by the outer
    training cutoff. Inner training uses only labels already available before
    the first inner validation decision. No subsequent full-outer refit is
    implied: the inner validation labels remain outside gradient training.
    """
    if outer_fold.phase != "development":
        raise ValueError("Inner validation is restricted to development blocks")
    if type(validation_business_days) is not int or validation_business_days < 1:
        raise ValueError("validation_business_days must be a positive integer")
    samples = dataset.samples
    outer_indices = np.asarray(outer_fold.train_indices, dtype=np.int64)
    if (not len(outer_indices) or np.unique(outer_indices).size != len(outer_indices)
            or outer_indices.min() < 0 or outer_indices.max() >= len(samples)):
        raise ValueError("Outer training indices must be nonempty, unique and valid")
    outer_cutoff = pd.Timestamp(outer_fold.training_cutoff)
    if outer_cutoff.tzinfo is None:
        raise ValueError("Outer training cutoff needs an explicit timezone")
    outer_cutoff = outer_cutoff.tz_convert("UTC")
    outer_samples = samples.iloc[outer_indices]
    if (not outer_samples["decision_at"].le(outer_cutoff).all()
            or not mature_samples(dataset, outer_cutoff)[outer_indices].all()):
        raise ValueError("Outer training contains origins or labels unavailable at its cutoff")
    if np.intersect1d(outer_indices, outer_fold.eval_indices).size:
        raise ValueError("Outer evaluation labels cannot enter the inner split")
    # These are observation dates, not publication timestamps. Validate their
    # ordering explicitly before deriving either inner subset.
    for horizon in dataset.horizons:
        published = pd.to_datetime(outer_samples[f"label_available_at_{horizon}"], utc=True)
        published_dates = published.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
        if (published_dates < outer_samples[f"target_date_{horizon}"]).any():
            raise ValueError("A label availability precedes its target observation")

    validation_end = pd.Timestamp(outer_samples["origin_date"].max())
    validation_start = validation_end - pd.offsets.BDay(validation_business_days - 1)
    inner_cutoff = decision_times([validation_start - pd.offsets.BDay(1)])[0]
    if inner_cutoff >= outer_cutoff:
        raise ValueError("Inner training cutoff must precede the outer cutoff")
    in_validation = outer_samples["origin_date"].between(validation_start, validation_end).to_numpy()
    validation_indices = outer_indices[in_validation]
    train_allowed = (outer_samples["decision_at"].le(inner_cutoff).to_numpy()
                     & mature_samples(dataset, inner_cutoff)[outer_indices])
    train_indices = outer_indices[train_allowed]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Inner training and validation require nonempty mature samples")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Inner validation labels cannot enter training")
    inner_train = samples.iloc[train_indices]
    inner_validation = samples.iloc[validation_indices]
    latest_train_label = max(inner_train[f"label_available_at_{h}"].max() for h in dataset.horizons)
    latest_validation_label = max(inner_validation[f"label_available_at_{h}"].max() for h in dataset.horizons)
    if latest_train_label > inner_cutoff or latest_validation_label > outer_cutoff:
        raise ValueError("An inner subset contains immature labels")
    fit_rows = np.unique(dataset.window_rows[train_indices])
    purged = np.setdiff1d(outer_indices, np.concatenate([train_indices, validation_indices]))
    summary = {
        "block_id": outer_fold.block_id,
        "outer_training_cutoff": outer_cutoff.isoformat(),
        "inner_training_cutoff": inner_cutoff.isoformat(),
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "validation_end": validation_end.strftime("%Y-%m-%d"),
        "validation_business_days": validation_business_days,
        "n_outer_train": len(outer_indices),
        "n_inner_train": len(train_indices),
        "n_inner_validation": len(validation_indices),
        "n_purged_from_outer_train": len(purged),
        "n_fit_rows": len(fit_rows),
        "first_inner_train_origin": inner_train["origin_date"].min().strftime("%Y-%m-%d"),
        "last_inner_train_origin": inner_train["origin_date"].max().strftime("%Y-%m-%d"),
        "latest_inner_train_label_available_at": latest_train_label.isoformat(),
        "latest_inner_validation_label_available_at": latest_validation_label.isoformat(),
    }
    return InnerSplit(train_indices, validation_indices, fit_rows, inner_cutoff,
                      validation_start, validation_end, outer_cutoff, summary)
