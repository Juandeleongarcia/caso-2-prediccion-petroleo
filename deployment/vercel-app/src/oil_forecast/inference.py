"""Read-only forecasts from a verified final bundle, without future labels.

The serving checkpoint is selected by the frozen development protocol. Loading
or predicting never trains, refits scalers, chooses a model or recalibrates a
quantile. A historical evaluation checkpoint remains dated explicitly when
used with a newer window; this module does not silently update its weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .acquisition import sha256
from .models import build_model
from .preprocessing import InputTransformer, TargetTransformer
from .supervised import _booleans, _dates, _instants
from .temporal import TIMEZONE, decision_times, target_dates
from .training import predict_scaled
from .uncertainty import apply_intervals


def _instant(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} requires a finite timestamp with an explicit timezone")
    return timestamp.tz_convert("UTC")


class ForecastService:
    """Loaded CPU model and frozen uncertainty state for the future application.

    Construct through :func:`load_forecaster`. ``predict`` accepts exactly one
    unscaled 60-row window with the same provenance columns as features.csv.
    Missing covariates can be imputed using the saved training-only statistics;
    the last known Brent reference itself must be available and finite.
    """

    def __init__(self, *, model, inputs: InputTransformer, targets: TargetTransformer,
                 protocol: dict, entry: dict, intervals: dict, metrics: pd.DataFrame) -> None:
        self._model = model
        self._inputs = inputs
        self._targets = targets
        self._protocol = protocol
        self._entry = entry
        self._intervals = intervals
        self._metrics = metrics.copy(deep=True)
        self._features = tuple(protocol["feature_names"])
        self._refit_at = _instant(entry["refit_at"], "model refit_at")
        self._training_cutoff = _instant(entry["training_cutoff"], "training_cutoff")
        if self._training_cutoff >= self._refit_at:
            raise ValueError("The model training cutoff must precede its refit")

    def evaluation_metrics(self, phase: str | None = None) -> pd.DataFrame:
        """Return historical results as a copy; test metrics never select a model."""
        if phase not in (None, "calibration", "test"):
            raise ValueError("Historical final metrics have calibration or test phases")
        rows = self._metrics if phase is None else self._metrics.loc[self._metrics["phase"].eq(phase)]
        return rows.copy(deep=True)

    def predict(self, window: pd.DataFrame, data_updated_at: object | None = None) -> dict:
        """Return direct prices, changes and frozen bands for 5/63/252 weekdays.

        ``origin_date`` is the last decision date, whereas
        ``reference_observation_date`` dates the last Brent actually available
        under the declared release policy. No future targets are accepted or
        needed. ``data_updated_at`` dates source acquisition, not model fitting.
        """
        if not isinstance(window, pd.DataFrame) or len(window) != self._protocol["lookback"]:
            raise ValueError("Prediction requires one complete 60-row raw window")
        required = {"decision_date", "decision_at"}
        for name in self._features:
            required.update([name, f"{name}__observation_date", f"{name}__available_at",
                             f"{name}__is_available", f"{name}__is_stale"])
        if not required.issubset(window.columns) or window.columns.duplicated().any():
            raise ValueError("The window lacks the frozen feature/provenance schema")
        source = window.copy(deep=True).reset_index(drop=True)
        dates = _dates(source["decision_date"], "decision_date")
        expected = pd.bdate_range(dates.iloc[0], dates.iloc[-1]).as_unit("ns")
        if not pd.DatetimeIndex(dates).equals(expected):
            raise ValueError("Window dates must be ordered, unique and a complete Mon-Fri calendar")
        decisions = _instants(source["decision_at"], "decision_at")
        expected_decisions = decision_times(dates, self._protocol.get("decision_hour", 8))
        if decisions.isna().any() or not pd.DatetimeIndex(decisions).equals(expected_decisions):
            raise ValueError("Window decision timestamps must match the frozen Madrid decision hour")
        origin = dates.iloc[-1]
        decision = decisions.iloc[-1]
        if decision < self._refit_at:
            raise ValueError("A checkpoint cannot forecast an origin before its refit")
        values = pd.DataFrame({name: pd.to_numeric(source[name], errors="raise") for name in self._features})
        if np.isinf(values.to_numpy(dtype=np.float64)).any():
            raise ValueError("Prediction inputs cannot contain infinity")
        observed_dates, available_at = {}, {}
        for name in self._features:
            observed_dates[name] = _dates(source[f"{name}__observation_date"], f"{name}__observation_date", True)
            available_at[name] = _instants(source[f"{name}__available_at"], f"{name}__available_at")
            known = _booleans(source[f"{name}__is_available"], f"{name}__is_available")
            stale = _booleans(source[f"{name}__is_stale"], f"{name}__is_stale")
            if not values[name].notna().equals(known) or (known & stale).any():
                raise ValueError(f"Input availability/staleness masks disagree with {name}")
            if (observed_dates[name] > dates).any() or (available_at[name] > decisions).any():
                raise ValueError(f"Future information in input {name}")
            if (known & (observed_dates[name].isna() | available_at[name].isna())).any():
                raise ValueError(f"Missing provenance for available input {name}")
            published_dates = available_at[name].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
            if (published_dates < observed_dates[name]).any():
                raise ValueError(f"Input availability precedes its observation: {name}")
        reference = float(values["brent"].iloc[-1])
        if not np.isfinite(reference):
            raise ValueError("The last known Brent reference must be finite and available")
        updated = _instant(data_updated_at if data_updated_at is not None else self._protocol["data_updated_at"],
                           "data_updated_at")
        if updated < decision:
            raise ValueError("Source acquisition cannot precede the last supplied decision row")
        transformed = self._inputs.transform(values.to_numpy(dtype=np.float64)[None, :, :]).astype(np.float32)
        scaled = predict_scaled(self._model, transformed)
        changes = self._targets.inverse_transform(scaled)[0]
        prices = reference + changes
        horizons = tuple(self._protocol["horizons"])
        records = [{"phase": "live", "model": self._entry["model"], "variant": self._entry["variant"],
                    "seed": self._entry["seed"], "horizon": horizon, "y_pred": float(prices[index])}
                   for index, horizon in enumerate(horizons)]
        records += [{"phase": "live", "model": "persistence", "variant": "baseline", "seed": -1,
                     "horizon": horizon, "y_pred": reference} for horizon in horizons]
        bands = apply_intervals(pd.DataFrame(records), self._intervals)
        forecast_dates = target_dates(origin, horizons)
        forecasts = []
        for index, horizon in enumerate(horizons):
            network = bands.loc[bands["model"].eq(self._entry["model"]) & bands["horizon"].eq(horizon)].iloc[0]
            baseline = bands.loc[bands["model"].eq("persistence") & bands["horizon"].eq(horizon)].iloc[0]
            forecasts.append({"horizon": horizon, "target_date": forecast_dates[horizon].strftime("%Y-%m-%d"),
                              "price": float(prices[index]), "change": float(changes[index]),
                              "lower": float(network["lower"]), "upper": float(network["upper"]),
                              "persistence_price": reference, "persistence_change": 0.0,
                              "persistence_lower": float(baseline["lower"]),
                              "persistence_upper": float(baseline["upper"])})
        refit_date = self._refit_at.tz_convert(TIMEZONE).tz_localize(None).normalize()
        due = refit_date + pd.DateOffset(months=6)
        overdue = bool(origin >= due)
        return {"model": self._entry["model"], "variant": self._entry["variant"], "seed": self._entry["seed"],
                "origin_date": origin.strftime("%Y-%m-%d"), "decision_at": decision.isoformat(),
                "reference_brent": reference,
                "reference_observation_date": observed_dates["brent"].iloc[-1].strftime("%Y-%m-%d"),
                "reference_available_at": available_at["brent"].iloc[-1].isoformat(),
                "data_updated_at": updated.isoformat(), "model_refit_at": self._refit_at.isoformat(),
                "training_cutoff": self._training_cutoff.isoformat(),
                "model_age_business_days": int(np.busday_count(refit_date.date(), origin.date())),
                "model_retraining_due_date": due.strftime("%Y-%m-%d"),
                "model_is_past_scheduled_refit": overdue,
                "model_age_warning": ("El checkpoint histórico ha superado su cadencia semestral; "
                                      "usar datos nuevos no actualiza sus pesos." if overdue else None),
                "interval_level": float(self._intervals["coverage"]),
                "interval_method": self._intervals["method"],
                "interval_limitation": "Cobertura nominal empírica; sin garantía exacta bajo dependencia temporal o cambios de régimen.",
                "availability_mode": self._protocol.get("availability_mode"), "forecasts": forecasts}


def load_forecaster(run_directory: Path | str) -> ForecastService:
    """Load the preselected seed/checkpoint only after verifying bundle hashes."""
    directory = Path(run_directory).resolve()
    report = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    hashes = report.get("output_sha256", {})
    required = {"protocol.json", "checkpoints.pt", "intervals.json", "metrics.csv"}
    if not required.issubset(hashes):
        raise ValueError("The final report must hash the serving artifacts")
    for name, digest in hashes.items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError(f"Final bundle output hash mismatch: {name}")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    deployment = report["deployment"]
    if (protocol["lookback"] != 60 or protocol["horizons"] != [5, 63, 252]
            or protocol["variants"] != {"RNN": "D", "LSTM": "B", "GRU": "C"}
            or deployment["selected_model"] != protocol["selected_model"]
            or deployment["selected_seed"] != protocol["selected_seed"]):
        raise ValueError("Serving selection differs from the frozen final protocol")
    if protocol["selected_model"] not in protocol["variants"] or protocol["selected_seed"] != 42:
        raise ValueError("The serving model and canonical seed must be frozen before test")
    torch.set_num_threads(protocol.get("threads", 1))
    bundle = torch.load(directory / "checkpoints.pt", map_location="cpu", weights_only=True)
    entry = bundle["models"][deployment["checkpoint_key"]]
    architecture, seed = protocol["selected_model"], protocol["selected_seed"]
    if (entry["model"] != architecture or entry["seed"] != seed
            or entry["variant"] != protocol["variants"][architecture]
            or entry["phase"] != "test" or entry["block_id"] != "test_2024H2"
            or _instant(entry["refit_at"], "refit_at") != _instant(deployment["model_refit_at"], "model_refit_at")):
        raise ValueError("Serving checkpoint metadata disagrees with the frozen selection")
    model = build_model(entry["model_config"])
    inputs = InputTransformer.from_dict(entry["transformers"]["input"])
    targets = TargetTransformer.from_dict(entry["transformers"]["target"])
    if (tuple(protocol["feature_names"]) != inputs.feature_names_ or len(inputs.feature_names_) != 10
            or "brent" not in inputs.feature_names_ or model.input_size != 20
            or model.architecture != architecture or model.hidden_size != 32 or model.num_layers != 1):
        raise ValueError("Serving model and transformer feature schemas disagree")
    model.load_state_dict(entry["state_dict"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Serving checkpoint contains non-finite weights")
    model.eval()
    intervals = json.loads((directory / "intervals.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(directory / "metrics.csv")
    if "phase" not in metrics or not metrics["phase"].isin(["calibration", "test"]).all():
        raise ValueError("Final historical metrics must identify calibration and test")
    return ForecastService(model=model, inputs=inputs, targets=targets, protocol=protocol,
                           entry=entry, intervals=intervals, metrics=metrics)
