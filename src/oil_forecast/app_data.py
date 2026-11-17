"""Read-only application data and explicit, separate market-data refreshes."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from .acquisition import download_snapshot, load_snapshot, sha256
from .academic import enrich_metrics
from .artifacts import _manifest
from .inference import ForecastService, load_forecaster
from .models import build_model
from .preprocessing import InputTransformer, TargetTransformer
from .supervised import _booleans, build_supervised
from .temporal import apply_availability, build_features, build_target, target_dates

VARIANTS = {"RNN": "D", "LSTM": "B", "GRU": "C", "persistence": "baseline"}
LABELS = {"RNN-D": "RNN", "LSTM-B": "LSTM", "GRU-C": "GRU", "Persistencia": "persistence"}


def load_app_data(bundle_dir: Path | str, dashboard_dir: Path | str | None = None,
                  academic_dir: Path | str | None = None) -> dict:
    """Validate payload hashes and load frames without accessing old experiments."""
    folder = Path(bundle_dir)
    manifest = _manifest(folder)
    result = {"manifest": manifest, "protocol": json.loads((folder / "protocol.json").read_text(encoding="utf-8")),
              "history": pd.read_csv(folder / "brent_history.csv"), "window": pd.read_csv(folder / "latest_window.csv"),
              "metrics": pd.read_csv(folder / "metrics.csv"),
              "interval_metrics": pd.read_csv(folder / "interval_metrics.csv"), "evaluation": {}}
    if dashboard_dir is not None:
        directory = Path(dashboard_dir)
        meta = _manifest(directory)
        files = meta["files"]
        if not files:
            raise ValueError("The evaluation supplement has no verified files")
        for name, info in files.items():
            expected = info.get("sha256") if isinstance(info, dict) else info
            if Path(name).name != name or sha256(directory / name) != expected:
                raise ValueError(f"Evaluation artifact integrity failure: {name}")
            if name.endswith(".csv"):
                result["evaluation"][Path(name).stem] = pd.read_csv(directory / name)
            elif name == "experiment_catalog.json":
                result["evaluation"]["experiment_catalog"] = json.loads((directory / name).read_text(encoding="utf-8"))
        result["evaluation_manifest"] = meta
    academic = (Path(academic_dir) if academic_dir is not None else
                Path(__file__).resolve().parents[2] / "artifacts/derived/brent-academic-v1")
    result["academic_available"] = academic.exists()
    if academic.exists():
        extension = _manifest(academic)
        if extension.get("bundle_id") != "brent-academic-v1" or set(extension["files"]) != {"metrics.csv"}:
            raise ValueError("Unrecognized academic metric supplement")
        scores = pd.read_csv(academic / "metrics.csv")
        result["metrics"] = enrich_metrics(result["metrics"], scores, experiment="final")
        if "metrics" in result["evaluation"]:
            result["evaluation"]["metrics"] = enrich_metrics(result["evaluation"]["metrics"], scores)
    return result


def load_services(bundle_dir: Path | str) -> dict[str, ForecastService]:
    """Load the three real seed-42 checkpoints; keep LSTM as preselected network."""
    folder = Path(bundle_dir)
    selected = load_forecaster(folder)
    _manifest(folder)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    checkpoints = torch.load(folder / "checkpoints.pt", map_location="cpu", weights_only=True)["models"]
    intervals = json.loads((folder / "intervals.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(folder / "metrics.csv")
    services = {"LSTM": selected}
    for architecture in ("RNN", "GRU"):
        entry = checkpoints[f"{architecture}_42_test_2024H2"]
        if (entry["model"] != architecture or entry["variant"] != VARIANTS[architecture]
                or entry["seed"] != 42 or entry["block_id"] != "test_2024H2"):
            raise ValueError("Application checkpoint differs from the frozen candidate")
        model = build_model(entry["model_config"])
        inputs = InputTransformer.from_dict(entry["transformers"]["input"])
        targets = TargetTransformer.from_dict(entry["transformers"]["target"])
        if (model.input_size != 20 or model.hidden_size != 32 or model.num_layers != 1
                or model.architecture != architecture or inputs.feature_names_ != tuple(protocol["feature_names"])):
            raise ValueError("Application checkpoint feature or architecture mismatch")
        model.load_state_dict(entry["state_dict"], strict=True)
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("Non-finite application checkpoint")
        model.eval()
        services[architecture] = ForecastService(model=model, inputs=inputs, targets=targets,
            protocol=protocol, entry=entry, intervals=intervals, metrics=metrics)
    return services


def forecast_for_model(services: dict, window: pd.DataFrame, model: str,
                       data_updated_at: object | None = None) -> dict:
    name = LABELS.get(model, model)
    if name not in VARIANTS:
        raise ValueError("Unknown frozen forecast model")
    result = services["LSTM" if name == "persistence" else name].predict(window, data_updated_at)
    if name == "persistence":
        result = copy.deepcopy(result)
        result.update(model="persistence", variant="baseline", seed=-1, model_refit_at=None,
                      training_cutoff=None, model_is_past_scheduled_refit=False, model_age_warning=None)
        for row in result["forecasts"]:
            row.update(price=row["persistence_price"], change=0., lower=row["persistence_lower"],
                       upper=row["persistence_upper"])
    result["prediction_source"] = "frozen_checkpoint" if name != "persistence" else "persistence"
    return result


def filter_history(history: pd.DataFrame, period: str = "MAX", anchor: object | None = None) -> pd.DataFrame:
    frame = history.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="raise")
    frame["y_brent"] = pd.to_numeric(frame["y_brent"], errors="raise")
    if frame["observation_date"].duplicated().any() or np.isinf(frame["y_brent"]).any():
        raise ValueError("History requires unique dates and finite observed prices")
    observed = _booleans(frame["observed"], "observed") if "observed" in frame else frame["y_brent"].notna()
    frame = frame.loc[observed & frame["y_brent"].notna()].sort_values("observation_date")
    periods = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
               "1Y": pd.DateOffset(years=1), "5Y": pd.DateOffset(years=5), "MAX": None}
    if period not in periods:
        raise ValueError("Unknown historical period")
    end = pd.Timestamp(anchor) if anchor is not None else frame["observation_date"].max()
    frame = frame.loc[frame["observation_date"].le(end)]
    if periods[period] is not None:
        frame = frame.loc[frame["observation_date"].ge(end - periods[period])]
    return frame.reset_index(drop=True)


def latest_market(history: pd.DataFrame, protocol: dict) -> dict:
    observed = filter_history(history)
    if observed.empty:
        return {"price": None, "change": None, "change_pct": None, "observation_date": None,
                "data_updated_at": protocol.get("data_updated_at")}
    latest = observed.iloc[-1]
    previous = float(observed.iloc[-2]["y_brent"]) if len(observed) > 1 else None
    price = float(latest["y_brent"])
    change = price - previous if previous is not None else None
    return {"price": price, "change": change, "change_pct": 100 * change / previous if previous else None,
            "observation_date": latest["observation_date"].strftime("%Y-%m-%d"),
            "data_updated_at": protocol.get("data_updated_at")}


def context_indicators(window: pd.DataFrame, *, include_brent: bool = False) -> pd.DataFrame:
    row = window.iloc[-1] if not window.empty else pd.Series(dtype=object)
    units = {"wti": "USD/barril", "gas_natural": "USD/MMBtu", "oro": "cotización futuro",
             "cobre": "cotización futuro", "indice_dolar": "índice", "vix": "índice",
             "tipo_bce": "%", "usd_por_eur": "USD/EUR", "cny_por_usd": "CNY/USD", "opep_orb": "USD/barril"}
    if include_brent:
        units = {"brent": "USD/barril"} | units
    return pd.DataFrame([{"variable": name, "value": float(row[name]) if name in row and pd.notna(row[name]) else None,
        "observation_date": row.get(f"{name}__observation_date"), "available_at": row.get(f"{name}__available_at"),
        "unit": unit, "is_available": bool(name in row and pd.notna(row[name]))} for name, unit in units.items()])


def aggregate_metrics(metrics: pd.DataFrame, phase: str, horizon: int, seed_mode: str = "mean") -> pd.DataFrame:
    if seed_mode not in {"mean", "42"}:
        raise ValueError("Metrics must show either all seeds or operational seed 42")
    rows = metrics.loc[metrics["phase"].eq(phase) & metrics["horizon"].eq(horizon)
                       & metrics["block_id"].eq("all")].copy()
    rows = rows.loc[[VARIANTS.get(model) == variant for model, variant in zip(rows["model"], rows["variant"])]]
    if seed_mode == "42":
        rows = rows.loc[rows["seed"].isin([42, -1])]
    result = []
    for model, part in rows.groupby("model", sort=False, observed=True):
        if part["seed"].duplicated().any() or part["n"].nunique() != 1:
            raise ValueError("Historical metric seeds must have identical pair counts and no duplicates")
        record = {"model": model, "variant": VARIANTS[model], "n_seeds": len(part), "n": int(part["n"].iloc[0])}
        for metric in ("mae", "mse", "rmse", "r2", "bias", "skill_mae_pct", "training_seconds"):
            values = pd.to_numeric(part[metric], errors="raise") if metric in part else pd.Series(dtype=float)
            record[metric] = float(values.mean()) if not values.empty else None
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0. if len(values) else None
        record["parameters"] = int(part["parameters"].iloc[0]) if "parameters" in part else None
        result.append(record)
    return pd.DataFrame(result)


def stored_forecast(predictions: pd.DataFrame, model: str, horizon: int, origin: object) -> dict:
    name = LABELS.get(model, model)
    rows = predictions.loc[predictions["model"].eq(name) & predictions["horizon"].eq(horizon)
        & pd.to_datetime(predictions["origin_date"]).eq(pd.Timestamp(origin))
        & predictions["seed"].eq(-1 if name == "persistence" else 42)
        & predictions["variant"].eq(VARIANTS[name])]
    if len(rows) != 1:
        raise ValueError("No unique stored forecast exists for that model, origin and horizon")
    row = rows.iloc[0]
    if pd.Timestamp(row["target_date"]) != target_dates(pd.Timestamp(origin))[horizon]:
        raise ValueError("Stored forecast target date is misaligned")
    if pd.Timestamp(row["refit_at"]) > pd.Timestamp(row["decision_at"]):
        raise ValueError("Stored checkpoint was not available at the forecast origin")
    return {"model": name, "variant": row["variant"], "seed": int(row["seed"]),
        "origin_date": pd.Timestamp(origin).strftime("%Y-%m-%d"), "reference_brent": float(row["reference_brent"]),
        "reference_observation_date": row.get("reference_observation_date"), "reference_available_at": row.get("reference_available_at"),
        "decision_at": row["decision_at"], "model_refit_at": row["refit_at"] if name != "persistence" else None,
        "prediction_source": "saved", "forecasts": [{"horizon": horizon, "target_date": str(row["target_date"])[:10],
            "price": float(row["y_pred"]), "change": float(row["y_pred"] - row["reference_brent"]),
            "lower": float(row["lower"]) if "lower" in row and pd.notna(row["lower"]) else None,
            "upper": float(row["upper"]) if "upper" in row and pd.notna(row["upper"]) else None,
            "observed": float(row["y_true"])}]}


def _last_decision_date(now: pd.Timestamp, hour: int) -> str:
    decision = pd.offsets.BDay().rollback(now.normalize()) + pd.Timedelta(hours=hour)
    if decision > now:
        decision -= pd.offsets.BDay(1)
    return decision.date().isoformat()


def refresh_market_data(config: dict, output_root: Path | str, snapshot_id: str | None = None) -> dict:
    """Explicit network action creating a fresh snapshot, never overwriting research."""
    settings = copy.deepcopy(config)
    settings["end"] = _last_decision_date(pd.Timestamp.now(tz="Europe/Madrid"), settings.get("decision_hour", 8))
    identifier = snapshot_id or datetime.now(timezone.utc).strftime("live-%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier):
        raise ValueError("Refresh snapshot identifiers cannot contain paths")
    settings["snapshot_id"] = identifier
    directory = Path(output_root) / identifier
    if directory.exists():
        raise FileExistsError("Refreshed snapshots are immutable; choose a new identifier")
    download_snapshot(settings, directory / "raw", Path(output_root) / ".cache")
    raw, manifest = load_snapshot(directory / "raw")
    available = apply_availability(raw, settings["sources"])
    features = build_features(available, settings["start"], settings["end"], settings["decision_hour"])
    history = build_target(available, settings["start"], settings["end"])
    names = tuple(name for name, spec in settings["sources"].items()
                  if spec.get("enabled", True) and name != "opep_orb")
    # An optional provider outage must remain a missing input, not remove the
    # frozen feature column or fabricate a replacement market observation.
    for name in names:
        if name not in features:
            features[name] = np.nan
            features[f"{name}__observation_date"] = pd.NaT
            features[f"{name}__available_at"] = pd.Series(pd.NaT, index=features.index, dtype="datetime64[ns, UTC]")
            for flag in ("is_available", "is_stale", "is_observed_today", "is_carried"):
                features[f"{name}__{flag}"] = False
            features[f"{name}__availability_basis"] = "unavailable_source"
            features[f"{name}__age_business_days"] = np.nan
    build_supervised(features, history, names, 60, (5, 63, 252))
    for name, frame in [("features.csv", features), ("target.csv", history)]:
        frame.to_csv(directory / name, index=False, lineterminator="\n")
    updated = {"snapshot_id": identifier, "data_updated_at": manifest["downloaded_at"],
               "config": settings, "sources": manifest["sources"],
               "output_sha256": {name: sha256(directory / name) for name in ["features.csv", "target.csv"]},
               "weights_updated": False, "historical_point_in_time": False}
    (directory / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"features": features, "window": features.tail(60).copy(), "history": history,
            "protocol": updated, "directory": str(directory)}
