"""Serving uses a frozen bundle and observed inputs, without future labels."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.acquisition import sha256
from oil_forecast.inference import load_forecaster
from oil_forecast.models import RecurrentForecaster, initialize_persistence_head
from oil_forecast.preprocessing import InputTransformer, TargetTransformer
from oil_forecast.temporal import decision_times, target_dates
from oil_forecast.training import set_reproducible


FEATURES = ["brent", *[f"feature_{index}" for index in range(9)]]


def window_fixture():
    dates = pd.bdate_range(end="2026-11-11", periods=60)
    result = pd.DataFrame({"decision_date": dates, "decision_at": decision_times(dates)})
    observations = dates - pd.offsets.BDay(10)
    for index, name in enumerate(FEATURES):
        result[name] = 70.0 if name == "brent" else float(index)
        result[f"{name}__observation_date"] = observations
        result[f"{name}__available_at"] = decision_times(dates)
        result[f"{name}__is_available"] = True
        result[f"{name}__is_stale"] = False
    return result


def write_bundle(directory):
    directory = Path(directory)
    inputs = InputTransformer().fit(np.arange(100, dtype=float).reshape(10, 10), FEATURES)
    targets = TargetTransformer().fit(np.array([[-2.0, -4.0, 0.0], [2.0, 8.0, 20.0]]))
    set_reproducible(42)
    model = RecurrentForecaster("LSTM")
    initialize_persistence_head(model, targets.mean_, targets.scale_)
    protocol = {"selected_model": "LSTM", "selected_seed": 42,
                "variants": {"RNN": "D", "LSTM": "B", "GRU": "C"},
                "feature_names": FEATURES, "lookback": 60, "horizons": [5, 63, 252],
                "data_updated_at": "2026-11-11T21:08:21+00:00", "decision_hour": 8,
                "threads": 1, "availability_mode": "retrospective_with_explicit_assumptions"}
    entry = {"model": "LSTM", "variant": "B", "seed": 42, "phase": "test", "block_id": "test_2024H2",
             "refit_at": "2024-07-01T06:00:00+00:00", "training_cutoff": "2024-06-28T06:00:00+00:00",
             "model_config": model.config(), "state_dict": model.state_dict(),
             "transformers": {"input": inputs.to_dict(), "target": targets.to_dict()}}
    torch.save({"models": {"LSTM_42_test_2024H2": entry}}, directory / "checkpoints.pt")
    (directory / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    groups = [{"model": architecture, "variant": variant, "seed": seed, "horizon": horizon,
               "n": 9, "rank": 8, "q": float(horizon) / 5,
               "latest_label_available_at": "2023-12-29T07:00:00+00:00"}
              for architecture, variant, seed in [("LSTM", "B", 42), ("persistence", "baseline", -1)]
              for horizon in [5, 63, 252]]
    state = {"schema_version": 1, "method": "prequential_absolute_error_order_statistic", "coverage": 0.8,
             "unit": "USD/barrel", "calibration_years": [2021, 2022],
             "available_before": "2024-01-01T07:00:00+00:00", "groups": groups}
    (directory / "intervals.json").write_text(json.dumps(state), encoding="utf-8")
    pd.DataFrame([{"phase": "calibration", "model": "LSTM", "mae": 3.0},
                  {"phase": "test", "model": "LSTM", "mae": 4.0}]).to_csv(directory / "metrics.csv", index=False)
    report = {"deployment": {"checkpoint_key": "LSTM_42_test_2024H2", "selected_model": "LSTM",
                              "selected_seed": 42, "model_refit_at": entry["refit_at"]},
              "output_sha256": {name: sha256(directory / name) for name in
                                ["protocol.json", "checkpoints.pt", "intervals.json", "metrics.csv"]}}
    (directory / "experiment.json").write_text(json.dumps(report), encoding="utf-8")
    return report


class FrozenInferenceContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.report = write_bundle(self.directory)
        self.service = load_forecaster(self.directory)
        self.window = window_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def test_zero_change_reconstructs_reference_and_future_calendar_without_future_labels(self):
        result = self.service.predict(self.window)
        self.assertEqual(result["origin_date"], "2026-11-11")
        self.assertEqual(result["reference_observation_date"], "2026-10-28")
        self.assertEqual(result["model_refit_at"], "2024-07-01T06:00:00+00:00")
        self.assertTrue(result["model_is_past_scheduled_refit"])
        self.assertIsNotNone(result["model_age_warning"])
        future = target_dates("2026-11-11")
        for forecast in result["forecasts"]:
            self.assertEqual(forecast["target_date"], future[forecast["horizon"]].strftime("%Y-%m-%d"))
            self.assertGreater(forecast["target_date"], result["origin_date"])
            self.assertAlmostEqual(forecast["price"], 70, places=6)
            self.assertAlmostEqual(forecast["change"], 0, places=6)
            self.assertEqual(forecast["persistence_price"], 70)
            self.assertLessEqual(forecast["lower"], forecast["price"])
            self.assertGreaterEqual(forecast["upper"], forecast["price"])
            self.assertAlmostEqual(forecast["upper"] - forecast["lower"], 2 * forecast["horizon"] / 5)
        json.dumps(result, allow_nan=False)

    def test_prediction_preserves_saved_statistics_weights_window_and_uses_missing_mask(self):
        original = self.window.copy(deep=True)
        input_state = copy.deepcopy(self.service._inputs.to_dict())
        target_state = copy.deepcopy(self.service._targets.to_dict())
        weights = {key: value.clone() for key, value in self.service._model.state_dict().items()}
        absent = self.window.copy(deep=True)
        absent.loc[0, "feature_0"] = np.nan
        absent.loc[0, "feature_0__is_available"] = False
        self.service.predict(absent)
        self.assertEqual(input_state, self.service._inputs.to_dict())
        self.assertEqual(target_state, self.service._targets.to_dict())
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, self.service._model.state_dict()[key]))
        pd.testing.assert_frame_equal(self.window, original)
        shuffled_columns = self.window.loc[:, self.window.columns[::-1]]
        self.assertEqual(self.service.predict(self.window), self.service.predict(shuffled_columns))

    def test_future_metadata_and_invalid_calendars_are_rejected(self):
        future_availability = self.window.copy()
        future_availability.loc[0, "feature_0__available_at"] += pd.Timedelta(hours=1)
        future_observation = self.window.copy()
        future_observation.loc[0, "feature_0__observation_date"] = pd.Timestamp("2027-01-01")
        duplicate_date = self.window.copy()
        duplicate_date.loc[1, "decision_date"] = duplicate_date.loc[0, "decision_date"]
        wrong_hour = self.window.copy()
        wrong_hour.loc[0, "decision_at"] += pd.Timedelta(hours=1)
        stale = self.window.copy()
        stale.loc[0, "feature_0__is_stale"] = True
        absent_reference = self.window.copy()
        absent_reference.loc[59, "brent"] = np.nan
        absent_reference.loc[59, "brent__is_available"] = False
        infinity = self.window.copy()
        infinity.loc[0, "feature_0"] = np.inf
        for bad in [future_availability, future_observation, duplicate_date, wrong_hour, stale,
                    absent_reference, infinity, self.window.iloc[:-1],
                    self.window.drop(columns=["feature_0__available_at"]), self.window.iloc[::-1]]:
            with self.subTest(case=bad.shape), self.assertRaises(ValueError):
                self.service.predict(bad)
        with self.assertRaises(ValueError):
            self.service.predict(self.window, data_updated_at="2026-11-10T21:00:00+00:00")

    def test_reload_is_exact_and_historical_metrics_cannot_change_selection(self):
        second = load_forecaster(self.directory)
        self.assertEqual(self.service.predict(self.window), second.predict(self.window))
        metrics = self.service.evaluation_metrics("test")
        metrics.loc[:, "mae"] = 999
        self.assertEqual(self.service.evaluation_metrics("test")["mae"].iloc[0], 4.0)
        self.assertEqual(self.service.predict(self.window)["model"], "LSTM")
        with self.assertRaises(ValueError):
            self.service.evaluation_metrics("development")

    def test_tampered_artifact_or_serving_selection_fails_before_checkpoint_loading(self):
        with (self.directory / "metrics.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_forecaster(self.directory)
        self.report = write_bundle(self.directory)
        self.report["deployment"]["selected_model"] = "RNN"
        (self.directory / "experiment.json").write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen final protocol"):
            load_forecaster(self.directory)


if __name__ == "__main__":
    unittest.main()
