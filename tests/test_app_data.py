"""Application outputs use frozen real artifacts and keep refreshes separate."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oil_forecast.acquisition import sha256
from oil_forecast.app_data import (aggregate_metrics, context_indicators, filter_history,
    forecast_for_model, latest_market, load_app_data, load_services, refresh_market_data, stored_forecast)


@unittest.skipUnless((ROOT / "artifacts/local/brent-frozen-v1/artifact_manifest.json").exists(), "Requires the frozen inference bundle")
class ApplicationData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "artifacts/local/brent-frozen-v1"
        cls.data = load_app_data(cls.folder)
        cls.services = load_services(cls.folder)

    def test_three_real_candidates_and_persistence_use_same_reference_and_direct_dates(self):
        before = {path.name: sha256(path) for path in self.folder.iterdir() if path.is_file()}
        for model in ["RNN-D", "LSTM-B", "GRU-C", "Persistencia"]:
            result = forecast_for_model(self.services, self.data["window"], model)
            self.assertEqual(result["origin_date"], "2026-11-11")
            self.assertEqual(result["reference_observation_date"], "2026-10-28")
            self.assertEqual([row["horizon"] for row in result["forecasts"]], [5, 63, 252])
            for row in result["forecasts"]:
                self.assertLessEqual(row["lower"], row["price"])
                self.assertLessEqual(row["price"], row["upper"])
                self.assertAlmostEqual(row["price"] - result["reference_brent"], row["change"], places=10)
                if model == "Persistencia":
                    self.assertEqual(row["price"], result["reference_brent"])
                    self.assertEqual(row["change"], 0.)
        self.assertEqual(before, {path.name: sha256(path) for path in self.folder.iterdir() if path.is_file()})

    def test_market_observation_differs_from_forecast_reference_and_missing_is_not_zero(self):
        market = latest_market(self.data["history"], self.data["protocol"])
        self.assertEqual(market["observation_date"], "2026-11-03")
        month = filter_history(self.data["history"], "1M")
        self.assertTrue(month.observation_date.ge(pd.Timestamp("2026-10-03")).all())
        self.assertEqual(month.observation_date.max(), pd.Timestamp("2026-11-03"))
        indicators = context_indicators(self.data["window"]).set_index("variable")
        self.assertFalse(indicators.loc["opep_orb", "is_available"])
        self.assertTrue(pd.isna(indicators.loc["opep_orb", "value"]))
        self.assertEqual(indicators.loc["wti", "value"], self.data["window"].iloc[-1].wti)
        audit = context_indicators(self.data["window"], include_brent=True)
        self.assertEqual(len(audit.loc[audit.variable.ne("opep_orb")]), 10)
        self.assertEqual(audit.set_index("variable").loc["brent", "value"], self.data["window"].iloc[-1].brent)

    def test_metric_means_do_not_pool_seeds_or_sum_shared_horizon_training_cost(self):
        source = self.data["metrics"]
        mean = aggregate_metrics(source, "test", 252).set_index("model")
        selected = aggregate_metrics(source, "test", 252, "42").set_index("model")
        originals = source[(source.phase == "test") & (source.model == "LSTM") & (source.horizon == 252) & (source.block_id == "all")]
        self.assertAlmostEqual(mean.loc["LSTM", "mae"], originals.mae.mean(), places=12)
        self.assertAlmostEqual(mean.loc["LSTM", "mae_std"], originals.mae.std(ddof=1), places=12)
        self.assertEqual(mean.loc["LSTM", "n"], 238)
        self.assertEqual(mean.loc["LSTM", "n_seeds"], 3)
        self.assertEqual(selected.loc["LSTM", "n_seeds"], 1)
        self.assertAlmostEqual(selected.loc["LSTM", "mae"], originals[originals.seed == 42].mae.iloc[0])
        short = aggregate_metrics(source, "test", 5).set_index("model")
        self.assertEqual(mean.loc["LSTM", "training_seconds"], short.loc["LSTM", "training_seconds"])

    def test_saved_forecast_is_not_simulated_with_a_future_checkpoint(self):
        frame = pd.read_csv(self.folder / "reference_predictions.csv")
        origin = frame.origin_date.iloc[0]
        result = stored_forecast(frame, "LSTM-B", 252, origin)
        truth = frame[frame.origin_date.eq(origin) & frame.horizon.eq(252)].iloc[0]
        self.assertEqual(result["forecasts"][0]["price"], truth.y_pred)
        self.assertEqual(result["prediction_source"], "saved")
        with self.assertRaises(ValueError):
            stored_forecast(frame, "LSTM-B", 252, "2020-01-02")
        altered = frame.copy()
        altered.loc[altered.origin_date.eq(origin), "refit_at"] = "2026-01-01T07:00:00+00:00"
        with self.assertRaises(ValueError):
            stored_forecast(altered, "LSTM-B", 252, origin)

    def test_academic_extension_preserves_original_scores_and_matches_each_real_seed(self):
        evidence = ROOT / "artifacts/local/brent-dashboard-v1"
        data = load_app_data(self.folder, evidence)
        self.assertTrue(data["academic_available"])
        original = pd.read_csv(self.folder / "metrics.csv")
        pd.testing.assert_frame_equal(data["metrics"].drop(columns="r2"), original)
        pd.testing.assert_frame_equal(data["evaluation"]["metrics"].drop(columns="r2"), pd.read_csv(evidence / "metrics.csv"))
        predictions = pd.read_csv(ROOT / "experiments/cpu-final-evaluation-v1/predictions.csv")
        for phase in ("calibration", "test"):
            for horizon in (5, 63, 252):
                expected = {}
                pairs = predictions.loc[predictions.phase.eq(phase) & predictions.horizon.eq(horizon)]
                for (model, seed), rows in pairs.groupby(["model", "seed"]):
                    truth, forecast = rows.y_true.to_numpy(), rows.y_pred.to_numpy()
                    expected[(model, seed)] = 1 - np.sum((forecast - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
                for mode in ("mean", "42"):
                    summary = aggregate_metrics(data["metrics"], phase, horizon, mode)
                    for row in summary.itertuples():
                        seeds = [-1] if row.model == "persistence" else [42] if mode == "42" else [42, 123, 2026]
                        self.assertAlmostEqual(row.r2, np.mean([expected[(row.model, seed)] for seed in seeds]), places=12)
        with tempfile.TemporaryDirectory() as temporary:
            without = load_app_data(self.folder, academic_dir=Path(temporary) / "absent")
            self.assertFalse(without["academic_available"])
            pd.testing.assert_frame_equal(without["metrics"], original)


class ExplicitRefresh(unittest.TestCase):
    def test_refresh_never_uses_a_future_or_weekend_decision(self):
        from oil_forecast.app_data import _last_decision_date
        for now, expected in [("2026-11-16 07:59", "2026-11-13"), ("2026-11-16 08:00", "2026-11-16"),
                              ("2026-11-15 12:00", "2026-11-13")]:
            self.assertEqual(_last_decision_date(pd.Timestamp(now, tz="Europe/Madrid"), 8), expected)

    def test_refresh_preserves_config_creates_new_snapshot_and_keeps_optional_failure_missing(self):
        config = {"start": "2025-01-01", "end": "2026-11-11", "decision_hour": 8,
            "sources": {"brent": {"lag_business_days": 10, "max_age_business_days": 20},
                        "oro": {"lag_business_days": 2, "max_age_business_days": 10}}}
        original = copy.deepcopy(config)
        def download(settings, directory, cache):
            directory.mkdir(parents=True)
            dates = pd.bdate_range(settings["start"], settings["end"])
            raw = pd.DataFrame({"observation_date": dates, "variable": "brent", "source": "fixture",
                                "value": 70 + np.sin(np.arange(len(dates))), "is_observed": True})
            raw.to_csv(directory / "observations.csv", index=False)
            meta = {"required_sources_unavailable": [], "observations_sha256": sha256(directory / "observations.csv"),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "sources": {"brent": {"status": "ok"}, "oro": {"status": "unavailable"}}}
            (directory / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.app_data.download_snapshot", side_effect=download) as network:
            result = refresh_market_data(config, temporary, "controlled-refresh")
            self.assertEqual(config, original)
            self.assertEqual(result["protocol"]["config"]["snapshot_id"], "controlled-refresh")
            self.assertFalse(result["protocol"]["weights_updated"])
            self.assertTrue(result["features"].oro.isna().all())
            self.assertFalse(result["features"].oro__is_available.any())
            self.assertEqual(len(result["window"]), 60)
            self.assertEqual(network.call_count, 1)
            with self.assertRaises(FileExistsError):
                refresh_market_data(config, temporary, "controlled-refresh")
            self.assertEqual(network.call_count, 1)


if __name__ == "__main__":
    unittest.main()
