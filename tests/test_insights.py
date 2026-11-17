"""Business explanations, costs and event markers preserve original-unit facts."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oil_forecast.app_data import aggregate_metrics, load_app_data, load_services, forecast_for_model, latest_market
from oil_forecast.insights import comparison_conclusion, cost_scenarios, event_points, explain_forecast
from oil_forecast.app_charts import history_chart


class DecisionSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_app_data(ROOT / "artifacts/local/brent-frozen-v1")
        cls.services = load_services(ROOT / "artifacts/local/brent-frozen-v1")

    def test_real_four_models_three_horizons_use_operational_not_mean_mae(self):
        before = self.data["metrics"].copy(deep=True)
        for model in ("persistence", "RNN", "LSTM", "GRU"):
            forecast = forecast_for_model(self.services, self.data["window"], model)
            for horizon in (5, 63, 252):
                explanation = explain_forecast(forecast, horizon, before, latest_market(self.data["history"], self.data["protocol"]))
                expected = before.loc[(before.model == model) & (before.seed == (-1 if model == "persistence" else 42))
                    & (before.phase == "test") & (before.block_id == "all") & (before.horizon == horizon)].iloc[0]
                self.assertEqual(explanation["mae"], expected.mae)
                self.assertIn("no una probabilidad de acierto", " ".join(explanation["paragraphs"]))
                self.assertIn("no garantiza", " ".join(explanation["paragraphs"]))
                if model == "LSTM" and horizon == 252:
                    self.assertAlmostEqual(explanation["mae"], 21.52538, places=5)
                    self.assertIn("menos precisa", " ".join(explanation["paragraphs"]))
        pd.testing.assert_frame_equal(before, self.data["metrics"])

    def test_direction_zero_reference_and_archived_quotes_are_handled(self):
        base = forecast_for_model(self.services, self.data["window"], "LSTM")
        for delta, direction in [(3., "subida"), (-3., "bajada"), (0., "estable")]:
            forecast = copy.deepcopy(base)
            row = forecast["forecasts"][0]
            row.update(price=forecast["reference_brent"] + delta, change=delta,
                       lower=forecast["reference_brent"] - 10, upper=forecast["reference_brent"] + 10)
            self.assertEqual(explain_forecast(forecast, 5, self.data["metrics"])["direction"], direction)
        forecast["prediction_source"] = "saved"
        text = explain_forecast(forecast, 5, self.data["metrics"], {"price": 999., "observation_date": "2099-01-01"})
        self.assertNotIn("2099", " ".join(text["paragraphs"]))
        forecast["reference_brent"] = 0.
        forecast["forecasts"][0].update(price=3., change=3., lower=-2., upper=8.)
        self.assertIsNone(explain_forecast(forecast, 5, self.data["metrics"])["change_pct"])

    def test_costs_preserve_endpoints_zero_volume_and_final_cent_rounding(self):
        forecast = forecast_for_model(self.services, self.data["window"], "LSTM")
        table = cost_scenarios(forecast, 63, 1000.).set_index("Escenario")
        row = forecast["forecasts"][1]
        self.assertEqual(table.loc["Referencia", "Coste (USD)"], 119200.)
        self.assertEqual(table.loc["Límite inferior · 80 %", "USD/barril"], row["lower"])
        self.assertEqual(table.loc["Límite superior · 80 %", "USD/barril"], row["upper"])
        self.assertTrue(cost_scenarios(forecast, 63, 0.)["Coste (USD)"].eq(0).all())
        small = copy.deepcopy(forecast)
        small["forecasts"][0].update(price=.005, change=.005 - small["reference_brent"], lower=-1., upper=1.)
        rounded = cost_scenarios(small, 5, 1.).set_index("Escenario")
        self.assertEqual(rounded.loc["Predicción", "Coste (USD)"], .01)
        self.assertEqual(rounded.loc["Límite inferior · 80 %", "Coste (USD)"], -1.)
        for invalid in (True, -1., float("nan"), float("inf"), 1e10):
            with self.assertRaises(ValueError):
                cost_scenarios(forecast, 5, invalid)

    def test_markers_are_real_observations_and_filtering_does_not_reselect_extrema(self):
        config = json.loads((ROOT / "configs/economic_events.json").read_text(encoding="utf-8"))
        points = event_points(self.data["history"], config["events"])
        self.assertEqual(len(points), 6)
        history = self.data["history"].copy()
        history["observation_date"] = pd.to_datetime(history.observation_date)
        for point in points.to_dict("records"):
            observed = history.loc[history.observation_date.eq(point["observation_date"])].iloc[0]
            self.assertTrue(observed.observed)
            self.assertEqual(point["price"], observed.y_brent)
            self.assertTrue(all(source["url"].startswith("https://") for source in point["sources"]))
        recent = history.loc[history.observation_date.ge("2026-01-01")]
        chart = history_chart(recent, events=points)
        actual = set(pd.to_datetime(chart.data[1].x))
        self.assertEqual(actual, {points.iloc[-1].observation_date})
        self.assertEqual(len(history_chart(recent, context_events=False, events=points).data), 1)
        clipped = history.loc[history.observation_date.between("2008-07-15", "2008-07-20")]
        self.assertTrue(event_points(clipped, config["events"][:1]).empty)

    def test_comparison_conclusions_match_real_horizons_metrics_and_seed_scope(self):
        for horizon in (5, 63, 252):
            for mode in ("mean", "42"):
                summary = aggregate_metrics(self.data["metrics"], "test", horizon, mode)
                before = summary.copy(deep=True)
                for metric in ("mae", "mse", "rmse", "r2"):
                    text = comparison_conclusion(summary, metric, "test", horizon, mode)
                    self.assertIn(f"a {horizon} días", text)
                    self.assertIn("238 orígenes", text)
                    self.assertIn("Persistencia presenta", text)
                    self.assertIn("media de las tres semillas" if mode == "mean" else "semilla operativa 42", text)
                    self.assertIn("selección previa a test", text)
                    self.assertIn("% mayor", text)
                    if metric == "r2" and horizon in (63, 252):
                        self.assertIn("mayor R²: -", text)
                    if horizon == 252:
                        self.assertIn("21,53" if mode == "42" else "13,68", text)
                pd.testing.assert_frame_equal(summary, before)

    def test_comparison_negative_r2_ties_and_phase_limits_keep_mae_separate(self):
        summary = pd.DataFrame({"model": ["persistence", "RNN", "LSTM", "GRU"],
            "n": 4, "n_seeds": 1, "mae": [2., 3., 4., 5.], "mse": [5., 10., 17., 26.],
            "rmse": np.sqrt([5., 10., 17., 26.]), "r2": [-2., -.5, -.5, -3.],
            "skill_mae_pct": [0., -50., -100., -150.]})
        before = summary.copy(deep=True)
        tied = comparison_conclusion(summary, "r2", "test", 63, "42")
        self.assertIn("RNN-D y LSTM-B comparten el mayor R²: -0,500", tied)
        self.assertIn("El MAE de LSTM-B es 4,00 frente a 2,00", tied)
        self.assertIn("100,00 % mayor", tied)
        for phase, expected in (("development", "no estiman por sí solos la generalización"),
                                ("calibration", "descriptiva y no demuestra una ventaja futura")):
            text = comparison_conclusion(summary, "mae", phase, 5, "42")
            self.assertIn("Persistencia presenta el menor MAE: 2,000", text)
            self.assertIn(expected, text)
        improved = summary.copy()
        improved.loc[improved.model.eq("LSTM"), ["mae", "skill_mae_pct"]] = [1., 50.]
        self.assertIn("50,00 % menor", comparison_conclusion(improved, "mae", "test", 5, "42"))
        pd.testing.assert_frame_equal(summary, before)

    def test_comparison_rejects_invalid_scopes_counts_scores_and_inconsistent_improvement(self):
        original = aggregate_metrics(self.data["metrics"], "test", 5, "42")
        for arguments in (("unknown", "test", 5, "42"), ("mae", "unknown", 5, "42"),
                          ("mae", "test", 1, "42"), ("mae", "test", 5, "unknown")):
            with self.assertRaises(ValueError):
                comparison_conclusion(original, *arguments)
        cases = [original.iloc[:3], pd.concat([original, original.iloc[:1]])]
        for column, value in (("mae", np.nan), ("r2", np.inf), ("n", 0), ("n", 237),
                              ("n_seeds", 3), ("skill_mae_pct", 10.)):
            changed = original.copy(deep=True)
            changed.loc[changed.model.eq("LSTM"), column] = value
            cases.append(changed)
        for case in cases:
            with self.assertRaises(ValueError):
                comparison_conclusion(case, "r2", "test", 5, "42")


if __name__ == "__main__":
    unittest.main()
