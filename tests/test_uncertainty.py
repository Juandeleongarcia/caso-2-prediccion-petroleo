"""Calibration-only empirical bands with finite, original-unit predictions."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.uncertainty import apply_intervals, fit_intervals, interval_metrics


def calibration():
    rows = []
    for seed in [42, 123]:
        for h, multiple in [(5, 1), (63, 2), (252, 3)]:
            for i, origin in enumerate(pd.bdate_range("2021-01-04", periods=9), start=1):
                error = multiple * (i + (10 if seed == 123 else 0))
                rows.append({"phase": "calibration", "model": "LSTM", "variant": "B", "seed": seed,
                             "horizon": h, "sample_id": i, "origin_date": origin,
                             "decision_at": origin.tz_localize("Europe/Madrid") + pd.Timedelta(hours=8),
                             "target_date": origin + pd.offsets.BDay(h),
                             "label_available_at": "2023-02-01T18:00:00+01:00",
                             "y_true": 100. + error, "y_pred": 100., "error": -float(error),
                             "reference_brent": 90., "predicted_change": 10.})
    return pd.DataFrame(rows)


class EmpiricalUncertainty(unittest.TestCase):
    def test_manual_order_statistic_separates_horizons_and_seeds_and_roundtrips_json(self):
        state = fit_intervals(calibration())
        self.assertEqual(len(state["groups"]), 6)
        quantiles = {(row["seed"], row["horizon"]): row for row in state["groups"]}
        for seed, offset in [(42, 0), (123, 10)]:
            for horizon, multiple in [(5, 1), (63, 2), (252, 3)]:
                row = quantiles[(seed, horizon)]
                self.assertEqual((row["n"], row["rank"]), (9, 8))
                self.assertEqual(row["q"], multiple * (8 + offset))
        fresh = json.loads(json.dumps(state, allow_nan=False))
        pd.testing.assert_frame_equal(apply_intervals(calibration(), state), apply_intervals(calibration(), fresh))

    def test_test_truth_cannot_change_bands_or_frozen_calibration_state(self):
        original = calibration()
        state = fit_intervals(original)
        state_before = copy.deepcopy(state)
        test = original.iloc[:3].copy().assign(phase="test", origin_date="2024-02-01")
        first = apply_intervals(test, state)
        test["y_true"] += 1e6
        test["label_available_at"] = "2026-01-01T18:00:00+01:00"
        second = apply_intervals(test, state)
        np.testing.assert_array_equal(first[["lower", "upper"]], second[["lower", "upper"]])
        self.assertEqual(state, state_before)
        pd.testing.assert_frame_equal(original, calibration())

    def test_forbidden_phase_year_late_label_duplicate_and_inconsistent_error_fail(self):
        base = calibration()
        wrong_year = base.copy()
        wrong_year.loc[0, "origin_date"] = pd.Timestamp("2024-01-04")
        at_deadline = base.copy()
        at_deadline.loc[0, "label_available_at"] = "2024-01-01T08:00:00+01:00"
        wrong_error = base.copy()
        wrong_error.loc[0, "error"] = 500.
        wrong_reconstruction = base.copy()
        wrong_reconstruction.loc[0, "predicted_change"] = 0.
        for bad in [base.assign(phase="test"), base.assign(phase="development"), wrong_year,
                    at_deadline, pd.concat([base, base.iloc[[0]]]), wrong_error, wrong_reconstruction,
                    base.assign(y_true=np.nan), base.assign(label_available_at="2023-02-01T18:00:00")]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                fit_intervals(bad)

    def test_manual_coverage_width_and_winkler_score_in_original_price_units(self):
        rows = calibration().iloc[:3].copy().assign(phase="test")
        rows["y_pred"] = [10., 10., 10.]
        rows["y_true"] = [8., 10., 13.]
        rows["lower"], rows["upper"], rows["interval_level"] = 8., 12., 0.8
        result = interval_metrics(rows).iloc[0]
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["mean_width"], 4.)
        self.assertEqual(result["median_width"], 4.)
        self.assertAlmostEqual(result["interval_score"], (4 + 4 + 14) / 3)
        changed = rows.copy()
        changed.loc[0, "lower"] = 11.
        with self.assertRaises(ValueError):
            interval_metrics(changed)

    def test_zero_error_cap_small_n_live_forecast_and_negative_limits_are_valid(self):
        frame = calibration().iloc[:1].copy()
        frame["y_true"], frame["error"] = frame["y_pred"], 0.
        state = fit_intervals(frame)
        self.assertEqual(state["groups"][0]["q"], 0.)
        self.assertEqual(state["groups"][0]["rank"], 1)
        live = frame.drop(columns=["y_true", "error", "label_available_at"]).assign(phase="live", y_pred=-2.)
        result = apply_intervals(live, state)
        self.assertEqual(result.iloc[0]["lower"], -2.)
        self.assertEqual(result.iloc[0]["upper"], -2.)
        state["groups"][0]["q"] = 5.
        result = apply_intervals(live, state)
        self.assertEqual(result.iloc[0]["lower"], -7.)
        self.assertEqual(result.iloc[0]["upper"], 3.)

    def test_missing_group_invalid_coverage_and_tampered_state_are_rejected(self):
        frame = calibration()
        state = fit_intervals(frame)
        for level in [0, 1, -0.1, np.nan, True]:
            with self.subTest(level=level), self.assertRaises(ValueError):
                fit_intervals(frame, coverage=level)
        with self.assertRaises(ValueError):
            fit_intervals(frame, available_before="2024-01-01")
        with self.assertRaises(ValueError):
            apply_intervals(frame.assign(model="GRU"), state)
        for change in [{"q": -1.}, {"q": np.inf}, {"rank": 1},
                       {"latest_label_available_at": "2024-01-01T07:00:00+00:00"}]:
            malformed = copy.deepcopy(state)
            malformed["groups"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_intervals(frame, malformed)


if __name__ == "__main__":
    unittest.main()
