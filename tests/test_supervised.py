"""Window shapes, exact calendar targets and input information boundaries."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.supervised import build_supervised
from oil_forecast.temporal import apply_availability, build_features, build_target, target_dates


def fixture(count=430):
    dates = pd.bdate_range("2020-01-01", periods=count)
    raw = pd.concat([
        pd.DataFrame({"observation_date": dates, "variable": variable,
                      "value": 70.0 + np.arange(count) / 10 - offset,
                      "source": "synthetic_fixture"})
        for variable, offset in [("brent", 0.0), ("wti", 2.0)]
    ], ignore_index=True)
    policies = {variable: {"enabled": True, "lag_business_days": 1, "max_age_business_days": 20}
                for variable in ["brent", "wti"]}
    available = apply_availability(raw, policies)
    start, end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    return build_features(available, start, end), build_target(available, start, end)


class SupervisedContracts(unittest.TestCase):
    def test_shape_exact_three_dates_and_reference_change(self):
        features, targets = fixture()
        data = build_supervised(features, targets, ["brent", "wti"])
        self.assertEqual(data.X.shape, (119, 60, 2))
        self.assertEqual(data.y_changes.shape, (119, 3))
        self.assertEqual(data.summary["candidate_windows"], 371)
        self.assertEqual(data.summary["warmup_excluded"], 59)
        first = data.samples.iloc[0]
        expected_dates = target_dates(first["origin_date"])
        for position, horizon in enumerate([5, 63, 252]):
            self.assertEqual(first[f"target_date_{horizon}"], expected_dates[horizon])
            expected_price = targets.set_index("observation_date").loc[expected_dates[horizon], "y_brent"]
            self.assertEqual(first[f"y_price_{horizon}"], expected_price)
            self.assertAlmostEqual(data.y_changes[0, position] + first["reference_brent"], expected_price)
        self.assertEqual(first["window_start_date"], features["decision_date"].iloc[0])
        np.testing.assert_array_equal(data.X[0], features[["brent", "wti"]].iloc[:60].to_numpy())

    def test_reference_is_last_available_price_not_same_date_target(self):
        features, targets = fixture()
        data = build_supervised(features, targets, ["brent", "wti"])
        first = data.samples.iloc[0]
        same_date_target = targets.set_index("observation_date").loc[first["origin_date"], "y_brent"]
        self.assertNotEqual(first["reference_brent"], same_date_target)
        self.assertEqual(first["reference_brent"], data.X[0, -1, 0])
        self.assertLess(first["reference_observation_date"], first["origin_date"])
        self.assertLessEqual(first["reference_available_at"], first["decision_at"])

    def test_missing_holiday_target_excludes_origin_for_all_horizons(self):
        features, targets = fixture()
        holiday = pd.Timestamp("2020-04-10")
        row = targets["observation_date"].eq(holiday)
        targets.loc[row, "y_brent"] = np.nan
        targets.loc[row, "observed"] = False
        targets.loc[row, "label_available_at"] = pd.NaT
        data = build_supervised(features, targets, ["brent", "wti"])
        origin = pd.Timestamp("2020-04-03")
        self.assertFalse(data.samples["origin_date"].eq(origin).any())
        rejected = data.exclusions.loc[data.exclusions["origin_date"].eq(origin)].iloc[0]
        self.assertEqual(rejected["target_date_5"], holiday)
        self.assertIn("missing_label_5", rejected["reason"])
        self.assertTrue(pd.isna(rejected["y_price_5"]))

    def test_target_without_availability_metadata_cannot_be_used(self):
        features, targets = fixture()
        targets.loc[64, "label_available_at"] = pd.NaT
        data = build_supervised(features, targets, ["brent", "wti"])
        excluded = data.exclusions.loc[data.exclusions["origin_date"].eq(features["decision_date"].iloc[59])].iloc[0]
        self.assertIn("missing_label_availability_5", excluded["reason"])

    def test_future_feature_mutation_cannot_change_past_windows(self):
        features, targets = fixture()
        cutoff = pd.Timestamp("2020-04-01")
        altered = features.copy()
        altered.loc[altered["decision_date"].gt(cutoff), "wti"] = 1e9
        original = build_supervised(features, targets, ["brent", "wti"])
        changed = build_supervised(altered, targets, ["brent", "wti"])
        past = original.samples["origin_date"].le(cutoff).to_numpy()
        self.assertTrue(past.any())
        np.testing.assert_array_equal(original.X[past], changed.X[past])
        pd.testing.assert_frame_equal(original.samples.loc[past], changed.samples.loc[past])

    def test_future_availability_or_observation_is_rejected(self):
        features, targets = fixture()
        for column, value in [
            ("wti__available_at", features["decision_at"].iloc[59] + pd.Timedelta(seconds=1)),
            ("wti__observation_date", features["decision_date"].iloc[60]),
        ]:
            with self.subTest(column=column):
                altered = features.copy()
                altered.loc[59, column] = value
                with self.assertRaisesRegex(ValueError, "Future information"):
                    build_supervised(altered, targets, ["brent", "wti"])

    def test_incomplete_calendar_and_duplicate_target_dates_are_rejected(self):
        features, targets = fixture()
        with self.assertRaisesRegex(ValueError, "complete Mon-Fri"):
            build_supervised(features.drop(index=80), targets, ["brent", "wti"])
        duplicate = pd.concat([targets, targets.iloc[10:11]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "ordered and unique"):
            build_supervised(features, duplicate, ["brent", "wti"])

    def test_repeatable_results_and_no_input_mutation(self):
        features, targets = fixture()
        feature_before, target_before = features.copy(deep=True), targets.copy(deep=True)
        original = build_supervised(features, targets, ["brent", "wti"])
        repeated = build_supervised(features, targets, ["brent", "wti"])
        np.testing.assert_array_equal(original.X, repeated.X)
        np.testing.assert_array_equal(original.y_changes, repeated.y_changes)
        pd.testing.assert_frame_equal(original.samples, repeated.samples)
        pd.testing.assert_frame_equal(features, feature_before)
        pd.testing.assert_frame_equal(targets, target_before)

    @unittest.skipUnless((ROOT / "data/processed/2026-11-11-official/manifest.json").exists(), "Local snapshot unavailable")
    def test_local_snapshot_and_professor_original_remain_unchanged(self):
        config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
        folder = ROOT / "data/processed" / config["snapshot_id"]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        source_paths = [folder / "features.csv", folder / "target.csv", ROOT / "data/petroleo_dataset.csv"]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
        data = build_supervised(pd.read_csv(source_paths[0]), pd.read_csv(source_paths[1]), manifest["selected_features"])
        self.assertGreater(len(data.samples), 1000)
        self.assertEqual(data.X.shape[1:], (60, 10))
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
        self.assertEqual(before, after)
        self.assertEqual(after[str(source_paths[-1])], config["original_sha256"])


if __name__ == "__main__":
    unittest.main()
