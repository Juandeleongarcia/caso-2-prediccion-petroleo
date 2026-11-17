"""Contratos de causalidad con ejemplos sintéticos, sin red ni entrenamiento."""

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.temporal import (
    apply_availability,
    build_features,
    build_target,
    eligible_training_labels,
    evaluation_schedule,
    target_dates,
)


def raw_rows(*rows):
    """Tuplas: fecha, variable, valor, publicación explícita opcional."""
    records = []
    for date, variable, value, available in rows:
        records.append({
            "observation_date": date,
            "variable": variable,
            "value": value,
            "source": "synthetic_fixture",
            "available_at": available,
            "availability_basis": "source_release_timestamp" if available else None,
        })
    return pd.DataFrame(records)


def policies(*variables, lag=1, max_age=20):
    return {
        variable: {
            "enabled": True,
            "lag_business_days": lag,
            "max_age_business_days": max_age,
        }
        for variable in variables
    }


def by_decision(frame):
    return frame.set_index("decision_date")


class TemporalContracts(unittest.TestCase):
    def test_duplicate_observation_rejected(self):
        raw = raw_rows(
            ("2024-01-02", "brent", 75.0, None),
            ("2024-01-02", "brent", 76.0, None),
        )
        with self.assertRaisesRegex(ValueError, "duplicad"):
            apply_availability(raw, policies("brent"))

    def test_negative_wti_and_identical_original_brent_prices_preserved(self):
        raw = raw_rows(
            ("2020-04-20", "wti", -36.98, None),
            ("2020-04-20", "brent", 17.36, None),
            ("2020-04-21", "brent", 17.36, None),
        )
        available = apply_availability(raw, policies("brent", "wti"))
        self.assertEqual(available.loc[available.variable.eq("wti"), "value"].iloc[0], -36.98)
        target = build_target(available, "2020-04-20", "2020-04-21")
        self.assertEqual(target["y_brent"].tolist(), [17.36, 17.36])
        self.assertTrue(target["observed"].all())
        features = build_features(available, "2020-04-20", "2020-04-22")
        self.assertIn(-36.98, features["wti"].dropna().tolist())

    def test_missing_target_is_never_forward_filled(self):
        raw = raw_rows(
            ("2024-12-24", "brent", 75.0, None),
            ("2024-12-26", "brent", 76.0, None),
        )
        target = build_target(
            apply_availability(raw, policies("brent")), "2024-12-24", "2024-12-26"
        ).set_index("observation_date")
        holiday = target.loc[pd.Timestamp("2024-12-25")]
        self.assertFalse(holiday["observed"])
        self.assertTrue(pd.isna(holiday["y_brent"]))
        self.assertTrue(pd.isna(holiday["label_available_at"]))

    def test_late_publication_respected_before_and_after_decision(self):
        raw = raw_rows(("2024-01-02", "brent", 75.0, "2024-01-04T10:00:00Z"))
        available = apply_availability(raw, policies("brent"))
        self.assertEqual(available["availability_basis"].iloc[0], "source_release_timestamp")
        features = by_decision(build_features(available, "2024-01-02", "2024-01-05"))
        before = features.loc[pd.Timestamp("2024-01-04")]
        after = features.loc[pd.Timestamp("2024-01-05")]
        self.assertFalse(before["brent__is_available"])
        self.assertTrue(pd.isna(before["brent"]))
        self.assertEqual(after["brent"], 75.0)
        self.assertLessEqual(after["brent__available_at"], after["decision_at"])

    def test_unknown_release_is_explicit_assumption_not_official_timestamp(self):
        raw = raw_rows(("2024-01-05", "brent", 75.0, None))
        available = apply_availability(raw, policies("brent", lag=2))
        self.assertEqual(available["availability_basis"].iloc[0], "assumed_business_lag")
        self.assertEqual(available["available_at"].iloc[0], pd.Timestamp("2024-01-09T07:00:00Z"))
        features = by_decision(build_features(available, "2024-01-05", "2024-01-09"))
        self.assertFalse(features.loc[pd.Timestamp("2024-01-08"), "brent__is_available"])
        self.assertTrue(features.loc[pd.Timestamp("2024-01-09"), "brent__is_available"])

    def test_no_backward_fill_before_coverage(self):
        raw = raw_rows(("2024-01-04", "oro", 2000.0, "2024-01-04T06:00:00Z"))
        features = by_decision(
            build_features(apply_availability(raw, policies("oro")), "2024-01-02", "2024-01-05")
        )
        self.assertTrue(features.loc[:"2024-01-03", "oro"].isna().all())
        self.assertFalse(features.loc[:"2024-01-03", "oro__is_available"].any())
        self.assertTrue(features.loc[pd.Timestamp("2024-01-04"), "oro__is_observed_today"])
        self.assertTrue(features.loc[pd.Timestamp("2024-01-05"), "oro__is_carried"])

    def test_old_value_becomes_missing_but_keeps_provenance(self):
        raw = raw_rows(("2024-01-02", "brent", 75.0, "2024-01-02T06:00:00Z"))
        features = by_decision(build_features(
            apply_availability(raw, policies("brent", max_age=2)), "2024-01-02", "2024-01-05"
        ))
        at_limit = features.loc[pd.Timestamp("2024-01-04")]
        stale = features.loc[pd.Timestamp("2024-01-05")]
        self.assertEqual(at_limit["brent"], 75.0)
        self.assertTrue(pd.isna(stale["brent"]))
        self.assertTrue(stale["brent__is_stale"])
        self.assertFalse(stale["brent__is_available"])
        self.assertEqual(stale["brent__age_business_days"], 3)
        self.assertEqual(stale["brent__observation_date"], pd.Timestamp("2024-01-02"))

    def test_altering_future_values_does_not_change_past_features(self):
        raw = raw_rows(
            ("2024-01-02", "brent", 75.0, None),
            ("2024-01-03", "brent", 76.0, None),
            ("2024-01-10", "brent", 77.0, None),
        )
        changed = raw.copy()
        changed.loc[changed.observation_date.eq("2024-01-10"), "value"] = 999999.0
        original_features = build_features(
            apply_availability(raw, policies("brent")), "2024-01-02", "2024-01-08"
        )
        changed_features = build_features(
            apply_availability(changed, policies("brent")), "2024-01-02", "2024-01-08"
        )
        pd.testing.assert_frame_equal(original_features, changed_features)

    def test_late_release_of_older_observation_cannot_regress_latest_known(self):
        raw = raw_rows(
            ("2024-01-02", "brent", 75.0, "2024-01-05T06:00:00Z"),
            ("2024-01-03", "brent", 76.0, "2024-01-05T06:00:00Z"),
            ("2024-01-01", "brent", 74.0, "2024-01-08T06:00:00Z"),
        )
        features = by_decision(
            build_features(apply_availability(raw, policies("brent")), "2024-01-05", "2024-01-08")
        )
        latest = features.loc[pd.Timestamp("2024-01-08")]
        self.assertEqual(latest["brent"], 76.0)
        self.assertEqual(latest["brent__observation_date"], pd.Timestamp("2024-01-03"))

    def test_publication_requires_timezone_and_cannot_predate_observation(self):
        for date, stamp in [
            ("2024-01-02", "2024-01-03 08:00:00"),
            ("2024-01-03", "2024-01-02T06:00:00Z"),
        ]:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                apply_availability(raw_rows((date, "brent", 75.0, stamp)), policies("brent"))

    def test_long_horizon_label_must_mature_before_training_cutoff(self):
        future_date = target_dates("2024-01-02")[252]
        self.assertEqual(future_date, pd.Timestamp("2024-12-19"))
        raw = raw_rows(("2024-12-19", "brent", 75.0, "2025-01-02T07:00:00Z"))
        target = build_target(apply_availability(raw, policies("brent")), "2024-12-19", "2024-12-19")
        self.assertFalse(eligible_training_labels(target, "2024-12-31T07:00:00Z").iloc[0])
        self.assertFalse(eligible_training_labels(target, "2025-01-02T06:59:59Z").iloc[0])
        self.assertTrue(eligible_training_labels(target, "2025-01-02T07:00:00Z").iloc[0])
        with self.assertRaises(ValueError):
            eligible_training_labels(target, "2025-01-02")

    def test_horizons_count_weekdays_without_compressing_missing_targets(self):
        dates = target_dates("2024-12-18")
        self.assertEqual(set(dates), {5, 63, 252})
        self.assertEqual(dates[5], pd.Timestamp("2024-12-25"))
        with self.assertRaises(ValueError):
            target_dates("2024-12-21")

    def test_madrid_decision_time_tracks_daylight_saving(self):
        raw = raw_rows(("2024-03-28", "brent", 75.0, None))
        features = by_decision(
            build_features(apply_availability(raw, policies("brent")), "2024-03-29", "2024-04-01")
        )
        self.assertEqual(features.loc[pd.Timestamp("2024-03-29"), "decision_at"], pd.Timestamp("2024-03-29T07:00:00Z"))
        self.assertEqual(features.loc[pd.Timestamp("2024-04-01"), "decision_at"], pd.Timestamp("2024-04-01T06:00:00Z"))

    def test_semiannual_schedule_is_identical_across_evaluation_phases(self):
        config = {"evaluation": {
            "development_origin_years": [2015, 2017, 2019],
            "calibration_origin_years": [2021, 2022],
            "test_origin_years": [2024],
            "refit_months": [1, 7],
        }}
        schedule = evaluation_schedule(config)
        self.assertEqual(schedule.groupby("phase").size().to_dict(), {
            "development": 6, "calibration": 4, "test": 2
        })
        self.assertTrue((schedule["training_cutoff"] < schedule["refit_at"]).all())
        self.assertTrue(schedule["origin_start"].dt.month.isin([1, 7]).all())
        self.assertTrue((schedule["origin_start"].dt.day_of_week < 5).all())
        self.assertTrue((schedule["origin_end"] >= schedule["origin_start"]).all())
        self.assertFalse(schedule.duplicated(["phase", "origin_start"]).any())


if __name__ == "__main__":
    unittest.main()
