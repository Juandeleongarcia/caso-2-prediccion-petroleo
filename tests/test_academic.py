"""Additional R², error diagnostics and immutable historical derivation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.academic import (EXPERIMENTS, derive_metrics, enrich_metrics,
                                   build_academic_metrics, detail_cases, error_periods)
from oil_forecast.acquisition import sha256
from oil_forecast.artifacts import _manifest


def saved_pairs(experiment="final", phase="test", model="LSTM", seed=42):
    return pd.DataFrame({"experiment": experiment, "phase": phase, "model": model,
                         "variant": "B" if model == "LSTM" else "baseline",
                         "seed": seed if model != "persistence" else -1, "horizon": 5,
                         "block_id": [f"{phase}_2024H1"] * 2 + [f"{phase}_2024H2"] * 2,
                         "sample_id": [1, 2, 3, 4],
                         "target_date": ["2024-01-02", "2024-02-05", "2024-07-02", "2024-08-05"],
                         "reference_brent": [2., 2., 8., 8.], "y_true": [1., 2., 9., 10.],
                         "y_pred": [0., 0., 0., 0.]})


class AcademicScores(unittest.TestCase):
    def test_manual_r2_perfect_negative_constant_and_single_observation(self):
        frame = saved_pairs()
        negative = derive_metrics(frame).query("block_id == 'all'").iloc[0]
        self.assertAlmostEqual(negative.r2, 1 - 186 / 65)
        self.assertAlmostEqual(negative.mse, 186 / 4)
        self.assertAlmostEqual(negative.mae, 5.5)
        frame["y_pred"] = frame["y_true"]
        perfect = derive_metrics(frame)
        self.assertTrue(perfect.r2.eq(1).all())
        for predicted in (2., 3.):
            constant = frame.assign(y_true=2., y_pred=predicted)
            self.assertTrue(derive_metrics(constant).r2.isna().all())
        self.assertTrue(derive_metrics(frame.iloc[:1]).r2.isna().all())

    def test_phase_totals_recompute_pairs_and_never_pool_seeds_horizons_or_phases(self):
        original = saved_pairs()
        frames = [original, original.assign(seed=123), original.assign(horizon=63),
                  saved_pairs(phase="calibration")]
        metrics = derive_metrics(pd.concat(frames, ignore_index=True))
        pooled = metrics.query("block_id == 'all'")
        self.assertEqual(len(pooled), 4)
        self.assertTrue(pooled.n.eq(4).all())
        individual_mean = metrics.query("block_id != 'all'").r2.mean()
        self.assertNotAlmostEqual(pooled.r2.iloc[0], individual_mean)
        pd.testing.assert_frame_equal(original, saved_pairs())

    def test_nonfinite_duplicate_invalid_phase_and_preaggregated_pairs_fail(self):
        original = saved_pairs()
        cases = [original.assign(y_true=np.nan), original.assign(y_pred=np.inf),
                 pd.concat([original, original.iloc[:1]], ignore_index=True),
                 original.assign(phase="unseen"), original.assign(horizon=7),
                 original.assign(block_id="all"), original.assign(phase="calibration")]
        for frame in cases:
            with self.subTest(case=frame.iloc[0].to_dict()), self.assertRaises(ValueError):
                derive_metrics(frame)

    def test_enrichment_only_appends_r2_and_reconciles_counts_existing_scores_and_keys(self):
        supplemental = derive_metrics(saved_pairs())
        original = supplemental.drop(columns="r2").assign(training_seconds=1.5)
        untouched = original.copy(deep=True)
        enriched = enrich_metrics(original, supplemental)
        pd.testing.assert_frame_equal(enriched.drop(columns="r2"), original)
        pd.testing.assert_frame_equal(original, untouched)
        np.testing.assert_allclose(enriched.r2, supplemental.r2)
        serving = original.drop(columns="experiment")
        pd.testing.assert_frame_equal(enrich_metrics(serving, supplemental, "final").drop(columns="r2"), serving)
        pd.testing.assert_frame_equal(enrich_metrics(enriched, supplemental), enriched)
        for column, value in [("mse", 99.), ("mae", 99.), ("rmse", 99.), ("n", 99), ("seed", 2026)]:
            changed = original.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                enrich_metrics(changed, supplemental)
        with self.assertRaises(ValueError):
            enrich_metrics(original, pd.concat([supplemental, supplemental.iloc[:1]]))

    def test_target_quarter_errors_bias_and_naive_use_identical_pairs(self):
        frame = saved_pairs()
        before = frame.copy(deep=True)
        cases = detail_cases(frame)
        np.testing.assert_allclose(cases.error, [-1, -2, -9, -10])
        np.testing.assert_allclose(cases.absolute_naive_error, [1, 0, 1, 2])
        periods = error_periods(frame)
        self.assertEqual(periods.period.tolist(), ["2024Q1", "2024Q3"])
        np.testing.assert_allclose(periods.mae, [1.5, 9.5])
        np.testing.assert_allclose(periods.bias, [-1.5, -9.5])
        np.testing.assert_allclose(periods.naive_mae, [.5, 1.5])
        np.testing.assert_allclose(periods.skill_mae_pct, [-200, 100 * (1 - 9.5 / 1.5)])
        pd.testing.assert_frame_equal(frame, before)
        zero_naive = frame.assign(reference_brent=frame.y_true)
        self.assertTrue(error_periods(zero_naive).skill_mae_pct.isna().all())
        for invalid in [pd.concat([frame, frame.assign(seed=123)]),
                        pd.concat([frame, frame.iloc[:1]]), frame.assign(reference_brent=np.nan)]:
            with self.assertRaises(ValueError):
                error_periods(invalid)


class AcademicPackaging(unittest.TestCase):
    def create_sources(self, root):
        for experiment, identifier in EXPERIMENTS.items():
            folder = root / identifier
            folder.mkdir(parents=True)
            phase = "test" if experiment == "final" else "development"
            frames = [saved_pairs(experiment, phase), saved_pairs(experiment, phase, "persistence")]
            predictions = pd.concat(frames, ignore_index=True)
            metrics = derive_metrics(predictions).drop(columns="r2")
            if experiment == "smoke":
                predictions = predictions.drop(columns=["seed", "variant"])
                metrics = metrics.drop(columns=["seed", "variant"])
            predictions.drop(columns="experiment").to_csv(folder / "predictions.csv", index=False)
            metrics.drop(columns="experiment").to_csv(folder / "metrics.csv", index=False)
            report = {"output_sha256": {name: sha256(folder / name) for name in ("metrics.csv", "predictions.csv")}}
            (folder / "experiment.json").write_text(json.dumps(report), encoding="utf-8")

    def test_new_supplement_has_source_hashes_leaves_sources_intact_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "experiments"
            self.create_sources(source)
            original = {str(path): sha256(path) for path in source.rglob("*") if path.is_file()}
            result = build_academic_metrics(source, root / "academic")
            self.assertEqual(result["metric_rows"], 24)
            manifest = _manifest(root / "academic")
            self.assertEqual(len(manifest["source_files"]), 12)
            self.assertFalse(manifest["models_trained"])
            self.assertFalse(manifest["original_results_modified"])
            self.assertEqual(original, {str(path): sha256(path) for path in source.rglob("*") if path.is_file()})
            with self.assertRaises(FileExistsError):
                build_academic_metrics(source, root / "academic")
            with self.assertRaises(ValueError):
                build_academic_metrics(source, source / "bad-output")
            (root / "academic/metrics.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                _manifest(root / "academic")

    def test_tampered_historical_source_is_rejected_before_output_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "experiments"
            self.create_sources(source)
            (source / EXPERIMENTS["final"] / "predictions.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_academic_metrics(source, root / "academic")
            self.assertFalse((root / "academic").exists())


if __name__ == "__main__":
    unittest.main()
