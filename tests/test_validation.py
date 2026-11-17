"""Verify the real snapshot's joins between samples, refits and saved transforms."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.preprocessing import InputTransformer
from oil_forecast.supervised import build_supervised
from oil_forecast.validation import build_folds


@unittest.skipUnless((ROOT / "data/processed/2026-11-11-official/manifest.json").exists(),
                     "Requires the local prepared snapshot")
class WalkForwardIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
        folder = ROOT / "data/processed" / cls.config["snapshot_id"]
        cls.features = pd.read_csv(folder / "features.csv")
        targets = pd.read_csv(folder / "target.csv")
        names = tuple(name for name in cls.config["sources"] if name in cls.features)
        cls.data = build_supervised(cls.features, targets, names)
        cls.folds = build_folds(cls.data, cls.config)

    def test_all_twelve_refits_use_only_mature_labels(self):
        self.assertEqual(len(self.folds), 12)
        previous_n = 0
        for fold in self.folds:
            train = self.data.samples.iloc[fold.train_indices]
            self.assertTrue(train["decision_at"].le(fold.training_cutoff).all())
            for horizon in self.data.horizons:
                self.assertTrue(train[f"label_available_at_{horizon}"].le(fold.training_cutoff).all())
            self.assertGreater(len(train), previous_n)
            self.assertLess(fold.training_cutoff, fold.refit_at)
            self.assertFalse(np.intersect1d(fold.train_indices, fold.eval_indices).size)
            previous_n = len(train)

    def test_sample_with_old_origin_but_unmatured_252_label_is_purged(self):
        samples = self.data.samples
        for fold in self.folds:
            bad = samples["decision_at"].le(fold.training_cutoff)
            bad &= samples["label_available_at_252"].gt(fold.training_cutoff)
            self.assertGreater(int(bad.sum()), 0)
            self.assertFalse(np.isin(np.flatnonzero(bad), fold.train_indices).any())

    def test_transformer_learns_unique_training_rows_and_ignores_future(self):
        fold = self.folds[0]
        expected_rows = np.unique(self.data.window_rows[fold.train_indices])
        np.testing.assert_array_equal(fold.fit_row_indices, expected_rows)
        values = self.data.feature_values
        transformer = InputTransformer().fit(values.iloc[expected_rows])
        # Distinct dates, not repeated copies from overlapping windows.
        self.assertEqual(transformer.n_fit_rows_, len(expected_rows))
        self.assertLess(transformer.n_fit_rows_, len(fold.train_indices) * 60)
        changed = values.copy()
        future = pd.to_datetime(self.features["decision_at"], utc=True).gt(fold.training_cutoff)
        changed.loc[future] = changed.loc[future] * 1000 + 99999
        rebuilt = InputTransformer().fit(changed.iloc[fold.fit_row_indices])
        self.assertEqual(transformer.to_dict(), rebuilt.to_dict())
        self.assertLess(transformer.n_fit_rows_, len(values))

    def test_evaluation_windows_can_use_prior_history_and_calibration_is_mature(self):
        calibration_deadline = pd.Timestamp(self.config["evaluation"]["calibration_errors_available_before"])
        for fold in self.folds:
            evaluate = self.data.samples.iloc[fold.eval_indices]
            self.assertTrue(evaluate["decision_at"].ge(fold.refit_at).all())
            # The first prediction legitimately needs history before its semester.
            self.assertLess(evaluate.iloc[0]["window_start_date"], evaluate.iloc[0]["origin_date"])
            if fold.phase == "calibration":
                for horizon in self.data.horizons:
                    self.assertTrue(evaluate[f"label_available_at_{horizon}"].le(calibration_deadline).all())

    def test_rebuilding_folds_preserves_exact_pair_indices(self):
        rebuilt = build_folds(self.data, self.config)
        for first, second in zip(self.folds, rebuilt):
            self.assertEqual(first.summary, second.summary)
            np.testing.assert_array_equal(first.train_indices, second.train_indices)
            np.testing.assert_array_equal(first.eval_indices, second.eval_indices)


if __name__ == "__main__":
    unittest.main()
