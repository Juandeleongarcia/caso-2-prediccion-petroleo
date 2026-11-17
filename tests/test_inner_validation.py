"""Nested chronology, label maturation and training-only scaling on real cuts."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.experiment import load_development_data
from oil_forecast.preprocessing import InputTransformer, TargetTransformer
from oil_forecast.temporal import decision_times
from oil_forecast.validation import build_inner_split


@unittest.skipUnless((ROOT / "data/supervised/2026-11-11-official/manifest.json").exists(),
                     "Requires the local supervised snapshot")
class InnerValidationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
        cls.data = load_development_data(ROOT / "data/supervised" / config["snapshot_id"], config)

    def test_six_inner_splits_have_three_mature_labels_and_no_outer_overlap(self):
        dataset = self.data.dataset
        for fold in self.data.folds:
            with self.subTest(block=fold.block_id):
                outer_before = fold.train_indices.copy(), fold.eval_indices.copy(), fold.training_cutoff
                split = build_inner_split(dataset, fold)
                train = dataset.samples.iloc[split.train_indices]
                valid = dataset.samples.iloc[split.validation_indices]
                self.assertTrue(train["decision_at"].le(split.training_cutoff).all())
                for horizon in dataset.horizons:
                    self.assertTrue(train[f"label_available_at_{horizon}"].le(split.training_cutoff).all())
                    self.assertTrue(valid[f"label_available_at_{horizon}"].le(fold.training_cutoff).all())
                self.assertFalse(np.intersect1d(split.train_indices, split.validation_indices).size)
                self.assertFalse(np.intersect1d(split.train_indices, fold.eval_indices).size)
                self.assertFalse(np.intersect1d(split.validation_indices, fold.eval_indices).size)
                np.testing.assert_array_equal(fold.train_indices, outer_before[0])
                np.testing.assert_array_equal(fold.eval_indices, outer_before[1])
                self.assertEqual(fold.training_cutoff, outer_before[2])

    def test_validation_tail_counts_calendar_positions_not_filtered_rows(self):
        samples = self.data.dataset.samples
        for fold in self.data.folds:
            split = build_inner_split(self.data.dataset, fold)
            self.assertEqual(len(pd.bdate_range(split.validation_start, split.validation_end)), 252)
            self.assertEqual(split.validation_end, samples.iloc[fold.train_indices]["origin_date"].max())
            expected = fold.train_indices[samples.iloc[fold.train_indices]["origin_date"].ge(split.validation_start)]
            np.testing.assert_array_equal(split.validation_indices, expected)
            self.assertLess(len(split.validation_indices), 252)

    def test_labels_crossing_inner_cutoff_are_purged_even_with_old_origins(self):
        samples = self.data.dataset.samples
        for fold in self.data.folds:
            split = build_inner_split(self.data.dataset, fold)
            candidates = samples.iloc[fold.train_indices]
            old_but_immature = candidates["decision_at"].le(split.training_cutoff)
            old_but_immature &= candidates["label_available_at_252"].gt(split.training_cutoff)
            excluded = fold.train_indices[old_but_immature]
            self.assertGreater(len(excluded), 0)
            self.assertFalse(np.isin(excluded, split.train_indices).any())
            self.assertGreater(split.summary["n_purged_from_outer_train"], 0)
            self.assertEqual(len(fold.train_indices), len(split.train_indices)
                             + len(split.validation_indices) + split.summary["n_purged_from_outer_train"])

    def test_scalers_use_only_unique_inner_training_rows_and_labels(self):
        dataset = self.data.dataset
        fold = self.data.folds[0]
        split = build_inner_split(dataset, fold)
        unique_rows = np.unique(dataset.window_rows[split.train_indices])
        np.testing.assert_array_equal(split.fit_row_indices, unique_rows)
        inputs = InputTransformer().fit(dataset.feature_values.iloc[unique_rows])
        targets = TargetTransformer().fit(dataset.y_changes[split.train_indices])
        self.assertEqual(inputs.n_fit_rows_, len(unique_rows))
        self.assertEqual(targets.n_fit_rows_, len(split.train_indices))
        self.assertLess(len(unique_rows), len(fold.fit_row_indices))
        changed_inputs = dataset.feature_values.copy()
        outside_rows = np.setdiff1d(np.arange(len(changed_inputs)), unique_rows)
        changed_inputs.iloc[outside_rows] = changed_inputs.iloc[outside_rows] * 1000 + 50000
        changed_targets = dataset.y_changes.copy()
        changed_targets[np.setdiff1d(np.arange(len(changed_targets)), split.train_indices)] += 50000
        self.assertEqual(inputs.to_dict(), InputTransformer().fit(changed_inputs.iloc[unique_rows]).to_dict())
        self.assertEqual(targets.to_dict(), TargetTransformer().fit(changed_targets[split.train_indices]).to_dict())

    def test_outer_future_labels_cannot_change_inner_split(self):
        dataset = self.data.dataset
        fold = self.data.folds[0]
        original = build_inner_split(dataset, fold)
        changed = dataset.samples.copy(deep=True)
        future = changed["decision_at"].gt(fold.training_cutoff)
        changed.loc[future, "y_price_252"] += 50000
        changed.loc[future, "label_available_at_252"] += pd.Timedelta(days=999)
        altered = build_inner_split(replace(dataset, samples=changed), fold)
        self.assertEqual(original.summary, altered.summary)
        np.testing.assert_array_equal(original.train_indices, altered.train_indices)
        np.testing.assert_array_equal(original.validation_indices, altered.validation_indices)
        np.testing.assert_array_equal(original.fit_row_indices, altered.fit_row_indices)

    def test_immature_outer_label_and_publication_before_observation_are_rejected(self):
        dataset = self.data.dataset
        fold = self.data.folds[0]
        split = build_inner_split(dataset, fold)
        index = split.validation_indices[-1]
        bad = dataset.samples.copy(deep=True)
        bad.loc[index, "label_available_at_252"] = fold.training_cutoff + pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            build_inner_split(replace(dataset, samples=bad), fold)
        bad = dataset.samples.copy(deep=True)
        target = bad.loc[index, "target_date_252"]
        bad.loc[index, "label_available_at_252"] = decision_times([target - pd.offsets.BDay(1)])[0]
        with self.assertRaisesRegex(ValueError, "precedes"):
            build_inner_split(replace(dataset, samples=bad), fold)

    def test_non_development_or_invalid_tail_is_rejected(self):
        dataset, fold = self.data.dataset, self.data.folds[0]
        with self.assertRaisesRegex(ValueError, "development"):
            build_inner_split(dataset, replace(fold, phase="test"))
        for count in [0, -1, True, 2.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_inner_split(dataset, fold, count)


if __name__ == "__main__":
    unittest.main()
