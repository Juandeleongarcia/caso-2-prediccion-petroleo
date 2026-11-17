"""Outer development data cannot choose inner transforms or early checkpoints."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.experiment import load_development_data, prepare_inner_development_fold
from oil_forecast.models import RecurrentForecaster
from oil_forecast.training import set_reproducible, train_with_early_stopping


@unittest.skipUnless((ROOT / "data/supervised/2026-11-11-official/manifest.json").exists(),
                     "Requires the local supervised snapshot")
class InnerOptimizationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
        cls.data = load_development_data(ROOT / "data/supervised" / config["snapshot_id"], config)

    def test_inner_transforms_use_only_inner_rows_and_outer_mutation_changes_neither_subset(self):
        for fold in self.data.folds:
            changed = copy.deepcopy(self.data)
            changed.dataset.X[fold.eval_indices] *= 100
            changed.dataset.y_changes[fold.eval_indices] += 1000
            with self.subTest(block=fold.block_id):
                original = prepare_inner_development_fold(self.data, fold.block_id, 252)
                altered = prepare_inner_development_fold(changed, fold.block_id, 252)
                inner = original["inner_split"]
                expected_rows = np.unique(self.data.dataset.window_rows[inner.train_indices])
                np.testing.assert_array_equal(inner.fit_row_indices, expected_rows)
                self.assertEqual(original["input_transformer"].n_fit_rows_, len(expected_rows))
                self.assertEqual(original["target_transformer"].n_fit_rows_, len(inner.train_indices))
                self.assertFalse(np.intersect1d(inner.train_indices, inner.validation_indices).size)
                self.assertFalse(np.intersect1d(inner.train_indices, fold.eval_indices).size)
                self.assertFalse(np.intersect1d(inner.validation_indices, fold.eval_indices).size)
                for key in ["X_train", "y_train", "X_inner_valid", "y_inner_valid"]:
                    np.testing.assert_array_equal(original[key], altered[key])
                self.assertEqual(original["input_transformer"].to_dict(), altered["input_transformer"].to_dict())
                self.assertEqual(original["target_transformer"].to_dict(), altered["target_transformer"].to_dict())
                self.assertFalse(np.array_equal(original["X_outer"], altered["X_outer"]))

    def test_altered_outer_windows_and_labels_cannot_change_selected_epoch_or_weights(self):
        fold = self.data.folds[0]
        changed = copy.deepcopy(self.data)
        changed.dataset.X[fold.eval_indices] *= 100
        changed.dataset.y_changes[fold.eval_indices] += 1000
        options = {"max_epochs": 2, "patience": 5, "min_delta": 0.0001,
                   "seed": 42, "threads": 1, "learning_rate": 0.001,
                   "weight_decay": 0.0001, "batch_size": 64,
                   "gradient_clip": 1.0, "optimizer": "Adam", "shuffle_train": True}
        states, results = [], []
        for data in [self.data, changed]:
            prepared = prepare_inner_development_fold(data, fold.block_id, 252)
            set_reproducible(42, 1)
            model = RecurrentForecaster("RNN", input_size=20, hidden_size=32, num_layers=1)
            result = train_with_early_stopping(
                model, prepared["X_train"], prepared["y_train"],
                prepared["X_inner_valid"], prepared["y_inner_valid"], options)
            states.append({name: value.clone() for name, value in model.state_dict().items()})
            results.append(result)
        self.assertEqual(results[0]["best_epoch"], results[1]["best_epoch"])
        self.assertEqual(results[0]["validation_loss_at_best"], results[1]["validation_loss_at_best"])
        np.testing.assert_array_equal(results[0]["inner_predictions"], results[1]["inner_predictions"])
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)


if __name__ == "__main__":
    unittest.main()
