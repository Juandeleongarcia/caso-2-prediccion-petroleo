"""Final policy choices are development-only and final labels never fit scalers."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oil_forecast.evaluation import freeze_final_choices, score_final_predictions
from oil_forecast.experiment import load_final_evaluation_data, prepare_final_fold


class FinalMetrics(unittest.TestCase):
    def test_phase_baselines_and_original_unit_metrics_remain_separate(self):
        rows = []
        for phase, block, actual in [("calibration", "calibration_2021H1", 20.), ("test", "test_2024H1", 30.)]:
            for model, variant, seed, prediction in [("persistence", "baseline", -1, 10.), ("LSTM", "B", 42, 15.)]:
                for horizon in (5, 63, 252):
                    rows.append(dict(phase=phase, block_id=block, model=model, variant=variant, seed=seed,
                                     sample_id=0 if phase == "calibration" else 1, horizon=horizon,
                                     y_true=actual, y_pred=prediction))
        result = score_final_predictions(pd.DataFrame(rows))
        test = result[(result.phase == "test") & (result.model == "LSTM") & (result.horizon == 5) & (result.block_id == "all")].iloc[0]
        self.assertEqual(test.mae, 15.)
        self.assertEqual(test.rmse, 15.)
        self.assertEqual(test.mse, 225.)
        self.assertEqual(test.bias, -15.)
        self.assertEqual(test.skill_mae_pct, 25.)


@unittest.skipUnless((ROOT / "experiments/cpu-refinement-persistence-init-v1/experiment.json").exists(), "Requires local development results")
class FrozenFinalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / "configs/data.json").read_text())
        cls.data = load_final_evaluation_data(ROOT / "data/supervised" / config["snapshot_id"], config)
        cls.dev = pd.read_csv(ROOT / "experiments/cpu-refinement-persistence-init-v1/metrics.csv")
        cls.runs = pd.DataFrame([run for directory in ["cpu-refinement-persistence-init-v1", "cpu-nested-ab-20epochs-seeds-v1"]
                                for run in json.loads((ROOT / "experiments" / directory / "experiment.json").read_text(encoding="utf-8"))["runs"]])

    def test_selected_variants_epochs_and_network_use_development_only(self):
        choices = freeze_final_choices(self.dev, self.runs)
        self.assertEqual(choices["epochs"], {"RNN": 4, "LSTM": 3, "GRU": 5})
        self.assertEqual(choices["selected_model"], "LSTM")
        self.assertEqual(choices["selected_seed"], 42)
        changed = self.dev.copy()
        changed.loc[0, "phase"] = "test"
        with self.assertRaises(ValueError):
            freeze_final_choices(changed, self.runs)

    def test_final_scalers_and_training_ignore_changed_outer_targets_and_features(self):
        self.assertEqual(len(self.data.folds), 6)
        for fold in self.data.folds:
            original = prepare_final_fold(self.data, fold.block_id)
            changed = copy.deepcopy(self.data)
            changed.dataset.y_changes[fold.eval_indices] += 100000
            changed.dataset.X[fold.eval_indices] *= 1000
            altered = prepare_final_fold(changed, fold.block_id)
            for name in ["X_train", "y_train"]:
                np.testing.assert_array_equal(original[name], altered[name])
            self.assertEqual(original["input_transformer"].to_dict(), altered["input_transformer"].to_dict())
            self.assertEqual(original["target_transformer"].to_dict(), altered["target_transformer"].to_dict())
            train = self.data.dataset.samples.iloc[fold.train_indices]
            self.assertFalse(train.origin_date.dt.year.eq(2024).any())
            for horizon in (5, 63, 252):
                self.assertTrue(train[f"label_available_at_{horizon}"].le(fold.training_cutoff).all())

    def test_final_pairs_are_original_baseline_and_boundary_is_visible_not_rewritten(self):
        baseline = self.data.baseline
        self.assertEqual(baseline[baseline.phase.eq("test")].sample_id.nunique(), 238)
        calibration = baseline[baseline.phase.eq("calibration")]
        self.assertEqual(calibration.sample_id.nunique(), 472)
        deadline = pd.Timestamp("2024-01-01T08:00:00+01:00")
        excluded = calibration.loc[~calibration.label_available_at.lt(deadline), "sample_id"].unique()
        self.assertEqual(excluded.tolist(), [4813])
        self.assertEqual(calibration.loc[~calibration.sample_id.isin(excluded)].sample_id.nunique(), 471)


if __name__ == "__main__":
    unittest.main()
