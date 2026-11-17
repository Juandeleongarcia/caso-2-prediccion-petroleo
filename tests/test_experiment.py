"""Real-snapshot guards for the short development-only neural experiment."""

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.experiment import (
    _relative_input_path,
    baseline_pairs,
    load_development_data,
    prepare_development_fold,
    require_development,
)


class PortableManifestPaths(unittest.TestCase):
    def test_windows_and_posix_keys_resolve_to_the_same_posix_project_path(self):
        expected = PurePosixPath("/study/data/processed/snapshot/features.csv")
        for key in (r"data\processed\snapshot\features.csv", "data/processed/snapshot/features.csv"):
            with self.subTest(key=key):
                relative = _relative_input_path(key)
                self.assertEqual(relative.as_posix(), "data/processed/snapshot/features.csv")
                self.assertEqual(PurePosixPath("/study").joinpath(*relative.parts), expected)

    def test_nonrelative_or_traversing_manifest_paths_are_rejected(self):
        for key in ("", ".", None, 42, "\x00", "/outside.csv", r"C:\outside.csv",
                    r"C:outside.csv", r"\\host\share\data.csv", "../data.csv", r"data\..\outside.csv"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                _relative_input_path(key)


@unittest.skipUnless((ROOT / "data/supervised/2026-11-11-official/manifest.json").exists(),
                     "Requires the local supervised snapshot")
class DevelopmentExperimentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
        cls.directory = ROOT / "data/supervised" / cls.config["snapshot_id"]
        cls.data = load_development_data(cls.directory, cls.config)

    def test_loader_exposes_exact_six_development_blocks_and_baseline_pairs(self):
        self.assertEqual(len(self.data.folds), 6)
        self.assertEqual({fold.phase for fold in self.data.folds}, {"development"})
        self.assertEqual(set(self.data.transformers), {fold.block_id for fold in self.data.folds})
        self.assertEqual(set(self.data.baseline["phase"]), {"development"})
        self.assertEqual(len(self.data.baseline), 2202)
        exported = pd.concat([baseline_pairs(self.data, fold.block_id) for fold in self.data.folds])
        self.assertFalse(exported.duplicated(["sample_id", "horizon"]).any())
        self.assertEqual(len(exported), 734 * 3)
        self.assertEqual(set(exported["horizon"]), {5, 63, 252})

    def test_every_development_cut_trains_only_on_three_mature_labels(self):
        for fold in self.data.folds:
            samples = self.data.dataset.samples.iloc[fold.train_indices]
            self.assertTrue(samples["decision_at"].le(fold.training_cutoff).all())
            for horizon in self.data.dataset.horizons:
                self.assertTrue(samples[f"label_available_at_{horizon}"].le(fold.training_cutoff).all())
            np.testing.assert_array_equal(fold.fit_row_indices,
                                          np.unique(self.data.dataset.window_rows[fold.train_indices]))
            self.assertFalse(np.intersect1d(fold.train_indices, fold.eval_indices).size)

    def test_preparing_each_fold_preserves_transformer_state_and_finite_shapes(self):
        before = copy.deepcopy(self.data.transformers)
        for fold in self.data.folds:
            prepared = prepare_development_fold(self.data, fold.block_id)
            for role, indices in [("train", fold.train_indices), ("valid", fold.eval_indices)]:
                inputs, outputs = prepared[f"X_{role}"], prepared[f"y_{role}"]
                self.assertEqual(inputs.shape, (len(indices), 60, 20))
                self.assertEqual(outputs.shape, (len(indices), 3))
                self.assertEqual(inputs.dtype, np.float32)
                self.assertEqual(outputs.dtype, np.float32)
                self.assertTrue(np.isfinite(inputs).all())
                self.assertTrue(np.isfinite(outputs).all())
            self.assertEqual(prepared["input_transformer"].to_dict(), before[fold.block_id]["input"])
            self.assertEqual(prepared["target_transformer"].to_dict(), before[fold.block_id]["target"])
        self.assertEqual(self.data.transformers, before)

    def test_calibration_and_test_requests_are_rejected(self):
        for phase in ["calibration", "test", "Development"]:
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(ValueError, "development only"):
                    require_development(phase)
        for block in ["calibration_2021H1", "test_2024H1"]:
            with self.subTest(block=block):
                with self.assertRaisesRegex(ValueError, "non-development"):
                    prepare_development_fold(self.data, block)
                with self.assertRaisesRegex(ValueError, "non-development"):
                    baseline_pairs(self.data, block)

    def test_changed_baseline_date_or_price_is_rejected_without_touching_disk(self):
        block = self.data.folds[0].block_id
        for column, alteration in [("target_date", pd.Timedelta(days=1)), ("y_true", 1.0)]:
            with self.subTest(column=column):
                changed = self.data.baseline.copy(deep=True)
                row = changed.index[changed["block_id"].eq(block)][0]
                changed.loc[row, column] = changed.loc[row, column] + alteration
                isolated = replace(self.data, baseline=changed)
                with self.assertRaises(ValueError):
                    baseline_pairs(isolated, block)

    def test_only_environment_hash_changes_are_recorded_and_inputs_stay_intact(self):
        self.assertTrue(set(self.data.environment_changes).issubset({"pyproject.toml", "poetry.lock"}))
        protected = [self.directory / name for name in ["manifest.json", "dataset.npz", "samples.csv",
                                                       "transformers.json", "baseline_predictions.csv"]]
        protected.append(ROOT / "data/petroleo_dataset.csv")
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        repeated = load_development_data(self.directory, self.config)
        self.assertEqual(repeated.environment_changes, self.data.environment_changes)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})
        self.assertEqual(before[protected[-1]], self.config["original_sha256"])

    @unittest.skipUnless((ROOT / "experiments/cpu-final-evaluation-v1/experiment.json").exists(),
                         "Requires the local final evaluation report")
    def test_all_52_historical_inputs_keep_hashes_with_portable_path_resolution(self):
        report = json.loads((ROOT / "experiments/cpu-final-evaluation-v1/experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["input_sha256"]), 52)
        for key, expected in report["input_sha256"].items():
            relative = _relative_input_path(key)
            path = ROOT.joinpath(*relative.parts)
            if relative.as_posix() in {"pyproject.toml", "poetry.lock"}:
                path = ROOT / "artifacts/local/research-environment" / relative.name
            with self.subTest(path=relative.as_posix()):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
