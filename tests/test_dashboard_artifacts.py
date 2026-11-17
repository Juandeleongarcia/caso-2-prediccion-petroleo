"""Saved dashboard evidence stays exact, bounded and reproducible."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from oil_forecast.acquisition import sha256
from oil_forecast.artifacts import _manifest
from prepare_dashboard_artifacts import EXPERIMENTS, build_evidence


@unittest.skipUnless((ROOT / "experiments/cpu-final-evaluation-v1/experiment.json").exists(), "Requires saved local experiments")
class DashboardEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.source = ROOT / "experiments"
        cls.output = cls.folder / "dashboard"
        cls.archive = cls.folder / "dashboard.zip"
        cls.built = build_evidence(cls.source, cls.output, cls.archive)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_saved_metrics_curves_and_predictions_are_not_recomputed(self):
        metrics = pd.read_csv(self.output / "metrics.csv")
        histories = pd.read_csv(self.output / "history.csv")
        predictions = pd.read_csv(self.output / "predictions.csv")
        self.assertEqual(set(metrics.experiment), set(EXPERIMENTS))
        for name, identifier in EXPERIMENTS.items():
            original = pd.read_csv(self.source / identifier / "metrics.csv")
            actual = metrics[metrics.experiment.eq(name)].reset_index(drop=True)
            for column in ["mse", "mae", "rmse", "skill_mae_pct"]:
                np.testing.assert_allclose(actual[column], original[column], rtol=1e-14, atol=1e-12)
            self.assertTrue(predictions[predictions.experiment.eq(name)].seed.isin([42, -1]).all())
        self.assertTrue(histories[histories.experiment.eq("final")].validation_loss.isna().all())
        self.assertEqual(len(predictions[predictions.experiment.eq("final") & predictions.phase.eq("test")]), 238 * 3 * 4)
        final_interval = self.source / EXPERIMENTS["final"] / "interval_metrics.csv"
        self.assertEqual(sha256(final_interval), sha256(self.output / "interval_metrics.csv"))
        catalog = json.loads((self.output / "experiment_catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["final"]["configuration"]["selected_model"], "LSTM")

    def test_identical_build_and_existing_destinations_are_preserved(self):
        second = build_evidence(self.source, self.folder / "second", self.folder / "second.zip")
        self.assertEqual(second["archive_sha256"], self.built["archive_sha256"])
        with self.assertRaises(FileExistsError):
            build_evidence(self.source, self.output, self.folder / "new.zip")
        altered = self.folder / "altered"
        shutil.copytree(self.output, altered)
        (altered / "metrics.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            _manifest(altered)

    def test_tampered_source_fails_before_publishing_any_payload(self):
        source = self.folder / "untrusted-sources"
        shutil.copytree(self.source / EXPERIMENTS["smoke"], source / EXPERIMENTS["smoke"])
        with (source / EXPERIMENTS["smoke"] / "history.csv").open("ab") as stream:
            stream.write(b"tampered")
        target = self.folder / "rejected"
        with self.assertRaisesRegex(ValueError, "Historical output hash mismatch"):
            build_evidence(source, target, self.folder / "rejected.zip")
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
