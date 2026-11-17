"""A fixed development gate cannot promote one lucky horizon, block or seed."""

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oil_forecast.evaluation import assess_development_refinement


def fixture():
    return pd.DataFrame([
        {"phase": "development", "variant": variant, "model": "LSTM", "seed": seed,
         "block_id": block, "horizon": horizon, "n": 100 if block != "all" else 600,
         "mae": 10.0 * factor, "skill_mae_pct": 100 * (1 - factor)}
        for variant, factor in [("B", 1.5), ("C", 1.2)] for seed in (42, 123, 2026)
        for block in [*[f"development_{year}H{half}" for year in (2015, 2017, 2019) for half in (1, 2)], "all"]
        for horizon in (5, 63, 252)
    ])


class RefinementGate(unittest.TestCase):
    def test_consistent_improvement_passes_and_reports_all_seeds(self):
        result = assess_development_refinement(fixture(), "LSTM", "C", (42, 123, 2026))
        self.assertTrue(result["passes"])
        self.assertAlmostEqual(result["relative_gain_pct"], 20)
        self.assertEqual(result["improved_blocks"], 6)
        self.assertEqual(result["improved_horizons"], 3)
        self.assertEqual(len(result["scores_by_seed"]), 3)

    def test_one_favorable_seed_cannot_override_a_worse_paired_seed(self):
        frame = fixture()
        mask = frame.variant.eq("C") & frame.seed.eq(2026)
        frame.loc[mask, "skill_mae_pct"] = -51
        self.assertFalse(assess_development_refinement(frame, "LSTM", "C", (42, 123, 2026))["passes"])

    def test_three_percent_gain_or_improvement_in_only_three_blocks_fails(self):
        frame = fixture()
        frame.loc[frame.variant.eq("C"), "skill_mae_pct"] = 100 * (1 - 1.5 * 0.97)
        self.assertFalse(assess_development_refinement(frame, "LSTM", "C")["passes"])
        frame = fixture()
        frame.loc[frame.variant.eq("C") & frame.block_id.isin([
            "development_2015H1", "development_2017H1", "development_2019H1"]), "skill_mae_pct"] = -50
        result = assess_development_refinement(frame, "LSTM", "C")
        self.assertEqual(result["improved_blocks"], 3)
        self.assertFalse(result["passes"])

    def test_forbidden_phase_missing_cells_duplicate_and_pair_mismatch_rejected(self):
        frame = fixture()
        forbidden = frame.copy()
        forbidden.loc[0, "phase"] = "test"
        unequal = frame.copy()
        unequal.loc[0, "n"] += 1
        for bad in [forbidden, frame.iloc[1:], pd.concat([frame, frame.iloc[[0]]]), unequal]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                assess_development_refinement(bad, "LSTM", "C", (42, 123, 2026))


if __name__ == "__main__":
    unittest.main()
