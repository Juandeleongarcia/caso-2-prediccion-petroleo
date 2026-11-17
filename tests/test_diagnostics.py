"""Manual diagnostic values and a bounded, development-only A/B selection."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.evaluation import diagnostics, select_development_configs


def prediction_fixture():
    return pd.DataFrame({
        "variant": ["A"] * 3, "model": ["RNN"] * 3, "seed": [42] * 3,
        "phase": ["development"] * 3, "block_id": ["2015H1", "2015H1", "2015H2"],
        "horizon": [252] * 3, "y_true": [110., 130., 120.], "y_pred": [105., 115., 120.],
        "reference_brent": [100., 100., 100.], "sample_id": [0, 1, 2],
    })


def selection_fixture():
    records = []
    for model in ["RNN", "LSTM", "GRU"]:
        for variant in ["A", "B"]:
            for block in range(6):
                ratio = (0.5 if block < 5 else 2.0) if variant == "A" else 0.9
                for horizon in [5, 63, 252]:
                    records.append({"phase": "development", "model": model, "variant": variant,
                                    "seed": 42, "block_id": f"block_{block}", "horizon": horizon,
                                    "skill_mae_pct": 100 * (1 - ratio), "n": 10 if block < 5 else 10000})
    return pd.DataFrame(records)


class DevelopmentDiagnostics(unittest.TestCase):
    def test_manual_bias_distribution_quantiles_and_delta_units(self):
        result = diagnostics(prediction_fixture())
        self.assertEqual(len(result), 3)
        pooled = result.loc[result["block_id"].eq("all")].iloc[0]
        self.assertEqual(pooled["n"], 3)
        self.assertAlmostEqual(pooled["bias"], -20 / 3)
        self.assertEqual(pooled["true_price_min"], 110.)
        self.assertEqual(pooled["pred_price_max"], 120.)
        self.assertEqual(pooled["true_delta_mean"], 20.)
        self.assertAlmostEqual(pooled["pred_delta_mean"], 40 / 3)
        self.assertEqual(pooled["true_delta_q05"], 11.)
        self.assertEqual(pooled["true_delta_q50"], 20.)
        self.assertEqual(pooled["true_delta_q95"], 29.)
        self.assertAlmostEqual(pooled["delta_std_ratio"], np.std([5., 15., 20.]) / np.std([10., 30., 20.]))
        self.assertAlmostEqual(pooled["delta_correlation"], np.corrcoef([10., 30., 20.], [5., 15., 20.])[0, 1])
        # Pool the individual errors; do not average differently sized block statistics.
        block_bias = result.loc[~result["block_id"].eq("all"), "bias"].mean()
        self.assertNotAlmostEqual(block_bias, pooled["bias"])

    def test_constant_delta_has_undefined_ratio_and_correlation(self):
        frame = prediction_fixture()
        frame["y_true"] = frame["reference_brent"] + 10
        frame["y_pred"] = frame["reference_brent"]
        result = diagnostics(frame)
        pooled = result.loc[result["block_id"].eq("all")].iloc[0]
        self.assertTrue(pd.isna(pooled["delta_std_ratio"]))
        self.assertTrue(pd.isna(pooled["delta_correlation"]))
        self.assertEqual(pooled["pred_delta_std"], 0.)

    def test_horizons_are_separated_and_negative_predictions_remain_visible(self):
        long = prediction_fixture()
        short = long.copy()
        short["horizon"] = 5
        short["y_pred"] = -1.
        combined = pd.concat([long, short], ignore_index=True)
        self.assertEqual(set(diagnostics(combined)["horizon"]), {252})
        result = diagnostics(combined, horizons=(5, 252))
        short_total = result.loc[result["horizon"].eq(5) & result["block_id"].eq("all")].iloc[0]
        self.assertEqual(short_total["negative_predicted_prices"], 3)
        self.assertEqual(short_total["pred_price_min"], -1.)

    def test_diagnostics_reject_non_development_nonfinite_and_duplicate_pairs(self):
        frame = prediction_fixture()
        with self.assertRaisesRegex(ValueError, "development only"):
            diagnostics(frame.assign(phase="test"))
        malformed = frame.copy()
        malformed.loc[0, "y_pred"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite original-unit"):
            diagnostics(malformed)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            diagnostics(pd.concat([frame, frame.iloc[:1]], ignore_index=True))

    def test_selection_weights_each_block_and_horizon_equally(self):
        frame = selection_fixture()
        # A's equal-cell ratio is 0.75 versus B's 0.90. Weighting by n would
        # reverse that conclusion; selecting by sample count is not the protocol.
        self.assertEqual(select_development_configs(frame), {"GRU": "A", "LSTM": "A", "RNN": "A"})

    def test_selection_ignores_stability_seeds_and_pooled_rows_and_resolves_ties(self):
        frame = selection_fixture()
        frame["skill_mae_pct"] = 20.
        extra_seed = frame.assign(seed=123)
        extra_seed.loc[extra_seed["variant"].eq("B"), "skill_mae_pct"] = 100.
        pooled = frame.iloc[:1].copy().assign(block_id="all", skill_mae_pct=-999.)
        combined = pd.concat([frame.iloc[::-1], extra_seed, pooled], ignore_index=True)
        self.assertEqual(set(select_development_configs(combined).values()), {"A"})

    def test_selection_rejects_non_development_missing_cells_and_unequal_pairs(self):
        frame = selection_fixture()
        with self.assertRaisesRegex(ValueError, "development only"):
            select_development_configs(frame.assign(phase="calibration"))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_development_configs(frame.drop(index=0))
        changed = frame.copy()
        changed.loc[0, "n"] += 1
        with self.assertRaisesRegex(ValueError, "same prediction-pair counts"):
            select_development_configs(changed)


if __name__ == "__main__":
    unittest.main()
