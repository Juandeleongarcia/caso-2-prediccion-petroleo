"""Manual numerical checks of the persistence baseline and original-scale errors."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.evaluation import persistence_predictions, regression_metrics, score_predictions


class EvaluationContracts(unittest.TestCase):
    def test_errors_match_manual_original_unit_example(self):
        result = regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 5.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_dropped"], 0)
        self.assertAlmostEqual(result["mse"], 5.0 / 3.0)
        self.assertEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(5.0 / 3.0))
        # No normalization: a tenfold price change multiplies RMSE/MAE by 10,
        # while the squared error is multiplied by 100.
        scaled_units = regression_metrics([10.0, 20.0, 30.0], [10.0, 30.0, 50.0])
        self.assertAlmostEqual(scaled_units["mse"], result["mse"] * 100)
        self.assertAlmostEqual(scaled_units["mae"], result["mae"] * 10)
        self.assertAlmostEqual(scaled_units["rmse"], result["rmse"] * 10)

    def test_nonfinite_pairs_are_explicitly_excluded(self):
        result = regression_metrics([10, np.nan, 30, np.inf, 50], [11, 20, np.nan, 40, 48])
        self.assertEqual(result, {"n": 2, "n_dropped": 3, "mse": 2.5, "mae": 1.5,
                                  "rmse": math.sqrt(2.5)})

    def test_no_valid_pairs_have_undefined_not_zero_errors(self):
        result = regression_metrics([np.nan], [80])
        self.assertEqual(result, {"n": 0, "n_dropped": 1, "mse": None, "mae": None, "rmse": None})
        self.assertEqual(regression_metrics([], [])["n"], 0)

    def test_shapes_cannot_broadcast_or_pool_horizons(self):
        for observed, predicted in [([1, 2], [1]), ([[1], [2]], [1, 2]),
                                    ([[1, 2, 3]], [[1, 2, 3]])]:
            with self.subTest(observed=observed, predicted=predicted):
                with self.assertRaisesRegex(ValueError, "identical shape"):
                    regression_metrics(observed, predicted)

    def test_persistence_repeats_exact_known_price_for_all_horizons(self):
        reference = np.array([107.03, 82.345678901, -36.98])
        predictions = persistence_predictions(reference)
        self.assertEqual(predictions.shape, (3, 3))
        for column in range(3):
            np.testing.assert_array_equal(predictions[:, column], reference)
        np.testing.assert_array_equal(predictions, persistence_predictions(reference))
        np.testing.assert_array_equal(reference, [107.03, 82.345678901, -36.98])

    def test_persistence_rejects_unknown_reference_and_invalid_horizons(self):
        with self.assertRaisesRegex(ValueError, "known prices"):
            persistence_predictions([np.nan])
        with self.assertRaisesRegex(ValueError, "known prices"):
            persistence_predictions([[80]])
        for horizons in [(), (5, 5), (5, 0), (True, 63)]:
            with self.subTest(horizons=horizons):
                with self.assertRaisesRegex(ValueError, "Horizons"):
                    persistence_predictions([80], horizons)

    def test_grouped_metrics_keep_phases_blocks_and_horizons_separate(self):
        frame = pd.DataFrame({
            "phase": ["development", "development", "development", "test", "test"],
            "block_id": ["2015-H1", "2015-H1", "2015-H2", "2024-H1", "2024-H1"],
            "horizon": [5, 5, 5, 5, 63], "y_true": [80, 84, 90, 100, 110],
            "y_pred": [81, 81, 94, 105, 100],
        })
        blocks = score_predictions(frame)
        self.assertEqual(len(blocks), 4)
        first = blocks.loc[blocks["block_id"].eq("2015-H1")].iloc[0]
        self.assertEqual(first["n"], 2)
        self.assertEqual(first["mse"], 5)
        self.assertEqual(first["mae"], 2)
        totals = score_predictions(frame, ("phase", "horizon"))
        development = totals.loc[totals["phase"].eq("development")].iloc[0]
        self.assertEqual(development["n"], 3)
        # Pool original errors [1, -3, 4], not an unweighted mean of two blocks.
        self.assertAlmostEqual(development["mse"], 26 / 3)
        self.assertAlmostEqual(development["mae"], 8 / 3)
        self.assertAlmostEqual(development["rmse"], math.sqrt(26 / 3))
        test_63 = totals.loc[totals["phase"].eq("test") & totals["horizon"].eq(63)].iloc[0]
        self.assertEqual(test_63["mae"], 10)

    def test_empty_or_malformed_prediction_export_is_handled(self):
        empty = pd.DataFrame(columns=["phase", "block_id", "horizon", "y_true", "y_pred"])
        self.assertTrue(score_predictions(empty).empty)
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            score_predictions(pd.DataFrame({"phase": ["test"]}))


if __name__ == "__main__":
    unittest.main()
