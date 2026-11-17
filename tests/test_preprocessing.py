"""Estadísticas manuales, límites train y persistencia exacta de parámetros."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.preprocessing import InputTransformer, TargetTransformer


class PreprocessingContracts(unittest.TestCase):
    def test_input_statistics_match_manual_train_calculation(self):
        train = np.array([[-2.0, np.nan], [0.0, 10.0], [2.0, 20.0]])
        transformer = InputTransformer().fit(train, ["wti", "other"])
        np.testing.assert_array_equal(transformer.median_, [0.0, 15.0])
        np.testing.assert_array_equal(transformer.mean_, [0.0, 15.0])
        np.testing.assert_allclose(transformer.scale_, [np.sqrt(8 / 3), np.sqrt(50 / 3)])
        scaled = transformer.transform(train)
        np.testing.assert_allclose(scaled[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(scaled[:, :2].std(axis=0), [1.0, 1.0])
        np.testing.assert_array_equal(scaled[:, 2:], [[0, 1], [0, 0], [0, 0]])
        self.assertEqual(transformer.n_fit_rows_, 3)

    def test_all_missing_and_constant_train_columns_have_explicit_fallback(self):
        transformer = InputTransformer().fit([[np.nan, 5.0], [np.nan, 5.0]], ["absent", "constant"])
        np.testing.assert_array_equal(transformer.all_missing_, [True, False])
        np.testing.assert_array_equal(transformer.scale_, [1.0, 1.0])
        # El dato futuro 7 no cambia la imputación aprendida: continúa siendo 0.
        np.testing.assert_array_equal(transformer.transform([[7.0, np.nan]]), [[7.0, 0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(transformer.median_, [0.0, 5.0])

    def test_evaluation_outliers_never_modify_learned_input_parameters(self):
        train = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        transformer = InputTransformer().fit(train, ["a", "b"])
        before_state = transformer.to_dict()
        before_train = transformer.transform(train)
        transformer.transform([[1e12, -1e12], [np.nan, 1e9]])
        self.assertEqual(before_state, transformer.to_dict())
        np.testing.assert_array_equal(before_train, transformer.transform(train))

    def test_window_shape_and_dataframe_column_order_are_preserved(self):
        train = pd.DataFrame({"wti": [-36.98, 8.91], "vix": [20.0, 40.0]})
        transformer = InputTransformer().fit(train)
        windows = np.broadcast_to(train.to_numpy(), (3, 2, 2)).copy()
        self.assertEqual(transformer.transform(windows).shape, (3, 2, 4))
        self.assertEqual(transformer.output_feature_names, ("wti", "vix", "wti__missing", "vix__missing"))
        with self.assertRaisesRegex(ValueError, "orden"):
            transformer.transform(train[["vix", "wti"]])
        np.testing.assert_allclose(transformer.transform(train)[:, 0], [-1.0, 1.0])

    def test_input_json_roundtrip_reproduces_transformations(self):
        transformer = InputTransformer().fit([[1.0, np.nan], [4.0, 3.0]], ["a", "b"])
        restored = InputTransformer.from_dict(json.loads(json.dumps(transformer.to_dict())))
        values = np.array([[[2.0, np.nan], [-4.0, 8.0]]])
        np.testing.assert_array_equal(restored.transform(values), transformer.transform(values))
        self.assertEqual(restored.to_dict(), transformer.to_dict())

    def test_target_statistics_and_inverse_match_manual_values(self):
        train = np.array([[-2.0, 10.0, 5.0], [2.0, 14.0, 5.0]])
        transformer = TargetTransformer().fit(train)
        np.testing.assert_array_equal(transformer.mean_, [0.0, 12.0, 5.0])
        np.testing.assert_array_equal(transformer.scale_, [2.0, 2.0, 1.0])
        np.testing.assert_array_equal(transformer.transform(train), [[-1, -1, 0], [1, 1, 0]])
        np.testing.assert_array_equal(transformer.inverse_transform([2.0, -3.0, 1.0]), [4.0, 6.0, 6.0])
        state = json.loads(json.dumps(transformer.to_dict()))
        restored = TargetTransformer.from_dict(state)
        future = [[-50.0, 123.0, 0.0]]
        np.testing.assert_allclose(restored.inverse_transform(restored.transform(future)), future, atol=1e-12)
        self.assertEqual(transformer.to_dict(), state)

    def test_target_evaluation_does_not_refit_and_requires_observed_values(self):
        transformer = TargetTransformer().fit([[1, 2, 3], [3, 4, 5]])
        before = transformer.to_dict()
        transformer.transform([[1e10, -1e9, 2000]])
        self.assertEqual(transformer.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "finitos"):
            transformer.transform([[1, np.nan, 3]])
        with self.assertRaisesRegex(ValueError, "variables"):
            TargetTransformer().fit([[1, 2]])

    def test_invalid_or_unfitted_transformers_fail_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "ajustado"):
            InputTransformer().transform([[1.0]])
        with self.assertRaisesRegex(ValueError, "ajustado"):
            TargetTransformer().inverse_transform([[1, 2, 3]])
        with self.assertRaises(ValueError):
            InputTransformer().fit(np.empty((0, 2)))
        with self.assertRaisesRegex(ValueError, "finitos"):
            InputTransformer().fit([[float("inf")]])
        with self.assertRaisesRegex(ValueError, "únicos"):
            InputTransformer().fit([[1, 2]], ["a", "a"])


if __name__ == "__main__":
    unittest.main()
