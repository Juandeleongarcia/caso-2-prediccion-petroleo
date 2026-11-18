"""Public aggregate-only study keeps scientific results and publication boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.public_study import comparison_rows, load_public_metrics

METRICS = ROOT / "artifacts/derived/brent-academic-v1"
ENTRY = ROOT / "deployment/public_app/app.py"
PAGES = ["Visión general", "Predicción", "Comparación", "Evaluación", "Metodología"]
NOTICE = "Demostración de resultados · Sin inferencia neuronal"
SCORES = ["mse", "mae", "rmse", "bias", "r2"]
PUBLIC_FILES = {
    "deployment/public_app/app.py", "deployment/public_app/requirements.txt",
    "deployment/public_app/constraints.txt",
    ".streamlit/config.toml", "src/oil_forecast/__init__.py",
    "src/oil_forecast/public_study.py", "src/oil_forecast/app.css",
    "artifacts/derived/brent-academic-v1/metrics.csv",
    "artifacts/derived/brent-academic-v1/artifact_manifest.json",
}


def digests(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def tearDownModule():
    from streamlit.testing.v1.app_test import TMP_DIR
    TMP_DIR.cleanup()


class AggregateStudyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_csv(METRICS / "metrics.csv")
        cls.metrics = load_public_metrics(METRICS)

    def test_loading_uses_only_verified_aggregates_and_preserves_original_bytes(self):
        before = digests(METRICS)
        loaded = load_public_metrics(METRICS)
        pd.testing.assert_frame_equal(loaded, self.original)
        self.assertEqual(loaded.shape, (1017, 14))
        self.assertGreaterEqual(int(loaded["n"].min()), 111)
        self.assertFalse({"y_true", "y_pred", "reference_brent", "price", "target_date"} & set(loaded))
        self.assertEqual(before, digests(METRICS))

    def test_altered_payload_or_rewritten_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            csv_bytes = (METRICS / "metrics.csv").read_bytes()
            manifest_bytes = (METRICS / "artifact_manifest.json").read_bytes()
            (folder / "metrics.csv").write_bytes(csv_bytes + b"\n")
            (folder / "artifact_manifest.json").write_bytes(manifest_bytes)
            with self.assertRaises(ValueError):
                load_public_metrics(folder)
            # A self-consistent manifest cannot replace the independently pinned one.
            manifest = json.loads(manifest_bytes)
            manifest["files"]["metrics.csv"]["sha256"] = hashlib.sha256(csv_bytes + b"\n").hexdigest()
            manifest["files"]["metrics.csv"]["size"] = len(csv_bytes) + 1
            (folder / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_public_metrics(folder)

    def test_twelve_final_mean_scores_match_saved_seeds_without_pooling_observations(self):
        before = self.metrics.copy(deep=True)
        for horizon in (5, 63, 252):
            source = self.original.loc[self.original.experiment.eq("final") & self.original.phase.eq("test")
                                       & self.original.block_id.eq("all") & self.original.horizon.eq(horizon)]
            actual = comparison_rows(self.metrics, horizon=horizon)
            self.assertEqual(set(actual.model), {"RNN", "LSTM", "GRU", "persistence"})
            self.assertTrue(actual.n.eq(238).all())
            for _, row in actual.iterrows():
                saved = source.loc[source.model.eq(row.model) & source.variant.eq(row.variant)]
                np.testing.assert_allclose(row[SCORES].to_numpy(dtype=float),
                                           saved[SCORES].mean().to_numpy(dtype=float), rtol=1e-12)
                self.assertEqual(int(row.seed_count), 1 if row.model == "persistence" else 3)
                baseline = source.loc[source.model.eq("persistence"), "mae"].iloc[0]
                self.assertAlmostEqual(row.skill_mae_pct, 100 * (1 - saved.mae.mean() / baseline))
        self.assertTrue(comparison_rows(self.metrics, horizon=252).r2.lt(0).all())
        pd.testing.assert_frame_equal(self.metrics, before)

    def test_operational_seed_is_not_replaced_by_a_better_test_seed(self):
        for horizon in (5, 63, 252):
            actual = comparison_rows(self.metrics, horizon=horizon, seed_mode="42")
            self.assertTrue(actual.seed_count.eq(1).all())
            for _, row in actual.iterrows():
                saved_seed = -1 if row.model == "persistence" else 42
                saved = self.original.loc[self.original.experiment.eq("final") & self.original.phase.eq("test")
                    & self.original.block_id.eq("all") & self.original.horizon.eq(horizon)
                    & self.original.model.eq(row.model) & self.original.seed.eq(saved_seed)].iloc[0]
                np.testing.assert_allclose(row[SCORES].to_numpy(dtype=float),
                                           saved[SCORES].to_numpy(dtype=float), rtol=1e-12)
        long = comparison_rows(self.metrics, horizon=252, seed_mode="42").set_index("model")
        self.assertAlmostEqual(long.loc["LSTM", "mae"], 21.525379558738585, places=5)
        self.assertLess(long.loc["LSTM", "skill_mae_pct"], 0)

    def test_calibration_blocks_and_horizons_never_mix_with_test(self):
        for phase, block, horizon in (("calibration", "all", 63),
                                      ("calibration", "calibration_2021H1", 252),
                                      ("test", "test_2024H2", 5)):
            actual = comparison_rows(self.metrics, phase=phase, block_id=block, horizon=horizon)
            source = self.original.loc[self.original.experiment.eq("final") & self.original.phase.eq(phase)
                & self.original.block_id.eq(block) & self.original.horizon.eq(horizon)]
            for _, row in actual.iterrows():
                saved = source.loc[source.model.eq(row.model)]
                self.assertEqual(int(row.n), int(saved.n.iloc[0]))
                self.assertAlmostEqual(row.mae, saved.mae.mean())
        for options in ({"phase": "unknown"}, {"horizon": 7}, {"seed_mode": "best"},
                        {"phase": "test", "block_id": "calibration_2021H1"}, {"experiment": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                comparison_rows(self.metrics, **options)

    def test_development_variants_remain_separate_instead_of_averaging_a_and_b(self):
        actual = comparison_rows(self.metrics, experiment="optimization", phase="development", horizon=63)
        for model in ("RNN", "LSTM", "GRU"):
            self.assertEqual(set(actual.loc[actual.model.eq(model), "variant"]), {"A", "B"})
        self.assertEqual(len(actual), 7)
        self.assertEqual(len(actual.loc[actual.model.eq("persistence")]), 1)


class PublicStudyPages(unittest.TestCase):
    def healthy(self, app):
        self.assertFalse(list(app.exception), [item.value for item in app.exception])
        self.assertFalse(list(app.error), [item.value for item in app.error])
        self.assertTrue(any(NOTICE in str(item.value) for item in app.warning))

    def test_all_five_pages_keep_public_scope_visible_and_offer_no_prediction_inputs(self):
        app = AppTest.from_file(str(ENTRY), default_timeout=30).run()
        self.healthy(app)
        self.assertEqual(list(app.radio(key="public_page").options), PAGES)
        for page in PAGES:
            app.radio(key="public_page").set_value(page).run()
            self.healthy(app)
            self.assertEqual(len(app.number_input), 0)
            self.assertEqual(len(app.file_uploader), 0)
            self.assertEqual(len(app.button), 0)
            if page == "Predicción":
                messages = " ".join(str(item.value) for name in ("info", "warning", "caption", "markdown")
                                    for item in getattr(app, name))
                self.assertIn("sin inferencia", messages.lower())

    def test_comparison_displays_negative_r2_and_the_real_operational_seed(self):
        app = AppTest.from_file(str(ENTRY), default_timeout=30).run()
        app.radio(key="public_page").set_value("Comparación").run()
        self.healthy(app)
        app.selectbox(key="public_compare_horizon").set_value(252).run()
        app.radio(key="public_compare_metric").set_value("R²").run()
        self.healthy(app)
        frames = [item.value for item in app.dataframe]
        self.assertTrue(any("R²" in frame and frame["R²"].lt(0).all() for frame in frames))
        app.radio(key="public_compare_seeds").set_value("42").run()
        self.healthy(app)
        tables = [item.value for item in app.dataframe if "Modelo" in item.value and "MAE" in item.value]
        self.assertTrue(tables)
        lstm = tables[0].loc[tables[0]["Modelo"].astype(str).str.contains("LSTM")].iloc[0]
        self.assertAlmostEqual(float(lstm["MAE"]), 21.52538, places=3)
        # Changing experiments must reset incompatible phases/configurations.
        for experiment in ("optimization", "refinement", "smoke", "final"):
            app.selectbox(key="public_compare_experiment").set_value(experiment).run()
            self.healthy(app)
            for horizon in (5, 63, 252):
                app.selectbox(key="public_compare_horizon").set_value(horizon).run()
                self.healthy(app)
        app.radio(key="public_page").set_value("Evaluación").run()
        for model in ("persistence", "RNN", "LSTM", "GRU"):
            app.selectbox(key="public_eval_model").set_value(model).run()
            for horizon in (5, 63, 252):
                app.selectbox(key="public_eval_horizon").set_value(horizon).run()
                self.healthy(app)


class PublicExportBoundaries(unittest.TestCase):
    def test_export_contains_exactly_public_files_and_runs_without_private_imports_or_network(self):
        from scripts.prepare_public_demo import build_public_demo
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "public-export"
            report = build_public_demo(ROOT, export)
            before = digests(export)
            self.assertEqual(set(before), PUBLIC_FILES)
            self.assertEqual(set(report["files"]), PUBLIC_FILES)
            for path, digest in before.items():
                self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                build_public_demo(ROOT, export)
            self.assertEqual(before, digests(export))
            program = r'''
import importlib.abc,json,socket,sys,urllib.request
from pathlib import Path
def forbidden(*args,**kwargs):
    raise AssertionError("The public study cannot access providers, private modules or model files")
class PublicOnly(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','yfinance'} or fullname in {
            'oil_forecast.artifacts','oil_forecast.inference','oil_forecast.acquisition',
            'oil_forecast.training','oil_forecast.app_data','oil_forecast.academic'}:
            forbidden()
sys.meta_path.insert(0,PublicOnly())
original_connect=socket.socket.connect
def local_loopback_only(sock,address):
    # asyncio uses a local socketpair on Windows to run Streamlit's test loop.
    if isinstance(address,tuple) and address[0] in {'127.0.0.1','::1'}:
        return original_connect(sock,address)
    forbidden()
socket.socket.connect=local_loopback_only
socket.create_connection=forbidden
urllib.request.urlopen=forbidden
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.app_test import TMP_DIR
try:
    assert not Path('data').exists() and not Path('experiments').exists()
    assert not Path('artifacts/local').exists() and not Path('artifacts/releases').exists()
    app=AppTest.from_file(str(Path('deployment/public_app/app.py').resolve()),default_timeout=30).run()
    for page in ['Visión general','Predicción','Comparación','Evaluación','Metodología']:
        app.radio(key='public_page').set_value(page).run()
        assert not list(app.exception),[x.value for x in app.exception]
        assert not list(app.error),[x.value for x in app.error]
        assert any('Demostración de resultados · Sin inferencia neuronal' in x.value for x in app.warning)
    assert 'torch' not in sys.modules and 'yfinance' not in sys.modules
    print(json.dumps({'public_pages':5,'private_resources':False,'network':False,'heavy_models_loaded':False}))
finally:
    TMP_DIR.cleanup()
'''
            environment = os.environ.copy()
            environment.update({"PYTHONIOENCODING": "utf-8", "BRENT_AUTO_PREPARE": "1",
                                "BRENT_AUTO_REFRESH": "1", "HF_TOKEN": "hf_test_only"})
            process = subprocess.run([sys.executable, "-I", "-X", "utf8", "-B", "-W", "error", "-c", program],
                cwd=export, env=environment, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["public_pages"], 5)
            self.assertFalse(result["network"])
            self.assertFalse(result["heavy_models_loaded"])
            self.assertEqual(before, digests(export))


if __name__ == "__main__":
    unittest.main()
