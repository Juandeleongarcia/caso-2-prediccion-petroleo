"""Exercise the actual Streamlit script, selectors and offline serving contract."""

from __future__ import annotations

import io
import base64
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

from oil_forecast.app_charts import forecast_chart
from oil_forecast.app_data import forecast_for_model, load_app_data, load_services
from oil_forecast.artifacts import fetch_bundle
from oil_forecast.acquisition import sha256


def tearDownModule():
    # Streamlit 1.67 creates this process-wide test directory on import.
    # Explicit cleanup prevents a ResourceWarning at interpreter shutdown.
    from streamlit.testing.v1.app_test import TMP_DIR
    TMP_DIR.cleanup()


class StreamlitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ROOT / "artifacts/local/brent-frozen-v1"
        cls.evidence = ROOT / "artifacts/local/brent-dashboard-v1"

    def setUp(self):
        self.live_temporary = tempfile.TemporaryDirectory()
        self.live_environment = patch.dict(os.environ, {"BRENT_AUTO_REFRESH": "0", "BRENT_LIVE_DIR": self.live_temporary.name})
        self.live_environment.start()
        self.addCleanup(self.live_environment.stop)
        self.addCleanup(self.live_temporary.cleanup)

    def fake_snapshot(self, config, root, snapshot_id=None):
        folder = Path(root) / snapshot_id
        folder.mkdir(parents=True)
        data = load_app_data(self.bundle)
        data["window"].to_csv(folder / "features.csv", index=False)
        data["history"].to_csv(folder / "target.csv", index=False)
        protocol = {"snapshot_id": snapshot_id, "data_updated_at": datetime.now(timezone.utc).isoformat(),
            "config": config, "weights_updated": False,
            "sources": {name: {"status": "ok"} for name in config["sources"]},
            "output_sha256": {name: sha256(folder / name) for name in ("features.csv", "target.csv")}}
        (folder / "manifest.json").write_text(json.dumps(protocol), encoding="utf-8")
        return {"window": data["window"], "history": data["history"], "protocol": protocol, "directory": str(folder)}

    def test_automatic_startup_checks_once_and_manual_double_click_is_debounced(self):
        with patch.dict(os.environ, {"BRENT_AUTO_REFRESH": "1"}), patch("oil_forecast.app_data.refresh_market_data", side_effect=self.fake_snapshot) as provider:
            app = self.app()
            self.assertEqual(provider.call_count, 1)
            for page in ("Predicción", "Comparación", "Visión general"):
                app.radio(key="page").set_value(page).run()
                self.healthy(app)
            self.assertEqual(provider.call_count, 1)
            app.button(key="refresh_data").click().run()
            self.healthy(app)
            self.assertEqual(provider.call_count, 1)

    def test_manual_refresh_activates_valid_snapshot_without_changing_models(self):
        protected = {file.name: sha256(file) for file in self.bundle.iterdir()}
        with patch("oil_forecast.app_data.refresh_market_data", side_effect=self.fake_snapshot) as provider:
            app = self.app()
            app.button(key="refresh_data").click().run()
            self.healthy(app)
            self.assertEqual(provider.call_count, 1)
            self.assertIn("live_market", app.session_state)
            self.assertTrue(app.session_state["live_market"]["protocol"]["snapshot_id"].startswith("live-"))
        self.assertEqual(protected, {file.name: sha256(file) for file in self.bundle.iterdir()})

    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.healthy(app)
        return app

    def healthy(self, app):
        self.assertFalse(list(app.exception), [str(item.value) for item in app.exception])
        self.assertFalse(list(app.error), [item.value for item in app.error])

    def test_all_pages_and_prediction_selectors_offline(self):
        with patch("oil_forecast.acquisition.urlopen", side_effect=AssertionError("Unexpected network access")), \
                patch("oil_forecast.artifacts.urlopen", side_effect=AssertionError("Unexpected artifact download")):
            app = self.app()
            for page in ("Predicción", "Comparación", "Evaluación", "Metodología", "Visión general"):
                app.radio(key="page").set_value(page).run()
                self.healthy(app)
            app.radio(key="page").set_value("Predicción").run()
            for model in ("persistence", "RNN", "LSTM", "GRU"):
                app.selectbox(key="forecast_model").set_value(model).run()
                for horizon in (5, 63, 252):
                    app.selectbox(key="forecast_horizon").set_value(horizon).run()
                    self.healthy(app)
                    self.assertEqual(len(app.get("plotly_chart")), 1)
            app.radio(key="forecast_source").set_value("Archivo de evaluación").run()
            self.healthy(app)
            self.assertTrue(app.selectbox(key="saved_origin").value.startswith("2024"))
            app.selectbox(key="saved_phase").set_value("calibration").run()
            self.healthy(app)

    def test_comparison_and_experiment_selectors(self):
        app = self.app()
        app.radio(key="page").set_value("Comparación").run()
        for phase in ("test", "calibration", "development"):
            app.selectbox(key="compare_phase").set_value(phase).run()
            for horizon in (5, 63, 252):
                app.selectbox(key="compare_horizon").set_value(horizon).run()
                self.healthy(app)
            app.radio(key="compare_seeds").set_value("Operativa 42").run()
            self.healthy(app)
        app.radio(key="page").set_value("Evaluación").run()
        for experiment in ("final", "optimization", "refinement", "smoke"):
            app.selectbox(key="eval_experiment").set_value(experiment).run()
            for model in ("LSTM", "RNN", "GRU", "persistence"):
                app.selectbox(key="eval_model").set_value(model).run()
                self.healthy(app)

    def test_three_horizon_table_matches_saved_scores_and_nonapplicable_plots_are_omitted(self):
        from oil_forecast.app_data import aggregate_metrics
        app = self.app()
        app.radio(key="page").set_value("Comparación").run()
        self.healthy(app)
        across = app.dataframe[1].value.set_index("Modelo")
        data = load_app_data(self.bundle)
        names = {"persistence": "Persistencia", "RNN": "RNN-D", "LSTM": "LSTM-B", "GRU": "GRU-C"}
        for days, column in [(5, "Una semana · 5 días"), (63, "Tres meses · 63 días"), (252, "Un año · 252 días")]:
            expected = aggregate_metrics(data["metrics"], "test", days)
            for row in expected.to_dict("records"):
                self.assertAlmostEqual(across.loc[names[row["model"]], column], row["mae"], places=12)
        app.radio(key="page").set_value("Evaluación").run()
        app.selectbox(key="eval_model").set_value("persistence").run()
        self.healthy(app)
        self.assertEqual(len(app.get("plotly_chart")), 3)
        self.assertTrue(any("no necesita entrenamiento" in item.value for item in app.info))
        app.selectbox(key="eval_experiment").set_value("optimization").run()
        self.healthy(app)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertTrue(all("_" not in option for option in app.selectbox(key="eval_block").options))
        for figure in app.get("plotly_chart"):
            self.assertTrue(json.loads(figure.proto.spec)["data"])

    def test_missing_artifacts_are_explained(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"BRENT_BUNDLE_DIR": temporary}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.error), 1)
            self.assertIn("artefactos", app.error[0].value)
            app.radio(key="page").set_value("Metodología").run()
            self.assertFalse(list(app.exception))
            self.assertTrue(any("Metodología" in item.value for item in app.title))

    def test_r2_and_error_analysis_show_real_scores_without_hiding_negative_values(self):
        from oil_forecast.app_data import aggregate_metrics
        data = load_app_data(self.bundle, self.evidence)
        app = self.app()
        app.radio(key="page").set_value("Comparación").run()
        app.radio(key="compare_metric").set_value("R²").run()
        for phase in ("test", "calibration", "development"):
            app.selectbox(key="compare_phase").set_value(phase).run()
            for horizon in (5, 63, 252):
                app.selectbox(key="compare_horizon").set_value(horizon).run()
                self.healthy(app)
        app.selectbox(key="compare_phase").set_value("test").run()
        app.selectbox(key="compare_horizon").set_value(252).run()
        figure = json.loads(app.get("plotly_chart")[0].proto.spec)
        expected = aggregate_metrics(data["metrics"], "test", 252).set_index("model").reindex(["persistence", "RNN", "LSTM", "GRU"])
        encoded = figure["data"][0]["y"]
        actual = np.frombuffer(base64.b64decode(encoded["bdata"]), dtype=encoded["dtype"]) if isinstance(encoded, dict) else encoded
        np.testing.assert_allclose(actual, expected.r2, rtol=1e-12)
        self.assertTrue((expected.r2 < 0).any())
        self.assertEqual(int(np.argmax(figure["data"][0]["marker"]["line"]["width"])), int(np.argmax(expected.r2)))
        app.radio(key="page").set_value("Evaluación").run()
        app.selectbox(key="eval_horizon").set_value(252).run()
        self.healthy(app)
        frame = data["evaluation"]["predictions"]
        pairs = frame.loc[frame.experiment.eq("final") & frame.phase.eq("test") & frame.model.eq("LSTM") & frame.seed.eq(42) & frame.horizon.eq(252)].copy()
        tables = [item.value for item in app.dataframe]
        periods = next(table for table in tables if "Trimestre objetivo" in table)
        worst = next(table for table in tables if "Error absoluto" in table)
        absolute = (pairs.y_pred - pairs.y_true).abs()
        self.assertAlmostEqual(worst["Error absoluto"].max(), round(absolute.max(), 3))
        self.assertEqual(int(periods.Orígenes.sum()), len(pairs))
        for row in periods.to_dict("records"):
            group = pairs.loc[pd.to_datetime(pairs.target_date).dt.to_period("Q").astype(str).eq(row["Trimestre objetivo"])]
            self.assertAlmostEqual(row["MAE"], round((group.y_pred-group.y_true).abs().mean(), 3))
            self.assertAlmostEqual(row["Sesgo"], round((group.y_pred-group.y_true).mean(), 3))
        app.radio(key="eval_error_cases").set_value("Menores errores").run()
        self.healthy(app)
        best = next(item.value for item in app.dataframe if "Error absoluto" in item.value)
        self.assertAlmostEqual(best["Error absoluto"].min(), round(absolute.min(), 3))

    def test_application_portable_without_research_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            shutil.copy2(ROOT / "app.py", isolated / "app.py")
            shutil.copytree(ROOT / "src", isolated / "src", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "configs", isolated / "configs")
            for source in (self.bundle, self.evidence):
                shutil.copytree(source, isolated / "artifacts/local" / source.name)
            academic = ROOT / "artifacts/derived/brent-academic-v1"
            shutil.copytree(academic, isolated / "artifacts/derived" / academic.name)
            self.assertFalse((isolated / "data").exists())
            self.assertFalse((isolated / "experiments").exists())
            with patch("oil_forecast.acquisition.urlopen", side_effect=AssertionError("Offline required")):
                app = AppTest.from_file(str(isolated / "app.py"), default_timeout=30).run()
                self.healthy(app)
                for page in ("Predicción", "Comparación", "Evaluación", "Metodología"):
                    app.radio(key="page").set_value(page).run()
                    self.healthy(app)
                    if page == "Comparación":
                        self.assertIn("R²", app.radio(key="compare_metric").options)

    def test_prediction_timing_pair_and_archived_context_match_backend(self):
        from oil_forecast.app_data import stored_forecast
        from oil_forecast.app_pages import money, date
        data = load_app_data(self.bundle, self.evidence)
        services = load_services(self.bundle)
        expected = forecast_for_model(services, data["window"], "LSTM")
        app = self.app()
        app.radio(key="page").set_value("Predicción").run()
        app.selectbox(key="forecast_model").set_value("LSTM").run()
        for horizon in (5, 63, 252):
            app.selectbox(key="forecast_horizon").set_value(horizon).run()
            self.healthy(app)
            row = next(item for item in expected["forecasts"] if item["horizon"] == horizon)
            html = "\n".join(item.proto.body for item in app.get("html"))
            for value in (money(row["price"]), money(row["persistence_price"]), date(row["target_date"]), date(expected["reference_observation_date"])):
                self.assertIn(value, html)
            self.assertIn("Origen lógico", html)
            self.assertIn("Mismo origen y objetivo", html)
            self.assertTrue(any("Último Brent observado en esta copia" in item.value for item in app.caption))
            provenance = next(item.value for item in app.dataframe if "Admisible" in item.value)
            self.assertEqual(len(provenance), 10)
            brent = provenance.set_index("Serie").loc["brent"]
            self.assertEqual(brent.Observación, date(expected["reference_observation_date"]))
            self.assertEqual(brent.Admisible, date(expected["reference_available_at"]))
        app.radio(key="forecast_source").set_value("Archivo de evaluación").run()
        for phase in ("test", "calibration"):
            app.selectbox(key="saved_phase").set_value(phase).run()
            self.healthy(app)
            self.assertFalse(any("Último Brent observado en esta copia" in item.value for item in app.caption))
            self.assertFalse(any("Admisible" in item.value for item in app.dataframe))
            saved = data["evaluation"]["predictions"]
            saved = saved.loc[saved.experiment.eq("final") & saved.phase.eq(phase)]
            expected_saved = stored_forecast(saved, "LSTM", 252, app.selectbox(key="saved_origin").value)
            html = "\n".join(item.proto.body for item in app.get("html"))
            self.assertIn(money(expected_saved["forecasts"][0]["price"]), html)
            self.assertFalse(any("Datos adquiridos el" in item.value for item in app.caption))

    def test_evaluation_labels_initial_epoch_and_retrospective_bands_are_explicit(self):
        app = self.app()
        app.radio(key="page").set_value("Evaluación").run()
        self.assertEqual(app.selectbox(key="eval_model").options, ["LSTM", "RNN", "GRU", "Persistencia"])
        app.selectbox(key="eval_experiment").set_value("optimization").run()
        app.selectbox(key="eval_model").set_value("RNN").run()
        for variant in ("A", "B"):
            app.selectbox(key="eval_variant").set_value(variant).run()
            self.healthy(app)
            self.assertEqual(app.selectbox(key="eval_model").value, "RNN")
            self.assertTrue(any("Época 0" in item.value for item in app.caption))
        app.selectbox(key="eval_experiment").set_value("refinement").run()
        app.selectbox(key="eval_variant").set_value("D").run()
        self.assertTrue(any("SmoothL1" in item.value for item in app.caption))
        app.selectbox(key="eval_experiment").set_value("final").run()
        app.selectbox(key="eval_model").set_value("LSTM").run()
        app.selectbox(key="eval_phase").set_value("calibration").run()
        self.healthy(app)
        self.assertTrue(any("Bandas retrospectivas" in item.value for item in app.caption))
        app.selectbox(key="eval_phase").set_value("test").run()
        app.selectbox(key="eval_horizon").set_value(252).run()
        self.healthy(app)
        self.assertTrue(any("238 orígenes" in item.value and "21,525" in item.value and "11,435" in item.value for item in app.info))

    def test_failed_refresh_keeps_offline_snapshot(self):
        with patch("oil_forecast.app_data.refresh_market_data", side_effect=OSError("Provider unavailable")):
            app = self.app()
            app.button(key="refresh_data").click().run()
            self.assertFalse(list(app.exception))
            self.assertTrue(any("última copia válida" in item.value for item in app.error))
            self.assertNotIn("live_market", app.session_state)
            self.assertEqual(len(app.get("plotly_chart")), 1)

    def test_future_points_and_intervals_come_from_real_models(self):
        data = load_app_data(self.bundle)
        services = load_services(self.bundle)
        for model in ("persistence", "RNN", "LSTM", "GRU"):
            forecast = forecast_for_model(services, data["window"], model)
            self.assertEqual([row["horizon"] for row in forecast["forecasts"]], [5, 63, 252])
            for row in forecast["forecasts"]:
                figure = forecast_chart(data["history"], forecast, row["horizon"])
                future = [trace for trace in figure.data if len(trace.x) == 1
                          and pd.Timestamp(trace.x[0]) > pd.Timestamp(forecast["origin_date"])]
                self.assertEqual(len(future), 1 if model == "persistence" else 2)
                estimate = future[-1]
                self.assertEqual(estimate.mode, "markers")
                self.assertEqual(float(estimate.y[0]), row["price"])
                self.assertAlmostEqual(float(estimate.error_y.array[0]), row["upper"] - row["price"])
                self.assertAlmostEqual(float(estimate.error_y.arrayminus[0]), row["price"] - row["lower"])


    def test_cold_cloud_start_downloads_pinned_private_bundles_without_research_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            shutil.copy2(ROOT / "app.py", isolated / "app.py")
            shutil.copytree(ROOT / "src", isolated / "src", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "configs", isolated / "configs")
            shutil.copytree(ROOT / "artifacts/derived", isolated / "artifacts/derived")
            environment = {
                "BRENT_BUNDLE_DIR": str(isolated / "artifacts/local/brent-frozen-v1"),
                "BRENT_DASHBOARD_DIR": str(isolated / "artifacts/local/brent-dashboard-v1"),
                "BRENT_AUTO_PREPARE": "1", "BRENT_HF_REPO_ID": "student/brent-models",
                "BRENT_HF_REVISION": "a" * 40, "HF_TOKEN": "hf_test_only",
                "BRENT_INFERENCE_URL": "", "BRENT_EVIDENCE_URL": "",
            }

            def response(request, timeout):
                self.assertEqual(request.get_header("Authorization"), "Bearer hf_test_only")
                self.assertIn("/resolve/" + "a" * 40 + "/", request.full_url)
                archive = ROOT / "artifacts/releases" / request.full_url.rsplit("/", 1)[1]
                result = io.BytesIO(archive.read_bytes())
                result.geturl = lambda: "https://cdn.example.test/authorized-download"
                return result

            with patch.dict(os.environ, environment), patch("oil_forecast.artifacts.build_opener") as opener, \
                    patch("oil_forecast.acquisition.urlopen", side_effect=AssertionError("Frozen mode required")):
                opener.return_value.open.side_effect = response
                app = AppTest.from_file(str(isolated / "app.py"), default_timeout=30).run()
                self.healthy(app)
                for page in ("Predicción", "Comparación", "Evaluación", "Metodología"):
                    app.radio(key="page").set_value(page).run()
                    self.healthy(app)
                self.assertEqual(opener.return_value.open.call_count, 2)
                self.assertTrue(app.session_state["resource_bootstrap_attempted"])
                self.assertFalse((isolated / "data").exists())
                self.assertFalse((isolated / "experiments").exists())
                self.assertFalse((isolated / "artifacts/releases").exists())
                for element in list(app.error) + list(app.warning) + list(app.info) + list(app.code):
                    self.assertNotIn("hf_test_only", str(element.value))

    def test_failed_cold_start_does_not_leak_secrets_or_retry_on_navigation(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary)
            environment = {
                "BRENT_BUNDLE_DIR": str(missing / "inference"),
                "BRENT_DASHBOARD_DIR": str(missing / "evidence"),
                "BRENT_AUTO_PREPARE": "1", "HF_TOKEN": "hf_test_only",
            }
            with patch.dict(os.environ, environment), \
                    patch("oil_forecast.deployment.prepare_application_resources",
                          side_effect=OSError("hf_test_only private transport detail")) as prepare:
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                self.assertFalse(list(app.exception))
                self.assertTrue(list(app.error))
                app.radio(key="page").set_value("Metodología").run()
                self.assertEqual(prepare.call_count, 1)
                for element in list(app.error) + list(app.warning) + list(app.info) + list(app.code):
                    self.assertNotIn("hf_test_only", str(element.value))
                app.button(key="prepare_resources").click().run()
                self.assertEqual(prepare.call_count, 2)
                self.assertFalse(list(app.exception))
                self.assertFalse((missing / "inference").exists())
                self.assertFalse((missing / "evidence").exists())

    def test_secret_bool_disables_provider_refresh_without_environment_export(self):
        environment = {key: value for key, value in os.environ.items() if key != "BRENT_AUTO_REFRESH"}
        with patch.dict(os.environ, environment, clear=True), \
                patch("oil_forecast.app_data.refresh_market_data",
                      side_effect=AssertionError("Provider use is disabled by the hosting secret")) as provider:
            self.assertNotIn("BRENT_AUTO_REFRESH", os.environ)
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            app.secrets["BRENT_AUTO_REFRESH"] = False
            app.secrets["BRENT_REFRESH_HOURS"] = 36
            app.run()
            self.healthy(app)
            self.assertFalse(app.toggle(key="auto_refresh").value)
            self.assertTrue(any("36 horas" in item.value for item in app.caption))
            app.radio(key="page").set_value("Predicción").run()
            self.healthy(app)
            self.assertFalse(app.toggle(key="auto_refresh").value)
            provider.assert_not_called()

    def test_registered_manifest_is_checked_before_loading_existing_frozen_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            inference, evidence = folder / "inference", folder / "evidence"
            shutil.copytree(self.bundle, inference)
            shutil.copytree(self.evidence, evidence)
            manifest_path = inference / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payloads = {file.name: sha256(file) for file in inference.iterdir() if file != manifest_path}
            manifest["logical_refit_at"] = "2099-01-01T00:00:00+00:00"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            altered_hash = sha256(manifest_path)
            environment = {"BRENT_BUNDLE_DIR": str(inference), "BRENT_DASHBOARD_DIR": str(evidence),
                           "BRENT_AUTO_PREPARE": "0"}
            with patch.dict(os.environ, environment), \
                    patch("oil_forecast.inference.torch.load",
                          side_effect=AssertionError("Reject the manifest before loading weights")) as weights, \
                    patch("oil_forecast.deployment.fetch_bundle",
                          side_effect=AssertionError("Existing resources must not be downloaded")) as fetch:
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                self.assertFalse(list(app.exception))
                self.assertTrue(list(app.error))
                weights.assert_not_called()
                fetch.assert_not_called()
            self.assertEqual(sha256(manifest_path), altered_hash)
            self.assertEqual(payloads, {file.name: sha256(file) for file in inference.iterdir()
                                        if file != manifest_path})


class RemoteBundleTests(unittest.TestCase):
    def test_fetch_requires_https_and_checks_digest_before_installing(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "payload"
            with self.assertRaises(ValueError):
                fetch_bundle("http://example.test/model.zip", destination, "a" * 64)
            response = io.BytesIO(b"wrong bundle")
            response.geturl = lambda: "https://example.test/model.zip"
            with patch("oil_forecast.artifacts.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    fetch_bundle("https://example.test/model.zip", destination, "a" * 64)
            self.assertFalse(destination.exists())
            registry = json.loads((ROOT / "configs/serving_bundle.json").read_text(encoding="utf-8"))
            response = io.BytesIO((ROOT / registry["archive"]["local_path"]).read_bytes())
            response.geturl = lambda: "https://example.test/model.zip"
            with patch("oil_forecast.artifacts.urlopen", return_value=response):
                verified = fetch_bundle("https://example.test/model.zip", destination, registry["archive"]["sha256"])
            self.assertEqual(verified["manifest"]["bundle_id"], "brent-frozen-v1")


if __name__ == "__main__":
    unittest.main()
