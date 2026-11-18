"""Provision real frozen archives offline, including private and legacy sources."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Barrier
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.artifacts import install_bundle, verify_bundle
from oil_forecast.deployment import hf_artifact_url, prepare_application_resources


def hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


class ImmutableModelsURLs(unittest.TestCase):
    def test_urls_require_safe_models_identifiers_and_full_commit(self):
        revision = "a" * 40
        self.assertEqual(
            hf_artifact_url("student/brent-models", revision, "brent-frozen-v1.zip"),
            f"https://huggingface.co/student/brent-models/resolve/{revision}/brent-frozen-v1.zip",
        )
        invalid = [
            ("../repo", revision, "brent-frozen-v1.zip"),
            ("student/repo/extra", revision, "brent-frozen-v1.zip"),
            ("student/repo?token=example", revision, "brent-frozen-v1.zip"),
            ("student\\repo", revision, "brent-frozen-v1.zip"),
            (None, revision, "brent-frozen-v1.zip"),
            ([], revision, "brent-frozen-v1.zip"),
            ("student/repo", "main", "brent-frozen-v1.zip"),
            ("student/repo", "a" * 39, "brent-frozen-v1.zip"),
            ("student/repo", "A" * 40, "brent-frozen-v1.zip"),
            ("student/repo", None, "brent-frozen-v1.zip"),
            ("student/repo", revision, "../brent-frozen-v1.zip"),
            ("student/repo", revision, "brent-frozen-v1.zip?token=example"),
            ("student/repo", revision, "another-model.zip"),
            ("student/repo", revision, None),
            ("student/repo", revision, []),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                hf_artifact_url(*arguments)


@unittest.skipUnless(
    all((ROOT / "artifacts/releases" / name).is_file()
        for name in ("brent-frozen-v1.zip", "brent-dashboard-v1.zip")),
    "Requires the two authorized local frozen archives",
)
class ApplicationProvisioning(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        self.registries = {}
        for filename in ("serving_bundle.json", "dashboard_bundle.json"):
            source = ROOT / "configs" / filename
            shutil.copy2(source, self.root / "configs" / filename)
            registry = json.loads(source.read_text(encoding="utf-8"))
            self.registries[registry["bundle_id"]] = registry
        self.bundle = self.root / "runtime/inference"
        self.evidence = self.root / "runtime/evidence"
        self.settings = {
            "BRENT_HF_REPO_ID": "student/brent-models",
            "BRENT_HF_REVISION": "a" * 40,
            "HF_TOKEN": "unit-test-placeholder",
        }
        self.source_hashes = {
            bundle_id: hashlib.sha256((ROOT / registry["archive"]["local_path"]).read_bytes()).hexdigest()
            for bundle_id, registry in self.registries.items()
        }
        self.addCleanup(self.check_sources_unchanged)

    def check_sources_unchanged(self):
        for bundle_id, registry in self.registries.items():
            source = ROOT / registry["archive"]["local_path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), self.source_hashes[bundle_id])

    def install_saved_archive(self, url, destination, expected_sha256, *, token=None):
        bundle_id = "brent-frozen-v1" if Path(destination) == self.bundle else "brent-dashboard-v1"
        registry = self.registries[bundle_id]
        self.assertEqual(expected_sha256, registry["archive"]["sha256"])
        return install_bundle(ROOT / registry["archive"]["local_path"], destination, expected_sha256)

    def test_local_archives_install_then_verify_without_overwrites_or_network(self):
        for registry in self.registries.values():
            source = ROOT / registry["archive"]["local_path"]
            copied = self.root / registry["archive"]["local_path"]
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copied)
        with patch("oil_forecast.deployment.fetch_bundle", side_effect=AssertionError("Unexpected network")):
            first = prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
            before = hashes(self.root)
            repeated = prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
        self.assertEqual(first, {name: "installed_local" for name in self.registries})
        self.assertEqual(repeated, {name: "verified" for name in self.registries})
        self.assertEqual(before, hashes(self.root))
        self.assertEqual(verify_bundle(self.bundle)["manifest"]["bundle_id"], "brent-frozen-v1")
        self.assertEqual(verify_bundle(self.evidence)["manifest"]["bundle_id"], "brent-dashboard-v1")

    def test_private_remote_uses_pinned_models_urls_without_returning_secrets(self):
        with patch("oil_forecast.deployment.fetch_bundle", side_effect=self.install_saved_archive) as fetch:
            result = prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
        self.assertEqual(result, {name: "installed_remote" for name in self.registries})
        self.assertEqual(fetch.call_count, 2)
        for call, registry in zip(fetch.call_args_list, self.registries.values()):
            expected = hf_artifact_url(self.settings["BRENT_HF_REPO_ID"], self.settings["BRENT_HF_REVISION"],
                                       registry["archive"]["filename"])
            self.assertEqual(call.args[0], expected)
            self.assertEqual(call.kwargs["token"], self.settings["HF_TOKEN"])
            self.assertNotIn(self.settings["HF_TOKEN"], call.args[0])
        self.assertNotIn(self.settings["HF_TOKEN"], repr(result))
        self.assertEqual(set(result), set(self.registries))
        self.assertNotIn("HF_TOKEN", repr(result))

    def test_legacy_https_sources_never_receive_the_private_hf_token(self):
        settings = {
            "BRENT_INFERENCE_URL": "https://downloads.example.test/inference.zip",
            "BRENT_EVIDENCE_URL": "https://downloads.example.test/evidence.zip",
            "HF_TOKEN": self.settings["HF_TOKEN"],
        }
        with patch("oil_forecast.deployment.fetch_bundle", side_effect=self.install_saved_archive) as fetch:
            result = prepare_application_resources(self.root, self.bundle, self.evidence, settings)
        self.assertEqual(result, {name: "installed_remote" for name in self.registries})
        self.assertEqual([call.args[0] for call in fetch.call_args_list],
                         [settings["BRENT_INFERENCE_URL"], settings["BRENT_EVIDENCE_URL"]])
        self.assertTrue(all(call.kwargs["token"] is None for call in fetch.call_args_list))

    def test_first_network_failure_publishes_neither_resource(self):
        with patch("oil_forecast.deployment.fetch_bundle", side_effect=OSError("Provider unavailable")) as fetch:
            with self.assertRaises(OSError):
                prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(self.bundle.exists())
        self.assertFalse(self.evidence.exists())
        self.assertFalse((self.root / "artifacts/releases").exists())

    def test_evidence_failure_preserves_verified_inference_on_retry(self):
        def provider(url, destination, expected_sha256, *, token=None):
            if Path(destination) == self.evidence:
                raise OSError("Evidence provider unavailable")
            return self.install_saved_archive(url, destination, expected_sha256, token=token)

        with patch("oil_forecast.deployment.fetch_bundle", side_effect=provider) as fetch:
            with self.assertRaises(OSError):
                prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
            self.assertTrue(self.bundle.is_dir())
            self.assertFalse(self.evidence.exists())
            before = hashes(self.bundle)
            forecast = verify_bundle(self.bundle)["forecast"]
            with self.assertRaises(OSError):
                prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
            self.assertEqual(fetch.call_count, 3)
        self.assertEqual(before, hashes(self.bundle))
        self.assertEqual(forecast, verify_bundle(self.bundle)["forecast"])
        self.assertFalse(self.evidence.exists())

    def test_installed_wrong_bundle_is_rejected_without_replacing_it(self):
        evidence = self.registries["brent-dashboard-v1"]
        install_bundle(ROOT / evidence["archive"]["local_path"], self.bundle, evidence["archive"]["sha256"])
        before = hashes(self.bundle)
        with patch("oil_forecast.deployment.fetch_bundle", side_effect=AssertionError("Unexpected network")):
            with self.assertRaisesRegex(ValueError, "trusted registry"):
                prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
        self.assertEqual(before, hashes(self.bundle))
        self.assertFalse(self.evidence.exists())

    def test_changed_manifest_metadata_is_rejected_before_loading_weights(self):
        registry = self.registries["brent-frozen-v1"]
        install_bundle(ROOT / registry["archive"]["local_path"], self.bundle, registry["archive"]["sha256"])
        original = hashes(self.bundle)
        manifest_path = self.bundle / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["logical_refit_at"] = "2099-01-01T00:00:00+00:00"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        altered = hashes(self.bundle)
        self.assertEqual({p: h for p, h in original.items() if p != "artifact_manifest.json"},
                         {p: h for p, h in altered.items() if p != "artifact_manifest.json"})
        with patch("oil_forecast.deployment.verify_bundle") as verify, \
                patch("oil_forecast.deployment.fetch_bundle") as fetch:
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)
        verify.assert_not_called()
        fetch.assert_not_called()
        self.assertEqual(altered, hashes(self.bundle))
        self.assertFalse(self.evidence.exists())

    def test_two_sessions_fetch_each_archive_only_once(self):
        start = Barrier(2)

        def provider(url, destination, expected_sha256, *, token=None):
            time.sleep(0.02)
            return self.install_saved_archive(url, destination, expected_sha256, token=token)

        def session():
            start.wait(timeout=10)
            return prepare_application_resources(self.root, self.bundle, self.evidence, self.settings)

        with patch("oil_forecast.deployment.fetch_bundle", side_effect=provider) as fetch:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(session) for _ in range(2)]
                results = [future.result(timeout=30) for future in futures]
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(Counter(Path(call.args[1]) for call in fetch.call_args_list),
                         Counter({self.bundle: 1, self.evidence: 1}))
        self.assertCountEqual(results, [
            {name: "installed_remote" for name in self.registries},
            {name: "verified" for name in self.registries},
        ])
        self.assertEqual(verify_bundle(self.bundle)["manifest"]["bundle_id"], "brent-frozen-v1")
        self.assertEqual(verify_bundle(self.evidence)["manifest"]["bundle_id"], "brent-dashboard-v1")


if __name__ == "__main__":
    unittest.main()
