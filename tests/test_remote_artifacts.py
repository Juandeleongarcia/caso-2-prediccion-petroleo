"""Authenticated HF artifact downloads preserve frozen files and secrets."""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.artifacts import _HttpsArtifactRedirects, fetch_bundle


class AuthenticatedArtifactDownloads(unittest.TestCase):
    token = "hf_" + "x" * 36
    revision = "a" * 40
    url = f"https://huggingface.co/student/brent-models/resolve/{revision}/brent-frozen-v1.zip"

    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT / "configs/serving_bundle.json").read_text(encoding="utf-8"))
        cls.payload = (ROOT / cls.registry["archive"]["local_path"]).read_bytes()
        cls.digest = cls.registry["archive"]["sha256"]

    def response(self, payload=None, *, final_url="https://cdn.example.test/signed-bundle.zip"):
        stream = io.BytesIO(self.payload if payload is None else payload)
        stream.geturl = lambda: final_url
        return stream

    def test_private_download_loads_original_frozen_cpu_bundle_and_scopes_token(self):
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.build_opener") as factory:
            factory.return_value.open.return_value = self.response()
            destination = Path(temporary) / "model"
            result = fetch_bundle(self.url, destination, self.digest, token=self.token)
            self.assertEqual(result["manifest"]["bundle_id"], "brent-frozen-v1")
            self.assertEqual(result["manifest"]["selected_seed"], 42)
            self.assertEqual(result["manifest"]["horizons"], [5, 63, 252])
            request = factory.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), f"Bearer {self.token}")
            self.assertNotIn("Authorization", request.headers)
            self.assertIsInstance(factory.call_args.args[0], _HttpsArtifactRedirects)

    def test_untrusted_hosts_moving_revisions_and_secret_urls_are_rejected_before_io(self):
        bad_urls = [
            "https://example.test/other.zip",
            self.url.replace("huggingface.co", "huggingface.co.example.test"),
            self.url.replace("huggingface.co", "cdn.huggingface.co"),
            self.url.replace("huggingface.co", "huggingface.co:8443"),
            self.url.replace(self.revision, "main"),
            self.url.replace(self.revision, "v1"),
            self.url + "?token=" + self.token,
            self.url + "#" + self.token,
            self.url.replace("huggingface.co", "user:password@huggingface.co"),
            self.url.replace("https://", "http://"),
            self.url.replace("brent-frozen-v1.zip", "nested/file.zip"),
        ]
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.build_opener") as factory, \
                patch("oil_forecast.artifacts.urlopen") as public:
            destination = Path(temporary) / "rejected"
            for url in bad_urls:
                with self.subTest(url=url), self.assertRaises(ValueError) as caught:
                    fetch_bundle(url, destination, self.digest, token=self.token)
                self.assertNotIn(self.token, str(caught.exception))
            factory.assert_not_called()
            public.assert_not_called()
            self.assertFalse(destination.exists())

    def test_all_hf_downloads_require_immutable_revision_even_without_authentication(self):
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.urlopen") as public:
            for suffix in ["main", "refs/pr/1", "a" * 39, "a" * 41]:
                with self.subTest(revision=suffix), self.assertRaises(ValueError):
                    fetch_bundle(self.url.replace(self.revision, suffix), Path(temporary) / "bad", self.digest)
            public.assert_not_called()

    def test_blank_whitespace_and_header_injection_tokens_are_rejected_before_io(self):
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.build_opener") as factory:
            for token in ["", " ", "bad token", "bad\r\nInjected: header", "bad\ttoken", "bad\0token", "é"]:
                with self.subTest(token=repr(token)), self.assertRaises(ValueError):
                    fetch_bundle(self.url, Path(temporary) / "bad", self.digest, token=token)
            factory.assert_not_called()

    def test_redirects_strip_authorization_and_reject_https_downgrades_before_following(self):
        request = Request(self.url, headers={"Authorization": f"Bearer {self.token}", "User-Agent": "BrentForecast"})
        request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        handler = _HttpsArtifactRedirects()
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example.test/signed.zip")
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(redirected.get_header("User-agent"), "BrentForecast")
        hub = handler.redirect_request(request, None, 302, "Found", {},
                                       "https://huggingface.co:443/api/resolve-cache/models/student/brent-models/file.zip")
        self.assertEqual(hub.get_header("Authorization"), f"Bearer {self.token}")
        self.assertNotIn("Authorization", hub.headers)
        for external in ["https://cdn.example.test/signed.zip", "https://huggingface.co:8443/file.zip",
                         "https://huggingface.co:0/file.zip", "https://huggingface.co.example.test/file.zip",
                         "https://cdn.huggingface.co/file.zip"]:
            with self.subTest(destination=external):
                detached = handler.redirect_request(hub, None, 302, "Found", {}, external)
                self.assertIsNone(detached.get_header("Authorization"))
        returned = handler.redirect_request(redirected, None, 302, "Found", {}, self.url)
        self.assertIsNone(returned.get_header("Authorization"))
        for bad in ["http://cdn.example.test/signed.zip", "ftp://cdn.example.test/signed.zip",
                    "https://user:password@cdn.example.test/signed.zip"]:
            with self.subTest(destination=bad), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "Found", {}, bad)

    def test_download_errors_do_not_expose_token_or_signed_url_or_publish_destination(self):
        signed = "https://cdn.example.test/archive.zip?secret=" + self.token
        failures = [OSError(signed), HTTPError(signed, 401, self.token, {}, None)]
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.build_opener") as factory:
            for index, failure in enumerate(failures):
                destination = Path(temporary) / f"failed-{index}"
                factory.return_value.open.side_effect = failure
                with self.subTest(error=type(failure).__name__), self.assertRaises(ValueError) as caught:
                    fetch_bundle(self.url, destination, self.digest, token=self.token)
                self.assertNotIn(self.token, str(caught.exception))
                self.assertNotIn(signed, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_corruption_and_size_limit_fail_without_installing(self):
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.build_opener") as factory:
            factory.return_value.open.return_value = self.response(b"not the original archive")
            destination = Path(temporary) / "corrupt"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fetch_bundle(self.url, destination, self.digest, token=self.token)
            self.assertFalse(destination.exists())
            factory.return_value.open.return_value = self.response()
            with patch("oil_forecast.artifacts.LIMIT", 100), self.assertRaisesRegex(ValueError, "size limit"):
                fetch_bundle(self.url, destination, self.digest, token=self.token)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_existing_valid_resources_are_never_overwritten_or_downloaded_again(self):
        with tempfile.TemporaryDirectory() as temporary, patch("oil_forecast.artifacts.build_opener") as factory:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_bytes(b"original resource")
            with self.assertRaises(FileExistsError):
                fetch_bundle(self.url, destination, self.digest, token=self.token)
            factory.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"original resource")


class PrivateArtifactCLI(unittest.TestCase):
    def arguments(self):
        return ["manage_artifacts.py", "fetch", "--url",
                "https://huggingface.co/student/brent-models/resolve/" + "a" * 40 + "/brent-frozen-v1.zip",
                "--registry", str(ROOT / "configs/serving_bundle.json"), "--destination", "unused-test-destination"]

    def test_cli_reads_token_only_when_requested_without_exposing_it_in_output(self):
        from scripts.manage_artifacts import main
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_only"}), \
                patch("scripts.manage_artifacts.fetch_bundle", return_value={"verified": True}) as fetch, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            with patch.object(sys, "argv", self.arguments()):
                main()
            self.assertIsNone(fetch.call_args.kwargs["token"])
            with patch.object(sys, "argv", self.arguments() + ["--use-hf-token"]):
                main()
            self.assertEqual(fetch.call_args.kwargs["token"], "hf_test_only")
            self.assertNotIn("hf_test_only", output.getvalue())

    def test_private_cli_requires_environment_secret_before_accessing_network(self):
        from scripts.manage_artifacts import main
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", self.arguments() + ["--use-hf-token"]), \
                patch("scripts.manage_artifacts.fetch_bundle") as fetch:
            with self.assertRaisesRegex(ValueError, "Configure HF_TOKEN"):
                main()
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
