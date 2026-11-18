"""Portable bundles preserve selected weights and reject untrusted archives."""

from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oil_forecast.acquisition import sha256
from oil_forecast.artifacts import build_bundle, install_bundle, verify_bundle


@unittest.skipUnless((ROOT / "experiments/cpu-final-evaluation-v1/experiment.json").exists(), "Requires frozen local final results")
class PortableArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.source = ROOT / "experiments/cpu-final-evaluation-v1"
        cls.processed = ROOT / "data/processed/2026-11-11-official"
        cls.output = cls.folder / "frozen"
        cls.archive = cls.folder / "bundle.zip"
        cls.before = {p.name: sha256(p) for p in cls.source.iterdir() if p.is_file()}
        cls.built = build_bundle(cls.source, cls.processed, cls.output, cls.archive)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_relocated_bundle_loads_without_original_paths_and_weights_are_exact(self):
        relocated = self.folder / "another-machine"
        installed = install_bundle(self.archive, relocated, self.built["archive_sha256"])
        self.assertEqual(installed["forecast"], verify_bundle(self.output)["forecast"])
        source = torch.load(self.source / "checkpoints.pt", map_location="cpu", weights_only=True)
        portable = torch.load(relocated / "checkpoints.pt", map_location="cpu", weights_only=True)
        self.assertEqual(len(portable["models"]), 3)
        for key, entry in portable["models"].items():
            self.assertEqual(entry["transformers"], source["models"][key]["transformers"])
            for name, value in entry["state_dict"].items():
                self.assertTrue(torch.equal(value, source["models"][key]["state_dict"][name]))
        self.assertEqual(self.before, {p.name: sha256(p) for p in self.source.iterdir() if p.is_file()})
        # An isolated process receives only source code and installed artifacts,
        # without the original repository's data/ or experiments/ directories.
        clone = self.folder / "standalone-copy"
        shutil.copytree(ROOT / "src", clone / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (clone / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/manage_artifacts.py", clone / "scripts/manage_artifacts.py")
        shutil.copyfile(self.archive, clone / "bundle.zip")
        (clone / "registry.json").write_text(json.dumps({"schema_version": 1, "bundle_id": "brent-frozen-v1",
            "archive": {"sha256": self.built["archive_sha256"]}}), encoding="utf-8")
        command = [sys.executable, "-I", "-X", "utf8", "-B", "-W", "error", "scripts/manage_artifacts.py", "install",
                   "--archive", "bundle.zip", "--registry", "registry.json", "--destination", "artifacts/frozen"]
        installed_cli = subprocess.run(command, cwd=clone, capture_output=True, text=True,
                                       encoding="utf-8", check=True)
        self.assertEqual(json.loads(installed_cli.stdout)["forecast"], installed["forecast"])
        program = """
import json,sys
from pathlib import Path
sys.path.insert(0,'src')
from oil_forecast.artifacts import verify_bundle
import oil_forecast.training as training
import oil_forecast.uncertainty as uncertainty
import oil_forecast.acquisition as acquisition
import urllib.request
def forbidden(*args,**kwargs):
    raise AssertionError('Portable inference must not train, calibrate or access the network')
training.train_short=training.train_with_early_stopping=training.train_fixed_epochs=forbidden
uncertainty.fit_intervals=acquisition.download_snapshot=acquisition.urlopen=urllib.request.urlopen=forbidden
assert not Path('data').exists() and not Path('experiments').exists()
print(json.dumps(verify_bundle(Path('artifacts/frozen'))['forecast']))
"""
        process = subprocess.run([sys.executable, "-I", "-X", "utf8", "-B", "-W", "error", "-c", program],
                                 cwd=clone, capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertEqual(json.loads(process.stdout), installed["forecast"])

    def test_second_build_is_identical_and_existing_destinations_are_preserved(self):
        second = build_bundle(self.source, self.processed, self.folder / "second", self.folder / "second.zip")
        self.assertEqual(self.built["archive_sha256"], second["archive_sha256"])
        with self.assertRaises(FileExistsError):
            build_bundle(self.source, self.processed, self.output, self.folder / "other.zip")
        with self.assertRaises(FileExistsError):
            install_bundle(self.archive, self.output, self.built["archive_sha256"])

    def test_changed_digest_or_payload_fails_without_publishing_destination(self):
        destination = self.folder / "rejected"
        with self.assertRaises(ValueError):
            install_bundle(self.archive, destination, "0" * 64)
        self.assertFalse(destination.exists())
        corrupted = self.folder / "corrupted.zip"
        with zipfile.ZipFile(self.archive) as original, zipfile.ZipFile(corrupted, "w") as changed:
            for member in original.infolist():
                changed.writestr(member, b"tampered" if member.filename == "metrics.csv" else original.read(member))
        with self.assertRaises(ValueError):
            install_bundle(corrupted, destination, sha256(corrupted))
        self.assertFalse(destination.exists())

    def test_traversal_absolute_duplicate_and_symlink_members_are_rejected(self):
        for index, name in enumerate(["../outside", "/absolute", "C:\\outside", "nested/file"]):
            archive = self.folder / f"unsafe-{index}.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(name, b"not allowed")
            with self.assertRaises(ValueError):
                install_bundle(archive, self.folder / f"unsafe-destination-{index}", sha256(archive))
        link = self.folder / "link.zip"
        with zipfile.ZipFile(link, "w") as stream:
            member = zipfile.ZipInfo("link")
            member.external_attr = 0o120777 << 16
            stream.writestr(member, b"target")
        with self.assertRaises(ValueError):
            install_bundle(link, self.folder / "link-destination", sha256(link))
        duplicate = self.folder / "duplicate.zip"
        with zipfile.ZipFile(duplicate, "w") as stream:
            stream.writestr("File.csv", b"first")
            stream.writestr("file.csv", b"second")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            install_bundle(duplicate, self.folder / "duplicate-destination", sha256(duplicate))


if __name__ == "__main__":
    unittest.main()
