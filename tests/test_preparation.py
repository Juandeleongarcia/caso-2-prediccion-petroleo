"""Source integrity and snapshot contracts; no network calls."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_forecast.acquisition import download_snapshot, load_snapshot, parse_fred


class SourceContracts(unittest.TestCase):
    def test_fred_missing_marker_is_not_an_observed_label(self):
        raw = parse_fred(b"observation_date,DCOILBRENTEU\n2024-01-02,80\n2024-01-03,.\n2024-01-04,80\n", "brent", "DCOILBRENTEU")
        self.assertEqual(raw["is_observed"].tolist(), [True, False, True])
        self.assertEqual(raw.loc[0, "value"], raw.loc[2, "value"])
        self.assertTrue(raw.loc[1:1, "value"].isna().all())

    def test_fred_schema_and_duplicate_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_fred(b"DATE,OTHER\n2024-01-02,80\n", "brent", "DCOILBRENTEU")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_fred(b"DATE,DCOILBRENTEU\n2024-01-02,80\n2024-01-02,81\n", "brent", "DCOILBRENTEU")

    def test_negative_original_value_survives_source_parser(self):
        raw = parse_fred(b"DATE,DCOILWTICO\n2020-04-20,-36.98\n", "wti", "DCOILWTICO")
        self.assertEqual(raw.loc[0, "value"], -36.98)
        self.assertTrue(raw.loc[0, "is_observed"])

    def test_tampered_snapshot_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = b"observation_date,value,variable,source\n2024-01-02,80,brent,FRED:DCOILBRENTEU\n"
            (folder / "observations.csv").write_bytes(raw)
            (folder / "manifest.json").write_text(json.dumps({"observations_sha256": hashlib.sha256(raw).hexdigest()}), encoding="utf-8")
            frame, _ = load_snapshot(folder)
            self.assertEqual(len(frame), 1)
            (folder / "observations.csv").write_bytes(raw.replace(b",80,", b",81,"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_snapshot(folder)

    def test_existing_snapshot_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                download_snapshot({}, Path(temp), Path(temp) / "cache")

    def test_required_source_failure_does_not_publish_a_dataset(self):
        spec = {"provider": "fred", "id": "DCOILBRENTEU", "required": True, "enabled": True}
        config = {"start": "2000-01-01", "end": "2026-11-11", "sources": {"brent": spec}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "snapshot"
            with patch("oil_forecast.acquisition._download_one", side_effect=OSError("offline")):
                with self.assertRaisesRegex(RuntimeError, "Required sources"):
                    download_snapshot(config, folder, Path(temp) / "cache")
            self.assertFalse((folder / "observations.csv").exists())
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["required_sources_unavailable"], ["brent"])


if __name__ == "__main__":
    unittest.main()
