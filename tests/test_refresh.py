"""Refresh scheduling, durable fallback and concurrency without provider access."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_forecast.acquisition import sha256
from oil_forecast.refresh import RefreshController, _process_lock

CONFIG = {"start": "2025-01-01", "decision_hour": 8,
          "sources": {"brent": {"required": True, "enabled": True, "lag_business_days": 10}}}


def write_snapshot(root, identifier, updated, config=None, value=70.):
    folder = Path(root) / identifier
    folder.mkdir(parents=True)
    dates = pd.bdate_range(end=pd.Timestamp(updated).tz_convert("Europe/Madrid").date(), periods=60)
    decisions = (dates.tz_localize("Europe/Madrid") + pd.Timedelta(hours=8)).tz_convert("UTC")
    features = pd.DataFrame({"decision_date": dates, "decision_at": decisions, "brent": value})
    history = pd.DataFrame({"observation_date": dates, "y_brent": value, "observed": True})
    features.to_csv(folder / "features.csv", index=False)
    history.to_csv(folder / "target.csv", index=False)
    protocol = {"snapshot_id": identifier, "data_updated_at": updated.isoformat(),
                "config": copy.deepcopy(config or CONFIG), "weights_updated": False,
                "sources": {"brent": {"status": "ok"}},
                "output_sha256": {name: sha256(folder / name) for name in ("features.csv", "target.csv")}}
    (folder / "manifest.json").write_text(json.dumps(protocol), encoding="utf-8")
    return folder


def hold_lock(path, ready, release):
    """Top-level target is compatible with Windows multiprocessing spawn."""
    with _process_lock(Path(path)) as acquired:
        ready.put(acquired)
        release.wait(10)


class RefreshContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "live"
        self.now = datetime(2026, 11, 16, 18, tzinfo=timezone.utc)
        self.calls = 0

    def tearDown(self):
        self.temporary.cleanup()

    def updater(self, config, directory, snapshot_id):
        self.calls += 1
        write_snapshot(directory, snapshot_id, self.now, config)

    def controller(self, **options):
        return RefreshController(self.directory, updater=self.updater, clock=lambda: self.now, **options)

    def test_fresh_legacy_snapshot_is_reused_until_24h_then_updated_once(self):
        folder = write_snapshot(self.directory, "legacy", self.now - timedelta(hours=23))
        before = {path.name: sha256(path) for path in folder.iterdir()}
        original = copy.deepcopy(CONFIG)
        controller = self.controller()
        self.assertEqual(controller.check(CONFIG)["status"], "current")
        self.assertEqual(self.calls, 0)
        self.now += timedelta(hours=1)
        result = controller.check(CONFIG)
        self.assertEqual(result["status"], "updated")
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(result["data"]["window"]), 60)
        self.assertEqual(result["last_success_at"], self.now.isoformat())
        self.assertEqual(controller.check(CONFIG)["status"], "current")
        self.assertEqual(self.calls, 1)
        self.assertEqual(CONFIG, original)
        self.assertEqual(before, {path.name: sha256(path) for path in folder.iterdir()})

    def test_manual_force_skips_schedule_but_guards_double_click_and_restart(self):
        controller = self.controller()
        first = controller.check(CONFIG, force=True, enabled=False)
        self.assertEqual(first["status"], "updated")
        self.assertEqual(controller.check(CONFIG, force=True)["status"], "cooldown")
        self.assertEqual(self.calls, 1)
        self.now += timedelta(seconds=61)
        second = self.controller().check(CONFIG, force=True)
        self.assertEqual(second["status"], "updated")
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertEqual(self.calls, 2)
        offline = self.controller().check(CONFIG, enabled=False)
        self.assertEqual(offline["snapshot_id"], second["snapshot_id"])
        self.assertEqual(offline["status"], "disabled")
        self.assertEqual(self.calls, 2)

    def test_failure_keeps_last_good_and_backoff_survives_restart(self):
        write_snapshot(self.directory, "previous", self.now - timedelta(days=2))
        def unavailable(*args, **kwargs):
            self.calls += 1
            raise OSError("provider unavailable")
        controller = RefreshController(self.directory, updater=unavailable, clock=lambda: self.now)
        failed = controller.check(CONFIG)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["snapshot_id"], "previous")
        self.assertEqual(failed["data"]["history"]["y_brent"].iloc[-1], 70.)
        restarted = RefreshController(self.directory, updater=unavailable, clock=lambda: self.now)
        self.assertEqual(restarted.check(CONFIG)["status"], "cooldown")
        self.assertEqual(self.calls, 1)
        self.now += timedelta(seconds=61)
        self.assertEqual(restarted.check(CONFIG, force=True)["status"], "failed")
        self.assertEqual(self.calls, 2)
        self.now += timedelta(minutes=15)
        self.assertEqual(restarted.check(CONFIG)["status"], "failed")
        self.assertEqual(self.calls, 3)

    def test_bad_payload_policy_future_time_and_failed_inference_never_replace_old(self):
        write_snapshot(self.directory, "old", self.now - timedelta(days=2))
        for kind in ("payload", "policy", "future", "inference"):
            def invalid(config, directory, snapshot_id):
                folder = write_snapshot(directory, snapshot_id, self.now, config, value=999.)
                if kind == "payload":
                    (folder / "features.csv").write_text("tampered", encoding="utf-8")
                elif kind in {"policy", "future"}:
                    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                    if kind == "policy":
                        manifest["config"]["sources"]["brent"]["lag_business_days"] = 0
                    else:
                        manifest["data_updated_at"] = (self.now + timedelta(days=1)).isoformat()
                    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            def validate(window):
                if kind == "inference" and window["brent"].iloc[-1] == 999.:
                    raise RuntimeError("not usable by frozen networks")
            self.now += timedelta(seconds=61)
            controller = RefreshController(self.directory, updater=invalid, clock=lambda: self.now)
            result = controller.check(CONFIG, force=True, validator=validate)
            self.assertEqual(result["status"], "failed", kind)
            self.assertEqual(result["snapshot_id"], "old", kind)
            restarted = RefreshController(self.directory, updater=invalid, clock=lambda: self.now)
            self.assertEqual(restarted.check(CONFIG, enabled=False, validator=validate)["snapshot_id"], "old", kind)

    def test_offline_without_local_state_does_not_create_or_download(self):
        result = self.controller().check(CONFIG, enabled=False)
        self.assertEqual(result["status"], "disabled")
        self.assertIsNone(result["data"])
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.calls, 0)

    def test_optional_yahoo_failure_keeps_last_complete_snapshot_without_changing_downloader_policy(self):
        config = copy.deepcopy(CONFIG)
        config["sources"]["oro"] = {"provider": "yahoo", "enabled": True, "required": False, "lag_business_days": 2}
        config["sources"]["opep_orb"] = {"enabled": True, "required": False}
        old = write_snapshot(self.directory, "complete", self.now - timedelta(days=2), config)
        protocol = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
        protocol["sources"].update(oro={"status": "ok"}, opep_orb={"status": "unavailable"})
        (old / "manifest.json").write_text(json.dumps(protocol), encoding="utf-8")
        before = {path.name: sha256(path) for path in old.iterdir()}
        def partial(settings, directory, snapshot_id):
            folder = write_snapshot(directory, snapshot_id, self.now, settings)
            current = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            current["sources"].update(oro={"status": "unavailable"}, opep_orb={"status": "unavailable"})
            (folder / "manifest.json").write_text(json.dumps(current), encoding="utf-8")
        result = RefreshController(self.directory, updater=partial, clock=lambda: self.now).check(config)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["snapshot_id"], "complete")
        self.assertFalse(config["sources"]["oro"]["required"])
        self.assertEqual(before, {path.name: sha256(path) for path in old.iterdir()})

    def test_acquisition_time_bounds_decisions_and_publication_metadata(self):
        for kind in ("decision", "publication"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = write_snapshot(root, "invalid", self.now)
                protocol = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                if kind == "decision":
                    protocol["data_updated_at"] = (self.now - timedelta(days=1)).isoformat()
                else:
                    frame = pd.read_csv(folder / "features.csv")
                    frame["brent__available_at"] = (self.now + timedelta(minutes=1)).isoformat()
                    frame.to_csv(folder / "features.csv", index=False)
                    protocol["output_sha256"]["features.csv"] = sha256(folder / "features.csv")
                (folder / "manifest.json").write_text(json.dumps(protocol), encoding="utf-8")
                result = RefreshController(root, clock=lambda: self.now).check(CONFIG, enabled=False)
                self.assertIsNone(result["data"], kind)

    def test_corrupt_state_and_unsafe_pointer_fall_back_to_verified_legacy(self):
        write_snapshot(self.directory, "valid", self.now - timedelta(hours=1))
        for text in ("not json", json.dumps({"schema_version": 1, "snapshot_id": "../outside", "rejected_snapshots": None})):
            (self.directory / "refresh_state.json").write_text(text, encoding="utf-8")
            result = self.controller().check(CONFIG, enabled=False)
            self.assertEqual(result["snapshot_id"], "valid")
            self.assertEqual(self.calls, 0)

    def test_two_controllers_in_threads_only_download_once(self):
        started, release = threading.Event(), threading.Event()
        results = []
        def slow(config, directory, snapshot_id):
            self.calls += 1
            started.set()
            self.assertTrue(release.wait(5))
            write_snapshot(directory, snapshot_id, self.now, config)
        one = RefreshController(self.directory, updater=slow, clock=lambda: self.now)
        two = RefreshController(self.directory, updater=slow, clock=lambda: self.now)
        worker = threading.Thread(target=lambda: results.append(one.check(CONFIG)))
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            self.assertEqual(two.check(CONFIG)["status"], "busy")
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0]["status"], "updated")
        self.assertEqual(two.check(CONFIG)["status"], "current")
        self.assertEqual(self.calls, 1)

    def test_lock_is_shared_with_another_process_and_released_after_exit(self):
        self.directory.mkdir()
        context = multiprocessing.get_context("spawn")
        ready, release = context.Queue(), context.Event()
        worker = context.Process(target=hold_lock, args=(str(self.directory / ".refresh.lock"), ready, release))
        worker.start()
        try:
            self.assertTrue(ready.get(timeout=30))
            self.assertEqual(self.controller().check(CONFIG)["status"], "busy")
            self.assertEqual(self.calls, 0)
        finally:
            release.set()
            worker.join(15)
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
            ready.close()
            ready.join_thread()
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(self.controller().check(CONFIG)["status"], "updated")
        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
