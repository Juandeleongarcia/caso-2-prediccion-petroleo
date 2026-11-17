"""Controlled market refreshes with an immutable last-good snapshot.

Only the explicit updater accesses providers. The coordinator never trains,
fits transformers or changes model artifacts. Its small state and OS lock may
live on ephemeral storage: losing them falls back to the packaged snapshot.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import errno
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable
from uuid import uuid4

import pandas as pd

from . import app_data
from .acquisition import sha256

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
POLICY_KEYS = ("start", "calendar", "decision_timezone", "decision_hour", "lookback_business_days",
               "horizons_business_days", "availability_mode", "sources")


def _utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Refresh timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _optional_time(value: object) -> datetime | None:
    try:
        return _utc(value) if value else None
    except (TypeError, ValueError):
        return None


@contextmanager
def _process_lock(path: Path):
    """Nonblocking OS lock, automatically released after a crash.

    Keep this file permanently: unlinking it would permit two processes to
    lock different inodes. Windows and Linux use their native stdlib APIs.
    """
    if path.is_symlink():
        raise ValueError("A refresh lock cannot be a symbolic link")
    with path.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as failure:
            if failure.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RefreshController:
    """Coordinate startup, periodic and manual checks across sessions/processes.

    ``check`` returns data or None (the UI then uses its frozen bundle), a
    status, UTC timestamps and the active snapshot identifier. ``force`` skips
    the 24 h schedule and failure backoff, subject to a 60 s double-click guard.
    ``enabled=False`` disables automatic provider access, not offline reading.
    The optional updater has the existing refresh_market_data signature.
    """

    def __init__(self, directory: Path | str, interval_hours: float = 24,
                 retry_minutes: float = 15, updater: Callable | None = None,
                 clock: Callable[[], datetime] | None = None):
        if (not math.isfinite(float(interval_hours)) or interval_hours <= 0
                or not math.isfinite(float(retry_minutes)) or retry_minutes <= 0):
            raise ValueError("Refresh and retry intervals must be positive and finite")
        self.directory = Path(directory).absolute()
        if self.directory.is_symlink():
            raise ValueError("The refresh directory cannot be a symbolic link")
        self.directory = self.directory.resolve()
        self.interval = timedelta(hours=float(interval_hours))
        self.retry = timedelta(minutes=float(retry_minutes))
        self.updater = updater
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.state_path = self.directory / "refresh_state.json"

    def _state(self) -> dict:
        try:
            if self.state_path.is_symlink() or self.state_path.stat().st_size > 1024 * 1024:
                return {"schema_version": 1}
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("schema_version") != 1:
                return {"schema_version": 1}
            rejected = state.get("rejected_snapshots", [])
            return {"schema_version": 1,
                    "snapshot_id": state.get("snapshot_id") if isinstance(state.get("snapshot_id"), str) else None,
                    "manifest_sha256": state.get("manifest_sha256"),
                    "last_attempt_at": state.get("last_attempt_at"),
                    "last_error": state.get("last_error") if isinstance(state.get("last_error"), str) else None,
                    "rejected_snapshots": [value for value in rejected if isinstance(value, str)][-100:] if isinstance(rejected, list) else []}
        except (OSError, ValueError, TypeError):
            return {"schema_version": 1}

    def _write_state(self, state: dict) -> None:
        if self.state_path.is_symlink():
            raise ValueError("A refresh pointer cannot be a symbolic link")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=".refresh-state-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, self.state_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(.01)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _load(self, identifier: str, config: dict, now: datetime, validator=None,
              expected_manifest: str | None = None) -> dict:
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise ValueError("Invalid refresh snapshot identifier")
        folder = self.directory / identifier
        if folder.is_symlink() or folder.resolve().parent != self.directory:
            raise ValueError("A snapshot must stay inside its refresh directory")
        manifest_path = folder / "manifest.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError("Invalid snapshot manifest")
        if expected_manifest is not None and sha256(manifest_path) != expected_manifest:
            raise ValueError("The last-good snapshot manifest changed")
        protocol = json.loads(manifest_path.read_text(encoding="utf-8"))
        updated = _utc(protocol["data_updated_at"])
        if updated > now or protocol["snapshot_id"] != identifier or protocol.get("weights_updated") is not False:
            raise ValueError("Incoherent snapshot provenance")
        saved_config = protocol.get("config", {})
        if {key: saved_config.get(key) for key in POLICY_KEYS} != {key: config.get(key) for key in POLICY_KEYS}:
            raise ValueError("The snapshot does not preserve the configured temporal policy")
        for name, specification in config.get("sources", {}).items():
            # The study's downloader permits missing Yahoo inputs. Serving is
            # more conservative: keep the previous complete snapshot whenever
            # any frozen input provider fails. ORB is not a model input.
            if specification.get("enabled", True) and name != "opep_orb":
                if protocol.get("sources", {}).get(name, {}).get("status") != "ok":
                    raise ValueError("A frozen input source is unavailable")
        for name in ("features.csv", "target.csv"):
            path = folder / name
            digest = protocol.get("output_sha256", {}).get(name)
            if (path.is_symlink() or not path.is_file() or path.stat().st_size > 100 * 1024 * 1024
                    or not isinstance(digest, str) or sha256(path) != digest):
                raise ValueError("Snapshot payload integrity failure")
        features = pd.read_csv(folder / "features.csv")
        history = pd.read_csv(folder / "target.csv")
        if len(features) < 60 or not {"decision_date", "decision_at"}.issubset(features):
            raise ValueError("A snapshot needs at least 60 dated decision rows")
        dates = pd.to_datetime(features["decision_date"], errors="raise")
        decisions = pd.to_datetime(features["decision_at"], utc=True, errors="raise")
        if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
                or (dates.dt.day_of_week >= 5).any() or decisions.isna().any()
                or decisions.gt(pd.Timestamp(now)).any() or decisions.gt(pd.Timestamp(updated)).any()):
            raise ValueError("A snapshot contains invalid or future decision dates")
        for column in (name for name in features if name.endswith("__available_at")):
            available = pd.to_datetime(features[column], utc=True, errors="raise")
            if (available.gt(decisions).any() or available.gt(pd.Timestamp(updated)).any()):
                raise ValueError("Snapshot availability metadata exceeds its decision or acquisition time")
        if history.empty or not {"observation_date", "y_brent"}.issubset(history):
            raise ValueError("A snapshot needs real Brent observations")
        observed = pd.to_datetime(history["observation_date"], errors="raise")
        prices = pd.to_numeric(history["y_brent"], errors="raise")
        today = pd.Timestamp(now).tz_convert("Europe/Madrid").normalize().tz_localize(None)
        if (observed.isna().any() or observed.duplicated().any() or not observed.is_monotonic_increasing
                or observed.gt(today).any() or prices.notna().sum() == 0
                or not prices.dropna().map(math.isfinite).all()):
            raise ValueError("A snapshot contains invalid or future Brent observations")
        window = features.tail(60).copy()
        if validator is not None:
            validator(window)
        return {"window": window, "history": history, "protocol": protocol, "directory": str(folder)}

    def _last_good(self, config, state, now, validator) -> dict | None:
        rejected = set(state.get("rejected_snapshots", []))
        identifier = state.get("snapshot_id")
        if identifier:
            try:
                return self._load(identifier, config, now, validator, state.get("manifest_sha256"))
            except Exception:
                rejected.add(identifier)
        if not self.directory.exists():
            return None
        candidates = []
        for folder in self.directory.iterdir():
            if folder.is_symlink() or not folder.is_dir() or folder.name in rejected or not IDENTIFIER.fullmatch(folder.name):
                continue
            try:
                path = folder / "manifest.json"
                if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                    continue
                updated = _utc(json.loads(path.read_text(encoding="utf-8"))["data_updated_at"])
                candidates.append((updated, folder.name))
            except Exception:
                continue
        for _, name in sorted(candidates, reverse=True):
            try:
                return self._load(name, config, now, validator)
            except Exception:
                continue
        return None

    def _result(self, status, data, state, now, message) -> dict:
        success = _optional_time(data["protocol"]["data_updated_at"]) if data else None
        attempt = _optional_time(state.get("last_attempt_at"))
        retry_at = attempt + self.retry if attempt and state.get("last_error") else None
        due = success + self.interval if success else now
        if retry_at is not None and (not success or attempt >= success):
            due = max(due, retry_at)
        return {"status": status, "data": data, "snapshot_id": data["protocol"]["snapshot_id"] if data else None,
                "last_success_at": success.isoformat() if success else None,
                "last_attempt_at": attempt.isoformat() if attempt else None,
                "next_due_at": due.isoformat(), "retry_at": retry_at.isoformat() if retry_at else None,
                "message": message, "failure": state.get("last_error")}

    def check(self, config: dict, *, validator=None, force: bool = False, enabled: bool = True) -> dict:
        now = _utc(self.clock().isoformat())
        state = self._state()
        try:
            data = self._last_good(config, state, now, validator)
        except OSError:
            data = None
        if not enabled and not force:
            return self._result("disabled", data, state, now, "Actualización automática desactivada; se utiliza la última copia válida.")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with _process_lock(self.directory / ".refresh.lock") as acquired:
                if not acquired:
                    return self._result("busy", data, state, now, "Otra sesión está actualizando; se conserva la última copia válida.")
                state = self._state()
                data = self._last_good(config, state, now, validator)
                if data is not None:
                    state.update(snapshot_id=data["protocol"]["snapshot_id"],
                                 manifest_sha256=sha256(Path(data["directory"]) / "manifest.json"))
                success = _optional_time(data["protocol"]["data_updated_at"]) if data else None
                attempt = _optional_time(state.get("last_attempt_at"))
                if attempt and attempt > now:
                    attempt = None
                if force and attempt and now < attempt + timedelta(seconds=60):
                    return self._result("cooldown", data, state, now, "Ya se ha consultado recientemente; espera un minuto antes de volver a intentarlo.")
                if not force and success and now < success + self.interval:
                    self._write_state(state)
                    return self._result("current", data, state, now, "Los datos están dentro de la frecuencia configurada.")
                if not force and attempt and now < attempt + self.retry and (not success or attempt >= success):
                    return self._result("cooldown", data, state, now, "Se mantiene la última copia válida; el próximo intento respeta el intervalo de reintento.")
                state.update(last_attempt_at=now.isoformat())
                self._write_state(state)
                identifier = f"live-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
                try:
                    updater = self.updater or app_data.refresh_market_data
                    updater(config, self.directory, snapshot_id=identifier)
                    fresh = self._load(identifier, config, _utc(self.clock().isoformat()), validator)
                    if success and _utc(fresh["protocol"]["data_updated_at"]) < success:
                        raise ValueError("A refresh cannot replace a newer successful snapshot")
                except Exception as failure:
                    rejected = list(dict.fromkeys([*state.get("rejected_snapshots", []), identifier]))
                    state.update(last_error=type(failure).__name__, rejected_snapshots=rejected[-100:])
                    self._write_state(state)
                    return self._result("failed", data, state, now, "No se pudo actualizar; se mantiene la última copia válida.")
                state.update(snapshot_id=identifier, manifest_sha256=sha256(Path(fresh["directory"]) / "manifest.json"), last_error=None)
                self._write_state(state)
                return self._result("updated", fresh, state, now, "Datos actualizados y verificados; los modelos permanecen congelados.")
        except (OSError, ValueError) as failure:
            state["last_error"] = type(failure).__name__
            return self._result("failed", data, state, now, "No se pudo acceder al estado de actualización; se conserva la copia disponible.")
