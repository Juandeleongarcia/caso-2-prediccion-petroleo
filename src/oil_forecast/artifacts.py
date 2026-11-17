"""Hash-verified portable artifacts; network access only through explicit fetch."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import stat
import tempfile
import zipfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
import torch

from .acquisition import sha256
from .inference import load_forecaster

LIMIT = 100 * 1024 * 1024
MANIFEST = "artifact_manifest.json"


def _json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _safe_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)) and name not in {".", ".."}


def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Existing artifacts cannot be overwritten: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Artifact destinations cannot have symlink ancestors")


def _manifest(directory: Path) -> dict:
    if directory.is_symlink() or any(parent.is_symlink() for parent in directory.parents):
        raise ValueError("Artifact directories cannot use symlinks")
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if manifest.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        raise ValueError("Unrecognized portable artifact manifest")
    if set(path.name for path in directory.iterdir()) != {*files, MANIFEST}:
        raise ValueError("Unexpected or missing portable artifact files")
    for name, expected in files.items():
        path = directory / name
        if (not _safe_name(name) or not isinstance(expected, dict) or path.is_symlink() or not path.is_file()
                or type(expected.get("size")) is not int or not 0 <= expected["size"] <= LIMIT
                or path.stat().st_size != expected["size"] or sha256(path) != expected.get("sha256")):
            raise ValueError(f"Portable artifact integrity failure: {name}")
    return manifest


def verify_bundle(directory: Path | str) -> dict:
    """Verify all payloads, then load and forecast without original experiments."""
    directory = Path(directory).absolute()
    manifest = _manifest(directory)
    if manifest.get("bundle_id") == "brent-dashboard-v1":
        required = {"metrics.csv", "history.csv", "predictions.csv", "interval_metrics.csv", "experiment_catalog.json"}
        if not required.issubset(manifest["files"]):
            raise ValueError("Incomplete dashboard evidence bundle")
        metrics = pd.read_csv(directory / "metrics.csv")
        predictions = pd.read_csv(directory / "predictions.csv")
        if (not {"phase", "model", "horizon", "mae", "rmse", "mse"}.issubset(metrics.columns)
                or not metrics["phase"].isin(["development", "calibration", "test"]).all()
                or not predictions["phase"].isin(["development", "calibration", "test"]).all()):
            raise ValueError("Invalid dashboard evidence phases or schema")
        return {"manifest": manifest, "verified_files": len(manifest["files"]),
                "metric_rows": len(metrics), "prediction_rows": len(predictions)}
    service = load_forecaster(directory)
    forecast = service.predict(pd.read_csv(directory / "latest_window.csv"))
    return {"manifest": manifest, "forecast": forecast, "verified_files": len(manifest["files"])}


def _zip_bytes(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(directory.iterdir()):
            member = zipfile.ZipInfo(path.name, date_time=(2000, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, path.read_bytes(), compresslevel=9)
    return buffer.getvalue()


def build_bundle(source_dir: Path | str, processed_dir: Path | str,
                 output_dir: Path | str, archive_path: Path | str) -> dict:
    """Package three frozen seed-42 networks and the data needed for inference.

    Historical outputs are copied or subsetted, never recomputed. The source
    run, data files, weights, transforms and uncertainty state remain untouched.
    """
    source, processed = Path(source_dir).resolve(), Path(processed_dir).resolve()
    output, archive = Path(output_dir).absolute(), Path(archive_path).absolute()
    _fresh(output)
    _fresh(archive)
    if output == archive or output in archive.parents or archive in output.parents:
        raise ValueError("The archive must be separate from the bundle directory")
    report = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    for name, digest in report["output_sha256"].items():
        if not _safe_name(name) or sha256(source / name) != digest:
            raise ValueError(f"Final evaluation source hash mismatch: {name}")
    prepared = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    for name in ["features.csv", "target.csv"]:
        if sha256(processed / name) != prepared["output_sha256"][name]:
            raise ValueError(f"Prepared data hash mismatch: {name}")
        suffix = f"data/processed/{processed.name}/{name}"
        expected = [digest for key, digest in report["input_sha256"].items()
                    if key.replace("\\", "/").endswith(suffix)]
        if expected != [sha256(processed / name)]:
            raise ValueError("Prepared data does not belong to the final evaluation")
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    serving_fold = pd.read_csv(source / "folds.csv").set_index("block_id").loc["test_2024H2"]
    if (protocol["selected_model"] != "LSTM" or protocol["selected_seed"] != 42
            or protocol["variants"] != {"RNN": "D", "LSTM": "B", "GRU": "C"}):
        raise ValueError("Portable artifacts must preserve the frozen final selection")
    previous = torch.load(source / "checkpoints.pt", map_location="cpu", weights_only=True)
    fields = ["model", "variant", "seed", "phase", "block_id", "refit_at", "training_cutoff",
              "model_config", "state_dict", "transformers"]
    models = {}
    for model, variant in protocol["variants"].items():
        key = f"{model}_42_test_2024H2"
        entry = previous["models"][key]
        if entry["variant"] != variant or entry["seed"] != 42:
            raise ValueError("Checkpoint differs from the frozen candidate")
        models[key] = {field: entry[field] for field in fields}
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=output.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        for name in ["protocol.json", "metrics.csv", "intervals.json", "interval_metrics.csv"]:
            (stage / name).write_bytes((source / name).read_bytes())
        checkpoint = io.BytesIO()
        torch.save({"schema_version": 1, "models": models}, checkpoint)
        (stage / "checkpoints.pt").write_bytes(checkpoint.getvalue())
        (stage / "brent_history.csv").write_bytes((processed / "target.csv").read_bytes())
        window = pd.read_csv(processed / "features.csv").tail(60)
        window.to_csv(stage / "latest_window.csv", index=False, lineterminator="\n")
        predictions = pd.read_csv(source / "predictions.csv")
        predictions.loc[predictions["model"].eq("LSTM") & predictions["seed"].eq(42)
                        & predictions["block_id"].eq("test_2024H2")].to_csv(
                            stage / "reference_predictions.csv", index=False, lineterminator="\n")
        packaged_report = {"schema_version": 1, "artifact_kind": "frozen_inference_bundle",
            "deployment": report["deployment"], "environment": report["environment"],
            "source_final_report_sha256": sha256(source / "experiment.json"),
            "limitations": report["limitations"],
            "output_sha256": {path.name: sha256(path) for path in sorted(stage.iterdir())}}
        (stage / "experiment.json").write_bytes(_json(packaged_report))
        manifest = {"schema_version": 1, "bundle_id": "brent-frozen-v1", "selected_model": "LSTM-B",
            "selected_seed": 42, "candidates": protocol["variants"], "epochs": protocol["epochs"],
            "horizons": protocol["horizons"], "model_refit_at": report["deployment"]["model_refit_at"],
            "logical_refit_at": report["deployment"]["model_refit_at"],
            "actual_training_execution_at": None,
            "training_execution_provenance": "Not recorded; logical backtest dates are not execution timestamps",
            "training_cutoff": report["deployment"]["training_cutoff"],
            "latest_train_origin": str(serving_fold["last_train_origin"]),
            "latest_train_label_available_at": str(serving_fold["latest_train_label_available_at"]),
            "data_updated_at": protocol["data_updated_at"], "snapshot": protocol["data_snapshot"],
            "latest_origin_date": str(window["decision_date"].iloc[-1]),
            "reference_observation_date": str(window["brent__observation_date"].iloc[-1]),
            "source_final_report_sha256": sha256(source / "experiment.json"),
            "source_checkpoint_sha256": sha256(source / "checkpoints.pt"),
            "source_protocol_sha256": sha256(source / "protocol.json"),
            "environment": report["environment"],
            "reproducibility_scope": "Verified Windows CPU; cross-platform floating-point equality is not guaranteed",
            "files": {path.name: {"sha256": sha256(path), "size": path.stat().st_size}
                      for path in sorted(stage.iterdir())}}
        (stage / MANIFEST).write_bytes(_json(manifest))
        verify_bundle(stage)
        zipped = _zip_bytes(stage)
        _fresh(output)
        _fresh(archive)
        # Each destination is new; no historical checkpoint or archive is replaced.
        with archive.open("xb") as stream:
            stream.write(zipped)
        stage.rename(output)
    return {"bundle_id": manifest["bundle_id"], "archive_sha256": hashlib.sha256(zipped).hexdigest(),
            "archive_size": len(zipped), "manifest": manifest}


def install_bundle(archive: Path | str, destination: Path | str, expected_sha256: str) -> dict:
    """Install a local trusted archive after digest and member-path checks."""
    archive, destination = Path(archive), Path(destination).absolute()
    _fresh(destination)
    if (archive.stat().st_size > LIMIT or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or sha256(archive) != expected_sha256):
        raise ValueError("Archive SHA256 differs from the trusted expected digest")
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        names = [entry.filename for entry in entries]
        if (not entries or len(entries) > 32 or len(set(name.casefold() for name in names)) != len(names)
                or sum(entry.file_size for entry in entries) > LIMIT
                or any(not _safe_name(entry.filename) or entry.file_size > LIMIT
                       or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1
                       or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED} for entry in entries)):
            raise ValueError("Unsafe, duplicate or oversized archive members")
        if MANIFEST not in names:
            raise ValueError("Archive has no artifact manifest")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "payload"
            stage.mkdir()
            for entry in entries:
                (stage / entry.filename).write_bytes(zipped.read(entry))
            verified = verify_bundle(stage)
            _fresh(destination)
            stage.rename(destination)
    return verified


def fetch_bundle(url: str, destination: Path | str, expected_sha256: str) -> dict:
    """Explicit HTTPS download anchored to an independently supplied digest.

    The caller decides when to fetch; inference never invokes this function.
    Existing bundles are immutable and network failures publish no destination.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Artifact downloads require an HTTPS URL without embedded credentials")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A trusted SHA256 is required before downloading")
    target = Path(destination).absolute()
    _fresh(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=target.parent) as temporary:
        archive = Path(temporary) / "bundle.zip"
        request = Request(url, headers={"User-Agent": "BrentForecast/1.0"})
        with urlopen(request, timeout=30) as response, archive.open("xb") as output:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Artifact redirects must retain HTTPS")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > LIMIT:
                    raise ValueError("Artifact download exceeds the size limit")
                output.write(chunk)
        return install_bundle(archive, target, expected_sha256)
