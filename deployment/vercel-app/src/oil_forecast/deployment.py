"""Provision verified frozen resources without training or changing originals."""

from __future__ import annotations

import json
from pathlib import Path
import re
from threading import Lock
from typing import Mapping
from urllib.parse import urlparse

from .acquisition import sha256
from .artifacts import fetch_bundle, install_bundle, verify_bundle

_PREPARATION_LOCK = Lock()


def verify_registered_manifest(directory: Path | str, expected_sha256: str) -> None:
    """Anchor installed metadata to the independent application registry."""
    if (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or sha256(Path(directory) / "artifact_manifest.json") != expected_sha256):
        raise ValueError("Installed manifest differs from its independently trusted registry")


def hf_artifact_url(repo_id: str, revision: str, filename: str) -> str:
    """Use an immutable Models revision, never a branch or a token in a URL."""
    component = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    if (not isinstance(repo_id, str) or not re.fullmatch(component + "/" + component, repo_id)
            or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or not isinstance(filename, str)
            or filename not in {"brent-frozen-v1.zip", "brent-dashboard-v1.zip"}):
        raise ValueError("Models requires a namespace/repository and a full immutable commit revision")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def prepare_application_resources(root: Path | str, bundle: Path | str, dashboard: Path | str,
                                  settings: Mapping[str, object] | None = None) -> dict[str, str]:
    """Install missing packages; verify existing ones, preserving any last-good copy.

    The caller opts into network access. Secrets are used only for transport and
    are never returned, cached, written into configuration or logged here.
    A process lock coordinates simultaneous Streamlit sessions. Archive installs
    remain atomic and refuse overwrites if another process publishes first.
    """
    root, settings = Path(root), settings or {}
    result = {}
    with _PREPARATION_LOCK:
        for registry_name, destination, url_name in (
            ("serving_bundle.json", Path(bundle), "BRENT_INFERENCE_URL"),
            ("dashboard_bundle.json", Path(dashboard), "BRENT_EVIDENCE_URL"),
        ):
            registry = json.loads((root / "configs" / registry_name).read_text(encoding="utf-8"))
            bundle_id = registry["bundle_id"]
            if registry.get("schema_version") != 1 or bundle_id not in {"brent-frozen-v1", "brent-dashboard-v1"}:
                raise ValueError("Unrecognized trusted resource registry")
            if destination.exists():
                verify_registered_manifest(destination, registry["manifest_sha256"])
                verified = verify_bundle(destination)
                if verified["manifest"]["bundle_id"] != bundle_id:
                    raise ValueError("Installed resource differs from its trusted registry")
                result[bundle_id] = "verified"
                continue
            archive = root / registry["archive"]["local_path"]
            if archive.exists():
                install_bundle(archive, destination, registry["archive"]["sha256"])
                result[bundle_id] = "installed_local"
                continue
            url = settings.get(url_name) or registry["archive"].get("published_url")
            if not url:
                repo, revision = settings.get("BRENT_HF_REPO_ID"), settings.get("BRENT_HF_REVISION")
                if not repo or not revision:
                    raise FileNotFoundError("No local archive or immutable remote Models source is configured")
                url = hf_artifact_url(repo, revision, registry["archive"]["filename"])
            if not isinstance(url, str):
                raise ValueError("The configured artifact URL must be text")
            token = settings.get("HF_TOKEN") if urlparse(url).hostname == "huggingface.co" else None
            fetch_bundle(url, destination, registry["archive"]["sha256"], token=token)
            result[bundle_id] = "installed_remote"
    return result
