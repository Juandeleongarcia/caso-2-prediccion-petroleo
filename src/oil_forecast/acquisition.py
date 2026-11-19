"""Unfilled source snapshots. No inference from repeated prices or model fitting."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree

import pandas as pd


def safe_acquisition_error(error: Exception) -> dict:
    """Only categories/codes; exception strings may include credentials or URLs."""
    details={'error_type':type(error).__name__}
    if isinstance(error,HTTPError):details.update(reason='http_error',http_status=error.code)
    elif isinstance(error,TimeoutError) or (isinstance(error,URLError) and isinstance(error.reason,TimeoutError)):
        details['reason']='timeout'
    elif isinstance(error,URLError):details['reason']='connection_or_tls_error'
    elif isinstance(error,ValueError):details['reason']='schema_or_observation_invalid'
    else:details['reason']='provider_error'
    return details


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous keys and malformed values instead of silently deduplicating."""
    required = {"observation_date", "variable", "value", "source"}
    if not required.issubset(frame):
        raise ValueError(f"Missing source columns: {sorted(required - set(frame))}")
    result = frame.copy()
    result["observation_date"] = pd.to_datetime(result["observation_date"], errors="raise")
    if result["observation_date"].isna().any():
        raise ValueError("Missing observation date")
    if result[["variable", "source"]].isna().any().any():
        raise ValueError("Missing variable/source")
    if result.duplicated(["variable", "observation_date"]).any():
        raise ValueError("Duplicate source observation date")
    result["value"] = pd.to_numeric(result["value"], errors="raise")
    if result["value"].isin([float("inf"), -float("inf")]).any():
        raise ValueError("Infinite source value")
    if not result["value"].notna().any():
        raise ValueError("Source contains no observed value")
    result["is_observed"] = result["value"].notna()
    return result.sort_values(["variable", "observation_date"]).reset_index(drop=True)


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "oil-forecast-academic/0.1",
                                   "Accept-Language": "en-GB,en;q=0.9"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def parse_fred(body: bytes, variable: str, series_id: str) -> pd.DataFrame:
    data = pd.read_csv(io.BytesIO(body), na_values=".")
    if data.shape[1] != 2 or data.columns[1] != series_id:
        raise ValueError(f"Unexpected FRED schema for {series_id}")
    data.columns = ["observation_date", "value"]
    return validate_observations(data.assign(variable=variable, source=f"FRED:{series_id}"))


def parse_opec(body: bytes) -> pd.DataFrame:
    """Only accept the official XML's explicit date/value records; never guess prices."""
    root = ElementTree.fromstring(body)
    records = []
    for element in root.iter():
        fields = {key.lower(): value for key, value in element.attrib.items()}
        if "date" in fields and "value" in fields:
            records.append({"observation_date": fields["date"], "value": fields["value"]})
    if not records:
        raise ValueError("OPEC XML schema not recognised; no data incorporated")
    data = pd.DataFrame(records)
    data["observation_date"] = pd.to_datetime(data["observation_date"], format="%Y-%m-%d", errors="raise")
    return validate_observations(data.assign(variable="opep_orb", source="OPEC:ORB"))


def _download_one(variable: str, spec: dict, config: dict, cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    start, end = config["start"], config["end"]
    provider, series_id = spec["provider"], spec["id"]
    metadata = {"provider": provider, "id": series_id, "unit": spec["unit"],
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "published_at_known": False, "historical_vintages": False}
    if provider == "fred":
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
        body = _fetch(url)
        frame = parse_fred(body, variable, series_id)
    elif provider == "opec":
        url = series_id
        body = _fetch(url)
        frame = parse_opec(body)
    elif provider == "yahoo":
        import yfinance as yf

        yf.set_tz_cache_location(str(cache_dir))
        # Yahoo's end is exclusive. Preserve missing records; no repair or filling.
        exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(series_id, start=start, end=exclusive_end, auto_adjust=False,
                           repair=False, keepna=True, progress=False, threads=False, timeout=30)
        if data.empty:
            raise ValueError(f"Yahoo returned no observations for {series_id}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        values = data["Close"]
        frame = validate_observations(pd.DataFrame({"observation_date": values.index.tz_localize(None),
                                                    "value": values.to_numpy(), "variable": variable,
                                                    "source": f"Yahoo:{series_id}"}))
        url = f"https://finance.yahoo.com/quote/{series_id}/history/"
        body = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    frame = frame.loc[frame["observation_date"].between(start, end)].copy()
    frame = validate_observations(frame)
    valid = frame.loc[frame["is_observed"], "observation_date"]
    metadata.update(url=url, response_sha256=hashlib.sha256(body).hexdigest(),
                    response_hash_kind="canonical_yfinance_close" if provider == "yahoo" else "http_body",
                    rows=len(frame), observed=int(frame["is_observed"].sum()),
                    missing=int(frame["value"].isna().sum()),
                    first_observed=valid.min().strftime("%Y-%m-%d"),
                    last_observed=valid.max().strftime("%Y-%m-%d"),
                    status="ok")
    return frame, metadata


def download_snapshot(config: dict, directory: Path, cache_dir: Path) -> None:
    """Create an immutable local snapshot; fail if a required source is unavailable."""
    if directory.exists():
        raise FileExistsError(f"Snapshot already exists: {directory}. Use a new --snapshot identifier.")
    specs = {key: spec for key, spec in config["sources"].items() if spec["enabled"]}
    frames, metadata = [], {}
    # Yahoo's shared client/cache is used sequentially; FRED sources are independent.
    fred = {key: spec for key, spec in specs.items() if spec["provider"] == "fred"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(_download_one, key, spec, config, cache_dir): key for key, spec in fred.items()}
        for future in as_completed(pending):
            key = pending[future]
            try:
                frame, info = future.result()
                frames.append(frame)
                metadata[key] = info
            except Exception as error:
                metadata[key] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}",
                                 "diagnostic":safe_acquisition_error(error),
                                 "provider": specs[key]["provider"], "id": specs[key]["id"]}
    required_failures = [key for key in fred if specs[key]["required"] and metadata[key]["status"] != "ok"]
    for key, spec in specs.items():
        if key in fred:
            continue
        if required_failures:
            metadata[key] = {"status": "not_attempted", "error": "Required FRED acquisition failed",
                             "provider": spec["provider"], "id": spec["id"]}
            continue
        try:
            frame, info = _download_one(key, spec, config, cache_dir)
            frames.append(frame)
            metadata[key] = info
        except Exception as error:
            metadata[key] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}",
                             "diagnostic":safe_acquisition_error(error),
                             "provider": spec["provider"], "id": spec["id"]}
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "start": config["start"],
                "end": config["end"], "sources": metadata,
                "limitations": "Unfilled current-vintage observations; historical publication timestamps unknown."}
    directory.mkdir(parents=True)
    failures = [key for key, spec in specs.items() if spec["required"] and metadata[key]["status"] != "ok"]
    if not failures:
        raw = validate_observations(pd.concat(frames, ignore_index=True))
        raw.to_csv(directory / "observations.csv", index=False, lineterminator="\n")
        manifest["observations_sha256"] = sha256(directory / "observations.csv")
    manifest["required_sources_unavailable"] = failures
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Required sources unavailable: {failures}; see {directory / 'manifest.json'}")


def load_snapshot(directory: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("required_sources_unavailable"):
        raise ValueError("Snapshot has failed required sources")
    path = directory / "observations.csv"
    if sha256(path) != manifest["observations_sha256"]:
        raise ValueError("Snapshot hash mismatch")
    return validate_observations(pd.read_csv(path)), manifest
