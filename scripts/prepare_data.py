"""Build a causal, unscaled view from a separate unfilled snapshot."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from oil_forecast.acquisition import download_snapshot, load_snapshot, sha256
from oil_forecast.temporal import apply_availability, build_features, build_target, evaluation_schedule


def validate_prepared(features: pd.DataFrame, targets: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Check the actual generated files' contracts before publishing their manifest."""
    for frame, key in [(features, "decision_date"), (targets, "observation_date")]:
        if not frame[key].is_monotonic_increasing or frame[key].duplicated().any():
            raise ValueError(f"Unordered or duplicate output dates: {key}")
    original_targets = raw.loc[raw["variable"].eq("brent")].set_index("observation_date")["value"]
    expected = targets["observation_date"].map(original_targets)
    if not expected.equals(targets["y_brent"]) or not targets["observed"].equals(expected.notna()):
        raise ValueError("Targets differ from exact unfilled original observations")
    for variable in raw["variable"].unique():
        published = pd.to_datetime(features[f"{variable}__available_at"], utc=True)
        if (published > features["decision_at"]).any():
            raise ValueError(f"Future availability in features: {variable}")
        stale = features[f"{variable}__is_stale"]
        known = features[f"{variable}__is_available"]
        if features.loc[stale | ~known, variable].notna().any():
            raise ValueError(f"Missing/stale observation used: {variable}")
    return {"ordered_unique_dates": True, "targets_match_unfilled_source": True,
            "no_future_availability_under_declared_policy": True, "stale_features_masked": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--snapshot", help="New raw snapshot name; never overwrites an existing one")
    parser.add_argument("--download", action="store_true", help="Acquire from providers before preparing")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["decision_timezone"] != "Europe/Madrid" or config["decision_hour"] != 8:
        raise ValueError("This frozen protocol uses 08:00 Europe/Madrid")
    if config["horizons_business_days"] != [5, 63, 252]:
        raise ValueError("The assignment requires the frozen horizons 5, 63, 252")
    snapshot = args.snapshot or config["snapshot_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", snapshot):
        raise ValueError("Snapshot must be a simple local identifier")
    original = ROOT / "data/petroleo_dataset.csv"
    protected = [original, ROOT / "scripts/download_data.py", ROOT / "docs/assignment/ENUNCIADO.pdf",
                 ROOT / "docs/README_profesor.md", ROOT / "pyproject.toml", ROOT / "poetry.lock"]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected if path.exists()}
    if original.exists() and before[str(original.relative_to(ROOT))] != config["original_sha256"]:
        raise ValueError("Original CSV differs from the audited snapshot; investigate before preparing")
    raw_dir = ROOT / "data/raw" / snapshot
    try:
        if args.download:
            download_snapshot(config, raw_dir, ROOT / "data/.cache/yfinance")
        raw, source_manifest = load_snapshot(raw_dir)
        if (source_manifest["start"], source_manifest["end"]) != (config["start"], config["end"]):
            raise ValueError("Snapshot range differs from the frozen configuration")
        available = apply_availability(raw, config["sources"])
        features = build_features(available, config["start"], config["end"], config["decision_hour"])
        targets = build_target(available, config["start"], config["end"])
        schedule = evaluation_schedule(config)
        checks = validate_prepared(features, targets, available)
        output = ROOT / "data/processed" / snapshot
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in {"features.csv": features, "target.csv": targets, "schedule.csv": schedule}.items():
            frame.to_csv(output / name, index=False, lineterminator="\n")
        observed = targets.loc[targets["observed"]]
        source_quality = {}
        for variable, group in raw.groupby("variable", sort=True):
            valid = group.loc[group["is_observed"]]
            weekend = valid["observation_date"].dt.day_of_week.ge(5)
            source_quality[variable] = {
                "observed_weekday_rows": int((~weekend).sum()),
                "observed_weekend_rows": int(weekend.sum()),
                "negative_observations": int(valid["value"].lt(0).sum()),
                "zero_observations": int(valid["value"].eq(0).sum()),
                "missing_feature_rows_after_policy": int(features[variable].isna().sum())
                if variable in features else None,
            }
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        manifest = {"snapshot_id": snapshot, "raw_manifest": str((raw_dir / "manifest.json").relative_to(ROOT)),
                    "raw_sha256": source_manifest["observations_sha256"],
                    "config_sha256": sha256(args.config), "config": config, "code_revision": revision,
                    "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                    [Path(__file__).resolve(), *sorted((ROOT / "src/oil_forecast").glob("*.py"))]},
                    "python": platform.python_version(), "pandas": pd.__version__,
                    "protected_input_sha256": before, "rows_decisions": len(features),
                    "observed_brent_labels": int(targets["observed"].sum()),
                    "last_observed_brent": str(observed["observation_date"].max().date()),
                    "source_coverage": source_manifest["sources"],
                    "source_quality": source_quality,
                    "selected_features": sorted(available["variable"].unique()),
                    "checks": checks,
                    "output_sha256": {name: sha256(output / name) for name in ["features.csv", "target.csv", "schedule.csv"]},
                    "scaled": False, "historical_point_in_time": False,
                    "limitations": ["available_at is an explicit lag assumption unless publication is supplied; not an official release time",
                                    "current source vintages may contain revisions; exact historical information sets are not guaranteed",
                                    "missing/stale features remain NaN; fit imputers/scalers only in the future training fold",
                                    "OPEC or Yahoo unavailable sources remain absent, with their failures recorded; no silent substitution"]}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output.relative_to(ROOT)), "decision_rows": len(features),
                          "observed_brent_labels": manifest["observed_brent_labels"],
                          "last_observed_brent": manifest["last_observed_brent"],
                          "unavailable_sources": [key for key, spec in source_manifest["sources"].items() if spec["status"] != "ok"],
                          "historical_point_in_time": False}, ensure_ascii=False, indent=2))
    finally:
        after = {str(path.relative_to(ROOT)): sha256(path) for path in protected if path.exists()}
        if before != after:
            raise RuntimeError("A protected original input changed while preparing the data")


if __name__ == "__main__":
    main()
