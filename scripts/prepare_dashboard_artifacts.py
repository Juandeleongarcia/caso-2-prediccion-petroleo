"""Copy a compact, verified selection of saved experiments for the dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from oil_forecast.acquisition import sha256
from oil_forecast.artifacts import MANIFEST, _fresh, _json, _manifest, _safe_name, _zip_bytes

EXPERIMENTS = {
    "smoke": "cpu-smoke-3epochs-seed42",
    "optimization": "cpu-nested-ab-20epochs-seeds-v1",
    "refinement": "cpu-refinement-persistence-init-v1",
    "final": "cpu-final-evaluation-v1",
}
PREDICTION_COLUMNS = [
    "experiment", "phase", "block_id", "model", "variant", "seed", "horizon", "sample_id", "origin_date",
    "target_date", "reference_brent", "y_true", "y_pred", "predicted_change", "error", "lower",
    "upper", "interval_level", "label_available_at", "training_cutoff", "refit_at",
    "decision_at", "reference_observation_date", "reference_available_at", "window_start_date",
]
RUN_COLUMNS = [
    "phase", "model", "variant", "seed", "block_id", "parameters", "best_epoch", "last_epoch",
    "epochs", "stopping_reason", "train_loss_at_best", "validation_loss_at_best", "training_seconds",
    "initial_train_loss", "final_train_loss", "initial_validation_loss", "final_validation_loss",
]


def _normalize(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    frame = frame.copy()
    if "phase" not in frame:
        frame["phase"] = frame["block_id"].str.split("_").str[0]
    if "seed" not in frame:
        frame["seed"] = 42
    if "variant" not in frame:
        frame["variant"] = "smoke"
    if "model" in frame:
        baseline = frame["model"].eq("persistence")
        frame.loc[baseline, "variant"] = "baseline"
        frame.loc[baseline, "seed"] = -1
    frame.insert(0, "experiment", experiment)
    return frame


def _resource_metadata(metrics: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Attach saved run resources; no error metric is recalculated or selected."""
    metrics = metrics.copy()
    if "parameters" not in metrics:
        metrics["parameters"] = float("nan")
    if "training_seconds" not in metrics:
        metrics["training_seconds"] = float("nan")
    for index, row in metrics.iterrows():
        if row["model"] == "persistence":
            metrics.loc[index, ["parameters", "training_seconds"]] = [0, 0]
            continue
        matching = runs.loc[runs["model"].eq(row["model"]) & runs["variant"].eq(row["variant"])
                            & runs["seed"].eq(row["seed"]) & runs["phase"].eq(row["phase"])]
        if row["block_id"] != "all":
            matching = matching.loc[matching["block_id"].eq(row["block_id"])]
        matching = matching.drop_duplicates(["model", "variant", "seed", "block_id"], keep="last")
        if matching.empty:
            raise ValueError("Missing saved resources for a dashboard metric")
        if pd.isna(row["parameters"]):
            metrics.loc[index, "parameters"] = matching["parameters"].iloc[0]
        if pd.isna(row["training_seconds"]):
            metrics.loc[index, "training_seconds"] = matching["training_seconds"].sum()
    return metrics


def build_evidence(source_root: Path | str, output_dir: Path | str, archive_path: Path | str) -> dict:
    """Build offline dashboard evidence without loading or fitting any model."""
    source_root = Path(source_root).resolve()
    output, archive = Path(output_dir).absolute(), Path(archive_path).absolute()
    _fresh(output)
    _fresh(archive)
    if output == archive or output in archive.parents or archive in output.parents:
        raise ValueError("Dashboard archive and payload directory must be separate")
    reports, protected = {}, {}
    for experiment, identifier in EXPERIMENTS.items():
        folder = source_root / identifier
        report_path = folder / "experiment.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protected[report_path] = sha256(report_path)
        for name, digest in report["output_sha256"].items():
            if not _safe_name(name) or sha256(folder / name) != digest:
                raise ValueError(f"Historical output hash mismatch: {identifier}/{name}")
            protected[folder / name] = digest
        reports[experiment] = report
    runs = {name: _normalize(pd.DataFrame(report["runs"]), name) for name, report in reports.items()}
    resources = pd.concat([runs["optimization"], runs["smoke"], runs["refinement"], runs["final"]], ignore_index=True)
    metrics, histories, predictions = [], [], []
    for experiment, identifier in EXPERIMENTS.items():
        folder = source_root / identifier
        read = lambda name: pd.read_csv(folder / name)
        metric = _normalize(read("metrics.csv"), experiment)
        metrics.append(_resource_metadata(metric, runs[experiment] if experiment == "smoke" else resources))
        histories.append(_normalize(read("history.csv"), experiment))
        saved = _normalize(read("predictions.csv"), experiment)
        saved = saved.loc[saved["seed"].isin([42, -1])].copy()
        if "predicted_change" not in saved:
            saved["predicted_change"] = saved["y_pred"] - saved["reference_brent"]
        predictions.append(saved.reindex(columns=PREDICTION_COLUMNS))
    catalog = {}
    for name, report in reports.items():
        table = runs[name]
        catalog[name] = {"run_id": report["run_id"], "configuration": report.get("config", report.get("protocol")),
                         "runs": table[[column for column in RUN_COLUMNS if column in table]].to_dict(orient="records")}
    tables = {"metrics.csv": pd.concat(metrics, ignore_index=True),
              "history.csv": pd.concat(histories, ignore_index=True),
              "predictions.csv": pd.concat(predictions, ignore_index=True)}
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dashboard-", dir=output.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        for name, table in tables.items():
            table.to_csv(stage / name, index=False, lineterminator="\n")
        (stage / "interval_metrics.csv").write_bytes((source_root / EXPERIMENTS["final"] / "interval_metrics.csv").read_bytes())
        (stage / "experiment_catalog.json").write_bytes(_json(catalog))
        manifest = {"schema_version": 1, "bundle_id": "brent-dashboard-v1", "artifact_kind": "saved_dashboard_evidence",
            "experiments": list(EXPERIMENTS), "prediction_seeds": [42], "metric_seeds": [42, 123, 2026],
            "selection_rule": "All saved metrics and curves; point predictions use canonical seed 42 and persistence. No favorable block or seed selection.",
            "missing_validation_rule": "Final fixed-epoch runs have no validation loss; missing values remain missing.",
            "metadata_normalizations": "Smoke seed 42 and variant smoke come from its fixed configuration; persistence uses seed -1 and variant baseline.",
            "source_files": {str(path.relative_to(source_root)).replace("\\", "/"): digest
                             for path, digest in sorted(protected.items())},
            "row_counts": {name: len(table) for name, table in tables.items()},
            "files": {path.name: {"sha256": sha256(path), "size": path.stat().st_size} for path in sorted(stage.iterdir())}}
        (stage / MANIFEST).write_bytes(_json(manifest))
        _manifest(stage)
        zipped = _zip_bytes(stage)
        if any(sha256(path) != digest for path, digest in protected.items()):
            raise ValueError("Dashboard packaging changed a historical source")
        _fresh(output)
        _fresh(archive)
        with archive.open("xb") as stream:
            stream.write(zipped)
        stage.rename(output)
    return {"bundle_id": manifest["bundle_id"], "archive_sha256": hashlib.sha256(zipped).hexdigest(),
            "archive_size": len(zipped), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/local/brent-dashboard-v1")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/releases/brent-dashboard-v1.zip")
    args = parser.parse_args()
    print(json.dumps(build_evidence(args.source_root, args.output, args.archive), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
