"""Read only the published aggregate scores; never load prices or model states."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_SHA256 = "dd1073129897e6df6c62a0582b5fbbd61e30eb24709dd915840827f26079ce8f"
KEYS = ["experiment", "phase", "model", "variant", "seed", "block_id", "horizon"]
SCORES = ["mse", "mae", "rmse", "bias", "r2"]
COLUMNS = KEYS + ["n", "n_dropped"] + SCORES
MODEL_ORDER = {"persistence": 0, "RNN": 1, "LSTM": 2, "GRU": 3}


def load_public_metrics(directory: Path | str) -> pd.DataFrame:
    """Verify the independently pinned manifest and its one public CSV.

    This supplement contains aggregate scores only. Historical source hashes
    in its metadata are provenance, not resources to open or download.
    """
    directory = Path(directory)
    payload = (directory / "artifact_manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("El manifiesto público no coincide con el registro verificado")
    manifest = json.loads(payload)
    if manifest.get("bundle_id") != "brent-academic-v1" or set(manifest["files"]) != {"metrics.csv"}:
        raise ValueError("Suplemento público no reconocido")
    expected = manifest["files"]["metrics.csv"]
    contents = (directory / "metrics.csv").read_bytes()
    if len(contents) != expected["size"] or hashlib.sha256(contents).hexdigest() != expected["sha256"]:
        raise ValueError("Las métricas públicas no coinciden con su SHA256")
    frame = pd.read_csv(io.BytesIO(contents))
    if (list(frame.columns) != COLUMNS or len(frame) != 1017 or frame[KEYS].isna().any().any()
            or frame.duplicated(KEYS).any() or frame["n"].lt(111).any()
            or not frame["model"].isin(MODEL_ORDER).all()
            or not frame["phase"].isin(["development", "calibration", "test"]).all()
            or not frame["horizon"].isin([5, 63, 252]).all()
            or not frame["seed"].isin([-1, 42, 123, 2026]).all()
            or not np.isfinite(frame[["mse", "mae", "rmse", "bias"]].to_numpy()).all()):
        raise ValueError("Contrato de métricas agregadas no válido")
    return frame


def comparison_rows(metrics: pd.DataFrame, *, experiment: str = "final", phase: str = "test",
                    horizon: int = 5, seed_mode: str = "mean", block_id: str = "all") -> pd.DataFrame:
    """Select stored scores without pooling seeds, horizons or evaluation phases.

    The mean uses available seeds of each variant and reports their count.
    Baseline rows remain unique. Skill is a descriptive ratio of these stored
    MAEs, never a new forecast or a reason to select a model from test.
    """
    seed_mode = str(seed_mode)
    if horizon not in (5, 63, 252) or seed_mode not in {"mean", "42", "123", "2026"}:
        raise ValueError("Horizonte o modo de semillas no reconocido")
    rows = metrics.loc[metrics["experiment"].eq(experiment) & metrics["phase"].eq(phase)
                       & metrics["horizon"].eq(horizon) & metrics["block_id"].eq(block_id)].copy()
    if seed_mode != "mean":
        rows = rows.loc[rows["model"].eq("persistence") | rows["seed"].eq(int(seed_mode))]
    if rows.empty or rows.duplicated(KEYS).any():
        raise ValueError("No hay un conjunto único de métricas guardadas para esa selección")
    baseline = rows.loc[rows["model"].eq("persistence")]
    if len(baseline) != 1 or int(baseline["seed"].iloc[0]) != -1:
        raise ValueError("Se necesita una única referencia de persistencia")
    if rows["n"].nunique() != 1:
        raise ValueError("Los recuentos de evaluación no coinciden")
    baseline_mae = float(baseline["mae"].iloc[0])
    records = []
    for (model, variant), group in rows.groupby(["model", "variant"], sort=False):
        if model not in MODEL_ORDER or group["seed"].duplicated().any():
            raise ValueError("Arquitectura o semillas duplicadas en la comparación")
        scores = {name: float(group[name].mean()) for name in SCORES}
        skill = (100. * (1. - scores["mae"] / baseline_mae) if baseline_mae > 0 else np.nan)
        if model == "persistence":
            skill = 0.
        records.append({"model": model, "variant": variant, "n": int(group["n"].iloc[0]),
                        "seed_count": int(group["seed"].nunique()), **scores, "skill_mae_pct": skill})
    result = pd.DataFrame(records)
    return (result.assign(_order=result["model"].map(MODEL_ORDER))
            .sort_values(["_order", "variant"]).drop(columns="_order").reset_index(drop=True))
