"""Four reproducible development diagnostics; plotting does not import PyTorch.

Run with an interpreter that already provides Matplotlib, NumPy and pandas.
Plots describe development selection, not an independent final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 63, 252)
ARCHITECTURES = ("RNN", "LSTM", "GRU")
COLORS = {"baseline": "#606770", "B": "#D55E00", "C": "#0072B2", "D": "#009E73"}
MARKERS = {"baseline": "s", "B": "o", "C": "^", "D": "D"}
BLOCK = "development_2019H1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(variant: str) -> str:
    return {"baseline": "Persistencia", "B": "B original",
            "C": "C: inicio persistente", "D": "D: inicio persistente + SmoothL1"}[variant]


def _candidate(report: dict, architecture: str) -> str:
    """Keep a visible diagnostic candidate when B remains the recommendation."""
    recommendation = report.get("recommendations", {}).get(architecture, "B")
    return recommendation if recommendation in {"C", "D"} else "C"


def _rows(table: pd.DataFrame, variant: str, architecture: str,
          *, horizon: int | None = None, seed: int | None = None,
          block: str | None = None) -> pd.DataFrame:
    model = "persistence" if variant == "baseline" else architecture
    selected = table.loc[table["variant"].eq(variant) & table["model"].eq(model)]
    if horizon is not None:
        selected = selected.loc[selected["horizon"].eq(horizon)]
    if seed is not None and variant != "baseline":
        selected = selected.loc[selected["seed"].eq(seed)]
    if block is not None:
        selected = selected.loc[selected["block_id"].eq(block)]
    return selected.copy()


def _save(figure, directory: Path, name: str) -> None:
    figure.savefig(directory / f"{name}.png", dpi=160,
                   metadata={"Software": "oil_forecast development diagnostics"})
    figure.savefig(directory / f"{name}.svg", metadata={"Date": None})
    plt.close(figure)


def _training_curves(history: pd.DataFrame, reference_history: pd.DataFrame,
                     runs: list[dict], reference_runs: list[dict], output: Path) -> None:
    combined = pd.concat([reference_history.loc[reference_history["variant"].eq("B")],
                          history], ignore_index=True)
    variants = [variant for variant in ("B", "C", "D")
                if not _rows(combined, variant, "LSTM", seed=42, block=BLOCK).empty]
    figure, axes = plt.subplots(len(variants), 1, figsize=(10, 2.8 * len(variants) + 1.2),
                                squeeze=False, layout="constrained")
    for axis, variant in zip(axes[:, 0], variants):
        rows = _rows(combined, variant, "LSTM", seed=42, block=BLOCK).sort_values("epoch")
        entries = [run for run in [*reference_runs, *runs]
                   if run["variant"] == variant and run["model"] == "LSTM"
                   and run["seed"] == 42 and run["block_id"] == BLOCK]
        if len(entries) != 1:
            raise ValueError("The representative training curve requires exactly one run summary")
        best = entries[0]["best_epoch"]
        axis.plot(rows["epoch"], rows["train_loss"], color=COLORS[variant],
                  linestyle="-", label="Entrenamiento interior")
        axis.plot(rows["epoch"], rows["validation_loss"], color="#222222",
                  linestyle="--", label="Validación interior")
        axis.axvline(best, color=COLORS[variant], alpha=0.65, linestyle=":",
                     label=f"Checkpoint: época {best}")
        axis.set(title=_label(variant), ylabel="MSE normalizado", ylim=(0, None))
        axis.set_xticks(rows["epoch"])
        axis.legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1, 0].set_xlabel("Época (0 es diagnóstico, sin seleccionar)")
    figure.suptitle("Selección mediante validación interior\n"
                    "LSTM · semilla 42 · desarrollo 2019H1 · criterio MSE común", fontsize=14)
    _save(figure, output, "01_training_curves")


def _price_predictions(predictions: pd.DataFrame, report: dict, output: Path) -> None:
    candidate = _candidate(report, "LSTM")
    variants = ["baseline", "B", candidate]
    selected = pd.concat([_rows(predictions, variant, "LSTM", seed=42)
                          for variant in variants], ignore_index=True)
    low = float(selected[["y_true", "y_pred"]].to_numpy().min())
    high = float(selected[["y_true", "y_pred"]].to_numpy().max())
    padding = (high - low) * 0.05
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharex=True, sharey=True,
                                layout="constrained")
    for axis, horizon in zip(axes, HORIZONS):
        for variant in variants:
            rows = _rows(predictions, variant, "LSTM", horizon=horizon, seed=42)
            axis.scatter(rows["y_true"], rows["y_pred"], s=11, alpha=0.38,
                         color=COLORS[variant], marker=MARKERS[variant],
                         label=f"{_label(variant)} (n={len(rows)})", linewidths=0)
        axis.plot([low - padding, high + padding], [low - padding, high + padding],
                  color="#333333", linestyle="--", linewidth=1, label="Predicción exacta")
        axis.set(title=f"{horizon} días hábiles", xlabel="Precio observado (USD/barril)",
                 xlim=(low - padding, high + padding), ylim=(low - padding, high + padding))
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Precio predicho (USD/barril)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    recommendation = report.get("recommendations", {}).get("LSTM", "B")
    note = "C se muestra como diagnóstico; la recomendación conserva B" if recommendation == "B" else \
           f"Variante candidata: {candidate}"
    figure.suptitle(f"LSTM: predicción frente a observación · desarrollo · semilla 42\n{note}", fontsize=14)
    _save(figure, output, "02_predicted_vs_observed")


def _block_errors(metrics: pd.DataFrame, diagnostic: pd.DataFrame,
                  report: dict, output: Path) -> None:
    candidate = _candidate(report, "LSTM")
    variants = ["baseline", "B", candidate]
    blocks = sorted(metrics.loc[~metrics["block_id"].eq("all"), "block_id"].unique())
    tick_labels = [block.replace("development_", "") for block in blocks]
    x = np.arange(len(blocks))
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, layout="constrained")
    for column, horizon in enumerate(HORIZONS):
        for variant in variants:
            cells = _rows(metrics, variant, "LSTM", horizon=horizon)
            cells = cells.loc[~cells["block_id"].eq("all")]
            count = cells["seed"].nunique()
            mae = cells.groupby("block_id", observed=True)["mae"].mean().reindex(blocks)
            described = _rows(diagnostic, variant, "LSTM", horizon=horizon)
            bias = described.groupby("block_id", observed=True)["bias"].mean().reindex(blocks)
            label = f"{_label(variant)} (n={count} semillas)" if variant != "baseline" else "Persistencia"
            for axis, values in [(axes[0, column], bias), (axes[1, column], mae)]:
                axis.plot(x, values, color=COLORS[variant], marker=MARKERS[variant],
                          linewidth=1.4, markersize=4, label=label)
        axes[0, column].axhline(0, color="#333333", linestyle="--", linewidth=0.9)
        axes[0, column].set_title(f"{horizon} días hábiles")
        axes[1, column].set_ylim(bottom=0)
        axes[1, column].set_xticks(x, tick_labels, rotation=35, ha="right")
        axes[1, column].set_xlabel("Bloque exterior de desarrollo")
    axes[0, 0].set_ylabel("Sesgo: predicho − observado\n(USD/barril)")
    axes[1, 0].set_ylabel("MAE (USD/barril)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    figure.suptitle("LSTM: errores y sesgo por periodo\n"
                    "Medias entre las semillas disponibles; los bloques no son independientes", fontsize=14)
    _save(figure, output, "03_bias_and_errors_by_block")


def _mae_comparison(metrics: pd.DataFrame, report: dict, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), sharey="col", layout="constrained")
    column_maxima = np.zeros(len(HORIZONS), dtype=float)
    for row, architecture in enumerate(ARCHITECTURES):
        candidate = _candidate(report, architecture)
        variants = ["baseline", "B", candidate]
        for column, horizon in enumerate(HORIZONS):
            axis = axes[row, column]
            for position, variant in enumerate(variants):
                cells = _rows(metrics, variant, architecture, horizon=horizon, block="all")
                if cells.empty or cells["seed"].duplicated().any():
                    raise ValueError("The MAE comparison requires one pooled row per variant/model/seed/horizon")
                values = cells["mae"].to_numpy(dtype=float)
                count = len(values)
                deviation = float(values.std(ddof=1)) if count >= 3 else 0.0
                mean = float(values.mean())
                column_maxima[column] = max(column_maxima[column], mean + deviation)
                axis.bar(position, mean, color=COLORS[variant], width=0.68,
                         yerr=deviation if count >= 3 else None, capsize=4, alpha=0.9)
                axis.annotate(f"{mean:.2f}\n" + (f"n={count}" if variant != "baseline" else "baseline"),
                              (position, mean + deviation), xytext=(0, 5), textcoords="offset points",
                              ha="center", va="bottom", fontsize=8)
            axis.set_title(f"{architecture} · {horizon} días hábiles")
            axis.set_xticks([0, 1, 2], ["Persistencia", "B", candidate])
            if column == 0:
                axis.set_ylabel("MAE (USD/barril)")
    # Shared axes need their final limits after all rows have contributed;
    # reserve headroom for both error bars and the two-line annotations.
    for column, maximum in enumerate(column_maxima):
        axes[0, column].set_ylim(0, max(1.0, maximum * 1.35))
    figure.suptitle("Comparación de MAE en desarrollo\n"
                    "Media entre semillas; barras de error = desviación muestral, no intervalo de predicción\n"
                    "Si se conserva B, C se muestra solo como candidata de diagnóstico", fontsize=13)
    _save(figure, output, "04_mae_comparison")


def render_refinement_figures(run_dir: Path) -> dict:
    """Read immutable development results and create exactly four figure pairs.

    Refuse any existing figures directory, preserving prior exports. A fresh
    run/configuration is needed to author a new result; no model is fitted here.
    """
    directory = Path(run_dir).resolve()
    output = directory / "figures"
    if output.exists():
        raise FileExistsError("Existing figure exports are immutable")
    report_path = directory / "experiment.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = Path(report["config"].get("reference_dir", "experiments/cpu-nested-ab-20epochs-seeds-v1"))
    if not reference.is_absolute():
        reference = ROOT / reference
    reference_report = json.loads((reference / "experiment.json").read_text(encoding="utf-8"))
    paths = {name: directory / name for name in ["predictions.csv", "metrics.csv", "diagnostics.csv", "history.csv"]}
    inputs = [report_path, *paths.values(), reference / "history.csv", reference / "experiment.json"]
    input_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in inputs}
    predictions = pd.read_csv(paths["predictions.csv"])
    metrics = pd.read_csv(paths["metrics.csv"])
    diagnostic = pd.read_csv(paths["diagnostics.csv"])
    history = pd.read_csv(paths["history.csv"])
    reference_history = pd.read_csv(reference / "history.csv")
    for table in [predictions, metrics, diagnostic]:
        if not table["phase"].eq("development").all():
            raise ValueError("Refinement figures permit development only")
    for table, names in [(predictions, ["y_true", "y_pred", "reference_brent"]),
                         (metrics, ["mae", "rmse", "mse"]),
                         (diagnostic, ["bias"]),
                         (history, ["train_loss", "validation_loss"])]:
        if not np.isfinite(table[names].to_numpy(dtype=float)).all():
            raise ValueError("Figure inputs must contain finite errors, prices and losses")
    if not history["variant"].isin(["C", "D"]).all():
        raise ValueError("Refinement history should contain only the new variants C/D")
    if not set(HORIZONS).issubset(set(predictions["horizon"])):
        raise ValueError("Missing forecast horizon")
    output.mkdir()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18,
                         "axes.axisbelow": True, "svg.hashsalt": "oil-forecast-refinement-v1"}):
        _training_curves(history, reference_history, report["runs"], reference_report["runs"], output)
        _price_predictions(predictions, report, output)
        _block_errors(metrics, diagnostic, report, output)
        _mae_comparison(metrics, report, output)
    if input_hashes != {str(path.relative_to(ROOT)): _sha256(path) for path in inputs}:
        raise RuntimeError("Plotting changed its input results")
    manifest = {"phase": "development", "figure_count": 4,
                "representative_curve": {"model": "LSTM", "seed": 42, "block_id": BLOCK},
                "prediction_scatter_seed": 42,
                "error_bars": "sample standard deviation between seeds when at least three are available; not prediction intervals",
                "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__, "interpreter": sys.executable},
                "input_sha256": input_hashes, "plot_code_sha256": _sha256(Path(__file__)),
                "output_sha256": {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


FINAL_VARIANTS = {"RNN": "D", "LSTM": "B", "GRU": "C"}
FINAL_MODELS = ("persistence", "RNN", "LSTM", "GRU")


def _final_rows(table: pd.DataFrame, model: str, *, horizon: int | None = None,
                seed: int | None = None, pooled: bool = False) -> pd.DataFrame:
    variant = "baseline" if model == "persistence" else FINAL_VARIANTS[model]
    rows = _rows(table, variant, model, horizon=horizon, seed=seed,
                 block="all" if pooled else None)
    return rows.sort_values("target_date") if "target_date" in rows else rows


def _final_label(model: str) -> str:
    return "Persistencia" if model == "persistence" else f"{model}-{FINAL_VARIANTS[model]}"


def _final_color(model: str) -> str:
    return COLORS["baseline" if model == "persistence" else FINAL_VARIANTS[model]]


def _final_date_axis(axis) -> None:
    from matplotlib import dates as mdates

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _final_error_comparison(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, metric in enumerate(("mae", "rmse")):
        for column, horizon in enumerate(HORIZONS):
            axis = axes[row, column]
            upper = 0.0
            labels = []
            for position, model in enumerate(FINAL_MODELS):
                cells = _final_rows(metrics, model, horizon=horizon, pooled=True)
                expected_seeds = {-1} if model == "persistence" else {42, 123, 2026}
                if set(cells["seed"]) != expected_seeds or cells["seed"].duplicated().any():
                    raise ValueError("Final metrics need one row per frozen model, seed and horizon")
                values = cells[metric].to_numpy(dtype=float)
                mean = float(values.mean())
                deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                upper = max(upper, mean + deviation)
                axis.bar(position, mean, color=_final_color(model), width=0.7,
                         yerr=deviation if len(values) > 1 else None, capsize=4, alpha=0.9)
                axis.annotate(f"{mean:.2f}", (position, mean + deviation), xytext=(0, 5),
                              textcoords="offset points", ha="center", va="bottom", fontsize=9)
                labels.append(f"{_final_label(model)}\nn={len(values)} semillas" if model != "persistence"
                              else "Persistencia")
            axis.set(title=f"{metric.upper()} · {horizon} días hábiles",
                     ylabel=f"{metric.upper()} (USD/barril)", ylim=(0, max(1.0, upper * 1.3)))
            axis.set_xticks(range(len(FINAL_MODELS)), labels, fontsize=8)
    figure.suptitle("Evaluación final en test · orígenes 2024\n"
                    "Media de tres semillas; barras = desviación muestral entre semillas, no intervalos predictivos",
                    fontsize=14)
    _save(figure, output, "01_test_mae_rmse")


def _final_prediction_curves(predictions: pd.DataFrame, seed: int, output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), layout="constrained")
    styles = {"persistence": "--", "RNN": ":", "LSTM": "-", "GRU": "-."}
    for axis, horizon in zip(axes, HORIZONS):
        truth = _final_rows(predictions, "persistence", horizon=horizon)
        axis.plot(truth["target_date"], truth["y_true"], color="#111111", linewidth=1.6,
                  label="Precio observado")
        for model in FINAL_MODELS:
            rows = _final_rows(predictions, model, horizon=horizon, seed=seed)
            axis.plot(rows["target_date"], rows["y_pred"], color=_final_color(model),
                      linestyle=styles[model], linewidth=1.25, alpha=0.9,
                      label=_final_label(model))
        axis.set(title=f"Horizonte {horizon} días hábiles · {len(truth)} orígenes",
                 ylabel="Precio (USD/barril)", xlabel="Fecha objetivo de la predicción")
        _final_date_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=9)
    figure.suptitle(f"Precios predichos y observados en test · semilla {seed}\n"
                    "Orígenes 2024; el eje X corresponde a cada fecha objetivo, incluida la de 2025 a 252 días",
                    fontsize=14)
    _save(figure, output, "02_test_predictions_by_target_date")


def _final_interval_curves(predictions: pd.DataFrame, selected_model: str,
                           seed: int, output: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="row", sharey="row",
                                layout="constrained")
    for row, horizon in enumerate(HORIZONS):
        for column, model in enumerate((selected_model, "persistence")):
            axis = axes[row, column]
            rows = _final_rows(predictions, model, horizon=horizon, seed=seed)
            covered = rows["y_true"].between(rows["lower"], rows["upper"], inclusive="both")
            width = float((rows["upper"] - rows["lower"]).mean())
            coverage = float(covered.mean())
            axis.fill_between(rows["target_date"], rows["lower"], rows["upper"],
                              color=_final_color(model), alpha=0.18,
                              label="Intervalo nominal del 80 %")
            axis.plot(rows["target_date"], rows["y_true"], color="#111111", linewidth=1.3,
                      label="Precio observado")
            axis.plot(rows["target_date"], rows["y_pred"], color=_final_color(model),
                      linewidth=1.2, linestyle="--", label="Predicción")
            misses = rows.loc[~covered]
            axis.scatter(misses["target_date"], misses["y_true"], marker="x", color="#AA3377",
                         s=17, linewidths=1, label="Fuera del intervalo", zorder=4)
            axis.set(title=f"{_final_label(model)} · {horizon} días\n"
                           f"Cobertura {coverage:.1%} · amplitud media {width:.2f} USD/barril",
                     xlabel="Fecha objetivo", ylabel="Precio (USD/barril)")
            _final_date_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=9)
    figure.suptitle(f"Intervalos predictivos nominales del 80 % en test · semilla {seed}\n"
                    f"{_final_label(selected_model)} seleccionada antes del test; cuantiles congelados con calibración",
                    fontsize=14)
    _save(figure, output, "03_test_intervals_80")


def _final_signed_errors(predictions: pd.DataFrame, seed: int, output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), layout="constrained")
    styles = {"persistence": "--", "RNN": ":", "LSTM": "-", "GRU": "-."}
    for axis, horizon in zip(axes, HORIZONS):
        for model in FINAL_MODELS:
            rows = _final_rows(predictions, model, horizon=horizon, seed=seed)
            errors = rows["y_pred"] - rows["y_true"]
            axis.plot(rows["target_date"], errors, color=_final_color(model),
                      linestyle=styles[model], linewidth=1.2, alpha=0.9,
                      label=_final_label(model))
        axis.axhline(0, color="#222222", linewidth=0.9)
        reference = _final_rows(predictions, "persistence", horizon=horizon)
        changes = (reference["y_true"] - reference["reference_brent"]).abs()
        extreme = reference.loc[changes.idxmax()]
        axis.axvline(extreme["target_date"], color="#777777", linewidth=0.9, linestyle=":")
        axis.text(0.99, 0.95, f"Mayor |cambio observado|: {float(changes.max()):.2f} USD/barril",
                  transform=axis.transAxes, ha="right", va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.set(title=f"Horizonte {horizon} días hábiles", xlabel="Fecha objetivo",
                 ylabel="Error: predicho − observado\n(USD/barril)")
        _final_date_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=9)
    figure.suptitle(f"Errores temporales en test · semilla {seed}\n"
                    "Línea vertical: mayor cambio realizado de cada horizonte, solo como diagnóstico de volatilidad",
                    fontsize=14)
    _save(figure, output, "04_test_signed_errors")


def render_final_figures(run_dir: Path) -> dict:
    """Export four test figures without selecting or recalibrating any model.

    The protocol fixes the displayed neural model and operational seed before
    test. Predictions are aligned by target date, rather than the origin date.
    Existing exports and all input results remain untouched.
    """
    directory = Path(run_dir).resolve()
    output = directory / "figures"
    if output.exists():
        raise FileExistsError("Existing figure exports are immutable")
    names = ["experiment.json", "protocol.json", "metrics.csv", "predictions.csv",
             "interval_metrics.csv", "intervals.json"]
    paths = {name: directory / name for name in names}
    input_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in paths.values()}
    protocol = json.loads(paths["protocol.json"].read_text(encoding="utf-8"))
    selected_model = protocol["selected_model"]
    seed = protocol.get("selected_seed", 42)
    if selected_model not in ARCHITECTURES or seed != 42:
        raise ValueError("Final figures require the pre-test neural selection and operational seed 42")
    predictions = pd.read_csv(paths["predictions.csv"])
    metrics = pd.read_csv(paths["metrics.csv"])
    interval_metrics = pd.read_csv(paths["interval_metrics.csv"])
    if (not set(predictions["phase"]).issubset({"calibration", "test"})
            or not set(metrics["phase"]).issubset({"calibration", "test"})):
        raise ValueError("Final results must be separate from development")
    predictions = predictions.loc[predictions["phase"].eq("test")].copy()
    metrics = metrics.loc[metrics["phase"].eq("test")].copy()
    if predictions.empty or metrics.empty:
        raise ValueError("Final figures require completed test results")
    if "phase" in interval_metrics and not set(interval_metrics["phase"]).issubset({"calibration", "test"}):
        raise ValueError("Interval metrics contain an unrelated evaluation phase")
    for name in ("origin_date", "target_date"):
        predictions[name] = pd.to_datetime(predictions[name], errors="raise")
    if not predictions["origin_date"].dt.year.eq(2024).all():
        raise ValueError("Final test figures require the frozen 2024 origin period")
    if not predictions["target_date"].gt(predictions["origin_date"]).all():
        raise ValueError("Forecast targets must follow their origins")
    if predictions.duplicated(["model", "variant", "seed", "sample_id", "horizon"]).any():
        raise ValueError("Duplicate final prediction pairs")
    if not np.isclose(predictions["interval_level"].to_numpy(dtype=float), 0.8, rtol=0, atol=1e-12).all():
        raise ValueError("Final figures require nominal 80 percent intervals")
    finite = predictions[["y_true", "y_pred", "reference_brent", "lower", "upper"]].to_numpy(dtype=float)
    if not np.isfinite(finite).all() or predictions["lower"].gt(predictions["upper"]).any():
        raise ValueError("Final prices and interval limits must be finite and coherent")
    if not np.isfinite(metrics[["mae", "rmse", "mse"]].to_numpy(dtype=float)).all():
        raise ValueError("Final point metrics must be finite")
    for model in FINAL_MODELS:
        expected_seeds = {-1} if model == "persistence" else {42, 123, 2026}
        rows = _final_rows(predictions, model)
        if set(rows["seed"]) != expected_seeds or set(rows["horizon"]) != set(HORIZONS):
            raise ValueError("Missing frozen architecture, seed or horizon")
    output.mkdir()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18,
                         "axes.axisbelow": True, "svg.hashsalt": "oil-forecast-final-v1"}):
        _final_error_comparison(metrics, output)
        _final_prediction_curves(predictions, seed, output)
        _final_interval_curves(predictions, selected_model, seed, output)
        _final_signed_errors(predictions, seed, output)
    if input_hashes != {str(path.relative_to(ROOT)): _sha256(path) for path in paths.values()}:
        raise RuntimeError("Plotting changed its final-evaluation inputs")
    manifest = {"phase": "test", "figure_count": 4, "selected_model": selected_model,
                "selected_seed": seed, "time_axis": "target_date, not origin_date",
                "error_bars": "sample standard deviation between three seeds, not prediction intervals",
                "intervals": "nominal 80 percent; empirical test coverage, no exact temporal coverage guarantee",
                "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__, "interpreter": sys.executable},
                "input_sha256": input_hashes, "plot_code_sha256": _sha256(Path(__file__)),
                "output_sha256": {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=None)
    parser.add_argument("--mode", choices=("refinement", "final"), default="refinement")
    args = parser.parse_args()
    directory = args.run_dir or ROOT / "experiments" / (
        "cpu-final-evaluation-v1" if args.mode == "final" else "cpu-refinement-persistence-init-v1")
    renderer = render_final_figures if args.mode == "final" else render_refinement_figures
    print(json.dumps(renderer(directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
