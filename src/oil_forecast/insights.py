"""Deterministic explanations, sourced event points and price-only cost scenarios."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import numpy as np
import pandas as pd

LABELS = {"persistence": "Persistencia", "RNN": "RNN-D", "LSTM": "LSTM-B", "GRU": "GRU-C"}


def _number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _output(forecast: dict, horizon: int) -> dict:
    rows = [row for row in forecast["forecasts"] if row["horizon"] == horizon]
    if horizon not in (5, 63, 252) or len(rows) != 1:
        raise ValueError("Select one of the three saved direct horizons")
    row = rows[0]
    values = [forecast["reference_brent"], row["price"], row["change"], row["lower"], row["upper"]]
    if not np.isfinite(values).all() or not row["lower"] <= row["price"] <= row["upper"]:
        raise ValueError("Forecast prices and interval must be finite and coherent")
    if not np.isclose(row["price"] - forecast["reference_brent"], row["change"], rtol=1e-10, atol=1e-10):
        raise ValueError("Forecast change does not match its reference")
    return row


def explain_forecast(forecast: dict, horizon: int, metrics: pd.DataFrame,
                     latest_quote: dict | None = None) -> dict:
    """Use only the selected real output and its frozen seed-specific test scores."""
    row = _output(forecast, horizon)
    model = forecast["model"]
    if model not in LABELS:
        raise ValueError("Unknown frozen model")
    seed = -1 if model == "persistence" else 42
    selected = metrics.loc[metrics["phase"].eq("test") & metrics["block_id"].eq("all")
        & metrics["horizon"].eq(horizon)]
    score = selected.loc[selected["model"].eq(model) & selected["seed"].eq(seed)
                         & selected["variant"].eq(forecast["variant"])]
    baseline = selected.loc[selected["model"].eq("persistence") & selected["seed"].eq(-1)]
    if len(score) != 1 or len(baseline) != 1:
        raise ValueError("A unique frozen test score is required for the selected seed")
    mae, naive_mae = float(score["mae"].iloc[0]), float(baseline["mae"].iloc[0])
    if not np.isfinite([mae, naive_mae]).all() or min(mae, naive_mae) < 0:
        raise ValueError("Historical errors must be finite and non-negative")
    reference, price, change = float(forecast["reference_brent"]), float(row["price"]), float(row["change"])
    percentage = 100 * change / reference if reference != 0 else None
    direction = "estable" if abs(change) < 1e-9 else "subida" if change > 0 else "bajada"
    target = pd.Timestamp(row["target_date"]).strftime("%d/%m/%Y")
    percentage_text = f"{_number(percentage)} %" if percentage is not None else "porcentaje no definido con referencia cero"
    movement = "sin cambio" if direction == "estable" else f"una {direction} de {_number(abs(change))} USD/barril"
    verb = "estimó" if forecast.get("prediction_source") == "saved" else "estima"
    paragraphs = [f"A {horizon} días lunes–viernes ({target}), {LABELS[model]} {verb} {_number(price)} USD/barril: "
                  f"{movement} frente a la referencia de {_number(reference)} USD/barril ({percentage_text})."]
    # A current quote must never be presented as known at an archived origin.
    if latest_quote and forecast.get("prediction_source") != "saved":
        quote, observed = latest_quote.get("price"), latest_quote.get("observation_date")
        if quote is not None and observed is not None and np.isfinite(quote) and (
                not np.isclose(quote, reference, rtol=1e-12, atol=1e-12)
                or str(observed)[:10] != str(forecast.get("reference_observation_date"))[:10]):
            paragraphs[0] += (f" La última cotización observada es {_number(quote)} USD/barril, del "
                f"{pd.Timestamp(observed).strftime('%d/%m/%Y')}; difiere de la referencia admitida por los retardos de publicación.")
    paragraphs.append(f"El intervalo nominal del 80 % va de {_number(row['lower'])} a {_number(row['upper'])} USD/barril. "
                      "Representa incertidumbre estimada con errores de calibración; no garantiza que el precio futuro quede dentro.")
    difference = 100 * (mae / naive_mae - 1) if naive_mae > 0 else None
    history = f"En test, el error absoluto medio de {LABELS[model]} fue {_number(mae)} USD/barril"
    if model != "persistence":
        history += f" con semilla 42, frente a {_number(naive_mae)} de persistencia"
        if difference is not None and abs(difference) > 1e-9:
            history += (f": un error {_number(abs(difference))} % {'mayor' if difference > 0 else 'menor'}. "
                        + ("La red fue menos precisa en este horizonte." if difference > 0 else "La red fue más precisa en este horizonte del test; no asegura una ventaja futura."))
    paragraphs.append(history.rstrip(".") + ". El MAE es un error medio histórico, no una probabilidad de acierto.")
    limitation = "Cambios en la oferta, demanda, divisas o riesgos geopolíticos pueden invalidar la estimación."
    if horizon == 252:
        limitation += " A un año, los cambios de régimen y de ciclo económico merecen especial atención."
    if model != "persistence":
        limitation += (f" Los pesos siguen congelados en el reajuste lógico del "
                       f"{pd.Timestamp(forecast['model_refit_at']).strftime('%d/%m/%Y')}; actualizar datos no los reentrena.")
    else:
        limitation += " Persistencia conserva el precio de referencia y no anticipa cambios de tendencia."
    if forecast.get("prediction_source") == "saved":
        limitation += " Las métricas de test se muestran como revisión posterior; no se conocían en el origen archivado."
    paragraphs.append(limitation)
    return {"paragraphs": paragraphs, "direction": direction, "change": change,
            "change_pct": percentage, "mae": mae, "persistence_mae": naive_mae,
            "mae_difference_pct": difference, "seed": seed}


def comparison_conclusion(summary: pd.DataFrame, metric: str, phase: str,
                          horizon: int, seed_mode: str) -> str:
    """Describe existing, comparable scores without changing model selection."""
    metrics = {"mae": ("MAE", "USD/barril"), "mse": ("MSE", "USD²/barril²"),
               "rmse": ("RMSE", "USD/barril"), "r2": ("R²", "sin unidad")}
    phases = {"development": "desarrollo", "calibration": "calibración", "test": "test"}
    if metric not in metrics or phase not in phases or seed_mode not in {"mean", "42"}:
        raise ValueError("Select a known comparison metric, phase and seed scope")
    if isinstance(horizon, bool) or horizon not in (5, 63, 252):
        raise ValueError("Comparison conclusions require a frozen forecast horizon")
    required = {"model", metric, "mae", "skill_mae_pct", "n", "n_seeds"}
    if not required.issubset(summary) or summary["model"].duplicated().any() or set(summary["model"]) != set(LABELS):
        raise ValueError("Comparison conclusions require each of the four frozen models once")
    values = pd.to_numeric(summary[metric], errors="raise").to_numpy(dtype=float)
    maes = pd.to_numeric(summary["mae"], errors="raise").to_numpy(dtype=float)
    counts = pd.to_numeric(summary["n"], errors="raise").to_numpy(dtype=float)
    seeds = pd.to_numeric(summary["n_seeds"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(np.r_[values, maes, counts, seeds]).all() or (maes < 0).any()
            or (metric != "r2" and (values < 0).any()) or (counts <= 0).any()
            or (counts != np.floor(counts)).any() or np.unique(counts).size != 1):
        raise ValueError("Comparison scores must be finite and use identical positive pair counts")
    expected_seeds = np.array([1 if model == "persistence" or seed_mode == "42" else 3
                               for model in summary["model"]])
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError("Comparison scores do not match the requested seed scope")
    best = float(values.max() if metric == "r2" else values.min())
    winners = summary.loc[np.isclose(values, best, rtol=1e-12, atol=1e-12), "model"]
    labels = [LABELS[model] for model in LABELS if model in set(winners)]
    names = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " y " + labels[-1]
    title, unit = metrics[metric]
    scope = "la media de las tres semillas neuronales" if seed_mode == "mean" else "la semilla operativa 42 de las redes"
    ranking = (f"{names} {'presenta' if len(labels) == 1 else 'comparten'} "
               f"el {'mayor' if metric == 'r2' else 'menor'} {title}: {_number(best, 3)} {unit}")
    parts = [f"En {phases[phase]}, a {horizon} días, con {scope} y {int(counts[0])} orígenes por modelo y semilla, {ranking}."]
    indexed = summary.set_index("model")
    naive_mae, lstm_mae = float(indexed.loc["persistence", "mae"]), float(indexed.loc["LSTM", "mae"])
    saved_skill = indexed.loc["LSTM", "skill_mae_pct"]
    skill = float(saved_skill) if pd.notna(saved_skill) else np.nan
    equal = bool(np.isclose(lstm_mae, naive_mae, rtol=1e-12, atol=1e-12))
    if equal:
        if naive_mae > 0 and (not np.isfinite(skill) or not np.isclose(skill, 0., rtol=1e-12, atol=1e-12)):
            raise ValueError("The saved MAE improvement contradicts equal MAEs")
        parts.append(f"LSTM-B y persistencia registran el mismo MAE: {_number(lstm_mae)} USD/barril.")
    else:
        statement = f"El MAE de LSTM-B es {_number(lstm_mae)} frente a {_number(naive_mae)} USD/barril de persistencia"
        if naive_mae > 0:
            if not np.isfinite(skill) or np.sign(skill) != np.sign(naive_mae - lstm_mae):
                raise ValueError("The saved MAE improvement contradicts the comparison")
            statement += f": un error medio {_number(abs(skill))} % {'mayor' if lstm_mae > naive_mae else 'menor'}"
        else:
            statement += "; la mejora porcentual no está definida con MAE ingenuo cero"
        parts.append(statement + ".")
    if phase == "development":
        parts.append("Son resultados de desarrollo utilizados en la selección; no estiman por sí solos la generalización.")
    elif phase == "calibration":
        parts.append("La comparación en calibración es descriptiva y no demuestra una ventaja futura.")
    parts.append("LSTM-B/42 conserva su selección previa a test; el ranking mostrado no cambia el modelo ni la semilla.")
    return " ".join(parts)


def cost_scenarios(forecast: dict, horizon: int, barrels: float) -> pd.DataFrame:
    """Multiply actual prices by fixed volume; round only the final USD amount."""
    row = _output(forecast, horizon)
    try:
        volume = Decimal(str(barrels))
    except InvalidOperation as error:
        raise ValueError("Barrels must be a finite, non-negative quantity") from error
    if isinstance(barrels, bool) or not volume.is_finite() or not 0 <= volume <= Decimal("1000000000"):
        raise ValueError("Barrels must be between zero and one billion")
    prices = {"Referencia": forecast["reference_brent"], "Persistencia": row.get("persistence_price", forecast["reference_brent"]),
              "Predicción": row["price"], "Límite inferior · 80 %": row["lower"], "Límite superior · 80 %": row["upper"]}
    return pd.DataFrame([{"Escenario": name, "USD/barril": float(price),
                          "Coste (USD)": float((volume * Decimal(str(price))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))}
                         for name, price in prices.items()])


def event_points(history: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Anchor documented episodes to original observed local prices, never forecasts."""
    rows = history.copy()
    rows["observation_date"] = pd.to_datetime(rows["observation_date"], errors="raise")
    if "observed" in rows:
        rows = rows.loc[rows["observed"].isin([True, "True", "true", 1])]
    rows = rows.loc[rows["y_brent"].notna()].sort_values("observation_date")
    points = []
    for event in events:
        part = rows.loc[rows["observation_date"].between(event["period_start"], event["period_end"])]
        if part.empty:
            continue
        rule = event["marker_rule"]
        if rule not in {"maximum", "minimum", "latest"} or not event.get("sources"):
            raise ValueError("Events need a documented source and a supported local marker rule")
        if rule != "latest" and (rows["observation_date"].min() > pd.Timestamp(event["period_start"])
                                  or rows["observation_date"].max() < pd.Timestamp(event["period_end"])):
            continue
        if not np.isfinite(part["y_brent"].to_numpy(dtype=float)).all():
            raise ValueError("Event prices must be observed and finite")
        point = part.iloc[-1] if rule == "latest" else part.loc[part["y_brent"].idxmax() if rule == "maximum" else part["y_brent"].idxmin()]
        points.append({**event, "observation_date": point["observation_date"], "price": float(point["y_brent"]),
                       "marker_basis": {"maximum": "Máximo observado del periodo", "minimum": "Mínimo observado del periodo", "latest": "Última observación del periodo"}[rule]})
    return pd.DataFrame(points)
