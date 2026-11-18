"""Five presentation views over verified data; no training or acquisition here."""

from __future__ import annotations

from html import escape
import re
import numpy as np
import pandas as pd
import streamlit as st

from .app_charts import (history_chart, forecast_chart, metrics_chart, loss_chart,
                         prediction_chart, errors_chart, coverage_chart)
from .app_data import (aggregate_metrics, context_indicators, filter_history, forecast_for_model,
                       latest_market, stored_forecast, VARIANTS)
from .insights import comparison_conclusion, cost_scenarios, event_points, explain_forecast
from .academic import detail_cases, error_periods

LABELS = {"persistence": "Persistencia", "RNN": "RNN-D", "LSTM": "LSTM-B", "GRU": "GRU-C"}
HORIZONS = {5: "Una semana · 5 días", 63: "Tres meses · 63 días", 252: "Un año · 252 días"}
PHASES = {"test": "Test · 2024", "calibration": "Calibración · 2021–2022", "development": "Desarrollo · 2015/17/19"}
EXPERIMENTS = {"final": "Evaluación final", "refinement": "Mejoras · inicio y pérdida", "optimization": "Optimización · validación interior", "smoke": "Prueba inicial · 3 épocas"}
VARIANT_NAMES = {"baseline": "Sin entrenamiento", "smoke": "Inicial", "A": "Configuración A", "B": "Configuración B", "C": "C · Inicio en persistencia", "D": "D · Pérdida robusta"}
METRICS = {"MAE": "mae", "RMSE": "rmse", "MSE": "mse", "R²": "r2"}
ARCHITECTURES = {"LSTM": "LSTM", "RNN": "RNN", "GRU": "GRU", "persistence": "Persistencia"}


def number(value, decimals=2):
    if value is None or not np.isfinite(float(value)):
        return "No disponible"
    return f"{float(value):,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def date(value):
    return "No disponible" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%d/%m/%Y")


def timestamp(value):
    if value is None or pd.isna(value):
        return "No disponible"
    moment = pd.Timestamp(value)
    moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment
    return moment.tz_convert("Europe/Madrid").strftime("%d/%m/%Y · %H:%M %Z")


def money(value):
    return "No disponible" if value is None or pd.isna(value) else f"${number(value)}"


def semester(value):
    match = re.search(r"(\d{4})H([12])$", str(value))
    return f"{'Enero–junio' if match[2] == '1' else 'Julio–diciembre'} de {match[1]}" if match else str(value)


def status(items):
    st.html('<div class="status-strip">' + ''.join(f'<span>{escape(label)} <b>{escape(str(value))}</b></span>' for label, value in items) + '</div>')


def ordered(summary):
    return summary.set_index("model").reindex(list(LABELS)).dropna(subset=["mae"]).reset_index()


def score_table(summary, seed_mode):
    """Present saved scores compactly, highlighting their numeric extrema."""
    table = pd.DataFrame({"Modelo": summary["model"].map(LABELS)})
    displayed = [("mae", "MAE"), ("rmse", "RMSE"), ("mse", "MSE")]
    if "r2" in summary and summary["r2"].notna().any():
        displayed.append(("r2", "R²"))
    for metric, title in displayed:
        table[title] = summary[metric].map(lambda value: number(value, 3))
        if metric != "mse" and seed_mode == "mean":
            table[title] += [f" ± {number(row[f'{metric}_std'], 3)}" if row["n_seeds"] > 1 else "" for row in summary.to_dict("records")]
    table["Mejora MAE (%)"] = summary["skill_mae_pct"].map(lambda value: number(value, 2))
    winners = {title: set(summary.index[np.isclose(summary[metric], summary[metric].max() if metric == "r2" else summary[metric].min(), rtol=1e-12, atol=1e-12)]) for metric, title in displayed}
    winners["Mejora MAE (%)"] = set(summary.index[np.isclose(summary["skill_mae_pct"], summary["skill_mae_pct"].max())])
    style = table.style.apply(lambda col: ["background-color:#194139;color:#C8F5E8;font-weight:600" if idx in winners.get(col.name, set()) else "" for idx in table.index])
    st.dataframe(style, hide_index=True, width="stretch", row_height=40)


def kpi(label, value, foot="", tone=""):
    st.html(f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value {tone}">{escape(str(value))}</div><div class="kpi-foot">{escape(str(foot))}</div></div>')


def chart(figure, key):
    st.plotly_chart(figure, theme=None, width="stretch", key=key, config={"displaylogo": False})


def heading(title, subtitle):
    st.title(title)
    st.html(f'<div class="intro">{escape(subtitle)}</div>')


def overview(data, services):
    heading("Visión general del mercado", "La evolución del Brent y las estimaciones del proyecto, con sus fechas y límites a la vista.")
    latest = latest_market(data["history"], data["protocol"])
    forecast = forecast_for_model(services, data["window"], "persistence", data["protocol"]["data_updated_at"])
    neural = forecast_for_model(services, data["window"], "LSTM", data["protocol"]["data_updated_at"])
    status([("Última adquisición válida", timestamp(latest["data_updated_at"])), ("Origen lógico de las predicciones", date(forecast["origin_date"])), ("Unidad", "USD por barril")])
    a, b, c, d = st.columns(4)
    with a: kpi("ÚLTIMO PRECIO OBSERVADO", money(latest["price"]), f"Cotización del {date(latest['observation_date'])}")
    with b: kpi("CAMBIO ENTRE ÚLTIMAS COTIZACIONES", f"{number(latest['change_pct'])} %", f"{money(latest['change'])} por barril", "positive" if (latest["change"] or 0) >= 0 else "negative")
    with c: kpi("REFERENCIA PARA PREDECIR", money(forecast["reference_brent"]), f"Cotización del {date(forecast['reference_observation_date'])}")
    with d: kpi("ESTADO DEL MODELO", "Histórico" if neural.get("model_is_past_scheduled_refit") else "Congelado", "Julio de 2024 · revisión pendiente" if neural.get("model_is_past_scheduled_refit") else "Reajuste lógico: julio de 2024", "gold")
    st.caption("El histórico observado es contexto de la copia descargada. La referencia para predecir respeta los retardos asumidos y puede ser anterior a la última cotización. Las observaciones recientes se usan en inferencia; actualizar datos conserva los pesos históricos.")
    event_config = data.get("economic_events", {})
    points = event_points(data["history"], event_config.get("events", []))
    market, outlook = st.columns([2.5, 1], gap="large")
    with market:
        st.subheader("Histórico del Brent")
        period = st.radio("Periodo", ["1 mes", "3 meses", "1 año", "5 años", "Máximo"], index=2,
                          horizontal=True, label_visibility="collapsed", key="market_period")
        periods = dict(zip(["1 mes", "3 meses", "1 año", "5 años", "Máximo"], ["1M", "3M", "1Y", "5Y", "MAX"]))
        show_events = st.toggle("Mostrar acontecimientos económicos", value=True, key="market_events")
        chart(history_chart(filter_history(data["history"], periods[period]), context_events=show_events, events=points), "market_history")
    with outlook:
        st.subheader("Tres horizontes")
        st.caption("Referencia principal: persistencia. Red seleccionada antes de test: LSTM-B/42.")
        cards = []
        for row, network in zip(forecast["forecasts"], neural["forecasts"]):
            cards.append(f'<div class="forecast-card"><div class="horizon">{escape(HORIZONS[row["horizon"]])}</div>'
                         f'<div class="forecast-value"><span class="price">{money(row["price"])}</span><span>{date(row["target_date"])}</span></div><div class="detail">'
                         f'Persistencia · 80 % nominal: {money(row["lower"])} – {money(row["upper"])}</div>'
                         f'<div class="neural">LSTM-B/42: {money(network["price"])}</div></div>')
        st.html('<div class="forecast-stack">' + ''.join(cards) + '</div>')
    if show_events and not points.empty:
        with st.expander("Ciclo económico y acontecimientos", expanded=True):
            events = {event["id"]: event for event in event_config["events"]}
            identifier = st.selectbox("Episodio económico", list(events), index=len(events) - 1,
                                      format_func=lambda key: events[key]["title"], key="economic_episode")
            event = events[identifier]
            point = points.loc[points["id"].eq(identifier)]
            st.write(f"**Hecho documentado.** {event['documented_fact']}")
            st.write(f"**Lectura económica.** {event['economic_interpretation']}")
            if not point.empty:
                selected = point.iloc[0]
                st.caption(f"{selected['marker_basis']}: {money(selected['price'])}/barril el {date(selected['observation_date'])}. La fecha del marcador corresponde a una cotización real; no equivale necesariamente a la fecha causal del acontecimiento.")
            for source in event["sources"]:
                st.markdown(f"[{source['title']}]({source['url']}) · publicación {date(source.get('published_at'))}")
            st.caption("Contexto retrospectivo, sin atribuir cada variación a una única causa. Usa Máximo para ver los episodios antiguos. Revisión de fuentes: " + date(event_config.get("reviewed_at")))
    st.subheader("Contexto económico disponible")
    context = context_indicators(data["window"])
    context.loc[context["variable"].eq("opep_orb"), "value"] = np.nan
    context.loc[context["variable"].eq("opep_orb"), "observation_date"] = None
    context.loc[context["variable"].eq("opep_orb"), ["value", "observation_date", "available_at"]] = [np.nan, None, None]
    names = {"wti": "WTI", "gas_natural": "Gas natural", "oro": "Oro", "cobre": "Cobre", "indice_dolar": "Índice dólar", "vix": "VIX", "tipo_bce": "Tipo BCE", "usd_por_eur": "USD por EUR", "cny_por_usd": "CNY por USD", "opep_orb": "Cesta OPEP"}
    context["Indicador"] = context["variable"].map(names)
    context["Valor"] = context["value"].map(number)
    context["Observación"] = context["observation_date"].map(date)
    st.dataframe(context[["Indicador", "Valor", "unit", "Observación"]].rename(columns={"unit": "Unidad"}), hide_index=True, width="stretch")
    st.caption("Valores admitidos por la política temporal. La cesta OPEP no se incorpora: el histórico diario y sus condiciones de redistribución siguen sin verificarse; las adquisiciones anteriores registraron HTTP 403.")


def prediction(data, services):
    heading("Predicción del precio", "Una estimación para la fecha elegida, comparada con mantener el último Brent conocido.")
    a, b, c = st.columns([1.1, 1.2, 1.5])
    with a: model = st.selectbox("Modelo", list(LABELS), format_func=LABELS.get, key="forecast_model")
    with b: horizon = st.selectbox("Horizonte", [5, 63, 252], format_func=HORIZONS.get, key="forecast_horizon")
    with c: source = st.radio("Datos de la predicción", ["Ventana disponible", "Archivo de evaluación"], format_func={"Ventana disponible": "Últimos datos", "Archivo de evaluación": "Histórico evaluado"}.get, key="forecast_source", horizontal=True)
    if source == "Ventana disponible":
        available = str(data["window"]["decision_date"].iloc[-1])[:10]
        origin_column, context_column = st.columns([1, 2])
        with origin_column:
            st.selectbox("Fecha de origen disponible", [available], key="live_origin", format_func=date)
        with context_column:
            st.caption(f"Datos adquiridos el {timestamp(data['protocol']['data_updated_at'])}. El origen es un corte lógico; la consulta se realiza después de adquirir los datos.")
        forecast = forecast_for_model(services, data["window"], model, data["protocol"]["data_updated_at"])
    else:
        saved = data["evaluation"].get("predictions", pd.DataFrame())
        if saved.empty:
            st.info("El archivo de evaluación no está instalado. La ventana disponible sigue permitiendo inferencia real.")
            return
        saved = saved.loc[saved["experiment"].eq("final")]
        phase = st.selectbox("Periodo guardado", ["test", "calibration"], format_func=lambda value: "Test · orígenes 2024" if value == "test" else "Calibración · 2021–2022", key="saved_phase")
        saved = saved.loc[saved["phase"].eq(phase)]
        dates = sorted(saved.loc[saved["model"].eq(model), "origin_date"].unique(), reverse=True)
        origin = st.selectbox("Fecha de origen guardada", dates, format_func=date, key="saved_origin")
        forecast = stored_forecast(saved, model, horizon, origin)
        row = forecast["forecasts"][0]
        baseline = stored_forecast(saved, "persistence", horizon, origin)["forecasts"][0]
        row.update(persistence_price=baseline["price"], persistence_lower=baseline["lower"], persistence_upper=baseline["upper"])
        st.caption("Predicción guardada del backtest de su propio periodo: se consulta el resultado original, sin usar un reajuste posterior a ese origen.")
        if phase == "calibration":
            st.caption("Las bandas de calibración son retrospectivas: los cuantiles se congelaron antes de test, después de estos orígenes; no son intervalos emitidos entonces.")
    row = next(item for item in forecast["forecasts"] if item["horizon"] == horizon)
    change_pct = 100 * row["change"] / forecast["reference_brent"] if forecast["reference_brent"] else None
    a, b, c, d = st.columns(4)
    with a: kpi("REFERENCIA DISPONIBLE", money(forecast["reference_brent"]), f"Cotización del {date(forecast.get('reference_observation_date'))}")
    with b: kpi("PRECIO ESTIMADO", money(row["price"]), f"Para el {date(row['target_date'])}", model.lower())
    with c: kpi("CAMBIO FRENTE A LA REFERENCIA", money(row["change"]), f"{number(change_pct)} %", "positive" if row["change"] >= 0 else "negative")
    with d: kpi("INTERVALO NOMINAL DEL 80 %", f"{money(row['lower'])} – {money(row['upper'])}", "Límite inferior – límite superior", "range")
    status([("Modelo", LABELS[model]), ("Origen lógico", date(forecast["origin_date"])), ("Fecha objetivo", date(row["target_date"])), ("Reajuste lógico", "No entrena" if model == "persistence" else date(forecast["model_refit_at"]))])
    archived = forecast.get("prediction_source") == "saved"
    quote = None if archived else latest_market(filter_history(data["history"], anchor=forecast["origin_date"]), data["protocol"])
    if quote is not None:
        st.caption(f"Último Brent observado en esta copia: {money(quote['price'])}/barril, del {date(quote['observation_date'])}. Es contexto; la referencia admisible del modelo es la mostrada arriba.")
    with st.expander("Fechas y entradas admitidas", expanded=False):
        status([("Decisión lógica", timestamp(forecast["decision_at"])), ("Referencia observada", date(forecast.get("reference_observation_date"))), ("Referencia admisible desde", timestamp(forecast.get("reference_available_at")))])
        st.caption("Calendario lunes–viernes, sin comprimir festivos. La disponibilidad se establece con retardos asumidos; el origen lógico no certifica una emisión real a esa hora ni una versión histórica de la fuente.")
        if not archived:
            inputs = context_indicators(data["window"], include_brent=True)
            inputs = inputs.loc[inputs["variable"].ne("opep_orb")].copy()
            inputs["Observación"] = inputs["observation_date"].map(date)
            inputs["Admisible"] = inputs["available_at"].map(date)
            st.dataframe(inputs[["variable", "value", "Observación", "Admisible"]].rename(columns={"variable": "Serie", "value": "Valor"}), hide_index=True, width="stretch")
            st.caption("Última fila de la ventana utilizada: diez series, con sus fechas de observación y disponibilidad bajo la política. El preprocesamiento conserva los transformadores de entrenamiento.")
    selected_network = forecast if model == "LSTM" else (stored_forecast(saved, "LSTM", horizon, origin) if archived else forecast_for_model(services, data["window"], "LSTM", data["protocol"]["data_updated_at"]))
    selected_row = next(item for item in selected_network["forecasts"] if item["horizon"] == horizon)
    st.html(f'<div class="rule">Mismo origen y objetivo · <b>Persistencia: {money(row["persistence_price"])}</b> · <b>LSTM-B/42: {money(selected_row["price"])}</b> USD/barril. La selección previa de la red no implica que supere a persistencia.</div>')
    if selected_network.get("model_is_past_scheduled_refit"):
        st.info(f"Pesos del reajuste lógico del {date(selected_network['model_refit_at'])}; la cadencia semestral de revisión ha vencido. Actualizar observaciones no actualiza pesos, escaladores ni cuantiles.")
    chart(forecast_chart(data["history"], forecast, horizon), "forecast_price")
    st.caption("Cada marcador futuro es una salida directa para el horizonte elegido. El histórico observado aporta contexto; la red estima un cambio respecto a la referencia, sin generar una trayectoria diaria intermedia.")
    st.subheader("Cómo leer el resultado")
    explanation = explain_forecast(forecast, horizon, data["metrics"], quote)
    for paragraph in explanation["paragraphs"][:3]:
        st.write(paragraph)
    st.caption(explanation["paragraphs"][3])
    with st.expander("Planificación de compras y escenarios de costes", expanded=False):
        st.write("Compara el presupuesto de compras con el precio de referencia y el estimado; revisa la exposición energética y escenarios de coste. Las bandas ayudan a dimensionar incertidumbre, sin decidir automáticamente compras, coberturas o inversiones.")
        barrels = st.number_input("Volumen de petróleo (barriles)", min_value=0., max_value=1_000_000_000., value=1000., step=100., format="%.2f", key="business_barrels")
        scenarios = cost_scenarios(forecast, horizon, barrels)
        st.dataframe(scenarios, hide_index=True, width="stretch", row_height=38,
                     column_config={"USD/barril": st.column_config.NumberColumn(format="%.2f"), "Coste (USD)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Volumen fijo × precio de cada escenario; redondeo final a céntimos. No incluye primas, transporte, almacenamiento, impuestos ni conversión de divisas. Los límites no son un coste garantizado ni una recomendación financiera.")


def comparison(data):
    heading("Comparación de modelos", "La misma evaluación para todas las arquitecturas. Menor error y mayor R² indican mejor ajuste.")
    evidence = data["evaluation"].get("metrics", pd.DataFrame())
    phases = ["test", "calibration"] + (["development"] if not evidence.empty else [])
    a, b, c = st.columns(3)
    with a: phase = st.selectbox("Conjunto", phases, format_func=PHASES.get, key="compare_phase")
    with b: horizon = st.selectbox("Horizonte", [5, 63, 252], format_func=HORIZONS.get, key="compare_horizon")
    with c: mode = st.radio("Semillas", ["Media de tres", "Operativa 42"], horizontal=True, key="compare_seeds")
    source = evidence.loc[evidence["experiment"].eq("refinement")] if phase == "development" else data["metrics"]
    seed_mode = "mean" if mode == "Media de tres" else "42"
    summary = ordered(aggregate_metrics(source, phase, horizon, seed_mode))
    choices = list(METRICS) if summary["r2"].notna().all() else ["MAE", "RMSE", "MSE"]
    metric = st.radio("Métrica del gráfico", choices, horizontal=True, key="compare_metric")
    metric_key = METRICS[metric]
    winner = summary.loc[summary[metric_key].idxmax() if metric == "R²" else summary[metric_key].idxmin()]
    unit = "sin unidad" if metric == "R²" else "USD²/barril²" if metric == "MSE" else "USD/barril"
    st.html(f'<div class="rule">{"Mayor" if metric == "R²" else "Menor"} {metric}: <b>{escape(LABELS[winner["model"]])}</b> · {number(winner[metric_key], 3)} {unit}, a {horizon} días en {escape(PHASES[phase].split(" · ")[0].lower())}.</div>')
    if data.get("academic_available"):
        st.caption("R² = 1 − suma de errores al cuadrado / variación de los valores reales. Si es negativo, el error cuadrático del modelo es mayor que al usar la media del conjunto evaluado: una referencia retrospectiva, distinta de persistencia. No es una probabilidad de acierto.")
    else:
        st.info("R² requiere el suplemento académico incluido con el código. Las métricas originales continúan disponibles.")
    detail, horizons = st.tabs(["Horizonte seleccionado", "Los tres horizontes"])
    with detail:
        chart(metrics_chart(summary, metric_key), "model_comparison")
        score_table(summary, seed_mode)
        st.caption(f"{int(summary['n'].iloc[0])} orígenes por modelo y semilla. MAE/RMSE: USD/barril; MSE: USD²/barril²; R²: sin unidad. Las celdas destacadas indican los mejores valores; una mejora MAE negativa indica peor resultado que persistencia.")
        st.caption("± indica desviación entre las tres semillas de las redes, no un intervalo de predicción. Persistencia tiene un único resultado y no entrena." if seed_mode == "mean" else "Resultados de la semilla operativa 42; persistencia no utiliza semilla. No se elige la semilla más favorable después de test.")
        if seed_mode == "mean":
            st.caption("Se promedian las métricas calculadas por separado para cada semilla; no se combinan sus predicciones en un nuevo modelo ni se triplica el número de observaciones.")
    with horizons:
        st.subheader(f"{metric} en los tres horizontes")
        across = pd.DataFrame({"Modelo": list(LABELS.values())})
        for days, title in HORIZONS.items():
            values = ordered(aggregate_metrics(source, phase, days, seed_mode))
            across[title] = values[metric_key].to_numpy()
        numeric = list(HORIZONS.values())
        style = across.style.format({col: lambda value: number(value, 3) for col in numeric})
        style = style.highlight_max(subset=numeric, color="#194139") if metric == "R²" else style.highlight_min(subset=numeric, color="#194139")
        columns = {title: st.column_config.NumberColumn(f"{days} días", width=76, format="%.3f") for days, title in HORIZONS.items()}
        columns["Modelo"] = st.column_config.TextColumn("Modelo", width=100)
        st.dataframe(style, hide_index=True, width="stretch", row_height=42, column_config=columns)
        st.caption(f"{unit}. Cada columna conserva sus propias fechas objetivo; el resaltado marca {'el mayor R²' if metric == 'R²' else 'el menor error'} de ese horizonte. La variación de los valores reales cambia entre horizontes y conjuntos.")
    table = summary.copy()
    table["Modelo"] = table["model"].map(LABELS)
    with st.expander("Parámetros y tiempo de entrenamiento", expanded=False):
        st.dataframe(table[["Modelo", "parameters", "training_seconds"]].rename(columns={"parameters": "Parámetros entrenables", "training_seconds": "Optimización por semilla (s)"}).round(3), hide_index=True, width="stretch")
        st.caption("Tiempo de optimización medido por semilla en este conjunto, compartido por los tres horizontes. No se suma tres veces el mismo entrenamiento.")
    st.markdown("**Lectura de esta selección.** " + comparison_conclusion(summary, metric_key, phase, horizon, seed_mode))


def evaluation(data):
    heading("Evaluación y resultados", "Del experimento inicial a la evaluación final: aprendizaje, predicciones y errores sobre datos reales.")
    evidence = data["evaluation"]
    if not evidence or "history" not in evidence:
        st.info("Instala el paquete de evidencia para consultar curvas y predicciones históricas. La comparación final continúa disponible.")
        return
    history, metrics, predictions = evidence["history"], evidence["metrics"], evidence["predictions"]
    a, b, c, d = st.columns([1.3, 1, .8, 1.2])
    with a: experiment = st.selectbox("Experimento", list(EXPERIMENTS), format_func=EXPERIMENTS.get, key="eval_experiment")
    with b: model = st.selectbox("Arquitectura", list(ARCHITECTURES), format_func=ARCHITECTURES.get, key="eval_model")
    available = metrics.loc[metrics["experiment"].eq(experiment) & metrics["model"].eq(model)]
    variants = sorted(available["variant"].dropna().unique())
    with c: variant = st.selectbox("Configuración", variants, format_func=lambda value: VARIANT_NAMES.get(value, value), index=variants.index(VARIANTS[model]) if VARIANTS[model] in variants else 0, key="eval_variant")
    with d: horizon = st.selectbox("Horizonte", [5, 63, 252], format_func=HORIZONS.get, key="eval_horizon")
    available = available.loc[available["variant"].eq(variant)]
    phases = sorted(available["phase"].unique(), reverse=True)
    phase_column, block_column = st.columns([1, 2])
    with phase_column:
        phase = st.selectbox("Conjunto del experimento", phases, format_func=PHASES.get, key="eval_phase")
    blocks = sorted(available.loc[available["phase"].eq(phase) & ~available["block_id"].eq("all"), "block_id"].unique())
    preferred = "development_2019H1" if phase == "development" else blocks[0]
    def semester_label(value):
        match = re.search(r"(\d{4})H([12])$", value)
        return f"{'Enero–junio' if match[2] == '1' else 'Julio–diciembre'} de {match[1]}" if match else value
    with block_column:
        block = st.selectbox("Semestre evaluado", blocks, format_func=semester_label, index=blocks.index(preferred) if preferred in blocks else 0, key="eval_block")
    st.caption("Ejemplos con semilla 42, fijada previamente. Se muestra por defecto enero–junio de 2019 en desarrollo y el primer semestre del conjunto final; la elección no depende del error obtenido.")
    if experiment == "final" and phase == "test" and model == "LSTM" and variant == "B" and horizon == 252:
        full = available.loc[available["phase"].eq("test") & available["block_id"].eq("all") & available["seed"].eq(42) & available["horizon"].eq(252)].iloc[0]
        naive = metrics.loc[metrics["experiment"].eq("final") & metrics["phase"].eq("test") & metrics["block_id"].eq("all") & metrics["model"].eq("persistence") & metrics["horizon"].eq(252)].iloc[0]
        st.info(f"LSTM-B/42 a un año, test completo ({int(full['n'])} orígenes): MAE {number(full['mae'], 3)} frente a {number(naive['mae'], 3)} USD/barril de persistencia; sesgo {number(full['bias'], 3)} USD/barril" + (f" y R² {number(full['r2'], 3)}." if "r2" in full else ".") + " La red sobreestima y es menos precisa. Las gráficas y métricas del semestre conservan su ámbito propio.")
    selected_history = history.loc[history["experiment"].eq(experiment)]
    phase_saved = predictions.loc[predictions["experiment"].eq(experiment) & predictions["model"].eq(model)
        & predictions["variant"].eq(variant) & predictions["phase"].eq(phase) & predictions["horizon"].eq(horizon)
        & predictions["seed"].eq(-1 if model == "persistence" else 42)]
    saved = phase_saved.loc[phase_saved["block_id"].eq(block)]
    curve = selected_history.loc[selected_history["model"].eq(model) & selected_history["variant"].eq(variant)
        & selected_history["seed"].eq(42) & selected_history["block_id"].eq(block)]
    if not curve.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Aprendizaje por época")
            chart(loss_chart(selected_history, model, variant, 42, block), "training_losses")
            if experiment == "final":
                st.caption("MSE normalizado. Las épocas finales se fijaron antes de test; este ajuste no utiliza validación ni Early Stopping.")
            elif experiment == "smoke":
                st.caption("MSE normalizado. Prueba inicial de tres épocas; la curva de validación corresponde al bloque exterior de desarrollo, sin selección de época.")
            else:
                st.caption("MSE normalizado de entrenamiento y validación interior. Early Stopping conserva la época de menor error interior entre las épocas optimizadas; la época 0 solo se registró como diagnóstico.")
            st.caption("Época 0: evaluación inicial, antes de actualizar pesos. Cada época posterior representa una pasada de optimización por entrenamiento.")
            if model == "RNN" and variant == "D":
                st.caption("RNN-D optimiza SmoothL1; la curva muestra MSE normalizado de diagnóstico para mantener una escala de comparación común.")
    else:
        st.info("Persistencia mantiene el último precio conocido: no necesita entrenamiento ni tiene una curva de pérdida." if model == "persistence" else "Este experimento no conserva una curva de aprendizaje para la selección actual.")
        right = st.container()
    with right:
        st.subheader("Predicción frente a observación")
        if not saved.empty:
            chart(prediction_chart(saved), "saved_predictions")
        else:
            st.info("No hay predicciones archivadas para esta selección.")
        st.caption("Eje temporal: fecha objetivo, no fecha de origen.")
        if phase == "calibration":
            st.caption("Bandas retrospectivas: los cuantiles se calcularon después de estos orígenes de calibración y se congelaron antes de test. No son intervalos emitidos en aquellas fechas.")
    selected_scores = available.loc[available["phase"].eq(phase) & available["block_id"].eq(block)
                                    & available["horizon"].eq(horizon) & available["seed"].eq(-1 if model == "persistence" else 42)]
    if not selected_scores.empty:
        st.subheader("Métricas del semestre seleccionado")
        columns = {"n": "Orígenes", "mae": "MAE", "mse": "MSE", "rmse": "RMSE"}
        if "r2" in selected_scores:
            columns["r2"] = "R²"
        st.dataframe(selected_scores[list(columns)].rename(columns=columns).round(4), hide_index=True, width="stretch")
        st.caption("MAE/RMSE en USD/barril; MSE en USD²/barril²; R² sin unidad, mayor es mejor y puede ser negativo. Se calcula para el semestre, modelo y horizonte seleccionados, sin mezclar conjuntos.")
    st.subheader("Evolución del error")
    if not saved.empty:
        chart(errors_chart(saved), "saved_errors")
        error = saved["y_pred"] - saved["y_true"]
        st.caption(f"{len(saved)} orígenes · Sesgo medio ${number(error.mean())}/barril · Error máximo absoluto ${number(error.abs().max())}/barril.")
    if not phase_saved.empty:
        with st.expander("Periodos y casos con mayor o menor error", expanded=False):
            st.write("Un error positivo indica sobreestimación; uno negativo, subestimación. Los casos cercanos al valor real se muestran sin inventar un umbral de acierto.")
            periods = error_periods(phase_saved)
            period_names = {"period": "Trimestre objetivo", "n": "Orígenes", "mae": "MAE", "rmse": "RMSE", "bias": "Sesgo", "naive_mae": "MAE persistencia", "skill_mae_pct": "Mejora MAE (%)"}
            st.dataframe(periods[list(period_names)].sort_values("mae", ascending=False).rename(columns=period_names).round(3), hide_index=True, width="stretch",
                         column_config={"Trimestre objetivo": st.column_config.TextColumn("Trimestre", width=90), "Orígenes": st.column_config.NumberColumn("n", width=45), "MAE persistencia": st.column_config.NumberColumn("MAE ingenuo", width=95), "Mejora MAE (%)": st.column_config.NumberColumn("Mejora (%)", width=90)})
            st.caption(f"{PHASES[phase]} completo, semilla 42 (persistencia sin semilla), horizonte {horizon}. Trimestres según fecha objetivo, ordenados por MAE; los gráficos anteriores muestran solo el semestre de orígenes seleccionado. La mejora compara exactamente las mismas observaciones con persistencia.")
            mode = st.radio("Casos a consultar", ["Mayores errores", "Menores errores"], horizontal=True, key="eval_error_cases")
            cases = detail_cases(phase_saved)
            cases["better_than_naive"] = cases["absolute_error"] < cases["absolute_naive_error"]
            cases = cases.sort_values(["absolute_error", "origin_date"], ascending=[mode == "Menores errores", True]).head(10)
            for col in ("origin_date", "target_date"):
                cases[col] = pd.to_datetime(cases[col]).dt.strftime("%d/%m/%Y")
            case_names = {"origin_date": "Origen", "target_date": "Fecha objetivo", "y_true": "Brent real", "y_pred": "Predicción", "error": "Error firmado", "absolute_error": "Error absoluto", "absolute_naive_error": "Error persistencia", "better_than_naive": "Mejora a persistencia"}
            st.dataframe(cases[list(case_names)].rename(columns=case_names).round(3), hide_index=True, width="stretch",
                         column_order=["Origen", "Fecha objetivo", "Brent real", "Predicción", "Error absoluto", "Error persistencia"],
                         column_config={"Origen": st.column_config.TextColumn(width=95), "Fecha objetivo": st.column_config.TextColumn("Objetivo", width=95), "Brent real": st.column_config.NumberColumn("Real", width=75), "Predicción": st.column_config.NumberColumn("Estimado", width=85), "Error absoluto": st.column_config.NumberColumn("|Error|", width=80), "Error persistencia": st.column_config.NumberColumn("|Ingenuo|", width=85)})
            st.caption("Precios y errores en USD/barril. El menú de columnas permite mostrar también el error firmado y si mejora a persistencia. Un caso favorable no demuestra superioridad en el conjunto ni garantiza aciertos futuros.")
    st.subheader("Cobertura y amplitud de los intervalos")
    intervals = data["interval_metrics"]
    cells = intervals.loc[intervals["phase"].eq(phase) & intervals["block_id"].eq("all")
        & intervals["horizon"].eq(horizon) & (intervals["model"].eq(model) | intervals["model"].eq("persistence"))
        & intervals["seed"].isin([42, -1])]
    if not cells.empty:
        chart(coverage_chart(cells), "coverage_width")
        st.caption(f"{PHASES[phase]} completo: todos sus semestres, semilla 42; las curvas anteriores muestran solo {semester_label(block).lower()}. Cobertura alta con bandas amplias no equivale a mayor precisión. No se recalibra con test.")
        if phase == "calibration":
            st.caption("La cobertura en calibración es descriptiva y retrospectiva: se mide sobre los residuos usados para fijar las bandas, no sobre una evaluación independiente.")
    else:
        st.info("En desarrollo no se calcularon intervalos: los cuantiles se fijaron posteriormente con los residuos de calibración. Por eso no se muestra una gráfica de cobertura en esta fase.")
    with st.expander("De la prueba inicial a las configuraciones seleccionadas"):
        evolution = metrics.loc[metrics["phase"].eq("development") & metrics["model"].eq(model)
            & metrics["block_id"].eq("all") & metrics["horizon"].eq(horizon) & metrics["seed"].isin([42, -1])]
        evolution = evolution[["experiment", "variant", "mae", "rmse", "skill_mae_pct"]].drop_duplicates().copy()
        evolution["experiment"] = evolution["experiment"].map(EXPERIMENTS)
        evolution["variant"] = evolution["variant"].map(VARIANT_NAMES)
        st.dataframe(evolution.rename(columns={"experiment": "Experimento", "variant": "Configuración", "mae": "MAE", "rmse": "RMSE", "skill_mae_pct": "Mejora MAE (%)"}).round(3), hide_index=True, width="stretch", row_height=38)
        st.caption("Resultados de desarrollo, semilla 42. La validación interior cambió el historial de ajuste frente a la prueba inicial; la diferencia no se debe únicamente al número de épocas.")


def methodology(data):
    heading("Metodología y arquitectura", "De las series económicas a una predicción verificable, con una política temporal explícita.")
    steps = [("01", "Datos económicos", "FRED/EIA y mercados de Yahoo"), ("02", "Preprocesamiento", "Fechas de disponibilidad, ausencias y escalado"), ("03", "Ventanas temporales", "60 pasos de 20 entradas cada uno"), ("04", "Redes recurrentes", "RNN, LSTM y GRU · 32 unidades"), ("05", "Predicciones", "Cambios a 5, 63 y 252 días"), ("06", "Evaluación", "Errores, persistencia e intervalos del 80 %")]
    st.html('<div class="pipeline">' + ''.join(f'<div><b>{n}</b>{escape(title)}<small>{escape(text)}</small></div>' for n, title, text in steps) + '</div>')
    st.caption("Entradas: Brent, WTI, gas natural, oro, cobre, índice dólar, VIX, tipo BCE, USD/EUR y CNY/USD. FRED/EIA y Yahoo; OPEP no disponible en el snapshot congelado.")
    st.html('<div class="tensor-flow"><span><b>Entrada</b> 60 × 20</span><span class="arrow">→</span><span><b>Memoria</b> 32 unidades</span><span class="arrow">→</span><span><b>Salida</b> 3 cambios de precio</span><span class="arrow">→</span><span><b>Reconstrucción</b> USD/barril</span></div>')
    for column, (model, name, text) in zip(st.columns(3), [("rnn", "RNN · memoria simple", "Actualiza un estado que resume la secuencia. Es la arquitectura más pequeña: 1.827 parámetros."), ("lstm", "LSTM · memoria regulada", "Usa compuertas y una memoria separada para conservar información. 7.011 parámetros; LSTM-B/42 se seleccionó antes de test."), ("gru", "GRU · estructura compacta", "Regula qué información conserva o renueva con menos parámetros que LSTM: 5.283.")]):
        with column: st.html(f'<div class="panel {model}"><h3>{escape(name)}</h3><p>{escape(text)}</p></div>')
    with st.expander("Entrenamientos acreditados", expanded=False):
        st.write("Los registros conservan 204 ajustes principales y tres repeticiones adicionales de reproducibilidad: al menos 207 entrenamientos y 1.481 épocas de optimización acreditadas.")
        st.table(pd.DataFrame({"Etapa": ["Prueba inicial", "Optimización A/B", "Mejoras C/D", "Evaluación final", "Repeticiones de reproducibilidad"], "Ajustes": [18, 72, 60, 54, 3], "Épocas": [54, 623, 571, 216, 17]}))
        st.caption("Cada ajuste inicia una red y aprende conjuntamente los tres horizontes; no son 207 modelos por horizonte ni una única red entrenada durante 1.481 épocas. La época 0 es un diagnóstico inicial y no cuenta como optimización. Ejecución documentada en CPU, con semillas 42, 123 y 2026; la prueba inicial usa 42.")
    left, right = st.columns(2)
    with left:
        st.subheader("Separación temporal")
        st.table(pd.DataFrame({"Conjunto": ["Desarrollo", "Calibración", "Test"], "Orígenes": ["2015 · 2017 · 2019", "2021–2022", "2024"], "Uso": ["Elegir configuraciones", "Calibrar intervalos", "Evaluar política congelada"]}))
        st.write("La validación avanza en el tiempo y amplía el historial con reajustes semestrales. En desarrollo, Early Stopping usa un máximo de 20 épocas y paciencia de 5. Las épocas finales se fijaron antes de calibración y test: RNN 4, LSTM 3 y GRU 5.")
        st.caption("Para cada arquitectura se tomó la mediana de sus 18 mejores épocas interiores (seis bloques de desarrollo × tres semillas) y se redondeó hacia arriba: RNN-D 3,5 → 4; LSTM-B 3 → 3; GRU-C 5 → 5. Los reajustes finales usan ese número fijo sobre el historial permitido, sin elegir épocas con calibración ni test. Trasladar esta regla al historial ampliado es una limitación del diseño.")
    with right:
        st.subheader("Qué aprende la red")
        st.code("Precio futuro = Brent conocido + cambio predicho", language=None)
        st.write("Cada ventana contiene diez series y diez indicadores de ausencia: 20 entradas por paso. Una capa recurrente resume los 60 pasos y estima tres cambios. Se suman a la referencia disponible, sin inventar valores intermedios.")
        st.write("Las bandas del 80 % se obtienen de errores de calibración anteriores a test. Se conservan separadas por modelo, semilla y horizonte; una cobertura alta puede acompañarse de intervalos demasiado amplios.")
    st.html('<div class="metric-guide"><div><b>MAE · USD/barril</b>Error absoluto medio: cuánto se desvía el precio en promedio.</div><div><b>RMSE · USD/barril</b>Da mayor peso a los errores grandes.</div><div><b>MSE · USD²/barril²</b>Promedio del error elevado al cuadrado.</div><div><b>R² · sin unidad</b>Compara el error cuadrático con la variación de los valores reales del conjunto evaluado. Mayor es mejor; puede ser negativo y no es una probabilidad.</div></div>')
    with st.expander("Controles contra información futura", expanded=False):
        st.write("Las series conservan observaciones y ausencias originales; la unión temporal solo toma valores admitidos por los retardos asumidos. La imputación con medianas de entrenamiento y las máscaras distinguen entradas ausentes; las etiquetas futuras nunca se rellenan.")
        st.markdown("- Etiquetas de los tres horizontes maduras al corte de cada reajuste.\n- Imputación y escalado ajustados solo con filas permitidas de entrenamiento.\n- Ninguna muestra con origen en test entra en entrenamiento.\n- Calendario lunes–viernes sin comprimir festivos; se excluyen etiquetas no observadas.\n- La selección de red y cuantiles no cambia al observar los resultados de test.")
        st.caption("Test online: un reajuste posterior puede reutilizar precios ya publicados como etiquetas de orígenes antiguos. No es un holdout estático de todos los precios.")
    st.subheader("Conclusiones y límites")
    st.write("Persistencia es la referencia con menor error medio en los tres horizontes de test, al promediar las tres semillas de cada red. Las redes presentan sesgo y variación entre semillas, especialmente a 252 días. LSTM-B fue la mejor red en desarrollo; esa elección no la convierte en el mejor predictor global. La aplicación permite examinar los resultados negativos y los periodos favorables, sin afirmar una ventaja predictiva inexistente.")
    st.caption("Los retardos de publicación son supuestos; no se garantiza que cada dato revisado existiera en su fecha histórica. La cesta OPEP quedó ausente por un HTTP 403. Los pesos corresponden al reajuste lógico de julio de 2024; no se registró la fecha real de ejecución del entrenamiento. Actualizar datos no modifica pesos ni cuantiles.")
