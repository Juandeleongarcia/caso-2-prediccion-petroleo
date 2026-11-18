"""Public academic results only: no restricted artifacts or neural inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oil_forecast.public_study import comparison_rows, load_public_metrics

PAGES = ["Visión general", "Predicción", "Comparación", "Evaluación", "Metodología"]
EXPERIMENTS = {"final": "Evaluación final", "refinement": "Mejoras C/D",
               "optimization": "Optimización A/B", "smoke": "Prueba inicial"}
PHASES = {"test": "Test · orígenes de 2024", "calibration": "Calibración · 2021–2022",
          "development": "Desarrollo · 2015/17/19"}
MODELS = {"persistence": "Persistencia", "RNN": "RNN", "LSTM": "LSTM", "GRU": "GRU"}
METRICS = {"MAE": "mae", "MSE": "mse", "RMSE": "rmse", "R²": "r2",
           "Sesgo": "bias", "Mejora MAE (%)": "skill_mae_pct"}
COLORS = {"persistence": "#A9B6CA", "RNN": "#7297DF", "LSTM": "#D9B970", "GRU": "#5CBAA5"}
DIRECTORY = ROOT / "artifacts/derived/brent-academic-v1"

st.set_page_config(page_title="Brent Forecast · Resultados académicos", page_icon="◈", layout="wide")
st.html(f"<style>{(ROOT / 'src/oil_forecast/app.css').read_text(encoding='utf-8')}</style>")
with st.sidebar:
    st.html('<div class="eyebrow">ESTUDIO ACADÉMICO · UAX</div><div class="brand">BRENT<span> FORECAST</span></div>')
    page = st.radio("Navegación", PAGES, key="public_page", label_visibility="collapsed")
    st.caption("Resultados históricos agregados. Persistencia es la mejor referencia media de test.")
    st.caption("Red seleccionada antes de test: LSTM-B, semilla 42.")

st.warning("Demostración de resultados · Sin inferencia neuronal")
st.caption("Versión pública parcial: consulta el estudio cerrado sin cargar precios individuales ni modelos.")


@st.cache_data(show_spinner=False)
def verified_metrics(revision):
    return load_public_metrics(DIRECTORY)


def label(row):
    return "Persistencia" if row["model"] == "persistence" else f"{row['model']}-{row['variant']}"


def display_table(rows):
    seed_counts = rows["seed_count"].where(~rows["model"].eq("persistence"), 0)
    result = pd.DataFrame({"Modelo": rows.apply(label, axis=1), "Orígenes": rows["n"],
                           "Semillas": seed_counts, "MAE": rows["mae"], "MSE": rows["mse"],
                           "RMSE": rows["rmse"], "R²": rows["r2"], "Sesgo": rows["bias"],
                           "Mejora MAE (%)": rows["skill_mae_pct"]})
    st.dataframe(result.round(6), hide_index=True, width="stretch")


def metric_chart(rows, metric, title):
    figure = go.Figure(go.Bar(x=rows.apply(label, axis=1), y=rows[metric],
                             marker_color=rows["model"].map(COLORS)))
    figure.update_layout(title=title, height=350, paper_bgcolor="#081321", plot_bgcolor="#081321",
                         font_color="#F1F5F9", margin=dict(l=30, r=20, t=55, b=55))
    figure.add_hline(y=0, line_dash="dot", line_color="#A9B6CA")
    figure.update_yaxes(title_text="Sin unidad" if metric == "r2" else "%" if metric == "skill_mae_pct"
                       else "USD²/barril²" if metric == "mse" else "USD/barril")
    st.plotly_chart(figure, width="stretch", theme=None)


def seed_options(frame):
    available = sorted(set(frame.loc[frame["seed"].ge(0), "seed"].astype(int)))
    return ["mean", *[str(value) for value in available]]


def seed_label(value):
    return "Media de semillas disponibles" if value == "mean" else f"Semilla {value}"


def overview(metrics):
    st.title("Visión general del estudio")
    st.write("Se compararon RNN, LSTM y GRU con persistencia a 5, 63 y 252 días lunes–viernes. "
             "Las tablas y gráficos proceden de los resultados reales ya calculados.")
    columns = st.columns(3)
    columns[0].metric("Modelos comparados", "4")
    columns[1].metric("Horizontes", "5 · 63 · 252")
    columns[2].metric("Grupos de métricas", f"{len(metrics):,}".replace(",", "."))
    records = []
    for horizon in (5, 63, 252):
        rows = comparison_rows(metrics, horizon=horizon)
        for row in rows.to_dict("records"):
            records.append({"Horizonte": horizon, "Modelo": label(row), "MAE": row["mae"],
                            "R²": row["r2"], "Mejora MAE (%)": row["skill_mae_pct"]})
    st.subheader("Resultados medios de test")
    st.dataframe(pd.DataFrame(records).round(6), hide_index=True, width="stretch")
    st.info("Persistencia supera en MAE medio a las tres redes en los tres horizontes. "
            "Se mantienen los resultados negativos y la selección previa de LSTM-B/42.")
    st.caption("Las medias se calculan sobre métricas por semilla; no son un ensemble ni multiplican los orígenes evaluados.")


def prediction(metrics):
    st.title("Predicción")
    st.info("Esta versión pública no ejecuta RNN, LSTM ni GRU: sus pesos y datos de apoyo no se distribuyen "
            "mientras no se acredite el permiso aplicable. No genera precios futuros ni sustituye la inferencia por una simulación.")
    st.write("La aplicación completa conservada localmente realiza inferencia real en CPU con los pesos congelados. "
             "Esta demostración pública permite consultar sus resultados experimentales, sin presentarlos como nuevas predicciones.")
    st.subheader("Errores históricos de la red operativa y persistencia")
    records = []
    for horizon in (5, 63, 252):
        selected = comparison_rows(metrics, horizon=horizon, seed_mode="42")
        for row in selected.loc[selected["model"].isin(["LSTM", "persistence"])].to_dict("records"):
            records.append({"Horizonte": horizon, "Modelo": label(row), "Orígenes": row["n"],
                            "MAE": row["mae"], "MSE": row["mse"], "RMSE": row["rmse"], "R²": row["r2"]})
    st.dataframe(pd.DataFrame(records).round(6), hide_index=True, width="stretch")
    st.caption("Evaluación test de 238 orígenes por horizonte; LSTM con semilla 42 y persistencia sin semilla. "
               "MAE/RMSE en USD/barril; MSE en USD²/barril²; R² sin unidad. No son cotizaciones ni pronósticos emitidos ahora.")
    st.warning("Los pesos corresponden al reajuste lógico de julio de 2024; la fecha real de ejecución no quedó registrada. "
               "El estudio no acredita una herramienta de predicción financiera fiable.")


def comparison(metrics):
    st.title("Comparación de modelos")
    experiment = st.selectbox("Experimento", list(EXPERIMENTS), format_func=EXPERIMENTS.get,
                              key="public_compare_experiment")
    source = metrics.loc[metrics["experiment"].eq(experiment)]
    phases = [phase for phase in PHASES if phase in set(source["phase"])]
    a, b, c = st.columns(3)
    with a:
        phase = st.selectbox("Conjunto", phases, format_func=PHASES.get, key="public_compare_phase")
    with b:
        horizon = st.selectbox("Horizonte", [5, 63, 252], key="public_compare_horizon")
    selected = source.loc[source["phase"].eq(phase) & source["horizon"].eq(horizon)]
    blocks = ["all", *sorted(set(selected["block_id"]) - {"all"})]
    with c:
        block = st.selectbox("Periodo", blocks, format_func=lambda value: "Conjunto completo" if value == "all" else value,
                             key="public_compare_block")
    mode = st.radio("Semillas", seed_options(selected), format_func=seed_label, horizontal=True,
                    key="public_compare_seeds")
    rows = comparison_rows(metrics, experiment=experiment, phase=phase, horizon=horizon,
                           seed_mode=mode, block_id=block)
    variant = st.selectbox("Configuración", ["Todas", *sorted(set(rows["variant"]))], key="public_compare_variant")
    if variant != "Todas":
        # Preserve the naive reference when selecting a neuronal configuration.
        rows = rows.loc[rows["variant"].eq(variant) | rows["model"].eq("persistence")]
    metric = st.radio("Métrica", list(METRICS), horizontal=True, key="public_compare_metric")
    metric_chart(rows, METRICS[metric], f"{metric} · {horizon} días")
    display_table(rows)
    st.caption("Menor MAE/MSE/RMSE y mayor R² indican mejor ajuste. Sesgo positivo: sobreestimación. "
               "Mejora MAE = 100 × (1 − MAE/MAE persistencia); un valor negativo significa peor resultado.")
    st.caption("El número de semillas se conserva por configuración: algunas solo tienen 42. "
               "Persistencia no utiliza semilla ni entrena. No se mezclan semestres con sus totales "
               "ni semillas como observaciones independientes.")


def evaluation(metrics):
    st.title("Evaluación y resultados agregados")
    experiment = st.selectbox("Experimento", list(EXPERIMENTS), format_func=EXPERIMENTS.get,
                              key="public_eval_experiment")
    source = metrics.loc[metrics["experiment"].eq(experiment)]
    phase = st.selectbox("Conjunto", [phase for phase in PHASES if phase in set(source["phase"])],
                         format_func=PHASES.get, key="public_eval_phase")
    source = source.loc[source["phase"].eq(phase)]
    model = st.selectbox("Arquitectura", [name for name in MODELS if name in set(source["model"])],
                         format_func=MODELS.get, key="public_eval_model")
    candidates = source.loc[source["model"].eq(model)]
    variant = st.selectbox("Configuración", sorted(set(candidates["variant"])), key="public_eval_variant")
    horizon = st.selectbox("Horizonte", [5, 63, 252], key="public_eval_horizon")
    candidates = candidates.loc[candidates["variant"].eq(variant) & candidates["horizon"].eq(horizon)]
    mode = st.radio("Semillas", seed_options(candidates),
                    format_func=(lambda value: "Resultado único, sin semilla") if model == "persistence" else seed_label,
                    horizontal=True,
                    key="public_eval_seed")
    records = []
    for block in sorted(set(candidates["block_id"]) - {"all"}):
        rows = comparison_rows(metrics, experiment=experiment, phase=phase, horizon=horizon,
                               seed_mode=mode, block_id=block)
        selected = rows.loc[rows["model"].eq(model) & rows["variant"].eq(variant)]
        for row in selected.to_dict("records"):
            records.append({"Semestre": block, "Orígenes": row["n"],
                            "Semillas": 0 if model == "persistence" else row["seed_count"],
                            "MAE": row["mae"], "MSE": row["mse"], "RMSE": row["rmse"],
                            "R²": row["r2"], "Sesgo": row["bias"], "Mejora MAE (%)": row["skill_mae_pct"]})
    frame = pd.DataFrame(records)
    st.dataframe(frame.round(6), hide_index=True, width="stretch")
    if not frame.empty:
        figure = go.Figure(go.Bar(x=frame["Semestre"], y=frame["MAE"], marker_color=COLORS[model]))
        figure.update_layout(title="MAE por semestre evaluado", yaxis_title="USD/barril", height=340)
        st.plotly_chart(figure, width="stretch", theme=None)
    st.info("Esta vista utiliza errores agregados reales. No incluye observaciones individuales, curvas por época "
            "ni gráficos de precio real frente a predicción; no se reconstruyen esos datos a partir de métricas.")
    st.caption("Las tablas separan experimento, configuración, conjunto, horizonte y semilla. "
               "Los resultados desfavorables y los R² negativos se conservan.")


def methodology():
    st.title("Metodología y limitaciones")
    st.write("El estudio usa ventanas de 60 pasos con diez series y diez máscaras de ausencia. "
             "RNN, LSTM y GRU tienen una capa de 32 unidades y tres salidas directas residuales a 5, 63 y 252 días lunes–viernes.")
    st.caption("Variables del estudio: Brent, WTI, gas natural, tipo del BCE, USD/EUR, CNY/USD, VIX, "
               "índice del dólar, oro y cobre. Se añaden máscaras de ausencia; las observaciones originales no se cargan en esta versión.")
    st.dataframe(pd.DataFrame({"Conjunto": ["Desarrollo", "Calibración", "Test"],
                              "Orígenes": ["2015, 2017 y 2019", "2021–2022", "2024"],
                              "Uso": ["Variantes, épocas y selección", "Incertidumbre antes de test", "Evaluación final"]}),
                 hide_index=True, width="stretch")
    st.write("El walk-forward es expansivo con reajustes semestrales. Las tres etiquetas deben estar publicadas "
             "al corte según los retardos asumidos; imputación y escalado se ajustan únicamente sobre entrenamiento. "
             "Validación interior y Early Stopping seleccionan en desarrollo; el reajuste final utiliza épocas prefijadas.")
    st.write("LSTM-B/42 fue seleccionada antes de consultar test. Persistencia conserva la referencia más reciente admisible. "
             "El test es online con política congelada: un reajuste posterior puede usar etiquetas antiguas ya publicadas, "
             "sin entrenar con orígenes de 2024 ni elegir hiperparámetros mirando test.")
    st.caption("Los intervalos nominales del 80 % del estudio se calibraron con 471 orígenes; las métricas puntuales "
               "conservan 472 de calibración. Esta vista pública no carga cuantiles ni genera intervalos nuevos.")
    st.warning("Limitaciones: vintages históricos no certificados, retardos de publicación asumidos, OPEP ausente, "
               "dependencia temporal y horizontes solapados, inestabilidad entre semillas y pesos de julio de 2024. "
               "La fecha real de entrenamiento no quedó registrada.")
    st.write("La aplicación completa, las redes y los resultados originales se conservan sin modificaciones. "
             "Esta alternativa pública es una demostración parcial de resultados, no satisface por sí sola "
             "el requisito de probar nuevas predicciones neuronales. Ese servicio requiere permisos y artefactos autorizados.")
    st.caption("Autoría individual y asistencia de IA declarada en la documentación. La identificación y declaración "
               "académica final corresponden al estudiante. No se asigna una licencia ficticia a datos o derivados.")


try:
    revision = "".join(hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                       for name in ("artifact_manifest.json", "metrics.csv"))
    metrics = verified_metrics(revision)
except (OSError, ValueError, KeyError):
    st.error("No se han podido verificar las métricas públicas del estudio.")
    st.stop()

if page == "Visión general":
    overview(metrics)
elif page == "Predicción":
    prediction(metrics)
elif page == "Comparación":
    comparison(metrics)
elif page == "Evaluación":
    evaluation(metrics)
else:
    methodology()

st.caption("Proyecto académico UAX · Resultados agregados congelados · Sin inferencia neuronal ni asesoramiento financiero.")
