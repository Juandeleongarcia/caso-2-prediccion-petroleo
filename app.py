"""Brent Forecast | Energy Intelligence — local, read-only model application."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

import oil_forecast.app_pages as pages
from oil_forecast.acquisition import sha256
from oil_forecast.app_data import load_app_data, load_services, latest_market
from oil_forecast.artifacts import fetch_bundle, install_bundle, verify_bundle
from oil_forecast.refresh import RefreshController

st.set_page_config(page_title="Brent Forecast | Energy Intelligence", page_icon="◈", layout="wide",
                   initial_sidebar_state="auto")
st.html(f"<style>{(ROOT / 'src/oil_forecast/app.css').read_text(encoding='utf-8')}</style>")

PAGES = ["Visión general", "Predicción", "Comparación", "Evaluación", "Metodología"]
bundle = Path(os.environ.get("BRENT_BUNDLE_DIR", ROOT / "artifacts/local/brent-frozen-v1"))
dashboard = Path(os.environ.get("BRENT_DASHBOARD_DIR", ROOT / "artifacts/local/brent-dashboard-v1"))
academic = ROOT / "artifacts/derived/brent-academic-v1"
live_directory = Path(os.environ.get("BRENT_LIVE_DIR", ROOT / "data/live"))
app_config = json.loads((ROOT / "configs/app.json").read_text(encoding="utf-8"))
source_config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
refresh_hours = float(os.environ.get("BRENT_REFRESH_HOURS", app_config["refresh_interval_hours"]))
automatic_default = os.environ.get("BRENT_AUTO_REFRESH", str(app_config["auto_refresh"])).lower() in {"true", "1", "yes"}


def signature(directory):
    manifest = directory / "artifact_manifest.json"
    return sha256(manifest) if manifest.exists() else "missing"


@st.cache_data(show_spinner=False)
def cached_data(bundle_path, evidence_path, academic_path, revision):
    return load_app_data(Path(bundle_path), Path(evidence_path) if evidence_path else None, Path(academic_path))


@st.cache_resource(show_spinner=False)
def cached_services(bundle_path, revision):
    return load_services(Path(bundle_path))


@st.cache_resource(show_spinner=False)
def cached_controller(directory, hours, retry_minutes):
    return RefreshController(Path(directory), interval_hours=hours, retry_minutes=retry_minutes)


def validate_live_window(services, window):
    # Validate compatibility before publishing a last-good snapshot. Prediction
    # never fits weights, scalers or intervals; the displayed run later uses
    # the snapshot's actual acquisition timestamp, not this validation clock.
    checked_at = datetime.now(timezone.utc).isoformat()
    for service in services.values():
        service.predict(window, data_updated_at=checked_at)


@st.fragment(run_every=app_config["check_interval_seconds"])
def update_panel(services, fallback_updated_at):
    automatic = st.toggle("Actualización automática", value=automatic_default, key="auto_refresh")
    manual = st.button("Actualizar datos ahora", key="refresh_data", type="primary", width="stretch", disabled=services is None)
    if services is None:
        st.caption("Prepara los modelos y datos para activar la actualización.")
        return
    controller = cached_controller(str(live_directory), refresh_hours, app_config["retry_minutes"])
    with st.spinner("Comprobando datos y fuentes…"):
        result = controller.check(source_config, validator=lambda window: validate_live_window(services, window),
                                  force=manual, enabled=automatic)
    if result["data"] is not None:
        st.session_state["live_market"] = result["data"]
    st.session_state["refresh_status"] = {key: value for key, value in result.items() if key != "data"}
    labels = {"current": "Datos al día", "updated": "Actualización completada", "failed": "Se conserva la copia válida",
              "busy": "Actualización en otra sesión", "disabled": "Automática pausada", "cooldown": "Próximo intento programado"}
    st.caption(labels[result["status"]])
    retained_time = st.session_state.get("live_market", {}).get("protocol", {}).get("data_updated_at", fallback_updated_at)
    st.caption(f"Última actualización válida: {pages.timestamp(result['last_success_at'] or retained_time)}")
    if manual and result["status"] == "failed":
        st.error(result["message"])
    elif result["status"] in {"busy", "cooldown", "failed"}:
        st.caption(result["message"])
    with st.expander("Frecuencia y funcionamiento"):
        st.caption(f"Frecuencia: {refresh_hours:g} horas. Comprobación al inicio y cada {app_config['check_interval_seconds'] // 60} minutos con una sesión activa. No se descargan series en cada interacción.")
        st.caption("Si el alojamiento se suspende no hay consultas en segundo plano; se comprueba de nuevo al iniciarse. Los datos nuevos no cambian los pesos ni los cuantiles.")
        st.caption(f"Próxima actualización prevista: {pages.timestamp(result['next_due_at'])}")
    selected_id = st.session_state.get("live_market", {}).get("protocol", {}).get("snapshot_id", "frozen")
    if "rendered_snapshot_id" in st.session_state and st.session_state["rendered_snapshot_id"] != selected_id:
        st.session_state["rendered_snapshot_id"] = selected_id
        st.rerun()


def setting(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return None


def prepare_resources():
    for name, directory, registry_file, url_name in [
        ("inferencia", bundle, "serving_bundle.json", "BRENT_INFERENCE_URL"),
        ("evaluación", dashboard, "dashboard_bundle.json", "BRENT_EVIDENCE_URL")]:
        if directory.exists():
            verify_bundle(directory)
            continue
        registry = json.loads((ROOT / "configs" / registry_file).read_text(encoding="utf-8"))
        archive = ROOT / registry["archive"]["local_path"]
        if archive.exists():
            install_bundle(archive, directory, registry["archive"]["sha256"])
        else:
            url = setting(url_name) or registry["archive"].get("published_url")
            if not url:
                raise FileNotFoundError(f"No hay paquete local ni origen remoto configurado para {name}.")
            fetch_bundle(url, directory, registry["archive"]["sha256"])
    cached_data.clear()
    cached_services.clear()


with st.sidebar:
    st.html('<div class="eyebrow">ENERGY INTELLIGENCE</div><div class="brand">BRENT<span> FORECAST</span></div><div class="brand-sub">ANÁLISIS · MODELOS · EVIDENCIA</div>')
    page = st.radio("Navegación", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.html('<div class="eyebrow">SISTEMA PREDICTIVO</div><span class="badge">CPU · Semilla 42</span>')
    st.caption("Persistencia: mejor resultado medio en test. LSTM-B/42: red elegida antes de test.")
    st.html('<span class="badge badge-amber">Pesos congelados · JUL 2024</span>')
    with st.expander("Disponibilidad de recursos"):
        st.caption("Con los datos y modelos instalados puedes navegar sin conexión. Este botón verifica su integridad o prepara los recursos disponibles.")
        if st.button("Preparar o verificar recursos", key="prepare_resources", width="stretch"):
            try:
                with st.spinner("Verificando paquetes…"):
                    prepare_resources()
                st.success("Recursos verificados.")
                st.rerun()
            except Exception as failure:
                st.error(str(failure))

st.html('<div class="topline"><div class="identity">Brent Forecast | Energy Intelligence</div><span class="badge">ESTUDIO ACADÉMICO · UAX</span></div>')
st.caption("Predicción del precio del petróleo mediante redes neuronales recurrentes.")

try:
    evidence = str(dashboard) if dashboard.exists() else None
    with st.spinner("Cargando datos y modelos verificados…"):
        data = cached_data(str(bundle), evidence, str(academic), signature(bundle) + signature(dashboard) + signature(academic))
        services = cached_services(str(bundle), signature(bundle))
    with st.sidebar:
        update_panel(services, data["protocol"]["data_updated_at"])
    if "live_market" in st.session_state:
        fresh = st.session_state["live_market"]
        data["history"], data["window"] = fresh["history"], fresh["window"]
        data["protocol"] = data["protocol"] | {"data_updated_at": fresh["protocol"]["data_updated_at"]}
    data["economic_events"] = json.loads((ROOT / "configs/economic_events.json").read_text(encoding="utf-8"))
    st.session_state["rendered_snapshot_id"] = st.session_state.get("live_market", {}).get("protocol", {}).get("snapshot_id", "frozen")
    with st.sidebar:
        st.caption(f"Última cotización: {pages.date(latest_market(data['history'], data['protocol'])['observation_date'])}")
    if page == "Visión general":
        pages.overview(data, services)
    elif page == "Predicción":
        pages.prediction(data, services)
    elif page == "Comparación":
        pages.comparison(data)
    elif page == "Evaluación":
        pages.evaluation(data)
    else:
        pages.methodology(data)
except (OSError, ValueError, KeyError) as failure:
    st.error("No se han podido cargar datos o artefactos válidos para esta vista.")
    st.info("Utiliza «Preparar o verificar recursos» en el menú lateral. Los paquetes se instalan sin volver a entrenar modelos.")
    with st.expander("Detalle de disponibilidad"):
        st.code(str(failure), language=None)
    if page == "Metodología":
        pages.methodology({})

st.html('<div class="app-footer">Proyecto académico UAX · No constituye asesoramiento financiero. Los intervalos no garantizan cobertura futura. Consulta las limitaciones de datos y modelos en Metodología.</div>')
