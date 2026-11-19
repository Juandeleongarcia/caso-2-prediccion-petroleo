"""Local WSGI API using the original frozen scientific implementation."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import RLock
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from oil_forecast.app_data import (load_app_data, load_services, forecast_for_model,
    aggregate_metrics, filter_history, context_indicators, latest_market, stored_forecast)
from oil_forecast.app_charts import (history_chart, forecast_chart, prediction_chart,
    errors_chart, loss_chart, metrics_chart, coverage_chart)
from oil_forecast.insights import explain_forecast, cost_scenarios, event_points
from oil_forecast.academic import error_periods, detail_cases
from oil_forecast.deployment import verify_registered_manifest
from oil_forecast.refresh import RefreshController
from oil_forecast.live_hub import HubReader, LiveHubController

MODELS = ['persistence', 'RNN', 'LSTM', 'GRU']
FILES = {'checkpoints.pt', 'protocol.json', 'intervals.json', 'load_models.py', 'README.md'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SecureRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = parsed.hostname or ''
        if parsed.scheme != 'https' or not (host == 'huggingface.co' or host.endswith(('.huggingface.co', '.hf.co'))):
            raise ValueError('Unexpected artifact redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected:
            redirected.remove_header('Authorization')
        return redirected

def verify_minimal(folder, trusted):
    if not re.fullmatch('[0-9a-f]{64}', trusted or '') or digest(folder / 'manifest.json') != trusted:
        raise ValueError('Untrusted remote manifest')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['files']) != FILES:
        raise ValueError('Unexpected remote payload')
    for name, info in manifest['files'].items():
        if (folder / name).stat().st_size != info['bytes'] or digest(folder / name) != info['sha256']:
            raise ValueError('Remote payload integrity failure')

def download_minimal(repo, revision, trusted, destination, token=None):
    """Explicit pinned download; never returns tokens or signed download URLs."""
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo or '') or not re.fullmatch('[0-9a-f]{40}', revision or ''):
        raise ValueError('An immutable Models commit is required')
    if not re.fullmatch('[0-9a-f]{64}', trusted or ''):
        raise ValueError('An independent manifest hash is required')
    destination = Path(destination)
    if destination.exists():
        verify_minimal(destination, trusted)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            folder = Path(temporary)
            opener = build_opener(SecureRedirect())
            for name in ['manifest.json', *sorted(FILES)]:
                request = Request(f'https://huggingface.co/{repo}/resolve/{revision}/{name}')
                if token:
                    request.add_unredirected_header('Authorization', f'Bearer {token}')
                with opener.open(request, timeout=30) as response:
                    content = response.read(32 * 1024 * 1024 + 1)
                if len(content) > 32 * 1024 * 1024:
                    raise ValueError('Artifact too large')
                (folder / name).write_bytes(content)
                if name == 'manifest.json' and digest(folder / name) != trusted:
                    raise ValueError('Untrusted remote manifest')
            verify_minimal(folder, trusted)
            # New revision directory only; existing resources are never replaced.
            folder.rename(destination)
        return destination
    except Exception:
        raise ValueError('No se pudo descargar una revisión verificada de Models; se conserva la copia anterior.') from None

def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))

def chart(figure):
    return json.loads(figure.to_json())

def choose(query, name, default, allowed):
    value = query.get(name, [str(default)])[0]
    if value not in [str(x) for x in allowed]:
        raise ValueError(f'Selección inválida: {name}')
    return value

class Study:
    def __init__(self, initial_refresh_network=True):
        self.lock = RLock()
        self.bundle = Path(os.environ.get('BRENT_BUNDLE_DIR', ROOT / 'artifacts/local/brent-frozen-v1'))
        evidence = Path(os.environ.get('BRENT_DASHBOARD_DIR', ROOT / 'artifacts/local/brent-dashboard-v1'))
        for folder, registry in [(self.bundle, 'serving_bundle.json'), (evidence, 'dashboard_bundle.json')]:
            if folder.exists():
                verify_registered_manifest(folder, json.loads((ROOT / 'configs' / registry).read_text())['manifest_sha256'])
        self.data = load_app_data(self.bundle, evidence if evidence.exists() else None)
        self.source = os.environ.get('BRENT_MODEL_SOURCE', 'local')
        if self.source == 'hf':
            revision = os.environ.get('BRENT_HF_REVISION')
            target = Path(os.environ.get('BRENT_MODEL_CACHE_DIR', ROOT / 'artifacts/local/web-models')) / str(revision)
            folder = download_minimal(os.environ.get('BRENT_HF_REPO_ID'), revision,
                os.environ.get('BRENT_MANIFEST_SHA256'), target, os.environ.get('HF_TOKEN'))
            # Verified against independent metadata BEFORE importing downloaded code.
            spec = importlib.util.spec_from_file_location('brent_verified_models', folder / 'load_models.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.services = module.load_models(folder, os.environ['BRENT_MANIFEST_SHA256'])
        elif self.source == 'local':
            self.services = load_services(self.bundle)
        else:
            raise ValueError('BRENT_MODEL_SOURCE must be local or hf')
        self.config = json.loads((ROOT / 'configs/data.json').read_text(encoding='utf-8'))
        self.events = json.loads((ROOT / 'configs/economic_events.json').read_text(encoding='utf-8'))['events']
        self.controller = RefreshController(Path(os.environ.get('BRENT_LIVE_DIR', ROOT / 'data/live')))
        self.published = None
        if os.environ.get('BRENT_LIVE_ENABLED') == '1':
            if os.environ.get('BRENT_LIVE_ACCESS_AUTHORIZED') != '1':
                raise ValueError('Remote live data requires separate acquisition/distribution authorization')
            self.published = LiveHubController(self.controller.directory / 'published',
                HubReader(os.environ.get('BRENT_LIVE_REPO_ID', os.environ.get('BRENT_HF_REPO_ID')),
                          os.environ.get('HF_TOKEN')), os.environ.get('BRENT_MANIFEST_SHA256'),
                interval_seconds=int(os.environ.get('BRENT_LIVE_POLL_SECONDS', '300')),
                minimum_updated_at=self.data['protocol']['data_updated_at'])
        self.refresh_status = {'status': 'disabled'}
        self.updated = self.data['protocol']['data_updated_at']
        self.refresh(False, allow_network=initial_refresh_network)

    def refresh(self, force=False, allow_network=True):
        if self.published is not None:
            def validate_live(window):
                for service in self.services.values():
                    service.predict(window, data_updated_at=pd.Timestamp.now(tz='UTC'))
            fresh,status=self.published.check(self.config,validate_live)
            if fresh is not None:
                self.data['history'],self.data['window']=fresh['history'],fresh['window']
                self.updated=fresh['protocol']['data_updated_at']
                self.data['protocol']=self.data['protocol'] | {'data_updated_at':self.updated}
            self.refresh_status=status
            return {'status':status.get('status'),'data_updated_at':self.updated,'weights_updated':False}
        enabled = allow_network and os.environ.get('BRENT_ALLOW_PROVIDER_ACCESS') == '1'
        def validate(window):
            for service in self.services.values():
                service.predict(window, data_updated_at=pd.Timestamp.now(tz='UTC'))
        result = self.controller.check(self.config, enabled=enabled, force=force and enabled, validator=validate)
        if result['data'] is not None:
            fresh = result['data']
            self.data['history'], self.data['window'] = fresh['history'], fresh['window']
            self.updated = fresh['protocol']['data_updated_at']
            self.data['protocol'] = self.data['protocol'] | {'data_updated_at': self.updated}
        self.refresh_status = {k: v for k, v in result.items() if k not in {'data', 'message'}}
        return {'status': result['status'], 'data_updated_at': self.updated, 'weights_updated': False}

    def response(self, endpoint, q):
        with self.lock:
            if self.published is not None:self.refresh(False)
            return self._response(endpoint, q)

    def _response(self, endpoint, q):
        model = choose(q, 'model', 'LSTM', MODELS)
        horizon = int(choose(q, 'horizon', 5, [5, 63, 252]))
        data = self.data
        if endpoint == 'models':
            return {'models': MODELS, 'source': self.source, 'horizons': [5, 63, 252],
                'model_refit_at': '2024-07-01T06:00:00+00:00', 'actual_training_execution_at': None,
                'data_updated_at': self.updated, 'refresh': self.refresh_status,
                'can_refresh': os.environ.get('BRENT_ALLOW_PROVIDER_ACCESS') == '1' and os.environ.get('BRENT_PUBLIC_ACCESS_AUTHORIZED') != '1',
                'automatic_refresh': self.published is not None or os.environ.get('BRENT_ALLOW_PROVIDER_ACCESS') == '1',
                'data_refresh_mode': 'github_actions' if self.published is not None else 'provider_or_frozen'}
        if endpoint == 'overview':
            period = choose(q, 'period', '1Y', ['1M', '3M', '1Y', '5Y', 'MAX'])
            self.refresh(False)
            forecast = forecast_for_model(self.services, data['window'], 'LSTM', self.updated)
            events = event_points(data['history'], self.events)
            return {'market': latest_market(data['history'], data['protocol']), 'forecast': forecast,
                'refresh':self.refresh_status,
                'indicators': records(context_indicators(data['window'])),
                'events': records(events), 'charts': {'history': chart(history_chart(filter_history(data['history'], period), events=events))}}
        if endpoint == 'archive-origins':
            phase = choose(q, 'phase', 'test', ['test', 'calibration'])
            saved = data['evaluation']['predictions']
            rows = saved.loc[saved.experiment.eq('final') & saved.phase.eq(phase) & saved.model.eq(model) & saved.horizon.eq(horizon)]
            return {'origins': sorted(rows.origin_date.astype(str).unique())}
        if endpoint == 'predict':
            source = choose(q, 'source', 'current', ['current', 'archive'])
            if source == 'current':
                output = forecast_for_model(self.services, data['window'], model, self.updated)
            else:
                phase = choose(q, 'phase', 'test', ['test', 'calibration'])
                saved = data['evaluation']['predictions']
                saved = saved.loc[saved.experiment.eq('final') & saved.phase.eq(phase)]
                origins = sorted(saved.origin_date.astype(str).unique())
                origin = choose(q, 'origin', origins[-1], origins)
                output = stored_forecast(saved, model, horizon, origin)
            context = data['history']
            if source == 'archive':
                context = context.loc[pd.to_datetime(context.observation_date).le(pd.Timestamp(output['origin_date']))]
            result = {'forecast': output, 'mode': source, 'charts': {'forecast': chart(forecast_chart(context, output, horizon))}}
            if source == 'current':
                result['explanation'] = explain_forecast(output, horizon, data['metrics'], latest_market(data['history'], data['protocol']))
            if 'barrels' in q:
                barrels = float(q['barrels'][0])
                if not 0 <= barrels <= 1e9:
                    raise ValueError('Volumen fuera de rango')
                result['costs'] = records(cost_scenarios(output, horizon, barrels))
            return result
        if endpoint == 'comparison':
            phase = choose(q, 'phase', 'test', ['test', 'calibration', 'development'])
            seed = choose(q, 'seed', 'mean', ['mean', '42'])
            metric = choose(q, 'metric', 'mae', ['mae', 'rmse', 'mse', 'r2'])
            scores = data['metrics'] if phase != 'development' else data['evaluation']['metrics'].loc[data['evaluation']['metrics'].experiment.eq('refinement')]
            summary = aggregate_metrics(scores, phase, horizon, seed)
            return {'rows': records(summary), 'phase': phase, 'horizon': horizon, 'seed': seed,
                'charts': {'comparison': chart(metrics_chart(summary, metric))}}
        if endpoint == 'evaluation':
            e = data['evaluation']
            if not e:
                raise ValueError('Se necesita evidencia original para esta vista')
            experiment = choose(q, 'experiment', 'final', list(e['metrics'].experiment.unique()))
            scores = e['metrics'].loc[e['metrics'].experiment.eq(experiment) & e['metrics'].model.eq(model)]
            variants = list(scores.variant.unique())
            variant = choose(q, 'variant', variants[0], variants)
            scores = scores.loc[scores.variant.eq(variant)]
            phases = list(scores.phase.unique())
            phase = choose(q, 'phase', 'test' if 'test' in phases else phases[0], phases)
            blocks = sorted(scores.loc[scores.phase.eq(phase) & ~scores.block_id.eq('all'), 'block_id'].unique())
            preferred = 'development_2019H1' if 'development_2019H1' in blocks else blocks[0]
            block = choose(q, 'block', preferred, blocks)
            pred = e['predictions']
            saved = pred.loc[pred.experiment.eq(experiment) & pred.model.eq(model) & pred.variant.eq(variant)
                & pred.phase.eq(phase) & pred.horizon.eq(horizon) & pred.seed.eq(-1 if model == 'persistence' else 42)]
            selected = saved.loc[saved.block_id.eq(block)]
            curves = e['history'].loc[e['history'].experiment.eq(experiment)]
            charts = {'predictions': chart(prediction_chart(selected)), 'errors': chart(errors_chart(selected))}
            if model != 'persistence':
                charts['loss'] = chart(loss_chart(curves, model, variant, 42, block))
            interval = data['interval_metrics']
            interval = interval.loc[interval.phase.eq(phase) & interval.block_id.eq('all') & interval.horizon.eq(horizon)
                & interval.model.isin([model, 'persistence']) & interval.seed.isin([42, -1])]
            if not interval.empty:
                charts['coverage'] = chart(coverage_chart(interval))
            detail = scores.loc[scores.phase.eq(phase) & scores.block_id.eq(block) & scores.horizon.eq(horizon)
                & scores.seed.isin([42, -1])]
            case_order = choose(q, 'case_order', 'worst', ['worst', 'best'])
            return {'charts': charts, 'rows': records(detail), 'options': {'variants': variants, 'phases': phases, 'blocks': blocks},
                'selected': {'variant': variant, 'phase': phase, 'block': block},
                'periods': records(error_periods(saved)) if len(saved) else [],
                'cases': records(detail_cases(saved).sort_values('absolute_error', ascending=case_order == 'best').head(10)) if len(saved) else [],
                'scope': {'experiment': experiment, 'phase': phase, 'block': block, 'seed': 42 if model != 'persistence' else None,
                    'horizon': horizon, 'intervals_retrospective': phase == 'calibration'}}
        raise ValueError('Endpoint desconocido')

class Application:
    """Only the frontend directory is served; repository files are never static."""
    def __init__(self):
        self.study = None
        self.lock = RLock()
        self.retry_after = 0.0

    def resources(self):
        with self.lock:
            if self.study is None:
                if time.monotonic() < self.retry_after:
                    raise RuntimeError('Resource retry cooldown')
                try:
                    self.study = Study()
                except Exception:
                    self.retry_after = time.monotonic() + 30
                    raise RuntimeError('Verified model resources unavailable') from None
            return self.study

    def __call__(self, env, start_response):
        path = env.get('PATH_INFO', '/')
        method = env.get('REQUEST_METHOD', 'GET')
        headers = [('X-Content-Type-Options', 'nosniff'), ('Cache-Control', 'no-store')]
        origin = env.get('HTTP_ORIGIN')
        allowed = os.environ.get('BRENT_WEB_ORIGIN')
        if allowed and origin == allowed:
            headers += [('Access-Control-Allow-Origin', allowed), ('Vary', 'Origin')]
        try:
            if method not in {'GET', 'POST'}:
                return self._send(start_response, '405 Method Not Allowed', {'error': 'Método no permitido'}, headers)
            if path == '/api/health':
                if method != 'GET':
                    return self._send(start_response, '405 Method Not Allowed', {'error': 'GET requerido'}, headers)
                return self._send(start_response, '200 OK', {'status': 'ok', 'inference_loaded': self.study is not None}, headers)
            if path.startswith('/api/'):
                self.resources()
                if path == '/api/ready' and method == 'GET':
                    result = {'status': 'ready', 'source': self.study.source, 'horizons': [5, 63, 252]}
                elif path == '/api/refresh':
                    host = env.get('HTTP_HOST', '')
                    if os.environ.get('BRENT_PUBLIC_ACCESS_AUTHORIZED') == '1' or method != 'POST' or host not in {'127.0.0.1:8765', 'localhost:8765'} or (origin and origin not in {'http://127.0.0.1:8765', 'http://localhost:8765'}):
                        return self._send(start_response, '403 Forbidden', {'error': 'Actualización manual solo desde el servidor local'}, headers)
                    if os.environ.get('BRENT_ALLOW_PROVIDER_ACCESS') != '1':
                        return self._send(start_response, '403 Forbidden', {'error': 'Acceso a proveedores desactivado hasta acreditar permisos'}, headers)
                    with self.study.lock:
                        result = self.study.refresh(True)
                elif method == 'GET':
                    result = self.study.response(path.removeprefix('/api/'), parse_qs(env.get('QUERY_STRING', ''), max_num_fields=20))
                else:
                    return self._send(start_response, '405 Method Not Allowed', {'error': 'GET requerido'}, headers)
                return self._send(start_response, '200 OK', result, headers)
            if method != 'GET':
                return self._send(start_response, '405 Method Not Allowed', {'error': 'GET requerido'}, headers)
            root = ROOT / 'web/frontend'
            file = (root / (path.lstrip('/') or 'index.html')).resolve()
            if not file.is_relative_to(root.resolve()) or not file.is_file():
                return self._send(start_response, '404 Not Found', {'error': 'No encontrado'}, headers)
            content = file.read_bytes()
            start_response('200 OK', headers + [('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')])
            return [content]
        except (ValueError, KeyError, TypeError) as failure:
            return self._send(start_response, '400 Bad Request', {'error': str(failure) if path != '/api/models' else 'Configuración o artefactos inválidos'}, headers)
        except Exception:
            return self._send(start_response, '503 Service Unavailable', {'error': 'Inferencia o evidencia no disponible. Comprueba los artefactos y permisos del backend; no se generan resultados simulados.'}, headers)

    @staticmethod
    def _send(start, status, body, headers):
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        start(status, headers + [('Content-Type', 'application/json; charset=utf-8')])
        return [content]

app = Application()
