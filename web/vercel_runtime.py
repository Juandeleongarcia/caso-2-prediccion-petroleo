"""Lazy WSGI adapter for Vercel: same scientific service, verified private resources."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
import time
from urllib.parse import parse_qs, urlencode

ROOT=Path(__file__).resolve().parents[1]
ENDPOINTS={'health','ready','models','overview','predict','archive-origins','comparison','evaluation','refresh'}
MODEL_REPO='Juaandeleoon/caso-2-prediccion-petroleo'
MODEL_REVISION='b9ad781cd90e1adcb080009358fca143e0bf0931'
MODEL_MANIFEST='0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570'


def settings(env):
    if env.get('VERCEL_SUPPORT_LARGE_FUNCTIONS')!='1':
        raise ValueError('Large Functions must be explicitly enabled')
    if env.get('BRENT_PUBLIC_ACCESS_AUTHORIZED')!='1':
        raise ValueError('Public data/derivative access is not authorized')
    if env.get('BRENT_MODEL_SOURCE','hf')!='hf' or not env.get('HF_TOKEN'):
        raise ValueError('Private Models requires server authentication and HF mode')
    repo=env.get('BRENT_HF_REPO_ID',MODEL_REPO)
    revision=env.get('BRENT_HF_REVISION',MODEL_REVISION)
    trusted=env.get('BRENT_MANIFEST_SHA256',MODEL_MANIFEST)
    if not re.fullmatch(r'[\w.-]+/[\w.-]+',repo) or not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('[0-9a-f]{64}',trusted):
        raise ValueError('Models requires immutable independently trusted metadata')
    runtime=Path(env.get('BRENT_RUNTIME_DIR',str(Path(tempfile.gettempdir())/'brent-vercel-v1')))
    if not runtime.is_absolute() or runtime.resolve().is_relative_to((ROOT/'src').resolve()):
        raise ValueError('Use a separate absolute writable runtime directory')
    if env.get('VERCEL')=='1' and not runtime.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError('Vercel runtime resources must live in temporary storage')
    return {'BRENT_MODEL_SOURCE':'hf','BRENT_HF_REPO_ID':repo,'BRENT_HF_REVISION':revision,
        'BRENT_MANIFEST_SHA256':trusted,'BRENT_MODEL_CACHE_DIR':str(runtime/'models'),
        'BRENT_BUNDLE_DIR':str(runtime/'brent-frozen-v1'),
        'BRENT_DASHBOARD_DIR':str(runtime/'brent-dashboard-v1'),'BRENT_LIVE_DIR':str(runtime/'live')}


class VercelApplication:
    def __init__(self):
        self.delegate=None
        self.lock=RLock()
        self.retry_after=0.0

    def prepare(self):
        with self.lock:
            if self.delegate is not None:
                return self.delegate
            if time.monotonic()<self.retry_after:
                raise RuntimeError('Retry cooldown')
            try:
                configured=settings(os.environ)
                os.environ.update(configured)
                # Import PyTorch only after checking deployment configuration.
                import sys
                sys.path.insert(0,str(ROOT/'src'))
                import torch
                from oil_forecast.deployment import prepare_application_resources
                torch.set_num_threads(1)
                bundle=Path(configured['BRENT_BUNDLE_DIR'])
                dashboard=Path(configured['BRENT_DASHBOARD_DIR'])
                if not bundle.exists() or not dashboard.exists():
                    repo=os.environ.get('BRENT_DATA_REPO_ID','')
                    revision=os.environ.get('BRENT_DATA_REVISION','')
                    if not re.fullmatch(r'[\w.-]+/[\w.-]+',repo) or not re.fullmatch('[0-9a-f]{40}',revision):
                        raise ValueError('Missing authorized history/evidence source')
                    data_settings={'BRENT_HF_REPO_ID':repo,'BRENT_HF_REVISION':revision,'HF_TOKEN':os.environ['HF_TOKEN']}
                else:
                    data_settings={}
                # Existing registries anchor ZIP and manifest hashes; never trust remote replacements.
                prepare_application_resources(ROOT,bundle,dashboard,data_settings)
                from web.backend import Application, Study
                candidate=Application()
                # Load last-good observations, but defer provider acquisition from cold initialization.
                candidate.study=Study(initial_refresh_network=False)
                self.delegate=candidate
                return candidate
            except Exception:
                self.retry_after=time.monotonic()+30
                raise RuntimeError('Verified resources unavailable') from None

    @staticmethod
    def send(start,status,body):
        payload=json.dumps(body,ensure_ascii=False).encode('utf-8')
        start(status,[('Content-Type','application/json; charset=utf-8'),('Cache-Control','no-store'),
                      ('X-Content-Type-Options','nosniff')])
        return [payload]

    def __call__(self,env,start):
        try:
            query=parse_qs(env.get('QUERY_STRING',''),max_num_fields=24,keep_blank_values=True)
            routed=query.pop('__brent_endpoint',None)
            endpoint=(routed[0] if routed and len(routed)==1 else env.get('PATH_INFO','').removeprefix('/api/'))
            if routed is not None and len(routed)!=1:
                raise ValueError('Ambiguous route')
            if endpoint not in ENDPOINTS:
                return self.send(start,'404 Not Found',{'error':'Endpoint no encontrado'})
            if env.get('REQUEST_METHOD','GET') not in {'GET','POST'}:
                return self.send(start,'405 Method Not Allowed',{'error':'Método no permitido'})
            if endpoint=='health':
                return self.send(start,'200 OK',{'status':'ok','inference_loaded':self.delegate is not None}) if env.get('REQUEST_METHOD','GET')=='GET' else self.send(start,'405 Method Not Allowed',{'error':'GET requerido'})
            if endpoint=='refresh':
                return self.send(start,'403 Forbidden',{'error':'Actualización manual pública desactivada'})
            if env.get('REQUEST_METHOD','GET')!='GET':
                return self.send(start,'405 Method Not Allowed',{'error':'GET requerido'})
            translated=dict(env,PATH_INFO='/api/'+endpoint,QUERY_STRING=urlencode(query,doseq=True))
            return self.prepare()(translated,start)
        except ValueError:
            return self.send(start,'400 Bad Request',{'error':'Petición inválida'})
        except Exception:
            return self.send(start,'503 Service Unavailable',{'error':'Faltan permisos, autenticación o artefactos verificados de modelos/histórico. No se generan predicciones simuladas.'})


app=VercelApplication()
