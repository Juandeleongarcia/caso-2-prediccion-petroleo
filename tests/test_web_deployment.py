"""Deployment boundaries and authentic minimal-checkpoint parity; no providers."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from web.backend import Application, Study, download_minimal
from web.production import validate_environment

def request(app,path,method='GET',origin=None):
    result=[]
    body=b''.join(app({'PATH_INFO':path,'REQUEST_METHOD':method,'QUERY_STRING':'',
        'HTTP_HOST':'localhost:8765','HTTP_ORIGIN':origin},lambda status,headers:result.append((status,dict(headers)))))
    return result[0],json.loads(body)

class Deployment(unittest.TestCase):
    def test_public_runtime_rejects_manual_refresh_even_with_forged_local_host(self):
        app=Application();app.study=object()
        with patch.dict(os.environ,{'BRENT_PUBLIC_ACCESS_AUTHORIZED':'1','BRENT_ALLOW_PROVIDER_ACCESS':'1'}):
            self.assertTrue(request(app,'/api/refresh','POST')[0][0].startswith('403'))

    def test_no_repeated_network_after_model_cache_verified(self):
        source=ROOT/'artifacts/local/hf-minimal-v1'
        sha=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
        with patch('web.backend.build_opener') as network:
            self.assertEqual(download_minimal('a/b','a'*40,sha,source),source)
            network.assert_not_called()

    def test_retry_cooldown_and_readiness_do_not_claim_unavailable_inference(self):
        app=Application()
        with patch('web.backend.Study',side_effect=RuntimeError('secret')),patch('web.backend.time.monotonic',return_value=10) as clock:
            for _ in range(2):
                (status,headers),body=request(app,'/api/ready')
                self.assertTrue(status.startswith('503'))
                self.assertNotIn('secret',json.dumps(body))
            from web import backend
            self.assertEqual(backend.Study.call_count,1)
            clock.return_value=41
            request(app,'/api/ready')
            self.assertEqual(backend.Study.call_count,2)
        (status,_),body=request(app,'/api/health')
        self.assertTrue(status.startswith('200'));self.assertFalse(body['inference_loaded'])
        self.assertTrue(request(app,'/api/health','POST')[0][0].startswith('405'))

    def test_cors_only_exact_configured_origin_and_no_cache(self):
        with patch.dict(os.environ,{'BRENT_WEB_ORIGIN':'https://brent.example.org'}):
            for origin in ['https://brent.example.org','https://attacker.example.org',None]:
                (status,headers),_=request(Application(),'/api/health',origin=origin)
                self.assertEqual(headers.get('Access-Control-Allow-Origin'),origin if origin=='https://brent.example.org' else None)
                self.assertEqual(headers['Cache-Control'],'no-store')

    def test_public_runtime_requires_permissions_private_auth_pins_and_paths(self):
        valid={'BRENT_PUBLIC_ACCESS_AUTHORIZED':'1','BRENT_MODEL_SOURCE':'hf','BRENT_HF_REPO_ID':'a/b',
            'BRENT_HF_REVISION':'a'*40,'BRENT_MANIFEST_SHA256':'b'*64,'HF_TOKEN':'dummy-not-a-token',
            'BRENT_WEB_ORIGIN':'https://brent.example.org','BRENT_BUNDLE_DIR':'/bundles/frozen',
            'BRENT_DASHBOARD_DIR':'/bundles/dashboard','BRENT_LIVE_DIR':'/tmp/live'}
        settings=validate_environment(valid)
        self.assertEqual((settings['host'],settings['port'],settings['threads']),('127.0.0.1',8000,2))
        for key in ['BRENT_PUBLIC_ACCESS_AUTHORIZED','BRENT_MODEL_SOURCE','BRENT_HF_REVISION','BRENT_MANIFEST_SHA256',
                    'HF_TOKEN','BRENT_WEB_ORIGIN','BRENT_BUNDLE_DIR','BRENT_DASHBOARD_DIR','BRENT_LIVE_DIR']:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_environment(valid|{key:''})

    def test_hf_path_loads_authentic_cpu_models_and_matches_original_services(self):
        from oil_forecast.app_data import forecast_for_model
        source=ROOT/'artifacts/local/hf-minimal-v1'
        sha=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
        with patch.dict(os.environ,{'BRENT_MODEL_SOURCE':'local','BRENT_ALLOW_PROVIDER_ACCESS':'0'}):
            original=Study()
        def remote(req,timeout):
            return io.BytesIO((source/Path(urlparse(req.full_url).path).name).read_bytes())
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'BRENT_MODEL_SOURCE':'hf',
            'BRENT_HF_REPO_ID':'Juaandeleoon/caso-2-prediccion-petroleo','BRENT_HF_REVISION':'a'*40,
            'BRENT_MANIFEST_SHA256':sha,'BRENT_MODEL_CACHE_DIR':folder,'BRENT_ALLOW_PROVIDER_ACCESS':'0'}),patch('web.backend.build_opener') as network:
            network.return_value.open.side_effect=remote
            candidate=Study()
            for architecture in ['RNN','LSTM','GRU']:
                self.assertEqual(str(next(candidate.services[architecture]._model.parameters()).device),'cpu')
                expected=forecast_for_model(original.services,original.data['window'],architecture,original.updated)
                actual=forecast_for_model(candidate.services,original.data['window'],architecture,original.updated)
                # Supplementary metrics are loaded separately; scientific outputs must match.
                for key in ['forecasts','reference_brent','origin_date','model','seed']:
                    self.assertEqual(actual[key],expected[key])
            app=Application();app.study=candidate
            self.assertEqual(request(app,'/api/ready')[1]['source'],'hf')

if __name__=='__main__':unittest.main()
