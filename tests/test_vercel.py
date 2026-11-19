"""Concrete code-only package, WSGI routing and frozen CPU inference without remote calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from scripts.prepare_vercel import prepare, source_files
from web.vercel_runtime import VercelApplication, MODEL_REVISION, settings


def request(app,endpoint,query='',method='GET'):
    response=[]
    body=b''.join(app({'PATH_INFO':'/api/forecast','REQUEST_METHOD':method,
        'QUERY_STRING':'__brent_endpoint='+endpoint+('&'+query if query else '')},
        lambda s,h:response.append((int(s.split()[0]),dict(h)))))
    return response[0][0],json.loads(body)


class VercelPackage(unittest.TestCase):
    def test_git_import_snapshot_tracks_current_sources_and_has_automatic_deploys_disabled(self):
        folder=ROOT/'deployment/vercel-app'
        manifest=json.loads((folder/'source_manifest.json').read_text(encoding='utf-8'))
        from scripts.prepare_vercel import TEMPLATE_FILES
        for name in source_files():
            expected=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            self.assertEqual((folder/name).read_bytes(),expected,name)
        for name in TEMPLATE_FILES:
            expected=(ROOT/'deployment/vercel'/name).read_bytes().replace(b'\r\n',b'\n')
            self.assertEqual((folder/name).read_bytes(),expected,name)
        for name,sha in manifest['files'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha,name)
        config=json.loads((folder/'vercel.json').read_text(encoding='utf-8'))
        self.assertFalse(config['git']['deploymentEnabled'])
        self.assertFalse((folder/'public').exists())

    def test_only_runtime_closure_public_assets_and_trusted_registries_packaged(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/local') as folder:
            target=Path(folder)/'project'
            result=prepare(target)
            self.assertGreater(result['files'],20)
            manifest=json.loads((target/'source_manifest.json').read_text())
            self.assertFalse(manifest['private_artifacts_included'])
            for name,sha in manifest['files'].items():
                self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(),sha)
                self.assertNotIn(name.split('/')[0],['data','experiments','.venv','presentacion'])
                self.assertNotIn(Path(name).suffix,['.pt','.zip','.onnx'])
            self.assertNotIn('src/oil_forecast/app_pages.py',source_files())
            for name in ['src/oil_forecast/models.py','src/oil_forecast/preprocessing.py','web/backend.py']:
                self.assertEqual((target/name).read_bytes(),(ROOT/name).read_bytes().replace(b'\r\n',b'\n'))
            self.assertFalse((target/'poetry.lock').exists())
            self.assertFalse((target/'pyproject.toml').exists())
            with self.assertRaises(ValueError):prepare(target)

    def test_fluid_duration_routes_and_cdn_build_use_same_origin_without_tokens(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/local') as folder:
            target=Path(folder)/'project';prepare(target)
            config=json.loads((target/'vercel.json').read_text())
            self.assertTrue(config['fluid']);self.assertEqual(config['functions']['api/forecast.py']['maxDuration'],300)
            self.assertFalse(config['git']['deploymentEnabled'])
            self.assertEqual(config['outputDirectory'],'public')
            self.assertEqual(len(config['rewrites']),9)
            env=dict(os.environ,VERCEL_SUPPORT_LARGE_FUNCTIONS='1',HF_TOKEN='dummy-not-an-actual-token')
            result=subprocess.run(['node','build.mjs'],cwd=target,env=env,capture_output=True,encoding='utf-8',timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            public=[p for p in (target/'public').rglob('*') if p.is_file()]
            self.assertEqual(len(public),7)
            self.assertIn("window.BRENT_API_BASE = ''",(target/'public/config.js').read_text())
            self.assertNotIn('dummy-not-an-actual-token',''.join(p.read_text(encoding='utf-8') for p in public))
            self.assertGreater((target/'public/vendor/plotly.min.js').stat().st_size,4_500_000)

    def test_configuration_requires_beta_permission_auth_and_fixed_revision(self):
        base={'VERCEL_SUPPORT_LARGE_FUNCTIONS':'1','BRENT_PUBLIC_ACCESS_AUTHORIZED':'1','HF_TOKEN':'dummy'}
        self.assertEqual(settings(base)['BRENT_HF_REVISION'],MODEL_REVISION)
        for extra in [{'VERCEL_SUPPORT_LARGE_FUNCTIONS':'0'},{'BRENT_PUBLIC_ACCESS_AUTHORIZED':'0'},
                      {'HF_TOKEN':''},{'BRENT_HF_REVISION':'main'},{'BRENT_MANIFEST_SHA256':'invalid'},
                      {'BRENT_MODEL_SOURCE':'local'},{'BRENT_RUNTIME_DIR':'relative/path'}]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):settings(base|extra)
        with patch('web.vercel_runtime.tempfile.gettempdir',return_value=str(ROOT/'artifacts/local')):
            with self.assertRaises(ValueError):settings(base|{'VERCEL':'1','BRENT_RUNTIME_DIR':str(ROOT/'data')})

    def test_route_errors_and_health_do_not_load_models_or_expose_secrets(self):
        app=VercelApplication()
        self.assertEqual(request(app,'health'),(200,{'status':'ok','inference_loaded':False}))
        self.assertEqual(request(app,'health',method='POST')[0],405)
        self.assertEqual(request(app,'health','__brent_endpoint=predict')[0],400)
        self.assertEqual(request(app,'../../secrets')[0],404)
        with patch.dict(os.environ,{'HF_TOKEN':'private-value','BRENT_PUBLIC_ACCESS_AUTHORIZED':'0'}):
            code,body=request(app,'ready');self.assertEqual(code,503)
            self.assertNotIn('private-value',json.dumps(body))


class VercelResources(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/local');self.runtime=Path(self.temp.name)
        for name in ['brent-frozen-v1','brent-dashboard-v1']:
            shutil.copytree(ROOT/'artifacts/local'/name,self.runtime/name)
        shutil.copytree(ROOT/'artifacts/local/hf-minimal-v1',self.runtime/'models'/MODEL_REVISION)
        self.env=patch.dict(os.environ,{'VERCEL_SUPPORT_LARGE_FUNCTIONS':'1','VERCEL':'0',
            'BRENT_PUBLIC_ACCESS_AUTHORIZED':'1','BRENT_MODEL_SOURCE':'hf','HF_TOKEN':'dummy-not-real',
            'BRENT_RUNTIME_DIR':str(self.runtime),'BRENT_ALLOW_PROVIDER_ACCESS':'0'})
        self.env.start();self.app=VercelApplication()

    def tearDown(self):
        self.env.stop();self.temp.cleanup()

    def test_all_api_views_and_three_cpu_models_match_streamlit_with_same_inputs(self):
        from oil_forecast.app_data import load_services,forecast_for_model
        for endpoint in ['ready','models','overview','comparison','evaluation']:
            self.assertEqual(request(self.app,endpoint)[0],200,endpoint)
        study=self.app.delegate.study
        original=load_services(ROOT/'artifacts/local/brent-frozen-v1')
        for model in ['RNN','LSTM','GRU','persistence']:
            expected=forecast_for_model(original,study.data['window'],model,study.updated)
            for h in [5,63,252]:
                code,body=request(self.app,'predict',f'model={model}&horizon={h}')
                self.assertEqual(code,200)
                for key in ['forecasts','origin_date','reference_brent']:
                    self.assertEqual(body['forecast'][key],expected[key])
        self.assertEqual(request(self.app,'refresh',method='POST')[0],403)

    def test_missing_historical_data_returns_503_without_fabricating_or_falling_back(self):
        removed=(self.runtime/'brent-dashboard-v1').resolve()
        self.assertTrue(removed.is_relative_to((ROOT/'artifacts/local').resolve()))
        shutil.rmtree(removed)  # Only a temporary test-owned copy.
        with patch.dict(os.environ,{'BRENT_DATA_REPO_ID':'a/b','BRENT_DATA_REVISION':'a'*40}),patch('oil_forecast.deployment.prepare_application_resources') as downloader:
            downloader.side_effect=FileNotFoundError('private test URL')
            code,body=request(self.app,'ready')
        self.assertEqual(code,503);self.assertNotIn('private test URL',json.dumps(body))
        self.assertFalse(request(self.app,'health')[1]['inference_loaded'])

    def test_corrupt_registered_data_is_rejected_without_replacing_originals(self):
        path=self.runtime/'brent-frozen-v1/brent_history.csv';path.write_bytes(b'corrupt')
        self.assertEqual(request(self.app,'predict')[0],503)
        self.assertEqual(path.read_bytes(),b'corrupt')

    def test_hf_unavailable_preserves_existing_data_and_never_uses_other_weights(self):
        original=hashlib.sha256((self.runtime/'brent-frozen-v1/checkpoints.pt').read_bytes()).hexdigest()
        with patch('web.backend.download_minimal',side_effect=ValueError('signed-private-token')):
            code,body=request(self.app,'predict')
        self.assertEqual(code,503);self.assertNotIn('signed-private-token',json.dumps(body))
        self.assertEqual(hashlib.sha256((self.runtime/'brent-frozen-v1/checkpoints.pt').read_bytes()).hexdigest(),original)

    def test_provider_failure_retains_verified_window_reference_and_predictions(self):
        self.assertEqual(request(self.app,'ready')[0],200)
        before=request(self.app,'predict')[1]['forecast']
        study=self.app.delegate.study
        with patch.dict(os.environ,{'BRENT_ALLOW_PROVIDER_ACCESS':'1'}),patch.object(study.controller,'check',return_value={'status':'failed','data':None}):
            self.assertEqual(request(self.app,'overview')[0],200)
            after=request(self.app,'predict')[1]['forecast']
        self.assertEqual(before,after)
        self.assertEqual(study.refresh_status['status'],'failed')


if __name__=='__main__':unittest.main()
