"""Concrete code-only package, WSGI routing and frozen CPU inference without remote calls."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

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
    def test_linux_dependencies_pin_python_and_keep_torch_hash_without_index_shadowing(self):
        from packaging.specifiers import SpecifierSet
        folder=ROOT/'deployment/vercel'
        project=tomllib.loads((folder/'pyproject.toml').read_text(encoding='utf-8'))
        versions=SpecifierSet(project['project']['requires-python'])
        self.assertIn('3.13.7',versions)
        self.assertNotIn('3.14.8',versions)
        self.assertEqual((folder/'.python-version').read_text().strip(),'3.13')
        lines=[line.strip() for line in (folder/'requirements-linux.txt').read_text().splitlines()
               if line.strip() and not line.startswith('#')]
        self.assertIn('--index-url https://pypi.org/simple',lines)
        self.assertFalse(any('extra-index' in line or 'unsafe-best-match' in line for line in lines))
        self.assertEqual([line for line in lines if not line.startswith('-')],project['project']['dependencies'])
        constraints=[line.strip() for line in (folder/'constraints-linux.txt').read_text().splitlines()
                     if line.strip() and not line.startswith('#')]
        self.assertEqual(constraints,project['tool']['uv']['constraint-dependencies'])
        scientific=tomllib.loads((ROOT/'poetry.lock').read_text(encoding='utf-8'))
        torch=next(package for package in scientific['package'] if package['name']=='torch')
        wheel=next(item for item in torch['files'] if item['file'].endswith('cp313-cp313-manylinux_2_28_x86_64.whl'))
        direct=project['project']['dependencies'][0]
        self.assertIn(wheel['file'].replace('+','%2B'),direct)
        self.assertIn(wheel['hash'].replace('sha256:','sha256='),direct)
        lock=tomllib.loads((folder/'uv.lock').read_text(encoding='utf-8'))
        self.assertEqual(lock['requires-python'],'==3.13.*')
        locked={p['name']:p for p in lock['package']}
        self.assertEqual(locked['torch']['version'],torch['version'])
        self.assertEqual(locked['filelock']['version'],'4.2.12')
        self.assertEqual(locked['filelock']['source'],{'registry':'https://pypi.org/simple'})

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
        # Other/static-build installs root requirements.txt with uv's default
        # interpreter before the Python function builder can select 3.13.
        self.assertFalse((folder/'requirements.txt').exists())
        self.assertTrue((folder/'requirements-linux.txt').is_file())
        self.assertNotIn('installCommand',config)

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
            self.assertFalse((target/'requirements.txt').exists())
            self.assertTrue((target/'requirements-linux.txt').is_file())
            config=tomllib.loads((target/'pyproject.toml').read_text(encoding='utf-8'))
            self.assertEqual(config['project']['requires-python'],'>=3.13,<3.14')
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
                      {'HF_TOKEN':''},{'BRENT_HF_REVISION':'main'},
                      {'BRENT_HF_REVISION':'b9ad781cd90e1adcb08009358fca143e0bf0931'},
                      {'BRENT_MANIFEST_SHA256':'invalid'},
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
        with patch.dict(os.environ,{'VERCEL_SUPPORT_LARGE_FUNCTIONS':'1',
                                   'HF_TOKEN':'private-value','BRENT_PUBLIC_ACCESS_AUTHORIZED':'0'}):
            code,body=request(app,'ready');self.assertEqual(code,503)
            self.assertEqual(body['code'],'public_access_not_authorized')
            self.assertNotIn('private-value',json.dumps(body))

    def test_configuration_failure_codes_and_cooldown_never_expose_values(self):
        base={'VERCEL_SUPPORT_LARGE_FUNCTIONS':'1','BRENT_PUBLIC_ACCESS_AUTHORIZED':'1',
              'BRENT_MODEL_SOURCE':'hf','HF_TOKEN':'not-a-real-secret'}
        for changes,expected in [({'VERCEL_SUPPORT_LARGE_FUNCTIONS':'0'},'large_functions_disabled'),
                                 ({'BRENT_PUBLIC_ACCESS_AUTHORIZED':'0'},'public_access_not_authorized'),
                                 ({'HF_TOKEN':''},'model_authentication_missing'),
                                 ({'BRENT_HF_REVISION':'private-branch-value'},'model_configuration_invalid')]:
            app=VercelApplication()
            with self.subTest(expected=expected),patch.dict(os.environ,base|changes,clear=True),self.assertLogs('web.vercel_runtime',level='WARNING') as logs:
                code,body=request(app,'ready')
                self.assertEqual(code,503);self.assertEqual(body['code'],expected)
                self.assertEqual(request(app,'ready')[1]['code'],expected)
                combined=json.dumps(body)+' '.join(logs.output)
                self.assertNotIn('not-a-real-secret',combined)
                self.assertNotIn('private-branch-value',combined)
                self.assertFalse(request(app,'health')[1]['inference_loaded'])


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

    def test_empty_runtime_fetches_pinned_original_resources_then_runs_authentic_inference(self):
        # Real original bytes; only HTTPS transport is replaced. No publication or
        # real token: this isolated fixture is not a grant of production rights.
        for name in ['brent-frozen-v1','brent-dashboard-v1','models']:
            target=(self.runtime/name).resolve()
            self.assertTrue(target.is_relative_to((ROOT/'artifacts/local').resolve()))
            shutil.rmtree(target)
        calls=[]
        data_revision='d'*40
        def response(req,timeout):
            parsed=urlparse(req.full_url)
            self.assertEqual(parsed.scheme,'https');self.assertEqual(parsed.hostname,'huggingface.co')
            self.assertFalse(parsed.query);self.assertFalse(parsed.fragment)
            self.assertEqual(req.get_header('Authorization'),'Bearer dummy-not-real')
            self.assertNotIn('dummy-not-real',req.full_url)
            name=Path(parsed.path).name
            revision=parsed.path.split('/resolve/')[1].split('/')[0]
            if name.endswith('.zip'):
                self.assertEqual(revision,data_revision)
                payload=(ROOT/'artifacts/releases'/name).read_bytes()
            else:
                self.assertEqual(revision,MODEL_REVISION)
                payload=(ROOT/'artifacts/local/hf-minimal-v1'/name).read_bytes()
            calls.append((revision,name))
            stream=io.BytesIO(payload);stream.geturl=lambda:req.full_url
            return stream
        with patch.dict(os.environ,{'BRENT_DATA_REPO_ID':'Juaandeleoon/caso-2-prediccion-petroleo',
                                    'BRENT_DATA_REVISION':data_revision}),\
             patch('web.vercel_runtime.ROOT',ROOT/'deployment/vercel-app'),\
             patch('oil_forecast.artifacts.build_opener') as data_network,\
             patch('web.backend.build_opener') as model_network:
            data_network.return_value.open.side_effect=response
            model_network.return_value.open.side_effect=response
            code,ready=request(self.app,'ready')
            self.assertEqual(code,200);self.assertEqual(ready['source'],'hf')
            self.assertEqual(len(calls),8)  # Two ZIPs and six minimal model files.
            study=self.app.delegate.study
            from oil_forecast.app_data import load_services,forecast_for_model
            original=load_services(ROOT/'artifacts/local/brent-frozen-v1')
            for model in ['RNN','LSTM','GRU','persistence']:
                expected=forecast_for_model(original,study.data['window'],model,study.updated)
                network=study.services['LSTM' if model=='persistence' else model]._model
                with patch.object(network,'forward',wraps=network.forward) as forward:
                    for horizon in [5,63,252]:
                        code,body=request(self.app,'predict',f'model={model}&horizon={horizon}')
                        self.assertEqual(code,200)
                        self.assertEqual(body['forecast']['forecasts'],expected['forecasts'])
                        self.assertEqual(body['forecast']['origin_date'],expected['origin_date'])
                    if model!='persistence':
                        self.assertGreaterEqual(forward.call_count,3)  # Real neural execution, not saved pairs.
            for endpoint in ['models','overview','comparison','evaluation']:
                self.assertEqual(request(self.app,endpoint)[0],200)
            self.assertTrue(request(self.app,'health')[1]['inference_loaded'])
            self.assertEqual(len(calls),8)  # Warm requests reuse verified resources.
            self.assertEqual(request(self.app,'refresh',method='POST')[0],403)

    def test_missing_historical_data_returns_503_without_fabricating_or_falling_back(self):
        removed=(self.runtime/'brent-dashboard-v1').resolve()
        self.assertTrue(removed.is_relative_to((ROOT/'artifacts/local').resolve()))
        shutil.rmtree(removed)  # Only a temporary test-owned copy.
        with patch.dict(os.environ,{'BRENT_DATA_REPO_ID':'a/b','BRENT_DATA_REVISION':'a'*40}),patch('oil_forecast.deployment.prepare_application_resources') as downloader:
            downloader.side_effect=FileNotFoundError('private test URL')
            code,body=request(self.app,'ready')
        self.assertEqual(code,503);self.assertNotIn('private test URL',json.dumps(body))
        self.assertEqual(body['code'],'history_artifacts')
        self.assertFalse(request(self.app,'health')[1]['inference_loaded'])

    def test_missing_history_source_has_specific_safe_code(self):
        removed=(self.runtime/'brent-dashboard-v1').resolve()
        self.assertTrue(removed.is_relative_to((ROOT/'artifacts/local').resolve()))
        shutil.rmtree(removed)
        with patch.dict(os.environ,{'BRENT_DATA_REPO_ID':'','BRENT_DATA_REVISION':''}):
            code,body=request(self.app,'ready')
        self.assertEqual(code,503);self.assertEqual(body['code'],'history_source_missing')

    def test_corrupt_registered_data_is_rejected_without_replacing_originals(self):
        path=self.runtime/'brent-frozen-v1/brent_history.csv';path.write_bytes(b'corrupt')
        code,body=request(self.app,'predict')
        self.assertEqual(code,503);self.assertEqual(body['code'],'history_artifacts')
        self.assertEqual(path.read_bytes(),b'corrupt')

    def test_hf_unavailable_preserves_existing_data_and_never_uses_other_weights(self):
        original=hashlib.sha256((self.runtime/'brent-frozen-v1/checkpoints.pt').read_bytes()).hexdigest()
        with patch('web.backend.download_minimal',side_effect=ValueError('signed-private-token')),self.assertLogs('web.vercel_runtime',level='WARNING') as logs:
            code,body=request(self.app,'predict')
        self.assertEqual(code,503);self.assertNotIn('signed-private-token',json.dumps(body))
        self.assertEqual(body['code'],'model_artifacts')
        self.assertNotIn('signed-private-token',' '.join(logs.output))
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
