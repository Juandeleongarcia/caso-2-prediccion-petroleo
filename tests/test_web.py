"""HTTP contract, real-model parity, archived evidence and remote trust boundary."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode, urlparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.backend import Application, Study, download_minimal, verify_minimal
from oil_forecast.app_data import forecast_for_model

def call(app, path, query=None, method='GET', **headers):
    status = []
    body = b''.join(app({'PATH_INFO':path, 'REQUEST_METHOD':method,
        'QUERY_STRING':urlencode(query or {}), 'HTTP_HOST':'127.0.0.1:8765', **headers}, lambda s,h:status.append((s,h))))
    return int(status[0][0].split()[0]), body

class WebApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment=patch.dict(os.environ, {'BRENT_MODEL_SOURCE':'local','BRENT_ALLOW_PROVIDER_ACCESS':'0'})
        cls.environment.start()
        cls.app=Application()
        cls.app.study=Study()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()

    def test_real_predictions_identical_for_four_models_three_horizons(self):
        study=self.app.study
        for model in ['persistence','RNN','LSTM','GRU']:
            expected=forecast_for_model(study.services,study.data['window'],model,study.updated)
            for h in [5,63,252]:
                code,body=call(self.app,'/api/predict',{'model':model,'horizon':h})
                self.assertEqual(code,200,body)
                self.assertEqual(json.loads(body)['forecast'],expected)

    def test_metrics_negative_r2_and_archive_use_original_evidence(self):
        code,body=call(self.app,'/api/comparison',{'horizon':252,'metric':'r2'})
        self.assertEqual(code,200,body)
        rows=json.loads(body)['rows'];self.assertTrue(all(r['r2']<0 for r in rows))
        self.assertEqual(min(rows,key=lambda r:r['mae'])['model'],'persistence')
        for phase in ['calibration','test']:
            code,body=call(self.app,'/api/predict',{'source':'archive','phase':phase,'horizon':252})
            self.assertEqual(code,200,body)
            self.assertEqual(json.loads(body)['forecast']['prediction_source'],'saved')

    def test_evaluation_all_experiments_architectures_and_horizons(self):
        for experiment in ['final','optimization','refinement','smoke']:
            for model in ['RNN','LSTM','GRU','persistence']:
                for h in [5,63,252]:
                    code,body=call(self.app,'/api/evaluation',{'experiment':experiment,'model':model,'horizon':h})
                    self.assertEqual(code,200,body)
                    result=json.loads(body)
                    self.assertIn('predictions',result['charts'])
                    self.assertEqual(result['scope']['horizon'],h)

    def test_static_boundary_and_invalid_controls(self):
        for path in ['/','/app.mjs','/style.css','/vendor/plotly.min.js']:
            self.assertEqual(call(self.app,path)[0],200)
        for path in ['/../data/petroleo_dataset.csv','/../.env','/../artifacts/local/brent-frozen-v1/checkpoints.pt']:
            self.assertEqual(call(self.app,path)[0],404)
        for query in [{'model':'unknown'},{'horizon':6},{'source':'future'},{'barrels':'NaN'},{'barrels':-1}]:
            self.assertEqual(call(self.app,'/api/predict',query)[0],400)
        self.assertEqual(call(self.app,'/api/refresh',method='POST')[0],403)
        self.assertEqual(call(self.app,'/api/predict',method='POST')[0],405)

    def test_market_and_models_are_dated_and_provider_access_off(self):
        code,body=call(self.app,'/api/models');self.assertEqual(code,200)
        self.assertFalse(json.loads(body)['can_refresh'])
        code,body=call(self.app,'/api/overview');self.assertEqual(code,200,body)
        self.assertIn('data_updated_at',json.loads(body)['market'])

    def test_archive_origins_and_best_error_controls_use_saved_pairs(self):
        code,body=call(self.app,'/api/archive-origins',{'model':'GRU','horizon':252})
        self.assertEqual(code,200)
        origins=json.loads(body)['origins'];self.assertEqual(len(origins),238)
        code,body=call(self.app,'/api/predict',{'model':'GRU','horizon':252,'source':'archive','origin':origins[0]})
        self.assertEqual(code,200,body)
        for order in ['worst','best']:
            code,body=call(self.app,'/api/evaluation',{'horizon':252,'case_order':order})
            self.assertEqual(code,200,body)
            errors=[x['absolute_error'] for x in json.loads(body)['cases']]
            self.assertEqual(errors,sorted(errors,reverse=order=='worst'))

    def test_hf_failure_is_explicit_and_cannot_expose_token_or_fallback(self):
        with patch('web.backend.Study',side_effect=ValueError('private-test-token')):
            code,body=call(Application(),'/api/models')
        self.assertEqual(code,503)
        self.assertNotIn(b'private-test-token',body)

class RemoteTrust(unittest.TestCase):
    def test_pinned_mock_download_matches_local_and_tampering_is_rejected(self):
        source=ROOT/'artifacts/local/hf-minimal-v1'
        trusted=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
        requests=[]
        def remote(request,timeout):
            requests.append(request)
            return io.BytesIO((source/Path(urlparse(request.full_url).path).name).read_bytes())
        with tempfile.TemporaryDirectory() as temp,patch('web.backend.build_opener') as opener:
            opener.return_value.open.side_effect=remote
            dest=Path(temp)/'revision'
            download_minimal('Juaandeleoon/caso-2-prediccion-petroleo','a'*40,trusted,dest,token='test-only-not-a-real-token')
            verify_minimal(dest,trusted)
            self.assertEqual(len(requests),6)
            self.assertTrue(all('/'+'a'*40+'/' in r.full_url for r in requests))
            (dest/'checkpoints.pt').write_bytes(b'bad')
            with self.assertRaises(ValueError):verify_minimal(dest,trusted)

    def test_mutable_revision_and_missing_hash_fail_before_network(self):
        with patch('web.backend.build_opener') as opener:
            for rev,sha in [('main','a'*64),('a'*40,None)]:
                with self.assertRaises(ValueError):download_minimal('a/b',rev,sha,Path('unused'))
            opener.assert_not_called()

    def test_redirect_never_downgrades_or_forwards_authorization(self):
        from urllib.request import Request
        from web.backend import SecureRedirect
        req=Request('https://huggingface.co/a/b/resolve/'+'a'*40+'/checkpoints.pt')
        req.add_unredirected_header('Authorization','Bearer fictitious')
        handler=SecureRedirect()
        with self.assertRaises(ValueError):handler.redirect_request(req,None,302,'',{},'http://huggingface.co/data')
        red=handler.redirect_request(req,None,302,'',{},'https://cdn-lfs.huggingface.co/data')
        self.assertIsNone(red.get_header('Authorization'))

if __name__=='__main__':unittest.main()
