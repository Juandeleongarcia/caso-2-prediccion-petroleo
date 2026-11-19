"""Offline daily publication contracts. Synthetic fixtures are never production data."""
import copy
from datetime import datetime,timezone
import hashlib
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from oil_forecast.acquisition import sha256,validate_observations
from oil_forecast.temporal import apply_availability,build_features,build_target
from oil_forecast.live_data import build_daily,validate_live,merge_observations
from oil_forecast.live_hub import LiveHubController,HubReader,fetch_latest,pointer_for,FILES
from scripts.update_live_data import publish,run,main
from oil_forecast.acquisition import safe_acquisition_error

MODEL_SHA='a'*64


class LiveContracts(unittest.TestCase):
    def setUp(self):
        (ROOT/'artifacts/local').mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/local')
        self.directory=Path(self.temp.name)
        self.config=json.loads((ROOT/'configs/data.json').read_text())
        self.config.update(start='2026-01-01',end='2026-10-30')
        self.names=[n for n in self.config['sources'] if n!='opep_orb']
        self.now=pd.Timestamp('2026-11-02T09:00:00Z')
        raw=self.observations('2026-10-30')
        available=apply_availability(raw,self.config['sources'])
        self.window=build_features(available,'2026-01-01','2026-10-30').tail(60).reset_index(drop=True)
        self.history=build_target(available,'2026-01-01','2026-10-30')

    def tearDown(self):self.temp.cleanup()

    def observations(self,end):
        dates=pd.bdate_range('2026-01-01',end)
        records=[]
        for i,name in enumerate(self.names):
            spec=self.config['sources'][name]
            for j,date in enumerate(dates):
                records.append({'observation_date':date,'variable':name,'value':70+i+j/100,
                    'source':('FRED:' if spec['provider']=='fred' else 'Yahoo:')+spec['id']})
        return validate_observations(pd.DataFrame(records))

    def download(self,config,directory,cache):
        directory.mkdir(parents=True)
        frame=self.observations(config['end'])
        frame.to_csv(directory/'observations.csv',index=False)
        meta={'downloaded_at':self.now.isoformat(),'observations_sha256':sha256(directory/'observations.csv'),
              'sources':{name:{'status':'ok'} for name in self.names},'required_sources_unavailable':[]}
        (directory/'manifest.json').write_text(json.dumps(meta))

    def build(self,name='live-test',previous=None):
        folder=self.directory/name
        build_daily(self.config,folder,self.window,self.history,'2026-10-30T20:00:00Z',MODEL_SHA,
                    previous=previous,downloader=self.download,now=self.now)
        return folder

    def reader(self,folder):
        pointer=(json.dumps(pointer_for(folder))+'\n').encode()
        class Reader:
            def head(self):return 'b'*40
            def read(self,revision,path):
                if revision!='b'*40:raise AssertionError('Mutable revision used')
                if path=='live/latest.json':return pointer
                return (folder/Path(path).name).read_bytes()
        return Reader()

    def test_append_only_first_seen_policy_and_all_schemas(self):
        folder=self.build()
        data=validate_live(folder,self.config,self.now)
        features=pd.read_csv(folder/'features.csv')
        self.assertEqual(list(features),list(self.window))
        self.assertEqual(len(features),61)
        for column in self.window:
            # CSV round trip may normalize date/dtype representations only.
            before=io.StringIO();self.window.to_csv(before,index=False)
            expected=pd.read_csv(io.StringIO(before.getvalue()))
            actual=features[column].iloc[:60].reset_index(drop=True)
            want=expected[column]
            if column=='decision_at' or column.endswith('__available_at'):
                actual=pd.to_datetime(actual,utc=True,format='mixed');want=pd.to_datetime(want,utc=True,format='mixed')
            pd.testing.assert_series_equal(actual,want,check_names=False)
        actual=pd.read_csv(folder/'target.csv').iloc[:len(self.history)].reset_index(drop=True)
        expected=pd.read_csv(io.StringIO(self.history.to_csv(index=False)))
        for frame in [actual,expected]:frame['label_available_at']=pd.to_datetime(frame.label_available_at,utc=True,format='mixed')
        pd.testing.assert_frame_equal(actual,expected)
        raw=pd.read_csv(folder/'observations.csv')
        new=raw.loc[raw.first_seen_at.notna()]
        self.assertTrue((pd.to_datetime(new.available_at,utc=True)>=pd.to_datetime(new.first_seen_at,utc=True)).all())
        self.assertEqual(features.brent.iloc[-1],self.window.brent.iloc[-1])
        self.assertFalse(data['protocol']['weights_updated'])
        self.assertFalse(data['protocol']['historical_point_in_time'])

    def test_revisions_duplicate_and_future_observations_are_not_backdated(self):
        folder=self.build()
        old=pd.read_csv(folder/'observations.csv')
        incoming=self.observations('2026-11-02')
        index=incoming.loc[incoming.variable.eq('brent') & incoming.observation_date.eq(pd.Timestamp('2026-10-16'))].index[0]
        incoming.loc[index,'value']=999.
        merged,count=merge_observations(old,incoming,self.config,self.now)
        self.assertGreater(count,0)
        self.assertNotEqual(float(merged.loc[merged.variable.eq('brent') & merged.observation_date.eq(pd.Timestamp('2026-10-16')),'value'].iloc[0]),999.)
        with self.assertRaises(ValueError):merge_observations(old,pd.concat([incoming,incoming.iloc[[0]]]),self.config,self.now)
        with self.assertRaises(ValueError):merge_observations(old,self.observations('2026-11-03'),self.config,self.now)

    def test_missing_provider_and_corrupt_payload_never_publish(self):
        def failed(config,folder,cache):
            self.download(config,folder,cache)
            path=folder/'manifest.json';meta=json.loads(path.read_text())
            meta['sources']['oro']['status']='unavailable';path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            build_daily(self.config,self.directory/'live-bad',self.window,self.history,
                        '2026-10-30T20:00:00Z',MODEL_SHA,downloader=failed,now=self.now)
        folder=self.build();(folder/'features.csv').write_text('corrupt')
        with self.assertRaises(ValueError):validate_live(folder,self.config,self.now)

    def test_remote_refresh_reuses_last_good_and_rejects_rollback_or_wrong_models(self):
        folder=self.build()
        reader=self.reader(folder)
        controller=LiveHubController(self.directory/'consumer',reader,MODEL_SHA,clock=lambda:self.now.to_pydatetime())
        data,status=controller.check(self.config,lambda window:None)
        self.assertEqual(status['status'],'current');self.assertEqual(status['published_commit'],'b'*40)
        with patch.object(reader,'head',side_effect=RuntimeError('private signed URL')):
            controller.next_check=0
            kept,status=controller.check(self.config,lambda window:None)
        self.assertEqual(kept['directory'],data['directory']);self.assertEqual(status['status'],'failed')
        self.assertNotIn('private',json.dumps(status))
        with self.assertRaises(ValueError):fetch_latest(reader,self.directory/'wrong',self.config,'c'*64,now=self.now)
        controller.data={**data,'protocol':{**data['protocol'],'data_updated_at':'2026-11-03T09:00:00+00:00'}}
        controller.clock=lambda:pd.Timestamp('2026-11-04T09:00:00Z').to_pydatetime();controller.next_check=0
        kept,status=controller.check(self.config,lambda window:None)
        self.assertEqual(status['status'],'failed')
        self.assertEqual(kept['protocol']['data_updated_at'],'2026-11-03T09:00:00+00:00')

    def test_old_seed_bootstrap_defers_origin_until_first_seen_is_legitimately_available(self):
        self.now=pd.Timestamp('2026-12-15T09:00:00Z')
        first=self.build('live-bootstrap')
        meta=json.loads((first/'manifest.json').read_text())
        self.assertTrue(meta['origin_advance_deferred'])
        validate_live(first,self.config,self.now)
        self.now=pd.Timestamp('2026-12-16T09:00:00Z')
        second=self.build('live-next',previous=first)
        data=validate_live(second,self.config,self.now)
        self.assertFalse(data['protocol']['origin_advance_deferred'])
        self.assertEqual(str(data['window'].decision_date.iloc[-1])[:10],'2026-12-16')

    def test_publication_is_single_compare_and_swap_under_live_prefix_only(self):
        folder=self.build()
        contents={f'live/snapshots/{folder.name}/{name}':(folder/name).read_bytes() for name in FILES}
        contents['live/latest.json']=(json.dumps(pointer_for(folder),sort_keys=True)+'\n').encode()
        class API:
            def model_info(self,*args,**kwargs):return SimpleNamespace(private=True,sha='b'*40)
            def file_exists(self,*args,**kwargs):return False
            def create_commit(api,**kwargs):
                self.assertEqual(kwargs['parent_commit'],'b'*40)
                self.assertEqual(len(kwargs['operations']),5)
                for op in kwargs['operations']:
                    self.assertTrue(op.path_in_repo.startswith('live/'))
                    self.assertNotIn('checkpoints.pt',op.path_in_repo)
                return SimpleNamespace(oid='c'*40)
        with patch('scripts.update_live_data.HubReader') as reader:
            reader.return_value.read.side_effect=lambda revision,path:contents[path]
            result=publish(folder,'a/b','dummy','b'*40,api=API(),operation_factory=lambda **kw:SimpleNamespace(**kw))
        self.assertEqual(result,'c'*40)
        with self.assertRaises(ValueError):publish(folder,'a/b','dummy','d'*40,api=API(),operation_factory=lambda **kw:SimpleNamespace(**kw))
        with self.assertRaises(ValueError):run({},self.directory/'denied')

    def test_fixed_paths_and_workflow_has_no_data_artifacts_or_training(self):
        reader=HubReader('a/b','dummy')
        for path in ['checkpoints.pt','../secrets','live/snapshots/a/../../weights.pt']:
            with self.assertRaises(ValueError):reader.read('b'*40,path)
        workflow=(ROOT/'.github/workflows/update-live-data.yml').read_text()
        self.assertIn('workflow_dispatch:',workflow);self.assertIn('cron:',workflow)
        self.assertIn('contents: read',workflow);self.assertNotIn('uses: actions/upload-artifact',workflow)
        self.assertNotIn('run_final_evaluation.py',workflow)

    def test_provider_error_diagnostics_exclude_urls_tokens_and_exception_strings(self):
        error=HTTPError('https://private.example/?token=secret-value',503,'secret-value',{},None)
        diagnostic=safe_acquisition_error(error)
        self.assertEqual(diagnostic,{'error_type':'HTTPError','reason':'http_error','http_status':503})
        self.assertNotIn('secret-value',json.dumps(diagnostic))

    def test_failed_command_has_safe_stage_source_and_summary(self):
        folder=self.directory/'run';raw=folder/'live-example/raw';raw.mkdir(parents=True)
        (raw/'manifest.json').write_text(json.dumps({'sources':{'brent':{
            'status':'unavailable','error':'https://private/?secret=hidden-token',
            'diagnostic':{'error_type':'HTTPError','reason':'http_error','http_status':503}}}}))
        summary=self.directory/'summary.md'
        def failed(env,workdir,publish_requested,diagnostics):
            diagnostics['stage']='provider_acquisition_and_dataset_build'
            raise RuntimeError('hidden-token in a private exception')
        output=io.StringIO()
        with patch('scripts.update_live_data.run',side_effect=failed),\
             patch.dict('os.environ',{'GITHUB_STEP_SUMMARY':str(summary)}),\
             patch.object(sys,'argv',['update_live_data.py','--workdir',str(folder)]),\
             redirect_stdout(output),self.assertRaises(SystemExit) as exit:
            main()
        self.assertEqual(exit.exception.code,1)
        report=json.loads(output.getvalue())
        self.assertEqual(report['stage'],'provider_acquisition_and_dataset_build')
        self.assertEqual(report['sources']['brent']['http_status'],503)
        self.assertFalse(report['publish_requested'])
        self.assertNotIn('hidden-token',output.getvalue()+summary.read_text())


if __name__=='__main__':unittest.main()
