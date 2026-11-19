"""Daily observations only. Explicit authorization, CPU validation, atomic live/ commit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import sys
from urllib.error import HTTPError
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from oil_forecast.artifacts import fetch_bundle
from oil_forecast.deployment import hf_artifact_url
from oil_forecast.live_data import build_daily,validate_live
from oil_forecast.live_hub import FILES,HubReader,fetch_latest,pointer_for
from oil_forecast.app_data import load_app_data,load_services
from web.vercel_runtime import MODEL_REPO,MODEL_REVISION,MODEL_MANIFEST


def publish(folder,repo,token,parent,api=None,operation_factory=None):
    """Only add this new live snapshot and pointer; never delete or upload weights."""
    if api is None or operation_factory is None:
        from huggingface_hub import HfApi,CommitOperationAdd
        api=api or HfApi(token=token)
        operation_factory=operation_factory or CommitOperationAdd
    info=api.model_info(repo,revision='main')
    if info.private is not True or info.sha!=parent:raise ValueError('Private unchanged destination required')
    folder=Path(folder)
    prefix=f'live/snapshots/{folder.name}/'
    if api.file_exists(repo,prefix+'manifest.json',revision=parent):raise ValueError('Snapshot already published')
    pointer=(json.dumps(pointer_for(folder),sort_keys=True,allow_nan=False)+'\n').encode()
    operations=[operation_factory(path_in_repo=prefix+name,path_or_fileobj=folder/name) for name in FILES]
    operations.append(operation_factory(path_in_repo='live/latest.json',path_or_fileobj=pointer))
    allowed={prefix+name for name in FILES}|{'live/latest.json'}
    if {op.path_in_repo for op in operations}!=allowed:raise ValueError('Invalid upload inventory')
    commit=api.create_commit(repo_id=repo,repo_type='model',revision='main',parent_commit=parent,
        operations=operations,commit_message='Publish validated live observations '+folder.name)
    reader=HubReader(repo,token)
    for name in FILES:
        content=reader.read(commit.oid,prefix+name)
        if hashlib.sha256(content).digest()!=hashlib.sha256((folder/name).read_bytes()).digest():
            raise ValueError('Published data verification failed')
    if reader.read(commit.oid,'live/latest.json')!=pointer:raise ValueError('Published pointer verification failed')
    return commit.oid


def run(env,workdir,publish_requested=False,diagnostics=None):
    diagnostics=diagnostics if diagnostics is not None else {}
    diagnostics['stage']='authorization'
    if env.get('BRENT_LIVE_UPDATE_AUTHORIZED')!='1':
        raise ValueError('Daily acquisition and distribution authorization required')
    token=env.get('HF_TOKEN')
    if not token:raise ValueError('A private server token is required')
    repo=env.get('BRENT_LIVE_REPO_ID',MODEL_REPO)
    reader=HubReader(repo,token)
    diagnostics['stage']='hf_repository_metadata'
    parent=reader.head()
    # Resolve the mutable branch ONCE. Everything read/written below is anchored
    # to this commit; a concurrent publication makes the atomic CAS fail safely.
    class Pinned:
        def head(self):return parent
        def read(self,revision,path):return reader.read(revision,path)
    workdir=Path(workdir).resolve()
    if not workdir.is_relative_to((ROOT/'artifacts/local').resolve()) or workdir.exists():
        raise ValueError('Use a new ignored actions working directory')
    workdir.mkdir(parents=True)
    diagnostics['stage']='hf_frozen_archive'
    registry=json.loads((ROOT/'configs/serving_bundle.json').read_text())
    base=workdir/'frozen'
    data_repo=env.get('BRENT_DATA_REPO_ID',MODEL_REPO)
    data_revision=env.get('BRENT_DATA_REVISION','')
    fetch_bundle(hf_artifact_url(data_repo,data_revision,'brent-frozen-v1.zip'),base,registry['archive']['sha256'],token=token)
    diagnostics['stage']='frozen_models_cpu_load'
    services=load_services(base)
    # The frozen archive is independently registered and contains the exact same
    # operational model payload as the six-file repository; no training occurs.
    config=json.loads((ROOT/'configs/data.json').read_text())
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    now=datetime.now(timezone.utc)
    def validator(window):
        for service in services.values():service.predict(window,data_updated_at=pd.Timestamp.now(tz='UTC'))
    previous=None
    diagnostics['stage']='previous_live_snapshot'
    try:reader.read(parent,'live/latest.json')
    except HTTPError as error:
        if error.code!=404:raise
    else:
        previous=fetch_latest(Pinned(),workdir/'previous',config,MODEL_MANIFEST,validator,now)['directory']
    seed_window=pd.read_csv(base/'latest_window.csv');seed_history=pd.read_csv(base/'brent_history.csv')
    protocol=json.loads((base/'protocol.json').read_text())
    identifier='live-'+now.strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
    destination=workdir/identifier
    diagnostics['stage']='provider_acquisition_and_dataset_build'
    build_daily(config,destination,seed_window,seed_history,protocol['data_updated_at'],MODEL_MANIFEST,previous=previous)
    diagnostics['stage']='snapshot_and_neural_validation'
    data=validate_live(destination,config,datetime.now(timezone.utc),validator)
    report={'status':'validated','snapshot_id':identifier,'data_updated_at':data['protocol']['data_updated_at'],
            'weights_updated':False,'ignored_revision_count':data['protocol']['ignored_revision_count'],
            'latest_decision_date':str(data['window'].decision_date.iloc[-1])[:10],
            'origin_advance_deferred':data['protocol']['origin_advance_deferred']}
    if publish_requested:
        diagnostics['stage']='publication_and_remote_integrity'
        report['published_commit']=publish(destination,repo,token,parent)
        report['status']='published_and_verified'
    return report


def safe_source_statuses(workdir):
    folder=Path(workdir).resolve()
    if not folder.is_relative_to((ROOT/'artifacts/local').resolve()):return {}
    names=set(json.loads((ROOT/'configs/data.json').read_text())['sources'])
    result={}
    for path in folder.glob('live-*/raw/manifest.json'):
        meta=json.loads(path.read_text())
        for name,info in meta.get('sources',{}).items():
            if name not in names:continue
            status=info.get('status')
            entry={'status':status if status in {'ok','unavailable','not_attempted'} else 'invalid_status'}
            diagnostic=info.get('diagnostic',{})
            # Values originate from safe_acquisition_error, never from error text.
            for key in ['error_type','reason','http_status']:
                value=diagnostic.get(key)
                if key=='http_status' and isinstance(value,int) and 100<=value<=599:entry[key]=value
                elif key=='reason' and value in {'http_error','timeout','connection_or_tls_error','schema_or_observation_invalid','provider_error'}:entry[key]=value
                elif key=='error_type' and isinstance(value,str) and value.isidentifier() and len(value)<80:entry[key]=value
            result[name]=entry
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--workdir',type=Path,required=True)
    args=parser.parse_args()
    diagnostics={}
    try:report=run(os.environ,args.workdir,args.publish,diagnostics)
    except Exception as error:
        # Provider/SDK exceptions may contain private URLs. Never dump traceback.
        report={'status':'failed','error_type':type(error).__name__,'weights_updated':False,
                'stage':diagnostics.get('stage','initialization'),'publish_requested':args.publish}
        if isinstance(error,HTTPError):report['http_status']=error.code
        try:report['sources']=safe_source_statuses(args.workdir)
        except Exception:report['sources']={}
        print(json.dumps(report,sort_keys=True))
        summary=os.environ.get('GITHUB_STEP_SUMMARY')
        if summary:
            with Path(summary).open('a',encoding='utf-8') as stream:
                stream.write('## Brent update failed\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
        raise SystemExit(1) from None
    print(json.dumps(report,sort_keys=True))
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a',encoding='utf-8') as stream:
            stream.write('## Brent daily data\n\n'+json.dumps(report,indent=2)+'\n\nNo model weights were changed.\n')


if __name__=='__main__':main()
