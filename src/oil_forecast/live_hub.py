"""Read a moving data pointer, then only immutable HF revisions; models stay pinned."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from threading import RLock
import time
from urllib.parse import urlparse
from urllib.request import Request,build_opener

from .artifacts import _HttpsArtifactRedirects
from .live_data import PAYLOADS,validate_live
from .refresh import IDENTIFIER

FILES=(*PAYLOADS,'manifest.json')


class HubReader:
    def __init__(self,repo,token):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo or '') or not token:
            raise ValueError('Private live repository and server token required')
        self.repo,self.token=repo,token

    def _get(self,path,limit):
        request=Request('https://huggingface.co/'+path,headers={'Cache-Control':'no-cache'})
        request.add_unredirected_header('Authorization','Bearer '+self.token)
        with build_opener(_HttpsArtifactRedirects()).open(request,timeout=20) as response:
            body=response.read(limit+1)
        if len(body)>limit:raise ValueError('Live response too large')
        return body

    def head(self):
        meta=json.loads(self._get('api/models/'+self.repo+'/revision/main',1024*1024))
        revision=meta.get('sha','')
        if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Invalid resolved commit')
        return revision

    def read(self,revision,path):
        if (not re.fullmatch('[0-9a-f]{40}',revision) or
            not (path=='live/latest.json' or re.fullmatch(r'live/snapshots/[A-Za-z0-9_-]+/(?:features.csv|target.csv|observations.csv|manifest.json)',path))):
            raise ValueError('Only pinned live data paths are allowed')
        return self._get(self.repo+'/resolve/'+revision+'/'+path,(1 if path.endswith('.json') else 100)*1024*1024)


def pointer_for(folder):
    folder=Path(folder)
    meta=json.loads((folder/'manifest.json').read_text())
    return {'schema_version':1,'snapshot_id':folder.name,'data_updated_at':meta['data_updated_at'],
        'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        'model_manifest_sha256':meta['model_manifest_sha256']}


def fetch_latest(reader,directory,config,model_sha,validator=None,now=None):
    """Stage and validate the full snapshot before making it visible locally."""
    now=now or datetime.now(timezone.utc)
    revision=reader.head()
    pointer=json.loads(reader.read(revision,'live/latest.json'))
    identifier=pointer.get('snapshot_id','')
    if (pointer.get('schema_version')!=1 or not IDENTIFIER.fullmatch(identifier) or
        not identifier.startswith('live-') or pointer.get('model_manifest_sha256')!=model_sha or
        not re.fullmatch('[0-9a-f]{64}',pointer.get('manifest_sha256',''))):
        raise ValueError('Invalid live pointer')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/identifier
    if target.exists():
        data=validate_live(target,config,now,validator,pointer['manifest_sha256'])
    else:
        with tempfile.TemporaryDirectory(dir=directory,prefix='.live-fetch-') as temporary:
            stage=Path(temporary)/identifier;stage.mkdir()
            for name in FILES:
                (stage/name).write_bytes(reader.read(revision,f'live/snapshots/{identifier}/{name}'))
            data=validate_live(stage,config,now,validator,pointer['manifest_sha256'])
            if data['protocol']['model_manifest_sha256']!=model_sha:raise ValueError('Wrong model provenance')
            stage.rename(target)
            data['directory']=str(target)
    if data['protocol']['data_updated_at']!=pointer['data_updated_at']:
        raise ValueError('Pointer timestamp mismatch')
    data['published_commit']=revision
    return data


class LiveHubController:
    def __init__(self,directory,reader,model_sha,interval_seconds=300,clock=None,minimum_updated_at=None):
        if not 60<=interval_seconds<=86400:raise ValueError('Invalid polling interval')
        self.directory=Path(directory);self.reader=reader;self.model_sha=model_sha
        self.interval=interval_seconds;self.next_check=0.;self.data=None;self.status={}
        self.clock=clock or (lambda:datetime.now(timezone.utc));self.lock=RLock()
        self.minimum_updated_at=minimum_updated_at

    def check(self,config,validator):
        with self.lock:
            if time.monotonic()<self.next_check:return self.data,self.status
            self.next_check=time.monotonic()+self.interval
            now=self.clock()
            try:
                candidate=fetch_latest(self.reader,self.directory,config,self.model_sha,validator,now)
                stamp=datetime.fromisoformat(candidate['protocol']['data_updated_at'].replace('Z','+00:00'))
                floor=self.data['protocol']['data_updated_at'] if self.data else self.minimum_updated_at
                if floor and stamp<datetime.fromisoformat(floor.replace('Z','+00:00')):
                    raise ValueError('Remote rollback rejected')
                self.data=candidate
                self.status={'status':'current','last_attempt_at':now.isoformat(),
                    'last_success_at':candidate['protocol']['data_updated_at'],
                    'published_commit':candidate['published_commit'],'snapshot_id':candidate['protocol']['snapshot_id'],
                    'origin_advance_deferred':candidate['protocol'].get('origin_advance_deferred',False),
                    'mode':'github_actions','weights_updated':False}
            except Exception:
                # No exception strings/URLs/tokens in public metadata. The old
                # verified in-memory snapshot (or the frozen bundle) remains.
                self.status={**self.status,'status':'failed','last_attempt_at':now.isoformat(),
                    'mode':'github_actions','weights_updated':False}
            return self.data,self.status
