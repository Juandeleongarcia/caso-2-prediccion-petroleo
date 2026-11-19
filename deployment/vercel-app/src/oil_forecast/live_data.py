"""Prospective, immutable serving observations; never fits or changes science files."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .acquisition import download_snapshot, load_snapshot, sha256, validate_observations
from .app_data import _last_decision_date
from .refresh import IDENTIFIER, RefreshController
from .temporal import apply_availability, build_features, build_target

PAYLOADS=('features.csv','target.csv','observations.csv')


def _canonical(frame):
    result=frame.copy()
    for name in result:
        if name in {'decision_date','observation_date'} or name.endswith('__observation_date'):
            result[name]=pd.to_datetime(result[name],format='mixed').dt.strftime('%Y-%m-%d')
        elif name in {'decision_at','label_available_at','available_at','first_seen_at'} or name.endswith('__available_at'):
            result[name]=pd.to_datetime(result[name],utc=True,format='mixed').dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    return result


def seed_observations(window, config):
    records=[]
    for name,spec in config['sources'].items():
        if name=='opep_orb' or not spec.get('enabled',True):continue
        rows=window.loc[window[name].notna()].drop_duplicates(name+'__observation_date')
        for _,row in rows.iterrows():
            records.append({'observation_date':row[name+'__observation_date'],'variable':name,
                'value':row[name],'source':('FRED:' if spec['provider']=='fred' else 'Yahoo:')+spec['id'],
                'available_at':row[name+'__available_at'],
                'availability_basis':'frozen_snapshot_availability_assumption',
                'first_seen_at':None,'seeded':True})
    return validate_observations(pd.DataFrame(records))


def merge_observations(previous, incoming, config, acquired_at):
    """Keep first nonmissing vintages; a new value cannot become known in the past."""
    now=pd.Timestamp(acquired_at)
    if now.tzinfo is None:raise ValueError('Timezone required')
    previous=validate_observations(previous)
    incoming=validate_observations(incoming)
    names={n for n,s in config['sources'].items() if s.get('enabled',True) and n!='opep_orb'}
    if set(incoming.variable)!=names:raise ValueError('All ten original sources are required')
    dates=pd.to_datetime(incoming.observation_date)
    if (dates>now.tz_convert('Europe/Madrid').tz_localize(None).normalize()).any():
        raise ValueError('Future source observations')
    old=previous.set_index(['variable','observation_date'])
    fresh=incoming.loc[incoming.value.notna()].copy()
    indices=pd.MultiIndex.from_frame(fresh[['variable','observation_date']])
    known=indices.isin(old.index)
    ignored_revisions=0
    if known.any():
        existing=old.reindex(indices[known]).value.to_numpy(dtype=float)
        current=fresh.loc[known,'value'].to_numpy(dtype=float)
        ignored_revisions=int(np.sum(np.isfinite(existing)&(existing!=current)))
        known[known]=np.isfinite(existing)
    fresh=fresh.loc[~known].copy()
    # The downloader has already applied calendar/range restrictions; policies
    # here add the original lag AND the first acquisition instant.
    available=apply_availability(fresh,config['sources']) if len(fresh) else None
    if available is not None:
        fresh=available.drop(columns=['max_age_business_days'])
        fresh['available_at']=pd.to_datetime(fresh.available_at,utc=True).where(
            pd.to_datetime(fresh.available_at,utc=True)>=now,now)
        fresh['availability_basis']='max_first_seen_and_policy_lag'
        fresh['first_seen_at']=now.isoformat();fresh['seeded']=False
        replacing=pd.MultiIndex.from_frame(fresh[['variable','observation_date']])
        previous=previous.loc[~pd.MultiIndex.from_frame(previous[['variable','observation_date']]).isin(replacing)]
        merged=pd.concat([previous,fresh],ignore_index=True)
    else:merged=previous.copy()
    return validate_observations(merged),ignored_revisions


def validate_live(folder,config,now,validator=None,expected_manifest=None):
    folder=Path(folder)
    if not IDENTIFIER.fullmatch(folder.name) or folder.is_symlink():raise ValueError('Invalid snapshot path')
    meta_path=folder/'manifest.json'
    if expected_manifest and sha256(meta_path)!=expected_manifest:raise ValueError('Manifest integrity')
    protocol=json.loads(meta_path.read_text(encoding='utf-8'))
    if protocol.get('live_schema_version')!=1 or set(protocol['output_sha256'])!=set(PAYLOADS):
        raise ValueError('Invalid live schema')
    for name in PAYLOADS:
        path=folder/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size>100*1024*1024 or sha256(path)!=protocol['output_sha256'][name]:
            raise ValueError('Live payload integrity')
    observations=validate_observations(pd.read_csv(folder/'observations.csv'))
    if set(observations)!={'observation_date','variable','value','source','is_observed','available_at',
                           'availability_basis','first_seen_at','seeded'}:
        raise ValueError('Unexpected observation schema')
    acquired=pd.Timestamp(protocol['data_updated_at'])
    if acquired.tzinfo is None or acquired>pd.Timestamp(now):raise ValueError('Invalid acquisition time')
    if (pd.to_datetime(observations.observation_date)>acquired.tz_convert('Europe/Madrid').tz_localize(None).normalize()).any():
        raise ValueError('Future observations')
    first=pd.to_datetime(observations.first_seen_at,utc=True)
    available=pd.to_datetime(observations.available_at,utc=True)
    if (first>acquired).any() or (available<first).any():raise ValueError('Retrospective first-seen metadata')
    data=RefreshController(folder.parent)._load(folder.name,config,pd.Timestamp(now).to_pydatetime(),validator,expected_manifest)
    features=pd.read_csv(folder/'features.csv')
    expected={'decision_date','decision_at'}
    for name,spec in config['sources'].items():
        if name=='opep_orb' or not spec.get('enabled',True):continue
        expected.add(name)
        expected.update(name+'__'+suffix for suffix in ['observation_date','available_at','availability_basis',
            'age_business_days','is_available','is_stale','is_observed_today','is_carried'])
    if set(features)!=expected:raise ValueError('Unexpected feature schema')
    dates=pd.to_datetime(features.decision_date)
    if not pd.DatetimeIndex(dates).equals(pd.bdate_range(dates.iloc[0],dates.iloc[-1])):
        raise ValueError('Incomplete business calendar')
    return data


def build_daily(config, destination, seed_window, seed_history, seed_updated_at,
                model_manifest_sha256, *, previous=None, downloader=download_snapshot,now=None):
    """Download into a NEW directory; previous decisions/targets never get revised."""
    destination=Path(destination)
    if destination.exists() or not IDENTIFIER.fullmatch(destination.name):raise ValueError('New snapshot required')
    settings=copy.deepcopy(config)
    settings['end']=_last_decision_date(pd.Timestamp(now).tz_convert('Europe/Madrid') if now is not None else pd.Timestamp.now(tz='Europe/Madrid'),config.get('decision_hour',8))
    settings['snapshot_id']=destination.name
    acquisition=copy.deepcopy(settings)
    if 'opep_orb' in acquisition['sources']:acquisition['sources']['opep_orb']['enabled']=False
    downloader(acquisition,destination/'raw',destination.parent/'.cache')
    incoming,source_manifest=load_snapshot(destination/'raw')
    names=[n for n,s in config['sources'].items() if s.get('enabled',True) and n!='opep_orb']
    if any(source_manifest['sources'].get(n,{}).get('status')!='ok' for n in names):
        raise ValueError('A source failed; do not publish')
    acquired=pd.Timestamp(source_manifest['downloaded_at'])
    old_features=pd.read_csv(Path(previous)/'features.csv') if previous else seed_window.copy()
    old_history=pd.read_csv(Path(previous)/'target.csv') if previous else seed_history.copy()
    old_raw=pd.read_csv(Path(previous)/'observations.csv') if previous else seed_observations(seed_window,config)
    old_updated=json.loads((Path(previous)/'manifest.json').read_text())['data_updated_at'] if previous else seed_updated_at
    if acquired<=pd.Timestamp(old_updated):raise ValueError('Non-increasing acquisition timestamp')
    merged,revisions=merge_observations(old_raw,incoming,config,acquired)
    # Explicit availability provenance is already frozen above. Never reinterpret
    # acquisition time as historical publication time.
    prepared=apply_availability(merged,config['sources'])
    last=pd.Timestamp(old_features.decision_date.iloc[-1])
    end=pd.Timestamp(settings['end'])
    if end<last:raise ValueError('Cannot regress decision origin')
    if end>last:
        appended=build_features(prepared,(last+pd.offsets.BDay(1)).strftime('%Y-%m-%d'),settings['end'],config['decision_hour'])
        if set(appended)!=set(old_features):raise ValueError('Feature schema changed')
        features=pd.concat([old_features,appended.reindex(columns=old_features.columns)],ignore_index=True)
    else:features=old_features.copy()
    # A bootstrap acquisition after the 08:00 decision cannot be backdated.
    # Keep the old valid forecast origin while publishing first-seen records;
    # the next decision can legitimately use them. Never fabricate a reference.
    deferred=not np.isfinite(pd.to_numeric(features['brent'].iloc[-1],errors='raise'))
    if deferred:features=old_features.copy()
    target_end=pd.Timestamp(old_history.observation_date.iloc[-1])
    if end>target_end:
        target=build_target(prepared,(target_end+pd.offsets.BDay(1)).strftime('%Y-%m-%d'),settings['end'])
        if set(target)!=set(old_history):raise ValueError('Target schema changed')
        history=pd.concat([old_history,target.reindex(columns=old_history.columns)],ignore_index=True)
    else:history=old_history.copy()
    for name,frame in zip(PAYLOADS,[features,history,merged]):_canonical(frame).to_csv(destination/name,index=False,lineterminator='\n')
    protocol={'live_schema_version':1,'snapshot_id':destination.name,'data_updated_at':acquired.isoformat(),
        'config':settings,'sources':{n:{k:v for k,v in source_manifest['sources'][n].items() if k!='error'} for n in names},
        'weights_updated':False,'historical_point_in_time':False,'model_manifest_sha256':model_manifest_sha256,
        'origin_advance_deferred':bool(deferred),
        'revision_policy':'first nonmissing observation retained; old decisions immutable',
        'ignored_revision_count':revisions,'output_sha256':{name:sha256(destination/name) for name in PAYLOADS}}
    (destination/'manifest.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return protocol
