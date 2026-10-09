"""Explicit future-only collection duration; published openings remain immutable.

The original fresh-run physical configuration and reset declaration stay1800s.
Only new openings at the signed boundary use1200s;600s becomes explicit reserve.
"""
import json
from pathlib import Path
VERSION='fresh-run-prospective-collection-window-v2'


def profiles(amendment):
    fields={'version','first_round','previous_duration','duration','previous_hourly_policy','hourly_policy',
            'successor_first_round','successor_duration','successor_hourly_policy'}
    if type(amendment)is not dict or set(amendment)!=fields or amendment['version']!=VERSION:
        raise ValueError('exact prospective collection amendment')
    before,after,future=(amendment[k]for k in('previous_hourly_policy','hourly_policy','successor_hourly_policy'))
    if (type(amendment['first_round'])is not int or amendment['first_round']!=91
            or type(amendment['successor_first_round'])is not int or amendment['successor_first_round']!=93
            or amendment['previous_duration']!=600 or amendment['duration']!=1800
            or type(amendment['successor_duration'])is not int or amendment['successor_duration']!=1200
            or after!=dict(before,mine_seconds=1800,audit_seconds=60,train_publication_seconds=1500,weight_seconds=60)
            or future!=dict(after,mine_seconds=1200,slack_seconds=720)
            or any(type(v)is not int or v<0 for p in(after,future)for k,v in p.items()if k!='version')
            or any(sum(v for k,v in p.items()if k!='version')!=3600 for p in(after,future))):
        raise ValueError('exact original and prospective hourly allocations')
    return before,after,future


def validate(amendment,previous,config,state):
    before,after,future=profiles(amendment)
    if (type(state.get('round'))is not int or state['round']<91
            or previous['duration']!=600 or config['duration']!=1800
            or previous['hourly_execution_policy']!=before or config['hourly_execution_policy']!=after):
        raise ValueError('unchanged original fresh-run physical timing configuration')
    active=state.get('active')or{}
    if active.get('epoch'):
        if int(active['epoch'].rsplit('-',1)[-1])!=state['round']:
            raise ValueError('exact active opening round')
        path=Path(config['state'])/(active['epoch']+'-manifest.json')
        if path.exists():
            manifest=json.loads(path.read_bytes());manifest=manifest.get('payload',manifest)
            prospective=state['round']>=amendment['successor_first_round']
            duration=amendment['successor_duration']if prospective else amendment['duration']
            expected=future if prospective else after
            if (manifest['deadline']-manifest['start']!=duration
                    or manifest.get('hourly_execution_policy')!=expected):
                raise ValueError('cannot rewrite an already published collection window')
    return {'duration','hourly_execution_policy'}


def install(service,amendment):
    before,after,future=profiles(amendment)
    original=service.contract
    def contract(config,round_number):
        if type(round_number)is not int or round_number<0:raise ValueError('exact future collection round')
        result=original(config,round_number)
        if round_number<amendment['successor_first_round']:return result
        if result.get('duration')!=1800 or result.get('hourly_execution_policy')!=after:
            raise ValueError('exact original collection contract before prospective amendment')
        return dict(result,duration=amendment['successor_duration'],hourly_execution_policy=dict(future))
    service.contract=contract
    return original
