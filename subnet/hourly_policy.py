"""Prospective signed phase bounds; deadlines never imply sample fraud."""
import math
VERSION='bounded-hourly-phases-v1'
def validate(raw,duration):
    keys={'version','mine_seconds','freeze_seconds','audit_seconds','train_publication_seconds','weight_seconds','slack_seconds'}
    if type(raw)is not dict or set(raw)!=keys or raw['version']!=VERSION:raise ValueError('hourly phase policy')
    for key in keys-{'version'}:
        if type(raw[key])is not int or not 0<=raw[key]<=3600:raise ValueError('hourly phase seconds')
    if raw['mine_seconds']!=duration or not raw['freeze_seconds'] or not raw['audit_seconds'] or sum(raw[k]for k in keys-{'version'})>3600:raise ValueError('full hourly phase bound')
    return dict(raw)
def cutoff(manifest,phase):
    raw=manifest.get('hourly_execution_policy')
    if raw is None:return None
    p=validate(raw,raw.get('mine_seconds'))
    if not 0<manifest['deadline']-manifest['start']<=p['mine_seconds']:raise ValueError('signed mining duration')
    return manifest['deadline']+p['freeze_seconds']+(p['audit_seconds']if phase=='audit'else 0)
