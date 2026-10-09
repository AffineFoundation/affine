"""Prospective CPU-only timing admission; scientific fields remain outside its scope."""
import json
from pathlib import Path

VERSION = 'future-epoch-collection-window-v1'

def validate(amendment, previous, config, state):
    fields = {'version', 'first_round', 'previous_duration', 'duration', 'previous_hourly_policy', 'hourly_policy'}
    if not isinstance(amendment, dict) or set(amendment) != fields or amendment['version'] != VERSION:
        raise ValueError('explicit collection timing amendment required')
    if (amendment['first_round'] != 88 or amendment['previous_duration'] != 600
            or amendment['duration'] != 1200 or state['round'] < 88):
        raise ValueError('future88 collection boundary and approved duration')
    before = amendment['previous_hourly_policy']; after = amendment['hourly_policy']
    expected = dict(before, mine_seconds=1200, train_publication_seconds=1500)
    if (previous.get('duration') != 600 or config.get('duration') != 1200
            or previous.get('hourly_execution_policy') != before or config.get('hourly_execution_policy') != after
            or before.get('mine_seconds') != 600 or before.get('train_publication_seconds') != 2100
            or after != expected or sum(v for k, v in after.items() if k != 'version') != 3600):
        raise ValueError('exact original and future hourly timing allocation')
    active = state.get('active') or {}
    if active.get('epoch'):
        path = Path(config['state']) / (active['epoch'] + '-manifest.json')
        if path.exists():
            manifest = json.loads(path.read_bytes()); manifest = manifest.get('payload', manifest)
            if manifest['deadline'] - manifest['start'] != 1200 or manifest.get('hourly_execution_policy') != after:
                raise ValueError('cannot rewrite an already published shorter epoch')
    return {'duration', 'hourly_execution_policy'}
