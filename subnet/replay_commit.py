"""Commit consumed replay usage after successful checkpoint publication.

The epoch and counts share one atomic journal, so a retry cannot double-count
usage or mark an unpublished checkpoint as consumed. The caller supplies only
authenticated training metrics after RemoteController.train returns.
"""
import json
import os
import tempfile
from pathlib import Path


def commit(path, epoch, metrics):
    replay = metrics['replay_training']
    increments = replay['proposed_reuse_increments']
    targets = {row['target_sha256'] for row in replay['checks']}
    if set(increments) != targets or any(type(v) is not int or v != 1 for v in increments.values()):
        raise ValueError('exact consumed historical replay increments')
    for target in targets:
        if len(target) != 64 or any(c not in '0123456789abcdef' for c in target):
            raise ValueError('replay target hash')
    record = {'pool_sha256': replay['pool_sha256'], 'increments': increments,
              'checkpoint': metrics['checkpoint'],
              'replay_inputs_sha256': metrics['replay_inputs_sha256']}
    path = Path(path)
    journal = json.loads(path.read_bytes()) if path.exists() else {'counts': {}, 'epochs': {}}
    if epoch in journal['epochs']:
        if journal['epochs'][epoch] != record:
            raise ValueError('already committed epoch replay binding')
        return journal
    for target, amount in increments.items():
        before = journal['counts'].get(target, 0)
        if type(before) is not int or before < 0:
            raise ValueError('existing replay usage count')
        journal['counts'][target] = before + amount
    journal['epochs'][epoch] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(json.dumps(journal, sort_keys=True, separators=(',', ':')).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return journal
