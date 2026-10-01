"""Environment-bound manifests/batches, with read-only legacy artifact support."""
from . import harness


def entries(manifest):
    result = manifest.get('environments')
    if result is None:
        environment = manifest['environment']
        return [dict(env_id=environment.get('id', 'mastermind'), spec=environment,
                     harness=manifest.get('harness'), indices=manifest['indices'])]
    if not isinstance(result, list) or not 1 <= len(result) <= 64:
        raise ValueError('environment registry budget')
    if len({e['env_id'] for e in result}) != len(result):
        raise ValueError('duplicate environment definition')
    if manifest.get('harness_source_hash') != harness.source_hash():
        raise ValueError('trusted harness source mismatch')
    for entry in result:
        if entry['env_id'] != entry['spec'].get('id', 'mastermind'):
            raise ValueError('environment registry identity')
        indices = entry['indices']
        if not isinstance(indices, list) or len(indices) > 10000 or len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError('sample index budget')
        harness.normalize(entry['harness'])
    return result


def entry(manifest, env_id=None):
    values = entries(manifest)
    if env_id is None:
        if len(values) != 1:
            raise ValueError('environment identity required')
        return values[0]
    for value in values:
        if value['env_id'] == env_id:
            return value
    raise ValueError('environment not authorized')


def sample_key(batch):
    index = batch.get('sample_index', batch['index'])
    if batch.get('index', index) != index:
        raise ValueError('inconsistent sample index')
    return batch.get('env_id', 'mastermind'), index, batch.get('checkpoint', 'legacy')


def classification(rollout):
    # Archived records predate the explicit classifier; don't reinterpret new
    # reward scales as zero/one. New records must carry the pinned classification.
    return rollout.get('classification', 'positive' if rollout['reward'] == 1 else 'negative' if rollout['reward'] == 0 else 'neutral')
