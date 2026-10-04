"""Environment-bound manifests/batches, with read-only legacy artifact support."""
from . import harness
from .sample_harness import validate as validate_sample_harness, resolve as resolve_sample_harness,project,VERSION as INDEXED_VERSION


def entries(manifest):
    """Live admission always pins the imported, approved current harness code."""
    from .forced_sampling import binding, validate_harness
    context=binding(manifest)
    values=_entries(manifest,harness.source_hash())
    if context is not None:
        for definition in values:
            raw=definition['harness']
            if isinstance(raw,dict) and raw.get('version')==INDEXED_VERSION:
                for row in raw['by_index'].values():validate_harness(row)
            else:validate_harness(raw)
    return values


def read_only_archived_entries(manifest,expected_archive_harness_source_hash):
    """Metadata only; caller derives this pin from authenticated archive bytes.

    This does not load code or authorize fresh verification/optimizer execution.
    No submitted manifest field selects this API or disables live admission.
    """
    if (not isinstance(expected_archive_harness_source_hash,str) or
            len(expected_archive_harness_source_hash)!=64 or
            any(c not in '0123456789abcdef' for c in expected_archive_harness_source_hash)):
        raise ValueError('verified archive harness SHA required')
    return _entries(manifest,expected_archive_harness_source_hash)


def _entries(manifest,expected_source_hash):
    result = manifest.get('environments')
    if result is None:
        environment = manifest['environment']
        if isinstance(manifest.get('harness'),dict)and manifest['harness'].get('version')==INDEXED_VERSION:raise ValueError('indexed harness requires explicit environment registry')
        return [dict(env_id=environment.get('id', 'mastermind'), spec=environment,
                     harness=manifest.get('harness'), indices=manifest['indices'])]
    if not isinstance(result, list) or not 1 <= len(result) <= 64:
        raise ValueError('environment registry budget')
    if len({e['env_id'] for e in result}) != len(result):
        raise ValueError('duplicate environment definition')
    if manifest.get('harness_source_hash') != expected_source_hash:
        raise ValueError('trusted harness source mismatch')
    for entry in result:
        if entry['env_id'] != entry['spec'].get('id', 'mastermind'):
            raise ValueError('environment registry identity')
        indices = entry['indices']
        evaluation_only=entry.get('evaluation_only',False)
        if type(evaluation_only)is not bool or (evaluation_only and indices):
            raise ValueError('evaluation-only mining indices')
        if not isinstance(indices, list) or len(indices) > 10000 or len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError('sample index budget')
        validate_sample_harness(entry['harness'],indices)
    registry=manifest.get('sample_harness_registry')
    if registry is None and any(isinstance(e['harness'],dict)and e['harness'].get('version')==INDEXED_VERSION for e in result):raise ValueError('signed full indexed harness registry required')
    if registry is not None:
        if not isinstance(registry,dict)or set(registry)!={e['env_id']for e in result}:raise ValueError('exact full sample harness registry')
        for e in result:
            row=registry[e['env_id']]
            if not isinstance(row,dict)or set(row)!={'indices','harness'}or not isinstance(row['indices'],list)or any(type(i)is not int or i<0 or i>=e['spec']['num_samples']for i in row['indices']):raise ValueError('full registry sample geometry')
            if e.get('evaluation_only',False) and row['indices']:
                raise ValueError('evaluation-only full mining registry')
            if e['harness']!=project(row['harness'],e['indices'],row['indices']):raise ValueError('signed projected sample harness mismatch')
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


def harness_for(definition,index):
    return resolve_sample_harness(definition['harness'],index,definition['indices'])


def replay_harness_descriptor(definition,index):
    """Bind the selected indexed policy; plain/legacy descriptors stay unchanged."""
    if isinstance(definition.get('harness'),dict)and definition['harness'].get('version')==INDEXED_VERSION:
        import hashlib
        from .storage import canonical
        resolved=harness_for(definition,index)
        return dict(resolved_harness=resolved,resolved_harness_sha256=hashlib.sha256(canonical(resolved)).hexdigest())
    return {}
