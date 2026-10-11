"""Explicit one-successor adoption of an authentic local research optimizer.

This is not a production publication or ACK. The old research evidence remains
unaltered; the first real production request owns the adoption under the normal
exclusive cache lease. Its successor uses the ordinary export and ACK lifecycle.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import stat

from .distributed_roles import authenticate
from .persistent_cpu_adamw import checkpoint_id, sha

VERSION = 'authority-local-research-parent-v1'
USE_VERSION = 'local-research-parent-first-use-v1'
MODULE = 'subnet/research_parent_adoption.py'
REPLICA_SCOPE = 'ROOT-original-recipe-full7B-positive-SFT-replication-v1'
MATCHED32_SCOPE = 'fresh-base-matched1024-positive-SFT32-v1'


def _retained_name(scope):
    if scope.get('version') == REPLICA_SCOPE:
        return 'retained-Adam8'
    if scope.get('version') == MATCHED32_SCOPE:
        return 'retained-Adam32'
    raise ValueError('unsupported original research execution scope')


def _matched32_execution(scope, execution, descriptor, raw):
    """Admit a captured original NETWORK32 endpoint, never a reference control.

    This authenticates local optimizer provenance only. A separate scientific
    decision and prospective execution release are still required to deploy.
    No old replica receipt is relabeled as a newer numerical experiment.
    """
    checkpoint = execution.get('checkpoint')
    retained = execution.get('retained_state')
    expected_directory = str(_absolute(scope['workspace']) / 'retained-Adam32')
    for document, counts in ((scope, {'steps': 32, 'passes': 2}),
                             (execution, {'optimizer_updates': 32, 'passes': 2,
                                          'tasks': 1024, 'negative_gradients': 0})):
        if any(type(document.get(key)) is not int or document[key] != value
               for key, value in counts.items()):
            raise ValueError('matched32 exact integer execution counts')
    if (execution.get('version') != 'ROOT-matched1024-positive-SFT32-numerical-adoption-v1'
            or scope.get('arm') != 'NETWORK32' or execution.get('arm') != 'NETWORK32'
            or scope.get('retain_optimizer_locally') is not True
            or scope.get('production_changes') is not False
            or execution.get('production_changes') is not False
            or descriptor['optimizer_steps'] != 32 or execution.get('positive_only') is not True
            or execution.get('fresh_optimizer') is not True
            or execution.get('EOS_included') is not True
            or type(scope.get('positive_responses')) is not int or scope['positive_responses'] < 1024
            or type(execution.get('positives')) is not int
            or execution['positives'] != scope['positive_responses']
            or execution.get('original_scope_sha256') != hashlib.sha256(raw.encode()).hexdigest()
            or execution.get('full_checkpoint_hashes_verified') is not True
            or execution.get('original_processes_terminal') is not True
            or execution.get('GPU_released') is not True
            or not isinstance(checkpoint, dict) or set(checkpoint) != {'id', 'files'}
            or checkpoint['id'] != descriptor['inference_checkpoint']
            or not isinstance(checkpoint['files'], dict) or not checkpoint['files']
            or sha(checkpoint['files']) != checkpoint['id']
            or not isinstance(retained, dict)
            or set(retained) != {'descriptor_sha256', 'path', 'all_shards_readback_verified', 'optimizer_uploaded'}
            or retained['descriptor_sha256'] != sha(descriptor)
            or retained['path'] != expected_directory
            or retained['all_shards_readback_verified'] is not True
            or retained['optimizer_uploaded'] is not False
            or execution.get('endpoint_path') != str(_absolute(scope['workspace']) / 'endpoint')
            or execution.get('input_checkpoint') != scope.get('checkpoint', {}).get('id')
            or execution.get('input_checkpoint') != descriptor['input_checkpoint']
            or scope.get('genesis', {}).get('input_checkpoint') != descriptor['input_checkpoint']
            or descriptor['parent_state_sha256'] is not None):
        raise ValueError('authenticated original matched32 research execution and retained state')
    for name, digest in checkpoint['files'].items():
        if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
            raise ValueError('matched32 inference checkpoint member')
        checkpoint_id(digest)
    for key in ('preregistration_sha256', 'confirmation_reservation_sha256'):
        checkpoint_id(scope.get(key))
        if scope[key] != execution.get(key):
            raise ValueError('matched32 original study binding')
    checkpoint_id(execution.get('capture_index_sha256'))
    for key in ('worker_sha256', 'kernel_sha256', 'launcher_sha256', 'optimizer_threading_sha256', 'inputs_sha256'):
        checkpoint_id(scope.get(key))


def is_publication(envelope):
    return isinstance(envelope, dict) and envelope.get('payload', {}).get('version') == VERSION


def _absolute(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError('research adoption absolute path')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('research adoption normalized absolute path')
    return path


def validate_publication(envelope, binding, authority):
    publication = authenticate(envelope, authority)
    return _validate_body(publication, binding, authority)


def _validate_body(publication, binding, authority):
    from .persistent_training_protocol import state_pointer, safe_namespace
    from .persistent_training_state import validate_descriptor
    fields = {'version', 'namespace', 'adoption_id', 'descriptor_sha256', 'descriptor',
              'research_scope_json', 'research_execution', 'research_source_files_sha256',
              'successor', 'retained_directory'}
    if set(publication) != fields or publication['version'] != VERSION:
        raise ValueError('exact signed research adoption publication')
    from .cache_lifecycle import identifier
    adoption_id = identifier(publication['adoption_id'])
    if len(adoption_id) > 100 or not safe_namespace(publication['namespace']).endswith('/' + adoption_id):
        raise ValueError('research adoption namespace binding')
    if state_pointer(publication) != binding['parent']:
        raise ValueError('research adoption exact signed parent pointer')
    descriptor = publication['descriptor']
    validate_descriptor(descriptor, publication['descriptor_sha256'], binding['input_checkpoint'], binding['parameters'])
    if (descriptor['optimizer_steps'] != binding['global_step_before'] or
            descriptor['genesis_sha256'] != binding['genesis_sha256']):
        raise ValueError('research adoption exact counter and genesis')
    raw = publication['research_scope_json']
    if not isinstance(raw, str) or not 0 < len(raw.encode()) <= 1_000_000:
        raise ValueError('bounded original research scope document')
    scope = authenticate(json.loads(raw), authority)
    execution = authenticate(publication['research_execution'], authority)
    source = scope.get('source_files')
    if not isinstance(source, dict) or not source or sha(source) != publication['research_source_files_sha256']:
        raise ValueError('original research source inventory binding')
    for name, digest in source.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('original research source member')
        checkpoint_id(digest)
    if scope.get('version') == MATCHED32_SCOPE:
        _matched32_execution(scope, execution, descriptor, raw)
    elif (scope.get('version') != REPLICA_SCOPE or
            execution.get('version') != 'ROOT-original-full7B-SFT-replication-execution-v1' or
            execution.get('scope_sha256') != hashlib.sha256(raw.encode()).hexdigest() or
            execution.get('state_descriptor_sha256') != sha(descriptor) or
            execution.get('checkpoint') != descriptor['inference_checkpoint'] or
            execution.get('optimizer_steps') != 8 or descriptor['optimizer_steps'] != 8 or
            execution.get('actual_full7B_GPU_execution') is not True or
            execution.get('new_actual_Adam8_retained') is not True or
            execution.get('exact_original_checkpoint_equality') is not True or
            execution.get('exact_original_BF16_parameter_equality') is not True or
            execution.get('historical_optimizer_recovery_claim') is not False or
            scope.get('original_output_checkpoint') != descriptor['inference_checkpoint'] or
            scope.get('epoch') != descriptor['epoch'] or
            scope.get('parameters') != binding['parameters'] or
            sha(scope.get('genesis')) != descriptor['genesis_sha256']):
        raise ValueError('authenticated original research execution and lineage')
    if (scope.get('epoch') != descriptor['epoch'] or scope.get('parameters') != binding['parameters']
            or sha(scope.get('genesis')) != descriptor['genesis_sha256']):
        raise ValueError('authenticated original research execution and lineage')
    successor = publication['successor']
    if not isinstance(successor, dict) or set(successor) != {
            'epoch', 'source_sha256', 'execution_source_sha256', 'source_files', 'trainer_workspace', 'hostname', 'boot_id',
            'optimizer_step_before', 'checkpoint'}:
        raise ValueError('exact research first-successor scope')
    if (successor['epoch'] != binding['epoch'] or successor['source_sha256'] != binding['source_sha256'] or
            successor['optimizer_step_before'] != binding['global_step_before'] or
            successor['checkpoint'] != binding['input_checkpoint'] or
            successor['hostname'] != scope.get('hostname') or successor['boot_id'] != scope.get('boot_id')):
        raise ValueError('research first-successor epoch source model host counter binding')
    checkpoint_id(successor['execution_source_sha256'])
    files = successor['source_files']
    if not isinstance(files, dict) or MODULE not in files:
        raise ValueError('research adoption execution module source pin')
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('successor execution source member')
        checkpoint_id(digest)
    _absolute(successor['trainer_workspace'])
    retained = _absolute(publication['retained_directory'])
    if retained != _absolute(scope['workspace']) / _retained_name(scope):
        raise ValueError('exact original research retained-state directory')
    return publication


def validate_successor(envelope, job, manifest, authority, *, preparation=False):
    from .persistent_publication import local_state
    publication = validate_publication(envelope, manifest['trainer_state_binding'], authority)
    target = publication['successor']
    if (not local_state(manifest) or job.get('role') != 'train' or
            job.get('source_files') != target['source_files'] or
            manifest['source_bundle']['sha256'] != target['source_sha256']):
        raise ValueError('research adoption exact local successor source')
    if type(preparation) is not bool:
        raise ValueError('explicit draft preparation stage')
    from .cache_lifecycle import identifier
    identifier(job['job_id'])
    if job['job_id'] == publication['adoption_id']:
        raise ValueError('research adoption identity is not a production job identity')
    # RemoteJobs prepares optimizer capabilities before attaching its signed
    # execution grant. Pure draft validation may precede that grant; consuming
    # any local research bytes always takes the strict default path below.
    if preparation and 'unaudited_training_execution' not in job:
        return publication
    declaration = authenticate(job.get('unaudited_training_execution', {}), authority)
    if (declaration.get('version') != 'unaudited-training-execution-amendment-v6-positive-sft' or
            declaration.get('job_id') != job['job_id'] or declaration.get('epoch') != target['epoch'] or
            declaration.get('training_source_bundle', {}).get('sha256') != target['execution_source_sha256'] or
            declaration.get('execution_source_files') != target['source_files'] or
            declaration.get('original_source_bundle_sha256') != target['source_sha256'] or
            declaration.get('parent_descriptor_sha256') != publication['descriptor_sha256'] or
            declaration.get('optimizer_step_before') != target['optimizer_step_before']):
        raise ValueError('research adoption signed retained-SFT execution source and parent')
    return publication


def _private_directory(path):
    info = path.lstat()
    if (path != path.resolve() or not stat.S_ISDIR(info.st_mode) or
            info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError('research adoption private owned directory')


def prepare_cache(cache, descriptor, source):
    """Called under StateCache's exclusive lease; never deletes research bytes."""
    from .cache_lifecycle import snapshot
    from .optimizer_state_cache import STAT_VERSION, fd_snapshot, member
    if cache.fd is None or cache.policy['version'] != STAT_VERSION:
        raise ValueError('research parent requires leased retained-fd state cache')
    envelope = cache.job['persistent_training']['parent_publication']
    publication = validate_successor(envelope, cache.job, cache.manifest, cache.authority)
    target = publication['successor']
    if (descriptor != publication['descriptor'] or source != target['source_sha256'] or
            str(cache.workspace) != target['trainer_workspace'] or
            socket.gethostname() != target['hostname'] or
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() != target['boot_id']):
        raise ValueError('research adoption actual host workspace source descriptor')
    # No recovery, discard, fallback or counter comparison may erase a later
    # production state, or silently replay a job whose output is already staged.
    for name in ('current.json', 'pending.json', 'promotion.json'):
        if (cache.root / name).exists() or (cache.root / name).is_symlink():
            raise ValueError('research adoption blocked by existing production state; recover original ACK')
    receipt = dict(version=USE_VERSION, publication_sha256=sha(publication),
                   job_id=cache.job['job_id'], job_sha256=sha(cache.job),
                   descriptor_sha256=sha(descriptor), source_sha256=source,
                   workspace=str(cache.workspace))
    marker = cache.root / 'research-parent-use.json'
    if marker.exists() or marker.is_symlink():
        if snapshot(marker)['mode'] & 0o077 or json.loads(marker.read_bytes()) != receipt:
            raise ValueError('research parent already assigned to a different original job')
    else:
        # fsync both file and directory before opening optimizer shards. Exact
        # retries after a crash are allowed; another generated request is not.
        cache.save(marker, receipt)
        fd = os.open(cache.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    directory = Path(publication['retained_directory'])
    _private_directory(directory)
    descriptor_path = directory / 'descriptor.json'
    descriptor_stat = snapshot(descriptor_path)
    if descriptor_stat['mode'] & 0o077 or json.loads(descriptor_path.read_bytes()) != descriptor:
        raise ValueError('original retained descriptor bytes changed')
    shards = {row['name']: row for row in descriptor['shards']}
    if {path.name for path in directory.iterdir()} != set(shards) | {'descriptor.json'}:
        raise ValueError('exact original retained-state inventory')
    files = {}
    for name, row in shards.items():
        path = directory / member(name)
        before = snapshot(path)
        if before['mode'] & 0o077:
            raise ValueError('private original research state required')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if fd_snapshot(fd) != before:
                raise ValueError('research state fd changed before hash')
            digest = hashlib.sha256()
            size = 0
            while data := os.read(fd, 8 * 1024**2):
                digest.update(data)
                size += len(data)
            if ((digest.hexdigest(), size) != (row['sha256'], row['size']) or
                    fd_snapshot(fd) != before or snapshot(path) != before):
                raise ValueError('original research state shard bytes changed')
        finally:
            os.close(fd)
        files[name] = dict(sha256=row['sha256'], size=row['size'], stat=before)
    if snapshot(descriptor_path) != descriptor_stat:
        raise ValueError('research descriptor changed during admission')
    size = sum(row['size'] for row in shards.values())
    if size > cache.policy['max_checkpoint_bytes']:
        raise ValueError('research retained parent byte cap')
    # This in-memory handle is deliberately NOT written as current.json and
    # contains no fabricated ROOT_ack or former production job identity.
    cache.research_parent_directory = directory
    cache.current = dict(version=VERSION, job_id=publication['adoption_id'],
                         descriptor_sha256=sha(descriptor), files=files)
    cache.rows = shards
    cache.verified_rows = {name: row['stat'] for name, row in files.items()}
    cache.cache_evidence.append(dict(outcome='authenticated-research-parent',
        publication_sha256=sha(publication), descriptor_sha256=sha(descriptor),
        size=size, all_SHA_size_verified=True, original_files_preserved=True,
        first_successor_job_id=cache.job['job_id']))
    return size


def prospective_publication(*, adoption_id, namespace, research_scope_json, research_execution,
                            descriptor, successor, authority):
    """Return validated unsigned prospective metadata; caller explicitly signs.

    Authentic research receipts are checked here, but this does not assert live
    state commitment, admit a machine or synthesize a production execution.
    """
    from .persistent_training_protocol import state_pointer
    scope = authenticate(json.loads(research_scope_json), authority)
    body = dict(version=VERSION, namespace=namespace, adoption_id=adoption_id,
        descriptor_sha256=sha(descriptor), descriptor=descriptor,
        research_scope_json=research_scope_json, research_execution=research_execution,
        research_source_files_sha256=sha(scope['source_files']), successor=successor,
        retained_directory=str(_absolute(scope['workspace']) / _retained_name(scope)))
    binding = dict(parent=state_pointer(body), input_checkpoint=successor['checkpoint'],
        parameters=descriptor['parameters'], global_step_before=successor['optimizer_step_before'],
        genesis_sha256=descriptor['genesis_sha256'], epoch=successor['epoch'],
        source_sha256=successor['source_sha256'])
    _validate_body(body, binding, authority)
    return body
