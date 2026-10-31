"""Local checkpoint handover; public runs require an explicit release capability.

Historical function names remain compatible with isolated private rehearsals.
These hooks do not themselves write public discovery or chain weights.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from .data import digest, sha
from .publication import validate_publication
from .transaction import _write

VERSION = 'affine-mimo-private-current-v1'


@contextmanager
def _locked(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('ordinary private CURRENT ancestry')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield path
    finally:
        os.close(fd)


def _release_scope(binding, release_policy, manifest):
    if release_policy is None:
        if not binding['epoch'].startswith(('private-', 'test-', 'nonpayable-')):
            raise ValueError('public CURRENT requires an explicit validated release policy')
        return {}
    from ..mimo_release_policy import ReleasePolicy
    if not isinstance(release_policy, ReleasePolicy) or manifest is None:
        raise ValueError('validated release capability and original manifest required')
    # Reauthenticate the capability instead of trusting mutable instance fields.
    checked = ReleasePolicy(release_policy.envelope, release_policy.authority,
                            allow_public=release_policy.mode == 'public')
    checked.validate_manifest(manifest)
    if (binding['run'] != checked.run_id or binding['epoch'] != manifest['epoch'] or
            binding['checkpoint'] != manifest['checkpoint']['id']):
        raise ValueError('original update differs from qualified release manifest')
    return dict(release_policy_sha256=checked.policy_sha256, execution_mode=checked.mode)


def initialize_private_current(path, binding, *, source_closure_sha256,
                               release_policy=None, manifest=None):
    binding.validate()
    scope = _release_scope(vars(binding), release_policy, manifest)
    if not sha(source_closure_sha256):
        raise ValueError('authenticated source required')
    value = dict(version=VERSION, run=binding.run, checkpoint=binding.checkpoint,
        optimizer_step=binding.optimizer_step, source_closure_sha256=source_closure_sha256,
        initial=True, **scope)
    with _locked(path) as path:
        if path.exists():
            current = json.loads(path.read_bytes())
            if current != value:
                raise ValueError('existing private CURRENT cannot be reinitialized')
        else:
            _write(path, value)
    return value


def commit_private_current(path, transaction, envelope, *, authority, run,
                           source_closure_sha256, assert_trainer_lease,
                           release_policy=None, manifest=None):
    """CAS advance after signed R2 readback, before old local state retirement.

    The original report binds the input checkpoint and optimizer step. Retrying
    an already committed result is idempotent; a stale/different run cannot
    overwrite CURRENT. This does not expose a public opening or mint capabilities.
    """
    assert_trainer_lease()
    published = validate_publication(envelope, authority=authority, transaction=transaction,
                                     source_closure_sha256=source_closure_sha256)
    original = transaction['report']['input']
    scope = _release_scope(original, release_policy, manifest)
    if (original['run'] != run or
            original['checkpoint'] != published['input_checkpoint']):
        raise ValueError('run original update binding')
    if release_policy is not None:
        namespace = release_policy.namespace(manifest)
        for name, row in published['objects'].items():
            if row['key'] != namespace + '/checkpoints/' + published['output_checkpoint'] + '/model/' + name:
                raise ValueError('model publication is outside qualified release namespace')
    value = dict(version=VERSION, run=run, checkpoint=published['output_checkpoint'],
        optimizer_step=published['optimizer_step'], source_closure_sha256=source_closure_sha256,
        initial=False, epoch=published['epoch'], job_id=published['job_id'],
        input_checkpoint=published['input_checkpoint'], descriptor=published['descriptor'],
        publication_receipt_sha256=digest(envelope), report_sha256=transaction['report_sha256'], **scope)
    with _locked(path) as path:
        if not path.is_file() or path.is_symlink():
            raise ValueError('original initialized private CURRENT required')
        current = json.loads(path.read_bytes())
        if current == value:
            return value
        if (current.get('version') != VERSION or current.get('run') != run or
                current.get('checkpoint') != original['checkpoint'] or
                current.get('optimizer_step') != original['optimizer_step'] or
                current.get('source_closure_sha256') != source_closure_sha256 or
                current.get('release_policy_sha256') != scope.get('release_policy_sha256') or
                current.get('execution_mode') != scope.get('execution_mode')):
            raise ValueError('CURRENT changed from original update input')
        assert_trainer_lease()
        _write(path, value)
    return value


def private_current_callback(path, publisher, *, run, assert_trainer_lease,
                             release_policy=None, manifest=None):
    """Supply as execute_transaction(commit_current=...) for each private epoch."""
    def commit(transaction):
        envelope = publisher.receipt(transaction['job']['job_id'])
        return commit_private_current(path, transaction, envelope,
            authority=publisher.signer.id, run=run,
            source_closure_sha256=publisher.source, assert_trainer_lease=assert_trainer_lease,
            release_policy=release_policy, manifest=manifest)
    return commit
