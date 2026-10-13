"""Host-local, single-update journal for the prospective complete-group trainer.

The host authenticates job, source, native preparation and checkpoint inputs.
Callbacks own the actual model/optimizer export and idempotent model publication.
An interrupted update is NEVER retried here. Publication can resume from staged
bytes without another optimizer step. Optimizer files remain on the trainer.
This module is not a production controller or an optimizer rollback mechanism.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat as file_stat
import tempfile
import threading

VERSION = 'affine-mimo-grpo-update-transaction-v1'
TRAINER = 'affine-mimo-grpo-update-v1'
MAX_RECORD = 16 * 1024**2
MAX_FILES = 16384
MAX_ARTIFACT_BYTES = 256 * 1024**3


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def need(v, message):
    if not v:
        raise ValueError(message)


def sha(v):
    return type(v) is str and len(v) == 64 and set(v) <= set('0123456789abcdef')


class RecoveryRequired(RuntimeError):
    """Original update may have mutated parameters; recover original evidence."""


@dataclass(frozen=True)
class Job:
    # Only authenticated, credential-free pins; no remote URLs in this record.
    job_id: str
    epoch: str
    manifest_sha256: str
    source_closure_sha256: str
    objective_sha256: str
    update_pin_sha256: str
    input_checkpoint: str
    prepared_documents_sha256: tuple[str, ...]
    optimizer_step_before: int

    def validate(self):
        need(type(self) is Job, 'host-owned job required')
        for v in (self.job_id, self.epoch):
            need(type(v) is str and 0 < len(v) <= 200 and all(32 <= ord(c) < 127 for c in v), 'bounded job identity')
        need(all(sha(getattr(self, k)) for k in ('manifest_sha256', 'source_closure_sha256',
             'objective_sha256', 'update_pin_sha256', 'input_checkpoint')), 'complete authenticated job pins')
        need(type(self.prepared_documents_sha256) is tuple and 1 <= len(self.prepared_documents_sha256) <= 4096
             and all(sha(v) for v in self.prepared_documents_sha256)
             and len(set(self.prepared_documents_sha256)) == len(self.prepared_documents_sha256), 'unique prepared group documents')
        need(type(self.optimizer_step_before) is int and self.optimizer_step_before >= 0, 'original optimizer step')


def _json_copy(v):
    return json.loads(canonical(v))


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _write(path, value):
    raw = canonical(value)
    need(len(raw) <= MAX_RECORD, 'bounded transaction record')
    fd, temporary = tempfile.mkstemp(prefix='.journal-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _file(root, name):
    need(type(name) is str, 'relative staged artifact name')
    p = PurePosixPath(name)
    need(name and not p.is_absolute() and '..' not in p.parts and str(p) == name
         and p.parts[0] in ('model', 'optimizer'), 'scoped staged model or local optimizer artifact')
    path = root / name
    need(path.resolve() == path and path.is_file() and not path.is_symlink(), 'ordinary local staged file')
    return path


def _stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _ordinary_stamp(path):
    # Check every ancestor too: O_NOFOLLOW only protects the final component.
    need(path.resolve() == path, 'ordinary local staged file ancestry')
    value = path.lstat()
    need(file_stat.S_ISREG(value.st_mode), 'ordinary local staged file')
    return _stamp(value)


def _hash_file(path, *, memo=None):
    """Hash durable bytes, optionally reusing this uninterrupted owner's digest.

    The caller keeps this memo only inside one exclusive transaction.run call.
    It is never serialized or retained across publication failure/recovery. A
    fresh descriptor and full inode stamp are checked even on cache hits;
    changed ctime catches same-size writes whose mtime was restored. Changed
    stamps force a full read and comparison against the original expected hash.

    This is an exclusive-owner coherence optimization, not a defense against a
    concurrent malicious same-UID writer. Filesystem timestamps can share a
    clock tick, so metadata cannot prove that hostile writes never occurred.
    The private staging directory and held owner lock exclude legitimate
    concurrent writers; all cold/recovery checks read every artifact again.
    """
    before_path = _ordinary_stamp(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        need(file_stat.S_ISREG(before.st_mode) and _stamp(before) == before_path,
             'staged artifact changed before hashing')
        need(before.st_size <= MAX_ARTIFACT_BYTES, 'bounded artifact bytes')
        cached = None if memo is None else memo.get(path)
        if cached is not None and cached[0] == _stamp(before):
            os.fsync(f.fileno())
            need(_stamp(os.fstat(f.fileno())) == before_path == _ordinary_stamp(path),
                 'staged artifact changed during digest reuse')
            return dict(cached[1])
        h = hashlib.sha256()
        total = 0
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            total += len(chunk)
            need(total <= before.st_size, 'staged artifact grew during hashing')
            h.update(chunk)
        after = os.fstat(f.fileno())
        need(total == before.st_size and _stamp(before) == _stamp(after),
             'staged artifact changed during hashing')
        os.fsync(f.fileno())
        need(_stamp(os.fstat(f.fileno())) == before_path == _ordinary_stamp(path),
             'staged artifact changed after hashing')
    result = dict(sha256=h.hexdigest(), size=total)
    if memo is not None:
        memo[path] = (before_path, dict(result))
    return result


def _checkpoint(artifacts):
    need(type(artifacts) is dict and 2 <= len(artifacts) <= MAX_FILES, 'staged artifact metadata')
    for name, row in artifacts.items():
        p = PurePosixPath(name)
        need(type(name) is str and name and not p.is_absolute() and '..' not in p.parts
             and str(p) == name and p.parts[0] in ('model', 'optimizer'), 'scoped staged artifact metadata')
        need(type(row) is dict and set(row) == {'size','sha256'} and sha(row['sha256'])
             and type(row['size']) is int and 0 <= row['size'] <= MAX_ARTIFACT_BYTES, 'staged file digest and size')
    need(sum(r['size'] for r in artifacts.values()) <= MAX_ARTIFACT_BYTES, 'total staged artifact metadata cap')
    need(any(n.startswith('optimizer/') for n in artifacts), 'retained local optimizer metadata')
    models = {n.removeprefix('model/'): row for n, row in artifacts.items() if n.startswith('model/')}
    need(1 <= len(models) <= 32 and 'config.json' in models and any(n.endswith('.safetensors') for n in models)
         and all(re.fullmatch(r'[A-Za-z0-9_.-]+',n) and not n.startswith('.')
                 and Path(n).suffix in {'.json','.safetensors','.txt','.model','.jinja','.tiktoken'} for n in models),
         'checkpoint metadata allowlist')
    return digest({name: row['sha256'] for name,row in models.items()}), models


def _receipt(job, checkpoint, value):
    need(type(value) is dict and set(value) == {'job_id','output_checkpoint','model_full_get_verified','optimizer_state_uploaded','receipt_sha256'},
         'bounded model publication receipt')
    need(value['job_id'] == job.job_id and value['output_checkpoint'] == checkpoint
         and value['model_full_get_verified'] is True and value['optimizer_state_uploaded'] is False
         and sha(value['receipt_sha256']), 'publication must confirm original model, never optimizer upload')


def _inventory(root, names, *, hash_memo=None):
    need(type(names) is list and 2 <= len(names) <= MAX_FILES and all(type(n) is str for n in names)
         and len(names) == len(set(names)), 'explicit unique bounded model and optimizer files')
    paths = {name: _file(root, name) for name in sorted(names)}
    need(sum(p.stat().st_size for p in paths.values()) <= MAX_ARTIFACT_BYTES, 'total staged artifact cap')
    model_names = [n.removeprefix('model/') for n in names if n.startswith('model/')]
    need(1 <= len(model_names) <= 32 and 'config.json' in model_names
         and any(n.endswith('.safetensors') for n in model_names)
         and all(re.fullmatch(r'[A-Za-z0-9_.-]+', n) and not n.startswith('.')
                 and Path(n).suffix in {'.json','.safetensors','.txt','.model','.jinja','.tiktoken'}
                 for n in model_names), 'actual checkpoint file allowlist')
    result = {name: _hash_file(path, memo=hash_memo) for name, path in paths.items()}
    need(any(n.startswith('model/') for n in result) and any(n.startswith('optimizer/') for n in result),
         'stage model and local optimizer together')
    need(sum(r['size'] for r in result.values()) <= MAX_ARTIFACT_BYTES, 'total staged artifact cap')
    # File fsync alone does not persist a newly created directory entry.
    parents = {p for f in paths.values() for p in f.parents if p == root or root in p.parents}
    for path in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        _sync_directory(path)
    return result


def _validate_report(job, report):
    need(type(report) is dict and report.get('version') == TRAINER, 'actual MiMo group trainer report')
    pin = report.get('input')
    need(type(pin) is dict and digest(pin) == job.update_pin_sha256
         and pin.get('checkpoint') == job.input_checkpoint and pin.get('epoch') == job.epoch
         and pin.get('optimizer_step') == job.optimizer_step_before, 'original update report binding')
    need(type(report.get('optimizer_step')) is int and report['optimizer_step'] == job.optimizer_step_before + 1, 'exactly one original update')
    need(report.get('input_documents_sha256') == list(job.prepared_documents_sha256), 'original ordered training groups')
    need(digest(report.get('config')) == job.objective_sha256, 'original objective')
    need(report.get('runtime_context', {}).get('manifest_sha256') == job.manifest_sha256,
         'report authenticated manifest binding')


class Transaction:
    def __init__(self, workspace, job):
        job.validate()
        self.root = Path(workspace).absolute()
        need(self.root.resolve() == self.root, 'ordinary transaction directory ancestry')
        missing = []
        current = self.root
        while not current.exists():
            missing.append(current)
            current = current.parent
        for path in reversed(missing):
            try: path.mkdir(mode=0o700)
            except FileExistsError: need(path.is_dir() and not path.is_symlink(), 'ordinary concurrently created directory')
            _sync_directory(path.parent)
        need(self.root.resolve() == self.root and not self.root.is_symlink(), 'private ordinary transaction directory')
        need(self.root.stat().st_uid == os.getuid() and self.root.stat().st_mode & 0o077 == 0,
             'owner-only transaction workspace')
        self.job = job
        self.path = self.root / 'TRANSACTION.json'

    @contextmanager
    def exclusive(self):
        fd = os.open(self.root / '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def _read(self):
        if not self.path.exists(): return None
        need(not self.path.is_symlink() and self.path.is_file() and self.path.stat().st_size <= MAX_RECORD,
             'bounded ordinary journal')
        value = json.loads(self.path.read_bytes())
        need(type(value) is dict and value.get('version') == VERSION
             and value.get('job') == _json_copy(asdict(self.job)), 'journal original job identity')
        need(value.get('state') in ('prepared', 'updating', 'staged', 'published'), 'known transaction stage')
        return value

    def run(self, *, update, stage=None, publish_model, stage_with_hasher=None):
        """Run original update or resume model publication from complete staging.

        update() invokes the trusted trainer exactly once and returns its report.
        stage(report, root) must export/fsync both model and local optimizer state,
        returning all relative file names. Never overwrite a prior job directory.
        Alternatively stage_with_hasher(report, root, hash_artifact) may reuse
        guarded full reads performed while writing artifacts. The callback can
        request an actual hash, never seed a digest from a descriptor. It is
        confined to this process/thread and the active exclusive staging call.
        Every invocation/recovery starts with an empty in-memory receipt set.
        publish_model(request) uses immutable model objects and idempotent metadata;
        after lost acknowledgement it must reconcile original bytes, not retrain.
        It receives no optimizer file paths or bytes. Returned receipt is checked.
        A stage/update exception leaves `updating` for explicit original recovery.
        The caller separately holds the sole trainer/optimizer lease; this lock
        excludes concurrent executions of this job, not unrelated job directories.
        """
        need((callable(stage) and stage_with_hasher is None) or
             (stage is None and callable(stage_with_hasher)), 'one explicit staging callback protocol')
        with self.exclusive():
            # No object-level or persistent memo: even a retry on this same
            # Transaction object starts with cold artifact verification.
            hash_memo = {}
            value = self._read()
            if value is None:
                value = dict(version=VERSION, job=_json_copy(asdict(self.job)), state='prepared')
                _write(self.path, value)
            if value['state'] == 'updating':
                raise RecoveryRequired('interrupted original update; do not run another optimizer step')
            if value['state'] == 'published':
                # Retained journal must remain self-consistent even after local
                # model cache cleanup. Do not silently return a damaged pointer.
                need(value.get('report_sha256') == digest(value.get('report')), 'published report integrity')
                _validate_report(self.job, value['report'])
                checkpoint, _ = _checkpoint(value.get('artifacts'))
                need(value.get('output_checkpoint') == checkpoint, 'published checkpoint integrity')
                _receipt(self.job, checkpoint, value.get('publication'))
                need(value.get('authority_committed') is False and value.get('automatic_retraining') is False,
                     'published transaction is not authority commitment or retraining permission')
                return value
            if value['state'] == 'prepared':
                value['state'] = 'updating'
                _write(self.path, value)
                report = _json_copy(update())
                _validate_report(self.job, report)
                stage_hashes = {}
                if stage_with_hasher is None:
                    names = stage(report, self.root)
                else:
                    owner = (os.getpid(), threading.get_ident())
                    active = True
                    def hash_artifact(path):
                        need(active and owner == (os.getpid(), threading.get_ident()),
                             'artifact hasher requires its active staging owner')
                        path = Path(path)
                        need(path.is_absolute() and self.root in path.parents,
                             'artifact hasher stays within its original transaction')
                        relative = path.relative_to(self.root).as_posix()
                        path = _file(self.root, relative)
                        row = _hash_file(path, memo=hash_memo)
                        need(path not in stage_hashes or stage_hashes[path] == row,
                             'artifact changed after original staging hash')
                        stage_hashes[path] = dict(row)
                        return dict(row)
                    try:
                        names = stage_with_hasher(report, self.root, hash_artifact)
                    finally:
                        active = False
                inventory = _inventory(self.root, names, hash_memo=hash_memo)
                need(all(row == inventory[path.relative_to(self.root).as_posix()]
                         for path, row in stage_hashes.items()
                         if path.relative_to(self.root).as_posix() in inventory),
                     'staged artifact integrity differs from original write hash')
                value.update(state='staged', report=report, report_sha256=digest(report), artifacts=inventory)
                _write(self.path, value)
            need(value['report_sha256'] == digest(value['report']), 'staged report integrity')
            _validate_report(self.job, value['report'])
            need(_inventory(self.root, list(value['artifacts']), hash_memo=hash_memo) == value['artifacts'],
                 'staged artifact integrity')
            checkpoint, models = _checkpoint(value['artifacts'])
            request = dict(version=VERSION, job_id=self.job.job_id, epoch=self.job.epoch,
                input_checkpoint=self.job.input_checkpoint, output_checkpoint=checkpoint,
                report_sha256=value['report_sha256'], model_directory=str(self.root/'model'), model_files=models,
                optimizer_step=self.job.optimizer_step_before+1, optimizer_state_location='trainer-local',
                optimizer_state_uploaded=False, authority_committed=False)
            receipt = _json_copy(publish_model(request))
            _receipt(self.job, checkpoint, receipt)
            value.update(state='published', output_checkpoint=checkpoint, publication=receipt,
                         authority_committed=False, automatic_retraining=False)
            _write(self.path, value)
            return value
