"""Private bounded working-state journal for original complete-group attempts.

The journal is an execution aid, not a verifier or anti-adversarial ledger. The
server must independently enforce canonical nonces and one group per task.
Completed filtered/indeterminate task dispositions survive restarts, while their
large local attempt payloads are removed automatically after that disposition is
durable. Accepted payloads are removed only after the transport persists them.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .batches import compression_for_manifest
from .complete_group_codec import encode, decode, limits
from .storage import canonical

VERSION = 'complete-group-attempt-journal-v2'
MAX_JOURNAL_BYTES = 32 * 1024 * 1024


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, body):
    need(not path.is_symlink(), 'private group state must not follow symlinks')
    temp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


class AttemptJournal:
    def __init__(self, path, manifest, miner):
        self.path = Path(str(path)+'.groups.json') if path is not None else None
        self.directory = Path(str(path)+'.group-attempts') if path is not None else None
        self.manifest = manifest
        self.budget = limits(manifest)
        self.n = manifest['group_contract']['group_size']
        self.lock = None
        self.blobs = {}
        self.binding = dict(epoch=manifest['epoch'], checkpoint=manifest['checkpoint']['id'],
                            miner=miner, manifest_sha256=sha(canonical(manifest)))
        self.value = dict(version=VERSION, binding=self.binding, tasks={})
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        need(not self.path.is_symlink() and not self.directory.is_symlink(), 'private group journal path')
        lock_path = Path(str(self.path)+'.lock')
        self.lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.directory.mkdir(mode=0o700, exist_ok=True)
            self.directory.chmod(0o700)
            if self.path.exists():
                need(self.path.stat().st_size <= MAX_JOURNAL_BYTES, 'bounded group journal')
                value = json.loads(self.path.read_bytes())
                need(type(value) is dict and set(value) == {'version', 'binding', 'tasks'}
                     and value['version'] == VERSION and value['binding'] == self.binding,
                     'stale group journal: checkpoint, manifest, N, miner or source changed')
                need(type(value['tasks']) is dict and len(value['tasks']) <= 640000, 'bounded group task ledger')
                for key, row in value['tasks'].items():
                    need(type(key) is str and len(key) == 64 and set(key) <= set('0123456789abcdef'),
                         'canonical journal task key')
                    need(type(row) is dict and row.get('status') in ('pending', 'accepted', 'filtered', 'native_error')
                         and type(row.get('attempts')) is list and len(row['attempts']) <= self.n,
                         'group journal disposition or attempt count')
                    for nonce, item in enumerate(row['attempts']):
                        need(type(item) is dict and set(item) == {'nonce', 'sha256', 'size'}
                             and type(item['nonce']) is int and item['nonce'] == nonce
                             and type(item['sha256']) is str and len(item['sha256']) == 64
                             and set(item['sha256']) <= set('0123456789abcdef')
                             and type(item['size']) is int and 0 < item['size'] <= self.budget.compressed_bytes,
                             'original ordered attempt journal')
                self.value = value
            else:
                self._persist()
            self._recover_cleanup()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def __del__(self):
        self.close()

    def _persist(self):
        data = canonical(self.value)
        need(len(data) <= MAX_JOURNAL_BYTES, 'bounded group disposition journal')
        if self.path is not None:
            _write(self.path, data)

    def status(self, key):
        return self.value['tasks'].get(key, {}).get('status')

    def begin(self, key):
        row = self.value['tasks'].get(key)
        if row is None:
            row = dict(status='pending', attempts=[])
            self.value['tasks'][key] = row
            self._persist()
        need(row['status'] == 'pending', 'original group has a terminal disposition')
        records = []
        for item in row['attempts']:
            if self.directory is not None:
                path = self.directory / (item['sha256']+'.zip')
                need(not path.is_symlink() and path.stat().st_size == item['size'], 'private original attempt path/size')
                data = path.read_bytes()
            else:
                data = self.blobs[item['sha256']]
            need(len(data) == item['size'] and sha(data) == item['sha256'], 'original attempt SHA256 mismatch')
            decoded = decode(data, self.budget)
            need(len(decoded) == 1 and set(decoded[0][0]) == {'original_rollout'}, 'original attempt archive shape')
            rollout, arrays = decoded[0][0]['original_rollout'], decoded[0][1]
            need(type(rollout.get('seed')) is int and rollout['seed'] == item['nonce'] and len(arrays) == 1,
                 'original attempt nonce/array binding')
            records.append((rollout, arrays[0]))
        return records

    def append(self, key, rollout, arrays):
        row = self.value['tasks'][key]
        nonce = len(row['attempts'])
        need(row['status'] == 'pending' and nonce < self.n and type(rollout['seed']) is int
             and rollout['seed'] == nonce, 'append only the next original nonce')
        data = encode([(dict(original_rollout=rollout), [arrays])], self.budget,
                      level=compression_for_manifest(self.manifest))
        fingerprint = sha(data)
        if self.directory is not None:
            path = self.directory / (fingerprint+'.zip')
            need(not path.is_symlink(), 'private original attempt path')
            if path.exists():
                need(path.read_bytes() == data, 'immutable original attempt bytes')
            else:
                _write(path, data)
        else:
            self.blobs[fingerprint] = data
        row['attempts'].append(dict(nonce=nonce, sha256=fingerprint, size=len(data)))
        self._persist()

    def finish(self, key, status, **details):
        row = self.value['tasks'][key]
        need(row['status'] == 'pending' and status in ('filtered', 'accepted', 'native_error'),
             'one terminal disposition for original group')
        if status in ('accepted', 'filtered'):
            need(len(row['attempts']) == self.n, 'all N attempts before outcome filtering')
        row.update(status=status, **details)
        self._persist()
        self._remove_payloads(row)

    def _remove_payloads(self, row):
        for item in row['attempts']:
            if self.directory is not None:
                path = self.directory / (item['sha256']+'.zip')
                need(not path.is_symlink(), 'private original attempt cleanup path')
                path.unlink(missing_ok=True)
            else:
                self.blobs.pop(item['sha256'], None)

    def _recover_cleanup(self):
        """Finish interrupted disposal, never delete an accepted group's backup.

        The exclusive journal lock owns this directory. A hash-named object
        written before its journal append is an orphan, not a completed attempt;
        restarting redoes that same prescribed nonce. Accepted references stay
        until the separately persisted transport bytes have been reconciled.
        """
        for row in self.value['tasks'].values():
            if row['status'] in ('filtered', 'native_error'):
                self._remove_payloads(row)
        keep = {a['sha256']+'.zip' for row in self.value['tasks'].values()
                if row['status'] in ('pending', 'accepted') for a in row['attempts']}
        for path in self.directory.iterdir():
            owned = re.fullmatch(r'[0-9a-f]{64}\.zip(?:\.tmp-[0-9a-f]{32})?', path.name)
            if owned and path.name not in keep:
                need(not path.is_symlink() and path.is_file(), 'private orphan attempt cleanup path')
                path.unlink()
        for path in self.path.parent.glob(self.path.name+'.tmp-*'):
            if re.fullmatch(re.escape(self.path.name)+r'\.tmp-[0-9a-f]{32}', path.name):
                need(not path.is_symlink() and path.is_file(), 'private orphan journal temporary path')
                path.unlink()

    def reconcile_accepted(self, key, batch):
        row = self.value['tasks'].get(key)
        need(row is not None, 'restored group missing original attempt journal')
        expected = sha(canonical(batch))
        if row['status'] == 'accepted':
            need(row.get('batch_sha256') == expected, 'restored accepted group bytes changed')
            self._remove_payloads(row)
            return
        need(row['status'] == 'pending', 'terminal filtered group cannot become accepted')
        original = self.begin(key)
        need([r for r, _ in original] == batch['rollouts'], 'restored group differs from original attempts')
        self.finish(key, 'accepted', batch_sha256=expected)

    def require_accepted(self, keys):
        need({k for k, r in self.value['tasks'].items() if r['status'] == 'accepted'} == set(keys),
             'accepted group missing transport state')
