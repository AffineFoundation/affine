"""Private, bounded search progress. Never an upload or verification authority.

SQLite stores strict JSON metadata and the existing safe ZIP/NPY proof format.
A nonce is committed BEFORE generation: interruption may lose one attempt, but
will not silently replay it. The local lock excludes concurrent search/reset;
this is deliberately not a cross-machine nonce coordination service.
"""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import io
import zipfile

from .batches import pack, unpack
from .artifact_budget import for_manifest
from .forced_sampling import binding, receipt, MINER_VERSION
from .protocol import entries
from .sampling_uniqueness import content_digest
from .storage import canonical

MAX_PARTIAL_TASKS = 256
MAX_PARTIAL_BYTES = 64 * 1024 * 1024
MAX_PARTIAL_RAW_BYTES = 128 * 1024 * 1024
MAX_DATABASE_BYTES = 96 * 1024 * 1024
VERSION = 'private-miner-search-v1'


class NoncesExhausted(RuntimeError):
    """This task has no unused manifest-authorized local attempts remaining."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


class SearchState:
    def __init__(self, path, manifest, miner, *, retire_previous=False):
        self.db = None
        self.lock_fd = None
        try:
            self._initialize(path, manifest, miner, retire_previous=retire_previous)
        except BaseException:
            self.close()
            raise

    def _initialize(self, path, manifest, miner, *, retire_previous=False):
        self.manifest, self.miner = manifest, miner
        self.context = binding(manifest, miner)
        need(self.context['contract']['version'] == MINER_VERSION, 'v5 search journal required')
        self.maximum = self.context['contract']['max_attempts']
        self.definitions = {row['env_id']: row for row in entries(manifest)}
        self.allowed = {env: set(row['indices']) for env, row in self.definitions.items()}
        deadline = manifest.get('deadline')
        need(type(deadline) in (int, float) and math.isfinite(deadline), 'signed search deadline')
        self.scope = dict(version=VERSION, epoch=manifest['epoch'], miner=miner, deadline=deadline,
                          manifest_sha256=digest(canonical(manifest)))
        self.scope_bytes = canonical(self.scope)
        self.path = Path(path) if path else None
        self.lock_fd = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            need(not self.path.parent.is_symlink(), 'private search parent path')
            for file in (self.path, self.path.with_name(self.path.name + '.lock')):
                fd = os.open(file, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                os.fchmod(fd, 0o600)
                if file == self.path:
                    os.close(fd)
                else:
                    self.lock_fd = fd
        self.db = sqlite3.connect(str(self.path) if self.path else ':memory:', timeout=0)
        with self.locked():
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA auto_vacuum=FULL')
            page_size = self.db.execute('PRAGMA page_size').fetchone()[0]
            self.db.execute('PRAGMA max_page_count=' + str(MAX_DATABASE_BYTES // page_size))
            need(self.db.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'search database integrity')
            with self.db:
                self.db.execute('CREATE TABLE IF NOT EXISTS scope (singleton INTEGER PRIMARY KEY CHECK(singleton=1), body BLOB NOT NULL, sha TEXT NOT NULL)')
                self.db.execute('CREATE TABLE IF NOT EXISTS task (key TEXT PRIMARY KEY, next INTEGER NOT NULL, complete INTEGER NOT NULL, proof BLOB, sha TEXT NOT NULL, touched INTEGER NOT NULL)')
                row = self.db.execute('SELECT body,sha FROM scope WHERE singleton=1').fetchone()
                if row is not None:
                    need(row[1] == digest(row[0]), 'search scope digest')
                    old = json.loads(row[0])
                    if row[0] != self.scope_bytes:
                        need(retire_previous and old.get('version') == VERSION and old.get('miner') == miner
                             and old.get('epoch') != manifest['epoch']
                             and type(old.get('deadline')) in (int, float) and deadline > old['deadline'], 'stale local search binding')
                        # Caller has authenticated the next epoch. Atomic retirement,
                        # never relabel an old proof/cursor to a new manifest.
                        self.db.execute('DELETE FROM task')
                self.db.execute('INSERT OR REPLACE INTO scope VALUES (1,?,?)',
                                (self.scope_bytes, digest(self.scope_bytes)))
            self._check_scope()
            for key, nonce, complete, proof, sha, touched in self.db.execute('SELECT * FROM task'):
                self._check_row(key, nonce, complete, proof, sha)
            self._check_bounds()

    @contextlib.contextmanager
    def locked(self):
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('local miner search cache already in use') from exc
        try:
            yield
        finally:
            if self.lock_fd is not None:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def close(self):
        if getattr(self, 'db', None) is not None:
            self.db.close()
            self.db = None
        if getattr(self, 'lock_fd', None) is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _check_scope(self):
        row = self.db.execute('SELECT body,sha FROM scope WHERE singleton=1').fetchone()
        need(row == (self.scope_bytes, digest(self.scope_bytes)), 'changed local search binding')

    def _key(self, env, index):
        need(type(index) is int and env in self.allowed and index in self.allowed[env], 'authorized search task')
        return canonical([env, index]).decode()

    def _row_digest(self, key, nonce, complete, proof):
        return digest(canonical([digest(self.scope_bytes), key, nonce, complete,
                                 digest(proof) if proof is not None else None]))

    def _check_row(self, key, nonce, complete, proof, sha):
        parsed = json.loads(key)
        need(type(parsed) is list and len(parsed) == 2 and self._key(*parsed) == key, 'search task key')
        need(type(nonce) is int and 0 <= nonce <= self.maximum and complete in (0, 1), 'search cursor range')
        need(proof is None or (type(proof) is bytes and 0 < len(proof) <= MAX_PARTIAL_BYTES), 'bounded partial proof')
        need(not complete or proof is None, 'completed search state has no partial proof')
        need(sha == self._row_digest(key, nonce, complete, proof), 'search row digest')

    def _read(self, key):
        self._check_scope()
        row = self.db.execute('SELECT next,complete,proof,sha FROM task WHERE key=?', (key,)).fetchone()
        if row is None:
            return 0, 0, None
        self._check_row(key, *row)
        return row[:3]

    def _write(self, key, nonce, complete, proof):
        self.db.execute('INSERT OR REPLACE INTO task VALUES (?,?,?,?,?,?)',
                        (key, nonce, complete, proof, self._row_digest(key, nonce, complete, proof),
                         self.db.execute('SELECT COALESCE(MAX(touched),0)+1 FROM task').fetchone()[0]))

    def _check_bounds(self):
        count, size = self.db.execute('SELECT COUNT(proof),COALESCE(SUM(LENGTH(proof)),0) FROM task').fetchone()
        need(count <= MAX_PARTIAL_TASKS and size <= MAX_PARTIAL_BYTES, 'partial cache bounds')
        need(self.db.execute('SELECT COUNT(*) FROM task').fetchone()[0] <= sum(map(len, self.allowed.values())), 'search ledger task bound')

    def _batch(self, env, index, rolls):
        return dict(schema=2, epoch=self.manifest['epoch'], checkpoint=self.manifest['checkpoint']['id'],
                    env_id=env, environment_version=self.definitions[env]['spec']['version'],
                    sample_index=index, index=index, rollouts=rolls)

    def _validate(self, env, index, rolls, nonce):
        need(type(rolls) is list and len(rolls) <= self.manifest['K'] + self.manifest['L'], 'partial rollout count')
        attempts, contents, hashes = set(), set(), set()
        counts = {'positive': 0, 'negative': 0}
        expected = self._batch(env, index, [])
        for roll in rolls:
            need(type(roll) is dict, 'partial rollout object')
            need(all(roll.get(k) == expected[k] for k in ('env_id', 'environment_version', 'index', 'sample_index'))
                 and type(roll.get('index')) is int and type(roll.get('sample_index')) is int, 'partial same-task binding')
            attempt = roll.get('seed')
            need(type(attempt) is int and 0 <= attempt < nonce and roll.get('sampling') == receipt(self.context, attempt), 'partial attempt binding')
            content = content_digest(roll)
            need(attempt not in attempts and content not in contents, 'duplicate partial attempt or output')
            attempts.add(attempt); contents.add(content)
            task_hash = roll.get('task_hash')
            need(type(task_hash) is str and len(task_hash) == 64 and all(c in '0123456789abcdef' for c in task_hash), 'partial task hash')
            hashes.add(task_hash)
            kind = roll.get('classification')
            need(kind in counts, 'partial completed outcome required')
            counts[kind] += 1
        need(len(hashes) <= 1 and counts['positive'] <= self.manifest['K'] and counts['negative'] <= self.manifest['L'], 'partial outcome quotas')

    def _decode(self, env, index, nonce, proof):
        if proof is None:
            return [], []
        with zipfile.ZipFile(io.BytesIO(proof)) as archive:
            need(sum(row.file_size for row in archive.infolist()) <= MAX_PARTIAL_RAW_BYTES, 'partial decompression bound')
        rows = unpack(proof, budget=for_manifest(self.manifest))
        need(len(rows) == 1, 'one local partial task')
        batch, arrays = rows[0]
        need(batch == self._batch(env, index, batch.get('rollouts')), 'partial artifact scope')
        self._validate(env, index, batch['rollouts'], nonce)
        need(len(arrays) == len(batch['rollouts']) and all(len(a) == len(r['turns']) for r,a in zip(batch['rollouts'], arrays)), 'partial proof alignment')
        return batch['rollouts'], arrays

    def load(self, env, index):
        nonce, complete, proof = self._read(self._key(env, index))
        rolls, arrays = self._decode(env, index, nonce, proof)
        return nonce, bool(complete), rolls, arrays

    def reserve(self, env, index, start):
        need(type(start) is int and 0 <= start < self.maximum, 'search nonce start range')
        key = self._key(env, index)
        with self.db:
            nonce, complete, proof = self._read(key)
            need(not complete, 'task already completed locally')
            nonce = max(nonce, start)
            if nonce >= self.maximum:
                raise NoncesExhausted('all manifest-authorized task nonces exhausted')
            self._write(key, nonce + 1, 0, proof)
        return nonce

    def accept(self, env, index, rollout, arrays, attempt):
        key = self._key(env, index)
        with self.db:
            nonce, complete, proof = self._read(key)
            need(not complete, 'task already completed locally')
            rolls, old_arrays = self._decode(env, index, nonce, proof)
            need(type(attempt) is int and rollout.get('seed') == attempt, 'generated attempt matches reserved nonce')
            self._validate(env, index, [rollout], nonce)
            # Do not retain neutral, quota padding, repeated attempts or text.
            kind = rollout['classification']
            limit = self.manifest['K' if kind == 'positive' else 'L']
            if (sum(r['classification'] == kind for r in rolls) >= limit
                    or any(r['seed'] == rollout['seed'] or content_digest(r) == content_digest(rollout) for r in rolls)):
                return rolls, old_arrays
            rolls = rolls + [rollout]; old_arrays = old_arrays + [arrays]
            self._validate(env, index, rolls, nonce)
            need(len(old_arrays) == len(rolls) and all(len(a) == len(r['turns']) for r,a in zip(rolls, old_arrays)), 'partial proof alignment')
            need(sum(t.nbytes for a in old_arrays for t in a) + len(canonical(rolls)) < MAX_PARTIAL_RAW_BYTES, 'partial raw proof bound')
            body = pack([(self._batch(env, index, rolls), old_arrays)], budget=for_manifest(self.manifest), stable=True)
            need(len(body) <= MAX_PARTIAL_BYTES, 'one partial exceeds local cache bound')
            self._write(key, nonce, 0, body)
            while True:
                count, size = self.db.execute('SELECT COUNT(proof),COALESCE(SUM(LENGTH(proof)),0) FROM task').fetchone()
                if count <= MAX_PARTIAL_TASKS and size <= MAX_PARTIAL_BYTES:
                    break
                victim = self.db.execute('SELECT key,next,complete FROM task WHERE proof IS NOT NULL AND key<>? ORDER BY touched LIMIT 1', (key,)).fetchone()
                need(victim is not None, 'one partial exceeds local cache bound')
                self._write(*victim, None)
            self._check_bounds()
        return rolls, old_arrays

    def completed(self, batches):
        """Call only after complete batch state is durable, never after GPU output alone."""
        with self.db:
            self._check_scope()
            for batch, _ in batches:
                key = self._key(batch['env_id'], batch['index'])
                nonce, _, _ = self._read(key)
                next_nonce = max([nonce] + [r['seed'] + 1 for r in batch['rollouts']])
                self._write(key, next_nonce, 1, None)
