"""Opt-in research ledger: durable selection claims, not optimizer atomicity.

No production caller. An unresolved execution is never automatically retried.
Recovery requires the SAME original job's independently authenticated durable
state/receipt; caller supplies that validation, not a self-asserted boolean field.
"""
import copy
from contextlib import contextmanager, closing
import json
import os
from pathlib import Path
import sqlite3

from ops.paired_quota_qualification import digest, _sha, _text

APPLICATION_ID = 1095782993


class RecoveryRequired(RuntimeError):
    pass


def validate_revision(revision):
    fields = {'kind', 'slot_id', 'quota', 'pairs', 'contribution_units',
              'pair_weight_within_task', 'revision_id', 'duplicate_content_count'}
    if not isinstance(revision, dict) or set(revision) != fields:
        raise ValueError('exact selected research revision')
    value = copy.deepcopy(revision)
    _sha(value['slot_id']); _sha(value['revision_id'])
    quota = value['quota']
    if (type(quota) is not int or quota not in (1, 2) or value['kind'] != 'selected-task-revision-v1' or
            type(value['contribution_units']) is not int or value['contribution_units'] != 1 or
            type(value['pair_weight_within_task']) not in (int, float) or
            value['pair_weight_within_task'] != 1 / quota or
            type(value['duplicate_content_count']) is not int or value['duplicate_content_count'] < 0 or
            not isinstance(value['pairs'], list) or len(value['pairs']) != quota):
        raise ValueError('selected research quota/weight')
    executions, contents = set(), set()
    for pair in value['pairs']:
        if not isinstance(pair, dict) or set(pair) != {'positive', 'negative'}:
            raise ValueError('exact selected pair')
        for side in ('positive', 'negative'):
            row = pair[side]
            if (not isinstance(row, dict) or set(row) != {'execution_id', 'content_id', 'classification'} or
                    row['classification'] != side):
                raise ValueError('exact selected member')
            execution = _sha(row['execution_id']); content = _sha(row['content_id'])
            if execution in executions or content in contents:
                raise ValueError('selected pair member reuse')
            executions.add(execution); contents.add(content)
    frozen = {k: v for k, v in value.items() if k not in ('revision_id', 'duplicate_content_count')}
    if digest(frozen) != value['revision_id']:
        raise ValueError('selected revision digest mismatch')
    return frozen, sorted(executions), sorted(contents)


class ResearchTrainingLedger:
    def __init__(self, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research ledger requires explicit opt-in')
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError('create a new owned research ledger first')
        with self._connection() as database:
            if database.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID:
                raise ValueError('not an owned research ledger')
            if database.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise ValueError('research ledger schema mismatch')

    @classmethod
    def create(cls, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research ledger requires explicit opt-in')
        path = Path(path).resolve()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        with closing(sqlite3.connect(path)) as database:
            database.execute('PRAGMA journal_mode=WAL')
            database.execute('PRAGMA synchronous=FULL')
            database.executescript('''
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY, binding TEXT NOT NULL, binding_digest TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('reserved','executing','uncertain','complete')),
                    receipt TEXT);
                CREATE TABLE selections (
                    slot_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL REFERENCES jobs(id));
                CREATE TABLE members (
                    execution_id TEXT PRIMARY KEY, content_id TEXT NOT NULL UNIQUE,
                    slot_id TEXT NOT NULL REFERENCES selections(slot_id));
            ''')
            database.execute(f'PRAGMA application_id={APPLICATION_ID}')
            database.execute('PRAGMA user_version=1')
        return cls(path, enabled=True)

    @contextmanager
    def _connection(self):
        database = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        database.row_factory = sqlite3.Row
        database.execute('PRAGMA foreign_keys=ON')
        database.execute('PRAGMA synchronous=FULL')
        try:
            yield database
        finally:
            database.close()

    def inspect(self, job_id):
        with self._connection() as database:
            row = database.execute('SELECT * FROM jobs WHERE id=?', (_text(job_id),)).fetchone()
        if row is None:
            raise ValueError('unknown research job')
        return dict(job_id=row['id'], binding=json.loads(row['binding']),
                    binding_digest=row['binding_digest'], status=row['status'],
                    receipt=json.loads(row['receipt']) if row['receipt'] is not None else None)

    def reserve(self, job_id, *, parent_state_sha256, step_before, settings_sha256, revisions):
        _text(job_id); _sha(parent_state_sha256); _sha(settings_sha256)
        if type(step_before) is not int or step_before < 0 or not isinstance(revisions, list) or not revisions:
            raise ValueError('bounded research job selection required')
        if len(revisions) > 4096:
            raise ValueError('research job selection budget')
        checked = [validate_revision(r) for r in revisions]
        frozen = sorted((v for v, _, _ in checked), key=lambda v: v['slot_id'])
        if len({v['slot_id'] for v in frozen}) != len(frozen):
            raise ValueError('duplicate selected task slot')
        binding = dict(version='paired-quota-research-job-v1', job_id=job_id,
                       parent_state_sha256=parent_state_sha256, step_before=step_before,
                       settings_sha256=settings_sha256, selected_revisions=frozen)
        raw = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False)
        with self._connection() as database:
            database.execute('BEGIN IMMEDIATE')
            try:
                old = database.execute('SELECT binding FROM jobs WHERE id=?', (job_id,)).fetchone()
                if old is not None:
                    if old['binding'] != raw:
                        raise ValueError('research job binding changed')
                else:
                    database.execute('INSERT INTO jobs VALUES(?,?,?, ?,NULL)',
                                     (job_id, raw, digest(binding), 'reserved'))
                    for value, executions, contents in checked:
                        database.execute('INSERT INTO selections VALUES(?,?,?)',
                                         (value['slot_id'], digest(value), job_id))
                        for pair in value['pairs']:
                            for side in ('positive', 'negative'):
                                member = pair[side]
                                database.execute('INSERT INTO members VALUES(?,?,?)',
                                                 (member['execution_id'], member['content_id'], value['slot_id']))
                database.commit()
            except Exception:
                database.rollback()
                raise
        return self.inspect(job_id)

    def execute_once(self, job_id, execute, authenticate_receipt):
        """Claim before calling executor. COMPLETE redelivery never calls it again.

        A crash after claim, even before any optimizer work, remains ambiguous.
        There are no expiry leases or reset/retry transitions for such claims.
        """
        with self._connection() as database:
            database.execute('BEGIN IMMEDIATE')
            row = database.execute('SELECT status,receipt FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                database.rollback(); raise ValueError('unknown research job')
            if row['status'] == 'complete':
                database.commit(); return json.loads(row['receipt'])
            if row['status'] != 'reserved':
                database.rollback(); raise RecoveryRequired('original execution unresolved; never auto-reapply')
            database.execute("UPDATE jobs SET status='executing' WHERE id=?", (job_id,))
            database.commit()
        try:
            receipt = execute(self.inspect(job_id)['binding'])
            return self.recover_complete(job_id, receipt, authenticate_receipt)
        except BaseException:
            with self._connection() as database:
                database.execute("UPDATE jobs SET status='uncertain' WHERE id=? AND status='executing'", (job_id,))
            raise

    def recover_complete(self, job_id, receipt, authenticate_receipt):
        """Record original durable outcome only after independent authentication.

        This does not run an optimizer, publish a checkpoint, or mint authority.
        External receipt must bind the exact original job, inputs, parent and
        one-step optimizer lineage plus durable output state/checkpoint readback.
        """
        job = self.inspect(job_id)
        expected = dict(job_id=job_id, binding_digest=job['binding_digest'],
                        parent_state_sha256=job['binding']['parent_state_sha256'],
                        step_before=job['binding']['step_before'],
                        step_after=job['binding']['step_before'] + 1)
        if not isinstance(receipt, dict) or set(receipt) != set(expected) | {'output_state_sha256', 'output_checkpoint_sha256'}:
            raise ValueError('exact original outcome receipt')
        if any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('original outcome lineage mismatch')
        _sha(receipt['output_state_sha256']); _sha(receipt['output_checkpoint_sha256'])
        if not callable(authenticate_receipt) or authenticate_receipt(copy.deepcopy(receipt), copy.deepcopy(job['binding'])) is not True:
            raise ValueError('independently authenticated durable outcome required')
        raw = json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False)
        with self._connection() as database:
            database.execute('BEGIN IMMEDIATE')
            row = database.execute('SELECT status,receipt FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row['status'] == 'complete':
                if row['receipt'] != raw:
                    database.rollback(); raise ValueError('completed original receipt changed')
            elif row['status'] in ('executing', 'uncertain'):
                database.execute("UPDATE jobs SET status='complete',receipt=? WHERE id=?", (raw, job_id))
            else:
                database.rollback(); raise ValueError('original execution was not started')
            database.commit()
        return copy.deepcopy(receipt)
