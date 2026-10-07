"""Default-off research application claim; no production optimizer caller.

The trusted boundary derives a schedule from authenticated ORIGINAL inputs.
A parent in one authorized branch has one immutable application, independent of
job labels/URLs/wrapper bytes. Unresolved execution is never automatically freed.
"""
import copy
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from ops.paired_quota_qualification import digest, _sha
from ops.paired_quota_research_ledger import RecoveryRequired, validate_revision

VERSION = 'consumed-input-application-research-v1'
APPLICATION_ID = 1095782996


def canonical_binding(value):
    fields = {'version', 'branch_sha256', 'plan_sha256', 'parent_state_sha256',
              'input_checkpoint_sha256', 'step_before', 'settings_sha256',
              'selected_revisions', 'groups'}
    if not isinstance(value, dict) or set(value) != fields or value['version'] != VERSION:
        raise ValueError('exact authenticated application binding')
    value = copy.deepcopy(value)
    for name in fields - {'version', 'step_before', 'selected_revisions', 'groups'}:
        _sha(value[name])
    if type(value['step_before']) is not int or not 0 <= value['step_before'] < 2**31 - 32:
        raise ValueError('actual parent optimizer counter')
    revisions = value['selected_revisions']; groups = value['groups']
    if not isinstance(revisions, list) or not 1 <= len(revisions) <= 4096:
        raise ValueError('bounded selected task population')
    checked = [validate_revision(r)[0] for r in revisions]
    slots = [r['slot_id'] for r in checked]
    if len(set(slots)) != len(slots):
        raise ValueError('duplicate selected task')
    executions, contents = set(), set()
    for r in checked:
        for pair in r['pairs']:
            for member in pair.values():
                if member['execution_id'] in executions or member['content_id'] in contents:
                    raise ValueError('duplicate execution/content across selected tasks')
                executions.add(member['execution_id']); contents.add(member['content_id'])
    if not isinstance(groups, list) or not 1 <= len(groups) <= 32:
        raise ValueError('bounded exact declared update schedule')
    used = set()
    for group in groups:
        if (not isinstance(group, list) or not group or len(group) != len(set(group)) or
                any(s not in slots for s in group)):
            raise ValueError('distinct selected tasks within each update')
        used.update(group)
    if used != set(slots):
        raise ValueError('schedule must cover selected population')
    # Repetition across groups is intentional only if the trusted original plan
    # actually prescribed it. It is not repeated receipt delivery.
    value['selected_revisions'] = sorted(checked, key=lambda r: r['slot_id'])
    return value


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class ConsumedApplicationLedger:
    def __init__(self, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research opt-in required')
        self.path = Path(path).resolve()
        if not self.path.is_file() or self.path.stat().st_uid != os.getuid() or self.path.stat().st_mode & 0o077:
            raise ValueError('owned private ledger required')
        with closing(self._connect()) as db:
            if (db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                    db.execute('PRAGMA user_version').fetchone()[0] != 1):
                raise ValueError('owned application schema required')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.execute('PRAGMA synchronous=FULL'); db.execute('PRAGMA foreign_keys=ON')
        return db

    @classmethod
    def create(cls, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research opt-in required')
        path = Path(path).resolve()
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        with closing(sqlite3.connect(path)) as db:
            db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
            db.executescript('''
              CREATE TABLE applications(id TEXT PRIMARY KEY, parent_slot TEXT UNIQUE NOT NULL,
                binding TEXT NOT NULL, state TEXT CHECK(state IN ('reserved','executing','complete')) NOT NULL,
                receipt TEXT);
              CREATE TABLE originals(original_sha TEXT PRIMARY KEY, application_id TEXT NOT NULL
                REFERENCES applications(id), evidence TEXT NOT NULL);
              CREATE TRIGGER immutable_binding BEFORE UPDATE OF id,parent_slot,binding ON applications
                BEGIN SELECT RAISE(ABORT,'immutable application'); END;
              CREATE TRIGGER immutable_original_update BEFORE UPDATE ON originals
                BEGIN SELECT RAISE(ABORT,'immutable original'); END;
              CREATE TRIGGER immutable_original_delete BEFORE DELETE ON originals
                BEGIN SELECT RAISE(ABORT,'immutable original'); END;
              CREATE TRIGGER no_application_delete BEFORE DELETE ON applications
                BEGIN SELECT RAISE(ABORT,'immutable application'); END;
              CREATE TRIGGER monotonic_state BEFORE UPDATE ON applications
                WHEN OLD.state='complete' OR NOT
                ((OLD.state='reserved' AND NEW.state='executing' AND NEW.receipt IS NULL) OR
                 (OLD.state='executing' AND NEW.state='complete' AND NEW.receipt IS NOT NULL))
                BEGIN SELECT RAISE(ABORT,'unresolved application cannot reset'); END;
            ''')
            db.execute(f'PRAGMA application_id={APPLICATION_ID}'); db.execute('PRAGMA user_version=1')
        return cls(path, enabled=True)

    def reserve_original(self, original_sha256, evidence, authenticate_original):
        _sha(original_sha256)
        if not callable(authenticate_original):
            raise ValueError('trusted original boundary required')
        evidence = copy.deepcopy(evidence)
        encoded = raw(evidence)
        if len(encoded.encode()) > 2 * 1024**2:
            raise ValueError('bounded original evidence')
        binding = canonical_binding(authenticate_original(original_sha256, copy.deepcopy(evidence)))
        aid = digest(binding)
        parent_slot = digest({k: binding[k] for k in ('branch_sha256', 'parent_state_sha256')})
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT id,binding FROM applications WHERE parent_slot=?', (parent_slot,)).fetchone()
            if old and old != (aid, raw(binding)):
                raise ValueError('parent already reserved for different application')
            alias = db.execute('SELECT application_id,evidence FROM originals WHERE original_sha=?', (original_sha256,)).fetchone()
            if alias and alias != (aid, encoded):
                raise ValueError('original attempt binding/evidence changed')
            if not old:
                db.execute('INSERT INTO applications VALUES(?,?,?,\'reserved\',NULL)', (aid, parent_slot, raw(binding)))
            if not alias:
                db.execute('INSERT INTO originals VALUES(?,?,?)', (original_sha256, aid, encoded))
            db.commit()
        return aid

    def inspect(self, application_id):
        _sha(application_id)
        with closing(self._connect()) as db:
            row = db.execute('SELECT binding,state,receipt FROM applications WHERE id=?', (application_id,)).fetchone()
        if row is None:
            raise ValueError('unknown application')
        return dict(binding=json.loads(row[0]), state=row[1], receipt=json.loads(row[2]) if row[2] else None)

    def execute_once(self, application_id, execute, authenticate_publication):
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state,receipt FROM applications WHERE id=?', (application_id,)).fetchone()
            if row is None:
                raise ValueError('unknown application')
            if row[0] == 'complete':
                db.commit(); return json.loads(row[1])
            if row[0] != 'reserved':
                raise RecoveryRequired('original application unresolved; no automatic reapplication')
            db.execute("UPDATE applications SET state='executing' WHERE id=?", (application_id,)); db.commit()
        # Exceptions and process death both leave an irreversible unresolved
        # claim. This includes death BEFORE the first optimizer update.
        result = execute(copy.deepcopy(self.inspect(application_id)['binding']))
        return self.record_publication(application_id, result, authenticate_publication)

    def record_publication(self, application_id, evidence, authenticate_publication):
        job = self.inspect(application_id); binding = job['binding']
        if not callable(authenticate_publication):
            raise ValueError('independent actual publication authentication required')
        receipt = authenticate_publication(copy.deepcopy(evidence), copy.deepcopy(binding), application_id)
        expected = dict(version='consumed-application-publication-research-v1',
            application_id=application_id, parent_state_sha256=binding['parent_state_sha256'],
            step_before=binding['step_before'], step_after=binding['step_before']+len(binding['groups']))
        if (not isinstance(receipt, dict) or set(receipt) != set(expected) | {'output_state_sha256', 'output_checkpoint_sha256'} or
                any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items())):
            raise ValueError('authenticated publication exact application/lineage')
        _sha(receipt['output_state_sha256']); _sha(receipt['output_checkpoint_sha256'])
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state,receipt FROM applications WHERE id=?', (application_id,)).fetchone()
            if row[0] == 'complete':
                if row[1] != raw(receipt):
                    raise ValueError('original durable publication changed')
            elif row[0] == 'executing':
                db.execute("UPDATE applications SET state='complete',receipt=? WHERE id=?", (raw(receipt), application_id))
            else:
                raise ValueError('application not started')
            db.commit()
        return copy.deepcopy(receipt)
