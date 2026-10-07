"""Default-off durable captured-revision/native-outcome research join.

Only supplied trusted authenticators verify original capture and native outcome
signatures/provenance. Cheap eligibility remains unaudited; this module does not
run grading, inference audits, training or optimizer publication. No live caller.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing
from ops.paired_quota_revision_journal import ResearchRevisionJournal, scope, boundary_receipt
from ops.paired_quota_batch_adapter import CumulativeTaskSlot
from ops.paired_quota_nested_selector import selection_scope, select_nested
from ops.paired_quota_qualification import _sha, digest
from subnet.storage import canonical

APPLICATION_ID = 1095782995
VERSION = 'captured-revision-native-nested-quota-bridge-research-v1'


class ResearchQuotaBridge:
    def __init__(self, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research quota bridge requires explicit opt-in')
        p=Path(path)
        if p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o077:
            raise ValueError('existing private owned bridge required')
        self.path=p.resolve()
        with self._connect() as db:
            if db.execute('PRAGMA application_id').fetchone()[0]!=APPLICATION_ID or db.execute('PRAGMA user_version').fetchone()[0]!=1:
                raise ValueError('owned research bridge schema')

    def _connect(self):
        db=sqlite3.connect(self.path,timeout=10,isolation_level=None)
        db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA synchronous=FULL')
        return closing(db)

    @classmethod
    def create(cls,path,*,enabled=False):
        if enabled is not True:raise ValueError('research quota bridge requires explicit opt-in')
        p=Path(path);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
        with closing(sqlite3.connect(p))as db:
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
            db.executescript('''
                CREATE TABLE contexts(slot TEXT PRIMARY KEY,binding TEXT NOT NULL);
                CREATE TABLE outcomes(slot TEXT NOT NULL REFERENCES contexts(slot),attempt INTEGER NOT NULL,
                    grade TEXT NOT NULL,evidence TEXT NOT NULL,PRIMARY KEY(slot,attempt));
                CREATE TABLE packets(id TEXT PRIMARY KEY,slot TEXT NOT NULL REFERENCES contexts(slot),result TEXT NOT NULL);
                CREATE TRIGGER contexts_no_update BEFORE UPDATE ON contexts BEGIN SELECT RAISE(ABORT,'immutable context');END;
                CREATE TRIGGER contexts_no_delete BEFORE DELETE ON contexts BEGIN SELECT RAISE(ABORT,'immutable context');END;
                CREATE TRIGGER outcomes_no_update BEFORE UPDATE ON outcomes BEGIN SELECT RAISE(ABORT,'immutable outcome');END;
                CREATE TRIGGER outcomes_no_delete BEFORE DELETE ON outcomes BEGIN SELECT RAISE(ABORT,'immutable outcome');END;
                CREATE TRIGGER packets_no_update BEFORE UPDATE ON packets BEGIN SELECT RAISE(ABORT,'immutable packet');END;
                CREATE TRIGGER packets_no_delete BEFORE DELETE ON packets BEGIN SELECT RAISE(ABORT,'immutable packet');END;
            ''')
            db.execute(f'PRAGMA application_id={APPLICATION_ID}');db.execute('PRAGMA user_version=1')
        return cls(p,enabled=True)

    def select_revision(self,journal,adapter,miner_public_key,manifest_sha256,revision_id,
                        native_outcomes,*,eos_token_ids,authenticate_original,
                        authenticate_admitted_native):
        if type(journal)is not ResearchRevisionJournal:raise ValueError('original revision journal required')
        _sha(revision_id)
        binding=scope(adapter,miner_public_key,manifest_sha256);slot=adapter.task.slot_id(miner_public_key)
        captured=journal.frozen_original(slot,revision_id)
        if captured['binding']!=binding:raise ValueError('same authenticated checkpoint/task/epoch/public-key revision required')
        expected=boundary_receipt(captured['original_sha256'],digest(binding))
        if captured['boundary']!=expected or not callable(authenticate_original) or authenticate_original(captured['original_bytes'],copy.deepcopy(captured['evidence']),copy.deepcopy(binding))!=expected:
            raise ValueError('reauthenticated immutable original capture required')
        document=json.loads(captured['original_bytes'])
        if document['epoch']!=adapter.task.epoch or document['checkpoint']!=adapter.task.checkpoint or document['miner']!=miner_public_key:
            raise ValueError('original captured owner/epoch/checkpoint mismatch')
        state=CumulativeTaskSlot(adapter,miner_public_key)
        if state.add_revision(document['batch'])['revision_id']!=revision_id:
            raise ValueError('recomputed original cumulative identity mismatch')
        rows={}
        for row in adapter.normalize(document['batch']):
            attempt=row['attempt']
            if attempt in rows and rows[attempt]!=row:raise ValueError('conflicting captured normalized attempt')
            rows[attempt]=row
        if not callable(authenticate_admitted_native):raise ValueError('native authenticator required')
        if type(native_outcomes)is not list or len(native_outcomes)>len(adapter.task.approved_attempts)*4 or len(canonical(native_outcomes))>2_000_000:
            raise ValueError('bounded original native outcome packet')
        incoming={}
        for observation in native_outcomes:
            if type(observation)is not dict or set(observation)!={'attempt','evidence'}:
                raise ValueError('native packet cannot supply replacement rows/member IDs')
            attempt=observation['attempt']
            if type(attempt)is not int or attempt not in adapter.task.approved_attempts:
                raise ValueError('approved native attempt required')
            if attempt in incoming and incoming[attempt]!=observation['evidence']:
                raise ValueError('conflicting duplicate native original evidence')
            incoming[attempt]=copy.deepcopy(observation['evidence'])
        records=[dict(attempt=a,row=rows.get(a),evidence=ev)for a,ev in sorted(incoming.items())]
        grades={}
        def authenticate(record,expected_binding):
            if not callable(authenticate_admitted_native):raise ValueError('native authenticator required')
            grade=authenticate_admitted_native(record,expected_binding)
            grades[record['attempt']]=copy.deepcopy(grade)
            return grade
        selected=select_nested(adapter.task,miner_public_key,records,eos_token_ids=eos_token_ids,
                               authenticate_admitted_native=authenticate,enabled=True)
        bridge_binding=dict(version=VERSION,original_scope=binding,
                            selection_scope_sha256=digest(selection_scope(adapter.task,miner_public_key,eos_token_ids)))
        inventory=[dict(attempt=a,grade_sha256=digest(grades[a]),evidence_sha256=digest(incoming[a]))for a in sorted(incoming)]
        packet_id=digest(dict(binding=bridge_binding,revision_id=revision_id,native_inventory=inventory))
        result=dict(version=VERSION,slot_id=slot,captured_revision_id=revision_id,
                    captured_original_sha256=captured['original_sha256'],native_inventory=inventory,
                    selection=selected,input_assurance='cheap-eligible-unaudited',
                    native_grade_assurance='caller-authenticated-native-not-inference-proof',
                    inference_audit_required_for_training=False,inference_verified=False,
                    optimizer_application_performed=False)
        with self._connect()as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                old_context=db.execute('SELECT binding FROM contexts WHERE slot=?',(slot,)).fetchone()
                if old_context is not None and old_context[0]!=canonical(bridge_binding).decode():
                    raise ValueError('frozen original/tokenizer selection scope changed')
                cached=db.execute('SELECT result FROM packets WHERE id=?',(packet_id,)).fetchone()
                if cached is not None:
                    if json.loads(cached[0])!=result:raise ValueError('immutable cached selection changed')
                    db.commit();return dict(packet_id=packet_id,redelivery=True,result=json.loads(cached[0]))
                known={a:(g,e)for a,g,e in db.execute('SELECT attempt,grade,evidence FROM outcomes WHERE slot=?',(slot,))}
                if not known.keys()<=incoming.keys():raise ValueError('cumulative native packet removed original outcome')
                for attempt,(grade,evidence)in known.items():
                    if grade!=canonical(grades[attempt]).decode()or evidence!=canonical(incoming[attempt]).decode():
                        raise ValueError('original native outcome/evidence changed')
                if old_context is None:db.execute('INSERT INTO contexts VALUES(?,?)',(slot,canonical(bridge_binding).decode()))
                for attempt in incoming.keys()-known.keys():
                    db.execute('INSERT INTO outcomes VALUES(?,?,?,?)',(slot,attempt,canonical(grades[attempt]).decode(),canonical(incoming[attempt]).decode()))
                db.execute('INSERT INTO packets VALUES(?,?,?)',(packet_id,slot,canonical(result).decode()))
                db.commit();return dict(packet_id=packet_id,redelivery=False,result=result)
            except BaseException:
                db.rollback();raise
