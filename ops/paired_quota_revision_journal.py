"""Default-off durable research revisions; no live caller or optimizer claim.

Caller authenticates captured commitment/admission, original manifest and native
reset/tokenizer before supplying the CurrentBatchAdapter. authenticate_original
must independently authenticate the supplied original token-document bytes and
return the exact scope-bound receipt described by boundary_receipt(). It is a
trusted integration callback, NOT an unsigned assurance flag or a grade proof.
"""
import copy
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing

from ops.paired_quota_batch_adapter import CurrentBatchAdapter, CumulativeTaskSlot
from ops.paired_quota_qualification import _sha, digest
from subnet.storage import canonical
from subnet import protocol
from subnet.training_documents import VERSION, TOKEN_VERSION

APPLICATION_ID = 1095782994
MAX_ORIGINAL_BYTES = 2_000_000


def scope(adapter, miner_public_key, manifest_sha256):
    if type(adapter) is not CurrentBatchAdapter:
        raise ValueError('original current batch adapter required')
    _sha(miner_public_key); _sha(manifest_sha256)
    task = adapter.task
    if (adapter.definition.get('env_id') != task.env_id or task.index not in adapter.definition.get('indices', []) or
            digest(adapter.definition['spec']) != task.taskset_sha256 or
            digest(protocol.harness_for(adapter.definition, task.index)) != task.harness_sha256 or
            digest(adapter.sampling_context) != task.sampling_context_sha256):
        raise ValueError('adapter approved task/harness/draw binding changed')
    return dict(version='cumulative-research-slot-v1', miner_public_key=miner_public_key,
                manifest_sha256=manifest_sha256, epoch=task.epoch,
                task=task.task_binding(), harness_sha256=task.harness_sha256,
                sampling_context_sha256=task.sampling_context_sha256,
                approved_attempts=list(task.approved_attempts))


def boundary_receipt(original_sha256, scope_sha256):
    """Expected output of the caller's trusted document authenticator.

    This constructor does not authenticate anything. Callers MUST verify original
    signature/admission/model/source/task and exact captured SHA/size first.
    """
    _sha(original_sha256); _sha(scope_sha256)
    return dict(version='authenticated-current-batch-research-boundary-v1',
                original_sha256=original_sha256, scope_sha256=scope_sha256)


class ResearchRevisionJournal:
    def __init__(self, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research journal requires explicit opt-in')
        p = Path(path)
        if p.is_symlink() or not p.is_file() or p.stat().st_uid != os.getuid() or p.stat().st_mode & 0o077:
            raise ValueError('existing regular owned research journal required')
        self.path = p.resolve()
        with self._connect() as db:
            if db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or db.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise ValueError('owned research revision journal schema')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.execute('PRAGMA foreign_keys=ON'); db.execute('PRAGMA synchronous=FULL')
        return closing(db)

    @classmethod
    def create(cls, path, *, enabled=False):
        if enabled is not True:
            raise ValueError('research journal requires explicit opt-in')
        p = Path(path)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        with closing(sqlite3.connect(p)) as db:
            db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
            db.executescript('''
                CREATE TABLE slots (id TEXT PRIMARY KEY, binding TEXT NOT NULL, head TEXT NOT NULL);
                CREATE TRIGGER slots_no_rebind BEFORE UPDATE OF id,binding ON slots BEGIN SELECT RAISE(ABORT,'immutable slot binding'); END;
                CREATE TRIGGER slots_no_delete BEFORE DELETE ON slots BEGIN SELECT RAISE(ABORT,'immutable slot'); END;
                CREATE TABLE originals (sha256 TEXT PRIMARY KEY, bytes BLOB NOT NULL);
                CREATE TABLE revisions (slot TEXT NOT NULL REFERENCES slots(id), id TEXT NOT NULL,
                    batch TEXT NOT NULL, PRIMARY KEY(slot,id));
                CREATE TABLE observations (slot TEXT NOT NULL REFERENCES slots(id), original TEXT NOT NULL REFERENCES originals(sha256),
                    revision TEXT NOT NULL, boundary TEXT NOT NULL, evidence TEXT NOT NULL, PRIMARY KEY(slot,original),
                    FOREIGN KEY(slot,revision) REFERENCES revisions(slot,id));
                CREATE TRIGGER originals_no_update BEFORE UPDATE ON originals BEGIN SELECT RAISE(ABORT,'immutable original'); END;
                CREATE TRIGGER originals_no_delete BEFORE DELETE ON originals BEGIN SELECT RAISE(ABORT,'immutable original'); END;
                CREATE TRIGGER revisions_no_update BEFORE UPDATE ON revisions BEGIN SELECT RAISE(ABORT,'immutable revision'); END;
                CREATE TRIGGER revisions_no_delete BEFORE DELETE ON revisions BEGIN SELECT RAISE(ABORT,'immutable revision'); END;
                CREATE TRIGGER observations_no_update BEFORE UPDATE ON observations BEGIN SELECT RAISE(ABORT,'immutable observation'); END;
                CREATE TRIGGER observations_no_delete BEFORE DELETE ON observations BEGIN SELECT RAISE(ABORT,'immutable observation'); END;
            ''')
            db.execute(f'PRAGMA application_id={APPLICATION_ID}'); db.execute('PRAGMA user_version=1')
        return cls(p, enabled=True)

    def append_authenticated(self, adapter, miner_public_key, manifest_sha256,
                             original_bytes, evidence, authenticate_original):
        binding = scope(adapter, miner_public_key, manifest_sha256)
        if type(original_bytes) is not bytes or not 0 < len(original_bytes) <= MAX_ORIGINAL_BYTES:
            raise ValueError('bounded exact captured original bytes')
        import hashlib
        original_sha = hashlib.sha256(original_bytes).hexdigest()
        raw_evidence = canonical(evidence).decode()
        if len(raw_evidence.encode()) > MAX_ORIGINAL_BYTES:
            raise ValueError('bounded canonical original authentication evidence')
        expected = boundary_receipt(original_sha, digest(binding))
        if not callable(authenticate_original) or authenticate_original(original_bytes, copy.deepcopy(evidence), copy.deepcopy(binding)) != expected:
            raise ValueError('authenticated original boundary required, not self-asserted boolean')
        # Parse only the ORIGINAL bytes; never accept caller-provided member IDs.
        document = json.loads(original_bytes)
        if canonical(document) != original_bytes or type(document) is not dict or set(document) != {'version','epoch','checkpoint','miner','slot','batch'}:
            raise ValueError('canonical original token document')
        if document['version'] not in (VERSION, TOKEN_VERSION) or document['epoch'] != adapter.task.epoch or document['checkpoint'] != adapter.task.checkpoint or document['miner'] != miner_public_key or type(document['slot']) is not int or document['slot'] < 0:
            raise ValueError('original public-key/task document scope')
        batch = document['batch']
        # Validate adapter even on old redelivery: new caller configuration cannot
        # evade the original task, harness, sampling or token constraints.
        adapter.normalize(batch)
        slot_id = adapter.task.slot_id(miner_public_key)
        raw_binding = canonical(binding).decode()
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                old = db.execute('SELECT binding,head FROM slots WHERE id=?', (slot_id,)).fetchone()
                if old is not None and old[0] != raw_binding:
                    raise ValueError('original authenticated slot binding changed')
                replay = db.execute('SELECT revision,boundary,evidence FROM observations WHERE slot=? AND original=?', (slot_id, original_sha)).fetchone()
                if replay is not None:
                    if replay[1] != canonical(expected).decode() or replay[2] != raw_evidence: raise ValueError('original boundary/evidence changed')
                    db.commit()
                    return dict(slot_id=slot_id, revision_id=replay[0], head_revision_id=old[1], redelivery=True, original_recorded=False)
                state = CumulativeTaskSlot(adapter, miner_public_key)
                if old is not None:
                    previous = db.execute('SELECT batch FROM revisions WHERE slot=? AND id=?', (slot_id, old[1])).fetchone()
                    state.add_revision(json.loads(previous[0]))
                computed = state.add_revision(batch)
                revision = computed['revision_id']
                if old is None:
                    db.execute('INSERT INTO slots VALUES(?,?,?)', (slot_id, raw_binding, revision))
                already = db.execute('SELECT 1 FROM revisions WHERE slot=? AND id=?', (slot_id, revision)).fetchone() is not None
                db.execute('INSERT OR IGNORE INTO originals VALUES(?,?)', (original_sha, original_bytes))
                if not already:
                    db.execute('INSERT INTO revisions VALUES(?,?,?)', (slot_id, revision, canonical(batch).decode()))
                db.execute('INSERT INTO observations VALUES(?,?,?,?,?)', (slot_id, original_sha, revision, canonical(expected).decode(), raw_evidence))
                db.execute('UPDATE slots SET head=? WHERE id=?', (revision, slot_id))
                db.commit()
                return dict(slot_id=slot_id, revision_id=revision, head_revision_id=revision,
                            redelivery=already, original_recorded=True)
            except BaseException:
                db.rollback(); raise

    def inspect(self, slot_id):
        _sha(slot_id)
        with self._connect() as db:
            row = db.execute('SELECT binding,head FROM slots WHERE id=?', (slot_id,)).fetchone()
            if row is None: raise ValueError('unknown research task slot')
            return dict(binding=json.loads(row[0]), head_revision_id=row[1],
                        revision_count=db.execute('SELECT COUNT(*) FROM revisions WHERE slot=?', (slot_id,)).fetchone()[0],
                        original_count=db.execute('SELECT COUNT(*) FROM observations WHERE slot=?', (slot_id,)).fetchone()[0])
