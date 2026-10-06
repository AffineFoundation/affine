"""Atomic operator-side role leases. GPU identity authentication is not sampling proof."""
import base64
import hashlib
import json
import math
import secrets
import sqlite3
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from nacl.signing import VerifyKey
from .storage import canonical


# Metadata-only indexes keep idle claims out of stored signed envelopes/reports.
QUEUE_METADATA_INDEXES = {
    'affine_jobs_status_expires_v1': 'CREATE INDEX affine_jobs_status_expires_v1 ON jobs(status,expires)',
    'affine_jobs_status_lease_attempt_v1': 'CREATE INDEX affine_jobs_status_lease_attempt_v1 ON jobs(status,lease,attempt)',
    'affine_jobs_role_status_expires_lease_attempt_v1': 'CREATE INDEX affine_jobs_role_status_expires_lease_attempt_v1 ON jobs(role,status,expires,lease,attempt)',
    'affine_requests_expires_v1': 'CREATE INDEX affine_requests_expires_v1 ON requests(expires)',
}


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def authenticate(envelope, identity):
    if envelope.get('signer') != identity:
        raise ValueError('signer')
    VerifyKey(bytes.fromhex(identity)).verify(canonical(envelope['payload']), base64.b64decode(envelope['signature'], validate=True))
    return envelope['payload']


def validate_frozen_submissions(manifest, submissions):
    """Bind selected child capabilities to authenticated miner commitment slots."""
    if 'probability_artifact_policy' in manifest:
        from .probability_artifacts import validate_policy
        validate_policy(manifest['probability_artifact_policy'])
    frozen = manifest.get('audit_frozen_receipts', {})
    if type(submissions)is not list or not submissions or len(submissions) > 256:
        raise ValueError('frozen submission binding')
    if manifest.get('submission_transport_policy') is None:
        allowed = {r['sha256'] for r in frozen.values()}
        if any(s.get('sha256') not in allowed for s in submissions):
            raise ValueError('frozen submission binding')
        return
    from .commitment_transport import VERSIONS, validate, is_digest
    from urllib.parse import urlsplit, unquote
    if manifest['submission_transport_policy'] not in VERSIONS:
        raise ValueError('explicit child commitment transport')
    fields = {'miner','commitment_sha256','slot','env_id','index','batch_sha256','size','frozen_key'}
    seen = set()
    for obj in submissions:
        if type(obj)is not dict or set(obj)!={'url','sha256','commitment_miner','commitment_ref'} or type(obj['url'])is not str:
            raise ValueError('exact selected child object fields')
        ref = obj.get('commitment_ref')
        if (not isinstance(ref,dict) or set(ref)!=fields or
                obj.get('commitment_miner')!=ref['miner'] or
                type(ref['slot'])is not int or type(ref['index'])is not int or
                type(ref['size'])is not int or type(ref['frozen_key'])is not str or
                not all(is_digest(ref[k])for k in ('miner','commitment_sha256','batch_sha256'))):
            raise ValueError('exact selected child metadata')
        environments=[e for e in manifest.get('environments',[])if e.get('env_id')==ref['env_id']]
        if len(environments)!=1 or ref['index']not in environments[0].get('indices',[]):
            raise ValueError('selected child approved environment/index')
        receipt=frozen.get(ref['miner'])
        if not isinstance(receipt,dict) or receipt.get('sha256')!=ref['commitment_sha256']:
            raise ValueError('selected child parent/miner commitment')
        from nacl.exceptions import BadSignatureError
        try:
            envelope=validate(canonical(receipt['commitment_document']),manifest['epoch'],ref['miner'],manifest['max_batches'])
        except (KeyError,TypeError,ValueError,BadSignatureError)as error:
            raise ValueError('authenticated miner child commitment required')from error
        if digest(envelope)!=receipt['sha256']:
            raise ValueError('exact canonical parent commitment digest')
        payload=envelope['payload']
        if payload['version']!=manifest['submission_transport_policy']:
            raise ValueError('exact original miner commitment transport version')
        if (payload['source']!=manifest['source_bundle']['sha256'] or
                payload['checkpoint']!=manifest['checkpoint']['id']):
            raise ValueError('selected child source/checkpoint commitment')
        rows=[r for r in payload['batches']if r['slot']==ref['slot']]
        artifacts=[r for r in receipt['artifacts']if r['slot']==ref['slot']]
        if len(rows)!=1 or len(artifacts)!=1:
            raise ValueError('selected child exact slot')
        row=rows[0];artifact=artifacts[0]
        if (any(ref[k]!=row[k]for k in ('slot','env_id','index','batch_sha256','size')) or
                obj.get('sha256')!=row['sha256'] or
                any(artifact.get(k)!=row[k]for k in row) or
                ref['frozen_key']!=artifact.get('frozen_key')):
            raise ValueError('selected child inventory binding')
        expected_key='public/'+manifest['epoch']+'/submissions/'+ref['miner']+'/'+receipt['sha256']+'/'+str(ref['slot'])+'.zip'
        if ref['frozen_key']!=expected_key:
            raise ValueError('selected child exact epoch/miner/parent storage key')
        original_artifact=artifact
        if manifest.get('proof_copy_policy') is not None:
            from .selected_proof_copy import validate_policy
            validate_policy(manifest['proof_copy_policy'])
            copied=manifest.get('proof_copy_receipts',{}).get(ref['miner'],{}).get(str(ref['slot']))
            expected={k:artifact[k]for k in ('sha256','size','etag','key','frozen_key')}
            if type(copied)is not dict or set(copied)!=set(expected)|{'read_url'} or any(copied[k]!=v for k,v in expected.items()):raise ValueError('signed original selected proof copy receipt')
            original_artifact=copied
        url=urlsplit(obj['url']);original=urlsplit(original_artifact['read_url'])
        if (url.scheme!='https' or url.netloc!=original.netloc or
                unquote(url.path)!=unquote(original.path) or
                not unquote(url.path).endswith('/'+ref['frozen_key']) or url.fragment):
            raise ValueError('selected child frozen URL/key binding')
        identity=(ref['miner'],ref['slot'])
        if identity in seen:raise ValueError('duplicate selected child slot')
        seen.add(identity)


class Coordinator:
    """SQLite is the claim authority; R2 holds immutable history, never lock files.

    One SQLite database on the validator host is shared by HTTP handler threads.
    BEGIN IMMEDIATE serializes selection and lease replacement across processes.
    Workers possess an individual signing seed, never an operator or bucket key.
    """
    def __init__(self, path, authority, workers, lease_seconds=300, max_attempts=3, clock=time.time):
        if not 10 <= lease_seconds <= 86400 or not 1 <= max_attempts <= 10:
            raise ValueError('lease bounds')
        self.path = str(path); self.authority = authority; self.workers = workers
        self.lease_seconds = lease_seconds; self.max_attempts = max_attempts; self.clock = clock
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._queue_inode = None
        if Path(path).exists():
            identity = Path(path).stat()
            self._queue_inode = (identity.st_dev, identity.st_ino)
        with self.transaction() as db:
            schema = '''
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL, envelope TEXT NOT NULL,
              role TEXT NOT NULL, expires REAL NOT NULL, status TEXT NOT NULL,
              attempt INTEGER NOT NULL DEFAULT 0, worker TEXT, token TEXT, lease REAL,
              report TEXT, report_digest TEXT, report_request TEXT);
            CREATE TABLE IF NOT EXISTS requests (worker TEXT, nonce TEXT, expires REAL,
              PRIMARY KEY(worker,nonce));
            CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              job TEXT, at REAL, kind TEXT, detail TEXT);
            '''
            # executescript implicitly commits BEGIN; individual DDL preserves
            # atomic initialization and index admission under the same lease lock.
            for statement in schema.split(';'):
                if statement.strip(): db.execute(statement)
            for name, sql in QUEUE_METADATA_INDEXES.items():
                existing = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
                if existing is None: db.execute(sql)
                elif existing[0] != sql: raise ValueError('conflicting queue metadata index')
        Path(path).chmod(0o600)

    def _check_queue_identity(self):
        identity = Path(self.path).stat()
        observed = (identity.st_dev, identity.st_ino)
        if self._queue_inode is None: self._queue_inode = observed
        elif observed != self._queue_inode: raise ValueError('authoritative queue inode changed')

    def _begin_transaction(self, *, budget=120, monotonic=time.monotonic, sleep=time.sleep):
        """Retry only BEGIN acquisition; never repeat a body or failed commit."""
        if not 0 < budget <= 120: raise ValueError('queue acquisition budget')
        deadline = monotonic() + budget
        while True:
            if self._queue_inode is not None: self._check_queue_identity()
            remaining = deadline - monotonic()
            if remaining <= 0: raise TimeoutError('authoritative queue write lock unavailable')
            db = sqlite3.connect(self.path, timeout=min(.25, remaining), isolation_level=None)
            db.row_factory = sqlite3.Row
            try:
                self._check_queue_identity()
                db.execute('BEGIN IMMEDIATE')
                self._check_queue_identity()
                # Short BEGIN polls must not shorten the historical body/commit wait.
                db.execute('PRAGMA busy_timeout=30000')
                return db
            except sqlite3.OperationalError as error:
                db.close()
                if not any(word in str(error).lower() for word in ('locked', 'busy')): raise
                remaining = deadline - monotonic()
                if remaining <= 0: raise TimeoutError('authoritative queue write lock unavailable') from error
                sleep(min(.05, remaining))
            except BaseException:
                db.close(); raise

    @contextmanager
    def transaction(self):
        db = self._begin_transaction()
        try:
            yield db
            self._check_queue_identity()
            db.commit()
        except BaseException:
            db.rollback(); raise
        finally:
            db.close()

    def event(self, db, identifier, kind, **detail):
        db.execute('INSERT INTO events(job,at,kind,detail) VALUES(?,?,?,?)',
                   (identifier, self.clock(), kind, canonical(detail).decode()))

    def enqueue(self, envelope):
        job = authenticate(envelope, self.authority)
        manifest = authenticate(job['manifest'], self.authority)
        from .backend_profiles import resolve
        resolve(manifest)
        now = self.clock()
        if manifest.get('payable') is not False or not str(manifest['epoch']).startswith('nonpayable-'):
            raise ValueError('nonpayable queue only')
        if job.get('schema') != 1 or job.get('role') != 'verify' or not str(job['job_id']).replace('-','').replace('_','').isalnum():
            raise ValueError('queue role/job')
        if any(type(job.get(k)) not in (int,float) or not math.isfinite(job[k]) for k in ('created_at','expires_at')) or not 0 < job['expires_at']-job['created_at'] <= 86400:
            raise ValueError('signed job lifetime')
        submissions = job.get('submissions', [])
        frozen = manifest.get('audit_frozen_receipts', {})
        validate_frozen_submissions(manifest,submissions)
        with self.transaction() as db:
            old = db.execute('SELECT digest FROM jobs WHERE id=?', (job['job_id'],)).fetchone()
            if old:
                if old['digest'] != digest(job): raise ValueError('immutable job collision')
                return job['job_id']
            if not job['created_at'] <= now < job['expires_at']: raise ValueError('signed job lifetime')
            db.execute('INSERT INTO jobs(id,digest,envelope,role,expires,status) VALUES(?,?,?,?,?,?)',
                (job['job_id'],digest(job),canonical(envelope).decode(),job['role'],job['expires_at'],'queued'))
            self.event(db, job['job_id'], 'queued')
        return job['job_id']

    def request(self, envelope):
        worker = envelope.get('signer')
        if worker not in self.workers: raise ValueError('unknown worker')
        request = authenticate(envelope, worker); now = self.clock()
        timestamp = request.get('at'); nonce = request.get('nonce')
        if type(timestamp) not in (int,float) or not math.isfinite(timestamp) or abs(timestamp-now) > 60 or not isinstance(nonce,str) or not 16 <= len(nonce) <= 128:
            raise ValueError('request freshness')
        with self.transaction() as db:
            db.execute('DELETE FROM requests WHERE expires<?', (now,))
            try: db.execute('INSERT INTO requests VALUES(?,?,?)', (worker,nonce,now+120))
            except sqlite3.IntegrityError: raise ValueError('request replay') from None
            action = request.get('action')
            if action == 'claim':
                role = request.get('role')
                if role not in self.workers[worker] or role != 'verify': raise ValueError('worker role')
                db.execute("UPDATE jobs SET status='expired' WHERE status IN ('queued','leased') AND expires<=?", (now,))
                db.execute("UPDATE jobs SET status='failed' WHERE status='leased' AND lease<=? AND attempt>=?", (now,self.max_attempts))
                row = db.execute("SELECT * FROM jobs WHERE role=? AND expires>? AND attempt<? AND (status='queued' OR (status='leased' AND lease<=?)) ORDER BY rowid LIMIT 1", (role,now,self.max_attempts,now)).fetchone()
                if not row: return {'claim':None}
                token = secrets.token_hex(32); lease = min(now+self.lease_seconds,row['expires'])
                db.execute("UPDATE jobs SET status='leased',worker=?,token=?,lease=?,attempt=attempt+1 WHERE id=?", (worker,token,lease,row['id']))
                self.event(db,row['id'],'claimed',worker=worker,attempt=row['attempt']+1,lease_until=lease)
                return {'claim':dict(job=json.loads(row['envelope']),job_sha256=row['digest'],token=token,lease_until=lease,attempt=row['attempt']+1)}
            row = db.execute('SELECT * FROM jobs WHERE id=?', (request.get('job_id'),)).fetchone()
            if not row: raise ValueError('unknown job')
            if row['worker'] != worker or row['token'] != request.get('token'): raise ValueError('stale lease identity')
            if action == 'report' and row['status'] == 'complete':
                if digest(request['report']) != row['report_digest']: raise ValueError('conflicting duplicate report')
                return {'accepted':True,'duplicate':True}
            if row['status'] != 'leased' or now >= row['lease'] or now >= row['expires']:
                raise ValueError('expired lease')
            if action == 'renew':
                lease = min(now+self.lease_seconds,row['expires'])
                db.execute('UPDATE jobs SET lease=? WHERE id=?', (lease,row['id']))
                self.event(db,row['id'],'renewed',worker=worker,attempt=row['attempt'],lease_until=lease)
                return {'lease_until':lease}
            if action == 'fail':
                status = 'queued' if row['attempt'] < self.max_attempts else 'failed'
                db.execute('UPDATE jobs SET status=? WHERE id=?', (status,row['id']))
                self.event(db,row['id'],'worker_failed',worker=worker,attempt=row['attempt'])
                return {'status':status}
            if action != 'report': raise ValueError('request action')
            report = request['report']; job = json.loads(row['envelope'])['payload']; manifest = job['manifest']['payload']
            self.validate_report(report,job,manifest,row['digest'])
            db.execute("UPDATE jobs SET status='complete',report=?,report_digest=?,report_request=? WHERE id=?", (canonical(report).decode(),digest(report),canonical(envelope).decode(),row['id']))
            self.event(db,row['id'],'completed',worker=worker,report_sha256=digest(report))
            return {'accepted':True,'duplicate':False}

    def validate_report(self, report, job, manifest, job_digest):
        expected = dict(job_id=job['job_id'],job_sha256=job_digest,operator=self.authority,
                        role=job['role'],epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                        source_files=job['source_files'],runtime_versions=job['runtime_versions'],
                        backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],
                        chain_transactions=False,success=True)
        if any(canonical(report.get(k)) != canonical(v) for k,v in expected.items()):
            raise ValueError('report job/epoch/checkpoint/source/runtime binding')
        completed = report.get('completed_at')
        if type(completed) not in (int,float) or not math.isfinite(completed) or not job['created_at'] <= completed < job['expires_at'] or completed > self.clock()+5:
            raise ValueError('report signed deadline')
        if [r.get('submission_sha256') for r in report.get('audits',[])] != [r['sha256'] for r in job['submissions']]:
            raise ValueError('report frozen artifact binding')
        if manifest.get('submission_transport_policy') is not None:
            validate_frozen_submissions(manifest,job['submissions'])
            for audit,obj in zip(report['audits'],job['submissions'],strict=True):
                ref=obj['commitment_ref'];accepted=audit.get('accepted',[])
                if (len(accepted)>1 or any(type(b.get('index'))is not int or b.get('env_id')!=ref['env_id']or b.get('index')!=ref['index']for b in accepted)):
                    raise ValueError('report selected child environment/index binding')
                if any(digest(b)!=ref['batch_sha256']for b in accepted):
                    raise ValueError('report exact committed accepted batch digest')
        if any(r.get('epoch') != manifest['epoch'] or any(b.get('epoch') != manifest['epoch'] or b.get('checkpoint') != manifest['checkpoint']['id'] for b in r.get('accepted', [])) for r in report['audits']):
            raise ValueError('audit epoch/checkpoint binding')

    def status(self, identifier, *, timeout_seconds=30.0):
        """Read the committed job snapshot without acquiring a writer lease."""
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
            raise ValueError('status timeout')
        deadline = time.monotonic() + timeout_seconds
        while True:
            db = None
            try:
                db = sqlite3.connect(Path(self.path).resolve().as_uri()+'?mode=ro', uri=True,
                                     timeout=min(.25, max(0, deadline-time.monotonic())))
                db.row_factory = sqlite3.Row
                row = db.execute('SELECT status,report,report_digest,worker,attempt FROM jobs WHERE id=?',
                                 (identifier,)).fetchone()
                break
            except sqlite3.OperationalError as error:
                if db is not None:
                    db.close()
                    db = None
                if not any(word in str(error).lower() for word in ('locked', 'busy')):
                    raise
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(remaining, .025+secrets.randbelow(26)/1000))
            finally:
                if db is not None:
                    db.close()
        if not row: raise ValueError('unknown job')
        value = dict(row); value['report'] = json.loads(value['report']) if value['report'] else None
        return value

    def archive(self, identifier, bucket, prefix):
        """Operator-only conditional creation. No permanent R2 key leaves this host."""
        with self.transaction() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            events = [dict(r) for r in db.execute('SELECT * FROM events WHERE job=? ORDER BY sequence',(identifier,))]
        objects = {'job.json': json.loads(row['envelope'])}
        if row['status'] == 'complete':
            objects.update({'report.json':json.loads(row['report']), 'worker-report.json':json.loads(row['report_request']), 'history.json':dict(worker=row['worker'],attempt=row['attempt'],events=events,chain_transactions=False)})
        for name,value in objects.items():
            key = prefix.rstrip('/')+'/'+identifier+'/'+name; body = canonical(value)
            try:
                bucket.client.put_object(Bucket=bucket.name,Key=key,Body=body,ContentType='application/json',IfNoneMatch='*')
            except Exception as error:
                code = getattr(error,'response',{}).get('Error',{}).get('Code')
                if code not in ('PreconditionFailed','412'): raise
                if bucket.get(key) != body: raise ValueError('immutable role history collision') from None


class CoordinatorServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, coordinator, sign):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                try:
                    self.connection.settimeout(30)
                    size = int(self.headers.get('Content-Length','0'))
                    if self.path != '/request' or not 0 < size <= 16_000_000: raise ValueError('request bounds')
                    result = coordinator.request(json.loads(self.rfile.read(size)))
                    data = canonical(sign(result)); code = 200
                except Exception:
                    data = b'{"error":"request rejected"}'; code = 403
                self.send_response(code); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        super().__init__(address,Handler)
