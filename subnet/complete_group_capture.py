"""Authenticated compact capture and transactional unaudited group accounting.

The immutable snapshot is private even during an open epoch. Heavy proof bytes
remain declared until the auditor downloads/checks their committed hash. This
does not reintroduce a proof-verification prerequisite for learner eligibility.
"""
import sqlite3
import os
import time
from pathlib import Path
from .storage import canonical
from .complete_group_codec import loads,limits,need
from .complete_group_admission import (TRANSPORT,sha,digest,scope,validate_commitment,
                                       validate_document)
from .complete_group_transport import paths


class SubmissionRejected(ValueError):
    """Definite malformed/unavailable submitted input, never native/transport failure."""


def _submitted(condition, reason):
    if not condition:
        raise SubmissionRejected(reason)


def _submitted_validation(call, *args, **kwargs):
    # Only pure parsing/signature/admission of client-controlled bytes belongs
    # inside this boundary. Never wrap the native resolver or storage writes.
    from nacl.exceptions import BadSignatureError
    try:
        return call(*args, **kwargs)
    except (ValueError, TypeError, KeyError, IndexError, BadSignatureError) as error:
        raise SubmissionRejected(str(error)) from error


def _submission_response(bucket, method, key):
    from botocore.exceptions import ClientError
    try:
        return getattr(bucket.client, method)(Bucket=bucket.name, Key=key)
    except ClientError as error:
        code=error.response.get('Error',{}).get('Code')
        if code in ('NoSuchKey','NotFound','404'):
            raise SubmissionRejected('committed submission object is absent') from error
        raise


def _submission_get(bucket,key,limit):
    response=_submission_response(bucket,'get_object',key);stream=response['Body']
    try:
        size=response['ContentLength']
        _submitted(type(size) is int and 0<size<=limit,'submitted object exceeds framing bound')
        data=stream.read(size+1)
        # A short read says nothing about the bytes originally uploaded. Retry
        # the epoch rather than silently dropping an honest miner on outage.
        if len(data)!=size:
            raise OSError('incomplete submitted object transport read')
        return data,response['LastModified'].timestamp()
    finally:
        stream.close()


class AdmissionLedger:
    def __init__(self,path):
        self.path=Path(path);need(not self.path.is_symlink(),'private admission ledger path')
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        descriptor=os.open(self.path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        os.fchmod(descriptor,0o600);os.close(descriptor)
        with self._connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS groups (scope TEXT NOT NULL, miner TEXT NOT NULL, slot INTEGER NOT NULL, group_id TEXT NOT NULL, claim BLOB NOT NULL, PRIMARY KEY(scope,miner,slot), UNIQUE(scope,miner,group_id))')
            db.execute('CREATE TABLE IF NOT EXISTS commits (scope TEXT NOT NULL, miner TEXT NOT NULL, sha256 TEXT NOT NULL, snapshot TEXT NOT NULL, PRIMARY KEY(scope,miner,sha256))')
            # v8.2.3 replace-until-seal: append-only, hash-chained audit of every
            # admitted commitment; the newest row is the one the seal may use.
            db.execute('CREATE TABLE IF NOT EXISTS supersessions (scope TEXT NOT NULL, miner TEXT NOT NULL, sequence INTEGER NOT NULL, sha256 TEXT NOT NULL, received REAL NOT NULL, previous_sha256 TEXT, groups INTEGER NOT NULL, PRIMARY KEY(scope,miner,sequence))')
            db.execute('CREATE TABLE IF NOT EXISTS slots (scope TEXT NOT NULL, miner TEXT NOT NULL, slot INTEGER NOT NULL, artifact_sha256 TEXT NOT NULL, document_sha256 TEXT NOT NULL, PRIMARY KEY(scope,miner,slot))')
        self.path.chmod(0o600)

    def _connect(self):
        db=sqlite3.connect(self.path,timeout=10,isolation_level=None)
        db.execute('PRAGMA synchronous=FULL')
        return db

    def admit(self,manifest,miner,envelope,snapshot,assessments,received=None):
        # Inputs are the result of capture below; repeat pure checks rather than
        # assuming any arbitrary caller-owned dict came from that entry point.
        signed=canonical(envelope)
        validate_commitment(signed,manifest,miner,registered_miners={miner})
        claims=envelope['payload']['groups'];s=scope(manifest,miner)
        need(type(assessments) is list and len(assessments)==len(claims)
             and all(a['group_id']==c['group_id'] and a['batch_sha256']==c['batch_sha256']
                     and a['assurance']=='structurally-eligible-unaudited' for a,c in zip(assessments,claims)),
             'complete structural capture assessments')
        need(snapshot==f'private/{manifest["epoch"]}/complete-group-snapshots/{miner}/{sha(signed)}',
             'immutable private snapshot binding')
        db=self._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT slot,claim FROM groups WHERE scope=? AND miner=? ORDER BY slot',(s,miner)).fetchall()
            from .batch_lifetime import policy as lifetime_policy
            if lifetime_policy(manifest) is None:
                need(len(claims)>=len(old),'cumulative commitment regression')
                for slot,body in old:
                    need(slot<len(claims) and bytes(body)==canonical(claims[slot]),'previous committed task group cannot change or move slots')
                new=claims[len(old):]
            else:
                need(type(received) in (int,float),'superseding commitment needs its upload time')
                last=db.execute('SELECT sequence,sha256,received FROM supersessions WHERE scope=? AND miner=? ORDER BY sequence DESC LIMIT 1',(s,miner)).fetchone()
                if last is not None and last[1]==sha(signed):
                    new=[]
                else:
                    need(last is None or received>=last[2],'a commitment cannot supersede a newer one')
                    # Write-once slots (R2 enforces the signed If-None-Match too):
                    # a slot named by any earlier commitment keeps its exact bytes.
                    for row in claims:
                        prior=db.execute('SELECT artifact_sha256,document_sha256 FROM slots WHERE scope=? AND miner=? AND slot=?',(s,miner,row['slot'])).fetchone()
                        if prior is None:
                            db.execute('INSERT INTO slots(scope,miner,slot,artifact_sha256,document_sha256) VALUES(?,?,?,?,?)',
                                       (s,miner,row['slot'],row['artifact_sha256'],row['document_sha256']))
                        else:
                            need(tuple(prior)==(row['artifact_sha256'],row['document_sha256']),'write-once upload slot reused with other bytes')
                    db.execute('DELETE FROM groups WHERE scope=? AND miner=?',(s,miner))
                    new=claims
                    db.execute('INSERT INTO supersessions(scope,miner,sequence,sha256,received,previous_sha256,groups) VALUES(?,?,?,?,?,?,?)',
                               (s,miner,0 if last is None else last[0]+1,sha(signed),received,None if last is None else last[1],len(claims)))
            for row in new:
                db.execute('INSERT INTO groups(scope,miner,slot,group_id,claim) VALUES(?,?,?,?,?)',
                           (s,miner,row['slot'],row['group_id'],canonical(row)))
            db.execute('INSERT OR IGNORE INTO commits(scope,miner,sha256,snapshot) VALUES(?,?,?,?)',(s,miner,sha(signed),snapshot))
            db.execute('COMMIT')
            return dict(new_groups=len(new),total_groups=len(claims),assurance='structurally-eligible-unaudited',
                        fully_audited=False,commitment_sha256=sha(signed),snapshot=snapshot)
        except BaseException:
            if db.in_transaction:db.execute('ROLLBACK')
            raise
        finally:db.close()


def _get(bucket,key,limit):
    response=bucket.client.get_object(Bucket=bucket.name,Key=key);stream=response['Body']
    try:
        size=response['ContentLength'];need(type(size) is int and 0<size<=limit,'bounded actual object size before read')
        data=stream.read(size+1);need(len(data)==size,'complete exact object read')
        return data,response['LastModified'].timestamp()
    finally:stream.close()


def _immutable(bucket,key,data):
    # R2 S3 conditional PutObject is required for this new capture adapter. A
    # deployment must qualify it; do not silently fall back to overwrite.
    from botocore.exceptions import ClientError
    try:
        bucket.client.put_object(Bucket=bucket.name,Key=key,Body=data,ContentType='application/octet-stream',IfNoneMatch='*')
    except ClientError as error:
        if error.response.get('ResponseMetadata',{}).get('HTTPStatusCode')!=412:
            raise
        existing,_=_get(bucket,key,len(data));need(existing==data,'immutable capture collision')


def capture(bucket,ledger,manifest,miner,*,registered_miners,resolve_task,now=None):
    """Caller passes authenticated manifest and trusted native task resolver.

    `resolve_task(manifest, env_id, index)` returns TaskAuthority from a pinned
    operator task/prompt/checkpoint source. A submitted task hash cannot select
    or replace this authority. Audit/model work is not called here.
    """
    observed=time.time() if now is None else now
    need(type(observed) in (int,float),'capture observation time')
    bound=limits(manifest)
    data,received=_submission_get(bucket,paths(manifest,miner),bound.commitment_bytes)
    _submitted(manifest['start']<=received<manifest['deadline'] and received<=observed,'commitment upload outside signed epoch window')
    envelope=_submitted_validation(validate_commitment,data,manifest,miner,registered_miners=registered_miners)
    prefix=f'private/{manifest["epoch"]}/complete-group-snapshots/{miner}/{sha(data)}'
    documents=[];assessments=[];arrivals=[]
    for claim in envelope['payload']['groups']:
        proof_key,document_key=paths(manifest,miner,claim['slot'])
        body,at=_submission_get(bucket,document_key,bound.metadata_bytes)
        _submitted(manifest['start']<=at<manifest['deadline'] and at<=received,'compact document must precede committed snapshot')
        task=resolve_task(manifest,claim['env_id'],claim['index'])
        value,assessment=_submitted_validation(validate_document,body,manifest,miner,claim,task)
        # Cheap presence/time/length only; full proof digest and inference are
        # deliberately left to the asynchronous auditor.
        metadata=_submission_response(bucket,'head_object',proof_key)
        when=metadata['LastModified'].timestamp()
        _submitted(metadata['ContentLength']==claim['artifact_size'] and manifest['start']<=when<manifest['deadline']
             and when<=received,'declared proof presence/size/upload window')
        documents.append((claim['slot'],body));assessments.append(assessment);arrivals.append(when)
    # No eligibility accounting until every compact input is authenticated and
    # written to a private, immutable snapshot. Repeating publication is safe.
    for slot,body in documents:_immutable(bucket,prefix+f'/training/{slot}.json',body)
    _immutable(bucket,prefix+'/commitment.json',data)
    result=_submitted_validation(ledger.admit,manifest,miner,envelope,prefix,assessments,received)
    result['proof_status']='declared-not-fully-downloaded-or-verified'
    result['documents']=[dict(slot=i,sha256=sha(body),key=prefix+f'/training/{i}.json') for i,body in documents]
    from .claim_order import enabled
    if enabled(manifest):
        # Cloudflare R2 LastModified of each proof artifact (first-claim-wins).
        result['received_at']=received
        for row,when in zip(result['documents'],arrivals,strict=True):row['received_at']=when
    return result
