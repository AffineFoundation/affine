"""Authenticate terminal per-request cache-reference retirement, without mutation.

Authority-signed retirement entries affect only their named historical request.
Queued/leased/new requests always re-protect the checkpoint. Current checkpoint,
pending outputs, dispatch references and actual worker mappings are independent
protections and cannot be removed by this ledger.
"""
import json
import time
from subnet.distributed_roles import authenticate,Coordinator,digest
from subnet.storage import canonical
import hashlib


def proof_for_replaced_reference(original,replacement,decision_envelope,authority,*,now,max_attempts):
    if type(max_attempts)is not int or not 1<=max_attempts<=10:raise ValueError('exact attempt policy required')
    decision=authenticate(decision_envelope,authority)
    old=authenticate(json.loads(original['envelope']),authority)
    job=authenticate(json.loads(replacement['envelope']),authority)
    manifest=authenticate(job['manifest'],authority)
    if (original['status']!='failed' or type(original['attempt'])is not int or original['attempt']!=max_attempts
            or original.get('report')is not None or old['expires_at']>=now
            or original['id']!=old['job_id']or digest(old)!=original['digest']
            or original['expires']!=old['expires_at']):
        raise ValueError('original expired exhausted request required')
    expected=dict(schema='terminal-infrastructure-replacement-v1',original_job_id=old['job_id'],
        original_job_sha256=original['digest'],original_status='failed',original_attempts=max_attempts,
        replacement_job_id=job['job_id'],replacement_job_sha256=replacement['digest'],
        checkpoint=manifest['checkpoint']['id'],original_expiry_unchanged=True,
        expired_leases_extended=False,frozen_inputs_unchanged=True)
    if any(canonical(decision.get(k))!=canonical(v)for k,v in expected.items()):
        raise ValueError('authenticated terminal decision binding')
    if (replacement['status']!='complete' or replacement['id']!=job['job_id']or digest(job)!=replacement['digest']
            or old['manifest']!=job['manifest']or old['source_files']!=job['source_files']
            or old['runtime_versions']!=job['runtime_versions']or old['role']!=job['role']
            or [r['sha256']for r in old['submissions']]!=[r['sha256']for r in job['submissions']]):
        raise ValueError('completed replacement with identical science required')
    request=authenticate(json.loads(replacement['report_request']),replacement['worker'])
    report=json.loads(replacement['report'])
    if (digest(report)!=replacement['report_digest']or request.get('action')!='report'
            or request.get('job_id')!=job['job_id']or request.get('token')!=replacement['token']or request.get('report')!=report):
        raise ValueError('original authenticated replacement report')
    checker=Coordinator.__new__(Coordinator);checker.authority=authority;checker.clock=lambda:now
    checker.validate_report(report,job,manifest,replacement['digest'])
    return dict(original_job_id=old['job_id'],original_job_sha256=original['digest'],
        original_signed_expiry=old['expires_at'],original_attempts=max_attempts,
        replacement_job_id=job['job_id'],replacement_job_sha256=replacement['digest'],
        replacement_report_sha256=replacement['report_digest'],worker=replacement['worker'],
        checkpoint=manifest['checkpoint']['id'],decision_sha256=hashlib.sha256(canonical(decision_envelope)).hexdigest())


def remaining_queue_references(rows,ledger_envelope,authority,*,now,max_attempts,fetch_document):
    ledger=authenticate(ledger_envelope,authority)
    if ledger.get('authority')!=authority or ledger.get('schema')!='verifier-cache-reference-retirements-v1' or ledger.get('scope')!='named-terminal-queue-cache-references-only':
        raise ValueError('explicit scoped reference ledger required')
    entries=ledger.get('entries')
    if not isinstance(entries,list)or not 1<=len(entries)<=128:raise ValueError('bounded explicit retirement entries')
    indexed={r['id']:r for r in rows};retired={};protected=set()
    if len({e['original_job_id']for e in entries})!=len(entries):raise ValueError('duplicate reference entries')
    for entry in entries:
        original=indexed.get(entry['original_job_id']);replacement=indexed.get(entry['replacement_job_id'])
        if original is None or replacement is None:raise ValueError('original immutable history missing')
        # Revival or a new lease always protects its signed checkpoint.
        if original['status']in('queued','leased'):continue
        proof=proof_for_replaced_reference(original,replacement,entry['decision'],authority,now=now,max_attempts=max_attempts)
        if any(canonical(entry.get(k))!=canonical(v)for k,v in proof.items()):raise ValueError('signed retirement proof changed')
        documents=entry.get('canonical_r2_documents')
        if not isinstance(documents,dict)or len(documents)!=4:raise ValueError('complete canonical R2 request/report history required')
        prefix=ledger['history_prefix'].rstrip('/')
        expected={prefix+'/'+original['id']+'/job.json':json.loads(original['envelope']),
            prefix+'/'+replacement['id']+'/job.json':json.loads(replacement['envelope']),
            prefix+'/'+replacement['id']+'/report.json':json.loads(replacement['report']),
            prefix+'/'+replacement['id']+'/worker-report.json':json.loads(replacement['report_request'])}
        if set(documents)!=set(expected):raise ValueError('canonical named R2 history only')
        for key,value in expected.items():
            body=fetch_document(key)
            if hashlib.sha256(body).hexdigest()!=documents[key]or json.loads(body)!=value:
                raise ValueError('durable canonical history changed')
        retired[original['id']]=proof
    for row in rows:
        if row['status']=='complete':continue
        job=authenticate(json.loads(row['envelope']),authority)
        manifest=authenticate(job['manifest'],authority)
        if digest(job)!=row['digest']or row['id']!=job['job_id']:raise ValueError('queue identity changed')
        if row['id']not in retired:protected.add(manifest['checkpoint']['id'])
    return dict(protected_checkpoints=protected,retired_reference_ids=set(retired))
