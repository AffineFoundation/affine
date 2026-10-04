"""Authenticated normal-completion hook for disposable trainer ZIP downloads.

No CLI apply or deletion occurs on import. Root calls prepare_completion after
original child.wait0, authenticated report and durable successor publication.
Full archive readback is required before any narrow remote retirement grant.
"""
import hashlib,json,re
from pathlib import Path

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(v):return hashlib.sha256(canonical(v)).hexdigest()
def prepare_completion(job_envelope,report,remote_evidence,authority,successor_readback,*,validate_receipt_report,approved_execution_sources):
    from nacl.signing import VerifyKey
    import base64
    def signed(doc):
        if doc.get('signer')!=authority:raise ValueError('original authority')
        VerifyKey(bytes.fromhex(authority)).verify(canonical(doc['payload']),base64.b64decode(doc['signature'],validate=True))
        return doc['payload']
    job=signed(job_envelope);manifest=signed(job['manifest']);jobid=job['job_id'];t=remote_evidence['terminal']
    if (job['role']!='train' or not re.fullmatch('[A-Za-z0-9_-]{1,200}',jobid) or report.get('success') is not True or report.get('job_sha256')!=sha(job)
            or report.get('job_id')!=jobid or report.get('role')!='train' or t.get('job_id')!=jobid or t.get('phase')!='complete' or t.get('exit_code')!=0):
        raise ValueError('actual authenticated training completion required')
    if remote_evidence['job_sha256']!=sha(job_envelope) or remote_evidence['report_sha256']!=sha(report):raise ValueError('original remote request/report full bytes')
    source=manifest.get('source_bundle',{}).get('sha256');members=approved_execution_sources.get(source)
    if not isinstance(members,dict) or not job.get('source_files') or any(members.get(n)!=h for n,h in job['source_files'].items()):
        raise ValueError('signed approved original receipt execution source required')
    if report.get('source_files')!=job['source_files'] or report.get('runtime_versions')!=job['runtime_versions']:
        raise ValueError('original executed source/runtime report binding')
    validate_receipt_report(report,job,manifest,authority)
    cp=report['new_checkpoint']['id']
    objects=successor_readback.get('objects',{})
    complete_readback=(successor_readback.get('all_checkpoint_objects_fully_read') is True or
        successor_readback.get('all_ten_checkpoint_objects_fully_read') is True and len(objects)==10)
    if (not complete_readback or successor_readback.get('successor_checkpoint')!=cp
            or {n:r.get('sha256') for n,r in objects.items()}!=report['new_checkpoint']['files']
            or any(type(r.get('bytes')) is not int or r['bytes']<=0 for r in objects.values())):
        raise ValueError('durable successor full readback protects final output')
    if len(job['submissions'])>256:raise ValueError('bounded original submission population')
    root=Path(remote_evidence['workspace'])/'jobs'/jobid;rows=[];frozen=manifest['audit_frozen_receipts']
    for i,obj in enumerate(job['submissions']):
        matches=[r for r in frozen.values() if r['sha256']==obj['sha256'] and r['size']==obj['size']]
        if len(matches)!=1:raise ValueError('unique signed immutable archive')
        row=matches[0]
        if not re.fullmatch('public/'+re.escape(manifest['epoch'])+'/submissions/[0-9a-f]{64}\\.zip',row['frozen_key']):raise ValueError('exact frozen epoch archive')
        rows.append(dict(index=i,name='submission-'+str(i)+'.zip',archive_key=row['frozen_key'],sha256=obj['sha256'],size=obj['size'],verifier_receipt_sha256=sha(obj['verifier_receipt'])))
    return dict(kind='authenticated-receipt-training-download-completion-hook-v1',job_id=jobid,workspace=remote_evidence['workspace'],authority=authority,
        job_sha256=remote_evidence['job_sha256'],report_sha256=remote_evidence['report_sha256'],terminal_sha256=remote_evidence['terminal_sha256'],
        protected_checkpoint=cp,protected_final_export=report['new_checkpoint']['path'],successor_readback_sha256=sha(successor_readback),execution_source_sha256=source,objects=rows,
        original_reports_requests_sources_model_weights_and_optimizer_state_preserved=True,full_durable_archive_readback_required=True)

def verify_durable_archives(plan,read_object,record):
    """read_object supplies bounded actual current R2 byte chunks; no local copy."""
    results=[]
    for row in plan['objects']:
        h=hashlib.sha256();n=0
        for part in read_object(row['archive_key']):
            if not isinstance(part,bytes) or not 0<len(part)<=16*1024**2:raise ValueError('bounded actual archive bytes')
            n+=len(part)
            if n>row['size']:raise ValueError('oversized durable archive')
            h.update(part)
        if n!=row['size'] or h.hexdigest()!=row['sha256']:raise ValueError('durable archive SHA/size differs from signed receipt')
        result=dict(row,full_current_durable_readback=True);record(result);results.append(result)
    return results

def retirement_grant(plan,readbacks):
    """Unsigned proposal: root must authenticate actual readbacks before signing."""
    if len(readbacks)!=len(plan['objects']):raise ValueError('complete durable archive readbacks required')
    base={k:plan[k] for k in ('workspace','job_id','authority','job_sha256','report_sha256','terminal_sha256')};rows=[]
    for row,actual in zip(plan['objects'],readbacks):
        if any(actual.get(k)!=row[k] for k in ('index','name','archive_key','sha256','size','verifier_receipt_sha256')) or actual.get('full_current_durable_readback') is not True:raise ValueError('exact per-download durable proof')
        rows.append(dict(base,kind='submission',directory=str(Path(plan['workspace'])/'jobs'/plan['job_id']),submission_index=row['index'],files={row['name']:dict(sha256=row['sha256'],size=row['size'])},archive_verified=True,archive_authenticated=True))
    return dict(domain='affine-completed-receipt-download-retirement-v1',qualification_only=False,completion_hook=True,plans=rows,
        original_completion_plan_sha256=sha(plan),durable_readbacks_sha256=sha(readbacks),protected_checkpoint=plan['protected_checkpoint'],protected_final_export=plan['protected_final_export'],no_model_or_optimizer_deletion=True)


def on_authenticated_training_completion(job_envelope,report,remote_evidence,authority,successor_readback,*,validate_receipt_report,approved_execution_sources,read_object,record_readback,enqueue_authenticated_retirement):
    """Reusable completion lifecycle: authenticate → archive/read back → enqueue.

    The caller supplies durable storage and an operator-owned journal/dispatcher.
    The dispatcher signs only the narrow returned grant and uses the existing
    ops.training_retention removal primitive with fresh idle/open-file guards.
    This hook never deletes models, mutates jobs or owns permanent bucket keys.
    Re-entry is handled by the caller's exclusive original operation journal;
    uncertain retirement must be observed, not relaunched blindly.
    """
    plan=prepare_completion(job_envelope,report,remote_evidence,authority,successor_readback,validate_receipt_report=validate_receipt_report,approved_execution_sources=approved_execution_sources)
    readbacks=verify_durable_archives(plan,read_object,record_readback)
    grant=retirement_grant(plan,readbacks)
    return enqueue_authenticated_retirement(grant)
