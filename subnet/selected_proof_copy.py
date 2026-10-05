"""Prospective bounded conditional snapshots of post-commit selected proof slots.

Metadata inventories are not frozen proof bytes. A copy receipt is not SHA/audit
verification: the original verifier still reads and hashes the entire artifact.
"""
import copy,time,math
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from .storage import canonical,sha,SubmissionPolicyError
VERSION='selected-proof-copy-v1'
POLICY=dict(version=VERSION,copy_workers=4)

def validate_policy(value,manifest):
    if value!=POLICY or type(value)is not dict or type(value.get('copy_workers'))is not int:
        raise ValueError('selected proof copy policy')
    from .commitment_transport import VERSION as TRANSPORT
    p=manifest.get('audit_policy',{})
    if (manifest.get('submission_transport_policy')!=TRANSPORT or
            p.get('version')!='bounded-random-v1' or p.get('mode')!='sampled' or
            manifest.get('hourly_execution_policy')is None):
        raise ValueError('selected copy requires signed sampled commitments and phase bounds')
    from .hourly_policy import validate
    validate(manifest['hourly_execution_policy'],manifest['hourly_execution_policy'].get('mine_seconds'))
    return dict(POLICY)

def capture_inventory(gateway,epoch):
    """All documents were authenticated first; capture every cheap HEAD now."""
    from .commitment_transport import FreezeMetadataIncomplete
    state=gateway.epochs[epoch];pending=state['commitment_pending'];rejections=state['rejections']
    cutoff=state['commitment_binding']['freeze_until']
    def expired():return time.time()>=cutoff
    client=gateway.bucket.commitment_read_client()if hasattr(gateway.bucket,'commitment_read_client')else gateway.bucket.client
    def inspect(item):
        miner,slot,b=item;key='private/'+epoch+'/staging/'+miner+'/'+str(slot)+'.zip'
        if expired():return miner,slot,None,TimeoutError('metadata cutoff')
        try:
            meta=client.head_object(Bucket=gateway.bucket.name,Key=key)
            if (b['size']>state.get('upload_limit',100_000_000) or
                    meta['ContentLength']!=b['size'] or
                    not state['start']<=meta['LastModified'].timestamp()<state['deadline']):
                raise SubmissionPolicyError('artifact size/time')
            return miner,slot,dict(etag=meta['ETag'],received_at=meta['LastModified'].timestamp()),None
        except Exception as error:return miner,slot,None,error
    rows=[(m,b['slot'],b)for m,p in pending.items()if m not in rejections
          for b in p['document']['payload']['batches']if str(b['slot'])not in p['artifact_plans']]
    failures=[]
    try:
        with ThreadPoolExecutor(max_workers=4)as pool:
            for miner,slot,meta,error in pool.map(inspect,rows):
                if error is None:pending[miner]['artifact_plans'][str(slot)]=meta
                elif isinstance(error,SubmissionPolicyError)or(isinstance(error,ClientError)and error.response.get('Error',{}).get('Code')in('NoSuchKey','404','NotFound')):
                    rejections[miner]='missing/size/time-invalid completed artifact'
                else:failures.append(type(error).__name__)
                gateway.persist()
    finally:
        if client is not gateway.bucket.client:client.close()
    if failures:
        state['commitment_metadata_incomplete']=dict(reason='artifact_HEAD_infrastructure_incomplete',at=time.time(),error_types=failures)
        gateway.persist();raise FreezeMetadataIncomplete('all artifact metadata unavailable')
    if expired():
        state['commitment_metadata_incomplete']=dict(reason='artifact_inventory_cutoff',at=time.time())
        gateway.persist();raise FreezeMetadataIncomplete('inventory publication cutoff')
    receipts=state.setdefault('commitment_snapshots',{})
    for miner,progress in pending.items():
        if miner in rejections or miner in receipts:continue
        if expired():raise FreezeMetadataIncomplete('inventory publication cutoff')
        raw=canonical(progress['document'])
        if sha(raw)!=progress['sha256']or len(raw)!=progress['size']:raise ValueError('exact captured canonical commitment bytes')
        # Publish the exact already captured small bytes, never reread a mutable
        # commitment or copy any heavyweight proof during inventory freezing.
        gateway.bucket.put(progress['root']+'/commitment.json',raw,'application/json')
        artifacts=[]
        for batch in progress['document']['payload']['batches']:
            meta=progress['artifact_plans'][str(batch['slot'])]
            artifacts.append(dict(batch,key='private/'+epoch+'/staging/'+miner+'/'+str(batch['slot'])+'.zip',
                frozen_key=progress['root']+'/'+str(batch['slot'])+'.zip',**meta,
                snapshot_status='metadata_captured'))
        receipts[miner]=dict(sha256=progress['sha256'],size=progress['size'],received_at=progress['received_at'],
            commitment_document=progress['document'],commitment_key=progress['root']+'/commitment.json',
            artifacts=artifacts,proof_copy_policy=dict(POLICY),hash_assurance='declared-payload-hashes-until-selected-verifier')
        gateway.persist()
    if expired():
        state['commitment_metadata_incomplete']=dict(reason='artifact_inventory_publication_cutoff',at=time.time())
        gateway.persist();raise FreezeMetadataIncomplete('inventory publication cutoff')
    state['frozen_receipts']=dict(receipts);state['proof_inventory_completed_at']=time.time();gateway.persist()
    gateway.bucket.json('public/'+epoch+'/receipts.json',state['frozen_receipts']);return state['frozen_receipts']

def conditional_snapshot(bucket,artifact,until):
    """One transport-only copy; no reread/rebinding of the captured source."""
    start=time.time()
    if start>=until:return dict(status='budget_deferred',started_at=start,finished_at=start)
    client=None
    try:
        if hasattr(bucket,'selected_proof_client'):
            client=bucket.selected_proof_client(until-start)
            client.copy_object(Bucket=bucket.name,Key=artifact['frozen_key'],
                CopySource={'Bucket':bucket.name,'Key':artifact['key']},CopySourceIfMatch=artifact['etag'])
        else:bucket.copy(artifact['key'],artifact['frozen_key'],expected_etag=artifact['etag'])
        if time.time()>=until:raise TimeoutError('conditional copy finished after cutoff')
        head=(client or bucket.client).head_object(Bucket=bucket.name,Key=artifact['frozen_key'])
        if head['ContentLength']!=artifact['size']:raise ValueError('selected destination size changed')
    except Exception as error:
        return dict(status='budget_deferred'if time.time()>=until else'infrastructure_deferred',
            started_at=start,finished_at=time.time(),error_type=type(error).__name__)
    finally:
        if client is not None:client.close()
    finish=time.time()
    return dict(status='copied'if finish<until else'budget_deferred',started_at=start,
        finished_at=finish,conditional_copy_completed=True,full_SHA_verified=False,
        destination_etag=head['ETag'])

def selected_slots(manifest,receipts):
    """Recompute the original unpredictable draw, never replace failed slots."""
    from .audit_policy import selection
    counts=manifest['audit_policy']['submission_counts']
    return {m:selection(len(r['artifacts']),counts[r['sha256']],manifest['audit_seed'],r['sha256'])for m,r in receipts.items()}

def snapshot_selected(gateway,manifest,slots):
    validate_policy(manifest['proof_copy_policy'],manifest)
    from .hourly_policy import cutoff
    epoch=manifest['epoch'];state=gateway.epochs[epoch];receipts=state['frozen_receipts']
    if receipts!=manifest['audit_frozen_receipts']or slots!=selected_slots(manifest,receipts):raise ValueError('exact captured inventory/original audit draw')
    plan=dict(epoch=epoch,source=manifest['source_bundle']['sha256'],checkpoint=manifest['checkpoint']['id'],
        receipts_sha256=sha(canonical(receipts)),audit_seed=manifest['audit_seed'],slots=slots,policy=manifest['proof_copy_policy'],audit_until=cutoff(manifest,'audit'))
    if state.setdefault('selected_proof_copy_plan',copy.deepcopy(plan))!=plan:raise ValueError('original selected copy plan changed')
    gateway.persist();results=state.setdefault('selected_proof_copy_results',{})
    rows=[(m,b)for m,r in receipts.items()for b in r['artifacts']if b['slot']in slots[m]and m+'/'+str(b['slot'])not in results]
    def run(row):
        miner,artifact=row;return miner,artifact,conditional_snapshot(gateway.bucket,artifact,plan['audit_until'])
    with ThreadPoolExecutor(max_workers=4)as pool:
        for miner,artifact,outcome in pool.map(run,rows):
            results[miner+'/'+str(artifact['slot'])]=dict(outcome,miner=miner,slot=artifact['slot'],
                commitment_sha256=receipts[miner]['sha256'],source_key=artifact['key'],source_etag=artifact['etag'],
                source_received_at=artifact['received_at'],frozen_key=artifact['frozen_key'],size=artifact['size'],sha256=artifact['sha256'])
            gateway.persist()
    return copy.deepcopy(results)

def require_snapshot(manifest,miner,artifact):
    """Original job/admission gate: a planned destination is never enough."""
    validate_policy(manifest['proof_copy_policy'],manifest)
    receipts=manifest['audit_frozen_receipts'];row=manifest.get('selected_proof_copy_receipts',{}).get(miner+'/'+str(artifact['slot']))
    if artifact['slot']not in selected_slots(manifest,receipts)[miner]or type(row)is not dict:raise ValueError('original selected proof copy missing')
    expected=dict(miner=miner,slot=artifact['slot'],commitment_sha256=receipts[miner]['sha256'],
        source_key=artifact['key'],source_etag=artifact['etag'],source_received_at=artifact['received_at'],
        frozen_key=artifact['frozen_key'],size=artifact['size'],sha256=artifact['sha256'])
    from .hourly_policy import cutoff
    times=[row.get(k)for k in ('started_at','finished_at')]
    if (any(type(t)not in (int,float)or not math.isfinite(t)for t in times)or
            artifact.get('snapshot_status')!='metadata_captured' or
            receipts[miner].get('proof_copy_policy')!=POLICY or
            any(row.get(k)!=v for k,v in expected.items())or row.get('status')!='copied' or
            row.get('conditional_copy_completed')is not True or row.get('full_SHA_verified')is not False or
            not manifest['deadline']<=row.get('started_at',-1)<=row.get('finished_at',-1)<cutoff(manifest,'audit')):
        raise ValueError('exact timely selected conditional snapshot required')
    return row

def audit_inventory(manifest,bucket):
    projected=copy.deepcopy(manifest['audit_frozen_receipts'])
    for miner,receipt in projected.items():
        for artifact in receipt['artifacts']:
            row=manifest['selected_proof_copy_receipts'].get(miner+'/'+str(artifact['slot']))
            if row and row['status']=='copied':
                require_snapshot(manifest,miner,artifact)
                artifact['read_url']=bucket.presign(artifact['frozen_key'])
    return projected

def annotate_deferrals(manifest,miner,report):
    """Preserve actual successful audits and label unavailable selected slots."""
    receipt=manifest['audit_frozen_receipts'][miner]
    for artifact in receipt['artifacts']:
        row=manifest['selected_proof_copy_receipts'].get(miner+'/'+str(artifact['slot']))
        if row and row['status']!='copied':
            report['outcomes'][artifact['slot']].update(valid=None,fully_audited=False,failure_kind=row['status'])
    return report


def captured_inventory(manifest):
    """Remove renewable URLs only, retaining the exact originally drawn inventory."""
    receipts=copy.deepcopy(manifest['audit_frozen_receipts'])
    for miner,receipt in receipts.items():
        for artifact in receipt['artifacts']:
            if 'read_url'in artifact:
                require_snapshot(manifest,miner,artifact)
                del artifact['read_url']
    return receipts
