"""Inspect operator-owned GPU job evidence without executing a model or chain writes."""
import argparse
import hashlib
import json
from pathlib import Path

from subnet.backend_jobs import canonical, file_map, signed
from subnet.storage import Bucket, Identity


def inspect(state, bucket):
    authority=Identity(bytes.fromhex((state/'authority.seed').read_text().strip())).id
    complete=json.loads((state/'complete.json').read_text())
    if complete.get('success') is not True or complete.get('chain_transactions') is not False:
        raise ValueError('unfinished or payable GPU control')
    jobs={}; reports={}
    for path in sorted(state.glob('*-job.json')):
        job=signed(json.loads(path.read_text()),authority)
        manifest=signed(job['manifest'],authority)
        report=json.loads(path.with_name(path.name.replace('-job.json','-report.json')).read_text())
        expected={'job_id':job['job_id'],'role':job['role'],'operator':authority,
                  'checkpoint':manifest['checkpoint']['id'],'epoch':manifest['epoch'],
                  'source_files':job['source_files'],'runtime_versions':job['runtime_versions'],
                  'backend_profile':manifest['backend_profile'],'numerical_policy':manifest['numerical_policy'],
                  'job_sha256':hashlib.sha256(canonical(job)).hexdigest(),
                  'success':True,'chain_transactions':False}
        if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('job/report binding mismatch')
        if not job['created_at']<=report['completed_at']<job['expires_at']:raise ValueError('job completion outside window')
        if file_map(manifest['checkpoint']['files'])!=manifest['checkpoint']['id']:raise ValueError('input checkpoint identity')
        jobs[job['job_id']]=(job,manifest);reports[job['job_id']]=report
    if len(jobs)!=7:raise ValueError('expected complete seven-role control')
    frozen=[]
    for epoch in complete['epochs']:
        miners=[r for r in reports.values() if r['epoch']==epoch and r['role']=='mine']
        verifiers=[r for r in reports.values() if r['epoch']==epoch and r['role']=='verify']
        if len(miners)!=1 or len(verifiers)!=1:raise ValueError('missing fresh miner/verifier')
        miner=miners[0];verifier=verifiers[0];job,manifest=jobs[miner['job_id']]
        if manifest.get('payable',False):raise ValueError('payable control manifest')
        policy=json.loads((state/(epoch+'-freeze-policy.json')).read_text());receipt=policy['receipt']
        if policy.get('payable') is not False or not manifest['start']<=receipt['received_at']<manifest['deadline']:raise ValueError('invalid freeze timing/policy')
        body=bucket.get(receipt['frozen_key']);digest=hashlib.sha256(body).hexdigest()
        if digest!=receipt['sha256'] or digest!=miner['submission_sha256'] or len(body)!=receipt['size'] or len(body)!=miner['submission_size']:raise ValueError('frozen R2 bytes mismatch')
        if jobs[verifier['job_id']][0]['submissions'][0]['sha256']!=digest:raise ValueError('verifier submission mismatch')
        audit=verifier['audits'][0]
        if audit['submission_sha256']!=digest or len(audit['accepted'])!=miner['batches'] or any(o.get('valid') is not True or o.get('fully_audited') is not True for o in audit['outcomes']):raise ValueError('full audit missing')
        frozen.append(dict(epoch=epoch,checkpoint=manifest['checkpoint']['id'],sha256=digest,bytes=len(body),batches=miner['batches']))
    train=[r for r in reports.values() if r['role']=='train']
    if len(train)!=1:raise ValueError('training count')
    train=train[0];new=train['new_checkpoint']
    if new['id']!=file_map(new['files']) or new['id']!=complete['new_checkpoint']['id'] or new['id']==train['checkpoint']:raise ValueError('changed checkpoint identity')
    if frozen[1]['checkpoint']!=new['id']:raise ValueError('next epoch checkpoint handover')
    firstverify=next(r for r in reports.values() if r['role']=='verify' and r['epoch']==frozen[0]['epoch'])
    if train['audits']!=firstverify['audits'] or train['training']['steps']<1:raise ValueError('training independent audit binding')
    published=json.loads((state/'root-r2-checkpoint-independent-check.json').read_text())
    if published['checkpoint']!=new['id'] or published['published_file_hashes_verified'] is not True or {n:v['sha256'] for n,v in published['files'].items()}!=new['files']:raise ValueError('missing independent published byte audit')
    return dict(success=True,authority=authority,jobs_bound=len(jobs),frozen_epochs=frozen,
        new_checkpoint=new['id'],published_checkpoint_bytes=sum(v['bytes'] for v in published['files'].values()),
        chain_transactions=False,full_model_finetune=complete['full_model_finetune'],
        fresh_model_execution_in_this_check=False,remote_reports_are_operator_collected=True,
        continuous_controller_finalization_proven=False,resource_dependency_closure_proven=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--bucket-config',type=Path,default=Path('state/r2-direct.json'));a=p.parse_args()
    value=inspect(a.state,Bucket(json.loads(a.bucket_config.read_text())))
    (a.state/'root-role-independent-evidence.json').write_bytes(canonical(value))
    print(json.dumps(value))

if __name__=='__main__':main()
