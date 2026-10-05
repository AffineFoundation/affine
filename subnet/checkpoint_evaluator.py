"""Independent durable checkpoint evaluation, never a training prerequisite.

The queue keeps the original complete manifest and exact cohort configuration.
No checkpoint is called evaluated until its authenticated remote report passes
all existing heldout completeness, source, seed and worker-time checks.
"""
import argparse
import fcntl
import hashlib
import json
import logging
import time
from pathlib import Path
from .remote_backend import save, RemoteObservationTimeout
from .storage import canonical

VERSION='independent-checkpoints-v1'
CONFIG_FIELDS=('heldout','environment','environments','evaluation_experiment_id','evaluation_seed',
               'model_id','evaluation_state')

def evaluation_mode(config):
    mode=config.get('evaluation_mode','synchronous-v1')
    if mode not in ('synchronous-v1',VERSION):raise ValueError('evaluation mode')
    return mode

def enqueue(controller,manifest,cache,phase,steps,config,*,public_optimizer_steps=None):
    from .gpu_service import heldout
    if evaluation_mode(config)!=VERSION:raise ValueError('independent evaluation admission')
    # Validate the complete fixed plan before the controller advances. No
    # filtering, fresh seeds or relabeling of the configured comparison cohort.
    plan=heldout(config,manifest)
    if not plan or phase not in ('before','after') or type(steps) is not int or steps<0:
        raise ValueError('bounded checkpoint evaluation request')
    if public_optimizer_steps is not None and (type(public_optimizer_steps) is not int or public_optimizer_steps<0):
        raise ValueError('public optimizer counter')
    label=manifest['epoch']+'-eval-'+phase
    selected={k:config[k] for k in CONFIG_FIELDS if k in config}
    request=dict(version=VERSION,label=label,manifest=manifest,cache=cache,
                 phase=phase,training_steps=steps,public_optimizer_steps=public_optimizer_steps,config=selected,heldout_plan=plan)
    digest=hashlib.sha256(canonical(request)).hexdigest()
    path=controller.state/'checkpoint-evaluations'/(label+'.json')
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['request']!=request or previous['request_sha256']!=digest:
            raise ValueError('immutable checkpoint evaluation request changed')
        return previous
    record=dict(request=request,request_sha256=digest,queued_at=time.time(),status='queued')
    save(path,record)
    return record

def evaluate_one(controller,path):
    from .gpu_service import heldout,evaluate
    path=Path(path);record=json.loads(path.read_text());request=record['request']
    if (request.get('version')!=VERSION or
            hashlib.sha256(canonical(request)).hexdigest()!=record['request_sha256'] or
            heldout(request['config'],request['manifest'])!=request['heldout_plan']):
        raise ValueError('checkpoint evaluation queue binding')
    if record['status']=='complete':return record
    try:
        records=evaluate(controller,request['manifest'],request['cache'],
                         request['phase'],request['training_steps'],request['config'])
    except RemoteObservationTimeout as error:
        # RemoteJobs persists and reuses the SAME signed job and original
        # expiry on the next pass. Never generate replacement evaluation.
        record.update(status='observing_original_job',remote_job_id=error.job_id,
                      last_observed_at=time.time())
        save(path,record)
        return record
    for result in records:
        result['public_optimizer_steps']=request['public_optimizer_steps']
        save(Path(request['config'].get('evaluation_state','state/evaluations'))/(result['run_id']+'.json'),result)
    record.update(status='complete',records=records,
                  completed_at=max(r['timestamp'] for r in records))
    public=dict(version=VERSION,epoch=request['manifest']['epoch'],checkpoint=request['manifest']['checkpoint']['id'],
                request_sha256=record['request_sha256'],phase=request['phase'],training_steps=request['training_steps'],
                public_optimizer_steps=request['public_optimizer_steps'],status=record['status'],
                records=records,queued_at=record['queued_at'],completed_at=record['completed_at'])
    controller.bucket.json('public/'+request['manifest']['epoch']+'/evaluation-'+request['phase']+'.json',controller.signed(public))
    save(path,record)
    return record

def progress(state):
    """Latest training and evaluation are independent, explicitly labeled."""
    state=Path(state);status=json.loads((state/'controller.json').read_text())
    rows=[json.loads(p.read_text()) for p in (state/'checkpoint-evaluations').glob('*.json')]
    completed=[r for r in rows if r['status']=='complete' and all(v['status']=='complete' for v in r['records'])]
    latest=max(completed,key=lambda r:(r['request']['training_steps'],r['completed_at'])) if completed else None
    return dict(version=VERSION,latest_training_checkpoint=status['checkpoint']['id'],
                latest_evaluated_checkpoint=latest['request']['manifest']['checkpoint']['id'] if latest else None,
                latest_evaluation_report_ids=[r['run_id'] for r in latest['records']] if latest else [],
                pending_checkpoints=sum(r['status']!='complete' for r in rows),
                public_optimizer_steps=status.get('public_optimizer_steps'),
                evaluation_caught_up=bool(latest and latest['request']['manifest']['checkpoint']['id']==status['checkpoint']['id']))

def run(config,once=False):
    from .remote_backend import RemoteJobs
    from .controller import Controller
    from .storage import Bucket,Gateway
    if evaluation_mode(config)!=VERSION:raise ValueError('independent evaluator configuration')
    if config.get('preparation_only',False) or not config.get('activation_allowed',True):
        raise ValueError('evaluator activation is not allowed')
    state=Path(config['state']);state.mkdir(parents=True,exist_ok=True)
    # A single evaluator process owns one physical role. A second service fails
    # closed instead of racing launches or overwriting progress receipts.
    with (state/'checkpoint-evaluator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bucket=Bucket(config['bucket']);gateway=Gateway(bucket,state_path=state/'gateway.json',public_url='http://unused-gpu-operator.invalid',direct_r2=True)
        if not (state/'authority.seed').exists():raise ValueError('existing controller authority required')
        controller=Controller(bucket,gateway,state)
        remote=config['remote']
        endpoint=dict(remote.get('roles',{}).get('evaluate',remote),
                      job_ttl_seconds_by_role=remote.get('job_ttl_seconds_by_role',{}),retain_original_jobs=True)
        remote_jobs=RemoteJobs(endpoint,controller)
        class EvaluationJobs:
            def run(self,label,role,manifest,cache=None,**fields):
                if role!='evaluate':raise ValueError('independent evaluator cannot dispatch other roles')
                local=endpoint.get('checkpoint_caches',{}).get(manifest['checkpoint']['id']) if 'roles' in remote else cache
                return remote_jobs.run(label,role,manifest,local,**fields)
        controller.jobs=EvaluationJobs()
        while True:
            files=sorted((state/'checkpoint-evaluations').glob('*.json'),key=lambda p:json.loads(p.read_text())['queued_at'])
            for path in files:
                if json.loads(path.read_text())['status']=='complete':continue
                try:evaluate_one(controller,path)
                except Exception as error:
                    # Faults do not become zero rewards or completed reports.
                    save(state/'checkpoint-evaluator-health.json',dict(status='retry_original_request',request=path.name,error_type=type(error).__name__,time=time.time()))
                    logging.exception('independent evaluation retains original request')
                break
            value=progress(state);save(state/'checkpoint-evaluation-progress.json',value)
            bucket.json('public/streams/'+config['epoch_prefix']+'/evaluation-progress.json',controller.signed(value))
            if once:return
            time.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.once)

if __name__=='__main__':main()
