"""Independent ROOT-scoped cached diagnostics of genuinely committed checkpoints.

CPU orchestration may be upgraded separately from its signed, frozen GPU source.
Original production evaluation queues and issued jobs are never adopted here.
"""
import argparse,copy,fcntl,hashlib,json,shlex,time
from pathlib import Path

def authenticated_manifest(path,authority):
    from subnet.backend_jobs import signed
    job=signed(json.loads(Path(path).read_bytes()),authority)
    if job['role'] not in ('mine','evaluate'):raise ValueError('original approved inference job')
    return signed(job['manifest'],authority)

def durable_checkpoint(production,authority):
    """The existing durable-commit boundary includes mandatory full-state readback."""
    from subnet.backend_jobs import signed
    production=Path(production);status=json.loads((production/'controller.json').read_bytes())
    if status.get('persistent_state_committed')is not True:return None
    epoch=status.get('last_completed_epoch',{}).get('epoch')
    if not epoch:return None
    closure=signed(json.loads((production/(epoch+'-signed-learner-completion.json')).read_bytes()),authority)
    pointer=json.loads((production/'latest-trainer-state.json').read_bytes())
    if closure!=status['last_completed_epoch'] or closure['next_checkpoint']!=status['checkpoint']['id']:raise ValueError('genuine signed durable completion')
    if (pointer!=status.get('trainer_state') or pointer['inference_checkpoint']!=closure['next_checkpoint'] or pointer['optimizer_steps']!=status.get('public_optimizer_steps')):raise ValueError('actual durable optimizer lineage')
    return dict(checkpoint=copy.deepcopy(status['checkpoint']),optimizer_steps=pointer['optimizer_steps'],closure=closure)

def config_admission(config):
    from subnet.owned_cached_evaluation import POLICY,validate_policy
    if config.get('version')!='continuous-owned-cached-checkpoints-v1' or config.get('dispatch_allowed')is not True:raise ValueError('ROOT scoped cached evaluator dispatch')
    validate_policy(config.get('owned_evaluation_policy'))
    if 'trusted_evaluation_policy'in config:raise ValueError('mutually exclusive evaluation policies')
    if config.get('state')==config.get('production_state'):raise ValueError('separate original queue')
    if config.get('source_sha256')!='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373':raise ValueError('qualified GPU source4db')
    if config.get('source_bundle',{}).get('sha256')!=config['source_sha256']:raise ValueError('source descriptor')
    if len(config['heldout'])!=1 or len(config['heldout'][0]['indices'])!=32:raise ValueError('fixed32 diagnostic population')
    h=config['heldout'][0]['harness']
    if h!=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=128,temperature=.7,top_p=1.):raise ValueError('fixed128 explicit cached contract')
    if config.get('evaluation_mode')!='independent-checkpoints-v1':raise ValueError('independent evaluator mode')
    if config.get('legacy_evaluator_scheduler_must_remain_stopped')is not True:raise ValueError('exclusive independent evaluator scheduler')
    return POLICY

def queue_original(controller,config,manifest,steps,phase):
    from subnet.checkpoint_evaluator import enqueue
    # New signed diagnostic manifest; the production manifest remains immutable.
    m=copy.deepcopy(manifest);m['source_bundle']=copy.deepcopy(config['source_bundle']);m['payable']=False;m['capabilities']={}
    m['epoch']='nonpayable-owned-cached-fixed32-'+m['checkpoint']['id'][:24]
    return enqueue(controller,m,None,phase,steps,config,public_optimizer_steps=steps)

def observe(controller,config):
    base=authenticated_manifest(config['before_original_signed_job'],controller.authority.id)
    if base['checkpoint']['id']!=config['before_checkpoint'] or hashlib.sha256(Path(config['before_original_signed_job']).read_bytes()).hexdigest()!=config['before_original_signed_job_sha256']:raise ValueError('original BEFORE checkpoint scope')
    queued=[queue_original(controller,config,base,config['before_optimizer_steps'],'before')]
    latest=durable_checkpoint(config['production_state'],controller.authority.id)
    if latest and latest['optimizer_steps']>config['before_optimizer_steps']:
        m=copy.deepcopy(base);m['checkpoint']=latest['checkpoint']
        queued.append(queue_original(controller,config,m,latest['optimizer_steps'],'after'))
    return queued

def run(config,once=False):
    from subnet.backend_jobs import canonical
    from subnet.checkpoint_evaluator import QualifiedEvaluationJobs,pending_pass
    from subnet.controller import Controller
    from subnet.storage import Bucket,Gateway
    from subnet.remote_backend import save
    config_admission(config);state=Path(config['state']);state.mkdir(parents=True,exist_ok=True)
    with (state/'continuous-owned-cached-evaluator.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not(state/'authority.seed').is_file():raise ValueError('ROOT owned diagnostic authority required')
        bucket=Bucket(config['bucket']);controller=Controller(bucket,Gateway(bucket,state_path=state/'gateway.json',public_url='http://unused-owned-evaluator.invalid',direct_r2=True),state)
        jobs=QualifiedEvaluationJobs(controller,config['evaluation_source_routes'])
        if set(jobs.rows)!={config['source_sha256']} or not jobs.rows[config['source_sha256']]['new_dispatch_approved']:raise ValueError('ROOT exact one-source diagnostic route')
        probe=jobs.instance(config['source_sha256'])
        guard="import json;from pathlib import Path;root=Path("+repr(config['original_evaluator_workspace'])+");marker=json.loads((root/'runner-status'/"+repr(config['original_evaluator_job_id']+'.json')+").read_bytes());assert marker['phase']=='complete' and marker['exit_code']==0 and marker['runner_pid']==129037 and marker['runner_pid_ticks']=='220584019' and marker['child_pid']==129052 and marker['child_pid_ticks']=='220584046';assert not Path('/proc/129037').exists() and not Path('/proc/129052').exists();print('original terminal preserved')"
        probe.command(shlex.quote(probe.python)+' -I -B -c '+shlex.quote(guard),timeout=30)
        class IdleBoundJobs:
            def busy(self):
                if jobs.busy():return True
                # No GPU imports. Preserve any current physical process reservation.
                script="import subprocess;print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())"
                return bool(probe.command(shlex.quote(probe.python)+' -I -B -c '+shlex.quote(script),timeout=30).strip())
            def dispatch_eligible(self,request):return jobs.dispatch_eligible(request)
            def run(self,*args,**kwargs):
                # An existing original is observed; an idle check never redraws it.
                label=args[0]
                if not(state/'roles'/(label+'.json')).exists() and self.busy():
                    from subnet.remote_backend import RemoteObservationTimeout
                    raise RemoteObservationTimeout(label,'evaluate')
                return jobs.run(*args,**kwargs)
        controller.jobs=IdleBoundJobs()
        while True:
            try:
                from ops.owned_cached_evaluator_cleanup import retire_completed
                retire_completed(controller,jobs,config)
                observe(controller,config)
                pending_pass(controller,dispatch_order=config.get('evaluation_dispatch_order'))
                retire_completed(controller,jobs,config)
            except (OSError,ValueError)as error:
                # Atomic publication files may be read across two generations.
                # Keep the original requests and defer dispatch; no zero score.
                save(state/'owned-cached-observer-fault.json',dict(error_type=type(error).__name__,observed_at=time.time(),infrastructure_failure=True,model_reward=None,dispatch_deferred=True))
            save(state/'owned-cached-observer-progress.json',dict(version=config['version'],source_sha256=config['source_sha256'],policy=config['owned_evaluation_policy'],observed_at=time.time(),production_queue_modified=False))
            if once:return
            time.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    from subnet.backend_jobs import signed
    envelope=json.loads(Path(args.config).read_bytes());config=signed(envelope,'3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd')
    run(config,args.once)

if __name__=='__main__':main()
