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

FIXED32_INDICES=(6903,3689,47,3166,5601,4899,2214,7211,4292,437,2893,6498,5356,2077,6873,4012,6812,4327,335,3087,6665,4971,1849,6925,4414,433,3083,5656,5447,2235,7363,3574)

def config_admission(config):
    from subnet.owned_cached_evaluation import POLICY,validate_policy
    if config.get('version')not in ('continuous-owned-cached-checkpoints-v1','owned-cached-fixed32-1024-pair-v2','continuous-owned-cached-checkpoints-1024-v3','continuous-owned-cached-base-restart-1024-v4') or config.get('dispatch_allowed')is not True:raise ValueError('ROOT scoped cached evaluator dispatch')
    validate_policy(config.get('owned_evaluation_policy'))
    if 'trusted_evaluation_policy'in config:raise ValueError('mutually exclusive evaluation policies')
    if config.get('state')==config.get('production_state'):raise ValueError('separate original queue')
    if config.get('source_sha256')!='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373':raise ValueError('qualified GPU source4db')
    if config.get('source_bundle',{}).get('sha256')!=config['source_sha256']:raise ValueError('source descriptor')
    if len(config['heldout'])!=1 or len(config['heldout'][0]['indices'])!=32:raise ValueError('fixed32 diagnostic population')
    restart=config['version']=='continuous-owned-cached-base-restart-1024-v4'
    v2=restart or config['version']in('owned-cached-fixed32-1024-pair-v2','continuous-owned-cached-checkpoints-1024-v3')
    continuous=restart or config['version']=='continuous-owned-cached-checkpoints-1024-v3'
    if v2:
        if type(config.get('evaluation_token_cap'))is not int or config['evaluation_token_cap']!=1024 or config.get('stop_after_pair')is not (False if continuous else True):raise ValueError('explicit bounded1024 pair')
        if config['heldout'][0]['indices']!=list(FIXED32_INDICES) or config['heldout'][0].get('seed')!=20261002:raise ValueError('same original fixed32 indices and seeds')
        if type(config.get('evaluation_job_ttl_seconds'))is not int or config['evaluation_job_ttl_seconds']!=1800:raise ValueError('bounded1800 original job timeout')
        if config.get('before_optimizer_steps')!=(0 if restart else 10) or (not continuous and config.get('after_optimizer_steps')!=11):raise ValueError('explicit baseline optimizer steps')
        if continuous and not restart and (type(config.get('completed_history'))is not dict or config['completed_history'].get('state')in (config.get('state'),config.get('production_state'))):raise ValueError('explicit separate completed original history')
        if not config.get('evaluation_experiment_id','').endswith('-cap1024-v1'):raise ValueError('separate1024 experiment identity')
    elif 'evaluation_token_cap'in config or 'stop_after_pair'in config:raise ValueError('legacy128 profile cannot be relabeled')
    h=config['heldout'][0]['harness']
    if h!=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=1024 if v2 else 128,temperature=.7,top_p=1.):raise ValueError('explicit cached token-cap contract')
    if config.get('evaluation_mode')!='independent-checkpoints-v1':raise ValueError('independent evaluator mode')
    if config.get('legacy_evaluator_scheduler_must_remain_stopped')is not True:raise ValueError('exclusive independent evaluator scheduler')
    if restart:
        if config.get('completed_history')is not None or config.get('run_id')!='completed-math-base7b-restart-20261008-v1':raise ValueError('isolated fresh evaluation run')
        cp=config.get('base_checkpoint_descriptor',{})
        if cp.get('id')!=config['before_checkpoint'] or cp.get('id')!='6493a901bd009f0800d5eed97d19aee78947cc586afeb27abfb2c72032ad1924':raise ValueError('explicit original 7B base checkpoint')
    return POLICY

def queue_original(controller,config,manifest,steps,phase):
    from subnet.checkpoint_evaluator import enqueue
    # New signed diagnostic manifest; the production manifest remains immutable.
    m=copy.deepcopy(manifest);m['source_bundle']=copy.deepcopy(config['source_bundle']);m['payable']=False;m['capabilities']={}
    prefix='nonpayable-owned-cached-fixed32-cap1024-'if config.get('version')in('owned-cached-fixed32-1024-pair-v2','continuous-owned-cached-checkpoints-1024-v3','continuous-owned-cached-base-restart-1024-v4')else 'nonpayable-owned-cached-fixed32-'
    if config.get('version')=='continuous-owned-cached-base-restart-1024-v4':prefix='nonpayable-fixed32-restart-20261008-'
    m['epoch']=prefix+m['checkpoint']['id'][:24]
    return enqueue(controller,m,None,phase,steps,config,public_optimizer_steps=steps)

def observe(controller,config):
    base=authenticated_manifest(config['before_original_signed_job'],controller.authority.id)
    if base['checkpoint']['id']!=config['before_checkpoint'] or hashlib.sha256(Path(config['before_original_signed_job']).read_bytes()).hexdigest()!=config['before_original_signed_job_sha256']:raise ValueError('original BEFORE checkpoint scope')
    if config.get('version')=='continuous-owned-cached-base-restart-1024-v4':
        cp=config['base_checkpoint_descriptor']
        if cp['files']!=base['checkpoint']['files']:raise ValueError('original base model file binding')
        base['checkpoint']=copy.deepcopy(cp)
    queued=[queue_original(controller,config,base,config['before_optimizer_steps'],'before')]
    if config.get('version')=='owned-cached-fixed32-1024-pair-v2':
        from subnet.backend_jobs import signed,file_map
        # Immutable signed original closure survives later production advances.
        closure=signed(config['after_signed_completion'],controller.authority.id)
        cp=config['after_checkpoint'];file_map(cp['files'])
        if (closure['checkpoint']!=base['checkpoint']['id'] or closure['next_checkpoint']!=cp['id'] or cp['id']==base['checkpoint']['id']):raise ValueError('exact signed CP10 to CP11 closure')
        pointer=config['after_durable_pointer']
        if pointer['inference_checkpoint']!=cp['id'] or pointer['optimizer_steps']!=config['after_optimizer_steps']:raise ValueError('exact durable AFTER pointer')
        latest=dict(checkpoint=cp,optimizer_steps=config['after_optimizer_steps'])
    else:latest=durable_checkpoint(config['production_state'],controller.authority.id)
    if latest and latest['optimizer_steps']>config['before_optimizer_steps']:
        m=copy.deepcopy(base);m['checkpoint']=latest['checkpoint']
        queued.append(queue_original(controller,config,m,latest['optimizer_steps'],'after'))
    return queued

def pair_finished(controller,config):
    if config.get('stop_after_pair')is not True:return False
    rows=[json.loads(p.read_bytes())for p in(controller.state/'checkpoint-evaluations').glob('*.json')]
    expected={('before',config['before_checkpoint'],10),('after',config['after_checkpoint']['id'],11)}
    actual={(r['request']['phase'],r['request']['manifest']['checkpoint']['id'],r['request']['public_optimizer_steps'])for r in rows}
    if actual!=expected:return False
    for row in rows:
        if row.get('status')!='complete':return False
        # ACK disposal is written only after full bucket readback and terminal,
        # process, ownership, inode and active-lease guards have passed.
        role=json.loads((controller.state/'roles'/(row['request']['label']+'.json')).read_bytes())
        disposition=controller.state/'cache-disposal'/(role['job_id']+'.json')
        if not disposition.exists()or json.loads(disposition.read_bytes()).get('result',{}).get('status')!='complete':return False
    return True

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
        if config.get('version')in('owned-cached-fixed32-1024-pair-v2','continuous-owned-cached-checkpoints-1024-v3','continuous-owned-cached-base-restart-1024-v4'):
            from subnet.remote_backend import role_time_budget
            if role_time_budget(probe.config,'evaluate')!=config['evaluation_job_ttl_seconds']:raise ValueError('signed source route job timeout binding')
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
        if config.get('version')=='continuous-owned-cached-checkpoints-1024-v3':
            from ops.owned_cached_evaluator_history import inherit_completed
            inherit_completed(controller,jobs,config)
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
            if once or pair_finished(controller,config):return
            time.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    from subnet.backend_jobs import signed
    envelope=json.loads(Path(args.config).read_bytes());config=signed(envelope,'3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd')
    run(config,args.once)

if __name__=='__main__':main()
