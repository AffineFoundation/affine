"""Wait for the exact committed optimizer ACK before trainer-side calibration.

This changes orchestration only. The existing retirement worker and all of its
live-role, state, signature and lease guards remain authoritative.
"""
import json,shlex,time
from pathlib import Path
VERSION='committed-trainer-ACK-before-calibration-v1'

def await_cleanup(controller,job,report,pointer,*,budget=1800):
    from subnet import persistent_training_controller as training
    from subnet.remote_backend import save
    path=controller.state/'roles'/(job['job_id']+'-trainer-cache-cleanup.json')
    key=str(path);deadline=time.monotonic()+budget
    with training._cleanup_lock:thread=training._cleanup_threads.get(key)
    if thread is not None and thread.is_alive():
        thread.join(max(0,deadline-time.monotonic()))
        if thread.is_alive():raise TimeoutError('original trainer ACK still running before calibration')
    result=json.loads(path.read_bytes())if path.exists()else None
    if not isinstance(result,dict)or result.get('status')!='complete':
        if time.monotonic()>=deadline:raise TimeoutError('trainer ACK observation budget before calibration')
        controller.jobs.prepare_training_cache_ack(job,report,pointer)
        result=controller.jobs.retire_training_cache(job,report,pointer)
        save(path,result)
    if result.get('status')!='complete' or result.get('optimizer_cache_promotion',{}).get('promoted')is not True:
        raise ValueError('exact committed optimizer ACK not complete; calibration remains closed')
    return result

def promoted_head(trainer,authority,job,report,pointer):
    from subnet.storage import sha,canonical
    expected=dict(authority=authority,job_id=job['job_id'],job_sha256=sha(canonical(job)),report_sha256=sha(canonical(report)),pointer=pointer,checkpoint=report['new_checkpoint']['id'])
    code='DATA='+repr(expected)+'\nROOT='+repr(trainer.workspace)+'\nCODE='+repr(trainer.code)+'\n'+'''import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,CODE)
from subnet.backend_jobs import signed
from subnet.cache_lifecycle import snapshot
from subnet.optimizer_state_cache import sha,verification_body
root=Path(ROOT)
def read(path):
 st=snapshot(path)
 if st['mode']&0o077:raise ValueError('private owned original ACK head')
 raw=path.read_bytes()
 if snapshot(path)!=st:raise ValueError('ACK metadata changed during observation')
 return json.loads(raw)
current=read(root/'.optimizer-state-cache/current.json');guard=read(root/'.optimizer-state-cache/promotion.json');retention=read(root/'.cache-lifecycle/trainer-current-state.json')
pid=guard.get('child_pid');ticks=guard.get('child_ticks')
if type(pid)is not int or pid<=0:raise ValueError('original ACK child identity required')
try:process=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
except FileNotFoundError:process=None
if process and process[0]!='Z'and (ticks is None or process[19]==str(ticks)):raise ValueError('original ACK child still live')
ack=signed(current['ROOT_ack'],DATA['authority'])
if (ack['job_id']!=DATA['job_id'] or ack['job_sha256']!=DATA['job_sha256'] or ack['report_sha256']!=DATA['report_sha256'] or ack['trainer_state']!=DATA['pointer'] or ack['new_checkpoint']['id']!=DATA['checkpoint'] or ack.get('authority_state_committed')is not True):raise ValueError('exact latest published checkpoint/state ACK')
if (guard.get('phase')!='complete' or guard.get('child_terminal_confirmed')is not True or guard['ack']!=current['ROOT_ack'] or retention['ROOT_ack']!=current['ROOT_ack'] or (root/'.optimizer-state-cache/pending.json').exists()):raise ValueError('real original ACK still incomplete')
if (current['job_id']!=DATA['job_id'] or current['job_sha256']!=DATA['job_sha256'] or current['descriptor_sha256']!=DATA['pointer']['descriptor_sha256'] or current.get('promotion_verification_sha256')!=sha(verification_body(current))):raise ValueError('exact authenticated promoted optimizer head')
print(json.dumps(dict(ready=True,job_id=DATA['job_id'],descriptor_sha256=DATA['pointer']['descriptor_sha256'],optimizer_steps=DATA['pointer']['optimizer_steps'],checkpoint=DATA['checkpoint'])))
'''
    return json.loads(trainer.command(shlex.quote(trainer.python)+' -I -B -c '+shlex.quote(code),timeout=90))

def install():
    from subnet import successor_calibration,persistent_training_protocol as protocol
    from subnet.backend_jobs import signed
    from subnet.storage import sha,canonical
    from subnet.remote_backend import save
    original=successor_calibration.before_open
    def before_open(controller,config,status,opening):
        pointer=status.get('trainer_state')
        if pointer is None:return original(controller,config,status,opening)
        trainer=getattr(controller.jobs,'roles',{}).get('train')
        if trainer is None or trainer.config.get('optimizer_state_lifecycle')!='trainer-local-only-v1':
            return original(controller,config,status,opening)
        binding=protocol.opening_binding(config,status,status['active']['epoch'])
        document=protocol.read_json(controller.bucket,pointer['descriptor_key'])
        protocol.validate_parent(document,binding,controller.authority.id)
        publication=signed(document,controller.authority.id)
        job=signed(json.loads((controller.state/'roles'/(publication['job_id']+'-job.json')).read_bytes()),controller.authority.id)
        report=json.loads((controller.state/'roles'/(publication['job_id']+'-report.json')).read_bytes())
        manifest=signed(job['manifest'],controller.authority.id)
        if sha(canonical(job))!=publication['job_sha256'] or job['persistent_training']['output_namespace']!=pointer['namespace']:
            raise ValueError('original acknowledged trainer request binding')
        protocol.validate_report(report,job,manifest)
        started=time.monotonic();await_cleanup(controller,job,report,pointer)
        receipt=promoted_head(trainer,controller.authority.id,job,report,pointer)
        save(controller.state/(status['active']['epoch']+'-calibration-ACK-ordering.json'),dict(version=VERSION,at=time.time(),wait_seconds=time.monotonic()-started,**receipt))
        return original(controller,config,status,opening)
    successor_calibration.before_open=before_open
