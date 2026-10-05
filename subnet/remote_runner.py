"""Durable retained-host role liveness markers, independent of SSH sessions."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .backend_jobs import canonical

def ticks(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_bytes(canonical(value));temp.chmod(0o600);temp.replace(path)

def probe(workspace,identifier,physical=False):
    path=Path(workspace)/'runner-status'/(identifier+'.json')
    if not physical and (Path(workspace)/'jobs'/identifier/'report.json').is_file():return dict(phase='complete',job_id=identifier)
    if not path.exists():
        argument=(str(Path(workspace)/(identifier+'.json'))).encode()
        for item in Path('/proc').iterdir():
            if not item.name.isdigit():continue
            try:args=(item/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if argument in args:return dict(phase='running',job_id=identifier,discovered_pid=int(item.name),discovered_pid_ticks=ticks(item.name))
        return dict(phase='not_launched')
    value=json.loads(path.read_text())
    if value['phase']=='running':
        alive=any(value.get(k) and ticks(value[k])==value.get(k+'_ticks') for k in ('runner_pid','child_pid'))
        if not alive:value=dict(value,phase='failed',reason='all recorded job processes exited without terminal marker')
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--authority',required=True);p.add_argument('--workspace',required=True);p.add_argument('--checkpoint-cache');a=p.parse_args()
    envelope=json.loads(Path(a.job).read_text());identifier=envelope['payload']['job_id']
    from .backend_jobs import validate,signed
    validate(envelope,a.authority)
    manifest=signed(envelope['payload']['manifest'],a.authority)
    root=Path(a.workspace);markers=root/'runner-status';markers.mkdir(parents=True,exist_ok=True);markers.chmod(0o700);path=markers/(identifier+'.json')
    if path.exists():raise ValueError('refuse duplicate remote job launch')
    status=dict(phase='running',job_id=identifier,runner_pid=os.getpid(),runner_pid_ticks=ticks(os.getpid()),started_at=time.time());save(path,status)
    args=[sys.executable,'-B','-m','subnet.backend_jobs',a.job,'--authority',a.authority,'--workspace',a.workspace]
    if a.checkpoint_cache:args+=['--checkpoint-cache',a.checkpoint_cache]
    log=root/(identifier+'-worker.log')
    from .cache_lifecycle import CacheLifecycle
    os.environ['AFFINE_CACHE_LIFECYCLE_ROOT']=str(root.absolute())
    with CacheLifecycle(root).lease_checkpoint(manifest['checkpoint']['id'])as lease,log.open('wb') as output:
        if envelope['payload']['role']=='evaluate':
            from .evaluator_cache_lifecycle import retain
            retain(root,manifest['checkpoint']['id'])
        log.chmod(0o600);child=subprocess.Popen(args,stdout=output,stderr=subprocess.STDOUT,pass_fds=(lease,))
        status.update(child_pid=child.pid,child_pid_ticks=ticks(child.pid));save(path,status);code=child.wait()
    status.update(phase='complete' if code==0 else 'failed',exit_code=code,finished_at=time.time());save(path,status)
    if envelope['payload']['role']=='evaluate':
        from .evaluator_cache_lifecycle import retain
        try:retention=retain(root,manifest['checkpoint']['id'])
        except Exception as error:retention=dict(status='deferred',reason=type(error).__name__)
        save(markers/(identifier+'-cache-retention.json'),retention)
if __name__=='__main__':main()
