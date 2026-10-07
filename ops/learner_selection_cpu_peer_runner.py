"""ROOT-authorized, default-off durable CPU peer runner. No ambiguous retries."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path

def ticks(pid):
    try:
        fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z'else fields[19]
    except FileNotFoundError:return None

def save(path,value,exclusive=False):
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    if exclusive:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    else:
        temp=path.with_suffix('.tmp')
        fd=os.open(temp,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--authority',required=True);p.add_argument('--operator-root',required=True);p.add_argument('--scientific-root',required=True);p.add_argument('--entry',required=True);p.add_argument('--workspace',required=True);p.add_argument('--checkpoint-cache');p.add_argument('--fresh-bytecode-prefix',required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    prefix=Path(a.fresh_bytecode_prefix)
    if not prefix.is_absolute()or prefix.exists()or prefix.is_symlink():raise ValueError('fresh never-existing bytecode prefix')
    if sys.flags.isolated!=1 or sys.dont_write_bytecode is not True or sys.pycache_prefix!=str(prefix):raise ValueError('isolated -I -B -X pinned fresh bytecode required')
    entry=Path(a.entry)
    spec=importlib.util.spec_from_file_location('_root_CPU_peer_entry',entry);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    document=json.loads(Path(a.job).read_bytes());receipt,grant,_=module.prepare(document,a.authority,a.operator_root,a.scientific_root,entry)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=grant['peer_runner_sha256']:raise ValueError('separately signed actual peer runner SHA')
    if not a.apply:print(json.dumps(dict(review_only=True,**receipt),sort_keys=True));return
    if not grant['backend_execution_allowed']:raise ValueError('ROOT backend execution grant required')
    root=Path(a.workspace)
    if not root.is_absolute()or root.resolve()!=root or not root.is_dir()or root.stat().st_uid!=os.getuid():raise ValueError('ordinary owned existing workspace')
    job=document['payload'];identifier=job['job_id']
    if type(identifier)is not str or not identifier or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'for c in identifier):raise ValueError('safe original job identifier')
    markers=root/'runner-status';markers.mkdir(mode=0o700,exist_ok=True)
    if markers.resolve()!=markers or markers.stat().st_uid!=os.getuid():raise ValueError('owned marker directory')
    path=markers/(identifier+'.json')
    status=dict(phase='starting',job_id=identifier,runner_pid=os.getpid(),runner_pid_ticks=ticks(os.getpid()),started_at=time.time(),CPU_peer_admission=receipt,original_job_file_sha256=hashlib.sha256(Path(a.job).read_bytes()).hexdigest())
    save(path,status,exclusive=True)
    # Original lease protocol protects the actual current checkpoint. Metadata
    # override loading is explicit and separately ROOT-bound; numerical code is unchanged.
    backend=module.bind_backend(grant,a.operator_root,a.scientific_root)
    from subnet.cache_lifecycle import CacheLifecycle
    os.environ['AFFINE_CACHE_LIFECYCLE_ROOT']=str(root)
    argv=[sys.executable,'-I','-B','-X','pycache_prefix='+str(prefix),str(entry),'--job',a.job,'--authority',a.authority,'--operator-root',a.operator_root,'--scientific-root',a.scientific_root,'--workspace',a.workspace,'--execute-backend']
    if a.checkpoint_cache:argv+=['--checkpoint-cache',a.checkpoint_cache]
    try:
        with CacheLifecycle(root).lease_checkpoint(job['manifest']['payload']['checkpoint']['id'])as lease,(root/(identifier+'-worker.log')).open('xb')as output:
            output.name and os.chmod(output.name,0o600)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            child=subprocess.Popen(argv,stdout=output,stderr=subprocess.STDOUT,env=env,pass_fds=(lease,))
            status.update(phase='running',child_pid=child.pid,child_pid_ticks=ticks(child.pid),command=argv);save(path,status)
            code=child.wait()
        status.update(phase='complete'if code==0 else'failed',exit_code=code,finished_at=time.time(),actual_wait=True);save(path,status)
        raise SystemExit(code)
    except Exception as error:
        status.update(phase='failed',reason=type(error).__name__,finished_at=time.time(),uncertain_child_outcome=bool(status.get('child_pid')));save(path,status);raise
if __name__=='__main__':main()
