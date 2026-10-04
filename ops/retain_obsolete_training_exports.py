"""Retire at most one obsolete final export after complete public archive checks.

Current and pending checkpoints, unfinished verification jobs, original evidence
and R2 objects remain intact. This companion never starts or restarts a GPU job.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,re,shlex,sqlite3,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from subnet.storage import Bucket,canonical
from subnet.live_reward_bridge import signed
from subnet.remote_backend import RemoteJobs
from ops.live_reward_writer import approved_source_members
from ops.retain_completed_training import admitted_training,guard,sha,save
from ops.retain_verifier_downloads import verified_archive


def protected_checkpoints(config,process_record,authority):
    c,protected=guard(config,process_record)
    state=Path(c['state']);roles=state/'roles'
    active=json.loads((state/'controller.json').read_text()).get('active') or {}
    epoch=active.get('epoch')
    # Training can finish before the coordinator records next_checkpoint (for
    # example, while recovering its report/publication under disk pressure).
    # Never treat that original active epoch's completed successor as obsolete.
    if active.get('phase') in ('train','after') and epoch is not None:
        if not isinstance(epoch,str) or re.fullmatch('[A-Za-z0-9_-]{1,200}',epoch)is None:raise ValueError('active training epoch binding')
        priorpath=roles/(epoch+'-train.json')
        if priorpath.exists():
            prior=json.loads(priorpath.read_text());reportpath=roles/(prior['job_id']+'-report.json')
            if reportpath.exists():
                job=signed(json.loads((roles/(prior['job_id']+'-job.json')).read_text()),authority)
                manifest=signed(job['manifest'],authority)
                if job['role']!='train' or manifest['epoch']!=epoch:raise ValueError('pending original training epoch')
                checker=RemoteJobs.__new__(RemoteJobs);checker.state=roles;checker.controller=SimpleNamespace(authority=SimpleNamespace(id=authority))
                report=json.loads(reportpath.read_text());checker.checked(report,prior,manifest)
                protected.add(final_candidate(job,report)['checkpoint'])
    queue=roles/'verifier-queue.sqlite3'
    if queue.exists():
        with sqlite3.connect('file:'+str(queue)+'?mode=ro',uri=True) as db:
            for envelope, in db.execute('SELECT envelope FROM jobs WHERE status!="complete"'):
                job=signed(json.loads(envelope),authority)
                manifest=signed(job['manifest'],authority)
                protected.add(manifest['checkpoint']['id'])
    return c,protected


def final_candidate(job,report):
    cp=report['new_checkpoint'];files=cp['files'];path=Path(cp['path'])
    if (not path.is_absolute() or len(path.parents)<3 or
            path.name!='checkpoint-step-'+str(job['steps']) or
            path.parent.name!=job['job_id'] or path.parents[1].name!='jobs' or
            not isinstance(files,dict) or not 1<=len(files)<=32 or
            'config.json' not in files or not any(n.endswith('.safetensors') for n in files) or
            any(not isinstance(n,str) or re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',n)is None or
                not isinstance(d,str) or re.fullmatch('[0-9a-f]{64}',d)is None for n,d in files.items()) or
            hashlib.sha256(canonical(files)).hexdigest()!=cp['id']):
        raise ValueError('original complete final export binding')
    return dict(checkpoint=cp['id'],files=files,directory=str(path),workspace=str(path.parents[2]),step=job['steps'])


def run_cycle(config_path,writer_path,authority,output,process_record):
    os.umask(0o077);config_path=Path(config_path);output=Path(output)
    if config_path.is_symlink() or config_path.stat().st_mode&0o077:raise ValueError('private operator config')
    c,protected=protected_checkpoints(config_path,process_record,authority);state=Path(c['state']);roles=state/'roles'
    writer=signed(json.loads(Path(writer_path).read_text()),authority)
    if writer['compute_state']!=str(state):raise ValueError('original compute state binding')
    sources=approved_source_members(writer,authority)
    output.mkdir(parents=True,mode=0o700,exist_ok=True)
    if output.is_symlink() or output.stat().st_mode&0o077:raise ValueError('private retention state')
    out=output/('obsolete-final-'+str(time.time_ns()));out.mkdir(mode=0o700)
    candidates=[];checker=RemoteJobs.__new__(RemoteJobs);checker.state=roles;checker.controller=SimpleNamespace(authority=SimpleNamespace(id=authority))
    for p in sorted(roles.glob('*-train.json')):
        prior=json.loads(p.read_text());jobid=prior['job_id'];reportpath=roles/(jobid+'-report.json')
        if not reportpath.exists():continue
        envelopepath=roles/(jobid+'-job.json');job=signed(json.loads(envelopepath.read_text()),authority);manifest=signed(job['manifest'],authority)
        admitted_training(job,manifest,sources);report=json.loads(reportpath.read_text());checker.checked(report,prior,manifest)
        candidate=final_candidate(job,report)
        if candidate['checkpoint'] in protected:continue
        candidate.update(job_id=jobid,base=dict(workspace=candidate['workspace'],job_id=jobid,authority=authority,job_sha256=sha(envelopepath),report_sha256=sha(reportpath)))
        candidates.append(candidate)
    e=c['remote']['roles']['train'];peer=e.get('user','root')+'@'+e['host'];opts=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+e['known_hosts']]
    ssh=['ssh',*opts,'-p',str(e['port']),peer];scp=['scp','-q',*opts,'-P',str(e['port'])]
    def call(argv,timeout=240):
        r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        if r.returncode:
            save(out/('transport-failure-'+str(time.time_ns())+'.private.json'),dict(at=time.time(),exit_code=r.returncode,stderr=r.stderr))
            raise RuntimeError('obsolete-final retirement refused; inspect original evidence')
        return r.stdout.strip()
    def remote(code):return json.loads(call(ssh+[shlex.quote(e['python'])+' -I -B -c '+shlex.quote(code)]))
    probe='CANDIDATES='+repr(candidates)+'\n'+'''import json,subprocess
from pathlib import Path
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():print(json.dumps({'deferred':'trainer GPU occupied'}))
else:
 selected=None
 for row in CANDIDATES:
  marker=Path(row['workspace'])/'runner-status'/(row['job_id']+'.json')
  if not Path(row['directory']).exists() or not marker.is_file():continue
  terminal=json.loads(marker.read_text())
  if terminal.get('phase')=='complete' and terminal.get('exit_code')==0:selected=row;break
 print(json.dumps({'selected':selected}))
'''
    observed=remote(probe);save(out/'actual-presence.private.json',observed)
    selected=observed.get('selected')
    if selected is None:
        result=dict(removed_bytes=0,removed_replicas=0,deferred=observed.get('deferred'),current_and_pending_preserved=True)
        save(out/'completion.private.json',result);return result
    bucket=Bucket(c['bucket']);cp=selected['checkpoint']
    descriptor=signed(json.loads(bucket.get('public/checkpoints/'+cp+'/authorities/'+authority+'/checkpoint.json')),authority)
    if descriptor['id']!=cp or descriptor['files']!=selected['files']:raise ValueError('authenticated original public archive')
    def archive(item):
        name,digest=item;key='public/checkpoints/'+cp+'/'+name
        metadata=bucket.client.head_object(Bucket=bucket.name,Key=key);size=metadata['ContentLength']
        if type(size)is not int or not 0<size<=5*1024**3:raise ValueError('bounded archived checkpoint object')
        return name,verified_archive(bucket,dict(archive_key=key,sha256=digest,size=size))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:readbacks=dict(pool.map(archive,selected['files'].items()))
    save(out/'fresh-complete-public-readbacks.private.json',readbacks)
    _,protected=protected_checkpoints(config_path,process_record,authority)
    if cp in protected:raise ValueError('checkpoint became protected during archive verification')
    plan=dict(selected['base'],kind='checkpoint-export',directory=selected['directory'],step=selected['step'],checkpoint=cp,protected_checkpoints=sorted(protected),files={n:dict(sha256=v['sha256'],size=v['size']) for n,v in readbacks.items()},archive_verified=True,archive_authenticated=True)
    local=out/'operator-plan.private.json';save(local,plan);helper=Path(__file__).with_name('training_retention.py');location=str(Path(e['workspace'])/'private-training-retention'/out.name)
    call(ssh+[shlex.quote(e['python'])+' -I -B -c '+shlex.quote('from pathlib import Path;Path('+repr(location)+').mkdir(parents=True,mode=0o700,exist_ok=False)')])
    for p,name in [(helper,'retention.py'),(local,'plan.private.json')]:call(scp+[str(p),peer+':'+location+'/'+name])
    _,protected=protected_checkpoints(config_path,process_record,authority)
    if cp in protected:raise ValueError('checkpoint now protected; retirement refused')
    operation='ROOT='+repr(location)+'\nHELPER_SHA='+repr(sha(helper))+'\nPLAN_SHA='+repr(sha(local))+'\n'+'''import hashlib,importlib.util,json,os,time
from pathlib import Path
root=Path(ROOT);hp=root/'retention.py';p=root/'plan.private.json'
assert hashlib.sha256(hp.read_bytes()).hexdigest()==HELPER_SHA and hashlib.sha256(p.read_bytes()).hexdigest()==PLAN_SHA
spec=importlib.util.spec_from_file_location('retention',hp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);plan=json.loads(p.read_text());plan['terminal_sha256']=m.hash_file(Path(plan['workspace'])/'runner-status'/(plan['job_id']+'.json'))
before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize;result=m.remove_training_replica(plan)
receipt={'at':time.time(),'result':result,'free_before':before,'free_after':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize};(root/'actual-completion.private.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
    save(out/'original-retirement.private.json',dict(command_sha256=hashlib.sha256(operation.encode()).hexdigest(),helper_sha256=sha(helper),remote=location,checkpoint=cp))
    actual=remote(operation);save(out/'actual-retirement.private.json',actual);protected_checkpoints(config_path,process_record,authority)
    result=dict(removed_bytes=actual['result']['bytes'],removed_replicas=int(actual['result']['removed']),checkpoint=cp,current_and_pending_preserved=True,model_caches_changed=False,archive_objects_preserved=True)
    save(out/'completion.private.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','writer-cutover','authority','output','controller-process']:p.add_argument('--'+name,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=300);a=p.parse_args()
    if not 60<=a.interval<=3600:raise ValueError('bounded retention interval')
    os.umask(0o077);output=Path(a.output);output.mkdir(parents=True,mode=0o700,exist_ok=True)
    if output.is_symlink() or output.stat().st_mode&0o077:raise ValueError('private operator state')
    fd=os.open(output/'obsolete-final.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            print(json.dumps(run_cycle(a.config,a.writer_cutover,a.authority,output,a.controller_process)),flush=True)
            if not a.watch:return
            time.sleep(a.interval)

if __name__=='__main__':main()
