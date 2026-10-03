"""Bounded archive/readback/retirement of completed trainer intermediates and ZIPs.

Final learned exports and model caches are always retained. This is an operator
housekeeping process, never a model worker, epoch restart, or chain writer.
"""
import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.storage import Bucket,canonical
from subnet.live_reward_bridge import signed
from subnet.remote_backend import RemoteJobs
from ops.live_reward_writer import approved_source_members
from ops.retain_verifier_downloads import verified_archive


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with path.open('xb') as stream:stream.write(canonical(value))
    path.chmod(0o600)


def guard(config,process_record):
    record=json.loads(Path(process_record).read_text())
    fields=Path('/proc',str(record['child_pid']),'stat').read_text().rsplit(')',1)[1].split()
    if fields[0] in ('Z','X') or fields[19]!=record['child_ticks'] or record['config_sha256']!=sha(config):
        raise ValueError('original live controller/config required')
    c=json.loads(Path(config).read_text());state=json.loads((Path(c['state'])/'controller.json').read_text())
    protected={state['checkpoint']['id']}
    active=state.get('active') or {}
    if active.get('next_checkpoint'):protected.add(active['next_checkpoint']['id'])
    return c,protected


def admitted_training(job,manifest,sources):
    if (job.get('role')!='train' or manifest.get('payable') is not False or
        type(job.get('steps')) is not int or not 1<=job['steps']<=32 or
        re.fullmatch('[A-Za-z0-9_-]+',job.get('job_id','')) is None):
        raise ValueError('bounded original training job required')
    source=manifest['source_bundle']['sha256'];files=job.get('source_files',{})
    if source not in sources or not files or any(sources[source].get(n)!=h for n,h in files.items()):
        raise ValueError('approved original training source required')
    if not isinstance(job.get('submissions'),list) or not 1<=len(job['submissions'])<=256:
        raise ValueError('bounded original frozen submission population')


def submission_archives(job,manifest):
    receipts=manifest['audit_frozen_receipts'];result=[]
    for i,submission in enumerate(job['submissions']):
        matches=[r for r in receipts.values() if r['sha256']==submission['sha256']]
        if len(matches)!=1:raise ValueError('unique original training receipt required')
        r=matches[0]
        if (re.fullmatch('public/'+re.escape(manifest['epoch'])+'/submissions/[0-9a-f]{64}\\.zip',r['frozen_key']) is None or
            type(r['size']) is not int or not 0<r['size']<=2_000_000_000):
            raise ValueError('canonical bounded immutable training archive')
        result.append(dict(index=i,sha256=r['sha256'],size=r['size'],archive_key=r['frozen_key']))
    return result


def run_cycle(config_path,writer_path,authority,output,process_record):
    os.umask(0o077)
    config_path=Path(config_path);output=Path(output)
    if config_path.is_symlink() or config_path.stat().st_mode&0o077:raise ValueError('private operator config')
    c,protected=guard(config_path,process_record);state=Path(c['state']);roles=state/'roles'
    writer=signed(json.loads(Path(writer_path).read_text()),authority);sources=approved_source_members(writer,authority)
    if writer['compute_state']!=str(state):raise ValueError('original compute state binding')
    output.mkdir(parents=True,mode=0o700,exist_ok=True)
    if output.is_symlink() or output.stat().st_mode&0o077:raise ValueError('private retention state')
    out=output/('completed-training-'+str(time.time_ns()));out.mkdir(mode=0o700)
    endpoint=c['remote']['roles']['train'];peer=endpoint.get('user','root')+'@'+endpoint['host']
    options=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+endpoint['known_hosts']]
    ssh=['ssh',*options,'-p',str(endpoint['port']),peer];scp=['scp','-q',*options,'-P',str(endpoint['port'])]
    def call(argv,timeout=240):
        result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        if result.returncode:
            save(out/('transport-failure-'+str(time.time_ns())+'.private.json'),dict(at=time.time(),exit_code=result.returncode,stderr=result.stderr))
            raise RuntimeError('trainer housekeeping refused; original evidence retained')
        return result.stdout.strip()
    def remote(code,timeout=240):return json.loads(call(ssh+[shlex.quote(endpoint['python'])+' -I -B -c '+shlex.quote(code)],timeout))
    candidates=[]
    checker=RemoteJobs.__new__(RemoteJobs);checker.state=roles;checker.controller=SimpleNamespace(authority=SimpleNamespace(id=authority))
    for path in sorted(roles.glob('*-train.json')):
        prior=json.loads(path.read_text());jobid=prior['job_id'];reportpath=roles/(jobid+'-report.json')
        if not reportpath.exists():continue
        envelopepath=roles/(jobid+'-job.json');envelope=json.loads(envelopepath.read_text());job=signed(envelope,authority);manifest=signed(job['manifest'],authority)
        admitted_training(job,manifest,sources);report=json.loads(reportpath.read_text());checker.checked(report,prior,manifest)
        candidates.append(dict(job_id=jobid,base=dict(workspace=endpoint['workspace'],job_id=jobid,authority=authority,job_sha256=sha(envelopepath),report_sha256=sha(reportpath)),steps=job['steps'],job=job,manifest=manifest))
    probe='''import json,subprocess
from pathlib import Path
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():print(json.dumps({'deferred':'trainer GPU occupied'}))
else:
 rows=[]
 for c in CANDIDATES:
  root=Path(WORKSPACE)/'jobs'/c['job_id'];terminal=Path(WORKSPACE)/'runner-status'/(c['job_id']+'.json')
  if not terminal.exists():continue
  t=json.loads(terminal.read_text())
  if t.get('phase')!='complete' or t.get('exit_code')!=0:continue
  exports=[n for n in range(1,c['steps']) if (root/('checkpoint-step-'+str(n))).exists()][:2]
  zips=[i for i in range(c['submissions']) if (root/('submission-'+str(i)+'.zip')).exists()]
  if exports or zips:rows.append({'job_id':c['job_id'],'exports':exports,'submissions':zips});break
 print(json.dumps({'candidates':rows}))
'''
    probe='CANDIDATES='+repr([dict(job_id=x['job_id'],steps=x['steps'],submissions=len(x['job']['submissions'])) for x in candidates])+'\nWORKSPACE='+repr(endpoint['workspace'])+'\n'+probe
    observed=remote(probe);save(out/'actual-presence.private.json',observed)
    if not observed.get('candidates'):
        result=dict(removed_bytes=0,removed_replicas=0,deferred=observed.get('deferred'),final_exports_and_model_caches_preserved=True)
        save(out/'completion.private.json',result);return result
    selected=observed['candidates'][0];candidate=next(x for x in candidates if x['job_id']==selected['job_id']);job=candidate['job'];manifest=candidate['manifest'];base=candidate['base']
    location=str(Path(endpoint['workspace'])/'private-training-retention'/out.name);helper=Path(__file__).with_name('training_retention.py')
    call(ssh+[shlex.quote(endpoint['python'])+' -I -B -c '+shlex.quote('from pathlib import Path;Path('+repr(location)+').mkdir(parents=True,mode=0o700,exist_ok=False)')]);call(scp+[str(helper),peer+':'+location+'/retention.py'])
    prefix='ROOT='+repr(location)+'\nHELPER_SHA='+repr(sha(helper))+'\n'+'''import hashlib,importlib.util,json,os,time
from pathlib import Path
root=Path(ROOT);hp=root/'retention.py';assert hashlib.sha256(hp.read_bytes()).hexdigest()==HELPER_SHA
spec=importlib.util.spec_from_file_location('retention',hp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
'''
    inventory=remote('BASE='+repr(base)+'\nSELECTED='+repr(selected)+'\n'+prefix+'''
m.unreferenced([]);BASE['terminal_sha256']=m.hash_file(Path(BASE['workspace'])/'runner-status'/(BASE['job_id']+'.json'));m.completed_job(BASE)
exports={}
for step in SELECTED['exports']:
 p=Path(BASE['workspace'])/'jobs'/BASE['job_id']/('checkpoint-step-'+str(step));members=list(p.iterdir());assert 1<=len(members)<=32
 files={}
 for f in members:
  assert f.is_file() and not f.is_symlink() and 0<f.stat().st_size<=5*1024**3
  files[f.name]={'sha256':m.hash_file(f),'size':f.stat().st_size}
 exports[str(step)]={'directory':str(p),'files':files,'checkpoint':m.digest({n:v['sha256'] for n,v in files.items()})}
print(json.dumps({'base':BASE,'exports':exports}))
''',timeout=600)
    save(out/'actual-inventory.private.json',inventory);base=inventory['base'];bucket=Bucket(c['bucket']);uploads={}
    for step,row in inventory['exports'].items():
        if row['checkpoint'] in protected:raise ValueError('protected export cannot be retired')
        for name,file in row['files'].items():
            if re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',name) is None:raise ValueError('bounded export name')
            key='private/training-checkpoint-retention/'+job['job_id']+'/step-'+step+'/'+row['checkpoint']+'/'+name
            uploads[step+'/'+name]=dict(file,path=row['directory']+'/'+name,archive_key=key,put_url=bucket.presign(key,'put_object',expires=7200))
    capabilities=out/'upload-capabilities.private.json';save(capabilities,dict(base=base,files=uploads));call(scp+[str(capabilities),peer+':'+location+'/upload-capabilities.private.json'])
    operation='PLAN_SHA='+repr(sha(capabilities))+'\n'+prefix+'''
import requests
p=root/'upload-capabilities.private.json';assert m.hash_file(p)==PLAN_SHA;plan=json.loads(p.read_text());m.completed_job(plan['base']);receipts={}
for name,row in plan['files'].items():
 p=Path(row['path']);assert p.stat().st_size==row['size'] and m.hash_file(p)==row['sha256']
 with p.open('rb') as body:r=requests.put(row['put_url'],data=body,headers={'Content-Type':'application/octet-stream','If-None-Match':'*'},timeout=(30,900),allow_redirects=False)
 assert r.status_code in (200,201,204,412);receipts[name]={'status':r.status_code,'size':row['size'],'sha256':row['sha256']}
 (root/'upload-progress.private.json').write_text(json.dumps(receipts))
(root/'upload-completion.private.json').write_text(json.dumps(receipts));print(json.dumps({'uploaded_or_existing_objects':len(receipts)}))
'''
    save(out/'original-upload.private.json',dict(command_sha256=hashlib.sha256(operation.encode()).hexdigest(),helper_sha256=sha(helper),remote=location));actual=remote(operation,timeout=1800);save(out/'upload-completion.private.json',actual)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:archived=dict(zip(uploads,pool.map(lambda p:verified_archive(bucket,p),uploads.values())))
    save(out/'export-archive-readbacks.private.json',archived)
    key=SigningKey(bytes.fromhex((state/'authority.seed').read_text().strip()))
    if key.verify_key.encode().hex()!=authority:raise ValueError('operator archive signing identity')
    _,protected=guard(config_path,process_record);plans=[]
    for step,row in inventory['exports'].items():
        payload=dict(kind='archived-completed-training-intermediate-v1',job_id=job['job_id'],step=int(step),checkpoint=row['checkpoint'],files=row['files'],archives={n:{k:v for k,v in archived[step+'/'+n].items() if k!='put_url'} for n in row['files']},original_evidence=base)
        doc=dict(payload=payload,signer=authority,signature=base64.b64encode(key.sign(canonical(payload)).signature).decode());archive_key='private/training-checkpoint-retention/'+job['job_id']+'/step-'+step+'/'+row['checkpoint']+'/descriptor.json'
        # First write only: retries retain the earlier authenticated descriptor.
        from botocore.exceptions import ClientError
        try:bucket.client.put_object(Bucket=bucket.name,Key=archive_key,Body=canonical(doc),ContentType='application/json',IfNoneMatch='*')
        except ClientError as error:
            if str(error.response.get('Error',{}).get('Code')) not in ('PreconditionFailed','412'):raise
        readback=signed(json.loads(bucket.get(archive_key)),authority)
        for name in ('kind','job_id','step','checkpoint','files','original_evidence'):
            if readback[name]!=payload[name]:raise ValueError('immutable authenticated archive descriptor changed')
        for name in row['files']:
            for field in ('archive_key','sha256','size'):
                if readback['archives'][name][field]!=payload['archives'][name][field]:
                    raise ValueError('immutable archived object binding changed')
        save(out/('step-'+step+'-authenticated-archive.private.json'),readback)
        plans.append(dict(base,kind='checkpoint-export',directory=row['directory'],step=int(step),checkpoint=row['checkpoint'],protected_checkpoints=sorted(protected),files=row['files'],archive_verified=True,archive_authenticated=True))
    rows=submission_archives(job,manifest);selected_rows=[r for r in rows if r['index'] in selected['submissions']]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:readbacks=list(pool.map(lambda p:verified_archive(bucket,p),selected_rows))
    save(out/'submission-archive-readbacks.private.json',readbacks)
    for row in selected_rows:plans.append(dict(base,kind='submission',directory=endpoint['workspace']+'/jobs/'+job['job_id'],submission_index=row['index'],files={'submission-'+str(row['index'])+'.zip':dict(sha256=row['sha256'],size=row['size'])},archive_verified=True,archive_authenticated=True))
    planpath=out/'retirement-plans.private.json';save(planpath,plans);call(scp+[str(planpath),peer+':'+location+'/retirement-plans.private.json']);guard(config_path,process_record)
    operation='PLAN_SHA='+repr(sha(planpath))+'\n'+prefix+'''
p=root/'retirement-plans.private.json';assert m.hash_file(p)==PLAN_SHA;plans=json.loads(p.read_text());before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize;results=[]
for plan in plans:
 result=m.remove_training_replica(plan);results.append(result);(root/'retirement-progress.private.json').write_text(json.dumps(results))
result={'results':results,'free_before':before,'free_after':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,'at':time.time()};(root/'actual-completion.private.json').write_text(json.dumps(result));print(json.dumps(result))
'''
    save(out/'original-retirement.private.json',dict(command_sha256=hashlib.sha256(operation.encode()).hexdigest(),helper_sha256=sha(helper),remote=location));actual=remote(operation,timeout=1200);save(out/'actual-retirement.private.json',actual)
    result=dict(removed_bytes=sum(x['bytes'] for x in actual['results']),removed_replicas=sum(x['removed'] for x in actual['results']),final_exports_and_model_caches_preserved=True)
    save(out/'completion.private.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','writer-cutover','authority','output','controller-process'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=300);args=parser.parse_args()
    if not 60<=args.interval<=3600:raise ValueError('bounded retention interval')
    os.umask(0o077);output=Path(args.output);output.mkdir(parents=True,mode=0o700,exist_ok=True)
    if output.is_symlink() or output.stat().st_mode&0o077:raise ValueError('private retention output')
    fd=os.open(output/'retention.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            print(json.dumps(run_cycle(args.config,args.writer_cutover,args.authority,args.output,args.controller_process)),flush=True)
            if not args.watch:return
            time.sleep(args.interval)

if __name__=='__main__':main()
