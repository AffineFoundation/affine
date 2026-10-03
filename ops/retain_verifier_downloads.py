"""Original-signature/full-archive-checked removal of duplicate verifier ZIPs."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shlex,sqlite3,subprocess,time
from pathlib import Path
from subnet.storage import Bucket,canonical
from subnet.live_reward_bridge import signed
from ops.live_reward_writer import approved_source_members
from ops.submission_retention import completed_replicas

def verified_archive(bucket, plan):
 r=bucket.client.get_object(Bucket=bucket.name,Key=plan['archive_key']);body=r['Body'];h=hashlib.sha256();n=0
 try:
  if r['ContentLength']!=plan['size']:raise ValueError('archive size changed')
  for block in iter(lambda:body.read(1024*1024),b''):
   n+=len(block)
   if n>plan['size']:raise ValueError('archive stream exceeded size')
   h.update(block)
 finally:body.close()
 if n!=plan['size'] or h.hexdigest()!=plan['sha256']:raise ValueError('archive bytes changed')
 return dict(plan,archive_verified=True,archive_verified_at=time.time(),archive_read_bytes=n,archive_etag=r['ETag'])

def run_cycle(config_path, writer_path, authority, output, per_worker=8):
 os.umask(0o077)
 config_path=Path(config_path)
 if config_path.is_symlink() or config_path.stat().st_mode & 0o077:raise ValueError('private operator config required')
 c=json.loads(config_path.read_text());U=Path(output);U.mkdir(parents=True,mode=0o700,exist_ok=True)
 if U.is_symlink() or U.stat().st_mode & 0o077:raise ValueError('private retention output required')
 if type(per_worker) is not int or not 1<=per_worker<=8:raise ValueError('bounded worker replica limit')
 if len(bytes.fromhex(authority))!=32:raise ValueError('operator authority')
 out=U/('submission-replica-retention-'+str(time.time_ns()));out.mkdir(mode=0o700)
 writer=signed(json.loads(Path(writer_path).read_text()),authority);sources=approved_source_members(writer,authority)
 endpoints={e['worker_identity']:e for e in c['remote']['roles']['verify']}
 if len(endpoints)!=len(c['remote']['roles']['verify']) or not endpoints or set(endpoints)!=set(writer['verifier_identities']):raise ValueError('signed worker roster mismatch')
 workspaces={w:e['workspace'] for w,e in endpoints.items()}
 db=sqlite3.connect('file:'+str(Path(c['state'])/'roles/verifier-queue.sqlite3')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 active=json.loads((Path(c['state'])/'controller.json').read_text()).get('active')
 epoch=active.get('epoch') if active else None
 try:rows=[dict(row) for row in db.execute('select * from jobs where status="complete" and role="verify" order by rowid')]
 finally:db.close()
 plans={worker:[] for worker in endpoints}
 for row in rows:
  plans[row['worker']].extend(completed_replicas(row,authority,workspaces,sources,now=time.time()))
 helper=Path(__file__).with_name('submission_retention.py');helper_sha=hashlib.sha256(helper.read_bytes()).hexdigest();bucket=Bucket(c['bucket'])
 def apply(item):
  worker,candidates=item;e=endpoints[worker];local=out/worker;local.mkdir(mode=0o700)
  peer=e.get('user','root')+'@'+e['host'];options=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+e['known_hosts']]
  ssh=['ssh',*options,'-p',str(e['port']),peer];scp=['scp','-q',*options,'-P',str(e['port'])]
  remote=str(Path(e['workspace'])/'private-retention'/out.name)
  def call(argv):
   r=subprocess.run(argv,capture_output=True,text=True,timeout=180)
   if r.returncode:
    (local/'failure.private.json').write_bytes(canonical(dict(at=time.time(),code=r.returncode,stderr=r.stderr)))
    raise RuntimeError('duplicate cleanup refused; preserve original evidence')
   return r.stdout.strip()
  probe='from pathlib import Path;import json;paths='+repr([p['path'] for p in candidates])+';print(json.dumps([p for p in paths if Path(p).exists()]))'
  present=json.loads(call(ssh+[shlex.quote(e['python'])+' -I -B -c '+shlex.quote(probe)]))
  if len(set(present))!=len(present) or not set(present)<=set(p['path'] for p in candidates):raise ValueError('actual replica observation changed')
  (local/'fresh-local-presence.private.json').write_bytes(canonical(dict(at=time.time(),present=present)))
  selected=[p for p in candidates if p['path'] in present][:per_worker]
  verified=[verified_archive(bucket,plan) for plan in selected]
  (local/'fresh-archive-readback.private.json').write_bytes(canonical(verified))
  if not verified:return worker,dict(results=[],no_present_completed_replicas=True)
  call(ssh+[shlex.quote(e['python'])+' -I -B -c '+shlex.quote('from pathlib import Path;Path('+repr(remote)+').mkdir(parents=True,mode=0o700,exist_ok=False)')])
  for p,name in [(helper,'retention.py'),(local/'fresh-archive-readback.private.json','plan.private.json')]:call(scp+[str(p),peer+':'+remote+'/'+name])
  code='ROOT='+repr(remote)+'\nHELPER_SHA='+repr(helper_sha)+'\nPLAN_SHA='+repr(hashlib.sha256((local/'fresh-archive-readback.private.json').read_bytes()).hexdigest())+'\n'+'''import hashlib,importlib.util,json,os,time
from pathlib import Path
root=Path(ROOT);helper=root/'retention.py';plan=root/'plan.private.json'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
assert hashlib.sha256(plan.read_bytes()).hexdigest()==PLAN_SHA
spec=importlib.util.spec_from_file_location('reviewed_retention',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=json.loads(plan.read_text());before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
results=[m.remove_verified_replica(row) for row in rows]
after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
receipt=dict(at=time.time(),results=results,free_before=before,free_after=after,model_caches_changed=False,reports_or_jobs_removed=False)
(root/'actual-completion.private.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
  (local/'original-operation.private.json').write_bytes(canonical(dict(at=time.time(),helper_sha256=helper_sha,plan_sha256=hashlib.sha256((local/'fresh-archive-readback.private.json').read_bytes()).hexdigest(),remote=remote)))
  actual=json.loads(call(ssh+[shlex.quote(e['python'])+' -I -B -c '+shlex.quote(code)]))
  (local/'actual-completion.private.json').write_bytes(canonical(actual));return worker,actual
 results={}
 with concurrent.futures.ThreadPoolExecutor(min(4,len(endpoints))) as pool:
  futures={pool.submit(apply,item):item[0] for item in plans.items() if item[1]}
  for future,worker in futures.items():
   try:k,result=future.result();results[k]=result
   except Exception as error:
    failure=out/worker/'operation-error.private.json'
    failure.write_bytes(canonical(dict(at=time.time(),error_type=type(error).__name__)))
    results[worker]={'operation_failed':type(error).__name__,'original_evidence_retained':True}
 (out/'root-review.private.json').write_bytes(canonical(dict(at=time.time(),epoch=epoch,actual=results,authenticated_original_jobs=True,all_operations_succeeded=not any('operation_failed' in r for r in results.values()),successful_deletions_have_fresh_archive_readback=True,live_job_or_deadline_changed=False)))
 summary=dict(roles=len(results),removed_files=sum(row.get('removed',False) for result in results.values() for row in result.get('results',[])),removed_bytes=sum(row.get('bytes',0) for result in results.values() for row in result.get('results',[])),failures=sum('operation_failed' in result for result in results.values()),models_reports_and_jobs_preserved=True)
 return summary

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--config',required=True);parser.add_argument('--writer-cutover',required=True)
 parser.add_argument('--authority',required=True);parser.add_argument('--output',required=True)
 parser.add_argument('--per-worker',type=int,default=8)
 parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=300)
 args=parser.parse_args()
 if not 60<=args.interval<=3600:raise ValueError('retention interval bounds')
 output=Path(args.output);output.mkdir(parents=True,mode=0o700,exist_ok=True)
 if output.is_symlink() or output.stat().st_mode & 0o077:raise ValueError('private retention output required')
 fd=os.open(output/'retention.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   summary=run_cycle(args.config,args.writer_cutover,args.authority,output,args.per_worker)
   print(json.dumps(summary),flush=True)
   if not args.watch:
    if summary['failures']:raise SystemExit(1)
    return
   time.sleep(args.interval)

if __name__=='__main__':main()
