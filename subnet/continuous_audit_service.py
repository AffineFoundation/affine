"""Independent continuous audit scheduling and immutable hourly score snapshots.

Learner progress never waits for this service. It joins the original SQLite
queue without binding another HTTP listener. No blockchain writes occur here.
"""
import argparse,hashlib,json,os,secrets,sqlite3,time
from pathlib import Path
from .continuous_audit_policy import VERSION,policy,population,digest,random_selection,verifier_contract,admit_queue_reports,snapshot
from .distributed_roles import authenticate
from .storage import canonical


class InvalidCommittedArtifact(ValueError):pass

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 if path.is_symlink():raise ValueError('private audit journal path')
 temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with temporary.open('wb')as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
 temporary.chmod(0o600);temporary.replace(path)


def register_population(manifest_document,receipts,round,committed_at,authority):
 """Return a signed-service input proposal from original immutable captures."""
 from .commitment_transport import validate
 manifest=authenticate(manifest_document,authority);rows=[]
 for miner,receipt in sorted(receipts.items()):
  document=validate(canonical(receipt['commitment_document']),manifest['epoch'],miner,manifest['max_batches'])
  if digest(document)!=receipt['sha256']or document['payload']['checkpoint']!=manifest['checkpoint']['id']or document['payload']['source']!=manifest['source_bundle']['sha256']:raise ValueError('original signed committed population')
  for b in document['payload']['batches']:
   rows.append(dict(epoch=manifest['epoch'],round=round,checkpoint=manifest['checkpoint']['id'],miner=miner,env_id=b['env_id'],index=b['index'],batch_sha256=b['batch_sha256'],proof_sha256=b['sha256'],commitment_sha256=receipt['sha256'],verifier_contract_sha256=verifier_contract(manifest),committed_at=committed_at))
 return dict(version='continuous-audit-population-v1',round=round,committed_at=committed_at,manifest_document=manifest_document,receipts=receipts,records=population(rows))


class ContinuousAuditor:
 def __init__(self,controller,queue,*,directory,approved_sources,job_metadata,audit_policy,max_inflight=8,budget_per_tick=8,job_seconds=900):
  self.controller=controller;self.queue=queue;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True,mode=0o700);self.sources=approved_sources;self.metadata=job_metadata;self.policy=policy(audit_policy)
  if type(max_inflight)is not int or not 1<=max_inflight<=128 or type(budget_per_tick)is not int or not 1<=budget_per_tick<=128 or type(job_seconds)is not int or not 60<=job_seconds<=86400:raise ValueError('bounded continuous audit scheduler')
  self.max_inflight=max_inflight;self.budget=budget_per_tick;self.job_seconds=job_seconds
  self.state_path=self.directory/'audit-state.json';self.state=json.loads(self.state_path.read_text())if self.state_path.exists()else dict(populations={},draws={},jobs={},capture_failures={})
 def persist(self):atomic(self.state_path,self.state)
 def admit(self,document):
  p=authenticate(document,self.controller.authority.id)
  if p.get('version')!='continuous-audit-population-v1':raise ValueError('continuous immutable population admission')
  expected=register_population(p['manifest_document'],p['receipts'],p['round'],p['committed_at'],self.controller.authority.id)
  if p!=expected:raise ValueError('canonical signed population registration')
  epoch=authenticate(p['manifest_document'],self.controller.authority.id)['epoch'];old=self.state['populations'].get(epoch)
  if old is not None and old!=document:raise ValueError('immutable audit population cannot be replaced')
  self.state['populations'][epoch]=document;self.persist()
 def records(self):return [r for document in self.state['populations'].values()for r in authenticate(document,self.controller.authority.id)['records']]
 def _capture(self,row,p):
  manifest=authenticate(p['manifest_document'],self.controller.authority.id);receipt=p['receipts'][row['miner']];selected=[b for b in receipt['artifacts']if b['batch_sha256']==row['batch_sha256']and b['sha256']==row['proof_sha256']]
  if len(selected)!=1:raise ValueError('exact selected declared child')
  artifact=selected[0];key=artifact['key'];meta=self.controller.bucket.client.head_object(Bucket=self.controller.bucket.name,Key=key)
  if meta['ContentLength']!=artifact['size']or not manifest['start']<=meta['LastModified'].timestamp()<manifest['deadline']:raise InvalidCommittedArtifact('selected original proof size/window')
  frozen=artifact['frozen_key'];self.controller.bucket.copy(key,frozen,expected_etag=meta['ETag'])
  response=self.controller.bucket.client.get_object(Bucket=self.controller.bucket.name,Key=frozen);body=response['Body'];h=hashlib.sha256();size=0
  try:
   while True:
    block=body.read(8*1024**2)
    if not block:break
    size+=len(block);h.update(block)
    if size>artifact['size']:raise InvalidCommittedArtifact('copied proof exceeds committed size')
  finally:body.close()
  if size!=artifact['size']or h.hexdigest()!=row['proof_sha256']:raise InvalidCommittedArtifact('copied proof full SHA differs from signed commitment')
  captured=dict(artifact,etag=meta['ETag'],received_at=meta['LastModified'].timestamp(),read_url=self.controller.bucket.presign(frozen));return manifest,receipt,captured
 def tick(self,now=None):
  now=time.time()if now is None else now;available=self.max_inflight
  for jobid in self.state['jobs']:
   status=self.queue.status(jobid)
   if status and status['status']in('pending','leased'):available-=1
  if available<=0:return dict(enqueued=0,backpressure=True)
  rows=self.records();done=set(self.state['draws']);queued={r['row_sha256']for r in self.state['jobs'].values()};remaining=min(self.budget,available)
  retry=[d['row']for identity,d in self.state['draws'].items()if identity not in queued and self.state['capture_failures'].get(identity,{}).get('kind')!='confirmed_invalid_artifact'][:remaining]
  seed=secrets.token_hex(32);selected=random_selection(rows,seed,remaining-len(retry),done);enqueued=0
  # Persist selection before any mutable proof HEAD. Retries use the SAME draw.
  for row in selected:self.state['draws'][digest(row)]=dict(row=row,seed=seed,selected_at=now)
  self.persist()
  for row in retry+selected:
   identity=digest(row);seed=self.state['draws'][identity]['seed'];p=authenticate(self.state['populations'][row['epoch']],self.controller.authority.id)
   try:manifest,receipt,artifact=self._capture(row,p)
   except Exception as error:
    from botocore.exceptions import ClientError
    missing=isinstance(error,ClientError)and str(error.response.get('Error',{}).get('Code'))in('NoSuchKey','NotFound','404')
    invalid=isinstance(error,InvalidCommittedArtifact)or missing
    failure=dict(version='continuous-artifact-capture-failure-v1',row=row,outcome='confirmed_invalid'if invalid else'infrastructure_error',completed_at=time.time(),reason=type(error).__name__,original_selection_sha256=digest(self.state['draws'][identity]),scientific_model_execution_claim=False)
    document=self.controller.signed(failure);atomic(self.directory/(identity+'-capture-failure.json'),document)
    self.state['capture_failures'][identity]=dict(kind='confirmed_invalid_artifact'if invalid else'infrastructure_error',document=document,selected_row_sha256=identity);self.persist();continue
   audit_manifest=dict(manifest,audit_seed=seed,audit_frozen_receipts={row['miner']:dict(receipt,artifacts=[dict(b,**({k:artifact[k]for k in ('etag','received_at','read_url')}if b['slot']==artifact['slot']else{}))for b in receipt['artifacts']])})
   audit_manifest['audit_policy']=dict(manifest['audit_policy'],submission_counts={row['proof_sha256']:1})
   if manifest.get('proof_copy_policy')is not None:audit_manifest['proof_copy_receipts']={row['miner']:{str(artifact['slot']):{k:artifact[k]for k in ('sha256','size','etag','key','frozen_key','read_url')}}}
   ref=dict(miner=row['miner'],commitment_sha256=row['commitment_sha256'],**{k:artifact[k]for k in ('slot','env_id','index','batch_sha256','size','frozen_key')})
   metadata=self.metadata[manifest['source_bundle']['sha256']];jobid='continuous-audit-'+identity[:32]
   job=dict(schema=1,job_id=jobid,role='verify',created_at=now,expires_at=now+self.job_seconds,manifest=self.controller.signed(audit_manifest),**metadata,submissions=[dict(url=artifact['read_url'],sha256=row['proof_sha256'],commitment_miner=row['miner'],commitment_ref=ref)])
   envelope=self.controller.signed(job);atomic(self.directory/(jobid+'-job.json'),envelope);self.queue.enqueue(envelope);self.queue.archive(jobid,self.controller.bucket,'public/continuous-audit/jobs');self.state['jobs'][jobid]=dict(row_sha256=identity,job_sha256=digest(job));self.persist();enqueued+=1
  return dict(enqueued=enqueued,selected=len(selected),retried=len(retry),backpressure=False)
 def hourly_snapshot(self,epoch,round,checkpoint,cutoff):
  target=self.directory/('snapshot-'+str(cutoff)+'-'+epoch+'.json')
  if target.exists():return json.loads(target.read_text())
  queued=[]
  for jobid in self.state['jobs']:
   row=self.queue.status(jobid)
   if row and row['status']=='complete' and self.state['draws'][self.state['jobs'][jobid]['row_sha256']]['row']['round']<=round and self.state['draws'][self.state['jobs'][jobid]['row_sha256']]['row']['committed_at']<=cutoff:
    # status() intentionally exposes no execution report_request. Original row
    # is read transactionally from this same authenticated queue below.
    with self.queue.transaction()as db:actual=db.execute('select * from jobs where id=?',(jobid,)).fetchone()
    queued.append(dict(actual))
  records=[r for r in self.records() if r['round']<=round and r['committed_at']<=cutoff];admissions=admit_queue_reports(queued,records,self.controller.authority.id,self.queue.workers,self.sources)
  from .continuous_audit_policy import admit_artifact_failures
  failures=[f['document']for f in self.state['capture_failures'].values()if f['kind']=='confirmed_invalid_artifact' and authenticate(f['document'],self.controller.authority.id)['row']in records];admissions.update(admit_artifact_failures(failures,records,self.controller.authority.id))
  verifiers={**self.queue.workers,self.controller.authority.id:['operator-artifact-capture']}
  pointers=[dict(admitted_queue_job_sha256=key)for key in admissions];result=snapshot(records,pointers,verifiers,epoch=epoch,round=round,checkpoint=checkpoint,cutoff=cutoff,audit_policy=self.policy,admitted_jobs=admissions)
  document=self.controller.signed(result);atomic(target,document);self.controller.bucket.json('public/continuous-audit/snapshots/'+str(cutoff)+'-'+epoch+'.json',document);return document


def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');a=parser.parse_args(argv)
 from .storage import Bucket
 from .controller import Controller
 from .distributed_roles import Coordinator
 config=json.loads(Path(a.config).read_text());state=Path(config['state']);seed=state/'authority.seed'
 if seed.is_symlink()or not seed.is_file()or seed.stat().st_mode&0o077:raise ValueError('original private authority required; never generate another authority')
 controller=Controller(Bucket(config['bucket']),None,state);c=config['continuous_audit_service'];queue=Coordinator(state/'roles/verifier-queue.sqlite3',controller.authority.id,{e['worker_identity']:['verify']for e in config['remote']['roles']['verify']})
 sources=authenticate(c['source_admission'],controller.authority.id)
 if sources.get('version')!='continuous-audit-service-sources-v1':raise ValueError('operator admitted exact audit source/runtime metadata')
 service=ContinuousAuditor(controller,queue,directory=state/'continuous-audit',approved_sources=sources['approved_sources'],job_metadata=sources['job_metadata'],audit_policy=c['policy'],max_inflight=c.get('max_inflight',8),budget_per_tick=c.get('budget_per_tick',8),job_seconds=c.get('job_seconds',900))
 while True:
  for path in sorted(state.glob('*-continuous-audit-population.json')):service.admit(json.loads(path.read_text()))
  result=service.tick();atomic(state/'continuous-audit-health.json',dict(at=time.time(),**result))
  cutoff=int(time.time()//3600)*3600
  for path in sorted(state.glob('*-learner-completion.json')):
   completed=json.loads(path.read_text())
   if completed['completed_at']<=cutoff and completed['epoch']in service.state['populations']:service.hourly_snapshot(completed['epoch'],completed['round'],completed['inputcheckpoint'],cutoff)
  if a.once:return 0
  time.sleep(c.get('poll_seconds',10))

if __name__=='__main__':raise SystemExit(main())
