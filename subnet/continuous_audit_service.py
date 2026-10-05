"""Independent continuous audit scheduling and immutable hourly score snapshots.

Learner progress never waits for this service. It joins the original SQLite
queue without binding another HTTP listener. No blockchain writes occur here.
"""
import argparse,hashlib,json,os,secrets,sqlite3,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .continuous_audit_policy import VERSION,policy,population,digest,random_selection,verifier_contract,admit_queue_reports,snapshot
from .distributed_roles import authenticate
from .storage import canonical


class InvalidCommittedArtifact(ValueError):pass

def admitted_service_config(config,authority):
 sources=authenticate(config['source_admission'],authority)
 if sources.get('version')!='continuous-audit-service-sources-v1':raise ValueError('operator admitted exact audit source/runtime metadata')
 workers=config.get('capture_workers',1)
 if workers not in (1,4,8)or type(workers)is not int or sources.get('capture_workers',1)!=workers:raise ValueError('exact signed bounded capture concurrency')
 expected=policy(config['policy'])
 if canonical(sources.get('audit_policy'))!=canonical(expected):raise ValueError('continuous penalty policy requires exact signed admission')
 return sources

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 if path.is_symlink():raise ValueError('private audit journal path')
 temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with temporary.open('wb')as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
 temporary.chmod(0o600);temporary.replace(path)


def register_population(manifest_document,receipts,round,committed_at,authority,eligible_pairs=None,*,version='continuous-audit-population-v2'):
 """Return a signed-service input proposal from original immutable captures."""
 from .commitment_transport import validate
 if version not in ('continuous-audit-population-v1','continuous-audit-population-v2'):raise ValueError('explicit population ordering version')
 manifest=authenticate(manifest_document,authority);rows=[]
 for miner,receipt in sorted(receipts.items()):
  document=validate(canonical(receipt['commitment_document']),manifest['epoch'],miner,manifest['max_batches'])
  if digest(document)!=receipt['sha256']or document['payload']['checkpoint']!=manifest['checkpoint']['id']or document['payload']['source']!=manifest['source_bundle']['sha256']:raise ValueError('original signed committed population')
  for b in document['payload']['batches']:
   rows.append(dict(epoch=manifest['epoch'],round=round,checkpoint=manifest['checkpoint']['id'],miner=miner,env_id=b['env_id'],index=b['index'],batch_sha256=b['batch_sha256'],proof_sha256=b['sha256'],commitment_sha256=receipt['sha256'],verifier_contract_sha256=verifier_contract(manifest),committed_at=committed_at))
 result=dict(version=version,round=round,committed_at=committed_at,manifest_document=manifest_document,receipts=receipts,records=population(rows,ordered=version=='continuous-audit-population-v2'))
 if eligible_pairs is not None:
  if type(eligible_pairs)is not list or any(type(p)is not dict or set(p)!={'miner','commitment_sha256','batch_sha256','proof_sha256'}for p in eligible_pairs):raise ValueError('exact actual eligible immutable pairs')
  ids=[]
  for pair in eligible_pairs:
   matching=[r for r in rows if all(r[k]==v for k,v in pair.items())]
   if len(matching)!=1:raise ValueError('eligible pair original committed population')
   ids.append(digest(matching[0]))
  if len(ids)!=len(set(ids)):raise ValueError('duplicate eligible pair')
  result['eligible_evidence_ids']=sorted(ids)
 return result


class ContinuousAuditor:
 def __init__(self,controller,queue,*,directory,approved_sources,job_metadata,audit_policy,max_inflight=8,budget_per_tick=8,job_seconds=900,capture_workers=1,execution_evidence_policy=None):
  if type(capture_workers)is not int or capture_workers not in (1,4,8):raise ValueError('bounded capture workers')
  self.capture_workers=capture_workers;self.execution_evidence_policy=execution_evidence_policy
  self.controller=controller;self.queue=queue;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True,mode=0o700);self.sources=approved_sources;self.metadata=job_metadata;self.policy=policy(audit_policy)
  if type(max_inflight)is not int or not 1<=max_inflight<=128 or type(budget_per_tick)is not int or not 1<=budget_per_tick<=128 or type(job_seconds)is not int or not 60<=job_seconds<=86400:raise ValueError('bounded continuous audit scheduler')
  self.max_inflight=max_inflight;self.budget=budget_per_tick;self.job_seconds=job_seconds
  self.state_path=self.directory/'audit-state.json';self.state=json.loads(self.state_path.read_text())if self.state_path.exists()else dict(populations={},draws={},jobs={},capture_failures={})
 def persist(self):atomic(self.state_path,self.state)
 def admit(self,document):
  p=authenticate(document,self.controller.authority.id)
  if p.get('version')not in('continuous-audit-population-v1','continuous-audit-population-v2'):raise ValueError('continuous immutable population admission')
  if 'eligible_evidence_ids'not in p:raise ValueError('explicit learner eligibility required for audit reward population')
  selected=[r for r in p['records']if digest(r)in p['eligible_evidence_ids']]
  pairs=[{k:r[k]for k in ('miner','commitment_sha256','batch_sha256','proof_sha256')}for r in selected]
  expected=register_population(p['manifest_document'],p['receipts'],p['round'],p['committed_at'],self.controller.authority.id,eligible_pairs=pairs,version=p['version'])
  if p!=expected:raise ValueError('canonical signed population registration')
  epoch=authenticate(p['manifest_document'],self.controller.authority.id)['epoch'];old=self.state['populations'].get(epoch)
  if old is not None and old!=document:raise ValueError('immutable audit population cannot be replaced')
  self.state['populations'][epoch]=document;self.persist()
 def records(self):return [r for document in self.state['populations'].values()for r in authenticate(document,self.controller.authority.id)['records']]
 def dispatch_records(self):
  rows=[];deferred=0
  for document in self.state['populations'].values():
   p=authenticate(document,self.controller.authority.id);manifest=authenticate(p['manifest_document'],self.controller.authority.id);source=manifest['source_bundle']['sha256']
   if source not in self.sources or source not in self.metadata:deferred+=len(p['records']);continue
   rows.extend(p['records'])
  return rows,deferred
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
   if status and status['status']in('queued','leased'):available-=1
  if available<=0:return dict(enqueued=0,backpressure=True)
  rows,deferred=self.dispatch_records();runnable=set(digest(r)for r in rows);done=set(self.state['draws']);queued={r['row_sha256']for r in self.state['jobs'].values()};remaining=min(self.budget,available)
  retry=[d['row']for identity,d in self.state['draws'].items()if identity in runnable and identity not in queued and self.state['capture_failures'].get(identity,{}).get('kind')!='confirmed_invalid_artifact'][:remaining]
  seed=secrets.token_hex(32);selected=random_selection(rows,seed,remaining-len(retry),done);enqueued=0
  # Persist selection before any mutable proof HEAD. Retries use the SAME draw.
  for row in selected:self.state['draws'][digest(row)]=dict(row=row,seed=seed,selected_at=now)
  self.persist()
  candidates=retry+selected
  # Workers perform storage-only capture. All authority signing and journal/queue
  # changes remain on this owner thread, and selection was already persisted.
  def captured_rows():
   fresh=[]
   for row in candidates:
    identity=digest(row);original=self.directory/('continuous-audit-'+identity[:32]+'-job.json')
    if original.exists():yield row,None;continue
    p=authenticate(self.state['populations'][row['epoch']],self.controller.authority.id);fresh.append((row,p))
   with ThreadPoolExecutor(max_workers=self.capture_workers)as pool:
    futures={pool.submit(self._capture,row,p):row for row,p in fresh}
    for future in as_completed(futures):yield futures[future],future
  for row,future in captured_rows():
   identity=digest(row);seed=self.state['draws'][identity]['seed'];p=authenticate(self.state['populations'][row['epoch']],self.controller.authority.id)
   jobid='continuous-audit-'+identity[:32];original=self.directory/(jobid+'-job.json')
   if original.exists():
    envelope=json.loads(original.read_text());job=authenticate(envelope,self.controller.authority.id)
    if job['job_id']!=jobid:raise ValueError('original selected request identity')
    self.queue.enqueue(envelope);self.queue.archive(jobid,self.controller.bucket,'public/continuous-audit/jobs');self.state['jobs'][jobid]=dict(row_sha256=identity,job_sha256=digest(job));self.persist();enqueued+=1;continue
   try:manifest,receipt,artifact=future.result()
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
   created_at=time.time()
   job=dict(schema=1,job_id=jobid,role='verify',created_at=created_at,expires_at=created_at+self.job_seconds,manifest=self.controller.signed(audit_manifest),**metadata,submissions=[dict(url=artifact['read_url'],sha256=row['proof_sha256'],commitment_miner=row['miner'],commitment_ref=ref)])
   envelope=self.controller.signed(job);atomic(self.directory/(jobid+'-job.json'),envelope);self.queue.enqueue(envelope);self.queue.archive(jobid,self.controller.bucket,'public/continuous-audit/jobs');self.state['jobs'][jobid]=dict(row_sha256=identity,job_sha256=digest(job));self.persist();enqueued+=1
  return dict(enqueued=enqueued,selected=len(selected),retried=len(retry),backpressure=False,source_deferred=deferred)
 def publish_immutable(self,key,document):
  body=canonical(document)
  try:self.controller.bucket.client.put_object(Bucket=self.controller.bucket.name,Key=key,Body=body,ContentType='application/json',IfNoneMatch='*')
  except Exception as error:
   code=getattr(error,'response',{}).get('Error',{}).get('Code')
   if code not in ('PreconditionFailed','412'):raise
   if self.controller.bucket.get(key)!=body:raise ValueError('immutable hourly publication collision')from None
 def hourly_completed(self,completed,cutoff):
  from .continuous_audit_policy import hourly_aggregate
  target=self.directory/('hourly-weights-'+str(cutoff)+'.json')
  if target.exists():
   document=json.loads(target.read_text());self.publish_immutable('public/continuous-audit/hourly/'+str(cutoff)+'.json',document);return document
  selected=[c for c in completed if cutoff-3600<c['completed_at']<=cutoff and c['epoch']in self.state['populations']]
  # Empty current hour is a real zero-weight snapshot, not last-hour reuse.
  documents=[self.hourly_snapshot(c['epoch'],c['round'],c.get('inputcheckpoint',c.get('checkpoint')),cutoff)for c in sorted(selected,key=lambda c:(c['completed_at'],c['epoch']))]
  document=self.controller.signed(hourly_aggregate(documents,self.controller.authority.id,cutoff));atomic(target,document);self.publish_immutable('public/continuous-audit/hourly/'+str(cutoff)+'.json',document);return document
 def reconcile_hours(self,completed,current_cutoff):
  from .continuous_audit_policy import finite
  if type(current_cutoff)is not int or current_cutoff<0 or current_cutoff%3600:raise ValueError('real whole UTC cutoff')
  cutoffs={current_cutoff}
  for c in completed:
   at=finite(c['completed_at'],0,2**53,'original learner completion timestamp')
   if c['epoch']not in self.state['populations']:continue
   cutoff=int(__import__('math').ceil(at/3600))*3600
   if cutoff<=current_cutoff:cutoffs.add(cutoff)
  published=self.state.setdefault('published_hours',{});count=0
  for cutoff in sorted(cutoffs):
   target=self.directory/('hourly-weights-'+str(cutoff)+'.json');known=published.get(str(cutoff))
   if known is not None:
    if not target.is_file()or digest(json.loads(target.read_text()))!=known:raise ValueError('original published hourly journal mismatch')
    continue
   document=self.hourly_completed(completed,cutoff)
   # Set only AFTER exact conditional publication/readback succeeds. A crash
   # after local atomic write is retried with the saved immutable document.
   published[str(cutoff)]=digest(document);self.persist();count+=1
  return dict(completed_hours=count,current_cutoff=current_cutoff)
 def hourly_snapshot(self,epoch,round,checkpoint,cutoff):
  target=self.directory/('snapshot-'+str(cutoff)+'-'+epoch+'.json')
  if target.exists():
   document=json.loads(target.read_text());self.publish_immutable('public/continuous-audit/snapshots/'+str(cutoff)+'-'+epoch+'.json',document);return document
  queued=[]
  for jobid in self.state['jobs']:
   row=self.queue.status(jobid)
   if row and row['status']=='complete' and self.state['draws'][self.state['jobs'][jobid]['row_sha256']]['row']['round']<=round and self.state['draws'][self.state['jobs'][jobid]['row_sha256']]['row']['committed_at']<=cutoff:
    # status() intentionally exposes no execution report_request. Original row
    # is read transactionally from this same authenticated queue below.
    with self.queue.transaction()as db:actual=db.execute('select * from jobs where id=?',(jobid,)).fetchone()
    queued.append(dict(actual))
  records=[r for r in self.records() if r['round']<=round and r['committed_at']<=cutoff];admissions=admit_queue_reports(queued,records,self.controller.authority.id,self.queue.workers,self.sources,execution_evidence_policy=self.execution_evidence_policy if self.execution_evidence_policy is not None and cutoff>=self.execution_evidence_policy['effective_cutoff']else None,cutoff=cutoff)
  from .continuous_audit_policy import admit_artifact_failures
  failures=[f['document']for f in self.state['capture_failures'].values()if f['kind']=='confirmed_invalid_artifact' and authenticate(f['document'],self.controller.authority.id)['row']in records];admissions.update(admit_artifact_failures(failures,records,self.controller.authority.id))
  verifiers={**self.queue.workers,self.controller.authority.id:['operator-artifact-capture']}
  pointers=[dict(admitted_queue_job_sha256=key)for key in admissions];result=snapshot(records,pointers,verifiers,epoch=epoch,round=round,checkpoint=checkpoint,cutoff=cutoff,audit_policy=self.policy,admitted_jobs=admissions,eligible_evidence_ids=authenticate(self.state['populations'][epoch],self.controller.authority.id)['eligible_evidence_ids'],adjudications=[json.loads(path.read_text())for path in sorted(self.directory.glob('*-adjudication.json'))],authority=self.controller.authority.id)
  if self.execution_evidence_policy is not None and cutoff>=self.execution_evidence_policy['effective_cutoff']:
   result['execution_evidence_policy_sha256']=digest(self.execution_evidence_policy);result['execution_evidence_policy_version']=self.execution_evidence_policy['version'];result['os_resource_enforcement_claimed']=False;result['historical_execution_proven']=False
  document=self.controller.signed(result);atomic(target,document);self.publish_immutable('public/continuous-audit/snapshots/'+str(cutoff)+'-'+epoch+'.json',document);return document


def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');a=parser.parse_args(argv)
 from .storage import Bucket
 from .controller import Controller
 from .distributed_roles import Coordinator
 config=json.loads(Path(a.config).read_text());state=Path(config['state']);seed=state/'authority.seed'
 if seed.is_symlink()or not seed.is_file()or seed.stat().st_mode&0o077:raise ValueError('original private authority required; never generate another authority')
 controller=Controller(Bucket(config['bucket']),None,state);c=config['continuous_audit_service'];queue=Coordinator(state/'roles/verifier-queue.sqlite3',controller.authority.id,{e['worker_identity']:['verify']for e in config['remote']['roles']['verify']})
 sources=admitted_service_config(c,controller.authority.id)
 service=ContinuousAuditor(controller,queue,directory=state/'continuous-audit',approved_sources=sources['approved_sources'],job_metadata=sources['job_metadata'],audit_policy=c['policy'],max_inflight=c.get('max_inflight',8),budget_per_tick=c.get('budget_per_tick',8),job_seconds=c.get('job_seconds',900),capture_workers=c.get('capture_workers',1),execution_evidence_policy=sources.get('execution_evidence_policy'))
 while True:
  for path in sorted(state.glob('*-continuous-audit-population.json')):service.admit(json.loads(path.read_text()))
  result=service.tick();atomic(state/'continuous-audit-health.json',dict(at=time.time(),**result))
  cutoff=int(time.time()//3600)*3600
  service.reconcile_hours([json.loads(path.read_text())for path in sorted(state.glob('*-learner-completion.json'))],cutoff)
  if a.once:return 0
  time.sleep(c.get('poll_seconds',10))

if __name__=='__main__':raise SystemExit(main())
