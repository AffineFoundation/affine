"""Local prospective one-writer runner. Dry-run default; never starts compute roles."""
import argparse,ast,fcntl,hashlib,json,os,sqlite3,subprocess,time,stat,signal,math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.live_reward_bridge import signed,need,canonical,sha
from subnet.chain import ChainAdapter,OWNER
from subnet.distributed_roles import Coordinator,authenticate
from subnet.backend_jobs import _validate
from subnet.remote_backend import RemoteJobs
from subnet.source_bootstrap import admitted_files
from ops import live_reward_exporter as exporter,live_reward_submit as submit
from ops.verifier_workforce import authenticate_supplements,authorize_worker
from ops.live_reward_source_approval import apply_source_approvals,source_verifiers
UNITS=tuple(n+'.'+s for n in ('affine-transition-weights','affine-hourly-burn') for s in ('timer','service'))

class FinalizationPending(ValueError):
 """An active expired epoch has not produced either score artifact yet."""
 def __init__(self,epoch):
  self.epoch=epoch
  super().__init__('old-hour active finalize lacks original scores')

def read(path):return json.loads(Path(path).read_text())
def file_hash(path):
 p=Path(path);need(p.is_file() and not p.is_symlink(),'regular reviewed file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def process_identity():
 pid=os.getpid();fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 need(fields[0] in ('R','S','D','I'),'live writer process')
 return dict(writer_pid=pid,writer_start_ticks=fields[19],writer_process_state=fields[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def observe_units(run=subprocess.run):
 result=[]
 for unit in UNITS:
  r=run(['systemctl','--user','show',unit,'--property=LoadState,ActiveState,UnitFileState','--no-pager'],capture_output=True,text=True,timeout=15)
  need(r.returncode==0,'old writer status query failed')
  fields=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
  need(fields.get('LoadState')=='loaded' and fields.get('ActiveState')=='inactive' and fields.get('UnitFileState') in (('disabled','masked','static') if unit.endswith('.service') else ('disabled','masked')),'old writer must be disabled and inactive')
  result.append(dict(unit=unit,running=False,enabled=False,status_query_succeeded=True))
 return result

def authenticate_cutover(document,anchor_document,authority):
 c=signed(document,authority);a=signed(anchor_document,authority)
 need(c.get('version')=='live-single-writer-runtime-v1' and c.get('netuid')==120 and c.get('owner_hotkey')==OWNER,'exact owner/netuid cutover')
 need(c['anchor_sha256']==sha(anchor_document) and a.get('owner_hotkey')==OWNER and a.get('netuid')==120,'cutover anchor binding')
 need(a.get('compute_epoch_prefix')=='nonpayable-live-reward-math-v1-','prospective compute prefix')
 for key in ('compute_state','reward_state','chain_state','queue_database','authority_seed_file'):
  path=Path(c[key]);need(path.is_absolute() and path.resolve()==path,'absolute canonical operator path')
 expected=Path('/run/user')/str(os.getuid())/('affine-live-reward-120-'+hashlib.sha256(OWNER.encode()).hexdigest()+'.lock')
 need(c['global_lock_path']==str(expected),'canonical global owner/netuid lock')
 need(c['chain_state']==c['reward_state'],'one writer chain/reward state')
 need(set(c.get('runtime_versions',{}))=={'torch','transformers','toploc'},'exact original job three-package runtime pins')
 need(c.get('stale_registration_policy','deny-hour') in ('deny-hour','exclude-ineligible-v1'),'approved stale registration policy')
 first=c.get('stale_registration_policy_first_window',0)
 need(type(first)is int and first>=0 and first%3600==0,'integral first eligibility window')
 return c,a
@contextmanager
def global_lock(path):
 p=Path(path);need(p.parent.is_dir() and not p.is_symlink(),'existing trusted runtime lock directory')
 fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'a') as f:
  actual=os.fstat(f.fileno());need(stat.S_ISREG(actual.st_mode) and actual.st_uid==os.getuid() and actual.st_mode&0o077==0,'private owned global lock file')
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def guard_files(c):
 for row in c['legacy_guard_files']:
  need(file_hash(row['path'])==row['sha256'],'reviewed legacy guard/marker bytes changed')
 need(len(c['legacy_guard_files'])==2 and {row['kind'] for row in c['legacy_guard_files']}=={'validator-hook','suppression-marker'},'actual hook and marker required')
 marker=Path.home()/'.local/state/affine-transition/active'
 need(any(row['kind']=='suppression-marker' and Path(row['path'])==marker for row in c['legacy_guard_files']),'actual production suppression marker path')

def verify_audit_lineage(state,manifest,audit,authority,c,db,now,files,*,required_source_files=None):
 """Read actual worker-authenticated queue row and original signed request."""
 jobid=audit['remote_job_id'];need(isinstance(jobid,str) and all(x.isalnum() or x in '-_' for x in jobid),'job path identity')
 envelope=read(state/'roles'/(jobid+'-job.json'));job=signed(envelope,authority)
 need(job['role']=='verify' and signed(job['manifest'],authority)['epoch']==manifest['epoch'],'original verifier job epoch')
 # Validate at original creation, preserving original signed expiry. No source-dependent live imports.
 _validate(envelope,authority,now=job['created_at'],resolve_source=False,required_source_files=required_source_files)
 need(job['manifest']['payload']['checkpoint']==manifest['checkpoint'],'exact checkpoint descriptor')
 jm=job['manifest']['payload']
 need(jm['source_bundle']['sha256']==manifest['source_bundle']['sha256'],'original approved source')
 # Postfreeze allocation fields are permitted; all opening scientific/contract fields remain exact.
 opening={k:v for k,v in jm.items() if k not in ('audit_seed','audit_frozen_receipts')};opening['audit_policy']=dict(opening['audit_policy']);opening['audit_policy'].pop('submission_counts',None)
 need(canonical(opening)==canonical(manifest),'opening versus audit manifest')
 need(job['runtime_versions']==c['runtime_versions'],'approved runtime versions')
 need(all(files.get(name)==digest for name,digest in job['source_files'].items()),'actual approved archive module pins')
 row=db.execute('SELECT * FROM jobs WHERE id=?',(jobid,)).fetchone();need(row is not None and row['status']=='complete','actual queue completion')
 need(canonical(json.loads(row['envelope']))==canonical(envelope),'original queue signed request')
 worker=row['worker'];workforce=authorize_worker(worker,manifest,job,row,db,source_verifiers(c,manifest,job),c.get('_verifier_workforce',{}))
 request=authenticate(json.loads(row['report_request']),worker);remote=json.loads(row['report'])
 need(request['action']=='report' and request['job_id']==jobid and request['report']==remote,'actual authenticated worker report')
 need(hashlib.sha256(canonical(remote)).hexdigest()==row['report_digest'],'queue report bytes')
 checker=Coordinator.__new__(Coordinator);checker.authority=authority;checker.clock=lambda:now
 checker.validate_report(remote,job,jm,hashlib.sha256(canonical(job)).hexdigest())
 prior=dict(job_id=jobid,job_sha256=hashlib.sha256(canonical(job)).hexdigest(),role='verify',source_files=job['source_files'],runtime_versions=job['runtime_versions'],manifest_sha256=sha(jm))
 reader=RemoteJobs.__new__(RemoteJobs);reader.state=state/'roles';reader.controller=SimpleNamespace(authority=SimpleNamespace(id=authority))
 reader.checked(remote,prior,jm)
 if manifest.get('submission_transport_policy'):
  from subnet.commitment_transport import combine
  receipt=jm['audit_frozen_receipts'][audit['miner_identity']]if 'miner_identity'in audit else next(r for r in jm['audit_frozen_receipts'].values()if r['sha256']==audit['submission_sha256'])
  bound=combine(jm,receipt,remote)
 else:
  matches=[r for r in remote['audits'] if r['submission_sha256']==audit['submission_sha256']];need(len(matches)==1,'exact frozen report')
  bound=dict(matches[0],remote_job_id=remote['job_id'],backend_profile=remote['backend_profile'],execution_resources_enforced=remote['execution_resources_enforced'])
 need(canonical(bound)==canonical(audit),'original audit metadata, not caller score')
 return dict(job_id=jobid,job_sha256=sha(job),report_sha256=sha(remote),worker=worker,submission_sha256=audit['submission_sha256'],**(workforce or {}))

def approved_source_members(c,authority):
 """Authenticate every operator-approved archive, retaining older epoch pins."""
 primary=c['source'];sources=c.get('approved_sources',{primary['sha256']:primary})
 need(isinstance(sources,dict) and sources and sources.get(primary['sha256'])==primary,'primary approved source binding')
 inventories={}
 for digest,source in sources.items():
  need(isinstance(digest,str) and len(digest)==64 and all(x in '0123456789abcdef' for x in digest) and source.get('sha256')==digest,'approved source registry key')
  for field in ('archive_path','descriptor_path'):
   path=Path(source[field]);need(path.is_absolute() and path.resolve()==path,'canonical approved source path')
  need(file_hash(source['archive_path'])==digest,'approved source archive bytes')
  descriptor=signed(read(source['descriptor_path']),authority)
  need(descriptor['sha256']==digest and descriptor['size']==Path(source['archive_path']).stat().st_size,'signed source descriptor')
  members=admitted_files(Path(source['archive_path']).read_bytes(),descriptor)
  inventories[digest]={name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
 return inventories

def original_required_source_files(module):
 """Read the authenticated archive's pin declaration without executing it.

 Only a literal sequence or our exact tuple/path generator is accepted. The
 archive is authority-approved; a miner's job never selects this requirement.
 """
 tree=ast.parse(module);assignments=[]
 writes=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='SOURCE_FILES' and isinstance(n.ctx,(ast.Store,ast.Del))]
 need(len(writes)==1,'unambiguous original source pin writes')
 for node in ast.walk(tree):
  if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
   targets=node.targets if isinstance(node,ast.Assign) else [node.target]
   if any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in targets):assignments.append(node)
 need(len(assignments)==1 and assignments[0] in tree.body and isinstance(assignments[0],ast.Assign) and len(assignments[0].targets)==1,'unambiguous original source pin declaration')
 value=assignments[0].value
 try:required=ast.literal_eval(value)
 except (ValueError,TypeError):
  # SOURCE_FILES = tuple('subnet/'+n+'.py' for n in ('model', ...)).
  need(isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='tuple' and len(value.args)==1 and not value.keywords,'original source pin expression')
  generator=value.args[0];need(isinstance(generator,ast.GeneratorExp) and len(generator.generators)==1,'original source pin generator')
  clause=generator.generators[0];need(isinstance(clause.target,ast.Name) and not clause.ifs and not clause.is_async,'original source pin iterator')
  expected=ast.parse("'subnet/'+"+clause.target.id+"+'.py'",mode='eval').body
  need(ast.dump(generator.elt)==ast.dump(expected),'original source pin path expression')
  names=ast.literal_eval(clause.iter);need(isinstance(names,(tuple,list)) and all(isinstance(n,str) and n.isidentifier() for n in names),'original source module names')
  required=tuple('subnet/'+n+'.py' for n in names)
 need(isinstance(required,(tuple,list)) and required and len(required)==len(set(required)) and all(isinstance(n,str) and n.startswith('subnet/') and '..' not in n and n.endswith('.py') for n in required),'original source pin paths')
 return tuple(required)

def verify_completed_evidence(c,authority,now):
 state=Path(c['compute_state']);proofs=[]
 # Authenticate archive bytes once per invocation. A source upgrade must never
 # replace the archive or module inventory used by an older finalized epoch.
 inventories=approved_source_members(c,authority)
 primary=c['source'];sources=c.get('approved_sources',{primary['sha256']:primary});requirements={}
 for digest,source in sources.items():
  descriptor=signed(read(source['descriptor_path']),authority)
  body=Path(source['archive_path']).read_bytes();need(hashlib.sha256(body).hexdigest()==digest,'approved source archive bytes')
  members=admitted_files(body,descriptor)
  need({name:hashlib.sha256(data).hexdigest() for name,data in members.items()}==inventories[digest],'original source inventory changed')
  requirements[digest]=original_required_source_files(members['subnet/backend_jobs.py'])
  need(set(requirements[digest])<=set(inventories[digest]),'original required modules absent from archive')
 uri=Path(c['queue_database']).as_uri()+'?mode=ro'
 with sqlite3.connect(uri,uri=True) as db:
  db.row_factory=sqlite3.Row
  for first in sorted(state.glob('*-first-signed-manifest.json')):
   m=signed(read(first),authority);epoch=m['epoch'];scores=state/(epoch+'-signed-compute-scores.json')
   if not scores.exists():continue
   digest=m['source_bundle']['sha256'];need(digest in inventories,'approved live source only');files=inventories[digest]
   exclusion=m.get('audit_exclusion_snapshot')
   if exclusion is not None:
    from subnet.audit_exclusion import snapshot
    history=signed(exclusion['history'],authority)
    need(snapshot(history,exclusion['policy'])==exclusion['excluded_miners'],'exact confirmed-invalid temporary exclusion')
    for event in history['epochs']:
     for oldminer,oldreport in event['reports'].items():
      oldmanifest=signed(read(state/(event['epoch']+'-first-signed-manifest.json')),authority)
      actual=signed(read(state/(event['epoch']+'-signed-compute-audit-'+oldminer+'.json')),authority)
      need(actual==oldreport,'original historical confirmed-invalid audit')
      olddigest=oldmanifest['source_bundle']['sha256'];need(olddigest in inventories,'historical approved source')
      verify_audit_lineage(state,oldmanifest,actual,authority,c,db,now,inventories[olddigest],required_source_files=requirements[olddigest])
   score=signed(read(scores),authority)
   for miner in score['receipts']:
    audit=signed(read(state/(epoch+'-signed-compute-audit-'+miner+'.json')),authority)
    need(audit['submission_sha256']==score['receipts'][miner]['sha256'],'signed frozen receipt binding')
    if m.get('submission_transport_policy')and audit.get('commitment_status')in('budget_deferred','infrastructure_deferred'):
     from subnet.commitment_transport import validate_deferred
     audit_manifest=read(state/(epoch+'-audit-manifest.json'));challenge=read(state/(epoch+'-audit-challenge.json'))
     need(challenge['receipts']==score['receipts']and challenge['generated_after_freeze_at']>=m['deadline'],'deferred frozen population/challenge')
     validate_deferred(audit_manifest,score['receipts'][miner],audit)
     continue
    if m.get('submission_transport_policy')and audit.get('commitment_status')=='not_selected':
     from subnet.commitment_transport import validate_unchecked
     challenge=read(state/(epoch+'-audit-challenge.json'));plan=read(state/(epoch+'-audit-plan.json'));audit_manifest=read(state/(epoch+'-audit-manifest.json'))
     need(challenge['receipts']==score['receipts']and challenge['generated_after_freeze_at']>=m['deadline'],'original committed population/challenge')
     from subnet.audit_policy import allocate
     expected=allocate({k:(0 if k in m.get('audit_exclusion_snapshot',{}).get('excluded_miners',[])else len(v['artifacts']))for k,v in score['receipts'].items()},m['audit_policy'],challenge['seed'])
     need(expected==plan['allocations']and expected[miner]==0,'original zero-slot allocation')
     validate_unchecked(audit_manifest,score['receipts'][miner],audit)
     continue
    proofs.append(verify_audit_lineage(state,m,audit,authority,c,db,now,files,required_source_files=requirements[digest]))
 return proofs

def finalized_hour_watermark(state,prefix,window_end):
 """Observe actual controller state: unresolved old-window finalize blocks payout."""
 status=read(state/'controller.json');active=status.get('active')
 if active is None:return status
 need(isinstance(active,dict) and isinstance(active.get('epoch'),str) and active['epoch'].startswith(prefix),'actual prospective controller epoch')
 phase=active.get('phase')
 need(phase in ('opening','mine','collect','before','train','after'),'known actual controller phase')
 if phase in ('opening','mine','collect'):
  manifest_path=state/(active['epoch']+'-manifest.json')
  need(manifest_path.is_file(),'unresolved opening watermark')
  manifest=read(manifest_path);need(manifest['epoch']==active['epoch'],'controller watermark manifest epoch')
  deadline=manifest['deadline'];need(type(deadline) in (int,float) and math.isfinite(deadline) and deadline>=0,'controller watermark deadline')
  if deadline<=window_end:
   if not (state/(active['epoch']+'-scores.json')).exists() and not (state/(active['epoch']+'-signed-compute-scores.json')).exists():
    raise FinalizationPending(active['epoch'])
   need((state/(active['epoch']+'-scores.json')).is_file(),'old-hour active finalize lacks original scores')
   need((state/(active['epoch']+'-signed-compute-scores.json')).is_file(),'old-hour active finalize lacks signed scores')
 return status

def finalized_reward_completeness(c,anchor_document,authority,*,window_end):
 """Never close an hour while original finalized records lack signed evidence.

 Active epochs without original scores are ignored. A previously closed hour
 must already contain the exact immutable derived record, never import it late.
 """
 state=Path(c['compute_state']);reward=Path(c['reward_state'])
 anchor=signed(anchor_document,authority);prefix=anchor['compute_epoch_prefix']
 need(prefix=='nonpayable-live-reward-math-v1-','fresh compute evidence namespace')
 watermark=finalized_hour_watermark(state,prefix,window_end)
 ledgerpath=reward/'signed-reward-ledger.json';ledger=read(ledgerpath) if ledgerpath.exists() else []
 indexed={signed(doc,authority)['compute_epoch_id']:signed(doc,authority) for doc in ledger}
 ledger_documents={signed(doc,authority)['compute_epoch_id']:doc for doc in ledger}
 need(len(indexed)==len(ledger),'unique immutable compute reward records')
 cursor=read(reward/'writer-cursor.json') if (reward/'writer-cursor.json').exists() else {}
 closed=None
 if 'window_end' in cursor:
  end=cursor['window_end'];need(type(end)is int and end%3600==0,'completed hour cursor')
  closed=end if cursor.get('status') in ('submitted','already_submitted','zero_points_no_submission') else end-3600
 rawpaths={p.name[:-len('-scores.json')]:p for p in state.glob(prefix+'*-scores.json') if not p.name.endswith('-signed-compute-scores.json')}
 signedpaths={p.name[:-len('-signed-compute-scores.json')]:p for p in state.glob(prefix+'*-signed-compute-scores.json')}
 need(set(signedpaths)<=set(rawpaths),'signed scores need original raw finalization')
 observations=[]
 for epoch,path in sorted(rawpaths.items()):
  original=read(path);need(original.get('epoch_id')==epoch,'original finalized epoch path')
  needed=['first-signed-manifest','opening-attestation','signed-registrations','signed-compute-scores']
  need(all((state/(epoch+'-'+label+'.json')).is_file() for label in needed),'incomplete finalized reward sidecars')
  score=signed(read(state/(epoch+'-signed-compute-scores.json')),authority)
  need(canonical(score)==canonical(original),'signed versus original finalized score bytes')
  first=signed(read(state/(epoch+'-first-signed-manifest.json')),authority)
  need(canonical(first)==canonical(read(state/(epoch+'-manifest.json'))),'original final manifest binding')
  need(all((state/(epoch+'-signed-compute-audit-'+miner+'.json')).is_file() for miner in original['receipts']),'incomplete finalized audit sidecars')
  original_anchor=exporter.epoch_anchor(first,anchor_document,authority,c.get('approved_source_anchors'))
  expected=exporter.export_epoch(state,epoch,original_anchor,authority)
  if epoch in indexed:need(canonical(indexed[epoch])==canonical(expected),'immutable ledger versus original finalized evidence')
  if closed is not None and score['finalized_at']<closed:
   need(epoch in indexed,'late unexported reward for already closed hour')
   hour=(int(score['finalized_at'])//3600+1)*3600
   proposal_document=read(reward/('hour-'+str(hour)+'-reward-units.json'));proposal=signed(proposal_document,authority)
   need(proposal.get('version')=='live-reward-hour-units-v1' and proposal.get('window_end')==hour and sha(ledger_documents[epoch]) in proposal.get('source_reward_records',[]),'closed hour proposal omitted existing reward')
   if hour==cursor.get('window_end'):need(cursor.get('proposal_sha256')==sha(proposal_document),'closed cursor versus signed hour proposal')
  observations.append(dict(epoch=epoch,score_sha256=sha(original)))
 need(canonical(finalized_hour_watermark(state,prefix,window_end))==canonical(watermark),'controller watermark changed during evidence read')
 return observations

def choose_hour(state,anchor,now):
 cursor=read(state/'writer-cursor.json') if (state/'writer-cursor.json').exists() else {}
 if cursor.get('status')=='submitting':raise RuntimeError('uncertain chain outcome requires explicit root reconciliation')
 end=cursor.get('window_end')
 if end is not None and cursor.get('status') not in ('submitted','already_submitted','zero_points_no_submission'):return end
 first=(int(anchor['effective_at'])//3600+1)*3600
 end=max(first,(end+3600 if end is not None else first))
 return end if end<=int(now)//3600*3600 else None

def run_once(cutover_document,anchor_document,authority,*,execute=False,adapter_factory=ChainAdapter,verifier_workforce_supplements=None,compute_source_approvals=None):
 c,anchor=authenticate_cutover(cutover_document,anchor_document,authority);need(type(execute)is bool,'explicit execution flag')
 c,anchor_document=apply_source_approvals(c,anchor_document,authority,cutover_document,compute_source_approvals or [])
 anchor=signed(anchor_document,authority)
 if verifier_workforce_supplements:
  c['_verifier_workforce']=authenticate_supplements(verifier_workforce_supplements,authority,sha(cutover_document),c['verifier_identities'])
 with global_lock(c['global_lock_path']):
  guard_files(c);units=observe_units();identity=process_identity();now=time.time()
  seed=Path(c['authority_seed_file']);need(not seed.is_symlink() and seed.stat().st_mode&0o077==0,'private authority seed')
  key=SigningKey(bytes.fromhex(seed.read_text().strip()));need(key.verify_key.encode().hex()==authority,'operator signing authority')
  state=Path(c['reward_state']);state.mkdir(exist_ok=True,mode=0o700)
  end=choose_hour(state,anchor,now)
  if end is None:return {'status':'waiting_for_completed_live_hour','chain_executed':False}
  try:completeness=finalized_reward_completeness(c,anchor_document,authority,window_end=end)
  except FinalizationPending as pending:
   return dict(status='waiting_for_epoch_finalization',window_end=end,epoch=pending.epoch,chain_executed=False)
  evidence=verify_completed_evidence(c,authority,now)
  adapter=adapter_factory(state,netuid=120,expected_owner=OWNER)
  # Actual chain-derived identities before import/export, not a caller-supplied registry.
  registrations=adapter.registrations()
  # Recheck after potentially slow chain identity discovery, immediately before export.
  completeness=finalized_reward_completeness(c,anchor_document,authority,window_end=end)
  export_options={'source_anchors':c['approved_source_anchors']} if 'approved_source_anchors' in c else {}
  if 'stale_registration_policy' in c and end>=c.get('stale_registration_policy_first_window',0):export_options['stale_policy']=c['stale_registration_policy']
  exporter.run_once(c['compute_state'],state,anchor_document,authority,key,registrations,end,**export_options)
  # Fresh execution proof immediately before chain handoff, under the same held lock.
  guard_files(c);units=observe_units();identity=process_identity();now=time.time()
  receipt=exporter.sign(dict(version='single-live-reward-writer-v1',netuid=120,observed_at=now,global_writer_lock_held=True,legacy_validator_guard_verified=True,old_writers=units,**identity),key)
  exporter.atomic(state/'actual-writer-observation.json',receipt)
  proposal=read(state/('hour-'+str(end)+'-reward-units.json'))
  if execute:exporter.atomic(state/'writer-cursor.json',dict(window_end=end,status='submitting',proposal_sha256=sha(proposal)))
  try:
   result=submit.submit_hour(adapter,proposal,authority,receipt,now=now,boot_id=identity['boot_id'],writer_pid=identity['writer_pid'],writer_ticks=identity['writer_start_ticks'],execute=execute)
  except BaseException as error:
   # Leave submitting cursor intact: never automatically retry an uncertain transaction.
   exporter.atomic(state/'last-run.json',dict(version='actual-single-writer-run-v1',at=now,execute=execute,window_end=end,status='handoff_failed_outcome_uncertain' if execute else 'dry_run_failed',error_type=type(error).__name__,writer_receipt_sha256=sha(receipt)))
   raise
  if execute:exporter.atomic(state/'writer-cursor.json',dict(window_end=end,status=result['status'],proposal_sha256=sha(proposal)))
  exporter.atomic(state/'last-run.json',dict(version='actual-single-writer-run-v1',at=now,execute=execute,window_end=end,result=result,evidence=evidence,finalized_completeness=completeness,writer_receipt_sha256=sha(receipt)))
  return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cutover',required=True);p.add_argument('--anchor',required=True);p.add_argument('--authority',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--verifier-workforce-supplement',action='append',default=[]);p.add_argument('--compute-source-approval',action='append',default=[]);a=p.parse_args()
 def timed_out(*_):raise TimeoutError('bounded writer invocation')
 signal.signal(signal.SIGALRM,timed_out);signal.alarm(720)
 try:result=run_once(read(a.cutover),read(a.anchor),a.authority,execute=a.execute,verifier_workforce_supplements=[read(path)for path in a.verifier_workforce_supplement],compute_source_approvals=[read(path)for path in a.compute_source_approval])
 except Exception as error:
  print(json.dumps(dict(status='refused_or_failed',error_type=type(error).__name__,execute=a.execute)));raise SystemExit(1)
 finally:signal.alarm(0)
 print(json.dumps(dict(status=result['status'],window_end=result.get('window_end'),execute=a.execute)))
if __name__=='__main__':main()
