#!/usr/bin/env python3
"""Full v8.2.2 successor dry-run (chain v7 -> v8.2 -> v8.2.1 -> v8.2.2) on COPIES (never live dirs). Read-only w.r.t. live state.

Inputs: the REAL signed chain [v7 -> v8.2] + signed v8.2 policy, a fresh copy of the live lifecycle state
(CURRENT.json, auditor populations + state, evaluator openings + state) and a copy of the trainer-host
private CURRENT.json. Exercises exactly the in-code switch paths:
 AUDIT     every historical population authenticates under the chain guard (and is rejected by the single v8.2 policy)
 EVALUATOR evaluator state policy is a chain member; every stored opening validates against the member it names
 LIFECYCLE Lifecycle._adopt_succession on a CURRENT copy -> re-stamped onto v8.2 with a re-attested handover
           (signed by the on-box authority, written only into the copy), then the _current() release/config gate passes
 TRAINER   training_worker._adopt_release_succession on the trainer CURRENT copy, using a temp role config
           holding the signed chain; optimizer/model lineage preserved
Writes DRY-RUN-RESULT-v821.json into --out.
"""
import argparse,copy,hashlib,json,os,shutil,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
ap=argparse.ArgumentParser();ap.add_argument('--signed',required=True);ap.add_argument('--copy',required=True);ap.add_argument('--trainer-current',required=True)
ap.add_argument('--out',required=True);a=ap.parse_args()
SIG=Path(a.signed);C=Path(a.copy);OUT=Path(a.out)
LIVE=Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1').resolve()
for p in (C,OUT,Path(a.trainer_current)):assert LIVE not in p.resolve().parents and p.resolve()!=LIVE,'never run on live dirs'
sys.path.insert(0,str(SIG/'deployment-final-source-v822'))
from subnet.storage import Identity,canonical,sha
from subnet.backend_jobs import signed
from subnet.mimo_release_policy import ReleasePolicy
from subnet.mimo_release_succession import ReleaseChain
from subnet.mimo_audit_bridge import authenticate_population
from subnet import mimo_lifecycle,mimo_training_worker as tw
RA=Path('/home/const/subnet120-rewrite/state/root-audits')
life=json.loads((SIG/'lifecycle.v8.2.2.ACTIVE.private.json').read_bytes())
A='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
chain_doc=json.loads((SIG/'RELEASE-SUCCESSION.v7-v8.2-v8.2.1-v8.2.2.ROOT-SIGNED.private.json').read_bytes())
assert chain_doc==life['release_succession'],'lifecycle carries the signed chain'
chain=ReleaseChain(chain_doc,A,allow_public=True);v82=ReleasePolicy(life['release_policy'],A,allow_public=True)
assert chain.active.policy_sha256==v82.policy_sha256
res=dict(version='mimo-v822-dry-run-v1',at=time.time(),chain_sha256=chain.chain_sha256,members=[p.policy_sha256 for _,p in chain.members],
         first_rounds=[r for r,_ in chain.members],active_policy_sha256=v82.policy_sha256,copy=str(C),live_touched=False)
# AUDIT
ok=rej=0;fail=[];_POP={}
pops=sorted((C/'auditor').glob('*.population.json'))
for p in pops:
    doc=json.loads(p.read_bytes())
    try:_POP[p]=authenticate_population(doc,A,manifest_guard=chain.validate_manifest);ok+=1
    except Exception as e:fail.append((p.name[:16],repr(e)[:200]))
    try:authenticate_population(doc,A,manifest_guard=v82.validate_manifest)
    except Exception:rej+=1
res['audit']=dict(total=len(pops),chain_authenticated=ok,rejected_by_single_v82_policy=rej,failures=fail)
# EVALUATOR
st=json.loads((C/'evaluator/state.json').read_bytes());eok=0;efail=[]
ops=sorted((C/'evaluator').glob('*.opening.json'))
for p in ops:
    try:m=signed(json.loads(p.read_bytes()),A);chain.validate_manifest(m);eok+=1
    except Exception as e:efail.append((p.name[:16],repr(e)[:200]))
res['evaluator']=dict(state_policy=st['policy_sha256'][:12],state_policy_is_chain_member=chain.member(st['policy_sha256']) is not None,
                      openings_total=len(ops),openings_validated=eok,failures=efail,baseline_checkpoint=chain.baseline_checkpoint_id[:12])
# LIFECYCLE adoption on a copy
cur_path=C/'CURRENT.json';before=json.loads(cur_path.read_bytes())
seed=Path(life['authority_seed_file']);ident=Identity(bytes.fromhex(seed.read_text().strip()));assert ident.id==A
L=object.__new__(mimo_lifecycle.Lifecycle)if hasattr(mimo_lifecycle,'Lifecycle') else None
cls=[v for v in vars(mimo_lifecycle).values() if isinstance(v,type) and hasattr(v,'_adopt_succession')][0]
L=object.__new__(cls);L.chain=chain;L.policy=v82;L.identity=ident;L.collection_seconds=life['lifecycle']['collection_seconds'];L.max_tasks=life['lifecycle']['max_tasks'];L.root=C
adopted=L._adopt_succession(cur_path,copy.deepcopy(before));ident=None
auth=adopted.get('checkpoint_authorization');av=signed(auth,A) if auth else None
gate=adopted['version']==mimo_lifecycle.VERSION and adopted['policy_sha256']==v82.policy_sha256 and adopted['collection_seconds']==L.collection_seconds and adopted['max_tasks']==L.max_tasks
res['lifecycle']=dict(before=dict(policy=before['policy_sha256'][:12],epoch_index=before['epoch_index'],round=before['round'],optimizer_step=before['optimizer_step'],checkpoint=before['checkpoint']['id'][:12],max_tasks=before['max_tasks']),
  after=dict(policy=adopted['policy_sha256'][:12],epoch_index=adopted['epoch_index'],round=adopted['round'],optimizer_step=adopted['optimizer_step'],checkpoint=adopted['checkpoint']['id'][:12],max_tasks=adopted['max_tasks']),
  lineage_preserved=all(adopted[k]==before[k] for k in ('epoch_index','round','optimizer_step','checkpoint','previous_discovery_sha256')),
  handover_reattested=bool(av and av['policy_sha256']==v82.policy_sha256 and av['checkpoint']==adopted['checkpoint'] and av['input_checkpoint']==signed(before['checkpoint_authorization'],A)['input_checkpoint']),
  current_gate_passes=gate,rerun_idempotent=L._adopt_succession(cur_path,json.loads(cur_path.read_bytes()))==json.loads(cur_path.read_bytes()))
# TRAINER adoption on a copy
T=Path(tempfile.mkdtemp(prefix='v822-trainer-',dir=OUT));tc=T/'CURRENT.json';shutil.copy2(a.trainer_current,tc);os.chmod(tc,0o600)
key=T/'publisher.key';key.write_text('d'*64);os.chmod(key,0o600)
role=T/'role.json';role.write_text(json.dumps(dict(bucket='dry-run-placeholder',publisher_identity_file=str(key),release_succession=chain_doc)));os.chmod(role,0o600)
tcur=json.loads(tc.read_bytes())
binding=SimpleNamespace(run=tcur['run'],checkpoint=tcur['checkpoint'],optimizer_step=tcur['optimizer_step'],epoch='dry-run')
rel=json.loads((SIG/'deployment-final-source-v821/../SIGNING-RESULT.json').read_bytes()) if False else None
SRCSHA='d0b0475db692cc84b5c1759336c8cf9e7400b8317066ece15567edeb97f592c3'
tad=tw._adopt_release_succession(tc,copy.deepcopy(tcur),dict(training=dict(run_id=tcur['run'],role_config_file=str(role))),A,v82,SRCSHA,binding,allow_public=True)
res['trainer']=dict(before=dict(release=tcur.get('release_policy_sha256','')[:12],checkpoint=str(tcur['checkpoint'])[:12],optimizer_step=tcur['optimizer_step'],job_id=tcur.get('job_id')),
  after=dict(release=tad['release_policy_sha256'][:12],checkpoint=str(tad['checkpoint'])[:12],optimizer_step=tad['optimizer_step'],job_id=tad.get('job_id'),execution_mode=tad.get('execution_mode')),
  lineage_preserved=all(tad.get(k)==tcur.get(k) for k in ('run','checkpoint','optimizer_step','job_id')),bound_to_v82=tad['release_policy_sha256']==v82.policy_sha256,
  source_closure_note='dry-run stamps the v8.2 inventory sha; live worker stamps source_guard(job)')
shutil.rmtree(T)
# JOBS (fix 547573dd): build audit + eval jobs exactly as the schedulers do, with the fixed per-archive resolver and the
# REAL signed lifecycle config; check each with the REAL distributed_worker.source_for_job against each host's prepared
# v8.2.1 registry (paths mapped to local trees), and with the writer's approved_execution equality (admit_queue_reports).
import tempfile as _tf
from ops.mimo_lifecycle import archive_execution_resolver
from subnet.mimo_audit_bridge import build_audit_job
from subnet import distributed_worker as _dw
AD=SIG.parent/'activation-v822'
live={'source_files':life['adapters']['source_files'],'runtime_versions':life['adapters']['runtime_versions']}
resolve=archive_execution_resolver(chain,life['archive_execution'],live)
reader=signed(json.loads((SIG/'writer-reader.v8.2.2.ROOT-SIGNED.private.json').read_bytes()),A);ae=reader['approved_execution']
A7='69ae058be865ab47ccf7f4decbe855c4826bc87c96d87aa1173d69ef6fbab2f9';A82='8ae233ad397a72ecf35416c9c751bf40cc62acb5d5aa3fbb21d74594b3fba375'
ANEW=chain.active.scope['immutable_manifest']['source_bundle']['sha256']
A821=json.loads((SIG.parent/'staged-v822-successor/SUCCESSOR-BINDING.json').read_bytes())['previous_archive_sha256']
local={A82:SIG.parent/'release-v82-signed/deployment-final-source-v82',A821:SIG.parent/'release-v821-signed/deployment-final-source-v821',ANEW:SIG/'deployment-final-source-v822'}
WC=[v for v in vars(_dw).values() if isinstance(v,type) and 'source_for_job' in vars(v)][0]
placeholder=Path(_tf.mkdtemp(prefix='v7-noscan-',dir=OUT))
def worker(role):
    reg=json.loads((AD/'hosts'/role/'source-registry.qualified-v822s.json').read_bytes());regl={}
    for k,v in reg.items():regl[k]=dict(v,path=str((local.get(k) or placeholder).resolve()))
    return SimpleNamespace(source_registry=regl,backend_source=None,_source_inventory_checked={k for k in reg if k not in local})
W={r:worker(r) for r in ('verifier-1','verifier-2','verifier-3','evaluator','trainer')}
def wcheck(role,job):
    try:WC.source_for_job(W[role],job);return True
    except Exception as e:return repr(e)[:120]
def writer_ok(job):
    m=signed(job['manifest'],A);e=ae.get(m['source_bundle']['sha256'])
    return bool(e) and job['source_files']==e['source_files'] and job['runtime_versions']==e['runtime_versions']
from subnet.storage import Bucket as _B;_bucket=_B(life['bucket'])  # local presign only (as AuditScheduler); no R2 call
BUILT=[];now=time.time();J=dict(audit={},eval={},e39=None,counterfactual_old_live_inventory={});jfail=[]
for p in pops:
    pop=_POP[p];m=signed(pop['manifest_document'],A)
    arch=m['source_bundle']['sha256'][:8];row=pop['records'][0]
    job=build_audit_job(pop,[row],authority=A,manifest_guard=chain.validate_manifest,job_id='dry-audit-'+p.name[:16],created_at=now,expires_at=now+900,
        read_url=lambda k:_bucket.presign(k,'get_object',900),checkpoint_read_urls={},**resolve(m))
    BUILT.append(('audit',m['epoch'],m['source_bundle']['sha256'],job))
    r={v:wcheck(v,job) for v in ('verifier-1','verifier-2','verifier-3')};wo=writer_ok(job)
    d=J['audit'].setdefault(arch,dict(jobs=0,workers_ok=0,writer_ok=0,inventory_files=len(job['source_files'])));d['jobs']+=1
    d['workers_ok']+=all(x is True for x in r.values());d['writer_ok']+=wo
    if not(all(x is True for x in r.values()) and wo):jfail.append(('audit',p.name[:16],r,wo))
    old=dict(job,source_files=live['source_files']);c=J['counterfactual_old_live_inventory'].setdefault('audit-'+arch,dict(jobs=0,refused_by_verifier_1=0))
    c['jobs']+=1;c['refused_by_verifier_1']+=wcheck('verifier-1',old) is not True
def evaljob(opening):
    m=signed(opening,A);chain.validate_manifest(m)
    return dict(schema=1,job_id='dry-eval',role='evaluate',created_at=now,expires_at=now+14400,manifest=opening,checkpoint_read_urls={},**resolve(m))
for p in ops:
    job=evaljob(json.loads(p.read_bytes()));arch=signed(job['manifest'],A)['source_bundle']['sha256'][:8]
    BUILT.append(('eval',signed(job['manifest'],A)['epoch'],signed(job['manifest'],A)['source_bundle']['sha256'],job))
    r=wcheck('evaluator',job);wo=writer_ok(job)
    d=J['eval'].setdefault(arch,dict(jobs=0,evaluator_ok=0,writer_ok=0,inventory_files=len(job['source_files'])));d['jobs']+=1;d['evaluator_ok']+=r is True;d['writer_ok']+=wo
    if not(r is True and wo):jfail.append(('eval',p.name[:16],r,wo))
e39p=C/'e39-OPENING.json'
if e39p.exists():
    op=json.loads(e39p.read_bytes());job=evaljob(op);m=signed(op,A)
    J['e39']=dict(epoch=m['epoch'],archive=m['source_bundle']['sha256'][:8],member=chain.policy_for_manifest(m).policy_sha256[:12],inventory_files=len(job['source_files']),
        verifiers={v:wcheck(v,job) for v in ('verifier-1','verifier-2','verifier-3')},evaluator=wcheck('evaluator',job),writer_ok=writer_ok(job),
        equals_v82_approved=job['source_files']==ae[A82]['source_files'])
    if not(all(x is True for x in J['e39']['verifiers'].values()) and J['e39']['evaluator'] is True and J['e39']['writer_ok']):jfail.append(('e39',J['e39']))
else:jfail.append(('e39','opening copy missing'))
J['v821_member_archive']=dict(archive=ANEW[:8],live_inventory_registered_on_all_hosts=all(json.loads((AD/'hosts'/r/'source-registry.qualified-v822s.json').read_bytes())[ANEW]['source_files']==live['source_files'] for r in W),
    writer_approved=ae[ANEW]==live)
J['failures']=jfail;res['jobs']=J;shutil.rmtree(placeholder)
# ---------------- v8.2.2: SOURCE_GUARD (worker source-closure check) ----------------
# Real subnet.mimo_worker.source_guard: (1) in-process from the signed v8.2.2 tree (byte-identical to host source-final-v822b);
# (2) on every real worker host with the host's own worker python against source-final-v822b (post-activation) and
# source-final-v821b (running now). Host inputs carry ONLY source_files (no presigned URLs).
import subnet.mimo_worker as _mw, subprocess as _sp
assert Path(_mw.__file__).resolve().parents[1]==(SIG/'deployment-final-source-v822').resolve(),'in-process guard root'
A3=ANEW;e39_job=evaljob(json.loads(e39p.read_bytes())) if e39p.exists() else None
guard_in={}
for kind,ep,arch,job in BUILT:guard_in.setdefault(kind+'-'+arch[:8],dict(source_files=job['source_files'],epoch=ep))
if e39_job is not None:guard_in['eval-e39-'+signed(e39_job['manifest'],A)['source_bundle']['sha256'][:8]]=dict(source_files=e39_job['source_files'],epoch='e39')
guard_in['a3-'+A3[:8]+'-lifecycle-archive_execution']=dict(source_files=life['archive_execution'][A3]['source_files'],epoch='v8.2.2-era')
def sg(j):
    try:_mw.source_guard(j);return True
    except Exception as e:return str(e)[:80]
expect=lambda k:k.split('-')[-1] if False else (A821[:8] in k or A3[:8] in k)
G=dict(inprocess={k:sg(v) for k,v in guard_in.items()},hosts={})
PLAN=json.loads((AD/'ACTIVATION-PLAN.json').read_bytes())['hosts']
gj=OUT/'sg_jobs.json';gj.write_text(json.dumps({k:dict(source_files=v['source_files']) for k,v in guard_in.items()}));os.chmod(gj,0o600)
gs=OUT/'sg_check.py';gs.write_text('''import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
import subnet.mimo_worker as m
assert Path(m.__file__).resolve().parents[1]==root
out={}
for k,j in json.load(open(sys.argv[2])).items():
    try:m.source_guard(j);out[k]=True
    except Exception as e:out[k]=str(e)[:80]
print(json.dumps(out))
''');os.chmod(gs,0o600)
for role,hp in PLAN.items():
    if role=='trainer':continue
    ssha=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','UserKnownHostsFile=/home/const/.lium/known_hosts/'+hp['known_hosts']]
    hostpy=hp['owpy'].split(' -I -B ')[0]
    try:
        _sp.run(['scp','-q',*ssha,'-P',str(hp['port']),str(gj),str(gs),'root@'+hp['host']+':/root/'],check=True,timeout=120)
        G['hosts'][role]={}
        for tree in (hp['source_v822'],hp['source_v821']):
            r=_sp.run(['ssh','-T',*ssha,'-p',str(hp['port']),'root@'+hp['host'],f"cd /tmp && {hostpy} -I -B /root/sg_check.py {tree} /root/sg_jobs.json"],capture_output=True,text=True,timeout=300)
            G['hosts'][role][tree.rsplit('/',1)[1]]=json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode==0 else 'ERR '+r.stderr.strip()[-200:]
    finally:
        _sp.run(['ssh','-T',*ssha,'-p',str(hp['port']),'root@'+hp['host'],'rm -f /root/sg_jobs.json /root/sg_check.py'],timeout=60)
gj.unlink();gs.unlink()
def guard_expect(res_map):
    return type(res_map) is dict and set(res_map)==set(guard_in) and all((res_map[k] is True)==(A821[:8] in k or A3[:8] in k) for k in guard_in)
G['summary']=dict(inputs=sorted(guard_in),inprocess_as_expected=guard_expect(G['inprocess']),
    hosts_as_expected={r:{t:guard_expect(v) for t,v in trees.items()} if type(trees) is dict else trees for r,trees in G['hosts'].items()},
    rule='pass iff job archive in {%s (e40/v8.2.1), %s (v8.2.2)}; v7 69ae058b and e39 8ae233ad must fail'%(A821[:8],A3[:8]))
guard_ok=G['summary']['inprocess_as_expected'] and len(G['hosts'])==4 and all(all(v.values()) for v in G['summary']['hosts_as_expected'].values() if type(v) is dict) \
  and all(type(v) is dict for v in G['summary']['hosts_as_expected'].values())
res['source_guard']=G
# ---------------- v8.2.2: SCHEDULER (archive-scoped audit selection) ----------------
# Real AuditScheduler.tick through ops.mimo_lifecycle.archive_scoped_audit_scheduler on a COPY of the auditor dir and a
# sqlite SNAPSHOT of the live queue. Jobs are signed with a throwaway key; the R2 write is a local recorder (no network).
from ops.mimo_lifecycle import archive_scoped_audit_scheduler
from subnet import mimo_audit_scheduler as mas
import sqlite3 as _sq
from contextlib import contextmanager
SC=Path(tempfile.mkdtemp(prefix='v822-sched-',dir=OUT));AUD=SC/'auditor'
shutil.copytree(C/'auditor',AUD,ignore=shutil.ignore_patterns('audit.lock'));os.chmod(AUD,0o700)
src_db=_sq.connect('file:'+str(LIVE/'roles.sqlite3')+'?mode=ro',uri=True,timeout=10);qdb=_sq.connect(str(SC/'queue.sqlite3'));src_db.backup(qdb);src_db.close()
qdb.row_factory=_sq.Row
class SnapQueue:
    authority=A
    def __init__(s):s.enq=[]
    @contextmanager
    def transaction(s):yield qdb
    def enqueue(s,env):s.enq.append(env['payload']);qdb.execute('INSERT OR REPLACE INTO jobs (id,digest,envelope,role,expires,status,attempt) VALUES (?,?,?,?,?,?,0)',(env['payload']['job_id'],'dry-'+env['payload']['job_id'],'{}','verify',env['payload']['expires_at'],'queued'))
writes=[];_orig_imm=mas._immutable;mas._immutable=lambda b,k,raw:writes.append(k)
throw=Identity()
from subnet.mimo_release_policy import sign as _sign
cfg=dict(life['audit_scheduler']);assert cfg.pop('enabled') is True and cfg.pop('directory')==str(LIVE/'auditor'),cfg
assert cfg.get('max_execution_retries')==3,cfg
Base=mas.AuditScheduler;Scoped=archive_scoped_audit_scheduler(Base,resolve.auditable_archives)
q=SnapQueue();sched=Scoped(AUD,bucket=_bucket,queue=q,identity=SimpleNamespace(id=A),manifest_guard=chain.validate_manifest,
    metadata=lambda m:dict(resolve(m),checkpoint_read_urls={}),**cfg)
sched._signed=lambda job:_sign(throw,job)
base_view=Base.__new__(Base);base_view.__dict__.update(sched.__dict__)
before=json.loads((AUD/'state.json').read_bytes())
epoch_arch={ep:sched._population_facts(c)['archive'] for ep,c in before['populations'].items()}
excluded_epochs=sorted(ep for ep,a in epoch_arch.items() if a not in resolve.auditable_archives)
done_before={r['id'] for r in Base.completed_queue_rows(base_view)}
ticks=[];t0=time.time()
for i in range(4):
    out=sched.tick();ticks.append(out)
    for j in q.enq:
        if i%2==0:qdb.execute("UPDATE jobs SET status='failed' WHERE id=? AND status='queued'",(j['job_id'],))
after=json.loads((AUD/'state.json').read_bytes());mas._immutable=_orig_imm
new_keys=[k for k in after['draws'] if k not in before['draws']]
changed_excluded=[k for k,d in before['draws'].items() if d['row']['epoch'] in excluded_epochs and after['draws'].get(k)!=d]
renewed=[k for k,d in after['draws'].items() if k in before['draws'] and after['draws'][k]!=before['draws'][k]]
enq_epochs=sorted({signed(j['manifest'],A)['epoch'] for j in q.enq});enq_arch=sorted({signed(j['manifest'],A)['source_bundle']['sha256'][:8] for j in q.enq})
job_checks=[]
for j in q.enq:
    job_checks.append(dict(job=j['job_id'][:24],registry={v:wcheck(v,j) for v in ('verifier-1','verifier-2','verifier-3')},source_guard=sg(j)))
rep=json.loads((AUD/'UNAUDITABLE-BY-DESIGN.json').read_bytes())
done_after={r['id'] for r in sched.completed_queue_rows()}
S=dict(ticks=ticks,seconds=round(time.time()-t0,1),populations=len(before['populations']),excluded_epochs=excluded_epochs,
  auditable_epochs=sorted(ep for ep in epoch_arch if ep not in excluded_epochs),
  new_draws=len(new_keys),new_draw_epochs=sorted({after['draws'][k]['row']['epoch'] for k in new_keys}),
  excluded_draws_before=sum(1 for d in before['draws'].values() if d['row']['epoch'] in excluded_epochs),excluded_draws_changed=len(changed_excluded),
  renewed_or_dispatched_existing=[after['draws'][k]['row']['epoch'] for k in renewed],
  enqueued=len(q.enq),enqueued_epochs=enq_epochs,enqueued_archives=enq_arch,r2_writes=len(writes),
  max_generation=max([d.get('generation',0) for k,d in after['draws'].items() if d['row']['epoch'] not in excluded_epochs] or [0]),
  report=dict(excluded_epochs=len(rep['excluded_epochs']),excluded_records=rep['excluded_records'],excluded_draws_parked=rep['excluded_draws_parked']),
  completed_rows_before=len(done_before),completed_rows_preserved=done_before<=done_after,
  jobs_all_pass_both_checks=all(all(x is True for x in c['registry'].values()) and c['source_guard'] is True for c in job_checks))
sched_ok=(S['excluded_draws_changed']==0 and set(S['new_draw_epochs'])<=set(S['auditable_epochs']) and set(S['enqueued_epochs'])<=set(S['auditable_epochs'])
  and not set(S['renewed_or_dispatched_existing'])&set(excluded_epochs) and S['enqueued']>0 and S['jobs_all_pass_both_checks']
  and sorted(rep['excluded_epochs'])==excluded_epochs and S['completed_rows_preserved'] and S['max_generation']<=3 and S['r2_writes']==S['enqueued'])
res['scheduler']=dict(summary=S,job_checks=job_checks[:20])
shutil.rmtree(SC)
jobs_ok=(not jfail and J['v821_member_archive']['live_inventory_registered_on_all_hosts'] and J['v821_member_archive']['writer_approved']
  and all(c['refused_by_verifier_1']==c['jobs'] for k,c in J['counterfactual_old_live_inventory'].items() if k.startswith('audit-69ae')))
res['passed']=(not fail and ok==len(pops) and rej==len(pops) and not efail and eok==len(ops) and res['evaluator']['state_policy_is_chain_member']
  and res['lifecycle']['lineage_preserved'] and res['lifecycle']['handover_reattested'] and gate and res['trainer']['lineage_preserved'] and res['trainer']['bound_to_v82'] and jobs_ok and guard_ok and sched_ok)
p=OUT/'DRY-RUN-RESULT-v822.json';p.write_text(json.dumps(res,indent=1,sort_keys=True)+'\n');os.chmod(p,0o600)
print(json.dumps(dict(source_guard=res['source_guard']['summary'],scheduler=res['scheduler']['summary'])))
print(json.dumps(dict(passed=res['passed'],jobs=dict(audit=J['audit'],eval=J['eval'],e39=J['e39'],cf=J['counterfactual_old_live_inventory'],nfail=len(jfail)),audit=res['audit']['chain_authenticated'],evaluator=eok,lifecycle=res['lifecycle'],trainer=res['trainer']['after'])))
