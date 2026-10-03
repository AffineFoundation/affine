"""Prospective authenticated reward bridge; no chain/network/service/model operations.

Historical nonpayable documents are immutable and never promoted. Inputs here
are original immutable signed objects supplied by a trusted operator reader.
"""
import base64,hashlib,json,math
from fractions import Fraction
from nacl.signing import VerifyKey
from subnet.audit_policy import validate as audit_policy,penalties
from subnet.scoring import score,adjusted_point_fractions
from subnet.protocol import sample_key,classification
VERSION='live-verified-subset-reward-v1'
SCALE=1_000_000
CHAIN_SCOPE='operator-live-reward-bridge-only-v1'
BAD_PREFIXES=('nonpayable-','test-','mock-')
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda v:hashlib.sha256(canonical(v)).hexdigest()
def need(condition,message):
 if not condition:raise ValueError(message)
def integer(v,name,maximum=2**63-1):need(type(v)is int and 0<=v<=maximum,name);return v
def finite(v,name):need(type(v)in(int,float) and math.isfinite(v) and v>=0,name);return v
def signed(document,authority):
 need(isinstance(document,dict) and set(document)=={'payload','signer','signature'} and document['signer']==authority,'authority/signature envelope')
 VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
 need(isinstance(document['payload'],dict),'signed object');return document['payload']
def proposed_contract(epoch,start,checkpoint,source_sha,cutover_id,*,penalty_policy=None,reward_epoch=None):
 """Unsigned template only. Actual Controller must bind it before first publication."""
 return dict(version=VERSION,payable=True,epoch=epoch,reward_epoch=reward_epoch,starts_at=start,checkpoint=checkpoint,source_sha256=source_sha,cutover_id=cutover_id,
  netuid=120,basis='fully-audited-unique-observed-subset',unchecked_duplicate_claims='unresolved-no-global-uniqueness-claim',
  penalties=penalties(penalty_policy),units_per_point=SCALE,rounding='floor-after-hour-aggregation',compute_chain_transactions=False)
def contract(manifest,anchor):
 epoch=manifest['epoch'];need(isinstance(epoch,str) and anchor['compute_epoch_prefix']=='nonpayable-live-reward-math-v1-' and epoch.startswith(anchor['compute_epoch_prefix']),'exact prospective compute-only namespace')
 need(anchor['live_epoch_prefix'].startswith('live-') and not anchor['live_epoch_prefix'].startswith(BAD_PREFIXES),'separate live reward namespace')
 need(manifest.get('payable') is False and manifest.get('chain_execution_scope')==CHAIN_SCOPE,'explicit compute no-chain versus live reward scope')
 c=manifest.get('live_reward_contract');need(isinstance(c,dict),'fresh first-manifest reward contract')
 expected=proposed_contract(epoch,manifest['start'],manifest['checkpoint']['id'],manifest['source_bundle']['sha256'],anchor['cutover_id'],penalty_policy=c.get('penalties'),reward_epoch=anchor['live_epoch_prefix']+sha({'compute_epoch':epoch}))
 need(canonical(c)==canonical(expected),'exact reward contract/schema')
 need(integer(c['starts_at'],'reward start')>=anchor['effective_at'],'historical epoch excluded')
 need(c['source_sha256'] in anchor['approved_compute_sources'],'reviewed source only')
 need(manifest['deadline']>manifest['start'],'epoch time binding')
 policy=manifest['audit_policy']
 if policy.get('mode')=='sampled':
  policy=audit_policy(policy);need('submission_counts' not in manifest['audit_policy'],'opening has no postfreeze allocations')
  need(canonical(policy['penalties'])==canonical(c['penalties']),'signed penalty binding')
 else:raise ValueError('live reward bridge supports bounded-random-v1 only')
 return c

def project(*,manifest_document,opening_document,score_document,audit_documents,registrations_document,anchor_document,authority):
 """Authenticate original opening/report chain and recompute a separate reward record.

Reader must additionally authenticate remote jobs/source and immutable R2 first
publication. This API authenticates controller attestations; it performs no GPU
recompute and no R2 GET. Never accept caller-miner report assertions instead.
"""
 anchor=signed(anchor_document,authority);need(anchor.get('version')=='live-reward-cutover-v1' and anchor.get('netuid')==120,'cutover anchor')
 integer(anchor['effective_at'],'cutover time');need(isinstance(anchor['approved_compute_sources'],list) and anchor['approved_compute_sources'],'source approvals')
 m=signed(manifest_document,authority);c=contract(m,anchor);opening=signed(opening_document,authority)
 need(opening.get('version')=='immutable-first-manifest-v1' and opening.get('epoch')==m['epoch'] and opening.get('first_manifest_sha256')==sha(manifest_document),'immutable original opening binding')
 finite(opening['published_at'],'publication time');need(m['start']<=opening['published_at']<m['deadline'],'published during fresh epoch')
 regs=signed(registrations_document,authority);need(regs.get('epoch')==m['epoch'] and regs.get('snapshot_block')==m.get('registration_snapshot_block'),'epoch registration snapshot')
 rows=regs['registrations'];by_public={}
 for hotkey,r in rows.items():
  need(hotkey!=anchor['owner_hotkey'],'owner not reward miner');integer(r['uid'],'UID',65535)
  need(isinstance(r['public_key'],str) and len(r['public_key'])==64 and r['public_key'] not in by_public,'unique Ed25519 activated identity')
  by_public[r['public_key']]=(hotkey,r)
 need(set(m['capabilities'])==set(by_public),'signed epoch identity population')
 s=signed(score_document,authority);need(s.get('payable') is False and s.get('epoch_id')==m['epoch'] and s.get('checkpoint')==m['checkpoint']['id'],'original compute scores binding')
 finite(s['finalized_at'],'finalization time');need(s['finalized_at']>=m['deadline'],'no early rewards')
 receipts=s['receipts'];need(set(audit_documents)==set(receipts) and set(receipts)<=set(by_public),'all frozen audited submission identities')
 reports={}
 definitions={r['env_id']:r for r in m['environments']}
 for miner,doc in audit_documents.items():
  r=signed(doc,authority);need(r.get('epoch')==m['epoch'] and r.get('submission_sha256')==receipts[miner]['sha256'],'frozen artifact/audit binding')
  need(isinstance(r.get('remote_job_id'),str) and r['remote_job_id'],'authenticated verifier request reference')
  accepted=r['accepted'];seen=set()
  confirmed=[o for o in r.get('outcomes',[]) if o.get('valid') is True and o.get('fully_audited') is True]
  need(len(confirmed)==len(accepted),'every credited batch fully audited')
  confirmed_keys={(o.get('env_id'),o.get('index')) for o in confirmed}
  for b in accepted:
   d=definitions.get(b.get('env_id'));index=b.get('sample_index');integer(index,'sample index')
   need(d is not None and index in d['indices'] and type(b.get('index')) is int and b.get('index')==index and b.get('schema')==2 and b.get('epoch')==m['epoch'] and b.get('checkpoint')==m['checkpoint']['id'],'actual schema2 epoch/sample/checkpoint')
   need(b.get('environment_version')==d['spec']['version'] and (b['env_id'],index) in confirmed_keys,'approved native version/audited attribution')
   rolls=b.get('rollouts');need(isinstance(rolls,list) and len(rolls)==m['K']+m['L'],'real accepted rollout population')
   need(all(r.get('env_id')==b['env_id'] and type(r.get('index'))is int and r['index']==index for r in rolls),'rollout attribution')
   need(sum(classification(r)=='positive' for r in rolls)==m['K'] and sum(classification(r)=='negative' for r in rolls)==m['L'],'native K/L metadata quota')
   key=sample_key(b);need(key not in seen,'duplicate accepted task');seen.add(key)
  need(len(accepted)<=m['max_batches'],'signed quota')
  reports[miner]=r
 recomputed=score(reports,c['penalties'])
 fields=('points','adjusted_points','total','weights','penalty_policy','penalties','provisional','score_basis','unchecked_duplicate_claims_unresolved','duplicate_coverage')
 for field in fields:need(field in s and canonical(s[field])==canonical(recomputed[field]),'recomputed score/penalty '+field)
 earned={};identities={};raw={}
 raw_points,adjusted,_=adjusted_point_fractions(reports,c['penalties'])
 need(raw_points==recomputed['points'],'shared pure score identity')
 for miner,points in raw_points.items():
  value=adjusted[miner]
  hotkey,row=by_public[miner];earned[hotkey]=[value.numerator,value.denominator];raw[hotkey]=points
  identities[hotkey]={k:row[k] for k in ('uid','public_key')}
 return dict(version=VERSION,payable=True,epoch_id=c['reward_epoch'],compute_epoch_id=m['epoch'],finalized_at=s['finalized_at'],compute_payable=False,compute_chain_transactions=False,
  manifest_sha256=sha(manifest_document),opening_sha256=sha(opening_document),score_sha256=sha(score_document),registrations_sha256=sha(registrations_document),
  contract=c,cutover_sha256=sha(anchor_document),raw_unique_observed_points=raw,adjusted_point_fractions=earned,identities=identities,
  duplicate_coverage=recomputed['duplicate_coverage'],unchecked_duplicate_claims_unresolved=recomputed['unchecked_duplicate_claims_unresolved'],
  reward_basis=c['basis'],assurance='controller-authenticated-fully-audited-subset-not-fresh-bridge-inference',
  audit_document_sha256={k:sha(v) for k,v in audit_documents.items()})

def hourly_reward_units(documents,authority,window_end,*,fresh_registrations):
 """New reward-only reducer; original hourly_points remains unchanged for history.

Aggregate rational penalties before declared integer rounding. A changed UID or
public key refuses the entire hour, rather than renormalizing somebody away.
"""
 integer(window_end,'UTC hour');need(window_end%3600==0,'integral UTC hour')
 totals={};identities={};seen=set();record_hashes=[]
 for document in documents:
  r=signed(document,authority)
  need(r.get('version')==VERSION and r.get('payable') is True and not str(r.get('epoch_id','')).startswith(BAD_PREFIXES),'reward-only source record')
  need(r['epoch_id']==r['contract']['reward_epoch'],'live reward identity binding')
  need(r['contract']['version']==VERSION and r['contract']['units_per_point']==SCALE and r['contract']['rounding']=='floor-after-hour-aggregation','hour reward policy')
  finite(r['finalized_at'],'finite finalization')
  if not window_end-3600<=r['finalized_at']<window_end:continue
  need(r['epoch_id'] not in seen,'duplicate reward epoch');seen.add(r['epoch_id']);record_hashes.append(sha(document))
  need(set(r['adjusted_point_fractions'])==set(r['identities']),'reward identity coverage')
  for hotkey,pair in r['adjusted_point_fractions'].items():
   need(isinstance(pair,list) and len(pair)==2,'rational points');n=integer(pair[0],'point numerator',2**4096);d=integer(pair[1],'point denominator',2**4096);need(d>0,'positive denominator')
   identity=r['identities'][hotkey];fresh=fresh_registrations.get(hotkey)
   need(fresh is not None and all(fresh.get(k)==identity.get(k) for k in ('uid','public_key')),'stale registration denies hour')
   if hotkey in identities:need(identities[hotkey]==identity,'UID reuse inside hour')
   identities[hotkey]=identity;totals[hotkey]=totals.get(hotkey,Fraction(0))+Fraction(n,d)
 units={hotkey:int(points*SCALE) for hotkey,points in totals.items()}
 return dict(version='live-reward-hour-units-v1',window_end=window_end,units_per_point=SCALE,points=units,registrations=identities,source_reward_records=record_hashes,
  fractional_points={k:[p.numerator,p.denominator] for k,p in totals.items()},rounding='floor-after-hour-aggregation',chain_executed=False)

def writer_gate(receipt_document,authority,*,now,boot_id,writer_pid,writer_ticks):
 """Metadata refusal gate only; ROOT must obtain actual local process/unit proofs."""
 r=signed(receipt_document,authority);need(r.get('version')=='single-live-reward-writer-v1' and r.get('netuid')==120,'writer cutover receipt')
 need(r['boot_id']==boot_id and r['writer_pid']==writer_pid and str(r['writer_start_ticks'])==str(writer_ticks),'exact live writer process')
 finite(r['observed_at'],'writer observation');need(0<=now-r['observed_at']<=60,'fresh cutover observation')
 need(r.get('global_writer_lock_held') is True and r.get('legacy_validator_guard_verified') is True,'single writer and verified legacy hook')
 need(r.get('writer_process_state') in ('R','S','D','I'),'live nonstopped writer state')
 expected={'affine-transition-weights.timer','affine-transition-weights.service','affine-hourly-burn.timer','affine-hourly-burn.service'}
 need(isinstance(r.get('old_writers'),list) and len(r['old_writers'])==4 and {x.get('unit') for x in r['old_writers']}==expected,'exact old writer unit coverage')
 need(r.get('old_writers') and all(x.get('running') is False and x.get('enabled') is False and x.get('status_query_succeeded') is True for x in r['old_writers']),'old writers disabled/inactive known status')
 return r


def inject_opening_manifest(manifest,anchor_document,authority,registrations):
 """Trusted operator hook BEFORE first manifest signing; never touches history."""
 import copy
 m=copy.deepcopy(manifest);anchor=signed(anchor_document,authority)
 need('live_reward_contract' not in m and 'chain_execution_scope' not in m,'one fresh contract insertion only')
 need(anchor.get('version')=='live-reward-cutover-v1' and anchor.get('netuid')==120,'cutover anchor')
 need(isinstance(registrations,dict) and registrations,'fresh activated registrations')
 blocks={r['snapshot_block'] for r in registrations.values()};need(len(blocks)==1,'one registration chain snapshot')
 m['registration_snapshot_block']=blocks.pop();m['payable']=False;m['chain_execution_scope']=CHAIN_SCOPE
 policy=m['audit_policy'];penalty_policy=policy.get('penalties') if policy.get('mode')=='sampled' else anchor.get('penalty_policy')
 m['live_reward_contract']=proposed_contract(m['epoch'],m['start'],m['checkpoint']['id'],m['source_bundle']['sha256'],anchor['cutover_id'],penalty_policy=penalty_policy,reward_epoch=anchor['live_epoch_prefix']+sha({'compute_epoch':m['epoch']}))
 contract(m,anchor)
 need({r['public_key'] for r in registrations.values()}==set(m['capabilities']),'activated public identities match encrypted grants')
 return m

def emit_opening_documents(controller,manifest,registrations):
 """Called only AFTER final immutable manifest publication succeeds.

RemoteController calls after buffered max_batches/heldout fields are finalized.
No attestation is produced by the buffered intermediate Controller.open.
 """
 from pathlib import Path
 m=manifest;need('live_reward_contract' in m,'explicit prospective live contract')
 first=controller.signed(m)
 registration=controller.signed(dict(epoch=m['epoch'],snapshot_block=m['registration_snapshot_block'],registrations=registrations))
 opening_path=Path(controller.state)/(m['epoch']+'-opening-attestation.json')
 if opening_path.exists():
  opening=json.loads(opening_path.read_text());o=signed(opening,controller.authority.id)
  need(set(o)=={'version','epoch','first_manifest_sha256','published_at'} and o['version']=='immutable-first-manifest-v1' and o['epoch']==m['epoch'] and o['first_manifest_sha256']==sha(first),'existing original opening identity')
  finite(o['published_at'],'opening publication time');need(m['start']<=o['published_at']<m['deadline'],'existing original opening time')
 else:
  now=__import__('time').time();need(m['start']<=now<m['deadline'],'cannot attest expired missing opening')
  opening=controller.signed(dict(version='immutable-first-manifest-v1',epoch=m['epoch'],first_manifest_sha256=sha(first),published_at=now))
 for name,doc in [('first-signed-manifest',first),('opening-attestation',opening),('signed-registrations',registration)]:
  path=Path(controller.state)/(m['epoch']+'-'+name+'.json')
  if path.exists():need(canonical(json.loads(path.read_text()))==canonical(doc),'immutable opening attestation collision')
  else:
   with path.open('x') as f:json.dump(doc,f,sort_keys=True);f.write('\n')
   path.chmod(0o600)
 return first,opening,registration

def persist_signed_compute_evidence(controller,manifest,result,reports):
 """Additional local attestations after ordinary public score publication."""
 from pathlib import Path
 need(manifest.get('live_reward_contract') is not None and result.get('payable') is False,'live contract plus unchanged computational scores')
 objects=[('signed-compute-scores',result)]+[('signed-compute-audit-'+miner,r) for miner,r in reports.items()]
 for name,payload in objects:
  path=Path(controller.state)/(manifest['epoch']+'-'+name+'.json');document=controller.signed(payload)
  if path.exists():need(canonical(json.loads(path.read_text()))==canonical(document),'immutable finalized evidence collision')
  else:
   with path.open('x') as f:json.dump(document,f,sort_keys=True);f.write('\n')
   path.chmod(0o600)

def export_epoch(compute_state,epoch,anchor_document,authority):
 """Read original signed sidecars emitted by prospective controller hooks."""
 import re
 from pathlib import Path
 need(isinstance(epoch,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',epoch),'epoch path')
 state=Path(compute_state)
 def read(label):return json.loads((state/(epoch+'-'+label+'.json')).read_text())
 scores=read('signed-compute-scores');payload=signed(scores,authority)
 audits={miner:read('signed-compute-audit-'+miner) for miner in payload['receipts']}
 return project(manifest_document=read('first-signed-manifest'),opening_document=read('opening-attestation'),score_document=scores,audit_documents=audits,
  registrations_document=read('signed-registrations'),anchor_document=anchor_document,authority=authority)

def prevalidate_opening_arguments(epoch,checkpoint,miners,duration,audit,source,anchor_document,registrations,authority):
 """Refuse bad signatures/prefix/source/identity BEFORE Gateway creates grants."""
 if anchor_document is None:
  need(registrations is None,'reward registry cannot appear without prospective contract');return
 import time
 now=int(time.time())
 candidate=dict(epoch=epoch,payable=False,start=now,deadline=now+duration,checkpoint=checkpoint,source_bundle=source,
  capabilities={m:'not-yet-created' for m in miners},audit_policy=dict(audit or {'mode':'full','version':1}))
 inject_opening_manifest(candidate,anchor_document,authority,registrations)
