"""Explicit prospective statistical reward bridge, separate from strict receipts.

This reader authenticates operator snapshots; it does not claim every sample
was audited or recompute model inference. Actual chain submission still requires
the independent single-writer gate and ChainAdapter's current identity checks.
"""
import copy
from decimal import Decimal, ROUND_FLOOR
from .live_reward_bridge import signed,need,sha,canonical,integer,finite,writer_gate
from .continuous_audit_policy import hourly_aggregate
VERSION='continuous-statistical-reward-v1'
ANCHOR='continuous-statistical-reward-activation-v1'
UNITS='continuous-statistical-hour-units-v1'
SCALE=1_000_000

def activation(document,authority):
 a=signed(document,authority)
 need(set(a)=={'version','netuid','owner_hotkey','activation_id','effective_at','approved_sources','units_per_point','rounding'},'exact statistical reward activation')
 need(a['version']==ANCHOR and a['netuid']==120 and type(a['owner_hotkey'])is str and a['owner_hotkey']and type(a['activation_id'])is str and a['activation_id'],'statistical reward activation scope')
 finite(a['effective_at'],'activation time')
 need(type(a['approved_sources'])is list and a['approved_sources']and len(set(a['approved_sources']))==len(a['approved_sources'])and all(type(v)is str and len(v)==64 and all(c in '0123456789abcdef'for c in v)for v in a['approved_sources']),'approved statistical source inventory')
 need(a['units_per_point']==SCALE and a['rounding']=='floor-after-hour-aggregation','explicit statistical rounding')
 return a

def inject_opening(manifest,activation_document,authority,registrations):
 """Must run BEFORE first original manifest signing, never on historical epochs."""
 a=activation(activation_document,authority);m=copy.deepcopy(manifest)
 need(m.get('training_input_policy')=='committed-unaudited-training-v1','statistical learner policy')
 finite(m['start'],'opening time');need(m['start']>=a['effective_at']and m['source_bundle']['sha256']in a['approved_sources'],'prospective source/time admission')
 need(m.get('payable')is False and 'continuous_reward_contract'not in m,'fresh compute-only opening')
 need(type(registrations)is dict and registrations,'actual chain registration snapshot')
 public=[];blocks=set();uids=set()
 for hotkey,row in registrations.items():
  need(hotkey!=a['owner_hotkey'],'owner excluded');integer(row['uid'],'UID',65535);integer(row['snapshot_block'],'snapshot block')
  need(row['uid']not in uids,'unique UID');uids.add(row['uid']);blocks.add(row['snapshot_block']);public.append(row['public_key'])
 need(len(blocks)==1 and len(public)==len(set(public))and set(public)==set(m['capabilities']),'one exact activated identity snapshot')
 m['registration_snapshot_block']=next(iter(blocks))
 m['continuous_reward_contract']=dict(version=VERSION,activation_sha256=sha(activation_document),activation_id=a['activation_id'],epoch=m['epoch'],checkpoint=m['checkpoint']['id'],source_sha256=m['source_bundle']['sha256'],starts_at=m['start'],basis='unique-cheap-eligible-times-statistical-validity',unaudited_samples_claimed_verified=False)
 return m

def project(hourly_document,snapshot_documents,population_documents,opening_documents,registration_documents,completion_documents,activation_document,authority,*,fresh_registrations):
 """Authenticates original prospective openings and exact immutable score inputs.

Population documents include the whole prior audit cohort used by each snapshot;
completion documents are separate signed actual checkpoint-completion metadata.
No pending epoch and no historical strict epoch can earn through this bridge.
"""
 a=activation(activation_document,authority);h=signed(hourly_document,authority)
 need(h.get('version')=='continuous-hourly-weights-v1'and h.get('chain_transactions')is False,'explicit statistical hourly proposal')
 integer(h['cutoff'],'UTC cutoff');need(h['cutoff']%3600==0,'whole completed hour')
 expected=hourly_aggregate(snapshot_documents,authority,h['cutoff'])
 need(canonical(expected)==canonical(h),'exact original statistical hourly aggregate')
 epochs=h['epochs'];need(set(opening_documents)==set(epochs)==set(registration_documents)==set(completion_documents),'complete original hourly evidence')
 populations={};records=[]
 for doc in population_documents:
  p=signed(doc,authority);need(p.get('version')=='continuous-audit-population-v1'and type(p.get('eligible_evidence_ids'))is list,'explicit signed audit eligibility')
  m=signed(p['manifest_document'],authority);need(m['epoch']not in populations,'unique original population')
  from .continuous_audit_service import register_population
  pairs=[{k:r[k]for k in ('miner','commitment_sha256','batch_sha256','proof_sha256')}for r in p['records']if sha(r)in p['eligible_evidence_ids']]
  need(p==register_population(p['manifest_document'],p['receipts'],p['round'],p['committed_at'],authority,eligible_pairs=pairs),'original signed miner population and eligibility')
  populations[m['epoch']]=(doc,p,m);records.extend(p['records'])
 identities={}
 for doc in snapshot_documents:
  s=signed(doc,authority);epoch=s['epoch'];need(epoch in populations,'original epoch population')
  popdoc,p,m=populations[epoch]
  cohort=[r for r in records if r['round']<=s['round']and r['committed_at']<=h['cutoff']]
  from .continuous_audit_policy import population
  need(s['population_sha256']==sha(population(cohort))and s['eligible_evidence_ids']==p['eligible_evidence_ids'],'original full audit cohort and learner eligibility')
  need(s['checkpoint']==m['checkpoint']['id']and s['round']==p['round'],'original checkpoint and round')
  opening=signed(opening_documents[epoch],authority)
  need(opening.get('version')=='immutable-first-manifest-v1'and opening.get('epoch')==epoch and opening.get('first_manifest_sha256')==sha(p['manifest_document']),'immutable original first opening')
  need(m['start']<=opening['published_at']<m['deadline'],'first publication inside epoch')
  contract=m.get('continuous_reward_contract');need(type(contract)is dict,'prospective opening statistical contract required')
  stripped=copy.deepcopy(m);stripped.pop('continuous_reward_contract')
  regs=signed(registration_documents[epoch],authority)
  need(regs.get('epoch')==epoch and regs.get('snapshot_block')==m.get('registration_snapshot_block'),'original signed registration snapshot')
  original=inject_opening(stripped,activation_document,authority,regs['registrations'])
  need(original==m,'exact prospective statistical contract')
  completion=signed(completion_documents[epoch],authority)
  need(completion.get('epoch')==epoch and completion.get('round')==p['round']and completion.get('checkpoint')==m['checkpoint']['id']and completion.get('input_assurance')=='unaudited','actual learner completion provenance')
  finite(completion['completed_at'],'completion time');need(h['cutoff']-3600<completion['completed_at']<=h['cutoff']and completion['completed_at']>=m['deadline'],'actual completed epoch inside hourly window')
  need(type(completion.get('next_checkpoint'))is str and len(completion['next_checkpoint'])==64,'actual published successor')
  by_public={row['public_key']:(hotkey,row)for hotkey,row in regs['registrations'].items()}
  need(set(s['points'])<=set(by_public),'score identities from original registration')
  for miner in s['points']:
   hotkey,row=by_public[miner]
   if miner in identities:
    old_hotkey,old_row=identities[miner]
    need(old_hotkey==hotkey and all(old_row[k]==row[k]for k in ('uid','public_key')),'UID/key changed inside paid hour')
    if row['snapshot_block']>=old_row['snapshot_block']:identities[miner]=(hotkey,row)
   else:identities[miner]=(hotkey,row)
 units={};regs={};raw={}
 for miner,points in h['points'].items():
  hotkey,row=identities[miner];current=fresh_registrations.get(hotkey)
  need(current is not None and all(current.get(k)==row.get(k)for k in ('uid','public_key')),'stale identity denies whole hour')
  units[hotkey]=int((Decimal(str(points))*SCALE).to_integral_value(rounding=ROUND_FLOOR));regs[hotkey]=row;raw[hotkey]=points
 return dict(version=UNITS,window_end=h['cutoff'],points=units,registrations=regs,raw_statistical_points=raw,units_per_point=SCALE,rounding=a['rounding'],activation_sha256=sha(activation_document),original_hourly_sha256=sha(hourly_document),snapshot_sha256=[sha(d)for d in snapshot_documents],population_sha256=[sha(d)for d in population_documents],chain_executed=False,unaudited_samples_claimed_verified=False)

def submit_hour(adapter,document,authority,writer_receipt,*,now,boot_id,writer_pid,writer_ticks,execute=False):
 p=signed(document,authority)
 need(p.get('version')==UNITS and p.get('chain_executed')is False and p.get('units_per_point')==SCALE and p.get('unaudited_samples_claimed_verified')is False,'explicit statistical unit handoff')
 need(type(execute)is bool,'explicit execution flag');need(p['window_end']<=now,'completed hour only')
 need(all(type(v)is int and v>=0 for v in p['points'].values()),'nonnegative integer statistical units')
 if execute:writer_gate(writer_receipt,authority,now=now,boot_id=boot_id,writer_pid=writer_pid,writer_ticks=writer_ticks)
 return adapter.submit_hour(p['points'],p['registrations'],p['window_end'],execute=execute)

def emit_opening_documents(controller,manifest,registrations):
 """Only after final manifest PUT; durable immutable sidecars never backdated."""
 import json,time
 from pathlib import Path
 need(manifest.get('continuous_reward_contract',{}).get('version')==VERSION,'explicit statistical opening')
 first=controller.signed(manifest);path=Path(controller.state)/(manifest['epoch']+'-opening-attestation.json')
 if path.exists():
  opening=json.loads(path.read_bytes());o=signed(opening,controller.authority.id)
  need(o.get('version')=='immutable-first-manifest-v1'and o.get('epoch')==manifest['epoch']and o.get('first_manifest_sha256')==sha(first),'immutable statistical first opening')
  need(manifest['start']<=o['published_at']<manifest['deadline'],'actual original opening clock')
 else:
  now=time.time();need(manifest['start']<=now<manifest['deadline'],'expired missing statistical opening cannot be backdated')
  opening=controller.signed(dict(version='immutable-first-manifest-v1',epoch=manifest['epoch'],first_manifest_sha256=sha(first),published_at=now))
 registration=controller.signed(dict(epoch=manifest['epoch'],snapshot_block=manifest['registration_snapshot_block'],registrations=registrations))
 from .remote_backend import save
 for name,doc in [('first-signed-manifest',first),('opening-attestation',opening),('signed-registrations',registration)]:
  target=Path(controller.state)/(manifest['epoch']+'-'+name+'.json')
  if target.exists():need(canonical(json.loads(target.read_bytes()))==canonical(doc),'immutable statistical opening sidecar')
  else:save(target,doc)
 return first,opening,registration
