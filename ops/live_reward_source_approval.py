"""Signed per-source approvals; v1 additive workforce, v2 explicit retirement."""
import copy,hashlib,math,re
from pathlib import Path
from subnet.live_reward_bridge import canonical,need,sha,signed
from subnet.source_bootstrap import admitted_files
from ops.live_reward_exporter import epoch_anchor
FIELDS={'version','original_cutover_sha256','previous_anchor_sha256','effective_at','source','runtime_source_files','runtime_versions','verifier_identities','anchor_document','epoch_prefix','registration_policy'}
V2_FIELDS=FIELDS|{'previous_authorization_sha256','retirements'}
RETIREMENT_FIELDS={'version','original_cutover_sha256','previous_anchor_sha256','previous_authorization_sha256','previous_source_sha256','verifier_identity','effective_at','reason','evidence_path','evidence_sha256'}

def replacement_roster(a,ids,prior_ids,retired,current,authority,cutover_document,previous_authorization):
 need(a['previous_authorization_sha256']==previous_authorization,'ordered previous workforce authorization')
 retirements=a['retirements'];need(isinstance(retirements,list)and len(retirements)<=(8 if a['version']=='live-compute-source-approval-v3' else 6),'bounded explicit retirements')
 removed=prior_ids-set(ids);records={}
 for document in retirements:
  r=signed(document,authority)
  need(set(r)==RETIREMENT_FIELDS and r['version']=='live-compute-verifier-retirement-v1','exact verifier retirement schema')
  need(r['original_cutover_sha256']==sha(cutover_document)and r['previous_anchor_sha256']==sha(current)and r['previous_authorization_sha256']==previous_authorization,'retirement original ordered authorization binding')
  need(r['previous_source_sha256']==signed(current,authority)['approved_compute_sources'][-1],'retirement previous source binding')
  identity=r['verifier_identity'];need(isinstance(identity,str)and identity in removed and identity not in records,'exact distinct removed verifier retirement')
  need(type(r['effective_at'])in(int,float)and math.isfinite(r['effective_at'])and r['effective_at']==a['effective_at'],'retirement exact prospective effective time')
  need(r['reason']in('node_terminated','signing_key_unavailable','key_compromised','operator_retired'),'explicit retirement reason')
  path=Path(r['evidence_path']);need(path.is_absolute()and path.resolve()==path and path.is_file()and not path.is_symlink(),'canonical retirement evidence file')
  digest=r['evidence_sha256'];need(isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest)and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'exact retirement evidence bytes')
  records[identity]=copy.deepcopy(document)
 need(set(records)==removed,'every removed identity needs explicit retirement evidence')
 need(not set(ids)&retired,'retired identity cannot silently return')
 return records

ANCHOR_ID=('version','netuid','owner_hotkey','effective_at','compute_epoch_prefix','live_epoch_prefix','cutover_id')

def apply_source_approvals(c,anchor_document,authority,cutover_document,documents):
 need(isinstance(documents,list)and len(documents)<=16,'bounded source approvals')
 result=copy.deepcopy(c);current=anchor_document;grants={};prior_ids=set(c['verifier_identities']) if documents else set()
 previous_authorization=sha(cutover_document);retired=set();v2_seen=False;v3_seen=False;previous_effective=None
 for document in documents:
  a=signed(document,authority);version=a.get('version')
  need((version=='live-compute-source-approval-v1'and set(a)==FIELDS)or(version in ('live-compute-source-approval-v2','live-compute-source-approval-v3')and set(a)==V2_FIELDS),'exact source approval schema')
  need(not v2_seen or version in ('live-compute-source-approval-v2','live-compute-source-approval-v3'),'v1 cannot follow explicit retirement semantics')
  need(not v3_seen or version=='live-compute-source-approval-v3','cannot downgrade eight-node authorization semantics')
  need(a['original_cutover_sha256']==sha(cutover_document),'original writer authority binding')
  need(a['previous_anchor_sha256']==sha(current),'ordered additive source approval')
  need(type(a['effective_at'])in(int,float)and math.isfinite(a['effective_at'])and a['effective_at']>0,'finite prospective authorization time')
  need(a['registration_policy']=='all_activated_subnet'and a['epoch_prefix']=='nonpayable-live-reward-math-v1-','open authenticated compute scope')
  old=signed(current,authority);new=signed(a['anchor_document'],authority)
  need(all(old.get(k)==new.get(k)for k in ANCHOR_ID),'unchanged original cutover identity')
  s=a['source'];need(isinstance(s,dict)and set(s)=={'sha256','archive_path','descriptor_path'},'exact approved source paths')
  digest=s['sha256'];need(isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest),'source digest')
  need(digest not in old['approved_compute_sources']and new['approved_compute_sources']==old['approved_compute_sources']+[digest],'exact additive source approval')
  for name in ('archive_path','descriptor_path'):
   p=Path(s[name]);need(p.is_absolute()and p.resolve()==p and p.is_file()and not p.is_symlink(),'canonical admitted source file')
  import json
  desc=signed(json.loads(Path(s['descriptor_path']).read_bytes()),authority);body=Path(s['archive_path']).read_bytes()
  need(desc['sha256']==digest and desc['size']==len(body)and hashlib.sha256(body).hexdigest()==digest,'signed exact source archive')
  members=admitted_files(body,desc)
  from ops.live_reward_writer import original_required_source_files
  required=original_required_source_files(members['subnet/backend_jobs.py']);actual={n:hashlib.sha256(b).hexdigest()for n,b in members.items()if n.startswith('subnet/')and n.count('/')==1 and n.endswith('.py')}
  need(set(required)<=set(actual),'required modules in complete runtime inventory')
  need(a['runtime_source_files']==actual,'exact complete runtime source inventory')
  need(a['runtime_versions']==c['runtime_versions'],'unchanged runtime versions')
  ids=a['verifier_identities']
  if version=='live-compute-source-approval-v1':
   need(isinstance(ids,list)and 4<=len(ids)<=6 and len(set(ids))==len(ids) and prior_ids<=set(ids)and all(isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)for v in ids),'bounded distinct reviewed verifier workforce must preserve prior identities')
  else:
   need(isinstance(ids,list)and 4<=len(ids)<=(8 if version=='live-compute-source-approval-v3' else 6) and all(isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)for v in ids)and len(set(ids))==len(ids),'bounded distinct active verifier workforce')
   need(a['effective_at']>=(old['effective_at']if previous_effective is None else previous_effective),'prospective replacement authorization time')
   records=replacement_roster(a,ids,prior_ids,retired,current,authority,cutover_document,previous_authorization)
   retired.update(records);v2_seen=True;v3_seen=v3_seen or version=='live-compute-source-approval-v3'
  result.setdefault('approved_sources',{})[digest]=copy.deepcopy(s)
  result.setdefault('approved_source_anchors',{})[digest]=copy.deepcopy(a['anchor_document'])
  for prior in result['approved_source_anchors']:
   epoch_anchor({'source_bundle':{'sha256':prior}},a['anchor_document'],authority,result['approved_source_anchors'])
  grants[digest]=dict(a,approval_document_sha256=sha(document));current=a['anchor_document'];prior_ids=set(ids);previous_authorization=sha(document);previous_effective=a['effective_at']
 result['_source_authorizations']=grants
 return result,current

def source_verifiers(c,manifest,job):
 grant=c.get('_source_authorizations',{}).get(manifest['source_bundle']['sha256'])
 if grant is None:return c['verifier_identities']
 need(type(manifest['start'])in(int,float)and math.isfinite(manifest['start'])and manifest['start']>=grant['effective_at'],'source authorization precedes opening')
 need(manifest['epoch'].startswith(grant['epoch_prefix']),'approved prospective epoch namespace')
 need(job['source_files']==grant['runtime_source_files']and job['runtime_versions']==grant['runtime_versions'],'exact new-source request pins')
 return grant['verifier_identities']
