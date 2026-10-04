"""Authenticate epoch-scoped operational workers without changing scientific contracts."""
import json,math,re
from subnet.live_reward_bridge import signed,need,sha,canonical
FIELDS={'version','cutover_document_sha256','epoch','opening_manifest_sha256','live_reward_contract_sha256','source_sha256','source_files','runtime_versions','backend_profile','numerical_policy','checkpoint','existing_verifier_identities','additional_verifier_identities','role','effective_at','previous_supplement_sha256'}
def identities(values,count):
 need(isinstance(values,list)and len(values)==count and len(set(values))==count and all(isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)for v in values),'exact distinct worker identity list');return set(values)
def authenticate_supplements(documents,authority,cutover_sha256,existing):
 need(isinstance(documents,list)and len(documents)<=32,'bounded workforce supplements');result={}
 for document in documents:
  s=signed(document,authority);need(set(s)==FIELDS and s['version']=='operational-verifier-workforce-v1'and s['role']=='verify','exact operational verification supplement')
  need(s['cutover_document_sha256']==cutover_sha256,'original writer cutover binding')
  old=identities(s['existing_verifier_identities'],2)
  need(isinstance(s['additional_verifier_identities'],list)and 1<=len(s['additional_verifier_identities'])<=4,'one through four additional verifiers')
  new=identities(s['additional_verifier_identities'],len(s['additional_verifier_identities']))
  need(old==set(existing)and not old&new,'preserve original workers plus distinct verifiers')
  need(type(s['effective_at'])in(int,float)and math.isfinite(s['effective_at'])and s['effective_at']>0,'finite authorization time')
  need(isinstance(s['epoch'],str)and s['epoch'].startswith('nonpayable-live-reward-math-v1-'),'one named epoch authorization')
  for k in ('opening_manifest_sha256','live_reward_contract_sha256','source_sha256','cutover_document_sha256'):need(isinstance(s[k],str)and re.fullmatch('[0-9a-f]{64}',s[k]),'exact immutable digest')
  need(isinstance(s['source_files'],dict)and s['source_files']and all(isinstance(n,str)and n.startswith('subnet/')and n.endswith('.py')and '..'not in n and isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)for n,v in s['source_files'].items()),'exact runtime source filemap')
  need(isinstance(s['runtime_versions'],dict)and set(s['runtime_versions'])=={'torch','transformers','toploc'},'exact package pins')
  prior=result.get(s['epoch']);doc_sha=sha(document)
  if prior is None:
   need(s['previous_supplement_sha256'] is None,'first authorization has no parent')
   authorizations={worker:{'effective_at':s['effective_at'],'document_sha256':doc_sha}for worker in new}
  else:
   previous=prior['payload'];need(s['previous_supplement_sha256']==prior['document_sha256'],'exact immutable authorization chain parent')
   immutable=FIELDS-{'additional_verifier_identities','effective_at','previous_supplement_sha256'}
   need(all(canonical(s[k])==canonical(previous[k])for k in immutable),'unchanged authorization chain context')
   need(set(previous['additional_verifier_identities'])<new,'strict monotonic worker addition with no removal')
   need(s['effective_at']>=previous['effective_at'],'monotonic authorization time')
   authorizations=dict(prior['worker_authorizations'])
   authorizations.update({worker:{'effective_at':s['effective_at'],'document_sha256':doc_sha}for worker in new-set(authorizations)})
  result[s['epoch']]={'payload':s,'document_sha256':doc_sha,'worker_authorizations':authorizations}
 return result

def authorize_worker(worker,manifest,job,row,db,existing,supplements):
 entry=supplements.get(manifest['epoch'])
 if entry:
  s=entry['payload'];need(s['opening_manifest_sha256']==sha(manifest)and s['live_reward_contract_sha256']==sha(manifest.get('live_reward_contract')),'original opening and reward contract binding')
  need(s['source_sha256']==manifest['source_bundle']['sha256']and s['source_files']==job['source_files'],'exact original compute source')
  need(s['runtime_versions']==job['runtime_versions']and s['backend_profile']==manifest['backend_profile']and s['numerical_policy']==manifest['numerical_policy'],'original runtime and numerical checks')
  need(s['checkpoint']=={'id':manifest['checkpoint']['id'],'files':manifest['checkpoint']['files']},'original checkpoint identity and complete filemap')
 if worker in existing:return None
 need(entry is not None and worker in entry['payload']['additional_verifier_identities'],'approved verifier identity')
 s=entry['payload'];authorization=entry['worker_authorizations'][worker];events=[dict(r)for r in db.execute("select * from events where job=? and kind='claimed' order by sequence",(row['id'],))];claims=[]
 for event in events:
  detail=json.loads(event['detail'])
  if detail.get('worker')==worker:
   need(type(event['at'])in(int,float)and math.isfinite(event['at'])and event['at']>=authorization['effective_at'],'worker claim predates authorization')
   if detail.get('attempt')==row['attempt']:claims.append(event)
 need(len(claims)==1,'exact original worker lease claim')
 request=json.loads(row['report_request'])['payload'];need(claims[0]['at']<=request['at'],'report after authorized lease claim')
 return {'workforce_supplement_sha256':authorization['document_sha256'],'workforce_authorized_at':authorization['effective_at'],'worker_claimed_at':claims[0]['at']}
