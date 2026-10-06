"""Prospective immutable hourly scores from independently authenticated audits.

No inference, blockchain calls, or unaudited-sample validity claim. Caller must
admit the immutable commitment population and verifier execution/source pins.
"""
import hashlib,json,math
from .distributed_roles import authenticate
LEGACY_VERSION='continuous-probabilistic-audit-v1'
VERSION='continuous-probabilistic-audit-v2'
RESOLUTION_VERSION='continuous-probabilistic-audit-v3'
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
digest=lambda v:hashlib.sha256(canonical(v)).hexdigest()
def need(v,message):
 if not v:raise ValueError(message)
def integer(v,minimum,maximum,name):need(type(v)is int and minimum<=v<=maximum,name);return v
def finite(v,low,high,name):need(type(v)in(int,float)and math.isfinite(v)and low<=v<=high,name);return float(v)
def valid_digest(v):return type(v)is str and len(v)==64 and all(c in '0123456789abcdef'for c in v)
def policy(value):
 fields={'version','recent_epochs','decay','prior_alpha','prior_beta','invalid_multiplier','zero_epoch_after','blacklist_after','blacklist_epochs'}
 need(type(value)is dict and set(value)==fields and value['version']in(VERSION,LEGACY_VERSION,RESOLUTION_VERSION),'exact continuous audit policy')
 result=dict(value);integer(value['recent_epochs'],1,128,'recent cohort window');finite(value['decay'],0.01,1,'cohort decay')
 for name in ('prior_alpha','prior_beta'):finite(value[name],0.01,100,'bounded prior')
 finite(value['invalid_multiplier'],0,1,'invalid reward multiplier')
 for name in ('zero_epoch_after','blacklist_after'):integer(value[name],0,10000,name)
 integer(value['blacklist_epochs'],0,128,'blacklist duration');return result

def population(records,*,ordered=False):
 """All rows are immutable cheap-admitted submissions, not verified samples."""
 need(type(records)is list and len(records)<=1000000,'bounded immutable population');seen=set();result=[]
 for row in records:
  need(type(row)is dict and set(row)=={'epoch','round','checkpoint','miner','env_id','index','batch_sha256','proof_sha256','commitment_sha256','verifier_contract_sha256','committed_at'},'exact committed audit row')
  for field in ('checkpoint','miner','batch_sha256','proof_sha256','commitment_sha256','verifier_contract_sha256'):need(valid_digest(row[field]),'committed digest')
  integer(row['round'],0,2**31-1,'epoch round');integer(row['index'],0,2**31-1,'task index');finite(row['committed_at'],0,2**53,'completion time')
  need(type(row['epoch'])is str and 0<len(row['epoch'])<=200 and type(row['env_id'])is str and 0<len(row['env_id'])<=100,'epoch/environment')
  key=(row['epoch'],row['miner'],row['batch_sha256']);need(key not in seen,'duplicate committed batch');seen.add(key);result.append(dict(row))
 return sorted(result,key=lambda r:(r['round'],r['epoch'],r['miner'],r['batch_sha256']))if ordered else result

def random_selection(records,seed,count,already=()):
 """Unpredictable seed is committed only after this immutable population closes."""
 rows=population(records);need(valid_digest(seed),'postcommit randomness');integer(count,0,1000000,'audit count')
 excluded=set(already);need(all(valid_digest(x)for x in excluded),'previous audited evidence ids')
 ordered=sorted((row for row in rows if digest(row)not in excluded),key=lambda row:digest(dict(domain=VERSION,seed=seed,row=row)))
 return ordered[:count]

def observations(envelopes,records,verifiers,cutoff,*,admitted_jobs=None,adjudications=(),authority=None,numerical_resolution_policy=None,expected_numerical_resolution_policy_sha256=None,numerical_reference_archives=()):
 """Reject substitution/conflicts; repeats never increase confidence or penalties."""
 rows=population(records);lookup={(r['epoch'],r['miner'],r['batch_sha256']):r for r in rows};result={}
 finite(cutoff,0,2**53,'immutable hourly cutoff');need(type(envelopes)is list and len(envelopes)<=1000000,'bounded audit evidence')
 inputs=[]
 for envelope in envelopes:
  if type(envelope)is dict and set(envelope)=={'admitted_queue_job_sha256'}:
   admission=(admitted_jobs or {}).get(envelope['admitted_queue_job_sha256']);need(admission is not None,'original admitted queue pointer')
   inputs.extend((admission['verifier'],p)for p in admission['observations'])
  else:
   signer=envelope.get('signer');need(signer in verifiers,'admitted verifier identity');inputs.append((signer,authenticate(envelope,signer)))
 for signer,p in inputs:
  need(signer in verifiers,'admitted verifier identity')
  fields={'version','epoch','checkpoint','miner','batch_sha256','commitment_sha256','verifier_contract_sha256','outcome','completed_at','job_sha256'}
  need(set(p)==fields and p['version']=='continuous-audit-observation-v1','exact audit observation')
  finite(p['completed_at'],0,2**53,'audit completion');need(valid_digest(p['job_sha256']),'original audit execution request')
  need(p['outcome']in('verified_valid','confirmed_invalid','numerical_ambiguous','infrastructure_error'),'typed audit result')
  row=lookup.get((p['epoch'],p['miner'],p['batch_sha256']));need(row is not None,'audit original commitment population')
  need(all(p[k]==row[k]for k in ('epoch','checkpoint','miner','batch_sha256','commitment_sha256','verifier_contract_sha256')),'audit checkpoint/version/commitment binding')
  need(p['completed_at']>=row['committed_at'],'postcommit audit completion')
  if p['completed_at']>cutoff:continue
  admission=(admitted_jobs or {}).get(p['job_sha256'])
  need(admission is not None and admission['verifier']==signer,'original admitted execution request required')
  matches=[o for o in admission['observations']if all(o.get(k)==p[k]for k in fields)]
  need(len(matches)==1,'original queued audit result binding')
  key=digest(row);old=result.get(key)
  if old is not None:
   if old['outcome']==p['outcome']:continue
   # Failed infrastructure is not a scientific assertion. An authenticated
   # retry can resolve it without fabricating a fraud adjudication.
   if p['outcome']=='infrastructure_error':continue
   if old['outcome']=='infrastructure_error':
    result[key]=dict(p,round=row['round'],evidence_id=key,verifier=signer);continue
   resolutions=[authenticate(a,authority)for a in adjudications]if authority else []
   expected=dict(version='continuous-audit-adjudication-v1',evidence_id=key,original_job_sha256=old['job_sha256'],reference_job_sha256=p['job_sha256'],outcome=p['outcome'])
   need(expected in resolutions and old['outcome']=='numerical_ambiguous'and p['outcome']in('verified_valid','confirmed_invalid'),'conflicting authenticated audits require explicit reference adjudication')
  result[key]=dict(p,round=row['round'],evidence_id=key,verifier=signer)
 from .numerical_resolution import apply
 return apply(list(result.values()),rows,admitted_jobs or {},authority=authority,cutoff=cutoff,policy_document=numerical_resolution_policy,expected_policy_sha256=expected_numerical_resolution_policy_sha256,reference_archives=numerical_reference_archives)

def snapshot(records,envelopes,verifiers,*,epoch,round,checkpoint,cutoff,audit_policy,admitted_jobs=None,adjudications=(),authority=None,eligible_evidence_ids=None,numerical_resolution_policy=None,expected_numerical_resolution_policy_sha256=None,numerical_reference_archives=()):
 """Validity estimate can decrease; current cohort bounds historical reputation.

 The caller authenticates immutable opening/policy and signs this exact result.
 No pending/ambiguous/infra observation counts as valid or fraudulent.
 """
 p=policy(audit_policy);rows=population(records,ordered=p['version']in(VERSION,RESOLUTION_VERSION));integer(round,0,2**31-1,'snapshot round');need(valid_digest(checkpoint),'current immutable checkpoint')
 if expected_numerical_resolution_policy_sha256 is not None:need(p['version']==RESOLUTION_VERSION,'numerical resolution requires explicit UNKNOWN coverage policy')
 need(all(r['round']<=round and r['committed_at']<=cutoff for r in rows),'future or postcutoff committed population')
 current=[r for r in rows if r['epoch']==epoch];need(all(r['round']==round and r['checkpoint']==checkpoint for r in current),'current epoch/checkpoint binding')
 audits=observations(envelopes,rows,verifiers,cutoff,admitted_jobs=admitted_jobs,adjudications=adjudications,authority=authority,numerical_resolution_policy=numerical_resolution_policy,expected_numerical_resolution_policy_sha256=expected_numerical_resolution_policy_sha256,numerical_reference_archives=numerical_reference_archives);miners=sorted({r['miner']for r in current});points={};details={}
 eligible_set=None if eligible_evidence_ids is None else set(eligible_evidence_ids)
 if eligible_set is not None:need(all(valid_digest(v)for v in eligible_set)and eligible_set<=set(digest(r)for r in current),'actual admitted eligible population subset')
 score_rows=[r for r in current if eligible_set is None or digest(r)in eligible_set]
 counts={}
 for row in score_rows:
  key=(row['env_id'],row['index']);counts.setdefault(key,set()).add(row['miner'])
 for miner in miners:
  eligible=len({(r['env_id'],r['index'])for r in score_rows if r['miner']==miner and len(counts[(r['env_id'],r['index'])])==1})
  recent=[o for o in audits if o['miner']==miner and 0<=round-o['round']<p['recent_epochs'] and o['outcome']in('verified_valid','confirmed_invalid')]
  alpha=float(p['prior_alpha']);beta=float(p['prior_beta']);ca=alpha;cb=beta;invalid_current=0;invalid_recent=0;latest_bad_round=None
  for o in recent:
   weight=p['decay']**(round-o['round'])
   if o['outcome']=='verified_valid':alpha+=weight
   else:beta+=weight;invalid_recent+=1;latest_bad_round=max(o['round'],latest_bad_round if latest_bad_round is not None else o['round'])
   if o['epoch']==epoch and o['checkpoint']==checkpoint:
    if o['outcome']=='verified_valid':ca+=1
    else:cb+=1;invalid_current+=1
  overall=alpha/(alpha+beta);cohort=ca/(ca+cb);probability=min(overall,cohort)
  blacklisted=bool(p['blacklist_after'] and invalid_recent>=p['blacklist_after'] and latest_bad_round is not None and round-latest_bad_round<p['blacklist_epochs'])
  multiplier=0. if blacklisted or p['zero_epoch_after']and invalid_current>=p['zero_epoch_after'] else p['invalid_multiplier']**invalid_current
  coverage=1.;coverage_details={}
  if p['version']==RESOLUTION_VERSION:
   scientific=[o for o in audits if o['miner']==miner and 0<=round-o['round']<p['recent_epochs'] and o['outcome']!='infrastructure_error']
   resolved=sum(p['decay']**(round-o['round'])for o in scientific if o['outcome']!='numerical_ambiguous')
   unknown=sum(p['decay']**(round-o['round'])for o in scientific if o['outcome']=='numerical_ambiguous')
   current_scientific=[o for o in scientific if o['epoch']==epoch and o['checkpoint']==checkpoint]
   current_resolved=sum(o['outcome']!='numerical_ambiguous'for o in current_scientific)
   current_unknown=sum(o['outcome']=='numerical_ambiguous'for o in current_scientific)
   recent_coverage=resolved/(resolved+unknown)if resolved+unknown else 1.
   current_coverage=current_resolved/(current_resolved+current_unknown)if current_resolved+current_unknown else 1.
   coverage=min(current_coverage,recent_coverage)
   coverage_details=dict(resolution_coverage_factor=coverage,current_resolution_coverage=current_coverage,recent_resolution_coverage=recent_coverage,resolved_current=current_resolved,numerical_ambiguous_current=current_unknown,resolved_recent_weight=resolved,numerical_ambiguous_recent_weight=unknown,unresolved_is_fraud=False,infrastructure_counted_in_coverage=False)
  points[miner]=eligible*probability*multiplier*coverage
  details[miner]=dict(unique_eligible_batches=eligible,validity_probability=probability,recent_posterior_mean=overall,current_cohort_posterior_mean=cohort,confirmed_invalid_current=invalid_current,confirmed_invalid_recent=invalid_recent,reward_multiplier=multiplier,blacklisted=blacklisted,**coverage_details)
 total=sum(points.values());weights={m:(v/total if total else 0.)for m,v in points.items()}
 result=dict(version=p['version'],epoch=epoch,round=round,checkpoint=checkpoint,cutoff=cutoff,policy=p,population_sha256=digest(rows),eligible_evidence_ids=sorted(eligible_set)if eligible_set is not None else None,evidence_ids=sorted(o['evidence_id']for o in audits),miners=details,points=points,weights=weights,training_waits_for_audits=False,unaudited_samples_claimed_verified=False)
 if expected_numerical_resolution_policy_sha256 is not None:
  result['numerical_resolution_policy_sha256']=expected_numerical_resolution_policy_sha256
  result['numerical_resolution_observations']=[dict(evidence_id=o['evidence_id'],original_job_sha256=o['job_sha256'],original_outcome=o['original_outcome'],outcome=o['outcome'],original_observation_sha256=o['original_observation_sha256'],review_sha256=o['numerical_resolution_review_sha256'],sampler_and_grader_completion_claimed=False)for o in audits if 'numerical_resolution_review_sha256'in o]
 return result

def verifier_contract(manifest):
 """A change of source, sampler, numerics or runtime starts another cohort."""
 fields=('sampling_contract','sampling_source_hash','model_runtime_revision','backend_profile','numerical_policy','source_bundle')
 need(all(k in manifest for k in fields),'complete verifier execution contract')
 return digest({k:manifest[k]for k in fields})

class BackendEvidenceNotAdmitted(ValueError):
 """Authenticated report has no prospective authorization for its backend claim."""
 def __init__(self,job,source):
  self.job_sha256=digest(job);self.source_sha256=source
  super().__init__('authenticated backend evidence not prospectively admitted')

def admit_queue_reports(queue_rows,records,authority,verifiers,approved_sources,*,execution_evidence_policy=None,cutoff=None):
 """Authenticate original SQLite terminal report requests before estimating.

 approved_sources must come from actual authenticated ROOT source admission;
 its mapping binds complete executed runtime module hashes, not file labels.
 """
 rows=population(records);lookup={(r['epoch'],r['miner'],r['batch_sha256']):r for r in rows};admissions={}
 def parsed(value):return json.loads(value)if type(value)is str else value
 for queue in queue_rows:
  need(queue.get('status')=='complete'and queue.get('role')=='verify','actually completed verifier job')
  worker=queue['worker'];need(worker in verifiers,'admitted actual worker')
  job=authenticate(parsed(queue['envelope']),authority);manifest=authenticate(job['manifest'],authority);report=parsed(queue['report']);request=authenticate(parsed(queue['report_request']),worker)
  need(job['role']=='verify'and job['job_id']==queue['id']and digest(job)==queue['digest'],'original queue request digest')
  need(digest(report)==queue['report_digest']and request.get('action')=='report'and request.get('job_id')==job['job_id']and request.get('token')==queue['token']and request.get('report')==report,'original worker terminal report request')
  need(report.get('success')is True and report.get('role')=='verify'and report.get('job_id')==job['job_id']and report.get('job_sha256')==digest(job)and report.get('operator')==authority and report.get('epoch')==manifest['epoch']and report.get('checkpoint')==manifest['checkpoint']['id'],'executed audit identity/checkpoint')
  source=manifest['source_bundle']['sha256'];pins=approved_sources.get(source);need(type(pins)is dict and job.get('source_files')and all(pins.get(k)==v for k,v in job['source_files'].items()),'admitted executed source pins')
  expected_enforced=True
  if execution_evidence_policy is not None:
   ep=execution_evidence_policy
   need(type(ep)is dict and set(ep)=={'version','effective_cutoff','sources'}and ep['version']in('explicit-backend-execution-evidence-v1','explicit-backend-execution-evidence-v2'),'explicit operator execution evidence policy')
   finite(ep['effective_cutoff'],0,2**53,'prospective execution evidence cutoff');need(type(cutoff)in(int,float)and cutoff>=ep['effective_cutoff'],'prospective execution evidence cutoff not reached')
   entry=ep['sources'].get(source)
   if entry is not None:
    need(type(entry)is dict and set(entry)==({'backend','backend_module_sha256','model_runtime_revision','backend_profile','numerical_policy','runtime_versions','execution_resources_enforced'}|({'effective_cutoff'}if ep['version']=='explicit-backend-execution-evidence-v2'else set())),'exact admitted backend evidence scope')
    need(entry['backend']=='standard-backend-no-os-resource-enforcement-v1'and entry['execution_resources_enforced']is False and pins.get('subnet/backend_jobs.py')==entry['backend_module_sha256']and job['source_files']==pins,'complete exact standard backend source')
    need(manifest['model_runtime_revision']==entry['model_runtime_revision']and manifest['backend_profile']==entry['backend_profile']and manifest['numerical_policy']==entry['numerical_policy']and job['runtime_versions']==entry['runtime_versions'],'exact admitted backend runtime/profile/numerical scope')
    entry_cutoff=entry.get('effective_cutoff',ep['effective_cutoff']);finite(entry_cutoff,ep['effective_cutoff'],2**53,'prospective per-source backend admission cutoff')
    if cutoff>=entry_cutoff:expected_enforced=False
  need(report.get('source_files')==job['source_files']and all(report.get('runtime_versions',{}).get(k)==v for k,v in job['runtime_versions'].items())and report.get('backend_profile')==manifest['backend_profile']and report.get('numerical_policy')==manifest['numerical_policy'],'actual runtime/profile/numerical/source evidence')
  if report.get('execution_resources_enforced')is False and expected_enforced is True:raise BackendEvidenceNotAdmitted(job,source)
  need(report.get('execution_resources_enforced')is expected_enforced,'actual backend resource enforcement evidence')
  contract=verifier_contract(manifest);completed=finite(report['completed_at'],0,2**53,'original report completion');observed=[];native={}
  audits=report.get('audits');need(type(audits)is list and len(audits)==len(job['submissions']),'original full child report population')
  for obj,audit in zip(job['submissions'],audits):
   ref=obj['commitment_ref'];row=lookup.get((manifest['epoch'],ref['miner'],ref['batch_sha256']));need(row is not None and row['proof_sha256']==obj['sha256']and row['commitment_sha256']==ref['commitment_sha256']and row['checkpoint']==manifest['checkpoint']['id']and row['verifier_contract_sha256']==contract,'audit immutable committed child/execution cohort')
   need(audit.get('submission_sha256')==row['proof_sha256']and audit.get('epoch')==manifest['epoch'],'original audited proof digest')
   outcomes=audit.get('outcomes');need(type(outcomes)is list and len(outcomes)==1,'single committed batch outcome');o=outcomes[0]
   if o.get('fully_audited')is True and type(o.get('valid'))is bool and o['valid']:outcome='verified_valid'
   elif o.get('valid')is False and (o.get('fully_audited')is True and o.get('failure_kind')=='confirmed_invalid' or o.get('failure_kind')=='structural_invalid'):outcome='confirmed_invalid'
   elif o.get('valid')is None and o.get('failure_kind')=='numerical_ambiguous':outcome='numerical_ambiguous'
   else:outcome='infrastructure_error'
   observation=dict(version='continuous-audit-observation-v1',epoch=row['epoch'],checkpoint=row['checkpoint'],miner=row['miner'],batch_sha256=row['batch_sha256'],commitment_sha256=row['commitment_sha256'],verifier_contract_sha256=contract,outcome=outcome,completed_at=completed,job_sha256=digest(job));observed.append(observation)
   native[digest(observation)]=dict(reason=o.get('reason'),failure_kind=o.get('failure_kind'),fully_audited=o.get('fully_audited'),artifact_sha256=obj['sha256'])
  key=digest(job);value=dict(verifier=worker,observations=observed,original_report_request_sha256=digest(parsed(queue['report_request'])),original_report_sha256=digest(report),source_sha256=source,native_observations=native)
  need(key not in admissions or admissions[key]==value,'conflicting original queued job');admissions[key]=value
 return admissions

def admit_artifact_failures(documents,records,authority):
 """ROOT storage evidence, not a claim that a GPU evaluated the model."""
 lookup={digest(r):r for r in population(records)};result={}
 for document in documents:
  p=authenticate(document,authority)
  need(set(p)=={'version','row','outcome','completed_at','reason','original_selection_sha256','scientific_model_execution_claim'}and p['version']=='continuous-artifact-capture-failure-v1'and p['outcome']=='confirmed_invalid'and p['scientific_model_execution_claim']is False,'original operator invalid artifact evidence')
  row=lookup.get(digest(p['row']));need(row is not None and row==p['row']and valid_digest(p['original_selection_sha256']),'original immutable invalid artifact selection')
  finite(p['completed_at'],row['committed_at'],2**53,'actual invalid artifact observation');key=digest(document)
  observation=dict(version='continuous-audit-observation-v1',epoch=row['epoch'],checkpoint=row['checkpoint'],miner=row['miner'],batch_sha256=row['batch_sha256'],commitment_sha256=row['commitment_sha256'],verifier_contract_sha256=row['verifier_contract_sha256'],outcome='confirmed_invalid',completed_at=p['completed_at'],job_sha256=key)
  result[key]=dict(verifier=authority,observations=[observation],original_report_request_sha256=key,scientific_model_execution_claim=False)
 return result


def hourly_aggregate(documents,authority,cutoff):
 """Normalize summed raw points once; never average already normalized weights."""
 finite(cutoff,0,2**53,'hourly cutoff');need(cutoff%3600==0,'whole UTC hourly cutoff')
 points={};epochs=[];policies=[]
 for document in documents:
  result=authenticate(document,authority)
  need(result.get('version')in(VERSION,LEGACY_VERSION,RESOLUTION_VERSION) and result.get('cutoff')==cutoff,'original same-cutoff epoch snapshot')
  need(result['epoch']not in epochs,'epoch can earn once in hourly aggregate');epochs.append(result['epoch']);policies.append(digest(policy(result['policy'])))
  for miner,value in result['points'].items():
   need(valid_digest(miner),'hourly miner identity');finite(value,0,1e6,'bounded raw epoch points');points[miner]=points.get(miner,0.)+value
 need(len(set(policies))<=1,'single prospective penalty policy per hourly snapshot')
 total=sum(points.values())
 return dict(version='continuous-hourly-weights-v1',cutoff=cutoff,epochs=sorted(epochs),epoch_snapshot_sha256=sorted(digest(d)for d in documents),snapshot_bindings=[dict(epoch=d['payload']['epoch'],round=d['payload']['round'],checkpoint=d['payload']['checkpoint'],population_sha256=d['payload']['population_sha256'],signed_snapshot_sha256=digest(d))for d in documents],points=points,weights={m:v/total if total else 0. for m,v in points.items()},chain_transactions=False)
