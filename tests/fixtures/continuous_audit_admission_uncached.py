# Frozen pre-cache admission behavior for full-output/signature-order regression.
# This fixture is test-only; it is never used by the runtime.
def _admit_report_batch(queue_rows,records,authority,verifiers,approved_sources,*,execution_evidence_policy,cutoff,historical_report_admission,defer_backend):
 """Authenticate original SQLite terminal report requests before estimating.

 approved_sources must come from actual authenticated ROOT source admission;
 its mapping binds complete executed runtime module hashes, not file labels.
 """
 historical=historical_report_workers(historical_report_admission)
 rows=population(records);lookup={(r['epoch'],r['miner'],r['batch_sha256']):r for r in rows};admissions={};deferred=[]
 def parsed(value):return json.loads(value)if type(value)is str else value
 def admit_one(queue):
  need(queue.get('status')=='complete'and queue.get('role')=='verify','actually completed verifier job')
  worker=queue['worker'];need(worker in verifiers or historical.get(worker,{}).get(queue.get('digest'))==queue.get('report_digest') and queue.get('report_digest')is not None,'admitted actual worker')
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
 for queue in queue_rows:
  try:admit_one(queue)
  except BackendEvidenceNotAdmitted as error:
   if not defer_backend:raise
   deferred.append(dict(job_sha256=error.job_sha256,source_sha256=error.source_sha256,outcome='infrastructure_deferred',validity_credit=False,fraud_claim=False))
 return admissions,deferred
