"""Narrow authenticated diagnostic deferral; original scientific reports stay intact.

This is an operator evidence interpretation overlay, never a verifier acceptance
rule. It must be authorized alongside the sole writer composition. Only exact
original invalid reports listed by job/report/request digest may be deferred.
"""
import hashlib,json,math,time

VERSION='mimo-exact-report-diagnostic-quarantine-v1'
REASON='cached-sampling-boundary-under-independent-diagnostic-review'

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def hexsha(x):return type(x)is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def parsed(x):return json.loads(x)if type(x)is str else x

def validate(p,*,authority,clock=time.time,check_time=True):
 fields={'version','authority','created_at','effective_cutoff','expires_at','entries'}
 need(type(p)is dict and set(p)==fields and p['version']==VERSION and p['authority']==authority,'exact diagnostic quarantine scope')
 for k in ('created_at','effective_cutoff','expires_at'):
  need(type(p[k])is int and p[k]>=0,'integer diagnostic validity window')
 need(p['effective_cutoff']%3600==0 and p['created_at']<=p['effective_cutoff']<p['expires_at']<=p['created_at']+86400,'bounded prospective hourly quarantine')
 if check_time:need(p['created_at']<=clock()+5<p['expires_at'],'diagnostic quarantine expired or not yet issued; retain last-best assessment')
 need(type(p['entries'])is list and 1<=len(p['entries'])<=32,'bounded exact disputed reports')
 seen=set()
 for e in p['entries']:
  need(type(e)is dict and set(e)=={'job_id','job_sha256','report_sha256','report_request_sha256','worker','epoch','miner','checkpoint','diagnostic_sha256','reason'},'exact diagnostic entry')
  need(all(hexsha(e[k])for k in ('job_sha256','report_sha256','report_request_sha256','worker','miner','checkpoint','diagnostic_sha256')),'exact evidence digests and identities')
  need(type(e['job_id'])is str and 1<=len(e['job_id'])<=200 and type(e['epoch'])is str and 1<=len(e['epoch'])<=200,'bounded original identity')
  need(e['reason']==REASON,'explicit numerical diagnostic reason')
  need(e['job_sha256']not in seen,'one diagnostic disposition per original job');seen.add(e['job_sha256'])
 return p


def filter_admissions(admissions,rows,p,*,cutoff,clock=time.time):
 """Call only after original signature/source/receipt admission has succeeded."""
 validate(p,authority=p['authority'],clock=clock)
 need(type(cutoff)is int and cutoff%3600==0,'original hourly evidence cutoff')
 if cutoff<p['effective_cutoff']:return admissions,[]
 need(cutoff<p['expires_at'],'expired diagnostic cutoff; retain last-best assessment')
 by_digest={r.get('digest'):r for r in rows};kept=dict(admissions);deferred=[]
 for e in p['entries']:
  key=e['job_sha256']
  if key not in admissions:continue # Not-yet-admitted or post-cutoff evidence stays original.
  q=by_digest.get(key);need(q is not None,'original admitted queue row missing')
  report=parsed(q['report']);request=parsed(q['report_request']);job=parsed(q['envelope'])['payload']
  need(q['id']==e['job_id'] and q['worker']==e['worker'] and q['report_digest']==e['report_sha256'] and digest(report)==e['report_sha256'] and digest(request)==e['report_request_sha256'],'diagnostic original report/request binding')
  need(digest(job)==key and report['epoch']==e['epoch'] and report['checkpoint']==e['checkpoint'],'diagnostic original job/epoch/checkpoint')
  observations=admissions[key]['observations']
  need(len(observations)==1 and observations[0]['miner']==e['miner'] and observations[0]['epoch']==e['epoch'] and observations[0]['checkpoint']==e['checkpoint'] and observations[0]['outcome']=='confirmed_invalid','quarantine only one exact original invalid group')
  need(admissions[key]['verifier']==e['worker'] and admissions[key]['original_report_request_sha256']==e['report_request_sha256'],'authenticated original worker request')
  del kept[key]
  deferred.append(dict(job_sha256=key,original_report_sha256=e['report_sha256'],original_report_request_sha256=e['report_request_sha256'],diagnostic_sha256=e['diagnostic_sha256'],miner=e['miner'],epoch=e['epoch'],reason=e['reason'],expires_at=p['expires_at'],fraud_claim=False,validity_credit=False,counts_as_numerical_coverage_failure=False,original_evidence_preserved=True))
 return kept,deferred


def install(bridge,p,*,authority,clock=time.time):
 """Overlay isolated MiMo reader only; historical reducer/writer stay untouched."""
 validate(p,authority=authority,clock=clock,check_time=False)
 p=json.loads(canonical(p))
 need(not getattr(bridge,'_diagnostic_quarantine_installed',False),'one explicit diagnostic overlay')
 original_admit=bridge.admit_queue_reports;original_evidence=bridge.evidence
 active=[]
 def admit(rows,*args,**kwargs):
  admitted,refused=original_admit(rows,*args,**kwargs)
  kept,deferred=filter_admissions(admitted,rows,p,cutoff=kwargs['cutoff'],clock=clock)
  active.extend(deferred)
  return kept,refused
 def evidence(*args,**kwargs):
  active.clear();value=original_evidence(*args,**kwargs)
  value['deferred'].extend(active)
  value['evidence_hashes']['mimo_diagnostic_quarantine_sha256']=digest(p)
  return value
 bridge.admit_queue_reports=admit;bridge.evidence=evidence;bridge._diagnostic_quarantine_installed=True
 return dict(policy_sha256=digest(p),entries=len(p['entries']),expires_at=p['expires_at'])
