"""Prospective asynchronous quality evidence; never reverify training samples.

No deployed controller selects this module. A hold is a recommendation tied to
an immutable branch and exact authenticated reports, never a silent rollback.
"""
import hashlib,math
from .storage import canonical
from .distributed_roles import authenticate
VERSION='asynchronous-training-quality-v1'
DEFAULT=dict(minimum_confirmed_tasks=128,severe_drop=.2,minimum_drop=.05,repeated_regressions=3)
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def wilson(k,n):
 z=1.96;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return center-radius,center+radius

def compare(before_document,after_document,training_document,authority,*,policy=None):
 p=dict(DEFAULT if policy is None else policy)
 if set(p)!=set(DEFAULT)or type(p['minimum_confirmed_tasks'])is not int or p['minimum_confirmed_tasks']<32 or type(p['repeated_regressions'])is not int or p['repeated_regressions']<2 or not 0<p['minimum_drop']<=p['severe_drop']<=1:raise ValueError('explicit bounded quality policy')
 b=authenticate(before_document,authority);a=authenticate(after_document,authority);t=authenticate(training_document,authority)
 if b['phase']!='before'or a['phase']!='after'or b['epoch']!=a['epoch']or t['epoch']!=a['epoch']or t['input_checkpoint']!=b['checkpoint']or t['output_checkpoint']!=a['checkpoint']:raise ValueError('same epoch checkpoint branch')
 if t.get('version')!='authenticated-training-quality-input-v1':raise ValueError('explicit operator-admitted training diagnostics')
 if b.get('status')!='complete'or a.get('status')!='complete':return dict(version=VERSION,status='unknown_infrastructure',epoch=a['epoch'],hold=False)
 if t['optimizer_step_after']<=t['optimizer_step_before']or a['public_optimizer_steps']!=t['optimizer_step_after']or b['public_optimizer_steps']!=t['optimizer_step_before']:raise ValueError('actual optimizer lineage')
 d=t['diagnostics'];margins=d['training_pair_margin_delta']
 if not margins or any(type(v)not in (float,int)or not math.isfinite(v)for v in margins):raise ValueError('finite actual margin evidence')
 for update in d['updates']:
  if not math.isfinite(update['loss'])or not math.isfinite(update['gradient_norm_before_clip']):raise ValueError('finite gradient/loss evidence')
 if len(b['records'])!=len(a['records']):raise ValueError('same evaluation suites')
 suites=[]
 for br,ar in zip(b['records'],a['records']):
  fields=('env_id','dataset_id','taskset_hash','heldout_indices','fixed_task_ids','harness_config','model_runtime_revision','requested_count')
  if any(br.get(k)!=ar.get(k)for k in fields):raise ValueError('same heldout tasks/grader/harness/runtime')
  n=br['requested_count']
  if n<=0 or br['completed_count']!=n or ar['completed_count']!=n or br.get('evaluation_failures')or ar.get('evaluation_failures'):return dict(version=VERSION,status='unknown_infrastructure',epoch=a['epoch'],hold=False)
  bk,ak=br['successes'],ar['successes']
  if type(bk)is not int or type(ak)is not int or not 0<=bk<=n or not 0<=ak<=n:raise ValueError('success counts')
  drop=(bk-ak)/n;confirmed=n>=p['minimum_confirmed_tasks']and drop>=p['minimum_drop']and wilson(ak,n)[1]<wilson(bk,n)[0]
  suites.append(dict(env_id=br['env_id'],count=n,before=bk,after=ak,drop=drop,confirmed_regression=confirmed,severe=confirmed and drop>=p['severe_drop']))
 return dict(version=VERSION,status='complete',epoch=a['epoch'],input_checkpoint=b['checkpoint'],output_checkpoint=a['checkpoint'],optimizer_step_before=t['optimizer_step_before'],optimizer_step_after=t['optimizer_step_after'],parent_state_sha256=t['parent_state_sha256'],output_state_sha256=t['output_state_sha256'],evidence_id=digest([before_document,after_document,training_document,p]),policy=p,suites=suites,mean_margin_delta=sum(margins)/len(margins),inference_weights_changed=t['inference_weights_changed'],numeric_training_checks=True,hold=any(s['severe']for s in suites),convergence_claimed=False)

def reconcile(observations):
 seen=set();prior=None;streak=0;hold=False
 for row in sorted(observations,key=lambda x:x.get('optimizer_step_after',-1)):
  if row['status']!='complete':continue
  if row['evidence_id']in seen:continue
  seen.add(row['evidence_id'])
  if prior is not None and (row['input_checkpoint']!=prior['output_checkpoint']or row['optimizer_step_before']!=prior['optimizer_step_after']or row['parent_state_sha256']!=prior['output_state_sha256']):raise ValueError('unbroken checkpoint and optimizer-state lineage')
  regressed=any(s['confirmed_regression']for s in row['suites']);streak=streak+1 if regressed else 0;hold=hold or row['hold']or streak>=row['policy']['repeated_regressions'];prior=row
 return dict(version=VERSION,hold_future_training=hold,hold_future_publication=hold,confirmed_regression_streak=streak,observed_updates=len(seen),action='operator_parent_checkpoint_and_optimizer_lineage_review'if hold else'continue_observation',automatic_rollback=False,convergence_claimed=False)

def evaluation_indices(reserved,mining,round_number,*,budget=128,full_every=24):
 """Prospective fixed public rotation; never select a mining task for eval."""
 if type(round_number)is not int or round_number<0 or type(budget)is not int or budget<1 or type(full_every)is not int or full_every<1:raise ValueError('bounded evaluation schedule')
 if len(set(reserved))!=len(reserved)or set(reserved)&set(mining)or not reserved:raise ValueError('independent reserved heldout inventory')
 rows=sorted(reserved)
 if (round_number+1)%full_every==0:return rows
 start=round_number*budget%len(rows)
 return [rows[(start+i)%len(rows)]for i in range(min(budget,len(rows)))]
