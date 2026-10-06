"""Default-off metadata plan for four separately authenticated native cohorts."""
import hashlib,json
from subnet.backend_jobs import canonical,signed
SOURCE='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'
POLICY={'version':'owned-cached-native-evaluation-v1','trust_scope':'operator-owned-process-native-grader','proof_reverification':False}
def prepare(original,authority,*,selection_seed='cached-heldout128-qualification-v1'):
 job=signed(original,authority);manifest=signed(job['manifest'],authority)
 if job.get('role')!='evaluate'or job.get('owned_evaluation_policy')!=POLICY or manifest['source_bundle']['sha256']!=SOURCE:raise ValueError('original qualified owned evaluator')
 if type(selection_seed)is not str or not selection_seed:raise ValueError('explicit fixed cohort selection seed')
 definition=next(e for e in manifest['environments']if e['env_id']=='affine_math');training=definition['indices'];population=definition['spec']['num_samples'];previous=job['heldout']
 if type(population)is not int or population!=7496 or len(training)!=6746 or len(set(training))!=len(training)or any(type(i)is not int or not 0<=i<population for i in training):raise ValueError('complete signed native training split')
 if len(previous)!=1 or len(previous[0]['indices'])!=32 or previous[0]['harness']!={'version':'text-tools-long-kv-v3','policy':'autoregressive','max_output_tokens':1024,'temperature':.7,'top_p':1.}:raise ValueError('existing1024 scientific generation profile')
 split=sorted(set(range(population))-set(training));excluded=set(previous[0]['indices'])
 if not excluded<=set(split):raise ValueError('existing heldout32 split binding')
 chosen=sorted(set(split)-excluded,key=lambda i:hashlib.sha256(canonical([selection_seed,i])).hexdigest())[:128]
 if len(chosen)!=128:raise ValueError('heldout-only128 population')
 groups=[dict(group=i,env_id='affine_math',indices=chosen[i*32:(i+1)*32],seeds=[20261002+n*1000 for n in chosen[i*32:(i+1)*32]],harness=previous[0]['harness'])for i in range(4)]
 return dict(version='owned-cached-heldout128-partitioned-plan-v1',dispatch_allowed=False,signing_allowed=False,source_sha256=SOURCE,experiment_id='owned-cached-native-heldout128-cap1024-v1',selection_seed=selection_seed,original_signed_job_sha256=hashlib.sha256(canonical(original)).hexdigest(),signed_training_split_sha256=hashlib.sha256(canonical(training)).hexdigest(),heldout_split_sha256=hashlib.sha256(canonical(split)).hexdigest(),cohort_sha256=hashlib.sha256(canonical(groups)).hexdigest(),groups=groups,checkpoint=manifest['checkpoint'],task_count=128,token_cap=1024,policy=POLICY,per_checkpoint_original_jobs=4,paired_original_jobs=8,aggregate_only_after_all_authentic_complete_ACKs=True,partial_is_infrastructure_failure_not_wrong_answer=True,production_observer_unchanged=True,model_retention_requirement='reviewed exclusive owned checkpoint lease until four original groups terminal and full durable ACKs; automatic final disposal before next parent; no external-cache deletion')
def aggregate(plan,reports):
 if plan['dispatch_allowed']is not False or len(reports)!=4:raise ValueError('default-off full cohort aggregation')
 expected={(i,s)for g in plan['groups']for i,s in zip(g['indices'],g['seeds'])};values=[v for r in reports for v in r['heldout']]
 if any(r.get('heldout_failures')for r in reports)or len(values)!=128:raise ValueError('incomplete native cohort; no score')
 keys=[(v['index'],v['seed'])for v in values]
 if set(keys)!=expected or len(set(keys))!=128 or any(v.get('native_graded')is not True or v.get('verified')is not False or v.get('proof_verification_performed')is not False or v.get('trust_scope')!=POLICY['trust_scope']or type(v['reward'])not in(int,float)or v['classification']not in('positive','negative')or v['reward']!=(1 if v['classification']=='positive'else 0)for v in values):raise ValueError('exact independently native-graded population')
 return dict(count=128,successes=sum(v['reward']==1 for v in values),mean_reward=sum(v['reward']for v in values)/128,execution_authenticated_here=False,requires_original_ROOT_job_report_ACK_authentication=True,experiment_id=plan['experiment_id'],cohort_sha256=plan['cohort_sha256'])
