"""750-specific CPU aggregation; scientific4db evaluator remains byte-identical."""
SOURCE='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'
POLICY={'version':'owned-cached-native-evaluation-v1','trust_scope':'operator-owned-process-native-grader','proof_reverification':False}
from subnet.backend_jobs import canonical
import hashlib
EXPERIMENT='owned-cached-native-all750-cap1024-v1'
def aggregate(plan,reports):
 if plan.get('dispatch_allowed')is not False or plan['experiment_id']!=EXPERIMENT or len(reports)!=24:raise ValueError('complete default-off750 group aggregation')
 expected={(i,s)for g in plan['groups']for i,s in zip(g['indices'],g['seeds'])};values=[v for r in reports for v in r['heldout']]
 if len(expected)!=750 or len(values)!=750 or any(r.get('heldout_failures')for r in reports):raise ValueError('incomplete native750 cohort; no score')
 keys=[(v['index'],v['seed'])for v in values]
 if set(keys)!=expected or len(set(keys))!=750 or any(v.get('native_graded')is not True or v.get('verified')is not False or v.get('proof_verification_performed')is not False or v.get('trust_scope')!=POLICY['trust_scope']or type(v['reward'])not in(int,float)or v['classification']not in('positive','negative')or v['reward']!=(1 if v['classification']=='positive'else 0)for v in values):raise ValueError('exact independently native graded750 outcomes')
 return dict(count=750,successes=sum(v['reward']==1 for v in values),mean_reward=sum(v['reward']for v in values)/750,execution_authenticated_here=False,requires_original_ROOT_job_report_ACK_authentication=True,experiment_id=EXPERIMENT,cohort_sha256=plan['cohort_sha256'])
def paired_summary(base,current):
 from subnet.owned_cached_evaluation import paired_summary as compare
 if len(base)!=750 or len(current)!=750:raise ValueError('all1500 genuine matched rows required')
 result=compare(base,current)
 result.update(experiment_id=EXPERIMENT,total_authenticated_rows=1500)
 return result
