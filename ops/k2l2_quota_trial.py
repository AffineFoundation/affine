"""CPU-only prospective K2/L2 qualification; never signs or dispatches jobs."""
import argparse,copy,hashlib,json
from pathlib import Path

SOURCE='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'
VERSION='qualification-only-K2-L2-same-context-v1'

def prepare(manifest):
    from subnet.backend_jobs import file_map
    from subnet.forced_sampling import validate
    from subnet.probability_artifacts import for_manifest
    if manifest.get('source_bundle',{}).get('sha256')!=SOURCE:raise ValueError('approved immutable4db source')
    if manifest.get('K')!=1 or manifest.get('L')!=1:raise ValueError('original K1 L1 baseline')
    if for_manifest(manifest) is None:raise ValueError('compact artifacts required')
    file_map(manifest['checkpoint']['files'])
    contract=validate(manifest['sampling_contract'])
    if contract['max_attempts']<16:raise ValueError('fresh ROOT signed max16 sampler context required')
    # Public draws hash this entire contract and epoch. Operational search
    # budgets vary; both arms retain exactly the same scientific draw context.
    trial=copy.deepcopy(manifest);trial.update(K=2,L=2,payable=False,capabilities={})
    trial['epoch']='nonpayable-K2L2-qualification-'+hashlib.sha256(manifest['epoch'].encode()).hexdigest()[:24]
    # This unsigned draft cannot reuse an expired production window. ROOT
    # supplies a fresh exact600-second window with its exclusive GPU scope.
    trial.pop('start',None);trial.pop('deadline',None)
    return dict(version=VERSION,execute_allowed=False,dispatch_allowed=False,qualification_only=True,
        production_cutover_allowed=False,source_sha256=SOURCE,
        baseline_manifest_sha256=hashlib.sha256(_canonical(manifest)).hexdigest(),
        prerequisite='genuine E21 K1 completion and fresh exclusive idle resource scope',
        manifest=trial,collection_seconds=600,search_budgets=[8,16],
        task_weight_rule='mean-pair-within-task-then-mean-task-within-group',
        paired_rollouts_rule='two zipped positive-negative pairs; no implicit cross product',
        reward_rule='one point per valid task; never pair count',native_outcome_claims='require independent approved native replay',
        arms=[dict(name='attempt-budget-'+str(n),search_budget=n,workspace_suffix='qualification-K2L2-attempt'+str(n),
            sampler_context_sha256=hashlib.sha256(_canonical(dict(contract=contract,epoch=trial['epoch'],checkpoint=trial['checkpoint']['id']))).hexdigest())for n in (8,16)])

def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def native_outcomes(batch,replay):
    """Check independently replayed native results, never trust uploaded labels.

    The operator must pin the actual approved grader/spec/tokenizer used by the
    replay callable. This API performs no model inference or proof admission.
    """
    rows=batch['rollouts']
    if len(rows)!=4:raise ValueError('four exact K2 L2 trajectories')
    seen=set();counts={'positive':0,'negative':0};verified=[]
    task_hashes=set()
    for row in rows:
        if row['index']!=batch['index'] or row['env_id']!=batch['env_id']:raise ValueError('same task')
        signature=hashlib.sha256(_canonical([dict(prompt=t['prompt'],output=t['output'])for t in row['turns']])).hexdigest()
        if signature in seen:raise ValueError('duplicate trajectory')
        seen.add(signature);task_hashes.add(row['task_hash'])
        result=replay(copy.deepcopy(row))
        if (result.get('done')is not True or result.get('task_hash')!=row['task_hash'] or
            result.get('classification')not in counts or type(result.get('reward'))not in (int,float) or
            result['reward']!=(1 if result['classification']=='positive'else 0) or
            type(row.get('reward'))not in (int,float) or row.get('classification')!=result['classification'] or row.get('reward')!=result['reward']):raise ValueError('independent native outcome mismatch')
        counts[result['classification']]+=1;verified.append(dict(trajectory_sha256=signature,native_result=result))
    if len(task_hashes)!=1 or counts!={'positive':2,'negative':2}:raise ValueError('same native task and K2 L2 quota')
    return dict(native_outcomes_checked=True,model_proofs_verified=False,valid_task_points=1,counts=counts,trajectories=verified)

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-signed-job',required=True);p.add_argument('--authority',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from subnet.backend_jobs import signed
    raw=Path(a.original_signed_job).read_bytes();job=signed(json.loads(raw),a.authority)
    if job['role']!='mine':raise ValueError('original approved miner job')
    plan=prepare(signed(job['manifest'],a.authority));plan['original_signed_job_sha256']=hashlib.sha256(raw).hexdigest()
    output=Path(a.output)
    with output.open('xb')as f:f.write(_canonical(plan))
    output.chmod(0o600);print(json.dumps(dict(output=str(output),qualification_only=True,dispatch_allowed=False)))

if __name__=='__main__':main()
