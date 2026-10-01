"""NEW original Tau2 inventory split by complete user-instruction scenario.

Untouched original tasks selected across disjoint instruction groups. Persona and initial-state
variants cannot cross the split. Private instructions remain operator-only.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from subnet.native_tau2_probe import REVISION,data_inventory,digest,sanitized_env
from subnet.storage import canonical

SPLIT='original-tau2-user-instruction-group-disjoint-v1'

def scenario_hash(task):
    scenario=task.get('user_scenario')
    if not isinstance(scenario,dict) or not isinstance(scenario.get('instructions'),dict) or not scenario['instructions']:
        raise ValueError('complete original user instruction scenario')
    return digest(scenario['instructions'])

def select(rows,count):
    if type(count) is not int or not 4<=count<=64 or count%2:raise ValueError('even bounded disjoint inventory')
    if not isinstance(rows,list):raise ValueError('original task rows')
    groups={};ids=set()
    for task in rows:
        if not isinstance(task,dict) or not isinstance(task.get('id'),str) or not task['id'] or task['id'] in ids:raise ValueError('unique original task IDs')
        ids.add(task['id']);group=scenario_hash(task);groups.setdefault(group,[]).append(task)
    if len(groups)<2:raise ValueError('insufficient distinct original scenario groups')
    ordered=list(groups);pivot=len(ordered)//2;mid=count//2
    def take(keys):
        selected=[];offset=0
        while len(selected)<mid:
            progressed=False
            for key in keys:
                if offset<len(groups[key]):selected.append(groups[key][offset]);progressed=True
                if len(selected)==mid:break
            if not progressed:raise ValueError('insufficient untouched tasks in scenario partition')
            offset+=1
        return selected
    chosen=take(ordered[:pivot])+take(ordered[pivot:])
    train={scenario_hash(t) for t in chosen[:mid]};heldout={scenario_hash(t) for t in chosen[mid:]}
    if train&heldout:raise ValueError('scenario split overlap')
    return chosen,len(groups)

def materialize(data,out,count=32):
    data=Path(data).resolve();out=Path(out).resolve()
    if (data/'.tau2_revision').read_text().strip()!=REVISION:raise ValueError('original Tau2 data revision')
    inventory=data_inventory(data)
    required={"tau2/domains/telecom/"+n for n in ("tasks_full.json","tasks.json","db.toml","user_db.toml","main_policy.md")}
    if not required<=set(inventory):raise ValueError('complete original telecom resources')
    code="from tau2.run import load_tasks;import json;excluded={t.id for t in load_tasks('telecom','base')};rows=[t.model_dump(mode='json') for t in load_tasks('telecom','full') if t.id not in excluded];print(json.dumps({'rows':rows,'pool_count':len(rows),'excluded_base_count':len(excluded)}))"
    result=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],env=sanitized_env(data),text=True,timeout=90))
    chosen,groups=select(result['rows'],count);mid=count//2
    private={'schema':'original-affine-tau2-telecom-private-tasks-v1','data_revision':REVISION,'data_inventory':inventory,'split_policy':SPLIT,'tasks':[{'index':i,'task_hash':digest(t),'task':t} for i,t in enumerate(chosen)]}
    public={'schema':'original-affine-tau2-telecom-public-task-commitments-v1','domain':'telecom','data_revision':REVISION,'data_inventory_sha256':digest(inventory),'split_policy':SPLIT,'mining_indices':list(range(mid)),'heldout_indices':list(range(mid,count)),'tasks':[{'index':i,'task_id':t['id'],'task_hash':digest(t),'scenario_group_sha256':scenario_hash(t)} for i,t in enumerate(chosen)]}
    report={'schema':1,'split_policy':SPLIT,'original_data_revision':REVISION,'pool_count':result['pool_count'],'excluded_base_count':result['excluded_base_count'],'pool_scenario_groups':groups,'count':count,'mining_indices':public['mining_indices'],'heldout_indices':public['heldout_indices'],'mining_group_count':len({scenario_hash(t) for t in chosen[:mid]}),'heldout_group_count':len({scenario_hash(t) for t in chosen[mid:]}),'scenario_group_overlap':0,'public_sha256':hashlib.sha256(canonical(public)).hexdigest(),'private_sha256':hashlib.sha256(canonical(private)).hexdigest(),'data_inventory_sha256':digest(inventory),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_task_bodies_preserved':True,'simulation_ran':False,'model_proofs':False,'optimizer_ran':False,'payable':False,'chain_transactions':False}
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    for name,value in [('private-tasks.json',private),('public-tasks.json',public),('materialization-report.json',report)]:
        p=out/name;p.write_bytes(canonical(value));p.chmod(0o600)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=32);a=p.parse_args();print(json.dumps(materialize(a.data,a.out,a.count)))
if __name__=='__main__':main()
