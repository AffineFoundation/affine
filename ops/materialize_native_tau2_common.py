"""Operator-only original Tau2 task inventory for prospective common role epochs.

Private task/user/grader fields stay in a mode-600 collection. Public definitions
contain identifiers and commitments only; this does not execute a simulation.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from subnet.native_tau2_probe import REVISION,data_inventory,digest,sanitized_env
from subnet.storage import canonical

def materialize(data,out,count=32):
    data=Path(data).resolve();out=Path(out).resolve()
    if type(count) is not int or not 4<=count<=64 or count%2:raise ValueError('even bounded task inventory')
    if (data/'.tau2_revision').read_text().strip()!=REVISION:raise ValueError('original Tau2 data revision')
    inventory=data_inventory(data)
    required={"tau2/domains/telecom/"+n for n in ("tasks_full.json","tasks.json","db.toml","user_db.toml","main_policy.md")}
    if not required<=set(inventory):raise ValueError("complete original telecom resources")
    code="from tau2.run import load_tasks;import json;excluded={t.id for t in load_tasks('telecom','base')};rows=[t.model_dump(mode='json') for t in load_tasks('telecom','full') if t.id not in excluded];print(json.dumps({'rows':rows[:"+str(count)+"],'pool_count':len(rows),'excluded_base_count':len(excluded)}))"
    result=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],env=sanitized_env(data),text=True,timeout=90))
    if len(result['rows'])!=count or len({r['id'] for r in result['rows']})!=count:raise ValueError('unique original task inventory')
    out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    private={'schema':'original-affine-tau2-telecom-private-tasks-v1','data_revision':REVISION,'data_inventory':inventory,'tasks':[{'index':i,'task_hash':digest(t),'task':t} for i,t in enumerate(result['rows'])]}
    public={'schema':'original-affine-tau2-telecom-public-task-commitments-v1','domain':'telecom','data_revision':REVISION,'data_inventory_sha256':digest(inventory),'mining_indices':list(range(count//2)),'heldout_indices':list(range(count//2,count)),'tasks':[{'index':i,'task_id':t['id'],'task_hash':digest(t)} for i,t in enumerate(result['rows'])]}
    for name,value in [('private-tasks.json',private),('public-tasks.json',public)]:
        p=out/name
        if p.exists() and p.read_bytes()!=canonical(value):raise ValueError('immutable materialization conflict')
        p.write_bytes(canonical(value));p.chmod(0o600)
    report={'schema':1,'original_data_revision':REVISION,'pool_count':result['pool_count'],'excluded_base_count':result['excluded_base_count'],'count':count,'mining_indices':public['mining_indices'],'heldout_indices':public['heldout_indices'],'public_sha256':hashlib.sha256(canonical(public)).hexdigest(),'private_sha256':hashlib.sha256(canonical(private)).hexdigest(),'data_inventory_sha256':digest(inventory),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'simulation_ran':False,'model_proofs':False,'optimizer_ran':False,'payable':False,'chain_transactions':False}
    (out/'materialization-report.json').write_bytes(canonical(report));return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=32);a=p.parse_args();print(json.dumps(materialize(a.data,a.out,a.count)))
if __name__=='__main__':main()
