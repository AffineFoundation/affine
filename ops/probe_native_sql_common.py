"""Actual isolated Spider reset/step/final grading conformance; no model claims."""
import json,time
from pathlib import Path
from types import SimpleNamespace
from subnet.native_sql_actor import PublicSQLActor
from subnet.native_sql_adapter import NativeSQLAdapter,VERSION,task_hash
from subnet.native_sql_isolation import grade

def main():
 p=Path('state/native-sql-isolation');private=json.loads((p/'private-task.json').read_text());public=json.loads((p/'actor-public.json').read_text());actor_runtime=json.loads((p/'actor-runtime.json').read_text());grader=json.loads((p/'runtime.json').read_text());old=json.loads((p/'actor-controls.json').read_text())
 spec=SimpleNamespace(adapter='native_sql_controlled',version=VERSION,config=dict(dependency_scope='controlled-public-database-private-original-grader',public_tasks=[public],grader_runtime=grader),num_samples=1,max_turns=4,success_reward=1)
 def actor_factory(i,seed,h):
  assert i==0 and h==task_hash(public)
  return PublicSQLActor(actor_runtime,public)
 def terminal(i,text,h):
  assert i==0 and h==task_hash(public)
  return grade(private,text,grader)
 runs=[]
 for label,condition in [('positive','56'),('negative','999'),('fresh-positive','56')]:
  a=NativeSQLAdapter(spec,actor_factory,terminal)
  try:
   reset=a.reset(0,0);observations=[]
   for command in old['actions']:
    result=a.step(dict(text='',tool_calls=[dict(name='bash',arguments=dict(command=command))]))
    assert not result['done'] and result['reward']==0
    observations.append(json.loads(result['observations'][0]['content']))
   assert observations==old['observations']
   result=a.step(dict(text=f'```sql\nSELECT COUNT(*) FROM head WHERE age > {condition}\n```'))
   assert result['done'] and result['reward']==(0 if label=='negative' else 1)
   runs.append(dict(label=label,reset_task_hash=reset['task_hash'],observations=observations,reward=result['reward'],classification=result['classification']))
  finally:a.close()
 assert runs[0]['observations']==runs[2]['observations'] and runs[0]['reset_task_hash']==runs[2]['reset_task_hash']
 out=dict(completed=True,version=VERSION,original_source_sha256=public['original_source_sha256'],actor_runtime=actor_runtime,grader_runtime=grader,runs=runs,fresh_exact_replay=True,full_original_grader=True,model_proofs_generated=False,shared_epoch_verified=False,training_performed=False,chain_transactions=False,completed_at=time.time())
 target=p/'common-adapter-controls.json';target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('runs','actor_runtime','grader_runtime')}))
if __name__=='__main__':main()
