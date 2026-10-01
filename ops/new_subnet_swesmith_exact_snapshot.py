"""Preserve original default first SWE-smith task without all-language Arrow duplication."""
import pathlib,json,hashlib
import pyarrow.parquet as pq
from subnet.environments import build_spec,_taskset,_source_hash
spec=build_spec('swesmith',num_samples=1,max_turns=2);_taskset(spec)
import verifiers.v1 as vf
from swesmith_v1.taskset import SWESmithTask,SWESmithData,REPO_PATH
from swesmith.profiles import registry
root=pathlib.Path('/home/const/.cache/huggingface/hub/datasets--SWE-bench--SWE-smith-py/snapshots');revision=sorted(root.iterdir())[0];found=None
for p in sorted((revision/'data').glob('*.parquet')):
 for batch in pq.ParquetFile(p).iter_batches(batch_size=1):
  row=batch.to_pylist()[0]
  try:registry.get_from_inst(row)
  except KeyError:continue
  found=(p,row);break
 if found:break
if not found:raise RuntimeError('no original row with registered profile')
p,row=found;iid=row['instance_id'];t=SWESmithTask(SWESmithData(idx=0,name='py:'+iid,prompt=row['problem_statement'],image=row['image_name'],workdir=REPO_PATH,resources=vf.TaskResources(cpu=4,memory=4,disk=10),language='py',row=row,instance_id=iid,gold_patch=row.get('patch') or '',fail_to_pass=list(row.get('FAIL_TO_PASS') or []),pass_to_pass=list(row.get('PASS_TO_PASS') or [])),vf.TaskConfig())
out=pathlib.Path('state/original-task-snapshots/swesmith-original-first.tasks.json');out.write_text(json.dumps([{'task_class':type(t).__name__,'data':t.data.model_dump(mode='json'),'task_config':t.config.model_dump(mode='json')}])) ;out.chmod(0o400)
spec.config['task_snapshot']=str(out);spec=spec.__class__(**{**spec.to_dict(),'source_hash':_source_hash(spec)})
pathlib.Path('state/original-task-snapshots/swesmith-original-first.spec.json').write_text(json.dumps(spec.to_dict(),indent=2));r={'source':'swesmith','original_dataset':'SWE-bench/SWE-smith-py','revision':revision.name,'parquet':p.name,'parquet_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'task_name':t.data.name,'task_hash':t.hash,'image':t.data.image,'source_hash':spec.source_hash};pathlib.Path('state/multi-environment/swesmith-first-exact-task-report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
