"""Extract actual first live16k test task from cached pinned original parquet; no Arrow all-split staging."""
import hashlib,json,pathlib,os
import pyarrow.parquet as pq
from subnet.environments import build_spec,_taskset,_source_hash,create_session
spec=build_spec('affine_oolong',{'taskset':{'context_len':16384,'split':'test'}},num_samples=1,max_turns=2)
_taskset(spec)
from affine_oolong_v1.taskset import SYSTEM,task_name
from oolong_synth_v1.taskset import OolongSynthTask,OolongSynthData,OolongSynthTaskConfig,INSTRUCTIONS,WORKDIR
root=pathlib.Path('/home/const/.cache/huggingface/hub/datasets--oolongbench--oolong-synth/snapshots');revision=sorted(root.iterdir())[0];index=0;found=None
for path in sorted((revision/'data').glob('test-*.parquet')):
 for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
  row=batch.to_pylist()[0]
  if row['context_len']==16384:found=(path,row,index);break
  index+=1
 if found:break
if not found:raise RuntimeError('no actual16k test task')
path,row,index=found;at=row.get('answer_type','');at=at if at in ('ANSWER_TYPE.NUMERIC','ANSWER_TYPE.DATE') else ''
t=OolongSynthTask(OolongSynthData(idx=index,name=task_name(16384,index,'test'),system_prompt=SYSTEM,prompt=row['question']+'\n\n'+INSTRUCTIONS,question=row['question'],answer=row['answer'],context=row['context_window_text'],answer_type=at,workdir=WORKDIR),OolongSynthTaskConfig())
out=pathlib.Path('state/original-task-snapshots/oolong-original-first.tasks.json');out.write_text(json.dumps([{'task_class':type(t).__name__,'data':t.data.model_dump(mode='json'),'task_config':t.config.model_dump(mode='json')}]));out.chmod(0o400)
spec.config['task_snapshot']=str(out);spec=spec.__class__(**{**spec.to_dict(),'source_hash':_source_hash(spec)})
pathlib.Path('state/original-task-snapshots/oolong-original-first.spec.json').write_text(json.dumps(spec.to_dict(),indent=2))
s=create_session(spec)
try:
 reset=s.reset(0,0);report={'source':'affine_oolong','original_revision':revision.name,'parquet':path.name,'parquet_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'original_row_index':index,'task_name':reset['task_name'],'task_hash':reset['task_hash'],'source_hash':spec.source_hash,'reset':True,'context_bytes':len(row['context_window_text'].encode()),'status':'actual_original_reset_passed_exact_parquet_snapshot'}
 pathlib.Path('state/multi-environment/oolong-original-snapshot-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
finally:s.close()
