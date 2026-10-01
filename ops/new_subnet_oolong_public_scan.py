"""Bounded genuine live16k row public-prompt/context inspection; never reads gold fields."""
import pathlib,json
import pyarrow.parquet as pq
root=pathlib.Path('/home/const/.cache/huggingface/hub/datasets--oolongbench--oolong-synth/snapshots');rev=sorted(root.iterdir())[0];rows=[];index=0
for p in sorted((rev/'data').glob('test-*.parquet')):
 for batch in pq.ParquetFile(p).iter_batches(batch_size=1,columns=['context_len','question','context_window_text']):
  row=batch.to_pylist()[0]
  if row['context_len']==16384:
   rows.append({'original_index':index,'question':row['question'],'context_header':row['context_window_text'][:1200],'parquet':p.name,'revision':rev.name})
   if len(rows)==16:break
  index+=1
 if len(rows)==16:break
pathlib.Path('state/multi-environment/oolong-public-16-task-scan.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
