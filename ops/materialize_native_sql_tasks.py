"""Materialize indexed original Spider tasks with distinct private/public files."""
import argparse,hashlib,json,os
from pathlib import Path
from subnet.native_sql_actor import build,public_descriptor
from subnet.native_sql_isolation import SOURCE

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default='state/native-sql-tasks');parser.add_argument('--count',type=int,default=32);args=parser.parse_args()
 if not 2<=args.count<=64:raise ValueError('controlled original task count')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
 from datasets import load_dataset
 data=load_dataset('xlangai/spider',split='train')
 cache_hash='0c350918f3f29ec754f1181c65cdce76cd6c133c'
 if len(data.cache_files)!=1 or Path(data.cache_files[0]['filename']).parent.name!=cache_hash:raise ValueError('approved original dataset cache')
 cache_sha=hashlib.sha256(Path(data.cache_files[0]['filename']).read_bytes()).hexdigest()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 images={};public_records=[];private_records=[]
 for index in range(args.count):
  row=data[index];db=Path('/home/const/.cache/affine/spider/spider_data/database')/row['db_id']/(row['db_id']+'.sqlite')
  private=dict(db_path=str(db),database_sha256=hashlib.sha256(db.read_bytes()).hexdigest(),gold_sql=row['query'],ordered='order by' in row['query'].lower(),question=row['question'],db_id=row['db_id'])
  if row['db_id'] not in images:images[row['db_id']]=build(private,out/'images'/row['db_id'])[0]
  public=public_descriptor(private);task_id='spider-'+row['db_id']+'-'+hashlib.sha256((row['db_id']+'\n'+row['question']+'\n'+row['query']).encode()).hexdigest()[:10]
  public_records.append(dict(original_index=index,original_task_id=task_id,public=public,actor_runtime=images[row['db_id']]))
  private_records.append(dict(original_index=index,original_task_id=task_id,private=private))
 common=dict(dataset='xlangai/spider',split='train',dataset_cache_hash=cache_hash,dataset_cache_sha256=cache_sha,original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),train_indices=list(range(args.count//2)),heldout_indices=list(range(args.count//2,args.count)))
 privatefile=out/'private-tasks.json';privatefile.write_text(json.dumps(dict(common,tasks=private_records),indent=2)+'\n');privatefile.chmod(0o600)
 (out/'public-tasks.json').write_text(json.dumps(dict(common,tasks=public_records),indent=2)+'\n')
 (out/'materialization.json').write_text(json.dumps(dict(common,count=args.count,unique_actor_images=len(images),public_contains_gold=False,model_proofs_generated=False,shared_epoch_verified=False),indent=2)+'\n')
 print(json.dumps(dict(count=args.count,unique_actor_images=len(images),train_count=args.count//2,heldout_count=args.count-args.count//2)))
if __name__=='__main__':main()
