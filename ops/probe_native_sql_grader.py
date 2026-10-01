"""Original cached Spider grader controls, not generated miner samples."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    from datasets import load_dataset
    from subnet.native_sql_isolation import build,grade,SOURCE
    root=Path('state/native-sql-isolation');root.mkdir(parents=True,exist_ok=True)
    row=load_dataset('xlangai/spider',split='train')[0]
    db=Path('/home/const/.cache/affine/spider/spider_data/database')/row['db_id']/(row['db_id']+'.sqlite')
    task={'db_path':str(db),'database_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),
          'gold_sql':row['query'],'ordered':'order by' in row['query'].lower(),
          'question':row['question'],'db_id':row['db_id']}
    private=root/'private-task.json';private.write_text(json.dumps(task,indent=2));private.chmod(0o600)
    runtime=build(root/'grader-image');(root/'runtime.json').write_text(json.dumps(runtime,indent=2))
    results=[]
    controls=[('native-positive','```sql\n'+row['query']+'\n```',1),
              ('native-negative','```sql\nSELECT -987654321 AS incorrect\n```',0),
              ('host-file-function',"```sql\nSELECT readfile('/etc/passwd')\n```",0),
              ('extension-loading',"```sql\nSELECT load_extension('/tmp/attack')\n```",0),
              ('query-parser','```sql\nSELECT * FROM definitely_missing_table\n```',0)]
    for name,reply,expected in controls:
        result=grade(task,reply,runtime)
        if result['reward']!=expected:raise ValueError('original grade mismatch '+name)
        (root/(name+'.json')).write_text(json.dumps(result,indent=2))
        results.append({'name':name,'reward':result['reward'],'isolation':result['isolation']})
    before=time.monotonic();bounded=False
    try:grade(task,'```sql\nWITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t) SELECT max(n) FROM t\n```',runtime,timeout=2)
    except subprocess.TimeoutExpired:bounded=True
    if not bounded:raise ValueError('recursive SQL wall-clock limit not exercised')
    if subprocess.check_output(['docker','ps','-aq','--filter','label=affine.native-sql-controlled='+runtime['revision']],text=True).strip():raise ValueError('SQL control container leaked')
    if hashlib.sha256(db.read_bytes()).hexdigest()!=task['database_sha256']:raise ValueError('original DB changed')
    repeat=grade(task,controls[0][1],runtime)
    if repeat['reward']!=1:raise ValueError('fresh original positive changed')
    summary={'completed':True,'original_dataset':'xlangai/spider','split':'train','original_index':0,
             'runtime':runtime,'original_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             'original_database_sha256':task['database_sha256'],'controls':results,
             'recursive_query_wall_clock_bounded':True,'bound_and_cleanup_seconds':time.monotonic()-before,
             'fresh_positive_after_attacks':1,'original_database_unchanged':True,
             'model_proofs_generated':False,'miner_harness_admitted':False,'training_performed':False,
             'gold_used_only_for_grader_control':True,'chain_transactions':False,'completed_at':time.time()}
    (root/'controls.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
