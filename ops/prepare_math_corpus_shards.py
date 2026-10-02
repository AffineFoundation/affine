"""Export exact qualified catalogs to separately hydrated native MathTask shards."""
import argparse,gzip,hashlib,json,sqlite3,time
from pathlib import Path
from subnet.math_corpus import snapshot_row
from subnet.math_corpus_provider import VERSION,CORPORA,SHARD_ROWS
from subnet.math_corpus_assets import admit_bytes,MAX_RAW,MAX_COMPRESSED

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 catalog=Path(a.catalog);report=json.loads((catalog/'catalog-complete.json').read_text());dbpath=catalog/'question-registry.sqlite'
 if hashlib.sha256(dbpath.read_bytes()).hexdigest()!=report['database_sha256']:raise ValueError('qualified catalog changed')
 out=Path(a.output);out.mkdir(exist_ok=False);template=json.loads(Path(a.template).read_bytes())[0]
 db=sqlite3.connect('file:'+str(dbpath.resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 provider_sha=hashlib.sha256(Path('subnet/math_corpus_provider.py').read_bytes()).hexdigest();result={'version':VERSION,'started_at':time.time(),'catalog_sha256':report['database_sha256'],'shard_rows':SHARD_ROWS,'shards':[],'GPU_jobs':0,'common_training':False,'no_CoT_stored':True,'max_manifest_environments':64,'existing_MATH_all7496_excluded':True}
 for slug,(corpus,revision) in CORPORA.items():
  for fold in ('train','heldout','official_test'):
   cursor=db.execute("SELECT * FROM questions WHERE corpus=? AND fold=? AND status='eligible' ORDER BY key",(corpus,fold));ordinal=0
   while True:
    batch=cursor.fetchmany(SHARD_ROWS)
    if not batch:break
    adapted=[dict(corpus=corpus,source_index=r['source_index'],question=r['question'],reference=r['reference'],category=r['category'],difficulty=r['difficulty']) for r in batch]
    raw=canonical([snapshot_row(row,template) for row in adapted]);body=gzip.compress(raw,compresslevel=6,mtime=0);sha=hashlib.sha256(raw).hexdigest()
    binding={'version':VERSION,'corpus':corpus,'upstream_revision':revision,'catalog_sha256':report['database_sha256'],'fold':fold,'shard':ordinal,'rows':len(batch),'sha256':sha,'size':len(raw),'compressed_sha256':hashlib.sha256(body).hexdigest(),'compressed_size':len(body),'path':'assets/math-corpora/'+sha+'.tasks.json','provider_sha256':provider_sha}
    admit_bytes(body,binding)
    file=out/(sha+'.tasks.json.gz');file.write_bytes(body);file.chmod(0o444)
    result['shards'].append({'id':f'math_corpus_{slug}_{fold}_{ordinal:03d}','asset':binding,'object_file':file.name,'question_keys_sha256':hashlib.sha256(canonical([r['key'] for r in batch])).hexdigest(),'maximum_prompt_tokens':max(r['prompt_tokens'] for r in batch)})
    ordinal+=1
   print(json.dumps({'corpus':corpus,'fold':fold,'shards':ordinal}),flush=True)
 result['finished_at']=time.time();result['rows']=sum(x['asset']['rows'] for x in result['shards']);result['train_rows']=sum(x['asset']['rows'] for x in result['shards'] if x['asset']['fold']=='train');result['compressed_bytes']=sum(x['asset']['compressed_size'] for x in result['shards']);(out/'shards.json').write_bytes(canonical(result));print(json.dumps({k:result[k] for k in ('rows','train_rows','compressed_bytes')}),flush=True)
if __name__=='__main__':main()
