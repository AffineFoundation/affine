"""Independent bounded re-read of every exported asset against its pinned catalog."""
import argparse,hashlib,json,sqlite3,time
from pathlib import Path
from subnet.math_corpus_assets import admit_bytes
from subnet.math_corpus import question_key

def main():
 p=argparse.ArgumentParser();p.add_argument('--shards',required=True);p.add_argument('--catalog',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.shards);body=(root/'shards.json').read_bytes();manifest=json.loads(body);dbpath=Path(a.catalog)/'question-registry.sqlite'
 if hashlib.sha256(dbpath.read_bytes()).hexdigest()!=manifest['catalog_sha256']:raise ValueError('catalog source mismatch')
 db=sqlite3.connect('file:'+str(dbpath.resolve())+'?mode=ro',uri=True);seen=set();counts={};report={'started_at':time.time(),'shards_manifest_sha256':hashlib.sha256(body).hexdigest(),'catalog_sha256':manifest['catalog_sha256'],'GPU_jobs':0,'native_full_grade':False,'shards':[]}
 for entry in manifest['shards']:
  b=entry['asset'];raw=admit_bytes((root/entry['object_file']).read_bytes(),b);rows=json.loads(raw);cursor=db.execute("SELECT key,source_index,question,reference FROM questions WHERE corpus=? AND fold=? AND status='eligible' ORDER BY key LIMIT ? OFFSET ?",(b['corpus'],b['fold'],b['rows'],8192*b['shard']));expected=cursor.fetchall()
  if len(expected)!=len(rows):raise ValueError('catalog row count')
  for row,source in zip(rows,expected):
   key,index,question,reference=source;data=row['data']
   if key in seen or question_key(data['problem'])!=key or data['problem']!=question or data['answer']!=reference or data['idx']!=index:raise ValueError('asset duplicate/catalog mutation')
   seen.add(key)
  counts.setdefault(b['corpus'],{});counts[b['corpus']][b['fold']]=counts[b['corpus']].get(b['fold'],0)+len(rows);report['shards'].append({'id':entry['id'],'rows':len(rows),'raw_sha256':b['sha256'],'compressed_sha256':b['compressed_sha256']})
  print(entry['id'],flush=True)
 expected=db.execute("SELECT count(*) FROM questions WHERE status='eligible'").fetchone()[0]
 if len(seen)!=expected or expected!=manifest['rows']:raise ValueError('catalog incomplete')
 report.update(finished_at=time.time(),actual_rows=len(seen),counts=counts,normalized_question_overlap_between_shards=0,all_reference_question_source_index_bytes_match=True);Path(a.output).write_text(json.dumps(report,sort_keys=True));print(json.dumps({'actual_rows':len(seen),'counts':counts}),flush=True)
if __name__=='__main__':main()
