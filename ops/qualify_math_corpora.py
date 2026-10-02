"""Bounded projected-Parquet qualification of two pinned prose-math datasets.

No model/GPU calls; solution chains are streamed for boxed extraction and never
retained. Uses explicit immutable revisions, strict HTTP ranges, and capped reads.
"""
import argparse,collections,hashlib,io,json,sqlite3,time
from pathlib import Path
import requests,pyarrow.parquet as pq
from subnet.math_corpus import adapt_row,question_key,public_messages,VERSION

class RangeReader(io.RawIOBase):
 def __init__(self,url,size,budget=2_000_000_000):
  self.url=url;self.size=size;self.pos=0;self.session=requests.Session();self.network_bytes=0;self.budget=budget;self.ranges=[]
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if not 0<=self.pos<=self.size:raise ValueError('seek bounds')
  return self.pos
 def read(self,n=-1):
  n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
  if not n:return b''
  if n>64*1024**2 or self.network_bytes+n>self.budget:raise ValueError('bounded projected read')
  start=self.pos;end=start+n-1
  response=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=90)
  try:
   if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('HTTP range/size mismatch')
   if response.headers.get('Content-Encoding') not in (None,'identity'):raise ValueError('encoded range refused')
   data=response.raw.read(n+1)
   if len(data)!=n:raise ValueError('range body size')
   self.pos+=n;self.network_bytes+=n;self.ranges.append({'start':start,'size':n,'sha256':hashlib.sha256(data).hexdigest()});return data
  finally:response.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--existing-math',required=True);p.add_argument('--tokenizer',required=True);a=p.parse_args();root=Path(a.state);root.mkdir(exist_ok=True);dbpath=root/'question-registry.sqlite';assert not dbpath.exists(),'preserve prior qualification; use a fresh state'
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
 db=sqlite3.connect(dbpath);db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE questions (key TEXT PRIMARY KEY,corpus TEXT,source_index INTEGER,question TEXT,reference TEXT,category TEXT,difficulty TEXT,prompt_tokens INTEGER,fold TEXT,status TEXT)')
 existing=json.loads(Path(a.existing_math).read_text());protected={question_key(r['data']['problem']) for r in existing};assert len(existing)==7496
 db.executemany('INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?)',[(k,'existing-MATH',-1,'','','','',0,'protected','protected') for k in protected]);db.commit()
 summary={'version':VERSION,'started_at':time.time(),'existing_MATH_protected_rows':7496,'existing_MATH_unique_conservative_keys':len(protected),'max_context_tokens':8192,'output_reserve_tokens':256,'tokenizer_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(a.tokenizer).iterdir() if p.is_file()},'corpora':{},'whole_solution_corpora_saved':False,'GPU_jobs':0}
 for slug in ('DeepMath-103K','NuminaMath-CoT'):
  meta=json.loads((root/(slug+'-metadata.json')).read_text());files=json.loads((root/(slug+'-files.json')).read_text());counts=collections.Counter();file_reports=[];raw_index=0
  for file in files:
   url='https://huggingface.co/datasets/'+meta['id']+'/resolve/'+meta['revision']+'/'+file['path'];reader=RangeReader(url,file['size']);parquet=pq.ParquetFile(reader);columns=['question','final_answer','topic','difficulty'] if slug=='DeepMath-103K' else ['problem','solution','source'];split='test' if '/test-' in file['path'] else 'train';file_rows=0
   for batch in parquet.iter_batches(batch_size=256,columns=columns):
    rows=batch.to_pylist();adapted=[]
    for row in rows:
     idx=raw_index;raw_index+=1;file_rows+=1;counts['raw_'+split]+=1
     try:entry=adapt_row(slug,row,idx)
     except ValueError as error:counts['rejected:'+str(error)]+=1;continue
     adapted.append(entry)
    if adapted:
     conversations=[public_messages(x) for x in adapted]
     encoded=tokenizer.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True)
     encoded=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
     for entry,tokens in zip(adapted,encoded,strict=True):
      if len(tokens)+256>8192:counts['rejected:context_budget']+=1;continue
      old=db.execute('SELECT corpus,reference,fold,status FROM questions WHERE key=?',(entry['question_key'],)).fetchone()
      if old:
       if old[0]=='existing-MATH':counts['excluded:existing_MATH_overlap']+=1;continue
       counts['duplicate:within' if old[0]==slug else 'duplicate:between_corpora']+=1
       if old[2]=='official_test':counts['excluded:official_test_overlap']+=1
       # Conflicting textual references are conservatively quarantined, without
       # claiming a symbolic equivalence/semantic deduplication proof.
       normalize=lambda x:''.join(x.split())
       if normalize(old[1])!=normalize(entry['reference']):
        counts['reference_conflict_occurrences']+=1;db.execute("UPDATE questions SET status='reference_conflict' WHERE key=?",(entry['question_key'],))
       continue
      fold='official_test' if split=='test' else ('heldout' if int(entry['question_key'][:16],16)%100==0 else 'train')
      db.execute('INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?)',(entry['question_key'],slug,entry['source_index'],entry['question'],entry['reference'],entry['category'],entry['difficulty'],len(tokens),fold,'eligible'))
    db.commit()
   assert file_rows==parquet.metadata.num_rows
   report={'path':file['path'],'upstream_size':file['size'],'upstream_lfs_sha256':file['lfs']['oid'],'actual_metadata_rows':parquet.metadata.num_rows,'actual_streamed_rows':file_rows,'columns_read':columns,'network_bytes':reader.network_bytes,'range_count':len(reader.ranges),'range_commitment':hashlib.sha256(json.dumps(reader.ranges,sort_keys=True).encode()).hexdigest()};file_reports.append(report);(root/(slug+'-range-receipts.json')).write_text(json.dumps(file_reports,sort_keys=True));summary['corpora'][slug]={'metadata':meta,'counts':dict(counts),'files':file_reports};(root/'inventory-progress.json').write_text(json.dumps(summary,sort_keys=True));print(json.dumps({'corpus':slug,'file':file['path'],'rows':file_rows,'network_bytes':reader.network_bytes}),flush=True)
  summary['corpora'][slug]['counts']=dict(counts)
 for slug in ('DeepMath-103K','NuminaMath-CoT'):
  eligible=dict(db.execute("SELECT fold,count(*) FROM questions WHERE corpus=? AND status='eligible' GROUP BY fold",(slug,)).fetchall());quarantined=db.execute("SELECT count(*) FROM questions WHERE corpus=? AND status='reference_conflict'",(slug,)).fetchone()[0]
  stats=db.execute("SELECT min(prompt_tokens),max(prompt_tokens),count(*) FROM questions WHERE corpus=? AND status='eligible'",(slug,)).fetchone();summary['corpora'][slug].update(eligible_after_all_cross_corpus_dedup=eligible,reference_conflict_groups_quarantined=quarantined,prompt_tokens_min_max_count=list(stats),full_native_grader_coverage=False,real_model_proof=False,common_training=False)
 summary.update(finished_at=time.time(),cross_corpus_train_heldout_overlap=0,existing_MATH_protected_key_overlap_in_new_eligible=0,dedup_scope='conservative question text/LaTex normalization, not semantic equivalence',database_sha256_pending_checkpoint=True)
 db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();summary['database_sha256']=hashlib.sha256(dbpath.read_bytes()).hexdigest();summary['database_sha256_pending_checkpoint']=False;(root/'inventory-complete.json').write_text(json.dumps(summary,sort_keys=True));print(json.dumps({'inventory_complete':True,'corpora':{k:v['eligible_after_all_cross_corpus_dedup'] for k,v in summary['corpora'].items()}}),flush=True)
if __name__=='__main__':main()
