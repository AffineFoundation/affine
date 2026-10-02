"""Freeze a NEW catalog after official-test protection and bounded reference checks."""
import argparse,collections,hashlib,json,logging,multiprocessing,sqlite3,time
from pathlib import Path
from importlib.metadata import version
from subnet.math_corpus import last_boxed_body,question_key

def reference_check(reference):
 logging.getLogger('math_verify').setLevel(logging.CRITICAL)
 from math_verify import parse,verify
 if last_boxed_body('\\boxed{'+reference+'}')!=reference:return reference,False,'unbalanced_reference'
 try:
  parsed=parse('\\boxed{'+reference+'}',parsing_timeout=1)
  valid=bool(parsed) and verify(parsed,parsed,timeout_seconds=1)
  kind='symbolic' if any(not isinstance(x,str) for x in parsed) else 'string_fallback'
  return reference,valid,kind if valid else 'unparsed_or_not_self_equivalent'
 except Exception as error:return reference,False,type(error).__name__

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.input);out=Path(a.output);out.mkdir(exist_ok=True);assert not (out/'question-registry.sqlite').exists();original=json.loads((src/'inventory-complete.json').read_text());body=(src/'question-registry.sqlite').read_bytes();assert hashlib.sha256(body).hexdigest()==original['database_sha256'];(out/'question-registry.sqlite').write_bytes(body);del body
 db=sqlite3.connect(out/'question-registry.sqlite');db.execute('CREATE TABLE reference_checks(reference TEXT PRIMARY KEY,valid INTEGER,kind TEXT)')
 # Official Numina test protection applies globally even when a prior corpus
 # already claimed the same normalized question. Preserve the v1 scan unchanged.
 from ops.qualify_math_corpora import RangeReader
 import pyarrow.parquet as pq
 m=json.loads((src/'NuminaMath-CoT-metadata.json').read_text());f=next(x for x in json.loads((src/'NuminaMath-CoT-files.json').read_text()) if '/test-' in x['path']);reader=RangeReader('https://huggingface.co/datasets/'+m['id']+'/resolve/'+m['revision']+'/'+f['path'],f['size']);table=pq.ParquetFile(reader).read(columns=['problem']);keys={question_key(x['problem']) for x in table.to_pylist()};protected=[]
 for key in keys:
  row=db.execute('SELECT corpus,fold,status FROM questions WHERE key=?',(key,)).fetchone()
  if row and row[0]!='existing-MATH' and row[1]!='official_test':
   protected.append({'question_key':key,'corpus':row[0],'prior_fold':row[1],'prior_status':row[2]});db.execute("UPDATE questions SET status='protected_official_test' WHERE key=?",(key,))
 db.commit();references=[r[0] for r in db.execute("SELECT DISTINCT reference FROM questions WHERE status='eligible'")];progress={'started_at':time.time(),'v1_database_sha256':original['database_sha256'],'official_test_questions':len(keys),'additional_global_test_protection':protected,'unique_references':len(references),'reference_checks_complete':0,'GPU_jobs':0,'native_provider_full_task_coverage':False,'scope':'bounded MathVerify reference self-check; corpus ground-truth correctness not certified'};(out/'finalization-progress.json').write_text(json.dumps(progress,sort_keys=True))
 kinds=collections.Counter()
 with multiprocessing.Pool(8) as pool:
  for reference,valid,kind in pool.imap_unordered(reference_check,references,chunksize=64):
   db.execute('INSERT INTO reference_checks VALUES(?,?,?)',(reference,int(valid),kind));kinds[kind]+=1;progress['reference_checks_complete']+=1
   if progress['reference_checks_complete']%2000==0:
    db.commit();progress['kinds']=dict(kinds);(out/'finalization-progress.json').write_text(json.dumps(progress,sort_keys=True));print(json.dumps({'checked_unique_references':progress['reference_checks_complete'],'total':len(references)}),flush=True)
 db.commit();db.execute("UPDATE questions SET status='reference_not_qualified' WHERE status='eligible' AND reference IN (SELECT reference FROM reference_checks WHERE valid=0)");db.commit();counts={}
 for corpus in original['corpora']:
  counts[corpus]={'eligible_by_fold':dict(db.execute("SELECT fold,count(*) FROM questions WHERE corpus=? AND status='eligible' GROUP BY fold",(corpus,))), 'statuses':dict(db.execute('SELECT status,count(*) FROM questions WHERE corpus=? GROUP BY status',(corpus,))), 'reference_kinds':dict(db.execute("SELECT r.kind,count(*) FROM questions q JOIN reference_checks r ON q.reference=r.reference WHERE q.corpus=? AND q.status='eligible' GROUP BY r.kind",(corpus,)))}
 assert db.execute("SELECT count(*) FROM questions WHERE status='eligible' AND corpus!='existing-MATH' AND fold!='official_test' AND key IN (%s)"%(','.join('?' for _ in keys)),tuple(keys)).fetchone()[0]==0
 db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();profile={name:version(name) for name in ['math-verify','sympy','latex2sympy2_extended','verifiers']};progress.update(finished_at=time.time(),reference_checks_complete=len(references),kinds=dict(kinds),qualified_catalog_counts=counts,dependency_versions=profile,database_sha256=hashlib.sha256((out/'question-registry.sqlite').read_bytes()).hexdigest(),math_verify_timeout_seconds=1,official_test_global_overlap_remaining=0,full_native_provider_controls_pending=True,model_proof=False,common_training=False);(out/'catalog-complete.json').write_text(json.dumps(progress,sort_keys=True));print(json.dumps({'catalog_complete':True,'counts':counts}),flush=True)
if __name__=='__main__':main()
