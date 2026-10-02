"""Actual original MATH common-session controls for separately named corpus snapshots."""
import argparse,copy,hashlib,json,sqlite3,time
from pathlib import Path
from subnet.math_corpus import write_snapshot,public_messages
from subnet.math_corpus_native import record,replay,VERSION
from subnet.environments import build_spec,create_session,EnvironmentSpec

def main():
 p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--inventory',required=True);p.add_argument('--existing-math',required=True);p.add_argument('--output',required=True);a=p.parse_args();catalog=Path(a.catalog);inventory=Path(a.inventory);out=Path(a.output);out.mkdir(exist_ok=True);assert not (out/'native-controls.json').exists();complete=json.loads((catalog/'catalog-complete.json').read_text());assert hashlib.sha256((catalog/'question-registry.sqlite').read_bytes()).hexdigest()==complete['database_sha256'];db=sqlite3.connect('file:'+str((catalog/'question-registry.sqlite').resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;template=json.loads(Path(a.existing_math).read_text())[0];report={'started_at':time.time(),'catalog_sha256':complete['database_sha256'],'corpora':{},'model_proof':False,'GPU_jobs':0,'common_training':False,'scope':'original common MATH CPU grader controls; new corpus IDs/large-population transport not yet deployed'}
 for corpus in ('DeepMath-103K','NuminaMath-CoT'):
  # Take distinct producer categories first, then fill by public prompt length.
  query="""SELECT q.* FROM questions q JOIN (SELECT key FROM
   (SELECT q.key,q.prompt_tokens,ROW_NUMBER() OVER (PARTITION BY q.category ORDER BY q.prompt_tokens,q.key) AS ordinal
    FROM questions q JOIN reference_checks r ON q.reference=r.reference
    WHERE q.corpus=? AND q.fold='train' AND q.status='eligible' AND r.kind='symbolic')
   WHERE ordinal=1 ORDER BY prompt_tokens,key LIMIT 8) picked ON picked.key=q.key"""
  selected=[dict(row) for row in db.execute(query,(corpus,))]
  if len(selected)<8:
   chosen={x['key'] for x in selected}
   for row in db.execute("SELECT q.* FROM questions q JOIN reference_checks r ON q.reference=r.reference WHERE q.corpus=? AND q.fold='train' AND q.status='eligible' AND r.kind='symbolic' ORDER BY q.prompt_tokens,q.key LIMIT 16",(corpus,)):
    if row['key'] not in chosen:selected.append(dict(row));chosen.add(row['key'])
    if len(selected)==8:break
  assert len(selected)==8
  rows=[dict(corpus=corpus,source_index=x['source_index'],question=x['question'],reference=x['reference'],category=x['category'],difficulty=x['difficulty'],question_key=x['key']) for x in selected];asset=out/(corpus+'.tasks.json');assetinfo=write_snapshot(rows,asset,template);metadata=json.loads((inventory/(corpus+'-metadata.json')).read_text());binding={'version':VERSION,'corpus':corpus,'upstream_revision':metadata['revision'],'catalog_sha256':complete['database_sha256'],'adapter_sha256':hashlib.sha256(Path('subnet/math_corpus_native.py').read_bytes()).hexdigest()};spec=build_spec('affine_math',config={'task_snapshot':str(asset.resolve()),'seed':0,'corpus_qualification':binding},num_samples=8,max_turns=1,max_output_tokens=256);(out/(corpus+'.spec.json')).write_text(json.dumps(spec.to_dict(),sort_keys=True));controls=[]
  for index,row in enumerate(rows):
   positive={'text':'\\boxed{'+row['reference']+'}','tool_calls':[]};negative={'text':'\\boxed{314159265358979323846264338327950288419716939937510}','tool_calls':[]};pos=record(spec,index,20261002,positive);neg=record(spec,index,20261002,negative);assert pos['initial']['messages']==public_messages(row);assert pos['result']['reward']==1.0 and neg['result']['reward']==0.0;replay(spec,index,20261002,pos);replay(spec,index,20261002,neg);refusals=[]
   for field,value in [('index',(index+1)%8),('seed',20261003),('source_hash','0'*64),('action',negative),('result',{'done':True,'reward':1.0})]:
    forged=copy.deepcopy(pos);forged[field]=value
    try:replay(spec,index,20261002,forged)
    except ValueError:refusals.append(field)
    else:raise AssertionError('forged trace accepted')
   controls.append({'index':index,'source_index':row['source_index'],'question_key':row['question_key'],'prompt_sha256':hashlib.sha256(row['question'].encode()).hexdigest(),'positive':pos,'negative':neg,'fresh_original_replay_passed':True,'mutations_refused':refusals});print(json.dumps({'corpus':corpus,'index':index,'original_positive':1.0,'original_negative':0.0,'replays':2,'mutations_refused':5}),flush=True)
  bad=spec.to_dict();bad['source_hash']='0'*64
  try:create_session(EnvironmentSpec.from_dict(bad))
  except ValueError:source_refused=True
  else:raise AssertionError('wrong source accepted')
  report['corpora'][corpus]={'upstream':metadata,'runtime_grader_id':'affine_math','separate_Lean_Numina':True,'snapshot':assetinfo,'spec_source_hash':spec.source_hash,'controls':controls,'wrong_source_refused':source_refused,'no_solutions_or_reference_fields_in_reset_messages':True};(out/'native-controls-progress.json').write_text(json.dumps(report,sort_keys=True))
 report['finished_at']=time.time();(out/'native-controls.json').write_text(json.dumps(report,sort_keys=True));print(json.dumps({'native_controls_complete':True,'native_original_tasks':16,'fresh_replays':32,'mutation_refusals':80}),flush=True)
if __name__=='__main__':main()
