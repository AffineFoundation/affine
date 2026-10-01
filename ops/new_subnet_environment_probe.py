"""Operator-only actual-source smoke: no registration, chain or training writes."""
import argparse, json, time, traceback, hashlib, os
from importlib.metadata import version
from pathlib import Path
import numpy as np
from subnet.environments import EnvironmentSpec,create_session
from subnet.model import Runtime,model_files
from subnet.harness import render

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--snapshots',default='state/original-task-snapshots');p.add_argument('--output',default='state/environment-remote-probes');p.add_argument('--source',action='append');p.add_argument('--harness-json');args=p.parse_args()
 target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
 configs=list(sorted(Path(args.snapshots).glob('*.spec.json')))
 if args.source:configs=[p for p in configs if p.name.removesuffix('.spec.json') in args.source]
 if not configs:raise ValueError('no trusted task specs selected')
 harness={'version':'plain-transcript-v1','policy':'autoregressive','max_output_tokens':16,'temperature':0.7,'top_p':1.0}
 if args.harness_json:harness=json.loads(Path(args.harness_json).read_text())
 first=json.loads(configs[0].read_text());files=model_files(args.checkpoint)
 checkpoint={'id':Path(args.checkpoint).name,'files':files,'filemap_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'model_revision':Path(args.checkpoint).name,'upstream_model_revision':'not recovered by this probe; exact trained weights bound by file allowlist'}
 profile={'device':'cpu','dtype':'float32','attention':'eager','torch_threads':2,'torch':version('torch'),'transformers':version('transformers'),'verifiers':version('verifiers'),'native_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','MKL_CBWR','ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA')}}
 miner=Runtime(args.checkpoint,files,threads=2,environment=first,harness=harness)
 # Separately reload approved checkpoint, independently replay every computation.
 verifier=Runtime(args.checkpoint,files,threads=2,environment=first,harness=harness)
 profile['runtime_revision']='cpu-float32-eager-v2-bounded-toploc' if getattr(miner,'toploc_threads',None)==2 else 'unobserved-toploc-native-thread-setting'
 profile['toploc_parts_threads']=getattr(miner,'toploc_threads',None)
 rows=[]
 for path in configs:
  spec=json.loads(path.read_text());name=spec['id'];started=time.time();row={'checkpoint':checkpoint,'runtime_profile':profile,'source':name,'source_hash':spec['source_hash'],'sampling':'unconstrained target-model autoregressive' if harness['policy']=='autoregressive' else 'curated target-model candidate choice','harness':harness,'max_output_tokens':harness['max_output_tokens'],'reset':False,'proof_generated':False,'independent_verified':False,'training':False}
  session=None
  try:
   miner.configure(spec,harness);verifier.configure(spec,harness)
   session=create_session(EnvironmentSpec.from_dict(spec));initial=session.reset(0,0)
   prompt=render(miner.tokenizer,initial['messages'],initial['tools'],harness)
   row.update(reset=True,task_name=initial['task_name'],task_hash=initial['task_hash'],prompt_tokens=len(prompt),tools=len(initial['tools']))
   session.close();session=None
   if len(prompt)+harness['max_output_tokens']>min(getattr(miner.model.config,'max_position_embeddings',8192),8192):
    row.update(status='blocked_pilot_context_budget',error='original task context exceeds pinned small model context; task not replaced')
   else:
    rollout,arrays=miner.rollout(0,31415)
    row.update(proof_generated=True,reward=rollout['reward'],classification=rollout['classification'],turns=len(rollout['turns']))
    (target/(name+'.rollout.json')).write_text(json.dumps(rollout)+'\n');np.savez_compressed(target/(name+'.probabilities.npz'),**{'turn_'+str(i):v for i,v in enumerate(arrays)})
    row['independent_verified']=verifier.verify(rollout,arrays)
    row['status']='verified_original_rollout'
  except Exception as e:row.update(status='error',error=type(e).__name__+': '+str(e)[:500])
  finally:
   if session:
    try:session.close()
    except Exception:pass
  row['seconds']=round(time.time()-started,2);rows.append(row)
  (target/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
