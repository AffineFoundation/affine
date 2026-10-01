"""Cross-runtime numeric checks and frozen-artifact random segment audit experiment."""
import argparse,asyncio,base64,copy,json,math,platform,random,time
from pathlib import Path
import numpy as np
import torch,transformers
from toploc import build_proofs_base64,verify_proofs_base64
from toploc.poly import batch_activations
import pipeline as P

THRESHOLDS=[(0,0.,0.,1e-5),(0,.5,1.,.001),(1,1.,2.,.01),(4,2.,4.,.05),(8,4.,8.,.1),(16,8.,16.,.2),(32,16.,32.,.5),(64,32.,64.,1.)]

def result_row(r):return {'exp_mismatches':int(r.exp_mismatches),'mant_err_mean':float(r.mant_err_mean),'mant_err_median':float(r.mant_err_median)}
def good(r,t):return r['exp_mismatches']<=t[0] and r['mant_err_mean']<=t[1] and r['mant_err_median']<=t[2] and r['logprob_max_abs']<=t[3]
def validate_document(doc,arrays,policy,tok):
 assert doc['schema']==1 and doc['model']==policy['model'] and doc['revision']==policy['revision'],'model identity'
 assert doc['model_state']==policy['files'],'model state'
 assert doc['environment']==P.CONFIG,'environment configuration'
 assert 1<=len(doc['turns'])<=3 and len(arrays)==len(doc['turns']),'turn count'
 env,state,messages=P.initial()
 for i,(turn,a) in enumerate(zip(doc['turns'],arrays)):
  assert turn['prompt']==P.context(tok,messages),'context'
  out=turn['output'];assert 0<len(out)<=96 and all(type(x)==int and 0<=x<tok.vocab_size for x in out),'tokens'
  text=tok.decode(out,skip_special_tokens=True);assert text==turn['text'],'text'
  assert len(turn['proofs'])==1+math.ceil(len(out)/P.BATCH),'proof count'
  for proof in turn['proofs']:
   raw=base64.b64decode(proof,validate=True)
   assert len(raw)==2+2*P.TOPK,'proof byte count'
   assert 32768<=int.from_bytes(raw[:2],'big')<=65497,'proof modulus'
  assert a.shape==(len(out),tok.vocab_size) and np.isfinite(a).all(),'logprob shape/finiteness'
  done,feedback,reward=P.advance(env,state,text)
  assert (turn['feedback'],turn['done'],turn['reward'])==(feedback,done,reward),'environment replay'
  assert not done or i==len(doc['turns'])-1,'extra turns'
  messages.extend([{'role':'assistant','content':text},{'role':'user','content':feedback}])
 assert done,'truncated trajectory'

def metrics(doc,arrays,cache):
 rows=[]
 for i,(turn,claimed) in enumerate(zip(doc['turns'],arrays)):
  acts,probs=cache[i]
  res=verify_proofs_base64(acts,turn['proofs'],decode_batching_size=P.BATCH,topk=P.TOPK)
  assert len(res)==len(turn['proofs'])
  for j,r in enumerate(res):
   row=result_row(r);start=max(0,(j-1)*P.BATCH);end=min(len(turn['output']),j*P.BATCH)
   delta=np.abs(claimed[start:end]-probs[start:end]) if j else np.array([0.])
   row.update(turn=i,segment=j,kind='prefill' if j==0 else 'output',logprob_max_abs=float(delta.max()),logprob_mean_abs=float(delta.mean()),logprob_p99_abs=float(np.quantile(delta,.99)))
   rows.append(row)
 return rows

def native_segment_results(doc,arrays,cache):
 return [not good(r,THRESHOLDS[0]) for r in metrics(doc,arrays,cache)]

def cpu_controls(out,doc,cache):
 d=copy.deepcopy(doc);arrays=[]
 for i,t in enumerate(d['turns']):
  acts,probs=cache[i];t['proofs']=build_proofs_base64(acts,decode_batching_size=P.BATCH,topk=P.TOPK);arrays.append(probs)
 path=out/'cpu-honest.zip';sha=P.pack(path,d,arrays)
 return d,arrays,sha

def simulate(out,doc,arrays,cache,policy,tok,trials,seed):
 baseline_doc,baseline_arrays,sha=cpu_controls(out,doc,cache)
 validate_document(baseline_doc,baseline_arrays,policy,tok)
 honest_bad=native_segment_results(baseline_doc,baseline_arrays,cache);assert not any(honest_bad),'CPU honest baseline'
 refs=[(i,j) for i,t in enumerate(doc['turns']) for j in range(len(t['proofs']))];n=len(refs);cases=[]
 def add(name,d,a,locations=None):
  path=out/(name+'.zip');seal=P.pack(path,d,a)
  # Commitment precedes random challenge generation; verify the committed bytes.
  committed_doc,committed_arrays=P.unpack(path);assert P.digest(path)==seal
  try:validate_document(committed_doc,committed_arrays,policy,tok);bad=native_segment_results(committed_doc,committed_arrays,cache);structural=False
  except Exception as e:bad=[False]*n;structural=True;reason=str(e)
  rng=random.Random(seed)
  for k in [1,2,4,8,n]:
   hits=sum(structural or any(bad[j] for j in rng.sample(range(n),k)) for _ in range(trials));b=sum(bad)
   expected=1. if structural else (1-math.comb(n-b,k)/math.comb(n,k) if n-b>=k else 1.)
   cases.append({'case':name,'sha256':seal,'total_segments':n,'bad_segments':b,'bad_indices':[i for i,x in enumerate(bad) if x],'structural_rejection':structural,'reason':reason if structural else 'numeric segment checks','audit_size':k,'trials':trials,'detected':hits,'detection_rate':hits/trials,'expected_rate':expected,'empirical_minus_expected':hits/trials-expected,'declared_corruption_indices':locations})
  assert P.digest(path)==seal,'submission changed after commitment'
 add('spot-honest',baseline_doc,baseline_arrays,[])
 for name,indices in [('first_prefill',[0]),('last_segment',[n-1]),('three_scattered',[1,5,9]),('half_segments',[0,2,4,6,8,10]),('all_segments',list(range(n)))]:
  d=copy.deepcopy(baseline_doc)
  for index in indices:
   i,j=refs[index];raw=bytearray(base64.b64decode(d['turns'][i]['proofs'][j]));raw[2:] = bytes(len(raw)-2);d['turns'][i]['proofs'][j]=base64.b64encode(raw).decode()
  add('spot-'+name,d,baseline_arrays,indices)
 d=copy.deepcopy(baseline_doc);a=[x.copy() for x in baseline_arrays];a[-1][-1,0]+=1.;add('spot-one_logprob',d,a,[n-1])
 d=copy.deepcopy(baseline_doc);d['turns'][0]['proofs'].pop();add('spot-missing_proof',d,baseline_arrays)
 d=copy.deepcopy(baseline_doc);d['turns'][0]['proofs'][0]='not-valid-base64!';add('spot-malformed_proof',d,baseline_arrays)
 d=copy.deepcopy(baseline_doc);raw=bytearray(base64.b64decode(d['turns'][0]['proofs'][0]));raw[:2]=b'\0\0';d['turns'][0]['proofs'][0]=base64.b64encode(raw).decode();add('spot-zero_modulus',d,baseline_arrays)
 d=copy.deepcopy(baseline_doc);d['turns'][0]['feedback']='fake';add('spot-fake_observation',d,baseline_arrays)
 return {'seed':seed,'trials_per_configuration':trials,'baseline_sha256':sha,'policy':'Check all structure and environment transitions; randomly select k distinct global inference segments after commitment; reject if any selected proof/probability segment fails. Prefill checks proof only; output blocks check proof plus all-vocabulary logprobs.','cost_note':'Numeric failure masks are obtained from actual native TOPLOC and probability checks once, then challenges sampled over those immutable results. Full reference forward passes are cached; this measures detection, not reduced computation cost.','results':cases}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='artifacts');parser.add_argument('--threads',type=int,default=4);parser.add_argument('--trials',type=int,default=20000);args=parser.parse_args();root=Path(args.root);out=root/'extended';out.mkdir(exist_ok=True)
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 policy=json.loads((root/'trusted-policy-local.json').read_text());tok,model=P.load(policy,'cpu')
 doc,arrays=P.unpack(root/'honest.zip');validate_document(doc,arrays,policy,tok);cache=[]
 for i,t in enumerate(doc['turns']):
  start=time.monotonic();value=P.compute(model,t['prompt'],t['output']);cache.append(value);print('CPU reference turn',i,'seconds',time.monotonic()-start,flush=True)
 honest=metrics(doc,arrays,cache);print('Cross-runtime honest metrics:',honest,flush=True)
 attacks={}
 for name in ['altered_weights_proofs_only','actual_altered_weights','wrong_proof','wrong_logprobs']:
  d,a=P.unpack(root/(name+'.zip'));validate_document(d,a,policy,tok);attacks[name]=metrics(d,a,cache)
 grid=[]
 for threshold in THRESHOLDS:
  grid.append({'threshold':{'max_exponent_mismatches':threshold[0],'max_mantissa_mean_error':threshold[1],'max_mantissa_median_error':threshold[2],'logprob_atol':threshold[3]},'honest_accepted':all(good(x,threshold) for x in honest),'honest_bad_segments':sum(not good(x,threshold) for x in honest),'attack_accepted':{name:all(good(x,threshold) for x in values) for name,values in attacks.items()}})
 cross={'gpu_reference':{'hardware':'RTX3090','torch':'2.12.0+cu130','transformers':'4.57.1','dtype':'bfloat16'},'cpu_verifier':{'hardware':platform.processor(),'machine':platform.machine(),'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'bfloat16','threads':args.threads},'limitation':'CPU/GPU hardware AND torch/transformers versions differ. This is a cross-runtime test, not an isolated hardware-only comparison. No cross-GPU test was performed.','honest_artifact_sha256':P.digest(root/'honest.zip'),'honest_segments':honest,'attack_segments':attacks,'threshold_grid':grid}
 (out/'cross-runtime.json').write_text(json.dumps(cross,indent=2));spot=simulate(out,doc,arrays,cache,policy,tok,args.trials,20260930);(out/'spot-checks.json').write_text(json.dumps(spot,indent=2));print(json.dumps({'cross_runtime_strict_honest_accepted':grid[0]['honest_accepted'],'spot_configurations':len(spot['results']),'out':str(out)}),flush=True)

if __name__=='__main__':main()
