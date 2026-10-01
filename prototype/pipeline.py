"""GPU TOPLOC experiment. Untrusted artifacts never supply executable code or weights."""
import argparse,asyncio,base64,copy,hashlib,json,math,sys,time,zipfile,io
from pathlib import Path
if not __debug__:
 raise RuntimeError('Prototype verification requires assertions; do not use python -O')
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import HfApi,snapshot_download
from toploc import build_proofs_base64,verify_proofs_base64
sys.path.insert(0,str(Path(__file__).parent/'vendor/mastermind'))
import mastermind
import verifiers as vf
MODEL='HuggingFaceTB/SmolLM2-1.7B-Instruct'
CONFIG=dict(num_train_examples=1,num_eval_examples=0,code_length=2,num_symbols=4,max_turns=3,use_think=True,seed=42,use_candidate_reduction_reward=False)
TOPK=128; BATCH=16

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def trust(root):
 info=HfApi().model_info(MODEL)
 path=Path(snapshot_download(MODEL,revision=info.sha,allow_patterns=['*.json','*.safetensors','*.txt']))
 policy={'model':MODEL,'revision':info.sha,'snapshot':str(path),'files':{p.name:digest(p) for p in path.iterdir() if p.is_file()},'environment':CONFIG,'environment_files':{str(p.relative_to(Path(__file__).parent)):digest(p) for p in (Path(__file__).parent/'vendor/mastermind').rglob('*.py')}}
 (root/'trusted-policy.json').write_text(json.dumps(policy,indent=2))
 return policy

def load(policy,device='cuda'):
 for name,sha in policy['files'].items():assert digest(Path(policy['snapshot'])/name)==sha,'trusted weights changed'
 for name,sha in policy['environment_files'].items():assert digest(Path(__file__).parent/name)==sha,'trusted environment changed'
 tok=AutoTokenizer.from_pretrained(policy['snapshot'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(policy['snapshot'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
 return tok,model

def context(tok,messages):
 result=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
 return result['input_ids'] if isinstance(result,dict) or hasattr(result,'keys') else result

def compute(model,prompt,output):
 with torch.inference_mode():
  result=model(torch.tensor([prompt+output],device=next(model.parameters()).device),output_hidden_states=True,use_cache=False)
  hidden=result.hidden_states[-1][0].cpu().contiguous()
  probs=torch.log_softmax(result.logits[0,len(prompt)-1:len(prompt)+len(output)-1].float(),dim=-1).cpu().numpy()
 acts=[hidden[:len(prompt)]]+[hidden[i:i+1] for i in range(len(prompt),len(prompt)+len(output))]
 return acts,probs

def pack(path,doc,arrays):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
  z.writestr('rollout.json',json.dumps(doc,allow_nan=False))
  for i,a in enumerate(arrays):
   b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'logprobs-{i}.npy',b.getvalue())
 return digest(path)

def unpack(path):
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(set(names))==len(names),'duplicate entries'
  assert all(x.file_size<100_000_000 for x in z.infolist()),'oversized entry'
  doc=json.loads(z.read('rollout.json'))
  assert names==['rollout.json']+[f'logprobs-{i}.npy' for i in range(len(doc['turns']))],'unexpected entries'
  arrays=[np.load(io.BytesIO(z.read(f'logprobs-{i}.npy')),allow_pickle=False) for i in range(len(doc['turns']))]
 return doc,arrays

def initial():
 env=mastermind.load_environment(**CONFIG);row=env.dataset[0]
 state={'answer':row['answer'],'trajectory':[],'prompt_too_long':False}
 asyncio.run(env.setup_state(state))
 messages=[{'role':'system','content':env.system_prompt},{'role':'user','content':'Start: make your first guess.'}]
 return env,state,messages

def advance(env,state,text):
 state['trajectory'].append({'completion':[vf.AssistantMessage(content=text)]})
 done=asyncio.run(env.check_done(state))
 feedback=state['next_turn_response'][0].content
 return done,feedback,float(mastermind.solved_reward(state))

def generate(root,policy,tok,model,curated=False):
 env,state,messages=initial();doc={'schema':1,'model':policy['model'],'revision':policy['revision'],'model_state':policy['files'],'environment':CONFIG,'turns':[],'curated':curated};arrays=[]
 for i in range(3):
  prompt=context(tok,messages)
  if curated: output=tok.encode('<think>Try a candidate.</think><guess>'+['00','11','22'][i]+'</guess>',add_special_tokens=False)
  else:
   torch.manual_seed(100+i)
   with torch.inference_mode():
    ids=model.generate(torch.tensor([prompt],device='cuda'),max_new_tokens=96,do_sample=True,temperature=.7,pad_token_id=tok.eos_token_id,use_cache=True)
   output=ids[0,len(prompt):].tolist()
  text=tok.decode(output,skip_special_tokens=True);acts,probs=compute(model,prompt,output)
  proofs=build_proofs_base64(acts,decode_batching_size=BATCH,topk=TOPK)
  done,feedback,reward=advance(env,state,text)
  doc['turns'].append({'prompt':prompt,'output':output,'text':text,'proofs':proofs,'feedback':feedback,'done':done,'reward':reward,'sampling':{'seed':100+i,'temperature':.7,'max_new_tokens':96,'externally_curated':curated}})
  arrays.append(probs);messages.extend([{'role':'assistant','content':text},{'role':'user','content':feedback}])
  print('generated turn',i,'tokens',len(output),'done',done,flush=True)
  if done:break
 name='curated.zip' if curated else 'honest.zip';sha=pack(root/name,doc,arrays);(root/(name+'.sha256')).write_text(sha+'\n')
 return root/name

def verify(path,policy,tok,model):
 doc,arrays=unpack(path)
 assert doc['schema']==1 and doc['model']==policy['model'] and doc['revision']==policy['revision'],'model identity'
 assert doc['model_state']==policy['files'],'model state'
 assert doc['environment']==CONFIG,'environment configuration'
 assert 1<=len(doc['turns'])<=3,'turn count'
 env,state,messages=initial();nproof=0
 for i,(turn,claimed) in enumerate(zip(doc['turns'],arrays)):
  prompt=context(tok,messages);assert turn['prompt']==prompt,'context'
  out=turn['output'];assert 0<len(out)<=96 and all(type(x)==int and 0<=x<model.config.vocab_size for x in out),'tokens'
  text=tok.decode(out,skip_special_tokens=True);assert turn['text']==text,'decoded text'
  acts,probs=compute(model,prompt,out)
  assert claimed.shape==probs.shape and np.isfinite(claimed).all(),'probability shape/finiteness'
  assert np.allclose(claimed,probs,atol=1e-5,rtol=0),'probabilities'
  expected=1+math.ceil(len(out)/BATCH);assert len(turn['proofs'])==expected,'proof count'
  for proof in turn['proofs']:
   raw=base64.b64decode(proof,validate=True)
   assert len(raw)==2+2*TOPK,'proof byte count'
   assert 32768<=int.from_bytes(raw[:2],'big')<=65497,'proof modulus'
  results=verify_proofs_base64(acts,turn['proofs'],decode_batching_size=BATCH,topk=TOPK)
  assert len(results)==expected,'verification count'
  for r in results:
   assert r.exp_mismatches==0 and r.mant_err_mean==0 and r.mant_err_median==0,'TOPLOC mismatch'
  nproof+=expected
  done,feedback,reward=advance(env,state,text)
  assert turn['feedback']==feedback and turn['done']==done and turn['reward']==reward,'environment replay'
  assert not done or i==len(doc['turns'])-1,'extra turns'
  messages.extend([{'role':'assistant','content':text},{'role':'user','content':feedback}])
 assert done,'truncated trajectory'
 return {'valid':True,'turns':len(doc['turns']),'proofs':nproof}

def suite(root,policy,tok,model):
 doc,arrays=unpack(root/'honest.zip');cases=[]
 def run(name,d,a,expected):
  p=root/(name+'.zip');sha=pack(p,d,a);start=time.monotonic()
  try:r=verify(p,policy,tok,model);valid=True;reason='accepted'
  except Exception as e:valid=False;reason=str(e) or type(e).__name__
  row={'case':name,'expected':expected,'valid':valid,'matches_expectation':valid==expected,'reason':reason,'seconds':time.monotonic()-start,'resealed_sha256':sha};cases.append(row);print(row,flush=True)
 run('honest',doc,arrays,True)
 d,a=unpack(root/'curated.zip');run('curated',d,a,True)
 # A dishonest prover really computes proofs and probabilities with altered weights,
 # while claiming the approved model identity. Restore trusted weights before audit.
 d=copy.deepcopy(doc);a=[x.copy() for x in arrays]
 weight=model.model.layers[0].mlp.down_proj.weight
 saved=weight.detach().clone()
 with torch.no_grad():weight.mul_(1.1)
 acts,a[0]=compute(model,d['turns'][0]['prompt'],d['turns'][0]['output'])
 d['turns'][0]['proofs']=build_proofs_base64(acts,decode_batching_size=BATCH,topk=TOPK)
 with torch.no_grad():weight.copy_(saved)
 run('actual_altered_weights',d,a,False)
 run('altered_weights_proofs_only',d,arrays,False)
 d=copy.deepcopy(doc)
 d['turns'][0]['output'][0]=(d['turns'][0]['output'][0]+1)%model.config.vocab_size
 d['turns'][0]['text']=tok.decode(d['turns'][0]['output'],skip_special_tokens=True)
 run('altered_tokens_consistent_text',d,arrays,False)
 for name,mutation in [('wrong_weights',lambda d:d['model_state'].update({'model.safetensors':'0'*64})),('wrong_prompt',lambda d:d['turns'][0]['prompt'].__setitem__(0,d['turns'][0]['prompt'][0]+1)),('altered_tokens',lambda d:d['turns'][0]['output'].__setitem__(0,1)),('missing_proof',lambda d:d['turns'][0]['proofs'].pop()),('extra_proof',lambda d:d['turns'][0]['proofs'].append(d['turns'][0]['proofs'][0])),('wrong_proof',lambda d:d['turns'][0]['proofs'].__setitem__(0,d['turns'][-1]['proofs'][0])),('fake_observation',lambda d:d['turns'][0].update(feedback='fabricated')),('fake_reward',lambda d:d['turns'][-1].update(reward=9)),('truncated_turns',lambda d:d['turns'].pop())]:
  d=copy.deepcopy(doc);mutation(d);run(name,d,arrays[:len(d['turns'])],False)
 for name,value in [('wrong_logprobs',1.0),('nan_logprobs',float('nan'))]:
  a=[x.copy() for x in arrays];a[0][0,0]=value;run(name,doc,a,False)
 report={'model':policy['model'],'revision':policy['revision'],'gpu':torch.cuda.get_device_name(),'precision':'bfloat16','backend':'transformers eager','toploc_topk':TOPK,'decode_batching':BATCH,'probability_tolerance':1e-5,'results':cases,'passed':all(x['matches_expectation'] for x in cases)}
 (root/'report.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','verify','suite']);p.add_argument('--root',default='artifacts');p.add_argument('--artifact',default='honest.zip');args=p.parse_args();root=Path(args.root);root.mkdir(exist_ok=True,parents=True)
 policy=trust(root) if args.mode=='generate' else json.loads((root/'trusted-policy.json').read_text());tok,model=load(policy)
 if args.mode=='generate':generate(root,policy,tok,model);generate(root,policy,tok,model,True)
 elif args.mode=='suite':
  report=suite(root,policy,tok,model);print(json.dumps(report));sys.exit(0 if report['passed'] else 1)
 else:
  try:result=verify(root/args.artifact,policy,tok,model)
  except Exception as e:print(json.dumps({'valid':False,'reason':str(e)}));sys.exit(1)
  print(json.dumps(result))
