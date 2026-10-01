"""Independent CUDA pilot. Does not change controller, source adapters or weights."""
import argparse,copy,gc,hashlib,json,os,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from toploc import build_proofs_base64,verify_proofs_base64
from subnet import harness as policy
from subnet.environments import create_session
from subnet.model import Runtime,file_hash,model_files

class GPURuntime(Runtime):
    def __init__(self,checkpoint,files,environment,harness):
        torch.set_num_threads(2)
        import toploc.poly as poly
        from toploc.C.csrc.utils import get_fp_parts as native_get_fp_parts
        poly.get_fp_parts=lambda tensor:native_get_fp_parts(tensor,num_threads=2)
        actual=model_files(checkpoint)
        if actual!=files:raise ValueError('GPU checkpoint allowlist mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
        self.configure(environment,harness)
        self.build_proofs,self.verify_proofs=build_proofs_base64,verify_proofs_base64
    def compute(self,prompt,output):
        with torch.inference_mode():
            result=self.model(torch.tensor([prompt+output],device='cuda'),output_hidden_states=True,use_cache=False)
            hidden=result.hidden_states[-1][0].to(torch.bfloat16).cpu().contiguous()
            probabilities=torch.log_softmax(result.logits[0,len(prompt)-1:len(prompt)+len(output)-1].float(),-1).cpu().numpy()
        return [hidden[:len(prompt)]]+[hidden[i:i+1] for i in range(len(prompt),len(prompt)+len(output))],probabilities
    def sample_gpu(self,prompt,seed):
        rng=torch.Generator(device='cuda').manual_seed(seed);output=[]
        with torch.inference_mode():
            for _ in range(self.harness['max_output_tokens']):
                logits=self.model(torch.tensor([prompt+output],device='cuda'),use_cache=False).logits[0,-1].float()/self.harness['temperature']
                probabilities=torch.softmax(logits,-1)
                token=int(torch.multinomial(probabilities,1,generator=rng));output.append(token)
                if token==self.tokenizer.eos_token_id:break
        return output
    def rollout(self,index,seed):
        env_seed=int(self.spec.config.get('seed',0));session=create_session(self.spec)
        try:
            initial=session.reset(index,env_seed);messages,tools=initial['messages'],initial.get('tools',[]);turns=[];arrays=[]
            for i in range(self.spec.max_turns):
                prompt=self.prompt(messages,tools)
                if len(prompt)+self.harness['max_output_tokens']>min(self.model.config.max_position_embeddings,8192):raise ValueError('GPU model context budget')
                output=self.sample_gpu(prompt,seed+i);text=self.tokenizer.decode(output,skip_special_tokens=True);acts,probs=self.compute(prompt,output)
                proofs=self.build_proofs(acts,decode_batching_size=16,topk=128)
                if not proofs or any(p is None for p in proofs):raise ValueError('GPU proof construction')
                result=session.step(policy.action(text));turns.append(dict(prompt=prompt,output=output,text=text,proofs=proofs,observations=result['observations'],done=result['done'],reward=result['reward'],classification=result['classification']));arrays.append(probs)
                messages=messages+[dict(role='assistant',content=text)]+policy.observations(result['observations'],self.harness)
                if result['done']:break
            if not result['done']:raise ValueError('GPU environment did not terminate')
            return dict(schema=2,env_id=self.spec.id,environment_version=self.spec.version,index=index,sample_index=index,seed=seed,env_seed=env_seed,task_hash=initial['task_hash'],reward=result['reward'],classification=result['classification'],turns=turns),arrays
        finally:session.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--policy',required=True);p.add_argument('--spec',required=True);p.add_argument('--index',type=int,default=2);p.add_argument('--output',default='state/gpu-pilot');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time();miner=verifier=None
    trusted=json.loads(Path(args.policy).read_text());spec=json.loads(Path(args.spec).read_text());files=trusted['files']
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    harness={'version':'text-tools-v1','policy':'autoregressive','max_output_tokens':16,'temperature':.7,'top_p':1.0}
    report={'model':trusted['model'],'model_revision':trusted['revision'],'checkpoint_files':files,'checkpoint_id':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'environment':spec,'index':args.index,'training':False,'policy':harness,'runtime_source_sha256':file_hash(__file__),'profile':{'device':'cuda','GPU':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'eager','toploc_parts_threads':2,'tf32':False,'deterministic_algorithms':True,'torch':version('torch'),'transformers':version('transformers'),'torch_cuda':torch.version.cuda,'native_env':{k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_CBWR','ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA']}},'controls':[]}
    try:
        if model_files(args.checkpoint)!=files:raise ValueError('archive bytes differ from approved prior policy')
        miner=GPURuntime(args.checkpoint,files,spec,harness);rollout,arrays=miner.rollout(args.index,100)
        (out/'honest-rollout.json').write_text(json.dumps(rollout)+'\n');np.savez_compressed(out/'honest-probabilities.npz',**{'turn_'+str(i):a for i,a in enumerate(arrays)})
        del miner;miner=None;gc.collect();torch.cuda.empty_cache()
        verifier=GPURuntime(args.checkpoint,files,spec,harness)
        report.update(reward=rollout['reward'],classification=rollout['classification'],turns=len(rollout['turns']),prompt_tokens=len(rollout['turns'][0]['prompt']),output_tokens=len(rollout['turns'][0]['output']))
        report['controls'].append({'case':'honest independent same-GPU checkpoint reload + environment replay','expected':True,'passed':verifier.verify(rollout,arrays)})
        cases=[]
        tokens=copy.deepcopy(rollout);tokens['turns'][0]['output'][0]=(tokens['turns'][0]['output'][0]+1)%verifier.model.config.vocab_size;tokens['turns'][0]['text']=verifier.tokenizer.decode(tokens['turns'][0]['output'],skip_special_tokens=True);cases.append(('altered output token',tokens,arrays))
        proofs=copy.deepcopy(rollout);proofs['turns'][0]['proofs'][0]='AAAA';cases.append(('altered proof',proofs,arrays))
        wrong=[a.copy() for a in arrays];wrong[0][0,0]+=1;cases.append(('altered full probability',rollout,wrong))
        reward=copy.deepcopy(rollout);reward['reward']=1-rollout['reward'];cases.append(('altered environment score',reward,arrays))
        for name,doc,probs in cases:
            try:verifier.verify(doc,probs);row={'case':name,'expected':False,'passed':False,'error':'forgery accepted'}
            except Exception as e:row={'case':name,'expected':False,'passed':True,'rejection':type(e).__name__+': '+str(e)[:160]}
            report['controls'].append(row)
        corrupted=dict(files);corrupted['model.safetensors']='0'*64
        try:GPURuntime(args.checkpoint,corrupted,spec,harness);report['controls'].append({'case':'wrong approved weight hash','passed':False})
        except ValueError as e:report['controls'].append({'case':'wrong approved weight hash','passed':True,'rejection':str(e)})
        report['success']=all(r['passed'] for r in report['controls'])
        report['artifact_hashes']={n:file_hash(out/n) for n in ['honest-rollout.json','honest-probabilities.npz']}
    except Exception as e:report.update(success=False,error=type(e).__name__+': '+str(e)[:600])
    finally:
        miner=verifier=None;gc.collect();torch.cuda.empty_cache();report['GPU_allocated_bytes_after_cleanup']=torch.cuda.memory_allocated();report['seconds']=round(time.time()-start,2)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
