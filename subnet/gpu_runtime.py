"""Isolated strict same-GPU runtime pilot; not selected by the live controller."""
import gc
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .model import Runtime,model_files
from . import harness as policy
from .environments import create_session

PROFILE_VERSION='cuda-bf16-eager-sm86-v1'

class GPURuntime(Runtime):
    def __init__(self,checkpoint,files,environment,harness,*,runtime_revision=PROFILE_VERSION):
        from .backend_profiles import profile
        revision,approved_profile,_=profile(runtime_revision)
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('GPU deterministic CUBLAS workspace profile missing')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=tuple(approved_profile['sm']):
            raise ValueError('approved CUDA device unavailable for '+revision)
        self.runtime_revision=revision
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        if model_files(checkpoint)!=files:raise ValueError('GPU checkpoint allowlist mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
        self.configure(environment,harness)
        from toploc import build_proofs_base64
        from .proofs import verify_mapped_proofs
        from toploc.C.csrc.utils import get_fp_parts
        import toploc.poly as poly
        self.toploc_threads=2;poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=2)
        self.build_proofs,self.verify_proofs=build_proofs_base64,verify_mapped_proofs

    def compute(self,prompt,output):
        with torch.inference_mode():
            result=self.model(torch.tensor([prompt+output],device='cuda'),output_hidden_states=True,use_cache=False)
            hidden=result.hidden_states[-1][0].to(torch.bfloat16).cpu().contiguous()
            probs=torch.log_softmax(result.logits[0,len(prompt)-1:len(prompt)+len(output)-1].float(),-1).cpu().numpy()
        return [hidden[:len(prompt)]]+[hidden[i:i+1] for i in range(len(prompt),len(prompt)+len(output))],probs

    def sample(self,prompt,seed,messages,turn):
        config=policy.turn_config(self.harness,turn)
        rng=torch.Generator(device='cuda').manual_seed(seed)
        if config['policy']=='public-mrcr-shell-candidates':
            config={**config,'policy':'candidates','candidates':policy.mrcr_candidates(messages)}
        if config['policy']=='visible-copy-candidates':
            opening,closing=config['input_tags'];visible='\n'.join(m['content'] for m in messages if m['role']=='user');start=visible.rfind(opening)
            if start<0 or closing not in visible[start+len(opening):]:raise ValueError('visible span missing')
            text=visible[start+len(opening):].split(closing,1)[0];before,after=config['output_tags']
            config={**config,'policy':'candidates','candidates':[before+text+after,before+text+'!'+after]}
        if config['policy']=='candidates':
            candidates=[self.tokenizer.encode(text,add_special_tokens=False) for text in config['candidates']]
            if any(not ids or len(ids)>config['max_output_tokens'] for ids in candidates):raise ValueError('candidate token budget')
            with torch.inference_mode():
                scores=[]
                for ids in candidates:
                    logits=self.model(torch.tensor([prompt+ids],device='cuda'),use_cache=False).logits[0,len(prompt)-1:len(prompt)+len(ids)-1]
                    lp=torch.log_softmax(logits.float(),-1)
                    scores.append(lp.gather(1,torch.tensor(ids,device='cuda')[:,None]).sum())
                distribution=torch.softmax(torch.stack(scores)/config['temperature'],-1)
            return candidates[int(torch.multinomial(distribution,1,generator=rng))]
        if config['policy']!='autoregressive':raise ValueError('unsupported GPU sampling policy')
        output=[]
        with torch.inference_mode():
            for _ in range(config['max_output_tokens']):
                logits=self.model(torch.tensor([prompt+output],device='cuda'),use_cache=False).logits[0,-1].float()/config['temperature'];probs=torch.softmax(logits,-1)
                if config['top_p']<1:
                    values,indices=probs.sort(descending=True);values[values.cumsum(0)-values>config['top_p']]=0
                    probs=torch.zeros_like(probs).scatter(0,indices,values);probs/=probs.sum()
                token=int(torch.multinomial(probs,1,generator=rng));output.append(token)
                if token==self.tokenizer.eos_token_id:break
        return output

    def rollout(self,index,seed):
        env_seed=int(self.spec.config.get('seed',0));session=create_session(self.spec)
        try:
            initial=session.reset(index,env_seed);messages=initial['messages'];tools=initial.get('tools',[]);turns=[];arrays=[]
            for i in range(self.spec.max_turns):
                prompt=self.prompt(messages,tools)
                if len(prompt)+self.harness['max_output_tokens']>min(self.model.config.max_position_embeddings,8192):raise ValueError('GPU model context budget')
                output=self.sample(prompt,seed+i,messages,i);text=self.tokenizer.decode(output,skip_special_tokens=True);acts,probs=self.compute(prompt,output)
                proofs=self.build_proofs(acts,decode_batching_size=16,topk=128)
                if not proofs or any(p is None for p in proofs):raise ValueError('GPU proof construction')
                result=session.step(policy.action(text,self.harness));turns.append(dict(prompt=prompt,output=output,text=text,proofs=proofs,observations=result['observations'],done=result['done'],reward=result['reward'],classification=result['classification']));arrays.append(probs)
                messages=messages+[dict(role='assistant',content=text)]+policy.observations(result['observations'],self.harness)
                if result['done']:break
            if not result['done']:raise ValueError('GPU environment did not terminate')
            return dict(schema=2,env_id=self.spec.id,environment_version=self.spec.version,index=index,sample_index=index,seed=seed,env_seed=env_seed,task_hash=initial['task_hash'],reward=result['reward'],classification=result['classification'],turns=turns),arrays
        finally:session.close()

    def head_logprob(self,rollout):
        total=0;count=0
        for turn in rollout['turns']:
            prompt,output=turn['prompt'],turn['output'];ids=torch.tensor([prompt+output],device='cuda')
            # Explicit frozen-feature objective; no gradients through the decoder.
            # The tied input/output embedding is updated through the head only.
            with torch.no_grad():hidden=self.model.base_model(ids,use_cache=False).last_hidden_state[0,len(prompt)-1:len(prompt)+len(output)-1]
            logits=torch.nn.functional.linear(hidden.detach(),self.model.lm_head.weight,self.model.lm_head.bias if hasattr(self.model.lm_head,'bias') else None)
            total=total+torch.log_softmax(logits.float(),-1).gather(1,torch.tensor(output,device='cuda')[:,None]).sum();count+=len(output)
        return total/count

    def train(self,pairs,destination,steps=1):
        if not pairs:raise ValueError('no verified pairs')
        for param in self.model.parameters():param.requires_grad_(False)
        head=self.model.lm_head.weight;head.requires_grad_(True)
        with torch.no_grad():references=[float(self.head_logprob(p)-self.head_logprob(n)) for p,n in pairs]
        before=head.detach().clone();optimizer=torch.optim.AdamW([head],lr=1e-3);losses=[]
        for step in range(steps):
            pos,neg=pairs[step%len(pairs)];optimizer.zero_grad(set_to_none=True)
            margin=self.head_logprob(pos)-self.head_logprob(neg)-references[step%len(pairs)]
            loss=-torch.nn.functional.logsigmoid(.1*margin)
            if not torch.isfinite(loss):raise ValueError('nonfinite GPU training')
            loss.backward();torch.nn.utils.clip_grad_norm_([head],1);optimizer.step();losses.append(float(loss.detach()))
        changed=not torch.equal(before,head.detach())
        if not changed:raise ValueError('GPU optimizer did not change weights')
        destination=Path(destination)
        if destination.exists():raise ValueError('refuse to overwrite GPU checkpoint')
        destination.mkdir(parents=True);self.model.save_pretrained(destination,safe_serialization=True);self.tokenizer.save_pretrained(destination)
        tied=head.data_ptr()==self.model.get_input_embeddings().weight.data_ptr()
        del optimizer,before;gc.collect();torch.cuda.empty_cache()
        return {'steps':steps,'losses':losses,'weights_changed':changed,'objective':'reference-relative frozen-decoder-feature output-head preference',
                'trainable_parameters':head.numel(),'tied_input_embedding_updated':tied,'learning_rate':1e-3,'full_model_finetune':False}
