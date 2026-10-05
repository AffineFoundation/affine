"""Future opt-in generation ONLY; no live integration or replay/proof changes."""
import inspect
import time
import torch

def sample(model,prompt,*,seed,max_output_tokens,temperature,top_p,eos_token_id=None,mode='kv-last-logits-v1',capture_logits=False,telemetry=None):
 if model.config.model_type!='qwen2' or not {'past_key_values','use_cache','logits_to_keep'}<=set(inspect.signature(model.forward).parameters):raise ValueError('explicit Qwen2 cache/last-logits API required')
 if mode not in ('legacy','last-logits-v1','kv-last-logits-v1'):raise ValueError('explicit future sampling revision')
 if not prompt or any(type(t) is not int or not 0<=t<model.config.vocab_size for t in prompt):raise ValueError('actual token IDs')
 if type(max_output_tokens) is not int or max_output_tokens<1 or len(prompt)+max_output_tokens>model.config.max_position_embeddings:raise ValueError('actual context/output budget')
 if not 0<temperature or not 0<top_p<=1:raise ValueError('distribution settings')
 device=next(model.parameters()).device;rng=torch.Generator(device=device).manual_seed(seed);output=[];cache=None;logits=[]
 with torch.inference_mode():
  for _ in range(max_output_tokens):
   started=time.perf_counter()
   cached=mode=='kv-last-logits-v1';ids=[output[-1]] if cached and cache is not None else prompt+output
   kw={'use_cache':cached}
   if mode!='legacy':kw['logits_to_keep']=1
   if cache is not None:kw['past_key_values']=cache
   result=model(torch.tensor([ids],device=device),**kw)
   if cached:
    cache=result.past_key_values
    if cache is None or cache.get_seq_length()!=len(prompt)+len(output):raise ValueError('actual cache position/length')
   values=result.logits[0,-1].float()/temperature
   if capture_logits:logits.append(values.detach().cpu())
   probs=torch.softmax(values,-1)
   if top_p<1:
    sorted_values,indices=probs.sort(descending=True);sorted_values[sorted_values.cumsum(0)-sorted_values>top_p]=0
    probs=torch.zeros_like(probs).scatter(0,indices,sorted_values);probs/=probs.sum()
   token=int(torch.multinomial(probs,1,generator=rng));output.append(token)
   if telemetry is not None:telemetry(dict(input_tokens=len(ids),cache_reused=kw.get("past_key_values")is not None,forward_and_sampling_seconds=time.perf_counter()-started))
   if eos_token_id is not None and token==eos_token_id:break
 # Cache belongs to this call only and is never passed to proof/logprob replay.
 return output,logits
