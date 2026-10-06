"""Default-off research: prescribed CDF pass / invalid / numerical unknown.

No production caller imports this module. Unknown never certifies validity or
fraud, and no autoregressive model call can occur in this sampling checker.
"""
import math
from .audit_policy import InvalidSample
from .fast_prefill_audit import distribution, NumericalAmbiguity
VERSION='token-only-prefill-threeway-research-v1'
POLICY={'version':VERSION,'verification':'prefill-cdf-calibrated-threeway',
        'historical_execution_proof':False,'autoregressive_fallback':False,
        'numerical_inconclusive_not_valid':True}

def validate_policy(value):
 if(type(value)is not dict or value!=POLICY or
    any(type(value.get(k))is not str for k in ('version','verification'))or
    value.get('historical_execution_proof')is not False or
    value.get('autoregressive_fallback')is not False or
    value.get('numerical_inconclusive_not_valid')is not True):
  raise ValueError('explicit threeway research policy required')
 return dict(value)

def intervals(logprobs,tokens,uniforms,temperature,top_p,error):
 import torch
 if type(error)not in(int,float)or not math.isfinite(error)or not 0<=error<=1e-3:raise ValueError('bounded admitted CDF error')
 probs=distribution(logprobs,temperature,top_p);n,v=probs.shape
 if len(tokens)!=n or len(uniforms)!=n or any(type(t)is not int or not 0<=t<v for t in tokens):raise InvalidSample('forced sampling token framing')
 if any(type(u)not in(int,float)or not math.isfinite(u)or not 0<=u<1 for u in uniforms):raise ValueError('trusted forced uniform framing')
 indices=torch.tensor(tokens,dtype=torch.int64,device=probs.device);positions=torch.arange(n,device=probs.device);cdf=probs.cumsum(-1);cdf[:,-1]=1.
 high=cdf[positions,indices];mass=probs[positions,indices];low=high-mass;u=torch.tensor(uniforms,dtype=torch.float64,device=probs.device)
 supported=mass>0
 # A support exclusion never becomes PASS. Its collapsed CDF boundary is
 # checked against the SAME signed uncertainty bound as every position.
 # Outside that bound rejects; inside stays unknown without exact replay.
 outside=(u<low-error)|(u>=high+error)
 if bool(outside.any()):raise InvalidSample('forced CDF interval outside calibrated region')
 if bool((~supported).any()):raise NumericalAmbiguity('reference nucleus support excluded; no cached adjudication')
 # Unlike the existing v3 fast path, BOTH sides of each boundary uncertainty
 # band remain unknown, including a draw just inside the prefill interval.
 ambiguous=(u<low+error)|(u>=high-error)
 if bool(ambiguous.any()):raise NumericalAmbiguity('forced CDF boundary within calibrated uncertainty; no cached adjudication')
 return {'positions':n,'all_intervals_verified':True,'cdf_abs_error':error,
         'sampling_assurance':'calibrated-interior-prescribed-CDF',
         'autoregressive_fallback':False}

def verify_sampling(runtime,rollout,turn_index,prompt,output,logprobs):
 from .forced_sampling import uniform,validate_attempt
 context=runtime.sampling_context;config=runtime.harness;validate_attempt(context,rollout['seed']);stop=runtime.tokenizer.eos_token_id
 if not output or any(token==stop for token in output[:-1])or len(output)<config['max_output_tokens']and output[-1]!=stop:raise InvalidSample('forced generation stop condition')
 draws=[uniform(context,runtime.spec.id,rollout['task_hash'],rollout['index'],rollout['seed'],turn_index,i)for i in range(len(output))]
 return intervals(logprobs,output,draws,config['temperature'],config['top_p'],runtime.fast_sampling_calibration['cdf_abs_error'])
