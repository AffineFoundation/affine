"""Opt-in single-prefill CDF verification with bounded calibrated ambiguity.

A near-boundary result stays UNKNOWN unless opt-in v3 exact cached replay
adjudicates it. Unavailable replay never fabricates verification success.
Calibration is an executed-control artifact admitted by the signed opening.
"""
import hashlib,json,math
VERSION='forced-inverse-cdf-prefill-v2'
SUPPORT_VERSION='forced-inverse-cdf-prefill-support-v3'
THREEWAY_VERSION='forced-inverse-cdf-prefill-threeway-v4'
CALIBRATION='cached-prefill-calibration-v1'
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
digest=lambda v:hashlib.sha256(canonical(v)).hexdigest()
class NumericalAmbiguity(RuntimeError):pass
class CalibrationRequired(RuntimeError):pass
from .audit_policy import InvalidSample
class SupportMismatch(InvalidSample):pass

def calibration(value):
 fields={'version','checkpoint','model_runtime_revision','backend_profile_sha256','harness_sha256','report_sha256','cdf_abs_error','logprob_atol','toploc_exp_mismatches','toploc_mant_err_mean','toploc_mant_err_median'}
 if type(value)is not dict or set(value)!=fields or value['version']!=CALIBRATION:raise ValueError('exact executed calibration policy')
 for name in ('checkpoint','backend_profile_sha256','harness_sha256','report_sha256'):
  v=value[name]
  if type(v)is not str or len(v)!=64 or any(c not in '0123456789abcdef'for c in v):raise ValueError('calibration digest')
 if type(value['model_runtime_revision'])is not str or not value['model_runtime_revision']:raise ValueError('calibration runtime')
 bounds={'cdf_abs_error':1e-3,'logprob_atol':.1,'toploc_exp_mismatches':0,'toploc_mant_err_mean':1.,'toploc_mant_err_median':1.}
 for name,maximum in bounds.items():
  v=value[name]
  if type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<=maximum:raise ValueError('bounded calibrated numerical threshold')
 if type(value['toploc_exp_mismatches'])is not int:raise ValueError('strict TOPLOC exponent threshold')
 return dict(value)

def bind(manifest,harness):
 from .harness import normalize
 p=calibration(manifest['sampling_contract']['calibration']);h=normalize(harness)
 if p['checkpoint']!=manifest['checkpoint']['id']or p['model_runtime_revision']!=manifest['model_runtime_revision']or p['backend_profile_sha256']!=digest(manifest['backend_profile'])or p['harness_sha256']!=digest(h):raise CalibrationRequired('executed calibration checkpoint/profile/harness mismatch')
 return p

def distribution(logprobs,temperature,top_p):
 import torch
 if type(temperature)not in(int,float)or not math.isfinite(temperature)or temperature<=0 or type(top_p)not in(int,float)or not 0<top_p<=1:raise ValueError('sampling distribution settings')
 values=torch.as_tensor(logprobs,dtype=torch.float32)
 if values.ndim!=2 or values.shape[0]<1 or values.shape[1]<2 or not torch.isfinite(values).all():raise RuntimeError('nonfinite/incomplete model distributions')
 probabilities=torch.softmax(values/temperature,dim=-1)
 if top_p<1:
  sorted_probs,indices=torch.sort(probabilities,dim=-1,descending=True,stable=True)
  sorted_probs=torch.where(sorted_probs.cumsum(-1)-sorted_probs>=top_p,0.,sorted_probs)
  probabilities=torch.zeros_like(probabilities).scatter(-1,indices,sorted_probs)
 probabilities=probabilities.double();probabilities/=probabilities.sum(-1,keepdim=True)
 return probabilities

def verify_intervals(logprobs,tokens,uniforms,temperature,top_p,error):
 """All positions are checked in a single vectorized prefill distribution pass."""
 import torch
 from .audit_policy import InvalidSample
 if type(error)not in(int,float)or not math.isfinite(error)or not 0<=error<=1e-3:raise ValueError('bounded calibrated CDF error')
 probs=distribution(logprobs,temperature,top_p);n,v=probs.shape
 if len(tokens)!=n or len(uniforms)!=n or any(type(t)is not int or not 0<=t<v for t in tokens):raise InvalidSample('forced sampling token framing')
 if any(type(u)not in(int,float)or not math.isfinite(u)or not 0<=u<1 for u in uniforms):raise ValueError('trusted forced uniform framing')
 indices=torch.tensor(tokens,dtype=torch.int64,device=probs.device);positions=torch.arange(n,device=probs.device);cdf=probs.cumsum(-1);cdf[:,-1]=1.
 high=cdf[positions,indices];low=high-probs[positions,indices];u=torch.tensor(uniforms,dtype=torch.float64,device=probs.device);mass=probs[positions,indices]
 # A zero-mass nucleus exclusion is not legalized by boundary tolerance.
 if bool((mass<=0).any()):raise SupportMismatch('token excluded by reference nucleus')
 if bool(((u<low-error)|(u>=high+error)).any()):raise InvalidSample('forced CDF interval outside calibrated region')
 outside=(u<low)|(u>=high)
 if error and bool(outside.any()):raise NumericalAmbiguity('forced CDF boundary within calibrated numerical uncertainty')
 if bool(outside.any()):raise InvalidSample('forced CDF interval mismatch')
 return dict(positions=n,cdf_abs_error=error,all_intervals_verified=True)

def _threeway_unknown(reason,positions):
 error=NumericalAmbiguity(reason)
 found=positions.nonzero(as_tuple=False).flatten().tolist()
 error.uncertain_positions=found[:128];error.uncertain_position_count=len(found)
 return error

def verify_threeway_intervals(logprobs,tokens,uniforms,temperature,top_p,error):
 import torch
 from .audit_policy import InvalidSample
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
 if bool((~supported).any()):raise _threeway_unknown('reference nucleus support excluded; no cached adjudication',~supported)
 # Unlike the existing v3 fast path, BOTH sides of each boundary uncertainty
 # band remain unknown, including a draw just inside the prefill interval.
 ambiguous=(u<low+error)|(u>=high-error)
 if bool(ambiguous.any()):raise _threeway_unknown('forced CDF boundary within calibrated uncertainty; no cached adjudication',ambiguous)
 return {'positions':n,'all_intervals_verified':True,'cdf_abs_error':error,
         'sampling_assurance':'calibrated-interior-prescribed-CDF',
         'autoregressive_fallback':False}


def verify_sampling(runtime,rollout,turn_index,prompt,output,logprobs):
 from .forced_sampling import uniform
 from .audit_policy import InvalidSample
 context=runtime.sampling_context;config=runtime.harness;p=runtime.fast_sampling_calibration
 stop=runtime.tokenizer.eos_token_id
 if any(token==stop for token in output[:-1])or len(output)<config['max_output_tokens']and output[-1]!=stop:raise InvalidSample('forced generation stop condition')
 draws=[uniform(context,runtime.spec.id,rollout['task_hash'],rollout['index'],rollout['seed'],turn_index,i)for i in range(len(output))]
 if context['contract']['version']==THREEWAY_VERSION:
  return verify_threeway_intervals(logprobs,output,draws,config['temperature'],config['top_p'],p['cdf_abs_error'])
 try:return verify_intervals(logprobs,output,draws,config['temperature'],config['top_p'],p['cdf_abs_error'])
 except (SupportMismatch,NumericalAmbiguity) as uncertainty:
  if context['contract']['version']!=SUPPORT_VERSION:raise
  try:return verify_cached_reference(runtime,prompt,output,rollout['seed'],turn_index,rollout['index'],rollout['task_hash'])
  except InvalidSample:raise
  except Exception as exc:
   if isinstance(uncertainty,NumericalAmbiguity):raise NumericalAmbiguity('cached reference unavailable for ambiguous interval')from exc
   raise


def verify_cached_reference(runtime,prompt,output,attempt,turn,index,task_hash):
 """Nucleus support adjudication uses actual pinned cached sampling, not a bound.

 Recompute the claimed sequence under original public draws. Every prefix must
 match before feeding it back. No synthetic trace becomes valid via ambiguity.
 """
 import torch
 from .forced_sampling import uniform,pick,validate_attempt
 context=runtime.sampling_context;validate_attempt(context,attempt);config=runtime.harness;device=next(runtime.model.parameters()).device
 cache=None;tokens=torch.tensor([prompt],device=device)
 with torch.inference_mode():
  for position,claimed in enumerate(output):
   result=runtime.model(tokens,past_key_values=cache,use_cache=True)
   expected=pick(result.logits[0,-1],uniform(context,runtime.spec.id,task_hash,index,attempt,turn,position),config['temperature'],config['top_p'])
   if claimed!=expected:raise InvalidSample('original cached sampler reference mismatch')
   cache=result.past_key_values;tokens=torch.tensor([[expected]],device=device)
 return dict(positions=len(output),cached_reference_adjudication=True,all_intervals_verified=True)

def cached_sample(runtime,prompt,attempt,turn,index,task_hash):
 import torch
 from .forced_sampling import uniform,pick,validate_attempt,validate_harness
 context=runtime.sampling_context;validate_attempt(context,attempt);config=validate_harness(runtime.harness,context['contract']);device=next(runtime.model.parameters()).device
 output=[];cache=None;tokens=torch.tensor([prompt],device=device)
 with torch.inference_mode():
  for position in range(config['max_output_tokens']):
   result=runtime.model(tokens,past_key_values=cache,use_cache=True)
   token=pick(result.logits[0,-1],uniform(context,runtime.spec.id,task_hash,index,attempt,turn,position),config['temperature'],config['top_p'])
   output.append(token);cache=result.past_key_values;tokens=torch.tensor([[token]],device=device)
   if token==runtime.tokenizer.eos_token_id:break
 return output

def measure_cached_prefill(runtime,prompt,*,checkpoint,attempt=0,turn=0,index=0,task_hash,max_tokens=64):
 """Executed honest control, not a qualification synthesized from metadata.

 Operator separately admits these measured bounds and cross-device controls.
 This function never signs a policy or calls successful controls production.
 """
 import time,torch
 from .forced_sampling import uniform,pick,validate_attempt
 context=runtime.sampling_context;validate_attempt(context,attempt)
 if type(max_tokens)is not int or not 1<=max_tokens<=runtime.harness['max_output_tokens']:raise ValueError('bounded calibration tokens')
 config=runtime.harness;device=next(runtime.model.parameters()).device;tokens=torch.tensor([prompt],device=device);cache=None;output=[];logprobs=[];draws=[];started=time.monotonic()
 with torch.inference_mode():
  for position in range(max_tokens):
   result=runtime.model(tokens,past_key_values=cache,use_cache=True);logits=result.logits[0,-1];draw=uniform(context,runtime.spec.id,task_hash,index,attempt,turn,position)
   token=pick(logits,draw,config['temperature'],config['top_p']);draws.append(draw);output.append(token);logprobs.append(torch.log_softmax(logits.float(),-1).cpu());cache=result.past_key_values;tokens=torch.tensor([[token]],device=device)
   if token==runtime.tokenizer.eos_token_id:break
 generation_seconds=time.monotonic()-started;started=time.monotonic();acts,prefill=runtime.compute(prompt,output);prefill_seconds=time.monotonic()-started
 cached=torch.stack(logprobs);prefill_tensor=torch.as_tensor(prefill,dtype=torch.float32)
 c1=distribution(cached,config['temperature'],config['top_p']).cumsum(-1);c2=distribution(prefill_tensor,config['temperature'],config['top_p']).cumsum(-1)
 error=float((c1-c2).abs().max());maximum_logprob_error=float((cached-prefill_tensor).abs().max())
 return dict(version=CALIBRATION,qualification_claim=False,checkpoint=checkpoint,runtime_revision=getattr(runtime,'runtime_revision','cpu'),harness_sha256=digest(__import__('subnet.harness',fromlist=['normalize']).normalize(config)),fast_verifier_source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),prompt_tokens=len(prompt),output_tokens=len(output),output_sha256=digest(output),measured_cdf_abs_error=error,measured_logprob_abs_error=maximum_logprob_error,cached_generation_seconds=generation_seconds,single_prefill_seconds=prefill_seconds,actual_model_forwards=len(output)+1,used_actual_cached_generation=True,used_actual_teacherforced_prefill=True,cross_device_qualified=False)


def policy_from_executed_controls(reports,*,checkpoint,model_runtime_revision,backend_profile,harness,safety_factor=4.):
 """Unsigned proposal from real controls; ROOT independently admits its evidence.

 Re-run for every successor checkpoint. Bounds above the hard policy maxima
 refuse this runtime rather than silently widening the sampler acceptance.
 """
 if type(reports)is not list or not 2<=len(reports)<=16 or type(safety_factor)not in(int,float)or not 1<=safety_factor<=16:raise ValueError('bounded repeated executed calibration controls')
 from .harness import normalize
 for report in reports:
  if report.get('version')!=CALIBRATION or report.get('checkpoint')!=checkpoint or report.get('runtime_revision')!=model_runtime_revision or report.get('harness_sha256')!=digest(normalize(harness)) or report.get('used_actual_cached_generation')is not True or report.get('used_actual_teacherforced_prefill')is not True or report.get('actual_model_forwards')!=report.get('output_tokens',-1)+1:raise ValueError('same-checkpoint/runtime actual honest calibration controls')
 cdf=max(r['measured_cdf_abs_error']for r in reports)*safety_factor;prob=max(r['measured_logprob_abs_error']for r in reports)*safety_factor
 proposal=dict(version=CALIBRATION,checkpoint=checkpoint,model_runtime_revision=model_runtime_revision,backend_profile_sha256=digest(backend_profile),harness_sha256=digest(normalize(harness)),report_sha256=digest(reports),cdf_abs_error=max(cdf,1e-8),logprob_atol=max(prob,1e-5),toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
 return calibration(proposal)
