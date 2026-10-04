"""Prospective full-forward public-CDF checks; NOT the active E9 sampler.

The verifier must supply its independently recomputed conditional logprobs from
the approved causal model, after authenticating weights/context and checking the
complete probability arrays, TOPLOC and environment. Miner-claimed arrays are
never the authoritative sampling distribution. This module performs no inference.

For top_p=1, per-logprob uncertainty |error|<=delta gives multiplicative weight
uncertainty exp(+-delta/T). A CDF prefix of mass P is therefore in [L(P), U(P)]
where R=exp(2*delta/T), L=P/(P+R*(1-P)), U=R*P/(R*P+1-P).
Accept only draws inside a token's interval for EVERY admissible distribution.
Numerical ambiguity is a separate unpaid/retry outcome, not proof of fraud.

The bound is real-arithmetic mathematics. The additional explicit FP64 margin
assumes the qualified/pinned CPU libm exp and math.fsum implementation; synthetic
tests do not qualify that implementation or any GPU family. This checks sampler
consistency with full-forward conditional distributions, not historical execution
or unbiased selection, nor equivalence to the old prefix-replay sampler.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

VERSION = 'verified-full-forward-robust-cdf-v1'
LOGPROB_ATOL = 1e-5
ARITHMETIC_MARGIN = 1e-12
MAX_VOCABULARY = 200000
MAX_CONTEXT = 8192
MAX_OUTPUT = 2048
FIELDS = {'version','randomness','max_attempts','verification','generation',
          'distribution','logprob_atol','logprob_rtol','top_p','arithmetic_margin'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(value):
    if not isinstance(value,str)or re.fullmatch('[0-9a-f]{64}',value)is None:
        raise ValueError('public sampler SHA256 binding')
    return value


def make_contract(randomness,max_attempts=128):
    return validate_contract(dict(version=VERSION,randomness=randomness,max_attempts=max_attempts,
        verification='all-conditional-rows-robust-public-cdf',
        generation='causal-autoregressive-canonical-fp64-cdf',
        distribution='independently-recomputed-full-forward-logprobs',
        logprob_atol=LOGPROB_ATOL,logprob_rtol=0,top_p=1.0,arithmetic_margin=ARITHMETIC_MARGIN))


def validate_contract(value):
    if not isinstance(value,dict)or set(value)!=FIELDS:
        raise ValueError('exact prospective public-CDF contract')
    expected=dict(version=VERSION,verification='all-conditional-rows-robust-public-cdf',
        generation='causal-autoregressive-canonical-fp64-cdf',
        distribution='independently-recomputed-full-forward-logprobs',
        logprob_atol=LOGPROB_ATOL,logprob_rtol=0,top_p=1.0,arithmetic_margin=ARITHMETIC_MARGIN)
    if any(type(value.get(k))is bool or value.get(k)!=v for k,v in expected.items()):
        raise ValueError('prospective sampler constants/top-p policy')
    digest(value['randomness'])
    if type(value['max_attempts'])is not int or not 2<=value['max_attempts']<=128:
        raise ValueError('public attempt budget')
    return dict(value)


def context_for_manifest(manifest):
    """Caller authenticates the signed manifest; this enforces its exact pins."""
    epoch=manifest.get('epoch')
    if not isinstance(epoch,str)or not 1<=len(epoch)<=200:
        raise ValueError('public epoch binding')
    contract=validate_contract(manifest.get('sampling_contract'))
    if manifest.get('sampling_source_hash')!=source_hash():
        raise ValueError('approved prospective sampler source')
    return dict(epoch=epoch,checkpoint=digest(manifest.get('checkpoint',{}).get('id')),contract=contract)


def _position(context,env_id,task_hash,index,attempt,turn,position):
    if (not isinstance(context,dict)or set(context)!={'epoch','checkpoint','contract'}
            or not isinstance(context['epoch'],str)or not 1<=len(context['epoch'])<=200):
        raise ValueError('exact public draw context')
    validate_contract(context['contract']);digest(context['checkpoint']);digest(task_hash)
    if not isinstance(env_id,str)or not 1<=len(env_id)<=200:
        raise ValueError('public environment identity')
    if (type(attempt)is not int or not 0<=attempt<context['contract']['max_attempts']
            or type(index)is not int or not 0<=index<2**31
            or type(turn)is not int or not 0<=turn<32
            or type(position)is not int or not 0<=position<MAX_OUTPUT):
        raise ValueError('public attempt/position binding')


def uniform(context,env_id,task_hash,index,attempt,turn,position):
    _position(context,env_id,task_hash,index,attempt,turn,position)
    message=dict(context,environment=env_id,task_hash=task_hash,index=index,
                 attempt=attempt,turn=turn,position=position)
    integer=int.from_bytes(hashlib.sha256(canonical(message)).digest()[:8],'big')>>11
    return integer/2**53


def _temperature(temperature):
    if type(temperature)not in(int,float)or not math.isfinite(temperature)or not .05<=temperature<=4:
        raise ValueError('qualified temperature range')
    return float(temperature)


def _row(row):
    if (not isinstance(row,np.ndarray)or row.ndim!=1 or row.dtype not in(np.dtype('float32'),np.dtype('float64'))
            or not 1<=row.size<=MAX_VOCABULARY or not np.isfinite(row).all()):
        raise ValueError('complete finite recomputed logprob row')
    return row


def _draw(u):
    if type(u)not in(int,float)or not math.isfinite(u)or not 0<=u<1:
        raise ValueError('finite public uniform in [0,1)')


def distribution(row,temperature):
    """CPU FP64, token-ID ordering; no GPU softmax/reduction or top-p sorting."""
    row=_row(row);temperature=_temperature(temperature)
    scaled=[float(value)/temperature for value in row]
    maximum=max(scaled)
    weights=[math.exp(value-maximum)for value in scaled]
    total=math.fsum(weights)
    if not math.isfinite(total)or total<=0:raise ValueError('finite normalized sampling distribution')
    return weights,total


def cdf_uncertainty(mass,temperature):
    """Worst possible prefix interval over every per-logprob error <=1e-5."""
    temperature=_temperature(temperature)
    if type(mass)not in(int,float)or not math.isfinite(mass)or not 0<=mass<=1:
        raise ValueError('finite CDF mass')
    # expm1 avoids losing the small perturbation in exp(x)-1 near zero.
    change=math.expm1(2*LOGPROB_ATOL/temperature)
    lower=mass/(1+change*(1-mass))
    upper=(1+change)*mass/(1+change*mass)
    return lower,upper


def select_token(row,u,temperature):
    """Prospective miner helper; qualification must bind its CPU implementation.

    Miners should additionally compute the final full-forward rows and self-check
    the complete rollout, retrying another allowed attempt when ambiguous. A
    prefix pick alone does not establish the final full-forward acceptance claim.
    """
    _draw(u);weights,total=distribution(row,temperature)
    # Search boundaries using math.fsum on a prefix, rather than GPU cumulative
    # reductions or a cumulative rounding drift over a large vocabulary.
    low,high=0,len(weights)
    while low<high:
        middle=(low+high)//2
        boundary=math.fsum(weights[:middle+1])/total
        if boundary<=u:low=middle+1
        else:high=middle
    if low==len(weights)or weights[low]<=0:raise ValueError('invalid canonical sampling interval')
    return low


def check_token(row,u,token,temperature):
    """Return accepted, numerical_ambiguity or sampler_mismatch; no broad tolerance.

    The possible interval detects definite mismatches, but ONLY the robust
    all-distributions interval can earn acceptance. Accepting the possible
    interval would allow adversarial within-tolerance probability nudging.
    """
    _draw(u);weights,total=distribution(row,temperature)
    if type(token)is not int or not 0<=token<len(weights):raise ValueError('submitted token identity')
    before=math.fsum(weights[:token])/total
    through=math.fsum(weights[:token+1])/total
    low_before,high_before=cdf_uncertainty(before,temperature)
    low_through,high_through=cdf_uncertainty(through,temperature)
    margin=ARITHMETIC_MARGIN
    if weights[token]>0 and high_before+margin<u<low_through-margin:
        status='accepted'
    elif u<low_before-margin or u>=high_through+margin:
        status='sampler_mismatch'
    else:status='numerical_ambiguity'
    return dict(status=status,robust_lower=high_before,robust_upper=low_through,
                possible_lower=low_before,possible_upper=high_through,
                arithmetic_margin=margin)


def causal_rows(full_forward_logprobs,forward_input_ids,prompt_tokens,output_tokens):
    """Extract row prompt_len-1+t, which predicts submitted output token t.

    Requires the ORIGINAL inputs of the verifier's approved causal model call.
    These equality/geometry checks do not themselves prove attention causality;
    source/model/masks must be independently approved by the surrounding verifier.
    """
    if (not isinstance(prompt_tokens,list)or not prompt_tokens or not isinstance(output_tokens,list)or not output_tokens
            or not isinstance(forward_input_ids,list)or forward_input_ids!=prompt_tokens+output_tokens
            or len(forward_input_ids)>MAX_CONTEXT):
        raise ValueError('exact causal forward input/trajectory alignment')
    if (not isinstance(full_forward_logprobs,np.ndarray)or full_forward_logprobs.ndim!=2
            or full_forward_logprobs.shape[0]!=len(forward_input_ids)
            or full_forward_logprobs.dtype not in(np.dtype('float32'),np.dtype('float64'))
            or not 1<=full_forward_logprobs.shape[1]<=MAX_VOCABULARY):
        raise ValueError('exact full-forward probability geometry')
    if any(type(value)is not int or not 0<=value<full_forward_logprobs.shape[1]for value in forward_input_ids):
        raise ValueError('causal input token types/bounds')
    return full_forward_logprobs[len(prompt_tokens)-1:len(prompt_tokens)+len(output_tokens)-1]


def verify_turn(context,env_id,task_hash,index,attempt,turn,*,prompt_tokens,output_tokens,
                forward_input_ids,full_forward_logprobs,temperature,max_output_tokens,eos_token_id):
    """Check all choices and exact stopping, using no autoregressive model calls.

    This deliberately accepts no miner-provided draws, row offsets, temperature
    overrides, alternate top-p policy or claimed probability distribution. The
    caller obtains temperature/limits/EOS from the approved harness/tokenizer.
    """
    _position(context,env_id,task_hash,index,attempt,turn,0);_temperature(temperature)
    if type(max_output_tokens)is not int or not 1<=max_output_tokens<=MAX_OUTPUT:
        raise ValueError('signed output-token budget')
    if type(eos_token_id)is not int or not 0<=eos_token_id<MAX_VOCABULARY:
        raise ValueError('approved tokenizer EOS identity')
    rows=causal_rows(full_forward_logprobs,forward_input_ids,prompt_tokens,output_tokens)
    if eos_token_id>=full_forward_logprobs.shape[1]:
        raise ValueError('approved EOS must belong to the model vocabulary')
    if (not 1<=len(output_tokens)<=max_output_tokens or eos_token_id in output_tokens[:-1]
            or output_tokens[-1]!=eos_token_id and len(output_tokens)!=max_output_tokens):
        raise ValueError('exact EOS or output-budget stopping')
    for position,(token,row)in enumerate(zip(output_tokens,rows)):
        u=uniform(context,env_id,task_hash,index,attempt,turn,position)
        decision=check_token(row,u,token,temperature)
        if decision['status']!='accepted':
            return dict(version=VERSION,status=decision['status'],valid=False,
                checked_tokens=position+1,first_rejected_position=position,
                complete_output_checked=False,historical_execution_proven=False,
                numerical_ambiguity_is_fraud=False)
    return dict(version=VERSION,status='accepted',valid=True,checked_tokens=len(output_tokens),
        complete_output_checked=True,historical_execution_proven=False,
        numerical_ambiguity_is_fraud=False,
        assurance='public-sampler-consistency-with-verified-full-forward-conditionals')
