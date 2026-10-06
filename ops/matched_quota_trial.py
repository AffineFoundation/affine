"""Default-off research fork: authentic sixteen-stream quota comparison.

No production authority/state commit is exposed. Native/CDF verification is
required before a trajectory can enter either arm. A censored task enters
neither arm, preserving an identical matched task population.
"""
import hashlib
import math
from pathlib import Path
from subnet.storage import canonical

VERSION='matched-quota-research-v1'

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def task_plan(mining_indices,excluded,seed,count=8):
    if type(count)is not int or not 1<=count<=16 or type(seed)is not str or len(seed)!=64:
        raise ValueError('bounded predeclared task plan')
    pool=sorted(set(mining_indices)-set(excluded))
    if len(pool)<count or any(type(x)is not int or x<0 for x in pool):raise ValueError('task leakage/framing')
    return sorted(pool,key=lambda i:hashlib.sha256(bytes.fromhex(seed)+canonical(i)).digest())[:count]

def validate_stream(rows,index):
    if len(rows)!=16 or [x['attempt']for x in rows]!=list(range(16)):
        raise ValueError('all original prescribed attempts, no cherry picking')
    seen=set();task_hashes=set()
    for x in rows:
        if x['index']!=index or type(x.get('native_verified'))is not bool or type(x.get('sampler_verified'))is not bool:
            raise ValueError('exact native/sampler evidence types')
        if x['status'] not in ('verified','numerical_unknown','native_error'):raise ValueError('outcome framing')
        if x['status']=='verified':
            if not x['native_verified']or not x['sampler_verified']:raise ValueError('claims are not verification')
            r=x['rollout']; category=r['classification'];reward=r['reward'];task_hashes.add(r['task_hash'])
            if category not in ('positive','negative') or type(reward)not in(int,float) or not math.isfinite(reward) or reward!=int(category=='positive'):
                raise ValueError('binary native class')
            if r['seed']!=x['attempt']or r['index']!=index:raise ValueError('attempt/task binding')
            key=digest([dict(prompt=t['prompt'],output=t['output'])for t in r['turns']])
            if key in seen:raise ValueError('duplicate trajectory cannot fill quota')
            seen.add(key)
    if len(task_hashes)>1:raise ValueError('all prescribed streams must share the native task')
    return rows

def select_matched(streams,definition):
    arms={'1P1N':[],'2P2N':[]};supply=[]
    for index,rows in streams.items():
        validate_stream(rows,index)
        positive=[x for x in rows if x['status']=='verified'and x['rollout']['classification']=='positive']
        negative=[x for x in rows if x['status']=='verified'and x['rollout']['classification']=='negative']
        def completion(k,prefix):
            p=[x for x in positive if x['attempt']<prefix];n=[x for x in negative if x['attempt']<prefix]
            return max(p[k-1]['attempt'],n[k-1]['attempt'])+1 if min(len(p),len(n))>=k else None
        lengths={name:[sum(len(t['output'])for t in x['rollout']['turns'])for x in items]for name,items in [('positive',positive),('negative',negative)]}
        supply.append(dict(index=index,positive=len(positive),negative=len(negative),unknown=sum(x['status']=='numerical_unknown'for x in rows),native_errors=sum(x['status']=='native_error'for x in rows),first1_prefix8=completion(1,8),first2_prefix8=completion(2,8),first1_prefix16=completion(1,16),first2_prefix16=completion(2,16),matched_included=min(len(positive),len(negative))>=2,class_token_lengths=lengths,class_cap_counts={k:sum(n>=1024 for n in v)for k,v in lengths.items()}))
        if min(len(positive),len(negative))<2:continue
        for name,k in [('1P1N',1),('2P2N',2)]:
            arms[name].extend((definition,positive[i]['rollout'],negative[i]['rollout'])for i in range(k))
    return arms,supply

def verify_generated(runtime,index,attempt):
    """Official rollout and compact TOPLOC/CDF/native verification, not claims."""
    from subnet.fast_prefill_audit import NumericalAmbiguity
    from verifiers.v1.errors import TaskError
    try:r,arrays=runtime.rollout(index,attempt)
    except TaskError:return dict(index=index,attempt=attempt,status='native_error',native_verified=False,sampler_verified=False),None
    row=dict(index=index,attempt=attempt,rollout=r,native_verified=False,sampler_verified=False)
    try:
        if runtime.verify(r,arrays)is not True:raise ValueError('official verifier did not accept')
    except NumericalAmbiguity as e:
        row.update(status='numerical_unknown',native_verified=getattr(e,'environment_verification_complete',False));return row,arrays
    row.update(status='verified',native_verified=True,sampler_verified=True);return row,arrays

def train_arm(runtime,pairs,output,plan,parent,fetch):
    """One actual task-mean BF16 update restored from the SAME durable FP32 parent.

    BF16 branch checkpoint is exported; returned optimizer is disposable research
    RAM, never represented as a durable or production state advance.
    """
    from subnet.persistent_cpu_adamw import parameter_inventory
    from subnet.persistent_training_state import resource_plan,admit_resources,restore_state
    from subnet.task_normalized_training import train_epoch
    _,inventory=parameter_inventory(runtime.model.named_parameters())
    resource=resource_plan(inventory,bf16_export_bytes=plan['bf16_export_bytes'],concurrency=plan['restore_concurrency'])
    admission=admit_resources(output,resource)
    restored,evidence=restore_state(parent,plan['parent_descriptor_sha256'],plan['checkpoint'],inventory,workspace=output,fetch_shard=fetch,resource_admission=admission,concurrency=plan['restore_concurrency'])
    destination,optimizer,metrics=train_epoch(runtime,pairs,output,input_checkpoint=plan['checkpoint'],epoch=plan['epoch'],seed=plan['selection_seed'],steps=1,restored_state=restored,resource_admission=admission)
    if optimizer.global_step!=parent['optimizer_steps']+1:raise ValueError('one isolated parent-lineage update')
    return destination,metrics,evidence
