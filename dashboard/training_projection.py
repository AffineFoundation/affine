"""Closed-epoch public training counters from original ROOT-signed evidence.

No storage client, credentials, signing or rollout publication. The optional
receipt cache is disposable; every receipt must bind the signed finalization.
"""
import math
import re
import time
from .learner_projection import AUTHORITY, authenticated, digest
DIGEST=re.compile(r'[0-9a-f]{64}')

def finite(v):return type(v)in(int,float)and math.isfinite(v)
def integer(v):return type(v)is int and v>=0

def project(manifest_document,completion_document,training_document=None,*,authority=None,now=None):
    authority=AUTHORITY if authority is None else authority
    now=time.time()if now is None else now
    manifest=authenticated(manifest_document,authority)
    completion=authenticated(completion_document,authority)
    epoch=manifest['epoch'];start=manifest['checkpoint']['id'];end=completion['next_checkpoint']
    if (completion.get('epoch')!=epoch or completion.get('checkpoint')!=start
            or not DIGEST.fullmatch(start)or not DIGEST.fullmatch(end)
            or not finite(manifest.get('deadline'))or not finite(completion.get('completed_at'))
            or not manifest['deadline']<=completion['completed_at']<=now
            or not integer(completion.get('round'))):
        raise ValueError('exact closed epoch/checkpoint/time binding')
    result=dict(finalized=True,finalization_source='authenticated-learner-completion',
                completion_sha256=digest(completion_document))
    if training_document is None:return result
    training=authenticated(training_document,authority)
    if training.get('status')=='closed_no_eligible_batches':
        if set(training)!={'status','checkpoint'}or training['checkpoint']!=start or start!=end:
            raise ValueError('exact authenticated no-update closure')
        result['training']=dict(steps=0,losses=[],weights_changed=False,checkpoint=end,objective=None,input_pairs=0)
    else:
        if (training.get('source_epoch')!=epoch or training.get('input_checkpoint')!=start
                or training.get('checkpoint')!=end):raise ValueError('original signed training receipt scope')
        steps=training.get('steps');changed=training.get('weights_changed')
        if not integer(steps)or type(changed)is not bool:raise ValueError('authentic training step count')
        updates=training.get('updates');diagnostics=training.get('persistent_diagnostics')
        if updates is not None or diagnostics is not None:
            if (type(updates)is not list or len(updates)!=steps or type(diagnostics)is not dict
                    or training.get('state_authority_committed')is not True
                    or diagnostics.get('state_staged')is not True
                    or diagnostics.get('input_checkpoint')!=start
                    or any(type(u)is not dict for u in updates)
                    or diagnostics.get('optimizer_steps')!=steps
                    or not integer(diagnostics.get('pair_count'))
                    or not integer(diagnostics.get('task_count'))):raise ValueError('complete original update diagnostics')
            # Worker diagnostics retain complete=False before authority ACK;
            # the signed receipt's state_authority_committed is the final gate.
            losses=[u.get('loss')for u in updates]
            objective=training.get('training_policy')
            if (any(not finite(x)for x in losses)or type(objective)is not str
                    or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',objective)
                    or any(u.get('training_policy')!=objective for u in updates)):
                raise ValueError('reported finite losses and exact training policy')
            pairs=diagnostics['pair_count']
        else:
            # Historical top-level format remains explicit, never zero-filled.
            losses=training.get('losses');objective=training.get('objective');pairs=training.get('input_pairs')
            if losses is not None and(type(losses)is not list or any(not finite(x)for x in losses)):
                raise ValueError('historical finite loss list')
            if pairs is not None and not integer(pairs):raise ValueError('historical pair count')
            if objective is not None and(type(objective)is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',objective)):
                raise ValueError('historical objective name')
        result['training']=dict(steps=steps,losses=losses,weights_changed=changed,checkpoint=end,objective=objective,input_pairs=pairs)
    result['training_evidence_source']='authenticated-finalized-training-receipt'
    result['training_receipt_sha256']=digest(training_document)
    return result
