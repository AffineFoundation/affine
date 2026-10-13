"""Structural complete-group admission; rewards/proofs remain unaudited.

Callers authenticate the authority manifest/source first. Server callers obtain
TaskAuthority from their pinned native task/tokenizer resolver, never miner JSON.
This module verifies identity signatures, canonical bytes and all original slots;
it does not call a model or turn claimed grades into verified outcomes.
"""
import base64
from dataclasses import dataclass
import hashlib
import math
from nacl.signing import VerifyKey
from .storage import canonical
from .batch_contract import for_manifest,CompleteGroupContract,REQUIRED_CLIENT_FEATURES
from .complete_group_codec import limits,loads,need
from .complete_group_mining import validate_group,digest

TRANSPORT = 'complete-group-commitment-v1'
DOCUMENT = 'complete-group-training-document-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_digest(value):
    return type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef')


def context(manifest,miner):
    from .forced_sampling import binding
    contract=for_manifest(manifest,supported_features=REQUIRED_CLIENT_FEATURES)
    need(isinstance(contract,CompleteGroupContract) and manifest.get('submission_transport_policy')==TRANSPORT,
         'explicit complete-group transport contract')
    limits(manifest)
    need(manifest.get('transport_policy')=='direct-r2-v1' and 'artifact_policy' not in manifest
         and type(manifest.get('start')) is int and type(manifest.get('deadline')) is int
         and manifest['start']<manifest['deadline'], 'signed complete-group upload window and transport')
    return contract,binding(manifest,miner)


def scope(manifest,miner):
    _,bound=context(manifest,miner)
    source=manifest.get('source_bundle',{}).get('sha256')
    need(is_digest(source),'source bundle digest required')
    return digest(dict(version=TRANSPORT,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                       source=source,miner=miner,sampling_context_sha256=digest(bound),
                       group_transport_policy=manifest['group_transport_policy']))


def group_id(manifest,miner,env_id,index):
    from .protocol import entry,harness_for
    definition=entry(manifest,env_id);harness_for(definition,index)
    need(type(index) is int and index in definition['indices'],'canonical authorized task')
    return digest(dict(scope=scope(manifest,miner),env_id=env_id,index=index))


@dataclass(frozen=True)
class TaskAuthority:
    epoch: str
    checkpoint: str
    env_id: str
    environment_version: str
    index: int
    task_hash: str
    prompt: tuple[int,...]
    eos_token_id: int
    vocab_size: int
    eos_token_ids: tuple[int,...] = ()


def local_task(batch,manifest,*,eos_token_id,vocab_size,eos_token_ids=()):
    """Client-side consistency only. Never use this to mint server authority."""
    return TaskAuthority(manifest['epoch'],manifest['checkpoint']['id'],batch['env_id'],
        batch['environment_version'],batch['index'],batch['task_hash'],
        tuple(batch['rollouts'][0]['turns'][0]['prompt']),eos_token_id,vocab_size,tuple(eos_token_ids))


def structural_batch(batch,manifest,miner,task,arrays=None):
    from .protocol import entry,harness_for
    from .proofs import validate_framing
    contract,bound=context(manifest,miner)
    validate_group(batch,manifest,contract,bound,miner)
    need(type(task) is TaskAuthority,'caller-owned native task authority required')
    definition=entry(manifest,task.env_id);h=harness_for(definition,task.index)
    budget=limits(manifest)
    token_policy=manifest['group_transport_policy']
    from .complete_group_distribution import eos_ids
    stops = eos_ids(token_policy)
    actual_stops = task.eos_token_ids or (task.eos_token_id,)
    need(actual_stops == stops and task.eos_token_id in stops and task.vocab_size==token_policy['vocab_size'],
         'trusted tokenizer geometry differs from signed transport profile')
    need(task.epoch==manifest['epoch'] and task.checkpoint==manifest['checkpoint']['id']
         and type(task.index) is int and task.index in definition['indices']
         and task.environment_version==definition['spec']['version'] and is_digest(task.task_hash),
         'trusted native task scope')
    need(type(task.vocab_size) is int and 2<=task.vocab_size<=1_000_000 and type(task.eos_token_id) is int
         and 0<=task.eos_token_id<task.vocab_size and type(task.prompt) is tuple and task.prompt
         and all(type(t) is int and 0<=t<task.vocab_size for t in task.prompt)
         and len(task.prompt)+h['max_output_tokens']<=budget.context,'trusted canonical prompt/EOS/support geometry')
    need(all(batch[k]==getattr(task,k) for k in ('env_id','environment_version','index','task_hash')),'batch native task authority')
    if arrays is not None:
        need(type(arrays) is list and len(arrays)==contract.group_size,'all original selected probability arrays')
    row_fields={'schema','env_id','environment_version','index','sample_index','seed','env_seed','task_hash',
                'reward','classification','turns','sampling'}
    turn_fields={'prompt','output','text','proofs','observations','done','reward','classification',
                 'stop_reason','behavior_sampling'}
    counts={'positive':0,'negative':0,'neutral':0}
    for nonce,row in enumerate(batch['rollouts']):
        turn=row['turns'][0];budget_stop=turn['stop_reason']=='budget'
        extra={'native_classification'} if budget_stop else set()
        need(set(row)==row_fields|extra and row['schema']==2,'exact versioned rollout fields')
        need(set(turn)==turn_fields|extra,'exact versioned turn fields')
        need(type(row['env_seed']) is int and row['env_seed']==int(definition['spec'].get('config',{}).get('seed',0)),
             'native environment seed')
        output=turn['output'];need(tuple(turn['prompt'])==task.prompt,'regenerated canonical prompt required')
        need(all(type(t) is int and 0<=t<task.vocab_size for t in output),'supported output token IDs')
        need(not any(t in stops for t in output[:-1]),'no generated tokens after EOS')
        if budget_stop:
            need(len(output)==h['max_output_tokens'] and output[-1] not in stops
                 and row['classification']=='neutral' and row['native_classification'] in counts
                 and turn['native_classification']==row['native_classification'],'budget cap is unresolved')
        else:
            need(output[-1] in stops,'EOS claim must end at actual checkpoint EOS')
        need(type(turn['text']) is str and len(turn['text'].encode())<=64*h['max_output_tokens'], 'bounded generated text')
        need(type(turn['observations']) is list and len(turn['observations'])<=16,'bounded observations')
        for obs in turn['observations']:
            need(type(obs) is dict and set(obs)=={'role','content'} and obs['role'] in ('user','tool')
                 and type(obs['content']) is str and len(obs['content'].encode())<=4096,'explicit bounded observations')
        validate_framing(turn['proofs'],1+math.ceil(len(output)/16),128)
        if arrays is not None:
            import numpy as np
            value=arrays[nonce]
            need(type(value) is list and len(value)==1 and isinstance(value[0],np.ndarray)
                 and value[0].dtype==np.float32 and value[0].shape==(len(output),1)
                 and np.isfinite(value[0]).all() and (value[0]<=0).all(), 'selected-T1 arrays bind every actual output token')
        counts[row['classification']]+=1
    return dict(group_id=group_id(manifest,miner,task.env_id,task.index),batch_sha256=sha(canonical(batch)),
                counts=counts,rollouts=contract.group_size,assurance='structurally-eligible-unaudited',
                sampling_verified=False,native_outcomes_verified=False)


def document(batch,manifest,miner,slot):
    contract,_=context(manifest,miner)
    need(type(slot) is int and 0<=slot<contract.max_batches,'signed group slot')
    value=dict(version=DOCUMENT,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],miner=miner,
               manifest_sha256=digest(manifest),scope_sha256=scope(manifest,miner),slot=slot,
               group_id=group_id(manifest,miner,batch['env_id'],batch['index']),batch=batch)
    body=canonical(value);need(len(body)<=limits(manifest).metadata_bytes,'bounded complete-group compact document')
    return body


def validate_document(data,manifest,miner,claim,task):
    need(type(data) is bytes and sha(data)==claim['document_sha256'] and len(data)==claim['document_size'],
         'compact document committed bytes')
    value=loads(data,limits(manifest).metadata_bytes)
    need(type(value) is dict and set(value)=={'version','epoch','checkpoint','miner','manifest_sha256',
         'scope_sha256','slot','group_id','batch'} and value['version']==DOCUMENT,'exact complete-group document fields')
    need(value==loads(document(value['batch'],manifest,miner,claim['slot']),limits(manifest).metadata_bytes),
         'canonical document scope and slot')
    result=structural_batch(value['batch'],manifest,miner,task)
    need(result['batch_sha256']==claim['batch_sha256'] and result['group_id']==claim['group_id']
         and value['batch']['env_id']==claim['env_id'] and value['batch']['index']==claim['index'],
         'complete-group document declared identity')
    return value,result


def signed_commitment(identity,manifest,prepared):
    claims=[]
    for slot,(batch,data) in enumerate(prepared):
        body=document(batch,manifest,identity.id,slot)
        claims.append(dict(slot=slot,env_id=batch['env_id'],index=batch['index'],
                           group_id=group_id(manifest,identity.id,batch['env_id'],batch['index']),
                           batch_sha256=sha(canonical(batch)),artifact_sha256=sha(data),artifact_size=len(data),
                           document_sha256=sha(body),document_size=len(body)))
    payload=dict(version=TRANSPORT,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                 source=manifest['source_bundle']['sha256'],miner=identity.id,manifest_sha256=digest(manifest),
                 scope_sha256=scope(manifest,identity.id),groups=claims)
    signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode()
    data=canonical(dict(payload=payload,signer=identity.id,signature=signature))
    validate_commitment(data,manifest,identity.id,registered_miners={identity.id})
    return data


def validate_commitment(data,manifest,miner,*,registered_miners):
    need(is_digest(miner) and miner in registered_miners,'epoch-authorized miner identity')
    contract,_=context(manifest,miner);budget=limits(manifest)
    envelope=loads(data,budget.commitment_bytes)
    need(type(envelope) is dict and set(envelope)=={'payload','signer','signature'} and envelope['signer']==miner,
         'exact miner signature envelope')
    payload=envelope['payload'];need(type(payload) is dict,'commitment object')
    VerifyKey(bytes.fromhex(miner)).verify(canonical(payload),base64.b64decode(envelope['signature'],validate=True))
    need(set(payload)=={'version','epoch','checkpoint','source','miner','manifest_sha256','scope_sha256','groups'}
         and payload['version']==TRANSPORT and payload['epoch']==manifest['epoch']
         and payload['checkpoint']==manifest['checkpoint']['id'] and payload['source']==manifest['source_bundle']['sha256']
         and payload['miner']==miner and payload['manifest_sha256']==digest(manifest)
         and payload['scope_sha256']==scope(manifest,miner),'complete-group commitment scope')
    claims=payload['groups'];need(type(claims) is list and len(claims)<=contract.max_batches,'signed cumulative group cap')
    seen=set();total=0
    for slot,row in enumerate(claims):
        need(type(row) is dict and set(row)=={'slot','env_id','index','group_id','batch_sha256',
             'artifact_sha256','artifact_size','document_sha256','document_size'},'exact committed group fields')
        need(type(row['slot']) is int and row['slot']==slot and type(row['index']) is int,'canonical ordered slots/index')
        expected=group_id(manifest,miner,row['env_id'],row['index'])
        need(row['group_id']==expected and expected not in seen,'one canonical group per task');seen.add(expected)
        for name in ('batch_sha256','artifact_sha256','document_sha256'):
            need(is_digest(row[name]),'complete-group digest')
        need(type(row['artifact_size']) is int and 0<row['artifact_size']<=budget.compressed_bytes
             and type(row['document_size']) is int and 0<row['document_size']<=budget.metadata_bytes,
             'declared per-group byte limits')
        total+=row['artifact_size']
    need(total<=budget.cumulative_bytes,'cumulative group byte limit')
    return envelope


def validate_artifact(data,manifest,miner,claim,task,*,expected_document):
    """Auditor fetch boundary: check exact committed archive, never claim inference.

    Admission does not call this. A later auditor/learner can load all original
    selected-T1 vectors losslessly and require the same signed compact document.
    Native rewards and sampler log probabilities still require trusted replay.
    """
    from .complete_group_codec import decode
    need(type(data) is bytes and len(data)==claim['artifact_size'] and sha(data)==claim['artifact_sha256'],
         'exact committed group artifact bytes')
    document_value,_=validate_document(expected_document,manifest,miner,claim,task)
    batch,arrays=decode(data,limits(manifest))[0]
    need(batch==document_value['batch'],'artifact and compact document describe the same original group')
    assessment=structural_batch(batch,manifest,miner,task,arrays)
    return batch,arrays,assessment
