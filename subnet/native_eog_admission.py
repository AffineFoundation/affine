"""Bind an independently trusted model audit to fresh split native replay.

An authenticated operator model-audit receipt is evidence, not a cryptographic
proof that a GPU executed. This stage independently replays native tools and
grades; it does not recompute model activations itself.
"""
import hashlib
import json
from subnet.backend_jobs import signed, file_map
from subnet.harness import observations
from subnet.native_eog_isolation import canonical, sha
from subnet.native_eog_split import OperatorBroker, PublicActor, request

def validate_model_audit(envelope,authority,public_bytes,approved_model):
    if len(bytes.fromhex(authority))!=32:raise ValueError('independent model authority')
    payload=signed(envelope,authority);artifact=json.loads(public_bytes)
    if payload.get('kind')!='controlled-original-eog-model-audit-v1':raise ValueError('model audit kind')
    if payload.get('public_trace_sha256')!=hashlib.sha256(public_bytes).hexdigest():raise ValueError('public trace bytes')
    for field in ('checkpoint','runtime_profile','harness_source_sha256'):
        if payload.get(field)!=approved_model[field]:raise ValueError('approved model/profile/harness binding')
    if payload.get('checkpoint_files')!=approved_model['checkpoint_files'] or payload['checkpoint']!=file_map(payload['checkpoint_files']):
        raise ValueError('approved model weights')
    if payload.get('numerical_tolerances')!={'TOPLOC_errors':0,'logprobs_atol':1e-5,'logprobs_rtol':0}:
        raise ValueError('strict numerical model verification')
    if payload.get('full_model_recompute') is not True or payload.get('curated_target_model_computation') is not True or payload.get('originally_sampled') is not False:
        raise ValueError('model verification scope')
    events=artifact['events'];records=payload.get('records')
    if not isinstance(records,list) or len(records)!=len(events) or not events:raise ValueError('full trajectory coverage')
    messages=artifact['public']['messages']
    for i,(event,record) in enumerate(zip(events,records)):
        action={'name':event['name'],'arguments':event['arguments']}
        expected={'turn_index':i,'messages_sha256':sha(messages),'action_sha256':sha(action),
                  'observation_sha256':hashlib.sha256(event['observation'].encode()).hexdigest(),'full_proof_verified':True}
        if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('native/model exact turn binding')
        text=canonical({'tool_call':action}).decode()
        messages=messages+[{'role':'assistant','content':text}]+observations([{'role':'tool','content':event['observation']}],{'version':'text-tools-v1'})
    return artifact,payload

def admit(envelope,authority,public_bytes,approved_model,private_task,runtime):
    artifact,payload=validate_model_audit(envelope,authority,public_bytes,approved_model)
    if artifact['runtime']!=runtime:raise ValueError('approved native runtime binding')
    broker=OperatorBroker(private_task,runtime)
    actor=PublicActor(broker.endpoint,broker.actor_capability,sha(broker.public))
    try:
        public=actor.reset()
        if public['task_id']!=artifact['public']['task_id'] or public['messages']!=artifact['public']['messages'] or public['tools']!=artifact['public']['tools'] or public['seed_sha256']!=artifact['original_seed_sha256']:
            raise ValueError('approved original task/public source binding')
        for event in artifact['events']:
            if actor.call(event['name'],event['arguments'])!=event['observation']:
                raise ValueError('original native observation replay')
        actor.close()
        result=request(broker.endpoint,broker.operator_capability,'operator',{'operation':'grade'})
        if result['grade']['reward']!=artifact['claimed_reward']:raise ValueError('original native reward')
        return {'kind':'controlled-original-eog-model-native-admission-v2','passed':True,
                'public_trace_sha256':hashlib.sha256(public_bytes).hexdigest(),
                'model_audit_sha256':sha(envelope),'model_authority':authority,
                'checkpoint':payload['checkpoint'],'runtime_profile':payload['runtime_profile'],
                'public_descriptor_sha256':sha(broker.public),'private_grader_descriptor_sha256':sha(broker.private_grader),
                'native_runtime':runtime,'native_source_files':broker.public['source_files'],
                'original_seed_sha256':broker.public['seed_sha256'],
                'exact_native_tool_calls':result['calls'],'original_reward':result['grade']['reward'],
                'logical_database_sha256':result['logical_database_sha256'],
                'independent_native_replay_performed_here':True,'model_recompute_performed_here':False,
                'authenticated_model_verification_receipt':True,'shared_epoch_verified':False,
                'optimizer_steps':0,'chain_submission':False}
    finally:broker.close()
