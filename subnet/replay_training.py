"""Prospective signed historical replay admission for a current training job.

Historical full audits are operator admissions. Fresh current-model numerical
checks and original environment replay still precede optimizer consumption.
"""
import copy
from . import verified_replay_pool as r

REVISION='signed-current-checkpoint-balanced-training-replay-v1'

def merge_pairs(fresh,historical,reuse_counts):
    """One contribution per family, preferring genuine current-epoch data.

    Among historical candidates, rotate toward least-used targets. Multiple
    current pairs per family require a separately approved weighted objective.
    """
    live={}
    for row in fresh:
        name=row[0]['env_id']
        if name in live:raise ValueError('multiple fresh pairs per family need approved weighting')
        live[name]=row
    choices={}
    for row in historical:
        name=row[0]['env_id']
        if name in live:continue
        target=r.digest({'environment_id':name,'environment_index':row[1]['index'],'task_hash':row[1]['task_hash'],'positive_rollout_sha256':r.digest(row[1]),'negative_rollout_sha256':r.digest(row[2])})
        priority=(reuse_counts.get(target,0),row[1]['index'],target)
        if name not in choices or priority<choices[name][0]:choices[name]=(priority,row,target)
    pairs=[live[name] if name in live else choices[name][1] for name in sorted(set(live)|set(choices))]
    return pairs,{value[2] for value in choices.values()}

def admitted(manifest,envelope,authority):
    if not isinstance(envelope,dict) or set(envelope)!={'manifest','pool','reuse_counts'}:
        raise ValueError('exact signed replay inputs')
    current=r.authenticated(envelope['manifest'],authority)
    if manifest.get('payable') is not False or not r.exact(r.checkpoint(current),r.checkpoint(manifest)):
        raise ValueError('nonpayable current training checkpoint')
    r.compatibility(current);r.compatibility(manifest)
    if not isinstance(current.get('model_runtime_revision'),str) or not current['model_runtime_revision'] or not isinstance(current.get('backend_profile'),dict) or not current['backend_profile'] or not isinstance(current.get('numerical_policy'),dict) or not current['numerical_policy']:
        raise ValueError('explicit approved runtime and numerical profile')
    for field in ('model_id','model_runtime_revision','backend_profile','numerical_policy','tokenizer_binding'):
        if not r.exact(current.get(field),manifest.get(field)):
            raise ValueError('training/replay runtime and model compatibility')
    definitions=r.definitions(current);r.heldout_registry(current,definitions)
    if not r.exact(current['heldout_indices'],manifest.get('heldout_indices')):
        raise ValueError('live signed complete heldout registry')
    live=r.definitions(manifest)
    if set(definitions)!=set(live):raise ValueError('exact live environment inventory')
    for name,row in definitions.items():
        if not r.exact(row['spec'],live[name]['spec']) or not r.exact(row['harness'],live[name]['harness']):
            raise ValueError('live trusted environment/harness version')
    pool=r.authenticated(envelope['pool'],authority)
    if pool.get('current_manifest_sha256')!=r.digest(envelope['manifest']):
        raise ValueError('pool/current manifest lineage')
    selection=r.select_pool(envelope['pool'],authority,envelope['reuse_counts'])
    for entry in selection['selected']:
        if entry.get('version')!=r.VERSION or entry.get('current_manifest_sha256')!=r.digest(envelope['manifest']) or not r.exact(entry.get('current_checkpoint'),current['checkpoint']):
            raise ValueError('validated current replay entry')
        definition=definitions[entry['environment_id']]
        if entry['environment_index'] not in definition['indices'] or entry['environment_index'] in current['heldout_indices'][entry['environment_id']]:
            raise ValueError('approved training index exclusion')
        for label in ('positive','negative'):
            rollout=entry[label]
            if rollout.get('classification')!=label or rollout.get('env_id')!=entry['environment_id'] or rollout.get('index')!=entry['environment_index'] or rollout.get('task_hash')!=entry['task_hash']:
                raise ValueError('exact replay outcome/task')
    return current,selection

def verified_pairs(runtime,manifest,envelope,authority):
    current,selection=admitted(manifest,envelope,authority);definitions=r.definitions(current)
    pairs=[];checks=[]
    for entry in selection['selected']:
        definition=definitions[entry['environment_id']];runtime.configure(definition['spec'],definition['harness'])
        for label in ('positive','negative'):
            claimed=copy.deepcopy(entry[label]);arrays=[]
            for turn in claimed['turns']:
                activations,probabilities=runtime.compute(turn['prompt'],turn['output'])
                turn['proofs']=runtime.build_proofs(activations,decode_batching_size=16,topk=128);arrays.append(probabilities)
            if not runtime.verify(claimed,arrays):raise ValueError('fresh current probability/proof/native replay')
        pairs.append((definition,entry['positive'],entry['negative']))
        checks.append({'env_id':entry['environment_id'],'index':entry['environment_index'],'target_sha256':entry['target_sha256'],'current_checkpoint':manifest['checkpoint']['id'],'historical_probabilities_used_as_reference':False,'fresh_current_numerical_native_verification':True})
    return pairs,{'revision':REVISION,'checks':checks,'pool_sha256':selection['pool_sha256'],'proposed_reuse_increments':selection['proposed_reuse_increments'],'optimizer_performed':False}
