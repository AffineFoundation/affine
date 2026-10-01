"""Prospective operator coordinator contracts; no deployment or optimizer calls.

A NEW signed challenge must precede its actual role generation. Qualification
controls cannot be relabeled as current contributions. Full audits precede any
training plan, and current model references must be recomputed by the optimizer.
"""
from .native_tau2_common_artifacts import admit_frozen_sample
from .native_tau2_common_search_contract import validate_epoch,preference_pair,digest,canonical,sha
VERSION='native-tau2-common-frozen-agent-preference-bridge-v1'
REFERENCE='recompute-at-current-approved-agent-checkpoint-v1'

def training_plan(positive,negative,epoch,authority,fixed_user,uid):
    """positive/negative are trusted operator records with explicit freeze/raw/time."""
    manifest=validate_epoch(epoch,authority,fixed_user)
    admitted=[]
    for value in (positive,negative):
        if set(value)!={'freeze','raw','received_at'}:raise ValueError('exact private frozen role artifact inputs')
        admitted.append(admit_frozen_sample(value['freeze'],value['raw'],epoch,authority,fixed_user,uid,value['received_at']))
    pair=preference_pair(admitted[0]['view'],admitted[1]['view'])
    return {'version':VERSION,'manifest_sha256':digest(manifest),'registered_uid':uid,'current_agent_checkpoint':manifest['checkpoint'],'fixed_user_sha256':digest(fixed_user),'positive_zip_sha256':admitted[0]['zip_sha256'],'negative_zip_sha256':admitted[1]['zip_sha256'],'positive_freeze_sha256':admitted[0]['operator_freeze_sha256'],'negative_freeze_sha256':admitted[1]['operator_freeze_sha256'],'pair':pair,'reference_policy':REFERENCE,'historical_logprobs_used_as_current_reference':False,'loss_scope':'same-context-divergent-agent-decision-only','all_auxiliary_tokens_in_loss':False,'native_role_transport_validation_is_fresh_verification':False,'optimizer_executed_here':False,'payable':False,'chain_transactions':False}

def heldout_contract(epoch,authority,fixed_user,public_tasks):
    manifest=validate_epoch(epoch,authority,fixed_user)
    from .native_tau2_probe import REVISION as DATA_REVISION
    if public_tasks.get('schema')!='original-affine-tau2-telecom-public-task-commitments-v1' or public_tasks.get('data_revision')!=DATA_REVISION or public_tasks.get('data_inventory_sha256')!=manifest['environment']['data_inventory_sha256'] or public_tasks.get('split_policy')!='original-tau2-user-instruction-group-disjoint-v1' or digest(public_tasks)!=manifest['environment']['taskset_sha256']:raise ValueError('fixed original scenario-disjoint dataset')
    mining=public_tasks.get('mining_indices');heldout=public_tasks.get('heldout_indices')
    if canonical(mining)!=canonical(list(range(16))) or canonical(heldout)!=canonical(list(range(16,32))) or canonical(manifest.get('mining_indices'))!=canonical(mining) or canonical(manifest.get('heldout_indices'))!=canonical(heldout):raise ValueError('signed fixed original16/16 split')
    tasks=public_tasks.get('tasks',[])
    if not isinstance(tasks,list) or len(tasks)!=32 or any(type(r.get('index')) is not int for r in tasks) or len({r['index'] for r in tasks})!=32:raise ValueError('complete unique original32 inventory')
    by_index={r['index']:r for r in tasks}
    expected={r['index']:r for r in manifest['tasks']}
    for i in range(32):
        row=by_index.get(i);signed=expected.get(i)
        if row is None or signed is None or signed['task_hash']!=row['task_hash']:raise ValueError('signed original heldout task identity')
        sha(row['task_hash']);sha(row['scenario_group_sha256'])
    if {by_index[i]['scenario_group_sha256'] for i in mining}&{by_index[i]['scenario_group_sha256'] for i in heldout}:raise ValueError('heldout scenario overlap')
    agent=manifest['roles']['agent']
    geometry={k:agent[k] for k in ('model_runtime_revision','native_role_revision','runtime_profile','runtime_versions','interpreter_sha256','renderer','harness_source_sha256','source_files','max_context','max_output_tokens','vocab_size','generation_policy','candidate_policy')}
    geometry['config_tokenizer_files']={n:s for n,s in agent['checkpoint']['files'].items() if not n.endswith('.safetensors')}
    body={'version':'native-tau2-common-fixed-auxiliary-heldout16-v1','taskset_sha256':manifest['environment']['taskset_sha256'],'environment':manifest['environment'],'heldout_tasks':[expected[i] for i in heldout],'fixed_auxiliary_descriptor':fixed_user,'agent_geometry_and_policy':geometry,'native_execution_policy':manifest['native_execution_policy'],'checkpoint_excluded_from_dataset_id':True,'evaluation_performed_here':False,'original_native_grader_required':True,'all_model_roles_fresh_verified_required':True,'payable':False}
    return {**body,'dataset_id':digest(body)}
