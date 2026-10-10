"""Pure v4 training bookkeeping evidence, with no inference or chain writes.

This does not turn operator-collected reports into proofs of physical GPU
execution or convergence. It checks the signed job's task normalization, full
coverage, optimizer counters and precision diagnostics instead of legacy cyclic
single-pair attribution or an unconditional weights-changed requirement.
"""
import math

from .persistent_cpu_adamw import POLICY,HYPERPARAMETERS,sha
from .task_normalized_training import task_groups


def _finite(value):
    return type(value)in (int,float)and math.isfinite(value)


def validate_objective_evidence(report,job,*,tasks):
    """Check components only for the authenticated unit objective, without inference.

    ``validate_updates`` first reconstructs complete task/pair identities and
    exact gradient weights. Legacy and explicit-zero reports keep their shape.
    This is arithmetic bookkeeping, not independent proof of physical forwards.
    """
    from .unaudited_training_execution import configured_objective
    from .training_receipts import authenticate
    envelope=job.get('unaudited_training_execution')
    declaration=authenticate(envelope,job['manifest']['signer']) if envelope is not None else {}
    objective=configured_objective(declaration)
    if objective is not None and (not isinstance(objective,dict)or
            set(objective)!={'version','positive_nll_weight'}or
            objective['version']!='sequence-mean-preference-positive-nll-v1'or
            not _finite(objective['positive_nll_weight'])or objective['positive_nll_weight']not in (0,1)):
        raise ValueError('exact authenticated configured objective')
    diagnostics=report['training']['persistent_diagnostics'];updates=report['training']['updates']
    fields=('positive_mean_logprob','negative_mean_logprob','margin',
        'preference_loss','positive_nll','loss')
    legacy_claims={'training_objective','training_pair_positive_mean_logprob_before',
        'training_pair_negative_mean_logprob_before','training_pair_positive_mean_logprob_after',
        'training_pair_negative_mean_logprob_after'}
    locations=[diagnostics,*updates,*(row for update in updates for row in update['pairs'])]
    if any(legacy_claims.intersection(row)for row in locations):
        raise ValueError('unexpected alternate objective telemetry')
    if objective is None or objective['positive_nll_weight']==0:
        claims={'positive_nll_components','positive_nll_weight','beta',*fields}-{'margin','loss'}
        if any(claims.intersection(row)for row in locations):
            raise ValueError('zero objective keeps historical report shape')
        return objective

    def equal(a,b):
        return _finite(a)and _finite(b)and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6)

    def components(row,margin,reference):
        if not all(_finite(row.get(k))for k in fields if k!='margin')or not _finite(margin)or not _finite(reference):
            raise ValueError('finite complete unit objective components')
        p,n=row['positive_mean_logprob'],row['negative_mean_logprob']
        z=.1*(margin-reference);preference=max(-z,0)+math.log1p(math.exp(-abs(z)))
        if (p>0 or n>0 or not equal(p-n,margin)or not equal(-p,row['positive_nll'])or
                not equal(preference,row['preference_loss'])or
                not equal(preference-p,row['loss'])):
            raise ValueError('unit objective component/margin/loss arithmetic')

    value=diagnostics.get('positive_nll_components')
    expected={'version','positive_nll_weight','beta','learning_rate','reference_scope',
        'before','after','weighted_before','weighted_after','extra_model_forward_passes','tail_mask_applied'}
    if (not isinstance(value,dict)or set(value)!=expected or
            value['version']!='unit-positive-nll-shared-forward-components-v1'or
            not _finite(value['positive_nll_weight'])or value['positive_nll_weight']!=1 or
            not _finite(value['beta'])or value['beta']!=.1 or
            not _finite(value['learning_rate'])or value['learning_rate']!=5e-7 or
            value['reference_scope']!='immutable-BF16-epoch-input'or
            type(value['extra_model_forward_passes'])is not int or value['extra_model_forward_passes']!=0 or
            value['tail_mask_applied']is not False):
        raise ValueError('exact shared-forward unit objective diagnostics')
    references=diagnostics['training_pair_margin_before']
    for when in ('before','after'):
        rows=value[when];margins=diagnostics['training_pair_margin_'+when]
        if not isinstance(rows,list)or len(rows)!=len(references):
            raise ValueError('complete objective snapshot pair population')
        for i,row in enumerate(rows):
            if (not isinstance(row,dict)or set(row)!={'pair_index',*fields}or
                    type(row['pair_index'])is not int or row['pair_index']!=i or
                    not equal(row['margin'],margins[i])):
                raise ValueError('ordered objective snapshot matches original pair margins')
            components(row,row['margin'],references[i])
        summary=value['weighted_'+when]
        if not isinstance(summary,dict)or set(summary)!=set(fields):
            raise ValueError('exact weighted objective summary')
        totals={name:0. for name in fields};seen=[]
        for task in tasks:
            indices=task['pair_indices']
            if not indices:raise ValueError('nonempty objective task')
            for i in indices:
                if type(i)is not int or not 0<=i<len(rows):raise ValueError('objective task pair index')
                seen.append(i);weight=1/(len(tasks)*len(indices))
                for name in fields:totals[name]+=weight*rows[i][name]
        if sorted(seen)!=list(range(len(rows)))or any(not equal(summary[name],totals[name])for name in fields):
            raise ValueError('complete equal-task weighted objective summary')
    for update in updates:
        if not _finite(update.get('positive_nll_weight'))or update['positive_nll_weight']!=1:
            raise ValueError('update exact unit objective coefficient')
        totals={name:0. for name in fields if name!='margin'}
        for row in update['pairs']:
            if (not _finite(row.get('positive_nll_weight'))or row['positive_nll_weight']!=1 or
                    not _finite(row.get('beta'))or row['beta']!=.1):
                raise ValueError('pair exact unit objective coefficient/beta')
            components(row,row['margin_before'],row['reference_margin'])
            for name in totals:totals[name]+=row[name]*row['gradient_weight']
        if any(not equal(update.get(name),total)for name,total in totals.items()):
            raise ValueError('equal-task weighted update objective components')
    return objective


def validate_updates(report,job,manifest):
    training=report['training'];updates=training['updates'];diagnostics=training['persistent_diagnostics']
    binding=manifest['trainer_state_binding'];before=binding['global_step_before']
    if (not isinstance(updates,list)or len(updates)!=job['steps']or
            not isinstance(diagnostics,dict)or 'updates'in diagnostics or
            diagnostics.get('training_policy')!=POLICY or diagnostics.get('optimizer_steps')!=job['steps']or
            diagnostics.get('global_optimizer_step_before')!=before or
            diagnostics.get('global_optimizer_step_after')!=before+job['steps']or
            diagnostics.get('epoch')!=manifest['epoch']or
            diagnostics.get('input_checkpoint')!=manifest['checkpoint']['id']or
            diagnostics.get('heldout_gain_claimed')is not False or
            diagnostics.get('state_publication_required')is not True):
        raise ValueError('persistent training diagnostics exact policy/epoch/update counters')
    definitions={row['env_id']:row for row in manifest['environments']};pairs=[]
    from .training_receipts import VERSION as RECEIPT_POLICY
    inputs=report['training_admissions'] if job.get('training_input_policy') in (RECEIPT_POLICY,'authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1') else report['audits']
    for audit in inputs:
        batches=[audit['claimed_batch']]if job.get('training_input_policy')=='committed-unaudited-training-v1'else audit['accepted']
        for batch in batches:
            definition=definitions[batch['env_id']]
            positives=[r for r in batch['rollouts']if r['classification']=='positive']
            negatives=[r for r in batch['rollouts']if r['classification']=='negative']
            if type(manifest.get('K'))is int and manifest['K']>=2:
                if (manifest.get('L')!=manifest['K'] or len(positives)!=manifest['K'] or
                    len(negatives)!=manifest['L'] or len(batch['rollouts'])!=manifest['K']+manifest['L']):
                    raise ValueError('complete manifest task rollout quotas')
            pairs.extend((definition,p,n)for p,n in zip(positives,negatives))
    pairs,tasks,groups,identities=task_groups(pairs,job['steps'],manifest['training_coverage']['seed'],
        **({'required_pairs_per_task':manifest['K']} if type(manifest.get('K'))is int and manifest.get('K')==manifest.get('L') and manifest['K']>=2 else {}))
    if diagnostics.get('task_count')!=len(tasks)or diagnostics.get('pair_count')!=len(pairs):
        raise ValueError('persistent diagnostics distinct task/pair population')
    references=diagnostics.get('training_pair_margin_before');after=diagnostics.get('training_pair_margin_after')
    delta=diagnostics.get('training_pair_margin_delta')
    if (not isinstance(references,list)or not isinstance(after,list)or not isinstance(delta,list)or
            len(references)!=len(pairs)or len(after)!=len(pairs)or len(delta)!=len(pairs)or
            not all(_finite(v)for v in references+after+delta)or
            any(d!=a-b for b,a,d in zip(references,after,delta))):
        raise ValueError('persistent training pair margin diagnostics')
    seen=set();inventory=binding['parameters'];elements=sum(r['numel']for r in inventory)
    master_changed=False;bf16_changed=False
    for step,(update,group)in enumerate(zip(updates,groups)):
        global_step=before+step+1;seen.update(group)
        if (not isinstance(update,dict)or update.get('training_policy')!=POLICY or update.get('steps')!=1 or
                type(update.get('epoch_optimizer_step'))is not int or update['epoch_optimizer_step']!=step+1 or
                type(update.get('global_optimizer_step'))is not int or update['global_optimizer_step']!=global_step or
                update.get('input_checkpoint')!=binding['input_checkpoint']or
                update.get('reference_scope')!='immutable-BF16-epoch-input'or
                update.get('optimizer_lifecycle')!='persistent-across-epochs'or
                update.get('task_weight_rule')!='mean-pair-within-task-then-mean-task-within-group'or
                sha(update.get('hyperparameters'))!=sha(HYPERPARAMETERS)or
                update.get('full_model_finetune')is not True or update.get('gradient_tensors')!=len(inventory)or
                update.get('unique_tasks')!=len(tasks)or update.get('unique_verified_pairs')!=len(pairs)or
                update.get('gradient_tasks')!=len(group)or update.get('cumulative_unique_gradient_tasks')!=len(seen)or
                not _finite(update.get('loss'))or update['loss']<0 or
                not _finite(update.get('gradient_norm_before_clip'))or update['gradient_norm_before_clip']<0):
            raise ValueError('persistent optimizer update attribution/counter/objective')
        observations=update.get('pairs');expected=[]
        for task_index in group:
            task=tasks[task_index]
            expected.extend(dict(pair_index=i,task_index=task_index,task_sha256=task['task_sha256'],
                pair_sha256=identities[i],gradient_weight=1/(len(group)*len(task['pair_indices'])),
                reference_margin=references[i])for i in task['pair_indices'])
        if not isinstance(observations,list)or len(observations)!=len(expected)or update.get('gradient_pairs')!=len(expected):
            raise ValueError('persistent optimizer complete task-normalized gradient population')
        for actual,wanted in zip(observations,expected):
            if (not isinstance(actual,dict)or any(sha(actual.get(k))!=sha(v)for k,v in wanted.items())or
                    not _finite(actual.get('margin_before'))or not _finite(actual.get('loss'))or actual['loss']<0):
                raise ValueError('persistent optimizer task/pair weight/reference attribution')
        precision=update.get('precision',{})
        if (precision.get('optimizer_step')!=global_step or precision.get('master_dtype')!='torch.float32'or
                precision.get('optimizer_state_dtype')!='torch.float32'or precision.get('inference_dtype')!='torch.bfloat16'):
            raise ValueError('persistent optimizer precision/counter evidence')
        tensors=precision.get('parameters')
        if not isinstance(tensors,list)or len(tensors)!=len(inventory):raise ValueError('persistent precision full parameter inventory')
        for row,approved in zip(tensors,inventory):
            if row.get('name')!=approved['name']or row.get('elements')!=approved['numel']:
                raise ValueError('persistent precision parameter name/shape population')
            for key in ('master_changed_elements','bf16_changed_elements'):
                if type(row.get(key))is not int or not 0<=row[key]<=approved['numel']:
                    raise ValueError('persistent precision changed-element count')
            if any(not _finite(row.get(k))or row[k]<0 for k in ('master_delta_l2','master_delta_max_abs')):
                raise ValueError('persistent finite nonnegative master deltas')
        for key in ('master_changed_elements','bf16_changed_elements'):
            if type(precision.get(key))is not int or precision[key]!=sum(r[key]for r in tensors)or not 0<=precision[key]<=elements:
                raise ValueError('persistent precision aggregate changed-element count')
        master_changed|=precision['master_changed_elements']>0;bf16_changed|=precision['bf16_changed_elements']>0
    if (seen!=set(range(len(tasks)))or diagnostics.get('master_state_updated')is not master_changed or
            diagnostics.get('inference_tensors_changed_during_updates')is not bf16_changed):
        raise ValueError('persistent complete task coverage/precision summary')
    validate_objective_evidence(report,job,tasks=tasks)
    return dict(update_count=len(updates),distinct_tasks=len(tasks),distinct_pairs=len(pairs),
        global_step_before=before,global_step_after=before+len(updates),
        master_state_updated=master_changed,heldout_gain_claimed=False,
        historical_gpu_execution_proven=False)
