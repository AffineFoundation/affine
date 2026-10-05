"""Operator-only asynchronous reducers of original checked role evidence.

No inference, dispatch, signing, audit replay or publication is performed here.
"""
import copy, math, re
from .distributed_roles import authenticate
from .persistent_cpu_adamw import sha
from .storage import canonical
from .training_quality_monitor import evaluation_indices


def original(jobs, prior, report, job_document, authority):
    job=authenticate(job_document,authority)
    manifest=authenticate(job['manifest'],authority)
    if sha(job)!=prior['job_sha256'] or job['job_id']!=prior['job_id']:
        raise ValueError('exact original signed role job')
    jobs.checked(report,prior,manifest)
    return job,manifest


def training_input(jobs, prior, report, job_document, publication_document, authority):
    job,manifest=original(jobs,prior,report,job_document,authority)
    from .persistent_cpu_adamw import POLICY
    if job['role']!='train' or job.get('training_policy')!=POLICY:raise ValueError('original persistent trainer role')
    if publication_document is None:
        return dict(status='pending_durable_adoption',epoch=manifest['epoch'],hold=False,convergence_claimed=False)
    from .persistent_training_protocol import PUBLICATION_VERSION
    publication=authenticate(publication_document,authority)
    state=report['persistent_training_state'];descriptor=state['descriptor'];training=report['training']
    if (publication.get('version')!=PUBLICATION_VERSION or publication.get('job_id')!=job['job_id'] or
            publication.get('job_sha256')!=sha(job) or publication.get('namespace')!=state['namespace'] or
            publication.get('descriptor_sha256')!=sha(descriptor) or publication.get('descriptor')!=descriptor):
        raise ValueError('actual original durable trainer state adoption')
    diagnostics=copy.deepcopy(training['persistent_diagnostics']);updates=copy.deepcopy(training['updates'])
    margins=diagnostics['training_pair_margin_delta']
    if not margins or any(type(v)not in (int,float)or not math.isfinite(v)for v in margins):
        raise ValueError('finite original pair margins')
    for row in updates:
        if any(type(row.get(k))not in (int,float)or not math.isfinite(row[k])for k in ('loss','gradient_norm_before_clip')):
            raise ValueError('finite original gradient/loss')
    if (descriptor['inference_checkpoint']!=report['new_checkpoint']['id'] or
            descriptor['optimizer_steps']!=training['global_step_after'] or
            descriptor['parent_state_sha256']!=(manifest['trainer_state_binding']['parent']['descriptor_sha256'] if manifest['trainer_state_binding']['parent'] is not None else None)):
        raise ValueError('exact parent/output checkpoint optimizer lineage')
    return dict(version='authenticated-training-quality-input-v1',status='complete',epoch=manifest['epoch'],
        input_checkpoint=manifest['checkpoint']['id'],output_checkpoint=descriptor['inference_checkpoint'],
        optimizer_step_before=training['global_step_before'],optimizer_step_after=training['global_step_after'],
        parent_state_sha256=descriptor['parent_state_sha256'],output_state_sha256=sha(descriptor),
        inference_weights_changed=training['weights_changed'],diagnostics=dict(diagnostics,updates=updates),
        provenance=dict(original_job_sha256=sha(job),original_report_sha256=sha(report),
            adoption_document_sha256=sha(publication_document),execution_runtime_revision=report['execution_runtime_revision'],
            generation_runtime_revision=report['generation_runtime_revision'],source_files_sha256=sha(report['source_files']),
            backend_profile_sha256=sha(report['backend_profile']),runtime_versions_sha256=sha(report['runtime_versions'])),
        convergence_claimed=False)


def evaluation_input(jobs, prior, report, job_document, export_document, authority):
    job,manifest=original(jobs,prior,report,job_document,authority)
    if job['role']!='evaluate':raise ValueError('original evaluator role')
    if export_document is None:return dict(status='pending_evaluation_export',hold=False,convergence_claimed=False)
    export=authenticate(export_document,authority)
    if export.get('checkpoint')!=manifest['checkpoint']['id'] or export.get('epoch')!=manifest['epoch']:
        raise ValueError('signed evaluation export checkpoint branch')
    records=copy.deepcopy(export['records'])
    if len(records)!=len(job['heldout']):raise ValueError('original heldout suite inventory')
    for record,suite in zip(records,job['heldout']):
        values=[v for v in report['heldout']if v['env_id']==suite['env_id']]
        failures=[v for v in report.get('heldout_failures',[])if v['env_id']==suite['env_id']]
        expected=list(zip(suite['indices'],suite['seeds']))
        if sorted((v['index'],v['seed'])for v in values+failures)!=sorted(expected):raise ValueError('original exact evaluation tasks/seeds')
        if any(not isinstance(v.get('task_hash'),str)or re.fullmatch('[0-9a-f]{64}',v['task_hash'])is None or v.get('verified')is not True or v.get('classification')not in ('positive','negative')or
               v.get('reward')!=(1 if v['classification']=='positive'else 0) for v in values):
            raise ValueError('original verified evaluation outcomes')
        definition=next(d for d in manifest['environments']if d['env_id']==suite['env_id'])
        from .backend_profiles import resolve
        revision,profile,_=resolve(manifest)
        frozen=dict(env_id=suite['env_id'],environment=definition['spec'],harness=suite['harness'],
            indices=suite['indices'],seeds=suite['seeds'],model_runtime_revision=revision,backend_profile=profile,
            runtime_versions=report['runtime_versions'],harness_source_hash=manifest['harness_source_hash'],
            source_files={n:report['source_files'][n]for n in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')})
        expected_fields=dict(env_id=suite['env_id'],checkpoint=manifest['checkpoint']['id'],remote_job_id=job['job_id'],
            timestamp=report['completed_at'],heldout_indices=suite['indices'],harness_config=suite['harness'],
            model_runtime_revision=revision,backend_profile=profile,dataset_id=sha(frozen),taskset_hash=sha(frozen),
            requested_count=len(expected),completed_count=len(values),successes=sum(v['classification']=='positive'for v in values),
            fixed_task_ids=[v['task_hash']for v in values],task_hashes=[v['task_hash']for v in values],evaluation_failures=failures)
        if any(canonical(record.get(k))!=canonical(v)for k,v in expected_fields.items()):
            raise ValueError('signed evaluation export exact original report/runtime/task hashes')
        record['public_optimizer_steps']=export['public_optimizer_steps']
    return dict(export,version='authenticated-evaluation-quality-input-v1',records=records,provenance=dict(original_job_sha256=sha(job),original_report_sha256=sha(report),
        original_export_sha256=sha(export_document)),convergence_claimed=False)


def evaluation_schedule(reserved,mining,round_number,*,budget=128,full_every=24,chunk_size=64):
    if type(chunk_size)is not int or not 1<=chunk_size<=64:raise ValueError('existing bounded evaluator chunk')
    indices=evaluation_indices(reserved,mining,round_number,budget=budget,full_every=full_every)
    return dict(version='asynchronous-paired-heldout-schedule-v1',round=round_number,
        indices=indices,chunks=[indices[i:i+chunk_size]for i in range(0,len(indices),chunk_size)],
        before_after_same_indices_required=True,training_barrier=False,dispatch_performed=False)


def comparison_views(before_document,after_document,training_document,authority):
    """Relate original checkpoint evaluations to a training transition.

    Comparison labels never change original job epochs, phases or timestamps;
    provenance keeps those intact. Caller may ROOT-sign these derived views.
    """
    if any(v is None for v in (before_document,after_document,training_document)):
        return dict(status='pending_paired_evidence',hold=False,convergence_claimed=False)
    before=authenticate(before_document,authority);after=authenticate(after_document,authority)
    training=authenticate(training_document,authority)
    if training.get('version')!='authenticated-training-quality-input-v1' or training.get('status')!='complete':
        return dict(status='pending_durable_adoption',hold=False,convergence_claimed=False)
    if before.get('version')!='authenticated-evaluation-quality-input-v1' or after.get('version')!='authenticated-evaluation-quality-input-v1':
        raise ValueError('operator admitted evaluation quality views')
    if before['checkpoint']!=training['input_checkpoint'] or after['checkpoint']!=training['output_checkpoint']:
        raise ValueError('exact compared training checkpoint branch')
    views=[]
    for phase,original,document in [('before',before,before_document),('after',after,after_document)]:
        view=copy.deepcopy(original)
        view.update(epoch=training['epoch'],phase=phase,
            comparison_provenance=dict(original_document_sha256=sha(document),original_epoch=original['epoch'],
                original_phase=original['phase'],original_completed_at=original.get('completed_at')))
        views.append(view)
    return dict(status='ready',before=views[0],after=views[1],training=training,
        training_barrier=False,convergence_claimed=False)


def aggregate_evaluations(documents,authority,*,checkpoint,optimizer_steps,expected_indices):
    """Aggregate existing checked/signed chunk views into one fixed cohort.

    128 tasks split into two 64-task jobs count as 128 paired tasks, never two
    separately confirmed regressions. All chunk documents must be admitted by
    evaluation_input and ROOT-signed before entering this reducer.
    """
    if not documents:return dict(status='pending_paired_evidence',hold=False)
    if len(set(expected_indices))!=len(expected_indices):raise ValueError('distinct reserved cohort')
    rows=[];hashes={};successes=0;datasets=[];first=None;originals=[]
    for document in documents:
        payload=authenticate(document,authority)
        if payload.get('version')!='authenticated-evaluation-quality-input-v1' or payload.get('checkpoint')!=checkpoint or payload.get('public_optimizer_steps')!=optimizer_steps or 'provenance'not in payload:
            raise ValueError('admitted checkpoint chunk lineage')
        if payload.get('status')!='complete':return dict(status='unknown_infrastructure',hold=False)
        for record in payload['records']:
            if record.get('evaluation_failures')or record['completed_count']!=record['requested_count']:
                return dict(status='unknown_infrastructure',hold=False)
            common={k:record.get(k)for k in ('env_id','harness_config','model_runtime_revision','backend_profile','runtime_profile')}
            if first is None:first=common
            elif common!=first:raise ValueError('same aggregated environment/harness/runtime')
            if len(record['task_hashes'])!=len(record['heldout_indices']):raise ValueError('complete actual task hashes')
            for index,task in zip(record['heldout_indices'],record['task_hashes']):
                if index in hashes or index not in expected_indices:raise ValueError('chunk overlap or foreign cohort')
                hashes[index]=task
            datasets.append(dict(indices=record['heldout_indices'],dataset_id=record['dataset_id']))
            successes+=record['successes'];rows.append(record)
        originals.append(sha(document))
    if set(hashes)!=set(expected_indices):return dict(status='pending_paired_evidence',hold=False)
    record=dict(rows[0],**first,heldout_indices=list(expected_indices),
        fixed_task_ids=[hashes[i]for i in expected_indices],task_hashes=[hashes[i]for i in expected_indices],
        requested_count=len(expected_indices),completed_count=len(expected_indices),successes=successes,
        dataset_id=sha(sorted(datasets,key=lambda r:r['indices'])),taskset_hash=sha(sorted(datasets,key=lambda r:r['indices'])),evaluation_failures=[])
    return dict(version='authenticated-evaluation-quality-input-v1',status='complete',checkpoint=checkpoint,public_optimizer_steps=optimizer_steps,
        epoch=payload['epoch'],phase=payload['phase'],records=[record],
        provenance=dict(original_chunk_document_sha256=originals),convergence_claimed=False)
