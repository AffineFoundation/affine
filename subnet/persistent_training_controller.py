"""Prospective v4 coordinator integration; existing optimizer paths are unchanged."""
import json
from pathlib import Path

from .persistent_cpu_adamw import POLICY, sha
from .persistent_training_protocol import (validate_binding, validate_report,
    independently_commit, validate_pointer, validate_parent,read_json)
from .storage import canonical


def original_request(controller,epoch):
    from .backend_jobs import signed
    from .training_startup_recovery import label
    record=json.loads((controller.state/'roles'/(label(controller,epoch)+'.json')).read_text())
    job=signed(json.loads((controller.state/'roles'/(record['job_id']+'-job.json')).read_text()),controller.authority.id)
    if sha(job)!=record['job_sha256']:raise ValueError('original persistent training request record hash')
    return record,job


def commit_latest(controller,binding,pointer):
    """Monotonic journal permits recovery of the SAME original completed job."""
    from .remote_backend import save
    validate_pointer(pointer)
    path=controller.state/'latest-trainer-state.json'
    previous=json.loads(path.read_text())if path.exists()else None
    if previous==pointer:return
    if previous!=binding['parent']:raise ValueError('latest trainer state changed; stale parent replay forbidden')
    if pointer['optimizer_steps']<=binding['global_step_before']:raise ValueError('persistent state must advance optimizer counter')
    save(path,pointer)


def train(controller,manifest,reports,checkpoint_path,*,steps,replay=None):
    from .backend_jobs import signed,file_map
    from .remote_backend import save,training_submission_bytes
    from .training_policy import coverage_manifest
    from .forced_sampling import require_report
    from .training_receipts import prepare_submissions,amend_manifest,receipt_inventory,require_execution_amendment
    require_execution_amendment(controller,manifest)
    epoch=manifest['epoch'];binding=validate_binding(manifest.get('trainer_state_binding'),manifest)
    if replay is not None:raise ValueError('persistent historical replay needs separate admission')
    training_manifest=manifest
    if manifest.get('audit_policy',{}).get('version')=='bounded-random-v1':
        training_manifest=json.loads((controller.state/(epoch+'-audit-manifest.json')).read_text())
    if (training_manifest.get('trainer_state_binding')!=binding or
            training_manifest.get('training_policy')!=POLICY or training_manifest.get('checkpoint')!=manifest['checkpoint']):
        raise ValueError('original persistent audit manifest binding')
    receipts=json.loads((controller.state/(epoch+'-scores.json')).read_text())['receipts']
    challenge=json.loads((controller.state/(epoch+'-audit-challenge.json')).read_text())
    training_manifest=coverage_manifest(training_manifest,receipts,challenge)
    from .training_startup_recovery import declaration,apply,label
    recovery=declaration(controller,epoch)
    if recovery is not None:
        from .compact_training_inputs import receipt_inventory
        training_manifest,submissions=apply(controller,training_manifest,steps)
    elif (training_manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
        from .compact_training_inputs import prepare_submissions,receipt_inventory
        submissions=prepare_submissions(controller,training_manifest,reports,receipts)
    else:
        submissions=prepare_submissions(controller,training_manifest,reports,receipts)
        training_manifest=amend_manifest(controller,training_manifest,submissions,steps)
    cached=controller.state/(epoch+'-training-metrics.json')
    if cached.exists():
        metrics=json.loads(cached.read_text());record,job=original_request(controller,epoch)
        original_manifest=signed(job['manifest'],controller.authority.id)
        report=json.loads((controller.state/'roles'/(record['job_id']+'-report.json')).read_text())
        if (original_manifest!=training_manifest or job['steps']!=steps or
                receipt_inventory(job['submissions'])!=receipt_inventory(submissions)or
                metrics.get('verifier_receipt_inventory')!=receipt_inventory(submissions)):
            raise ValueError('cached persistent original request changed')
        validate_report(report,job,training_manifest)
        pointer=validate_pointer(metrics.get('trainer_state'))
        publication=read_json(controller.bucket,pointer['descriptor_key'])
        # Authenticate the current output as a parent commitment for recovery,
        # without fabricating a next epoch or a new train request.
        output_binding=dict(binding,input_checkpoint=metrics['new_checkpoint']['id'],
            genesis=None,parent=pointer,global_step_before=pointer['optimizer_steps'])
        validate_parent(publication,output_binding,controller.authority.id)
        committed=signed(publication,controller.authority.id)
        if (pointer['namespace']!=job['persistent_training']['output_namespace']or
                committed['job_id']!=job['job_id']or committed['job_sha256']!=sha(job)):
            raise ValueError('cached persistent publication original signed job namespace/hash/ID')
        if (metrics.get('source_epoch')!=epoch or metrics.get('training_policy')!=POLICY or
                metrics.get('input_checkpoint')!=binding['input_checkpoint']or metrics.get('steps')!=steps or
                metrics.get('original_job_sha256')!=sha(job)or metrics.get('trainer_binding_sha256')!=sha(binding)or
                metrics.get('checkpoint')!=file_map(metrics['new_checkpoint']['files'])or
                metrics.get('weights_changed')!=report['training']['weights_changed']or
                canonical(metrics.get('updates'))!=canonical(report['training']['updates'])or
                canonical(metrics.get('persistent_diagnostics'))!=canonical(report['training']['persistent_diagnostics'])or
                pointer['descriptor_sha256']!=report['persistent_training_state']['descriptor_sha256']or
                pointer['optimizer_steps']!=binding['global_step_before']+steps):
            raise ValueError('cached persistent checkpoint/state original lineage binding')
        receipt=json.loads((controller.state/(epoch+'-checkpoint-publication.json')).read_text())
        if (receipt.get('checkpoint')!=metrics['checkpoint']or receipt.get('operator_independent_hashes')is not True or
                {n:r['sha256']for n,r in receipt.get('objects',{}).items()}!=metrics['new_checkpoint']['files']):
            raise ValueError('cached persistent inference publication binding')
        commit_latest(controller,binding,pointer)
        output=controller.checkpoint_with_reads(metrics['new_checkpoint'])
        metrics=dict(metrics,new_checkpoint=output)
        controller.bucket.json('public/'+epoch+'/training.json',controller.signed(metrics))
        return output,metrics
    capacity=(controller.jobs.training_resume(label(controller,epoch),training_manifest,submissions,steps,None)
              if hasattr(controller.jobs,'training_resume')else None)
    if capacity is None:
        total=(sum(obj['size'] for obj in submissions) if (training_manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2')
               else training_submission_bytes(receipts,reports,manifest))
        probe=getattr(controller.jobs,'persistent_training_capacity',None)or getattr(controller.jobs,'training_capacity',None)
        if probe is None:raise ValueError('persistent training requires actual trainer resource probe')
        capacity=probe(training_manifest,steps,submission_bytes=total)
    remote=controller.jobs.run(label(controller,epoch),'train',training_manifest,checkpoint_path,
        submissions=submissions,steps=steps,training_policy=POLICY)
    record,job=original_request(controller,epoch)
    if signed(job['manifest'],controller.authority.id)!=training_manifest or job['steps']!=steps:
        raise ValueError('persistent original request exact manifest/steps')
    validate_report(remote,job,training_manifest)
    new=dict(remote['new_checkpoint']);path=new.pop('path')
    # BF16 values may still be identical while masters/moments advance. Publish
    # the actual immutable file map and keep weights_changed honest.
    from .persistent_publication import complete
    output,pointer,publication_timings=complete(controller,remote,job,training_manifest,path)
    commit_latest(controller,binding,pointer)
    metrics=dict(steps=steps,weights_changed=remote['training']['weights_changed'],state_updated=True,
        full_model_finetune=True,training_policy=POLICY,updates=remote['training']['updates'],
        persistent_diagnostics=remote['training']['persistent_diagnostics'],
        source_epoch=epoch,input_checkpoint=manifest['checkpoint']['id'],checkpoint=new['id'],
        new_checkpoint=output,checkpoint_path=path,trainer_state=pointer,capacity_preflight=capacity,
        remote_job_id=remote['job_id'],original_job_sha256=sha(job),trainer_binding_sha256=sha(binding),
        training_coverage=training_manifest['training_coverage'],training_input_policy=remote['training']['training_input_policy'],
        trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True,
        verifier_receipt_inventory=receipt_inventory(submissions),
        state_authority_committed=True,heldout_gain_claimed=False)
    if publication_timings is not None:metrics['publication_timings']=publication_timings
    save(cached,metrics);controller.bucket.json('public/'+epoch+'/training.json',controller.signed(metrics))
    return output,metrics
