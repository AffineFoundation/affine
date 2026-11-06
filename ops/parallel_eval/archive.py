"""Read-only verification of original signed past-epoch artifacts, not re-training."""
from pathlib import Path
from .durable import digest, file_sha, read


def audit_epoch(epoch_root, gate_epoch_root, authority, cached_model=None):
    from subnet.backend_jobs import signed
    root, gate = Path(epoch_root), Path(gate_epoch_root)
    job_envelope = read(root / 'TRAINING-JOB.json')
    job = signed(job_envelope, authority)
    manifest = signed(job['manifest'], authority)
    report_envelope = read(root / 'TRAINING-RESULT.json')
    request = signed(report_envelope, job['trainer_worker'])
    report = request['report']
    publication = signed(report['publication_receipt'], job['training']['publisher_id'])
    cp = publication['descriptor']
    decision = read(gate / 'DECISION.json')
    training = report['training']
    original = {k:v for k,v in training.items() if k not in ('output_checkpoint','training_requires_sampling_audit','model_only_publication')}
    publication_job = 'mimo-' + digest(dict(input=training['input'], documents=training['input_documents_sha256'], config=training['config']))
    checks = dict(operator_job_signature=True, opening_signature=True,
                  original_worker_report_signature=True, original_publisher_signature=True,
                  job_id_binding=request['job_id'] == job['job_id'] == report['job_id'],
                  publication_job_binding=publication['job_id'] == publication_job,
                  original_training_report_digest=publication['report_sha256'] == report['transaction_report_sha256'] == digest(original),
                  source_closure_binding=publication['source_closure_sha256'] == digest(job['source_files']),
                  publication_epoch_binding=publication['epoch'] == manifest['epoch'] == report['epoch'],
                  report_job_digest=report['job_sha256'] == digest(job),
                  original_success=report['success'] is True and report['training']['status'] == 'updated',
                  candidate_binding=cp['id'] == publication['output_checkpoint'] == report['output_checkpoint'] == decision['candidate'],
                  training_head_binding=publication['input_checkpoint'] == report['training']['input_checkpoint'] == manifest['checkpoint']['id'],
                  optimizer_step_binding=publication['optimizer_step'] == report['training']['optimizer_step'],
                  original_publication_model_only=publication['optimizer_state_uploaded'] is False,
                  checkpoint_file_map=digest(cp['files']) == cp['id'])
    if cached_model is not None:
        directory = Path(cached_model) / cp['id']
        checks['full_cached_model_readback'] = all(
            Path(name).name == name and not (directory / name).is_symlink()
            and file_sha(directory / name) == expected for name, expected in cp['files'].items())
    if not all(checks.values()):
        raise ValueError('archive binding failed: ' + ','.join(k for k,v in checks.items() if not v))
    return dict(scope='read-only-original-signed-archive', checks=checks,
                model_files=len(cp['files']), epoch=int(root.name),
                gate_decision=decision['decision'], live_z=decision['evaluated']['z'],
                reran_training=False, reran_gpu_probe=False, production_admission_replayed=False,
                production_end_to_end_passed=False)
