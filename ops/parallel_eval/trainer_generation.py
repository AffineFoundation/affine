"""Admitted isolated trainer generation selection/capture, never a live hot patch."""
from pathlib import Path
from .durable import Stage, atomic, digest, read
from .masters import MasterStore, metadata, install_generation_hooks
from .publication import checkpoint


def configure(trainer, root, request, certificate, root_authority, model_output):
    from subnet.backend_jobs import signed
    cert = signed(certificate, root_authority)
    if cert.get('version') != 'pe-isolated-trainer-admission-v1' or cert.get('payable') is not False or cert.get('stage_root') != str(Path(root)):
        raise ValueError('isolated trainer admission')
    body = signed(request, cert['replay_authority'])
    if body.get('version') != 'pe-native-train-request-v1' or body.get('payable') is not False or body.get('training_output_tokens') != 32768:
        raise ValueError('signed 32K replay request')
    if body['original_job_sha256'] != cert['original_job_sha256'] or body['node_id'] not in cert['allowed_node_ids']:
        raise ValueError('admitted replay job/node binding')
    stage, row = Stage(root), body['input']
    if Path(trainer.FP32_STORE).resolve() != stage.root / 'fp32-master':
        raise ValueError('admitted generation store location')
    if row['id'] != digest(row['head']) or row['head'] != dict(version='pe-head-v1', checkpoint=row['checkpoint']['id'], master_digest=row['master_digest'], generation_id=row['generation_id']):
        raise ValueError('generation head binding')
    if checkpoint(row['model']) != row['checkpoint'] or not Path(row['model']).resolve().is_relative_to(stage.root):
        raise ValueError('generation input model inventory')
    output_generation = body['output_generation_id']
    store = MasterStore(stage)
    if not Path(model_output).resolve().is_relative_to(stage.root):
        raise ValueError('isolated generation output')
    def select(opt, live):
        return row
    install_generation_hooks(trainer, store, select)
    original_save, original_finish = trainer._fp32_save, trainer._fp32_finish
    lineage = dict(request_sha256=digest(request), input_head=row['id'], original_job_sha256=body['original_job_sha256'])
    def save(opt, bf16_entry, native_meta, protect, emit):
        entry = native_meta['master_digest'] + '-' + output_generation
        store.entry_id(entry)
        native_meta = dict(native_meta, generation_id=output_generation, lineage=lineage)
        original_save(opt, entry, native_meta, set(protect) | {entry, row['master_entry']}, emit)
        target = store.root / entry
        canonical = metadata(model_output, target, optimizer_step=native_meta['optimizer_step'], lineage=lineage, generation_id=output_generation)
        if canonical['master_digest'] != native_meta['master_digest'] or canonical['bf16_sha256'] != bf16_entry:
            raise ValueError('native generation capture digest/projection')
        atomic(target / 'meta.json', canonical)
        store.pin(entry, 'training:' + body['command_id'])
        store.verify(entry, model_output)
    def finish(opt, info, emit):
        out = original_finish(opt, info, emit)
        out['output_store_entry'] = out['output_master_digest'] + '-' + output_generation
        out['output_generation_id'] = output_generation
        return out
    trainer._fp32_save, trainer._fp32_finish = save, finish
    return dict(input_head=row['id'], output_generation_id=output_generation, request_sha256=digest(request), input_master_entry=row['master_entry'])
