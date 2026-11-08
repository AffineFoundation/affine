"""Reuse the complete-group update journal for the MiMo training API."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .config import TrainerConfig
from .data import digest, sha
from .training import train_update
from .lifecycle import stage_for_publication
from .transaction import Job, Transaction, RecoveryRequired


def training_job(groups, binding, config, *, manifest_sha256, source_closure_sha256):
    """Derive original transaction identity solely from authenticated inputs."""
    documents = tuple(digest(asdict(group)) for group in groups)
    return Job(job_id='mimo-' + digest(dict(input=asdict(binding), documents=documents, config=config.document())),
        epoch=binding.epoch, manifest_sha256=manifest_sha256, source_closure_sha256=source_closure_sha256,
        objective_sha256=digest(config.document()), update_pin_sha256=digest(asdict(binding)),
        input_checkpoint=binding.checkpoint, prepared_documents_sha256=documents,
        optimizer_step_before=binding.optimizer_step)


def retire_obsolete_optimizer_states(history_root, *, run, keep_checkpoint, retire_models=False):
    """Normal completion retention: one committed generation + one in-flight.

    Only published journals in the explicitly owned run directory authorize
    deletion. Never remove an in-flight/updating generation or the current state.
    Journals/reports and bucket model history are retained.
    """
    root = Path(history_root).resolve()
    if not sha(keep_checkpoint):
        raise ValueError('immutable current checkpoint for retention')
    removed = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        path = child / 'TRANSACTION.json'
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 16 * 1024**2:
            continue
        value = json.loads(path.read_text())
        if value.get('state') != 'published' or value.get('report', {}).get('input', {}).get('run') != run:
            continue
        if value.get('output_checkpoint') == keep_checkpoint:
            continue
        # Validate original journal and publication receipt before using it as
        # a retention authority. Published reads do not require retired bytes.
        job = dict(value['job'])
        job['prepared_documents_sha256'] = tuple(job['prepared_documents_sha256'])
        original = Transaction(child, Job(**job)).run(
            update=lambda: (_ for _ in ()).throw(AssertionError('retention cannot train')),
            stage=lambda *args: (_ for _ in ()).throw(AssertionError('retention cannot stage')),
            publish_model=lambda *args: (_ for _ in ()).throw(AssertionError('retention cannot publish')))
        if original['output_checkpoint'] != value['output_checkpoint']:
            raise ValueError('original retention journal changed')
        optimizer = child / 'optimizer'
        if optimizer.is_symlink():
            raise ValueError('optimizer retention refuses symlink')
        if optimizer.is_dir():
            shutil.rmtree(optimizer)
            removed.append(str(optimizer))
        if retire_models:
            model = child / 'model'
            if model.is_symlink():
                raise ValueError('model retention refuses symlink')
            if model.is_dir():
                shutil.rmtree(model)
                removed.append(str(model))
    return removed


def execute_transaction(workspace, runtime, optimizer, groups, binding, *,
                        manifest_sha256, source_closure_sha256, assert_trainer_lease,
                        publish_model, config=TrainerConfig(), record=lambda row: None,
                        microbatch_rollouts=1, shard_bytes=256 * 1024**2,
                        commit_current=None, retire_published_models=False):
    """One original update or publication recovery, never automatic retraining.

    publish_model(request) must perform immutable/idempotent R2 publication and
    full content verification. It receives only model paths. The callback returns
    the bounded original receipt required by Transaction, including
    optimizer_state_uploaded=False. The caller authenticates receipt authority.
    """
    if not callable(assert_trainer_lease) or not callable(publish_model):
        raise ValueError('host-owned sole trainer lease and publisher required')
    if retire_published_models and not callable(commit_current):
        raise ValueError('model retirement requires an authenticated durable CURRENT callback')
    assert_trainer_lease()
    groups = tuple(groups)
    if not groups:
        report = train_update(runtime, optimizer, groups, binding, config=config,
            guard=assert_trainer_lease, record=record, microbatch_rollouts=microbatch_rollouts)
        return dict(state='no-update', output_checkpoint=binding.checkpoint,
                    report=report, publication=None)
    job = training_job(groups, binding, config, manifest_sha256=manifest_sha256,
                       source_closure_sha256=source_closure_sha256)
    documents = job.prepared_documents_sha256
    transaction = Transaction(workspace, job)
    original = transaction._read()
    if original is None or original['state'] == 'prepared':
        retire_obsolete_optimizer_states(transaction.root.parent, run=binding.run,
            keep_checkpoint=binding.checkpoint, retire_models=retire_published_models)

    def update():
        assert_trainer_lease()
        report = train_update(runtime, optimizer, groups, binding, config=config,
            guard=assert_trainer_lease, record=record, microbatch_rollouts=microbatch_rollouts)
        return dict(report, input_documents_sha256=list(documents),
                    runtime_context=dict(manifest_sha256=manifest_sha256,
                                         source_closure_sha256=source_closure_sha256))

    def stage(report, directory):
        assert_trainer_lease()
        staged = stage_for_publication(runtime, optimizer, report, directory, shard_bytes=shard_bytes)
        return list(staged['publish_files']) + list(staged['local_state_files'])

    def publish(request):
        assert_trainer_lease()
        return publish_model(request)

    result = transaction.run(update=update, stage=stage, publish_model=publish)
    assert_trainer_lease()
    output = result['output_checkpoint']
    if optimizer.global_step != result['report']['optimizer_step']:
        raise RecoveryRequired('published update requires original staged model and optimizer restoration')
    files = {name.removeprefix('model/'): row['sha256'] for name, row in result['artifacts'].items() if name.startswith('model/')}
    if runtime.checkpoint_id != output:
        if runtime.checkpoint_id != binding.checkpoint:
            raise RecoveryRequired('published update actor is neither original input nor committed output')
        acknowledge = getattr(runtime, 'acknowledge_model_publication', None)
        if not callable(acknowledge):
            raise RecoveryRequired('runtime needs authenticated checkpoint handover hook')
        acknowledge(output, files)
    optimizer.input_checkpoint = output
    # The host may atomically advance its authenticated CURRENT before old
    # state retirement. A failure leaves complete published evidence for cold
    # recovery; never rerun the optimizer to repair a discovery pointer.
    if commit_current is not None:
        commit_current(result)
    retired = retire_obsolete_optimizer_states(transaction.root.parent,
        run=binding.run, keep_checkpoint=output, retire_models=retire_published_models)
    if retired:
        record(dict(stage='optimizer-retention', removed_directories=retired,
                    retained_checkpoint=output, retained_committed_generations=1))
    return result
