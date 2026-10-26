"""v8.2.10: optimizer-state persistence OFF the critical path (engine.optimizer_persistence='none').

Authorized by Jacob (owner) 2026-10-22: "saving the optimizer state should not be important can we remove
this requirement". The epoch report must never wait on writing ~107 GB of fp32 masters + AdamW moments.

* stage_for_publication_model_only: identical model export / hashing / STAGED.json shape as
  v824_staging_fast.stage_for_publication_fast (same model bytes, names, checkpoint id), but the
  optimizer generation is a single small local marker `optimizer/NOT-PERSISTED.json` instead of 1,372
  shards. The marker satisfies the unchanged transaction journal (`optimizer/` artifact present) and
  binds step / checkpoints / inventory / objective. Nothing optimizer-related is published (unchanged).
  The live optimizer state stays in host memory in the persistent trainer process (unchanged).
* restore_transaction_state: cold-process recovery (after a trainer restart). A generation that has
  the legacy `optimizer/descriptor.json` restores exactly as before (original function, unchanged).
  A generation with the NOT-PERSISTED marker restores the published model and rebuilds the optimizer
  with FRESH AdamW moments and the step counter kept: fp32 masters = upcast of the exact bf16 output
  weights, exp_avg = exp_avg_sq = 0, step = N. Same method as the approved 2026-10-14 and 2026-10-22 (e88)
  recoveries (rebuild_trainer_state_pod.py), now automatic. All journal / report / receipt / model
  inventory authentication of the original function is kept.
"""
import json
import os
import sys
from pathlib import Path

from subnet.mimo_grpo.data import digest
from subnet.mimo_grpo.optimizer import _write_json

MARKER = 'NOT-PERSISTED.json'
MARKER_VERSION = 'affine-mimo-optimizer-not-persisted-v1'
RECOVERY = 'fresh-adamw-moments-step-kept-masters-from-output-bf16'


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def marker_document(optimizer, output_checkpoint):
    return dict(version=MARKER_VERSION, optimizer_step=optimizer.global_step,
                input_checkpoint=optimizer.input_checkpoint, checkpoint=output_checkpoint,
                inventory_sha256=digest(optimizer.inventory), config=optimizer.config.document(),
                state_location='trainer-process-memory', persisted=False, recovery=RECOVERY)


def stage_for_publication_model_only(runtime, optimizer, report, directory, *, shard_bytes=256 * 1024**2,
                                     workers=8, on_model_ready=None):
    """Same model staging + STAGED.json shape as stage_for_publication_fast; optimizer = marker only."""
    from v824_staging_fast import _hash_path
    from concurrent.futures import ThreadPoolExecutor
    root = Path(directory)
    if report.get('status') != 'updated' or report['optimizer_step'] != optimizer.global_step:
        raise ValueError('only a completed original optimizer update can be staged')
    if optimizer.failed or report['input_checkpoint'] != optimizer.input_checkpoint:
        raise ValueError('healthy original input lineage')
    root.mkdir(parents=True, exist_ok=True)
    model = root / 'model'
    if model.exists() or (root / 'optimizer').exists() or (root / 'STAGED.json').exists():
        raise ValueError('existing staging requires transaction reconciliation, never overwrite')
    runtime.save_model(model)
    paths = []
    for path in sorted(model.rglob('*')):
        if path.is_symlink():
            raise ValueError('model export cannot contain symlinks')
        if not path.is_file():
            continue
        relative = path.relative_to(model).as_posix()
        if 'optimizer' in relative.lower() or relative.endswith(('.pt', '.bin', '.pkl')):
            raise ValueError('model-only safetensors/config/tokenizer publication')
        paths.append((relative, path))

    def model_hash(item):
        relative, path = item
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
            before = _stamp(os.fstat(fd))
        finally:
            os.close(fd)
        value = _hash_path(path)
        if _stamp(os.lstat(path)) != before:
            raise ValueError('model file changed while hashing')
        return relative, path, before, value

    with ThreadPoolExecutor(max_workers=max(1, min(len(paths), workers)), thread_name_prefix='model-hash') as pool:
        hashed = list(pool.map(model_hash, paths))
    files = {rel: dict(sha256=h, bytes=st[2]) for rel, _, st, h in hashed}
    memo = {p: (st, dict(sha256=h, size=st[2])) for _, p, st, h in hashed}
    if not files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('actual exported model tensor files required')
    checkpoint_files = {name: row['sha256'] for name, row in files.items()}
    output_checkpoint = digest(checkpoint_files)
    if on_model_ready is not None:
        on_model_ready(model, {name: dict(size=row['bytes'], sha256=row['sha256']) for name, row in files.items()},
                       output_checkpoint)
    # v8.2.10: no optimizer shards. One small marker (the in-memory state is authoritative).
    if optimizer.failed:
        raise ValueError('healthy optimizer required')
    (root / 'optimizer').mkdir(parents=False, exist_ok=False)
    document = marker_document(optimizer, output_checkpoint)
    marker = root / 'optimizer' / MARKER
    _write_json(marker, document)
    fd = os.open(root / 'optimizer', os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    memo[marker] = (_stamp(os.lstat(marker)), dict(sha256=_hash_path(marker), size=marker.stat().st_size))
    state_files = {'optimizer/' + MARKER: dict(sha256=memo[marker][1]['sha256'], bytes=memo[marker][1]['size'])}
    staged = dict(version='affine-mimo-model-only-stage-v1', input_checkpoint=report['input_checkpoint'],
        output_checkpoint=output_checkpoint, checkpoint_files=checkpoint_files,
        optimizer_step=optimizer.global_step, report_sha256=digest(report),
        optimizer_descriptor_sha256=digest(document),
        publish_files={'model/' + name: row for name, row in files.items()},
        local_state_files=state_files, retain_committed_generations=1,
        max_local_optimizer_generations=2)
    _write_json(root / 'STAGED.json', staged)
    print(json.dumps(dict(component='trainer-host-v8210', event='optimizer-persistence-skipped',
                          optimizer_step=optimizer.global_step, output_checkpoint=output_checkpoint,
                          state_location='trainer-process-memory')), flush=True)
    return staged, memo


def make_restore(original_restore):
    """Wrap the original cold-recovery function; legacy generations use it unchanged."""
    def restore_transaction_state(workspace, *, expected_job, load_runtime, config=None,
                                  assert_trainer_lease=lambda: None):
        from subnet.mimo_grpo.config import TrainerConfig
        config = TrainerConfig() if config is None else config
        root = Path(workspace)
        if not (root / 'optimizer' / MARKER).exists() or (root / 'optimizer' / 'descriptor.json').exists():
            return original_restore(workspace, expected_job=expected_job, load_runtime=load_runtime,
                                    config=config, assert_trainer_lease=assert_trainer_lease)
        return _restore_fresh_moments(workspace, expected_job=expected_job, load_runtime=load_runtime,
                                      config=config, assert_trainer_lease=assert_trainer_lease)
    return restore_transaction_state


def _restore_fresh_moments(workspace, *, expected_job, load_runtime, config, assert_trainer_lease):
    """Original restore_transaction_state checks; optimizer = fresh moments at the recorded step."""
    import torch
    from subnet.mimo_grpo.optimizer import CPUAdamW, restorable_config
    from subnet.mimo_grpo.transaction import (Job, Transaction, RecoveryRequired, _checkpoint,
                                              _inventory, _validate_report, _receipt)
    config.validate()
    if not isinstance(expected_job, Job):
        raise ValueError('authenticated original Job required for cold recovery')
    assert_trainer_lease()
    transaction = Transaction(workspace, expected_job)
    with transaction.exclusive():
        value = transaction._read()
        if value is None or value['state'] not in ('staged', 'published'):
            raise RecoveryRequired('complete original staging required for cold recovery')
        if value.get('report_sha256') != digest(value.get('report')):
            raise ValueError('original recovery report integrity')
        _validate_report(expected_job, value['report'])
        if not restorable_config(value['report']['config'], config):
            raise ValueError('original recovery optimizer/objective settings')
        if _inventory(transaction.root, list(value['artifacts'])) != value['artifacts']:
            raise ValueError('original staged model and optimizer marker content required')
        if set(n for n in value['artifacts'] if n.startswith('optimizer/')) != {'optimizer/' + MARKER}:
            raise ValueError('not-persisted generation must hold exactly the marker')
        checkpoint, models = _checkpoint(value['artifacts'])
        if value['state'] == 'published':
            if value.get('output_checkpoint') != checkpoint:
                raise ValueError('published recovery output binding')
            _receipt(expected_job, checkpoint, value.get('publication'))
        marker = json.loads((transaction.root / 'optimizer' / MARKER).read_bytes())
        step = expected_job.optimizer_step_before + 1
        if (marker.get('version') != MARKER_VERSION or marker.get('optimizer_step') != step
                or marker.get('checkpoint') != checkpoint or marker.get('input_checkpoint') != expected_job.input_checkpoint
                or marker.get('persisted') is not False or not restorable_config(marker.get('config'), config)):
            raise ValueError('not-persisted optimizer marker binding')
        assert_trainer_lease()
        runtime = load_runtime(transaction.root / 'model',
            dict(id=checkpoint, files={name: row['sha256'] for name, row in models.items()}))
        try:
            if runtime.checkpoint_id != checkpoint:
                raise ValueError('cold recovery must load actual original output model')
            optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=checkpoint)
            if digest(optimizer.inventory) != marker['inventory_sha256']:
                optimizer.failed = True
                raise ValueError('not-persisted marker parameter inventory')
            for name, master in optimizer.masters.items():
                optimizer.optimizer.state[master] = dict(step=torch.tensor(float(step)),
                    exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master))
            optimizer.global_step = step
            assert_trainer_lease()
            print(json.dumps(dict(component='trainer-host-v8210', event='optimizer-fresh-moments-recovery',
                                  optimizer_step=step, checkpoint=checkpoint, recovery=RECOVERY)), flush=True)
            return runtime, optimizer, value
        except BaseException:
            close = getattr(runtime, 'close', None)
            if callable(close):
                close()
            raise
