"""Stage callbacks for the existing leased learner transaction.

The transaction must persist its updating/staged/published journal, reconcile
lost acknowledgements, authenticate receipts, and only then move discovery.
This module supplies the actual model/local-state staging operation. It does
not start a second controller, uploader, payout writer or training loop.
"""
from pathlib import Path
import os

from .data import digest
from .optimizer import _hash_file, _write_json


def stage_for_publication(runtime, optimizer, report, directory, *, shard_bytes=256 * 1024**2):
    """Export model bytes for R2 and local-only optimizer recovery shards.

    Returns explicit disjoint inventories. Only `publish_files` are sent to the
    model bucket; `local_state_files` stay on this trainer. Existing transaction
    retention keeps one current committed optimizer generation plus the new
    in-flight generation, at most two total, until publication is acknowledged.
    """
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
    files = {}
    for path in sorted(model.rglob('*')):
        if path.is_symlink():
            raise ValueError('model export cannot contain symlinks')
        if not path.is_file():
            continue
        relative = path.relative_to(model).as_posix()
        if 'optimizer' in relative.lower() or relative.endswith(('.pt', '.bin', '.pkl')):
            raise ValueError('model-only safetensors/config/tokenizer publication')
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        files[relative] = dict(sha256=_hash_file(path), bytes=path.stat().st_size)
    if not files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('actual exported model tensor files required')
    checkpoint_files = {name: row['sha256'] for name, row in files.items()}
    output_checkpoint = digest(checkpoint_files)
    descriptor = optimizer.save_local(root / 'optimizer', output_checkpoint=output_checkpoint,
                                      shard_bytes=shard_bytes)
    state_files = {path.relative_to(root).as_posix(): dict(sha256=_hash_file(path), bytes=path.stat().st_size)
                   for path in sorted((root / 'optimizer').iterdir()) if path.is_file()}
    staged = dict(version='affine-mimo-model-only-stage-v1', input_checkpoint=report['input_checkpoint'],
        output_checkpoint=output_checkpoint, checkpoint_files=checkpoint_files,
        optimizer_step=optimizer.global_step, report_sha256=digest(report),
        optimizer_descriptor_sha256=digest(descriptor),
        publish_files={'model/' + name: row for name, row in files.items()},
        local_state_files=state_files, retain_committed_generations=1,
        max_local_optimizer_generations=2)
    _write_json(root / 'STAGED.json', staged)
    return staged


def stage_callback(runtime, optimizer, *, shard_bytes=256 * 1024**2):
    """Adapter for existing transaction `stage(report, directory)` callbacks."""
    def stage(report, directory):
        return stage_for_publication(runtime, optimizer, report, directory, shard_bytes=shard_bytes)
    return stage
