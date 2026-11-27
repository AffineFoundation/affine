"""v8.2.4 candidate staging: same files, names, bytes and descriptors as
lifecycle.stage_for_publication / CPUAdamW.save_local, but
* optimizer shards are serialized in memory (safetensors.torch.save, byte-identical
  to save_file), hashed from those bytes, and written+fsynced by a thread pool;
* model files are hashed in parallel (one thread per file);
* the (inode stamp, digest) of every file is returned so the transaction's
  existing per-run hash memo can be seeded (cold/recovery paths still re-read);
* optional background mode: optimizer shards are written after the model is
  published; `barrier()` must complete before the next optimizer step (see
  ACTIVATION/RISKS for the crash-recovery semantics).
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib

from subnet.mimo_grpo.data import digest
from subnet.mimo_grpo.optimizer import _write_json


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _hash_path(path):
    h = hashlib.sha256()
    with open(path, 'rb', buffering=0) as f:
        while True:
            b = f.read(16 * 1024**2)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_bytes(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
        return _stamp(os.fstat(fd))
    finally:
        os.close(fd)


def save_local_parallel(optimizer, directory, *, output_checkpoint, shard_bytes=256 * 1024**2, workers=8):
    """Byte-identical to CPUAdamW.save_local; returns (descriptor, memo)."""
    from safetensors.torch import save as st_save
    from subnet.mimo_grpo.data import sha
    if optimizer.failed or not sha(output_checkpoint) or shard_bytes < 4096:
        raise ValueError('valid committed candidate and bounded shard size')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    elements = shard_bytes // 4
    jobs = []
    for index, (name, _) in enumerate(optimizer.parameters):
        master = optimizer.masters[name]
        state = optimizer.optimizer.state[master]
        if optimizer.global_step and int(state['step']) != optimizer.global_step:
            raise ValueError('all parameter moments at the same optimizer step')
        for slot in ('master', 'exp_avg', 'exp_avg_sq'):
            tensor = master.detach() if slot == 'master' else state.get(slot)
            zeros = tensor is None
            for start in range(0, master.numel(), elements):
                jobs.append((index, name, slot, start, zeros))

    def one(job):
        index, name, slot, start, zeros = job
        import torch
        master = optimizer.masters[name]
        tensor = torch.zeros_like(master) if zeros else (
            master.detach() if slot == 'master' else optimizer.optimizer.state[master][slot])
        value = tensor.reshape(-1)[start:start + elements].contiguous()
        data = st_save({'value': value})
        filename = f'{index:04d}-{slot}-{start:012d}.safetensors'
        stamp = _write_bytes(root / filename, data)
        return dict(file=filename, parameter=name, slot=slot, offset=start, elements=value.numel(),
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest()), stamp

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='opt-save') as pool:
        results = list(pool.map(one, jobs))  # ordered like the original loop
    chunks = [r[0] for r in results]
    descriptor = dict(version='affine-mimo-local-adamw-v1', optimizer_step=optimizer.global_step,
        input_checkpoint=optimizer.input_checkpoint, checkpoint=output_checkpoint,
        inventory=optimizer.inventory, config=optimizer.config.document(), chunks=chunks)
    _write_json(root / 'descriptor.json', descriptor)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    memo = {root / c['file']: (stamp, dict(sha256=c['sha256'], size=c['bytes'])) for c, stamp in results}
    return descriptor, memo


def stage_for_publication_fast(runtime, optimizer, report, directory, *, shard_bytes=256 * 1024**2,
                               workers=8, on_model_ready=None):
    """Same STAGED.json and inventories as stage_for_publication. Returns (staged, memo)."""
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
        # e.g. start the immutable model upload while optimizer shards are written
        on_model_ready(model, {name: dict(size=row['bytes'], sha256=row['sha256']) for name, row in files.items()},
                       output_checkpoint)
    descriptor, opt_memo = save_local_parallel(optimizer, root / 'optimizer',
        output_checkpoint=output_checkpoint, shard_bytes=shard_bytes, workers=workers)
    memo.update(opt_memo)
    desc_path = root / 'optimizer' / 'descriptor.json'
    memo[desc_path] = (_stamp(os.lstat(desc_path)), dict(sha256=_hash_path(desc_path), size=desc_path.stat().st_size))
    state_files = {path.relative_to(root).as_posix(): dict(sha256=memo[path][1]['sha256'], bytes=memo[path][1]['size'])
                   for path in sorted((root / 'optimizer').iterdir()) if path.is_file()}
    staged = dict(version='affine-mimo-model-only-stage-v1', input_checkpoint=report['input_checkpoint'],
        output_checkpoint=output_checkpoint, checkpoint_files=checkpoint_files,
        optimizer_step=optimizer.global_step, report_sha256=digest(report),
        optimizer_descriptor_sha256=digest(descriptor),
        publish_files={'model/' + name: row for name, row in files.items()},
        local_state_files=state_files, retain_committed_generations=1,
        max_local_optimizer_generations=2)
    _write_json(root / 'STAGED.json', staged)
    return staged, memo
