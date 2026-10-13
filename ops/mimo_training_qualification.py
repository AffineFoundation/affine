#!/usr/bin/env python3
"""One isolated real-model GRPO update from authenticated native task groups.

This writes only a private local workspace. It does not mine, mint labels,
publish to R2, advance public discovery, sign qualification or set chain weights.
The model pin is prepared by ops/mimo_qualification.py on the target runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.mimo_runtime import MimoRuntime, canonical, digest, file_hash
from subnet.mimo_grpo import TrainerConfig, execute_transaction, restore_transaction_state
from subnet.mimo_grpo.optimizer import CPUAdamW
from subnet.mimo_grpo.prepared import prepare_signed_dispatch
from subnet.mimo_grpo.resources import admit_fresh_optimizer
from subnet.mimo_grpo.transaction import Job


def read_json(path, maximum=128 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('bounded ordinary prepared input required')
    return json.loads(path.read_bytes())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def optimizer_digest(optimizer):
    """Stream resident FP32 state without allocating a second full copy."""
    result = hashlib.sha256(canonical(dict(step=optimizer.global_step,
        checkpoint=optimizer.input_checkpoint, inventory=optimizer.inventory)))
    for name, master in optimizer.masters.items():
        state = optimizer.optimizer.state[master]
        result.update(canonical([name, int(state['step'])]))
        for tensor in (master.detach(), state['exp_avg'], state['exp_avg_sq']):
            raw = memoryview(tensor.contiguous().numpy()).cast('B')
            for offset in range(0, len(raw), 8 * 1024**2):
                result.update(raw[offset:offset + 8 * 1024**2])
    return result.hexdigest()


def local_publisher(root):
    """Diagnostic-only durable copy/readback, explicitly not R2 qualification."""
    def publish(request):
        destination = root/request['output_checkpoint']
        destination.mkdir(parents=True, exist_ok=True)
        for name, descriptor in request['model_files'].items():
            if Path(name).name != name or 'optimizer' in name.lower():
                raise ValueError('model-only diagnostic publication')
            source, target = Path(request['model_directory'])/name, destination/name
            if not target.exists():
                temporary = destination/(name + '.partial')
                with source.open('rb') as src, temporary.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024**2)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(temporary, target)
            if target.stat().st_size != descriptor['size'] or file_hash(target) != descriptor['sha256']:
                raise ValueError('diagnostic model readback differs from original export')
        evidence = dict(mode='local-diagnostic-copy-readback-not-r2',
            checkpoint=request['output_checkpoint'], model_files=request['model_files'])
        return dict(job_id=request['job_id'], output_checkpoint=request['output_checkpoint'],
            model_full_get_verified=True, optimizer_state_uploaded=False,
            receipt_sha256=digest(evidence))
    return publish


def run(args):
    import torch
    from subnet.mimo_grpo.data import sha
    if not sha(args.source_closure_sha256) or not sha(args.pin_sha256):
        raise ValueError('authenticated source and runtime-pin sha256 required')
    pin = read_json(args.pin)
    if digest(pin) != args.pin_sha256:
        raise ValueError('runtime pin differs from operator-authenticated content')
    root = Path(args.workspace).absolute()
    if root.resolve() != root or root.exists():
        raise ValueError('new isolated ordinary qualification workspace required')
    root.mkdir(parents=True, mode=0o700)
    lock = os.open(root/'.trainer-lease', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    events = (root/'events.jsonl').open('x')
    runtime, optimizer = None, None
    timers, began = {}, time.monotonic()
    def record(row):
        events.write(json.dumps(dict(observed_at=time.time(), **row), sort_keys=True) + '\n')
        events.flush()
        print(json.dumps(row), flush=True)
    def lease():
        os.fstat(lock)
        if runtime is not None:
            runtime.guard()
    try:
        runtime = MimoRuntime.load(Path(args.checkpoint).resolve(), pin['checkpoint'],
            pin['mimo_runtime'], device='cuda', trainable=True)
        timers['initial_model_load_seconds'] = time.monotonic() - began
        prepared = prepare_signed_dispatch(read_json(args.opening), read_json(args.dispatch),
            authority=args.authority, documents_root=args.documents_root, runtime=runtime,
            optimizer_step=0, max_groups=args.max_groups)
        if not prepared['binding'].epoch.startswith(('private-', 'test-', 'nonpayable-')):
            raise ValueError('qualification accepts private/nonpayable epochs only')
        config = TrainerConfig(group_size=prepared['manifest']['group_contract']['group_size'])
        resources = admit_fresh_optimizer(runtime, root, extra_model_copies=1)
        record(dict(stage='resource-admission', **resources))
        write_new(root/'prepared-evidence.json', dict(
            binding=asdict(prepared['binding']), config=config.document(),
            native_receipts=prepared['receipts'], diagnostic_omitted_groups=prepared['diagnostic_omitted_groups']))
        optimizer = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=runtime.checkpoint_id)
        update_started = time.monotonic()
        transaction = execute_transaction(root/'transactions/update1', runtime, optimizer,
            prepared['groups'], prepared['binding'], manifest_sha256=digest(prepared['manifest']),
            source_closure_sha256=args.source_closure_sha256, assert_trainer_lease=lease,
            publish_model=local_publisher(root/'local-model-readback'), config=config, record=record)
        timers['update_and_export_seconds'] = time.monotonic() - update_started
        before_state = optimizer_digest(optimizer)
        trajectory = prepared['groups'][0].trajectories[0]
        before_scores = runtime.score(prepared['groups'][0].prompt, trajectory.tokens,
                                      gradients=False).detach().cpu()
        output_checkpoint = runtime.checkpoint_id
        job = dict(transaction['job'])
        job['prepared_documents_sha256'] = tuple(job['prepared_documents_sha256'])
        runtime.close()
        del optimizer
        optimizer = None
        runtime = None
        gc.collect()
        torch.cuda.empty_cache()
        started = time.monotonic()
        def load_output(path, descriptor):
            return MimoRuntime.load(path, descriptor, pin['mimo_runtime'], device='cuda', trainable=True)
        runtime, optimizer, restored_journal = restore_transaction_state(root/'transactions/update1',
            expected_job=Job(**job), load_runtime=load_output, config=config, assert_trainer_lease=lease)
        after_state = optimizer_digest(optimizer)
        after_scores = runtime.score(prepared['groups'][0].prompt, trajectory.tokens,
                                     gradients=False).detach().cpu()
        score_drift = float((before_scores - after_scores).abs().max())
        if (before_state != after_state or runtime.checkpoint_id != output_checkpoint or
                restored_journal != transaction or score_drift > config.old_actor_logprob_tolerance):
            raise ValueError('cold model/optimizer restore differs from original update')
        timers['cold_restore_seconds'] = time.monotonic() - started
        timers['total_seconds'] = time.monotonic() - began
        result = dict(version='affine-mimo-training-diagnostic-v1', qualification=False,
            scope='private-functional-training-and-local-readback-only',
            sampling_qualification=False, convergence_evidence=False,
            r2_publication_qualification=False, chain_weights_changed=False,
            source_closure_sha256=args.source_closure_sha256, runtime_pin_sha256=args.pin_sha256,
            runtime_profile=pin['mimo_runtime'], native_group_count=len(prepared['groups']),
            diagnostic_omitted_groups=prepared['diagnostic_omitted_groups'], resources=resources,
            update=transaction['report'], output_checkpoint=output_checkpoint,
            restored_optimizer_step=optimizer.global_step, optimizer_state_digest=after_state,
            cold_restore_exact=True, cold_restore_max_logprob_drift=score_drift,
            publication_mode='local-diagnostic-copy-readback-not-r2', timings=timers,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()))
        write_new(args.output, result)
        record(dict(stage='complete', output=str(args.output), qualification=False,
                    group_count=len(prepared['groups']), timings=timers))
    except BaseException as error:
        record(dict(stage='failed', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        events.close()
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pin', 'pin-sha256', 'opening', 'dispatch', 'authority',
                 'documents-root', 'workspace', 'output', 'source-closure-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-groups', type=int, default=1,
        help='complete native groups for a short functional diagnostic; no trajectory truncation')
    args = parser.parse_args()
    if not 1 <= args.max_groups <= 512:
        parser.error('--max-groups must be 1 through 512')
    run(args)


if __name__ == '__main__':
    main()
