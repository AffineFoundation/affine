#!/usr/bin/env python3
"""Run a private, signed MiMo trainer job through the actual worker and R2.

The coordinator must already have captured real miner documents, graded them
natively and signed the opening, dispatch and trainer job. This command neither
creates outcomes nor edits proof bindings. Its lease is an exclusive local
rehearsal lease, not a qualification of distributed queue ownership.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.storage import Identity, canonical
from subnet.mimo_grpo.data import digest
from subnet.mimo_training_worker import TrainingContext
from subnet.mimo_worker import validate, validate_report
from ops.mimo_training_qualification import optimizer_digest


def read_json(path, maximum=256 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('bounded ordinary original input required')
    return json.loads(path.read_bytes())


def private_directory(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('ordinary absolute private path required')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = path.stat()
    if mode.st_uid != os.getuid() or mode.st_mode & 0o077:
        raise ValueError('owner-only private rehearsal directory required')
    return path


def write_new(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def identity(path):
    path = Path(path)
    mode = path.stat()
    if path.is_symlink() or not stat.S_ISREG(mode.st_mode) or mode.st_uid != os.getuid() or mode.st_mode & 0o077:
        raise ValueError('owner-only worker signing seed required')
    return Identity(bytes.fromhex(path.read_text().strip()))


def signed(actor, value):
    return dict(payload=value, signer=actor.id,
        signature=base64.b64encode(actor.key.sign(canonical(value)).signature).decode())


def run(args):
    from subnet.backend_jobs import execute
    envelope = read_json(args.job)
    job, manifest = validate(envelope, args.authority)
    if job['role'] != 'train' or not manifest['epoch'].startswith(('private-', 'test-', 'nonpayable-')):
        raise ValueError('one prospective private training job required')
    worker = identity(args.worker_seed)
    if job['trainer_worker'] != worker.id:
        raise ValueError('private worker identity differs from assigned trainer')
    root = private_directory(args.workspace)
    state = private_directory(args.trainer_state_root)
    output = Path(args.output).absolute()
    if output.exists() or output.parent != root:
        raise ValueError('new summary file directly in the private workspace required')
    lease_path = state / '.private-rehearsal-lease'
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    context = TrainingContext()
    began = time.monotonic()
    def guard():
        os.fstat(lease_fd)
        if time.time() >= job['expires_at']:
            raise ValueError('original trainer job expired')
    def execute_original(label):
        workspace = private_directory(root / label)
        result = execute(envelope, args.authority, workspace,
            assert_lease=guard, trainer_state_root=state, training_context=context)
        validate_report(result, job, manifest, args.authority, digest(job), time.time())
        write_new(root / (label + '-report.signed.json'), signed(worker, result))
        return result
    try:
        first = execute_original('first')
        if first['training']['status'] != 'updated':
            raise ValueError('private functional training requires a genuine eligible mixed group')
        first_state = optimizer_digest(context.optimizer)
        first_step = context.optimizer.global_step
        first_receipt = digest(first['publication_receipt'])
        before_retry = time.monotonic()
        retry = None
        if args.cold_retry:
            # Destroy both actor and optimizer before re-entering through the
            # same authenticated job in a different attempt workspace.
            context.close()
            del context
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            context = TrainingContext()
            retry = execute_original('cold-retry')
            if (context.optimizer.global_step != first_step or
                    optimizer_digest(context.optimizer) != first_state or
                    retry['output_checkpoint'] != first['output_checkpoint'] or
                    digest(retry['publication_receipt']) != first_receipt):
                raise ValueError('cold retry changed the original update or publication')
        summary = dict(version='mimo-private-trainer-worker-rehearsal-v1',
            scope='actual-signed-trainer-worker-private-r2', private=True,
            public_discovery_changed=False, chain_transactions=False,
            convergence_evidence=False, distributed_queue_qualified=False,
            lease_kind='exclusive-local-private-rehearsal', job_sha256=digest(job),
            worker=worker.id, source_closure_sha256=digest(job['source_files']),
            input_checkpoint=manifest['checkpoint']['id'], output_checkpoint=first['output_checkpoint'],
            trained_groups=first['training']['group_count'], trained_rollouts=first['training']['rollout_count'],
            optimizer_step=first_step, optimizer_state_digest=first_state,
            publication_receipt_sha256=first_receipt, model_full_get_verified=True,
            optimizer_state_uploaded=False, cold_retry_exact=retry is not None,
            initial_execution_seconds=before_retry - began,
            cold_retry_seconds=time.monotonic() - before_retry if retry is not None else None,
            total_seconds=time.monotonic() - began)
        write_new(output, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        context.close()
        os.close(lease_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('job','authority','worker-seed','workspace','trainer-state-root','output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--cold-retry', action='store_true',
        help='destroy actor/optimizer and retry the original signed job without another update')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
